from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator

img2mse = lambda x, y: torch.mean((x - y) ** 2)

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad, mask):
        ctx.save_for_backward(gt_grad, mask)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, mask= ctx.saved_tensors
        # print('-----------gt_grad: ', gt_grad.size())
        # print('-----------grad_scale: ', grad_scale.size())
        # print('-----------mask: ', mask.size())
        gt_grad = gt_grad * grad_scale * mask
        return gt_grad, None, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=self.precision_t,
            safety_checker = None
        )

        if vram_O:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        
        #    set timesteps
        self.scheduler = self.pipe.scheduler
        self.num_inference_steps = 1000
        # self.num_inference_steps = 500 # for iteration
        self.strength = 0.75
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.timesteps, self.num_inference_steps = self.pipe.get_timesteps(self.num_inference_steps, self.strength, device=self.device)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        # 
        self.latents = torch.zeros(1, 4, 64, 64)
        self.latents = self.latents.to(self.device)
        self.noise = torch.zeros(1, 4, 64, 64)
        self.noise = self.noise.to(device)
        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    # sd for normal_map
    def train_step_sd_normal(self, i, mask, prompt, pred_normal_map, guidance_scale=100, normal_start = 0, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        from PIL import Image, ImageOps
        import numpy as np

        latent_size = 512
        pred_normal_map = F.interpolate(pred_normal_map, (latent_size, latent_size), mode='bilinear', align_corners=False)
        mask = torch.abs(mask) 
        mask = F.interpolate(mask, (latent_size, latent_size), mode='bilinear', align_corners=False)
        
        height = latent_size
        width = latent_size
        callback_steps = 1
        num_images_per_prompt = 1
        negative_prompt = None
        prompt_embeds = None
        negative_prompt_embeds = None
        # guidance_scale = 7.5
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = None
        strength = self.strength
        timesteps = self.timesteps
        eta = 0.0
        
        ###
        # 0. check inputs
        self.pipe.check_inputs(
            prompt,
            pred_normal_map,
            mask,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 1. Encode input prompt
        prompt_embeds = self.pipe._encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            text_encoder_lora_scale,
        )
    
        # 2. Preprocess mask and image 
        masked_image = pred_normal_map[:, :3, :, :] * (mask < 0.5)
        init_image = pred_normal_map[:, :3, :, :]

        viz_images = torch.cat([pred_normal_map, masked_image], dim=0)

        num_channels_latents = 4
        num_channels_unet = 9
        return_image_latents = False
        return_noise = True
        generator = None
        latents = None
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        is_strength_max = strength == 1.0

        # 4. Prepare mask latent variables
        mask, masked_image_latents = self.pipe.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            do_classifier_free_guidance,
        )
        init_image = init_image.to(device=self.device, dtype=masked_image_latents.dtype)
        init_latents = self.pipe._encode_vae_image(init_image, generator=generator)

        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
        
        for j in range(1):
            # add noise
            # t = self.max_step - (self.max_step - self.min_step) * (i)/10000
            t = self.max_step - (self.max_step - self.min_step) * np.sqrt((i-normal_start)/20000)
            t = int(t)
            t = torch.tensor(t)
            
            latents_outputs = self.pipe.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                self.device,
                generator,
                latents,
                init_image,
                t,
                is_strength_max,
                return_noise,
                return_image_latents,
            )

            if return_image_latents:
                latents, noise, image_latents = latents_outputs
            else:
                latents, noise = latents_outputs
            

        # 5. Unet
        with torch.no_grad():    
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            w = (1 - self.scheduler.alphas_cumprod[t])
            grad = w * (noise_pred - noise) #* mask
            grad = torch.nan_to_num(grad)
            
            viz_images = torch.cat([latents[:,:3,:,:], grad[:,:3,:,:]], dim=0)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.pipe.run_safety_checker(image, self.device, prompt_embeds.dtype)
            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            image = self.pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)
            image_inpainted = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

        loss = SpecifyGradient.apply(latents, grad, mask[0,:,:,:])

        return loss 
     
    # sd + original sds (Note that this is a gfood baseline)
    def train_step_sd(self, i, mask, prompt, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        from PIL import Image, ImageOps
        import numpy as np

        latent_size = 512
        pred_rgb = F.interpolate(pred_rgb, (latent_size, latent_size), mode='bilinear', align_corners=False)
        mask = torch.abs(mask) 
        mask = F.interpolate(mask, (latent_size, latent_size), mode='bilinear', align_corners=False)
        
        height = latent_size
        width = latent_size
        callback_steps = 1
        num_images_per_prompt = 1
        negative_prompt = None
        prompt_embeds = None
        negative_prompt_embeds = None
        # guidance_scale = 7.5
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = None
        strength = self.strength
        timesteps = self.timesteps
        eta = 0.0
        
        ###
        # 0. check inputs
        self.pipe.check_inputs(
            prompt,
            pred_rgb,
            mask,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 1. Encode input prompt
        prompt_embeds = self.pipe._encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            text_encoder_lora_scale,
        )
    
        # 2. Preprocess mask and image 
        masked_image = pred_rgb[:, :3, :, :] * (mask < 0.5)
        init_image = pred_rgb[:, :3, :, :]

        num_channels_latents = 4
        num_channels_unet = 9
        return_image_latents = False
        return_noise = True
        generator = None
        latents = None
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        is_strength_max = strength == 1.0

        # 4. Prepare mask latent variables
        mask, masked_image_latents = self.pipe.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            do_classifier_free_guidance,
        )
        init_image = init_image.to(device=self.device, dtype=masked_image_latents.dtype)
        init_latents = self.pipe._encode_vae_image(init_image, generator=generator)

        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
        

        # for i, t in enumerate(timesteps):
        for j in range(1):
            # add noise
          
            t = self.max_step - (self.max_step - self.min_step) * np.sqrt((i)/20000)
            # t = self.max_step - (self.max_step - self.min_step) * ((i)/10000)
            t = int(t)
            t = torch.tensor(t)
            
            latents_outputs = self.pipe.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                self.device,
                generator,
                latents,
                init_image,
                t,
                is_strength_max,
                return_noise,
                return_image_latents,
            )

            if return_image_latents:
                latents, noise, image_latents = latents_outputs
            else:
                latents, noise = latents_outputs
            
        # 5. Unet
        with torch.no_grad():    
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            w = (1 - self.scheduler.alphas_cumprod[t])
            # grad = w * (init_latents - latents) #* mask
            grad = w * (noise_pred - noise) #* mask
            grad = torch.nan_to_num(grad)
        
            viz_images = torch.cat([latents[:,:3,:,:], grad[:,:3,:,:]], dim=0)
            save_image(viz_images, "image_SD_noalpha.png")

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.pipe.run_safety_checker(image, self.device, prompt_embeds.dtype)
            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            image = self.pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)
            image_inpainted = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
           
        loss = SpecifyGradient.apply(latents, grad, mask[0,:,:,:])

        return loss # grad.mean()

    # sd + collaborative sds (Note that this is a good baseline) nn means number of neighbors and mask4 means for nerighbors
    def train_step_colla_sds(self, i, mask_nn, prompt, pred_rgb_nn, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        from PIL import Image, ImageOps
        import numpy as np

        latent_size = 512
        NN = pred_rgb_nn.size(0)
        latent_size2 = int(latent_size/8)
        grad = torch.zeros(1, 4, latent_size2, latent_size2, device=self.device)
        for i in range(NN):
            pred_rgb = pred_rgb_nn[i,:,:,:]
            mask = mask_nn[i,:,:,:]
            pred_rgb = pred_rgb.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
            pred_rgb = F.interpolate(pred_rgb, (latent_size, latent_size), mode='bilinear', align_corners=False)
            mask = torch.abs(mask) 
            mask = F.interpolate(mask, (latent_size, latent_size), mode='bilinear', align_corners=False)
            
            height = latent_size
            width = latent_size
            callback_steps = 1
            num_images_per_prompt = 1
            negative_prompt = None
            prompt_embeds = None
            negative_prompt_embeds = None
            batch_size = 1
            do_classifier_free_guidance = guidance_scale > 1.0
            text_encoder_lora_scale = None
            strength = self.strength
            timesteps = self.timesteps
            eta = 0.0
            
            ###
            # 0. check inputs
            self.pipe.check_inputs(
                prompt,
                pred_rgb,
                mask,
                height,
                width,
                strength,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
            )

            # 1. Encode input prompt
            prompt_embeds = self.pipe._encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                text_encoder_lora_scale,
            )
        
            # 2. Preprocess mask and image 
            masked_image = pred_rgb[:, :3, :, :] * (mask < 0.5)
            init_image = pred_rgb[:, :3, :, :]

            num_channels_latents = 4
            num_channels_unet = 9
            return_image_latents = False
            return_noise = True
            generator = None
            latents = None
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            is_strength_max = strength == 1.0

            # 4. Prepare mask latent variables
            mask, masked_image_latents = self.pipe.prepare_mask_latents(
                mask,
                masked_image,
                batch_size * num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                self.device,
                generator,
                do_classifier_free_guidance,
            )
            init_image = init_image.to(device=self.device, dtype=masked_image_latents.dtype)
            init_latents = self.pipe._encode_vae_image(init_image, generator=generator)

            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
            
            # for i, t in enumerate(timesteps):
            for j in range(1):
                # add noise
              
                t = self.max_step - (self.max_step - self.min_step) * i/10000
                t = int(t)
                t = torch.tensor(t)
                
                latents_outputs = self.pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    self.device,
                    generator,
                    latents,
                    init_image,
                    t,
                    is_strength_max,
                    return_noise,
                    return_image_latents,
                )

                if return_image_latents:
                    latents, noise, image_latents = latents_outputs
                else:
                    latents, noise = latents_outputs
                
            # 5. Unet
            with torch.no_grad():    
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                if num_channels_unet == 9:
                        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                w = (1 - self.scheduler.alphas_cumprod[t])
                
                grad += w * (noise_pred - noise) #* mask
                grad = torch.nan_to_num(grad)
               
                from datetime import datetime
                # Get the current time
                current_time = datetime.now()
                # Format the current time as a string
                time_string = current_time.strftime("%Y%m%d_%H%M%S")
                # Generate the filename with the time string
                viz_images = torch.cat([latents[:,:3,:,:], grad[:,:3,:,:]], dim=0)
                # save_image(viz_images, f"image_{time_string}_noalpha.png")

                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                image, has_nsfw_concept = self.pipe.run_safety_checker(image, self.device, prompt_embeds.dtype)
                if has_nsfw_concept is None:
                    do_denormalize = [True] * image.shape[0]
                else:
                    do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
                image = self.pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)
                image_inpainted = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
                # image_inpainted.images[0].save(f"image_sdo_{i}_inpainted.png")

            loss = SpecifyGradient.apply(latents, grad, mask)

        return loss # grad.mean()

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




