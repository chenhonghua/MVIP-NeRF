from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
# import thre3d_atom.thre3d_reprs.cross_attn as ca
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import time
from torch.cuda.amp import custom_bwd, custom_fwd

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)

        # dummy loss value
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device,
                 sd_version='2.1',
                 hf_key=None,
                 t_sched_start = 1500,
                 t_sched_freq = 500,
                 t_sched_gamma = 1.0, auth_token=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.t_sched_start = t_sched_start
        self.t_sched_freq = t_sched_freq
        self.t_sched_gamma = t_sched_gamma

        print(f'[INFO] loading stable diffusion...')

        use_auth_token = False
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == '1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
            use_auth_token = auth_token
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", use_auth_token=use_auth_token).to(
            self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer",
                                                       use_auth_token=use_auth_token)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder",
                                                          use_auth_token=use_auth_token).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet",
                                                         use_auth_token=use_auth_token).to(
            self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler",
                                                       use_auth_token=use_auth_token)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, self.device)

        self.min_step_ratio = 0.02
        self.min_step = int(self.num_train_timesteps * self.min_step_ratio)

        self.max_step_ratio = 0.98
        self.max_step = int(self.num_train_timesteps * self.max_step_ratio)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_max_step_ratio(self):
        return self.max_step_ratio

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    # def get_attn_map(self, prompt, pred_rgb, timestamp=0, indices_to_fetch=[7], guidance_scale=100,  logvar=None):
    #     prompt = [prompt]
    #     batch_size = len(prompt)
    #     controller = ca.AttentionStore()
    #     ca.register_attention_control(self.unet, controller)
    #     # interp to 512x512 to be fed into vae.

    #     with torch.no_grad():
    #         orig_im_h, orig_im_w = pred_rgb.shape[-2:]
    #         text_embeddings = self.get_text_embeds(prompt, '')
    #         pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
    #         t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
    #         if timestamp > 0:
    #             t = torch.as_tensor(timestamp, dtype=torch.long, device=self.device)
    #         latents = self.encode_imgs(pred_rgb_512)
    #         latents = latents.expand(batch_size, self.unet.in_channels, 512 // 8, 512 // 8).to(self.device)
    #         noise = torch.randn_like(latents)
    #         latents_noisy = self.scheduler.add_noise(latents, noise, t)
    #         latent_model_input = torch.cat([latents_noisy] * 2)
    #         noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #         noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
    #         latents = controller.step_callback(latents)

    #         attn_maps = None
    #         if indices_to_fetch is not None:
    #             attn_maps = ca.aggregate_and_get_max_attention_per_token(
    #                 prompts=prompt,
    #                 attention_store=controller,
    #                 indices_to_alter=indices_to_fetch, orig_im_h=orig_im_h, orig_im_w=orig_im_h
    #             )
    #     return attn_maps, t.item()


    # sd
    def train_step(self, i, masks, text, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        from PIL import Image, ImageOps
        import numpy as np

        # latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        save_image(pred_rgb_512, "pred_rgb_512.png")
        masks = torch.abs(masks) 
        masks = F.interpolate(masks, (512, 512), mode='bilinear', align_corners=False)

        # print('-----masks.shape: ', masks.size())
        # print('-----pred_rgb_512.shape: ', pred_rgb_512.size())

        latents = self.encode_imgs(pred_rgb_512)
        
        with torch.no_grad():
            # print('-------------masks: ', masks)
            
            # latent = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
            # pred_rgb2 = F.interpolate(pred_rgb[:, :3, :, :], (512, 512), mode='bilinear', align_corners=False) #* 2 - 1
            
            # pred_rgb_512 = self.decode_latents(pred_rgb)
            # rescaled_image = (pred_rgb2 + 1) / 2.0
            # Multiply by 255 to get the final image in the range [0, 255]
            # pred_rgb2 = (rescaled_image * 255).to(torch.uint8)
            
            
            # import global_variables
            # print('-----i: ', i)
            # if i == 1:
            image_inpainted, grad = self.pipe(text, pred_rgb_512, masks)
            # print('------type(): ', type(image_inpainted))
            image_inpainted = image_inpainted.images[0]
            image_inpainted.save("pred_rgb_512_inpainted.png")
            # global_variables.image_inpainted = image_inpainted
            # global_variables.grad = grad
        
            # image_inpainted = global_variables.image_inpainted
            # grad = global_variables.grad
        
            # print('--------grad.size: ', grad.size())
            # image_inpainted2 = image_inpainted.images[0]
            # image_inpainted.save("dog_inpainted300.png")

            # numpy_image = pred_rgb_512.squeeze()
            # numpy_image = numpy_image.cpu().detach().numpy()
            # # print('------------numpy_image size: ',numpy_image.shape)
            # numpy_image = np.transpose(numpy_image, (1, 2, 0))

            # numpy_image = (numpy_image - np.min(numpy_image)) / (np.max(numpy_image) - np.min(numpy_image))
            # numpy_image = (numpy_image * 255).astype(np.uint8)
        
            # # image_inpainted2 = np.array(image_inpainted2)

            # # print('numpy_image: ', numpy_image)
            # # print('image_inpainted: ', image_inpainted)

            # concatenated_array = np.concatenate((numpy_image, numpy_image), axis=1)  # Change axis to 0 for vertical concatenation
            # concatenated_image = Image.fromarray(concatenated_array)
            # concatenated_image.save("stable_diffusion.png")

            # import torchvision.transforms.functional as TF
            # # Assuming tensor is a GPU tensor with size 1x4x64x64
            # print('--------grad.size: ', grad)
            # tensor_cpu = grad.to('cpu')  # Move tensor to CPU
            # tensor_cpu = tensor_cpu.squeeze(0)
            # tensor_pil = TF.to_pil_image(tensor_cpu)  # Convert tensor to PIL Image
            # # Save the PIL Image
            # tensor_pil.save("stable-diffusion_grad.png")

        print('--------grad.size: ', grad.size())
        print('--------latents.size: ', latents.size())
        grad = grad[:, :3, :, :]
        loss = SpecifyGradient.apply(latents, grad)

        return loss # grad.mean()


    # sd + new formulation of latent sds + each render is a state of DDPM
    def train_step_sd_sds2(self, i, mask, prompt, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        from PIL import Image, ImageOps
        import numpy as np

        # print('mask shape: ', torch.unique(mask))
        mask = torch.abs(mask)
        # print('----------------------------------------')
        # print(mask < 0.5)

        masked_image = pred_rgb[:, :3, :, :] * (mask < 0.5)
        viz_images = torch.cat([masked_image, pred_rgb[:, :3, :, :]], dim=0)
        # save_image(viz_images, f"image_inputs.png")
        
        pred_rgb = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, (512, 512), mode='bilinear', align_corners=False)
        masked_image = F.interpolate(masked_image, (512, 512), mode='bilinear', align_corners=False)

        
        height = 512
        width = 512
        callback_steps = 1
        num_images_per_prompt = 1
        negative_prompt = None
        prompt_embeds = None
        negative_prompt_embeds = None
        guidance_scale = 25
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = None
        strength = self.strength
        timesteps = self.timesteps
        eta = 0.0

        # masks = F.interpolate(masks, (512, 512), mode='bilinear', align_corners=False)
        # latents = F.interpolate(pred_rgb[:, :3, :, :], (512, 512), mode='bilinear', align_corners=False) * 2 - 1
        
        ###
        # 0. check inputs
        self.pipe.check_inputs(
            prompt,
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
    
        # 2. Preprocess mask and image and latents
        # masked_image = pred_rgb[:, :3, :, :] * (mask < 0.5)
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
        init_latent = self.pipe._encode_vae_image(init_image, generator=generator)
        print('-------init_latent: ', init_latent.size())
        
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # add noise
        t = self.max_step - (self.max_step - self.min_step) * np.sqrt(i/10000)
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

        if i == 101:
            self.noise = noise
            self.latents = latents
        else:
            self.noise = (noise + self.noise)/2
            self.latents = (latents + self.latents)/2        

        noise = self.noise
        latents = self.latents
        
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

            
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        self.latents = latents
        
        w = (1 - self.scheduler.alphas_cumprod[t])
        # grad = w * (init_latent - latents) #* mask
        grad = w * (noise_pred - noise) #* mask
        grad = torch.nan_to_num(grad)
        loss = grad 

            # # # for visualization
            # viz_images = torch.cat([latents[:,:3,:,:], noise[:,:3,:,:], noise_pred[:,:3,:,:], grad[:,:3,:,:]], dim=0)
            # from datetime import datetime
            # # Get the current time
            # current_time = datetime.now()
            # # Format the current time as a string
            # time_string = current_time.strftime("%Y%m%d_%H%M%S")
            # # Generate the filename with the time string
            # # save_image(viz_images, f"image_{time_string}_noalpha.png")

            # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # image, has_nsfw_concept = self.pipe.run_safety_checker(image, self.device, prompt_embeds.dtype)
            # if has_nsfw_concept is None:
            #     do_denormalize = [True] * image.shape[0]
            # else:
            #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            # image = self.pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)
            
            # # print('--------------image.size(): ', image.size())
            
            # image_inpainted = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
            # # image_inpainted.images[0].save(f"image_{i}_{time_string}_inpainted.png")

        # loss_z = img2mse(init_latent, latents)
        # loss_dec = 0.0
        # # loss_dec = img2mse(pred_rgb, image2)
        # loss = loss_z + loss_dec
        # print('---------decoded loss: ', loss_dec)
        loss = SpecifyGradient.apply(latents, grad)

        return loss
    

    # sd + formulation of latent sds on sd
    def train_step_sd_sds_latents(self, i, mask, prompt, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        from PIL import Image, ImageOps
        import numpy as np

        # print('mask shape: ', torch.unique(mask))
        mask = torch.abs(mask)
        # print('----------------------------------------')
        # print(mask < 0.5)

        masked_image = pred_rgb[:, :3, :, :] * (mask < 0.5)
        viz_images = torch.cat([masked_image, pred_rgb[:, :3, :, :]], dim=0)
        # save_image(viz_images, f"image_inputs.png")
        
        pred_rgb = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, (512, 512), mode='bilinear', align_corners=False)
        masked_image = F.interpolate(masked_image, (512, 512), mode='bilinear', align_corners=False)

        
        height = 512
        width = 512
        callback_steps = 1
        num_images_per_prompt = 1
        negative_prompt = None
        prompt_embeds = None
        negative_prompt_embeds = None
        guidance_scale = 2.5
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = None
        strength = self.strength
        timesteps = self.timesteps
        eta = 0.0

        # masks = F.interpolate(masks, (512, 512), mode='bilinear', align_corners=False)
        # latents = F.interpolate(pred_rgb[:, :3, :, :], (512, 512), mode='bilinear', align_corners=False) * 2 - 1
        
        ###
        # 0. check inputs
        self.pipe.check_inputs(
            prompt,
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
    
        # 2. Preprocess mask and image and latents
        # masked_image = pred_rgb[:, :3, :, :] * (mask < 0.5)
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
        init_latent = self.pipe._encode_vae_image(init_image, generator=generator)
        
        
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
        
        # set t
        t = self.max_step - (self.max_step - self.min_step) * np.sqrt(i/20000)
        jt = int(t)
        jt = torch.tensor(jt)

        # add noise
        latents_outputs = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
            pred_rgb[:, :3, :, :],
            jt,
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
            # iterative denoising
            # while jt == 0:          
                # print('----------jt: ', jt)
                # add noise
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, jt)
            
            if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            
            noise_pred = self.unet(
                latent_model_input,
                jt,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents2 = self.scheduler.step(noise_pred, jt, latents, **extra_step_kwargs, return_dict=False)[0]
            
            w = (1 - self.scheduler.alphas_cumprod[jt])
            grad = w * (init_latent - latents2) #* mask
            # grad = (init_latent - latents) #* mask
            # grad = (noise_pred - noise) #* mask
            grad = torch.nan_to_num(grad)
            # loss = grad 

            # # for visualization
            viz_images = torch.cat([latents2[:,:3,:,:], noise[:,:3,:,:], noise_pred[:,:3,:,:], grad[:,:3,:,:]], dim=0)
            from datetime import datetime
            # Get the current time
            current_time = datetime.now()
            # Format the current time as a string
            time_string = current_time.strftime("%Y%m%d_%H%M%S")
            # Generate the filename with the time string
            save_image(viz_images, f"image_{time_string}_noalpha.png")

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image2, has_nsfw_concept = self.pipe.run_safety_checker(image, self.device, prompt_embeds.dtype)
            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            image2 = self.pipe.image_processor.postprocess(image2, output_type='pil', do_denormalize=do_denormalize)          
            image_inpainted = StableDiffusionPipelineOutput(images=image2, nsfw_content_detected=has_nsfw_concept)
            # image_inpainted.images[0].save(f"image_{i}_{time_string}_inpainted.png")
            jt = jt - 10

        # loss_z = img2mse(init_latent, latents)
        # loss_dec = 0.0
        # img_res = image_inpainted.images[0]
        # img_res_npy= np.array(img_res)
        # if image is None:
        # loss_dec = img2mse(pred_rgb, image)
        # loss = loss_z + 0.01*loss_dec
        # print('---------decoded loss: ', loss_dec)
        # print('---------latent loss: ', loss_z)
        loss = SpecifyGradient.apply(latents, grad)

        return loss

# sd for depth map
    def train_step_sd_depth(self, i, mask, prompt, pred_depth, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        from PIL import Image, ImageOps
        import numpy as np

        
        pred_depth = pred_depth.expand(1, 3, -1, -1)
        print('-------------prompt: ', prompt)
        print('-------------pred_depth: ', pred_depth.size())

        pred_depth = F.interpolate(pred_depth, (512, 512), mode='bilinear', align_corners=False)
        mask = torch.abs(mask) 
        mask = F.interpolate(mask, (512, 512), mode='bilinear', align_corners=False)
        
        height = 512
        width = 512
        callback_steps = 1
        num_images_per_prompt = 1
        negative_prompt = None
        prompt_embeds = None
        negative_prompt_embeds = None
        guidance_scale = 25
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
        masked_image = pred_depth[:, :3, :, :] * (mask < 0.5)
        init_image = pred_depth[:, :3, :, :]

        # viz_images = torch.cat([pred_depth, pred_depth], dim=0)
        # save_image(viz_images, f"image_inputs2.png")

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
            # t = torch.randint(self.min_step, self.max_step + 1 -400, [1], dtype=torch.long, device=self.device)
            # t = timesteps[300+i]

            t = self.max_step - (self.max_step - self.min_step) * np.sqrt(i/10000)
            t = int(t)
            t = torch.tensor(t)
            
            print('----------timesteps shape:', timesteps.size())
            print('----------t:', t)

            # noise = torch.randn_like(latents)
            # latents = self.pipe.scheduler.add_noise(latents, noise, t)
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
            
            # print('---------noise: ', torch.unique(noise))

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
            grad = 1.5 * w * (noise_pred - noise) #* mask
            grad = torch.nan_to_num(grad)
            # loss = grad 

            viz_images = torch.cat([latents[:,:3,:,:], noise[:,:3,:,:], noise_pred[:,:3,:,:], grad[:,:3,:,:]], dim=0)
            
            # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            from datetime import datetime
            # Get the current time
            current_time = datetime.now()
            # Format the current time as a string
            time_string = current_time.strftime("%Y%m%d_%H%M%S")
            # Generate the filename with the time string
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

        loss = SpecifyGradient.apply(latents, grad)

        return loss # grad.mean()
    

    def train_step_sd_rgbd(self, i, mask, prompt, pred_rgb, pred_depth, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        from PIL import Image, ImageOps
        import numpy as np

        
        pred_rgbd = torch.cat([pred_rgb, pred_depth], dim=1)
        print('-------------prompt: ', prompt)
        # print('-------------pred_rgbd: ', pred_rgbd.size())

        pred_rgbd = F.interpolate(pred_rgbd, (512, 512), mode='bilinear', align_corners=False)
        mask = torch.abs(mask) 
        mask = F.interpolate(mask, (512, 512), mode='bilinear', align_corners=False)
        
        height = 512
        width = 512
        callback_steps = 1
        num_images_per_prompt = 1
        negative_prompt = None
        prompt_embeds = None
        negative_prompt_embeds = None
        guidance_scale = 15
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
        masked_image = pred_rgbd[:, :4, :, :] * (mask < 0.5)
        init_image = pred_rgbd[:, :4, :, :]

        # viz_images = torch.cat([pred_rgbd, pred_rgbd], dim=0)
        # save_image(viz_images, f"image_inputs2.png")

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
            # t = torch.randint(self.min_step, self.max_step + 1 -400, [1], dtype=torch.long, device=self.device)
            # t = timesteps[300+i]

            t = self.max_step - (self.max_step - self.min_step) * np.sqrt(i/10000)
            t = int(t)
            t = torch.tensor(t)
            
            # print('----------timesteps shape:', timesteps.size())
            # print('----------t:', t)

            # noise = torch.randn_like(latents)
            # latents = self.pipe.scheduler.add_noise(latents, noise, t)
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
            
            # print('---------noise: ', torch.unique(noise))

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
            # loss = grad 

            viz_images = torch.cat([latents[:,:,:,:], noise[:,:,:,:], noise_pred[:,:,:,:], grad[:,:3,:,:]], dim=0)
            
            # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            from datetime import datetime
            # Get the current time
            current_time = datetime.now()
            # Format the current time as a string
            time_string = current_time.strftime("%Y%m%d_%H%M%S")
            # Generate the filename with the time string
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

        loss = SpecifyGradient.apply(latents, grad)

        return loss # grad.mean()
    

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, global_step=-1, logvar=None):
        # schedule max step:
        if global_step >= self.t_sched_start and global_step % self.t_sched_freq == 0:
            self.max_step_ratio = self.max_step_ratio * self.t_sched_gamma

            # if self.max_step_ratio < self.min_step_ratio * 2:

            if self.max_step_ratio < 0.27:
                #self.max_step_ratio = self.min_step_ratio * 2 # don't let it get too low!
                self.max_step_ratio = 0.27 # don't let it get too low!
            else:
                print(f"Updating max step to {self.max_step_ratio}")

        self.max_step = int(self.num_train_timesteps * self.max_step_ratio)

        # interp to 512x512 to be fed into vae.
        # _t = time.time()
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        save_image(pred_rgb_512, "pred_rgb_512.png")
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        from datetime import datetime
        # Get the current time
        current_time = datetime.now()
        # Format the current time as a string
        time_string = current_time.strftime("%Y%m%d_%H%M%S")
        viz_images = torch.cat([noise[:,:3,:,:], noise_pred[:,:3,:,:], grad[:,:3,:,:]],dim=0)
        save_image(viz_images, f"image_{time_string}_noalpha.png")

        ####
        if True:
            with torch.no_grad():
                if True:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1 
                b = len(noise_pred)
                a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(torch.float32)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(torch.float32))

                # TODO: also denoise all-the-way

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, 'stable-diffusion.png')
        ####

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        if logvar != None:
            grad = grad * torch.exp(-1 * logvar)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # _t = time.time()
        loss = SpecifyGradient.apply(latents, grad)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return loss

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

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
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()

class scoreDistillationLoss(nn.Module):
    def __init__(self,
                 device,
                 prompt,
                 t_sched_start = 1500,
                 t_sched_freq = 500,
                 t_sched_gamma = 1.0,
                 directional = True):
        super().__init__()
        self.dir_to_indx_dict = {}
        self.directional = directional
        
        # get sd model
        self.sd_model = StableDiffusion(device,
                                        "2.0",
                                        t_sched_start=t_sched_start,
                                        t_sched_freq=t_sched_freq,
                                        t_sched_gamma=t_sched_gamma)

        # encode text
        if directional:
            self.text_encodings = {}
            for dir_prompt in ['side', 'overhead', 'back', 'front']:
                print(f"Encoding text for \'{dir_prompt}\' direction")
                modified_prompt = prompt + f", {dir_prompt} view"
                self.text_encodings[dir_prompt] = self.sd_model.get_text_embeds(modified_prompt, '')
        else:
            self.text_encoding = self.sd_model.get_text_embeds(prompt, '')

    def get_current_max_step_ratio(self):
        return self.sd_model.get_max_step_ratio()

    def training_step(self, output, image_height, image_width, directions=None, global_step=-1, logvars=None):
        loss = 0
        if self.directional:
            assert (directions != None), f"Must supply direction if SDS loss is set to directional mode"
        # format output images
        out_imgs = torch.reshape(output, (-1, image_height, image_width, 3))
        out_imgs = out_imgs.permute((0, 3, 1, 2))

        # perform training step
        if not self.directional:
            loss = self.sd_model.train_step(self.text_encoding, out_imgs, global_step=global_step, logvar=logvars)
        else:
            for idx, dir_prompt in enumerate(directions):
                if logvars != None:
                    logvar = logvars[idx]
                else:
                    logvar = None
                encoding = self.text_encodings[dir_prompt]
                loss = loss + self.sd_model.train_step(encoding, out_imgs, global_step=global_step, logvar=logvar)

        return loss