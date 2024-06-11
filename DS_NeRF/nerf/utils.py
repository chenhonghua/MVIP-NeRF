import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def adjust_text_embeddings(embeddings, azimuth, opt):
    text_z_list = []
    weights_list = []
    K = 0
    for b in range(azimuth.shape[0]):
        text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth[b], opt)
        K = max(K, weights_.shape[0])
        text_z_list.append(text_z_)
        weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def get_view_direction(thetas, phis, overhead, front):
    #                   phis: [B,];          thetas: [B,]
    # front = 0             [-front/2, front/2)
    # side (cam left) = 1   [front/2, 180-front/2)
    # back = 2              [180-front/2, 180+front/2)
    # side (cam right) = 3  [180+front/2, 360-front/2)
    # top = 4               [0, overhead]
    # bottom = 5            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


#######************ add ***********#############
def rand_poses(size, device, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size, device=device),
                torch.abs(torch.randn(size, device=device)),
                torch.randn(size, device=device),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


class Pretrain_Model(object):
    def __init__(self, opt, device, guidance):

        self.opt = opt
        self.device = device #cuda

        self.global_step = 0

        # guide model
        self.guidance = guidance

        self.embeddings = {}

        # text prompt / images
        if self.guidance is not None:
            for key in self.guidance:
                for p in self.guidance[key].parameters():
                    p.requires_grad = False
                self.embeddings[key] = {}

        if self.guidance is not None:
            self.prepare_embeddings_text()

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings_text(self):

        # text embeddings (stable-diffusion)
        if self.opt.text is not None:

            # if 'SD' in self.guidance:
                # self.embeddings['SD']['default'] = self.guidance['SD'].get_text_embeds([self.opt.text])
                # self.embeddings['SD']['uncond'] = self.guidance['SD'].get_text_embeds([self.opt.negative])

                # for d in ['front', 'side', 'back']:
                #     self.embeddings['SD'][d] = self.guidance['SD'].get_text_embeds([f"{self.opt.text}, {d} view"])

            if 'clip' in self.guidance:
                self.embeddings['clip']['text'] = self.guidance['clip'].get_text_embeds(self.opt.text)

    @torch.no_grad()
    def prepare_embeddings_image(self):

        if self.opt.images is not None:

            if 'clip' in self.guidance:
                self.embeddings['clip']['image'] = self.guidance['clip'].get_img_embeds(self.rgb)

    def cal_loss(self, i, rgbs4_tensor, pre_normal_map, pred_depth, pred_rgb, rgb, masks, mask4, B=1):

        self.rgb = rgb  # 1,3,H,W
        self.pred_rgb = pred_rgb  # 1,3,H,W
        self.pred_depth = pred_depth
        self.pre_normal_map = pre_normal_map  # N,3,H,W
        self.rgbs4_tensor = rgbs4_tensor  # N,3,H,W

        if self.guidance is not None:
            self.prepare_embeddings_image()

        self.B = B

        self.masks = masks
        # self.mask_cropped = mask_cropped

        # random pose on the fly
        _, _, _, phis, _ = rand_poses(self.B, self.device, self.opt,
                                      radius_range=self.opt.radius_range,
                                      theta_range=self.opt.theta_range,
                                      phi_range=self.opt.phi_range, return_dirs=True,
                                      angle_overhead=self.opt.angle_overhead,
                                      angle_front=self.opt.angle_front,
                                      uniform_sphere_rate=self.opt.uniform_sphere_rate)

        # delta polar/azimuth/radius to default view
        delta_azimuth = phis - self.opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]

        data = {'azimuth': delta_azimuth}

        # interpolate text_z
        azimuth = data['azimuth']  # [-180, 180]


        self.global_step += 1
        # experiment iterations ratio
        # i.e. what proportion of this experiment have we completed (in terms of iterations) so far?
        exp_iter_ratio = (self.global_step - self.opt.exp_start_iter) / (
                    self.opt.exp_end_iter - self.opt.exp_start_iter)

        # progressively relaxing view range
        if self.opt.progressive_view:
            r = min(1.0, self.opt.progressive_view_init_ratio + 2.0 * exp_iter_ratio)
            self.opt.phi_range = [self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[0] * r,
                                  self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[1] * r]
            self.opt.theta_range = [self.opt.default_polar * (1 - r) + self.opt.full_theta_range[0] * r,
                                    self.opt.default_polar * (1 - r) + self.opt.full_theta_range[1] * r]
            self.opt.radius_range = [self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                                     self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r]
            self.opt.fovy_range = [self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                                   self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r]


        loss = 0

        if 'SD' in self.guidance:
            if True:                
                if self.opt.is_rgb_guidance:
                    if self.opt.is_crop:
                        loss = loss + self.guidance['SD'].train_step_sd(i, mask_cropped, self.opt.text, self.pred_rgb, as_latent=True,
                                                                    guidance_scale=self.opt.guidance_scale,
                                                                    grad_scale=self.opt.lambda_guidance,
                                                                    save_guidance_path=self.opt.save_guidance_path)
                    else:
                        loss = loss + self.guidance['SD'].train_step_sd(i, masks, self.opt.text, self.pred_rgb, as_latent=True,
                                                                guidance_scale=self.opt.rgb_guidance_scale,
                                                                grad_scale=self.opt.lambda_guidance,
                                                                save_guidance_path=self.opt.save_guidance_path)

                if self.opt.is_colla_guidance and i > 0:
                    loss = loss + self.guidance['SD'].train_step_colla_sds(i, mask4, self.opt.text, self.rgbs4_tensor, as_latent=True,
                                                                guidance_scale=self.opt.colla_guidance_scale,
                                                                grad_scale=self.opt.lambda_guidance,
                                                                save_guidance_path=self.opt.save_guidance_path)
                
                if self.opt.is_normal_guidance and i > self.opt.normal_start:
                    loss = 1.0*loss + 1.0*self.guidance['SD'].train_step_sd_normal(i, masks, self.opt.text_normal, self.pre_normal_map, as_latent=True,
                                                                guidance_scale=self.opt.normal_guidance_scale, normal_start = self.opt.normal_start,
                                                                grad_scale=self.opt.lambda_guidance,
                                                                save_guidance_path=self.opt.save_guidance_path)

        # if 'clip' in self.guidance:
        #     # empirical, far view should apply smaller CLIP loss
        #     lambda_guidance = 7.5 #10 * (1 - abs(azimuth) / 180) * self.opt.lambda_guidance

        #     loss = loss + 1*self.guidance['clip'].train_step(self.embeddings['clip'], self.pred_rgb,
        #                                                    grad_scale=lambda_guidance)
        
        return loss
