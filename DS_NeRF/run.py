import copy
import os
import sys
import threading

import numpy as np
import imageio
from PIL import Image
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import tkinter as tk

import matplotlib.pyplot as plt

from load_nerd import load_nerd_data
from load_blender import load_blender_data
from run_nerf_helpers_tcnn import NeRF_TCNN
from run_nerf_helpers import *
from correspondence_utils import *

from load_llff import load_llff_data, load_colmap_depth
from load_dtu import load_dtu_data

from loss import SigmaLoss

from data import RayDataset
from torch.utils.data import DataLoader

from utils.generate_renderpath import generate_renderpath
import cv2
import lpips
import scipy.spatial as sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### add
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
device_ids = list(range(num_gpus))
###

# torch.cuda.set_device(0)
np.random.seed(0)
DEBUG = False

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,   #############
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=0.01,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=float, default=10,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_test_ray", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true',
                        help='render the train set instead of render_poses path')
    parser.add_argument("--render_mypath", action='store_true',
                        help='render the test path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / custom')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=1000000,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000,  #100000
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    # debug
    parser.add_argument("--debug", action='store_true')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')
    parser.add_argument("--colmap_depth", action='store_true',
                        help="Use depth supervision by colmap.")
    parser.add_argument("--depth_loss", action='store_true',
                        help="Use depth supervision by colmap - depth loss.")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help="Depth lambda used for loss.")
    parser.add_argument("--sigma_loss", action='store_true',
                        help="Use depth supervision by colmap - sigma loss.")
    parser.add_argument("--sigma_lambda", type=float, default=0.1,
                        help="Sigma lambda used for loss.")
    parser.add_argument("--weighted_loss", action='store_true',
                        help="Use weighted loss by reprojection error.")
    parser.add_argument("--relative_loss", action='store_true',
                        help="Use relative loss.")
    parser.add_argument("--depth_with_rgb", action='store_true',
                        help="single forward for both depth and rgb")
    parser.add_argument("--normalize_depth", action='store_true',
                        help="normalize depth before calculating loss")

    parser.add_argument("--no_tcnn", action='store_true',
                        help='set to not use tinycudann and use the original NeRF')

    parser.add_argument("--clf_weight", type=float, default=0.01,
                        help='The weight of the classification loss')
    parser.add_argument("--clf_reg_weight", type=float, default=0.01,
                        help='The weight of the classification regularizer')
    parser.add_argument("--feat_weight", type=float,
                        default=0.01, help='The weight of the feature loss')
    parser.add_argument("--i_feat", type=int, default=10,
                        help='frequency of calculating the feature loss')
    parser.add_argument("--prepare", action='store_true',
                        help='Prepare depths for inpainting')
    parser.add_argument("--lpips", action='store_true',
                        help='use perceptual loss for rgb inpainting')
    parser.add_argument("--N_gt", type=int, default=0,
                        help='Number of ground truth inpainted samples')
    parser.add_argument("--N_train", type=int, default=None,
                        help='Number of training images used for optimization')
    parser.add_argument("--train_gt", action='store_true',
                        help='Use the gt inpainted images to train a NeRF')
    parser.add_argument("--masked_NeRF", action='store_true',
                        help='Only train NeRF on unmasked pixels')
    parser.add_argument("--object_removal", action='store_true',
                        help='Remove the object and shrink the masks')
    parser.add_argument("--tmp_images", action='store_true',
                        help='Use images in lama_images_tmp for ablation studies')
    parser.add_argument("--no_geometry", action='store_true',
                        help='Stop using inpainted depths for training')

    ##dreamfusion
    #######************* add **********#############
    parser.add_argument("--lpips_render_factor", type=int, default=1,
                        help='The stride (render factor) used for sampling patches for the perceptual loss')
    parser.add_argument("--patch_len_factor", type=int, default=2,
                        help='The resizing factor to obtain the side lengths of the patches for the perceptual loss')
    parser.add_argument("--lpips_batch_size", type=int, default=4,
                        help='The number of patches used in each iteration for the perceptual loss')

    ##diffusion part
    #######************* add **********#############
    parser.add_argument('--save_guidance_path', default='dream_fusion.png', type=str, help="save_guidance_path")
    parser.add_argument('--text_normal', default='A stone bench on a grass ground', help="text prompt") # a cylindric stone pedestal on the grass in front of building walls and trees
    parser.add_argument('--text_depth', default='A stone bench on a grass ground', help="text prompt") # a cylindric stone pedestal on the grass in front of building walls and trees
    parser.add_argument('--text', default='A stone bench on a grass ground', help="text prompt") # a cylindric stone pedestal on the grass in front of building walls and trees
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--image', default='', help="image prompt")

    parser.add_argument('--guidance', type=str, nargs='*', default=['SD'], help='guidance model')
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")

    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    parser.add_argument('--guidance_scale', type=float, default=75, help="diffusion model classifier-free guidance scale")

    # @ for ours setting (config.txt)
    #######************* add **********#############
    parser.add_argument('--is_normal_guidance', action='store_true', help="diffusion model classifier-free guidance scale")
    parser.add_argument('--normal_guidance_scale', type=float, default=7.5, help="diffusion model classifier-free guidance scale")
    parser.add_argument("--normal_start", type=int, default=0, help='start normal sds')

    parser.add_argument('--is_rgb_guidance', action='store_true', help="diffusion model classifier-free guidance scale")
    parser.add_argument('--rgb_guidance_scale', type=float, default=7.5, help="diffusion model classifier-free guidance scale")
    parser.add_argument("--rgb_start", type=int, default=0, help='start rgb sds')

    parser.add_argument('--is_colla_guidance', action='store_true', help="diffusion model classifier-free guidance scale")
    parser.add_argument('--colla_guidance_scale', type=float, default=7.5, help="diffusion model classifier-free guidance scale")
    parser.add_argument("--colla_start", type=int, default=0, help='start colla sds')

    parser.add_argument('--is_depth_guidance', action='store_true', help="diffusion model classifier-free guidance scale")
    parser.add_argument('--depth_guidance_scale', type=float, default=7.5, help="diffusion model classifier-free guidance scale")

    parser.add_argument('--is_crop', action='store_true', help="crop bbox for diffusion")
    parser.add_argument("--first_stage", action='store_true', help="first_stage.")
    parser.add_argument("--second_stage", action='store_true', help="second_stage.")
    parser.add_argument('--sds_loss_weight', type=float, default=0.0001, help="weight for sds loss")
    parser.add_argument('--normalmap_render_factor', type=int, default=4,
                        help='The stride (render factor) used for rendering normal map for the normal sds loss')
    # end

    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[3.0, 3.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[10, 30], help="training camera fovy range")

    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0, help="likelihood of sampling camera location uniformly on the sphere surface area")

    parser.add_argument('--exp_start_iter', type=int, default=None, help="start iter # for experiment, to calculate progressive_view and progressive_level")
    parser.add_argument('--exp_end_iter', type=int, default=None, help="end iter # for experiment, to calculate progressive_view and progressive_level")

    parser.add_argument('--progressive_view', action='store_true', help="progressively expand view sampling range from default to full")
    parser.add_argument('--progressive_view_init_ratio', type=float, default=0.2, help="initial ratio of final range, used for progressive_view")

    parser.add_argument('--progressive_level', action='store_true', help="progressively increase gridencoder's max_level")

    parser.add_argument("--sds_loss", action='store_true',
                        help="Use SDS supervision.")

    #######************ add ***********#############

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    args.train_gt = True

    gnrt_losses = []
    disc_losses = []

    if args.lpips:
        LPIPS = lpips.LPIPS(net='vgg')
        # LPIPS.eval()
        for param in LPIPS.parameters():
            param.requires_grad = False

    # Load data
    if args.dataset_type == 'custom':
        _, images, poses, render_poses, masks, inpainted_depths, mask_indices, depths = load_custom_data(args.datadir,
                                                                                                         args.factor,
                                                                                                         recenter=True,
                                                                                                         bd_factor=.75,
                                                                                                         spherify=args.spherify,
                                                                                                         prepare=args.prepare,
                                                                                                         args=args)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        i_test = []
        print('Loaded custom', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto custom holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0]))])
        else:
            i_train = np.array([i for i in args.train_scene if
                                (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        near = 0.1
        far = 6.
        print('NEAR FAR', near, far)

        # if args.prepare:  # todo uncomment?
        #     masks = np.abs(masks)

        if args.object_removal:
            masks = np.abs(masks)

        if args.N_gt > 0:
            if not args.train_gt:
                i_test = i_train[:args.N_gt]
                if args.N_train is None:
                    i_train = i_train[args.N_gt:]
                else:
                    i_train = i_train[args.N_gt:args.N_gt + args.N_train]
            else:
                i_test = i_train
                i_train = i_train[:args.N_gt]

    elif args.dataset_type == 'llff':
        if args.colmap_depth:
            depth_gts = load_colmap_depth(args.datadir, factor=args.factor, bd_factor=.75, prepare=args.prepare)
        images, poses, bds, render_poses, i_test, masks, inpainted_depths, mask_indices = load_llff_data(args.datadir,
                                                                                                         args.factor,
                                                                                                         recenter=True,
                                                                                                         bd_factor=.75,
                                                                                                         spherify=args.spherify,
                                                                                                         prepare=args.prepare,
                                                                                                         args=args)

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir, poses.shape)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0]))])
        else:
            i_train = np.array([i for i in args.train_scene if
                                (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        if args.object_removal:
            masks = np.abs(masks)

        if args.N_gt > 0:
            if not args.train_gt:
                i_test = i_train[:args.N_gt]
                if args.N_train is None:
                    i_train = i_train[args.N_gt:]
                else:
                    i_train = i_train[args.N_gt:args.N_gt + args.N_train]
            else:
                i_test = i_train
                i_train = i_train[:args.N_gt]

    elif args.dataset_type == 'dtu':
        images, poses, hwf = load_dtu_data(args.datadir)
        print('Loaded DTU', images.shape, poses.shape, hwf, args.datadir)
        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])
        else:
            i_train = np.array([i for i in args.train_scene if
                                (i not in i_test and i not in i_val)])

        near = 0.1
        far = 5.0
        if args.colmap_depth:
            depth_gts = load_colmap_depth(
                args.datadir, factor=args.factor, bd_factor=.75)
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, masks, objects = load_blender_data(args.datadir,
                                                                                      args.half_res,
                                                                                      args.testskip)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * \
                images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
    elif args.dataset_type == 'nerd':
        images, poses, bds, render_poses, i_test, masks, objects = load_nerd_data(args.datadir, args.factor,
                                                                                  recenter=True, bd_factor=.75,
                                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        # plt.imshow(images[0])
        # plt.savefig('sample.png')
        # plt.clf()
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])
    elif args.render_mypath:
        render_poses = generate_renderpath(
            np.array(poses[i_test])[3:4], focal, sc=1)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    if args.no_tcnn:
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
            args)
    else:
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf_tcnn(
            args)

    gui = threading.Thread(target=gui_application,
                           args=(args, render_kwargs_test,))
    gui.start()

    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only: # False
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = images[i_train]

            if args.render_test:
                testsavedir = os.path.join(
                    basedir, expname, 'renderonly_{}_{:06d}'.format('test', start))
            elif args.render_train:
                testsavedir = os.path.join(
                    basedir, expname, 'renderonly_{}_{:06d}'.format('train', start))
            else:
                testsavedir = os.path.join(
                    basedir, expname, 'renderonly_{}_{:06d}'.format('path', start))
            os.makedirs(testsavedir, exist_ok=True)

            if args.render_test_ray:
                index_pose = i_train[0]
                rays_o, rays_d = get_rays_by_coord_np(H, W, focal, poses[index_pose, :3, :4],
                                                      depth_gts[index_pose]['coord'])
                rays_o, rays_d = torch.Tensor(rays_o).to(
                    device), torch.Tensor(rays_d).to(device)
                rgb, sigma, z_vals, depth_maps = render_test_ray(rays_o, rays_d, hwf,
                                                                 network=render_kwargs_test['network_fine'],
                                                                 **render_kwargs_test)
                visualize_sigma(sigma[0, :].cpu().numpy(), z_vals[0, :].cpu().numpy(),
                                os.path.join(testsavedir, 'rays.png'))
                print("colmap depth:", depth_gts[index_pose]['depth'][0])
                print("Estimated depth:", depth_maps[0].cpu().numpy())
                print(depth_gts[index_pose]['coord'])
            else:
                rgbs, disps, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images,
                                             savedir=testsavedir, render_factor=args.render_factor, need_alpha=True)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(
                    testsavedir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
                disps[np.isnan(disps)] = 0
                print('Depth stats', np.mean(disps), np.max(
                    disps), np.percentile(disps, 95))
                imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(disps / np.percentile(disps, 95)), fps=30,
                                 quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand  # H * W
    use_batching = not args.no_batching # True
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p)
                        for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        if args.debug: # False
            print('rays.shape:', rays.shape)
        print('done, concats')
        labels = np.expand_dims(masks, axis=-1)  # [N, H, W, 1]
        labels = np.repeat(labels[:, None], 3, axis=1)  # [N, 3, H, W, 1]
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        print('rays_rgb.shape and labels.shape:', rays_rgb.shape,
                labels.shape, images.shape, masks.shape, poses.shape)
        rays_rgb = np.concatenate([rays_rgb, labels], -1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                            for i in i_train], 0)  # train images only

        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 4])
        rays_rgb = rays_rgb.astype(np.float16)

        # for depth_inpainting rays
        rays = np.stack([get_rays_np(H, W, focal, p)
                        for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        labels = np.expand_dims(inpainted_depths, axis=-1)  # [N, H, W, 1]
        labels = np.repeat(labels[:, None], 3, axis=1)  # [N, 3, H, W, 1]
        # [N, ro+rd+rgb, H, W, 3]
        rays_inp = np.concatenate([rays, images[:, None]], 1)
        print("########################", images.shape,
                rays.shape, rays_inp.shape, labels.shape)
        rays_inp = np.concatenate([rays_inp, labels], -1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_inp = np.transpose(rays_inp, [0, 2, 3, 1, 4])
        rays_inp = np.stack([rays_inp[i]
                            for i in i_train], 0)  # train images only

        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_inp = np.reshape(rays_inp, [-1, 3, 4])
        rays_inp = rays_inp.astype(np.float16)

        rays_depth = None
        if args.colmap_depth:
            print('get depth rays')
            rays_depth_list = []
            for i in i_train:
                if not args.prepare:
                    indices = [_ for _ in range(len(depth_gts[i]['coord']))
                                if masks[i][
                                    min(int(depth_gts[i]['coord'][_]
                                        [1]), masks[i].shape[0] - 1)
                    ][
                                    min(int(depth_gts[i]['coord'][_]
                                        [0]), masks[i].shape[1] - 1)
                    ] == 0
                    ]
                    depth_gts[i]['coord'] = depth_gts[i]['coord'][indices]
                    depth_gts[i]['weight'] = depth_gts[i]['weight'][indices]
                    depth_gts[i]['depth'] = depth_gts[i]['depth'][indices]
                rays_depth = np.stack(get_rays_by_coord_np(H, W, focal, poses[i, :3, :4], depth_gts[i]['coord']),
                                        axis=0)  # 2 x N x 3
                # print(rays_depth.shape)
                rays_depth = np.transpose(rays_depth, [1, 0, 2])
                depth_value = np.repeat(
                    depth_gts[i]['depth'][:, None, None], 3, axis=2)  # N x 1 x 3
                weights = np.repeat(
                    depth_gts[i]['weight'][:, None, None], 3, axis=2)  # N x 1 x 3
                rays_depth = np.concatenate(
                    [rays_depth, depth_value, weights], axis=1)  # N x 4 x 3
                rays_depth_list.append(rays_depth)

            rays_depth = np.concatenate(rays_depth_list, axis=0)
            print('rays_weights mean:', np.mean(rays_depth[:, 3, 0]))
            print('rays_weights std:', np.std(rays_depth[:, 3, 0]))
            print('rays_weights max:', np.max(rays_depth[:, 3, 0]))
            print('rays_weights min:', np.min(rays_depth[:, 3, 0]))
            print('rays_depth.shape:', rays_depth.shape)
            rays_depth = rays_depth.astype(np.float16)
            print('shuffle depth rays')
            # np.random.shuffle(rays_depth)

        if rays_depth is not None:
            max_depth = np.max(rays_depth[:, 3, 0])
        print('done')
        i_batch = 0

        if False:  # args.train_gt or args.prepare: # train_gt=True
            rays_rgb_clf = rays_rgb.reshape(-1, 3, 4)
        else:
            rays_rgb_clf = rays_rgb[rays_rgb[:, :, 3] == 0].reshape(-1, 3, 4)
        
        rays_rgb_sds = rays_rgb.reshape(-1, 3, 4)
        if not args.prepare:                                               ################# modified #####################
            rays_rgb = rays_rgb[rays_rgb[:, :, 3] == 1].reshape(-1, 3, 4)  ################# modified #####################
        
        print('shuffle rays')
        print(
            f'rays_rgb shape is {rays_rgb.shape} and rays_rgb_clf shape is {rays_rgb_clf.shape} and ray_rgb_full_sds shape is {rays_rgb_sds.shape}')
    
    if args.debug:
        return

    # print(rays_rgb.shape,rays_inp.shape,rays_rgb_clf.shape,rays_rgb_sds.shape)
    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if args.first_stage and use_batching:
        rays_depth = torch.Tensor(rays_depth).to(device) if rays_depth is not None else None
        raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), batch_size=N_rand, shuffle=True, num_workers=0,
                                       generator=torch.Generator(device=device)))
        raysINP_iter = iter(DataLoader(RayDataset(rays_inp), batch_size=N_rand, shuffle=True, num_workers=0,
                                       generator=torch.Generator(device=device)))
        raysDepth_iter = iter(DataLoader(RayDataset(rays_depth), batch_size=N_rand, shuffle=True, num_workers=0,
                                         generator=torch.Generator(device=device))) if rays_depth is not None else None
        raysRGBCLF_iter = iter(DataLoader(RayDataset(rays_rgb_clf), batch_size=N_rand, shuffle=True, num_workers=0,
                                          generator=torch.Generator(device=device)))
        raysRGBSDS_iter = iter(DataLoader(RayDataset(rays_rgb_sds), batch_size=N_rand, shuffle=True, num_workers=0,
                                          generator=torch.Generator(device=device)))
    elif args.second_stage and use_batching:
        raysRGBCLF_iter = iter(DataLoader(RayDataset(rays_rgb_clf), batch_size=N_rand, shuffle=True, num_workers=0,
                                          generator=torch.Generator(device=device)))
        raysDepth_iter = iter(DataLoader(RayDataset(rays_depth), batch_size=N_rand, shuffle=True, num_workers=0,
                                         generator=torch.Generator(device=device))) if rays_depth is not None else None
        raysINP_iter = iter(DataLoader(RayDataset(rays_inp), batch_size=N_rand, shuffle=True, num_workers=0,
                                       generator=torch.Generator(device=device)))


    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    ############## add ##############
    # guide model

    args.images = []
    if args.image is not None:
        args.images += [args.image]

    guidance = nn.ModuleDict()
    opt = args

    opt.exp_start_iter = opt.exp_start_iter or 0
    opt.exp_end_iter = opt.exp_end_iter or opt.N_iters

    if opt.progressive_view:

        opt.uniform_sphere_rate = 0
        # back up full range
        opt.full_radius_range = opt.radius_range
        opt.full_theta_range = opt.theta_range
        opt.full_phi_range = opt.phi_range
        opt.full_fovy_range = opt.fovy_range

    if 'SD' in opt.guidance:
        from guidance.sd_utils import StableDiffusion
        guidance['SD'] = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key,
                                         opt.t_range)
        guidance['SD'] = nn.DataParallel(guidance['SD'], device_ids=device_ids)
        if isinstance(guidance['SD'], nn.DataParallel):
            guidance['SD'] = guidance['SD'].module
    if 'clip' in opt.guidance:
        from guidance.clip_utils import CLIP

        guidance['clip'] = CLIP(device)

    from nerf.utils import Pretrain_Model

    pre_model = Pretrain_Model(args, device, guidance)

    ############ add ############

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    img_i = 0
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        # print('----------------------i: ', i)
        if args.first_stage and use_batching: # actually we do not need this stage
            # Random over all images
            # batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            try:
                batch_inp = next(raysINP_iter).to(device)
            except StopIteration:
                raysINP_iter = iter(DataLoader(RayDataset(rays_inp), batch_size=N_rand, shuffle=True, num_workers=0))
                batch_inp = next(raysINP_iter).to(device)
            try:
                batch = next(raysRGB_iter).to(device)
            except StopIteration:
                raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), batch_size=N_rand, shuffle=True, num_workers=0))
                batch = next(raysRGB_iter).to(device)
            try:
                batch_clf = next(raysRGBCLF_iter).to(device)
            except StopIteration:
                raysRGBCLF_iter = iter(DataLoader(RayDataset(rays_rgb_clf), batch_size=N_rand, shuffle=True, num_workers=0))
                batch_clf = next(raysRGBCLF_iter).to(device)
            try: #######################add for SDS
                batch_sds = next(raysRGBSDS_iter).to(device)
            except StopIteration:
                raysRGBSDS_iter = iter(DataLoader(RayDataset(rays_rgb_sds), batch_size=N_rand, shuffle=True, num_workers=0))
                batch_sds = next(raysRGBSDS_iter).to(device)

            ################ add ################

            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            batch_rays = batch_rays[:, :, :-1]
            target_s, label_s = target_s[:, :3], target_s[:, 3]

            batch_inp = torch.transpose(batch_inp, 0, 1)
            batch_inp, target_inp = batch_inp[:2], batch_inp[2]
            batch_inp = batch_inp[:, :, :-1]
            target_inp, depth_inp = target_inp[:, 3], target_inp[:, 3]

            batch_clf = torch.transpose(batch_clf, 0, 1)
            batch_rays_clf, target_clf = batch_clf[:2], batch_clf[2]
            batch_rays_clf = batch_rays_clf[:, :, :-1]
            target_clf, label_s_clf = target_clf[:, :3], target_clf[:, 3]


            batch_sds = torch.transpose(batch_sds, 0, 1)
            batch_rays_sds, target_sds = batch_sds[:2], batch_sds[2]
            batch_rays_sds = batch_rays_sds[:, :, :-1]
            target_sds, label_s_sds = target_sds[:, :3], target_sds[:, 3]

            if args.colmap_depth:
                # batch_depth = rays_depth[i_batch:i_batch+N_rand]
                try:
                    batch_depth = next(raysDepth_iter).to(device)
                except StopIteration:
                    raysDepth_iter = iter(DataLoader(RayDataset(rays_depth), batch_size=N_rand, shuffle=True, num_workers=0,generator=torch.Generator(device=device)))
                    batch_depth = next(raysDepth_iter).to(device)

                batch_depth = torch.transpose(batch_depth, 0, 1)
                batch_rays_depth = batch_depth[:2]  # 2 x B x 3
                target_depth = batch_depth[2, :, 0]  # B
                ray_weights = batch_depth[3, :, 0]
        elif args.second_stage: # For the second stage, only optimize the masked region
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]
            mask = masks[img_i]
            # image to rays
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                        -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
           
            # obtain masked rays
            mask = mask.flatten()
            select_inds = np.argwhere(mask == 1)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            select_coords = select_coords.squeeze(1)
            # print(select_coords)
            rays_o = rays_o[select_coords[:, 0],
                            select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0],
                            select_coords[:, 1]]  # (N_rand, 3)
            masked_batch_rays = torch.stack([rays_o, rays_d], 0)
            masked_target_s = target[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)

            # # obtain unmasked image
            try:
                batch_clf = next(raysRGBCLF_iter).to(device)
            except StopIteration:
                raysRGBCLF_iter = iter(DataLoader(RayDataset(rays_rgb_clf), batch_size=N_rand, shuffle=False, num_workers=0))
                batch_clf = next(raysRGBCLF_iter).to(device)
            
            batch_clf = torch.transpose(batch_clf, 0, 1)
            batch_rays_clf, target_clf = batch_clf[:2], batch_clf[2]
            batch_rays_clf = batch_rays_clf[:, :, :-1]
            target_clf, label_s_clf = target_clf[:, :3], target_clf[:, 3]
            
            try:
                batch_inp = next(raysINP_iter).to(device)
            except StopIteration:
                raysINP_iter = iter(DataLoader(RayDataset(rays_inp), batch_size=N_rand, shuffle=True, num_workers=0))
                batch_inp = next(raysINP_iter).to(device)
            batch_inp = torch.transpose(batch_inp, 0, 1)
            batch_inp, target_inp = batch_inp[:2], batch_inp[2]
            batch_inp = batch_inp[:, :, :-1]
            target_inp, depth_inp = target_inp[:, 3], target_inp[:, 3]
                

        #####  Core optimization loop  #####
        if args.first_stage and use_batching:
            # for uninpainted rgb region
            rgb, disp, _, _, _, _ = render(H, W, focal, chunk=args.chunk, rays=batch_rays_clf,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        else:
            # render masked regions--for rgb sds
            rgb, disp, _, _, _ = render(H, W, focal, chunk=args.chunk, rays=masked_batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
            
            # combin with unmasked regions and form a new image
            combin_rgb = images[img_i].clone().detach()
            combin_rgb = torch.Tensor(combin_rgb).to(device)
            mask = masks[img_i]
            mask = torch.Tensor(mask).to(device)
            combin_rgb[select_coords[:, 0], select_coords[:, 1]] = rgb # HW3
            # from torchvision.utils import save_image
            combin_rgb = combin_rgb.permute(2, 0, 1).unsqueeze(0) # 13HW
            mask = mask.unsqueeze(0).unsqueeze(0)
            # save_image(combin_rgb, f"combin_rgb.png")

            # crop a bbox of the masked region (we do not use it)
            if args.is_crop: 
                masked = np.where(masks[img_i] != 0)
                min_x = masked[0].min()
                max_x = masked[0].max()
                min_y = masked[1].min()
                max_y = masked[1].max()
                bb = 0
                combin_rgb_cropped1 = combin_rgb[:, :, min_x-bb:max_x+bb, min_y-bb:max_y+bb]
                # save_image(combin_rgb_cropped1, f"combin_rgb_masked1.png")
                mask_cropped = mask[:, :, min_x-bb:max_x+bb, min_y-bb:max_y+bb]

            # render masked regions--for normal sds
            # Render downsampled for speed
            if args.is_normal_guidance:
                H_r = H // args.normalmap_render_factor
                W_r = W // args.normalmap_render_factor
                focal_r = focal / args.normalmap_render_factor
                K = np.array([
                    [focal_r, 0, W_r / 2],
                    [0, focal_r, H_r / 2],
                    [0, 0, 1]
                ])
                K = torch.Tensor(K)
            
                pose_i = poses[img_i]
                _, _, _, depth1, _ = render(H_r, W_r, focal_r, chunk=args.chunk, c2w=pose_i[:3, :4],
                                                        verbose=i < 10, retraw=True, **render_kwargs_train)
                points = depth2xyz_torch(depth1.reshape(H_r, W_r), K)  
                points_tensor = points.unsqueeze(0).transpose(2,3).transpose(1,2) #1,3,h,w
                normalized_normal_map_tensor = depth2normal_geo(points_tensor) #1,3,h,w
                normalized_normal_map_tensor = (normalized_normal_map_tensor + 1) / 2
            
            # collaborative rendering
            if args.is_colla_guidance:
                rgbs4, _, mask4 = render_path_4view(i, masks, poses, hwf, args.chunk, render_kwargs_test,
                                                    render_factor=args.normalmap_render_factor, need_alpha=True)
                rgbs4_tensor = torch.tensor(rgbs4, dtype=torch.float16, device=device)
                rgbs4_tensor = rgbs4.permute(0, 3, 1, 2)
                mask4_tensor = torch.tensor(mask4, dtype=torch.float16, device=device)
                mask4_tensor = mask4_tensor.unsqueeze(1)
             # collaborative rendering end 

            # render for unmasked pixels (unmasked RGBD supervision).
            rgb2, _, _, _, extras2 = render(H, W, focal, chunk=args.chunk, rays=batch_rays_clf,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

            _, disp2, _, _, _ = render(H, W, focal, chunk=args.chunk, rays=batch_inp,
            verbose=i < 10, retraw=True,
            **render_kwargs_train)
        ############## add guidance loss for all pixels

        if args.first_stage and use_batching: # actually, we do not need this step
            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_clf)
            depth_loss = 0.0
            # only for uninpainted regions
            if args.depth_loss and not args.colmap_depth:
                depth_loss = img2mse(disp, target_inp)
                # print('------depth_loss: ', depth_loss)
            loss = img_loss + args.depth_lambda * depth_loss
        else:
            optimizer.zero_grad()

            # compute the unmasked RGB loss
            img_loss = img2mse(rgb2, target_clf)

            # compute the SDS loss
            if args.is_colla_guidance:
                loss_rgb_sds = pre_model.cal_loss(i, rgbs4_tensor, normalized_normal_map_tensor, None, combin_rgb, None, mask.reshape(1,1,H,W), mask_cropped, mask4_tensor, 1)
            elif args.is_normal_guidance:
                if args.is_crop:
                    loss_rgb_sds = pre_model.cal_loss(i, None, normalized_normal_map_tensor, None, combin_rgb_cropped1, None, mask.reshape(1,1,H,W), mask_cropped, None, 1)
                else:
                    loss_rgb_sds = pre_model.cal_loss(i, None, normalized_normal_map_tensor, None, combin_rgb, None, mask.reshape(1,1,H,W), None, None, 1)
            else:
                if not args.is_crop:
                    loss_rgb_sds = pre_model.cal_loss(i, None, None, None, combin_rgb, None, mask.reshape(1,1,H,W), mask_cropped, None, 1)
                else:
                    loss_rgb_sds = pre_model.cal_loss(i, None, None, None, combin_rgb_cropped1, None, mask.reshape(1,1,H,W), mask_cropped, None, 1)

            # compute the unmasked depth loss
            depth_loss = 0.0
            depth_loss = img2mse(disp2, target_inp)
            # print('------depth_loss: ', depth_loss)

            loss = img_loss + args.depth_lambda * depth_loss

            if 'rgb0' in extras2 and not args.no_coarse:
                img_loss0 = img2mse(extras2['rgb0'], target_clf)
                loss = loss + img_loss0

            loss = loss + args.sds_loss_weight*loss_rgb_sds
            # print('---------------loss: ', loss)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() if render_kwargs_train[
                    'network_fn'] is not None else None,
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train[
                    'network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if args.i_video > 0 and i % args.i_video == 0 and i >= 0:  # todo replace i > 4000 with i > 0
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                                             render_factor=args.render_factor, need_alpha=True)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname,
                                     '{}_lpips_{}_prepare_{}_{:06d}{}{}{}_'.format(
                                         expname,
                                         str(args.lpips),
                                         str(args.prepare),
                                         i,
                                         '_masked_nerf' if args.masked_NeRF else '',
                                         '_N_train_' +
                                         str(args.N_train) if args.N_train is not None else '',
                                         '_no_geo' if args.no_geometry else ''
                                     ))
            if args.train_gt:
                moviebase = os.path.join(basedir, expname,
                                         '{}_gt_images_{:06d}_'.format(
                                             expname,
                                             i,
                                         ))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.nanmax(disps)), fps=30, quality=8)

            with torch.no_grad():
                rgbs, disps, _ = render_path(poses[i_test], hwf, args.chunk, render_kwargs_test,
                                             render_factor=args.render_factor)
            print('Done, saving', rgbs.shape, disps.shape)
            imageio.mimwrite(moviebase + 'test.mp4',
                             to8b(rgbs), fps=30, quality=8)

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()} ")

        global_step += 1

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs2, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs2, [-1, inputs2.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs2.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs2 = torch.reshape(outputs_flat, list(
        inputs2.shape[:-1]) + [outputs_flat.shape[-1]])
    
    return outputs2


def batchify_rays(rays_flat, chunk=1024 * 32, need_alpha=False, detach_weights=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(
            rays_flat[i:i + chunk], need_alpha=need_alpha, detach_weights=detach_weights, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None, depths=None, need_alpha=False, detach_weights=False,
           patch=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        if patch is not None:
            i, j, len1, len2 = patch
            rays_o = rays_o[i:i + len1, j:j + len2, :]
            rays_d = rays_d[i:i + len1, j:j + len2, :]
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # B x 8
    if depths is not None:
        rays = torch.cat([rays, depths.reshape(-1, 1)], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    
    # Render and reshape
    all_ret = batchify_rays(
        rays, chunk, need_alpha=need_alpha, detach_weights=detach_weights, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0,
                disp_require_grad=False, need_alpha=False, rgb_require_grad=False, detach_weights=False,
                patch_len=None, masks=None):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])
    if savedir is not None:
        np.savetxt(
            os.path.join(savedir, 'intrinsics.txt'),
            K
        )

    rgbs = []
    disps = []

    Xs = []
    Ys = []

    for i, c2w in enumerate(render_poses):
        if disp_require_grad or rgb_require_grad:
            if patch_len is not None:
                masked = np.where(masks[i] != 0)
                masked = (masked[0] // render_factor,
                          masked[1] // render_factor)
                
                Xs.append(random.randint(
                    masked[0].min(),
                    max(masked[0].max() - patch_len[0], masked[0].min())
                ))
                Ys.append(random.randint(
                    masked[1].min(),
                    max(masked[1].max() - patch_len[1], masked[1].min())
                ))
                patch = (Xs[-1], Ys[-1], patch_len[0], patch_len[1])
            else:
                patch = None
            rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4],
                                                   retraw=True, need_alpha=need_alpha,
                                                   detach_weights=detach_weights, patch=patch,
                                                   **render_kwargs)
        else:
            with torch.no_grad():
                rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4],
                                                       retraw=True, need_alpha=need_alpha, **render_kwargs)

        if disp_require_grad:
            disps.append(disp)
        else:
            disps.append(disp.detach().cpu().numpy())

        if rgb_require_grad:
            rgbs.append(rgb)
        else:
            rgbs.append(rgb.detach().cpu().numpy())

        if savedir is not None:
            rgb_dir = os.path.join(savedir, 'rgb')
            depth_dir = os.path.join(savedir, 'depth')
            disp_dir = os.path.join(savedir, 'disp')
            weight_dir = os.path.join(savedir, 'weight')
            alpha_dir = os.path.join(savedir, 'alpha')
            gt_img_dir = os.path.join(savedir, 'images')
            z_dir = os.path.join(savedir, 'z')
            pose_dir = os.path.join(savedir, 'pose')
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(gt_img_dir, exist_ok=True)
            os.makedirs(weight_dir, exist_ok=True)
            os.makedirs(z_dir, exist_ok=True)
            os.makedirs(pose_dir, exist_ok=True)
            os.makedirs(disp_dir, exist_ok=True)
            if need_alpha:
                os.makedirs(alpha_dir, exist_ok=True)

            rgb8 = to8b(rgbs[-1])
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(rgb_dir, '{:06d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            if gt_imgs is not None:
                gt_filename = os.path.join(gt_img_dir, '{:06d}.png'.format(i))
                try:
                    imageio.imwrite(gt_filename, to8b(
                        gt_imgs[i].detach().cpu().numpy()))
                except:
                    imageio.imwrite(gt_filename, to8b(gt_imgs[i]))

            depth = depth.cpu().numpy()

            np.save(
                os.path.join(depth_dir, '{:06d}.npy'.format(i)),
                depth
            )
            np.save(
                os.path.join(disp_dir, '{:06d}.npy'.format(i)),
                disp.cpu().numpy()
            )
            np.save(
                os.path.join(weight_dir, '{:06d}.npy'.format(i)),
                extras['weights'].cpu().numpy()
            )
            np.save(
                os.path.join(z_dir, '{:06d}.npy'.format(i)),
                extras['z_vals'].cpu().numpy()
            )
            if need_alpha:
                np.save(
                    os.path.join(alpha_dir, '{:06d}.npy'.format(i)),
                    extras['alpha'].cpu().numpy()
                )

            render_pose = np.concatenate(
                [render_poses[i, :3, :4].detach().cpu().numpy(),
                 np.array([[0, 0, 0, 1]])],
                axis=0
            )
            np.savetxt(
                os.path.join(pose_dir, '{:06d}.txt'.format(i)),
                render_pose
            )

    if disp_require_grad:
        disps = torch.stack(disps, 0)
    else:
        disps = np.stack(disps, 0)

    if rgb_require_grad:
        rgbs = torch.stack(rgbs, 0)
    else:
        rgbs = np.stack(rgbs, 0)

    return rgbs, disps, (Xs, Ys)


def render_path_4view(iter, all_masks, render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0,
                disp_require_grad=False, need_alpha=False, rgb_require_grad=False, detach_weights=False,
                patch_len=None, masks=None):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])
    
    rgbs = []
    disps = []

    Xs = []
    Ys = []

    # Select nearby elements
    neighborhood_size = 4
    iter = iter % 60#149 # 60
    selected_poses = render_poses[max(0, iter - neighborhood_size) : min(len(render_poses), iter + neighborhood_size + 1) : 2]
    selected_masks = all_masks[max(0, iter - neighborhood_size) : min(len(all_masks), iter + neighborhood_size + 1) : 2]

    for i, c2w in enumerate(selected_poses):
        rgb, disp, _, _, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4],
                                                       retraw=True, need_alpha=need_alpha, **render_kwargs)
        disps.append(disp)
        rgbs.append(rgb)
    disps = torch.stack(disps, 0)
    rgbs = torch.stack(rgbs, 0)

    return rgbs, disps, selected_masks

    
def render_path_projection(render_poses, hwf, chunk, render_kwargs, render_factor=0):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])

    z_vals = []
    weights = []
    c2ws = []
    for i, c2w in enumerate(render_poses):
        with torch.no_grad():
            rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4], retraw=True,
                                                   **render_kwargs)

        z_vals.append(extras['z_vals'].cpu().numpy())
        weights.append(extras['weights'].cpu().numpy())
        c2ws.append(convert_pose(np.concatenate(
            [render_poses[i, :3, :4].detach().cpu().numpy(), np.array([[0, 0, 0, 1]])], axis=0
        )))

    return z_vals, weights, c2ws, K


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def render_test_ray(rays_o, rays_d, hwf, ndc, near, far, use_viewdirs, N_samples, network, network_query_fn, **kwargs):
    H, W, focal = hwf
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    z_vals = near * (1. - t_vals) + far * (t_vals)

    z_vals = z_vals.reshape([rays_o.shape[0], N_samples])

    rgb, sigma, depth_maps = sample_sigma(
        rays_o, rays_d, viewdirs, network, z_vals, network_query_fn)

    return rgb, sigma, z_vals, depth_maps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    if args.alpha_model_path is None:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        ##### added
        model = nn.DataParallel(model, device_ids=device_ids)
        #####
        grad_vars = list(model.parameters())
    else:
        alpha_model = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                           input_ch=input_ch, output_ch=output_ch, skips=skips,
                           input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        ##### added
        alpha_model = nn.DataParallel(alpha_model, device_ids=device_ids)
        #####
        print('Alpha model reloading from', args.alpha_model_path)
        ckpt = torch.load(args.alpha_model_path)
        alpha_model.load_state_dict(ckpt['network_fine_state_dict'])
        if not args.no_coarse:
            model = NeRF_RGB(D=args.netdepth, W=args.netwidth,
                             input_ch=input_ch, output_ch=output_ch, skips=skips,
                             input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, alpha_model=alpha_model).to(
                device)
            grad_vars = list(model.parameters())
        else:
            model = None
            grad_vars = []

    model_fine = None
    if args.N_importance > 0:
        if args.alpha_model_path is None:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        else:
            model_fine = NeRF_RGB(D=args.netdepth_fine, W=args.netwidth_fine,
                                  input_ch=input_ch, output_ch=output_ch, skips=skips,
                                  input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                                  alpha_model=alpha_model).to(device)
        grad_vars += list(model_fine.parameters())
        ##### added
        model_fine = nn.DataParallel(model_fine, device_ids=device_ids)
        #####

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(inputs, viewdirs, network_fn,
                                                                           embed_fn=embed_fn,
                                                                           embeddirs_fn=embeddirs_fn,
                                                                           netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        render_kwargs_train['ndc'] = True

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if args.sigma_loss:
        render_kwargs_train['sigma_loss'] = SigmaLoss(
            args.N_samples, args.perturb, args.raw_noise_std)

    ##########################

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def create_nerf_tcnn(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = lambda inp: inp, 3

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = lambda inp: inp, 3
    output_ch = 5 if args.N_importance > 0 else 4
    if args.alpha_model_path is None:
        model = NeRF_TCNN(
            encoding="hashgrid",
        )
        ##### added
        model = nn.DataParallel(model, device_ids=device_ids)
        #####
        grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        if args.alpha_model_path is None:
            model_fine = NeRF_TCNN(
                encoding="hashgrid",
            )
            ###### added
            model_fine = nn.DataParallel(model_fine, device_ids=device_ids)
            ######
        grad_vars += list(model_fine.parameters())

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(inputs, viewdirs, network_fn,
                                                                           embed_fn=embed_fn,
                                                                           embeddirs_fn=embeddirs_fn,
                                                                           netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    # if args.masked_NeRF or args.object_removal:
    #     ckpts = []
    # ckpts = []  # todo remove this line!

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        render_kwargs_train['ndc'] = True

    
    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                pytest=False,
                sigma_loss=None,
                verbose=False,
                need_alpha=False,
                detach_weights=False
                ):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # print('-----near:', near)
    # print('-----far:', far)
    
    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(device)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    if network_fn is not None: # true
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                            white_bkgd, pytest=pytest,
                                                                            need_alpha=need_alpha,
                                                                            detach_weights=detach_weights)
    else:
        if network_fine.alpha_model is not None:
            raw = network_query_fn(pts, viewdirs, network_fine.alpha_model)
            rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                white_bkgd, pytest=pytest,
                                                                                need_alpha=need_alpha,
                                                                                detach_weights=detach_weights)
        else:
            raw = network_query_fn(pts, viewdirs, network_fine)
            rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                white_bkgd, pytest=pytest,
                                                                                need_alpha=need_alpha,
                                                                                detach_weights=detach_weights)

    if N_importance > 0: 
        rgb_map_0, disp_map_0, acc_map_0, alpha0 = rgb_map, disp_map, acc_map, alpha

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                   None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                            white_bkgd, pytest=pytest,
                                                                            need_alpha=need_alpha,
                                                                            detach_weights=detach_weights)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map,
           'weights': weights, 'z_vals': z_vals}
    if retraw:
        ret['raw'] = raw
    if need_alpha:
        ret['alpha'] = alpha
        ret['alpha0'] = alpha0
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    if sigma_loss is not None and ray_batch.shape[-1] > 11:
        depths = ray_batch[:, 8]
        ret['sigma_loss'] = sigma_loss.calculate_loss(rays_o, rays_d, viewdirs, near, far, depths, network_query_fn,
                                                      network_fine)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def estimate_normals(depth_map):
    # Compute gradients along x and y directions
    depth_grad_x = np.gradient(depth_map, axis=1)
    depth_grad_y = np.gradient(depth_map, axis=0)

    # Compute surface normals using the gradients
    normals_x = -depth_grad_x
    normals_y = -depth_grad_y
    normals_z = np.ones_like(depth_map)  # Assume the camera is looking in the z direction

    # Normalize the normals to the range [0, 1]
    normals_x = (normals_x + 1) / 2
    normals_y = (normals_y + 1) / 2
    normals_z = (normals_z + 1) / 2

    # Map the normals to RGB values
    normal_map = np.dstack((normals_x, normals_y, normals_z))

    return normal_map

def depth2xyz(depth_map, depth_cam_matrix, flatten=False, depth_scale=1.0):
    fx, fy = depth_cam_matrix [0,0] , depth_cam_matrix[1,1]
    cx, cy = depth_cam_matrix[0,2], depth_cam_matrix[1,2]
    h,w=np.mgrid [0:depth_map.shape[0],0:depth_map.shape[1]]
    z=depth_map/depth_scale
    x=(w-cx)*z/fx
    y=(h-cy)*z/fy
    xyz=np.dstack((x,y,z)) if(flatten==False) else np.dstack((x,y,z)).reshape(-1,3)
    return xyz.astype(np.float16)

def cal_normal(pcd, knn=30):
    # input: tensor(N,3){, int(knn)}
    # output: tensor(N,3)

    data_np= pcd.detach().cpu().numpy()
    kdtree = sp.cKDTree(data_np)
    _,idxs = kdtree.query(data_np, k = knn)
    neighbors = pcd[idxs,:] # N,knn,3
    average_neighbors = torch.mean(neighbors, axis=1) # N,3
    decentration_matrix = neighbors - average_neighbors.unsqueeze(1) # N,knn,3
    H = torch.matmul(decentration_matrix.transpose(2,1), decentration_matrix)  #N,3,3
    eigenvectors, eigenvalues, eigenvectors_T = torch.linalg.svd(H, full_matrices=False) #(N,3,3 )X3
    normals = eigenvectors[:, 2] #N,3
    return normals

def cal_normal_np(pcd, knn=30):
    # input: array(N,3){, int(knn)}
    # output: array(N,3)

    data_np = pcd
    kdtree = sp.cKDTree(data_np)
    _, idxs = kdtree.query(data_np, k=knn)
    neighbors = pcd[idxs, :]  # N,knn,3
    average_neighbors = np.mean(neighbors, axis=1)  # N,3
    decentration_matrix = neighbors - np.expand_dims(average_neighbors, axis=1)  # N,knn,3
    H = np.matmul(decentration_matrix.transpose(0, 2, 1), decentration_matrix)  # N,3,3
    eigenvectors, _, _ = np.linalg.svd(H)  # (N,3,3 )X3
    normals = eigenvectors[:, :, 2]  # N,3
    return normals

def depth2xyz_torch(depth_map, depth_cam_matrix, depth_scale=1.0): 
    # input: tensor(h,w), tensor(3,3){, float(scale)}
    # output: tensor(h,w,3)
    fx, fy = depth_cam_matrix [0,0] , depth_cam_matrix[1,1]
    cx, cy = depth_cam_matrix[0,2], depth_cam_matrix[1,2]
    h,w=np.mgrid[0:depth_map.shape[0],0:depth_map.shape[1]]  
    h = torch.Tensor(h)
    w = torch.Tensor(w)
    z=depth_map/depth_scale
    x=(w-cx)*z/fx
    y=(h-cy)*z/fy 
    xyz=torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)],axis = -1)

    return xyz

def depth2normal_geo(depth, k = 31):  
    # input: tensor(b,3,h,w){, int(knn)}
    # output: tensor(b,3,h,w)
    B,C,H,W = depth.shape
    point_martx = torch.nn.functional.unfold(depth, (k, k), dilation=1, padding=int((k-1)/2), stride=1) # b,(3*k*k),h*w 
    matrix_a = point_martx.transpose(1,2).reshape(B,H,W,C,k*k).transpose(-1,-2)  #b,h,w,k*k,3
    matrix_a_zero = torch.zeros_like(matrix_a, dtype=torch.float16)

    matrix_a_trans = matrix_a.transpose(-1,-2) #b,h,w,3,k*k
    matrix_b = torch.ones([B, H, W, k * k, 1])

    point_multi = torch.matmul(matrix_a_trans, matrix_a) #b,h,w,3,3
    inv_matrix = torch.linalg.inv(point_multi) #b,h,w,3,3 ###### torch>1.8 -> torch.linalg.inv
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans),matrix_b) #b,h,w,3,1
    normals = generated_norm.squeeze(-1).transpose(2,3).transpose(1,2) #b,3,h,w

    return normals


def gui_application(args, render_kwargs_test):
    root = tk.Tk()
    root.geometry("300x520")

    def set_values():
        args.feat_weight = float(feat.get())
        args.i_video = int(i_video.get())
        args.render_factor = int(render_factor.get())
        if white_bkgd.get() == 1:
            args.white_bkgd = True
        else:
            args.white_bkgd = False
        render_kwargs_test['white_bkgd'] = args.white_bkgd

    tk.Label(root, text="Feature weight").pack()
    feat = tk.Entry(root, textvariable=tk.StringVar(
        root, value=str(args.feat_weight)))
    feat.pack()
    tk.Label(root, text="i_video").pack()
    i_video = tk.Entry(root, textvariable=tk.StringVar(
        root, value=str(args.i_video)))
    i_video.pack()
    tk.Label(root, text="render factor").pack()
    render_factor = tk.Entry(root, textvariable=tk.StringVar(
        root, value=str(args.render_factor)))
    render_factor.pack()

    white_bkgd = tk.IntVar()
    tk.Checkbutton(root, text='White BG', onvalue=1,
                   offvalue=0, variable=white_bkgd).pack()

    tk.Button(root, text='Submit', command=set_values).pack()
    root.mainloop()

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
