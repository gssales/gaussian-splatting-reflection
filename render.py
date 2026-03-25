#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
from ppisp import PPISP
from gaussian_renderer import render, render_env_map
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians: GaussianModel, ppisp: PPISP, pipeline, background, render_normals, render_refl):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if render_normals:
        normals_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
        makedirs(normals_path, exist_ok=True)
        
    if render_refl:
        refl_path = os.path.join(model_path, name, "ours_{}".format(iteration), "refl")
        makedirs(refl_path, exist_ok=True)

    # save env light
    if gaussians.env_map != None:
        ltres = render_env_map(gaussians)
        torchvision.utils.save_image(ltres['env_cood1'], os.path.join(model_path, 'light1_{}.png'.format(iteration)))
        torchvision.utils.save_image(ltres['env_cood2'], os.path.join(model_path, 'light2_{}.png'.format(iteration)))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rgb_raw = torch.clamp(render_pkg["render"], 0.0, 1.0)

        gt = view.original_image[0:3, :, :]
        gt_alpha_mask = view.gt_alpha_mask
        if gt_alpha_mask is not None:
            gt = gt * gt_alpha_mask + (1-gt_alpha_mask) * background[:, None, None]
            alpha = torch.clamp(render_pkg["alpha"], 0.0, 1.0)
            rgb_raw = rgb_raw * alpha + (1-alpha) * background[:, None, None]
        
        frame_idx = idx if name == "train" else -1
        rgb_out = apply_ppisp(ppisp, rgb_raw, frame_idx=frame_idx, clamp=True)

        torchvision.utils.save_image(rgb_out, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if render_normals:
            normals = render_pkg["rend_normal"]
            normals = normals*0.5+0.5
            torchvision.utils.save_image(normals, os.path.join(normals_path, '{0:05d}'.format(idx) + ".png"))
            
        if render_refl:
            refl = render_pkg["refl_strength_map"].repeat(3,1,1)
            torchvision.utils.save_image(refl, os.path.join(refl_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_normals : bool, render_refl : bool, checkpoint=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        ppisp = PPISP(num_cameras=len(scene.getTrainCameras()), num_frames=len(scene.getTrainCameras()))

        if checkpoint:
            opt = OptimizationParams()
            ckpt = torch.load(checkpoint)
            if isinstance(ckpt, tuple):
                model_params, _ = ckpt
                gaussians.restore(model_params, opt)
            else:
                gaussians.restore(ckpt["gaussians"], opt)
                if "ppisp" in ckpt:
                    ppisp.load_state_dict(ckpt["ppisp"])

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, ppisp, pipeline, background, render_normals, render_refl)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, ppisp, pipeline, background, render_normals, render_refl)

_PIXEL_COORD_CACHE = {}
def get_pixel_coords(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Returns pixel coordinates with shape [H, W, 2] in (x, y) order as float32.
    Cached per (H, W, device).
    """
    key = (height, width, str(device))
    if key not in _PIXEL_COORD_CACHE:
        ys = torch.arange(height, device=device, dtype=torch.float32)
        xs = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).contiguous()  # [H, W, 2]
        _PIXEL_COORD_CACHE[key] = coords
    return _PIXEL_COORD_CACHE[key]

def apply_ppisp(ppisp, rgb_raw_chw, frame_idx, clamp=False):
    """
    rgb_raw_chw: [3, H, W]
    returns rgb_out_chw: [3, H, W]
    """
    _, H, W = rgb_raw_chw.shape
    pixel_coords = get_pixel_coords(H, W, rgb_raw_chw.device)
    camera_idx = 0

    rgb_raw_hwc = rgb_raw_chw.permute(1, 2, 0).contiguous()
    rgb_out_hwc = ppisp(
        rgb_raw_hwc,
        pixel_coords,
        resolution=(W, H),
        camera_idx=camera_idx,
        frame_idx=frame_idx,
    )
    rgb_out_chw = rgb_out_hwc.permute(2, 0, 1).contiguous()

    if clamp:
        rgb_out_chw = torch.clamp(rgb_out_chw, 0.0, 1.0)
    return rgb_out_chw

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_normals", action="store_true")
    parser.add_argument("--render_refl", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.render_normals, args.render_refl, args.start_checkpoint)
