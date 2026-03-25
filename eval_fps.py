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
import time
import numpy as np
from tqdm import tqdm
from ppisp import PPISP
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_fps(dataset : ModelParams, iteration : int, pipeline : PipelineParams, renders_per_view : int = 100, checkpoint=None):
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

        render_times = []
        views = scene.getTestCameras()
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            for i in range(renders_per_view):
                t1 = time.time()
                res = render(view, gaussians, pipeline, background, initial_stage=False)
                rgb_raw = res["render"]
                apply_ppisp(ppisp, rgb_raw, frame_idx=-1, clamp=True)
                render_time = time.time() - t1
                render_times.append(render_time)
        with open(dataset.model_path + "/fps.txt", 'w') as fp:
            fps = 1.0/np.array(render_times).mean()
            fp.write('fps:{}\n'.format(fps))
            fp.write('count:{}\n'.format(len(gaussians.get_xyz)))

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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = get_combined_args(parser)
    print("Measuring FPS for " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_fps(model.extract(args), args.iteration, pipeline.extract(args), checkpoint=args.start_checkpoint)