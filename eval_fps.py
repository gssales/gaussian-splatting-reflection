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
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_fps(dataset : ModelParams, iteration : int, pipeline : PipelineParams, renders_per_view : int = 100):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_times = []
        views = scene.getTestCameras()
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            for i in range(renders_per_view):
                t1 = time.time()
                render(view, gaussians, pipeline, background, initial_stage=False)
                render_time = time.time() - t1
                render_times.append(render_time)
        with open(dataset.model_path + "/fps.txt", 'w') as fp:
            fps = 1.0/np.array(render_times).mean()
            fp.write('fps:{}\n'.format(fps))
            fp.write('count:{}\n'.format(len(gaussians.get_xyz)))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Measuring FPS for " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_fps(model.extract(args), args.iteration, pipeline.extract(args))