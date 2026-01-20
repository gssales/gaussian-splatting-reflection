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
from gaussian_renderer import render, render_env_map
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

G = {}
G["visible_gaussians"] = torch.empty(0)

def render_set(model_path, name, iteration, views, gaussians: GaussianModel, pipeline, background, render_normals, render_refl, count_visible = True):
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

    visible_gaussians_ = torch.zeros_like(gaussians.get_opacity)

    render_times = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)

        gt_alpha_mask = view.gt_alpha_mask
        if gt_alpha_mask is not None:
            gt = gt * gt_alpha_mask + (1-gt_alpha_mask) * background[:, None, None]
            rendering = rendering * gt_alpha_mask + (1-gt_alpha_mask) * background[:, None, None]

        mask = render_pkg["is_rendered"] == 1
        visible_gaussians_[mask] = 1

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        if render_normals:
            normals = render_pkg["rend_normal"]
            normals = normals*0.5+0.5
            torchvision.utils.save_image(normals, os.path.join(normals_path, '{0:05d}'.format(idx) + ".png"))
            
        if render_refl:
            refl = render_pkg["refl_strength_map"].repeat(3,1,1)
            torchvision.utils.save_image(refl, os.path.join(refl_path, '{0:05d}'.format(idx) + ".png"))

    if count_visible:
        G["visible_gaussians"] = visible_gaussians_

        with open(model_path + "/visible.txt", 'w') as fp:
            fp.write('visible_gaussians:{}/{}'.format(torch.sum(G["visible_gaussians"]), len(G["visible_gaussians"])))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, render_normals : bool, render_refl : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "all", scene.loaded_iter, scene.getTestCameras() + scene.getTrainCameras(), gaussians, pipeline, background, render_normals, render_refl)

        mask = G["visible_gaussians"] == 1
        print(G["visible_gaussians"].shape, torch.sum(G["visible_gaussians"]))
        gaussians._opacity[mask] = -10
        inv_mask = G["visible_gaussians"] == 0
        gaussians._opacity[inv_mask] = 10

        render_set(dataset.model_path, "hidden", scene.loaded_iter, scene.getTestCameras() + scene.getTrainCameras(), gaussians, pipeline, background, render_normals, render_refl, count_visible=False)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--render_normals", action="store_true")
    parser.add_argument("--render_refl", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.render_normals, args.render_refl)