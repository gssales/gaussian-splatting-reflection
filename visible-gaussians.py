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

from matplotlib import pyplot as plt
import matplotlib
import torch
from scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

marked_gaussians = torch.empty(0)

def render_set(model_path, name, iteration, views, gaussians: GaussianModel, pipeline, background, count_visible = True):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    marked_gaussians = torch.zeros_like(gaussians.get_opacity).bool()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)

        mask = render_pkg["gaussian_weights"] > 0.0
        marked_gaussians = torch.logical_or(marked_gaussians, mask)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    if count_visible:
        with open(model_path + "/visible.txt", 'w') as fp:
            fp.write('visible_gaussians:{}/{}'.format(torch.sum(marked_gaussians), len(marked_gaussians)))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, opt: OptimizationParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.training_setup(opt)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "all", scene.loaded_iter, scene.getTestCameras() + scene.getTrainCameras(), gaussians, pipeline, background)

        mask = marked_gaussians.squeeze(-1).cpu()
        
        #calculate distances of all gaussians to the center
        visible_gaussians_xyz = gaussians.get_xyz
        distances = torch.norm(visible_gaussians_xyz, dim=1)
        scales = torch.max(gaussians.get_scaling, dim=1).values
        print("Distance to center: min {}, max {}, mean {}".format(distances.min().item(), distances.max().item(), distances.mean().item()))
        print("Scale: min {}, max {}, mean {}".format(scales.min().item(), scales.max().item(), scales.mean().item()))

        # Convert to numpy
        dist = distances.detach().cpu().numpy()
        sizes = scales.detach().cpu().numpy()

        # Bin
        num_bins = 100
        bins = np.linspace(dist.min(), dist.max(), num_bins)
        bin_indices = np.digitize(dist, bins)

        bin_centers = []
        bin_counts = []
        bin_size_mean = []

        for b in range(1, len(bins)):
            mask = bin_indices == b
            if np.any(mask):
                bin_centers.append((bins[b] + bins[b-1]) / 2)
                bin_counts.append(mask.sum())
                bin_size_mean.append(sizes[mask].mean())

        # Plot
        plt.figure(label="ours")
        plt.plot(bin_centers, bin_size_mean)
        plt.xlabel("Distance to origin")
        plt.ylabel("Average Gaussian size")
        plt.title("Gaussian Size vs Radius")
        plt.show(block=False)

        bin_centers = np.array(bin_centers)
        bin_counts = np.array(bin_counts)
        bin_size_mean = np.array(bin_size_mean)
        marker_sizes = (200 * (bin_size_mean / bin_size_mean.max()))  # Scale marker sizes for better visibility
        plt.figure(label="ours", figsize=(6, 5))
        plt.scatter(bin_centers, bin_counts, s=marker_sizes, alpha=0.2)
        plt.xlabel("Distance to origin")
        plt.ylabel("Gaussian count")
        plt.yscale("log")
        plt.title("Count vs distance (marker size = avg Gaussian size)")
        plt.show(block=False)

        plt.figure(label="ours")
        plt.hist2d(dist, sizes, bins=100, norm=matplotlib.colors.LogNorm())
        plt.xlabel("Distance to origin")
        plt.ylabel("Gaussian size")
        plt.title("Count of Gaussians (distance vs size)")
        plt.colorbar(label="Count")
        plt.show()
        
        gaussians.max_radii2D = torch.zeros_like(gaussians._opacity)
        gaussians.prune_points_no_grad(~mask)
        scene.save(iteration=9999999999)

        render_set(dataset.model_path, "hidden", scene.loaded_iter, scene.getTestCameras() + scene.getTrainCameras(), gaussians, pipeline, background, count_visible=False)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), opt.extract(args))
