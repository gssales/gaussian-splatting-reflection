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
import os
import numpy as np
import torch
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, get_combined_args
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def report(dataset: ModelParams, opt: OptimizationParams, iteration: int):

    gaussians = GaussianModel(dataset.sh_degree, opt.refl_init_value, dataset.cubemap_resol)
    gaussians.load_ply(os.path.join(dataset.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(iteration),
                                                           "point_cloud.ply"))
    # scene = Scene(dataset, gaussians, load_iteration=iteration)

    print(len(gaussians.get_xyz), "gaussians loaded for reporting.")

    #calculate distances of all gaussians to the center
    distances = torch.norm(gaussians.get_xyz, dim=1)
    scales = torch.max(gaussians.get_scaling, dim=1).values
    print("Distance to center: min {}, max {}, mean {}".format(distances.min().item(), distances.max().item(), distances.mean().item()))
    print("Scale: min {}, max {}, mean {}".format(scales.min().item(), scales.max().item(), scales.mean().item()))
    print("Opacity: min {}, max {}, mean {}".format(gaussians.get_opacity.min().item(), gaussians.get_opacity.max().item(), gaussians.get_opacity.mean().item()))

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

    plt.figure(figsize=(10, 10))
    # Plot
    plt.subplot(2, 2, 1)
    plt.plot(bin_centers, bin_size_mean)
    plt.xlabel("Distance to origin")
    plt.ylabel("Max Gaussian size")
    plt.title("Gaussian Size vs Radius")
    plt.grid()

    bin_centers = np.array(bin_centers)
    bin_counts = np.array(bin_counts)
    bin_size_mean = np.array(bin_size_mean)
    marker_sizes = (200 * (bin_size_mean / bin_size_mean.max()))  # Scale marker sizes for better visibility
    plt.subplot(2, 2, 2)
    plt.scatter(bin_centers, bin_counts, s=marker_sizes, alpha=0.5)
    plt.xlabel("Distance to origin")
    plt.ylabel("Gaussian count")
    plt.yscale("log")
    plt.title("Count vs distance (marker size = max Gaussian size)")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.hist2d(dist, sizes, bins=100, norm=matplotlib.colors.LogNorm())
    plt.xlabel("Distance to origin")
    plt.ylabel("Gaussian size")
    plt.title("Count of Gaussians (distance vs size)")
    plt.colorbar(label="Count")
    plt.grid()
    
    plt.subplot(2, 2, 4)
    plt.title("Distance to center histogram")
    plt.hist(distances.cpu().detach().numpy(), bins=50, log=True)
    plt.xlabel("Distance to Origin")
    plt.ylabel("Count")
    plt.grid()
    
    plt.savefig(args.model_path + "/ball_distances_{}.png".format(iteration))


    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Opacity histogram")
    opacity_values = gaussians.get_opacity.cpu().detach().numpy()
    plt.hist(opacity_values, bins=50, range=(0.0, 1.0), log=True)
    plt.xlabel("Opacity")
    plt.ylabel("Count")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.title("Refl histogram")
    refl_values = gaussians.get_refl.cpu().detach().numpy()
    plt.hist(refl_values, bins=50, range=(0.0, 1.0), log=True)
    plt.xlabel("Reflection Strength")
    plt.ylabel("Count")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.title("Max Scale histogram")
    scaling_values = gaussians.get_scaling
    max_scale = torch.max(scaling_values, dim=1).values.cpu().detach().numpy()
    min_scale = torch.min(scaling_values, dim=1).values.cpu().detach().numpy()
    plt.hist(max_scale, bins=50, range=(0.0, max_scale.max()), log=True)
    plt.xlabel("Scale")
    plt.ylabel("Count")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.title("Min Scale histogram")
    plt.hist(min_scale, bins=50, range=(0.0, min_scale.max()), log=True)
    plt.xlabel("Scale")
    plt.ylabel("Count")
    plt.grid()
    
    #save the figure as png
    plt.savefig(args.model_path + "/opacity_histogram_{}.png".format(iteration))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    
    print("Tensorboard reporting " + args.model_path)
        
    report(lp.extract(args), op.extract(args), args.iteration)

    # All done
    print("\nReport Complete.")
