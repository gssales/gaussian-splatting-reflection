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
import torch
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, get_combined_args
import matplotlib.pyplot as plt

def report(dataset: ModelParams, opt: OptimizationParams, iteration: int):

    gaussians = GaussianModel(dataset.sh_degree, opt.refl_init_value, dataset.cubemap_resol)
    gaussians.load_ply(os.path.join(dataset.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(iteration),
                                                           "point_cloud.ply"))
    # scene = Scene(dataset, gaussians, load_iteration=iteration)

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
