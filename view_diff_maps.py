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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from utils.image_utils import psnr_map, to_3ch, colormap, gradient_map
from utils.mae_utils import angular_error_map, compute_mae
from argparse import ArgumentParser
import traceback

try:
    from fused_ssim import fused_ssim_map
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readNormalsImages(renders_dir, gt_dir):
    render_normals = []
    for fname in os.listdir(renders_dir):
        normal = Image.open(renders_dir / fname)
        render_normals.append(tf.to_tensor(normal).unsqueeze(0)[:, :3, :, :].cuda())

    gt_normals = []
    alphas = []
    for fname in os.listdir(gt_dir):
        if fname.endswith("_normal.png"):
            normal = Image.open(gt_dir / fname)
            gt_normals.append(tf.to_tensor(normal).unsqueeze(0)[:, :3, :, :].cuda())
        if fname.endswith("_alpha.png"):
            alpha = Image.open(gt_dir / fname)
            alpha_tensor = tf.to_tensor(alpha).unsqueeze(0)[:, :1, :, :].cuda()
            # should shape like [1,3,H,W]
            alpha_tensor = alpha_tensor.repeat(1,3,1,1)
            alphas.append(alpha_tensor)


    return render_normals, gt_normals, alphas

def evaluate(model_path, args):
    try:
        print(f"Processing scene: {model_path}")
          
        if not model_path.is_dir():
            print(f"[!] {model_path} is not a directory")
            return

        test_dir = Path(model_path) / "test"
        if not test_dir.exists():
            print(f"[!] No test directory found in {model_path}, skipping.")
            return

        for method in os.listdir(test_dir):
            print(f"  [+] Processing Method: {method}")

            method_dir = test_dir / method

            base_save_dir = method_dir / "diff_maps"
            os.makedirs(base_save_dir, exist_ok=True)

            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            if args.eval_normals:
                normal_renders_dir = method_dir / "normals"
                normal_gts_dir = Path(args.normal_path) / "test"
                normal_renders, normal_gts, alphas = readNormalsImages(normal_renders_dir, normal_gts_dir)

            # normals_mean = 0.0
            # nromals_std = 0.0
            # if args.eval_normals:
            #     for idx in range(len(normal_renders)):
            #         alpha = alphas[idx]
            #         normal_gt = normal_gts[idx]
            #         normal_gt[alpha < 0.01] = 0
            #         normal_gt = (normal_gt-0.5)*2
            #         normal_render = normal_renders[idx]
            #         normal_render[alpha < 0.01] = 0
            #         normal_render = (normal_render-0.5)*2
            #         angular_error = compute_mae(normal_render, normal_gt)
            #         normals_mean += angular_error.mean().item()
            #         nromals_std += angular_error.std().item()
            #     normals_mean /= len(normal_renders)
            #     nromals_std /= len(normal_renders)
            #     print(f"    Normal Angular Error - Mean: {normals_mean}, Std: {nromals_std}")

            tiles = []
            for idx in tqdm(range(len(renders)), desc="Generating tiles"):
                tiles.append(to_3ch(gts[idx]))
                tiles.append(to_3ch(renders[idx]))

                ssim_map_ = fused_ssim_map(renders[idx], gts[idx])
                tiles.append(to_3ch(ssim_map_))

                psnr_map_ = psnr_map(renders[idx], gts[idx])
                max_finite_value = torch.max(psnr_map_[torch.isfinite(psnr_map_)])
                psnr_map_[torch.isinf(psnr_map_)] = max_finite_value #* 2 # Or any other large value
                psnr_map_ = (psnr_map_ - psnr_map_.min()) / (psnr_map_.max() - psnr_map_.min() + 1e-8)
                tiles.append(to_3ch(psnr_map_ / psnr_map_.max()))

                l1_map_ = torch.abs(renders[idx] - gts[idx])
                tiles.append(to_3ch(l1_map_ / l1_map_.max()))

                if args.eval_normals:
                    normal_gt = normal_gts[idx][0]
                    normal_gt = (normal_gt-0.5)*2

                    normal_render = normal_renders[idx][0]
                    normal_render = (normal_render-0.5)*2

                    angular_error = angular_error_map(normal_render, normal_gt)

                    mean_angular_error = angular_error.mean().item()
                    std_angular_error = angular_error.std().item()
                    min_angular_error = angular_error.min().item()
                    max_angular_error = angular_error.max().item()
                    # if len(alphas) > idx:
                    #     alpha = alphas[idx][0,0,:,:]
                    #     print(alpha.shape, angular_error.shape)
                    #     angular_error[alpha < 0.01] = 0

                    #     angular_error_window = angular_error[alpha >= 0.01]
                    #     mean_angular_error = angular_error_window.mean().item()
                    #     std_angular_error = angular_error_window.std().item()
                    #     min_angular_error = angular_error_window.min().item()
                    #     max_angular_error = angular_error_window.max().item()
                    #     print(angular_error_window.min(), angular_error_window.max(), angular_error_window.mean(), angular_error_window.std())
                    
                    #using mean and std to normalize angular error map
                    # angular_error[angular_error > mean_angular_error] = 0.0
                    normalized_angular_error = (angular_error - mean_angular_error) / (std_angular_error + 1e-8)
                    # normalized_angular_error = (angular_error-min_angular_error-std_angular_error/2.0) / (std_angular_error/2.0)
                    normalized_angular_error[normalized_angular_error > 1.0] = 1.0
                    # angular_error = (angular_error - angular_error.min()) / (angular_error.max() - angular_error.min() + 1e-8)
                    tiles.append(to_3ch(normalized_angular_error))

                    # save_image(to_3ch(normalized_angular_error), os.path.join(base_save_dir, f"normal_error_{idx:03d}.png"))
                    # save_image(to_3ch(normal_gt/2 + 0.5), os.path.join(base_save_dir, f"normal_gt_{idx:03d}.png"))
                    # save_image(to_3ch(normal_render/2 + 0.5), os.path.join(base_save_dir, f"normal_render_{idx:03d}.png"))

                tiles = [t for t in tiles if t is not None]
                if len(tiles) > 0:
                    cat = torch.cat(tiles, dim=0)  # (N,3,H,W) – now safe
                    grid = make_grid(
                        cat,
                        nrow=3,  # all attributes in one row for this camera
                        padding=2,
                    )

                    img_name = f"{idx:03d}.png"
                    save_path = os.path.join(base_save_dir, img_name)
                    save_image(grid, save_path)
                tiles = []

            del renders, gts, image_names

    except Exception as e:
        print(f"[!] Unable to compute metrics for model {model_path}")
        print(e)
        traceback.print_exc()

def process_root(root_dir: Path, args):
    """
    Process all scene subdirectories inside root_dir.
    A "scene" is any directory that contains cfg_args.
    """
    print(f"Root directory: {root_dir}")

    if not root_dir.is_dir():
        print(f"[!] {root_dir} is not a directory")
        return

    any_scene = False
    for sub in root_dir.iterdir():
        if sub.is_dir() and (sub / "cfg_args").exists():
            any_scene = True
            evaluate(sub, args)
        elif sub.is_dir() and sub.name not in ["point_cloud","progress","train","test","cubemap"]:
            process_root(sub, args)

    if not any_scene:
        # Also allow running the script directly inside a single scene directory
        if (root_dir / "cfg_args").exists():
            evaluate(root_dir, args)
        else:
            print("[!] No scenes with cfg_args found.")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_path', '-m', required=True, type=str, default="")
    parser.add_argument("--eval_normals", action="store_true")
    parser.add_argument("--normal_path", default="E:\\Research\\data\\shiny_blender\\ball", type=str)
    args = parser.parse_args()
    process_root(Path(args.model_path), args)
