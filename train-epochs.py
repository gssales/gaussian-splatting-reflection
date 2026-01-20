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
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import traceback
import torchvision
from torchvision.utils import make_grid, save_image
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def training(dataset: ModelParams, opt: OptimizationParams, pipe, saving_epochs):

    view_render_options = ['RGB', 'Alpha', 'Normal', 'Depth', "Base Color", "Refl. Strength", "Normal", "Refl. Color", "Edge", "Curvature", "Mask"]

    first_epoch = 0
    tb_writer = prepare_output_and_logger(dataset)

    total_epochs = opt.iterations
    
    if opt.use_env_scope:
        center = [float(c) for c in opt.env_scope_center]
        env_scope_center = torch.tensor(center, device='cuda')
        env_scope_radius = opt.env_scope_radius
        refl_mask_loss_weight = 0.4

    gaussians = GaussianModel(dataset.sh_degree, opt.refl_init_value, dataset.cubemap_resol)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    epoch_start = torch.cuda.Event(enable_timing = True)
    epoch_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_epoch, total_epochs-1), desc="Training progress")
    first_epoch += 1
    for epoch in range(first_epoch, total_epochs):

        epoch_start.record()

        gaussians.update_learning_rate(epoch)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # deferred_reflection delays the sh optimization
        # if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))

        loss = 0.0
        means2D_grad = []
        while viewpoint_stack:
            # Pick a random Camera
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
	
            # Render
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, initial_stage=True, env_scope_center=opt.env_scope_center, env_scope_radius=opt.env_scope_radius)
            image, viewspace_point_tensor, visibility_filter, radii, alpha = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["rend_alpha"]
            env_scope_mask = render_pkg["env_scope_mask"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask

            image = image * alpha + (1-alpha) * background[:, None, None]
            if gt_alpha_mask is not None:
                gt_alpha_mask = gt_alpha_mask.cuda()
                gt_image = gt_image * gt_alpha_mask + (1-gt_alpha_mask) * background[:, None, None]

                Ll1 = l1_loss(image, gt_image)
                if FUSED_SSIM_AVAILABLE:
                    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                else:
                    ssim_value = ssim(image, gt_image)
                loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            def get_outside_msk():
                return None if not opt.use_env_scope else \
                    torch.sum((gaussians.get_xyz - env_scope_center[None])**2, dim=-1) > env_scope_radius**2
                    
            if opt.use_env_scope:
                refls = gaussians.get_refl
                refl_msk_loss = refls[get_outside_msk()].mean()
                loss += refl_mask_loss_weight * refl_msk_loss

            
            # Keep track of max radii in image-space for pruning
            # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            # means2D_grad.append(viewspace_point_tensor)
            # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)


        loss.backward()

        epoch_end.record()
        
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        training_report(epoch, scene, render, (pipe, bg), dataset.model_path)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            loss_dict = {
                "Loss": f"{ema_loss_for_log:.{5}f}",
                "Points": f"{len(gaussians.get_xyz)}"
            }
            progress_bar.set_postfix(loss_dict)
            progress_bar.update(1)
            if epoch == total_epochs-1:
                progress_bar.close()

            bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            if epoch % 100 == 0:
                os.makedirs(os.path.join(dataset.model_path, 'cubemap'), exist_ok = True)
                for i in range(6):
                    torchvision.utils.save_image(torch.sigmoid(gaussians.env_map.params['Cubemap_texture'][i]), os.path.join(dataset.model_path, 'cubemap/{}_{}.png'.format(i, epoch)))

            # if epoch == 100:
            #     gaussians.double_env_map()

            # if epoch > 200:
            #     gaussians.freeze_xyz()

            if (epoch in saving_epochs or epoch == total_epochs-1):
                print("\n[EPOCH {}] Saving Gaussians".format(epoch))
                scene.save(epoch)

            # Densification
            # if epoch < 10000:
            #     for viewspace_point_tensor in means2D_grad:
            #         gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
            #     # if opt.normal_propagation and (opt.init_until_iter < iteration <= normal_prop_until_iter):
            #     #     densification_interval = opt.densification_interval_when_prop
            #     # else:
            #     #     densification_interval = opt.densification_interval
            #     # if epoch > opt.densify_from_iter and epoch % densification_interval == 0:
            #     size_threshold = 20 #if epoch > opt.opacity_reset_interval else None
            #     gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                

            # Optimizer step
            if epoch < total_epochs-1:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(iteration, scene : Scene, renderFunc, renderArgs,
    model_path,  # <-- NEW: to get dataset.model_path
):

    # Report test and samples of training set
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    from utils.image_utils import colormap as img_colormap  # different colormap function

    def to_3ch(t: torch.Tensor) -> torch.Tensor:
        """
        Normalize various image-like tensor shapes to (B, 3, H, W) on CPU.
        Accepted inputs:
        (H, W)
        (C, H, W)
        (H, W, C)
        (B, C, H, W)
        (B, H, W, C)
        (1, H, W)
        """
        if t is None:
            return None

        t = t.detach().cpu()

        if t.dim() == 2:
            # (H, W)
            t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        elif t.dim() == 3:
            # Could be (C,H,W) or (H,W,C)
            if t.shape[0] in (1, 3):
                # (C,H,W)
                t = t.unsqueeze(0)  # (1,C,H,W)
            elif t.shape[2] in (1, 3):
                # (H,W,C)
                t = t.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
            else:
                # Ambiguous, treat as single-channel
                t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        elif t.dim() == 4:
            # Could be (B,C,H,W) or (B,H,W,C)
            if t.shape[1] in (1, 3):
                # already (B,C,H,W)
                pass
            elif t.shape[-1] in (1, 3):
                # (B,H,W,C)
                t = t.permute(0, 3, 1, 2)  # (B,C,H,W)
            else:
                # Ambiguous, assume second dim is channel
                # but keep shape, we'll just repeat later if needed
                pass
        else:
            raise ValueError(f"Unsupported tensor dim {t.dim()} for image-like data")

        # Ensure 3 channels
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        elif t.shape[1] != 3:
            # In weird cases, just squeeze to one channel and repeat
            t = t.mean(dim=1, keepdim=True)  # (B,1,H,W)
            t = t.repeat(1, 3, 1, 1)

        return t


    for config in validation_configs:
        config_name = config['name']
        if config['cameras'] and len(config['cameras']) > 0:

            # base folder: dataset.model_path/progress/{train|test}
            base_save_dir = os.path.join(model_path, 'progress', config_name)
            os.makedirs(base_save_dir, exist_ok=True)

            for idx, viewpoint in enumerate(config['cameras']):
                render_pkg = renderFunc(viewpoint, scene.gaussians, initial_stage=False, *renderArgs)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                base_color = torch.clamp(render_pkg["base_color_map"], 0.0, 1.0).to("cuda")
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                # ---- Save grid per camera (by attributes) ----
                if base_save_dir is not None:
                    tiles = []

                    # 1) Ground truth (B,C,H,W)
                    tiles.append(to_3ch(gt_image))

                    # 2) Render (B,C,H,W)
                    tiles.append(to_3ch(image))
                    tiles.append(to_3ch(base_color))

                    # 3) Depth (colored)
                    try:
                        depth = render_pkg["surf_depth"]          # possible shapes: (1,1,H,W) or (1,H,W)
                        norm = depth.max().clamp(min=1e-8)
                        depth_norm = depth / norm                 # same shape as depth

                        # Convert to (H,W) for colormap
                        # depth_np = depth_norm.squeeze().cpu().numpy()  # (H,W)
                        depth_color = img_colormap(depth_norm, cmap='turbo') # (H,W,3)
                        # depth_t = torch.from_numpy(depth_color).float()  # (H,W,3)
                        tiles.append(to_3ch(depth_color))
                    except Exception as e:
                        print(e)
                        traceback.print_exc()

                    # 4) Other attributes if present
                    try:
                        if "rend_normal" in render_pkg:
                            rn = render_pkg["rend_normal"] * 0.5 + 0.5   # usually (1,3,H,W)
                            tiles.append(to_3ch(rn))

                        if "surf_normal" in render_pkg:
                            sn = render_pkg["surf_normal"] * 0.5 + 0.5
                            tiles.append(to_3ch(sn))

                        if "refl_strength_map" in render_pkg:
                            rm = render_pkg["refl_strength_map"]         # maybe (1,1,H,W) or (1,H,W)
                            tiles.append(to_3ch(rm))

                        # if "rend_dist" in render_pkg:
                        #     rd = render_pkg["rend_dist"]                 # (1,1,H,W) or (1,H,W)
                        #     # rd_np = rd.squeeze().cpu().numpy()           # (H,W)
                        #     rd_color = img_colormap(rd)                   # (H,W,3)
                        #     tiles.append(to_3ch(rd_color))
                    except Exception:
                        pass

                    tiles = [t for t in tiles if t is not None]
                    if len(tiles) > 0:
                        cat = torch.cat(tiles, dim=0)  # (N,3,H,W) – now safe
                        grid = make_grid(
                            cat,
                            nrow=3,  # all attributes in one row for this camera
                            padding=2,
                        )

                        img_name = f"{config_name}_{idx:03d}_{iteration}.png"
                        save_path = os.path.join(base_save_dir, img_name)
                        save_image(grid, save_path)

    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_epochs.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_epochs)

    # All done
    print("\nTraining complete.")
