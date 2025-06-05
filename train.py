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
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import traceback
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

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset: ModelParams, opt: OptimizationParams, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    view_render_options = ["RGB", "Depth"]
    if dataset.surfel_splatting:
        view_render_options = ['RGB', 'Alpha', 'Normal', 'Depth']
    if dataset.deferred_reflection:
        view_render_options += ["Base Color", "Refl. Strength", "Normal", "Refl. Color"]

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # if dataset.deferred_reflection and opt.densify_until_iter == 15_000:
    #     opt.densify_until_iter = 30_000
    if dataset.surfel_splatting and opt.opacity_cull == 0.005:
        opt.opacity_cull = 0.05

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    total_iterations = opt.iterations + 1
    densify_until_iteration = opt.densify_until_iter
    if opt.normal_propagation:
        total_iterations += opt.longer_prop_iter
        densify_until_iteration += opt.longer_prop_iter
        normal_propagation_iterations = opt.normal_prop_until_iter + opt.longer_prop_iter
    
    if opt.use_env_scope:
        center = [float(c) for c in opt.env_scope_center]
        env_scope_center = torch.tensor(center, device='cuda')
        env_scope_radius = opt.env_scope_radius
        refl_mask_loss_weight = 0.4

    gaussians = GaussianModel(dataset.sh_degree, dataset.deferred_reflection, dataset.surfel_splatting, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    if opt.normal_propagation:
        print('propagation until: {}'.format(normal_propagation_iterations))
    print('densify until: {}'.format(densify_until_iteration))
    print('total iter: {}'.format(total_iterations))

    progress_bar = tqdm(range(first_iter, total_iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, total_iterations):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if dataset.deferred_reflection:
            # deferred_reflection delays the sh optimization
            if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
                gaussians.oneupSHdegree()
        else:
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, initial_stage=iteration<opt.init_until_iter)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        def get_outside_msk():
            return None if not opt.use_env_scope else \
                torch.sum((gaussians.get_xyz - env_scope_center[None])**2, dim=-1) > env_scope_radius**2

        if opt.use_env_scope and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            loss += refl_mask_loss_weight * refl_msk_loss

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

         # regularization
        if opt.normal_consistency_loss:
            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            rend_normal  = render_pkg['normal_map']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            loss += normal_loss
            normal_loss = normal_loss.item()
        else:
            normal_loss = 0

            
        if opt.depth_distortion_loss:
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
            rend_dist = render_pkg["rend_dist"]
            dist_loss = lambda_dist * (rend_dist).mean()
            loss += dist_loss
            dist_loss = dist_loss.item()
        else:
            dist_loss = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                if dataset.surfel_splatting:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "distort": f"{ema_dist_for_log:.{5}f}",
                        "normal": f"{ema_normal_for_log:.{5}f}",
                        "Points": f"{len(gaussians.get_xyz)}"
                    }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if iteration == total_iterations:
                progress_bar.close()

            # Log and save
            if dataset.surfel_splatting and tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations or iteration == total_iterations-1):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < densify_until_iteration:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if opt.normal_propagation and iteration <= normal_propagation_iterations:
                    densification_interval = opt.densification_interval_when_prop
                    opacity_lr = 0.05
                else:
                    densification_interval = opt.densification_interval
                    opacity_lr = opt.opacity_lr

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, radii)

                if opt.normal_propagation:
                    opacity_reset = False
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        opacity_reset = True
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity_norm_prop0()
                        if dataset.deferred_reflection:
                            gaussians.reset_refl(exclusive_msk=outside_msk)

                    if  opt.opac_lr0_interval > 0 and (opt.init_until_iter < iteration <= normal_propagation_iterations) and iteration % opt.opac_lr0_interval == 0: ## 200->50
                        gaussians.set_opacity_lr(opacity_lr)

                    if  (opt.init_until_iter < iteration <= normal_propagation_iterations) and iteration % 1000 == 0:
                        if not opacity_reset:
                            outside_msk = get_outside_msk()
                            gaussians.reset_opacity_norm_prop1(exclusive_msk=outside_msk)
                            gaussians.reset_scale(exclusive_msk=outside_msk)
                            if opt.opac_lr0_interval > 0 and iteration != normal_propagation_iterations:
                                gaussians.set_opacity_lr(0.0)

                elif iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    gaussians.set_opacity_lr(opacity_lr)
                    if dataset.deferred_reflection:
                        outside_msk = get_outside_msk()
                        gaussians.reset_refl(exclusive_msk=outside_msk)

            if opt.color_sabotage and (opt.init_until_iter < iteration <= opt.color_sabotage_until_iter) and iteration % 1000 == 0:
                outside_msk = get_outside_msk()
                gaussians.dist_color(exclusive_msk=outside_msk)

            # Optimizer step
            if iteration < total_iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(view_render_options)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                        net_image = render_net_image(render_pkg, view_render_options, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "it": iteration,
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    traceback.print_exc()
                    print(e)
                    network_gui.conn = None

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    if 'normal_map' in render_pkg:
                        normal = torch.clamp(render_pkg["normal_map"], 0.0, 1.0)
                        normal = (normal+1)/2
                    if 'refl_strength_map' in render_pkg:
                        refl_str = torch.clamp(render_pkg["refl_strength_map"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if 'normal_map' in render_pkg:
                            tb_writer.add_images(config['name'] + "_view_{}/normal_map".format(viewpoint.image_name), normal[None], global_step=iteration)
                        if 'refl_strength_map' in render_pkg:
                            tb_writer.add_images(config['name'] + "_view_{}/refl_strength_map".format(viewpoint.image_name), refl_str[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
