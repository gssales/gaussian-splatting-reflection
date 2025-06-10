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

def training(dataset: ModelParams, opt: OptimizationParams, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):

    view_render_options = ['RGB', 'Alpha', 'Normal', 'Depth', "Base Color", "Refl. Strength", "Normal", "Refl. Color", "Edge", "Curvature", "Mask"]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    total_iterations = opt.iterations + 1
    densify_until_iteration = opt.densify_until_iter
    
    if opt.use_env_scope:
        center = [float(c) for c in opt.env_scope_center]
        env_scope_center = torch.tensor(center, device='cuda')
        env_scope_radius = opt.env_scope_radius
        refl_mask_loss_weight = 0.4

    gaussians = GaussianModel(dataset.sh_degree, opt.refl_init_value)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    print('densify until: {}'.format(densify_until_iteration))
    print('total iter: {}'.format(total_iterations))
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, total_iterations):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # deferred_reflection delays the sh optimization
        if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, initial_stage=iteration<opt.init_until_iter, env_scope_center=opt.env_scope_center, env_scope_radius=opt.env_scope_radius)
        image, viewspace_point_tensor, visibility_filter, radii, alpha = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["rend_alpha"]
        env_scope_mask = render_pkg["env_scope_mask"]

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

        if opt.use_env_scope:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            loss += refl_mask_loss_weight * refl_msk_loss

         # regularization
        if not opt.disable_normal_consistentcy_loss:
            lambda_normal = opt.lambda_normal if iteration > opt.init_until_iter else 0.0
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            if opt.use_env_scope:
                normal_error = normal_error * env_scope_mask
            normal_loss = lambda_normal * (normal_error).mean()
            loss += normal_loss
            normal_loss = normal_loss.item()
        else:
            normal_loss = 0

        if not opt.disable_depth_distortion_loss:
            lambda_dist = opt.lambda_dist if iteration > opt.init_until_iter else 0.0
            rend_dist = render_pkg["rend_dist"]
            if opt.use_env_scope:
                rend_dist = rend_dist * env_scope_mask
            dist_loss = lambda_dist * (rend_dist).mean()
            loss += dist_loss
            dist_loss = dist_loss.item()
        else:
            dist_loss = 0

        # in synthetic scenes, forces gaussian of the same color as the background to be transparent
        if opt.synthetic:
            threshold = 0.99 if dataset.white_background else 0.01
            msk = (image < threshold).all(dim=0)
            alpha_error = torch.zeros_like(alpha, dtype=alpha.dtype, device=alpha.device)
            alpha_error[:,msk] = alpha[:,msk]
            loss += 0.1 * alpha_error.mean()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == total_iterations-1:
                progress_bar.close()

            # Log and save
            loss_report = {
                'l1_loss': Ll1.item(),
                'dist_loss': dist_loss,
                'normal_loss': normal_loss,
                'total_loss': loss.item()
            }
            training_report(tb_writer, iteration, loss_report, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations or iteration == total_iterations-1):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < densify_until_iteration:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if opt.normal_propagation and iteration <= opt.normal_prop_until_iter:
                    densification_interval = opt.densification_interval_when_prop
                else:
                    densification_interval = opt.densification_interval
                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                opacity_reset_0 = False
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    opacity_reset_0 = True
                    gaussians.reset_opacity()
                
                if  opt.opac_lr0_interval > 0 and iteration % opt.opac_lr0_interval == 0 and (opt.init_until_iter < iteration <= opt.normal_prop_until_iter): ## 200->50
                    gaussians.set_opacity_lr(opt.opacity_lr)

                if (iteration-500) % opt.normal_prop_interval == 0 and (opt.init_until_iter < iteration <= opt.normal_prop_until_iter):
                    if not opacity_reset_0 and opt.normal_propagation:
                        outside_msk = get_outside_msk()
                        opacity_old = gaussians.get_opacity
                        opac_mask = (opacity_old > 0.9).flatten()
                        if outside_msk is not None:
                            opac_mask = torch.logical_or(opac_mask, outside_msk)
                        gaussians.reset_opacity(reset_value=0.9, exclusive_msk=opac_mask)

                        refl = gaussians.get_refl
                        scale_mask = (refl < 0.02).flatten()
                        if outside_msk is not None:
                            scale_mask = torch.logical_or(scale_mask, outside_msk)
                        gaussians.reset_scale(enlarge_scale=1.5, exclusive_msk=scale_mask)

                        gaussians.reset_refl()
                        
                        if opt.opac_lr0_interval > 0 and iteration != opt.normal_prop_until_iter:
                            gaussians.set_opacity_lr(0.0)

                    if opt.color_sabotage:
                        color_mask = (refl > 0.1).flatten()                        
                        refl = gaussians.get_refl
                        outside_msk = get_outside_msk()
                        if outside_msk is not None:
                            color_mask = torch.logical_or(color_mask, outside_msk)
                        gaussians.dist_color(exclusive_msk=color_mask)
                    

            # Optimizer step
            if iteration < total_iterations-1:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


            if network_gui.conn == None:
                network_gui.try_connect(view_render_options)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer, initial_stage=False, env_scope_center=opt.env_scope_center, env_scope_radius=opt.env_scope_radius)   
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

@torch.no_grad()
def training_report(tb_writer, iteration, train_loss_report, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        for tag,loss in train_loss_report.items():
            tb_writer.add_scalar('train_loss_patches/{}'.format(tag), loss, iteration)
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
                    render_pkg = renderFunc(viewpoint, scene.gaussians, initial_stage=False, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            refl_map = render_pkg['refl_strength_map']
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/refl_map".format(viewpoint.image_name), refl_map[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

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

    testing_iterations = [i for i in range(5000, args.iterations+1, 5000)]

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), testing_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
