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
from gaussian_renderer import render, render_fast, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import plot_cubemap, psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import traceback
from utils.general_utils import colormap  # used repeatedly
from ppisp import PPISP, PPISPConfig
from utils.post_process_utils import apply_ppisp
try:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
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

    view_render_options = ['RGB', 'Alpha', 'Normal', 'Depth', "Base Color", "Refl. Strength", "", "Refl. Color", "RGB raw"]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    densify_until_iteration = opt.densify_until_iter + opt.longer_prop_iter
    # if not opt.disable_normal_propagation:
    normal_prop_until_iter = opt.normal_prop_until_iter + opt.longer_prop_iter
    # if not opt.disable_color_sabotage:
    color_sabotage_until_iter = opt.color_sabotage_until_iter + opt.longer_prop_iter
    
    if opt.use_env_scope:
        center = [float(c) for c in opt.env_scope_center]
        env_scope_center = torch.tensor(center, device='cuda')
        env_scope_radius = opt.env_scope_radius
        refl_mask_loss_weight = 0.4
    def get_outside_msk():
        return None if not opt.use_env_scope else \
            torch.sum((gaussians.get_xyz - env_scope_center[None])**2, dim=-1) > env_scope_radius**2

    gaussians = GaussianModel(dataset.sh_degree, opt.refl_init_value, dataset.cubemap_resol)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    ppisp = None
    if dataset.post_process:
        ppisp_config = PPISPConfig(
            use_controller=True,
            controller_distillation=True,
            controller_activation_ratio=(opt.iterations - 5000) / opt.iterations, 
        )
        ppisp = PPISP(num_cameras=1, num_frames=len(scene.getTrainCameras()), config=ppisp_config).cuda()
        ppisp.train()
        ppisp_optimizers = ppisp.create_optimizers()
        ppisp_schedulers = ppisp.create_schedulers(ppisp_optimizers, opt.iterations)
    scene_frozen = False
    
    if checkpoint:
        ckpt = torch.load(checkpoint)
        if isinstance(ckpt, tuple):
            # backward compatibility with original 3DGS checkpoint format
            model_params, first_iter = ckpt
            gaussians.restore(model_params, opt)
        elif dataset.post_process and ("ppisp" in ckpt):
            first_iter = ckpt["iteration"]
            gaussians.restore(ckpt["gaussians"], opt)
            if "ppisp" in ckpt:
                ppisp.load_state_dict(ckpt["ppisp"])
            if "ppisp_optimizers" in ckpt:
                for opt_idx, state in enumerate(ckpt["ppisp_optimizers"]):
                    if opt_idx < len(ppisp_optimizers):
                        ppisp_optimizers[opt_idx].load_state_dict(state)
            if "ppisp_schedulers" in ckpt:
                for sch_idx, state in enumerate(ckpt["ppisp_schedulers"]):
                    if sch_idx < len(ppisp_schedulers):
                        ppisp_schedulers[sch_idx].load_state_dict(state)
        else:
            raise ValueError("Unrecognized checkpoint format")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_ppisp_loss_for_log = 0.0

    print('Total Iterations: {}'.format(opt.iterations))
    print('Densify until: {}'.format(densify_until_iteration))
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations+1): 
        iter_start.record()

        if dataset.post_process:       
            scene_frozen = ppisp_config.use_controller and (iteration >= ppisp_config.controller_activation_ratio * opt.iterations)

        if not scene_frozen and dataset.post_process:
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
        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background, 
            initial_stage=iteration<opt.init_until_iter, 
            env_scope_center=opt.env_scope_center, 
            env_scope_radius=opt.env_scope_radius)
        rgb_raw, alpha = render_pkg["render"], render_pkg["rend_alpha"]
        env_scope_mask = render_pkg["env_scope_mask"]

        gt_image = viewpoint_cam.original_image.cuda()
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask
        if gt_alpha_mask is not None:
            gt_alpha_mask = gt_alpha_mask.cuda()
            gt_image = gt_image * gt_alpha_mask + (1-gt_alpha_mask) * background[:, None, None]
            rgb_raw = rgb_raw * alpha + (1-alpha) * background[:, None, None]

        # Apply PPISP
        if dataset.post_process:
            image = apply_ppisp(ppisp, rgb_raw, frame_idx=vind)
        else:
            image = rgb_raw

        # Loss
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        # 3DGS original loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if opt.use_env_scope:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            loss += refl_mask_loss_weight * refl_msk_loss

            sh4_ = gaussians.sh4_refl(viewpoint_cam)
            sh4_msk_loss = sh4_[get_outside_msk()].mean()
            loss += refl_mask_loss_weight * sh4_msk_loss

         # regularization
        if not opt.disable_normal_consistentcy_loss:
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            if opt.use_env_scope:
                normal_error = normal_error * env_scope_mask
            normal_loss = opt.lambda_normal * (normal_error).mean()
            loss += normal_loss
        else:
            normal_loss = torch.tensor(0.0)

        # Add PPISP regularization loss to other losses
        if dataset.post_process:
            ppisp_loss = ppisp.get_regularization_loss()
            loss = loss + ppisp_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_ppisp_loss_for_log = 0.4 * ppisp_loss.item() + 0.6 * ema_ppisp_loss_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "PPISP": f"{ema_ppisp_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            loss_report = {
                'l1_loss': Ll1.item(),
                'normal_loss': normal_loss.item(),
                'ppisp_loss': ppisp_loss.item(),
                'total_loss': loss.item()
            }
            bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            training_report(tb_writer, iteration, loss_report, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, ppisp, render, pipe, bg, initial_stage=iteration<=opt.init_until_iter)

            if iteration == densify_until_iteration:
                gaussians.double_env_map()

            if iteration > opt.iterations - 10000:
                gaussians.freeze_xyz()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if (not scene_frozen) and iteration < densify_until_iteration:
                # Keep track of max radii in image-space for pruning
                viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                render_weight = render_pkg["gaussian_weights"]
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_weight)

                if not opt.disable_normal_propagation and (opt.init_until_iter < iteration <= normal_prop_until_iter):
                    densification_interval = opt.densification_interval_when_prop
                else:
                    densification_interval = opt.densification_interval
                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_mean, scene.cameras_extent, size_threshold)
                
                opacity_reset_0 = False
                if iteration % opt.opacity_reset_interval == 0:
                    opacity_reset_0 = True
                    gaussians.reset_opacity()
                
                if opt.opac_lr0_interval > 0 and (iteration-500) % opt.opac_lr0_interval == 0 and (opt.init_until_iter < iteration <= normal_prop_until_iter): ## 200->50
                    gaussians.set_opacity_lr(opt.opacity_lr)

                if (iteration-500) % opt.normal_prop_interval == 0 and (opt.init_until_iter < iteration <= normal_prop_until_iter):
                    if not opacity_reset_0 and not opt.disable_normal_propagation:
                        outside_msk = get_outside_msk()
                        opacity_old = gaussians.get_opacity
                        opac_mask = (opacity_old > 0.9).flatten()
                        if outside_msk is not None:
                            opac_mask = torch.logical_or(opac_mask, outside_msk)
                        gaussians.reset_opacity(reset_value=0.9, exclusive_msk=opac_mask)

                        refl = gaussians.sh4_refl(viewpoint_cam)
                        scale_mask = (refl < 0.02).flatten()
                        if outside_msk is not None:
                            scale_mask = torch.logical_or(scale_mask, outside_msk)
                        gaussians.reset_scale(enlarge_scale=1.5, exclusive_msk=scale_mask)

                        gaussians.reset_sh_refl()
                        
                        if opt.opac_lr0_interval > 0 and iteration != normal_prop_until_iter:
                            gaussians.set_opacity_lr(0.0)

            if (iteration-500) % opt.color_sabotage_interval == 0 and (opt.init_until_iter < iteration <= color_sabotage_until_iter):
                if not opt.disable_color_sabotage:
                    refl = gaussians.sh4_refl(viewpoint_cam)
                    color_mask = (refl > 0.1).flatten()                        
                    outside_msk = get_outside_msk()
                    if outside_msk is not None:
                        color_mask = torch.logical_or(color_mask, outside_msk)
                    gaussians.dist_color_rgb(exclusive_msk=color_mask)

            # Optimizer step
            if iteration < opt.iterations:
                if dataset.post_process:
                    for ppisp_opt in ppisp_optimizers:
                        ppisp_opt.step()
                        ppisp_opt.zero_grad(set_to_none=True)

                    for sched in ppisp_schedulers:
                        sched.step()

                if not scene_frozen:
                    # NOTE: Do NOT step gaussians.exposure_optimizer here.
                    # PPISP replaces the learned exposure path in 3DGS.
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.zero_grad(set_to_none=True)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt = {
                    "gaussians": gaussians.capture(),
                    "iteration": iteration,
                    "ppisp": ppisp.state_dict(),
                    "ppisp_optimizers": [o.state_dict() for o in ppisp_optimizers],
                    "ppisp_schedulers": [s.state_dict() for s in ppisp_schedulers],
                }
                torch.save(ckpt, scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
            if network_gui.conn == None:
                network_gui.try_connect(view_render_options)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                        render_pkg = render_fast(custom_cam, gaussians, pipe, bg, scaling_modifer, initial_stage=False, env_scope_center=opt.env_scope_center, env_scope_radius=opt.env_scope_radius)   
                        rgb_raw = render_pkg["render"]
                        rgb_out = apply_ppisp(ppisp, rgb_raw, frame_idx=-1, clamp=True)
                        net_image = render_net_image(rgb_out, render_pkg, view_render_options, render_mode, custom_cam)
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
def training_report(
    tb_writer, 
    iteration,
    train_loss_report, 
    l1_loss, 
    elapsed,
    testing_iterations,
    scene : Scene, 
    ppisp: PPISP,
    renderFunc, 
    pipe,
    bg,
    initial_stage,
):
    if tb_writer:
        for tag,loss in train_loss_report.items():
            tb_writer.add_scalar('train_loss_patches/{}'.format(tag), loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_histogram("scene/refl_histogram", scene.gaussians.get_refl, iteration)
        textures = torch.sigmoid(scene.gaussians.env_map.params['Cubemap_texture'])
        grid = plot_cubemap(textures)
        tb_writer.add_image("env_cubemap", grid, iteration)

        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(), "use_known_frame_idx": False}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)], "use_known_frame_idx": True})

        with torch.no_grad():        
            for config in validation_configs:
                config_name = config['name']
                l1_test = 0.0
                psnr_test = 0.0
                if config['cameras'] and len(config['cameras']) > 0:
                    for idx, viewpoint in enumerate(config['cameras']):
                        render_pkg = renderFunc(
                            viewpoint, 
                            scene.gaussians, 
                            pipe,
                            bg, 
                            initial_stage=initial_stage)
                        raw_image = render_pkg["render"]
                    
                        gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                        gt_alpha_mask = viewpoint.gt_alpha_mask
                        if gt_alpha_mask is not None:
                            gt_alpha_mask = gt_alpha_mask.cuda()
                            gt_image = gt_image * gt_alpha_mask + (1-gt_alpha_mask) * bg[:, None, None]
                            
                            alpha = render_pkg["rend_alpha"]
                            raw_image = raw_image * alpha + (1-alpha) * bg[:, None, None]

                        if ppisp is not None:
                            frame_idx = idx if config["use_known_frame_idx"] else -1
                            rgb_out = apply_ppisp(ppisp, raw_image, frame_idx=frame_idx, clamp=True)
                        else:
                            rgb_out = raw_image

                        # ---- TensorBoard logging ----
                        if tb_writer and (idx < 5):
                            tb_writer.add_images( f"{config_name}_view_{viewpoint.image_name}/render", rgb_out[None], global_step=iteration)
                            
                            rend_alpha = render_pkg['rend_alpha']
                            rend_alpha = colormap(rend_alpha.detach().cpu().numpy()[0], cmap='turbo')
                            tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/rend_alpha", rend_alpha[None], global_step=iteration)

                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/rend_normal", rend_normal[None], global_step=iteration)

                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)

                            if "base_color_map" in render_pkg:
                                base_color_map = render_pkg["base_color_map"]
                                tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/base_color", base_color_map[None], global_step=iteration)

                            if "refl_strength_map" in render_pkg:
                                refl_map = render_pkg['refl_strength_map']
                                tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/refl_map", refl_map[None], global_step=iteration)

                            if "surf_normal" in render_pkg:
                                surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                                tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/surf_normal", surf_normal[None], global_step=iteration)
        
                            if "surf_depth" in render_pkg:
                                depth = render_pkg["surf_depth"]
                                norm = depth.max()
                                depth = depth / norm
                                depth = colormap(depth.detach().cpu().numpy()[0], cmap='turbo')
                                tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/depth", depth[None], global_step=iteration)

                        # ---- Metrics ----
                        l1_test += l1_loss(rgb_out, gt_image).mean().double()
                        psnr_test += psnr(rgb_out, gt_image).mean().double()


                # ---- Average metrics per config ----
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config_name}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(f"{config_name}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config_name}/loss_viewpoint - psnr", psnr_test, iteration)

    if tb_writer:
        tb_writer.flush()
    if ppisp is not None:
        ppisp.train()
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
    parser.add_argument("--auto_test", action="store_true", help="If set, test iterations will be set to every 5000 iterations.")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--progress_report_iterations", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.auto_test:
        args.test_iterations = [500, 1000, 1500, 3000] + [i for i in range(5000, args.iterations+1, 5000)]

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
