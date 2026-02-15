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
from utils.general_utils import colormap  # used repeatedly
from utils.image_utils import colormap as img_colormap  # different colormap function
from utils.image_utils import to_3ch
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

def training(dataset: ModelParams, opt: OptimizationParams, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, progress_iterations):

    view_render_options = ['RGB', 'Alpha', 'Normal', 'Depth', "Base Color", "Refl. Strength", "Normal", "Refl. Color", "Edge", "Curvature", "Mask"]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    densify_until_iteration = opt.densify_until_iter + opt.longer_prop_iter
    if not opt.disable_normal_propagation:
        normal_prop_until_iter = opt.normal_prop_until_iter + opt.longer_prop_iter
    if not opt.disable_color_sabotage:
        color_sabotage_until_iter = opt.color_sabotage_until_iter + opt.longer_prop_iter
    
    if opt.use_env_scope:
        center = [float(c) for c in opt.env_scope_center]
        env_scope_center = torch.tensor(center, device='cuda')
        env_scope_radius = opt.env_scope_radius
        refl_mask_loss_weight = 0.4

    gaussians = GaussianModel(dataset.sh_degree, opt.refl_init_value, dataset.cubemap_resol)
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

    print('Total Iterations: {}'.format(opt.iterations))
    print('Densify until: {}'.format(densify_until_iteration))
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations+1):

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

        if opt.random_background_color:
            background = torch.rand(3, dtype=torch.float32, device="cuda")

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, initial_stage=iteration<opt.init_until_iter, env_scope_center=opt.env_scope_center, env_scope_radius=opt.env_scope_radius)
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
            lambda_normal = opt.lambda_normal if opt.init_until_iter < iteration else 0.0
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
            lambda_dist = opt.lambda_dist if opt.init_until_iter < iteration else 0.0
            rend_dist = render_pkg["rend_dist"]
            if opt.use_env_scope:
                rend_dist = rend_dist * env_scope_mask
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
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    # "distort": f"{ema_dist_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            loss_report = {
                'l1_loss': Ll1.item(),
                'dist_loss': dist_loss,
                'normal_loss': normal_loss,
                'total_loss': loss.item()
            }
            bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            training_report(tb_writer, iteration, loss_report, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, bg), bg, initial_stage=iteration<=opt.init_until_iter)
            progress_report(iteration, progress_iterations, scene, render, (pipe, bg), bg, initial_stage=iteration<=opt.init_until_iter, model_path=dataset.model_path)

            if iteration % 10000 == 0:
                os.makedirs(os.path.join(dataset.model_path, 'cubemap'), exist_ok = True)
                for i in range(6):
                    torchvision.utils.save_image(torch.sigmoid(gaussians.env_map.params['Cubemap_texture'][i]), os.path.join(dataset.model_path, 'cubemap/{}_{}.png'.format(i, iteration)))

            if iteration == densify_until_iteration or iteration == 45000:
                gaussians.double_env_map()

            if iteration > densify_until_iteration*2:
                gaussians.freeze_xyz()

            # if total_iterations > iteration >= densify_until_iteration and iteration % 5000 == 0:
            #     gaussians.filter_env_map()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < densify_until_iteration:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if not opt.disable_normal_propagation and (opt.init_until_iter < iteration <= normal_prop_until_iter):
                    densification_interval = opt.densification_interval_when_prop
                else:
                    densification_interval = opt.densification_interval
                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                opacity_reset_0 = False
                if iteration % opt.opacity_reset_interval == 0:
                    opacity_reset_0 = True
                    gaussians.reset_opacity()
                
            if opt.opac_lr0_interval > 0 and (iteration-500) % opt.opac_lr0_interval == 0 and (opt.init_until_iter < iteration <= normal_prop_until_iter): ## 200->50
                gaussians.set_opacity_lr(opt.opacity_lr)

            if (iteration-500) % opt.normal_prop_interval == 0 and (opt.init_until_iter < iteration <= normal_prop_until_iter):
                if not opacity_reset_0 and opt.disable_normal_propagation:
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
                    
                    if opt.opac_lr0_interval > 0 and iteration != normal_prop_until_iter:
                        gaussians.set_opacity_lr(0.0)

            if (iteration-500) % opt.color_sabotage_interval == 0 and (opt.init_until_iter < iteration <= color_sabotage_until_iter):
                if not opt.disable_color_sabotage:
                    refl = gaussians.get_refl
                    color_mask = (refl > 0.1).flatten()                        
                    outside_msk = get_outside_msk()
                    if outside_msk is not None:
                        color_mask = torch.logical_or(color_mask, outside_msk)
                    gaussians.dist_color(exclusive_msk=color_mask)

            # Optimizer step
            if iteration < opt.iterations:
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
                        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                        render_pkg = render(custom_cam, gaussians, pipe, bg, scaling_modifer, initial_stage=False, env_scope_center=opt.env_scope_center, env_scope_radius=opt.env_scope_radius)   
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
def progress_report(
    iteration,
    progress_iterations, 
    scene : Scene, 
    renderFunc, 
    renderArgs,
    bg,
    initial_stage,
    model_path,
):
    if iteration in progress_iterations:
        torch.cuda.empty_cache()
        test_cam_interval = 5 if len(scene.getTestCameras()) > 30 else 1
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(test_cam_interval-1, len(scene.getTestCameras()), test_cam_interval)]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(4, len(scene.getTrainCameras()), 5)]})

        torch.cuda.empty_cache()
        for config in validation_configs:
            config_name = config['name']
            if config['cameras'] and len(config['cameras']) > 0:

                # base folder: dataset.model_path/progress/{train|test}
                base_save_dir = os.path.join(model_path, 'progress', config_name)
                os.makedirs(base_save_dir, exist_ok=True)
                
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, initial_stage=initial_stage)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    alpha = torch.clamp(render_pkg["rend_alpha"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_alpha_mask = torch.clamp(viewpoint.gt_alpha_mask.to("cuda"), 0.0, 1.0)

                    image = image * alpha + (1-alpha) * bg[:, None, None]
                    if gt_alpha_mask is not None:
                        gt_alpha_mask = gt_alpha_mask.cuda()
                        gt_image = gt_image * gt_alpha_mask + (1-gt_alpha_mask) * bg[:, None, None]

                    # ---- Save grid per camera (by attributes) ----
                    if base_save_dir is not None:
                        tiles = []

                        tiles.append(to_3ch(gt_image))
                        tiles.append(to_3ch(image))

                        try:
                            if "base_color_map" in render_pkg:
                                base_color = torch.clamp(render_pkg["base_color_map"], 0.0, 1.0).to("cuda")   # usually (1,3,H,W)
                                tiles.append(to_3ch(base_color))
                        except Exception:
                            pass

                        try:
                            depth = render_pkg["surf_depth"]     
                            norm = depth.max().clamp(min=1e-8)
                            depth_norm = depth / norm   

                            # Convert to (H,W) for colormap
                            depth_color = img_colormap(depth_norm, cmap='turbo') 
                            tiles.append(to_3ch(depth_color))
                        except Exception as e:
                            print(e)
                            traceback.print_exc()

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
                        except Exception:
                            pass

                        # avoid saving very large images/grids
                        if gt_image.shape[2] > 512 or gt_image.shape[1] > 512:
                            large_dim = max(gt_image.shape[1], gt_image.shape[2])
                            scale_factor = 512 / large_dim
                            dimens = (int(gt_image.shape[1] * scale_factor), int(gt_image.shape[2] * scale_factor))
                            for i in range(len(tiles)):
                                tiles[i] = torchvision.transforms.functional.resize(tiles[i], dimens, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

                        tiles = [t for t in tiles if t is not None]
                        if len(tiles) > 0:
                            cat = torch.cat(tiles, dim=0)  # (N,3,H,W) – now safe
                            grid = make_grid(cat, nrow=3, padding=2)

                            img_name = f"{config_name}_{idx:03d}_{iteration}.png"
                            save_path = os.path.join(base_save_dir, img_name)
                            save_image(grid, save_path)

@torch.no_grad()
def training_report(
    tb_writer, 
    iteration,
    train_loss_report, 
    l1_loss, 
    elapsed,
    testing_iterations,
    scene : Scene, 
    renderFunc, 
    renderArgs,
    bg,
    initial_stage,
):
    if tb_writer:
        for tag,loss in train_loss_report.items():
            tb_writer.add_scalar('train_loss_patches/{}'.format(tag), loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_histogram("scene/refl_histogram", scene.gaussians.get_refl, iteration)
        max_scale = torch.max(scene.gaussians.get_scaling, dim=1).values
        min_scale = torch.min(scene.gaussians.get_scaling, dim=1).values
        tb_writer.add_histogram("scene/max_scale", max_scale, iteration)
        tb_writer.add_histogram("scene/min_scale", min_scale, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, len(scene.getTrainCameras()), 5)]})


        for config in validation_configs:
            config_name = config['name']
            if config['cameras'] and len(config['cameras']) > 0:

                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, initial_stage=initial_stage)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    alpha = torch.clamp(render_pkg["rend_alpha"], 0.0, 1.0).to("cuda")
                    base_color = torch.clamp(render_pkg["base_color_map"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_alpha_mask = torch.clamp(viewpoint.gt_alpha_mask.to("cuda"), 0.0, 1.0)

                    image = image * alpha + (1-alpha) * bg[:, None, None]
                    if gt_alpha_mask is not None:
                        gt_alpha_mask = gt_alpha_mask.cuda()
                        gt_image = gt_image * gt_alpha_mask + (1-gt_alpha_mask) * bg[:, None, None]

                    # ---- TensorBoard logging ----
                    if tb_writer and (idx < 5):
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.detach().cpu().numpy()[0], cmap='turbo')

                        tb_writer.add_images(
                            f"{config_name}_view_{viewpoint.image_name}/depth",
                            depth[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"{config_name}_view_{viewpoint.image_name}/render",
                            image[None],
                            global_step=iteration,
                        )

                        try:
                            refl_map = render_pkg['refl_strength_map']
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5

                            tb_writer.add_images(
                                f"{config_name}_view_{viewpoint.image_name}/rend_normal",
                                rend_normal[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                f"{config_name}_view_{viewpoint.image_name}/surf_normal",
                                surf_normal[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                f"{config_name}_view_{viewpoint.image_name}/rend_alpha",
                                rend_alpha[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                f"{config_name}_view_{viewpoint.image_name}/refl_map",
                                refl_map[None],
                                global_step=iteration,
                            )
                        except Exception:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{config_name}_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None],
                                global_step=iteration,
                            )

                    # ---- Metrics ----
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()


            # ---- Average metrics per config ----
            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            print(f"\n[ITER {iteration}] Evaluating {config_name}: L1 {l1_test} PSNR {psnr_test}")
            if tb_writer:
                tb_writer.add_scalar(
                    f"{config_name}/loss_viewpoint - l1_loss",
                    l1_test,
                    iteration,
                )
                tb_writer.add_scalar(
                    f"{config_name}/loss_viewpoint - psnr",
                    psnr_test,
                    iteration,
                )

    if tb_writer:
        tb_writer.flush()
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

    progress_iterations = [500, 1000, 1500, 3000] + [i for i in range(5000, args.iterations+1, 5000)]

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, progress_iterations)

    # All done
    print("\nTraining complete.")
