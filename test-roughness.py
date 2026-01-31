
import os
import torch
from PIL import Image
from argparse import ArgumentParser
from torchvision.utils import save_image
import tqdm
import nvdiffrast.torch

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render, get_refl_color, sample_camera_rays, reflection
from utils.general_utils import PILtoTorch, inverse_sigmoid
from cubemapencoder import CubemapEncoder
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

MIPMAP_LEVELS = 7

@torch.no_grad()
def load_cubemap_mipmap(cubemap_mipmap_path: str):
    cubemap_levels = []
    for level in range(MIPMAP_LEVELS):
        level_faces = []
        for i in range(6):
            img = Image.open(f"{cubemap_mipmap_path}/{i}_{level}.png")
            cubemap_face = PILtoTorch(img, img.size)[0:3]
            cubemap_face = inverse_sigmoid(cubemap_face)
            level_faces.append(cubemap_face.cuda())
        level_faces = torch.stack(level_faces, dim=0)  # (6, 3, H, W)
        envmap = CubemapEncoder(output_dim=3, resolution=level_faces.shape[2]).cuda()
        envmap.set_textures(level_faces)
        cubemap_levels.append(envmap)
    return cubemap_levels

@torch.no_grad()
def roughness_renderer(cubemap_mipmaps, camera, render_args):

    rendered = render(camera, *render_args, initial_stage=False)

    base_color = rendered['base_color_map']
    refl_map = rendered['refl_strength_map']
    normal = rendered['rend_normal']
    alpha_map = rendered['rend_alpha']
    original_refl_color = rendered['refl_color_map']

    roughness_map = refl_map.clone()
    roughness_map[refl_map > (refl_map.mean() + 0.2)] = 0.0
    roughness_map[roughness_map > 0.0] = 0.75
    roughness_map[alpha_map < 0.5] = 0.0

    refl_map_ = refl_map.clone()
    refl_map_[roughness_map > 0.0] = refl_map.mean() + 0.3
    # refl_map_[alpha_map < 0.5] = 0.0

    refl_colors = []
    normal = normal.permute(1,2,0)
    for level in range(MIPMAP_LEVELS):
        cubemap = cubemap_mipmaps[level]
        refl_color = get_refl_color(cubemap, camera.HWK, camera.R, camera.T, normal)
        refl_color[refl_color.isnan()] = 0.0
        refl_colors.append(refl_color)

    refl_color = refl_colors[0].clone()

    # progress_bar = tqdm.tqdm(total=refl_map.shape[1] * refl_map.shape[2], desc="Calculating final reflection color")
    # for r in range(refl_map.shape[1]):
    #     for c in range(refl_map.shape[2]):
    #         roughness = roughness_map[0, r, c].item()
    #         mip_level = min(int(roughness * (MIPMAP_LEVELS - 1)), MIPMAP_LEVELS - 1)
    #         mip_level_1 = min(mip_level + 1, MIPMAP_LEVELS - 1)
    #         weight = roughness * (MIPMAP_LEVELS - 1) - mip_level
    #         if mip_level != 0:
    #             print(mip_level, mip_level_1, weight)
    #         refl_color[:, r, c] = (1 - weight) * refl_colors[mip_level][:, r, c] + weight * refl_colors[mip_level_1][:, r, c]
    #         progress_bar.update(1)
    # progress_bar.close()
    mip_level = 4
    mip_level_1 = min(mip_level + 1, MIPMAP_LEVELS - 1)
    weight = 0.5
    selected_pixels = roughness_map.repeat(3, 1, 1) > 0.0
    refl_4 = refl_colors[mip_level][selected_pixels]
    refl_5 = refl_colors[mip_level_1][selected_pixels]
    refl_color[selected_pixels] = (1 - weight) * refl_4 + weight * refl_5

    # for c in range(refl_color.shape[1]):
    #     for r in range(refl_color.shape[2]):
    #         if roughness_map[0, r, c] > 0.5:
    #             refl_color[:, r, c] = (1-weight) * refl_colors[mip_level][:, r, c] + weight * refl_colors[mip_level_1][:, r, c]

    final_color = base_color * (1 - refl_map_) + refl_color * refl_map_

    return final_color, refl_map_, roughness_map, refl_color, original_refl_color, rendered['render']



def test_roughness(model: ModelParams, pipe: PipelineParams, output_dir: str, iteration: int, camera_index: int, cubemap_mipmap_path: str):

    gaussians = GaussianModel(model.sh_degree)
    scene = Scene(model, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cubemap_mipmaps = load_cubemap_mipmap(cubemap_mipmap_path)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    original_psnr_values = []
    original_ssim_values = []
    original_lpips_values = []

    for camera_index in range(0, 101, 10):
        print(f"Testing camera index {camera_index}...")
        camera = scene.getTestCameras()[camera_index]

        with torch.no_grad():
            rendered, refl_map, roughness_map, refl_color, original_refl_color, original_rendered = roughness_renderer(cubemap_mipmaps, camera, (gaussians, pipe, background))

            save_image(rendered, os.path.join(output_dir, f"image_{camera_index}.png"))
            save_image(refl_map, os.path.join(output_dir, f"refl_map_{camera_index}.png"))
            save_image(roughness_map, os.path.join(output_dir, f"roughness_map_{camera_index}.png"))
            save_image(refl_color, os.path.join(output_dir, f"refl_color_{camera_index}.png"))
            save_image(original_refl_color, os.path.join(output_dir, f"original_refl_color_{camera_index}.png"))

            rendered = torch.clamp(rendered, 0.0, 1.0)
            original_rendered = torch.clamp(original_rendered, 0.0, 1.0)
            gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
            
            gt_alpha_mask = camera.gt_alpha_mask
            if gt_alpha_mask is not None:
                gt_image = gt_image * gt_alpha_mask + (1-gt_alpha_mask) * background[:, None, None]

            psnr_value = psnr(rendered, gt_image).mean()
            ssim_value = ssim(rendered, gt_image).mean()
            lpips_value = lpips(rendered, gt_image).mean()

            original_psnr_value = psnr(original_rendered, gt_image).mean()
            original_ssim_value = ssim(original_rendered, gt_image).mean()
            original_lpips_value = lpips(original_rendered, gt_image).mean()

            print(f"method \t\t PSNR \t SSIM \t LPIPS")
            print(f"roughness \t {psnr_value.item():.2f} \t {ssim_value.item():.3f} \t {lpips_value.item():.3f}")
            print(f"original \t {original_psnr_value.item():.2f} \t {original_ssim_value.item():.3f} \t {original_lpips_value.item():.3f}")

            psnr_values.append(psnr_value.item())
            ssim_values.append(ssim_value.item())
            lpips_values.append(lpips_value.item())
            original_psnr_values.append(original_psnr_value.item())
            original_ssim_values.append(original_ssim_value.item())
            original_lpips_values.append(original_lpips_value.item())

    print("==== Final Averages ====")
    print(f"Roughness-modulated render:")
    print(f"Average over all views - PSNR: {sum(psnr_values)/len(psnr_values):.2f}, SSIM: {sum(ssim_values)/len(ssim_values):.3f}, LPIPS: {sum(lpips_values)/len(lpips_values):.3f}")
    print(f"Original render:")
    print(f"Average over all views - PSNR: {sum(original_psnr_values)/len(original_psnr_values):.2f}, SSIM: {sum(original_ssim_values)/len(original_ssim_values):.3f}, LPIPS: {sum(original_lpips_values)/len(original_lpips_values):.3f}")

def test_nvdiffrast(model: ModelParams, pipe: PipelineParams, output_dir: str, iteration: int, camera_index: int, cubemap_mipmap_path: str):
    gaussians = GaussianModel(model.sh_degree)
    scene = Scene(model, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    textures = []
    cubemap_path = "E:\\Research\\projects\\gaussian-splatting-reflection\\cubemap"
    for i in range(6):
        img = Image.open(f"{cubemap_path}/{i}_40000.png")
        cubemap_face = PILtoTorch(img, img.size)[0:3]
        textures.append(cubemap_face.cuda())

    textures = torch.stack(textures, dim=0)  # (6, 3, H, W)
    # reshape to [minibatch_size, 6, tex_height, tex_width, tex_channels]
    textures = textures.permute(0,2,3,1).unsqueeze(0)
    
    for camera_index in range(0, 101, 10):
        print(f"Testing camera index {camera_index}...")
        camera = scene.getTrainCameras()[camera_index]

        with torch.no_grad():
            rendered = render(camera, gaussians, pipe, background, initial_stage=False)
            
            base_color = rendered['base_color_map']
            refl_map = rendered['refl_strength_map']
            roughness_map = rendered['roughness_map']
            normal = rendered['rend_normal'].permute(1,2,0)
            alpha_map = rendered['rend_alpha']
            original_refl_color = rendered['refl_color_map']

            rays_d = sample_camera_rays(camera.HWK, camera.R, camera.T)
            rays_d = reflection(rays_d, normal).unsqueeze(0)

            roughness_map = roughness_map / roughness_map.max()
            mip_level = (1-roughness_map) * (MIPMAP_LEVELS - 1)

            encoding = nvdiffrast.torch.texture(
                textures.contiguous(),
                rays_d.contiguous(),
                mip_level_bias=mip_level*(MIPMAP_LEVELS - 1),
                boundary_mode="cube",
                max_mip_level=MIPMAP_LEVELS - 1,
            )

            ## encodig has shape (1, H, W, 3)
            encoding = encoding[0].permute(2,0,1)  # (3, H, W)
        
            # save_image(rendered, os.path.join(output_dir, f"image_{camera_index}.png"))
            save_image(refl_map, os.path.join(output_dir, f"refl_map_{camera_index}.png"))
            save_image(roughness_map, os.path.join(output_dir, f"roughness_map_{camera_index}.png"))
            save_image(encoding, os.path.join(output_dir, f"refl_color_{camera_index}.png"))
            save_image(original_refl_color, os.path.join(output_dir, f"original_refl_color_{camera_index}.png"))


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--iteration', type=int, default=7000)
    parser.add_argument('--camera_index', type=int, default=0)
    parser.add_argument('--cubemap_mipmap_path', type=str, default="E:\\output\\ours\\eval3\\shiny_blender\\ball\\roughness_test\\cubemap")
    parser.add_argument('--output_dir', type=str, default="E:\\output\\ours\\eval3\\shiny_blender\\ball\\roughness_test")

    args = get_combined_args(parser)
    
    test_nvdiffrast(lp.extract(args), pp.extract(args), args.output_dir, args.iteration, args.camera_index, args.cubemap_mipmap_path)