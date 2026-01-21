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

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    # [ B, C, H, W ] -> [ B, 1 ]
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_map(img1, img2):
    # 
    # [ B, C, H, W ] -> [ B, 1, H, W ] -> mean per pixel
    mse = torch.mean(((img1 - img2)) ** 2, dim=1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
    mse = (((img1 - img2)) ** 2) #.view(img1.shape[0], img1.shape[1], -1).mean(2, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    # map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'mask':
        net_image = render_pkg["env_scope_mask"].repeat(3,1,1)
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'base color':
        net_image = render_pkg["base_color_map"]
    elif output == 'refl. strength':
        net_image = render_pkg["refl_strength_map"].repeat(3,1,1)
    elif output == 'refl. color':
        net_image = render_pkg["refl_color_map"]
    elif output == 'edge':
        net_image = render_pkg["surf_normal"]
        net_image = (net_image+1)/2
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image

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

