#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from utils.general_utils import build_rotation
from functools import partial
# import open3d as o3d
# import trimesh

class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']

            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)

            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            # self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            self.normals.append(normal)
            # self.depth_normals.append(depth_normal.cpu())
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            # gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            normal = self.normals[idx].permute(1,2,0).cpu().numpy()
            #toycar
            # R = build_rotation(torch.tensor([[0, 0.5735764, 0.819152, 0]])).cpu().numpy()
            #sedan
            R = build_rotation(torch.tensor([[ 0, 0.2164396, 0.976296, 0 ]])).cpu().numpy()
            #garden
            # R = build_rotation(torch.tensor([[0, 0.4226183, 0.9063078,0 ]])).cpu().numpy()
            h, w, _ = normal.shape
            normals_flat = normal.reshape(-1, 3)
            rotated_normals = (R[0] @ normals_flat.T).T
            rotated_normals = rotated_normals.reshape(h, w, 3)
            # Re-normaliza para garantir que os vetores continuem unitários
            norm = np.linalg.norm(rotated_normals, axis=2, keepdims=True)
            rotated_normals = rotated_normals / np.maximum(norm, 1e-6)
            save_img_u8(rotated_normals * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
