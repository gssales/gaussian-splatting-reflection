import torch
import numpy as np
from torch import nn
import nvdiffrast.torch
from torchvision.transforms import functional as F

class CubemapMipEncoder(nn.Module):
    def __init__(
        self,
        n_levels: int = 9,
        max_level: int = 4,
        resolution: int = 256,
        rand_init: bool = False
    ):
        super(CubemapMipEncoder, self).__init__()
        self.n_levels = n_levels
        self.max_level = max_level
        
        self.register_parameter("texture", nn.Parameter(torch.zeros(1, 6, resolution, resolution, 3)),)
        
        if rand_init:
            self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.uniform_(self.texture, -0.5, 0.5)

    def resize(self, new_resolution: int) -> None:
        self.n_levels = int(np.log2(new_resolution)) + 1
        # from (1, 6, H, W, C) to (6, C, H, W)
        texture = self.texture.squeeze(0).permute(0,3,1,2)
        texture = nn.functional.interpolate(texture, size=(new_resolution, new_resolution), mode='bilinear', align_corners=False).permute(0,2,3,1).unsqueeze(0)
        self.texture = nn.Parameter(texture)
        
    def filter(self, activation, inverse_activation, factor = 2.0):
        textures = self.texture
        textures = activation(textures)
        textures =  F.adjust_sharpness(textures, factor)
        textures = torch.clamp(textures, min=1e-3, max=1-1e-3)
        textures = inverse_activation(textures)
        self.texture = textures

    def forward(self, rays, mip_levels):
        level = mip_levels.contiguous()

        enc = nvdiffrast.torch.texture(
            self.texture.contiguous(),
            rays.contiguous(),
            mip_level_bias=level*self.n_levels,
            boundary_mode="cube",
            max_mip_level=self.max_level,
        )
        
        enc = enc[0].permute(2, 0, 1)
        return enc