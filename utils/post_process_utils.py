
import torch


_PIXEL_COORD_CACHE = {}
def get_pixel_coords(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Returns pixel coordinates with shape [H, W, 2] in (x, y) order as float32.
    Cached per (H, W, device).
    """
    key = (height, width, str(device))
    if key not in _PIXEL_COORD_CACHE:
        ys = torch.arange(height, device=device, dtype=torch.float32)
        xs = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).contiguous()  # [H, W, 2]
        _PIXEL_COORD_CACHE[key] = coords
    return _PIXEL_COORD_CACHE[key]

def apply_ppisp(ppisp, rgb_raw_chw, frame_idx, clamp=False):
  """
  rgb_raw_chw: [3, H, W]
  returns rgb_out_chw: [3, H, W]
  """
  _, H, W = rgb_raw_chw.shape
  pixel_coords = get_pixel_coords(H, W, rgb_raw_chw.device)
  camera_idx = 0

  rgb_raw_hwc = rgb_raw_chw.permute(1, 2, 0).contiguous()
  rgb_out_hwc = ppisp(
    rgb_raw_hwc,
    pixel_coords,
    resolution=(W, H),
    camera_idx=camera_idx,
    frame_idx=frame_idx,
  )
  rgb_out_chw = rgb_out_hwc.permute(2, 0, 1).contiguous()

  if clamp:
      rgb_out_chw = torch.clamp(rgb_out_chw, 0.0, 1.0)
  return rgb_out_chw