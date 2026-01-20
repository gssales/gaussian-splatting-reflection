import torch

def angular_error_map(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
    """
    pred, gt: torch tensors of shape [1, 3, H, W], normalized

    returns: torch tensor of shape [H, W] with angular error in degrees
             invalid pixels set to NaN
    """
    # Per-pixel dot product → [H, W]
    dot = torch.sum(pred * gt, dim=0)

    # Per-pixel norms → [H, W]
    norm_pred = torch.linalg.norm(pred, dim=0)
    norm_gt   = torch.linalg.norm(gt, dim=0)

    # Cosine of angle
    cos_ang = dot / (norm_pred * norm_gt + eps)
    cos_ang = torch.clamp(cos_ang, -1.0, 1.0)

    # Angle in degrees
    ang_deg = torch.acos(cos_ang) * (180.0 / torch.pi)

    # Mask invalid pixels
    invalid = (norm_pred <= eps) | (norm_gt <= eps) | torch.isnan(ang_deg)
    ang_deg = ang_deg.clone()
    ang_deg[invalid] = float("nan")
    
    return ang_deg


def compute_mae(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
    """
    pred, gt: torch tensors of shape [1, 3, H, W]
              pred assumed in [0,255], gt in [0,65535]

    returns: torch tensor of shape [H, W] with angular error in degrees
             invalid pixels set to NaN
    """

    if pred.ndim != 4 or gt.ndim != 4:
        raise ValueError(f"Expected 4D tensors [B,3,H,W], got pred {pred.shape}, gt {gt.shape}")

    if pred.shape[0] != 1 or gt.shape[0] != 1:
        raise ValueError("This function expects batch size = 1")

    if pred.shape[1] != 3 or gt.shape[1] != 3:
        raise ValueError("Expected channel dimension = 3")

    # Remove batch dimension → [3, H, W]
    pred = pred[0]
    gt = gt[0]

    # Convert to float + normalize
    pred = pred.to(torch.float32)
    gt   = gt.to(torch.float32)
    if pred.max() > 1.0:
        pred = pred / 255.0
    if gt.max() > 1.0:
        gt = gt / 65535.0

    angular_error = angular_error_map(pred, gt, eps)

    return angular_error.mean()
