import numpy as np
import os, sys
import tqdm
from PIL import Image
import torch
import cv2
import torchvision.transforms.functional as tf
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

def compute_mae(pred, gt, eps=1e-8):
  """
  gt, pred: arrays of shape (H, W, 3), holding either normals or RGB vectors.
  returns: mean angular error in degrees
  """
  # flatten to (N,3)
  gt_v   = gt.reshape(-1, 3)
  pred_v = pred.reshape(-1, 3)
  # normalize to unit length
  gt_v = gt_v.astype(np.float32) / 65535.0
  pred_v = pred_v.astype(np.float32) / 255.0
  # dot products and norms
  dot = np.sum(gt_v * pred_v, axis=1)
  norm_gt   = np.linalg.norm(gt_v,   axis=1)
  norm_pred = np.linalg.norm(pred_v, axis=1)
  cos_ang = dot / (norm_gt * norm_pred + eps)
  # clamp and compute angle
  cos_ang = np.clip(cos_ang, -1.0, 1.0)
  ang_rad = np.arccos(cos_ang)
  ang_deg = np.degrees(ang_rad)
  # mask out any pixels you don’t trust (e.g. depth invalid)
  valid = ~np.isnan(ang_deg)
  return np.mean(ang_deg[valid])

def compute_mean_angular_error(render_path, gt_path):
  gt_normals, alphas = readGTImages(gt_path)

  for method in os.listdir(render_path):
    print(method)
    render_normals, image_names = readImages(f"{render_path}/{method}")
    total_mae = 0.0
    for idx in range(len(render_normals)):
      alpha = alphas[idx]
      normal = render_normals[idx][alpha > 0]
      normal = (normal-0.5)*2
      
      gt_normal = gt_normals[idx][alpha > 0]
      gt_normal = (gt_normal-0.5)*2

      # print(alpha.shape)
      # print(normal.shape)
      # print(gt_normal.shape)
      # cv2.imshow("alpha", alpha)
      # cv2.imshow("normal", render_normals[idx])
      # cv2.imshow("gt normal", gt_normals[idx])
      # cv2.waitKey(0)

      # print(image_names[idx])
      total_mae += compute_mae(normal, gt_normal)
  
    print(f"Method: {method}  MÂE: {total_mae/len(render_normals)}")

def readImages(renders_dir):
  renders = []
  image_names = []
  for fname in os.listdir(renders_dir):
    render = cv2.imread(os.path.join(renders_dir, fname), cv2.IMREAD_UNCHANGED)
    renders.append(render)
    image_names.append(fname)
  return renders, image_names

def readGTImages(gt_dir):
  images = []
  alphas = []
  for index in range(200):
    image = cv2.imread(os.path.join(gt_dir, f"r_{index}_normal.png"), cv2.IMREAD_UNCHANGED)[..., :3]
    try:
      alpha = cv2.imread(os.path.join(gt_dir, f"r_{index}.png"), cv2.IMREAD_UNCHANGED)[..., 3] / 255.0
    except:
      alpha = cv2.imread(os.path.join(gt_dir, f"r_{index}_alpha.png"), cv2.IMREAD_UNCHANGED) / 255.0
    images.append(image)
    alphas.append(alpha)

  return images, alphas


if __name__ == "__main__":
  # Set up command line argument parser
  parser = ArgumentParser(description="Testing script parameters")
  parser.add_argument("--render_path", type=str)
  parser.add_argument("--gt_path", default="../../datasets/shiny_blender", type=str)
  
  cmdlne_string = sys.argv[1:]
  args = parser.parse_args(cmdlne_string)

  compute_mean_angular_error(args.render_path, args.gt_path)