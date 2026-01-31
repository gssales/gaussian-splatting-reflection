import os, sys
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import json
import torchvision


from utils.general_utils import PILtoTorch
from utils.mae_utils import compute_mae

def compute_mean_angular_error(model_path, gt_path):
  print(f"Evaluating Normals in Path: {model_path}")
  gt_normals, gt_image_names, alphas = readGTImages(gt_path)

  mae_eval_values = {}

  test_dir = model_path / "test"
  for method in os.listdir(test_dir):
    print(f"  [+] Processing Method: {method}")
    render_normals, image_names = readImages(test_dir / method)

    total_mae = 0.0
    for idx in tqdm(range(len(render_normals))):
      alpha = alphas[idx]
      normal = render_normals[idx][alpha > 0]
      normal = (normal-0.5)*2
      
      gt_normal = gt_normals[idx][alpha > 0]
      gt_normal = (gt_normal-0.5)*2

      torchvision.utils.save_image(render_normals[idx], f"test/{image_names[idx]}_pred.png")
      torchvision.utils.save_image(gt_normals[idx], f"test/{image_names[idx]}_gt.png")

      print(normal[0], gt_normal[0])

      total_mae += compute_mae(normal, gt_normal)
  
    mae_eval_values[method] = {}
    mae_eval_values[method]["MAE"] = total_mae/len(render_normals)
    print(f"Method: {method}  MÂE: {total_mae/len(render_normals)}")

  with open(model_path / "mae_eval.json", "w") as fp:
    json.dump(mae_eval_values, fp, indent=True)

def readImages(renders_dir):
  renders = []
  image_names = []
  for fname in os.listdir(renders_dir / "normals"):
    render = Image.open(renders_dir / "normals" / fname)
    render = PILtoTorch(render, render.size)[:3, :, :]
    renders.append(render)
    image_names.append(fname)
  return renders, image_names

def readGTImages(gt_dir):
  gt_normals = []
  images_names = []
  alphas = []
  for fname in os.listdir(gt_dir):
    if fname.endswith("_normal.png"):
      normal = Image.open(gt_dir / fname)
      normal = PILtoTorch(normal, normal.size)[:3, :, :]
      gt_normals.append(normal)
      images_names.append(fname)
    if fname.endswith("_alpha.png"):
      alpha = Image.open(gt_dir / fname)
      alpha = PILtoTorch(alpha, alpha.size).repeat(3,1,1)
      alphas.append(alpha)
    # elif not fname.endswith("_normal.png") and not fname.endswith("_alpha.png"):
    #   rgba_image = Image.open(gt_dir / fname)
    #   rgba_image = PILtoTorch(rgba_image, rgba_image.size)
    #   alphas.append(rgba_image)
  return gt_normals, images_names, alphas


if __name__ == "__main__":
  # Set up command line argument parser
  parser = ArgumentParser(description="Testing script parameters")
  parser.add_argument("--model_path", type=str)
  parser.add_argument("--gt_path", default="E:\\Research\\data\\shiny_blender", type=str)
  
  cmdlne_string = sys.argv[1:]
  args = parser.parse_args(cmdlne_string)

  compute_mean_angular_error(Path(args.model_path), Path(args.gt_path))