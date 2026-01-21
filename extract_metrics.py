
import os
from argparse import ArgumentParser
import json

# ref_real_scenes = ["ref_real/gardenspheres", "ref_real/sedan", "ref_real/toycar"]
# shiny_blender_scenes = ["shiny_blender/ball","shiny_blender/car","shiny_blender/coffee","shiny_blender/helmet","shiny_blender/teapot","shiny_blender/toaster"]
# nerf_synthetic_scenes = ["nerf_synthetic/chair","nerf_synthetic/drums","nerf_synthetic/ficus","nerf_synthetic/hotdog","nerf_synthetic/lego","nerf_synthetic/materials","nerf_synthetic/mic","nerf_synthetic/ship"]
# glossy_synthetic_scenes = ["GlossySynthetic/angel","GlossySynthetic/bell","GlossySynthetic/cat","GlossySynthetic/horse","GlossySynthetic/luyu","GlossySynthetic/potion","GlossySynthetic/tbell","GlossySynthetic/teapot"]
# materials_scenes = ["materials2/mirror2", "materials2/metal2", "materials2/glossy2", "materials2/black2", "materials2/white2"]
# materials_scenes = ["materials_sparse/mirror", "materials_sparse/metal", "materials_sparse/glossy", "materials_sparse/diffuse", "materials_sparse/black", "materials_sparse/white", "materials_sparse/far", "materials_sparse/cube_mirror", "materials_sparse/cube_metal", "materials_sparse/cube_glossy", "materials_sparse/cube_diffuse",
#   "materials_sparse/mirror_sky", "materials_sparse/metal_sky", "materials_sparse/glossy_sky", "materials_sparse/black_sky", "materials_sparse/white_sky",
#   "materials_sparse/mirror_const_env", "materials_sparse/metal_const_env", "materials_sparse/glossy_const_env"]
materials_scenes = ["materials_regular/black", "materials_regular/white", "materials_regular/diffuse", "materials_regular/glossy", "materials_regular/metal", "materials_regular/mirror"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--output_path", default="E:\\output\\3dgs-dr\\eval")

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(materials_scenes)

metrics = {}

for scene in all_scenes:
  fps = os.path.join(args.output_path, scene, 'fps.txt')
  with open(fps, 'r') as fp:
    fps_value = fp.readline().replace("fps:", "").replace('.', ',')[:-1]
    count = fp.readline().replace("count:", "")

  results_path = os.path.join(args.output_path, scene, 'results.json')
  print(results_path)

  with open(results_path, 'r') as file:
    results = json.load(file)

  max_key = list(results.keys())[0]
  for key in results.keys():
    if key > max_key:
      max_key = key
  # max_key = "ref_gs_30000"

  # timings_path = os.path.join(args.output_path, 'timing.json')
  # with open(timings_path, 'r') as fp:
  #   timings = json.load(fp)
  
  metrics[scene] = [
    results[max_key]["PSNR"],
    results[max_key]["SSIM"],
    results[max_key]["LPIPS"],
    fps_value,
    count,
    # timings[scene]
  ]

with open(os.path.join(args.output_path, 'results.csv'), 'w') as results:
  for method in metrics.keys():
    results.write(method+"\t")
  results.write("\n")
  for i in range(0,5):
    for method in metrics:
      results.write(str(metrics[method][i]).replace('.',',')+"\t")
    results.write("\n")

  
