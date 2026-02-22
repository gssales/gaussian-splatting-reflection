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
from argparse import ArgumentParser
import time
import json
import yaml

ref_real_scenes = []#"ref_real/gardenspheres", "ref_real/sedan", "ref_real/toycar"]
refnerf_scenes = ["shiny_blender/ball","shiny_blender/car","shiny_blender/coffee","shiny_blender/helmet","shiny_blender/teapot","shiny_blender/toaster"]
nerf_synthetic_scenes = ["nerf_synthetic/chair","nerf_synthetic/drums","nerf_synthetic/ficus","nerf_synthetic/hotdog","nerf_synthetic/lego","nerf_synthetic/materials","nerf_synthetic/mic","nerf_synthetic/ship"]
glossy_synthetic_scenes = ["GlossySynthetic/angel","GlossySynthetic/bell","GlossySynthetic/cat","GlossySynthetic/horse","GlossySynthetic/luyu","GlossySynthetic/potion","GlossySynthetic/tbell","GlossySynthetic/teapot"]


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--output_path", default="/mnt/output/ours/eval")
parser.add_argument('--source', type=str, default="/mnt/data")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(ref_real_scenes)
all_scenes.extend(refnerf_scenes)
all_scenes.extend(nerf_synthetic_scenes)
all_scenes.extend(glossy_synthetic_scenes)

scene_args = {}
with open("scene_args.yaml", 'r') as file:
    try:
        scene_args = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

print("Starting report for all scenes...")

for scene in all_scenes:
    print("Report scene:", scene)

    # dataset/scene
    dataset = scene.split('/')[0]
    render_args = ""
    if dataset in scene_args["data"]["realDatasets"]:
        render_args += scene_args["real"]["render"]
    if dataset in scene_args["data"]["syntheticDatasets"]:
        render_args += scene_args["synthetic"]["render"]
        
    output_path = os.path.join(args.output_path, scene)
    report_command = "python report.py -m " + output_path + " " + render_args

    os.system(report_command)
