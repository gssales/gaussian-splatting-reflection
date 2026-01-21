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

materials_scenes = ["specular/specular_100", "specular/specular_75", "specular/specular_50"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="/mnt/output/ours/eval")


args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(materials_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--materials', type=str, default="/mnt/data")
    args = parser.parse_args()
if not args.skip_training:
    common_args = " --disable_viewer --quiet --eval --save_iterations 30000 --normal_propagation --color_sabotage --init_until_iter 20 --densification_interval_when_prop 500 -w"

    scene_times = {}
    
    start_time = time.time()
    for scene in materials_scenes:
        source = args.materials + "/" + scene

        scene_time = time.time()
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        scene_times[scene] = (time.time() - start_time)/60.0
    materials_timing = (time.time() - start_time)/60.0

    with open(os.path.join(args.output_path,"specular_timing.txt"), 'w') as file:
        file.write(f"materials: {materials_timing} minutes \n")
    with open(os.path.join(args.output_path,"specular_timing.json"), 'w') as file:
        json.dump(scene_times, file, indent=True)

if not args.skip_rendering:
    all_sources = []
    for scene in materials_scenes:
        all_sources.append(args.materials + "/" + scene + " --iteration 31000 --render_normals --render_refl -w")
   
    print(all_sources)

    common_args = " --quiet --eval --skip_train"

    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)

