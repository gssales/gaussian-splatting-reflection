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
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_fps", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--skip_collect_results", action="store_true")
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

if not args.skip_training:
    print("Starting training for all scenes...")
    common_args = " --disable_viewer --quiet --eval --test_iterations -1"

    scene_times = {}
    
    for scene in all_scenes:
        print("Training scene:", scene)
        source = args.source + "/" + scene

        # dataset/scene
        dataset = scene.split('/')[0]
        train_args = ""
        if scene in scene_args["args"]:
            train_args = scene_args["args"][scene]
        if dataset in scene_args["data"]["realDatasets"]:
            train_args += scene_args["real"]["train"]
        if dataset in scene_args["data"]["syntheticDatasets"]:
            train_args += scene_args["synthetic"]["train"]

        output_path = os.path.join(args.output_path, scene)
        train_command = "python train.py -s " + source + " -m " + output_path + common_args + train_args
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "train_command.sh"), 'w') as file:
            file.write(train_command)

        scene_time = time.time()
        os.system(train_command)
        scene_times[scene] = (time.time() - scene_time)/60.0

    with open(os.path.join(args.output_path,"timing.json"), 'w') as file:
        json.dump(scene_times, file, indent=True)

if not args.skip_rendering:
    print("Starting rendering for all scenes...")
    common_args = " --quiet --eval --skip_train --render_normals"

    for scene in all_scenes:
        print("Rendering scene:", scene)
        source = args.source + "/" + scene

        # dataset/scene
        dataset = scene.split('/')[0]
        render_args = ""
        if dataset in scene_args["data"]["realDatasets"]:
            render_args += scene_args["real"]["render"]
        if dataset in scene_args["data"]["syntheticDatasets"]:
            render_args += scene_args["synthetic"]["render"]
            
        output_path = os.path.join(args.output_path, scene)
        render_command = "python render.py -s " + source + " -m " + output_path + common_args + render_args
        with open(os.path.join(output_path, "train_command.sh"), 'a') as file:
            file.write(render_command)

        os.system(render_command)
        
if not args.skip_fps:
    print("Starting FPS evaluation for all scenes...")

    for scene in all_scenes:
        print("FPS eval scene:", scene)            
        output_path = os.path.join(args.output_path, scene)
        fps_eval_command = "python eval_fps.py -m " + output_path
        os.system(fps_eval_command)

if not args.skip_metrics:
    print("Starting metrics computation for all scenes...")
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)

if not args.skip_collect_results:
    output_path = args.output_path
    print("Collecting results in:", output_path)
    collect_command = "python collect_results.py --tsv --output_path " + output_path
    os.system(collect_command)

print("Done with full evaluation for all scenes!")
