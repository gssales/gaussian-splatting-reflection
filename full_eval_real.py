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

ref_real_scenes = ["ref_real/gardenspheres", "ref_real/sedan", "ref_real/toycar"]
envgs_scenes = ["envgs/dog", "envgs/audi"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="/mnt/output/ours/real_scenes2")


extra_args = {
    "ref_real/sedan": " -r 8 --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138",
    "ref_real/gardenspheres": " -r 6 --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974",
    "ref_real/toycar": " -r 4 --env_scope_center 0.486 1.108 3.72 --env_scope_radius 2.507",
    "envgs/dog": " --env_scope_center -0.032 0.808 0.751 --env_scope_radius 4",
    "envgs/audi": " --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 5",
}


args, _ = parser.parse_known_args()

all_scenes = []

all_scenes.extend(ref_real_scenes)
all_scenes.extend(envgs_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--ref_real', type=str, default="/mnt/data")
    parser.add_argument('--envgs', type=str, default="/mnt/data")
    args = parser.parse_args()
if not args.skip_training:
    common_args = " --quiet --eval --iterations 60000 --test_iterations -1 --normal_propagation --color_sabotage --densification_interval_when_prop 500 --densification_interval 500 --init_until_iter 3000 --refl_init_value 1e-4 "
    
    scene_times = {}
    
    start_time = time.time()
    for scene in ref_real_scenes:
        source = args.ref_real + "/" + scene
        extra = extra_args[scene]
        more_args = " --longer_prop_iter 30_000 --use_env_scope"

        scene_time = time.time()
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra + more_args)
        scene_times[scene] = (time.time() - scene_time)/60.0
    ref_real_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene in envgs_scenes:
        source = args.envgs + "/" + scene
        extra = extra_args[scene]
        more_args = " --longer_prop_iter 30_000 --use_env_scope"
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra + more_args)
        scene_times[scene] = (time.time() - scene_time)/60.0
    envgs_timing = (time.time() - start_time)/60.0

    with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
        file.write(f"ref_real: {ref_real_timing} minutes \n envgs: {envgs_timing} minutes \n")
    with open(os.path.join(args.output_path,"timing.json"), 'w') as file:
        json.dump(scene_times, file, indent=True)

if not args.skip_rendering:
    all_sources = []
    for scene in ref_real_scenes:
        all_sources.append(args.ref_real + "/" + scene)
    for scene in envgs_scenes:
        all_sources.append(args.envgs + "/" + scene)

    common_args = " --quiet --eval --skip_train"

    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 60000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 90000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python view_diff_maps.py -m " + args.output_path + "/" + scene)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)

