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

envgs_scenes = ["envgs/dog", "envgs/audi"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="/mnt/output/ours/eval")


extra_args = {
    "envgs/dog": " --env_scope_center -0.032 0.808 0.751 --env_scope_radius 4",
    "envgs/audi": " --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 5",
}


args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(envgs_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--envgs', type=str, default="/mnt/data")
    args = parser.parse_args()
if not args.skip_training:
    common_args = " --quiet --eval --iterations 55000 --test_iterations -1 --normal_propagation --color_sabotage --densification_interval_when_prop 500 --init_until_iter 3000 --refl_init_value 1e-1 "
    
    start_time = time.time()
    for scene in envgs_scenes:
        source = args.envgs + "/" + scene
        extra = extra_args[scene]
        more_args = " --longer_prop_iter 15_000 --use_env_scope"
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra + more_args)
    envgs_timing = (time.time() - start_time)/60.0

    with open(os.path.join(args.output_path,"envgs_timing.txt"), 'w') as file:
        file.write(f"envgs: {envgs_timing} minutes\n")

if not args.skip_rendering:
    all_sources = []
    for scene in envgs_scenes:
        all_sources.append(args.envgs + "/" + scene)
    
    common_args = " --quiet --eval --skip_train"

    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 70000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)
