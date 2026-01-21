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
refnerf_scenes = ["shiny_blender/ball","shiny_blender/car","shiny_blender/coffee","shiny_blender/helmet","shiny_blender/teapot","shiny_blender/toaster"]
nerf_synthetic_scenes = ["nerf_synthetic/chair","nerf_synthetic/drums","nerf_synthetic/ficus","nerf_synthetic/hotdog","nerf_synthetic/lego","nerf_synthetic/materials","nerf_synthetic/mic","nerf_synthetic/ship"]
glossy_synthetic_scenes = ["GlossySynthetic/angel","GlossySynthetic/bell","GlossySynthetic/cat","GlossySynthetic/horse","GlossySynthetic/luyu","GlossySynthetic/potion","GlossySynthetic/tbell","GlossySynthetic/teapot"]


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="/mnt/output/ours/eval")

extra_args = {
    "ref_real/sedan": " -r 8 --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138 --iterations 55000 --longer_prop_iter 15000 --refl_init_value 1e-1",
    "ref_real/gardenspheres": " -r 6 --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974 --iterations 50000 --longer_prop_iter 10000",
    "ref_real/toycar": " -r 4 --env_scope_center 0.486 1.108 3.72 --env_scope_radius 2.507 --iterations 55000 --longer_prop_iter 15000",
    "shiny_blender/ball": " -w --init_until_iter 0 --synthetic",
    "shiny_blender/car": " -w --opacity_reset_interval 1000 --synthetic",
    "shiny_blender/coffee": " -w --init_until_iter 3000 --synthetic  --densification_interval_when_prop 500",
    "shiny_blender/helmet": " -w --init_until_iter 0 --synthetic",
    "shiny_blender/teapot": " -w --opacity_reset_interval 1000 --synthetic",
    "shiny_blender/toaster": " --opacity_reset_interval 1000 --synthetic --refl_init_value 1e-3",
    "GlossySynthetic/angel": " -w --init_until_iter 20 --synthetic --densification_interval_when_prop 500",
    "GlossySynthetic/bell": " -w --opacity_reset_interval 1000 --synthetic",
    "GlossySynthetic/cat": " -w --opacity_reset_interval 1000 --synthetic",
    "GlossySynthetic/horse": " -w --init_until_iter 20 --synthetic --densification_interval_when_prop 500",
    "GlossySynthetic/luyu": " -w --opacity_reset_interval 1000 --synthetic",
    "GlossySynthetic/potion": " -w --init_until_iter 20 --synthetic",
    "GlossySynthetic/tbell": " -w --init_until_iter 20 --synthetic --refl_init_value 1e-3",
    "GlossySynthetic/teapot": " -w --opacity_reset_interval 1000 --synthetic",
    "nerf_synthetic/chair": " -w --opacity_reset_interval 1000 --synthetic",
    "nerf_synthetic/drums": " -w --opacity_reset_interval 1000 --synthetic",
    "nerf_synthetic/ficus": " -w --opacity_reset_interval 1000 --synthetic",
    "nerf_synthetic/hotdog": " -w --opacity_reset_interval 1000 --synthetic",
    "nerf_synthetic/lego": " --opacity_reset_interval 1000 --synthetic --refl_init_value 1e-3",
    "nerf_synthetic/materials": " --opacity_reset_interval 1000 --synthetic --refl_init_value 1e-3",
    "nerf_synthetic/mic": " --opacity_reset_interval 1000 --synthetic",
    "nerf_synthetic/ship": " -w --opacity_reset_interval 1000 --synthetic"
}


args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(ref_real_scenes)
all_scenes.extend(refnerf_scenes)
all_scenes.extend(nerf_synthetic_scenes)
all_scenes.extend(glossy_synthetic_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--ref_real', type=str, default="/mnt/data")
    parser.add_argument('--refnerf', type=str, default="/mnt/data")
    parser.add_argument('--nerf_synthetic', type=str, default="/mnt/data")
    parser.add_argument('--glossy_synthetic', type=str, default="/mnt/data")
    args = parser.parse_args()
if not args.skip_training:
    common_args = " --disable_viewer --quiet --eval --test_iterations -1 --save_iterations 30000 --normal_propagation --color_sabotage"

    scene_times = {}
    
    start_time = time.time()
    for scene in ref_real_scenes:
        source = args.ref_real + "/" + scene
        extra = extra_args[scene]
        more_args = " --init_until_iter 3000 --use_env_scope  --densification_interval_when_prop 500 --densification_interval 500"

        scene_time = time.time()
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra + more_args)
        scene_times[scene] = (time.time() - scene_time)/60.0
    ref_real_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in refnerf_scenes:
        source = args.refnerf + "/" + scene
        extra = extra_args[scene]

        scene_time = time.time()
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra)
        scene_times[scene] = (time.time() - start_time)/60.0
    refnerf_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in nerf_synthetic_scenes:
        source = args.nerf_synthetic + "/" + scene
        extra = extra_args[scene]

        scene_time = time.time()
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra)
        scene_times[scene] = (time.time() - start_time)/60.0
    nerf_synthetic_timing = (time.time() - start_time)/60.0
    
    start_time = time.time()
    for scene in glossy_synthetic_scenes:
        source = args.glossy_synthetic + "/" + scene
        extra = extra_args[scene]

        scene_time = time.time()
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + extra)
        scene_times[scene] = (time.time() - start_time)/60.0
    glossy_synthetic_timing = (time.time() - start_time)/60.0

    with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
        file.write(f"ref_real: {ref_real_timing} minutes \n shiny_blender: {refnerf_timing} minutes \n nerf_synthetic: {nerf_synthetic_timing} minutes \n GlossySynthetic: {glossy_synthetic_timing} minutes \n")
    with open(os.path.join(args.output_path,"timing.json"), 'w') as file:
        json.dump(scene_times, file, indent=True)

if not args.skip_rendering:
    all_sources = []
    #for scene in ref_real_scenes:
        #all_sources.append(args.ref_real + "/" + scene + " --iteration 60000 --render_normals")
    all_sources.append(args.ref_real + "/" + ref_real_scenes[0] + " --iteration 60000 --render_normals")
    all_sources.append(args.ref_real + "/" + ref_real_scenes[1] + " --iteration 70000 --render_normals")
    all_sources.append(args.ref_real + "/" + ref_real_scenes[2] + " --iteration 70000 --render_normals")
    for scene in refnerf_scenes:
        all_sources.append(args.refnerf + "/" + scene + " --iteration 40000 --render_normals")
    for scene in nerf_synthetic_scenes:
        all_sources.append(args.nerf_synthetic + "/" + scene + " --iteration 40000")
    for scene in glossy_synthetic_scenes:
        all_sources.append(args.glossy_synthetic + "/" + scene + " --iteration 40000")
   
    print(all_sources)

    common_args = " --quiet --eval --skip_train"

    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)
