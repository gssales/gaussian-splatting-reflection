#!/usr/bin/env python3
from pathlib import Path
from argparse import Namespace


def process_scene(scene_dir: Path, dataset_dir: Path):
    """
    Process a single scene directory:
      - Read transforms_train.json
      - Split frames: every `step`-th frame (starting at start_index) -> test
      - Write updated transforms_train.json and new transforms_test.json
      - Rename 'train' folder to 'rgb'
      - Update file_path entries: change 'train' -> 'rgb' and remove file extension
    """
    print(f"\n=== Processing scene: {scene_dir} ===")
    
    cfgfilepath = scene_dir / "cfg_args"
    if not cfgfilepath.exists():
        print(f"  [!] No cfg_args found in {scene_dir}, skipping.")
        return

    config = "Namespace()"
    with open(cfgfilepath, "r", encoding="utf-8") as f:
        config = f.read()
    cfg_args = eval(config)

    print(f"  [+] Loaded cfg_args from {cfgfilepath.name}")
    cfg_args.model_path = str(scene_dir)
    print(f"  [+] Updated model_path to: {cfg_args.model_path}")

    # /mnt/data/materials/camera_regular/forest_env_sphere/sparse_rand/glossy => {dataset_dir}\\materials\\camera_regular\\forest_env_sphere\\sparse_rand\\glossy
    # replace /mnt/data with dataset_dir
    cfg_args.source_path = str(dataset_dir) + cfg_args.source_path.split("/mnt/data")[-1].replace("/", "\\")    
    print(f"  [+] Updated source_path to: {cfg_args.source_path}")

    cfg_args.white_background = True

    print(cfg_args.__dict__)
    with open(cfgfilepath, "w", encoding="utf-8") as f:
        f.write(str(cfg_args))

def process_root(root_dir: Path, dataset_dir: Path):
    """
    Process all scene subdirectories inside root_dir.
    A "scene" is any directory that contains cfg_args.
    """
    print(f"Root directory: {root_dir}")

    if not root_dir.is_dir():
        print(f"[!] {root_dir} is not a directory")
        return

    any_scene = False
    for sub in root_dir.iterdir():
        if sub.is_dir() and (sub / "cfg_args").exists():
            any_scene = True
            process_scene(sub, dataset_dir)
        elif sub.is_dir() and sub.name not in ["point_cloud","progress","train","test","cubemap"]:
            process_root(sub, dataset_dir)

    if not any_scene:
        # Also allow running the script directly inside a single scene directory
        if (root_dir / "cfg_args").exists():
            process_scene(root_dir, dataset_dir)
        else:
            print("[!] No scenes with cfg_args found.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Update cfg_args in scenes:\n"
            "- Create a consistent train/test split (every N-th image to test)\n"
            "- Rename 'train' folder to 'rgb'\n"
            "- Update JSON file_path entries (train->rgb, remove extension)"
        )
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root directory containing scene folders (or a single scene directory)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Dataset base directory to set source_path",
        default="E:\\Research\\data"
    )

    args = parser.parse_args()
    process_root(Path(args.root), Path(args.dataset_dir))