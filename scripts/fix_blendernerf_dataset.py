#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
from copy import deepcopy


def process_scene(scene_dir: Path, step: int = 8, start_index: int = 0):
    """
    Process a single scene directory:
      - Read transforms_train.json
      - Split frames: every `step`-th frame (starting at start_index) -> test
      - Write updated transforms_train.json and new transforms_test.json
      - Rename 'train' folder to 'rgb'
      - Update file_path entries: change 'train' -> 'rgb' and remove file extension
    """
    print(f"\n=== Processing scene: {scene_dir} ===")

    train_json_path = scene_dir / "transforms_train.json"
    if not train_json_path.exists():
        print(f"  [!] No transforms_train.json found in {scene_dir}, skipping.")
        return

    # back up the original transforms_train.json
    backup_path = scene_dir / "transforms_train_original.json.bak"
    if not backup_path.exists():
        shutil.copy2(train_json_path, backup_path)
        print(f"  [+] Backed up transforms_train.json to {backup_path.name}")

    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    frames = train_data.get("frames", [])
    if not isinstance(frames, list) or len(frames) == 0:
        print("  [!] No frames found in transforms_train.json, skipping.")
        return

    # ---- Step 1: normalize file paths: 'train' -> 'rgb' and remove extension ----
    def fix_file_path(fp: str) -> str:
        """
        Given a file_path string, replace leading 'train' folder with 'rgb'
        and remove file extension.
        Examples:
          'train/r_000.png'   -> 'rgb/r_000'
          './train/r_000.png' -> 'rgb/r_000'
        """
        p = Path(fp)

        # If it starts with "train", replace it with "rgb"
        parts = list(p.parts)
        if parts and parts[0] == "train":
            parts[0] = "rgb"
        # Handle possible "./train/..." or similar (e.g., '.', 'train', 'r_000.png')
        if len(parts) >= 2 and parts[0] in (".", "") and parts[1] == "train":
            parts[1] = "rgb"

        new_p = Path(*parts)

        # Remove extension (e.g., '.png')
        new_p = new_p.with_suffix("")  # drops extension

        # Use POSIX-style paths for JSON
        return new_p.as_posix()

    for fr in frames:
        if "file_path" in fr:
            fr["file_path"] = fix_file_path(fr["file_path"])

    # ---- Step 2: split into train/test by sampling every `step`-th frame ----
    train_frames = []
    test_frames = []

    for i, fr in enumerate(frames):
        # every `step`-th image starting at `start_index` goes to test
        if (i - start_index) % step == 0:
            test_frames.append(fr)
        else:
            train_frames.append(fr)

    print(f"  [+] Total frames: {len(frames)}")
    print(f"      Train frames: {len(train_frames)}")
    print(f"      Test  frames: {len(test_frames)}")

    # ---- Step 3: write updated transforms_train.json ----
    new_train_data = deepcopy(train_data)
    new_train_data["frames"] = train_frames
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(new_train_data, f, indent=2)
    print("  [+] Updated transforms_train.json")

    # ---- Step 4: create new transforms_test.json from the same base ----
    test_json_path = scene_dir / "transforms_test.json"
    new_test_data = deepcopy(train_data)
    new_test_data["frames"] = test_frames
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(new_test_data, f, indent=2)
    print("  [+] Created new transforms_test.json")

    # ---- Step 5: rename 'train' dir to 'rgb' if needed ----
    train_dir = scene_dir / "train"
    rgb_dir = scene_dir / "rgb"

    if train_dir.exists():
        if rgb_dir.exists():
            print("  [!] 'rgb' directory already exists; not renaming 'train'.")
        else:
            train_dir.rename(rgb_dir)
            print("  [+] Renamed 'train' directory to 'rgb'")
    else:
        print("  [i] No 'train' directory to rename (maybe already 'rgb').")


def process_root(root_dir: Path, step: int = 8, start_index: int = 0):
    """
    Process all scene subdirectories inside root_dir.
    A "scene" is any directory that contains transforms_train.json.
    """
    print(f"Root directory: {root_dir}")

    if not root_dir.is_dir():
        print(f"[!] {root_dir} is not a directory")
        return

    any_scene = False
    for sub in root_dir.iterdir():
        if sub.is_dir() and (sub / "transforms_train.json").exists():
            any_scene = True
            process_scene(sub, step=step, start_index=start_index)

    if not any_scene:
        # Also allow running the script directly inside a single scene directory
        if (root_dir / "transforms_train.json").exists():
            process_scene(root_dir, step=step, start_index=start_index)
        else:
            print("[!] No scenes with transforms_train.json found.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fix BlenderNeRF Camera On Sphere datasets:\n"
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
        "--step",
        type=int,
        default=8,
        help="Sample every N-th frame to the test set (default: 8)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index offset for sampling (default: 0 -> frames 0,8,16,... in test)",
    )

    args = parser.parse_args()
    process_root(Path(args.root), step=args.step, start_index=args.start_index)
