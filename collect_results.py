import os
import re
import json
import csv
from pathlib import Path
from argparse import ArgumentParser


ENV_SHAPES = {"forest_env_sphere", "forest_env_cube", "constant_env_sphere", "constant_env_cube"}
DENSITIES = {"dense_rand", "dense_regular", "sparse_rand", "sparse_regular"}
MATERIALS = {"diffuse", "glossy", "metal", "mirror", "black", "white"}


def parse_fps_txt(fps_path: Path):
    """
    Expected:
      fps: 123.45
      count: 999
    Returns (fps_str, count_str) keeping original formatting as strings.
    """
    fps_value = ""
    count_value = ""
    with open(fps_path, "r", encoding="utf-8") as fp:
        line1 = fp.readline().strip()
        line2 = fp.readline().strip()

    if line1.lower().startswith("fps:"):
        fps_value = line1.split(":", 1)[1].strip()
    else:
        fps_value = line1.strip()

    if line2.lower().startswith("count:"):
        count_value = line2.split(":", 1)[1].strip()
    else:
        count_value = line2.strip()

    return fps_value, count_value


def extract_iteration(key: str):
    """
    Extract a numeric iteration from keys like:
      'ref_gs_30000' -> 30000
      'ours_31000'   -> 31000
    If none found, return -1 so it loses in max().
    """
    m = re.findall(r"(\d+)", key)
    if not m:
        return -1
    return int(m[-1])


def pick_best_key(results: dict):
    """
    Pick the key with the highest numeric iteration.
    Fallback: first key if parsing fails.
    """
    if not results:
        return None

    keys = list(results.keys())
    best = max(keys, key=lambda k: extract_iteration(k))
    return best


def is_valid_scene_path(scene_dir: Path):
    """
    Validate the last 3 components are: <env_shape>/<density>/<material>
    under .../camera_regular/...
    """
    parts = scene_dir.parts
    # ... camera_regular env density material
    if len(parts) < 4:
        return False
    env, dens, mat = parts[-3], parts[-2], parts[-1]
    return (env in ENV_SHAPES) and (dens in DENSITIES) and (mat in MATERIALS)


def find_scene_dirs(output_root: Path, camera_folder: str = "camera_regular"):
    """
    Finds all scene directories under:
      output_root/camera_regular/<env>/<density>/<material>
    """
    base = output_root / camera_folder
    if not base.is_dir():
        raise FileNotFoundError(f"Not found: {base}")

    # Expect exactly 3 levels below camera_folder
    # (env_shape)/(density_sampling)/(material)
    scene_dirs = []
    for env_dir in base.iterdir():
        if not env_dir.is_dir() or env_dir.name not in ENV_SHAPES:
            continue
        for dens_dir in env_dir.iterdir():
            if not dens_dir.is_dir() or dens_dir.name not in DENSITIES:
                continue
            for mat_dir in dens_dir.iterdir():
                if not mat_dir.is_dir() or mat_dir.name not in MATERIALS:
                    continue
                scene_dirs.append(mat_dir)

    return sorted(scene_dirs)


def main():
    parser = ArgumentParser(description="Collect PSNR/SSIM/LPIPS + FPS into a CSV across all materials models.")
    parser.add_argument("--output_path", default=r"E:\output\3dgs-dr\eval",
                        help="Root eval output path (contains camera_regular/...)")
    parser.add_argument("--camera_folder", default="camera_regular",
                        help="Subfolder name under output_path (default: camera_regular)")
    parser.add_argument("--csv_name", default="results_all.csv",
                        help="Output CSV filename (written inside output_path)")
    parser.add_argument("--tsv", action="store_true",
                        help="Write TSV instead of CSV")
    args, _ = parser.parse_known_args()

    output_root = Path(args.output_path)
    out_file = output_root / args.csv_name

    scene_dirs = find_scene_dirs(output_root, camera_folder=args.camera_folder)
    print(f"Found {len(scene_dirs)} scene folders under {output_root / args.camera_folder}")

    rows = []
    missing = []

    for scene_dir in scene_dirs:
        # scene_dir ends with .../<env>/<density>/<material>
        env_shape = scene_dir.parent.parent.name
        density = scene_dir.parent.name
        material = scene_dir.name

        fps_path = scene_dir / "fps.txt"
        results_path = scene_dir / "results.json"

        if not results_path.exists():
            missing.append((scene_dir, "results.json missing"))
            continue

        fps_value = ""
        count_value = ""
        if fps_path.exists():
            fps_value, count_value = parse_fps_txt(fps_path)
        else:
            missing.append((scene_dir, "fps.txt missing"))

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        best_key = pick_best_key(results)
        if best_key is None or best_key not in results:
            missing.append((scene_dir, "results.json empty/invalid"))
            continue

        entry = results[best_key]
        # Tolerate missing fields
        psnr = entry.get("PSNR", "")
        ssim = entry.get("SSIM", "")
        lpips = entry.get("LPIPS", "")

        rows.append({
            "env_shape": env_shape,
            "density_sampling": density,
            "material": material,
            "scene_rel": str(scene_dir.relative_to(output_root)),
            "key": best_key,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips,
            "fps": fps_value,
            "count": count_value,
        })

    # Write table
    delimiter = "\t" if args.tsv else ","
    fieldnames = ["env_shape", "density_sampling", "material", "scene_rel", "key", "PSNR", "SSIM", "LPIPS", "fps", "count"]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to: {out_file}")

    if missing:
        print("\nWarnings (missing files):")
        for scene_dir, reason in missing:
            print(f"  - {scene_dir}: {reason}")


if __name__ == "__main__":
    main()

