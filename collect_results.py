import re
import json
import csv
from pathlib import Path
from argparse import ArgumentParser

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


def find_scene_dirs(base_path: Path):
    if (base_path / "cfg_args").exists():
        return [base_path]

    scene_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            sub = find_scene_dirs(item)
            if sub is not None:
                scene_dirs.extend(sub)
    return scene_dirs


def main():
    parser = ArgumentParser(description="Collect PSNR/SSIM/LPIPS + FPS into a CSV across all scenes.")
    parser.add_argument("--output_path", default=r"E:\\output\\ours\\eval",
                        help="Root eval output path (contains camera_regular/...)")
    parser.add_argument("--csv_name", default="results_all.csv",
                        help="Output CSV filename (written inside output_path)")
    parser.add_argument("--tsv", action="store_true",
                        help="Write TSV instead of CSV")
    args, _ = parser.parse_known_args()

    output_root = Path(args.output_path)
    out_file = output_root / args.csv_name

    scene_dirs = find_scene_dirs(output_root)
    print(f"Found {len(scene_dirs)} scene folders under {output_root}")

    rows = []
    missing = []

    for scene_dir in scene_dirs:
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
            "scene": str(scene_dir.relative_to(output_root)),
            "key": best_key,
            "PSNR": str(psnr).replace(".", ","),
            "SSIM": str(ssim).replace(".", ","),
            "LPIPS": str(lpips).replace(".", ","),
            "fps": str(fps_value).replace(".", ","),
            "count": count_value,
        })

    # Write table
    delimiter = "\t" if args.tsv else ","
    fieldnames = ["scene", "key", "PSNR", "SSIM", "LPIPS", "fps", "count"]

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

