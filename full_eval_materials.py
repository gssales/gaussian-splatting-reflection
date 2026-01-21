import os
import sys
import time
import json
import shlex
import platform
import subprocess
from pathlib import Path
from argparse import ArgumentParser

materials_scenes = [
    "camera_regular/forest_env_sphere/dense_rand/diffuse","camera_regular/forest_env_sphere/dense_rand/glossy","camera_regular/forest_env_sphere/dense_rand/metal","camera_regular/forest_env_sphere/dense_rand/mirror","camera_regular/forest_env_sphere/dense_rand/black","camera_regular/forest_env_sphere/dense_rand/white",
    "camera_regular/forest_env_sphere/sparse_rand/diffuse","camera_regular/forest_env_sphere/sparse_rand/glossy","camera_regular/forest_env_sphere/sparse_rand/metal","camera_regular/forest_env_sphere/sparse_rand/mirror","camera_regular/forest_env_sphere/sparse_rand/black","camera_regular/forest_env_sphere/sparse_rand/white",
    "camera_regular/forest_env_sphere/dense_regular/diffuse","camera_regular/forest_env_sphere/dense_regular/glossy","camera_regular/forest_env_sphere/dense_regular/metal","camera_regular/forest_env_sphere/dense_regular/mirror","camera_regular/forest_env_sphere/dense_regular/black","camera_regular/forest_env_sphere/dense_regular/white",
    "camera_regular/forest_env_sphere/sparse_regular/diffuse","camera_regular/forest_env_sphere/sparse_regular/glossy","camera_regular/forest_env_sphere/sparse_regular/metal","camera_regular/forest_env_sphere/sparse_regular/mirror","camera_regular/forest_env_sphere/sparse_regular/black","camera_regular/forest_env_sphere/sparse_regular/white",
    
    "camera_regular/forest_env_cube/dense_rand/diffuse","camera_regular/forest_env_cube/dense_rand/glossy","camera_regular/forest_env_cube/dense_rand/metal","camera_regular/forest_env_cube/dense_rand/mirror","camera_regular/forest_env_cube/dense_rand/black","camera_regular/forest_env_cube/dense_rand/white",
    "camera_regular/forest_env_cube/sparse_rand/diffuse","camera_regular/forest_env_cube/sparse_rand/glossy","camera_regular/forest_env_cube/sparse_rand/metal","camera_regular/forest_env_cube/sparse_rand/mirror","camera_regular/forest_env_cube/sparse_rand/black","camera_regular/forest_env_cube/sparse_rand/white",
    "camera_regular/forest_env_cube/dense_regular/diffuse","camera_regular/forest_env_cube/dense_regular/glossy","camera_regular/forest_env_cube/dense_regular/metal","camera_regular/forest_env_cube/dense_regular/mirror","camera_regular/forest_env_cube/dense_regular/black","camera_regular/forest_env_cube/dense_regular/white",
    "camera_regular/forest_env_cube/sparse_regular/diffuse","camera_regular/forest_env_cube/sparse_regular/glossy","camera_regular/forest_env_cube/sparse_regular/metal","camera_regular/forest_env_cube/sparse_regular/mirror","camera_regular/forest_env_cube/sparse_regular/black","camera_regular/forest_env_cube/sparse_regular/white",
    
    "camera_regular/constant_env_sphere/dense_rand/glossy","camera_regular/constant_env_sphere/dense_rand/metal","camera_regular/constant_env_sphere/dense_rand/mirror","camera_regular/constant_env_sphere/dense_rand/black","camera_regular/constant_env_sphere/dense_rand/white",
    "camera_regular/constant_env_sphere/sparse_rand/glossy","camera_regular/constant_env_sphere/sparse_rand/metal","camera_regular/constant_env_sphere/sparse_rand/mirror","camera_regular/constant_env_sphere/sparse_rand/black","camera_regular/constant_env_sphere/sparse_rand/white",
    "camera_regular/constant_env_sphere/dense_regular/glossy","camera_regular/constant_env_sphere/dense_regular/metal","camera_regular/constant_env_sphere/dense_regular/mirror","camera_regular/constant_env_sphere/dense_regular/black","camera_regular/constant_env_sphere/dense_regular/white",
    "camera_regular/constant_env_sphere/sparse_regular/glossy","camera_regular/constant_env_sphere/sparse_regular/metal","camera_regular/constant_env_sphere/sparse_regular/mirror","camera_regular/constant_env_sphere/sparse_regular/black","camera_regular/constant_env_sphere/sparse_regular/white",
    
    "camera_regular/constant_env_cube/dense_rand/glossy","camera_regular/constant_env_cube/dense_rand/metal","camera_regular/constant_env_cube/dense_rand/mirror","camera_regular/constant_env_cube/dense_rand/black","camera_regular/constant_env_cube/dense_rand/white",
    "camera_regular/constant_env_cube/sparse_rand/glossy","camera_regular/constant_env_cube/sparse_rand/metal","camera_regular/constant_env_cube/sparse_rand/mirror","camera_regular/constant_env_cube/sparse_rand/black","camera_regular/constant_env_cube/sparse_rand/white",
    "camera_regular/constant_env_cube/dense_regular/glossy","camera_regular/constant_env_cube/dense_regular/metal","camera_regular/constant_env_cube/dense_regular/mirror","camera_regular/constant_env_cube/dense_regular/black","camera_regular/constant_env_cube/dense_regular/white",
    "camera_regular/constant_env_cube/sparse_regular/glossy","camera_regular/constant_env_cube/sparse_regular/metal","camera_regular/constant_env_cube/sparse_regular/mirror","camera_regular/constant_env_cube/sparse_regular/black","camera_regular/constant_env_cube/sparse_regular/white"]


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def hr():
    return "-" * 80


def log(msg: str):
    print(f"[{now_str()}] {msg}", flush=True)


def fmt_minutes(seconds: float) -> str:
    return f"{seconds/60.0:.2f} min"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_cmd_capture(cmd: str, log_path: Path, dry_run: bool = False):
    """
    TQDM-safe runner:
    - Does NOT stream to terminal
    - Captures stdout/stderr
    - Writes everything to log_path
    - On failure, prints a short tail + log path
    """
    ensure_dir(log_path.parent)

    log(hr())
    log(f"RUN: {cmd}")
    log(f"LOG: {log_path}")

    if dry_run:
        log("DRY-RUN: command not executed.")
        return 0, 0.0

    start = time.time()
    res = subprocess.run(
        cmd,
        shell=True,                 # your original used os.system; keep shell style
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - start

    # Write full output to file (keeps terminal clean for tqdm)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(res.stdout or "")

    log(f"EXIT CODE: {res.returncode} | ELAPSED: {fmt_minutes(elapsed)}")

    if res.returncode != 0:
        # Print a short tail for quick diagnosis without destroying tqdm formatting
        tail_lines = 40
        out = (res.stdout or "").splitlines()
        tail = "\n".join(out[-tail_lines:]) if out else "(no output captured)"
        log(f"[!] Command failed. Last {tail_lines} lines:\n{tail}")
        log(f"[!] Full log saved at: {log_path}")

    return res.returncode, elapsed


def main():
    parser = ArgumentParser(description="Full evaluation script parameters (verbose, tqdm-safe)")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_rendering", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--output_path", default="/mnt/output/ours/materials_2/")
    parser.add_argument("--materials", type=str, default="/mnt/data/materials")
    parser.add_argument("--iteration", type=int, default=31000)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--logs_dir", default="logs", help="Logs folder inside output_path")

    args = parser.parse_args()

    output_path = Path(args.output_path)
    materials_root = Path(args.materials)
    logs_root = output_path / args.logs_dir
    ensure_dir(output_path)
    ensure_dir(logs_root)

    all_scenes = list(materials_scenes)

    log(hr())
    log("STARTING MATERIALS PIPELINE (VERBOSE, TQDM-SAFE)")
    log(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}")
    log(f"Materials root: {materials_root}")
    log(f"Output root:    {output_path}")
    log(f"Logs root:      {logs_root}")
    log(f"Scenes:         {len(all_scenes)}")
    log(f"skip_training={args.skip_training}, skip_rendering={args.skip_rendering}, skip_metrics={args.skip_metrics}")
    log(f"render_iteration={args.iteration}")
    if args.dry_run:
        log("NOTE: --dry_run enabled")
    log(hr())

    train_common_args = (
        " --disable_viewer --quiet --eval --normal_propagation --color_sabotage "
        "--init_until_iter 20 --densification_interval_when_prop 500 --random_background_color"
    )
    render_common_args = " --quiet --eval --skip_train"
    render_extra_args = " --render_normals --render_refl -w"

    # --- TRAINING ---
    if not args.skip_training:
        log("STAGE 1/3: TRAINING")
        scene_times = {}
        t0_all = time.time()

        for idx, scene in enumerate(materials_scenes, start=1):
            scene_rel = Path(scene)
            source = materials_root / scene_rel
            out_model_dir = output_path / scene_rel

            log(hr())
            log(f"[TRAIN {idx}/{len(materials_scenes)}] {scene}")
            log(f"  Source: {source}")
            log(f"  Output: {out_model_dir}")

            cmd = f"python train.py -s {shlex.quote(str(source))} -m {shlex.quote(str(out_model_dir))}{train_common_args}"
            log_path = logs_root / "train" / scene_rel / "train.log"

            rc, elapsed = run_cmd_capture(cmd, log_path=log_path, dry_run=args.dry_run)

            scene_times[scene] = {
                "return_code": rc,
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60.0,
                "log": str(log_path),
            }

            if rc != 0:
                log(f"[!] Training failed for '{scene}' (exit {rc}). Continuing...")

        elapsed_all = time.time() - t0_all
        timing_txt = output_path / "materials_timing.txt"
        timing_json = output_path / "materials_timing.json"

        with open(timing_txt, "w", encoding="utf-8") as f:
            f.write(f"materials: {elapsed_all/60.0:.2f} minutes\n")

        with open(timing_json, "w", encoding="utf-8") as f:
            json.dump(scene_times, f, indent=2)

        log(hr())
        log(f"TRAINING DONE. Total elapsed: {fmt_minutes(elapsed_all)}")
        log(f"Wrote: {timing_txt}")
        log(f"Wrote: {timing_json}")

    # --- RENDERING ---
    if not args.skip_rendering:
        log("STAGE 2/3: RENDERING")

        for idx, scene in enumerate(all_scenes, start=1):
            scene_rel = Path(scene)
            source = materials_root / scene_rel
            out_model_dir = output_path / scene_rel

            source_with_flags = f"{source} --iteration {args.iteration}{render_extra_args}"

            log(hr())
            log(f"[RENDER {idx}/{len(all_scenes)}] {scene}")
            log(f"  Source+flags: {source_with_flags}")
            log(f"  Output:       {out_model_dir}")

            cmd = f"python render.py -s {str(source_with_flags)} -m {str(out_model_dir)}{render_common_args}"
            log_path = logs_root / "render" / scene_rel / "render.log"

            rc, _ = run_cmd_capture(cmd, log_path=log_path, dry_run=args.dry_run)
            if rc != 0:
                log(f"[!] Rendering failed for '{scene}' (exit {rc}). Continuing...")

        log(hr())
        log("RENDERING DONE.")

    # --- METRICS ---
    if not args.skip_metrics:
        log("STAGE 3/3: METRICS")

        scene_paths = [output_path / Path(scene) for scene in all_scenes]
        scenes_string = " ".join(shlex.quote(str(p)) for p in scene_paths)

        cmd = f"python metrics.py -m {scenes_string}"
        log_path = logs_root / "metrics.log"

        rc, _ = run_cmd_capture(cmd, log_path=log_path, dry_run=args.dry_run)
        if rc != 0:
            log(f"[!] Metrics failed (exit {rc}). See log: {log_path}")
        else:
            log("METRICS DONE.")

    log(hr())
    log("ALL DONE.")
    log(hr())


if __name__ == "__main__":
    main()

