#!/usr/bin/env python3
import argparse
import sys
import subprocess
import json
from pathlib import Path


def find_checkpoint_dirs(checkpoints_root: Path):
    if not checkpoints_root.exists():
        return []
    # Expect structure: checkpoints/<run>/<seed>
    dirs = []
    for run_dir in sorted(d for d in checkpoints_root.iterdir() if d.is_dir()):
        for seed_dir in sorted(d for d in run_dir.iterdir() if d.is_dir()):
            dirs.append(seed_dir)
    return dirs


def corresponding_results_path(seed_checkpoint_dir: Path) -> Path:
    # Replace 'checkpoints' with 'results' and append results.npz
    results_dir = Path(str(seed_checkpoint_dir).replace("checkpoints", "results", 1))
    return results_dir / "results.npz"


def detect_example_script(seed_checkpoint_dir: Path, default_script: Path) -> Path:
    """Detect the right example script based on the saved config.json.

    Rules:
    - ACCEL: if use_accel == true -> use maze_plr.py
    - PLR: if PLR-specific keys present (e.g., score_function, replay_prob, level_buffer_capacity,
      staleness_coeff, topk_k, prioritization) -> use maze_plr.py
    - DR: otherwise -> use maze_dr.py
    """
    cfg_path = seed_checkpoint_dir / "config.json"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return default_script

    examples_dir = default_script.parent
    plr_script = examples_dir / "maze_plr.py"
    dr_script = examples_dir / "maze_dr.py"

    # ACCEL case
    if bool(cfg.get("use_accel", False)):
        return plr_script if plr_script.exists() else default_script

    # PLR heuristics based on presence of known PLR keys
    plr_keys = {
        "score_function",
        "replay_prob",
        "level_buffer_capacity",
        "staleness_coeff",
        "topk_k",
        "prioritization",
        "buffer_duplicate_check",
    }
    if any(k in cfg for k in plr_keys):
        return plr_script if plr_script.exists() else default_script

    # Otherwise assume DR
    return dr_script if dr_script.exists() else default_script


def run_eval(example_script: Path, checkpoint_dir: Path, pass_args: list[str]) -> int:
    # Default to 250 eval attempts unless user overrides
    has_eval_attempts = any(
        (arg == "--eval_num_attempts") or arg.startswith("--eval_num_attempts=") for arg in pass_args
    )
    cmd = [
        sys.executable,
        str(example_script),
        "--mode",
        "eval",
        "--checkpoint_directory",
        str(checkpoint_dir),
    ]
    if not has_eval_attempts:
        cmd += ["--eval_num_attempts", "250"]
    cmd += pass_args

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        print(
            f"ERROR evaluating {checkpoint_dir} using {example_script} (exit {proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            file=sys.stderr,
        )
    else:
        # Stream some stdout summary
        print(proc.stdout)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints under checkpoints/<run>/<seed>.")
    parser.add_argument("--checkpoints_root", default="./checkpoints", help="Root directory containing run/seed checkpoints")
    parser.add_argument("--example_script", default="examples/maze_plr.py", help="Default evaluation entry script to call if auto-detect fails")
    parser.add_argument("--no_auto_detect", action="store_true", help="Disable auto-detection of example script from config.json")
    parser.add_argument(
        "--checkpoint_to_eval",
        type=int,
        default=118,
        help="Checkpoint step to evaluate; forwarded to the example script (default: 118)",
    )
    # Any additional args after '--' will be forwarded verbatim to the eval script
    parser.add_argument("pass_through", nargs=argparse.REMAINDER, help="Arguments to pass through to the eval script (e.g. --checkpoint_to_eval 118)")

    args = parser.parse_args()

    checkpoints_root = Path(args.checkpoints_root).resolve()
    default_example_script = Path(args.example_script).resolve()

    # Normalize pass-through: strip a leading '--' separator if present
    pass_args = list(args.pass_through or [])
    if pass_args and pass_args[0] == "--":
        pass_args = pass_args[1:]

    # Ensure a checkpoint is specified unless already provided via pass-through
    has_checkpoint_to_eval = any(
        (arg == "--checkpoint_to_eval") or arg.startswith("--checkpoint_to_eval=") for arg in pass_args
    )
    if not has_checkpoint_to_eval:
        pass_args += ["--checkpoint_to_eval", str(args.checkpoint_to_eval)]

    # Ensure default example script exists
    if not default_example_script.exists():
        print(f"Default example script not found: {default_example_script}", file=sys.stderr)
        sys.exit(1)

    seed_dirs = find_checkpoint_dirs(checkpoints_root)
    if not seed_dirs:
        print(f"No checkpoint directories found under {checkpoints_root}")
        return 0

    num_skipped = 0
    num_success = 0
    num_failed = 0

    for seed_dir in seed_dirs:
        results_npz = corresponding_results_path(seed_dir)
        if results_npz.exists():
            print(f"Skip {seed_dir} -> results exists at {results_npz}")
            num_skipped += 1
            continue

        # Ensure results directory exists to mirror script expectation
        results_dir = results_npz.parent
        results_dir.mkdir(parents=True, exist_ok=True)

        example_script = (
            default_example_script
            if args.no_auto_detect
            else detect_example_script(seed_dir, default_example_script)
        )

        rc = run_eval(example_script, seed_dir, pass_args)
        if rc == 0 and results_npz.exists():
            num_success += 1
        elif rc == 0 and not results_npz.exists():
            print(f"WARNING: Eval succeeded but results file not found at {results_npz}", file=sys.stderr)
            num_failed += 1
        else:
            num_failed += 1

    print(f"Done. Success: {num_success}, Skipped: {num_skipped}, Failed: {num_failed}")
    return 0 if num_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
