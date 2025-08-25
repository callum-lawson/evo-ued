#!/usr/bin/env python3
# pyright: reportMissingImports=false
import argparse
import sys
from pathlib import Path
from typing import Optional


# The examples save: states, cum_rewards, episode_lengths, levels
# We'll extract a tidy table with columns:
# run_name, seed, attempt, level_index, level_name, cum_reward, episode_length, checkpoint


def parse_run_seed(path: Path) -> tuple[str, str]:
    parts = path.resolve().parts
    if "results" in parts:
        idx = parts.index("results")
        if len(parts) > idx + 2:
            return parts[idx + 1], parts[idx + 2]
    return "unknown_run", "unknown_seed"


def infer_checkpoint(npz_path: Path) -> Optional[int]:
    # Mirror results/<run>/<seed>/ to checkpoints/<run>/<seed>/models/
    parts = list(npz_path.resolve().parts)
    if "results" not in parts:
        return None
    idx = parts.index("results")
    mirrored = Path(*parts[:idx], "checkpoints", *parts[idx + 1:-1])  # drop filename
    models_dir = mirrored / "models"
    if not models_dir.exists() or not models_dir.is_dir():
        return None
    steps = []
    for p in models_dir.iterdir():
        if p.is_dir():
            try:
                steps.append(int(p.name))
            except ValueError:
                continue
    return max(steps) if steps else None


def extract_single_npz(npz_path: Path, out_dir: Optional[Path], force: bool) -> Optional[Path]:
    try:
        import numpy as np  # lazy import
    except ImportError as e:
        print(f"NumPy is required to read {npz_path}: {e}", file=sys.stderr)
        return None

    if out_dir is None:
        out_dir = npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "results_extracted.parquet"
    if out_path.exists() and not force:
        print(f"Skip {npz_path} -> exists: {out_path}")
        return out_path

    data = np.load(str(npz_path), allow_pickle=True)
    required = ["cum_rewards", "episode_lengths", "levels"]
    for k in required:
        if k not in data.files:
            print(f"Missing key '{k}' in {npz_path}", file=sys.stderr)
            return None

    cum_rewards = np.asarray(data["cum_rewards"])  # (A, L)
    episode_lengths = np.asarray(data["episode_lengths"])  # (A, L)
    levels = np.asarray(data["levels"])  # (L,)

    if cum_rewards.ndim == 1:
        cum_rewards = cum_rewards[None, :]
    if episode_lengths.ndim == 1:
        episode_lengths = episode_lengths[None, :]

    attempts, num_levels = cum_rewards.shape

    try:
        level_names = levels.astype(str)
    except (TypeError, ValueError):
        level_names = np.array([str(x) for x in levels])

    run_name, seed = parse_run_seed(npz_path)
    checkpoint = infer_checkpoint(npz_path)

    rows = []
    for a in range(attempts):
        for l in range(num_levels):
            rows.append(
                (
                    run_name,
                    seed,
                    a,
                    l,
                    level_names[l],
                    float(cum_rewards[a, l]),
                    int(episode_lengths[a, l]),
                    checkpoint if checkpoint is not None else None,
                    str(npz_path),
                )
            )

    try:
        import pandas as pd  # lazy import
    except ImportError as e:
        print(
            "pandas (and a Parquet engine like pyarrow or fastparquet) are required for Parquet output: "
            f"{e}",
            file=sys.stderr,
        )
        return None

    df = pd.DataFrame(
        rows,
        columns=[
            "run_name",
            "seed",
            "attempt",
            "level_index",
            "level_name",
            "cum_reward",
            "episode_length",
            "checkpoint",
            "source_npz",
        ],
    )

    df.to_parquet(out_path, index=False)
    print(f"Wrote Parquet: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract results.npz files to a Parquet table per file.")
    parser.add_argument("--results_root", default="./results", help="Root directory to search for results.npz")
    parser.add_argument("--output_dir", default=None, help="Optional directory to write extracted files. Defaults to same dir as each npz.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted files instead of skipping")

    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    if not results_root.exists():
        print(f"Results root not found: {results_root}", file=sys.stderr)
        return 1

    npz_files = sorted(results_root.rglob("results.npz"))
    if not npz_files:
        print(f"No results.npz files found under {results_root}")
        return 0

    num_ok = 0
    num_fail = 0

    for npz_path in npz_files:
        try:
            out = extract_single_npz(npz_path, output_dir, force=args.force)
            if out is not None:
                num_ok += 1
            else:
                num_fail += 1
        except (RuntimeError, ValueError, OSError, ImportError) as e:
            print(f"ERROR extracting {npz_path}: {e}", file=sys.stderr)
            num_fail += 1

    print(f"Done extracting. Success: {num_ok}, Failed: {num_fail}")
    return 0 if num_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
