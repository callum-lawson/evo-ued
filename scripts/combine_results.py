#!/usr/bin/env python3
# pyright: reportMissingImports=false
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def load_config(config_path: Path) -> Tuple[List[dict], dict]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    entries = cfg.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("Config must contain a non-empty 'entries' list")
    for e in entries:
        if "run_name" not in e:
            raise ValueError("Each entry must include 'run_name'")
    return entries, cfg


def resolve_output_path(
    cfg: dict, cli_output: Optional[str], default_dir: Path
) -> Path:
    # Precedence: CLI --output > cfg['output'] > (cfg['output_dir']+cfg['output_filename'])
    if cli_output:
        out = Path(cli_output)
        return out.resolve()
    if "output" in cfg:
        return Path(cfg["output"]).resolve()
    if "output_dir" in cfg and "output_filename" in cfg:
        return (Path(cfg["output_dir"]) / cfg["output_filename"]).resolve()
    # If nothing specified, default to default_dir/combined.parquet
    return (default_dir / "combined.parquet").resolve()


def find_candidate_parquets(
    results_root: Path, run_name: str, seed: Optional[str], checkpoint: Optional[int]
) -> List[Path]:
    base = results_root / run_name
    candidates: List[Path] = []

    if seed is not None:
        seed_dir = base / seed
        # Prefer nested checkpoint directories
        if checkpoint is not None:
            candidates = [seed_dir / str(checkpoint) / "results_extracted.parquet"]
        else:
            candidates = sorted(seed_dir.glob("*/results_extracted.parquet"))
        # Fallback to legacy flat location
        legacy = seed_dir / "results_extracted.parquet"
        if legacy.exists():
            candidates.append(legacy)
        return candidates

    # No seed specified: search across seeds
    if checkpoint is not None:
        candidates = sorted(base.glob(f"*/{checkpoint}/results_extracted.parquet"))
    else:
        candidates = sorted(base.glob("*/*/results_extracted.parquet"))
    # Fallback to legacy flat per-seed locations
    candidates += sorted(base.glob("*/results_extracted.parquet"))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine extracted Parquet files into one.")
    parser.add_argument("--config", required=True, help="Path to JSON config specifying entries (run_name, optional seed/metadata). May also include 'output' or ('output_dir' + 'output_filename').")
    parser.add_argument("--results_root", default="./results", help="Base results directory")
    parser.add_argument("--output", required=False, help="Output Parquet path (overrides config)")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first missing/failed file instead of skipping")
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it exists")

    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    results_root = Path(args.results_root).resolve()

    try:
        entries, cfg = load_config(config_path)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(f"Invalid config: {e}", file=sys.stderr)
        return 1

    # Determine output path (from CLI or config)
    default_combined_dir = (Path.cwd() / "results/combined").resolve()
    output_path = resolve_output_path(cfg, args.output, default_dir=default_combined_dir)

    if output_path.exists() and not args.force:
        print(
            f"Refusing to overwrite existing file: {output_path}.\n"
            "Re-run with --force or change the output name in the config (--output or 'output').",
            file=sys.stderr,
        )
        return 2

    try:
        import pandas as pd  # lazy import
    except ImportError as e:
        print("pandas (and a Parquet engine like pyarrow/fastparquet) are required: " f"{e}", file=sys.stderr)
        return 1

    frames = []
    skipped: List[str] = []

    # Global checkpoint specified once at the top-level config (optional)
    global_checkpoint_opt = int(cfg["checkpoint"]) if "checkpoint" in cfg else None

    for e in entries:
        run_name = e["run_name"]
        seed_opt = None  # seed is no longer specified in configs; aggregate across seeds
        checkpoint_opt = (
            global_checkpoint_opt
            if global_checkpoint_opt is not None
            else int(e["checkpoint"]) if "checkpoint" in e else None
        )

        # Build list of candidate parquet paths (supports nested checkpoint directories)
        candidate_paths = find_candidate_parquets(
            results_root, run_name, seed_opt, checkpoint_opt
        )
        if not candidate_paths:
            msg = f"No extracted files found under {results_root / run_name}"
            if checkpoint_opt is not None:
                msg += f" (checkpoint={checkpoint_opt})"
            if args.fail_fast:
                print(msg, file=sys.stderr)
                return 1
            print(msg)
            skipped.append(str(results_root / run_name))
            continue

        for parquet_path in candidate_paths:
            if not parquet_path.exists():
                msg = f"Missing extracted file: {parquet_path}"
                if args.fail_fast:
                    print(msg, file=sys.stderr)
                    return 1
                print(msg)
                skipped.append(str(parquet_path))
                continue
            try:
                df = pd.read_parquet(parquet_path)
                # Extract seed from directory name; supports nested checkpoint dir
                parent_dir = parquet_path.parent
                parsed_seed = parent_dir.name
                try:
                    # If parent is numeric, it's the checkpoint; go up one for seed
                    int(parsed_seed)
                    parsed_seed = parent_dir.parent.name
                except ValueError:
                    pass
                if "seed" not in df.columns:
                    df["seed"] = parsed_seed
                # Ensure run_name column exists
                if "run_name" not in df.columns:
                    df["run_name"] = run_name
                # Attach optional metadata
                for k in ["algo", "checkpoint"]:
                    if k in e and k not in df.columns:
                        df[k] = e[k]
                # If checkpoint column still missing, fill from global checkpoint if provided
                if "checkpoint" not in df.columns and global_checkpoint_opt is not None:
                    df["checkpoint"] = global_checkpoint_opt
                frames.append(df)
            except (OSError, ValueError) as e_read:
                msg = f"ERROR reading {parquet_path}: {e_read}"
                if args.fail_fast:
                    print(msg, file=sys.stderr)
                    return 1
                print(msg)
                skipped.append(str(parquet_path))

    if not frames:
        print("No frames to combine.", file=sys.stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    print(f"Wrote combined Parquet: {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} files (missing or failed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
