#!/usr/bin/env python3
"""Download full history (all columns) for all wandb runs to local cache.

This script fetches complete history data from wandb and stores it locally
for offline analysis and Google Drive backup.

Usage:
    # Test with one run first (shows file size)
    python scripts/download_full_histories.py --test

    # Download all runs
    python scripts/download_full_histories.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils_wandb import WandbDataClient


ENTITY = "callumrlawson-pibbss"
PROJECT = "JAXUED_TEST"

# Config files that define all runs we care about
CONFIG_FILES = [
    "config/eval250_updates30k_lr1e-4.json",
    "config/eval250_updates30k_lr3e-4.json",
    "config/eval250_updates30k_lr5e-5.json",
    "config/eval250_updates50k_lr1e-4.json",
    "config/updates30k_lr_sweep.json",
]


def load_all_run_names(project_root: Path) -> Set[str]:
    """Load all run names from config files."""
    run_names: Set[str] = set()
    for config_file in CONFIG_FILES:
        path = project_root / config_file
        if not path.exists():
            print(f"Warning: Config file not found: {path}")
            continue
        with open(path, "r") as f:
            data = json.load(f)
        groups = data.get("groups", {})
        for runs in groups.values():
            run_names.update(runs)
    return run_names


def get_cached_run_ids(cache_dir: Path) -> Set[str]:
    """Get run IDs that are already cached."""
    if not cache_dir.exists():
        return set()
    return {d.name for d in cache_dir.iterdir() if d.is_dir()}


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def download_full_history(
    client: WandbDataClient,
    run_id: str,
    *,
    refresh: bool = False,
) -> tuple[bool, int]:
    """Download full history for a single run.

    Returns (success, file_size_bytes).
    """
    try:
        # Fetch with keys=None to get all columns
        data = client.fetch_run_data(run_id, keys=None, refresh=refresh)

        # Find the history file that was created
        run_cache_dir = client.cache_dir / client.entity / client.project / data.key.run_id
        parquet_files = list(run_cache_dir.glob("history_*.parquet"))

        if parquet_files:
            # Get the most recently modified parquet file
            latest = max(parquet_files, key=lambda p: p.stat().st_mtime)
            return True, latest.stat().st_size
        return True, 0
    except Exception as e:
        print(f"  Error: {e}")
        return False, 0


def test_single_run(client: WandbDataClient, run_names: Set[str]) -> None:
    """Test downloading one run to check file size."""
    print("\n=== Testing single run download ===\n")

    # Get all runs from wandb to map names to IDs
    all_runs = client.list_runs()
    name_to_run: Dict[str, object] = {}
    for run in all_runs:
        orig_name = None
        if hasattr(run, "config") and run.config:
            orig_name = run.config.get("run_name")
        if not orig_name:
            orig_name = getattr(run, "name", None)
        if orig_name and orig_name in run_names:
            name_to_run[orig_name] = run

    if not name_to_run:
        print("No matching runs found!")
        return

    # Pick one run to test
    test_name = next(iter(run_names))
    if test_name not in name_to_run:
        test_name = next(iter(name_to_run.keys()))

    run = name_to_run[test_name]
    run_id = getattr(run, "id", str(run))

    print(f"Testing run: {test_name} (ID: {run_id})")
    print("Downloading full history (all columns)...")

    success, size = download_full_history(client, run_id, refresh=True)

    if success:
        print(f"\nSuccess! File size: {format_size(size)}")
        estimated_total = size * len(run_names)
        print(f"Estimated total for {len(run_names)} runs: {format_size(estimated_total)}")
        print("\nTo download all runs, run without --test flag.")
    else:
        print("\nFailed to download test run.")


def download_all_runs(client: WandbDataClient, run_names: Set[str]) -> None:
    """Download full history for all runs."""
    print(f"\n=== Downloading full history for {len(run_names)} runs ===\n")

    # Get all runs from wandb to map names to IDs
    print("Fetching run list from wandb...")
    all_runs = client.list_runs()
    name_to_run: Dict[str, object] = {}
    for run in all_runs:
        orig_name = None
        if hasattr(run, "config") and run.config:
            orig_name = run.config.get("run_name")
        if not orig_name:
            orig_name = getattr(run, "name", None)
        if orig_name and orig_name in run_names:
            name_to_run[orig_name] = run

    print(f"Found {len(name_to_run)} matching runs in wandb")

    # Find missing runs
    missing = run_names - set(name_to_run.keys())
    if missing:
        print(f"\nWarning: {len(missing)} runs not found in wandb:")
        for name in sorted(missing)[:10]:
            print(f"  - {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Download each run
    total_size = 0
    success_count = 0
    fail_count = 0

    for i, (name, run) in enumerate(sorted(name_to_run.items()), 1):
        run_id = getattr(run, "id", str(run))
        print(f"[{i}/{len(name_to_run)}] {name} (ID: {run_id})... ", end="", flush=True)

        success, size = download_full_history(client, run_id, refresh=True)

        if success:
            print(f"{format_size(size)}")
            total_size += size
            success_count += 1
        else:
            print("FAILED")
            fail_count += 1

    print(f"\n=== Summary ===")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total size: {format_size(total_size)}")
    print(f"\nCache location: {client.cache_dir / ENTITY / PROJECT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download full wandb history for all runs"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test with a single run first to check file size",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory (default: ~/.cache/evo-ued/wandb)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Load all run names from config files
    run_names = load_all_run_names(project_root)
    print(f"Found {len(run_names)} unique run names in config files")

    # Create client with long TTL since we want to keep cached data
    client = WandbDataClient(
        ENTITY,
        PROJECT,
        cache_dir=args.cache_dir,
        cache_ttl_seconds=365 * 24 * 60 * 60,  # 1 year
        use_cache=True,
    )

    if args.test:
        test_single_run(client, run_names)
    else:
        download_all_runs(client, run_names)


if __name__ == "__main__":
    main()
