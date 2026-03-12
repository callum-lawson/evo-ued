#!/usr/bin/env python3
"""Generate initial trade-off analysis outputs for the DR baseline seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.training_analyses import (
    compute_maze_correlations_across_seeds,
    compute_seed_maze_matrix,
    load_checkpoint_results_for_runs,
    select_representative_seeds,
    select_tradeoff_pairs,
    summarize_dr_baseline_seed_maze_results,
)
from scripts.training_plots import (
    plot_seed_maze_heatmap,
    plot_seed_profiles,
    plot_tradeoff_scatter,
)


def _save_plot(ax, path: Path) -> None:
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table suffix: {path.suffix}")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "group"


def _load_group_runs(
    config_path: Path, group_name: str
) -> tuple[list[str], int | None]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    groups = cfg.get("groups", {})
    if group_name not in groups:
        raise ValueError(f"Group '{group_name}' not found in {config_path}")
    run_names = [str(name) for name in groups[group_name]]
    checkpoint = cfg.get("checkpoint")
    return run_names, int(checkpoint) if checkpoint is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate checkpoint-result trade-off analysis outputs for a configured run group."
    )
    parser.add_argument(
        "--config",
        default="./config/eval250_updates30k_lr1e-4.json",
        help="JSON config containing a 'groups' mapping.",
    )
    parser.add_argument(
        "--group",
        default="DR (baseline)",
        help="Group name from the config file to analyze.",
    )
    parser.add_argument("--results_root", default="./results")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for summary tables and plots.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint to analyze. Defaults to the config checkpoint if present.",
    )
    parser.add_argument("--top_tradeoff_pairs", type=int, default=3)
    parser.add_argument(
        "--expected_num_seeds",
        type=int,
        default=None,
        help="Expected number of runs/seeds. Defaults to the size of the selected group.",
    )
    parser.add_argument("--expected_num_mazes", type=int, default=8)
    parser.add_argument("--expected_attempts", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = Path(args.config).resolve()
    run_names, config_checkpoint = _load_group_runs(config_path, args.group)
    checkpoint = args.checkpoint if args.checkpoint is not None else config_checkpoint
    if checkpoint is None:
        raise ValueError("No checkpoint provided and no checkpoint found in config.")

    if args.output_dir is None:
        output_dir = (
            Path(args.results_root).resolve()
            / "analysis"
            / config_path.stem
            / f"{_slugify(args.group)}_ckpt{checkpoint}"
        )
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    attempt_df = load_checkpoint_results_for_runs(
        run_names=run_names,
        results_root=args.results_root,
        checkpoint=checkpoint,
    )
    if attempt_df.empty:
        raise ValueError("No checkpoint results found for the requested group.")

    summary_df = summarize_dr_baseline_seed_maze_results(attempt_df)
    seed_maze_matrix = compute_seed_maze_matrix(summary_df, value_col="mean_return")
    solve_rate_matrix = compute_seed_maze_matrix(summary_df, value_col="solve_rate")
    maze_corr = compute_maze_correlations_across_seeds(
        summary_df, value_col="mean_return"
    )
    tradeoff_pairs = select_tradeoff_pairs(
        summary_df,
        value_col="mean_return",
        top_n=args.top_tradeoff_pairs,
    )
    representative_seeds = select_representative_seeds(
        summary_df,
        value_col="mean_return",
    )

    num_seeds = summary_df["seed"].nunique()
    num_mazes = summary_df["maze"].nunique()
    attempt_counts = sorted(summary_df["num_attempts"].dropna().unique().tolist())
    expected_num_seeds = (
        args.expected_num_seeds
        if args.expected_num_seeds is not None
        else len(run_names)
    )
    if num_seeds != expected_num_seeds:
        raise ValueError(f"Expected {expected_num_seeds} seeds, found {num_seeds}.")
    if num_mazes != args.expected_num_mazes:
        raise ValueError(
            f"Expected {args.expected_num_mazes} mazes, found {num_mazes}."
        )
    if attempt_counts != [args.expected_attempts]:
        raise ValueError(
            f"Expected num_attempts == {args.expected_attempts}, found {attempt_counts}."
        )

    _write_table(attempt_df, output_dir / "attempt_level.parquet")
    _write_table(summary_df, output_dir / "seed_maze_summary.parquet")
    seed_maze_matrix.to_csv(output_dir / "seed_maze_mean_return_matrix.csv")
    solve_rate_matrix.to_csv(output_dir / "seed_maze_solve_rate_matrix.csv")
    maze_corr.to_csv(output_dir / "maze_correlations_mean_return.csv")
    tradeoff_pairs.to_csv(output_dir / "tradeoff_pairs_mean_return.csv", index=False)

    heatmap_ax = plot_seed_maze_heatmap(
        seed_maze_matrix,
        title=f"{args.group} by maze (mean return)",
        cmap="viridis",
        annotate=False,
    )
    _save_plot(heatmap_ax, output_dir / "seed_maze_heatmap_mean_return.png")

    solve_ax = plot_seed_maze_heatmap(
        solve_rate_matrix,
        title=f"{args.group} by maze (solve rate)",
        cmap="mako",
        annotate=False,
    )
    _save_plot(solve_ax, output_dir / "seed_maze_heatmap_solve_rate.png")

    corr_ax = plot_seed_maze_heatmap(
        maze_corr,
        title=f"Maze correlations across {args.group}",
        cmap="vlag",
        annotate=True,
    )
    _save_plot(corr_ax, output_dir / "maze_correlations_mean_return.png")

    profile_ax = plot_seed_profiles(
        summary_df,
        seeds=representative_seeds,
        value_col="mean_return",
        title=f"Representative {args.group} profiles",
    )
    _save_plot(profile_ax, output_dir / "representative_seed_profiles_mean_return.png")

    for _, row in tradeoff_pairs.iterrows():
        maze_x = str(row["maze_x"])
        maze_y = str(row["maze_y"])
        scatter_ax = plot_tradeoff_scatter(
            summary_df,
            maze_x=maze_x,
            maze_y=maze_y,
            value_col="mean_return",
            title=f"Trade-off scatter: {maze_x} vs {maze_y}",
        )
        safe_x = maze_x.replace("/", "_")
        safe_y = maze_y.replace("/", "_")
        _save_plot(
            scatter_ax,
            output_dir / f"tradeoff_scatter_{safe_x}__{safe_y}.png",
        )

    print(f"Wrote analysis outputs to {output_dir}")
    print(f"Group: {args.group}")
    print(f"Checkpoint: {checkpoint}")
    print(
        f"Seeds: {num_seeds}, Mazes: {num_mazes}, Attempts per maze: {attempt_counts}"
    )
    print(f"Representative seeds: {representative_seeds}")
    if not tradeoff_pairs.empty:
        print("Top trade-off pairs:")
        print(tradeoff_pairs.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
