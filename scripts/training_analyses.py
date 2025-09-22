"""
Helpers for assembling W&B run histories into tidy DataFrames and aggregations.

This keeps notebooks focused on visualization while the data plumbing lives here.
"""

from __future__ import annotations

from typing import Optional, Sequence, Any, Dict, Tuple, List
from dataclasses import dataclass

import pandas as pd

import warnings

# In this section, we are going to build a pipeline:
# aggregate functions: by quantile, with the option to add the smoothing afterwards
# these functions should work EITHER at the evals level within a run (250 evaluations)
# OR when we've already aggregated those into runs (10 runs)


# ------------------------- Final-evaluation solve table -------------------------


# ------------------------------ Data assembly ------------------------------
# (Utilities consolidated here; use this module for performance table helpers.)


def compute_maze_similarity_linkage(
    df: pd.DataFrame,
    agg: str = "mean",
    method: str = "pearson",
    linkage_method: str = "average",
):
    """Compute maze similarity and hierarchical linkage.

    Returns a tuple (similarity_df, linkage_Z). The similarity matrix is
    [maze x maze] using the specified correlation method; the linkage is built
    from distance d = 1 - similarity, clipped to [0, 1] and NaNs filled.
    """
    from scipy.spatial.distance import squareform  # type: ignore
    from scipy.cluster.hierarchy import linkage  # type: ignore

    sim = compute_maze_similarity_overall(df, agg=agg, method=method)
    if sim is None or sim.empty:
        return pd.DataFrame(), None

    # Convert similarity to distance suitable for hierarchical clustering
    dist = (1.0 - sim).clip(lower=0).fillna(1.0)
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=linkage_method)
    return sim, Z


import math

from .utils_wandb import (
    WandbDataClient,
    _choose_step_key,
    _extract_original_name,
)


# ------------------------------ Per-run/per-maze tables ------------------------------


def final_values_per_run(
    df: pd.DataFrame,
    step_key: str,
    *,
    extra_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return one row per run with the final value at the max step.

    Input columns: [step_key, value, run_id, group] (+ optional extras).
    Output columns: [group, run_id, step, value] (+ extras if requested).
    """
    if df is None or df.empty:
        return pd.DataFrame(
            {
                "group": pd.Series(dtype="object"),
                "run_id": pd.Series(dtype="object"),
                "step": pd.Series(dtype="float64"),
                "value": pd.Series(dtype="float64"),
            }
        )

    sorted_df = df.sort_values(by=["run_id", step_key])
    idx = sorted_df.groupby("run_id")[step_key].idxmax()
    base_cols = ["group", "run_id", step_key, "value"]
    cols: List[str] = base_cols
    if extra_cols:
        # Keep only extras that are present to avoid KeyError
        extras_present = [c for c in extra_cols if c in sorted_df.columns]
        cols = base_cols + extras_present
    per_run = (
        sorted_df.loc[idx, cols]
        .rename(columns={step_key: "step"})
        .reset_index(drop=True)
    )
    return per_run


@dataclass(frozen=True)
class PerformanceBuildConfig:
    entity: str
    project: str
    runname_to_group: Dict[str, str]
    step_key: str
    metric_prefix: str  # e.g., "return/" or "solve_rate/"
    at_step: Optional[float] = None  # None => take last/max available
    use_cache: bool = True
    refresh: bool = False
    cache_ttl_seconds: int = 6 * 60 * 60


def _is_maze_metric_column(column: str, prefix: str) -> bool:
    if not column.startswith(prefix):
        return False
    tail = column[len(prefix) :]
    bad_suffixes = {"mean", "median", "std", "min", "max", "p25", "p75"}
    if tail in bad_suffixes:
        return False
    if any(s in tail.lower() for s in ("smooth", "ewm", "ema")):
        return False
    return True


def _value_at_step(
    df: pd.DataFrame, step_col: str, metric_col: str, at_step: Optional[float]
) -> Optional[float]:
    if metric_col not in df.columns:
        return None
    sdf = df[[step_col, metric_col]].dropna()
    if sdf.empty:
        return None
    # Convert to plain Python lists to avoid pandas typing issues
    step_list = [float(x) for x in pd.Series(sdf[step_col]).to_list()]
    metric_list = [float(x) for x in pd.Series(sdf[metric_col]).to_list()]
    if not metric_list:
        return None
    if at_step is None:
        return float(metric_list[-1])
    leq_indices: List[int] = [i for i, s in enumerate(step_list) if s <= float(at_step)]
    if leq_indices:
        return float(metric_list[leq_indices[-1]])
    # Fallback: nearest step
    nearest_idx = min(
        range(len(step_list)), key=lambda i: abs(step_list[i] - float(at_step))
    )
    return float(metric_list[nearest_idx])


def _select_representative_runs(
    client: WandbDataClient,
    runname_to_group: Dict[str, str],
    step_key: str,
) -> List[Tuple[Any, str]]:
    """Return list of (run, group_label) selecting one best run per original name."""
    all_runs = list(client.list_runs())
    candidates_by_name: Dict[str, List[Any]] = {}
    label_by_name: Dict[str, str] = {}

    for run in all_runs:
        orig_name = _extract_original_name(run)
        if not orig_name:
            continue
        label = None
        for k in (
            orig_name,
            getattr(run, "name", None),
            getattr(run, "id", None),
            getattr(run, "group", None),
        ):
            if isinstance(k, str) and k in runname_to_group:
                label = runname_to_group[k]
                break
        if label is None:
            continue
        candidates_by_name.setdefault(orig_name, []).append(run)
        label_by_name[orig_name] = label

    def completeness_score(r: Any) -> float:
        s_obj = getattr(r, "summary", None)
        if s_obj is not None:
            keys = (step_key, "num_updates", "_step", "Step", "global_step")
            get_attr = None
            try:
                get_attr = s_obj.get  # type: ignore[attr-defined]
            except (AttributeError, KeyError):
                get_attr = None
            if callable(get_attr):
                for k in keys:
                    try:
                        v = get_attr(k, None)  # type: ignore[misc]
                        if isinstance(v, (int, float)):
                            return float(v)
                    except (TypeError, KeyError, ValueError):
                        continue
            for k in keys:
                try:
                    v = s_obj[k]  # type: ignore[index]
                    if isinstance(v, (int, float)):
                        return float(v)
                except (KeyError, TypeError, ValueError):
                    continue
        try:
            dfh = r.history(keys=[step_key], samples=500)
            if step_key in dfh.columns and not dfh.empty:
                return float(dfh[step_key].max())
            return float(len(dfh))
        except (AttributeError, RuntimeError, ValueError):
            return 0.0

    chosen: List[Tuple[Any, str]] = []
    for name, runs in candidates_by_name.items():
        if len(runs) == 1:
            chosen.append((runs[0], label_by_name[name]))
            continue
        scored = sorted(
            ((completeness_score(r), r) for r in runs), key=lambda x: x[0], reverse=True
        )
        best_score, best = scored[0]
        others = ", ".join(f"{getattr(r, 'id', '?')}:{s}" for s, r in scored[1:])
        warnings.warn(
            f"Duplicate runs for '{name}'. Using {getattr(best, 'id', '?')} (score={best_score}). Others: {others}"
        )
        chosen.append((best, label_by_name[name]))
    return chosen


def build_performance_table(cfg: PerformanceBuildConfig) -> pd.DataFrame:
    """Assemble long-form per-maze performance at a chosen timestep.

    Returns columns: [group, run_id, run_name, maze, value, step, step_col].
    """
    client = WandbDataClient(
        cfg.entity,
        cfg.project,
        cache_ttl_seconds=cfg.cache_ttl_seconds,
        use_cache=cfg.use_cache,
    )

    chosen = _select_representative_runs(client, cfg.runname_to_group, cfg.step_key)

    rows: List[Dict[str, object]] = []
    for run, label in chosen:
        rd = client.fetch_run_data(run.id, keys=None, samples=None, refresh=cfg.refresh)
        df = rd.history_df
        step_col = _choose_step_key(df.columns, cfg.step_key)
        if step_col is None:
            continue
        maze_cols = [
            c for c in df.columns if _is_maze_metric_column(c, cfg.metric_prefix)
        ]
        if not maze_cols:
            continue

        at_step = cfg.at_step
        if at_step is None:
            series = df[step_col].dropna()
            at_step = float(series.max()) if not series.empty else None

        rname = _extract_original_name(rd.run) or rd.run.id

        for col in maze_cols:
            val = _value_at_step(df, step_col, col, at_step)
            if val is None or (
                isinstance(val, float) and (math.isnan(val) or math.isinf(val))
            ):
                continue
            maze = col[len(cfg.metric_prefix) :]
            rows.append(
                {
                    "group": label,
                    "run_id": rd.key.run_id,
                    "run_name": rname,
                    "maze": maze,
                    "value": float(val),
                    "step": at_step,
                    "step_col": step_col,
                }
            )

    if not rows:
        return pd.DataFrame(
            {
                "group": pd.Series(dtype="object"),
                "run_id": pd.Series(dtype="object"),
                "run_name": pd.Series(dtype="object"),
                "maze": pd.Series(dtype="object"),
                "value": pd.Series(dtype="float64"),
                "step": pd.Series(dtype="float64"),
                "step_col": pd.Series(dtype="object"),
            }
        )
    return pd.DataFrame(rows)


# -------------------------------- Computations --------------------------------


def _aggregate_algo_maze(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    if df.empty:
        return df
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")
    func = agg
    return (
        df.groupby(["group", "maze"], as_index=False)["value"]
        .agg(func)
        .rename(columns={"value": agg})  # type: ignore[arg-type]
    )


def compute_maze_similarity_overall(
    df: pd.DataFrame, agg: str = "mean", method: str = "pearson"
) -> pd.DataFrame:
    """Similarity between mazes based on algorithm-aggregated performance."""
    if df.empty:
        return pd.DataFrame()
    agg_df = _aggregate_algo_maze(df, agg=agg)
    mat = agg_df.pivot(index="maze", columns="group", values=agg)
    mat = mat.dropna(how="all")
    if mat.empty:
        return pd.DataFrame()
    if method in {"pearson", "spearman"}:
        return mat.transpose().corr(method=method)  # type: ignore[arg-type]
    if method == "cosine":
        from numpy.linalg import norm  # type: ignore
        import numpy as np  # type: ignore

        X = mat.fillna(mat.mean()).to_numpy(dtype=float)
        n = X.shape[0]
        sims = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                a, b = X[i], X[j]
                denom = norm(a) * norm(b)
                sims[i, j] = float(a.dot(b) / denom) if denom else 0.0
        return pd.DataFrame(sims, index=mat.index, columns=mat.index)
    raise ValueError("method must be 'pearson', 'spearman', or 'cosine'")


def compute_maze_similarity_per_algorithm(
    df: pd.DataFrame, method: str = "pearson"
) -> Dict[str, pd.DataFrame]:
    """For each algorithm, similarity between mazes using across-seed variation."""
    result: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return result
    for algo, dfa in df.groupby("group"):
        pivot = dfa.pivot_table(
            index="run_name", columns="maze", values="value", aggfunc="mean"
        )
        pivot = pivot.dropna(axis=1, how="all")
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            result[algo] = pd.DataFrame()
            continue
        corr = pivot.corr(method=method)  # type: ignore[arg-type]
        result[algo] = corr
    return result


def compute_winners_by_maze(
    df: pd.DataFrame, agg: str = "mean"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return winners per maze and the full (maze x algorithm) value table."""
    if df.empty:
        return (
            pd.DataFrame(
                {
                    "maze": pd.Series(dtype="object"),
                    "winner": pd.Series(dtype="object"),
                    "best_value": pd.Series(dtype="float64"),
                }
            ),
            pd.DataFrame(),
        )
    agg_df = _aggregate_algo_maze(df, agg=agg)
    values = agg_df.pivot(index="maze", columns="group", values=agg)
    best_vals = values.max(axis=1).rename("best_value")
    winners = values.idxmax(axis=1).astype(str).rename("winner").to_frame()
    # Construct a new DataFrame to keep typing happy
    winners = pd.DataFrame(
        {
            "maze": winners.index.astype(str),
            "winner": winners["winner"].astype(str).values,
            "best_value": best_vals.values,
        }
    )
    return winners, values


def compute_algorithm_similarity(
    df: pd.DataFrame, agg: str = "mean", method: str = "spearman"
) -> pd.DataFrame:
    """Similarity between algorithms based on per-maze aggregated performance."""
    if df.empty:
        return pd.DataFrame()
    agg_df = _aggregate_algo_maze(df, agg=agg)
    mat = agg_df.pivot(index="group", columns="maze", values=agg)
    mat = mat.dropna(how="all", axis=1)
    if mat.shape[1] < 2 or mat.shape[0] < 2:
        return pd.DataFrame()
    return mat.transpose().corr(method=method)  # type: ignore[arg-type]


__all__ = [
    # Per-run tables
    "final_values_per_run",
    # Performance table
    "PerformanceBuildConfig",
    "build_performance_table",
    # Similarities & winners
    "compute_maze_similarity_overall",
    "compute_maze_similarity_per_algorithm",
    "compute_winners_by_maze",
    "compute_algorithm_similarity",
    # Quantiles & smoothing
    "aggregate_training_quantiles",
    "aggregate_training_iqr",
    "aggregate_worst_case",
    "smooth_ewm",
    # Clustering helper
    "compute_maze_similarity_linkage",
]


def aggregate_training_quantiles(
    df: pd.DataFrame,
    step_key: str,
    *,
    q_low: float,
    q_center: float,
    q_high: float,
) -> pd.DataFrame:
    """Aggregate per group and step into arbitrary low/center/high quantiles.

    Returns columns: [group, step_key, q_low, q_center, q_high, count].
    """
    if df.empty:
        return pd.DataFrame(
            {
                "group": pd.Series(dtype="object"),
                step_key: pd.Series(dtype="float64"),
                "q_low": pd.Series(dtype="float64"),
                "q_center": pd.Series(dtype="float64"),
                "q_high": pd.Series(dtype="float64"),
                "count": pd.Series(dtype="int64"),
            }
        )

    def _q(series: pd.Series, q: float) -> float:
        return float(series.quantile(q))

    grouped = df.groupby(["group", step_key])["value"]
    agg = grouped.agg(
        q_low=lambda s: _q(s, q_low),
        q_center=lambda s: _q(s, q_center),
        q_high=lambda s: _q(s, q_high),
        count="count",
    ).reset_index()
    return agg


def aggregate_training_iqr(df: pd.DataFrame, step_key: str) -> pd.DataFrame:
    """Aggregate into 0.25, 0.5, 0.75 quantiles with median as center.

    Returns columns: [group, step_key, q_low, q_center, q_high, count], where
    q_center corresponds to the median (0.5).
    """
    return aggregate_training_quantiles(
        df, step_key, q_low=0.25, q_center=0.50, q_high=0.75
    )


def aggregate_worst_case(df: pd.DataFrame, step_key: str) -> pd.DataFrame:
    """Aggregate into 0.01, 0.05, 0.10 quantiles for worst-case analysis."""
    return aggregate_training_quantiles(
        df, step_key, q_low=0.01, q_center=0.05, q_high=0.10
    )


# -------------------------- Plotting functions --------------------------


def plot_quantiles(
    agg_df: pd.DataFrame,
    step_key: str,
    *,
    lower_col: str = "q_low",
    center_col: Optional[str] = "median",
    upper_col: str = "q_high",
    groups: Optional[Sequence[str]] = None,
    ax=None,
    label_fmt: str = "{group}",
    fill_alpha: float = 0.2,
    palette_map: Optional[dict[str, Any]] = None,
    y_label: str = "value",
    x_label: Optional[str] = None,
):
    """Plot a quantile envelope (and optional center) per group.

    Expects an aggregated DataFrame with columns:
      - "group", step_key
      - lower_col (default "q_low") and upper_col (default "q_high")
      - optionally center_col (default "median"); set to None to skip the line
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    groups_to_plot = groups or list(agg_df["group"].unique())
    any_plotted = False
    for g in groups_to_plot:
        gdf = agg_df[agg_df["group"] == g]
        if gdf.empty:
            continue
        color_kwargs = {}
        if palette_map is not None and g in palette_map:
            color_kwargs = {"color": palette_map[g]}
        if center_col is not None and center_col in gdf.columns:
            ax.plot(
                gdf[step_key],
                gdf[center_col],
                label=label_fmt.format(group=g),
                **color_kwargs,
            )
        if lower_col in gdf.columns and upper_col in gdf.columns:
            ax.fill_between(
                gdf[step_key],
                gdf[lower_col],
                gdf[upper_col],
                alpha=fill_alpha,
                **color_kwargs,
            )
        any_plotted = True

    xlab = (
        x_label
        if x_label is not None
        else ("update" if step_key == "num_updates" else step_key)
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(y_label)
    if any_plotted:
        ax.legend()
    return ax


# -------------------------- Step alignment diagnostics --------------------------
def summarize_max_steps(df: pd.DataFrame, step_key: str) -> pd.DataFrame:
    """Return one row per run with its maximum recorded step.

    Input is the long-form history assembled by this module, with columns
    [step_key, value, run_id, group]. Output columns are [group, run_id, max_step].
    """
    if df is None or df.empty:
        return pd.DataFrame(
            {
                "group": pd.Series(dtype="object"),
                "run_id": pd.Series(dtype="object"),
                "max_step": pd.Series(dtype="float64"),
            }
        )

    max_by_run = (
        df.groupby("run_id")[step_key]
        .max()
        .reset_index()
        .rename(columns={step_key: "max_step"})
    )
    group_by_run = df.groupby("run_id")["group"].first().reset_index()
    out = max_by_run.merge(group_by_run, on="run_id")[["group", "run_id", "max_step"]]
    # Use MultiIndex sort for better typing compatibility
    out = out.set_index(["group", "run_id"]).sort_index().reset_index()
    return out


def check_update_step_alignment(
    df: pd.DataFrame,
    step_key: str,
    *,
    per_group: bool = True,
    atol: float = 0.0,
) -> pd.DataFrame:
    """Warn if runs end at different update steps.

    Args:
        df: Long-form history with columns [step_key, value, run_id, group].
        step_key: Name of the step column, e.g. "num_updates".
        per_group: If True, also check within each algorithm group.
        atol: Absolute tolerance when comparing steps (useful if steps differ by
              a few due to logging cadence).

    Returns:
        DataFrame with per-run max steps: columns [group, run_id, max_step].
    """
    steps = summarize_max_steps(df, step_key)
    if steps.empty:
        return steps

    def _warn_if_inconsistent(scope: str, sub: pd.DataFrame) -> None:
        vals = sub["max_step"].to_numpy()
        if len(vals) <= 1:
            return
        ref = float(vals[0])
        mismatched = [v for v in vals if abs(float(v) - ref) > float(atol)]
        if mismatched:
            counts = sub.groupby("max_step").size().sort_index()
            warnings.warn(
                f"Inconsistent max {step_key} across {scope}: "
                f"{dict(counts.to_dict())}. Consider aligning by a fixed step."
            )

    # Global check
    _warn_if_inconsistent("all runs", steps)

    # Per-group check
    if per_group:
        for g, sub in steps.groupby("group"):
            _warn_if_inconsistent(f"group '{g}'", sub)

    return steps


# ------------------------- Final-evaluation solve table -------------------------


def summarize_final_values(
    per_run_df: pd.DataFrame,
    *,
    group_order: Optional[Sequence[str]] = None,
    decimals: int = 2,
    as_wide_formatted: bool = False,
) -> pd.DataFrame:
    """Summarize final per-run values into mean ± std per group.

    Args:
        per_run_df: Output of final_values_per_run().
        group_order: Optional explicit order of groups in the result.
        decimals: Number of decimals for the formatted string.
        as_wide_formatted: If True, return a single-row wide table with one
            column per group containing the formatted value "mean ± std".

    Returns:
        If as_wide_formatted is False (default): columns [group, mean, std, n].
        If True: single-row DataFrame with columns == groups and string values.
    """
    if per_run_df is None or per_run_df.empty:
        if as_wide_formatted:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "group": pd.Series(dtype="object"),
                "mean": pd.Series(dtype="float64"),
                "std": pd.Series(dtype="float64"),
                "n": pd.Series(dtype="int64"),
            }
        )

    summary = (
        per_run_df.groupby("group")["value"].agg(["mean", "std", "count"]).reset_index()
    )
    summary = summary.rename(columns={"count": "n"})

    if group_order is not None:
        # Preserve specified ordering
        order_map = {g: i for i, g in enumerate(group_order)}
        summary = (
            summary.assign(_ord=summary["group"].map(lambda g: order_map.get(g)))
            .sort_values(by=["_ord", "group"], na_position="last")
            .drop(columns=["_ord"])
            .reset_index(drop=True)
        )

    if not as_wide_formatted:
        return summary

    # Build a single-row wide formatted table
    def _fmt(r: pd.Series) -> str:
        try:
            return f"{r['mean']:.{decimals}f} ± {r['std']:.{decimals}f}"
        except Exception:
            return ""

    formatted = {str(row["group"]): _fmt(row) for _, row in summary.iterrows()}
    wide = pd.DataFrame([formatted])
    return wide
