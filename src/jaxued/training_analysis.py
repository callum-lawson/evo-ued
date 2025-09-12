"""
Helpers for assembling W&B run histories into tidy DataFrames and aggregations.

This keeps notebooks focused on visualization while the data plumbing lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any
from pathlib import Path

import pandas as pd

from jaxued.wandb_client import WandbDataClient
import warnings


@dataclass
class HistoryCollectionConfig:
    entity: str
    project: str
    groups: Sequence[str]
    step_key: str
    metric_key: str
    samples: Optional[int] = None
    max_rows: Optional[int] = None
    use_cache: bool = True
    refresh: bool = False
    cache_ttl_seconds: int = 6 * 60 * 60


def _choose_step_key(columns, preferred: str) -> Optional[str]:
    candidates = [preferred, "num_updates", "Step", "_step", "global_step", "num_env_steps"]
    for c in candidates:
        if c in columns:
            return c
    return None


def _collect_group(
    client: WandbDataClient,
    group: str,
    step_key: str,
    metric_key: str,
    *,
    samples: Optional[int],
    max_rows: Optional[int],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    runs = client.list_runs(group=group)
    for run in runs:
        try:
            data = client.fetch_run_data(
                run.id,
                keys=[step_key, metric_key],
                samples=samples,
                max_rows=max_rows,
                refresh=False,
            )
            df = data.history_df
            sk = _choose_step_key(df.columns, step_key)
            if sk is None or metric_key not in df.columns:
                continue
            slim = (
                df[[sk, metric_key]]
                .dropna()
                .rename(columns={metric_key: "value", sk: step_key})
                .assign(run_id=run.id, group=group)
            )
            frames.append(slim)
        except Exception:
            # Try once more with refresh in case of stale cache
            try:
                data = client.fetch_run_data(
                    run.id,
                    keys=[step_key, metric_key],
                    samples=samples,
                    max_rows=max_rows,
                    refresh=True,
                )
                df = data.history_df
                sk = _choose_step_key(df.columns, step_key)
                if sk is None or metric_key not in df.columns:
                    continue
                slim = (
                    df[[sk, metric_key]]
                    .dropna()
                    .rename(columns={metric_key: "value", sk: step_key})
                    .assign(run_id=run.id, group=group)
                )
                frames.append(slim)
            except Exception:
                continue

    if not frames:
        return pd.DataFrame({step_key: pd.Series(dtype="float64"), "value": pd.Series(dtype="float64"), "run_id": pd.Series(dtype="object"), "group": pd.Series(dtype="object")})
    return pd.concat(frames, ignore_index=True)


def collect_histories(cfg: HistoryCollectionConfig) -> pd.DataFrame:
    """Collect histories for multiple W&B groups into one tidy DataFrame.

    Returns columns: [step_key, value, run_id, group].
    """
    client = WandbDataClient(
        cfg.entity,
        cfg.project,
        cache_ttl_seconds=cfg.cache_ttl_seconds,
        use_cache=cfg.use_cache,
    )

    frames = [
        _collect_group(
            client,
            group,
            cfg.step_key,
            cfg.metric_key,
            samples=cfg.samples,
            max_rows=cfg.max_rows,
        )
        for group in cfg.groups
    ]
    return pd.concat(frames, ignore_index=True) if len(frames) > 0 else pd.DataFrame({cfg.step_key: pd.Series(dtype="float64"), "value": pd.Series(dtype="float64"), "run_id": pd.Series(dtype="object"), "group": pd.Series(dtype="object")})


# ----------------------------- Mapping-based grouping -----------------------------
def load_runname_group_map_from_file(path: str | Path) -> Dict[str, str]:
    """Load a mapping {run_name -> group_label} from a JSON config in the new format.

    Expected structure:
        { "groups": { "AlgoA": ["run_a", ...], "AlgoB": ["run_b", ...] }, ... }
    """
    import json

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    groups = data.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("Config must contain a 'groups' mapping of {algo: [run_names...]}.")

    mapping: Dict[str, str] = {}
    for algo, run_list in groups.items():
        if not isinstance(run_list, list):
            continue
        for rn in run_list:
            mapping[str(rn)] = str(algo)
    return mapping


def collect_histories_from_run_mapping(
    entity: str,
    project: str,
    runname_to_group: Dict[str, str],
    *,
    step_key: str,
    metric_key: str,
    samples: Optional[int] = None,
    max_rows: Optional[int] = None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_ttl_seconds: int = 6 * 60 * 60,
) -> pd.DataFrame:
    """Collect histories by mapping specific run identifiers to custom groups.

    A run is matched if any of these keys exists in the mapping:
      - run.group
      - run.name
      - run.config.get("run_name")
      - run.id
    The mapping value is used as the 'group' label in the output.
    """
    client = WandbDataClient(
        entity,
        project,
        cache_ttl_seconds=cache_ttl_seconds,
        use_cache=use_cache,
    )

    # Build candidate runs grouped by original name, attach desired group label
    all_runs = list(client.list_runs())
    candidates_by_name: Dict[str, List[Any]] = {}
    label_by_name: Dict[str, str] = {}

    for run in all_runs:
        # Prefer original name logged in config; fall back to display name
        try:
            orig_name = run.config.get("run_name")
        except Exception:
            orig_name = None
        if not orig_name:
            orig_name = getattr(run, "name", None)
        if not orig_name:
            continue

        # Only consider runs that appear in our mapping (by original name, or fallbacks)
        label = None
        for k in (orig_name, getattr(run, "name", None), getattr(run, "id", None), getattr(run, "group", None)):
            if isinstance(k, str) and k in runname_to_group:
                label = runname_to_group[k]
                break
        if label is None:
            continue

        candidates_by_name.setdefault(orig_name, []).append(run)
        label_by_name[orig_name] = label

    # Select the most complete run per original name
    def completeness_score(r: Any) -> float:
        # 1) Use summary step if available
        try:
            s = dict(r.summary)
            for k in (step_key, "num_updates", "_step", "Step", "global_step"):
                if k in s and isinstance(s[k], (int, float)):
                    return float(s[k])
        except Exception:
            pass
        # 2) Fallback: history max(step_key) or number of rows
        try:
            dfh = r.history(keys=[step_key], samples=500)
            if step_key in dfh.columns and not dfh.empty:
                return float(dfh[step_key].max())
            return float(len(dfh))
        except Exception:
            return 0.0

    chosen: List[tuple[Any, str]] = []
    for name, runs in candidates_by_name.items():
        if len(runs) == 1:
            chosen.append((runs[0], label_by_name[name]))
            continue
        scored = sorted(((completeness_score(r), r) for r in runs), key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        others = ", ".join(f"{getattr(r, 'id', '?')}:{s}" for s, r in scored[1:])
        warnings.warn(
            f"Duplicate runs for '{name}'. Using {getattr(best, 'id', '?')} (score={best_score}). Others: {others}"
        )
        chosen.append((best, label_by_name[name]))

    # Fetch histories only for chosen runs
    frames: List[pd.DataFrame] = []
    for run, label in chosen:
        try:
            data = client.fetch_run_data(
                run.id,
                keys=[step_key, metric_key],
                samples=samples,
                max_rows=max_rows,
                refresh=refresh,
            )
            df = data.history_df
            sk = _choose_step_key(df.columns, step_key)
            if sk is None or metric_key not in df.columns:
                continue
            slim = (
                df[[sk, metric_key]]
                .dropna()
                .rename(columns={metric_key: "value", sk: step_key})
                .assign(run_id=run.id, group=label)
            )
            frames.append(slim)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame({step_key: pd.Series(dtype="float64"), "value": pd.Series(dtype="float64"), "run_id": pd.Series(dtype="object"), "group": pd.Series(dtype="object")})
    return pd.concat(frames, ignore_index=True)


def aggregate_mean_std(df: pd.DataFrame, step_key: str) -> pd.DataFrame:
    """Aggregate per group and step into mean/std/count for column 'value'."""
    if df.empty:
        return pd.DataFrame({"group": pd.Series(dtype="object"), step_key: pd.Series(dtype="float64"), "mean": pd.Series(dtype="float64"), "std": pd.Series(dtype="float64"), "count": pd.Series(dtype="int64")})
    agg = (
        df.groupby(["group", step_key])["value"].agg(["mean", "std", "count"]).reset_index()
    )
    return agg


def smooth_ewm(df: pd.DataFrame, value_col: str = "mean", span: int = 20) -> pd.DataFrame:
    """Exponentially-weighted smoothing for nicer plots."""
    if df.empty:
        return df
    out = df.copy()
    out[value_col] = out.groupby("group")[value_col].transform(lambda s: s.ewm(span=span).mean())
    return out


def plot_mean_std(
    agg_df: pd.DataFrame,
    step_key: str,
    *,
    groups: Optional[Sequence[str]] = None,
    ax=None,
    label_fmt: str = "{group}",
    fill_alpha: float = 0.2,
    palette_map: Optional[Dict[str, Any]] = None,
    y_label: str = "value",
    x_label: Optional[str] = None,
):
    """Quick matplotlib plot of mean +/- std per group.

    Returns the matplotlib Axes used.
    """
    import matplotlib.pyplot as plt  # Local import to avoid hard dep at import time

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
        ax.plot(gdf[step_key], gdf["mean"], label=label_fmt.format(group=g), **color_kwargs)
        ax.fill_between(
            gdf[step_key], gdf["mean"] - gdf["std"], gdf["mean"] + gdf["std"], alpha=fill_alpha, **color_kwargs
        )
        any_plotted = True

    xlab = x_label if x_label is not None else ("update" if step_key == "num_updates" else step_key)
    ax.set_xlabel(xlab)
    ax.set_ylabel(y_label)
    if any_plotted:
        ax.legend()
    return ax


def aggregate_quantiles(
    df: pd.DataFrame, step_key: str, *, q_low: float = 0.25, q_high: float = 0.75
) -> pd.DataFrame:
    """Aggregate per group and step into median and quantile envelope.

    No interpolation/resampling: only uses existing rows.
    """
    if df.empty:
        return pd.DataFrame({"group": pd.Series(dtype="object"), step_key: pd.Series(dtype="float64"), "median": pd.Series(dtype="float64"), "q_low": pd.Series(dtype="float64"), "q_high": pd.Series(dtype="float64"), "count": pd.Series(dtype="int64")})

    def _q(series: pd.Series, q: float) -> float:
        return float(series.quantile(q))

    grouped = df.groupby(["group", step_key])["value"]
    agg = grouped.agg(
        median="median",
        q_low=lambda s: _q(s, q_low),
        q_high=lambda s: _q(s, q_high),
        count="count",
    ).reset_index()
    return agg


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
    palette_map: Optional[Dict[str, Any]] = None,
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
                gdf[step_key], gdf[lower_col], gdf[upper_col], alpha=fill_alpha, **color_kwargs
            )
        any_plotted = True

    xlab = x_label if x_label is not None else ("update" if step_key == "num_updates" else step_key)
    ax.set_xlabel(xlab)
    ax.set_ylabel(y_label)
    if any_plotted:
        ax.legend()
    return ax


def aggregate_quantiles_custom(
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
    agg = (
        grouped.agg(
            q_low=lambda s: _q(s, q_low),
            q_center=lambda s: _q(s, q_center),
            q_high=lambda s: _q(s, q_high),
            count="count",
        ).reset_index()
    )
    return agg


def plot_median_interquartile(
    agg_df: pd.DataFrame,
    step_key: str,
    *,
    groups: Optional[Sequence[str]] = None,
    ax=None,
    label_fmt: str = "{group}",
    fill_alpha: float = 0.2,
    palette_map: Optional[Dict[str, Any]] = None,
    y_label: str = "value",
    x_label: Optional[str] = None,
):
    """Convenience wrapper for plotting median with IQR band (0.25, 0.5, 0.75)."""
    return plot_quantiles(
        agg_df,
        step_key,
        lower_col="q_low",
        center_col="median",
        upper_col="q_high",
        groups=groups,
        ax=ax,
        label_fmt=label_fmt,
        fill_alpha=fill_alpha,
        palette_map=palette_map,
        y_label=y_label,
        x_label=x_label,
    )


def plot_worst_case(
    df: pd.DataFrame,
    step_key: str,
    *,
    groups: Optional[Sequence[str]] = None,
    ax=None,
    label_fmt: str = "{group}",
    fill_alpha: float = 0.2,
    palette_map: Optional[Dict[str, Any]] = None,
    y_label: str = "value",
    x_label: Optional[str] = None,
):
    """Plot a "worst-case" envelope using (0.01, 0.05, 0.10) quantiles.

    This aggregates the raw long-form history first, then plots with the 5th
    percentile as the center line and 1st-10th percentile as the band.
    """
    agg = aggregate_quantiles_custom(
        df,
        step_key,
        q_low=0.01,
        q_center=0.05,
        q_high=0.10,
    )
    return plot_quantiles(
        agg,
        step_key,
        lower_col="q_low",
        center_col="q_center",
        upper_col="q_high",
        groups=groups,
        ax=ax,
        label_fmt=label_fmt,
        fill_alpha=fill_alpha,
        palette_map=palette_map,
        y_label=y_label,
        x_label=x_label,
    )



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

    max_by_run = df.groupby("run_id")[step_key].max().rename("max_step").reset_index()
    group_by_run = df.groupby("run_id")["group"].first().reset_index()
    out = max_by_run.merge(group_by_run, on="run_id")[["group", "run_id", "max_step"]]
    return out.sort_values(["group", "run_id"]).reset_index(drop=True)


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
def final_values_per_run(df: pd.DataFrame, step_key: str) -> pd.DataFrame:
    """Return one row per run with the final value at the max step.

    Args:
        df: Long-form history with columns [step_key, value, run_id, group].
        step_key: Name of the step column, e.g. "num_updates".

    Returns:
        DataFrame with columns [group, run_id, step, value]. One row per run.
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

    # For each run, select the row with the maximum step
    # Sort first to make idxmax deterministic when duplicates exist
    sorted_df = df.sort_values(["run_id", step_key])
    idx = sorted_df.groupby("run_id")[step_key].idxmax()
    per_run = (
        sorted_df.loc[idx, ["group", "run_id", step_key, "value"]]
        .rename(columns={step_key: "step"})
        .reset_index(drop=True)
    )
    return per_run


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
        summary = summary.assign(_ord=summary["group"].map(order_map)).sort_values(
            by=["_ord", "group"], na_position="last"
        ).drop(columns=["_ord"]).reset_index(drop=True)

    if not as_wide_formatted:
        return summary

    # Build a single-row wide formatted table
    def _fmt(r: pd.Series) -> str:
        try:
            return f"{r['mean']:.{decimals}f} ± {r['std']:.{decimals}f}"
        except Exception:
            return ""

    formatted = {row["group"]: _fmt(row) for _, row in summary.iterrows()}
    wide = pd.DataFrame([formatted])
    return wide
