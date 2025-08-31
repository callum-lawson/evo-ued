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
        label = (
            runname_to_group.get(orig_name)
            or runname_to_group.get(getattr(run, "name", None))
            or runname_to_group.get(getattr(run, "id", None))
            or runname_to_group.get(getattr(run, "group", None))
        )
        if not label:
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
):
    """Quick matplotlib plot of mean +/- std per group.

    Returns the matplotlib Axes used.
    """
    import matplotlib.pyplot as plt  # Local import to avoid hard dep at import time

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    groups_to_plot = groups or list(agg_df["group"].unique())
    any_plotted = False
    for g in groups_to_plot:
        gdf = agg_df[agg_df["group"] == g]
        if gdf.empty:
            continue
        ax.plot(gdf[step_key], gdf["mean"], label=label_fmt.format(group=g))
        ax.fill_between(
            gdf[step_key], gdf["mean"] - gdf["std"], gdf["mean"] + gdf["std"], alpha=fill_alpha
        )
        any_plotted = True

    ax.set_xlabel(step_key)
    ax.set_ylabel("value")
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


def plot_median_quantiles(
    agg_df: pd.DataFrame,
    step_key: str,
    *,
    groups: Optional[Sequence[str]] = None,
    ax=None,
    label_fmt: str = "{group}",
    fill_alpha: float = 0.2,
):
    """Plot median with a quantile band per group (no smoothing or interpolation)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    groups_to_plot = groups or list(agg_df["group"].unique())
    any_plotted = False
    for g in groups_to_plot:
        gdf = agg_df[agg_df["group"] == g]
        if gdf.empty:
            continue
        ax.plot(gdf[step_key], gdf["median"], label=label_fmt.format(group=g))
        ax.fill_between(
            gdf[step_key], gdf["q_low"], gdf["q_high"], alpha=fill_alpha
        )
        any_plotted = True

    ax.set_xlabel(step_key)
    ax.set_ylabel("value")
    if any_plotted:
        ax.legend()
    return ax


