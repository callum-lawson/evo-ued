"""
Analysis utilities for per-maze performance comparisons and similarity plots.

This module builds on the W&B client utilities to assemble a tidy table of
per-maze performance at a chosen timestep (default: last), and provides helpers
to compute and visualize:

- Similarity between maze types (overall and per algorithm)
- Where each algorithm is strongest (per-maze winners)
- Similarity between algorithms (rank- and winner-overlap based)

The functions are designed for use from notebooks: heavy lifting lives here;
notebooks pass parameters and call plotting functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence

import math
import warnings

import pandas as pd

from .wandb_client import WandbDataClient
from .training_analysis import _choose_step_key  # reuse step selection heuristic


# ------------------------------ Data assembly ------------------------------

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
    # Exclude common aggregate keys like mean/std/min/max, and summary rollups
    tail = column[len(prefix) :]
    bad_suffixes = {"mean", "median", "std", "min", "max", "p25", "p75"}
    if tail in bad_suffixes:
        return False
    # Avoid nested aggregates like "return/mean_smooth" or similar
    if any(s in tail.lower() for s in ("smooth", "ewm", "ema")):
        return False
    # Keep metrics that look like specific maze names
    return True


def _extract_original_name(run) -> Optional[str]:
    cfg = getattr(run, "config", None)
    if cfg is not None:
        try:
            val = cfg.get("run_name")  # type: ignore[assignment]
            if isinstance(val, str) and val:
                return val
        except (AttributeError, TypeError, ValueError, KeyError):
            pass
    name = getattr(run, "name", None)
    if isinstance(name, str) and name:
        return name
    return None


def _select_representative_runs(
    client: WandbDataClient,
    runname_to_group: Dict[str, str],
    step_key: str,
) -> List[Tuple["any", str]]:
    """Return list of (run, group_label) selecting one best run per original name.

    Matches runs by any of: config.run_name, display name, id, or wandb group,
    and then picks the most complete run per original name using the highest
    step seen in summary or history.
    """
    all_runs = list(client.list_runs())
    candidates_by_name: Dict[str, List["any"]] = {}
    label_by_name: Dict[str, str] = {}

    for run in all_runs:
        orig_name = _extract_original_name(run)
        if not orig_name:
            continue
        label = None
        for k in (orig_name, getattr(run, "name", None), getattr(run, "id", None), getattr(run, "group", None)):
            if isinstance(k, str) and k in runname_to_group:
                label = runname_to_group[k]
                break
        if label is None:
            continue
        candidates_by_name.setdefault(orig_name, []).append(run)
        label_by_name[orig_name] = label

    def completeness_score(r: "any") -> float:
        # Prefer summary step (robust to W&B Summary proxy behavior)
        s_obj = getattr(r, "summary", None)
        if s_obj is not None:
            keys = (step_key, "num_updates", "_step", "Step", "global_step")
            # Try .get first
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
            # Fallback to item access
            for k in keys:
                try:
                    v = s_obj[k]  # type: ignore[index]
                    if isinstance(v, (int, float)):
                        return float(v)
                except (KeyError, TypeError, ValueError):
                    continue
        # Fallback to history size/max(step)
        try:
            dfh = r.history(keys=[step_key], samples=500)
            if step_key in dfh.columns and not dfh.empty:
                return float(dfh[step_key].max())
            return float(len(dfh))
        except (AttributeError, RuntimeError, ValueError):
            return 0.0

    chosen: List[Tuple["any", str]] = []
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

    return chosen


def _value_at_step(df: pd.DataFrame, step_col: str, metric_col: str, at_step: Optional[float]) -> Optional[float]:
    if metric_col not in df.columns:
        return None
    sdf = df[[step_col, metric_col]].dropna()
    if sdf.empty:
        return None
    if at_step is None:
        # Last non-null value
        return float(sdf[metric_col].iloc[-1])
    # Prefer the last value with step <= at_step, else nearest by absolute distance
    leq = sdf[sdf[step_col] <= at_step]
    if not leq.empty:
        return float(leq.iloc[-1][metric_col])
    # Nearest overall
    idx = (sdf[step_col] - at_step).abs().idxmin()
    return float(sdf.loc[idx, metric_col])


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
        # Identify maze metric columns present in this run
        maze_cols = [c for c in df.columns if _is_maze_metric_column(c, cfg.metric_prefix)]
        if not maze_cols:
            continue

        # Determine step to evaluate at (max seen in this run if None)
        at_step = cfg.at_step
        if at_step is None:
            series = df[step_col].dropna()
            at_step = float(series.max()) if not series.empty else None

        # Resolve human-friendly run name
        rname = _extract_original_name(rd.run) or rd.run.id

        # Gather values per maze
        for col in maze_cols:
            val = _value_at_step(df, step_col, col, at_step)
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
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


# ------------------------------ Computations -------------------------------

def _aggregate_algo_maze(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    if df.empty:
        return df
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")
    func = agg
    return (
        df.groupby(["group", "maze"], as_index=False)["value"].agg(func)
        .rename(columns={"value": agg})
    )


def compute_maze_similarity_overall(df: pd.DataFrame, agg: str = "mean", method: str = "pearson") -> pd.DataFrame:
    """Similarity between mazes based on algorithm-aggregated performance.

    Steps:
    - Aggregate values per (algorithm, maze)
    - Pivot to matrix [maze x algorithm]
    - Compute pairwise similarity between mazes (rows)
    """
    if df.empty:
        return pd.DataFrame()
    agg_df = _aggregate_algo_maze(df, agg=agg)
    mat = agg_df.pivot(index="maze", columns="group", values=agg)
    # Drop mazes with all-NaN
    mat = mat.dropna(how="all")
    if mat.empty:
        return pd.DataFrame()
    if method in {"pearson", "spearman"}:
        return mat.transpose().corr(method=method)
    if method == "cosine":
        # Row-wise cosine similarity
        from numpy.linalg import norm
        import numpy as np

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


def compute_maze_similarity_per_algorithm(df: pd.DataFrame, method: str = "pearson") -> Dict[str, pd.DataFrame]:
    """For each algorithm, similarity between mazes using across-seed variation.

    For a fixed algorithm, build matrix [run/seed x maze] and compute
    correlation across columns (mazes).
    """
    result: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return result
    for algo, dfa in df.groupby("group"):
        pivot = dfa.pivot_table(index="run_name", columns="maze", values="value", aggfunc="mean")
        # Drop mazes with all-NaN or single row which makes correlation undefined
        pivot = pivot.dropna(axis=1, how="all")
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            result[algo] = pd.DataFrame()
            continue
        corr = pivot.corr(method=method)
        result[algo] = corr
    return result


def compute_winners_by_maze(df: pd.DataFrame, agg: str = "mean") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return winners per maze and the full (maze x algorithm) value table.

    Returns:
        winners: columns [maze, winner, best_value]
        values:  pivot table [maze x algorithm] of aggregated values
    """
    if df.empty:
        return pd.DataFrame(columns=["maze", "winner", "best_value"]), pd.DataFrame()
    agg_df = _aggregate_algo_maze(df, agg=agg)
    values = agg_df.pivot(index="maze", columns="group", values=agg)
    # Identify winners per row
    winners = values.idxmax(axis=1).rename("winner").to_frame().reset_index()
    best_vals = values.max(axis=1).rename("best_value").to_frame().reset_index(drop=True)
    winners["best_value"] = best_vals
    return winners, values


def compute_algorithm_similarity(df: pd.DataFrame, agg: str = "mean", method: str = "spearman") -> pd.DataFrame:
    """Similarity between algorithms based on per-maze aggregated performance.

    Uses rank correlation by default to compare across mazes.
    """
    if df.empty:
        return pd.DataFrame()
    agg_df = _aggregate_algo_maze(df, agg=agg)
    mat = agg_df.pivot(index="group", columns="maze", values=agg)
    mat = mat.dropna(how="all", axis=1)
    if mat.shape[1] < 2 or mat.shape[0] < 2:
        return pd.DataFrame()
    return mat.transpose().corr(method=method)


def plot_maze_algorithm_bars(values: pd.DataFrame, title: Optional[str] = None, normalize: bool = False):
    """Grouped bar plot: each maze on x-axis, bars for each algorithm.

    Args:
        values: pivot table with index=maze, columns=algorithm, values=metric.
        title: optional figure title.
        normalize: if True, scale each maze's values to [0, 1] for comparability.
    """
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns  # type: ignore
        have_seaborn = True
    except ImportError:
        have_seaborn = False

    if values is None or values.empty:
        _, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    plot_df = values.copy()
    if normalize:
        plot_df = plot_df.divide(plot_df.max(axis=1).replace(0, float("nan")), axis=0)

    if have_seaborn:
        long_df = plot_df.reset_index().melt(id_vars=["maze"], var_name="algorithm", value_name="value")
        width = max(10, len(values.index) * 0.7)
        _, ax = plt.subplots(figsize=(width, 4))
        # Use a consistent palette based on algorithms
        algos = sorted(long_df["algorithm"].unique().tolist())
        palette = make_algorithm_palette(algos)
        sns.barplot(data=long_df, x="maze", y="value", hue="algorithm", dodge=True, ax=ax, palette=palette)
        ax.set_xlabel("maze")
        ax.set_ylabel("normalized value" if normalize else "value")
        ax.tick_params(axis='x', rotation=60)
        if title:
            ax.set_title(title)
        ax.legend(title="algorithm")
        return ax

    # Matplotlib fallback: manual grouped bars
    algos = list(plot_df.columns)
    mazes = list(map(str, plot_df.index))
    num_mazes = len(mazes)
    num_algos = len(algos)
    x = list(range(num_mazes))
    total_width = 0.8
    bar_w = total_width / max(1, num_algos)
    _, ax = plt.subplots(figsize=(max(10, num_mazes * 0.7), 4))
    # Consistent colors
    color_map = make_algorithm_palette(algos)

    for j, algo in enumerate(algos):
        offsets = [xi - total_width/2 + j * bar_w + bar_w/2 for xi in x]
        ax.bar(offsets, plot_df[algo].tolist(), width=bar_w, label=str(algo), color=color_map.get(str(algo)))

    ax.set_xticks(x)
    ax.set_xticklabels(mazes, rotation=60, ha="right")
    ax.set_xlabel("maze")
    ax.set_ylabel("normalized value" if normalize else "value")
    if title:
        ax.set_title(title)
    ax.legend(title="algorithm")
    return ax

# ------------------------------ Palette helpers ----------------------------

def make_algorithm_palette(algorithms: Sequence[str]) -> Dict[str, str]:
    """Create a consistent color mapping for algorithms.

    Deterministic by algorithm name (sorted), independent of input order.
    Uses seaborn/matplotlib tab10-like palette if available.
    """
    algos = sorted(set(map(str, algorithms)))
    try:
        import seaborn as sns  # type: ignore
        # Prefer seaborn's colorblind-safe palette
        colors = sns.color_palette("colorblind", n_colors=max(10, len(algos)))
    except ImportError:
        from matplotlib.cm import get_cmap
        cmap = get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(max(10, len(algos)))]
    return {alg: colors[i % len(colors)] for i, alg in enumerate(algos)}

# --------------------------- Faceted algorithm bars ------------------------

def plot_algorithm_facets_over_mazes(values: pd.DataFrame, title: Optional[str] = None, normalize: bool = False, ncols: int = 2):
    """Facet bar plots: one subplot per algorithm, bars over mazes.

    Bars are colored using the algorithm's color to keep consistency across plots.
    """
    import matplotlib.pyplot as plt
    if values is None or values.empty:
        _, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    plot_df = values.copy()
    if normalize:
        plot_df = plot_df.divide(plot_df.max(axis=1).replace(0, float("nan")), axis=0)

    algos = list(map(str, plot_df.columns))
    mazes = list(map(str, plot_df.index))
    palette = make_algorithm_palette(algos)

    n = len(algos)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(max(10, ncols * 6), max(4, nrows * 3)), squeeze=False)

    for idx, algo in enumerate(algos):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        vals = plot_df[algo].tolist()
        x = list(range(len(mazes)))
        ax.bar(x, vals, color=palette.get(algo))
        ax.set_xticks(x)
        ax.set_xticklabels(mazes, rotation=60, ha="right")
        ax.set_title(str(algo))
        ax.set_ylabel("normalized value" if normalize else "value")

    # Hide any unused subplots
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].set_visible(False)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig.tight_layout()
    return axes

# --------------------------------- Plotting --------------------------------

def plot_similarity_heatmap(matrix: pd.DataFrame, title: Optional[str] = None, cmap: str = "vlag", annotate: bool = False, fmt: str = ".2f"):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns  # type: ignore
        have_seaborn = True
    except ImportError:
        have_seaborn = False

    if matrix is None or matrix.empty:
        _, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    _, ax = plt.subplots(figsize=(max(6, len(matrix) * 0.5), max(4, len(matrix) * 0.5)))
    if have_seaborn:
        sns.heatmap(matrix, cmap=cmap, annot=annotate, fmt=fmt, square=True, cbar=True, ax=ax)
    else:
        im = ax.imshow(matrix.to_numpy(), cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(list(map(str, matrix.columns)), rotation=45, ha="right")
        ax.set_yticklabels(list(map(str, matrix.index)))
        if annotate:
            for i, row in enumerate(matrix.index):
                for j, col in enumerate(matrix.columns):
                    ax.text(j, i, format(matrix.loc[row, col], fmt), ha="center", va="center", color="black")
    if title:
        ax.set_title(title)
    return ax


# Removed previous heatmap/winner-bar helpers in favor of grouped bars



