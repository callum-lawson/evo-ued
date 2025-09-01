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




# ------------------------ Maze similarity clustering ------------------------

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


def plot_maze_similarity_clustering(
    df: pd.DataFrame,
    agg: str = "mean",
    method: str = "pearson",
    linkage_method: str = "average",
    flat_threshold: Optional[float] = 0.6,
    title_prefix: Optional[str] = "Maze similarity",
):
    """Plot dendrogram and clustered heatmap for maze similarity.

    - Computes similarity across mazes aggregated over algorithms
    - Builds hierarchical clustering linkage
    - Plots a dendrogram with maze labels
    - Plots a clustered heatmap using the same linkage (seaborn if available,
      otherwise falls back to a reordered heatmap)

    Returns a dict with keys: {"Z", "similarity", "cluster_assignments", "order"}.
    """
    import matplotlib.pyplot as plt  # type: ignore
    from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list  # type: ignore

    try:
        import seaborn as sns  # type: ignore
        have_seaborn = True
    except ImportError:
        have_seaborn = False

    sim, Z = compute_maze_similarity_linkage(
        df, agg=agg, method=method, linkage_method=linkage_method
    )
    if sim is None or sim.empty or Z is None:
        # Produce a placeholder figure
        _, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return {"Z": None, "similarity": pd.DataFrame(), "cluster_assignments": None, "order": None}

    maze_labels = sim.index.tolist()

    # 1) Dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=maze_labels, leaf_rotation=90)
    title = f"{title_prefix} dendrogram" if title_prefix else "Maze similarity dendrogram"
    plt.title(title)
    plt.ylabel("Distance (1 - correlation)")
    plt.tight_layout()
    plt.show()

    # Compute leaf order and optional flat clusters
    order_idx = leaves_list(Z)
    ordered_labels = [maze_labels[i] for i in order_idx]
    clusters = None
    if flat_threshold is not None:
        clusters = fcluster(Z, t=float(flat_threshold), criterion="distance")
        # Map back to labels
        clusters = pd.Series(clusters, index=maze_labels, name="cluster")

    # 2) Clustered heatmap using the same linkage if seaborn is available
    if have_seaborn:
        sns.clustermap(
            sim,
            row_linkage=Z,
            col_linkage=Z,
            cmap="vlag",
            center=0.0,
            linewidths=0.5,
            figsize=(10, 10),
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": f"{method.title()} r" if method in {"pearson", "spearman"} else "similarity"},
        )
        if title_prefix:
            plt.suptitle(f"{title_prefix} clustermap", y=1.02)
        plt.show()
    else:
        # Fallback: reorder similarity matrix and plot with matplotlib
        sim_ord = sim.loc[ordered_labels, ordered_labels]
        _, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(sim_ord.to_numpy(), cmap="vlag", vmin=-1.0, vmax=1.0)
        plt.colorbar(im, ax=ax, label=f"{method.title()} r" if method in {"pearson", "spearman"} else "similarity")
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_yticks(range(len(ordered_labels)))
        ax.set_xticklabels(ordered_labels, rotation=90)
        ax.set_yticklabels(ordered_labels)
        if title_prefix:
            ax.set_title(f"{title_prefix} (reordered)")
        plt.tight_layout()
        plt.show()

    return {"Z": Z, "similarity": sim, "cluster_assignments": clusters, "order": ordered_labels}


# --------------- Maze similarity clustering for a single algorithm -----------

def compute_maze_similarity_linkage_for_algorithm(
    df: pd.DataFrame,
    algorithm: str,
    method: str = "pearson",
    linkage_method: str = "average",
):
    """Compute per-maze similarity and linkage using only one algorithm's runs.

    Steps:
    - Filter to rows where group == algorithm
    - Pivot to [run/seed x maze]
    - Compute correlation across columns (mazes) to get [maze x maze] similarity
    - Build hierarchical clustering linkage from distance d = 1 - similarity
    """
    from scipy.spatial.distance import squareform  # type: ignore
    from scipy.cluster.hierarchy import linkage  # type: ignore

    if df is None or df.empty:
        return pd.DataFrame(), None

    dfa = df[df["group"] == algorithm]
    if dfa.empty:
        return pd.DataFrame(), None

    pivot = dfa.pivot_table(index="run_name", columns="maze", values="value", aggfunc="mean")
    pivot = pivot.dropna(axis=1, how="all")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return pd.DataFrame(), None

    if method in {"pearson", "spearman"}:
        sim = pivot.corr(method=method)
    else:
        raise ValueError("method must be 'pearson' or 'spearman' for per-algorithm similarity")

    dist = (1.0 - sim).clip(lower=0).fillna(1.0)
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=linkage_method)
    return sim, Z


def plot_maze_similarity_clustering_for_algorithm(
    df: pd.DataFrame,
    algorithm: str,
    method: str = "pearson",
    linkage_method: str = "average",
    flat_threshold: Optional[float] = 0.6,
    title_prefix: Optional[str] = None,
):
    """Plot dendrogram and clustered heatmap for a single algorithm's maze similarity."""
    import matplotlib.pyplot as plt  # type: ignore
    from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list  # type: ignore

    try:
        import seaborn as sns  # type: ignore
        have_seaborn = True
    except ImportError:
        have_seaborn = False

    sim, Z = compute_maze_similarity_linkage_for_algorithm(
        df=df,
        algorithm=algorithm,
        method=method,
        linkage_method=linkage_method,
    )

    if sim is None or sim.empty or Z is None:
        _, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No data for {algorithm}", ha="center", va="center")
        ax.set_axis_off()
        return {"Z": None, "similarity": pd.DataFrame(), "cluster_assignments": None, "order": None}

    maze_labels = sim.index.tolist()
    ttl_prefix = title_prefix if title_prefix is not None else f"Maze similarity ({algorithm})"

    # 1) Dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=maze_labels, leaf_rotation=90)
    plt.title(f"{ttl_prefix} dendrogram")
    plt.ylabel("Distance (1 - correlation)")
    plt.tight_layout()
    plt.show()

    order_idx = leaves_list(Z)
    ordered_labels = [maze_labels[i] for i in order_idx]
    clusters = None
    if flat_threshold is not None:
        clusters = fcluster(Z, t=float(flat_threshold), criterion="distance")
        clusters = pd.Series(clusters, index=maze_labels, name="cluster")

    # 2) Clustered heatmap
    if have_seaborn:
        sns.clustermap(
            sim,
            row_linkage=Z,
            col_linkage=Z,
            cmap="vlag",
            center=0.0,
            linewidths=0.5,
            figsize=(10, 10),
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": f"{method.title()} r"},
        )
        plt.suptitle(f"{ttl_prefix} clustermap", y=1.02)
        plt.show()
    else:
        sim_ord = sim.loc[ordered_labels, ordered_labels]
        _, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(sim_ord.to_numpy(), cmap="vlag", vmin=-1.0, vmax=1.0)
        plt.colorbar(im, ax=ax, label=f"{method.title()} r")
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_yticks(range(len(ordered_labels)))
        ax.set_xticklabels(ordered_labels, rotation=90)
        ax.set_yticklabels(ordered_labels)
        ax.set_title(ttl_prefix)
        plt.tight_layout()
        plt.show()

    return {"Z": Z, "similarity": sim, "cluster_assignments": clusters, "order": ordered_labels}
