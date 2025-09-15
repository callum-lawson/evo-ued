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

from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .training_analyses import compute_maze_similarity_linkage

# -------------------------- Plotting functions --------------------------


def plot_training_runs(
    agg_df: pd.DataFrame,
    step_key: str,
    *,
    lower_col: str = "q_low",
    center_col: str = "q_center",
    upper_col: str = "q_high",
    groups: Optional[Sequence[str]] = None,
    ax=None,
    label_fmt: str = "{group}",
    fill_alpha: float = 0.2,
    palette_map: Optional[dict[str, Any]] = None,
    y_label: str = "value",
    x_label: Optional[str] = None,
):
    """Plot a quantile envelope and center per group.

    Expects an aggregated DataFrame with columns:
      - "group", step_key
      - lower_col (default "q_low") and upper_col (default "q_high")
      - center_col (default "q_center")
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
        if center_col in gdf.columns:
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


# -------------------------- Maze algorithm bars --------------------------


def plot_maze_algorithm_bars(
    values: pd.DataFrame,
    title: Optional[str] = None,
    normalize: bool = False,
    y_label: Optional[str] = None,
):
    """Grouped bar plot: each maze on x-axis, bars for each algorithm.

    Args:
        values: pivot table with index=maze, columns=algorithm, values=metric.
        title: optional figure title.
        normalize: if True, scale each maze's values to [0, 1] for comparability.
    """
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore
    except ImportError:
        sns = None  # type: ignore[assignment]

    if values is None or values.empty:
        _, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    plot_df = values.copy()
    if normalize:
        plot_df = plot_df.divide(plot_df.max(axis=1).replace(0, float("nan")), axis=0)

    if sns is not None:
        long_df = plot_df.reset_index().melt(
            id_vars=["maze"], var_name="algorithm", value_name="value"
        )
        width = max(10, len(values.index) * 0.7)
        _, ax = plt.subplots(figsize=(width, 4))
        # Use a consistent palette based on algorithms
        algos = sorted(long_df["algorithm"].unique().tolist())
        palette = make_algorithm_palette(algos)
        sns.barplot(
            data=long_df,
            x="maze",
            y="value",
            hue="algorithm",
            dodge=True,
            ax=ax,
            palette=palette,
        )
        ax.set_xlabel("maze")
        ax.set_ylabel(
            y_label
            if y_label is not None
            else ("normalized value" if normalize else "value")
        )
        ax.tick_params(axis="x", rotation=60)
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
        offsets = [xi - total_width / 2 + j * bar_w + bar_w / 2 for xi in x]
        ax.bar(
            offsets,
            plot_df[algo].tolist(),
            width=bar_w,
            label=str(algo),
            color=color_map.get(str(algo)),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(mazes, rotation=60, ha="right")
    ax.set_xlabel("maze")
    ax.set_ylabel(
        y_label
        if y_label is not None
        else ("normalized value" if normalize else "value")
    )
    if title:
        ax.set_title(title)
    ax.legend(title="algorithm")
    return ax


# ------------------------------ Palette helpers ----------------------------


def make_algorithm_palette(algorithms: Sequence[str]) -> Dict[str, Any]:
    """Create a consistent color mapping for algorithms.

    Deterministic by algorithm name (sorted), independent of input order.
    Uses seaborn/matplotlib tab10-like palette if available.
    """
    algos = sorted(set(map(str, algorithms)))
    try:
        import seaborn as sns  # type: ignore

        colors = sns.color_palette("colorblind", n_colors=max(10, len(algos)))
    except ImportError:
        from matplotlib.cm import get_cmap

        cmap = get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(max(10, len(algos)))]
    return {alg: colors[i % len(colors)] for i, alg in enumerate(algos)}


# --------------------------- Faceted algorithm bars ------------------------


def plot_algorithm_facets_over_mazes(
    values: pd.DataFrame,
    title: Optional[str] = None,
    normalize: bool = False,
    ncols: int = 2,
    y_label: Optional[str] = None,
    y_lim_01: bool = False,
):
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
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(10, ncols * 6), max(4, nrows * 3)),
        squeeze=False,
    )

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
        ax.set_ylabel(
            y_label
            if y_label is not None
            else ("normalized value" if normalize else "value")
        )
        if y_lim_01:
            ax.set_ylim(0.0, 1.0)

    # Hide any unused subplots
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].set_visible(False)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    else:
        fig.tight_layout()
    return axes


# --------------------------------- Plotting --------------------------------


def plot_similarity_heatmap(
    matrix: pd.DataFrame,
    title: Optional[str] = None,
    cmap: str = "vlag",
    annotate: bool = False,
    fmt: str = ".2f",
):
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore
    except ImportError:
        sns = None  # type: ignore[assignment]

    if matrix is None or matrix.empty:
        _, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    _, ax = plt.subplots(figsize=(max(6, len(matrix) * 0.5), max(4, len(matrix) * 0.5)))
    if sns is not None:
        sns.heatmap(
            matrix, cmap=cmap, annot=annotate, fmt=fmt, square=True, cbar=True, ax=ax
        )
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
                    ax.text(
                        j,
                        i,
                        format(matrix.loc[row, col], fmt),
                        ha="center",
                        va="center",
                        color="black",
                    )
    if title:
        ax.set_title(title)
    return ax


# Removed previous heatmap/winner-bar helpers in favor of grouped bars


# ------------------------ Maze similarity clustering ------------------------


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
    except ImportError:
        sns = None  # type: ignore[assignment]

    sim, Z = compute_maze_similarity_linkage(
        df, agg=agg, method=method, linkage_method=linkage_method
    )
    if sim is None or sim.empty or Z is None:
        # Produce a placeholder figure
        _, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return {
            "Z": None,
            "similarity": pd.DataFrame(),
            "cluster_assignments": None,
            "order": None,
        }

    maze_labels = sim.index.tolist()

    # 1) Dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=maze_labels, leaf_rotation=90)
    title = (
        f"{title_prefix} dendrogram" if title_prefix else "Maze similarity dendrogram"
    )
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
    if sns is not None:
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
            cbar_kws={
                "label": (
                    f"{method.title()} r"
                    if method in {"pearson", "spearman"}
                    else "similarity"
                )
            },
        )
        if title_prefix:
            plt.suptitle(f"{title_prefix} clustermap", y=1.02)
        plt.show()
    else:
        # Fallback: reorder similarity matrix and plot with matplotlib
        sim_ord = sim.loc[ordered_labels, ordered_labels]
        _, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(sim_ord.to_numpy(), cmap="vlag", vmin=-1.0, vmax=1.0)
        plt.colorbar(
            im,
            ax=ax,
            label=(
                f"{method.title()} r"
                if method in {"pearson", "spearman"}
                else "similarity"
            ),
        )
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_yticks(range(len(ordered_labels)))
        ax.set_xticklabels(ordered_labels, rotation=90)
        ax.set_yticklabels(ordered_labels)
        if title_prefix:
            ax.set_title(f"{title_prefix} (reordered)")
        plt.tight_layout()
        plt.show()

    return {
        "Z": Z,
        "similarity": sim,
        "cluster_assignments": clusters,
        "order": ordered_labels,
    }


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

    pivot = dfa.pivot_table(
        index="run_name", columns="maze", values="value", aggfunc="mean"
    )
    pivot = pivot.dropna(axis=1, how="all")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return pd.DataFrame(), None

    if method in {"pearson", "spearman"}:
        sim = pivot.corr(method=method)  # type: ignore[arg-type]
    else:
        raise ValueError(
            "method must be 'pearson' or 'spearman' for per-algorithm similarity"
        )

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
    except ImportError:
        sns = None  # type: ignore[assignment]

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
        return {
            "Z": None,
            "similarity": pd.DataFrame(),
            "cluster_assignments": None,
            "order": None,
        }

    maze_labels = sim.index.tolist()
    ttl_prefix = (
        title_prefix if title_prefix is not None else f"Maze similarity ({algorithm})"
    )

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
    if sns is not None:
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

    return {
        "Z": Z,
        "similarity": sim,
        "cluster_assignments": clusters,
        "order": ordered_labels,
    }
