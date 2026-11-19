"""Plot generation for the analysis report package.

Uses a fixed, colorblind-validated categorical palette (see the project's dataviz
notes) so condition identity stays distinguishable without relying on hue alone;
each figure also gets a legend so color is never the only channel carrying identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

_CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]

_SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

_GRID_COLOR = "#c8c7c2"


def _condition_colors(conditions: Sequence[str]) -> dict[str, str]:
    if len(conditions) > len(_CATEGORICAL_PALETTE):
        # Falls outside the validated 8-slot palette; cycle rather than fail.
        colors = [_CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)] for i in range(len(conditions))]
    else:
        colors = _CATEGORICAL_PALETTE[: len(conditions)]
    return dict(zip(conditions, colors))


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


_INK, _MUTED, _SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
# Raw samples are drawn as dots only when there are few enough to read; past this the box plot
# alone shows the distribution, and the per-recording means carry the individual points.
_MAX_RAW_POINTS = 150


def _header(fig: plt.Figure, title: str, key: str) -> tuple[float, float, float, float]:
    """Title and a one-line key at the top-left, placed in inches from the top so they never
    overlap, whatever the figure's height. Returns the rect for tight_layout below them."""
    height = fig.get_size_inches()[1]
    fig.text(0.012, 1 - 0.12 / height, title, fontsize=12, color=_INK, ha="left", va="top")
    fig.text(0.012, 1 - 0.40 / height, key, fontsize=8.5, color=_MUTED, ha="left", va="top")
    return (0, 0, 1, 1 - 0.62 / height)


def plot_condition_comparison(
    df: pd.DataFrame,
    signal_col: str,
    condition_col: str,
    output_path: str | Path,
    unit_cols: Sequence[str] = (),
) -> str:
    """The signal's distribution by condition: a box plot of every sample, with one dot per
    recording (its mean) when recording units are known. Recordings, not samples, are the
    independent observations the significance tests compare, and drawing thousands of
    autocorrelated samples as dots only hides the box. With no units and few samples, the
    samples themselves are drawn instead."""
    conditions = sorted(df[condition_col].dropna().unique(), key=str)
    colors = _condition_colors(conditions)
    values_by_condition = [df.loc[df[condition_col] == c, signal_col].to_numpy(dtype=float) for c in conditions]

    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(conditions) + 2.5), 4.5))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    _style_axes(ax)
    box = ax.boxplot(
        values_by_condition,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": _INK, "linewidth": 2},
        whiskerprops={"color": _MUTED, "linewidth": 1},
        capprops={"color": _MUTED, "linewidth": 1},
    )
    for patch, condition in zip(box["boxes"], conditions):
        patch.set_facecolor(colors[condition])
        patch.set_alpha(0.25)
        patch.set_edgecolor(colors[condition])
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(0)
    present_units = [c for c in unit_cols if c in df.columns]
    note = "box: all samples"
    for i, condition in enumerate(conditions, start=1):
        rows = df[df[condition_col] == condition]
        if present_units:
            points = rows.groupby(present_units, dropna=False)[signal_col].mean().to_numpy(dtype=float)
            note = "box: all samples · dots: one per recording (its mean)"
        elif rows.shape[0] <= _MAX_RAW_POINTS:
            points = rows[signal_col].to_numpy(dtype=float)
            note = "box and dots: all samples"
        else:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=points.size)
        ax.scatter(np.full(points.size, i) + jitter, points, s=46, color=colors[condition],
                   edgecolors=_SURFACE, linewidths=1.5, zorder=3)

    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions)
    ax.set_ylabel(signal_col)
    rect = _header(fig, f"{signal_col} by {condition_col}", note)

    fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return str(output_path)


def plot_complexity_metrics(
    complexity_df: pd.DataFrame,
    signal_col: str,
    metrics: Sequence[str],
    output_path: str | Path,
    unit_complexity: pd.DataFrame | None = None,
) -> str | None:
    """One small panel per metric comparing conditions: a dot per recording and a line at the
    condition mean (from complexity_df), plus a dashed line at the shuffled-surrogate mean when
    surrogates were computed. Panels wrap at three per row so many metrics stay legible, and
    each has its own y-axis, since the metrics live on incompatible scales. Without per-recording
    values, only the condition means are drawn."""
    signal_rows = complexity_df[complexity_df["signal"] == signal_col].sort_values("condition")
    available_metrics = [m for m in metrics if m in signal_rows.columns]
    if signal_rows.empty or not available_metrics:
        return None
    units = None
    if unit_complexity is not None and not unit_complexity.empty:
        units = unit_complexity[unit_complexity["signal"] == signal_col]

    conditions = list(signal_rows["condition"])
    colors = _condition_colors(sorted(set(conditions), key=str))
    n_cols = min(3, len(available_metrics))
    n_rows = int(np.ceil(len(available_metrics) / n_cols))
    width_per_panel = max(2.6, 1.1 * len(conditions) + 1.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_per_panel * n_cols + 0.6, 3.4 * n_rows + 0.8),
                             squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    rng = np.random.default_rng(0)
    for ax in axes.flat[len(available_metrics):]:
        ax.set_visible(False)

    for ax, metric in zip(axes.flat, available_metrics):
        ax.set_facecolor(_SURFACE)
        _style_axes(ax)
        for x, condition in enumerate(conditions):
            mean = float(signal_rows.loc[signal_rows["condition"] == condition, metric].iloc[0])
            if units is not None and metric in units.columns:
                points = units.loc[units["condition"] == condition, metric].dropna().to_numpy(dtype=float)
                if points.size:
                    ax.scatter(x + rng.uniform(-0.14, 0.14, size=points.size), points, s=34,
                               color=colors[condition], edgecolors=_SURFACE, linewidths=1.2, zorder=3)
            if np.isfinite(mean):
                ax.hlines(mean, x - 0.3, x + 0.3, color=_INK, linewidth=2.2, zorder=4)
                ax.annotate(f"{mean:.3g}", (x + 0.32, mean), fontsize=8.5, color=_INK, va="center")
            surrogate = f"{metric}_surrogate_mean"
            if units is not None and surrogate in units.columns:
                baseline = units.loc[units["condition"] == condition, surrogate].mean()
                if np.isfinite(baseline):
                    ax.hlines(baseline, x - 0.3, x + 0.3, color=_MUTED, linewidth=1.2, linestyles=(0, (3, 2)),
                              zorder=2)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions)
        ax.set_xlim(-0.6, len(conditions) - 0.2)
        ax.set_title(metric.replace("_", " "), loc="left", fontsize=10.5)

    has_surrogates = units is not None and any(f"{m}_surrogate_mean" in units.columns for m in available_metrics)
    key = "dots: one per recording · solid line: condition mean"
    if has_surrogates:
        key += " · dashed: shuffled-surrogate mean"
    rect = _header(fig, f"{signal_col}: complexity metrics by condition", key)
    fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return str(output_path)


def plot_mutual_information_heatmap(
    cross_signal_df: pd.DataFrame,
    output_path: str | Path,
) -> str | None:
    """Heatmap (one panel per condition) of pairwise mutual information between signals."""
    if cross_signal_df.empty:
        return None

    conditions = sorted(cross_signal_df["condition"].unique(), key=str)
    signals = sorted(set(cross_signal_df["signal_a"]) | set(cross_signal_df["signal_b"]), key=str)
    n = len(signals)

    fig, axes = plt.subplots(1, len(conditions), figsize=(3.5 * len(conditions) + 1, 3.5), squeeze=False)
    axes = axes[0]

    vmax = float(cross_signal_df["mutual_information"].max()) or 1.0
    last_im = None
    for ax, condition in zip(axes, conditions):
        matrix = np.full((n, n), np.nan)
        rows = cross_signal_df[cross_signal_df["condition"] == condition]
        for _, row in rows.iterrows():
            i, j = signals.index(row["signal_a"]), signals.index(row["signal_b"])
            matrix[i, j] = row["mutual_information"]
            matrix[j, i] = row["mutual_information"]

        last_im = ax.imshow(matrix, cmap=_SEQUENTIAL_BLUE, vmin=0, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_xticklabels(signals, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(signals)
        ax.set_title(str(condition))
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#0b0b0b")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.046, pad=0.04, label="mutual information (bits)")
    fig.suptitle("Cross-signal mutual information")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_degree_distribution(
    graphs: dict[str, nx.Graph],
    output_path: str | Path,
) -> str | None:
    """Small-multiple degree-distribution bar charts, one panel per condition/graph."""
    graphs = {name: g for name, g in graphs.items() if g.number_of_nodes() > 0}
    if not graphs:
        return None

    names = sorted(graphs, key=str)
    colors = _condition_colors(names)

    fig, axes = plt.subplots(1, len(names), figsize=(3.5 * len(names), 3.5), squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, names):
        _style_axes(ax)
        degrees = [d for _, d in graphs[name].degree()]
        max_degree = max(degrees) if degrees else 0
        bins = np.arange(0, max_degree + 2) - 0.5
        ax.hist(degrees, bins=bins, color=colors[name], zorder=2)
        ax.set_xlabel("degree")
        ax.set_ylabel("node count")
        ax.set_title(str(name))

    fig.suptitle("Degree distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)
