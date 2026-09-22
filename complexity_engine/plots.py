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
    "#4a3aa7",  # violet
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


def plot_condition_comparison(
    df: pd.DataFrame,
    signal_col: str,
    condition_col: str,
    output_path: str | Path,
) -> str:
    """Box plot (with jittered raw points) of the raw signal by condition."""
    conditions = sorted(df[condition_col].dropna().unique(), key=str)
    colors = _condition_colors(conditions)
    values_by_condition = [df.loc[df[condition_col] == c, signal_col].to_numpy(dtype=float) for c in conditions]

    fig, ax = plt.subplots(figsize=(max(4, 1.4 * len(conditions) + 2), 4))
    _style_axes(ax)

    box = ax.boxplot(
        values_by_condition,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#0b0b0b", "linewidth": 2},
        whiskerprops={"color": "#52514e", "linewidth": 1},
        capprops={"color": "#52514e", "linewidth": 1},
        flierprops={"markersize": 3, "markerfacecolor": "#52514e", "markeredgecolor": "none"},
    )
    for patch, condition in zip(box["boxes"], conditions):
        patch.set_facecolor(colors[condition])
        patch.set_alpha(0.35)
        patch.set_edgecolor(colors[condition])
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(0)
    for i, values in enumerate(values_by_condition, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax.scatter(np.full(values.size, i) + jitter, values, color="#52514e", alpha=0.5, s=12, zorder=3)

    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions)
    ax.set_ylabel(signal_col)
    ax.set_title(f"{signal_col} by {condition_col}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def plot_complexity_metrics(
    complexity_df: pd.DataFrame,
    signal_col: str,
    metrics: Sequence[str],
    output_path: str | Path,
) -> str | None:
    """Small-multiple bar charts (one per metric) comparing conditions.

    Each metric gets its own y-axis rather than one shared axis, since sample
    entropy, spectral entropy, and the DFA exponent live on incompatible scales.
    """
    signal_rows = complexity_df[complexity_df["signal"] == signal_col].sort_values("condition")
    available_metrics = [m for m in metrics if m in signal_rows.columns]
    if signal_rows.empty or not available_metrics:
        return None

    conditions = list(signal_rows["condition"])
    colors = _condition_colors(sorted(set(conditions), key=str))

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(3.2 * len(available_metrics), 4), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, available_metrics):
        _style_axes(ax)
        values = signal_rows[metric].to_numpy(dtype=float)
        bar_colors = [colors[c] for c in conditions]
        bars = ax.bar(range(len(conditions)), values, color=bar_colors, width=0.6, zorder=2)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            offset = 0.02 * (np.nanmax(np.abs(values)) or 1.0)
            va = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset if value >= 0 else value - offset,
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=8,
                color="#0b0b0b",
            )
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions)
        ax.set_title(metric.replace("_", " "))
        ax.axhline(0, color=_GRID_COLOR, linewidth=1, zorder=1)

    fig.suptitle(f"{signal_col}: complexity metrics by condition")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
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
