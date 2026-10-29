import logging
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from chap_core.explainability.plot import parse_coefficients
from chap_core.models.utils import CHAP_RUNS_DIR
from chap_core.util import generate_run_name

logger = logging.getLogger(__name__)

"""
This files contains code from the LIME equivlant code for saving and plotting the explanation result.

This is just for testing and is work in progress.

Some parts of the code have been altered, by most of it is direclty used from LIME.
"""

def save_shap_explanation(
    results: list[tuple[str, float]],
    model_name: str | None,
    location: str,
    horizon: int,
    prediction: float,
    baseline_prediction: float,
    params: dict,
    include_plot: bool = False,
) -> Path:
    """
    Based on implementation from LIME pipeline currently
    """
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in (model_name or "unknown"))

    run_name = generate_run_name()
    timestamp = run_name.rsplit("_", 1)[0]
    run_dir = CHAP_RUNS_DIR / "explainability" / safe_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# SHAP Explanation",
        "",
        f"**Model:** {model_name or 'unknown'}  ",
        f"**Location:** {location}  ",
        f"**Horizon:** {horizon}  ",
        f"**Generated:** {timestamp}  ",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in params.items():
        lines.append(f"| {k} | {v} |")

    shap_sum = sum(value for _, value in results)

    lines += [
        "",
        "## Additivity Check",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Prediction, f(x) | {prediction:.4f} |",
        f"| Baseline prediction | {baseline_prediction:.4f} |",
        f"| sum(SHAP) | {shap_sum:.4f} |",
        f"| Prediction - baseline | {prediction - baseline_prediction:.4f} |",
        f"| Additivity gap | {shap_sum - (prediction - baseline_prediction):+.4f} |",
        f"| Features | {len(results)} |",
        "",
        "## Feature Importance",
        "",
        "| Feature | SHAP value |",
        "|---------|------------|",
    ]
    for name, value in results:
        lines.append(f"| {name} | {value:+.4f} |")

    if include_plot:
        lines += [
            "",
            "## Plots",
            "",
            "### How much each variable influenced the prediction",
            "![Contribution percentages](percentage_plot.png)",
            "",
            "### When each variable raised or lowered the prediction",
            "![Feature importance](importance_plot.png)",            
        ]

    md_path = run_dir / "explanation.md"
    md_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Explanation saved to {md_path}")
    return md_path


def plot_shap_percentages(
    coefficients: list[tuple[str, float]],
    save_path: Path | None = None,
):
    temp_columns, static_columns = parse_coefficients(coefficients)

    totals = {name: sum(abs(v) for v in lags.values()) for name, lags in temp_columns.items()}
    totals.update({name: abs(value) for name, value in static_columns.items()})

    total = sum(totals.values())

    names = sorted(totals, key=totals.__getitem__, reverse=True)
    percentages = [100.0 * totals[name] / total for name in names]

    fig, ax = plt.subplots(figsize=(6, max(2.0, 0.5 * len(names))))
    ax.barh(names, percentages, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Share of total influence (%)")
    ax.set_xlim(0, max(percentages) * 1.15)
    for i, pct in enumerate(percentages):
        ax.text(pct + max(percentages) * 0.02, i, f"{pct:.0f}%", va="center", fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Percentage plot saved to {save_path}")
    else:
        plt.show()


def plot_shap_values(
    coefficients: list[tuple[str, float]],
    hist_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    segment_indices: dict[str, dict[int, tuple[int, int]]],
    save_path: Path | None = None,
):
    """
    Based on plot_importance in plot.py
    """
    temp_columns, static_columns = parse_coefficients(coefficients)

    max_temp = max((abs(v) for inner in temp_columns.values() for v in inner.values()), default=0.0)
    max_static = max((abs(v) for v in static_columns.values()), default=0.0)
    max_val = max(max_temp, max_static) or 1.0

    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.SymLogNorm(linthresh=max_val / 100, vmin=-max_val, vmax=max_val)

    def label(ax, x, value):
        ax.text(x, 0.95, f"{value:+.3f}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)

    sorted_vars = sorted(temp_columns.keys())
    has_static = bool(static_columns)
    n_rows = len(sorted_vars) + (1 if has_static else 0)

    fig = plt.figure(figsize=(14, max(4, n_rows * 2)), constrained_layout=True)
    gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[85, 15])

    first_ax_hist = None
    for i, col in enumerate(sorted_vars):
        ax_hist = fig.add_subplot(gs[i, 0], sharex=first_ax_hist)
        ax_fut = fig.add_subplot(gs[i, 1])
        if first_ax_hist is None:
            first_ax_hist = ax_hist

        # Historical
        hist_indices = segment_indices.get(col)
        if col in hist_df.columns and hist_indices is not None:
            y = hist_df[col].tolist()
            ax_hist.plot(range(len(y)), y)
            for lag, (startx, endx) in hist_indices.items():
                ax_hist.axvline(x=endx, color="darkgray", linestyle="--", linewidth=1.0)
                value = temp_columns[col].get(lag, 0.0)
                ax_hist.axvspan(startx, endx, facecolor=cmap(norm(value)), alpha=0.5, zorder=0)
                label(ax_hist, (startx + endx) / 2, value)
        else:
            ax_hist.axis("off")

        ax_hist.set_title(col, fontsize=9)
        ax_hist.tick_params(labelsize=7)

        # Future: single bar (horizon=1) or line + axvspan coloring (horizon>=2)
        fut_coeffs = {abs(lag): val for lag, val in temp_columns[col].items() if lag < 0}
        if col in fut_df.columns and fut_coeffs:
            y_fut = fut_df[col].tolist()
            if len(y_fut) == 1:
                value = fut_coeffs.get(1, 0.0)
                ax_fut.bar(0, y_fut[0], color=cmap(norm(value)), alpha=0.6, edgecolor="none")
                label(ax_fut, 0, value)
            else:
                ax_fut.plot(range(len(y_fut)), y_fut)
                for j in range(len(y_fut)):
                    value = fut_coeffs.get(j + 1, 0.0)
                    ax_fut.axvspan(j - 0.5, j + 0.5, facecolor=cmap(norm(value)), alpha=0.5, zorder=0)
                    label(ax_fut, j, value)
                for j in range(len(y_fut) - 1):
                    ax_fut.axvline(x=j + 0.5, color="darkgray", linestyle="--", linewidth=1.0)
            ax_fut.set_xticks(range(len(y_fut)))
            ax_fut.set_xticklabels([f"+{k + 1}" for k in range(len(y_fut))], fontsize=7)
            ax_fut.set_title("future", fontsize=8)
            ax_fut.tick_params(labelsize=7)
        else:
            ax_fut.axis("off")

    # Static features: one colored box per feature in the bottom row
    if has_static:
        ax_static = fig.add_subplot(gs[n_rows - 1, :])
        names = sorted(static_columns.keys())
        colors = [cmap(norm(static_columns[n])) for n in names]
        ax_static.bar(range(len(names)), [1] * len(names), color=colors, alpha=0.6, edgecolor="none")
        for k, n in enumerate(names):
            label(ax_static, k, static_columns[n])
        ax_static.set_xticks(range(len(names)))
        ax_static.set_xticklabels(names, fontsize=9)
        ax_static.set_yticks([])
        ax_static.set_title("Static features", fontsize=9)

    ticks = [s * max_val / d for s in (-1, 1) for d in (1, 10, 100)] + [0.0]
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=fig.axes,
        label="SHAP value (change in prediction from baseline)",
        alpha=0.5,
        fraction=0.02,
        pad=0.01,
        ticks=sorted(ticks),
    )
    cbar.ax.set_yticklabels([f"{t:+.3g}" for t in sorted(ticks)], fontsize=7)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Importance plot saved to {save_path}")
    else:
        plt.show()