import logging
import math
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from chap_core.explainability.perturb import SampleModel
from chap_core.explainability.plot import parse_coefficients
from chap_core.models.utils import CHAP_RUNS_DIR
from chap_core.util import generate_run_name

logger = logging.getLogger(__name__)

"""
This files contains code from the LIME equivlant code for saving and plotting the explanation result.

This is just for testing and is work in progress.

Some parts of the code have been altered, by most of it is direclty used from LIME.
"""

def make_explainability_run_dir(model_name: str | None, run_name: str) -> Path:
    """Creates and return runs/explainability/<model_name>/<run_name>."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in (model_name or "unknown"))
    run_dir = CHAP_RUNS_DIR / "explainability" / safe_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    run_name = generate_run_name()
    timestamp = run_name.rsplit("_", 1)[0]
    run_dir = make_explainability_run_dir(model_name, run_name)

    lines = [
        "# SHAP Explanation",
        "",
        f"**Model:** {model_name or 'unknown'}  ",
        f"**Location:** {location}  ",
        f"**Horizon:** {horizon}  ",
        f"**Generated:** {timestamp}  ",
        "",
    ]
    lines += [
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
            "### How each variable moved the prediction away from the baseline",
            "![Waterfall](waterfall_plot.png)",
        ]
        # Grouped (fast) runs have one value per variable, so there is no per-segment importance plot
        if any("_seg_" in name or "_fut_" in name for name, _ in results):
            lines += [
                "",
                "### When each variable raised or lowered the prediction",
                "![Feature importance](importance_plot.png)",
            ]

    md_path = run_dir / "explanation.md"
    md_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Explanation saved to {md_path}")
    return md_path


def variable_totals(coefficients: list[tuple[str, float]]) -> dict[str, float]:
    """Sum the SHAP values of all segments and future steps of each variable."""
    temp_columns, static_columns = parse_coefficients(coefficients)
    totals = {name: sum(lags.values()) for name, lags in temp_columns.items()}
    totals.update(static_columns)
    return totals


def draw_waterfall(
    ax,
    totals: dict[str, float],
    baseline_prediction: float,
    prediction: float,
    colors: dict[str, tuple],
    fontsize: int = 9,
):
    """Baseline -> one step per variable (largest |SHAP| first) -> prediction, on a single axis."""
    names = sorted(totals, key=lambda name: abs(totals[name]), reverse=True)
    labels = ["Baseline", *names, "Prediction"]
    width = 0.6

    def label(x, top, text):
        ax.annotate(
            text, (x, top), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize
        )

    ax.bar(0, baseline_prediction, width, color="darkgray")
    label(0, baseline_prediction, f"{baseline_prediction:.1f}")

    level = baseline_prediction
    for i, name in enumerate(names, start=1):
        value = totals[name]
        ax.plot([i - 1 + width / 2, i - width / 2], [level, level], color="gray", linewidth=0.8)
        ax.bar(i, value, width, bottom=level, color=colors[name])
        label(i, max(level, level + value), f"{value:+.1f}")
        level += value

    end = len(labels) - 1
    ax.plot([end - 1 + width / 2, end - width / 2], [level, level], color="gray", linewidth=0.8)
    ax.bar(end, prediction, width, color="black")
    label(end, prediction, f"{prediction:.1f}")

    # Headroom so the value labels don't run into the title
    levels = baseline_prediction + np.cumsum([0.0] + [totals[name] for name in names])
    low, high = min(0.0, *levels, prediction), max(0.0, *levels, prediction)
    ax.set_ylim(low - 0.05 * (high - low), high + 0.15 * (high - low))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.spines[["top", "right"]].set_visible(False)


def plot_shap_waterfall(
    coefficients: list[tuple[str, float]],
    baseline_prediction: float,
    prediction: float,
    save_path: Path | None = None,
):
    """
    Waterfall from the baseline prediction to the prediction, with one step per variable.

    The SHAP values of all segments and future steps of a variable are summed.
    """
    totals = variable_totals(coefficients)
    names = sorted(totals, key=lambda name: abs(totals[name]), reverse=True)

    # One colour per variable group
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % cmap.N) for i, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(1.4 * (len(names) + 2) + 1, 4.5))
    draw_waterfall(ax, totals, baseline_prediction, prediction, colors)
    ax.set_ylabel("Prediction (disease cases)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Waterfall plot saved to {save_path}")
    else:
        plt.show()


def plot_shap_waterfall_grid(
    explanations: dict[str, tuple[list[tuple[str, float]], float, float]],
    save_path: Path | None = None,
):
    """
    One waterfall per location in a grid, based on plot_all_locations.py.

    ``explanations`` maps location -> (SHAP results, prediction, baseline prediction), as returned
    by kernel_shap.explain. Each variable has the same colour in every panel. Y-axes are not shared.
    """
    totals = {loc: variable_totals(results) for loc, (results, _, _) in explanations.items()}

    # Colours by mean |SHAP| over locations, panels by prediction (highest first)
    variables = pd.DataFrame(totals).T.fillna(0.0).abs().mean().sort_values(ascending=False).index
    cmap = plt.get_cmap("tab10")
    colors = {var: cmap(i % cmap.N) for i, var in enumerate(variables)}
    locations = sorted(explanations, key=lambda loc: explanations[loc][1], reverse=True)

    n = len(locations)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)

    for ax, loc in zip(axes.flat, locations, strict=False):
        _, prediction, baseline_prediction = explanations[loc]
        draw_waterfall(ax, totals[loc], baseline_prediction, prediction, colors, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(loc, fontsize=11, loc="left")
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    handles = [Rectangle((0, 0), 1, 1, color=color, label=var) for var, color in colors.items()]
    fig.legend(handles=handles, loc="upper right", ncols=len(handles), frameon=False, fontsize=9)
    fig.suptitle("SHAP explanation per location. Y-axes are not shared.", x=0.01, ha="left", fontsize=12)
    fig.supylabel("Prediction (disease cases)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Waterfall grid saved to {save_path}")
    else:
        plt.show()


def plot_shap_values(
    coefficients: list[tuple[str, float]],
    hist_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    segment_indices: dict[str, dict[int, tuple[int, int]]],
    sampler: SampleModel | None = None,
    save_path: Path | None = None,
):
    """
    Based on plot_importance in plot.py

    If sampler is given, the baseline (the values a feature is replaced with when turned off) is drawn as a red line.
    """
    temp_columns, static_columns = parse_coefficients(coefficients)

    max_temp = max((abs(v) for inner in temp_columns.values() for v in inner.values()), default=0.0)
    max_static = max((abs(v) for v in static_columns.values()), default=0.0)
    max_val = max(max_temp, max_static) or 1.0

    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)

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
            baseline = None
            if sampler is not None:
                baseline = [float("nan")] * len(y)
                for startx, endx in hist_indices.values():
                    baseline[startx:endx] = sampler.sample(hist_df, (startx, endx), col, endx - startx)
                ax_hist.plot(range(len(y)), baseline, color="red", linewidth=1.2, alpha=0.7, label="baseline")
                if i == 0:
                    ax_hist.legend(fontsize=7, loc="upper left")
            for lag, (startx, endx) in hist_indices.items():
                ax_hist.axvline(x=endx, color="darkgray", linestyle="--", linewidth=1.0)
                value = temp_columns[col].get(lag, 0.0)
                if baseline is None:
                    ax_hist.axvspan(startx, endx, facecolor=cmap(norm(value)), alpha=0.5, zorder=0)
                else:
                    # Fill only the gap between input and baseline; include endx so neighbouring segments touch
                    stop = min(endx + 1, len(y))
                    ax_hist.fill_between(
                        range(startx, stop),
                        y[startx:stop],
                        baseline[startx:stop],
                        color=cmap(norm(value)),
                        alpha=0.7,
                        linewidth=0,
                    )
                label(ax_hist, (startx + endx) / 2, value)
        else:
            ax_hist.axis("off")

        ax_hist.set_title(col, fontsize=9)
        ax_hist.tick_params(labelsize=7)

        # Future: single bar (horizon=1) or line + axvspan coloring (horizon>=2)
        fut_coeffs = {abs(lag): val for lag, val in temp_columns[col].items() if lag < 0}
        if col in fut_df.columns and fut_coeffs:
            y_fut = fut_df[col].tolist()
            baseline_fut = None
            if sampler is not None and hasattr(sampler, "sample_future"):
                baseline_fut = [sampler.sample_future(hist_df, fut_df, col, j) for j in range(len(y_fut))]
                if len(y_fut) == 1:
                    ax_fut.hlines(baseline_fut[0], -0.4, 0.4, color="red", linewidth=1.2, alpha=0.7, zorder=3)
                else:
                    ax_fut.plot(range(len(y_fut)), baseline_fut, color="red", linewidth=1.2, alpha=0.7, zorder=3)
            if len(y_fut) == 1:
                value = fut_coeffs.get(1, 0.0)
                bottom = baseline_fut[0] if baseline_fut is not None else 0.0
                ax_fut.bar(0, y_fut[0] - bottom, bottom=bottom, color=cmap(norm(value)), alpha=0.6, edgecolor="none")
                label(ax_fut, 0, value)
            else:
                ax_fut.plot(range(len(y_fut)), y_fut)
                for j in range(len(y_fut)):
                    value = fut_coeffs.get(j + 1, 0.0)
                    if baseline_fut is None:
                        ax_fut.axvspan(j - 0.5, j + 0.5, facecolor=cmap(norm(value)), alpha=0.5, zorder=0)
                    else:
                        # Fill the gap over the half-step on each side of j, interpolating both lines
                        xs = [max(j - 0.5, 0), j, min(j + 0.5, len(y_fut) - 1)]
                        ax_fut.fill_between(
                            xs,
                            np.interp(xs, range(len(y_fut)), y_fut),
                            np.interp(xs, range(len(y_fut)), baseline_fut),
                            color=cmap(norm(value)),
                            alpha=0.7,
                            linewidth=0,
                        )
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

    ticks = [s * max_val * f for s in (-1, 1) for f in (1, 0.5)] + [0.0]
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
