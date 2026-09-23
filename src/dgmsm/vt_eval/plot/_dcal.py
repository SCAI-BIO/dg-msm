# _dcal.py
from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from ._base import (
    _OKABE_ITO,
    _SUPTITLE_SIZE,
    _SUPTITLE_WEIGHT,
    _add_panel_label,
    _blank_panel,
    _pub_rc,
    _save_figure,
)

logger = logging.getLogger(__name__)

_ANNOT_LABEL_W = 12
_TABLE_FONT_SIZE = 8.5

def _format_scientific(value: Any, sig_figs: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    if number == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(number))))
    mantissa = number / (10 ** exponent)

    if round(abs(mantissa), sig_figs) >= 10:
        mantissa /= 10
        exponent += 1

    sign = "-" if number < 0 else ""
    return rf"${sign}{abs(mantissa):.{sig_figs}f}\times10^{{{exponent}}}$"

def _format_scalar(value: Any, spec: str = ".3f") -> str:
    """Format a scalar safely for a plot annotation or table."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"

    return format(number, spec) if np.isfinite(number) else "NA"

def _format_pvalue(
    value: Any,
    small_threshold: float = 1e-3,
    sig_figs: int = 1,
) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(p) or p < 0:
        return "NA"

    if p == 0.0:
        return r"$<2.2\times10^{-16}$"

    if p >= small_threshold:
        return f"{p:.2f}"

    return _format_scientific(p, sig_figs=sig_figs)


def _validate_table_width_ratio(
    value: float,
    *,
    parameter_name: str,
) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{parameter_name} must be a positive finite number."
        ) from exc

    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(
            f"{parameter_name} must be a positive finite number."
        )

    return ratio

def _blank_summary_panel(ax: plt.Axes, text: str = "n/a") -> None:
    """Clear and disable a dedicated summary-table axes."""
    ax.clear()
    ax.set_axis_off()

    if text:
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="0.5",
        )


def _draw_dcal_summary_table(
    ax: plt.Axes,
    rows: Sequence[tuple[str, str]],
) -> None:
    """Draw D-calibration diagnostics in a dedicated table axes."""
    if not rows:
        _blank_summary_panel(ax)
        return

    ax.clear()
    ax.set_axis_off()


    # One additional row is used for the header.
    n_table_rows = len(rows) + 1
    table_height = min(0.18 * n_table_rows, 0.82)


    table = ax.table(
        cellText=[list(row) for row in rows],
        colLabels=["Metric", "Value"],
        colWidths=[0.70, 0.30],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.50 - table_height / 2, 1.0, table_height],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(_TABLE_FONT_SIZE)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.PAD = 0.08
        cell.set_edgecolor("0.78")
        cell.set_linewidth(0.45)
        cell.set_facecolor("white")
        cell.visible_edges = "BT" if row_idx == 0 else "B"

        text = cell.get_text()
        text.set_color("0.15")
        text.set_ha("right" if col_idx == 1 else "left")

        if row_idx == 0:
            cell.set_facecolor("#F2F2F2")
            text.set_fontweight("bold")


def _normalise_bin_proportions(prop: np.ndarray) -> np.ndarray:
    """Validate and normalise a one-dimensional vector of bin proportions."""
    prop = np.asarray(prop, dtype=float)

    if prop.ndim != 1:
        raise ValueError("'bin_proportions' must be one-dimensional.")
    if len(prop) == 0:
        raise ValueError("'bin_proportions' is empty.")
    if np.any(~np.isfinite(prop)):
        raise ValueError("'bin_proportions' contains non-finite values.")
    if np.any(prop < 0):
        raise ValueError("'bin_proportions' contains negative values.")

    total = float(prop.sum())
    if total <= 0:
        raise ValueError("'bin_proportions' must sum to a positive value.")

    if not np.isclose(total, 1.0, atol=1e-3):
        logger.warning(
            "'bin_proportions' sums to %.6f, not 1. Normalising for plotting.",
            total,
        )
        prop = prop / total

    return prop


def _plot_landmark_reliability_panel(
    ax,
    dcal,
    *,
    plot_band: bool = False,
    show_folds: bool = False,
    plot_fill: bool = False,
    show_pvalue: bool = True,
    summary_ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plot one D-calibration P-P panel.

    If ``summary_ax`` is supplied, diagnostics are drawn in a dedicated
    table instead of the small annotation box inside the plot.
    """
    if dcal is None:
        _blank_panel(ax)
        if summary_ax is not None:
            _blank_summary_panel(summary_ax)
        return

    primary = _OKABE_ITO[0]

    prop = _normalise_bin_proportions(dcal.bin_proportions)
    nbins = len(prop)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    cum_obs = np.clip(
        np.concatenate([[0.0], np.cumsum(prop)]),
        0.0,
        1.0,
    )

    if show_folds and dcal.bin_proportions_folds is not None:
        folds = np.asarray(dcal.bin_proportions_folds, dtype=float)

        if folds.ndim == 1:
            folds = folds[None, :]

        if folds.ndim == 2 and folds.shape[1] == nbins:
            for fold_prop in folds:
                if np.any(~np.isfinite(fold_prop)) or fold_prop.sum() <= 0:
                    continue

                cum_fold = np.clip(
                    np.concatenate(
                        [
                            [0.0],
                            np.cumsum(_normalise_bin_proportions(fold_prop)),
                        ]
                    ),
                    0.0,
                    1.0,
                )

                ax.plot(
                    edges,
                    cum_fold,
                    lw=0.6,
                    color=primary,
                    alpha=0.22,
                    zorder=2,
                )
        else:
            logger.warning(
                "Ignoring 'bin_proportions_folds': expected shape "
                "[n_folds, %d], got %s.",
                nbins,
                folds.shape,
            )

    if dcal.bin_proportions_std is not None and plot_band:
        std_prop = np.asarray(dcal.bin_proportions_std, dtype=float)

        if std_prop.shape == prop.shape and np.all(np.isfinite(std_prop)):
            cum_upper = np.clip(
                np.concatenate(
                    [[0.0], np.cumsum(np.clip(prop + std_prop, 0.0, 1.0))]
                ),
                0.0,
                1.0,
            )
            cum_lower = np.clip(
                np.concatenate(
                    [[0.0], np.cumsum(np.clip(prop - std_prop, 0.0, 1.0))]
                ),
                0.0,
                1.0,
            )

            ax.fill_between(
                edges,
                cum_lower,
                cum_upper,
                color=primary,
                alpha=0.15,
                lw=0,
                zorder=2.5,
                label="±1 SD across folds",
            )
        else:
            logger.warning(
                "Ignoring 'bin_proportions_std': expected shape %s, got %s.",
                prop.shape,
                std_prop.shape,
            )

    if plot_fill:
        ax.fill_between(
            edges,
            edges,
            cum_obs,
            where=(cum_obs >= edges),
            color="#C0392B",
            alpha=0.12,
            lw=0,
            zorder=1.5,
        )
        ax.fill_between(
            edges,
            cum_obs,
            edges,
            where=(cum_obs < edges),
            color=primary,
            alpha=0.12,
            lw=0,
            zorder=1.5,
        )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        ls="--",
        lw=0.8,
        color="0.45",
        label="Perfect calibration",
        zorder=3,
    )
    ax.plot(
        edges,
        cum_obs,
        marker="o",
        ms=3.0,
        lw=1.4,
        zorder=4,
        color=primary,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="DG-MSM observed CDF",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])

    cdf_max_deviation = float(np.abs(cum_obs - edges).max())

    stats_rows = [
        ("Max. CDF dev.", _format_scalar(cdf_max_deviation)),
    ]


    if show_pvalue:
        stats_rows.extend(
            [
                ("R(θ)", _format_scientific(getattr(dcal, "r_theta", None))),
                ("Cohen's w", _format_scalar(getattr(dcal, "cohens_w", None), spec=".2f")),
                ("p(χ²)", _format_pvalue(getattr(dcal, "pvalue", None))),
            ]
        )

    if summary_ax is not None:
        _draw_dcal_summary_table(summary_ax, stats_rows)
        return

    text_lines = [
        f"{label:<{_ANNOT_LABEL_W}} = {value}"
        for label, value in stats_rows
    ]


    ax.text(
        0.96, 0.04, "\n".join(text_lines),
        transform=ax.transAxes, va="bottom", ha="right", fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.78", lw=0.5, alpha=0.90),
        zorder=5,
    )


def plot_landmark_reliability(
    soj_dcal=None,
    os_dcal=None,
    state_name = None,
    save_path  = None,
    show: bool = True,
    max_cols: int = 4,
    panel_size: float = 3.0,
    dpi: int = 300,
    plot_band: bool = False,
    show_pvalue: bool = True,
    target = None,
    show_folds: bool = False,
    close: bool = False,
) -> plt.Figure:
    """Plot sojourn-time and overall-survival D-calibration P-P panels."""
    soj_title = f"Sojourn time in {state_name}" if state_name else "Sojourn time"
    os_title = f"Survival from {state_name}" if state_name else "Overall survival"

    panels = [
        (soj_title, soj_dcal),
        (os_title, os_dcal),
    ]

    plot_kwargs = {
        "plot_band": plot_band,
        "show_folds": show_folds,
        "show_pvalue": show_pvalue,
    }

    with plt.rc_context(_pub_rc()):
        if target is not None:
            if len(target) != 2:
                raise ValueError(
                    "target must contain exactly two Axes/SubFigures."
                )

            fig = None

            for obj, (title, dcal) in zip(target, panels):
                if isinstance(obj, plt.Axes):
                    ax = obj
                    current_fig = obj.figure
                    ax.set_title(title, fontweight="bold")
                elif hasattr(obj, "subplots"):
                    ax = obj.subplots(1, 1)
                    current_fig = getattr(obj, "figure", None) or ax.figure
                    setter = obj.suptitle if hasattr(obj, "suptitle") else ax.set_title
                    setter(title, fontsize=9, fontweight="bold")
                else:
                    raise TypeError(
                        "target entries must be matplotlib Axes or "
                        "SubFigure-like."
                    )

                if fig is None:
                    fig = current_fig

                _plot_landmark_reliability_panel(ax, dcal, **plot_kwargs)
                ax.set_xlabel("Expected cumulative fraction")
                ax.set_ylabel("Observed cumulative fraction")

            if fig is None:
                fig = plt.gcf()

            if save_path is not None:
                _save_figure(fig, save_path, dpi=dpi)
            if show:
                plt.show()
            if close:
                plt.close(fig)

            return fig

        n = len(panels)
        ncols = min(max_cols, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(panel_size * ncols, panel_size * nrows + 0.5),
            dpi=dpi,
            squeeze=False,
        )

        for i, (title, dcal) in enumerate(panels):
            ax = axes[i // ncols, i % ncols]
            _plot_landmark_reliability_panel(ax, dcal, **plot_kwargs)
            ax.set_title(title, fontweight="bold")

        for row_idx in range(nrows):
            axes[row_idx, 0].set_ylabel("Observed cumulative fraction")

        for col_idx in range(ncols):
            axes[nrows - 1, col_idx].set_xlabel(
                "Expected cumulative fraction"
            )

        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        fig.suptitle(
            f"D-calibration P-P plot"
            f"{' – ' + str(state_name) if state_name else ''}",
            fontsize=_SUPTITLE_SIZE,
            fontweight=_SUPTITLE_WEIGHT,
        )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(len(handles), 4),
                frameon=False,
                bbox_to_anchor=(0.5, 0.02),
            )

        fig.tight_layout(rect=[0.02, 0.11, 1.0, 0.92], pad=1.2)

        if save_path is not None:
            _save_figure(fig, save_path, dpi=dpi)
        if show:
            plt.show()
        if close:
            plt.close(fig)

    return fig


def plot_cv_landmark_pp(
    cv_result,
    landmark_labels = None,
    row_keys = None,
    show: bool = True,
    save_path = None,
    dpi: int = 300,
    panel_size: float = 3.0,
    show_folds: bool = True,
    show_pvalue: bool = True,
    plot_band: bool = False,
    close: bool = False,
    plot_fill: bool = False,
    dcal_table_width_ratio: float = 0.82,
) -> plt.Figure:
    """CV-aggregated D-calibration P-P grid, optionally with tables."""
    states = cv_result.states()

    if not states:
        raise ValueError("cv_result has no landmark states.")

    if row_keys is None:
        row_keys = [
            ("soj_dcal", "Sojourn Time"),
            ("os_dcal", "Overall Survival"),
        ]

    table_ratio = _validate_table_width_ratio(
        dcal_table_width_ratio,
        parameter_name="dcal_table_width_ratio",
    )

    n_state_rows = len(states)
    n_endpoint_cols = len(row_keys)


    cell_width = 1.0 + table_ratio

    with plt.rc_context(_pub_rc()):
        fig = plt.figure(
            figsize=(
                panel_size * n_endpoint_cols * cell_width,
                panel_size * n_state_rows + 0.65,
            ),
            dpi=dpi,
        )

        gs = fig.add_gridspec(
            n_state_rows,
            n_endpoint_cols,
            left=0.075,
            right=0.985,
            bottom=0.075,
            top=0.92,
            wspace=0.16,
            hspace=0.16,
        )

        dcal_axes: list[plt.Axes] = []
        panel_idx = 0

        for state_row, state in enumerate(states):
            entry = cv_result.entry(state)

            state_label = (
                str(landmark_labels[state])
                if landmark_labels and state in landmark_labels
                else entry["state_label"]
            )

            for endpoint_col, (dcal_key, title_fmt) in enumerate(row_keys):
                outer_spec = gs[state_row, endpoint_col]

                inner = outer_spec.subgridspec(
                    1,
                    2,
                    width_ratios=[1.0, table_ratio],
                    wspace=0.05,
                )
                ax = fig.add_subplot(inner[0, 0])
                summary_ax = fig.add_subplot(inner[0, 1])


                dcal_axes.append(ax)

                _plot_landmark_reliability_panel(
                    ax,
                    entry.get(dcal_key),
                    plot_band=plot_band,
                    show_folds=show_folds,
                    plot_fill=plot_fill,
                    show_pvalue=show_pvalue,
                    summary_ax=summary_ax,
                )

                _add_panel_label(ax, panel_idx)

                endpoint_label = title_fmt.format("").rstrip(" —").strip()

                # Endpoint names are column headings.
                if state_row == 0:
                    ax.set_title(
                        endpoint_label,
                        fontsize=9,
                        fontweight="bold",
                    )

                # Landmark/state names identify rows.
                ax.set_ylabel(
                    f"{state_label}\nObserved CDF"
                    if endpoint_col == 0
                    else ""
                )

                ax.set_xlabel(
                    "Expected cumulative fraction"
                    if state_row == n_state_rows - 1
                    else ""
                )

                panel_idx += 1

        fig.suptitle(
            "D-Calibration",
            fontsize=_SUPTITLE_SIZE,
            fontweight=_SUPTITLE_WEIGHT,
            y=0.975,
        )

        for ax in dcal_axes:
            handles, labels = ax.get_legend_handles_labels()

            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=min(len(handles), 4),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.018),
                    fontsize=8,
                )
                break

        if save_path is not None:
            _save_figure(fig, save_path, dpi=dpi)

        if show:
            plt.show()

        if close:
            plt.close(fig)

    return fig