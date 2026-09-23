# _pi.py
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

_PI_LO = 0.4
_PI_TICKS = np.round(np.arange(0.4, 1.0 + 1e-9, 0.1), 1)
_DEFAULT_REFERENCE_ALPHAS: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
_TABLE_FONT_SIZE = 8.0


def _format_scalar(value: Any, spec: str = ".3f") -> str:
    """Format a scalar safely for a table or annotation."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "NA"

    return format(number, spec) if np.isfinite(number) else "NA"


def _format_count(value: Any) -> str:
    """Format an integer-like count safely."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return "NA"



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


def _draw_pi_summary_table(
    ax: plt.Axes,
    rows: Sequence[tuple[str, str]],
) -> None:
    """Draw PI-coverage diagnostics in a dedicated table axes."""
    if not rows:
        _blank_summary_panel(ax)
        return

    ax.clear()
    ax.set_axis_off()


    # Includes the header row.
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


def _pi_summary_rows(
    nominal: np.ndarray,
    empirical: np.ndarray,
    *,
    n: Optional[int] = None,
    n_events: Optional[int] = None,
    show_n: bool = False,
    det_lower_bound: Optional[np.ndarray] = None,
    reference_alphas: Sequence[float] = _DEFAULT_REFERENCE_ALPHAS,
) -> list[tuple[str, str]]:
    """Create table rows for a PI-coverage summary."""
    nominal = np.asarray(nominal, dtype=float)
    empirical = np.asarray(empirical, dtype=float)

    rows: list[tuple[str, str]] = []

    if show_n:
        rows.append(("N", _format_count(n)))

        if n_events is not None:
            rows.append(("Events", _format_count(n_events)))

    last_idx = None

    for reference_alpha in reference_alphas:
        try:
            alpha = float(reference_alpha)
        except (TypeError, ValueError):
            continue

        idx = int(np.argmin(np.abs(nominal - alpha)))

        if np.isclose(nominal[idx], alpha, atol=1e-6):
            rows.append(
                (
                    f"PICP({alpha:.2f})",
                    _format_scalar(empirical[idx]),
                )
            )
            last_idx = idx

    if det_lower_bound is not None and last_idx is not None:
        det_lower_bound = np.asarray(det_lower_bound, dtype=float)

        if det_lower_bound.shape == empirical.shape:
            gap = abs(empirical[last_idx] - det_lower_bound[last_idx])

            rows.append(
                (
                    f"IPCW–det. gap @ {nominal[last_idx]:.2f}",
                    _format_scalar(gap),
                )
            )

    return rows


def _pi_annotation_lines(
    nominal: np.ndarray,
    empirical: np.ndarray,
    *,
    n: Optional[int] = None,
    n_events: Optional[int] = None,
    show_n: bool = False,
    det_lower_bound: Optional[np.ndarray] = None,
    reference_alphas: Sequence[float] = _DEFAULT_REFERENCE_ALPHAS,
) -> list[str]:
    """Create backwards-compatible in-plot PI annotation text."""
    nominal = np.asarray(nominal, dtype=float)
    empirical = np.asarray(empirical, dtype=float)

    lines: list[str] = []

    if show_n:
        n_text = f"N    = {_format_count(n)}"

        if n_events is not None:
            n_text += f" (n={_format_count(n_events)})"

        lines.append(n_text)

    last_idx = None

    for reference_alpha in reference_alphas:
        try:
            alpha = float(reference_alpha)
        except (TypeError, ValueError):
            continue

        idx = int(np.argmin(np.abs(nominal - alpha)))

        if np.isclose(nominal[idx], alpha, atol=1e-6):
            lines.append(
                f"PICP({alpha:.2f})  = {_format_scalar(empirical[idx])}"
            )
            last_idx = idx

    if det_lower_bound is not None and last_idx is not None:
        det_lower_bound = np.asarray(det_lower_bound, dtype=float)

        if det_lower_bound.shape == empirical.shape:
            gap = abs(empirical[last_idx] - det_lower_bound[last_idx])

            lines.append(
                f"IPCW-det. gap @ {nominal[last_idx]:.2f} = "
                f"{_format_scalar(gap)}"
            )

    return lines





def _plot_pi_panel(
    ax: plt.Axes,
    pi_cov=None,
    *,
    show_folds: bool = True,
    show_det_bound: bool = False,
    plot_band: bool = False,
    annotate: bool = True,
    primary: str = _OKABE_ITO[0],
    det_col: str = _OKABE_ITO[1],
    summary_ax: p = None,
    summary_show_n: bool = False,
    summary_reference_alphas: Sequence[float] = _DEFAULT_REFERENCE_ALPHAS,
) -> None:
    """
    Draw one PICP(alpha) panel.

    If ``summary_ax`` is provided, PI diagnostics are drawn in that axes as
    a table rather than in a small annotation box inside the plot.
    """
    if pi_cov is None:
        _blank_panel(ax)

        if summary_ax is not None:
            _blank_summary_panel(summary_ax)

        return

    nominal = np.asarray(pi_cov.alphas, dtype=float)
    empirical = np.asarray(pi_cov.empirical, dtype=float)

    if nominal.ndim != 1 or empirical.ndim != 1:
        raise ValueError("'alphas' and 'empirical' must be one-dimensional.")

    if nominal.shape != empirical.shape:
        raise ValueError(
            "'alphas' and 'empirical' must have identical shapes; "
            f"got {nominal.shape} and {empirical.shape}."
        )

    if nominal.size == 0:
        _blank_panel(ax)

        if summary_ax is not None:
            _blank_summary_panel(summary_ax)

        return

    raw_det_lower_bound = getattr(pi_cov, "det_lower_bound", None)

    det_lower_bound = (
        np.asarray(raw_det_lower_bound, dtype=float)
        if show_det_bound and raw_det_lower_bound is not None
        else None
    )

    n_events = getattr(pi_cov, "n_events", None)

    ax.plot(
        [_PI_LO, 1.0],
        [_PI_LO, 1.0],
        ls="--",
        lw=0.8,
        color="0.45",
        label="Perfect coverage",
        zorder=3,
    )

    if det_lower_bound is not None and det_lower_bound.shape == empirical.shape:
        ax.fill_between(
            nominal,
            det_lower_bound,
            empirical,
            where=(empirical >= det_lower_bound),
            color=primary,
            alpha=0.12,
            lw=0,
            zorder=2,
            label="PICP(α) IPCW-imputed gap",
        )

    empirical_folds = getattr(pi_cov, "empirical_folds", None)

    if show_folds and empirical_folds is not None:
        folds = np.asarray(empirical_folds, dtype=float)

        if folds.ndim == 2 and folds.shape[1] == nominal.size:
            for fold_empirical in folds:
                ax.plot(
                    nominal,
                    fold_empirical,
                    lw=0.6,
                    color=primary,
                    alpha=0.22,
                    zorder=2,
                )
        else:
            logger.warning(
                "Ignoring 'empirical_folds': expected shape [n_folds, %d], "
                "got %s.",
                nominal.size,
                folds.shape,
            )

    empirical_std = getattr(pi_cov, "empirical_std", None)

    if plot_band and empirical_std is not None:
        std = np.asarray(empirical_std, dtype=float)

        if std.shape == empirical.shape and np.all(np.isfinite(std)):
            ax.fill_between(
                nominal,
                np.clip(empirical - std, 0.0, 1.0),
                np.clip(empirical + std, 0.0, 1.0),
                color=primary,
                alpha=0.15,
                lw=0,
                zorder=2.5,
                label="±1 SD across folds",
            )
        else:
            logger.warning(
                "Ignoring 'empirical_std': expected shape %s with finite "
                "values, got %s.",
                empirical.shape,
                std.shape,
            )

    if det_lower_bound is not None and det_lower_bound.shape == empirical.shape:
        ax.plot(
            nominal,
            det_lower_bound,
            lw=1.2,
            ls=":",
            color=det_col,
            marker="s",
            ms=3.5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            zorder=3.5,
            label="PICP(α) lower bound (assumption-free)",
        )

    ax.plot(
        nominal,
        empirical,
        lw=1.4,
        marker="o",
        ms=4.0,
        color=primary,
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=4,
        label="PICP(α), IPCW-weighted",
    )

    ax.set_xlim(_PI_LO, 1.0)
    ax.set_ylim(_PI_LO, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(_PI_TICKS)
    ax.set_yticks(_PI_TICKS)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="major", lw=0.4, alpha=0.35)

    if summary_ax is not None:
        if annotate:
            rows = _pi_summary_rows(
                nominal,
                empirical,
                n=getattr(pi_cov, "n", None),
                n_events=n_events,
                show_n=summary_show_n,
                det_lower_bound=det_lower_bound,
                reference_alphas=summary_reference_alphas,
            )
            _draw_pi_summary_table(summary_ax, rows)
        else:
            _blank_summary_panel(summary_ax, text="")

        return

    if annotate:
        lines = _pi_annotation_lines(
            nominal,
            empirical,
            n=getattr(pi_cov, "n", None),
            n_events=n_events,
            show_n=summary_show_n,
            det_lower_bound=det_lower_bound,
            reference_alphas=summary_reference_alphas,
        )

        if lines:
            ax.text(
                0.96,
                0.04,
                "\n".join(lines),
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=6.5,
                bbox=dict(
                    boxstyle="round,pad=0.30",
                    fc="white",
                    ec="0.78",
                    lw=0.5,
                    alpha=0.90,
                ),
                zorder=5,
            )


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
def plot_cv_landmark_pi_coverage(
    cv_result,
    landmark_labels: Optional[[Mapping[object, str]]] = None,
    row_keys: Optional[Sequence[tuple[str, str]]] = None,
    show: bool = True,
    save_path = None,
    dpi: int = 300,
    panel_size: float = 3.0,
    show_folds: bool = True,
    show_det_bound: bool = False,
    plot_band: bool = True,
    close: bool = False,
    pi_table_width_ratio: float = 0.82,
) -> plt.Figure:
    """CV-aggregated PI-coverage grid, optionally with summary tables."""
    states = cv_result.states()

    if not states:
        raise ValueError("cv_result has no landmark states.")

    if row_keys is None:
        row_keys = [
            ("soj_pi", "Sojourn Time"),
            ("os_pi", "Overall Survival"),
        ]

    table_ratio = _validate_table_width_ratio(
        pi_table_width_ratio,
        parameter_name="pi_table_width_ratio",
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

        pi_axes: list[plt.Axes] = []
        panel_idx = 0

        for state_row, state in enumerate(states):
            entry = cv_result.entry(state)

            state_label = (
                str(landmark_labels[state])
                if landmark_labels and state in landmark_labels
                else entry["state_label"]
            )

            for endpoint_col, (pi_key, title_fmt) in enumerate(row_keys):
                outer_spec = gs[state_row, endpoint_col]

                inner = outer_spec.subgridspec(
                    1,
                    2,
                    width_ratios=[1.0, table_ratio],
                    wspace=0.05,
                )
                ax = fig.add_subplot(inner[0, 0])
                summary_ax = fig.add_subplot(inner[0, 1])


                pi_axes.append(ax)

                _plot_pi_panel(
                    ax,
                    entry.get(pi_key),
                    show_folds=show_folds,
                    show_det_bound=show_det_bound,
                    plot_band=plot_band,
                    summary_ax=summary_ax,
                    summary_show_n=False,
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
                    f"{state_label}\nPICP(α)"
                    if endpoint_col == 0
                    else ""
                )

                ax.set_xlabel(
                    "Nominal level α"
                    if state_row == n_state_rows - 1
                    else ""
                )

                panel_idx += 1

        fig.suptitle(
            "PI coverage",
            fontsize=_SUPTITLE_SIZE,
            fontweight=_SUPTITLE_WEIGHT,
            y=0.975,
        )

        for ax in pi_axes:
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


def plot_landmark_pi_coverage(
    pi_sojourn=None,
    pi_os=None,
    state_name=None,
    *,
    save_path = None,
    show: bool = True,
    max_cols: int = 4,
    panel_size: float = 3.0,
    dpi: int = 300,
    target = None,
    close: bool = False,
    show_det_bound: bool = False,
) -> plt.Figure:
    """Plot sojourn-time and overall-survival PI-coverage panels."""
    soj_title = (
        f"Sojourn time in {state_name}"
        if state_name
        else "Sojourn time"
    )
    os_title = (
        f"Survival from {state_name}"
        if state_name
        else "Overall survival"
    )

    panels = [
        (soj_title, pi_sojourn, _OKABE_ITO[0]),
        (os_title, pi_os, _OKABE_ITO[0]),
    ]

    def _draw(ax: plt.Axes, spec: tuple) -> None:
        _title, pi_cov, color = spec

        _plot_pi_panel(
            ax,
            pi_cov,
            show_det_bound=show_det_bound,
            primary=color,
        )

    with plt.rc_context(_pub_rc()):
        if target is not None:
            if len(target) != 2:
                raise ValueError(
                    "target must contain exactly two Axes/SubFigures."
                )

            fig = None

            for obj, spec in zip(target, panels):
                title = spec[0]

                if isinstance(obj, plt.Axes):
                    ax = obj
                    current_fig = obj.figure
                    ax.set_title(title, fontweight="bold")
                elif hasattr(obj, "subplots"):
                    ax = obj.subplots(1, 1)
                    current_fig = getattr(obj, "figure", None) or ax.figure
                    setter = (
                        obj.suptitle
                        if hasattr(obj, "suptitle")
                        else ax.set_title
                    )
                    setter(title, fontsize=9, fontweight="bold")
                else:
                    raise TypeError(
                        "target entries must be matplotlib Axes or "
                        "SubFigure-like."
                    )

                if fig is None:
                    fig = current_fig

                _draw(ax, spec)
                ax.set_xlabel("Nominal level α")
                ax.set_ylabel("PICP(α)")

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

        for i, spec in enumerate(panels):
            ax = axes[i // ncols, i % ncols]
            _draw(ax, spec)
            ax.set_title(spec[0], fontweight="bold")

        for row_idx in range(nrows):
            axes[row_idx, 0].set_ylabel("PICP(α)")

        for col_idx in range(ncols):
            axes[nrows - 1, col_idx].set_xlabel("Nominal level α")

        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        fig.suptitle(
            f"PI coverage{' – ' + str(state_name) if state_name else ''}",
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