# _metrics.py
from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D

from ._base import (
    _OKABE_ITO,
    _SUPTITLE_SIZE,
    _SUPTITLE_WEIGHT,
    _add_panel_label,
    _pub_rc,
    _save_figure,
)
from ._dcal import _plot_landmark_reliability_panel
from ._pi import _plot_pi_panel


_METRICS_GROUP_LABEL: dict[str, str] = {
    "os": "Overall Survival",
    "sojourn": "Sojourn Time",
}

_PANEL_W: float = 3.5
_PANEL_H: float = 2.8

_CORE_LABELS: dict[str, str] = {
    "adj Antolini": "C-Index",
    "adj Antolini IPCW": "C-Index (IPCW)",
    "IBS": "IBS",
    "AUC": "Mean AUC(t)",
    "INBLL": "INBLL",
    "IBS (trunc)": "IBS (trunc.)",
    "AUC (trunc)": "AUC (trunc.)",
    "INBLL (trunc)": "INBLL (trunc.)",
    "adj Antolini IPCW (trunc)": "adj. C-IPCW (trunc.)",
}

_DEFAULT_METRICS: tuple[str, ...] = (
    # "adj Antolini",
    "adj Antolini IPCW",
    "IBS",
    "AUC",
)


def _support_label_map(
    support_df: pd.DataFrame | None,
    *,
    mode: str = "total",
) -> dict[int, str]:
    """
    Map landmark state -> a short ``n=...`` annotation string from a
    support_summary()-shaped DataFrame.
    """
    if support_df is None or support_df.empty:
        return {}

    out: dict[int, str] = {}

    for _, row in support_df.iterrows():
        state = int(row["state"])
        n_folds = int(row.get("n_folds", 1) or 1)
        total = row.get("total")

        if (
            mode == "total_range"
            and n_folds > 1
            and {"min", "max"}.issubset(support_df.columns)
            and total is not None
        ):
            out[state] = (
                f"n={int(total):,}\n"
                f"({int(row['min'])}–{int(row['max'])}/fold)"
            )
        elif (
            mode == "median_range"
            and n_folds > 1
            and {"min", "max", "median"}.issubset(support_df.columns)
        ):
            out[state] = (
                f"n≈{int(round(row['median'])):,} "
                f"({int(row['min'])}–{int(row['max'])})"
            )
        elif (
            mode == "mean_std"
            and n_folds > 1
            and {"mean", "std"}.issubset(support_df.columns)
        ):
            out[state] = f"n={row['mean']:.0f}±{row['std']:.0f}"
        elif total is not None:
            out[state] = f"n={int(total):,}"
        else:
            out[state] = f"n={int(row.get('median', 0)):,}"

    return out


def _decorate_metric_ax(
    ax: plt.Axes,
    *,
    tick_positions: np.ndarray,
    tick_labels: list[str],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    tick_rotation: int = 0,
) -> None:
    """Apply shared styling to a metric panel."""
    ax.set_ylabel(ylabel)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        rotation=tick_rotation,
        ha="right" if tick_rotation > 0 else "center",
    )
    ax.margins(x=0.05)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.6)
    ax.tick_params(direction="out", length=3, width=0.8)

    if ylim is not None:
        ax.set_ylim(*ylim)


def _effective_metrics(
    requested: list[str] | None,
    available: list[str],
) -> list[str]:
    """Resolve the metrics to plot."""
    if requested is not None:
        selected = [metric for metric in requested if metric in available]

        if not selected:
            raise ValueError(
                f"None of the requested metrics are available.\n"
                f"Available: {available}"
            )

        return selected

    selected = [metric for metric in _DEFAULT_METRICS if metric in available]
    return selected if selected else list(available)


def _metric_display_name(
    raw: str,
    name_map: dict[str, str] | None = None,
) -> str:
    """Return a publication-friendly metric label."""
    if name_map and raw in name_map:
        return name_map[raw]

    return _CORE_LABELS.get(raw, raw)


def _method_legend_handles(
    methods: Sequence[str],
    prepared: dict[str, Any],
    *,
    linewidth: float = 1.4,
    markersize: float = 4.5,
) -> list[Line2D]:
    """Create legend handles for metric methods."""
    handles: list[Line2D] = []

    for method in methods:
        mode = prepared["resolved_mode"][method]

        if mode == "line":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=prepared["color_map"][method],
                    linestyle=prepared["linestyle_map"][method],
                    marker=prepared["marker_map"][method],
                    linewidth=linewidth,
                    markersize=markersize,
                    label=method,
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=prepared["color_map"][method],
                    linestyle="None",
                    marker=prepared["marker_map"][method],
                    markersize=max(markersize, 5.0),
                    label=method,
                )
            )

    return handles


# ======================================================================
# Landmark metrics plot (single-run, wide-format input)
# ======================================================================

def plot_landmark_metrics(
    metric_frames: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
    metric_name_map: dict[str, str] | None = None,
    landmark_labels: dict[int, str] | None = None,
    figsize: tuple[float, float] | None = None,
    ylim_dict: dict[str, tuple[float, float]] | None = None,
    sharey: bool = False,
    show_plot: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
    suptitle: str | None = None,
    line_color: str = _OKABE_ITO[0],
    line_label: str = "DG-MSM",
    show_panel_labels: bool = True,
    tick_rotation: int = 0,
    support: pd.DataFrame | None = None,
    support_mode: str = "total",
    close: bool = False,
) -> plt.Figure:
    """Plot per-landmark metric values from wide-format input."""
    if ylim_dict is None:
        ylim_dict = {}

    non_empty = {
        key: value
        for key, value in metric_frames.items()
        if value is not None and not value.empty
    }

    if not non_empty:
        raise ValueError("All metric_frames are empty.")

    first_df = next(iter(non_empty.values()))

    value_cols = sorted(
        (column for column in first_df.columns if column != "metric"),
        key=lambda column: (
            int(column)
            if str(column).lstrip("-").isdigit()
            else str(column)
        ),
    )

    if not value_cols:
        raise ValueError("No value columns found in metric_frames.")

    x_vals = np.array(value_cols, dtype=int)
    tick_positions = x_vals.astype(float)
    support_labels = _support_label_map(support, mode=support_mode)

    tick_labels = [
        (
            landmark_labels.get(state, str(state))
            if landmark_labels
            else str(state)
        )
        + (
            f"\n{support_labels[state]}"
            if state in support_labels
            else ""
        )
        for state in x_vals
    ]

    row_groups: list[tuple[str, list[str], pd.DataFrame]] = []

    for group_label, frame_df in non_empty.items():
        available = list(frame_df["metric"].unique())
        selected = _effective_metrics(metrics, available)

        if selected:
            row_groups.append((group_label, selected, frame_df))

    if not row_groups:
        raise ValueError(
            "No metrics to plot after applying 'metrics' filter."
        )

    use_row_titles = len(row_groups) > 1
    n_rows = len(row_groups)
    ncols_eff = max(len(selected) for _, selected, _ in row_groups)

    if figsize is None:
        figsize = (
            _PANEL_W * ncols_eff,
            _PANEL_H * n_rows
            + (0.35 if suptitle else 0.0)
            + (0.20 * n_rows if use_row_titles else 0.0),
        )

    with plt.rc_context(_pub_rc()):
        fig = plt.figure(figsize=figsize, dpi=dpi)

        if suptitle:
            fig.suptitle(
                suptitle,
                fontsize=_SUPTITLE_SIZE,
                fontweight=_SUPTITLE_WEIGHT,
            )

        gs = fig.add_gridspec(
            n_rows,
            ncols_eff,
            hspace=0.55 if use_row_titles else 0.40,
            wspace=0.32,
        )

        row_ref_ax: list[plt.Axes | None] = [None] * n_rows
        panel_idx = 0

        for row_idx, (row_title, row_metrics, frame_df) in enumerate(
            row_groups
        ):
            for col_idx in range(ncols_eff):
                reference_axis = (
                    row_ref_ax[row_idx]
                    if sharey and row_ref_ax[row_idx] is not None
                    else None
                )

                ax = fig.add_subplot(
                    gs[row_idx, col_idx],
                    sharey=reference_axis,
                )

                if row_ref_ax[row_idx] is None:
                    row_ref_ax[row_idx] = ax

                if col_idx >= len(row_metrics):
                    ax.set_visible(False)
                    continue

                metric_name = row_metrics[col_idx]

                y = (
                    frame_df[frame_df["metric"] == metric_name][value_cols]
                    .mean(axis=0)
                    .to_numpy(dtype=float)
                )

                mask = np.isfinite(y)

                if mask.any():
                    ax.plot(
                        x_vals[mask],
                        y[mask],
                        marker="o",
                        ms=3.5,
                        lw=1.4,
                        color=line_color,
                        markerfacecolor="white",
                        markeredgewidth=0.9,
                        label=line_label,
                        zorder=3,
                    )

                if use_row_titles and col_idx == 0 and row_title:
                    ax.annotate(
                        row_title,
                        xy=(0.0, 1.0),
                        xycoords="axes fraction",
                        xytext=(0, 22),
                        textcoords="offset points",
                        fontsize=9,
                        fontweight=_SUPTITLE_WEIGHT,
                        va="bottom",
                        ha="left",
                        annotation_clip=False,
                    )

                if show_panel_labels:
                    _add_panel_label(ax, panel_idx)

                _decorate_metric_ax(
                    ax,
                    tick_positions=tick_positions,
                    tick_labels=tick_labels,
                    ylabel=_metric_display_name(
                        metric_name,
                        metric_name_map,
                    ),
                    ylim=ylim_dict.get(metric_name),
                    tick_rotation=tick_rotation,
                )

                panel_idx += 1

        fig.tight_layout()

        if save_path is not None:
            _save_figure(fig, save_path, dpi=dpi)

        if show_plot:
            plt.show()

        if close:
            plt.close(fig)

    return fig


# ======================================================================
# CV landmark metrics plot (fold-level, long-format input)
# ======================================================================

def plot_cv_landmark_metrics(
    landmark_cv_df: pd.DataFrame,
    landmark_labels: Mapping[object, str] | Sequence[str] | None = None,
    metric_name_map: dict[str, str] | None = None,
    metrics_order: Sequence[str] | None = None,
    methods_order: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    ylim_dict: dict[str, tuple[float, float]] | None = None,
    ncols: int = 3,
    sharey: bool = False,
    show_plot: bool = True,
    dpi: int = 300,
    suptitle: str | None = None,
    save_path: str | os.PathLike[str] | None = None,
    color_map: dict[str, str] | None = None,
    linestyle_map: dict[str, str] | None = None,
    marker_map: dict[str, str] | None = None,
    show_legend: bool = True,
    show_panel_labels: bool = True,
    error: str = "std",
    error_style: str = "band",
    alpha_fill: float = 0.10,
    linewidth: float = 1.4,
    markersize: float = 4.5,
    method_plot_mode: dict[str, str] | None = None,
    box_width: float = 0.12,
    box_alpha: float = 0.18,
    benchmark_offset_span: float = 0.16,
    show_benchmark_means: bool = True,
    show_benchmark_fold_points: bool = False,
    benchmark_point_jitter: float = 0.022,
    support: pd.DataFrame | None = None,
    support_mode: str = "total",
    close: bool = False,
) -> plt.Figure:
    """Plot CV-aggregated landmark performance metrics."""
    prepared = _prepare_cv_landmark_metrics(
        landmark_cv_df,
        landmark_labels=landmark_labels,
        metric_name_map=metric_name_map,
        metrics_order=metrics_order,
        methods_order=methods_order,
        error=error,
        error_style=error_style,
        method_plot_mode=method_plot_mode,
        color_map=color_map,
        linestyle_map=linestyle_map,
        marker_map=marker_map,
        support_labels=_support_label_map(
            support,
            mode=support_mode,
        ),
    )

    n_metrics = len(prepared["selected_metrics"])
    ncols = max(1, min(ncols, n_metrics))
    nrows = math.ceil(n_metrics / ncols)

    if figsize is None:
        figsize = (_PANEL_W * ncols, _PANEL_H * nrows)

    with plt.rc_context(_pub_rc()):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
            sharey=sharey,
            squeeze=False,
        )

        axes_flat = axes.ravel()

        plotted_methods = _draw_cv_landmark_metrics_panels(
            axes_flat,
            prepared,
            metric_name_map=metric_name_map,
            ylim_dict=ylim_dict,
            alpha_fill=alpha_fill,
            linewidth=linewidth,
            markersize=markersize,
            box_width=box_width,
            box_alpha=box_alpha,
            benchmark_offset_span=benchmark_offset_span,
            show_benchmark_means=show_benchmark_means,
            show_benchmark_fold_points=show_benchmark_fold_points,
            benchmark_point_jitter=benchmark_point_jitter,
            show_panel_labels=show_panel_labels,
        )

        if show_legend and plotted_methods:
            handles = _method_legend_handles(
                plotted_methods,
                prepared,
                linewidth=linewidth,
                markersize=markersize,
            )

            if handles:
                fig.legend(
                    handles=handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.01),
                    ncol=min(len(handles), 4),
                    frameon=False,
                    handlelength=1.8,
                    columnspacing=1.0,
                    handletextpad=0.5,
                )

        if suptitle is not None:
            fig.suptitle(
                suptitle,
                fontsize=_SUPTITLE_SIZE,
                fontweight=_SUPTITLE_WEIGHT,
                y=0.98,
            )

        fig.tight_layout(
            rect=[0.02, 0.06, 1.0, 1.0],
            pad=1.5,
        )

        if save_path is not None:
            _save_figure(fig, save_path, dpi=dpi)

        if show_plot:
            plt.show()

        if close:
            plt.close(fig)

    return fig


def _prepare_cv_landmark_metrics(
    landmark_cv_df: pd.DataFrame,
    *,
    landmark_labels=None,
    metric_name_map=None,
    metrics_order=None,
    methods_order=None,
    error="std",
    error_style="band",
    method_plot_mode=None,
    color_map=None,
    linestyle_map=None,
    marker_map=None,
    support_labels: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Validate and prepare CV metric data for plotting."""
    if landmark_cv_df is None or landmark_cv_df.empty:
        raise ValueError("landmark_cv_df is empty.")

    error = str(error).lower()
    error_style = str(error_style).lower()

    if error not in {"none", "std", "sem", "ci95"}:
        raise ValueError(
            "error must be one of {none, std, sem, ci95}."
        )

    if error_style not in {"band", "bar"}:
        raise ValueError(
            "error_style must be one of {band, bar}."
        )

    if method_plot_mode is None:
        method_plot_mode = {
            "DG-MSM": "line",
            "CoxPH": "point",
            "PyCox": "point",
        }

    def _ordered_unique(values) -> list:
        return list(
            dict.fromkeys(
                value
                for value in values
                if not pd.isna(value)
            )
        )

    def _select_in_order(available, requested) -> list:
        if requested is None:
            return list(available)

        available_set = set(available)
        return [value for value in requested if value in available_set]

    def _style_map(keys, user_map, defaults) -> dict:
        return {
            key: (
                user_map[key]
                if user_map and key in user_map
                else defaults[index % len(defaults)]
            )
            for index, key in enumerate(keys)
        }

    def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()

        for column in [
            "landmark_idx",
            "value",
            "mean",
            "std",
            "sem",
            "ci95",
            "ci_lower",
            "ci_upper",
            "count",
        ]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        return frame

    def _build_summary(raw: pd.DataFrame) -> pd.DataFrame:
        summary = (
            raw.groupby(
                ["method", "metric", "landmark_idx"],
                dropna=False,
            )["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        summary["sem"] = np.where(
            summary["count"] > 0,
            summary["std"] / np.sqrt(summary["count"]),
            np.nan,
        )

        summary["ci95"] = 1.96 * summary["sem"]
        summary["ci_lower"] = summary["mean"] - summary["ci95"]
        summary["ci_upper"] = summary["mean"] + summary["ci95"]

        return summary

    def _coerce_summary(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()

        if "std" not in frame.columns:
            frame["std"] = np.nan

        if "sem" not in frame.columns:
            frame["sem"] = (
                frame["std"] / np.sqrt(frame["count"])
                if "count" in frame.columns
                else np.nan
            )

        if "ci95" not in frame.columns:
            if {"ci_lower", "ci_upper"}.issubset(frame.columns):
                frame["ci95"] = np.maximum(
                    (frame["mean"] - frame["ci_lower"]).abs(),
                    (frame["ci_upper"] - frame["mean"]).abs(),
                )
            else:
                frame["ci95"] = 1.96 * frame["sem"]

        if "ci_lower" not in frame.columns:
            frame["ci_lower"] = frame["mean"] - frame["ci95"]

        if "ci_upper" not in frame.columns:
            frame["ci_upper"] = frame["mean"] + frame["ci95"]

        return frame

    df_input = _coerce(landmark_cv_df)

    missing = {"method", "metric", "landmark_idx"} - set(
        df_input.columns
    )

    if missing:
        raise ValueError(
            f"landmark_cv_df missing columns: {sorted(missing)}"
        )

    has_raw = "value" in df_input.columns

    if has_raw:
        raw_df = df_input.dropna(
            subset=["method", "metric", "landmark_idx", "value"]
        ).copy()

        summary_df = _build_summary(raw_df)
    else:
        raw_df = pd.DataFrame(columns=df_input.columns)

        missing_summary = {
            "method",
            "metric",
            "landmark_idx",
            "mean",
        } - set(df_input.columns)

        if missing_summary:
            raise ValueError(
                f"Pre-aggregated input missing: "
                f"{sorted(missing_summary)}"
            )

        summary_df = _coerce_summary(
            df_input.dropna(
                subset=["method", "metric", "landmark_idx", "mean"]
            ).copy()
        )

    if summary_df.empty:
        raise ValueError("No valid rows remain after filtering.")

    metrics_available = _ordered_unique(
        summary_df["metric"].tolist()
    )

    methods_available = _ordered_unique(
        summary_df["method"].tolist()
    )

    if metric_name_map is not None:
        allowed = [
            metric
            for metric in metric_name_map
            if metric in set(metrics_available)
        ]

        if not allowed:
            raise ValueError(
                "None of the metrics in metric_name_map are present "
                "in the data."
            )

        selected_metrics = _select_in_order(
            allowed,
            metrics_order,
        )
    else:
        selected_metrics = _effective_metrics(
            metrics_order,
            metrics_available,
        )

    selected_methods = _select_in_order(
        methods_available,
        methods_order,
    )

    if not selected_metrics:
        raise ValueError(
            "No metrics available after applying metrics_order."
        )

    if not selected_methods:
        raise ValueError(
            "No methods available after applying methods_order."
        )

    summary_df = summary_df[
        summary_df["metric"].isin(selected_metrics)
        & summary_df["method"].isin(selected_methods)
    ].copy()

    if has_raw:
        raw_df = raw_df[
            raw_df["metric"].isin(selected_metrics)
            & raw_df["method"].isin(selected_methods)
        ].copy()

    landmark_idx_values = sorted(
        _ordered_unique(summary_df["landmark_idx"].tolist())
    )

    if not landmark_idx_values:
        raise ValueError("No landmark indices available.")

    x_position_map = {
        landmark: index
        for index, landmark in enumerate(landmark_idx_values)
    }

    tick_positions = np.arange(
        len(landmark_idx_values),
        dtype=float,
    )

    if landmark_labels is None:
        landmark_label_map = {
            landmark: str(landmark)
            for landmark in landmark_idx_values
        }
    elif isinstance(landmark_labels, Mapping):
        landmark_label_map = {
            key: str(value)
            for key, value in landmark_labels.items()
        }
    elif isinstance(landmark_labels, (list, tuple)):
        if len(landmark_labels) != len(landmark_idx_values):
            raise ValueError(
                "landmark_labels sequence length must match number "
                "of landmarks."
            )

        landmark_label_map = {
            landmark: str(label)
            for landmark, label in zip(
                landmark_idx_values,
                landmark_labels,
            )
        }
    else:
        raise TypeError(
            "landmark_labels must be a mapping or sequence of strings."
        )

    tick_labels = [
        landmark_label_map.get(landmark, str(landmark))
        + (
            f"\n{support_labels[landmark]}"
            if support_labels and landmark in support_labels
            else ""
        )
        for landmark in landmark_idx_values
    ]

    default_markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
    default_linestyles = ["-", "--", "-.", ":"]

    preferred_colors = {
        "DG-MSM": _OKABE_ITO[0],
        "CoxPH": _OKABE_ITO[1],
        "PyCox": _OKABE_ITO[1],
    }

    preferred_markers = {
        "DG-MSM": "o",
        "CoxPH": "s",
        "PyCox": "s",
    }

    preferred_linestyles = {
        "DG-MSM": "-",
        "CoxPH": "None",
        "PyCox": "None",
    }

    color_map_resolved = _style_map(
        selected_methods,
        {**preferred_colors, **(color_map or {})},
        _OKABE_ITO,
    )

    marker_map_resolved = _style_map(
        selected_methods,
        {**preferred_markers, **(marker_map or {})},
        default_markers,
    )

    linestyle_map_resolved = _style_map(
        selected_methods,
        {**preferred_linestyles, **(linestyle_map or {})},
        default_linestyles,
    )

    resolved_mode: dict[str, str] = {}

    for method in selected_methods:
        mode = str(method_plot_mode.get(method, "point")).lower()

        if mode not in {"line", "box", "point"}:
            raise ValueError(
                f"Invalid plot mode for '{method}': {mode!r}"
            )

        if mode == "box" and not has_raw:
            mode = "point"

        resolved_mode[method] = mode

    method_z = {
        method: (
            25.0
            if "dgmsm" in str(method).lower()
            else 30.0
            if resolved_mode[method] == "line"
            else 10.0
        )
        for method in selected_methods
    }

    return {
        "summary_df": summary_df,
        "raw_df": raw_df,
        "has_raw": has_raw,
        "selected_metrics": selected_metrics,
        "selected_methods": selected_methods,
        "landmark_idx_values": landmark_idx_values,
        "x_position_map": x_position_map,
        "tick_positions": tick_positions,
        "tick_labels": tick_labels,
        "color_map": color_map_resolved,
        "marker_map": marker_map_resolved,
        "linestyle_map": linestyle_map_resolved,
        "resolved_mode": resolved_mode,
        "method_z": method_z,
        "error": error,
        "error_style": error_style,
    }


def _draw_cv_landmark_metrics_panels(
    axes_flat: Sequence[plt.Axes],
    prepared: dict[str, Any],
    *,
    metric_name_map=None,
    ylim_dict=None,
    alpha_fill=0.10,
    linewidth=1.4,
    markersize=4.5,
    box_width=0.12,
    box_alpha=0.18,
    benchmark_offset_span=0.16,
    show_benchmark_means=True,
    show_benchmark_fold_points=False,
    benchmark_point_jitter=0.022,
    show_panel_labels=True,
    panel_label_start=0,
    rng_seed=42,
) -> list[str]:
    """Draw metric panels using preprocessed CV metric data."""
    selected_metrics = prepared["selected_metrics"]
    selected_methods = prepared["selected_methods"]
    summary_df = prepared["summary_df"]
    raw_df = prepared["raw_df"]
    has_raw = prepared["has_raw"]
    landmark_idx_values = prepared["landmark_idx_values"]
    x_position_map = prepared["x_position_map"]
    tick_positions = prepared["tick_positions"]
    tick_labels = prepared["tick_labels"]
    color_map = prepared["color_map"]
    marker_map = prepared["marker_map"]
    linestyle_map = prepared["linestyle_map"]
    resolved_mode = prepared["resolved_mode"]
    method_z = prepared["method_z"]
    error = prepared["error"]
    error_style = prepared["error_style"]

    def _get_err(
        sub: pd.DataFrame,
        kind: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if kind == "none":
            return None, None, None

        mean = sub["mean"].to_numpy(float)

        if kind == "std":
            err = sub["std"].to_numpy(float)
            return err, mean - err, mean + err

        if kind == "sem":
            err = sub["sem"].to_numpy(float)
            return err, mean - err, mean + err

        lower = sub["ci_lower"].to_numpy(float)
        upper = sub["ci_upper"].to_numpy(float)

        return (
            np.maximum(
                np.abs(mean - lower),
                np.abs(upper - mean),
            ),
            lower,
            upper,
        )

    def _reindex_to_all(
        sub: pd.DataFrame,
        all_landmarks: Sequence[Any],
    ) -> pd.DataFrame:
        sub = sub.groupby(
            "landmark_idx",
            as_index=False,
        ).first()

        return (
            sub.set_index("landmark_idx")
            .reindex(all_landmarks)
            .rename_axis("landmark_idx")
            .reset_index()
        )

    rng = np.random.default_rng(rng_seed)
    plotted_methods: set[str] = set()

    for idx, metric_name in enumerate(selected_metrics):
        ax = axes_flat[idx]
        ax.set_axisbelow(True)

        metric_summary = summary_df[
            summary_df["metric"] == metric_name
        ].copy()

        metric_raw = (
            raw_df[raw_df["metric"] == metric_name].copy()
            if has_raw
            else pd.DataFrame()
        )

        benchmark_methods = [
            method
            for method in selected_methods
            if resolved_mode[method] in {"box", "point"}
            and not metric_summary[
                metric_summary["method"] == method
            ].empty
        ]

        offset_map = (
            dict(
                zip(
                    benchmark_methods,
                    np.linspace(
                        -benchmark_offset_span / 2,
                        benchmark_offset_span / 2,
                        len(benchmark_methods),
                    ),
                )
            )
            if len(benchmark_methods) > 1
            else {
                method: 0.0
                for method in benchmark_methods
            }
        )

        for method in selected_methods:
            mode = resolved_mode[method]
            base_z = method_z[method]

            sub = (
                metric_summary[
                    metric_summary["method"] == method
                ]
                .dropna(subset=["landmark_idx", "mean"])
                .sort_values("landmark_idx")
                .copy()
            )

            if sub.empty:
                continue

            if mode == "line":
                sub_line = _reindex_to_all(
                    sub,
                    landmark_idx_values,
                )

                x = tick_positions
                y = sub_line["mean"].to_numpy(float)
                _, lower, upper = _get_err(sub_line, error)

                if error != "none" and lower is not None:
                    valid = (
                        np.isfinite(y)
                        & np.isfinite(lower)
                        & np.isfinite(upper)
                    )

                    if valid.any():
                        if error_style == "band":
                            ax.fill_between(
                                x,
                                lower,
                                upper,
                                where=valid,
                                color=color_map[method],
                                alpha=alpha_fill,
                                linewidth=0,
                                zorder=base_z - 2,
                            )
                        else:
                            yerr = np.vstack(
                                [
                                    y[valid] - lower[valid],
                                    upper[valid] - y[valid],
                                ]
                            )

                            ax.errorbar(
                                x[valid],
                                y[valid],
                                yerr=yerr,
                                fmt="none",
                                ecolor=color_map[method],
                                elinewidth=0.9,
                                capsize=2,
                                alpha=0.95,
                                zorder=base_z - 1,
                            )

                valid_points = np.isfinite(y)

                if valid_points.any():
                    ax.plot(
                        x,
                        y,
                        color=color_map[method],
                        linestyle=linestyle_map[method],
                        linewidth=linewidth,
                        zorder=base_z,
                    )

                    marker = marker_map[method]

                    if marker not in {None, "None", "none", ""}:
                        ax.scatter(
                            x[valid_points],
                            y[valid_points],
                            marker=marker,
                            s=max(markersize, 1.0) ** 2,
                            color=color_map[method],
                            linewidth=0.6,
                            zorder=base_z + 1,
                        )

                    plotted_methods.add(method)

                continue

            sub = sub.groupby(
                "landmark_idx",
                as_index=False,
            ).first()

            x = sub["landmark_idx"].map(
                x_position_map
            ).to_numpy(float)

            y = sub["mean"].to_numpy(float)
            _, lower, upper = _get_err(sub, error)

            x_offset = x + offset_map.get(method, 0.0)
            drew = False

            if has_raw and show_benchmark_fold_points:
                sub_raw = (
                    metric_raw[
                        metric_raw["method"] == method
                    ]
                    .dropna(subset=["landmark_idx", "value"])
                    .copy()
                )

                for landmark in landmark_idx_values:
                    values = sub_raw.loc[
                        sub_raw["landmark_idx"] == landmark,
                        "value",
                    ].astype(float).to_numpy()

                    values = values[np.isfinite(values)]

                    if values.size == 0:
                        continue

                    position = (
                        x_position_map[landmark]
                        + offset_map.get(method, 0.0)
                    )

                    jitter = rng.uniform(
                        -benchmark_point_jitter,
                        benchmark_point_jitter,
                        values.size,
                    )

                    ax.scatter(
                        np.full(values.size, position) + jitter,
                        values,
                        s=12,
                        color=color_map[method],
                        alpha=0.35,
                        linewidth=0,
                        zorder=base_z - 2,
                    )

                    drew = True

            if mode == "box" and has_raw:
                sub_raw = (
                    metric_raw[
                        metric_raw["method"] == method
                    ]
                    .dropna(subset=["landmark_idx", "value"])
                    .copy()
                )

                box_data: list[np.ndarray] = []
                box_positions: list[float] = []

                for landmark in landmark_idx_values:
                    values = sub_raw.loc[
                        sub_raw["landmark_idx"] == landmark,
                        "value",
                    ].astype(float).to_numpy()

                    values = values[np.isfinite(values)]

                    if values.size > 0:
                        box_data.append(values)
                        box_positions.append(
                            x_position_map[landmark]
                            + offset_map.get(method, 0.0)
                        )

                if box_data:
                    boxplot = ax.boxplot(
                        box_data,
                        positions=box_positions,
                        widths=box_width,
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False,
                        zorder=base_z - 1,
                    )

                    for patch in boxplot["boxes"]:
                        patch.set(
                            facecolor=color_map[method],
                            edgecolor=color_map[method],
                            alpha=box_alpha,
                            linewidth=0.9,
                        )

                    for artist in (
                        boxplot["whiskers"]
                        + boxplot["caps"]
                        + boxplot["medians"]
                    ):
                        artist.set(
                            color=color_map[method],
                            linewidth=0.9,
                        )

                    drew = True

            if error != "none" and lower is not None:
                valid = (
                    np.isfinite(x_offset)
                    & np.isfinite(y)
                    & np.isfinite(lower)
                    & np.isfinite(upper)
                )

                if valid.any():
                    yerr = np.vstack(
                        [
                            y[valid] - lower[valid],
                            upper[valid] - y[valid],
                        ]
                    )

                    ax.errorbar(
                        x_offset[valid],
                        y[valid],
                        yerr=yerr,
                        fmt="none",
                        ecolor=color_map[method],
                        elinewidth=0.9,
                        capsize=2,
                        alpha=0.95,
                        zorder=base_z,
                    )

                    drew = True

            if show_benchmark_means:
                valid = np.isfinite(x_offset) & np.isfinite(y)

                if valid.any():
                    ax.scatter(
                        x_offset[valid],
                        y[valid],
                        marker=marker_map[method],
                        s=28,
                        color=color_map[method],
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=base_z + 1,
                    )

                    drew = True

            if drew:
                plotted_methods.add(method)

        if show_panel_labels:
            _add_panel_label(
                ax,
                panel_label_start + idx,
            )

        _decorate_metric_ax(
            ax,
            tick_positions=tick_positions,
            tick_labels=tick_labels,
            ylabel=_metric_display_name(
                metric_name,
                metric_name_map,
            ),
            ylim=(ylim_dict or {}).get(metric_name),
        )

    for index in range(len(selected_metrics), len(axes_flat)):
        axes_flat[index].set_visible(False)

    return [
        method
        for method in selected_methods
        if method in plotted_methods
    ]