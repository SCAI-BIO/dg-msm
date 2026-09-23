import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from typing import Optional

def _rescale_times(
    times: np.ndarray,
    time_delta: float,
    time_label: str,
) -> tuple[np.ndarray, str]:
    """
    Rescale a time grid to plot scale and return (plot_times, xlabel).
    """
    times = times.astype(float)
    plot_times = times * time_delta

    if time_label == "days":
        xlabel = "Time (days)"
    elif time_label == "years":
        plot_times = plot_times / 365.25
        xlabel = "Time (years)"
    else:
        xlabel = "Time"

    return plot_times, xlabel


def _format_td_axis(
    ax: plt.Axes,
    xlabel: str,
    y_label: str,
    y_lim: Optional[tuple[float, float] ],
    time_label: str,
) -> None:
    """Apply consistent formatting to time-dependent metric axes."""
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlim(left=0.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", which="major", labelsize=8, length=3)

    if time_label == "years":
        ax.xaxis.set_major_locator(MultipleLocator(1.0))  # 1-year spacing

    ax.grid(True, which="major", linestyle=":", linewidth=0.4, alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# CV curves with mean + CI (multiple folds)
def plot_td_metric(
    curves,
    state_label,
    ax=None,
    color="C0",
    alpha_band=0.25,
    lw_mean=2.0,
    show_folds=False,
    folds_alpha=0.2,
    folds_lw=0.8,
    ci_mode="ci",        # "sd" or "ci"
    show_legend=True,
    label_mean: Optional[str ] = None,  # auto-set from y_label
    label_band: Optional[str ] = None,  # auto-set from ci_mode
    time_delta: float = 1.0,        # length of one model time step in DAYS
    time_label: str = "grid",       # "grid", "days", or "years"
    y_label: str = "AUC(t)",
    y_lim: Optional[tuple[float, float] ] = None,
    clip_range: Optional[tuple[float, float] ] = None,  # optional value clipping
    grid_times_list: Optional[list[np.ndarray] ] = None,
):
    """
    Plot time-dependent metric for one state across multiple folds.
    """
    # collect per-fold curves, possibly converted to real time
    times_list = []
    vals_list = []

    if grid_times_list is not None and len(grid_times_list) != len(curves):
        raise ValueError("grid_times_list must have same length as curves.")

    for fold_idx, fold_dict in enumerate(curves):
        if state_label not in fold_dict:
            continue

        curve_obj = fold_dict[state_label]
        idx = curve_obj.index.values.astype(float)   # model time indices
        v = curve_obj.values.astype(float)

        if grid_times_list is not None:
            grid = grid_times_list[fold_idx]
            if grid is None:
                # no grid for this fold -> treat idx as model steps
                t_real = idx
            else:
                # map indices to real time (days) for THIS fold
                g_idx = np.arange(len(grid), dtype=float)
                t_real = np.interp(idx, g_idx, grid.astype(float))
        else:
            # no per-fold grid provided; work in index space
            t_real = idx

        times_list.append(t_real)
        vals_list.append(v)

    if not times_list:
        raise ValueError(f"No curves found for state_label={state_label!r}")

    # common time grid (in whatever units t_real uses: days if grid_times_list is not None,
    # otherwise in "model steps")
    t_min_common = max(t.min() for t in times_list)
    t_max_common = min(t.max() for t in times_list)
    if t_max_common <= t_min_common:
        raise ValueError(
            "No overlapping time range across folds for state "
            f"{state_label!r}"
        )

    n_points = min(t.size for t in times_list)
    common_times = np.linspace(t_min_common, t_max_common, n_points)

    # interpolate each fold onto the common grid (in real time or index space)
    val_matrix = np.vstack([
        np.interp(common_times, t, v)
        for t, v in zip(times_list, vals_list)
    ])

    if clip_range is not None:
        val_matrix = np.clip(val_matrix, *clip_range)

    mean_vals = val_matrix.mean(axis=0)
    std_vals = val_matrix.std(axis=0)
    n_folds = val_matrix.shape[0]

    if ci_mode == "sd":
        lower = mean_vals - std_vals
        upper = mean_vals + std_vals
        default_label_band = "±1 SD"
    elif ci_mode == "ci":
        se = std_vals / np.sqrt(max(n_folds, 1))
        delta = 1.96 * se
        lower = mean_vals - delta
        upper = mean_vals + delta
        default_label_band = "95% CI"
    else:
        raise ValueError("ci_mode must be 'sd' or 'ci'.")

    if clip_range is not None:
        lower = np.clip(lower, *clip_range)
        upper = np.clip(upper, *clip_range)

    if label_band is None:
        label_band = default_label_band
    if label_mean is None:
        label_mean = f"Mean {y_label}"

    # Rescale times for plotting
    if grid_times_list is not None:
        plot_times, xlabel = _rescale_times(
            common_times,
            time_delta=1.0,         # already in days
            time_label=time_label,
        )
    else:
        plot_times, xlabel = _rescale_times(
            common_times,
            time_delta=time_delta,  # index -> days
            time_label=time_label,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.0, 2.3), dpi=300)

    # optional: show individual folds
    if show_folds:
        for row in val_matrix:
            ax.plot(
                plot_times,
                row,
                color=color,
                alpha=folds_alpha,
                linewidth=folds_lw,
            )

    # mean curve
    ax.plot(
        plot_times,
        mean_vals,
        color=color,
        linewidth=lw_mean,
        label=label_mean,
    )

    # uncertainty band
    ax.fill_between(
        plot_times,
        lower,
        upper,
        color=color,
        alpha=alpha_band,
        label=label_band,
        linewidth=0,
    )

    # axis formatting (shared helper)
    _format_td_axis(ax, xlabel, y_label, y_lim, time_label)

    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=2,
            frameon=False,
            fontsize=8,
            borderaxespad=0.5,
        )
    return ax


def plot_td_metric_per_state(
    curves,
    metric_name: str,
    state_labels=None,
    ncols=2,
    figsize_per_subplot=(3.0, 2.3),
    show_folds=False,
    suptitle: Optional[str ] = None,      # main title (bold)
    subtitle: Optional[str ] = None,      # if None, auto-generated from metric_name
    time_delta: float = 1.0,
    time_label: str = "grid",         # "grid", "days", or "years"
    sharex: bool = False,
    y_lim: Optional[tuple[float, float] ] = None,
    clip_range: Optional[tuple[float, float] ] = None,
    grid_times_list: Optional[list[np.ndarray] ] = None,
):
    """
    Plot time-dependent metric curves per state (CV: multiple folds).
    """
    if not curves:
        raise ValueError("curves is empty.")

    # infer state indices from first fold
    first_fold = curves[0]
    state_indices = sorted(first_fold.keys())
    if not state_indices:
        raise ValueError("No state indices found in curves.")

    # derive display labels for each state index
    if state_labels is None:
        display_labels = [str(k) for k in state_indices]
    else:
        display_labels = [state_labels[k] for k in state_indices]

    n_states = len(state_indices)
    nrows = math.ceil(n_states / ncols)
    fig_width = figsize_per_subplot[0] * ncols
    fig_height = figsize_per_subplot[1] * nrows

    # create figure; we manage layout manually to leave room for titles
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharey=True,
        sharex=sharex,
        constrained_layout=False,
        dpi=300,
    )
    axes = np.array(axes, dtype=object).reshape(-1)

    first_handles, first_labels = None, None
    global_xmax = np.inf

    for i, (state_idx, state_name) in enumerate(zip(state_indices, display_labels)):
        ax = axes[i]

        ax = plot_td_metric(
            curves=curves,
            state_label=state_idx,
            ax=ax,
            show_folds=show_folds,
            show_legend=False,
            time_delta=time_delta,
            time_label=time_label,
            y_label=metric_name,
            y_lim=y_lim,
            clip_range=clip_range,
            grid_times_list=grid_times_list,
        )
        ax.set_title(str(state_name), fontsize=11)

        _, xmax = ax.get_xlim()
        global_xmax = max(global_xmax, xmax)

        if i % ncols != 0:
            ax.set_ylabel("")

        if i == 0:
            first_handles, first_labels = ax.get_legend_handles_labels()

    if np.isfinite(global_xmax) and sharex:
        # fix lower x limit at 0
        for i in range(n_states):
            axes[i].set_xlim(left=0.0, right=global_xmax)

        # if x-axis is in years, set ticks at each whole year
        if time_label == "years":
            max_year = max(1, math.floor(global_xmax))
            year_ticks = np.arange(0, max_year + 1, 1.0)
            axes[0].set_xticks(year_ticks)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # leave space at the top for titles
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])

    # auto-generate subtitle if not provided
    if subtitle is None:
        subtitle = f"{metric_name} for {len(curves)}-fold stratified CV"

    # main title (bold)
    if suptitle is not None:
        fig.text(
            0.5,
            1.0,
            suptitle,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
    # subtitle (smaller, in brackets)
    fig.text(
        0.5,
        0.92,
        subtitle,
        ha="center",
        va="top",
        fontsize=10,
    )

    if first_handles:
        fig.legend(
            first_handles,
            first_labels,
            loc="lower center",
            frameon=False,
            fontsize=9,
            ncol=2,
            bbox_to_anchor=(0.5, -0.10),
        )

    return fig, axes

# Single-run versions (no folds, no CI)
def plot_td_single_run_helper(
    curve: pd.Series,
    ax=None,
    color: str = "C0",
    lw: float = 2.0,
    time_delta: float = 1.0,        # length of one model time step in DAYS
    time_label: str = "grid",       # "grid", "days", or "years"
    y_label: str = "AUC(t)",
    y_lim: Optional[tuple[float, float]] = None,
    clip_range: Optional[tuple[float, float] ] = None,
    grid_times: Optional[np.ndarray ] = None,
):
    """
    Plot a single time-dependent metric curve (no folds, no CI).
    """
    idx = curve.index.values.astype(float)
    v = curve.values.astype(float)

    if clip_range is not None:
        v = np.clip(v, *clip_range)

    if grid_times is not None:
        g_idx = np.arange(len(grid_times), dtype=float)
        t_real = np.interp(idx, g_idx, grid_times.astype(float))  # days
        plot_times, xlabel = _rescale_times(
            t_real,
            time_delta=1.0,          # already in days
            time_label=time_label,
        )
    else:
        plot_times, xlabel = _rescale_times(
            idx,
            time_delta=time_delta,   # index -> days
            time_label=time_label,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.0, 2.3), dpi=300)

    ax.plot(plot_times, v, color=color, linewidth=lw)

    _format_td_axis(ax, xlabel, y_label, y_lim, time_label)

    return ax


def plot_td_metric_single_run(
    curves: dict[int, pd.Series],
    metric_name: str,
    state_labels=None,
    ncols=2,
    figsize_per_subplot=(3.0, 2.3),
    suptitle: Optional[str] = None,      # main title (bold)
    subtitle: Optional[str] = None,      # default: "<metric_name> (single run)"
    time_delta: float = 1.0,
    time_label: str = "grid",         # "grid", "days", or "years"
    sharex: bool = False,
    y_lim: Optional[tuple[float, float]] = None,
    clip_range: Optional[tuple[float, float]] = None,
    grid_times: Optional[np.ndarray] = None,
):
    """
    Plot time-dependent metric curves per state for a single test run.
    """
    if not curves:
        raise ValueError("curves is empty.")

    state_indices = sorted(curves.keys())
    if not state_indices:
        raise ValueError("No state indices found in curves.")

    if state_labels is None:
        display_labels = [str(k) for k in state_indices]
    else:
        display_labels = [state_labels[k] for k in state_indices]

    n_states = len(state_indices)
    nrows = math.ceil(n_states / ncols)
    fig_width = figsize_per_subplot[0] * ncols
    fig_height = figsize_per_subplot[1] * nrows

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharey=True,
        sharex=sharex,
        constrained_layout=False,
        dpi=300,
    )
    axes = np.array(axes, dtype=object).reshape(-1)

    global_xmax = np.inf

    for i, (state_idx, state_name) in enumerate(zip(state_indices, display_labels)):
        ax = axes[i]
        curve = curves[state_idx]

        ax = plot_td_single_run_helper(
            curve=curve,
            ax=ax,
            time_delta=time_delta,
            time_label=time_label,
            y_label=metric_name,
            y_lim=y_lim,
            clip_range=clip_range,
            grid_times=grid_times,
        )
        ax.set_title(str(state_name), fontsize=11)

        _, xmax = ax.get_xlim()
        global_xmax = max(global_xmax, xmax)

        if i % ncols != 0:
            ax.set_ylabel("")

    if np.isfinite(global_xmax) and sharex:
        for i in range(n_states):
            axes[i].set_xlim(left=0.0, right=global_xmax)

        if time_label == "years":
            max_year = max(1, math.floor(global_xmax))
            year_ticks = np.arange(0, max_year + 1, 1.0)
            axes[0].set_xticks(year_ticks)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])

    if subtitle is None:
        subtitle = f"{metric_name} (single run)"

    if suptitle is not None:
        fig.text(
            0.5,
            1.0,
            suptitle,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.92,
        subtitle,
        ha="center",
        va="top",
        fontsize=10,
    )

    return fig, axes