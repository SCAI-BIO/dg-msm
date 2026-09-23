import math
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def metric_boxplots(
    df_metrics: pd.DataFrame,
    state_name: str,
    metric_names: Sequence[str],
    models_order: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 200,
    palette_models: Optional[List[str]] = None,
    show_fliers: bool = False,
    box_width: float = 0.55,
    box_linewidth: float = 1.8,
    whisker  = 1.5,
    show_mean: bool = False,
    mean_size: int = 90,
    grid_y: bool = True,
    y_limits: Optional[Tuple[float, float]] = None,
    y_pad_top_frac: float = 0.1,
    ref_line: Optional[float] = None,
    ref_line_style: str = "--",
    ref_line_color: str = "gray",
    ref_line_alpha: float = 0.7,
    title: Optional[str] = None,
    title_fontsize: int = 22,
    label_fontsize: int = 16,
    tick_fontsize: int = 13,
    legend_fontsize: int = 14,
    legend_ncol: int = 1,
    yticks: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    set_theme: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Boxplot of multiple metrics for a single state, grouped by model.

    x-axis: metric
    hue: model
    y-axis: value
    """

    df = df_metrics.copy()
    df = df[df["state"] == state_name].copy()
    if df.empty:
        raise ValueError(f"No rows for state {state_name!r} in df_metrics.")

    metric_names = list(metric_names)
    df = df[df["metric"].isin(metric_names)].copy()
    if df.empty:
        raise ValueError(
            f"No rows for requested metrics {metric_names} in state {state_name!r}."
        )

    # Model order
    if models_order is None:
        models_order = sorted(df["model"].unique().tolist())
    else:
        missing_models = [m for m in models_order if m not in df["model"].unique()]
        if missing_models:
            raise ValueError(
                f"Some models in 'models_order' are not present in the data: {missing_models}"
            )

    # Colors per model (colorblind-friendly, well-separated)
    if palette_models is None:
        default_colors = [
            "#0072B2",  # blue
            "#E69F00",  # orange
            "#009E73",  # green
            "#D55E00",  # red
            "#CC79A7",  # pink
            "#56B4E9",  # light blue
            "#F0E442",  # yellow
            "#999999",  # gray
        ]
        if len(models_order) > len(default_colors):
            pal = sns.color_palette("colorblind", n_colors=len(models_order))
        else:
            pal = default_colors[: len(models_order)]
    else:
        if len(palette_models) < len(models_order):
            raise ValueError("palette_models must have at least as many colors as models.")
        pal = palette_models
    pal_map = {m: pal[i] for i, m in enumerate(models_order)}

    rc_bold = {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }

    with plt.rc_context(rc_bold):
        if set_theme:
            sns.set_theme(style="whitegrid", context="talk")

        created_fig = False
        if ax is None:
            created_fig = True
            try:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")
            except TypeError:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                fig.set_constrained_layout(True)
        else:
            fig = ax.figure

        # --- Boxplots: x = metric, hue = model ---
        sns.boxplot(
            x="metric",
            y="value",
            hue="model",
            data=df,
            order=metric_names,
            hue_order=models_order,
            palette=pal_map,
            width=box_width,
            whis=whisker,
            fliersize=3 if show_fliers else 0,
            linewidth=box_linewidth,
            dodge=0.7,
            ax=ax,
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        # --- Means as diamonds ---
        if show_mean and not df.empty:
            means = df.groupby(["metric", "model"], as_index=False)["value"].mean()

            pointplot_kwargs = dict(
                x="metric",
                y="value",
                hue="model",
                data=means,
                order=metric_names,
                hue_order=models_order,
                dodge=0.4,
                markers="D",
                linestyle="none",
                markersize=np.sqrt(mean_size),
                palette=pal_map,
                ax=ax,
            )

            try:
                sns.pointplot(errorbar=None, **pointplot_kwargs)
            except TypeError:
                sns.pointplot(ci=None, **pointplot_kwargs)

            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        # --- Legend ---
        handles = [Patch(facecolor=pal_map[m], edgecolor="none", label=m) for m in models_order]
        ax.legend(
            handles=handles,
            title="Model",
            ncol=legend_ncol,
            prop={"size": legend_fontsize, "weight": "bold"},
        )

        # --- Reference line ---
        if ref_line is not None:
            ax.axhline(
                ref_line,
                ls=ref_line_style,
                color=ref_line_color,
                alpha=ref_line_alpha,
                linewidth=1.5,
            )

        # --- Labels & title ---
        ax.set_xlabel("", fontsize=label_fontsize, fontweight="bold")
        ax.set_ylabel("Value", fontsize=label_fontsize, fontweight="bold")
        if title is None:
            title = f"{state_name}: metrics per model"
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)

        # --- Grid & spines ---
        if grid_y:
            ax.yaxis.grid(True, which="major", alpha=0.3)
        ax.xaxis.grid(False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # --- Y-limits (consider yticks first) ---
        if yticks is not None and len(yticks) > 0:
            ymin, ymax = min(yticks), max(yticks)
            span = max(1e-6, ymax - ymin)
            pad_top = span * y_pad_top_frac
            ax.set_ylim(ymin, ymax + pad_top)
        elif y_limits is not None:
            ax.set_ylim(*y_limits)
        else:
            ymin = float(df["value"].min()) if not df.empty else 0.0
            ymax = float(df["value"].max()) if not df.empty else 1.0
            pad = max(1e-6, (ymax - ymin)) * y_pad_top_frac
            ax.set_ylim(ymin, ymax + pad)
        ax.margins(x=0.04)

        # --- Ensure all tick labels exist, then style them ---
        fig.canvas.draw()  # force creation of final tick labels

        # Set explicit y-ticks if provided
        if yticks is not None:
            ax.set_yticks(yticks)

        ax.tick_params(axis="both", labelsize=tick_fontsize)
        for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick.set_fontweight("bold")

        # --- Save / show ---
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)

        if show:
            plt.show()
        elif created_fig:
            plt.close(fig)

    return fig, ax


def multi_state_metric_boxplots(
    df_metrics: pd.DataFrame,
    state_names: Sequence[str],
    metric_names: Sequence[str],
    models_order: Optional[Sequence[str]] = None,
    ncols: int = 1,
    nrows: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    suptitle: Optional[str] = None,
    suptitle_fontsize: int = 30,
    shared_legend: bool = True,
    show: bool = True,
    set_theme: bool = True,
    save_path: Optional[str] = None,
    y_ticks: Optional[Sequence[float]] = None,  # same y-ticks for all states (comparable metrics)
    **metric_boxplots_kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Multi-panel figure with one subplot per state.

    For each state:
      - x-axis: metric (subset of metric_names)
      - hue: model
      - y-axis: value

    This is the "swapped" counterpart to multi_metric_state_boxplots,
    designed for comparable metrics where you want to compare states.
    """

    state_names = list(state_names)
    metric_names = list(metric_names)
    n_states = len(state_names)
    if n_states == 0:
        raise ValueError("state_names must contain at least one state.")

    if nrows is None:
        nrows = math.ceil(n_states / ncols)

    # Default overall figsize: grid of states
    if figsize is None:
        base_w, base_h = 4.5, 4.5
        figsize = (base_w * ncols, base_h * nrows)

    # These are controlled here, not via **kwargs
    for key in ("ax", "show", "set_theme", "save_path", "figsize", "dpi", "title", "yticks"):
        metric_boxplots_kwargs.pop(key, None)

    # --- Create figure and axes grid ---
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()

    # --- Plot each state on its own Axes ---
    first_ax_with_data = None
    for i, state_name in enumerate(state_names):
        ax = axes_flat[i]

        fig_i, ax_i = metric_boxplots(
            df_metrics=df_metrics,
            state_name=state_name,
            metric_names=metric_names,
            models_order=list(models_order) if models_order is not None else None,
            ax=ax,
            show=False,
            set_theme=set_theme and i == 0,
            title="",
            yticks=y_ticks,
            **metric_boxplots_kwargs,
        )
        if first_ax_with_data is None:
            first_ax_with_data = ax_i

    # Hide any unused axes
    for j in range(n_states, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # --- Hide repeated x-tick labels on non-bottom rows ---
    if nrows > 1:
        for row in range(nrows - 1):
            for col in range(ncols):
                ax = axes[row, col]
                if not ax.get_visible():
                    continue
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")

    # --- Subplot labels: use state names instead of A, B, C ---
    for state_name, ax in zip(state_names, axes_flat[:n_states]):
        if not ax.get_visible():
            continue
        ax.text(
            0.01,
            0.99,
            str(state_name),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,      # you can adjust size
            fontweight="bold",
        )

    # --- Layout: leave room for suptitle (top) and legend (bottom) ---
    fig.tight_layout(rect=[0.0, 0.1, 1.0, 0.95])

    # --- Figure-level suptitle ---
    if suptitle is None:
        suptitle = "States: " + ", ".join(state_names)
    fig.suptitle(suptitle, fontsize=suptitle_fontsize, fontweight="bold", y=0.96)

    # --- Shared legend below the figure ---
    if shared_legend and first_ax_with_data is not None:
        handles, labels = first_ax_with_data.get_legend_handles_labels()

        # Remove legends from individual axes
        for ax in axes_flat:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        legend_fontsize = metric_boxplots_kwargs.get("legend_fontsize", 16)
        n_models = len(handles)
        ncol = math.ceil(max(1, n_models) / 2)  # ~2 rows

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            borderaxespad=0.0,
            ncol=ncol,
            prop={"size": legend_fontsize, "weight": "bold"},
        )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes