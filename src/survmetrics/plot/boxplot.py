from typing import Optional, Sequence, Tuple, List, Dict
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D 


def state_boxplots(
    df_metrics: pd.DataFrame,
    metric_name: str,
    state_names: Optional[List[str]] = None,
    models_order: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 200,
    palette_models: Optional[List[str]] = None,
    show_fliers: bool = False,
    box_width: float = 0.55,
    box_linewidth: float = 1.8,
    whisker = 1.5,
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
    """Boxplot of one metric per state and model."""

    df = df_metrics.copy()
    df = df[df["metric"] == metric_name].copy()
    if df.empty:
        raise ValueError(f"No rows for metric {metric_name!r} in df_metrics.")
    df = df.drop(columns=["metric"])

    # state_order
    if not state_names:  # covers None and []
        state_names = sorted(df["state"].unique().tolist())
    else:
        missing_states = [s for s in state_names if s not in df["state"].unique()]
        if missing_states:
            raise ValueError(
                f"Some states in 'state_names' are not present in the data: {missing_states}"
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
            # fall back to seaborn if more colors are needed
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

        # --- Boxplots ---
        sns.boxplot(
            x="state",
            y="value",
            hue="model",
            data=df,
            order=state_names,
            hue_order=models_order,
            palette=pal_map,
            width=box_width,
            whis=whisker,
            fliersize=3 if show_fliers else 0,
            linewidth=1.3,
            dodge=0.7,
            ax=ax,
            showmeans=True,
            #boxprops=dict(alpha=0.6, linewidth=1.3),
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            meanprops=dict(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=6,
                zorder=5,
            ),
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


        # --- Legend ---
        model_handles = [
            Patch(facecolor=pal_map[m], edgecolor="none", label=m)
            for m in models_order
        ]

        handles = list(model_handles)

        mean_handle = Line2D(
            [], [],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="Mean",
        )
        handles.append(mean_handle)

        ax.legend(
            handles=handles,
            title=None,
            ncol=legend_ncol,
            prop={"size": legend_fontsize, "weight": "bold"},
            frameon=False,
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
        ax.set_ylabel(metric_name, fontsize=label_fontsize, fontweight="bold")
        if title is None:
            title = f"{metric_name} per state and model"
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)

        # --- Grid & spines ---
        if grid_y:
            ax.yaxis.grid(True, which="major", alpha=0.3)
        ax.xaxis.grid(False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # --- Y-limits (now consider yticks first) ---

        if yticks is not None and len(yticks) > 0:
            ymin, ymax = min(yticks), max(yticks)
            span = max(1e-6, ymax - ymin)
            pad_top = span * y_pad_top_frac   # same fraction you use in the auto case
            ax.set_ylim(ymin, ymax + pad_top) # extra space above last tick
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


        # (bold tick styling as you like)
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


def multi_metric_state_boxplots(
    df_metrics: pd.DataFrame,
    metric_names: Optional[Sequence[str]] = None,
    state_names: Optional[Sequence[str]] = None,
    models_order: Optional[Sequence[str]] = None,
    ncols: int = 1,
    nrows: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    suptitle: Optional[str] = None,
    suptitle_fontsize: int = 30,
    suptitle_y: float = 1.0,
    shared_legend: bool = True,
    show: bool = True,
    set_theme: bool = True,
    save_path: Optional[str] = None,
    y_ticks: Optional[Dict[str, Sequence[float]]] = None,
    legend_ncol: int = 2,
    **state_boxplots_kwargs,
) -> tuple[plt.Figure, np.ndarray]:

    if not metric_names:
        metric_names = sorted(df_metrics.metric.unique())
    if not state_names:
        state_names = sorted(df_metrics["state"].unique().tolist())

    n_metrics = len(metric_names)
    if n_metrics == 0:
        raise ValueError("metric_names must contain at least one metric.")

    if nrows is None:
        nrows = n_metrics

    # Default overall figsize: wide, with enough height per metric
    if figsize is None:
        base_w, base_h = 12.0, 4.5   # more vertical space per metric
        figsize = (base_w, base_h * nrows)

    # These are controlled here, not via **kwargs
    for key in ("ax", "show", "set_theme", "save_path", "figsize", "dpi", "title"):
        state_boxplots_kwargs.pop(key, None)

    # --- Create figure and axes grid, SHARED X-AXIS ---
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharex=True,
    )
    axes_flat = axes.ravel()

    # --- Plot each metric on its own Axes ---
    for i, metric_name in enumerate(metric_names):
        ax = axes_flat[i]

        state_boxplots(
            df_metrics=df_metrics,
            metric_name=metric_name,
            state_names=list(state_names),
            models_order=list(models_order) if models_order is not None else None,
            ax=ax,
            show=False,
            set_theme=set_theme and i == 0,
            title="",
            yticks=y_ticks.get(metric_name, None) if y_ticks is not None else None,
            **state_boxplots_kwargs,
        )
    # No unused axes to remove: nrows == n_metrics, ncols == 1

    # --- Hide repeated x-tick labels & x-labels on all but bottom axis ---
    if nrows > 1:
        for row in range(nrows - 1):        # all rows except last
            ax = axes[row, 0]
            if not ax.get_visible():
                continue
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")

    labels = [f"{c})" for c in string.ascii_lowercase]
    for i, ax in enumerate(axes_flat[:n_metrics]):
        ax.text(
            0.01, 0.99,
            labels[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=22,
            fontweight="bold",
        )
    # --- Layout: leave room for suptitle (top) and legend (bottom) ---
    fig.tight_layout(
        rect=[0.0, 0.05, 1.0, 0.95],
        h_pad=5.0,
        w_pad=5.0,
    )

    # --- Figure-level suptitle ---
    if suptitle is None:
        suptitle = "Metrics: " + ", ".join(metric_names)
    fig.suptitle(suptitle, fontsize=suptitle_fontsize, fontweight="bold", y=suptitle_y)

    # --- Shared legend below the figure, 2 rows ---
    if shared_legend:
        first_ax = axes_flat[0]
        leg = first_ax.get_legend()
        if leg is None:
            raise RuntimeError("No legend found on first axis; state_boxplots must create it.")


        handles = leg.legend_handles
        labels = [t.get_text() for t in leg.get_texts()]

        # Remove legends from individual axes
        for ax in axes_flat:
            lg = ax.get_legend()
            if lg is not None:
                lg.remove()

        legend_fontsize = state_boxplots_kwargs.get("legend_fontsize", 16)

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.00),
            borderaxespad=0.0,
            ncol=legend_ncol,
            prop={"size": legend_fontsize, "weight": "bold"},
            frameon=True,
        )
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes