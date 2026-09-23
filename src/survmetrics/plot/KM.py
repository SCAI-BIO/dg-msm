from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter


def plot_km(
    durations: pd.Series,
    events: pd.Series,
    ax: Optional[plt.Axes] = None,
    label: str = "KM curve",
    xlabel: Optional[str] = "Time",
    ylabel: Optional[str] = "Probability",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    ylim: Tuple[float, float] = (0.0, 1.0),
    xlim: Optional[Tuple[float, float]] = None,
    show: bool = True,
) -> plt.Axes:
    # Create or reuse axes
    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            constrained_layout=True,
            dpi=dpi,
        )
    else:
        fig = ax.figure

    # Fit KM and plot
    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events, label=label)
    kmf.plot_survival_function(
        ax=ax,
        ci_show=False,
        linewidth=2.2,
    )

    # Axis limits
    ax.set_ylim(*ylim)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        d = durations.dropna()
        if len(d) > 0:
            x_min = max(0.0, float(d.min()))
            x_max = float(d.max())
            ax.set_xlim(x_min, x_max)

    # Labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold")

    # Subtle y-grid
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25)

    # Legend
    ax.legend(loc="best", frameon=False)

    if show:
        plt.show()

    return ax