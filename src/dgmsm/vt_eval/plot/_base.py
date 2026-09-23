# _base.py
from __future__ import annotations

import logging
import os
import string
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


_SUPTITLE_SIZE   = 11
_SUPTITLE_WEIGHT = "bold"

_OKABE_ITO: list[str] = [
    "#0072B2",  # blue       – DG-MSM / first method
    "#D55E00",  # vermillion – benchmark 1
    "#009E73",  # teal
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

def _pub_rc() -> dict[str, Any]:
    """Publication-quality rc params shared by all figures."""
    return {
        "figure.dpi":             300,
        "savefig.dpi":            300,
        "font.family":            "sans-serif",
        "font.sans-serif":        ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":              9,
        "axes.titlesize":         10,
        "axes.labelsize":         9,
        "legend.fontsize":        8,
        "xtick.labelsize":        8,
        "ytick.labelsize":        8,
        "axes.linewidth":         0.6,
        "xtick.major.width":      0.6,
        "ytick.major.width":      0.6,
        "xtick.major.size":       3.0,
        "ytick.major.size":       3.0,
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "axes.grid":              True,
        "grid.alpha":             0.22,
        "grid.linewidth":         0.4,
        "grid.color":             "0.82",
        "lines.solid_capstyle":   "round",
        "patch.linewidth":        0.5,
        "pdf.fonttype":           42,
        "ps.fonttype":            42,
        "svg.fonttype":           "none",
        "figure.constrained_layout.use": False,
    }


def _save_figure(
    fig: plt.Figure,
    save_path: str | os.PathLike[str],
    *,
    dpi: int = 300,
) -> None:
    """Save *fig* to *save_path* and write a PDF copy unless already .pdf."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    logger.info("Saved figure → %s", path)


def _panel_label(idx: int) -> str:
    alpha = string.ascii_lowercase
    if idx < 26:
        return alpha[idx]
    hi, lo = divmod(idx - 26, 26)
    return alpha[hi] + alpha[lo]




def _blank_panel(ax: plt.Axes, text: str = "n/a") -> None:
    """Turn an axes into an empty 'no data' placeholder panel."""
    ax.axis("off")
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
             ha="center", va="center", fontsize=8, color="0.5")


def _add_panel_label(
    ax: plt.Axes,
    idx: int,
    *,
    fontsize: int = 8,
    fontweight: str = "bold",
) -> None:
    ax.text(
        0.03, 0.97, f"{_panel_label(idx)})",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=fontsize, fontweight=fontweight,
        zorder=5, clip_on=False,
    )


