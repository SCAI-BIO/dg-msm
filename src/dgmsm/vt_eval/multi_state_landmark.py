# multi_state_landmark.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
import matplotlib.pyplot as plt

from ..utils import set_global_seed
from .state_landmark import state_landmark, DEFAULT_PI_ALPHAS
from ._results import CVLandmarkResult, MultiStateLandmarkResult, StateLandmarkResult
from .plot import (
    plot_landmark_metrics,
    plot_cv_landmark_pp,
    plot_cv_landmark_pi_coverage,
)

logger = logging.getLogger(__name__)


def multi_state_landmark(
    model, dataset, trafo, state_names, n_sim=1000, batch_size=128,
    device=None, d_cal_bins=10, pi_alphas=None, seed=None, **kwargs,
) -> MultiStateLandmarkResult:
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model = model.to(device).eval()
    if seed is not None:
        set_global_seed(seed)
    pi_alphas = pi_alphas or DEFAULT_PI_ALPHAS

    n_states = model.graph.n_nonterminal_states
    present_states = dataset.states_present()
    eligible = [s for s in range(1, n_states + 1) if s in present_states]

    name_map = {s: state_names[s - 1] if s <= len(state_names) else str(s) for s in eligible}

    by_state: dict[int, StateLandmarkResult] = {}
    for s in eligible:
        try:
            by_state[s] = state_landmark(
                model=model, dataset=dataset, trafo=trafo, state_names=state_names,
                n_sim=n_sim, state=s, batch_size=batch_size, device=device,
                d_cal_bins=d_cal_bins, pi_alphas=pi_alphas, seed=None, **kwargs,
            )
            logger.info("Landmark state=%d (%s) done.", s, name_map[s])
        except Exception:
            logger.exception("Landmark state=%d failed", s)

    return MultiStateLandmarkResult(by_state=by_state, name_map=name_map)


def plot_multi_state_landmark(
    result: MultiStateLandmarkResult,
    *,
    plot_metrics: bool = True,
    plot_reliability: bool = True,
    plot_pi_coverage: bool = True,
    metrics_to_plot: Optional[list[str]] = None,
    metric_name_map: Optional[dict[str, str]] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
    show: bool = True,
    save_dir = None,
) -> dict[str, plt.Figure]:

    figures: dict[str, plt.Figure] = {}
    save_dir_path = Path(save_dir) if save_dir is not None else None
    cv_result = CVLandmarkResult.from_single_run(result)
    def _fig_path(stem: str):
        if save_dir_path is None:
            return None
        save_dir_path.mkdir(parents=True, exist_ok=True)
        return save_dir_path / f"{stem}.png"

    if plot_metrics:
        extra = dict(plot_kwargs or {})
        for k in ("metrics", "landmark_labels", "save_path", "show_plot"):
            extra.pop(k, None)
        try:
            figures["metrics"] = plot_landmark_metrics(
                metric_frames={
                    "Overall Survival": result.metrics_wide_frame("os"),
                    "Sojourn Time": result.metrics_wide_frame("sojourn"),
                },
                metrics=metrics_to_plot,
                metric_name_map=metric_name_map,
                landmark_labels=result.name_map,
                save_path=_fig_path("metrics"),
                show_plot=show,
                **extra,
            )
        except Exception:
            logger.exception("plot_multi_state_landmark: metrics plot failed")

    if plot_reliability and result.states():
        try:
            figures["reliability"] = plot_cv_landmark_pp(
                cv_result=cv_result,
                landmark_labels=result.name_map,
                show=show,
                save_path=_fig_path("reliability"),
                show_folds=False,
                plot_band=False,
            )
        except Exception:
            logger.exception("plot_multi_state_landmark: reliability plot failed")

    if plot_pi_coverage and result.states():
        try:
            figures["pi_coverage"] = plot_cv_landmark_pi_coverage(
                cv_result=cv_result,
                landmark_labels=result.name_map,
                show=show,
                save_path=_fig_path("pi_coverage"),
                show_folds=False,
                plot_band=False,
            )
        except Exception:
            logger.exception("plot_multi_state_landmark: pi_coverage plot failed")

    return figures