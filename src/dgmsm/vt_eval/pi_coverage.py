from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from typing import Optional
from ._results import PICoverageResult

logger = logging.getLogger(__name__)

DEFAULT_PI_ALPHAS: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)


def _ipcw_weights(obs_times: np.ndarray, events: np.ndarray) -> np.ndarray:
    obs_times = np.asarray(obs_times, dtype=float)
    events = np.asarray(events, dtype=int)
    kmf = KaplanMeierFitter()
    kmf.fit(obs_times, event_observed=(1 - events))
    weights = np.zeros(len(obs_times), dtype=float)
    uncens = events == 1
    t_uncens = np.maximum(obs_times[uncens] - 1e-6, 0.0)
    g_vals = np.asarray(kmf.predict(t_uncens), dtype=float)
    weights[uncens] = 1.0 / np.maximum(g_vals, 1e-6)
    return weights


def pi_coverage(
    sim_times: np.ndarray,
    obs_times: np.ndarray,
    events: np.ndarray,
    alphas: tuple[float, ...] = DEFAULT_PI_ALPHAS,
    label: str = "",
    weights: Optional[np.ndarray] = None,
) -> PICoverageResult:
    sim_times = np.asarray(sim_times, dtype=float)
    obs_times = np.asarray(obs_times, dtype=float)
    events = np.asarray(events, dtype=int)

    if weights is None:
        weights = _ipcw_weights(obs_times, events)

    uncens = events == 1  
    n_events = int(uncens.sum())
    n_censored = int(events.shape[0] - n_events)

    alphas_arr = np.asarray(alphas, dtype=float)
    n_alpha = alphas_arr.size

    ipcw_hits = np.zeros(n_alpha)
    ipcw_weight_sum = np.zeros(n_alpha)
    det_hits = np.zeros(n_alpha)
    n_cens_above_U = np.zeros(n_alpha)
    width_sum = np.zeros(n_alpha)

    for j, alpha in enumerate(alphas_arr):
        q_lo, q_hi = (1 - alpha) / 2, (1 + alpha) / 2
        L = np.quantile(sim_times, q_lo, axis=1)
        U = np.quantile(sim_times, q_hi, axis=1)
        in_pi = (obs_times >= L) & (obs_times <= U)

        w_sum = weights[uncens].sum()
        ipcw_hits[j] = float((weights[uncens] * in_pi[uncens]).sum())
        ipcw_weight_sum[j] = float(w_sum)

        cens_above_U = (events == 0) & (obs_times > U)
        n_cens_above_U[j] = int(cens_above_U.sum())
        det_hits[j] = int(in_pi[uncens].sum())
        width_sum[j] = float((U[uncens] - L[uncens]).sum())

        logger.debug(
            "PI coverage [%s] alpha=%.2f  PICP=%.3f  width=%.2f",
            label, alpha,
            ipcw_hits[j] / w_sum if w_sum > 0 else float("nan"),
            width_sum[j] / max(n_events, 1),
        )

    return PICoverageResult(
        alphas=alphas_arr,
        ipcw_hits=ipcw_hits,
        ipcw_weight_sum=ipcw_weight_sum,
        det_hits=det_hits,
        n_cens_above_U=n_cens_above_U,
        width_sum=width_sum,
        n_events=n_events,
        n_censored=n_censored,
    )


def pi_coverage_to_frame(pi_cov: PICoverageResult) -> pd.DataFrame:
    if pi_cov is None or pi_cov.alphas.size == 0:
        return pd.DataFrame(columns=[
            "alpha", "ipcw_coverage", "det_lower_bound", "mean_PI_width",
            "n_events", "n_censored", "n_folds_pooled",
        ])
    return pd.DataFrame({
        "alpha": pi_cov.alphas,
        "ipcw_coverage": pi_cov.empirical,
        "det_lower_bound": pi_cov.det_lower_bound,
        "mean_PI_width": pi_cov.mean_pi_width,
        "n_events": pi_cov.n_events,
        "n_censored": pi_cov.n_censored,
        "n_folds_pooled": pi_cov.n_folds_pooled,
    })