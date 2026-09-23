from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _aj_estimator(
    Tstart: np.ndarray,
    Tstop: np.ndarray,
    frm: np.ndarray,      # 0-based from-state per row
    to: np.ndarray,       # 0-based to-state per row
    ev: np.ndarray,       # event indicator per row (0 = censored/no transition)
    K: int,
    p0: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    etimes = np.unique(Tstop[ev != 0])
    p = np.asarray(p0, float).copy()
    out = np.empty((etimes.size, K), float)

    for i, t in enumerate(etimes):
        at_risk = (Tstart < t) & (t <= Tstop)
        Y = np.array([np.count_nonzero(at_risk & (frm == j)) for j in range(K)])

        trans = (Tstop == t) & (ev != 0)
        A = np.zeros((K, K))
        for j in range(K):
            fj = trans & (frm == j)
            if not fj.any() or Y[j] == 0:
                continue
            for k in range(K):
                dN = np.count_nonzero(fj & (to == k))
                if dN:
                    A[j, k] = dN / Y[j]
            A[j, j] = -A[j].sum()

        p = p @ (np.eye(K) + A)
        out[i] = p

    return etimes, out



def _eval_aj_on_grid(etimes: np.ndarray, P: np.ndarray,
                     p0: np.ndarray, grid: np.ndarray) -> np.ndarray:
    K = P.shape[1] if P.size else len(p0)
    out = np.tile(np.asarray(p0, float), (len(grid), 1))
    if etimes.size:
        idx = np.searchsorted(etimes, grid, side="right") - 1
        valid = idx >= 0
        out[valid] = P[idx[valid]]
    return out


def _states_to_aj_arrays(msp, df):
    """Extract 0-based (Tstart, Tstop, frm, to, ev) arrays for _aj_estimator."""
    Tstart = df["Tstart"].to_numpy(float)
    Tstop = df["Tstop"].to_numpy(float)
    frm = df["state"].to_numpy(int) - 1
    to = df["next_state"].to_numpy(int) - 1
    ev = df["event"].to_numpy(int)
    return Tstart, Tstop, frm, to, ev


def aj_divergence_per_state(
    real_msp,
    synthetic_msp,
    *,
    time_scale: float = 1.0,
    n_points: int = 500,
    censor_fix: bool = False,
    warn: bool = False,
) -> pd.DataFrame:
    """
    State-specific Aalen–Johansen divergence between two MultiStatePrep objects.

        d_AJ,j = (1/T) ∫ |P_real,j(t) - P_synth,j(t)| dt

    Returns one row per state with divergence and similarity score.
    """
    real_msp.assert_same_metadata(synthetic_msp)
    K = real_msp.n_states_total

    df_r = real_msp._closed_states(warn=warn) if censor_fix else real_msp.states
    df_s = synthetic_msp._closed_states(warn=warn) if censor_fix else synthetic_msp.states

    Ts_r, Te_r, frm_r, to_r, ev_r = _states_to_aj_arrays(real_msp, df_r)
    Ts_s, Te_s, frm_s, to_s, ev_s = _states_to_aj_arrays(synthetic_msp, df_s)

    # Fixed entry-state distribution for a fair dynamics-only comparison
    p0 = np.zeros(K)
    p0[0] = 1.0

    g_r, P_r = _aj_estimator(Ts_r, Te_r, frm_r, to_r, ev_r, K, p0)
    g_s, P_s = _aj_estimator(Ts_s, Te_s, frm_s, to_s, ev_s, K, p0)

    # scale to desired time unit
    g_r = g_r / time_scale
    g_s = g_s / time_scale
    Ts_r_max = Te_r.max() / time_scale
    Ts_s_max = Te_s.max() / time_scale

    t_max = min(Ts_r_max, Ts_s_max)
    if t_max <= 0 or not np.isfinite(t_max):
        return pd.DataFrame(columns=["AJ divergence", "AJ Similarity Score"])

    grid = np.linspace(0.0, t_max, n_points)
    Pr = _eval_aj_on_grid(g_r, P_r, p0, grid)
    Ps = _eval_aj_on_grid(g_s, P_s, p0, grid)

    rows = []
    for j in range(K):
        trapz = getattr(np, 'trapezoid', None) or np.trapz
        d = trapz(np.abs(Pr[:, j] - Ps[:, j]), grid) / t_max
        rows.append({
            "state_name": real_msp.state_names[j],
            "AJ divergence": float(d),
            "AJ Similarity Score": float((1.0 - d) * 100.0),
        })
    return pd.DataFrame(rows).set_index("state_name")