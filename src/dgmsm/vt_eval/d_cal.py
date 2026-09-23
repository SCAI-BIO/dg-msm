"""
D-Calibration for discrete-time survival models.

Overview
--------
D-Calibration (Haider et al., JMLR 2020) tests whether a model's predicted
survival distribution is *marginally calibrated* via the probability-integral
transform (PIT): if F_theta(t | x) is correct, the PIT of each subject's event
time is Uniform(0, 1).  Partition [0, 1] into B equal bins; under perfect
calibration each bin should receive exactly 1/B of the total mass.  A Pearson
chi-squared goodness-of-fit test against the uniform (df = B - 1) yields the
D-Calibration statistic and p-value.


References
----------
H. Haider, B. Hoehn, S. Davis, R. Greiner.
"Effective Ways to Build and Evaluate Individual Survival Distributions."
Journal of Machine Learning Research, 21(85):1-63, 2020.

M. Goldstein, X. Han, A. Puli, A. Perotte, R. Ranganath.
"X-CAL: Explicit Calibration for Survival Analysis."
Advances in Neural Information Processing Systems (NeurIPS), 33, 2020.
"""

import logging
from typing import Optional
import numpy as np
import scipy.stats

logger = logging.getLogger(__name__)


def _spread_uniform_mass(lo: np.ndarray, hi: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Distribute a unit mass per subject uniformly over the PIT interval [lo, hi]
    onto the bin grid `edges` (length n_bins + 1).
    This is the closed-form expectation of the randomized PIT U ~ Uniform(lo, hi).
    """
    n_bins = len(edges) - 1
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    degenerate = (hi - lo) <= 0.0
    bins = np.zeros(n_bins, dtype=float)

    if (~degenerate).any():
        lo_v = lo[~degenerate][:, None]
        hi_v = hi[~degenerate][:, None]
        width = hi_v - lo_v
        left = np.maximum(edges[None, :-1], lo_v)
        right = np.minimum(edges[None, 1:], hi_v)
        overlap = np.clip(right - left, 0.0, None)
        bins += (overlap / width).sum(axis=0)

    if degenerate.any():
        idx = np.clip((lo[degenerate] * n_bins).astype(int), 0, n_bins - 1)
        np.add.at(bins, idx, 1.0)

    return bins


def _dcal_stats(bins: np.ndarray, n: int, n_bins: int) -> dict:
    """
    Chi^2 goodness-of-fit + effect-size diagnostics for one bin vector.
    """
    expected = n / n_bins
    statistic = float(np.sum((bins - expected) ** 2 / expected))
    pvalue = float(scipy.stats.chi2.sf(statistic, df=n_bins - 1))  # tail-stable

    proportions = bins / n
    target = 1.0 / n_bins
    abs_dev = np.abs(proportions - target)

    # KS-type CUMULATIVE deviation: sup_t |F_hat(t) - t|.

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    cum_obs = np.concatenate([[0.0], np.cumsum(proportions)])
    cdf_max_deviation = float(np.abs(cum_obs - edges).max())

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "bins": bins,
        "bin_proportions": proportions,
        "target_proportion": target,
        "max_abs_deviation": float(abs_dev.max()),   # per-bin (PMF); chi^2 companion
        "mean_abs_deviation": float(abs_dev.mean()),
        "cdf_max_deviation": cdf_max_deviation,       # KS-type, sup_t|F_hat(t)-t|
        "chi2_per_n": statistic / n,
        "n": int(n),
    }


def _point_mass_counts(u: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Vectorized point-mass assignment (Haider Algorithm 2, delta=1 branch).

    Each value in `u` deposits a full unit mass into bucket
    j* = ceil(n_bins * u) (0-based, clipped to [0, n_bins-1]); u == 0 -> bin 0.
    """
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    idx = np.clip(np.ceil(u * n_bins).astype(int) - 1, 0, n_bins - 1)
    bins = np.zeros(n_bins, dtype=float)
    np.add.at(bins, idx, 1.0)
    return bins


def d_calibration(
    cdf: np.ndarray,
    event_time_idx: np.ndarray,
    event_observed: np.ndarray,
    *,
    n_bins: int = 10,
    uncensored_mode: str = "spread",
    event_time_frac: Optional[np.ndarray] = None,
    log: Optional[logging.Logger] = None,
) -> dict:
    """
    Single-event D-Calibration (Haider et al., 2020) with deterministic
    mass-spreading (= exact randomized-PIT expectation) so discrete time bins
    do not cause pile-up.

    """
    log = log or logger

    if uncensored_mode not in ("spread", "point", "interp"):
        raise ValueError("`uncensored_mode` must be 'spread', 'point' or 'interp'")
    if uncensored_mode == "interp" and event_time_frac is None:
        raise ValueError("uncensored_mode='interp' requires `event_time_frac`")

    cdf = np.asarray(cdf, dtype=float)
    if cdf.ndim == 3 and cdf.shape[-1] == 1:
        cdf = cdf[..., 0]
    if cdf.ndim != 2:
        raise ValueError(f"`cdf` must be (N, K), got shape {cdf.shape}")
    N, K = cdf.shape

    event_time_idx = np.asarray(event_time_idx).astype(int)
    event_observed = np.asarray(event_observed).astype(int)
    if event_time_idx.shape != (N,) or event_observed.shape != (N,):
        raise ValueError("`event_time_idx` and `event_observed` must both have shape (N,)")

    is_event = event_observed == 1
    is_cens = ~is_event

    # events index a PMF cell 0..K-1; censored counts may reach the sentinel K
    if is_event.any():
        ev_t = event_time_idx[is_event]
        if ev_t.min() < 0 or ev_t.max() > K - 1:
            raise ValueError("event `time` out of range [0, K-1]")
    if is_cens.any():
        c_all = event_time_idx[is_cens]
        if c_all.min() < 0 or c_all.max() > K:
            raise ValueError("censored `time` out of range [0, K]")

    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0), axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = np.arange(N)
    k = event_time_idx

    F_k = cdf[rows, np.clip(k, 0, K - 1)]
    F_km1 = np.where(k > 0, cdf[rows, np.clip(k - 1, 0, K - 1)], 0.0)

    bins = np.zeros(n_bins, dtype=float)

    # (a) uncensored
    if is_event.any():
        if uncensored_mode == "point":
            bins += _point_mass_counts(F_k[is_event], n_bins)
        elif uncensored_mode == "interp":
            frac_e = np.clip(np.asarray(event_time_frac)[is_event], 0.0, 1.0)
            lo_e = F_km1[is_event]
            hi_e = np.maximum(F_k[is_event], lo_e)
            u = lo_e + frac_e * (hi_e - lo_e)     # exact PIT position
            bins += _point_mass_counts(u, n_bins)
        else:  # spread
            lo_e = F_km1[is_event]
            hi_e = np.maximum(F_k[is_event], lo_e)
            bins += _spread_uniform_mass(lo_e, hi_e, edges)

    # (b) censored: tail blur over [Fu_cens, 1]
    if is_cens.any():
        c = event_time_idx[is_cens]

        if uncensored_mode == "interp" and event_time_frac is not None:
            frac_c = np.clip(np.asarray(event_time_frac)[is_cens], 0.0, 1.0)

            # F(k-1): lower edge of censoring bin
            lo_idx = c - 1
            F_prev = np.where(
                lo_idx >= 0,
                cdf[rows[is_cens], np.clip(lo_idx, 0, K - 1)],
                0.0,                              # c == 0 -> F(-1) = 0
            )
            # F(k): upper edge (np.clip(K,0,K-1)=K-1, so horizon degenerates correctly)
            F_curr = cdf[rows[is_cens], np.clip(c, 0, K - 1)]

            # Interpolated Fu: F(k-1) + frac * delta_F
            # At horizon (c=K): F_prev=F_curr=F(K-1), delta=0 -> Fu=F(K-1)
            delta = np.maximum(F_curr - F_prev, 0.0)
            Fu_cens = F_prev + frac_c * delta

        else:  # spread / point: use F(k-1), correct without sub-bin info
            lo_idx = c - 1
            Fu_cens = np.where(
                lo_idx >= 0,
                cdf[rows[is_cens], np.clip(lo_idx, 0, K - 1)],
                0.0,
            )

        bins += _spread_uniform_mass(Fu_cens, np.ones(int(is_cens.sum())), edges)

    stats = _dcal_stats(bins, n=N, n_bins=n_bins)
    stats.update(
        n_events=int(is_event.sum()),
        n_censored=int(is_cens.sum()),
        uncensored_mode=uncensored_mode,
    )

    log.info(
        "D-Cal[%s]: chi2=%.4f, p=%.4f, n=%d (event=%d, cens=%d), "
        "chi2/n=%.5f, max_dev_pmf=%.4f, max_dev_cdf=%.4f",
        uncensored_mode,
        stats["statistic"], stats["pvalue"], N,
        stats["n_events"], stats["n_censored"],
        stats["chi2_per_n"], stats["max_abs_deviation"], stats["cdf_max_deviation"],
    )

    stats["n_bins"] = n_bins
    return stats