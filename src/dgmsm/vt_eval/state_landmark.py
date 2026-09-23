# state_landmark.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ..utils import set_global_seed


from survmetrics import SurvMetrics
from .d_cal import d_calibration
from .pi_coverage import pi_coverage, DEFAULT_PI_ALPHAS

from ._results import StateLandmarkResult, DCalResult, EndpointResult

from .plot import (
    plot_landmark_reliability,
    plot_landmark_pi_coverage,
)

logger = logging.getLogger(__name__)

_RESULT_LABEL: dict[str, str] = {"os": "OS", "sojourn": "Sojourn"}


def state_landmark(
    model,
    dataset,
    trafo,
    state_names,
    n_sim: int = 1000,
    state: int = 1,
    batch_size: int = 128,
    device=None,
    seed=None,
    d_cal_bins: Optional[int] = 10,
    uncensored_mode: str = "interp",
    pi_alphas: Optional[tuple[float, ...]] = None,
    plot_reliability: bool = False,
    plot_pi_coverage: bool = False,
    reliability_target=None,
    pi_coverage_target=None,
    save_path: Optional[str] = None,
    **kwargs,
) -> StateLandmarkResult:
    """
    Landmark evaluation for one non-terminal state: survival discrimination
    metrics (SurvMetrics), D-Calibration, and PICP(alpha) prediction-interval
    coverage, for both overall survival (OS) and sojourn time.
    """

    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        set_global_seed(seed)
    if model.n_time != trafo.n_time:
        raise ValueError(f"model.n_time={model.n_time} \u2260 trafo.n_time={trafo.n_time}.")

    device = torch.device(device)
    model = model.to(device).eval()
    pi_alphas = pi_alphas or DEFAULT_PI_ALPHAS

    landmark_state_name = state_names[state - 1]
    n_time = model.n_time
    dt = trafo.time_delta

    terminal_event_ids = torch.as_tensor(
        model.graph.terminal_event_ids, dtype=torch.long
    ).cpu().flatten()

    # 0. Prepare landmarking dataset and extract target labels
    landmark_ds = dataset.state_hitting_subset(state=state, mask_future=False)
    landmark_idx = np.asarray(landmark_ds.hit_positions, dtype=int)

    os_durations, os_events, os_fracs = _get_os_labels(
        landmark_ds=landmark_ds, terminal_event_ids=terminal_event_ids, dt=dt,
    )
    sojourn_durations, sojourn_events, sojourn_fracs = _get_sojourn_labels(
        landmark_ds=landmark_ds
    )

    # 1. Run simulations
    loader = DataLoader(landmark_ds, batch_size=batch_size, shuffle=False)
    out_chunks: Dict[str, list] = {}

    with torch.inference_mode():
        for (x_base, m_base), (x_state, m_state), labels in loader:
            x_base, m_base = x_base.to(device), m_base.to(device)
            x_state, m_state = x_state.to(device), m_state.to(device)
            labels = labels.to(device)

            sim_out = model.simulate_from_state(
                x_base=x_base, m_base=m_base,
                x_state=x_state, m_state=m_state,
                labels=labels, state=state,
                n_sim=n_sim, return_labels=True,
                **kwargs,
            )
            for k, v in sim_out.items():
                if isinstance(v, torch.Tensor):
                    out_chunks.setdefault(k, []).append(v.cpu())
                elif isinstance(v, list):
                    out_chunks.setdefault(k, []).extend(v)
                else:
                    out_chunks.setdefault(k, []).append(v)

    result: Dict[str, Any] = {
        k: (torch.cat(chunks, dim=0) if isinstance(chunks[0], torch.Tensor) else chunks)
        for k, chunks in out_chunks.items()
        if chunks
    }

    stop_time_rel = torch.as_tensor(result["stop_time_rel"]).cpu().numpy()
    event_sim = torch.as_tensor(result["event"]).cpu().numpy().astype(bool)
    labels_sim = result["labels"]

    landmark_idx_from_sim = (
        torch.as_tensor(result["landmark_visits"]).cpu().numpy().astype(int) - 1
    )
    if not np.array_equal(landmark_idx_from_sim, landmark_idx):
        raise RuntimeError(
            "Mismatch between dataset.hit_positions and simulate_from_state landmark_visits."
        )

    lm_rows_sim = _extract_landmark_rows_from_sim(
        labels_sim=labels_sim, landmark_indices=landmark_idx,
    )
    soj_time_sim = lm_rows_sim[:, :, 1].float().cpu().numpy()
    soj_event_sim = (lm_rows_sim[:, :, 2] > 0).cpu().numpy()

    # 2. Per-endpoint calibration: SurvMetrics + D-Calibration + PICP(alpha)
    endpoint_inputs = {
        "os": dict(
            durations_obs=os_durations, events_obs=os_events, fracs_obs=os_fracs,
            durations_sim=stop_time_rel, events_sim=event_sim,
            K=_infer_dcal_horizon_bins(
                observed_durations=os_durations, observed_fracs=os_fracs,
                simulated_durations=stop_time_rel, simulated_events=event_sim,
                min_K=n_time,
            ),
        ),
        "sojourn": dict(
            durations_obs=sojourn_durations, events_obs=sojourn_events, fracs_obs=sojourn_fracs,
            durations_sim=soj_time_sim, events_sim=soj_event_sim,
            K=n_time,
        ),
    }

    endpoints: dict[str, EndpointResult] = {}
    for ep, inputs in endpoint_inputs.items():
        endpoints[ep] = _compute_endpoint_calibration(
            endpoint=ep,
            label=f"{_RESULT_LABEL[ep]} | state={landmark_state_name}",
            d_cal_bins=d_cal_bins, uncensored_mode=uncensored_mode, pi_alphas=pi_alphas,
            **inputs,
        )

    # 3. Optional plots
    if plot_pi_coverage or pi_coverage_target is not None:
        pi_save_path = None
        if save_path and pi_coverage_target is None:
            root, ext = os.path.splitext(save_path)
            pi_save_path = f"{root}_pi_coverage{ext}"

        plot_landmark_pi_coverage(
            endpoints["sojourn"].pi_cov,
            endpoints["os"].pi_cov,
            state_name=landmark_state_name,
            save_path=pi_save_path,
            show=pi_coverage_target is None,
            target=pi_coverage_target,
        )

    if plot_reliability or reliability_target is not None:
        plot_landmark_reliability(
            endpoints["sojourn"].d_cal,
            endpoints["os"].d_cal,
            save_path=None if reliability_target is not None else save_path,
            show=reliability_target is None,
            plot_band=False,
            target=reliability_target,
            state_name=landmark_state_name,
        )
    return StateLandmarkResult(
        state=state,
        state_name=landmark_state_name,
        n_at_risk=len(landmark_ds),
        endpoints=endpoints,
        sim_result=result,
    )


def _step_lookup_matrix(
    time_grid: np.ndarray, values: np.ndarray, query_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate an already-computed right-continuous step function at new
    query points via lookup, rather than re-estimating it from raw samples.
    """
    idx = np.searchsorted(time_grid, query_grid, side="right") - 1
    idx = np.clip(idx, 0, len(time_grid) - 1)
    return values[idx, :]


def _cdf_from_survival_matrix(
    time_grid: np.ndarray, surv_mat: np.ndarray, n_time: int,
) -> np.ndarray:
    """Per-subject discrete CDF F = 1 - S on integer grid 0..n_time-1."""
    grid = np.arange(n_time, dtype=float)
    surv_at_grid = _step_lookup_matrix(time_grid, surv_mat, grid)   # (n_time, N)
    return np.clip(1.0 - surv_at_grid.T, 0.0, 1.0)                  # (N, n_time)


def _compute_endpoint_calibration(
    *, endpoint: str, label: str,
    durations_obs, events_obs, fracs_obs, durations_sim, events_sim, K,
    d_cal_bins, uncensored_mode, pi_alphas,
) -> EndpointResult:
    time_grid, surv_mat = _mc_survival_from_simulation(durations_sim, events_sim)

    t_lo, t_hi = int(durations_obs.min()), int(durations_obs.max())
    times_k = time_grid[(time_grid >= t_lo) & (time_grid <= t_hi)]
    if times_k.size == 0:
        times_k = time_grid

    metrics = dict(SurvMetrics(
        surv=pd.DataFrame(surv_mat, index=time_grid),
        durations=durations_obs.astype(int),
        events=events_obs.astype(int),
    ).compute(time_grid=times_k))

    cdf = _cdf_from_survival_matrix(time_grid, surv_mat, n_time=K)
    eti = _event_time_idx_for_dcal(durations_obs, events_obs, K=K, log_name=f"D-calibration [{label}]")
    nb = d_cal_bins if d_cal_bins is not None else _resolve_n_bins(int(events_obs.sum()))

    d_cal_dict = d_calibration(
        cdf=cdf, event_time_idx=eti, event_observed=events_obs.astype(int),
        event_time_frac=fracs_obs, uncensored_mode=uncensored_mode, n_bins=nb,
    )

    pi_cov = pi_coverage(
        sim_times=durations_sim, obs_times=durations_obs.astype(float),
        events=events_obs, label=label, alphas=pi_alphas,
    )

    return EndpointResult(
        endpoint=endpoint,
        label=label,
        metrics=metrics,
        d_cal=DCalResult.from_d_calibration(d_cal_dict, horizon_bins=K),
        pi_cov=pi_cov,
    )


# Survival / CDF helpers
def _mc_survival_from_simulation(durations, events):
    durations = torch.as_tensor(durations).cpu().numpy().astype(float)
    events = torch.as_tensor(events).cpu().numpy().astype(bool)
    time_grid = np.unique(np.concatenate([[0.0], durations.ravel()]))

    surv_mat = np.stack(
        [_km_from_samples(durations[i], events[i], time_grid)
         for i in range(durations.shape[0])],
        axis=1,
    )
    return time_grid, surv_mat


def _km_from_samples(times: np.ndarray, events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    grid = np.asarray(grid, dtype=float)

    event_times = np.unique(times[events])
    if event_times.size == 0:
        return np.ones_like(grid, dtype=float)

    surv_at_event = np.empty_like(event_times, dtype=float)
    s = 1.0
    for j, t in enumerate(event_times):
        n_risk = np.sum(times >= t)
        d_j = np.sum((times == t) & events)
        s *= 1.0 - d_j / max(n_risk, 1)
        surv_at_event[j] = s

    idx = np.searchsorted(event_times, grid, side="right") - 1
    out = np.ones_like(grid, dtype=float)
    mask = idx >= 0
    out[mask] = surv_at_event[idx[mask]]
    return out


def _resolve_n_bins(
    n_events: int, target_per_bin: int = 5, min_bins: int = 5, max_bins: int = 20,
) -> int:
    """Adaptive bin count: ~target_per_bin events per bin, clipped to [min, max]."""
    if n_events <= 0:
        return min_bins
    return int(max(min_bins, min(max_bins, n_events // target_per_bin)))


# Label extraction helpers
def _get_sojourn_labels(landmark_ds) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = torch.stack([torch.as_tensor(lb) for lb in landmark_ds.labels], dim=0)
    labels_cont = torch.stack([torch.as_tensor(lc) for lc in landmark_ds.labels_cont], dim=0)
    landmark_indices = torch.as_tensor(landmark_ds.hit_positions, dtype=torch.long)
    b = torch.arange(labels.size(0))

    durations = labels[b, landmark_indices, 1].float().numpy()
    events = (labels[b, landmark_indices, 2] > 0).numpy().astype(int)
    fracs = labels_cont[b, landmark_indices, 1].float().numpy()

    return durations, events, fracs


def _get_os_labels(
    landmark_ds, terminal_event_ids: torch.Tensor, dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = torch.stack([torch.as_tensor(lb) for lb in landmark_ds.labels], dim=0)
    labels_cont = torch.stack([torch.as_tensor(lc) for lc in landmark_ds.labels_cont], dim=0)
    landmark_indices = torch.as_tensor(landmark_ds.hit_positions, dtype=torch.long)
    terminal_event_ids = terminal_event_ids.to(dtype=torch.long)

    N, max_path, _ = labels.shape
    obs_mask = labels[:, :, 0] != 0
    n_obs = obs_mask.sum(dim=1)

    event_ids = labels[:, :, 2].long()
    positions = torch.arange(max_path).unsqueeze(0)

    term_mask = torch.isin(event_ids, terminal_event_ids)
    term_mask = term_mask & (event_ids > 0) & obs_mask
    term_mask = term_mask & (positions >= landmark_indices.unsqueeze(1))

    has_event = term_mask.any(dim=1)
    first_term_idx = term_mask.float().argmax(dim=1).long()
    last_obs_idx = n_obs.clamp_min(1) - 1
    end_idx = torch.where(has_event, first_term_idx, last_obs_idx)

    events = has_event.numpy().astype(int)

    from_mask = positions >= landmark_indices.unsqueeze(1)
    to_mask = positions <= end_idx.unsqueeze(1)
    range_mask = (from_mask & to_mask).float()

    soj_time_all = labels_cont[:, :, 0].float()
    total_cont = (soj_time_all * range_mask).sum(dim=1)
    total_bins = total_cont / float(dt)

    eps = 1e-6 * torch.clamp(total_bins.abs(), min=1.0)
    durations_t = (total_bins + eps).floor().clamp_min(0).long()
    frac_os = (total_bins - durations_t.float()).clamp(0.0, 1.0 - 1e-9)

    return durations_t.numpy(), events, frac_os.numpy()


def _extract_landmark_rows_from_sim(labels_sim, landmark_indices: np.ndarray) -> torch.Tensor:
    labels_t = torch.as_tensor(labels_sim)

    if labels_t.ndim != 4 or labels_t.shape[-1] < 5:
        raise ValueError(f"Expected labels_sim shape [N, S, P, 5], got {labels_t.shape}")

    N, S, P, C = labels_t.shape
    landmark_indices = np.asarray(landmark_indices, dtype=int)

    if landmark_indices.shape[0] != N:
        raise ValueError(
            f"landmark_indices has length {landmark_indices.shape[0]}, but labels_sim has N={N}."
        )
    if np.any(landmark_indices < 0) or np.any(landmark_indices >= P):
        raise ValueError("Some landmark indices are outside the simulated path length.")

    idx = torch.as_tensor(landmark_indices, dtype=torch.long, device=labels_t.device)
    gather_idx = idx.view(N, 1, 1, 1).expand(N, S, 1, C)

    return labels_t.gather(dim=2, index=gather_idx).squeeze(2)


def _infer_dcal_horizon_bins(
    observed_durations: np.ndarray,
    observed_fracs: Optional[np.ndarray ] = None,
    simulated_durations: Optional[np.ndarray ] = None,
    simulated_events: Optional[np.ndarray ] = None,
    min_K: int = 1,
) -> int:
    """
    Infer number of discrete CDF columns for D-calibration.

    Returned K gives CDF columns 0..K-1.
    Event indices must be in 0..K-1.
    Censored observations may use K as sentinel.
    """
    obs_d = np.asarray(observed_durations, dtype=float).ravel()

    if observed_fracs is None:
        obs_f = np.zeros_like(obs_d, dtype=float)
    else:
        obs_f = np.asarray(observed_fracs, dtype=float).ravel()
        if obs_f.shape != obs_d.shape:
            raise ValueError(
                f"observed_fracs shape {obs_f.shape} does not match "
                f"observed_durations shape {obs_d.shape}"
            )

    observed_time = obs_d + obs_f
    observed_time = observed_time[np.isfinite(observed_time)]

    max_times = []
    if observed_time.size > 0:
        max_times.append(float(observed_time.max()))

    if simulated_durations is not None:
        sim_t = np.asarray(simulated_durations, dtype=float)
        if simulated_events is not None:
            sim_e = np.asarray(simulated_events, dtype=bool)
            if sim_e.shape != sim_t.shape:
                raise ValueError(
                    f"simulated_events shape {sim_e.shape} does not match "
                    f"simulated_durations shape {sim_t.shape}"
                )
            sim_t = sim_t[sim_e]
        else:
            sim_t = sim_t.ravel()
        sim_t = sim_t[np.isfinite(sim_t)]
        if sim_t.size > 0:
            max_times.append(float(sim_t.max()))

    if not max_times:
        return int(min_K)
    return max(int(min_K), int(np.ceil(max(max_times))) + 1)


def _event_time_idx_for_dcal(
    durations: np.ndarray, events: np.ndarray, K: int, log_name: str = "D-calibration",
) -> np.ndarray:
    """
    Build event_time_idx for d_calibration.
    """
    K = int(K)
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")

    durations_idx = np.floor(np.asarray(durations, dtype=float)).astype(int)
    events_bool = np.asarray(events).astype(bool)

    if durations_idx.shape != events_bool.shape:
        raise ValueError(
            f"durations shape {durations_idx.shape} does not match events shape {events_bool.shape}"
        )

    event_oob = events_bool & ((durations_idx < 0) | (durations_idx >= K))
    cens_oob = (~events_bool) & ((durations_idx < 0) | (durations_idx > K))

    if event_oob.any() or cens_oob.any():
        logger.warning(
            "%s: event_time_idx still requires clipping. "
            "event_oob=%d, cens_oob=%d, K=%d, max_duration_idx=%d",
            log_name, int(event_oob.sum()), int(cens_oob.sum()), K, int(durations_idx.max()),
        )

    idx = np.empty_like(durations_idx, dtype=int)
    idx[events_bool] = np.clip(durations_idx[events_bool], 0, K - 1)
    idx[~events_bool] = np.clip(durations_idx[~events_bool], 0, K)
    return idx