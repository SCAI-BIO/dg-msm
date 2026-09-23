import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

from msprep import MultiStatePrep
from msprep.template import MSPTemplate, align_dtypes_like_schema
from .utils import discrete_variables_transformation, MultiStateData

logger = logging.getLogger(__name__)


@dataclass
class PosteriorEvalResult:
    synthetic_msp: MultiStatePrep
    id_mapping: pd.DataFrame
    eligible_person_ids: list
    n_conditioning_visits: int
    time_delta: Optional[float] = None


def _decode_state_covariates(cov_enc: np.ndarray, model) -> np.ndarray:
    """Decode state covariates from the model's encoded representation."""
    cov_tensor = torch.from_numpy(cov_enc).float()
    cov_dec = discrete_variables_transformation(
        cov_tensor, model.state_ftypes
    )
    if isinstance(cov_dec, torch.Tensor):
        return cov_dec.cpu().numpy()
    return cov_dec


@torch.no_grad()
def baseline_simulation(
    model,
    msp_template: "MSPTemplate",
    dataset: MultiStateData,
    n_sim: int,
    real_msp: MultiStatePrep,
    time_delta: Optional[float] = None,
    max_future_steps: Optional[int] = None,
    dataset_name: Optional[str] = None,
    temperature: float = 1.0,
) -> PosteriorEvalResult:
    """
    Simulate fully random trajectories conditioned ONLY on each patient's
    baseline covariates (no observed states).

    For every real patient in `dataset`, the model draws `n_sim` forward
    rollouts starting from:
        baseline -> s, z_0, h_0 -> sampled initial state v_1 -> trajectory

    This calls `model.simulate(..., x_state=None, m_state=None)`, i.e. the
    baseline-only branch of `_encode_history`. Covariates and labels are then
    decoded back to the original feature space and wrapped into a new
    MultiStatePrep using `msp_template` metadata only (no real data rows).
    """
    device = next(model.parameters()).device

    base_feat_names  = list(msp_template.baseline_features.keys())
    state_feat_names = list(msp_template.state_features.keys())
    tmat             = msp_template.tmat
    state_names      = msp_template.state_names
    event_names      = msp_template.event_names
    baseline_schema  = msp_template.baseline_schema
    states_schema    = msp_template.states_schema

    real_baseline_df      = real_msp.baseline
    persons_list          = dataset.persons
    real_baseline_indexed = real_baseline_df.set_index("person_id")

    synth_baseline_rows: List[Dict[str, Any]] = []
    synth_states_rows:   List[Dict[str, Any]] = []
    id_mapping_rows:     List[Dict[str, Any]] = []
    synth_pid = 1

    all_state_cols = (
        ["person_id", "state", "Tstart", "Tstop", "time", "event"]
        + state_feat_names
    )

    n_skipped_not_found  = 0
    n_skipped_sim_failed = 0
    person_ids_used: List[Any] = []

    for i in range(len(dataset)):
        (x_base, m_base), (x_state, m_state), labels = dataset[i]

        orig_pid = persons_list[i]
        if orig_pid not in real_baseline_indexed.index:
            logger.warning(
                "Patient %s not found in real_msp baseline; skipping.", orig_pid
            )
            n_skipped_not_found += 1
            continue

        base_feat_vals = real_baseline_indexed.loc[orig_pid, base_feat_names].values

        x_base = x_base.to(device)
        m_base = m_base.to(device)

        # --- Baseline-only simulation: no observed states ---
        try:
            sim_out = model.simulate(
                x_base=x_base,
                m_base=m_base,
                x_state=None,
                m_state=None,
                n_sim=n_sim,
                deterministic_history=False,
                max_future_steps=max_future_steps,
                temperature=temperature,
                return_covariates=True,
                return_labels=True,
            )
        except ValueError as e:
            logger.warning(
                "Patient index %d: simulation failed (%s); skipping.", i, e
            )
            n_skipped_sim_failed += 1
            continue

        if "labels" not in sim_out:
            n_skipped_sim_failed += 1
            continue

        person_ids_used.append(orig_pid)

        labels_sim     = sim_out["labels"]      # [n_sim, max_path, 5]
        covariates_sim = sim_out["covariates"]  # List[n_sim] of np.ndarray [L, state_x_dim]
        n_sim_i        = labels_sim.shape[0]

        for s in range(n_sim_i):
            labels_s  = labels_sim[s].numpy()                               # [max_path, 5]
            cov_s_dec = _decode_state_covariates(covariates_sim[s], model)  # [L, n_feats]
            L         = cov_s_dec.shape[0]

            # Baseline row (carry over the real patient's baseline covariates)
            synth_base_row = {"person_id": synth_pid}
            for k, fname in enumerate(base_feat_names):
                synth_base_row[fname] = base_feat_vals[k]
            synth_baseline_rows.append(synth_base_row)

            # Write the full simulated trajectory
            n_written = 0
            for j in range(labels_s.shape[0]):
                state_id_j = int(labels_s[j, 0])
                if state_id_j == 0:
                    break  # padding / after terminal

                dur_disc    = float(labels_s[j, 1])
                event_id_j  = int(labels_s[j, 2])
                Tstart_disc = float(labels_s[j, 3])
                Tstop_disc  = float(labels_s[j, 4])

                if time_delta is not None:
                    td       = float(time_delta)
                    Tstart_r = Tstart_disc * td
                    Tstop_r  = Tstop_disc  * td
                    dur_r    = dur_disc    * td
                else:
                    Tstart_r, Tstop_r, dur_r = Tstart_disc, Tstop_disc, dur_disc

                cov_idx   = min(j, L - 1)
                feat_vals = cov_s_dec[cov_idx]

                row = {col: np.nan for col in all_state_cols}
                row.update({
                    "person_id": synth_pid,
                    "state":     state_id_j,
                    "Tstart":    Tstart_r,
                    "Tstop":     Tstop_r,
                    "time":      dur_r,
                    "event":     event_id_j,
                })
                for k, fname in enumerate(state_feat_names):
                    row[fname] = float(feat_vals[k])

                synth_states_rows.append(row)
                n_written += 1

            if n_written == 0:
                synth_baseline_rows.pop()
                continue

            id_mapping_rows.append({
                "synthetic_person_id": synth_pid,
                "real_person_id":      orig_pid,
                "sim_index":           s,
                "patient_index":       i,
            })
            synth_pid += 1

    logger.info(
        "Baseline simulation (temp=%.2f): %d synthetic patients from %d real "
        "patients (%d sims each). Skipped: %d not found, %d sim failed.",
        temperature, synth_pid - 1, len(person_ids_used), n_sim,
        n_skipped_not_found, n_skipped_sim_failed,
    )

    if not synth_baseline_rows:
        raise RuntimeError("No synthetic patients generated.")

    synth_baseline_df = pd.DataFrame(synth_baseline_rows)
    synth_states_df   = pd.DataFrame(synth_states_rows)
    synth_baseline_df = align_dtypes_like_schema(synth_baseline_df, baseline_schema)
    synth_states_df   = align_dtypes_like_schema(synth_states_df,   states_schema)
    synth_baseline_df["person_id"] = synth_baseline_df["person_id"].astype(int)
    synth_states_df  ["person_id"] = synth_states_df  ["person_id"].astype(int)

    if dataset_name is None:
        dataset_name = "Synthetic Data (baseline-conditioned)"

    synthetic_msp = MultiStatePrep.from_dataframes(
        dataset_name=dataset_name,
        baseline=synth_baseline_df,
        states=synth_states_df,
        baseline_features=dict(msp_template.baseline_features),
        state_features=dict(msp_template.state_features),
        state_names=list(state_names),
        event_names=list(event_names),
        tmat=tmat,
    )

    return PosteriorEvalResult(
        synthetic_msp=synthetic_msp,
        id_mapping=pd.DataFrame(id_mapping_rows),
        n_conditioning_visits=0,          # no observed visits
        eligible_person_ids=person_ids_used,
        time_delta=time_delta,
    )

@torch.no_grad()
def prior_sample_msp(
    model,
    msp_template: MSPTemplate,   # metadata only
    n: int,
    time_delta = None,
    dataset_name: Optional[str] = None,
    temperature: Optional[float] = 1.0,
) -> MultiStatePrep:

    # 1) Generate synthetic encoded data & labels from the model's prior
    synth = model.sample(
        N = n,
        temperature = temperature,
        )
    x_base_enc       = synth["x_base"]          # [N, base_x_dim]
    x_state_enc_full = synth["x_state"]         # [N, max_visits, 1 + v_dim + state_x_dim]
    labels           = synth["labels"]          # [N, max_visits, 5]

    N, max_path, _ = labels.shape
    assert N == n, "Mismatch between requested n and generated N"

    v_dim = model.v_dim

    # 2) Decode encoded covariates back to original feature space
    base_feat_names  = list(msp_template.baseline_features.keys())
    state_feat_names = list(msp_template.state_features.keys())

    # Baseline: x_base_enc -> [N, n_base_features]
    x_base_dec = discrete_variables_transformation(
        x_base_enc, model.base_ftypes
    )  # [N, n_base_features]
    if isinstance(x_base_dec, torch.Tensor):
        x_base_dec = x_base_dec.cpu().numpy()
    baseline_feat_df = pd.DataFrame(x_base_dec, columns=base_feat_names)

    # State covariates: strip Tstart + state one-hot, keep only covariates
    # x_state_enc_full: [N, max_path, 1 + v_dim + state_x_dim]
    x_state_cov_enc = x_state_enc_full[..., 1 + v_dim :]   # [N, max_path, state_x_dim]
    x_state_flat = x_state_cov_enc.reshape(N * max_path, -1)

    x_state_dec_flat = discrete_variables_transformation(
        x_state_flat, model.state_ftypes
    )  # [N*max_path, n_state_features]
    if isinstance(x_state_dec_flat, torch.Tensor):
        x_state_dec_flat = x_state_dec_flat.cpu().numpy()
    x_state_dec = x_state_dec_flat.reshape(N, max_path, -1)  # [N, max_path, n_state_features]

    # 3) Extract labels and optionally rescale times
    state_ids = labels[..., 0].long()   # [N, max_path], 1-based, 0 = padding
    dur       = labels[..., 1].float()  # [N, max_path]
    event_ids = labels[..., 2].long()   # [N, max_path]
    Tstart    = labels[..., 3].float()  # [N, max_path]
    Tstop     = labels[..., 4].float()  # [N, max_path]

    if time_delta is not None:
        Tstart = Tstart * float(time_delta)
        Tstop  = Tstop  * float(time_delta)
        dur    = dur    * float(time_delta)

    # 4) Build baseline_df (person-level)
    person_ids = np.arange(1, N + 1)

    baseline_df = pd.DataFrame({"person_id": person_ids})
    baseline_df = pd.concat([baseline_df, baseline_feat_df], axis=1)

    # Align dtypes using baseline_schema from msp_template (metadata only)
    baseline_df = align_dtypes_like_schema(baseline_df, msp_template.baseline_schema)
    baseline_df["person_id"] = baseline_df["person_id"].astype(int)

    # 5) Build states_df (interval-level; one row per visit)
    rows_states = []
    for i in range(N):
        pid = person_ids[i]
        for j in range(max_path):
            state_id_ij = int(state_ids[i, j].item())
            if state_id_ij == 0:
                continue  # padding / after terminal

            event_id_ij = int(event_ids[i, j].item())
            Tstart_ij   = float(Tstart[i, j].item())
            Tstop_ij    = float(Tstop[i, j].item())
            time_ij     = float(dur[i, j].item())

            feat_vals = x_state_dec[i, j, :]  # [n_state_features]

            row = {
                "person_id": pid,
                "state": state_id_ij,
                "Tstart": Tstart_ij,
                "Tstop":  Tstop_ij,
                "time":   time_ij,
                "event":  event_id_ij,
            }
            for k, fname in enumerate(state_feat_names):
                row[fname] = feat_vals[k]
            rows_states.append(row)

    states_df = pd.DataFrame(rows_states)
    states_df = align_dtypes_like_schema(states_df, msp_template.states_schema)
    states_df["person_id"] = states_df["person_id"].astype(int)
    if dataset_name is None:
        dataset_name = "Synthetic Data (prior)"
        
    # 6) Wrap into new MultiStatePrep
    synthetic_msp = MultiStatePrep.from_dataframes(
        dataset_name=dataset_name,
        baseline=baseline_df,
        states=states_df,
        baseline_features=dict(msp_template.baseline_features),
        state_features=dict(msp_template.state_features),
        state_names=list(msp_template.state_names),
        event_names=list(msp_template.event_names),
        tmat=msp_template.tmat,
    )

    return synthetic_msp