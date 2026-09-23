import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader

def split_features(x, ftypes):
    data_list = []
    idx = 0
    for feat in ftypes:
        width = int(feat['nclass'])
        data_list.append(x[:, idx: idx + width])
        idx += width
    return data_list

def _normalize_feature(
    d: torch.Tensor,             # [N, width_i] (usually width_i=1 for real/pos)
    miss_col: torch.Tensor,      # [N] mask 1=observed, 0=missing (float/int/bool)
    ftype: str,
    mean: Optional[torch.Tensor] = None,   # scalar, same dtype/device as d
    var: Optional[torch.Tensor]  = None,   # scalar
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize a single feature column (or block) d given a missingness mask and
    feature type. If mean/var are None, compute them from observed data.
    Returns:
        normalized_d: [N, width_i]
        mean: scalar tensor (for this feature)
        var:  scalar tensor (for this feature)
    """
    # observed mask: True where we *have* data
    obs_mask = (miss_col != 0)
    observed = d[obs_mask]  # [N_obs, width_i]

    # If no observed data: return zeros and safe defaults
    if observed.numel() == 0:
        norm_d = torch.zeros_like(d)
        mean_out = torch.tensor(0.0, dtype=d.dtype, device=d.device)
        var_out  = torch.tensor(1.0, dtype=d.dtype, device=d.device)
        norm_d = torch.nan_to_num(norm_d, nan=0.0, posinf=0.0, neginf=0.0)
        return norm_d, mean_out, var_out

    ftype = ftype.lower()

    # Categorical/ordinal: leave unchanged (store dummy mean/var)
    if ftype not in {"real", "pos"}:
        norm_d = d.clone()
        mean_out = torch.tensor(0.0, dtype=d.dtype, device=d.device)
        var_out  = torch.tensor(1.0, dtype=d.dtype, device=d.device)
        norm_d = torch.nan_to_num(norm_d, nan=0.0, posinf=0.0, neginf=0.0)
        return norm_d, mean_out, var_out

    # Continuous features: "real" or "pos"
    # Transform domain if needed, then apply mean/var
    if ftype == "real":
        # If no precomputed stats, compute on observed data
        if mean is None or var is None:
            var_out, mean_out = torch.var_mean(observed, unbiased=False)
            var_out = torch.clamp(var_out, min=1e-6, max=1e20)
        else:
            mean_out = mean.to(d.device, dtype=d.dtype)
            var_out  = var.to(d.device, dtype=d.dtype)

        std = torch.sqrt(torch.clamp(var_out, min=1e-6))
        norm_obs = (observed - mean_out) / std

    elif ftype == "pos":
        # Clamp to strictly positive then log1p
        observed_clamped = torch.clamp(observed, min=1e-6)
        observed_log = torch.log1p(observed_clamped)

        if mean is None or var is None:
            var_out, mean_out = torch.var_mean(observed_log, unbiased=False)
            var_out = torch.clamp(var_out, min=1e-6, max=1e20)
        else:
            mean_out = mean.to(d.device, dtype=d.dtype)
            var_out  = var.to(d.device, dtype=d.dtype)

        std = torch.sqrt(torch.clamp(var_out, min=1e-6))
        norm_obs = (observed_log - mean_out) / std

    # Scatter back into full tensor
    norm_d = torch.zeros_like(d)
    norm_d[obs_mask] = norm_obs
    norm_d[~obs_mask] = 0.0

    norm_d = torch.nan_to_num(norm_d, nan=0.0, posinf=0.0, neginf=0.0)
    return norm_d, mean_out, var_out

def batch_normalization(
    batch_data_list: List[torch.Tensor],
    miss_list: torch.Tensor,          # [N, n_features]
    feat_types_list,
):
    """
    Compute per-feature normalization parameters from a batch (mean, var)
    and return normalized data + params.
    """
    normalized_data: List[torch.Tensor] = []
    normalization_parameters: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i, d in enumerate(batch_data_list):
        ftype = feat_types_list[i]["type"]
        miss_col = miss_list[:, i]  # [N]

        norm_d, mean_i, var_i = _normalize_feature(
            d=d,
            miss_col=miss_col,
            ftype=ftype,
            mean=None,
            var=None,
        )
        normalized_data.append(norm_d)
        normalization_parameters.append((mean_i, var_i))

    return normalized_data, normalization_parameters

@torch.no_grad()
def compute_norm_params_from_dataloader(
    config,
    dataloader,
    device: torch.device,
) -> tuple[list[Tuple[torch.Tensor, torch.Tensor]], list[Tuple[torch.Tensor, torch.Tensor]]]:

    # Convert dict[name -> schema] to list[schema] in the same order as model_kwargs()
    base_ftypes_list = [config.base_ftypes[name] for name in config.base_ftypes.keys()]
    state_ftypes_list = [config.state_ftypes[name] for name in config.state_ftypes.keys()]

    base_obs_acc = None
    m_base_acc = []

    state_obs_acc = None
    m_state_acc = []

    v_dim = config.n_nonterminal_states
    if v_dim is None:
        raise ValueError("config.n_nonterminal_states is None; call compute_tmat_stats in ModelConfig.__post_init__.")

    for (x_base, m_base), (x_state, m_state), _ in dataloader:
        x_base  = x_base.to(device)   # [B, D_base_enc]
        m_base  = m_base.to(device)   # [B, n_base_features]

        x_state = x_state.to(device)  # [B, S, D_state_total]
        m_state = m_state.to(device)  # [B, S, n_state_features + 2]

        # Drop visit mask + state mask (keep only feature masks)
        m_state_feat = m_state[..., 2:]  # [B, S, n_state_features]

        # Drop time + state one-hot from x_state, keep only covariates
        # x_state structure: [t (1), v_onehot (v_dim), state_x_enc (state_x_dim)]
        x_state_cov = x_state[..., v_dim + 1 :]  # [B, S, state_x_dim]

        B, S, _ = x_state_cov.shape

        # -------- Baseline features --------
        base_data = split_features(x_base, base_ftypes_list)  # list of [B, width_i]
        base_data_obs = [
            d * m_base[:, i].view(B, 1)
            for i, d in enumerate(base_data)
        ]

        base_data_obs_cpu = [d.detach().cpu() for d in base_data_obs]
        if base_obs_acc is None:
            base_obs_acc = [[] for _ in base_data_obs_cpu]
        for i, d_cpu in enumerate(base_data_obs_cpu):
            base_obs_acc[i].append(d_cpu)

        m_base_acc.append(m_base.detach().cpu())

        # -------- State features --------
        x_flat = x_state_cov.view(B * S, -1)                       # [B*S, state_x_dim]
        m_flat = m_state_feat.view(B * S, len(state_ftypes_list))  # [B*S, n_state_features]

        state_data = split_features(x_flat, state_ftypes_list)  # list of [B*S, width_i]
        state_data_obs = [
            d * m_flat[:, i].view(B * S, 1)
            for i, d in enumerate(state_data)
        ]

        state_data_obs_cpu = [d.detach().cpu() for d in state_data_obs]
        if state_obs_acc is None:
            state_obs_acc = [[] for _ in state_data_obs_cpu]
        for i, d_cpu in enumerate(state_data_obs_cpu):
            state_obs_acc[i].append(d_cpu)

        m_state_acc.append(m_flat.detach().cpu())

    if base_obs_acc is None or state_obs_acc is None:
        raise ValueError("Dataset/dataloader appears to be empty.")

    # Concatenate all batches along the sample dimension
    base_data_obs_all = [torch.cat(lst, dim=0) for lst in base_obs_acc]
    m_base_all = torch.cat(m_base_acc, dim=0)                  # [N_total, n_base_features]

    state_data_obs_all = [torch.cat(lst, dim=0) for lst in state_obs_acc]
    m_state_all = torch.cat(m_state_acc, dim=0)                # [N_total*S, n_state_features]

    # Compute normalization params on CPU for all data
    _, base_norm_params = batch_normalization(
        base_data_obs_all, m_base_all, base_ftypes_list
    )
    _, state_norm_params = batch_normalization(
        state_data_obs_all, m_state_all, state_ftypes_list
    )

    return base_norm_params, state_norm_params


@torch.no_grad()
def apply_normalization_from_params(
    data_list: List[torch.Tensor],
    miss_list: torch.Tensor,          # [N, n_features]
    feat_types_list,
    norm_params: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[torch.Tensor]:
    """
    Apply precomputed normalization parameters (mean, var) to a list of feature tensors.
    """
    normalized_data: List[torch.Tensor] = []

    for i, d in enumerate(data_list):
        ftype = feat_types_list[i]["type"]
        miss_col = miss_list[:, i]
        mean_i, var_i = norm_params[i]

        norm_d, _, _ = _normalize_feature(
            d=d,
            miss_col=miss_col,
            ftype=ftype,
            mean=mean_i,
            var=var_i,
        )
        normalized_data.append(norm_d)

    return normalized_data
def apply_normalization_from_buffers(
    data_list: List[torch.Tensor],
    miss_list: torch.Tensor,
    feat_types_list,
    mean_buf: torch.Tensor,   # [n_features]
    var_buf: torch.Tensor,    # [n_features]
) -> List[torch.Tensor]:
    """
    Same normalization logic, but using mean/var from 1D buffers.
    """
    normalized_data: List[torch.Tensor] = []

    for i, d in enumerate(data_list):
        ftype = feat_types_list[i]["type"]
        miss_col = miss_list[:, i]
        mean_i = mean_buf[i]
        var_i  = var_buf[i]

        norm_d, _, _ = _normalize_feature(
            d=d,
            miss_col=miss_col,
            ftype=ftype,
            mean=mean_i,
            var=var_i,
        )
        normalized_data.append(norm_d)

    return normalized_data