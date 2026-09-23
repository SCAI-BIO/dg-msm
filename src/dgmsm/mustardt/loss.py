import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MuStaRDTLoss(nn.Module):
    def __init__(
        self,
        aux_weight: float = 1.0,
        eps: float = 1e-6,
        ):
        super().__init__()
        self.aux_weight = aux_weight
        self.eps        = eps

    def _nll(
        self,
        p_logits: torch.Tensor,   # [batch_size, max_path, E]
        f_logits: torch.Tensor,   # [batch_size, max_path, E, K]
        labels: torch.Tensor,     # [batch_size, max_path, 5]
    ) -> torch.Tensor:
        batch_size, max_path, E, K = f_logits.shape
        device = f_logits.device

        states = labels[..., 0].long()
        times  = labels[..., 1].long()
        events = labels[..., 2].long()

        visit_mask = (states != 0).view(-1)
        if visit_mask.sum() == 0:
            return p_logits.new_zeros(())

        time_flat = times.view(-1)[visit_mask]
        event_flat = events.view(-1)[visit_mask]

        bad_time = (time_flat < 0) | (time_flat > K)
        if bad_time.any():
            raise ValueError(f"Times must be in 0..{K}.")

        p_flat = p_logits.view(-1, E)[visit_mask]       # [N_vis, E]
        f_flat = f_logits.view(-1, E, K)[visit_mask]    # [N_vis, E, K]

        total_nll = p_flat.new_zeros(())
        total_count = 0

        log_h   = F.logsigmoid(f_flat)      # [N_vis, E, K], log h(k)
        log_1mh = F.logsigmoid(-f_flat)     # [N_vis, E, K], log(1-h(k))

        # Cumulative log-survival
        cum_log_surv = torch.zeros_like(log_1mh)
        cum_log_surv[..., 1:] = torch.cumsum(log_1mh[..., :-1], dim=-1)

        full_cum_log_surv = torch.cumsum(log_1mh, dim=-1)  # [N_vis, E, K]


        log_pmf = log_h + cum_log_surv  # [N_vis, E, K]

        # Event-choice log-probs (unchanged)
        logp = F.log_softmax(p_flat, dim=-1)  # [N_vis, E]

        obs_mask = (event_flat != 0)
        N_obs = int(obs_mask.sum().item())

        if N_obs > 0:
            obs_event_raw = event_flat[obs_mask]
            obs_time = time_flat[obs_mask]

            wrong_event = (obs_event_raw < 1) | (obs_event_raw > E)
            if wrong_event.any():
                raise ValueError(f"Observed event ids must be in 1..{E}.")
            obs_event = obs_event_raw - 1
            if ((obs_time < 0) | (obs_time >= K)).any():
                raise ValueError("Observed event times must be in 0..K-1.")

            ce_p = -logp[obs_mask].gather(-1, obs_event[:, None]).squeeze(-1)

            log_pmf_obs = log_pmf[obs_mask]
            log_f_at_event_time = log_pmf_obs[
                torch.arange(N_obs, device=device), obs_event, obs_time
            ]
            ce_t = -log_f_at_event_time

            
            loss_obs = ce_p + ce_t              # [N_obs]
            total_nll = total_nll + loss_obs.sum()
            total_count += N_obs

        cens_mask = ~obs_mask
        N_cens = int(cens_mask.sum().item())

        if N_cens > 0:
            cens_time = time_flat[cens_mask]               # allowed: 0..K
            logp_cens = logp[cens_mask]                    # [N_cens, E]
            full_cum_cens = full_cum_log_surv[cens_mask]   # [N_cens, E, K]

            log_S_e = torch.empty_like(logp_cens)          # [N_cens, E]

            no_full_bin = cens_time == 0
            inside = (cens_time > 0) & (cens_time < K)
            at_tail = cens_time == K


            if no_full_bin.any():
                log_S_e[no_full_bin] = 0.0

            if inside.any():
                idx = (cens_time[inside] - 1).view(-1, 1, 1).expand(-1, E, 1)
                log_S_e[inside] = full_cum_cens[inside].gather(-1, idx).squeeze(-1)

            if at_tail.any():
                log_S_e[at_tail] = full_cum_cens[at_tail, :, -1]

            log_surv = torch.logsumexp(logp_cens + log_S_e, dim=-1)
            loss_cens = -log_surv

            total_nll = total_nll + loss_cens.sum()
            total_count += N_cens

        return total_nll / max(total_count, 1)


    def _aux_loss(self, p_logits, f_logits, labels):
        B, T, E, K = f_logits.shape
        device = f_logits.device

        # S_e(k) = P(T_e > k), for k=0..K-1
        log_1mh = F.logsigmoid(-f_logits)              # [B, T, E, K]
        log_S_e = torch.cumsum(log_1mh, dim=-1)       # [B, T, E, K]
        S_e = log_S_e.exp()

        # Mixture over event types
        p = F.softmax(p_logits, dim=-1)               # [B, T, E]
        S = (p.unsqueeze(-1) * S_e).sum(dim=2)        # [B, T, K], S(k)=P(T > k)

        # CDF on 0..K-1
        F_main = 1.0 - S                              # [B, T, K], F(k)=P(T <= k)

        # Augment with tail point K, where CDF must be 1
        F_aug = torch.cat(
            [F_main, torch.ones_like(F_main[..., :1])],
            dim=-1
        )                                             # [B, T, K+1]

        state_id = labels[..., 0].long()
        duration_idx = labels[..., 1].long()
        event_idx = labels[..., 2].long()

        # Use SAME valid-mask convention as _NLL
        valid_mask = (state_id != 0)

        if valid_mask.any():
            bad_time = (duration_idx[valid_mask] < 0) | (duration_idx[valid_mask] > K)
            if bad_time.any():
                raise ValueError(f"CRPS expects times in 0..{K}.")

            bad_obs = valid_mask & (event_idx > 0) & (
                (duration_idx < 0) | (duration_idx >= K)
            )
            if bad_obs.any():
                raise ValueError(f"Observed event times must be in 0..{K-1}.")

        exact_mask = valid_mask & ((event_idx > 0) | ((event_idx == 0) & (duration_idx == K)))

        # Proper right-censoring inside horizon
        cens_mask = valid_mask & (event_idx == 0) & (duration_idx < K)

        k_grid = torch.arange(K + 1, device=device).view(1, 1, K + 1)
        dur_expanded = duration_idx.unsqueeze(-1)


        exact_target_cdf = (k_grid >= dur_expanded).float()   # [B, T, K+1]
        crps_exact = (F_aug - exact_target_cdf).pow(2).sum(dim=-1)  # [B, T]


        known_event_free_upto = (duration_idx - 1).unsqueeze(-1)    # [B, T, 1]
        survived_through = (k_grid <= known_event_free_upto).float()  # [B, T, K+1]
        crps_cens = (F_aug.pow(2) * survived_through).sum(dim=-1)     # [B, T]


        total_score = f_logits.new_zeros(())
        total_count = 0.0

        if exact_mask.any():
            total_score = total_score + (crps_exact * exact_mask.float()).sum()
            total_count += exact_mask.float().sum().item()

        if cens_mask.any():
            total_score = total_score + (crps_cens * cens_mask.float()).sum()
            total_count += cens_mask.float().sum().item()

        loss = total_score / max(total_count, 1.0)
        return loss / (K + 1)


    def forward(
        self,
        p_logits: torch.Tensor,
        f_logits: torch.Tensor,
        labels: torch.Tensor,
        return_components: bool = False,
    ):
        nll = self._nll(p_logits=p_logits, f_logits=f_logits, labels=labels)
        total_loss = nll

        # unweighted component values, default 0
        aux = f_logits.new_zeros(())

        if self.aux_weight != 0.0:
            aux = self._aux_loss(p_logits=p_logits, f_logits=f_logits, labels=labels)
            total_loss = total_loss + self.aux_weight * aux

        if not return_components:
            return total_loss

        components = {
            "loss":            total_loss.detach(),
            "nll":             nll.detach(),
            "aux_weighted":    (self.aux_weight * aux).detach(),
            "aux_raw":          aux.detach(),
        }
        return total_loss, components