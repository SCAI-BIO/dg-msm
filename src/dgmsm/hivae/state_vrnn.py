import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple

from .normalization import (
    batch_normalization,
    apply_normalization_from_buffers,
)

from .feature_decoder import FeatureDecoder


class StateVRNN(nn.Module):
    def __init__(
        self,
        state_x_dim: int,
        state_ftypes: List[Dict[str, Any]],
        y_dim: int,
        z_dim: int,
        h_dim: int,
        v_dim: int,
        s_dim: int,
        state_norm_params: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        super().__init__()
        self.state_ftypes = state_ftypes
        self.state_x_dim = state_x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.v_dim = v_dim
        self.s_dim = s_dim

        z_prior_in_dim  = 1 + self.v_dim + self.h_dim + self.s_dim
        z_post_in_dim   = self.state_x_dim + 1 + self.v_dim + self.h_dim + self.s_dim

        # normal distributed  prior for the nth visited state v_n at Tstart t_n, given history h_n-1
        self.z_n_prior = nn.Sequential(
            nn.Linear(z_prior_in_dim, z_prior_in_dim),
            nn.ReLU(),
            nn.Linear(z_prior_in_dim, 2 * self.z_dim),
        )        # z_n ~ p(z_n | t_n, v_n, h_n-1, s)
                
        # approximate posterior given state cov x_n, current state index v_n, current memory h_n-1
        # z_n ~ q(z_n | x_n, t_n, v_n, h_n-1)
        # mu(x_n, t_n, v_n, h_n-1), logvar(x_n, t_n, v_n, h_n-1)
        self.z_n_posterior = nn.Sequential(
            nn.Linear(z_post_in_dim, z_post_in_dim),
            nn.ReLU(),
            nn.Linear(z_post_in_dim, 2 * self.z_dim),
        )
                                
        self.phi_z = nn.Sequential(nn.Linear(self.z_dim, self.h_dim), nn.ReLU())
        self.phi_v = nn.Sequential(nn.Linear(1 + self.v_dim, self.h_dim), nn.ReLU())

        # Decoder (theta) for state covariates
        self.state_decoder = FeatureDecoder(
            feat_types=self.state_ftypes,
            y_dim=y_dim,
            s_dim=self.s_dim,
            y_in_dim=self.z_dim + self.h_dim,  # latent_in = [z_n, h_{n-1}]
        )

        # Use an RNN to store the past trajectories of each patient
        # h_n = RNN(v_n, t_n, z_n, h_n-1)
        self.memory_rnn   = nn.GRU(
            input_size=2 * self.h_dim,  # phi_v + phi_z
            hidden_size=self.h_dim,
            batch_first=True,
        )

        # Store normalisation parameters
        self.state_norm_params = state_norm_params

        # Register state normalization buffers
        if state_norm_params is not None:
            state_means = torch.stack([m for (m, _) in state_norm_params])  # [n_state_features]
            state_vars  = torch.stack([v for (_, v) in state_norm_params])  # [n_state_features]
            self.register_buffer("state_norm_mean", state_means)
            self.register_buffer("state_norm_var",  state_vars)
        else:
            self.state_norm_mean = None
            self.state_norm_var  = None


    def normalize(
        self,
        x_state: torch.Tensor,        # [B, max_path, state_x_dim]
        m_state_feat: torch.Tensor,   # [B, max_path, n_state_features]
    ) -> tuple[torch.Tensor, list[Any]]:
        """
        Split, mask, and normalize state covariates per visit.

        Returns:
            state_data_norm: [B, max_path, state_x_dim] (normalized),
            state_normalization_params: list of (mean,var) per feature.
        """
        B, max_path, _ = x_state.shape
        x_flat = x_state.reshape(B * max_path, -1)
        m_flat = m_state_feat.reshape(B * max_path, len(self.state_ftypes))

        state_data = self.split_features(x_flat)

        if self.state_norm_mean is not None and self.state_norm_var is not None:
            state_list = apply_normalization_from_buffers(
                data_list=state_data,
                miss_list=m_flat,
                feat_types_list=self.state_ftypes,
                mean_buf=self.state_norm_mean,
                var_buf=self.state_norm_var,
            )
            state_data_norm_flat = torch.cat(state_list, dim=1)
            state_data_norm = state_data_norm_flat.view(B, max_path, -1)
            state_normalization_params = self.state_norm_params
        else:
            state_data_observed = [
                d * m_flat[:, i].view(B * max_path, 1)
                for i, d in enumerate(state_data)
            ]
            state_list, state_normalization_params = batch_normalization(
                state_data_observed,
                m_flat,
                self.state_ftypes,
            )
            state_data_norm_flat = torch.cat(state_list, dim=1)
            state_data_norm = state_data_norm_flat.view(B, max_path, -1)

        return state_data_norm, state_normalization_params
    
    def update_memory(
        self,
        z_n: torch.Tensor,     # [B, z_dim]
        v_n: torch.Tensor,     # [B, v_dim] one-hot
        t_n: torch.Tensor,     # [B, 1]
        h_prev: torch.Tensor,  # [B, h_dim]
    ) -> torch.Tensor:
        """
        One GRU update step:
            h_n = GRU([phi_v(t_n, v_n), phi_z(z_n)], h_prev)

        All inputs are batched over B.
        Returns:
            h_n: [B, h_dim]
        """
        # Feature transforms
        phi_v_n = self.phi_v(torch.cat([t_n, v_n], dim=-1))   # [B, h_dim]
        phi_z_n = self.phi_z(z_n)                             # [B, h_dim]

        rnn_in = torch.cat([phi_v_n, phi_z_n], dim=-1).unsqueeze(1)   # [B,1,2*h_dim]
        h_prev_rnn = h_prev.unsqueeze(0)                               # [1,B,h_dim]

        _, h_n = self.memory_rnn(rnn_in, h_prev_rnn)                   # h_n: [1,B,h_dim]
        return h_n.squeeze(0)                                          # [B,h_dim]

    def encode(
        self,
        state_data_norm: torch.Tensor,   # [batch_size, max_path, state_x_dim]
        v: torch.Tensor,                 # [batch_size, max_path, v_dim] one-hot states
        t: torch.Tensor,                 # [batch_size, max_path, 1]
        h0: torch.Tensor,                # [batch_size, h_dim]
        s: torch.Tensor,                 # [batch_size, s_dim]
    ):
        batch_size, max_path, _ = state_data_norm.shape

        # GRU hidden state has shape [num_layers=1, batch_size, h_dim]
        h = h0.unsqueeze(0)              # [1, batch_size, h_dim]
        s_expanded = s.unsqueeze(1).expand(batch_size, max_path, -1)  # [B, T, s_dim]
        mean_q_list, logvar_q_list = [], []
        mean_p_list, logvar_p_list = [], []
        z_list, h_list = [], []

        for n in range(max_path):
            x_n = state_data_norm[:, n, :]   # [batch_size, state_x_dim]
            v_n = v[:, n, :]                 # [batch_size, v_dim]
            t_n = t[:, n, :]            # [batch_size, 1]
            h_prev = h[-1]                   # [batch_size, h_dim] = h_{n-1}

            # store h_{n-1} for visit n
            h_list.append(h_prev)

            # Prior: p(z_n | t_n, v_n, h_{n-1}, s)
            prior_in = torch.cat([t_n, v_n, h_prev, s_expanded[:, n, :]], dim=-1)
            prior_params = self.z_n_prior(prior_in)
            mean_p, logvar_p = torch.chunk(prior_params, 2, dim=-1)
            logvar_p = torch.clamp(logvar_p, -15.0, 15.0)

            # Posterior: q(z_n | x_n, t_n, v_n, h_{n-1})
            post_in = torch.cat(
                [x_n, t_n, v_n, h_prev, s_expanded[:, n, :]],
                dim=-1
            )  # [B, state_x_dim+1+v_dim+h_dim+s_dim]
            post_params = self.z_n_posterior(post_in)
            mean_q, logvar_q = torch.chunk(post_params, 2, dim=-1)
            logvar_q = torch.clamp(logvar_q, -15.0, 15.0)

            # Reparameterized sample z_n
            eps = torch.randn_like(mean_q)
            z_n = mean_q + torch.exp(0.5 * logvar_q) * eps  # [batch_size, z_dim]

            # Store params and sample
            mean_q_list.append(mean_q)
            logvar_q_list.append(logvar_q)
            mean_p_list.append(mean_p)
            logvar_p_list.append(logvar_p)
            z_list.append(z_n)

            # RNN update (belief state): h_n = GRU([phi_v(v_n, t_n), phi_z(z_n)], h_{n-1})
            h = self.update_memory(z_n=z_n, v_n=v_n, t_n=t_n, h_prev=h_prev).unsqueeze(0)

        # Stack over time
        mean_q   = torch.stack(mean_q_list, dim=1)      # [batch_size, max_path, z_dim]
        logvar_q = torch.stack(logvar_q_list, dim=1)    # [batch_size, max_path, z_dim]
        mean_p   = torch.stack(mean_p_list, dim=1)      # [batch_size, max_path, z_dim]
        logvar_p = torch.stack(logvar_p_list, dim=1)    # [batch_size, max_path, z_dim]
        z_samples = torch.stack(z_list, dim=1)          # [batch_size, max_path, z_dim]
        h_seq     = torch.stack(h_list, dim=1)          # [batch_size, max_path, h_dim], h_seq[:, n, :] = h_{n-1}

        # NOTE: h_seq[:, n, :] = h_{n-1} (memory BEFORE observing visit n)
        # This means the decoder/MuStaRDT at position n sees the memory state
        # that does NOT yet incorporate information from visit n.

        q_params_state = {"z_n": (mean_q, logvar_q)}
        p_params_state = {"z_n": (mean_p, logvar_p)}
        samples_state  = {"z_n": z_samples}

        return q_params_state, p_params_state, samples_state, h_seq
    

    def decode(
        self,
        samples: Dict[str, torch.Tensor],   # expects "z_n" and "s"
        x: torch.Tensor,                    # [batch_size, max_path, state_x_dim]
        m_state_feat: torch.Tensor,         # [batch_size, max_path, n_state_features]
        h_seq: torch.Tensor,                # [batch_size, max_path, h_dim]  (h_{n-1})
        state_normalization_params: List[Any],
        n_generated_sample: int = 1,
    ):
        B, T, _ = x.shape
        z = samples["z_n"]                    # [B,T,z_dim]
        z_flat = z.view(B*T, self.z_dim)
        h_flat = h_seq.view(B*T, self.h_dim)
        latent_in = torch.cat([z_flat, h_flat], dim=-1)

        s_rep = samples["s"].repeat_interleave(T, dim=0)  # [B*T, s_dim]
        x_flat = x.reshape(B*T, -1)
        m_flat = m_state_feat.reshape(B*T, len(self.state_ftypes))
        out = self.state_decoder(
            latent_in=latent_in,
            s=s_rep,
            x_raw=x_flat,
            miss=m_flat,
            normalization_params=state_normalization_params,
            n_generated_sample=n_generated_sample,
        )

        params_x_state        = out["params_x"]
        log_p_x_state         = out["log_p_x"]
        log_p_x_state_missing = out["log_p_x_missing"]
        samples_x_state_list  = out["samples_x_list"]

        state_samples_merged_flat = torch.cat(samples_x_state_list, dim=-1)  # [G, B*T, D_state]
        G, BT, D_state = state_samples_merged_flat.shape
        assert BT == B * T
        state_samples_merged = state_samples_merged_flat.view(G, B, T, D_state)

        p_params_state = {"x_state": params_x_state}
        samples["x_state"] = state_samples_merged

        log_p_state = {
            "log_p_x_state":         log_p_x_state,
            "log_p_x_state_missing": log_p_x_state_missing,
        }
        return p_params_state, log_p_state, samples


    def encode_deterministic(
        self,
        state_data_norm: torch.Tensor,   # [batch_size, max_path, state_x_dim]
        v: torch.Tensor,                 # [batch_size, max_path, v_dim] one-hot states
        t: torch.Tensor,                 # [batch_size, max_path, 1]
        h0: torch.Tensor,                # [batch_size, h_dim]
        s: torch.Tensor,                 # [batch_size, s_dim]
    ):
        """
        Deterministic visit-level encoder:
        - z_n: posterior means q(z_n | x_n, v_n, h_{n-1}),
        - h_seq: GRU memory updated with mean z_n (no noise).
        """
        batch_size, max_path, _ = state_data_norm.shape
        h = h0.unsqueeze(0)              # [1, B, h_dim]
        s_expanded = s.unsqueeze(1).expand(batch_size, max_path, -1)  # [B,T,s_dim]

        z_list = []
        h_list = []

        for n in range(max_path):
            x_n = state_data_norm[:, n, :]   # [batch_size, state_x_dim]
            v_n = v[:, n, :]                 # [batch_size, v_dim]
            t_n = t[:, n, :]                 # [batch_size, 1]
            h_prev = h[-1]                   # [batch_size, h_dim]

            h_list.append(h_prev)

            # Posterior: q(z_n | x_n, t_n, v_n, h_{n-1})
            post_in = torch.cat([x_n, t_n, v_n, h_prev, s_expanded[:, n, :]], dim=-1)
            post_params = self.z_n_posterior(post_in)       # [batch_size, 2*z_dim]
            mean_q, _ = torch.chunk(post_params, 2, dim=-1) # [batch_size, z_dim]

            z_n = mean_q
            z_list.append(z_n)

            # GRU update (belief state): h_n = GRU([phi_v(v_n, t_n), phi_z(z_n)], h_{n-1})
            h = self.update_memory(z_n=z_n, v_n=v_n, t_n=t_n, h_prev=h_prev).unsqueeze(0)

        z_det   = torch.stack(z_list, dim=1)   # [B, max_path, z_dim]
        h_seq   = torch.stack(h_list, dim=1)   # [B, max_path, h_dim]

        samples = {
            "z_n": z_det,
        }
        return samples, h_seq

    def elbo(
        self,
        log_p_x_state: torch.Tensor,               # [n_features, B*max_path]
        p_zn: tuple[torch.Tensor, torch.Tensor],   # (mean_pzn, logvar_pzn), [B, max_path, z_dim]
        q_zn: tuple[torch.Tensor, torch.Tensor],   # (mean_qzn, logvar_qzn), [B, max_path, z_dim]
        visit_mask: torch.Tensor,                  # [B, max_path], 1 = real visit, 0 = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state VRNN contributions to the ELBO:

            loss_re_state: E_q[log p(x_{1:T} | z_{1:T}, h_{0:T-1}, s)]   (per sample)
            KL_z_n:        sum_t KL(q(z_t|...) || p(z_t|...))            (per sample)

        Returns:
            loss_re_state: [B]
            KL_z_n:        [B]
        """
        n_features, batch_time = log_p_x_state.shape   # batch_time = B * max_path
        B, max_path = visit_mask.shape
        assert batch_time == B * max_path, "log_p_x_state shape inconsistent with visit_mask"

        # 1) Reconstruction term for x_state
        # reshape to [n_features, B, max_path]
        log_p_x_state_reshaped = log_p_x_state.view(n_features, B, max_path)
        visit_mask_b = visit_mask.view(1, B, max_path)  # broadcast over features

        # sum over features and time, masking padding
        loss_re_state = (log_p_x_state_reshaped * visit_mask_b).sum(dim=(0, 2))  # [B]

        # 2) KL for z_n: KL(q(z_n|...) || p(z_n|...)), summed over time with mask
        mean_pzn, logvar_pzn = p_zn      # [B, max_path, z_dim]
        mean_qzn, logvar_qzn = q_zn      # [B, max_path, z_dim]

        KL_z_n_steps = self._kl_gaussian(
            mean_qzn, logvar_qzn,
            mean_pzn, logvar_pzn,
            dim=-1,
        )  # [B, max_path]

        KL_z_n_steps = KL_z_n_steps * visit_mask  # mask padding
        KL_z_n = KL_z_n_steps.sum(dim=1)          # [B]

        return loss_re_state, KL_z_n


  
    def split_features(self, x):
        data_list = []
        idx = 0
        for feat in self.state_ftypes:
            width = int(feat['nclass'])
            data_list.append(x[:, idx: idx + width])
            idx += width
        return data_list


    @staticmethod
    def _kl_gaussian(mu_q, logvar_q, mu_p, logvar_p, dim: int = -1) -> torch.Tensor:
        """
        KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ) summed over dim.
        All inputs: same shape.
        """
        return 0.5 * torch.sum(
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
            - 1.0,
            dim=dim,
        )

    def forward(
        self,
        x_state: torch.Tensor,        # [B, max_path, state_x_dim]
        m_state_feat: torch.Tensor,   # [B, max_path, n_state_features]
        v: torch.Tensor,              # [B, max_path, v_dim] one-hot states
        t: torch.Tensor,              # [B, max_path, 1]
        h0: torch.Tensor,             # [B, h_dim]
        s: torch.Tensor,              # [B, s_dim]
        n_generated_sample: int = 1,
    ) -> dict[str, Any]:
        """
        Full state VRNN block:
        - normalize x_state,
        - q(z_n | x_n, t_n, v_n, h_{n-1}, s),
        - p(z_n | t_n, v_n, h_{n-1}, s),
        - p(x_n | z_n, h_{n-1}, s).
        """
        # 1) normalize
        state_data_norm, state_norm_params = self.normalize(
            x_state=x_state,
            m_state_feat=m_state_feat,
        )

        # 2) encode (VRNN)
        q_params_state, p_params_state_z, samples_state, h_seq = self.encode(
            state_data_norm=state_data_norm,
            v=v,
            t=t,
            h0=h0,
            s=s,
        )

        # 3) decode x_n
        # Build a minimal samples dict for decode: need z_n and s
        samples_for_dec = {
            "z_n": samples_state["z_n"],   # [B, max_path, z_dim]
            "s":   s,                      # [B, s_dim]
        }

        p_params_x, log_p_state, samples_dec = self.decode(
            samples=samples_for_dec,
            x=x_state,
            m_state_feat=m_state_feat,
            h_seq=h_seq,
            state_normalization_params=state_norm_params,
            n_generated_sample=n_generated_sample,
        )

        p_params = {
            "x_state": p_params_x["x_state"],
            "z_n":     p_params_state_z["z_n"],     # (mean_pzn, logvar_pzn)
        }
        q_params = {
            "z_n": q_params_state["z_n"],          # (mean_qzn, logvar_qzn)
        }

        samples_out = {
            "z_n":     samples_state["z_n"],        # [B, max_path, z_dim]
            "x_state": samples_dec["x_state"],      # [G, B, max_path, state_x_dim]
        }

        return {
            "state_data_norm": state_data_norm,
            "state_norm_params": state_norm_params,
            "q_params": q_params,
            "p_params": p_params,
            "log_p": log_p_state,     # log_p_x_state, log_p_x_state_missing
            "samples": samples_out,
            "h_seq": h_seq,           # [B, max_path, h_dim], h_seq[:, n, :] = h_{n-1}
        }

    def forward_deterministic(
        self,
        x_state: torch.Tensor,        # [B, max_path, state_x_dim]
        m_state_feat: torch.Tensor,   # [B, max_path, n_state_features]
        v: torch.Tensor,              # [B, max_path, v_dim]
        t: torch.Tensor,              # [B, max_path, 1]
        h0: torch.Tensor,             # [B, h_dim]
        s: torch.Tensor,              # [B, s_dim]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Deterministic state encoder:
        - normalize,
        - encode_deterministic -> z_n, h_seq.
        Returns:
        state_data_norm, samples_det ({"z_n"}), h_seq_det
        """
        state_data_norm, _ = self.normalize(
            x_state=x_state,
            m_state_feat=m_state_feat,
        )
        samples_det, h_seq_det = self.encode_deterministic(
            state_data_norm=state_data_norm,
            v=v,
            t=t,
            h0=h0,
            s=s,
        )
        return state_data_norm, samples_det, h_seq_det

    @torch.no_grad()
    def sample_state_cov(
        self,
        h_prev: torch.Tensor,        # [B, h_dim] = h_{n-1}
        v_next_idx: torch.Tensor,    # [B] int indices of next state (0..v_dim-1)
        t_next: torch.Tensor,        # [B, 1] time for next visit
        s: torch.Tensor,             # [B, s_dim]
        n_generated_sample: int = 1,
        deterministic_generation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample next-visit latent and covariates from the generative prior.
        Does not update the RNN memory (use update_memory).

        Samples from 
            1. z_n ~ p(z_n | h_{n-1}, v_n, t_n, s)
            2. x_n ~ p(x_n | z_n, h_{n-1}, s)
        """
        device = h_prev.device
        B = h_prev.size(0)

        # 1) One-hot v_n
        v_next = torch.zeros(B, self.v_dim, device=device)
        v_next[torch.arange(B, device=device), v_next_idx] = 1.0  # [B, v_dim]

        # 2) Prior p(z_n | t_n, v_n, h_{n-1}, s)
        prior_in = torch.cat([t_next, v_next, h_prev, s], dim=-1)
        prior_params = self.z_n_prior(prior_in)
        mean_p, logvar_p = torch.chunk(prior_params, 2, dim=-1)
        logvar_p = torch.clamp(logvar_p, -15.0, 15.0)

        if deterministic_generation:
            z_next = mean_p
        else:
            eps = torch.randn_like(mean_p)
            z_next = mean_p + torch.exp(0.5 * logvar_p) * eps

        # 3) Sample x_n ~ p(x_n | z_n, h_{n-1}, s) via state_decoder
        latent_in = torch.cat([z_next, h_prev], dim=-1)             # [B, z_dim + h_dim]

        # Dummy raw x and full-observed mask (we only care about samples, not log p)
        x_dummy = torch.zeros(B, self.state_x_dim, device=device)
        m_state_flat = torch.ones(B, len(self.state_ftypes), device=device)

        assert self.state_norm_params is not None, "state_norm_params must be set on StateVRNN."

        out = self.state_decoder(
            latent_in=latent_in,
            s=s,                                 # [B, s_dim]
            x_raw=x_dummy,
            miss=m_state_flat,
            normalization_params=self.state_norm_params,
            n_generated_sample=n_generated_sample,
        )

        samples_x_state_list = out["samples_x_list"]                # list[G, B, width_i]
        state_samples_merged_flat = torch.cat(samples_x_state_list, dim=-1)  # [G, B, state_x_dim]
        G, B_check, D_state = state_samples_merged_flat.shape
        assert B_check == B
        x_next_raw = state_samples_merged_flat[0]                   # [B, state_x_dim]

        # 4) Normalize x_next_raw so phi_x sees the same scale as during training
        x_next_raw_seq     = x_next_raw.unsqueeze(1)                # [B,1,state_x_dim]
        m_state_feat_seq   = m_state_flat.unsqueeze(1)              # [B,1,n_state_features]
        x_next_norm_seq, _ = self.normalize(x_next_raw_seq, m_state_feat_seq)
        x_next_norm = x_next_norm_seq[:, 0, :]                      # [B, state_x_dim]

        return {
            "x_next_norm": x_next_norm,  # [B, state_x_dim] (normalized)
            "x_next_raw":  x_next_raw,
            "z_next":      z_next,       # [B, z_dim]
        }

    @torch.no_grad()
    def decode_state_sample(
        self,
        z_n: torch.Tensor,                        # [N, max_path, z_dim]
        h_seq: torch.Tensor,                      # [N, max_path, h_dim] (h_{n-1})
        s_samples: torch.Tensor,                  # [N, s_dim]
        n_generated_sample: int = 1,
    ) -> torch.Tensor:
        """
        Sample state covariates x_n ~ p(x_n | z_n, h_{n-1}, s) for all visits
        using StateVRNN.decode.
        """
        N, max_path, _ = z_n.shape
        device = z_n.device

        # Dummy raw x and mask (all observed)
        x_dummy = torch.zeros(N, max_path, self.state_x_dim, device=device)
        m_dummy = torch.ones(N, max_path, len(self.state_ftypes), device=device)

        # Build samples dict expected by StateVRNN.decode
        samples = {
            "s":   s_samples,        # [N, s_dim]
            "z_n": z_n,              # [N, max_path, z_dim]
        }

        # Use stored normalization params from training
        state_norm_params = self.state_norm_params
        assert state_norm_params is not None, "state_norm_params must be set on StateVRNN."

        # Decode
        _, _, samples_out = self.decode(
            samples=samples,
            x=x_dummy,
            m_state_feat=m_dummy,
            h_seq=h_seq,
            state_normalization_params=state_norm_params,
            n_generated_sample=n_generated_sample,
        )

        # samples_out["x_state"]: [G, N, max_path, state_x_dim]
        return samples_out["x_state"]


    def generate_first_visit(
        self,
        state_idx: torch.Tensor,  # [N]
        h0: torch.Tensor,         # [N, h_dim]
        s: torch.Tensor,          # [N, s_dim]
    ) -> Dict[str, torch.Tensor]:
        """Sample z_1, x_1 ~ p(· | h_0, v_1, t_1=0, s) for the first visit."""
        device = h0.device
        N = h0.size(0)

        v_onehot = F.one_hot(state_idx, num_classes=self.v_dim).float()
        t0 = torch.zeros(N, 1, device=device)

        prior_in = torch.cat([t0, v_onehot, h0, s], dim=-1)
        z_params = self.z_n_prior(prior_in)
        mean_p, logvar_p = torch.chunk(z_params, 2, dim=-1)
        logvar_p = torch.clamp(logvar_p, -15.0, 15.0)
        z_curr = mean_p + torch.exp(0.5 * logvar_p) * torch.randn_like(mean_p)

        x_state_samples = self.decode_state_sample(
            z_n=z_curr.unsqueeze(1),
            h_seq=h0.unsqueeze(1),
            s_samples=s,
        )
        x_curr_raw = x_state_samples[0, :, 0, :]

        m_state_feat_seq = torch.ones(N, 1, len(self.state_ftypes), device=device)
        x_norm_seq, _ = self.normalize(x_curr_raw.unsqueeze(1), m_state_feat_seq)

        return {
            "z_curr": z_curr,
            "x_curr_raw": x_curr_raw,
            "x_curr_norm": x_norm_seq[:, 0, :],
            "m_state_feat_curr": m_state_feat_seq[:, 0, :],
        }

    @torch.no_grad()
    def sample_z(
        self,
        h_prev: torch.Tensor,
        v_next_idx: torch.Tensor,
        t_next: torch.Tensor,
        s: torch.Tensor,
        deterministic_generation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.prior_z(
            h_prev=h_prev,
            v_next_idx=v_next_idx,
            t_next=t_next,
            s=s,
            deterministic=deterministic_generation,
        )

    def prior_z(
        self,
        h_prev: torch.Tensor,        # [B, h_dim]
        v_next_idx: torch.Tensor,    # [B]
        t_next: torch.Tensor,        # [B, 1]
        s: torch.Tensor,             # [B, s_dim]
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Differentiable prior sample/mean from p(z_n | t_n, v_n, h_{n-1}, s).
        """
        v_next = F.one_hot(v_next_idx.long(), num_classes=self.v_dim).float()

        prior_in = torch.cat([t_next, v_next, h_prev, s], dim=-1)
        prior_params = self.z_n_prior(prior_in)
        mean_p, logvar_p = torch.chunk(prior_params, 2, dim=-1)
        logvar_p = torch.clamp(logvar_p, -15.0, 15.0)

        if deterministic:
            z_next = mean_p
        else:
            eps = torch.randn_like(mean_p)
            z_next = mean_p + torch.exp(0.5 * logvar_p) * eps

        return {
            "mean": mean_p,
            "logvar": logvar_p,
            "z_next": z_next,
        }