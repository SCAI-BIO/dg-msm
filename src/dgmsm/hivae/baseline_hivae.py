import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from .normalization import (
    batch_normalization,
    apply_normalization_from_buffers,
)

from .feature_decoder import FeatureDecoder

class BaseHIVAE(nn.Module):
    def __init__(
        self,
        base_x_dim: int,
        base_ftypes: List[Dict[str, Any]],
        y_dim: int,
        s_dim: int,
        z_dim: int,
        base_norm_params: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        super().__init__()

        self.base_ftypes = base_ftypes
        self.base_x_dim = base_x_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.y_dim = y_dim

        # Gaussian mixture model for the z_0 posterior
        # learnable logits for p(s), initial uniform (all zeros)
        self.s_prior_logits = nn.Parameter(torch.zeros(self.s_dim), requires_grad=True)
        self.s_posterior = nn.Linear(self.base_x_dim, self.s_dim)                       # s ~ q(s | x_0)
        z0_prior_in_dim = self.s_dim
        z0_post_in_dim  = self.base_x_dim + self.s_dim

        self.z_0_prior = nn.Sequential(
            nn.Linear(z0_prior_in_dim, z0_prior_in_dim),
            nn.ReLU(),
            nn.Linear(z0_prior_in_dim, 2 * self.z_dim),
        )# z_0 ~ p(z_0 | s)

        self.z_0_posterior = nn.Sequential(
            nn.Linear(z0_post_in_dim, z0_post_in_dim),
            nn.ReLU(),
            nn.Linear(z0_post_in_dim, 2 * self.z_dim),
        ) # z_0 ~ q(z_0 | s, x_0)


        self.base_decoder = FeatureDecoder(
            feat_types=self.base_ftypes,
            y_dim=self.y_dim,           # per-feature y-dim
            s_dim=self.s_dim,           # subtype dimension
            y_in_dim=self.z_dim,        # latent_in is z_0
        )

        # store normalization params (optional, e.g. from train set)
        self.base_norm_params = base_norm_params
        if base_norm_params is not None:
            means = torch.stack([m for (m, _) in base_norm_params])  # [n_base_features]
            vars_ = torch.stack([v for (_, v) in base_norm_params])  # [n_base_features]
            self.register_buffer("base_norm_mean", means)
            self.register_buffer("base_norm_var", vars_)
        else:
            self.base_norm_mean = None
            self.base_norm_var  = None

    def normalize(
        self,
        x_base: torch.Tensor,  # [B, base_x_dim]
        m_base: torch.Tensor,  # [B, n_base_features]
    ) -> tuple[torch.Tensor, list[Any]]:
        """
        Split, mask, and normalize baseline features.
        Returns:
            base_data_norm: [B, base_x_dim] (normalized),
            base_normalization_params: list of (mean,var) per feature.
        """
        B = x_base.shape[0]
        base_data = self.split_features(x=x_base)

        if self.base_norm_mean is not None and self.base_norm_var is not None:
            # use stored global stats
            base_list = apply_normalization_from_buffers(
                data_list=base_data,
                miss_list=m_base,
                feat_types_list=self.base_ftypes,
                mean_buf=self.base_norm_mean,
                var_buf=self.base_norm_var,
            )
            base_data_norm = torch.cat(base_list, dim=1)
            base_normalization_params = self.base_norm_params
        else:
            # compute batch normalization stats on the fly
            base_data_observed = [
                d * m_base[:, i].view(B, 1)
                for i, d in enumerate(base_data)
            ]
            base_list, base_normalization_params = batch_normalization(
                base_data_observed,
                m_base,
                self.base_ftypes,
            )
            base_data_norm = torch.cat(base_list, dim=1)

        return base_data_norm, base_normalization_params

    def encode(
        self,
        base_data_norm: torch.Tensor,   # [batch_size, base_x_dim]
        tau: float,
    ):
        # q(s | x_0): categorical with Gumbel-Softmax relaxation
        logits_s = self.s_posterior(base_data_norm)          # [batch_size, s_dim]

        # relaxed sample of s
        samples_s = F.gumbel_softmax(
            logits_s, tau=tau, hard=False, dim=-1
        )  # [batch_size, s_dim]

        # q(z_0 | s, x_0): Gaussian
        z0_in = torch.cat([base_data_norm, samples_s], dim=-1)  # [batch_size, base_x_dim + s_dim]
        z0_params = self.z_0_posterior(z0_in)                   # [batch_size, 2*z_dim]

        mean_qz0, logvar_qz0 = torch.chunk(z0_params, 2, dim=-1)  # each [batch_size, z_dim]
        logvar_qz0 = torch.clamp(logvar_qz0, -15.0, 15.0)

        eps = torch.randn_like(mean_qz0)
        samples_z0 = mean_qz0 + torch.exp(0.5 * logvar_qz0) * eps  # [batch_size, z_dim]

        q_params = {
            "s":   logits_s,             # logits for q(s | x_0)
            "z_0": (mean_qz0, logvar_qz0),  # params of q(z_0 | s, x_0)
        }

        samples = {
            "s":   samples_s,   # relaxed one-hot
            "z_0": samples_z0,
        }
        return q_params, samples
    
    def encode_deterministic(
        self,
        base_data_norm: torch.Tensor,   # [batch_size, base_x_dim]
    ):
        """
        Deterministic baseline encoder:
        - s: posterior softmax (no Gumbel noise),
        - z_0: posterior mean of q(z_0 | s, x_0).
        """
        # q(s | x_0): logits_s, use softmax as "sample"
        logits_s = self.s_posterior(base_data_norm)          # [B, s_dim]
        s_det    = F.softmax(logits_s, dim=-1)               # [B, s_dim]

        # q(z_0 | s, x_0): use mean
        z0_in    = torch.cat([base_data_norm, s_det], dim=-1)
        z0_params = self.z_0_posterior(z0_in)                # [B, 2*z_dim]
        mean_qz0, _ = torch.chunk(z0_params, 2, dim=-1)      # [B, z_dim]

        samples = {
            "s":   s_det,   # [B, s_dim]
            "z_0": mean_qz0,
        }
        return samples

    def decode(
        self,
        samples: Dict[str, torch.Tensor],
        x_base: torch.Tensor,                 # [B, base_x_dim]
        m_base: torch.Tensor,                 # [B, n_base_features]
        base_normalization_params: List[Any],
        n_generated_sample: int = 1,
    ):
        """
        Baseline decoder: p(x_0 | z_0, s) via FeatureDecoder.
        """

        z_0 = samples["z_0"]          # [B, z_dim]
        s   = samples["s"]            # [B, s_dim]

        out = self.base_decoder(
            latent_in=z_0,
            s=s,
            x_raw=x_base,
            miss=m_base,
            normalization_params=base_normalization_params,
            n_generated_sample=n_generated_sample,
        )

        params_x_base    = out["params_x"]
        log_p_x_base     = out["log_p_x"]
        log_p_x_base_missing = out["log_p_x_missing"]
        samples_x_list   = out["samples_x_list"]

        p_params_base = {"x_base": params_x_base}
        samples["x_base"] = samples_x_list

        log_p_base = {
            "log_p_x_base":         log_p_x_base,
            "log_p_x_base_missing": log_p_x_base_missing,
        }

        return p_params_base, log_p_base, samples

    def elbo(
        self,
        log_p_x_base: torch.Tensor,        # [F_base, batch_size]
        p_z0: tuple[torch.Tensor, torch.Tensor],  # (mean_pz0, logvar_pz0), [B, z_dim]
        q_s_logits: torch.Tensor,          # logits for q(s|x_0), [B, s_dim]
        q_z0: tuple[torch.Tensor, torch.Tensor],  # (mean_qz0, logvar_qz0), [B, z_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute baseline HIVAE contributions to the ELBO:

            loss_re_base:   E_q[log p(x_0 | z_0, s)]   (per sample)
            KL_z0:          KL(q(z_0|s,x_0) || p(z_0|s))  (per sample)
            KL_s:           KL(q(s|x_0) || p(s))          (scalar, averaged over batch)

        Returns:
            loss_re_base: [batch_size]
            KL_z0:        [batch_size]
            KL_s:         scalar
        """
        # 1) Reconstruction term for x_0
        # log_p_x_base: [F_base, B] (sum over features)
        loss_re_base = log_p_x_base.sum(dim=0)  # [batch_size]

        # 2) KL for s: KL(q(s|x_0) || p(s))
        log_pi = q_s_logits                      # [batch_size, s_dim]
        q = F.softmax(log_pi, dim=-1)           # q(s|x_0)
        log_q = F.log_softmax(log_pi, dim=-1)   # log q(s|x_0)

        log_p_s = F.log_softmax(self.s_prior_logits, dim=-1)  # [s_dim]
        log_p_s = log_p_s.unsqueeze(0)                   # [1, s_dim] broadcast

        # per-sample KL_s
        KL_s = torch.sum(q * (log_q - log_p_s), dim=-1)  # [batch_size]
        #KL_s = KL_s_per_sample.mean()                               # scalar

        # 3) KL for z_0: KL(q(z_0|s,x_0) || p(z_0|s))
        mean_pz0, logvar_pz0 = p_z0      # [batch_size, z_dim]
        mean_qz0, logvar_qz0 = q_z0      # [batch_size, z_dim]

        KL_z0 = self._kl_gaussian(
            mean_qz0, logvar_qz0,
            mean_pz0, logvar_pz0,
            dim=-1,
        )  # [batch_size]
        return loss_re_base, KL_z0, KL_s


    def split_features(self, x):
        data_list = []
        idx = 0
        for feat in self.base_ftypes:
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
        x_base: torch.Tensor,
        m_base: torch.Tensor,
        tau: float = 1.0,
        n_generated_sample: int = 1,
    ) -> dict[str, Any]:
        """
        Full baseline block:
          - normalize x_base,
          - q(s|x0), q(z0|s,x0),
          - p(z0|s),
          - p(x0|z0,s).
        """
        # 1) normalize
        base_data_norm, base_norm_params = self.normalize(
            x_base=x_base,
            m_base=m_base,
        )

        # 2) encode: q(s|x0), q(z0|s,x0)
        q_params, samples = self.encode(
            base_data_norm=base_data_norm,
            tau=tau,
        )

        # 3) prior p(z0 | s)
        mean_pz0, logvar_pz0 = torch.chunk(
            self.z_0_prior(samples["s"]), 2, dim=1
        )
        logvar_pz0 = torch.clamp(logvar_pz0, -15.0, 15.0)

        # 4) decode: p(x0 | z0, s)
        p_params_x, log_p_x, samples = self.decode(
            samples=samples,
            x_base=x_base,
            m_base=m_base,
            base_normalization_params=base_norm_params,
            n_generated_sample=n_generated_sample,
        )

        p_params = {
            "x_base": p_params_x["x_base"],
            "z_0":    (mean_pz0, logvar_pz0),
        }

        return {
            "base_data_norm": base_data_norm,
            "base_norm_params": base_norm_params,
            "q_params": q_params,   # "s", "z_0"
            "p_params": p_params,   # "x_base", "z_0"
            "log_p": log_p_x,       # log_p_x_base, log_p_x_base_missing
            "samples": samples,     # includes s, z_0, x_base samples
        }

    @torch.no_grad()
    def forward_deterministic(
        self,
        x_base: torch.Tensor,  # [B, base_x_dim]
        m_base: torch.Tensor,  # [B, n_base_features]
    ) -> dict[str, Any]:
        """
        Deterministic baseline block:
          - normalize x_base,
          - s_det = softmax q(s|x0),
          - z0_det = mean q(z0 | s_det, x0).

        Returns:
            {
                "base_data_norm": [B, base_x_dim],
                "samples": {
                    "s":   [B, s_dim],
                    "z_0": [B, z_dim],
                },
            }
        """
        # 1) normalize
        base_data_norm, _ = self.normalize(
            x_base=x_base,
            m_base=m_base,
        )

        # 2) deterministic encoding
        samples_det = self.encode_deterministic(base_data_norm)

        return {
            "base_data_norm": base_data_norm,
            "samples": samples_det,
        }


    @torch.no_grad()
    def sample_from_prior(
        self,
        n_samples: int,
        device: torch.device,
    ):
        """
        Sample from the baseline prior:
          s ~ p(s), z0 ~ p(z0|s), x0 ~ p(x0|z0,s),
        and return x0 and its normalized version.
        """
        assert self.base_norm_params is not None, "sample_from_prior requires stored base_norm_params"
        s_dim = self.s_dim

        # 1) s ~ p(s)
        log_ps = self.s_prior_logits.to(device)      # [s_dim]
        ps     = F.softmax(log_ps, dim=-1)
        s_idx  = torch.multinomial(ps, n_samples, replacement=True)     # [N]
        s_samples = F.one_hot(s_idx, num_classes=s_dim).float()         # [N,s_dim]

        # 2) z0 ~ p(z0|s)
        z0_params = self.z_0_prior(s_samples)        # [N, 2*z_dim]
        mean_pz0, logvar_pz0 = torch.chunk(z0_params, 2, dim=-1)
        logvar_pz0 = torch.clamp(logvar_pz0, -15.0, 15.0)
        eps0 = torch.randn_like(mean_pz0)
        z0  = mean_pz0 + torch.exp(0.5 * logvar_pz0) * eps0

        # 3) x0 ~ p(x0|z0,s)
        samples = {"s": s_samples, "z_0": z0}
        x_dummy = torch.zeros(n_samples, self.base_x_dim, device=device)
        m_dummy = torch.ones(n_samples, len(self.base_ftypes), device=device)
        base_norm_params = self.base_norm_params

        _, _, samples = self.decode(
            samples=samples,
            x_base=x_dummy,
            m_base=m_dummy,
            base_normalization_params=base_norm_params,
            n_generated_sample=1,
        )
        x_base_list = samples["x_base"]
        x_base = torch.cat(x_base_list, dim=-1)[0]            # [N, base_x_dim]

        # 4) normalize x0 if needed
        base_data_norm, _ = self.normalize(x_base=x_base, m_base=m_dummy)

        return {
            "s":      s_samples,            # [N,s_dim]
            "z_0":    z0,                   # [N,z_dim]
            "x_base": x_base,               # [N,base_x_dim]
            "x_base_norm": base_data_norm,  # [N,base_x_dim]
        }