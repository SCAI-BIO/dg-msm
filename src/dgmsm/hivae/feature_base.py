import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math
from torch.distributions import Normal, Categorical


class FeatureBase(nn.Module):
    """
    Base class for a single feature's distribution:
      - builds its own θ-network(s),
      - computes θ from (y, s, mask),
      - computes log-likelihood & samples from (x, θ, normalization_params).

    Subclasses must implement forward().
    """    
    _registry: Dict[str, type["FeatureBase"]] = {}
    def __init__(
        self,
        feat_config: Dict[str, Any],  # e.g. {"type": "real", "nclass": 1, ...}
        y_dim: int,
        s_dim: int,
    ) -> None:
        super().__init__()
        self.feat_config = feat_config
        self.ftype = feat_config["type"]
        self.nclass = int(feat_config["nclass"])
        self.y_dim = y_dim
        self.s_dim = s_dim
    
    @classmethod
    def register(cls, ftype: str):
        """
        Decorator to register a FeatureBase subclass for a given feature type string.
        """
        def decorator(subcls: "FeatureBase"):
            cls._registry[ftype] = subcls
            return subcls
        return decorator

    @classmethod
    def from_config(
        cls,
        feat_config: Dict[str, Any],
        y_dim: int,
        s_dim: int,
    ) -> "FeatureBase":
        """
        Factory: returns an instance of the appropriate subclass based on feat_config["type"].
        """
        ftype = feat_config["type"]
        if ftype not in cls._registry:
            raise ValueError(f"Unknown feature type {ftype!r} for FeatureBase.")
        subcls = cls._registry[ftype]
        return subcls(feat_config, y_dim, s_dim)

    @staticmethod
    def _observed_data_layer(
        observed_data: torch.Tensor,
        missing_data: torch.Tensor,
        condition_indices,
        layer: nn.Module,
    ) -> torch.Tensor:
        """
        Apply `layer` to observed data with grad, to missing data without grad,
        and combine back in original order.
        """
        obs_output = layer(observed_data)
        with torch.no_grad():
            miss_output = layer(missing_data)

        output = torch.empty_like(torch.cat([miss_output, obs_output], dim=0))
        output[condition_indices[0]] = miss_output  # missing indices
        output[condition_indices[1]] = obs_output   # observed indices
        return output

    def forward(
        self,
        y: torch.Tensor,                 # [N, y_dim]
        s: torch.Tensor,                 # [N, s_dim]
        x: torch.Tensor,                 # [N, width]
        mask: torch.Tensor,              # [N] (1=observed, 0=missing)
        norm_params: Tuple[torch.Tensor, torch.Tensor],
        n_generated_sample: int = 1,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@FeatureBase.register("real")
class RealFeature(FeatureBase):
    def __init__(self, feat_config: Dict[str, Any], y_dim: int, s_dim: int):
        super().__init__(feat_config, y_dim, s_dim)
        self.mean_layer  = nn.Linear(y_dim + s_dim, 1, bias=False)
        self.sigma_layer = nn.Linear(s_dim, 1, bias=False)
        
    def forward(
        self,
        y: torch.Tensor,                 # [N, y_dim]
        s: torch.Tensor,                 # [N, s_dim]
        x: torch.Tensor,                 # [N, 1]
        mask: torch.Tensor,              # [N]
        norm_params: Tuple[torch.Tensor, torch.Tensor],
        n_generated_sample: int = 1,
    ) -> Dict[str, torch.Tensor]:

        mask = mask.bool()
        observed_y, missing_y = y[mask], y[~mask]
        observed_s, missing_s = s[mask], s[~mask]
        condition_indices = [~mask, mask]

        # θ: mean(y,s), sigma(s)
        est_mean = self._observed_data_layer(
            torch.cat([observed_y, observed_s], dim=1),
            torch.cat([missing_y, missing_s], dim=1),
            condition_indices,
            layer=self.mean_layer,
        )  # [N,1]

        est_var = self._observed_data_layer(
            observed_s,
            missing_s,
            condition_indices,
            layer=self.sigma_layer,
        )  # [N,1]

        # loglik_real logic
        epsilon = 1e-3
        est_var = torch.clamp(F.softplus(est_var), min=epsilon, max=10.0)
        est_mean = torch.clamp(est_mean, min=-5.0, max=5.0)

        data_mean, data_var = norm_params
        data_mean = data_mean.to(est_mean.device, est_mean.dtype)
        data_var  = data_var.to(est_mean.device, est_mean.dtype)

        est_mean = torch.sqrt(data_var) * est_mean + data_mean
        est_var  = data_var * est_var

        # Gaussian log-likelihood
        log_norm = -0.5 * math.log(2 * math.pi)
        log_var_term = -0.5 * torch.sum(torch.log(est_var), dim=1)
        log_exp = -0.5 * torch.sum((x - est_mean) ** 2 / est_var, dim=1)
        log_p_x_full = log_exp + log_var_term + log_norm  # [N]

        missing_mask = (~mask).float()
        obs_mask = mask.float()

        log_p_x = log_p_x_full * obs_mask
        log_p_x_missing = log_p_x_full * missing_mask

        samples = Normal(est_mean, torch.sqrt(est_var)).sample(
            sample_shape=(n_generated_sample,)
        )  # [G,N,1]

        return {
            "params": [est_mean, est_var],
            "log_p_x": log_p_x,
            "log_p_x_missing": log_p_x_missing,
            "samples": samples,
        }

@FeatureBase.register("cat")
class CatFeature(FeatureBase):
    def __init__(self, feat_config: Dict[str, Any], y_dim: int, s_dim: int):
        super().__init__(feat_config, y_dim, s_dim)
        n_class = self.nclass
        self.logits_layer = nn.Linear(y_dim + s_dim, n_class - 1, bias=False)

    def forward(
        self,
        y: torch.Tensor, s: torch.Tensor,
        x: torch.Tensor, mask: torch.Tensor,
        norm_params: Tuple[torch.Tensor, torch.Tensor],  # unused
        n_generated_sample: int = 1,
    ) -> Dict[str, torch.Tensor]:
        mask = mask.bool()
        observed_y, missing_y = y[mask], y[~mask]
        observed_s, missing_s = s[mask], s[~mask]
        condition_indices = [~mask, mask]

        h2_log_pi_partial = self._observed_data_layer(
            torch.cat([observed_y, observed_s], dim=1),
            torch.cat([missing_y, missing_s], dim=1),
            condition_indices,
            layer=self.logits_layer,
        )  # [N, n_class-1]

        zeros_col = torch.zeros(h2_log_pi_partial.shape[0], 1,
                                device=h2_log_pi_partial.device,
                                dtype=h2_log_pi_partial.dtype)
        logits = torch.cat([zeros_col, h2_log_pi_partial], dim=1)  # [N, n_class]

        target = x.argmax(dim=1)  # one-hot -> index
        log_p_x_full = -F.cross_entropy(logits, target, reduction="none")  # [N]

        obs_mask = mask.float()
        missing_mask = (~mask).float()
        log_p_x = log_p_x_full * obs_mask
        log_p_x_missing = log_p_x_full * missing_mask

        samples = F.one_hot(
            Categorical(logits=logits).sample(sample_shape=(n_generated_sample,)),
            num_classes=self.nclass,
        )  # [G,N,n_class]

        return {
            "params": logits,
            "log_p_x": log_p_x,
            "log_p_x_missing": log_p_x_missing,
            "samples": samples,
        }

@FeatureBase.register("pos")
class PosFeature(FeatureBase):
    """
    Log-normal model for positive real-valued features (type 'pos').
    """

    def __init__(self, feat_config: Dict[str, Any], y_dim: int, s_dim: int):
        super().__init__(feat_config, y_dim, s_dim)
        self.mean_layer  = nn.Linear(y_dim + s_dim, 1, bias=False)
        self.sigma_layer = nn.Linear(s_dim, 1, bias=False)

    def forward(
        self,
        y: torch.Tensor,                 # [N, y_dim]
        s: torch.Tensor,                 # [N, s_dim]
        x: torch.Tensor,                 # [N, 1] (original scale, >= 0)
        mask: torch.Tensor,              # [N] (1=observed, 0=missing)
        norm_params: Tuple[torch.Tensor, torch.Tensor],  # (mean_log, var_log) from normalization
        n_generated_sample: int = 1,
    ) -> Dict[str, torch.Tensor]:
        mask_bool = mask.bool()
        observed_y, missing_y = y[mask_bool], y[~mask_bool]
        observed_s, missing_s = s[mask_bool], s[~mask_bool]
        condition_indices = [~mask_bool, mask_bool]

        # θ in latent (log) space: mean(y,s), sigma(s)
        est_mean = self._observed_data_layer(
            torch.cat([observed_y, observed_s], dim=1),
            torch.cat([missing_y, missing_s], dim=1),
            condition_indices,
            layer=self.mean_layer,
        )  # [N,1]
        est_mean = torch.clamp(est_mean, min=-5.0, max=5.0)

        est_var = self._observed_data_layer(
            observed_s,
            missing_s,
            condition_indices,
            layer=self.sigma_layer,
        )  # [N,1]

        # Log-normal likelihood (same as your loglik_pos)
        epsilon = 1e-3

        # data_log = log1p(x)
        data = x
        data_log = torch.log1p(data)              # [N,1]
        missing_mask = (mask == 0).float()
        obs_mask = (mask != 0).float()

        # Ensure positivity of est_var
        est_var = F.softplus(est_var).clamp(min=epsilon, max=1.0)

        # Normalization params in log-space
        data_mean_log, data_var_log = norm_params
        data_mean_log = data_mean_log.to(est_mean.device, est_mean.dtype)
        data_var_log  = data_var_log.to(est_mean.device, est_mean.dtype)

        # Affine transformation back to normalized log-space
        est_mean = torch.sqrt(data_var_log) * est_mean + data_mean_log
        est_var  = data_var_log * est_var

        # Gaussian log-likelihood in log-space + Jacobian term
        log_p_x_full = (
            -0.5 * torch.sum((data_log - est_mean) ** 2 / est_var, dim=1)
            - 0.5 * torch.sum(torch.log(2 * math.pi * est_var), dim=1)
            - torch.sum(data_log, dim=1)
        )  # [N]

        log_p_x = log_p_x_full * obs_mask
        log_p_x_missing = log_p_x_full * missing_mask

        # Sample in log-space and transform back: exp(sample) - 1
        samples_pos_log = Normal(est_mean, torch.sqrt(est_var)).sample(
            sample_shape=(n_generated_sample,)
        )  # [G,N,1]
        samples_pos = torch.exp(samples_pos_log) - 1.0
        samples_pos = torch.clamp(samples_pos, min=0.0)

        return {
            "params": [est_mean, est_var],
            "log_p_x": log_p_x,
            "log_p_x_missing": log_p_x_missing,
            "samples": samples_pos,
        }

@FeatureBase.register("ord")
class OrdFeature(FeatureBase):
    """
    Ordinal likelihood using cumulative probabilities (type 'ord').

    x: thermometer/binary encoding [N, K] where K = nclass.
    """

    def __init__(self, feat_config: Dict[str, Any], y_dim: int, s_dim: int):
        super().__init__(feat_config, y_dim, s_dim)
        n_class = self.nclass
        self.theta_layer = nn.Linear(s_dim, n_class - 1, bias=False)
        # mean layer to shift thresholds
        self.mean_layer  = nn.Linear(y_dim + s_dim, 1, bias=False)

    def forward(
        self,
        y: torch.Tensor,                 # [N, y_dim]
        s: torch.Tensor,                 # [N, s_dim]
        x: torch.Tensor,                 # [N, K] thermometer encoding
        mask: torch.Tensor,              # [N] (1=observed, 0=missing)
        norm_params: Any,                # unused for ordinal
        n_generated_sample: int = 1,
    ) -> Dict[str, torch.Tensor]:
        epsilon = 1e-6
        N, K = x.shape
        assert K == self.nclass, "x width must equal nclass for ordinal feature."

        mask_bool = mask.bool()
        observed_y, missing_y = y[mask_bool], y[~mask_bool]
        observed_s, missing_s = s[mask_bool], s[~mask_bool]
        condition_indices = [~mask_bool, mask_bool]

        # θ partition parameters from s
        partition_param = self._observed_data_layer(
            observed_s,
            missing_s,
            condition_indices,
            layer=self.theta_layer,
        )  # [N, n_class-1]

        # mean shift from [y,s]
        mean_param = self._observed_data_layer(
            torch.cat([observed_y, observed_s], dim=1),
            torch.cat([missing_y, missing_s], dim=1),
            condition_indices,
            layer=self.mean_layer,
        )  # [N,1]

        # Same as your loglik_ord
        mean_value = mean_param.view(-1, 1)  # [N,1]

        theta_values = torch.cumsum(
            torch.clamp(F.softplus(partition_param), min=epsilon, max=1e7),
            dim=1,
        )  # [N,n_class-1]

        sigmoid_est_mean = torch.sigmoid(theta_values - mean_value)  # [N,n_class-1]

        batch_size = sigmoid_est_mean.shape[0]
        device = sigmoid_est_mean.device
        dtype = sigmoid_est_mean.dtype

        ones_col = torch.ones((batch_size, 1), device=device, dtype=dtype)
        zeros_col = torch.zeros((batch_size, 1), device=device, dtype=dtype)

        mean_probs = torch.cat([sigmoid_est_mean, ones_col], dim=1) - \
                     torch.cat([zeros_col, sigmoid_est_mean], dim=1)  # [N,n_class]

        mean_probs = torch.clamp(mean_probs, min=epsilon, max=1.0)

        # target class index from thermometer encoding
        nclass = self.nclass
        idx = x.sum(dim=1).long() - 1          # may be -1 for all-zero rows
        idx = idx.clamp(min=0, max=nclass - 1)

        missing_mask = (mask == 0).float()
        obs_mask     = (mask != 0).float()

        log_p_all = -F.cross_entropy(
            torch.log(mean_probs), idx, reduction="none"
        )  # [N]

        log_p_x         = log_p_all * obs_mask
        log_p_x_missing = log_p_all * missing_mask

        # Sample from ordinal distribution using probs
        sampled_values = Categorical(probs=mean_probs).sample(
            sample_shape=(n_generated_sample,)
        )  # [G,N]

        samples = []
        for g in range(n_generated_sample):
            # build thermometer encoding for each sampled category
            s_g = sampled_values[g]  # [N]
            samples.append(
                (
                    torch.arange(nclass, device=device)
                    .unsqueeze(0) < (s_g + 1).unsqueeze(1)
                ).float().unsqueeze(0)
            )
        samples = torch.cat(samples, dim=0)  # [G,N,nclass]

        return {
            "params": mean_probs,
            "log_p_x": log_p_x,
            "log_p_x_missing": log_p_x_missing,
            "samples": samples,
        }