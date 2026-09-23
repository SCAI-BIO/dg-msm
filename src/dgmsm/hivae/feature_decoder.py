import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from .feature_base import FeatureBase

class FeatureDecoder(nn.Module):
    """
    Generic HIVAE-style decoder block:
    """
    def __init__(
        self,
        feat_types: List[Dict[str, Any]],
        y_dim: int,
        s_dim: int,
        y_in_dim: int,
    ) -> None:
        super().__init__()
        self.feat_types = feat_types
        self.s_dim = s_dim
        self.y_dim = y_dim

        self.n_features = len(self.feat_types)
        self.total_y_dim = self.y_dim * self.n_features

        # y-layer: maps latent_in -> concatenated y for all features
        self.y_layer = nn.Linear(y_in_dim, self.total_y_dim)

        # One FeatureBase subclass per feature, selected via registry
        self.feature_dists = nn.ModuleList(
            [
                FeatureBase.from_config(feat_config=feat, y_dim=y_dim, s_dim=s_dim)
                for feat in self.feat_types
            ]
        )

        # Precompute slices to split x_raw into per-feature blocks
        self._x_slices = []
        idx = 0
        for feat in self.feat_types:
            width = int(feat["nclass"])
            self._x_slices.append(slice(idx, idx + width))
            idx += width
                

    def forward(
        self,
        latent_in: torch.Tensor,             # [N, y_in_dim]
        s: torch.Tensor,                     # [N, s_dim]
        x_raw: torch.Tensor,                 # [N, x_dim] (concatenated features)
        miss: torch.Tensor,                  # [N, n_features] (1=observed, 0=missing)
        normalization_params: List[Any],
        n_generated_sample: int = 1,
    ) -> Dict[str, Any]:

        N = latent_in.size(0)

        # 1) y-layer: [N, y_in_dim] -> [N, total_y_dim]
        y_flat = self.y_layer(latent_in)  # [N, total_y_dim]

        # 2) partition y into per-feature slices of size y_dim
        grouped_y = [
            y_flat[:, i * self.y_dim : (i + 1) * self.y_dim]  # [N, y_dim]
            for i in range(self.n_features)
        ]

        params_x_list = []
        log_p_list = []
        log_p_missing_list = []
        samples_x_list = []

        # 3) loop over features and delegate to each FeatureBase module
        for i, dist in enumerate(self.feature_dists):
            y_i = grouped_y[i]                          # [N, y_dim]
            x_i = x_raw[:, self._x_slices[i]]          # [N, width_i]
            mask_i = miss[:, i]                        # [N]
            norm_i = normalization_params[i]           # (mean, var) or unused

            out_i = dist(
                y=y_i,
                s=s,
                x=x_i,
                mask=mask_i,
                norm_params=norm_i,
                n_generated_sample=n_generated_sample,
            )

            params_x_list.append(out_i["params"])
            log_p_list.append(out_i["log_p_x"])             # [N]
            log_p_missing_list.append(out_i["log_p_x_missing"])  # [N]
            samples_x_list.append(out_i["samples"])         # [G, N, width_i]

        # 4) stack log-likelihoods into [n_features, N]
        log_p_x = torch.stack(log_p_list, dim=0)              # [n_features, N]
        log_p_x_missing = torch.stack(log_p_missing_list, 0)  # [n_features, N]

        return {
            "params_x": params_x_list,        # list per feature
            "log_p_x": log_p_x,               # [n_features, N]
            "log_p_x_missing": log_p_x_missing,
            "samples_x_list": samples_x_list, # list[G, N, width_i]
        }

