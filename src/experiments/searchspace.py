from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Sequence, Dict, Any

import optuna
import numpy as np
import yaml

from dgmsm.config import ModelConfig


@dataclass
class SearchSpace:
    # ---- discrete time grid ----
    n_time_min: int = 50
    n_time_max: int = 100
    n_time_step: int = 10

    # ---- latent dimensions ----
    z_dim_min: int = 4
    z_dim_max: int = 10
    z_dim_step: int = 2

    s_dim_min: int = 4
    s_dim_max: int = 10
    s_dim_step: int = 2

    h_dim_min: int = 16 #old: 32  #new:16
    h_dim_max: int = 128 #old: 128 #new: 64
    h_dim_step: int = 8 #old: 32  #new:16

    y_dim_min: int = 6
    y_dim_max: int = 10
    y_dim_step: int = 2

    # ---- MSM  (state-specific MLP) ----
    msm_hidden_size_min: int  = 32  #old: 64  #new: 32
    msm_hidden_size_max: int  = 64 #old: 128 #new: 64
    msm_hidden_size_step: int = 8 #old: 32  #new:16

    msm_n_layers_min: int = 1
    msm_n_layers_max: int = 2
    msm_n_layers_step: int = 1

    f_latent_min: int = 32
    f_latent_max: int = 64
    f_latent_step: int = 16

    # ---- regularization ----
    dropout_min: float = 0.0
    dropout_max: float = 0.4

    beta_elbo_choices: Sequence[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.2, 0.1, 0.01, 0.001],
        )
    # 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 

    aux_weight_choices: Sequence[float] = field(
        default_factory=lambda: [1.0, 2.0, 5.0, 10.0, 20.0, 100.0],
    )
    
    # ---- optimizer hyperparameters ----
    lr_min: float = 0.0001 
    lr_max: float = 0.005
    lr_log: bool = True

    wd_min: float = 1e-6
    wd_max: float = 1e-4
    wd_log: bool = True


    # Fixed hyperparameters
    learn_init_state: bool = False
    use_kendall: bool = True
    rollout_weight: int = 0.0 # backwards compatibility
    
    # ---------- sampling ----------
    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        # Time grid
        n_time = trial.suggest_int("n_time", self.n_time_min, self.n_time_max, step=self.n_time_step)

        # Latent dimensions
        z_dim = trial.suggest_int("z_dim", self.z_dim_min, self.z_dim_max, step=self.z_dim_step)
        s_dim = trial.suggest_int("s_dim", self.s_dim_min, self.s_dim_max, step=self.s_dim_step)
        y_dim = trial.suggest_int("y_dim", self.y_dim_min, self.y_dim_max, step=self.y_dim_step)

        # MSM architecture
        msm_hidden_size = trial.suggest_int(
            "msm_hidden_size", self.msm_hidden_size_min, self.msm_hidden_size_max,
            step=self.msm_hidden_size_step,
        )
        msm_n_layers = trial.suggest_int(
            "msm_n_layers", self.msm_n_layers_min, self.msm_n_layers_max,
            step=self.msm_n_layers_step,
        )
        f_dims = [msm_hidden_size] * msm_n_layers

        f_latent = trial.suggest_int(
            "f_latent", self.f_latent_min, self.f_latent_max, step=self.f_latent_step,
        )

        # History RNN
        h_dim = trial.suggest_int(
            "h_dim", self.h_dim_min, self.h_dim_max, step=self.h_dim_step,
        )

        # Regularization / nonlinearity, optimizer
        dropout = trial.suggest_float("dropout", self.dropout_min, self.dropout_max)
        beta_elbo = trial.suggest_categorical("beta_elbo", list(self.beta_elbo_choices))
        aux_weight = trial.suggest_categorical("aux_weight", list(self.aux_weight_choices))

        if self.lr_log:
            learning_rate = trial.suggest_float("learning_rate", self.lr_min, self.lr_max, log=True)
        else:
            learning_rate = trial.suggest_float("learning_rate", self.lr_min, self.lr_max)

        if self.wd_log:
            weight_decay = trial.suggest_float("weight_decay", self.wd_min, self.wd_max, log=True)
        else:
            weight_decay = trial.suggest_float("weight_decay", self.wd_min, self.wd_max)

        return {
            "n_time": n_time,
            "z_dim": z_dim,
            "s_dim": s_dim,
            "y_dim": y_dim,
            "h_dim": h_dim,
            "f_dims": f_dims,
            "f_latent": f_latent,
            "dropout": dropout,
            "beta_elbo": beta_elbo,
            "aux_weight": aux_weight,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "learn_init_state": self.learn_init_state,
            "use_kendall": self.use_kendall,
        }

    def suggest_model_config(
        self,
        trial: optuna.Trial,
        *,
        tmat: np.ndarray,
        base_ftypes: Dict[str, Dict[str, Any]],
        state_ftypes: Dict[str, Dict[str, Any]],
    ) -> ModelConfig:
        params = self.suggest(trial)

        return ModelConfig(
            tmat=tmat,
            base_ftypes=base_ftypes,
            state_ftypes=state_ftypes,
            n_time=params["n_time"],
            z_dim=params["z_dim"],
            s_dim=params["s_dim"],
            y_dim=params["y_dim"],
            h_dim=params["h_dim"],
            f_dims=params["f_dims"],
            f_latent=params["f_latent"],
            dropout=params["dropout"],
            beta_elbo=params["beta_elbo"],
            aux_weight=params["aux_weight"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            learn_init_state=self.learn_init_state,
            use_kendall=self.use_kendall,
        )

    # ---------- (de-)serialization ----------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSpace":
        valid_fields = {f.name for f in fields(cls)}
        unknown = set(data) - valid_fields
        if unknown:
            raise AttributeError(f"Unknown field(s) in SearchSpace: {unknown}")
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "SearchSpace":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)