from __future__ import annotations

from dataclasses import dataclass, asdict, fields, field
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import torch

import numpy as np
import logging
import yaml

from .model import DGMSM

from .utils import compute_tmat_stats

@dataclass
class ModelConfig:
    tmat: np.ndarray
    base_ftypes: Dict[str, Dict[str, Any]]
    state_ftypes: Dict[str, Dict[str, Any]]
    n_time: int
    z_dim: int
    s_dim: int
    y_dim: int
    h_dim: int
    f_dims: Sequence[int]
    f_latent: int

    dropout:        float = 0.1
    beta_elbo:      float = 1.0
    aux_weight:     float = 1.0
    use_kendall: bool = True
    learn_init_state: bool = False
    # Path cutoff (optional)
    max_path: int = 20 #Optional[int] = None

    # Derived from tmat:
    n_nonterminal_states: Optional[int] = None

    # training hyperparameters
    batch_size: int = 128
    epochs: int = 300
    learning_rate: float = 1e-3
    scheduler_name: str = "ReduceLROnPlateau"
    lr_decay_rate: float = 1e-5
    patience: int = 10
    min_delta: float = 1e-3
    weight_decay: float = 1e-5
    eval_every: int = 1

    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure tmat is an array
        self.tmat = np.asarray(self.tmat, dtype=float)

        # Use compute_tmat_stats to derive max_path and n_nonterminal_states
        stats = compute_tmat_stats(self.tmat, max_path=self.max_path)
        self.max_path = stats["max_path"]
        self.n_nonterminal_states = stats["n_nonterminal_states"]

    # model construction

    def model_kwargs(self, log: Optional[logging.Logger] = None) -> Dict[str, Any]:
        base_ftypes_model = [self.base_ftypes[name] for name in self.base_ftypes.keys()]
        state_ftypes_model = [self.state_ftypes[name] for name in self.state_ftypes.keys()]

        kwargs: Dict[str, Any] = {
            "tmat":         self.tmat,
            "n_time":       self.n_time,
            "z_dim":        self.z_dim,
            "s_dim":        self.s_dim,
            "y_dim":        self.y_dim,
            "h_dim":        self.h_dim,
            "base_ftypes":  base_ftypes_model,
            "state_ftypes": state_ftypes_model,
            "f_dims":       list(self.f_dims),
            "f_latent":     self.f_latent,
            "dropout":      self.dropout,
            "beta_elbo":    self.beta_elbo,
            "use_kendall":  self.use_kendall,
            "aux_weight":   self.aux_weight,
            "log":          log,
            "max_path":     self.max_path,
            "n_nonterminal_states": self.n_nonterminal_states,
            "learn_init_state": self.learn_init_state,
            **self.extra_kwargs,
        }
        return kwargs

    def build_model(
        self,
        log: Optional[logging.Logger] = None,
        **change_kwargs: Any,
    ) -> DGMSM:
        kwargs = self.model_kwargs(log=log)
        # let explicit overrides win over config
        kwargs.update(change_kwargs)
        return DGMSM(**kwargs)

    def dataset_kwargs(self) -> Dict[str, Any]:
        return {
            "n_nonterminal_states":    self.n_nonterminal_states,
            "max_path":                 self.max_path,
            "baseline_ftypes": self.base_ftypes,
            "states_ftypes": self.state_ftypes,
        }
    
    def training_kwargs(self) -> Dict[str, Any]:
        return {
            "batch_size":    self.batch_size,
            "epochs":        self.epochs,
            "learning_rate": self.learning_rate,
            "scheduler_name": self.scheduler_name,
            "lr_decay_rate":  self.lr_decay_rate,
            "patience":       self.patience,
            "min_delta":      self.min_delta,
            "weight_decay":   self.weight_decay,
            "eval_every":     self.eval_every,
        }


    def to_dict(self) -> Dict[str, Any]:
        cfg = asdict(self)
        cfg["tmat"] = self.tmat.tolist()
        return cfg

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ModelConfig":
        cfg = dict(cfg)
        if isinstance(cfg.get("tmat"), list):
            cfg["tmat"] = np.asarray(cfg["tmat"], dtype=float)
        return cls(**cfg)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        valid_fields = {f.name for f in fields(cls)}
        unknown = set(data) - valid_fields
        if unknown:
            raise AttributeError(f"Unknown field(s) in ModelConfig YAML: {unknown}")

        return cls.from_dict(data)


    @classmethod
    def load_model_from_checkpoint(
        cls,
        ckpt_path: str | Path,
        device: str | torch.device = "cpu",
        log: Optional[logging.Logger] = None,
    ) -> tuple[DGMSM, "ModelConfig"]:
        """
        Load a trained DGMSM model and its ModelConfig from a checkpoint
        saved as:
            {"state_dict": model.state_dict(), "model_config": cfg.to_dict()}
        """

        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        # 1) reconstruct config
        cfg = cls.from_dict(ckpt["model_config"])
        state_dict = ckpt["state_dict"]

        # 2) reconstruct base_norm_params from buffers in state_dict (if present)
        base_norm_params = None
        if (
            "base_hivae.base_norm_mean" in state_dict
            and "base_hivae.base_norm_var" in state_dict
        ):
            base_mean = state_dict["base_hivae.base_norm_mean"]  # [n_base_features]
            base_var  = state_dict["base_hivae.base_norm_var"]   # [n_base_features]
            assert base_mean.dim() == 1 and base_var.shape == base_mean.shape
            base_norm_params = [
                (base_mean[i].clone(), base_var[i].clone())
                for i in range(base_mean.shape[0])
            ]

        # 3) reconstruct state_norm_params from buffers in state_dict (if present)
        state_norm_params = None
        if (
            "state_vrnn.state_norm_mean" in state_dict
            and "state_vrnn.state_norm_var" in state_dict
        ):
            state_mean = state_dict["state_vrnn.state_norm_mean"]  # [n_state_features]
            state_var  = state_dict["state_vrnn.state_norm_var"]   # [n_state_features]
            assert state_mean.dim() == 1 and state_var.shape == state_mean.shape
            state_norm_params = [
                (state_mean[i].clone(), state_var[i].clone())
                for i in range(state_mean.shape[0])
            ]

        # 4) build model with reconstructed norm params
        model = cfg.build_model(
            log=log,
            base_norm_params=base_norm_params,
            state_norm_params=state_norm_params,
        )

        # 5) load full state_dict
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()

        return model, cfg


    def __str__(self) -> str:
        EXCLUDE = {"base_ftypes", "state_ftypes", "tmat", "extra_kwargs"}

        sections = {
            "Architecture": ["n_time", "z_dim", "s_dim", "y_dim", "h_dim", "f_dims", "f_latent",
                            "n_nonterminal_states", "max_path", "learn_init_state"],
            "Loss weights":  ["beta_elbo", "aux_weight", "use_kendall"],
            "Regularisation": ["dropout"],
            "Training":      ["batch_size", "epochs", "learning_rate", "scheduler_name",
                            "lr_decay_rate", "patience", "min_delta", "weight_decay", "eval_every"],
        }

        # Collect all explicitly listed field names
        listed = {f for names in sections.values() for f in names}

        # Catch any new fields not yet assigned to a section
        other = [
            f.name for f in fields(self)
            if f.name not in listed and f.name not in EXCLUDE
        ]

        lines = ["ModelConfig"]
        lines.append("=" * 40)

        for section, keys in sections.items():
            lines.append(f"  [{section}]")
            for k in keys:
                lines.append(f"    {k:<26} = {getattr(self, k)}")

        if other:
            lines.append("  [Other]")
            for k in other:
                lines.append(f"    {k:<26} = {getattr(self, k)}")

        lines.append("=" * 40)
        return "\n".join(lines)