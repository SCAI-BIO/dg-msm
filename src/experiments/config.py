from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, replace
from typing import Dict, Any, Optional

import numpy as np
import yaml
import optuna

from .searchspace import SearchSpace

@dataclass(frozen=True)
class MetaConfig:
    # --- dataset / experiment metadata ---
    dataset_path: str
    experiment_name: str = "default"

    # --- Optuna / CV arguments ----
    n_trials: Optional[int] = None
    n_splits: Optional[int] = None
    val_size: Optional[float] = None
    device: Optional[str] = None
    save_dir: Optional[str] = None
    use_db: bool = False
    # --- Time-split arguments ---
    train_frac: Optional[float] = None
    cut_date: Optional[int] = None

    # --- Additional parameters ---
    target_metric: str = "adj Antolini IPCW"  # e.g. "IBS" 
    num_workers: int = 0
    seed: int = 42

    search_space = None

    def __post_init__(self):
        if self.search_space is None:
            object.__setattr__(self, "search_space", SearchSpace())

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "MetaConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaConfig":
        data = dict(data)
        valid_fields = {f.name for f in fields(cls)}
        unknown = set(data) - valid_fields
        if unknown:
            raise AttributeError(f"Unknown field(s) in MetaConfig: {unknown}")

        ss = data.get("search_space")

        if isinstance(ss, dict):
            data["search_space"] = SearchSpace.from_dict(ss)

        return cls(**data)

    def suggest_model_config(
        self,
        trial: optuna.Trial,
        *,
        tmat: Optional[np.ndarray] = None,
        base_ftypes: Optional[Dict[str, Dict[str, Any]]] = None,
        state_ftypes: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        if tmat is None or base_ftypes is None or state_ftypes is None:
            raise ValueError(
                "tmat, base_ftypes and state_ftypes must be provided."
            )
        return self.search_space.suggest_model_config(
            trial=trial,
            tmat=tmat,
            base_ftypes=base_ftypes,
            state_ftypes=state_ftypes,
        )

    def with_updates(self, **kwargs) -> "MetaConfig":
        """
        Return a new MetaConfig with specified fields overridden.
        """
        valid_fields = {f.name for f in fields(type(self))}
        unknown = set(kwargs) - valid_fields
        if unknown:
            raise AttributeError(
                f"Unknown field(s) in MetaConfig.with_updates: {unknown}"
            )
        return replace(self, **kwargs)
