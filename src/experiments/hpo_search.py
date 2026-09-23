from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import optuna
import torch
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from msprep import MultiStatePrep

from dgmsm.config import ModelConfig
from dgmsm.utils import MultiStateData, LabelTrafo

from dgmsm.hivae.normalization import compute_norm_params_from_dataloader

from .utils import TestResult, set_global_seed, METRIC_DIRECTIONS


class HPOSearch:
    def __init__(
        self,
        msp_train: MultiStatePrep,
        msp_test: MultiStatePrep,
        meta_cfg,
        baseline_ftypes_global: Dict[str, Dict[str, Any]],
        state_ftypes_global: Dict[str, Dict[str, Any]],
        experiment_name: Optional[str] = None,
        save_dir: Optional[str] = None,
        log: Optional[logging.Logger] = None,
        seed: Optional[int] = None,
        eval_mask: Optional[torch.Tensor] = None,  # optional input for state-specific splits
    ):
        self.meta_cfg = meta_cfg
        self.log = log or logging.getLogger(__name__)
        self.experiment_name = experiment_name or meta_cfg.experiment_name
        self.seed = seed if seed is not None else meta_cfg.seed
        set_global_seed(self.seed)

        self.baseline_ftypes_global = baseline_ftypes_global
        self.state_ftypes_global = state_ftypes_global

        # Optuna / CV config
        self.n_trials = meta_cfg.n_trials if meta_cfg.n_trials is not None else 50
        self.val_size = meta_cfg.val_size if meta_cfg.val_size is not None else 0.2
        self.num_workers = meta_cfg.num_workers
        self.target_metric = meta_cfg.target_metric

        # Device
        if meta_cfg.device is not None:
            self.device = torch.device(meta_cfg.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but no CUDA available.")

        # Save folder
        self.save_dir = (
            Path(save_dir)
            if save_dir is not None
            else Path(meta_cfg.save_dir or ".")
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Data / state meta
        self.msp_train_full = msp_train
        self.msp_test = msp_test

        self.state_names = msp_train.nonterminal_state_names
        self.n_states = len(msp_train.nonterminal_state_indices)
        self.eval_mask = eval_mask

        # Transition matrix for ModelConfig
        self.tmat = np.asarray(msp_train.tmat, dtype=float)

        # Split outer train into inner-train and inner-val
        self._prep_data()

        # Optuna study
        self.study: Optional[optuna.Study] = None

    # ---------- data prep ----------
    def _prep_data(self) -> None:
        """
        Split msp_train_full into inner train and validation sets on person level.
        """
        states = self.msp_train_full.state_labels  # dataframe with trajectories only
        y_person = states.groupby("person_id", sort=False)["state"].max()

        person_ids = y_person.index.to_numpy()
        y = y_person.to_numpy()

        train_ids, val_ids = train_test_split(
            person_ids,
            test_size=self.val_size,
            stratify=y,
            random_state=self.seed,
        )

        self.msp_train = self.msp_train_full.subset_persons(person_ids=set(train_ids))
        self.msp_val = self.msp_train_full.subset_persons(person_ids=set(val_ids))

    # ---------- main run ----------
    def run(self) -> Tuple[TestResult, pd.DataFrame]:
        self._log_state_counts()

        try:
            direction = METRIC_DIRECTIONS[self.target_metric]
        except KeyError:
            raise ValueError(
                f"Unknown target_metric={self.target_metric!r}. "
                f"Known: {list(METRIC_DIRECTIONS.keys())}"
            )
        self.log.info(
            "Optuna target metric: %s (direction=%s)",
            self.target_metric, direction
        )
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.NopPruner()
        if self.meta_cfg.use_db:
            try:
                from experiments.db_cfg import get_optuna_db
                storage = get_optuna_db()
                # delete existing study with same name (if any)
                existing = optuna.study.get_all_study_summaries(storage=storage)
                if any(s.study_name == self.experiment_name for s in existing):
                    self.log.info(
                        "Deleting existing Optuna study '%s'", self.experiment_name
                    )
                    optuna.delete_study(
                        study_name=self.experiment_name,
                        storage=storage,
                    )
            except Exception as e:
                self.log.warning(
                    "Optuna DB not available (%s); falling back to in-memory storage.",
                    e,
                )
                storage = None
        else: 
            storage = None # use memory
        
        self.study = optuna.create_study(
            study_name=self.experiment_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
        )

        self.study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)

        best = self.study.best_trial
        best_cfg_dict = best.user_attrs["model_config"]
        best_model_cfg = ModelConfig.from_dict(best_cfg_dict)

        self.log.info(
            "%s optimization finished. Best value=%.4f, params=%s, best_epoch=%s",
            self.experiment_name,
            best.value,
            best.params,
            str(best.user_attrs.get("best_epoch", None)),
        )

        results, metrics_df = self._eval_final_model(best_model_cfg=best_model_cfg)

        config_path = self.save_dir / "best_model_config.yaml"
        best_model_cfg.to_yaml(config_path)
        self.log.info("Saved best model config to: %s", config_path)

        metrics_path = self.save_dir / "final_test_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        self.log.info("Saved final test metrics to: %s", metrics_path)

        return results, metrics_df

    # ---------- Optuna objective ----------
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective: validation metric on a fixed inner train/val split.
        """
        set_global_seed(self.seed + trial.number)

        # Sample ModelConfig from MetaConfig.search_space
        cfg = self.meta_cfg.suggest_model_config(
            trial=trial,
            tmat=self.tmat,
            base_ftypes=self.baseline_ftypes_global,
            state_ftypes=self.state_ftypes_global,
        )
        trial.set_user_attr("model_config", cfg.to_dict())

        # Label transform
        trafo = LabelTrafo(n_time=cfg.n_time)

        states_train_disc = trafo.fit_transform(self.msp_train.states)
        train_loader = self._build_loader(
            discrete_states_df=states_train_disc,
            baseline_df=self.msp_train.baseline,
            cfg=cfg,
            shuffle=True,
        )

        states_val_disc = trafo.transform(self.msp_val.states)
        val_loader = self._build_loader(
            discrete_states_df=states_val_disc,
            baseline_df=self.msp_val.baseline,
            cfg=cfg,
            shuffle=False,
        )

        base_norm_params, state_norm_params = compute_norm_params_from_dataloader(
            config=cfg,
            dataloader=train_loader,
            device=self.device,
        )

        model = cfg.build_model(
            base_norm_params=base_norm_params,
            state_norm_params=state_norm_params,
        ).to(self.device)

        best_epoch = None
        try:
            train_kwargs = cfg.training_kwargs()
            stats = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                trial=trial,
                restore_best=True,
                step_offset=0,
                verbose=False,
                **train_kwargs,
            )
            best_epoch = stats.get("best_epoch", None)

            evaluator = model.eval_surv(
                dataset=val_loader.dataset,
                device=self.device,
                state_names=self.state_names,
            )
            metric = float(evaluator.mean_metric(metric=self.target_metric))
            if np.isnan(metric):
                direction = METRIC_DIRECTIONS[self.target_metric]
                self.log.warning(
                    "Validation metric %r is NaN for trial %d (direction=%s). "
                    "Penalizing this trial with %s.",
                    self.target_metric,
                    trial.number,
                    direction,
                    "-inf" if direction == "maximize" else "inf",
                )
                if direction == "maximize":
                    metric = -np.inf
                else:
                    metric = np.inf
        finally:
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if best_epoch is not None:
            trial.set_user_attr("best_epoch", int(best_epoch))

        return metric

    # ---------- final training + test evaluation ----------
    def _eval_final_model(self, best_model_cfg: ModelConfig) -> Tuple[TestResult, pd.DataFrame]:
        set_global_seed(self.seed)

        # Label transform on full outer train
        trafo = LabelTrafo(n_time=best_model_cfg.n_time)
        states_train_full_disc = trafo.fit_transform(self.msp_train_full.states)
        train_loader = self._build_loader(
            discrete_states_df=states_train_full_disc,
            baseline_df=self.msp_train_full.baseline,
            cfg=best_model_cfg,
            shuffle=True,
        )

        # Test dataset / loader
        states_test_disc = trafo.transform(self.msp_test.states)
        test_dataset = MultiStateData(
            states_df=states_test_disc,
            baseline_df=self.msp_test.baseline,
            **best_model_cfg.dataset_kwargs(),
        )

        final_grid_times = trafo.discrete_times

        # Training length: use best_epoch from inner search if available
        best_trial = self.study.best_trial
        best_epoch = best_trial.user_attrs.get("best_epoch", None)
        if best_epoch is not None:
            max_epochs = int(best_epoch)
        else:
            max_epochs = best_model_cfg.training_kwargs().get("epochs", 100)

        # Final training kwargs: fixed number of epochs, effectively no early stopping
        final_train_kwargs = best_model_cfg.training_kwargs()
        final_train_kwargs["epochs"] = max_epochs
        final_train_kwargs["patience"] = max_epochs

        base_norm_params, state_norm_params = compute_norm_params_from_dataloader(
            config=best_model_cfg,
            dataloader=train_loader,
            device=self.device,
        )


        model = best_model_cfg.build_model(
            base_norm_params=base_norm_params,
            state_norm_params=state_norm_params,
        ).to(self.device)
        try:
            stats = model.fit(
                train_loader=train_loader,
                val_loader=None,
                device=self.device,
                restore_best=False,
                **final_train_kwargs,
            )

            # Save trained model
            model_path = self.save_dir / "best_final_model.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": best_model_cfg.to_dict(),
                },
                model_path,
            )
            self.log.info("Saved final trained model to: %s", model_path)

            evaluator = model.eval_surv(
                dataset=test_dataset,
                device=self.device,
                state_names=self.state_names,
                eval_mask=self.eval_mask,
            )
            metrics = evaluator.metrics
            auc_curves = evaluator.auc_curves or {}
            brier_curves = evaluator.brier_curves or {}
            metrics_df = evaluator.metrics_df.copy()

            # long format: one row per (metric, state, value)
            metrics_df = metrics_df.melt(
                id_vars="metric",
                value_vars=self.state_names,
                var_name="state",
                value_name="value",
            )

            self.log.info(
                "%s test evaluation:\n%s",
                self.experiment_name,
                metrics_df.to_string(index=False),
            )

            results = TestResult(
                metrics=metrics,
                train_stats=stats,
                auc_curves=auc_curves,
                brier_curves=brier_curves,
                model_cfg=best_model_cfg,
                final_grid_times=final_grid_times,
            )
            return results, metrics_df
        finally:
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


    def _build_loader(
        self,
        discrete_states_df,
        baseline_df,
        cfg: ModelConfig,
        shuffle: bool,
    ) -> DataLoader:
        
        dataset = MultiStateData(
            states_df=discrete_states_df,
            baseline_df=baseline_df,
            **cfg.dataset_kwargs(),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            pin_memory=(self.device.type == "cuda"),
            num_workers=self.num_workers,
        )
        return dataloader

    def _log_state_counts(self) -> None:
        self.log.info(
            "Per-state event summary (training data):\n%s",
            self.msp_train.event_summary().to_string(index=False),
        )
        self.log.info(
            "Per-state event summary (validation data):\n%s",
            self.msp_val.event_summary().to_string(index=False),
        )