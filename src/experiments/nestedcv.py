from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from msprep import MultiStatePrep
from .config import MetaConfig
from .utils import TestResult, plot_curves, cv_summary, set_global_seed

from .hpo_search import HPOSearch


class NestedCrossVal:
    def __init__(
        self,
        msp: MultiStatePrep,
        meta_cfg: MetaConfig,
        log: Optional[logging.Logger] = None,
    ) -> None:
        # Logging
        self.log = log or logging.getLogger(__name__)
        # Config
        self.meta_cfg = meta_cfg
        self.seed = meta_cfg.seed
        self.experiment_name = meta_cfg.experiment_name

        # Data
        self.msp = msp
        self.baseline_ftypes_global = msp.baseline_ftypes
        self.state_ftypes_global = msp.state_ftypes

        # Save folder
        if meta_cfg.save_dir is not None:
            self.save_dir_path = Path(meta_cfg.save_dir)
        else:
            self.save_dir_path = Path(meta_cfg.experiment_name)

        # Number of CV splits
        if meta_cfg.n_splits is None:
            raise ValueError("MetaConfig.n_splits must be set for NestedCrossVal.")
        self.n_splits = meta_cfg.n_splits

    def run(self) -> Dict[str, Any]:
        set_global_seed(self.seed)
        self.save_dir_path.mkdir(parents=True, exist_ok=True)

        per_fold: List[TestResult] = []
        metrics_tables: List[pd.DataFrame] = []
        hp_rows: List[Dict[str, Any]] = []

        states = self.msp.state_labels  # trajectories only

        # Person-level label: max state reached
        y_person = states.groupby("person_id", sort=False)["state"].max()

        # Persons to split on (ensure consistent ordering)
        person_ids = y_person.index.to_numpy()
        y = y_person.to_numpy()

        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )

        for fold, (train_idx, test_idx) in enumerate(
            sgkf.split(X=person_ids, y=y, groups=person_ids),
            start=1,
        ):
            self.log.info("==== Fold %d/%d ====", fold, self.n_splits)

            train_ids = person_ids[train_idx].tolist()
            test_ids = person_ids[test_idx].tolist()

            # Create per-fold MSPs
            # Adjust subset_persons signature if needed (e.g. person_ids=set(train_ids))
            msp_train = self.msp.subset_persons(
                train_ids,
                dataset_name=f"{self.msp.dataset_name}_fold{fold}_train",
            )
            msp_test = self.msp.subset_persons(
                test_ids,
                dataset_name=f"{self.msp.dataset_name}_fold{fold}_test",
            )

            fold_exp_name = f"{self.experiment_name}_fold_{fold}"
            fold_save_dir = str(self.save_dir_path / f"fold_{fold}")

            inner_kwargs = {
                "msp_train": msp_train,
                "msp_test": msp_test,
                "meta_cfg": self.meta_cfg,
                "experiment_name": fold_exp_name,
                "save_dir": fold_save_dir,
                "log": self.log,
                "seed": self.seed + fold * 1000,
            }
            inner_loop = HPOSearch(
                baseline_ftypes_global= self.baseline_ftypes_global,
                state_ftypes_global= self.state_ftypes_global,
                **inner_kwargs,
            )


            fold_result, metrics_df_fold = inner_loop.run()
            per_fold.append(fold_result)
            metrics_df_fold["fold"] = fold
            metrics_tables.append(metrics_df_fold)

            # Collect best hyperparameters for this fold
            best_trial = inner_loop.study.best_trial
            row = dict(best_trial.params)  # tuned params
            row["fold"] = fold
            row["best_value"] = best_trial.value
            hp_rows.append(row)

        metrics_df = pd.concat(metrics_tables, ignore_index=True)

        summary = cv_summary(df=metrics_df, state_names=self.msp.state_names)

        self.log.info(
            "Cross-validation finished. Summary per state and metric:\n%s",
            summary.to_string(index=False),
        )

        hp_df = pd.DataFrame(hp_rows)

        ncv_results = {
            "per_fold": per_fold,
            "metrics_df": metrics_df,
            "summary_df": summary,
            "hyperparams_df": hp_df,
        }

        # Save CV results
        metrics_path = self.save_dir_path / "cv_metrics.csv"
        summary_path = self.save_dir_path / "cv_summary.csv"
        hp_path = self.save_dir_path / "cv_best_hyperparams.csv"

        metrics_df.to_csv(metrics_path, index=False)
        summary.to_csv(summary_path, index=False)
        hp_df.to_csv(hp_path, index=False)

        # Plotting
        plot_curves(
            results=ncv_results,
            state_names=self.msp.state_names,
            suptitle=self.experiment_name,
            show=False,
            save_to=self.save_dir_path,
        )

        return ncv_results