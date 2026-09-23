from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from msprep import MultiStatePrep
from msprep.utils.syndat_scores import syndat_scores

from ._plot_syndat import plot_syndat_summary_cv

from experiments.config import MetaConfig
from dgmsm.config import ModelConfig
from dgmsm.synthetic_msp import baseline_simulation
from dgmsm.utils import MultiStateData, LabelTrafo


@dataclass
class BaselineSimCVReport:
    sampling_all: Optional[pd.DataFrame] = None
    sampling_summary: Optional[pd.DataFrame] = None
    sampling_raw: dict[int, dict[str, Any]] = field(default_factory=dict)
    sampling_figure: Optional[Any] = None


class BaselineSimCV:
    """
    Cross-validated synthetic-data evaluation using baseline-conditioned
    simulation.

    For each outer CV fold:
    1. load the saved fold model
    2. rebuild the training split (as MultiStatePrep and MultiStateData)
    3. simulate fully random trajectories conditioned only on each patient's
       baseline covariates (no observed states)
    4. score synthetic vs. real training data
    5. aggregate across folds
    6. optionally plot a CV summary figure
    """

    def __init__(
        self,
        meta_cfg_path: str | Path,
        device: Optional[torch.device | str] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.log = log or logging.getLogger(__name__)

        self.meta_cfg_path = Path(meta_cfg_path)
        self.log.info("Loading MetaConfig from: %s", self.meta_cfg_path)
        self.meta_cfg = MetaConfig.from_yaml(self.meta_cfg_path)


        if self.meta_cfg.n_splits is None:
            raise ValueError("MetaConfig.n_splits must be set.")

        self.seed = self.meta_cfg.seed
        self.n_splits = self.meta_cfg.n_splits

        if device is None:
            if self.meta_cfg.device is not None:
                device = self.meta_cfg.device
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.log.info(
            "Initializing baseline-sim CV evaluator | n_splits=%d | seed=%s | device=%s",
            self.n_splits,
            str(self.seed),
            str(self.device),
        )

        self.log.info("Loading dataset from: %s", self.meta_cfg.dataset_path)
        self.msp = MultiStatePrep.load(self.meta_cfg.dataset_path)
        self.dataset_name = getattr(self.msp, "dataset_name", "dataset")

        if self.meta_cfg.save_dir is not None:
            self.save_dir = Path(self.meta_cfg.save_dir)
        else:
            self.save_dir = Path(self.meta_cfg.experiment_name)
        self.log.info("Using model directory: %s", self.save_dir)

        self.fold_splits = self._make_outer_fold_splits()
        self.folds = list(sorted(self.fold_splits.keys()))

    def _make_outer_fold_splits(self) -> dict[int, tuple[list[Any], list[Any]]]:
        self.log.info("Reconstructing outer CV splits using StratifiedGroupKFold")

        states = self.msp.state_labels
        y_person = states.groupby("person_id", sort=False)["state"].max()

        person_ids = y_person.index.to_numpy()
        y = y_person.to_numpy()

        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )

        fold_splits: dict[int, tuple[list[Any], list[Any]]] = {}

        for fold, (train_idx, test_idx) in enumerate(
            sgkf.split(X=person_ids, y=y, groups=person_ids),
            start=1,
        ):
            train_ids = person_ids[train_idx].tolist()
            test_ids = person_ids[test_idx].tolist()
            fold_splits[fold] = (train_ids, test_ids)

            self.log.info(
                "Prepared fold %d/%d | train=%d persons | test=%d persons",
                fold,
                self.n_splits,
                len(train_ids),
                len(test_ids),
            )

        return fold_splits

    def _load_fold_model_and_train_data(
        self,
        fold: int,
    ):
        model, cfg = self._load_fold_model(fold)
        msp_train = self._build_fold_train_msp(fold)
        return model, msp_train, cfg

    def _build_fold_train_msp(self, fold: int) -> MultiStatePrep:
        if fold not in self.fold_splits:
            raise ValueError(f"Fold {fold} not found.")

        train_ids, _ = self.fold_splits[fold]

        msp_train = self.msp.subset_persons(
            train_ids,
            dataset_name=f"{self.dataset_name}_fold{fold}_train",
        )
        self.log.info(
            "Built training MSP for fold %d | persons=%d",
            fold,
            msp_train.n_persons,
        )
        return msp_train

    def _load_fold_model(self, fold: int):
        ckpt_path = self.save_dir / f"fold_{fold}" / "best_final_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        self.log.info("Loading fold %d model from: %s", fold, ckpt_path)
        model, cfg = ModelConfig.load_model_from_checkpoint(
            ckpt_path,
            device=self.device,
        )
        model.eval()
        return model, cfg

    def _build_fold_dataset(
        self,
        model: Any,
        msp_train: MultiStatePrep,
        cfg: ModelConfig,
    ) -> tuple[MultiStateData, float]:
        trafo = LabelTrafo(n_time=model.n_time)
        states_train_disc = trafo.fit_transform(msp_train.states)

        dataset = MultiStateData(
            states_df=states_train_disc,
            baseline_df=msp_train.baseline,
            n_nonterminal_states=model.graph.n_nonterminal_states,
            max_path=model.max_path,
            baseline_ftypes=cfg.base_ftypes,
            states_ftypes=cfg.state_ftypes,
        )

        self.log.info(
            "Built training dataset | persons=%d | time_delta=%.4g",
            msp_train.n_persons,
            trafo.time_delta,
        )

        return dataset, trafo.time_delta

    @staticmethod
    def _aggregate_mean_std(
        df: pd.DataFrame,
        group_cols: list[str],
        value_col: str = "value",
    ) -> pd.DataFrame:
        return (
            df.groupby(group_cols, as_index=False, dropna=False)
            .agg(
                mean=(value_col, "mean"),
                std=(value_col, "std"),
                n_folds=("fold", "nunique"),
            )
            .sort_values(group_cols)
            .reset_index(drop=True)
        )



    @staticmethod
    def _melt_sampling_scores(
        scores_df: pd.DataFrame,
        fold: int,
    ) -> pd.DataFrame:
        df = scores_df.copy()

        if isinstance(df.index, pd.MultiIndex):
            index_names = [
                name if name is not None else f"index_{i}"
                for i, name in enumerate(df.index.names)
            ]
            df.index = df.index.set_names(index_names)
        else:
            df.index = df.index.rename(df.index.name or "state_name")

        df = df.reset_index()

        id_cols = [c for c in ["state_name", "visit", "state_id"] if c in df.columns]
        value_cols = [
            c for c in df.columns
            if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not value_cols:
            return pd.DataFrame(columns=id_cols + ["metric", "value", "fold"])

        long_df = df.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name="metric",
            value_name="value",
        ).copy()

        long_df["fold"] = fold

        aj_metric = "AJ Similarity Score"
        if "visit" in long_df.columns and (long_df["metric"] == aj_metric).any():
            is_aj = long_df["metric"] == aj_metric
            aj_rows = (
                long_df[is_aj]
                .drop_duplicates(subset=["state_name", "metric", "fold"])
                .assign(visit=pd.NA)
            )
            long_df = pd.concat([long_df[~is_aj], aj_rows], ignore_index=True)

        return long_df

    def plot_sampling_summary(
        self,
        report: BaselineSimCVReport,
        title: Optional[str] = None,
        figsize: tuple[float, float] = (16.0, 4.8),
        dpi: int = 300,
        show_points: bool = False,
        show: bool = True,
        score_ylim: tuple[float, float] = (50, 100),
        save_path: Optional[Any] = None,
    ) -> Any:
        if report.sampling_all is None or report.sampling_all.empty:
            raise ValueError("report.sampling_all is empty. Run the evaluation first.")

        self.log.info(
            "Plotting CV sampling summary | save_path=%s",
            str(save_path) if save_path is not None else "<none>",
        )

        return plot_syndat_summary_cv(
            scores_long=report.sampling_all,
            title=title,
            figsize=figsize,
            dpi=dpi,
            show_points=show_points,
            show=show,
            score_ylim=score_ylim,
            save_path=None if save_path is None else str(save_path),
        )

    def run(
        self,
        n_sim: int = 1,
        temperature: float = 1.0,
        max_future_steps: Optional[int] = None,
        stratify_by_visit: bool = False,
        title: Optional[str] = None,
        save_path = None,
        min_rows_per_group: int = 30,
        plot_summary: bool = True,
        figsize: tuple[float, float] = (16.0, 4.8),
        dpi: int = 300,
        show_points: bool = False,
        score_ylim: tuple[float, float] = (50, 100),
    ) -> BaselineSimCVReport:
        report = BaselineSimCVReport()
        sampling_parts: list[pd.DataFrame] = []

        total_start = time.perf_counter()


        for fold_idx, fold in enumerate(self.folds, start=1):
            fold_start = time.perf_counter()

            self.log.info("==== Fold %d/%d (fold_id=%d) ====", fold_idx, len(self.folds), fold)

            model = None
            msp_train = None
            dataset = None
            synthetic_msp = None
            synthetic_msp_path = None
            sim_result = None
            score_df = None

            try:
                model, msp_train, cfg = self._load_fold_model_and_train_data(fold)

                msp_template = msp_train.get_template()
                dataset, time_delta = self._build_fold_dataset(model, msp_train, cfg)

                self.log.info(
                    "Fold %d | simulating baseline trajectories | persons=%d | n_sim=%d",
                    fold,
                    msp_train.n_persons,
                    n_sim,
                )
                sim_start = time.perf_counter()
                sim_result = baseline_simulation(
                    model=model,
                    msp_template=msp_template,
                    dataset=dataset,
                    n_sim=n_sim,
                    real_msp=msp_train,
                    time_delta=time_delta,
                    max_future_steps=max_future_steps,
                    temperature=temperature,
                )
                synthetic_msp = sim_result.synthetic_msp
                self.log.info(
                    "Fold %d | simulation done | synthetic_n=%d | elapsed=%.2fs",
                    fold,
                    synthetic_msp.n_persons,
                    time.perf_counter() - sim_start,
                )



                self.log.info("Fold %d | scoring synthetic vs. real", fold)
                score_start = time.perf_counter()
                score_df = syndat_scores(
                    real_msp=msp_train,
                    synthetic_msp=synthetic_msp,
                    stratify_by_visit=stratify_by_visit,
                    min_rows_per_group=min_rows_per_group,
                )
                self.log.info(
                    "Fold %d | scoring done | elapsed=%.2fs",
                    fold,
                    time.perf_counter() - score_start,
                )

                report.sampling_raw[fold] = {
                    "scores": score_df,
                    "n_train": msp_train.n_persons,
                    "n_synthetic": synthetic_msp.n_persons,
                    "n_sim": n_sim,
                    "id_mapping": sim_result.id_mapping,
                }

                long_df = self._melt_sampling_scores(score_df, fold)
                sampling_parts.append(long_df)

                self.log.info(
                    "Fold %d done | score_rows=%d | synthetic_n=%d | elapsed=%.2fs",
                    fold,
                    len(long_df),
                    synthetic_msp.n_persons,
                    time.perf_counter() - fold_start,
                )

            except Exception:
                self.log.exception("Fold %d failed during baseline-sim evaluation", fold)
                raise

            finally:
                del model
                del msp_train
                del dataset
                del synthetic_msp
                del sim_result
                del score_df

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        if sampling_parts:
            self.log.info("Aggregating sampling scores across %d folds", len(sampling_parts))
            report.sampling_all = pd.concat(sampling_parts, ignore_index=True)

            group_cols = ["metric"]
            for col in ["state_name", "visit", "state_id"]:
                if col in report.sampling_all.columns and report.sampling_all[col].notna().any():
                    group_cols.append(col)

            report.sampling_summary = self._aggregate_mean_std(
                report.sampling_all,
                group_cols=group_cols,
                value_col="value",
            )
        else:
            self.log.warning("No sampling scores were collected across folds.")



        if plot_summary and report.sampling_all is not None and not report.sampling_all.empty:
            report.sampling_figure = self.plot_sampling_summary(
                report=report,
                title=title,
                figsize=figsize,
                dpi=dpi,
                show_points=show_points,
                score_ylim=score_ylim,
                save_path=save_path,
            )

        return report

    def save_all_training_msps(
        self,
        directory: str | Path,
    ) -> dict[int, Path]:
        root_dir = Path(directory)

        root_dir.mkdir(parents=True, exist_ok=True)
        self.log.info("Saving all training MSPs to: %s", root_dir)

        saved_paths: dict[int, Path] = {}

        for fold in self.folds:
            fold_dir = root_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            msp_train = self._build_fold_train_msp(fold)
            meta_path = msp_train.save(fold_dir)

            saved_paths[fold] = meta_path
            self.log.info("Fold %d | saved training MSP to: %s", fold, meta_path)

        return saved_paths

    def save_synthetic_msp(
        self,
        synthetic_msp: MultiStatePrep,
        directory: str | Path,
    ) -> Path:
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        return synthetic_msp.save(save_dir)