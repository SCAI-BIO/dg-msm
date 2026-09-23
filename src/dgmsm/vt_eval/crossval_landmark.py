# crossval_landmark.py
from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from ..config import ModelConfig
from ..utils import LabelTrafo, MultiStateData
from experiments.config import MetaConfig
from msprep import MultiStatePrep

from ._results import CVFoldResult, CVLandmarkResult
from .plot import (
    plot_cv_landmark_metrics,
    plot_cv_landmark_pp,
    plot_cv_landmark_pi_coverage,
)

from .multi_state_landmark import multi_state_landmark
from .state_landmark import DEFAULT_PI_ALPHAS

logger = logging.getLogger(__name__)


class LandmarkCV:
    def __init__(
        self,
        meta_cfg_path,
        device= None,
        log = None,
        plot_dir = None,
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
            device = self.meta_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            self.log.warning("CUDA is not available, falling back to CPU.")
            device = torch.device("cpu")

        self.device = device

        self.log.info(
            "Initializing landmark CV evaluator | n_splits=%d | seed=%s | device=%s",
            self.n_splits, str(self.seed), str(self.device),
        )

        self.log.info("Loading dataset from: %s", self.meta_cfg.dataset_path)
        self.msp = MultiStatePrep.load(self.meta_cfg.dataset_path)

        self.save_dir = (
            Path(self.meta_cfg.save_dir)
            if self.meta_cfg.save_dir is not None
            else Path(self.meta_cfg.experiment_name)
        )
        self.log.info("Using model directory: %s", self.save_dir)

        self.plot_dir = Path(plot_dir) if plot_dir is not None else self.save_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.log.info("Using plot directory: %s", self.plot_dir)

        self.fold_splits = self._make_outer_fold_splits()
        self.folds = sorted(self.fold_splits)
        self.landmark_labels = {i + 1: name for i, name in enumerate(self.msp.nonterminal_state_names)}

        # Populated by .run(); external benchmark tables, kept outside the
        # typed hierarchy on purpose (see _metric_frames docstring).
        self._benchmark_frames: dict[str, list[pd.DataFrame]] = {}


    def run(
        self,
        suptitle: Optional[str] = None,
        n_sim: int = 1000,
        d_cal_bins: Optional[int]  = 10,
        pi_alphas: Optional[tuple[float, ...]] = None,
        landmark_benchmark_os_experiments: Optional[Mapping[str, str | Path]] = None,
        landmark_benchmark_sojourn_experiments: Optional[Mapping[str, str | Path]] = None,
        plot_metrics: bool = False,
        plot_reliability: bool = False,
        plot_pi_coverage: bool = False,
        metric_name_map: Optional[Mapping[str, str]] = None,
        show_fold_lines: bool = True,
        save_plots: bool = False,
        show_support: bool = True,
        support_mode: str = "total",
        **kwargs: Any,
    ) -> CVLandmarkResult:
        pi_alphas = pi_alphas or DEFAULT_PI_ALPHAS
        save_dir = self.plot_dir if save_plots else None

        folds: list[CVFoldResult] = []
        total_start = time.perf_counter()
        self.log.info("Starting landmark CV evaluation | folds=%s", self.folds)

        for fold_idx, fold in enumerate(self.folds, start=1):
            fold_start = time.perf_counter()
            self.log.info("==== Fold %d/%d (fold_id=%d) ====", fold_idx, len(self.folds), fold)
            model = msp_test = msp_train = test_dataset = trafo = None
            try:
                model, msp_train, msp_test, test_dataset, trafo = self._load_fold_inputs(fold)

                result = multi_state_landmark(
                    model=model, dataset=test_dataset, trafo=trafo,
                    state_names=self.msp.state_names, n_sim=n_sim,
                    d_cal_bins=d_cal_bins, pi_alphas=pi_alphas, device=self.device,
                    seed=None if self.seed is None else self.seed + fold * 1000,
                    **kwargs,
                )

                train_ids, test_ids = self.fold_splits[fold]
                folds.append(CVFoldResult(
                    fold=fold, n_train=len(train_ids), n_test=len(test_ids), result=result,
                ))

                self.log.info(
                    "Fold %d done | states=%s | elapsed=%.2fs",
                    fold, sorted(result.states()), time.perf_counter() - fold_start,
                )

            except Exception:
                self.log.exception("Fold %d failed during landmark evaluation", fold)
                raise
            finally:
                del model, msp_train, msp_test, test_dataset, trafo
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        self.report = CVLandmarkResult(folds=folds, landmark_labels=self.landmark_labels)

        # External benchmarks: loaded once here, merged only at plotting
        self._benchmark_frames = {"Overall Survival": [], "Sojourn Time": []}
        for method_name, exp_path in (landmark_benchmark_os_experiments or {}).items():
            self._benchmark_frames["Overall Survival"].append(
                self._load_external_landmark_benchmark(
                    experiment_dir=exp_path, method_name=method_name, start_visit_offset=1,
                )
            )
        for method_name, exp_path in (landmark_benchmark_sojourn_experiments or {}).items():
            self._benchmark_frames["Sojourn Time"].append(
                self._load_external_landmark_benchmark(
                    experiment_dir=exp_path, method_name=method_name, start_visit_offset=1,
                )
            )

        if any((plot_metrics, plot_reliability, plot_pi_coverage, save_dir)):
            self.plot_landmark_cv(
                report=self.report,
                metric_name_map=metric_name_map,
                show_plot=True,
                suptitle=suptitle,
                plot_metric_panels=plot_metrics,
                plot_reliability=plot_reliability,
                plot_pi_coverage=plot_pi_coverage,
                show_fold_lines=show_fold_lines,
                save_dir=save_dir,
                show_support=show_support,
                support_mode=support_mode,
            )

        self.log.info(
            "Landmark CV evaluation completed | states=%s | elapsed=%.2fs",
            self.report.states(), time.perf_counter() - total_start,
        )
        return self.report


    def _resolve_report(self, report: Optional[CVLandmarkResult]) -> CVLandmarkResult:
        """Resolve an explicit ``report=`` argument or fall back to
        ``self.report``; raise a single, consistent error if neither exists."""
        rpt = report or getattr(self, "report", None)
        if rpt is None:
            raise RuntimeError("No report available. Call .run() first or pass report= explicitly.")
        return rpt

    @staticmethod
    def _fig_path(save_dir_path: Optional[Path], stem: str):
        return save_dir_path / f"{stem}.png" if save_dir_path is not None else None

    @contextlib.contextmanager
    def _safe_plot(self, name: str):
        try:
            yield
        except Exception:
            self.log.exception("%s failed", name)

    @staticmethod
    def _metrics_long_frame(
        cv_result: CVLandmarkResult, ep: str, method: str = "DG-MSM",
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for f in cv_result.folds:
            for s, r in f.result.by_state.items():
                ep_result = r.endpoints.get(ep)
                if ep_result is None:
                    continue
                for k, v in ep_result.metrics.items():
                    if not np.isscalar(v):
                        continue
                    rows.append({
                        "metric": k, "landmark": f"landmark_{s}", "landmark_idx": int(s),
                        "value": float(v), "method": method, "fold": int(f.fold),
                    })
        if not rows:
            return pd.DataFrame(columns=["metric", "landmark", "value", "landmark_idx", "method", "fold"])
        return pd.DataFrame(rows)

    def _metric_frames(self, cv_result: CVLandmarkResult) -> dict[str, pd.DataFrame]:
        label_to_ep = {"Overall Survival": "os", "Sojourn Time": "sojourn"}
        out: dict[str, pd.DataFrame] = {}
        for label, ep in label_to_ep.items():
            parts = [self._metrics_long_frame(cv_result, ep)]
            parts += self._benchmark_frames.get(label, [])
            parts = [p for p in parts if p is not None and not p.empty]
            out[label] = (
                pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=["metric", "landmark", "value", "landmark_idx", "method", "fold"])
            )
        return out

    # Plotting
    def plot_landmark_cv(
        self,
        report: CVLandmarkResult,
        metric_name_map: Optional[Mapping[str, str]] = None,
        methods_order: Optional[Sequence[str]] = None,
        show_plot: bool = True,
        suptitle: Optional[str] = None,
        plot_metric_panels: bool = True,
        plot_reliability: bool = False,
        plot_pi_coverage: bool = False,
        show_fold_lines: bool = True,
        reliability_plot_band: bool = False,
        show_support: bool = True,
        support_mode: str = "total",
        save_dir: Optional[str | Path] = None,
    ) -> dict[str, plt.Figure]:
        landmark_all = self._metric_frames(report)
        save_dir_path = Path(save_dir) if save_dir is not None else None
        figures: dict[str, plt.Figure] = {}
        support_df = report.support_summary() if show_support else None

        n_folds = len(report.folds)
        cv_note = (
            f"{n_folds}-fold cross-validation"
            if n_folds > 1
            else "held-out evaluation"
        )

        title_prefix = suptitle or "Dynamically Updated Predictive Performance"

        if plot_metric_panels:
            for group_label, df in landmark_all.items():
                if df.empty:
                    continue
                metrics_suptitle = (
                    f"{title_prefix}\n"
                    f"{group_label} across all states ({cv_note})"
                )
                with self._safe_plot(f"plot_cv_landmark_metrics[{group_label}]"):
                    available_methods = (
                        df["method"].dropna().astype(str).drop_duplicates().tolist()
                        if "method" in df.columns else []
                    )
                    
                    base_order = list(methods_order or []) + available_methods
                    m_order = [m for m in dict.fromkeys(str(x) for x in base_order) if m in available_methods]
                    if "DG-MSM" in m_order:
                        m_order.remove("DG-MSM")
                        m_order.insert(0, "DG-MSM")
                    
                    figures[group_label] = plot_cv_landmark_metrics(
                            landmark_cv_df=df,
                            metric_name_map=metric_name_map,
                            landmark_labels=self.landmark_labels,
                            methods_order=m_order,
                            show_plot=show_plot,
                            suptitle=metrics_suptitle,
                            save_path=self._fig_path(save_dir_path, f"landmark_metrics_{group_label}"),
                            support=support_df,
                            support_mode=support_mode,
                        )

        if plot_reliability and report.states():
            with self._safe_plot("reliability"):
                figures["reliability"] = plot_cv_landmark_pp(
                    cv_result=report,
                    landmark_labels=self.landmark_labels,
                    show=show_plot,
                    save_path=self._fig_path(save_dir_path, "reliability_cv"),
                    show_folds=show_fold_lines,
                    plot_band=reliability_plot_band,
                )

        if plot_pi_coverage and report.states():
            with self._safe_plot("pi_coverage"):
                figures["pi_coverage"] = plot_cv_landmark_pi_coverage(
                    cv_result=report,
                    landmark_labels=self.landmark_labels,
                    show=show_plot,
                    save_path=self._fig_path(save_dir_path, "pi_coverage_cv"),
                    show_folds=show_fold_lines,
                    plot_band=reliability_plot_band,
                )


        return figures

    def plot_reliability_cv(
        self,
        report: Optional[CVLandmarkResult] = None,
        show_folds: bool = True,
        save_path: Optional[str] = None,
        show: bool = True,
        plot_fill: bool = False,
        plot_band: bool = False,
    ) -> "plt.Figure":
        rpt = self._resolve_report(report)
        if not rpt.states():
            raise ValueError("No reliability data found. Run .run() first.")
        return plot_cv_landmark_pp(
            cv_result=rpt,
            landmark_labels=self.landmark_labels,
            show=show,
            save_path=save_path,
            show_folds=show_folds,
            plot_band=plot_band,
            plot_fill=plot_fill,
        )





    # CV split reconstruction
    def _make_outer_fold_splits(self) -> dict[int, tuple[list[Any], list[Any]]]:
        self.log.info("Reconstructing outer CV splits using StratifiedGroupKFold")
        states = self.msp.state_labels
        y_person = states.groupby("person_id", sort=False)["state"].max()
        person_ids = y_person.index.to_numpy()
        y = y_person.to_numpy()

        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        fold_splits: dict[int, tuple[list[Any], list[Any]]] = {}
        for fold, (train_idx, test_idx) in enumerate(
            sgkf.split(X=person_ids, y=y, groups=person_ids), start=1
        ):
            train_ids = person_ids[train_idx].tolist()
            test_ids = person_ids[test_idx].tolist()
            fold_splits[fold] = (train_ids, test_ids)
            self.log.info(
                "Prepared fold %d/%d | train=%d persons | test=%d persons",
                fold, self.n_splits, len(train_ids), len(test_ids),
            )
        return fold_splits


    def _build_test_dataset(
        self,
        model,
        cfg: ModelConfig,
        msp_train: MultiStatePrep,
        msp_test: MultiStatePrep,
    ) -> tuple[MultiStateData, LabelTrafo]:
        trafo = LabelTrafo(n_time=model.n_time)
        trafo.fit_transform(msp_train.states)
        states_test_disc = trafo.transform(msp_test.states)
        dataset = MultiStateData(
            states_df=states_test_disc,
            baseline_df=msp_test.baseline,
            n_nonterminal_states=model.graph.n_nonterminal_states,
            max_path=model.max_path,
            baseline_ftypes=cfg.base_ftypes,
            states_ftypes=cfg.state_ftypes,
        )
        return dataset, trafo

    def _load_fold_inputs(
        self, fold: int,
    ) -> tuple[Any, MultiStatePrep, MultiStatePrep, MultiStateData, LabelTrafo]:
        if fold not in self.fold_splits:
            raise ValueError(f"Fold {fold} not found.")

        train_ids, test_ids = self.fold_splits[fold]
        msp_train = self.msp.subset_persons(
            train_ids, dataset_name=f"{self.msp.dataset_name}_fold{fold}_train",
        )
        msp_test = self.msp.subset_persons(
            test_ids, dataset_name=f"{self.msp.dataset_name}_fold{fold}_test",
        )

        ckpt_path = self.save_dir / f"fold_{fold}" / "best_final_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        model, cfg = ModelConfig.load_model_from_checkpoint(ckpt_path, device=self.device)
        model.eval()

        test_dataset, trafo = self._build_test_dataset(
            model=model, cfg=cfg, msp_train=msp_train, msp_test=msp_test,
        )
        return model, msp_train, msp_test, test_dataset, trafo

    @staticmethod
    def _aggregate_mean_std(df: pd.DataFrame, group_cols: list[str], value_col: str = "value") -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[*group_cols, "mean", "std", "n_folds"])
        out = (
            df.groupby(group_cols, as_index=False)
            .agg(mean=(value_col, "mean"), std=(value_col, "std"), n_folds=("fold", "nunique"))
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
        out["std"] = out["std"].fillna(0.0)
        return out

    @staticmethod
    def _load_external_landmark_benchmark(
        experiment_dir: str | Path,
        method_name: Optional[str] = None,
        start_visit_offset: int = 1,
        landmark_idx_offset: int = 1,
    ) -> pd.DataFrame:
        experiment_dir = Path(experiment_dir)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Benchmark directory not found: {experiment_dir}")

        method = method_name or experiment_dir.name

        for candidate in [
            experiment_dir / "cv_metrics_all_start_visits.csv",
            experiment_dir / "cv_summary_all_start_visits.csv",
        ]:
            if candidate.exists():
                df = pd.read_csv(candidate).copy()
                break
        else:
            parts: list[pd.DataFrame] = []
            for subdir in sorted(experiment_dir.glob("start_visit_*")):
                if not subdir.is_dir():
                    continue
                metrics_path = next(
                    (p for p in [subdir / "cv_metrics.csv", subdir / "cv_summary.csv"] if p.exists()),
                    None,
                )
                if metrics_path is None:
                    continue
                try:
                    start_visit = int(subdir.name.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                part = pd.read_csv(metrics_path).copy()
                if "start_visit" not in part.columns:
                    part["start_visit"] = start_visit
                parts.append(part)

            if not parts:
                raise FileNotFoundError(f"No benchmark metrics found in {experiment_dir}.")
            df = pd.concat(parts, ignore_index=True)

        if "metric" not in df.columns and "metric_name" in df.columns:
            df = df.rename(columns={"metric_name": "metric"})
        if "value" not in df.columns:
            if "mean" in df.columns:
                df = df.rename(columns={"mean": "value"})
            elif "score" in df.columns:
                df = df.rename(columns={"score": "value"})
        if "start_visit" not in df.columns:
            if "landmark_idx" in df.columns:
                df["start_visit"] = pd.to_numeric(df["landmark_idx"], errors="coerce") + start_visit_offset
            elif "landmark" in df.columns:
                tmp = df["landmark"].astype(str).str.extract(r"(\d+)", expand=False)
                df["start_visit"] = pd.to_numeric(tmp, errors="coerce") + start_visit_offset

        missing = {"start_visit", "metric", "value"} - set(df.columns)
        if missing:
            raise ValueError(f"Benchmark missing required columns: {sorted(missing)}")

        df["metric"] = df["metric"].astype(str)
        df["start_visit"] = pd.to_numeric(df["start_visit"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[df["start_visit"].notna() & df["value"].notna()].copy()
        df["start_visit"] = df["start_visit"].astype(int)

        if "fold" not in df.columns:
            df["fold"] = 1
        else:
            df["fold"] = pd.to_numeric(df["fold"], errors="coerce").fillna(1).astype(int)

        df["landmark_idx"] = df["start_visit"] - int(start_visit_offset) + int(landmark_idx_offset)
        df = df[df["landmark_idx"] >= 0].copy()
        df["landmark_idx"] = df["landmark_idx"].astype(int)
        df["landmark"] = df["landmark_idx"].map(lambda k: f"landmark_{int(k)}")
        df["method"] = str(method)

        return (
            df[["method", "metric", "landmark", "landmark_idx", "value", "fold"]]
            .sort_values(["method", "metric", "landmark_idx", "fold"])
            .reset_index(drop=True)
        )

    def get_support_summary(self, report: Optional[CVLandmarkResult] = None) -> pd.DataFrame:
        rpt = self._resolve_report(report)
        return rpt.support_summary()

    def get_support_frame(self, report: Optional[CVLandmarkResult] = None) -> pd.DataFrame:
        rpt = self._resolve_report(report)
        frames = []
        for f in rpt.folds:
            df = f.result.support_frame()
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "fold", f.fold)
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["fold", "state", "state_name", "n_at_risk"])
        return pd.concat(frames, ignore_index=True)


    def get_dcal_summary(
        self,
        state: int = 1,
        report: Optional[CVLandmarkResult] = None,
        endpoint: str = "both",  # "os" | "sojourn" | "both"
        tidy: bool = True,
    ) -> pd.DataFrame | dict:
        """
        Cross-validated D-Calibration summary for one landmark state.
        """
        rpt = self._resolve_report(report)
        endpoint = str(endpoint).lower()
        if endpoint not in {"both", "os", "sojourn"}:
            raise ValueError(f"endpoint must be one of {{'both','os','sojourn'}}, got {endpoint!r}")
        endpoints = ("sojourn", "os") if endpoint == "both" else (endpoint,)

        raw: dict[str, Any] = {}
        rows: list[dict] = []
        for ep in endpoints:
            d = rpt.try_pooled_dcal(state, ep)
            raw[ep] = d
            if d is None:
                continue
            n_folds_total = sum(1 for f in rpt.folds if state in f.result.by_state)
            rows.append({
                "endpoint": "OS" if ep == "os" else "Sojourn",
                "n_folds_total": n_folds_total,
                "n_folds_pooled": d.n_folds_pooled,
                "n_pooled": d.n,
                "n_events_pooled": d.n_events,
                "n_censored_pooled": d.n_censored,
                "statistic": d.statistic,
                "pvalue": d.pvalue,
                "chi2_per_n": d.chi2_per_n,
                "cohens_w": d.cohens_w,
                "r_theta": d.r_theta, 
                "max_abs_deviation": d.max_abs_deviation,
                "mean_abs_deviation": d.mean_abs_deviation,
                "cdf_max_deviation": d.cdf_max_deviation,
                "n_bins": d.n_bins,
            })

        empty_cols = [
            "endpoint", "n_folds_total", "n_folds_pooled", "n_pooled", "n_events_pooled",
            "n_censored_pooled", "statistic", "pvalue", "chi2_per_n", "cohens_w", "r_theta",
            "max_abs_deviation", "mean_abs_deviation", "cdf_max_deviation", "n_bins",
        ]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=empty_cols)
        if not tidy:
            return raw
        return df.set_index("endpoint") if not df.empty else df

    def get_coverage_summary(
        self,
        state: int = 1,
        report: Optional[CVLandmarkResult] = None,
        endpoint: str = "both",  # "os" | "sojourn" | "both"
        tidy: bool = True,
    ) -> pd.DataFrame | dict:
        """
        Cross-validated PICP(alpha) prediction-interval coverage summary for
        one landmark state.
        """
        rpt = self._resolve_report(report)
        endpoint = str(endpoint).lower()
        if endpoint not in {"both", "os", "sojourn"}:
            raise ValueError(f"endpoint must be one of {{'both','os','sojourn'}}, got {endpoint!r}")
        endpoints = ("sojourn", "os") if endpoint == "both" else (endpoint,)

        raw: dict[str, Any] = {}
        rows: list[dict] = []
        for ep in endpoints:
            pc = rpt.try_pooled_pi_cov(state, ep)
            raw[ep] = pc
            if pc is None:
                continue
            label = "OS" if ep == "os" else "Sojourn"
            for a, emp, lb, w in zip(pc.alphas, pc.empirical, pc.det_lower_bound, pc.mean_pi_width):
                rows.append({
                    "endpoint": label, "nominal": float(a),
                    "empirical_coverage": float(emp), "det_lower_bound": float(lb),
                    "mean_PI_width": float(w),
                    "n_folds_pooled": pc.n_folds_pooled,
                    "n_events": pc.n_events, "n_censored": pc.n_censored,
                })

        empty_cols = [
            "endpoint", "nominal", "empirical_coverage", "det_lower_bound",
            "mean_PI_width", "n_folds_pooled", "n_events", "n_censored",
        ]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=empty_cols)
        if not tidy:
            return raw
        return df.set_index(["endpoint", "nominal"]) if not df.empty else df

    def get_metric_summary(
        self,
        state: int = 1,
        report: Optional[CVLandmarkResult] = None,
        endpoint: str = "both",  # "Overall Survival" | "Sojourn Time" | "both"
        method: Optional[str] = "DG-MSM",
    ) -> pd.DataFrame:
        """
        Cross-validated time-to-event discrimination/error metric summary
        (C-index, IPCW C-index, AUC(t), IBS, ...) for one landmark state.
        """
        rpt = self._resolve_report(report)
        frames = self._metric_frames(rpt)

        if endpoint == "both":
            groups = list(frames.keys())
        else:
            if endpoint not in frames:
                raise ValueError(
                    f"endpoint={endpoint!r} not found (available: {list(frames)})."
                )
            groups = [endpoint]

        parts = []
        for g in groups:
            df = frames.get(g)
            if df is None or df.empty:
                continue
            sub = df[df["landmark_idx"] == state]
            if method is not None:
                sub = sub[sub["method"] == method]
            if sub.empty:
                continue
            agg = self._aggregate_mean_std(
                sub, group_cols=["method", "metric", "landmark", "landmark_idx"],
            )
            agg.insert(0, "endpoint", g)
            parts.append(agg)

        if not parts:
            return pd.DataFrame(columns=[
                "endpoint", "method", "metric", "landmark", "landmark_idx", "mean", "std", "n_folds",
            ])

        return (
            pd.concat(parts, ignore_index=True)
            .sort_values(["endpoint", "method", "metric"])
            .reset_index(drop=True)
        )


