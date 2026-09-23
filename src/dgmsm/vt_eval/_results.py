# _results.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .d_cal import _dcal_stats

logger = logging.getLogger(__name__)

_EP_KEYS: dict[str, tuple[str, str]] = {
    "sojourn": ("soj_dcal", "soj_pi"),
    "os":      ("os_dcal",  "os_pi"),
}

def _fold_std(values_per_fold: np.ndarray) -> np.ndarray:
    ddof = 1 if values_per_fold.shape[0] > 1 else 0
    return values_per_fold.std(axis=0, ddof=ddof)



@dataclass(frozen=True)
class DCalResult:
    bins: np.ndarray
    n: int
    n_bins: int
    n_events: int
    n_censored: int
    uncensored_mode: str
    horizon_bins: int
    statistic: float
    pvalue: float
    max_abs_deviation: float
    mean_abs_deviation: float
    cdf_max_deviation: float
    chi2_per_n: float
    n_folds_pooled: int = 1
    bin_proportions_folds: Optional[np.ndarray] = None   # [n_folds, n_bins] - pooled only
    bin_proportions_std: Optional[np.ndarray] = None     # [n_bins]          - pooled only


    @property
    def cohens_w(self) -> float:
        """Cohen's w = sqrt(chi2/n) — sample-size-free standardized effect size."""
        return float(np.sqrt(max(self.chi2_per_n, 0.0)))

    @property
    def r_theta(self) -> float:
        """
        X-CAL calibration score (Goldstein et al., 2020):
            R(theta) = sum_b (p_b - 1/B)^2 = chi2/(n*B) = w^2 / B
        Bin-count- and sample-size-free effect size, directly comparable
        across different B and n.
        """
        return float(self.chi2_per_n / self.n_bins)

    @property
    def bin_proportions(self) -> np.ndarray:
        return self.bins / self.n if self.n else self.bins

    @property
    def target_proportion(self) -> float:
        return 1.0 / self.n_bins

    @classmethod
    def from_d_calibration(cls, d: dict, horizon_bins: int) -> "DCalResult":
        return cls(
            bins=np.asarray(d["bins"], dtype=float),
            n=int(d["n"]), n_bins=int(d["n_bins"]),
            n_events=int(d["n_events"]), n_censored=int(d["n_censored"]),
            uncensored_mode=d["uncensored_mode"], horizon_bins=int(horizon_bins),
            statistic=float(d["statistic"]), pvalue=float(d["pvalue"]),
            max_abs_deviation=float(d["max_abs_deviation"]),
            mean_abs_deviation=float(d["mean_abs_deviation"]),
            cdf_max_deviation=float(d["cdf_max_deviation"]),
            chi2_per_n=float(d["chi2_per_n"]),
        )

    @classmethod
    def pool(cls, results: list["DCalResult"]) -> "DCalResult":
        if not results:
            raise ValueError("DCalResult.pool: empty list.")

        modal_n_bins = max(
            {r.n_bins for r in results},
            key=lambda nb: (sum(r.n_bins == nb for r in results), nb),
        )
        usable = [r for r in results if r.n_bins == modal_n_bins]
        n_dropped = len(results) - len(usable)
        if n_dropped:
            logger.warning(
                "DCalResult.pool: dropping %d/%d fold(s) with n_bins != %d "
                "(pass a fixed d_cal_bins to avoid this).",
                n_dropped, len(results), modal_n_bins,
            )

        pooled_bins = np.sum([r.bins for r in usable], axis=0)
        n_total = int(round(float(pooled_bins.sum())))
        stats = _dcal_stats(pooled_bins, n=n_total, n_bins=modal_n_bins)
        proportions = np.stack([r.bin_proportions for r in usable], axis=0)

        return cls(
            bins=pooled_bins, n=n_total, n_bins=modal_n_bins,
            n_events=sum(r.n_events for r in usable),
            n_censored=sum(r.n_censored for r in usable),
            uncensored_mode=usable[0].uncensored_mode,
            horizon_bins=usable[0].horizon_bins,
            statistic=stats["statistic"], pvalue=stats["pvalue"],
            max_abs_deviation=stats["max_abs_deviation"],
            mean_abs_deviation=stats["mean_abs_deviation"],
            cdf_max_deviation=stats["cdf_max_deviation"],
            chi2_per_n=stats["chi2_per_n"],
            n_folds_pooled=len(usable),
            bin_proportions_folds=proportions,
            bin_proportions_std=_fold_std(proportions),
        )


@dataclass(frozen=True)
class PICoverageResult:
    alphas: np.ndarray
    ipcw_hits: np.ndarray
    ipcw_weight_sum: np.ndarray
    det_hits: np.ndarray
    n_cens_above_U: np.ndarray
    width_sum: np.ndarray
    n_events: int
    n_censored: int
    n_folds_pooled: int = 1
    empirical_folds: Optional[np.ndarray] = None 
    empirical_std: Optional[np.ndarray] = None 

    @property
    def n(self) -> int:
        return self.n_events + self.n_censored

    @property
    def empirical(self) -> np.ndarray:
        """PICP(alpha), IPCW-weighted."""
        return np.divide(self.ipcw_hits, self.ipcw_weight_sum,
                          out=np.full_like(self.ipcw_hits, np.nan),
                          where=self.ipcw_weight_sum > 0)

    @property
    def det_lower_bound(self) -> np.ndarray:
        """Assumption-free lower bound on PICP(alpha)."""
        den = self.n_cens_above_U + self.n_events
        return np.divide(self.det_hits, den,
                          out=np.full(den.shape, np.nan), where=den > 0)

    @property
    def mean_pi_width(self) -> np.ndarray:
        if self.n_events <= 0:
            return np.full_like(self.width_sum, np.nan)
        return self.width_sum / self.n_events

    @classmethod
    def pool(cls, results: list["PICoverageResult"]) -> "PICoverageResult":
        if not results:
            raise ValueError("PICoverageResult.pool: empty list.")
        alphas = results[0].alphas
        if any(not np.array_equal(r.alphas, alphas) for r in results):
            raise ValueError("PICoverageResult.pool: alpha grids differ across folds.")

        def _sum(field: str) -> np.ndarray:
            return np.sum([getattr(r, field) for r in results], axis=0)

        emp_folds = np.stack([r.empirical for r in results], axis=0)

        return cls(
            alphas=alphas,
            ipcw_hits=_sum("ipcw_hits"), ipcw_weight_sum=_sum("ipcw_weight_sum"),
            det_hits=_sum("det_hits"), n_cens_above_U=_sum("n_cens_above_U"),
            width_sum=_sum("width_sum"),
            n_events=sum(r.n_events for r in results),
            n_censored=sum(r.n_censored for r in results),
            n_folds_pooled=len(results),
            empirical_folds=emp_folds,
            empirical_std=_fold_std(emp_folds),
        )

@dataclass(frozen=True)
class EndpointResult:
    """One endpoint ("os" or "sojourn") of one landmark state."""
    endpoint: str
    label: str
    metrics: dict[str, float]
    d_cal: DCalResult
    pi_cov: PICoverageResult


@dataclass(frozen=True)
class StateLandmarkResult:
    """One landmark state, both endpoints, for one (model, dataset) run."""
    state: int
    state_name: str
    n_at_risk: int                             # == len(landmark_ds); computed once, here
    endpoints: dict[str, EndpointResult]       # {"os": ..., "sojourn": ...}
    sim_result: dict[str, Any]                 # raw simulate_from_state() output, for debugging/plots



@dataclass(frozen=True)
class MultiStateLandmarkResult:
    """
    Result of one multi_state_landmark() call: every eligible landmark
    state, both endpoints, for a single (model, dataset) pair.
    """
    by_state: dict[int, StateLandmarkResult]
    name_map: dict[int, str]

    def states(self) -> list[int]:
        return sorted(self.by_state)

    def endpoint(self, ep: str) -> dict[int, EndpointResult]:
        return {s: r.endpoints[ep] for s, r in self.by_state.items()}

    def _metrics_by_state(self, ep: str) -> dict[int, dict[str, float]]:
        return {s: r.endpoints[ep].metrics for s, r in self.by_state.items() if ep in r.endpoints}

    def metrics_frame(self, ep: str) -> pd.DataFrame:
        rows = self._metrics_by_state(ep)
        if not rows:
            return pd.DataFrame(columns=["state", "metric", "value"])
        keys = sorted({k for v in rows.values() for k in v})
        return pd.DataFrame(
            {s: {k: rows[s].get(k, float("nan")) for k in keys} for s in rows}
        ).T.rename_axis("state").reset_index().melt(
            id_vars="state", var_name="metric", value_name="value"
        )

    def metrics_wide_frame(self, ep: str) -> pd.DataFrame:
        rows = self._metrics_by_state(ep)
        if not rows:
            return pd.DataFrame()
        metric_keys = sorted({k for v in rows.values() for k in v})
        return pd.DataFrame([
            {"metric": mk, **{s: rows[s].get(mk, float("nan")) for s in sorted(rows)}}
            for mk in metric_keys
        ])

    def support_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"state": s, "state_name": r.state_name, "n_at_risk": r.n_at_risk}
            for s, r in self.by_state.items()
        ])


@dataclass(frozen=True)
class CVFoldResult:
    fold: int
    n_train: int
    n_test: int
    result: MultiStateLandmarkResult


@dataclass(frozen=True)
class CVLandmarkResult:
    """
    Full cross-validated landmarking result.
    """
    folds: list[CVFoldResult]
    landmark_labels: dict[int, str]

    @classmethod
    def from_single_run(
        cls,
        result: MultiStateLandmarkResult,
        *,
        fold: int = 1,
        n_train: int = 0,
        n_test: int = 0,
    ) -> "CVLandmarkResult":
        # Wrap a single (non-CV) run as a one-fold CV result
        return cls(
            folds=[CVFoldResult(fold=fold, n_train=n_train, n_test=n_test, result=result)],
            landmark_labels=result.name_map,
        )

    def states(self) -> list[int]:
        return sorted({s for f in self.folds for s in f.result.by_state})

    def _leaf(self, state: int, ep: str, which: str):
        for f in self.folds:
            r = f.result.by_state.get(state)
            if r is not None:
                yield getattr(r.endpoints[ep], which)

    def _pool_leaf(self, state: int, ep: str, which: str, pool_fn):
        items = list(self._leaf(state, ep, which))
        return pool_fn(items) if items else None

    def try_pooled_dcal(self, state: int, ep: str):
        return self._pool_leaf(state, ep, "d_cal", DCalResult.pool)

    def try_pooled_pi_cov(self, state: int, ep: str):
        return self._pool_leaf(state, ep, "pi_cov", PICoverageResult.pool)

    def pooled_dcal(self, state: int, ep: str) -> DCalResult:
        value = self.try_pooled_dcal(state, ep)
        if value is None:
            raise ValueError(f"No D-Cal data for state={state}, endpoint={ep!r}")
        return value

    def pooled_pi_cov(self, state: int, ep: str) -> PICoverageResult:
        value = self.try_pooled_pi_cov(state, ep)
        if value is None:
            raise ValueError(f"No PI-coverage data for state={state}, endpoint={ep!r}")
        return value

    def entry(self, state: int) -> dict[str, Any]:
        label = self.landmark_labels.get(state, str(state))
        out: dict[str, Any] = {"state_label": label, "n_folds": len(self.folds)}
        for ep, (dcal_key, pi_key) in _EP_KEYS.items():
            out[dcal_key] = self.try_pooled_dcal(state, ep)
            out[pi_key] = self.try_pooled_pi_cov(state, ep)
        return out

    def metric_summary(self, state: int, ep: str) -> pd.DataFrame:
        rows = [
            {"fold": f.fold, "metric": k, "value": v}
            for f in self.folds
            if (r := f.result.by_state.get(state)) is not None
            for k, v in r.endpoints[ep].metrics.items()
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return (df.groupby("metric")["value"]
                  .agg(mean="mean", std="std")
                  .assign(n_folds=df.groupby("metric")["fold"].nunique())
                  .reset_index())

    def support_summary(self) -> pd.DataFrame:
        """
        Per-landmark support across folds.
        """
        rows = [
            {"fold": f.fold, "state": s, "state_name": r.state_name, "n_at_risk": r.n_at_risk}
            for f in self.folds for s, r in f.result.by_state.items()
        ]
        cols = ["state", "state_name", "n_folds", "total", "median", "min", "max", "mean", "std"]
        if not rows:
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(rows)
        out = (
            df.groupby(["state", "state_name"])["n_at_risk"]
              .agg(n_folds="count", total="sum", median="median",
                   min="min", max="max", mean="mean", std="std")
              .reset_index()
        )
        out["std"] = out["std"].fillna(0.0)
        for col in ["n_folds", "total", "min", "max"]:
            out[col] = out[col].astype(int)
        return out[cols]


    def dcal_fold_frame(
        self,
        state = None,
        ep = None,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for f in self.folds:
            for s, r in f.result.by_state.items():
                if state is not None and s != state:
                    continue
                for ep_key, ep_result in r.endpoints.items():
                    if ep is not None and ep_key != ep:
                        continue
                    d = ep_result.d_cal
                    rows.append(dict(
                        fold=f.fold, state=s, state_name=r.state_name,
                        endpoint="OS" if ep_key == "os" else "Sojourn",
                        n=d.n, n_events=d.n_events, n_censored=d.n_censored,
                        n_bins=d.n_bins,
                        statistic=d.statistic, pvalue=d.pvalue,
                        chi2_per_n=d.chi2_per_n,
                        cohens_w=d.cohens_w,
                        r_theta=d.r_theta,
                        max_abs_deviation=d.max_abs_deviation,
                        mean_abs_deviation=d.mean_abs_deviation,
                        cdf_max_deviation=d.cdf_max_deviation,
                    ))

        cols = [
            "fold", "state", "state_name", "endpoint", "n", "n_events", "n_censored",
            "n_bins", "statistic", "pvalue", "chi2_per_n", "cohens_w", "r_theta",
            "max_abs_deviation", "mean_abs_deviation", "cdf_max_deviation",
        ]
        if not rows:
            return pd.DataFrame(columns=cols)
        return (
            pd.DataFrame(rows, columns=cols)
            .sort_values(["endpoint", "state", "fold"])
            .reset_index(drop=True)
        )

    def dcal_effect_size_summary(self, state: int, ep: str) -> dict[str, Any]:
        pooled = self.try_pooled_dcal(state, ep)
        fdf = self.dcal_fold_frame(state=state, ep=ep)
        return {
            "pooled_w": pooled.cohens_w if pooled else float("nan"),
            "pooled_r_theta": pooled.r_theta if pooled else float("nan"),
            "fold_mean_w": fdf["cohens_w"].mean() if not fdf.empty else float("nan"),
            "fold_std_w": fdf["cohens_w"].std() if not fdf.empty else float("nan"),
            "fold_mean_r_theta": fdf["r_theta"].mean() if not fdf.empty else float("nan"),
            "fold_std_r_theta": fdf["r_theta"].std() if not fdf.empty else float("nan"),
        }