from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from survmetrics import SurvMetrics
from survmetrics.plot import plot_td_metric_per_state
from dgmsm import ModelConfig
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


METRIC_DIRECTIONS = {
    "Antolini": "maximize",
    "adj Antolini": "maximize",
    "Antolini IPCW": "maximize",
    "adj Antolini IPCW": "maximize",
    "AUC": "maximize",
    "IBS": "minimize",
    "INBLL": "minimize",
}

@dataclass
class TestResult:
    metrics: Union[pd.DataFrame, dict[str, Any]]
    auc_curves: dict[str, Any] = field(default_factory=dict)
    brier_curves: dict[str, Any] = field(default_factory=dict)
    train_stats: dict[str, Any] = field(default_factory=dict)
    fold: Optional[int ] = None
    best_params: Optional[Any ] = None
    model_cfg: Optional[ModelConfig ] = None
    final_grid_times: Optional[np.ndarray] = None


def _restrict_time_grid(surv: pd.DataFrame, durations: np.ndarray) -> np.ndarray:
    grid = np.asarray(surv.index, dtype=float)
    if durations.size == 0:
        return grid[:0]
    return grid[(grid >= durations.min()) & (grid <= durations.max())]


def _log_eval_summary(
    logger: logging.Logger,
    state_label: str,
    surv: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
) -> None:
    n = len(durations)
    n_events = int(events.sum())
    n_cens = n - n_events

    s_vals = surv.values
    non_monotone = bool(
        s_vals.shape[0] > 1 and (np.diff(s_vals, axis=0) > 1e-4).any()
    )

    logger.info(
        "[Eval] state=%s | n_test=%d | events=%d (%.2f%%) | censored=%d | "
        "t_min=%.3f | t_max=%.3f | surv_shape=%s | S_min=%.3f | S_max=%.3f | non_monotone=%s",
        state_label,
        n,
        n_events,
        (100.0 * n_events / n) if n else 0.0,
        n_cens,
        float(durations.min()) if n else float("nan"),
        float(durations.max()) if n else float("nan"),
        s_vals.shape,
        float(np.nanmin(s_vals)) if s_vals.size else float("nan"),
        float(np.nanmax(s_vals)) if s_vals.size else float("nan"),
        non_monotone,
    )


def _plot_metric_curves(
    per_fold: list[TestResult],
    curves_attr: str,
    metric_name: str,
    file_name: str,
    suptitle: str,
    state_names: list[str],
    time_delta: int,
    show: bool,
    save_path: Optional[Path],
    y_lim: Optional[tuple[float, float]] = None,
) -> None:
    curves_list = []
    grid_times_list = []

    for fr in per_fold:
        curves = getattr(fr, curves_attr, None)
        if curves:
            curves_list.append(curves)
            grid_times_list.append(fr.final_grid_times)

    if not curves_list:
        return

    fig, _ = plot_td_metric_per_state(
        curves=curves_list,
        metric_name=metric_name,
        state_labels=state_names,
        ncols=2,
        show_folds=False,
        time_delta=time_delta,
        time_label="years",
        suptitle=suptitle,
        y_lim=y_lim,
        grid_times_list=grid_times_list,
    )

    if save_path is not None:
        fig.savefig(save_path / file_name, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_curves(
    results: dict[str, Any],
    suptitle: str,
    state_names: list[str],
    time_delta: int = 1,
    show: bool = True,
    save_to = None,
) -> None:
    per_fold: list[TestResult] = results["per_fold"]

    save_path = Path(save_to) if save_to is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    _plot_metric_curves(
        per_fold=per_fold,
        curves_attr="auc_curves",
        metric_name="AUC(t)",
        file_name="auc_per_state.png",
        suptitle=suptitle,
        state_names=state_names,
        time_delta=time_delta,
        show=show,
        save_path=save_path,
        y_lim=(0.0, 1.0),
    )

    _plot_metric_curves(
        per_fold=per_fold,
        curves_attr="brier_curves",
        metric_name="Brier score",
        file_name="brier_per_state.png",
        suptitle=suptitle,
        state_names=state_names,
        time_delta=time_delta,
        show=show,
        save_path=save_path,
    )


def cv_summary(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    long = df[["metric", "state", "value"]]
    grouped = long.groupby(["state", "metric"])["value"].agg(["mean", "std"])

    summary = grouped.unstack("metric")
    summary.columns = [
        f"{metric}_{stat}" for stat, metric in summary.columns.to_flat_index()
    ]
    return summary.reset_index()


def get_msm_eval(
    model,
    data_loader,
    device: torch.device,
    fold: Optional[int] = None,
    state_names: Optional[list[str]] = None,
    eval_mask: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Evaluate survival predictions for all states."""
    evaluator = model.eval_surv(
        data_loader=data_loader,
        device=device,
        state_names=state_names,
        eval_mask=eval_mask,
    )
    metrics = evaluator.metrics
    auc_curves = evaluator.auc_curves or {}
    brier_curves = evaluator.brier_curves or {}
    metrics_df = evaluator.metrics_df.copy()

    if fold is not None:
        metrics_df.insert(0, "fold", fold)

    return metrics, metrics_df, auc_curves, brier_curves


def get_state_eval(
    surv: pd.DataFrame,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    state_name: Optional[str] = None,
    log: Optional[logging.Logger] = None,
    truncation_percentile: float = 95.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    logger = log or logging.getLogger(__name__)
    state_label = state_name or "unknown"

    durations_test = np.asarray(durations_test, dtype=float)
    events_test = (np.asarray(events_test) != 0).astype(int)

    _log_eval_summary(logger, state_label, surv, durations_test, events_test)

    if durations_test.size == 0:
        empty_metrics = pd.DataFrame(columns=["metric", "value"])
        empty_curve = pd.Series(dtype=float)
        return empty_metrics, empty_curve, empty_curve

    time_grid = _restrict_time_grid(surv, durations_test)
    if time_grid.size == 0:
        time_grid = np.asarray(surv.index, dtype=float)

    ev = SurvMetrics(
        surv=surv,
        durations=durations_test,
        events=events_test,
        censor_surv="km",
        logger=logger,
    )
    metrics, state_auc, state_brier = ev.compute(
        time_grid=time_grid,
        percentile=truncation_percentile,
        return_curves=True,
    )

    metrics_df = pd.DataFrame({
        "metric": list(metrics.keys()),
        "value": list(metrics.values()),
    })
    return metrics_df, state_auc, state_brier