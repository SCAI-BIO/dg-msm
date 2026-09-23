import logging
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List, Tuple
from survmetrics import SurvMetrics


class EvalMuStaRDT:
    def __init__(
        self,
        model,
        dataset,
        device = "cpu",
        censor_surv: str = "km",
        clamp_durations: bool = True,
        logger: Optional[logging.Logger] = None,
        state_names: Optional[List[str]] = None,
        eval_mask: Optional[torch.Tensor] = None,
        min_visits_for_state: int = 10,
        truncation_percentile: float = 95.0,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.censor_surv = censor_surv
        self.clamp_durations = clamp_durations
        self.logger = logger or logging.getLogger(__name__)
        self.state_names = list(state_names) if state_names is not None else None
        self.eval_mask = eval_mask
        self.min_visits_for_state = min_visits_for_state
        self.truncation_percentile = truncation_percentile

        self._metrics: Optional[Dict[int, Dict[str, float]]] = None
        self._auc_curves: Optional[Dict[int, pd.Series]] = None
        self._brier_curves: Optional[Dict[int, pd.Series]] = None

    def compute(
        self,
        times: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict[str, float]]:
        self.logger.info(
            "Starting EvalMuStaRDT.compute(): censor_surv=%s, clamp_durations=%s, truncation_percentile=%s",
            self.censor_surv,
            self.clamp_durations,
            self.truncation_percentile,
        )

        metrics: Dict[int, Dict[str, float]] = {}
        auc_curves: Dict[int, pd.Series] = {}
        brier_curves: Dict[int, pd.Series] = {}

        states, evt, dur_base, S = self._prep_labels()
        device = states.device

        valid = (states != 0)

        if self.eval_mask is not None:
            mask = self.eval_mask.to(device)
            if mask.shape != valid.shape:
                raise ValueError(
                    f"eval_mask shape {mask.shape} != states shape {valid.shape}"
                )
            valid = valid & mask

        N, max_path, max_time = S.shape
        v_dim = getattr(self.model, "v_dim", int(states.max().item()))

        states_flat = states.view(-1)
        evt_flat = evt.view(-1)
        dur_flat = dur_base.view(-1)
        valid_flat = valid.view(-1)
        S_flat = S.view(-1, max_time)

        surv_cols_by_state: list[np.ndarray] = [np.empty((0, 0)) for _ in range(v_dim)]
        durations_by_state: list[np.ndarray] = [np.empty((0,), dtype=int) for _ in range(v_dim)]
        events_by_state: list[np.ndarray] = [np.empty((0,), dtype=int) for _ in range(v_dim)]

        for s_idx in range(v_dim):
            state_id = s_idx + 1
            mask_s = valid_flat & (states_flat == state_id)

            n_valid = int(mask_s.sum().item())
            if n_valid == 0:
                self.logger.debug("State %d: no valid samples, skipping", s_idx)
                continue

            idx_flat = mask_s.nonzero(as_tuple=False).squeeze(-1)

            S_s = S_flat[idx_flat, :]
            evt_s = evt_flat[idx_flat]
            dur_s = dur_flat[idx_flat]

            d_out, e_out = self._transform_durations_events(
                dur_base=dur_s,
                evt=evt_s,
                max_time=max_time,
                state_idx=s_idx,
            )

            surv_mat = S_s.transpose(0, 1).cpu().numpy()
            durations_np = d_out.cpu().numpy().astype(int)
            events_np = e_out.cpu().numpy().astype(int)

            surv_cols_by_state[s_idx] = surv_mat
            durations_by_state[s_idx] = durations_np
            events_by_state[s_idx] = events_np

            self.logger.debug(
                "State %d: collected %d samples", s_idx, len(durations_by_state[s_idx])
            )

        for s_idx in range(v_dim):
            durations_np = durations_by_state[s_idx]
            if durations_np.size == 0:
                continue

            surv_mat = surv_cols_by_state[s_idx]
            events_np = events_by_state[s_idx]

            surv_df = pd.DataFrame(
                surv_mat,
                index=np.arange(surv_mat.shape[0], dtype=float),
            )

            ev_eval = SurvMetrics(
                surv=surv_df,
                durations=durations_np,
                events=events_np,
                censor_surv=self.censor_surv,
                logger=self.logger,
            )

            times_s = surv_df.index.values.astype(float) if times is None else np.asarray(times, dtype=float)
            times_s = np.unique(times_s)

            # Restrict to prediction horizon
            times_s = times_s[
                (times_s >= float(surv_df.index.min())) &
                (times_s <= float(surv_df.index.max()))
            ]

            # Restrict to observed support for this state
            t_min = float(durations_np.min())
            t_max = float(durations_np.max())
            times_s = times_s[(times_s >= t_min) & (times_s <= t_max)]

            if times_s.size == 0:
                times_s = surv_df.index.values.astype(float)

            metrics_s, auc_series, brier_series = ev_eval.compute(
                time_grid=times_s,
                percentile=self.truncation_percentile,
                return_curves=True,
            )

            vals = auc_series.to_numpy(dtype=float)
            n_nan_auc = int(np.isnan(vals).sum())
            n_finite_auc = int(np.isfinite(vals).sum())
            if n_nan_auc > 0:
                self.logger.debug(
                    "State %d: AUC(t) contains NaN values (#NaN=%d, #finite=%d).",
                    s_idx,
                    n_nan_auc,
                    n_finite_auc,
                )

            metrics[s_idx] = metrics_s
            auc_curves[s_idx] = auc_series
            brier_curves[s_idx] = brier_series

            label = self._state_label(s_idx)
            metrics_str = ", ".join(
                f"{name}={self._fmt_metric(val)}"
                for name, val in metrics_s.items()
            )
            self.logger.info("State %s: %s", label, metrics_str)

        self._metrics = metrics
        self._auc_curves = auc_curves
        self._brier_curves = brier_curves

        self.logger.info("EvalMuStaRDT.compute() finished.")
        return metrics

    @staticmethod
    def _fmt_metric(v, fmt: str = ".4f") -> str:
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return str(v)

    def _state_label(self, idx: int) -> str:
        if self.state_names is not None and 0 <= idx < len(self.state_names):
            return self.state_names[idx]
        return f"state_{idx}"

    def _prep_labels(self):
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        ds = self.dataset

        x_base = torch.stack(ds.x_base, dim=0).to(device)
        m_base = torch.stack(ds.m_base, dim=0).to(device)
        x_state = torch.stack(ds.x_state, dim=0).to(device)
        m_state = torch.stack(ds.m_state, dim=0).to(device)
        labels = torch.stack(ds.labels, dim=0).to(device)

        states = labels[..., 0].long()
        dur_base = labels[..., 1].long()
        evt = labels[..., 2].long()

        valid = (states != 0)

        if self.eval_mask is not None:
            mask = self.eval_mask.to(device)
            if mask.shape != valid.shape:
                raise ValueError(
                    f"eval_mask shape {mask.shape} != states shape {valid.shape}"
                )
            valid = valid & mask

        states_flat = states.view(-1)
        evt_flat = evt.view(-1)
        valid_flat = valid.view(-1)

        v_dim = getattr(self.model, "v_dim", int(states_flat.max().item()))

        for s_idx in range(v_dim):
            state_id = s_idx + 1
            mask_s = valid_flat & (states_flat == state_id)

            n_visits = int(mask_s.sum().item())
            if n_visits == 0:
                self.logger.warning(
                    "EvalMuStaRDT: no valid test visits for state %d; metrics for this state will be omitted.",
                    state_id,
                )
                continue

            n_events = int((evt_flat[mask_s] > 0).sum().item())

            if n_events == 0:
                self.logger.warning(
                    "EvalMuStaRDT: state %d has %d valid test visits but no events; concordance metrics may be undefined or NaN.",
                    state_id,
                    n_visits,
                )
            elif n_visits < self.min_visits_for_state:
                self.logger.warning(
                    "EvalMuStaRDT: state %d has only %d valid test visits (events=%d); metrics may be unstable.",
                    state_id,
                    n_visits,
                    n_events,
                )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                S = self.model.predict_survival_eval(
                    x_base=x_base,
                    m_base=m_base,
                    x_state=x_state,
                    m_state=m_state,
                )
        finally:
            if was_training:
                self.model.train()

        return states, evt, dur_base, S

    def _transform_durations_events(
        self,
        dur_base: torch.Tensor,
        evt: torch.Tensor,
        max_time: int,
        state_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dur_raw = dur_base.long()
        in_range = (dur_raw >= 0) & (dur_raw < max_time)

        e_out = ((evt > 0) & in_range).long()

        out_of_range = ~in_range
        n_out = int(out_of_range.sum().item())
        n_out_events = int(((evt > 0) & out_of_range).sum().item())
        if n_out > 0:
            self.logger.info(
                "State %d: %d/%d observations (including %d events) have durations outside [0, %d) and are treated as censored w.r.t. the prediction horizon.",
                state_idx,
                n_out,
                int(dur_raw.numel()),
                n_out_events,
                max_time,
            )

        if self.clamp_durations:
            d_out = dur_raw.clamp(min=0, max=max_time - 1)
        else:
            d_out = dur_raw

        return d_out, e_out

    @property
    def metrics(self) -> Dict[int, Dict[str, float]]:
        if self._metrics is None:
            self.compute()
        return self._metrics

    @property
    def metrics_df(self) -> pd.DataFrame:
        metrics = self.metrics
        if not metrics:
            return pd.DataFrame()

        rows = []
        metric_keys = list(next(iter(metrics.values())).keys())
        for metric_key in metric_keys:
            row = {"metric": metric_key}
            for state_idx, vals in sorted(metrics.items()):
                label = self._state_label(state_idx)
                row[label] = vals[metric_key]
            rows.append(row)

        return pd.DataFrame(rows)

    @property
    def auc_curves(self) -> Dict[int, pd.Series]:
        if self._metrics is None:
            self.compute()
        return self._auc_curves or {}

    @property
    def brier_curves(self) -> Dict[int, pd.Series]:
        if self._metrics is None:
            self.compute()
        return self._brier_curves or {}

    def __call__(
        self,
        times: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict[str, float]]:
        return self.compute(times=times)

    def __str__(self) -> str:
        df = self.metrics_df
        if df is None or df.empty:
            return "EvalMuStaRDT: no metrics available."
        return df.to_string(index=False)

    def worst_metric(
        self,
        metric: str = "Antolini",
        direction: str = "min",
    ) -> float:
        if direction not in ("min", "max"):
            raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")

        metrics = getattr(self, "metrics", None) or {}
        if not metrics:
            return float("nan")

        values: list[float] = []
        for state_idx, m in metrics.items():
            if metric not in m:
                raise KeyError(f"Metric {metric!r} missing for state {state_idx}")
            v = m[metric]
            if v is None:
                continue
            v = float(v)
            if not np.isnan(v):
                values.append(v)

        if not values:
            return float("nan")

        return min(values) if direction == "min" else max(values)

    def mean_metric(
        self,
        metric: str = "adj Antolini",
    ) -> float:
        metrics = self.metrics
        if not metrics:
            return float("nan")

        vals: list[float] = []
        for _, m in metrics.items():
            v = m.get(metric)
            if v is None:
                continue
            v = float(v)
            if not np.isnan(v):
                vals.append(v)

        return float(np.mean(vals)) if vals else float("nan")