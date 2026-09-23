
import warnings
import numpy as np
import pandas as pd
from .concordance import concordance_td, concordance_td_ipcw
from . import admin
from . import ipcw
from sklearn.metrics import roc_auc_score
from .utils import kaplan_meier, idx_at_times
import scipy
import scipy.stats
import scipy.integrate
import logging
from collections.abc import Callable
from typing import Any

# This SurvMetrics class is based on pycox.evaluation.EvalSurv with the following additions:
# - Import: from sklearn.metrics import roc_auc_score
# - New methods include
#     * SurvMetrics.auc_ipcw(...)
#     * SurvMetrics.integrated_auc_ipcw(...)

class SurvMetrics:
    """Class for evaluating predictions.
    
    Arguments:
        surv {pd.DataFrame} -- Survival predictions.
        durations {np.array} -- Durations of test set.
        events {np.array} -- Events of test set.

    Keyword Arguments:
        censor_surv {str, pd.DataFrame, SurvMetrics} -- Censoring distribution.
            If provided data frame (survival function for censoring) or SurvMetrics object,
            this will be used. 
            If 'km', we will fit a Kaplan-Meier to the dataset.
            (default: {None})
        censor_durations {np.array}: -- Administrative censoring times. (default: {None})
        steps {str} -- For durations between values of `surv.index` choose the higher index 'pre'
            or lower index 'post'. For a visualization see `help(SurvMetrics.steps)`. (default: {'post'})
    """
    SAFE_EXCEPTIONS = (ZeroDivisionError, ValueError, FloatingPointError)

    def __init__(
        self,
        surv,
        durations,
        events,
        censor_surv='km',
        censor_durations=None,
        steps='post',
        logger = None,
    ):
        assert (type(durations) == type(events) == np.ndarray), \
            'Need `durations` and `events` to be arrays'

        self.logger = logger or logging.getLogger(__name__)
        self.surv = surv
        self.durations = durations
        self.events = events
        self.steps = steps
        self.censor_durations = censor_durations
        self.censor_surv = censor_surv

        assert pd.Series(self.index_surv).is_monotonic_increasing

    @property
    def censor_surv(self):
        """Estimated survival for censorings. 
        Also an SurvMetrics object.
        """
        return self._censor_surv

    @censor_surv.setter
    def censor_surv(self, censor_surv):
        if isinstance(censor_surv, SurvMetrics):
            self._censor_surv = censor_surv
        elif type(censor_surv) is str:
            if censor_surv == 'km':
                self.add_km_censor()
            else:
                raise ValueError(f"censor_surv cannot be {censor_surv}. Use e.g. 'km'")
        elif censor_surv is not None:
            self.add_censor_est(censor_surv)
        else:
            self._censor_surv = None

    @property
    def index_surv(self):
        return self.surv.index.values

    @property
    def steps(self):
        """How to handle predictions that are between two indexes in `index_surv`.

        For a visualization, run the following:
            ev = SurvMetrics(pd.DataFrame(np.linspace(1, 0, 7)), np.empty(7), np.ones(7), steps='pre')
            ax = ev[0].plot_surv()
            ev.steps = 'post'
            ev[0].plot_surv(ax=ax, style='--')
            ax.legend(['pre', 'post'])
        """
        return self._steps

    @steps.setter
    def steps(self, steps):
        vals = ['post', 'pre']
        if steps not in vals:
            raise ValueError(f"`steps` needs to be {vals}, got {steps}")
        self._steps = steps

    def add_censor_est(self, censor_surv, steps='post'):
        """Add censoring estimates so one can use inverse censoring weighting.
        `censor_surv` are the survival estimates trained on (durations, 1-events),
        
        Arguments:
            censor_surv {pd.DataFrame} -- Censor survival curves.

        Keyword Arguments:
            round {str} -- For durations between values of `surv.index` choose the higher index 'pre'
                            or lower index 'post'. If `None` use `self.steps` (default: {None})
        """
        if not isinstance(censor_surv, SurvMetrics):
            censor_surv = self._constructor(
                censor_surv,
                self.durations,
                1 - self.events,
                None,
                steps=steps,
                logger=self.logger,
            )
        self.censor_surv = censor_surv
        return self

    def add_km_censor(self, steps='post'):
        """Add censoring estimates obtained by Kaplan-Meier on the test set
        (durations, 1-events).
        """
        km = kaplan_meier(self.durations, 1-self.events)
        surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(self.durations), axis=1),
                            index=km.index)
        return self.add_censor_est(surv, steps)

    @property
    def censor_durations(self):
        """Administrative censoring times."""
        return self._censor_durations
    
    @censor_durations.setter
    def censor_durations(self, val):
        if val is not None:
            assert (self.durations[self.events == 0] == val[self.events == 0]).all(),\
                'Censored observations need same `durations` and `censor_durations`'
            assert (self.durations[self.events == 1] <= val[self.events == 1]).all(),\
                '`durations` cannot be larger than `censor_durations`'
            if (self.durations == val).all():
                warnings.warn("`censor_durations` are equal to `durations`." +
                              " `censor_durations` are likely wrong!")
            self._censor_durations = val
        else:
            self._censor_durations = val

    @property
    def _constructor(self):
        return SurvMetrics

    def __getitem__(self, index):
        if not (hasattr(index, '__iter__') or type(index) is slice):
            index = [index]
        surv = self.surv.iloc[:, index]
        durations = self.durations[index]
        events = self.events[index]
        new = self._constructor(
            surv,
            durations,
            events,
            None,
            steps=self.steps,
            logger=self.logger,
        )
        if self.censor_surv is not None:
            new.censor_surv = self.censor_surv[index]
        return new

    def plot_surv(self, **kwargs):
        """Plot survival estimates. 
        kwargs are passed to `self.surv.plot`.
        """
        if len(self.durations) > 50:
            raise RuntimeError("We don't allow to plot more than 50 lines. Use e.g. `ev[1:5].plot()`")
        if 'drawstyle' in kwargs:
            raise RuntimeError(f"`drawstyle` is set by `self.steps`. Remove from **kwargs")
        return self.surv.plot(drawstyle=f"steps-{self.steps}", **kwargs)

    def idx_at_times(self, times):
        """Get the index (iloc) of the `surv.index` closest to `times`.
        I.e. surv.loc[tims] (almost)= surv.iloc[idx_at_times(times)].

        Useful for finding predictions at given durations.
        """
        return idx_at_times(self.index_surv, times, self.steps)

    def _duration_idx(self):
        return self.idx_at_times(self.durations)

    def surv_at_times(self, times):
        idx = self.idx_at_times(times)
        return self.surv.iloc[idx]

    # def prob_alive(self, time_grid):
    #     return self.surv_at_times(time_grid).values

    def concordance_td(self, method='adj_antolini'):
        """Time dependent concorance index from
        Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A time-dependent discrimination
        index for survival data. Statistics in Medicine 24:3927–3944.

        If 'method' is 'antolini', the concordance from Antolini et al. is computed.
    
        If 'method' is 'adj_antolini' (default) we have made a small modifications
        for ties in predictions and event times.
        We have followed step 3. in Sec 5.1. in Random Survival Forests paper, except for the last
        point with "T_i = T_j, but not both are deaths", as that doesn't make much sense.
        See 'metrics._is_concordant'.

        Keyword Arguments:
            method {str} -- Type of c-index 'antolini' or 'adj_antolini' (default {'adj_antolini'}).

        Returns:
            float -- Time dependent concordance index.
        """
        return concordance_td(self.durations, self.events, self.surv.values,
                              self._duration_idx(), method)

    def brier_score(self, time_grid, max_weight=np.inf):
        """Brier score weighted by the inverse censoring distribution.
        See Section 3.1.2 or [1] for details of the wighting scheme.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute Brier score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bs = ipcw.brier_score(time_grid, self.durations, self.events, self.surv.values,
                              self.censor_surv.surv.values, self.index_surv,
                              self.censor_surv.index_surv, max_weight, True, self.steps,
                              self.censor_surv.steps)
        return pd.Series(bs, index=time_grid).rename('brier_score')

    def nbll(self, time_grid, max_weight=np.inf):
        """Negative binomial log-likelihood weighted by the inverse censoring distribution.
        See Section 3.1.2 or [1] for details of the wighting scheme.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute the score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bll = ipcw.binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                           self.censor_surv.surv.values, self.index_surv,
                                           self.censor_surv.index_surv, max_weight, True, self.steps,
                                           self.censor_surv.steps)
        return pd.Series(-bll, index=time_grid).rename('nbll')

    def integrated_brier_score(self, time_grid, max_weight=np.inf):
        """Integrated Brier score weighted by the inverse censoring distribution.
        Essentially an integral over values obtained from `brier_score(time_grid, max_weight)`.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError("Need to add censor_surv to compute briser score. Use 'add_censor_est'")
        return ipcw.integrated_brier_score(time_grid, self.durations, self.events, self.surv.values,
                                           self.censor_surv.surv.values, self.index_surv,
                                           self.censor_surv.index_surv, max_weight, self.steps,
                                           self.censor_surv.steps)

    def integrated_nbll(self, time_grid, max_weight=np.inf):
        """Integrated negative binomial log-likelihood weighted by the inverse censoring distribution.
        Essentially an integral over values obtained from `nbll(time_grid, max_weight)`.
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError("Need to add censor_surv to compute the score. Use 'add_censor_est'")
        ibll = ipcw.integrated_binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                                       self.censor_surv.surv.values, self.index_surv,
                                                       self.censor_surv.index_surv, max_weight, self.steps,
                                                       self.censor_surv.steps)
        return -ibll

    def brier_score_admin(self, time_grid):
        """The Administrative Brier score proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event. 
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError("Need to provide `censor_durations` (censoring durations) to use this method")
        bs = admin.brier_score(time_grid, self.durations, self.censor_durations, self.events,
                               self.surv.values, self.index_surv, True, self.steps)
        return pd.Series(bs, index=time_grid).rename('brier_score')

    def integrated_brier_score_admin(self, time_grid):
        """The Integrated administrative Brier score proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event. 
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError("Need to provide `censor_durations` (censoring durations) to use this method")
        ibs = admin.integrated_brier_score(time_grid, self.durations, self.censor_durations, self.events,
                                           self.surv.values, self.index_surv, self.steps)
        return ibs

    def nbll_admin(self, time_grid):
        """The negative administrative binomial log-likelihood proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event. 
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError("Need to provide `censor_durations` (censoring durations) to use this method")
        bll = admin.binomial_log_likelihood(time_grid, self.durations, self.censor_durations, self.events,
                                           self.surv.values, self.index_surv, True, self.steps)
        return pd.Series(-bll, index=time_grid).rename('nbll')

    def integrated_nbll_admin(self, time_grid):
        """The Integrated negative administrative binomial log-likelihood score proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event. 
        
        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError("Need to provide `censor_durations` (censoring durations) to use this method")
        ibll = admin.integrated_binomial_log_likelihood(time_grid, self.durations, self.censor_durations,
                                                        self.events, self.surv.values, self.index_surv,
                                                        self.steps)
        return -ibll

    def auc_ipcw(
        self,
        time_grid = None,
        max_weight: float = np.inf,
        min_cases: int = 5,
        min_controls: int = 5,
    ) -> pd.Series:
        """
        Time-dependent AUC(t) using IPCW (PyCox-style) for this SurvMetrics object.

        Differences from original PyCox:
        - This method does not exist in upstream PyCox.
        - It uses ipcw.brier_score(..., reduce=False) to obtain IPCW weights

          per (time, individual) and then computes weighted ROC AUC(t) using
          sklearn.metrics.roc_auc_score.

        Behaviour:
        - Uses self.surv (predicted survival for TEST individuals).
        - Uses self.durations, self.events (TEST data).
        - Uses self.censor_surv; if None, calls self.add_km_censor() to fit a

          Kaplan-Meier censoring model on the TEST set (consistent with Brier).
        - Evaluates AUC(t) only on an informative window:

              t in [t_min_event, t_max_duration],
          where t_min_event is the earliest event time in TEST.
        - Returns a pd.Series indexed by time_grid, with NaN at times where

          AUC(t) is undefined (e.g. no cases/controls or too few).

        Parameters
        ----------
        time_grid : np.ndarray or None
            Times at which to evaluate AUC(t). If None, uses self.index_surv.
        max_weight : float
            Max IPCW weight (passed to ipcw.brier_score).
        min_cases : int
            Minimum number of (weighted) cases required at a time t
            to attempt computing AUC(t).
        min_controls : int
            Minimum number of (weighted) controls required at a time t.
        """
        # ensure censor_surv exists (KM on TEST if not provided)
        if self.censor_surv is None:
            self.add_km_censor()

        # default time_grid: survival index
        if time_grid is None:
            time_grid = self.index_surv
        time_grid = np.asarray(time_grid, dtype=float)

        durations = self.durations.astype(float)
        events = self.events.astype(int)

        # informative window: only times where AUC(t) can in principle be defined
        event_times = durations[events == 1]
        if event_times.size == 0:
            # no events -> AUC not defined anywhere
            return pd.Series(np.nan, index=time_grid).rename("auc")

        t_min = float(event_times.min())
        t_max = float(durations.max())

        mask_window = (time_grid >= t_min) & (time_grid <= t_max)
        time_grid_inf = time_grid[mask_window]
        if time_grid_inf.size == 0:
            return pd.Series(np.nan, index=time_grid).rename("auc")

        # survival at those times, respecting self.steps
        surv_at_grid = self.surv_at_times(time_grid_inf).values  # (len(time_grid_inf), N)
        risk_mat = 1.0 - surv_at_grid                            # risk(t) = 1 - S(t)

        # IPCW weights via ipcw.brier_score(..., reduce=False)
        # brier_score returns (scores, weights) when reduce=False
        _, weights = ipcw.brier_score(
            time_grid_inf,
            durations,
            events,
            self.surv.values,
            self.censor_surv.surv.values,
            self.index_surv,
            self.censor_surv.index_surv,
            max_weight,
            False,                  # reduce=False -> (scores, weights)
            self.steps,
            self.censor_surv.steps,
        )
        # weights.shape == (len(time_grid_inf), N)

        auc_vals_inf = np.full(time_grid_inf.shape[0], np.nan, dtype=float)

        for k, t in enumerate(time_grid_inf):
            w = weights[k]         # (N,)
            valid = w > 0
            if not valid.any():
                continue

            # dynamic cases/controls at time t
            cases = valid & (events == 1) & (durations <= t)
            controls = valid & ((events == 0) | (durations > t))

            n_cases = int(cases.sum())
            n_controls = int(controls.sum())

            if n_cases < min_cases or n_controls < min_controls:
                continue  # leave NaN

            y_true = np.concatenate([
                np.ones(n_cases, dtype=int),
                np.zeros(n_controls, dtype=int),
            ])
            scores = np.concatenate([
                risk_mat[k, cases],
                risk_mat[k, controls],
            ])
            w_use = np.concatenate([
                w[cases],
                w[controls],
            ])

            try:
                auc_vals_inf[k] = roc_auc_score(y_true, scores, sample_weight=w_use)
            except ValueError:
                # keep NaN if ROC cannot be computed
                pass

        # map back to full time_grid: NaN outside informative window
        auc_full = np.full(time_grid.shape[0], np.nan, dtype=float)
        auc_full[mask_window] = auc_vals_inf

        return pd.Series(auc_full, index=time_grid).rename("auc")


    def integrated_auc_ipcw(
        self,
        time_grid = None,
        max_weight: float = np.inf,
        min_cases: int = 5,
        min_controls: int = 5,
    ) -> float:
        """
        Mean time-dependent AUC over the informative window, using IPCW.

        Differences from original PyCox:
        - This method does not exist in upstream PyCox.
        - It is analogous to `integrated_brier_score`, but uses AUC(t) instead
          of Brier.

        Behaviour:
        - Calls `self.auc_ipcw(...)`.
        - Returns the average of finite AUC(t) values.
        - Returns NaN if no finite AUC(t) can be computed.

        Parameters
        ----------
        time_grid : np.ndarray or None
            Times at which to evaluate AUC(t). If None, uses self.index_surv.
        max_weight : float
            Max IPCW weight (passed to auc_ipcw/ipcw.brier_score).
        min_cases : int
            Minimum number of (weighted) cases at a time t.
        min_controls : int
            Minimum number of (weighted) controls at a time t.
        """
        auc_series = self.auc_ipcw(
            time_grid=time_grid, max_weight=max_weight,
            min_cases=min_cases, min_controls=min_controls,
        )
        vals = auc_series.values.astype(float)
        times = np.asarray(auc_series.index, dtype=float)
        finite = np.isfinite(vals)
        if not finite.any():
            return float("nan")

        # Use only finite values for integration
        finite_times = times[finite]
        finite_vals = vals[finite]

        if len(finite_times) < 2:
            return float(finite_vals[0])

        integral = scipy.integrate.simpson(finite_vals, x=finite_times)
        return integral / (finite_times[-1] - finite_times[0])

    def _ipcw_weights_event_time(self, eps: float = 1e-8) -> np.ndarray:
        """
        IPCW weights w_i = delta_i / G_hat(X_i)^2 using self.censor_surv.
        """
        if self.censor_surv is None:
            self.add_km_censor()

        return ipcw.ipcw_event_weights_from_censor_surv(
            self.durations,
            self.events,
            self.censor_surv.surv.values,
            self.censor_surv.index_surv,
            steps_censor=self.censor_surv.steps,
            eps=eps,
        )

    def concordance_td_ipcw(self, method: str = 'adj_antolini') -> float:
        """
        IPCW-weighted time-dependent C-index (Antolini-type).
        """
        if self.censor_surv is None:
            self.add_km_censor()

        w = self._ipcw_weights_event_time()
        return concordance_td_ipcw(
            self.durations,
            self.events,
            self.surv.values,
            self._duration_idx(),
            w,
            method=method,
        )

    def _compute_ipcw_max_weight(self, percentile: float = 95.0) -> float:
        """
        Compute a data-driven max_weight by examining the censoring distribution.
        
        Strategy: Evaluate G(t) at all event times, compute 1/G(t) weights,
        and cap at the given percentile.

        Parameters
        ----------
        percentile : float
            Percentile of the weight distribution to use as cap (default: 95).

        Returns
        -------
        float
            max_weight value to pass to brier_score / nbll / integrated_* methods.
        """
        if self.censor_surv is None:
            self.add_km_censor()

        # Get G(t) at each event time
        event_mask = self.events == 1
        event_durations = self.durations[event_mask]

        if event_durations.size == 0:
            return np.inf

        # G(t_i) for event individuals
        idx = idx_at_times(
            self.censor_surv.index_surv,
            event_durations,
            'pre',
            assert_sorted=True,
        )
        if self.censor_surv.steps == 'post':
            idx = (idx - 1).clip(0)

        # For marginal KM censor, all columns are identical → take column 0
        G_vals = self.censor_surv.surv.values[idx, 0]
        G_vals = np.maximum(G_vals, 1e-8)

        # Weights ~ 1/G(t)
        weights = 1.0 / G_vals

        # Cap at percentile
        cap = float(np.percentile(weights, percentile))
        return cap

    def integrated_brier_score_truncated(
        self,
        time_grid,
        percentile: float = 95.0,
    ) -> float:
        """
        Integrated Brier Score with IPCW weight truncation.
        
        Uses a data-driven max_weight derived from the percentile of the
        weight distribution (Cole & Hernán, 2008).

        Parameters
        ----------
        time_grid : np.ndarray
            Times at which to evaluate.
        percentile : float
            Percentile for weight truncation (default: 95).
        """
        max_weight = self._compute_ipcw_max_weight(percentile=percentile)
        return self.integrated_brier_score(time_grid, max_weight=max_weight)

    def integrated_nbll_truncated(
        self,
        time_grid,
        percentile: float = 95.0,
    ) -> float:
        """
        Integrated NBLL with IPCW weight truncation.
        """
        max_weight = self._compute_ipcw_max_weight(percentile=percentile)
        return self.integrated_nbll(time_grid, max_weight=max_weight)

    def integrated_auc_ipcw_truncated(
        self,
        time_grid=None,
        percentile: float = 95.0,
        min_cases: int = 5,
        min_controls: int = 5,
    ) -> float:
        """
        Integrated AUC(t) with IPCW weight truncation.
        """
        max_weight = self._compute_ipcw_max_weight(percentile=percentile)
        return self.integrated_auc_ipcw(
            time_grid=time_grid,
            max_weight=max_weight,
            min_cases=min_cases,
            min_controls=min_controls,
        )

    def concordance_td_ipcw_truncated(
        self,
        method: str = 'adj_antolini',
        percentile: float = 95.0,
        eps: float = 1e-8,
    ) -> float:
        """
        IPCW-weighted time-dependent C-index with weight truncation (Cole & Hernán, 2008).

        Parameters
        ----------
        method : str
            'adj_antolini' or 'antolini'.
        percentile : float
            Percentile at which to cap non-zero weights (default: 95).
        eps : float
            Floor for G(t) to avoid division by zero.
        """
        if self.censor_surv is None:
            self.add_km_censor()

        w = self._ipcw_weights_event_time(eps=eps)

        nonzero = w[w > 0]
        if nonzero.size == 0:
            return float('nan')

        cap = np.percentile(nonzero, percentile)
        w_truncated = np.minimum(w, cap)

        return concordance_td_ipcw(
            self.durations,
            self.events,
            self.surv.values,
            self._duration_idx(),
            w_truncated,
            method=method,
        )

    def _log_metric_failure(self, kind: str, name: str, exc: Exception) -> None:
        msg = f"SurvMetrics.compute: {kind} '{name}' failed with {type(exc).__name__}: {exc}"

        if isinstance(exc, self.SAFE_EXCEPTIONS):
            self.logger.warning(
                "%s. Returning NaN%s.",
                msg,
                " curve" if kind == "curve" else "",
                exc_info=self.logger.isEnabledFor(logging.DEBUG),
            )
        else:
            self.logger.exception(
                "%s. Returning NaN%s.",
                msg,
                " curve" if kind == "curve" else "",
            )

    def _safe_run(
        self,
        *,
        kind: str,
        name: str,
        fn: Callable[[], Any],
        fallback: Callable[[], Any],
    ) -> Any:
        try:
            return fn()
        except Exception as exc:
            self._log_metric_failure(kind, name, exc)
            return fallback()

    def _nan_curve(self, name: str, time_grid: np.ndarray) -> pd.Series:
        time_grid = np.asarray(time_grid, dtype=float)
        return pd.Series(np.nan, index=time_grid, dtype=float, name=name)

    def _coerce_curve(
        self,
        name: str,
        out: Any,
        time_grid: np.ndarray,
    ) -> pd.Series:
        time_grid = np.asarray(time_grid, dtype=float)

        if isinstance(out, pd.Series):
            if len(out) != len(time_grid):
                raise ValueError(
                    f"Curve '{name}' length {len(out)} != len(time_grid) {len(time_grid)}"
                )
            return out.astype(float).rename(name)

        arr = np.asarray(out, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"Curve '{name}' must be 1D, got shape {arr.shape}")
        if arr.shape[0] != time_grid.shape[0]:
            raise ValueError(
                f"Curve '{name}' length {arr.shape[0]} != len(time_grid) {time_grid.shape[0]}"
            )

        return pd.Series(arr, index=time_grid, dtype=float, name=name)

    def _safe_scalar_metric(self, name: str, fn: Callable[[], Any]) -> float:
        out = self._safe_run(
            kind="metric",
            name=name,
            fn=fn,
            fallback=lambda: float("nan"),
        )
        try:
            return float(out)
        except Exception as exc:
            self._log_metric_failure("metric", f"{name} [float-cast]", exc)
            return float("nan")

    def _safe_curve_metric(
        self,
        name: str,
        fn: Callable[[], Any],
        time_grid: np.ndarray,
    ) -> pd.Series:
        time_grid = np.asarray(time_grid, dtype=float)
        return self._safe_run(
            kind="curve",
            name=name,
            fn=lambda: self._coerce_curve(name, fn(), time_grid),
            fallback=lambda: self._nan_curve(name, time_grid),
        )
    

    def compute(
        self,
        time_grid,
        percentile: float = 95.0,
        return_curves: bool = False,
    ):
        time_grid = np.asarray(time_grid, dtype=float)


        metric_fns = {
            "Antolini": lambda: self.concordance_td(method="antolini"),
            "adj Antolini": lambda: self.concordance_td(method="adj_antolini"),
            "Antolini IPCW": lambda: self.concordance_td_ipcw(method="antolini"),
            "adj Antolini IPCW": lambda: self.concordance_td_ipcw(method="adj_antolini"),
            "IBS": lambda: self.integrated_brier_score(time_grid=time_grid),
            "AUC": lambda: self.integrated_auc_ipcw(time_grid=time_grid),
            "INBLL": lambda: self.integrated_nbll(time_grid=time_grid),
            "IBS (trunc)": lambda: self.integrated_brier_score_truncated(
                time_grid=time_grid,
                percentile=percentile,
            ),
            "AUC (trunc)": lambda: self.integrated_auc_ipcw_truncated(
                time_grid=time_grid,
                percentile=percentile,
            ),
            "INBLL (trunc)": lambda: self.integrated_nbll_truncated(
                time_grid=time_grid,
                percentile=percentile,
            ),
            "adj Antolini IPCW (trunc)": lambda: self.concordance_td_ipcw_truncated(
                method="adj_antolini",
                percentile=percentile,
            ),
        }

        metrics = {
            name: self._safe_scalar_metric(name, fn)
            for name, fn in metric_fns.items()
        }

        if not return_curves:
            return metrics

        auc_curve = self._safe_curve_metric(
            "AUC(t)",
            lambda: self.auc_ipcw(time_grid=time_grid),
            time_grid=time_grid,
        )
        brier_curve = self._safe_curve_metric(
            "Brier(t)",
            lambda: self.brier_score(time_grid=time_grid),
            time_grid=time_grid,
        )

        return metrics, auc_curve, brier_curve