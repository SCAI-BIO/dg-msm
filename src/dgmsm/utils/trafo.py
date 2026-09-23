import numpy as np
import pandas as pd
import logging


class LabelTrafo:
    """
    Duration-based (clock-reset) discretiser for multi-state survival data.

    Maps continuous intervals (Tstart, Tstop, event) onto a uniform discrete
    time grid of K bins.  The discrete label `time` encodes the number of
    complete bins survived within a single state sojourn, making it
    translation-invariant (semi-Markov / clock-reset semantics): only the
    duration  d = Tstop - Tstart  matters, not the absolute position on the
    timeline.
    """

    def __init__(
        self,
        n_time: int,
        use_event_stops_only: bool = False,
        logger = None,
        verbose: bool = False,
    ) -> None:
        if n_time <= 0:
            raise ValueError("n_time must be a positive integer.")

        # Number of bins K on the absolute time axis.
        self._n_time = int(n_time)
        self.use_event_stops_only = use_event_stops_only

        # Learned in fit(): cuts_ = (c_0,...,c_K), c_0=0, c_K=t_max.
        self.cuts_= None  # shape (K + 1,)
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose
        
    def fit(self, df: pd.DataFrame) -> "LabelTrafo":
        """
        Learn the uniform time grid from training data.
        Sets  cuts = linspace(0, t_max, K+1)  where t_max is the maximum
        Tstop value in df (or among event rows only when
        use_event_stops_only=True).
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty.")

        tstop = df["Tstop"].to_numpy(dtype=float)
        events = df["event"].to_numpy()

        # choose which times define the grid
        if self.use_event_stops_only:
            # use only event stops to define t_max
            mask_ev = events > 0
            if not mask_ev.any():
                raise ValueError("No events with event>0 found to define grid.")
            base_stop = tstop[mask_ev]
        else:
            # use all Tstop values (events + censored)
            base_stop = tstop

        if base_stop.size == 0:
            raise ValueError("No Tstop values found.")

        K = self._n_time

        t_min = 0.0
        t_max_raw = float(base_stop.max())
        if t_max_raw <= t_min:
            raise ValueError(f"Require t_max > 0, got t_max={t_max_raw}")

        # Equally spaced cuts: c_0=0, c_K=t_max_raw.
        cuts = np.linspace(t_min, t_max_raw, K + 1)

        self.cuts_ = cuts
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.cuts_ is None:
            raise RuntimeError("Must call fit() before transform().")
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty.")

        df_disc = df.copy()

        tstart = df_disc["Tstart"].to_numpy(dtype=float)
        tstop  = df_disc["Tstop"].to_numpy(dtype=float)
        events = df_disc["event"].to_numpy()

        K  = self._n_time
        dt = self.time_delta

        # Duration (sojourn), the only quantity the label depends on
        duration = tstop - tstart
        if np.any(duration < -1e-9):
            raise ValueError("LabelTrafo: encountered Tstop < Tstart before discretization.")
        duration = np.clip(duration, 0.0, None)

        time_floor = np.floor(duration / dt + 1e-9).astype(np.int64)

        time_event = np.clip(time_floor, 0, K - 1)
        time_cens  = np.clip(time_floor, 0, K)
        time = np.where(events > 0, time_event, time_cens)
        frac = np.clip(duration / dt - time, 0.0, 1.0).astype(np.float32)


        mask_clip = (events > 0) & (time_floor >= K)
        if mask_clip.any() and self.verbose:
            n_problem = int(mask_clip.sum())
            n_obs     = int((events > 0).sum())
            self.logger.warning(
                "LabelTrafo: %d/%d observed rows (%.4f%%) have duration >= K*dt "
                "(event beyond horizon) and are folded into bin K-1.",
                n_problem, n_obs, 100.0 * n_problem / max(n_obs, 1),
            )

        mask_subbin = (events > 0) & (time_floor == 0)
        if mask_subbin.any() and self.verbose:
            n_sub = int(mask_subbin.sum())
            self.logger.info(
                "LabelTrafo: %d observed events have duration < dt (sub-bin, bin 0). "
                "Not an anomaly — resolvable only by finer dt / larger K.",
                n_sub,
            )

        df_disc["time"]  = time
        df_disc["event"] = events

        df_disc["Tstart"] = np.floor(tstart / dt + 1e-9).astype(np.int64)
        df_disc["Tstop"]  = np.floor(tstop  / dt + 1e-9).astype(np.int64)

        df_disc["sojourn_time"] = duration.astype(np.float32)   # continuous, pre-floor
        df_disc["sojourn_frac"] = frac.astype(np.float32)       # sub-bin offset in [0,1]


        return df_disc

    @property
    def cuts(self) -> np.ndarray:
        if self.cuts_ is None:
            raise RuntimeError("Must call fit() before accessing cuts.")
        return self.cuts_

    @property
    def n_time(self) -> int:
        return self._n_time

    @property
    def discrete_times(self) -> np.ndarray:
        return self.cuts[1:]

    @property
    def time_delta(self) -> float:
        if self.cuts_ is None:
            raise RuntimeError("Must call fit() before accessing time_delta.")
        if self._n_time <= 0:
            raise RuntimeError("Invalid n_time, must be positive.")
        return float(self.cuts_[1] - self.cuts_[0])