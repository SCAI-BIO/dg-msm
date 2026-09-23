from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Sequence, List
import numpy as np
import pandas as pd
import logging
from .utils.alignment import align_person_tables

from .utils import save_prep, load_prep, state_event_summary

logger = logging.getLogger(__name__)

FEATURE_DICT = dict[str, str]

BASELINE_CORE_COLS = ["person_id"]
STATE_CORE_COLS_REQUIRED = ["person_id", "state", "Tstart", "Tstop", "time", "event"]
STATE_CORE_COLS = ["person_id", "visit", "state", "Tstart", "Tstop", "time", "event", "next_state"]

@dataclass
class MultiStatePrep:
    dataset_name: str
    state_names: Optional[List[str]] = None
    event_names: Optional[List[str]] = None
    tmat: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    _baseline: pd.DataFrame = field(init=False, repr=False)
    _states: pd.DataFrame = field(init=False, repr=False)

    # Feature metadata: {feature_name (column): category}
    baseline_features: FEATURE_DICT = field(init=False, default_factory=dict, repr=False)
    state_features: FEATURE_DICT = field(init=False, default_factory=dict, repr=False)

    # Construction from DataFrames
    @classmethod
    def from_dataframes(
        cls,
        dataset_name: str,
        baseline: pd.DataFrame,
        states: pd.DataFrame,
        tmat: np.ndarray,
        baseline_features: Optional[FEATURE_DICT] = None,
        state_features: Optional[FEATURE_DICT] = None,
        *,
        state_names: Optional[List[str]] = None,
        event_names: Optional[List[str]] = None,
    ) -> "MultiStatePrep":
        # core column presence / NaN checks
        if "person_id" not in baseline.columns:
            raise ValueError("Baseline is missing person_id")

        missing_states = set(STATE_CORE_COLS_REQUIRED) - set(states.columns)
        if missing_states:
            raise ValueError(
                f"states is missing required columns: {sorted(missing_states)}"
            )

        if baseline['person_id'].isna().any():
            raise ValueError("baseline['person_id'] contains missing values, which is not allowed.")

        for col in STATE_CORE_COLS_REQUIRED:
            if states[col].isna().any():
                raise ValueError(
                    f"states['{col}'] contains missing values, which is not allowed."
                )

        obj = cls(
            dataset_name=dataset_name,
            state_names=state_names,
            event_names=event_names,
        )

        obj._baseline = baseline.reset_index(drop=True).copy()
        obj._states = states.reset_index(drop=True).copy()

        # Alignment + invariants
        obj._baseline, obj._states = align_person_tables(
            obj._baseline, obj._states, step="Alignment in from_dataframes"
        )
        if (obj._states["time"] < 0).any():
            logger.error("Negative sojourn time found in states.")
            raise ValueError("Negative sojourn time found in states.")

        obj._states = obj._states.sort_values(["person_id", "Tstart"], kind="mergesort")
        obj._states["visit"] = obj._states.groupby("person_id").cumcount() + 1

        # centralised tmat + next_state logic
        obj._attach_tmat_and_next_state(tmat)

        # state/event names
        if obj.state_names is None or len(obj.state_names) != obj.n_states_total:
            if obj.state_names is not None:
                logger.warning("Provided state_names don't match n_states.")
            obj.state_names = [f"State {n + 1}" for n in range(obj.n_states_total)]

        if obj.event_names is None or len(obj.event_names) != obj.n_events:
            if obj.event_names is not None:
                logger.warning("Provided event_names don't match n_events.")
            obj.event_names = [f"Event {n + 1}" for n in range(obj.n_events)]

        # feature registries
        obj.baseline_features = dict(baseline_features) if baseline_features is not None else {}
        obj.state_features = dict(state_features) if state_features is not None else {}

        # warn about overlapping feature names between baseline and state
        overlap = set(obj.baseline_features) & set(obj.state_features)
        if overlap:
            logger.warning(
                "MultiStatePrep '%s': feature(s) defined both as baseline and state "
                "features: %s. This will cause `msp.features` to raise and may lead "
                "to ambiguous columns in merged data.",
                obj.dataset_name,
                sorted(overlap),
            )


        return obj

    def _ensure_built(self) -> None:
        """
        Ensure that the object has been properly initialized.
        """
        if (
            not hasattr(self, "_baseline")
            or not hasattr(self, "_states")
            or self._baseline is None
            or self._states is None
        ):
            raise RuntimeError(
                "MultiStatePrep must be constructed via from_dataframes(...) "
                "before accessing data."
            )


    #################################################################################################

    def terminal_labels(self, competing_risks: bool = False) -> pd.DataFrame:
        self._ensure_built()

        if self.tmat is None:
            raise ValueError("tmat must be set to derive terminal labels.")

        tm = self.tmat
        # nonterminal = rows with any outgoing transitions
        nonterminal_mask = ~np.all(np.isnan(tm), axis=1)
        # terminal states = rows with no outgoing transitions
        terminal_states = np.where(~nonterminal_mask)[0] + 1  # 1-based

        states = self._states.copy()

        # Rows where a terminal state is reached via a real event (>0)
        terminal_rows = states[
            (states["next_state"].isin(terminal_states)) & (states["event"] > 0)
        ].copy()

        if not terminal_rows.empty:
            # first time we hit a terminal state per person
            terminal_rows = terminal_rows.sort_values(["person_id", "Tstop"])
            first_terminal = (
                terminal_rows
                .groupby("person_id", as_index=False)
                .first()[["person_id", "Tstop", "event"]]
                .rename(columns={"Tstop": "time_to_terminal"})
            )
        else:
            first_terminal = pd.DataFrame(
                columns=["person_id", "time_to_terminal", "event"]
            )

        # Last observed time per person (for censoring)
        last_times = (
            states.groupby("person_id", as_index=False)["Tstop"]
            .max()
            .rename(columns={"Tstop": "last_time"})
        )

        # Merge with baseline person list to ensure all persons appear
        persons = self._baseline[["person_id"]].drop_duplicates()
        out = persons.merge(first_terminal, on="person_id", how="left")
        out = out.merge(last_times, on="person_id", how="left")

        # If no terminal event, use last_time and mark event=0 (censored)
        mask_no_event = out["time_to_terminal"].isna()
        out.loc[mask_no_event, "time_to_terminal"] = out.loc[mask_no_event, "last_time"]
        out["event"] = out["event"].fillna(0).astype(int)

        # Optionally collapse to binary event indicator (no competing risks)
        if not competing_risks:
            out["event"] = (out["event"] != 0).astype(int)

        return out[["person_id", "time_to_terminal", "event"]]

    def survival_analysis(
        self,
        competing_risks: bool = False,
        start_visit: int = 1,
    ) -> pd.DataFrame:
        """
        Build a per-person survival dataset for standard time-to-event models.
        """
        self._ensure_built()

        terminal    = self.terminal_labels(competing_risks=competing_risks)
        terminal    = terminal.merge(self.baseline, on="person_id", how="left")
        start_visit_df = self.states[self.states["visit"] == start_visit]

        n_total = self.n_persons
        n_at_visit = start_visit_df["person_id"].nunique()
        if start_visit_df.empty:
            raise ValueError(f"No persons found at visit {start_visit}.")
        if n_at_visit < n_total:
            logger.info(
                "survival_analysis '%s': %d of %d persons have no visit %d "
                "and will be excluded.",
                self.dataset_name,
                n_total - n_at_visit,
                n_total,
                start_visit,
            )

        start_visit_df = start_visit_df[["person_id", "Tstart"] + self.list_state_features()]
        df = start_visit_df.merge(terminal, on="person_id", how="inner")
        df["time_to_terminal"] = df["time_to_terminal"] - df["Tstart"]
        cols = ["person_id", "time_to_terminal", "event"] + self.list_baseline_features() + self.list_state_features()
        return df[cols].reset_index(drop=True)


    @property
    def state_labels(self) -> pd.DataFrame:
        self._ensure_built()
        missing = [c for c in STATE_CORE_COLS if c not in self._states.columns]
        if missing:
            raise KeyError(
                f"Missing expected columns in _states: {missing}"
            )
        return self._states[STATE_CORE_COLS].copy()


    @property
    def n_persons(self) -> int:
        self._ensure_built()
        return int(self._baseline["person_id"].nunique())

    @property
    def n_states_total(self) -> int:
        self._ensure_built()
        if self._states.empty or "state" not in self._states.columns:
            return 0
        if self.tmat is not None:
            return int(self.tmat.shape[0])
        s = pd.to_numeric(self._states["state"], errors="coerce")
        return int(s.max()) if s.notna().any() else 0

    @property
    def n_states(self) -> int:
        self._ensure_built()
        return np.sum(~np.all(np.isnan(self.tmat), axis=1))

    @property
    def n_events(self) -> int:
        self._ensure_built()
        if self.tmat is not None:
            vals = self.tmat[~np.isnan(self.tmat)]
            return int(vals.max()) if vals.size > 0 else 0
        # Fallback: infer from event column
        if self._states.empty or "event" not in self._states.columns:
            return 0
        return int(self._states["event"].max())


    @property
    def n_events_per_state(self) -> List[int]:
        self._ensure_built()
        return (~np.isnan(self.tmat)).sum(axis=1).astype(int).tolist()


    @property
    def nonterminal_state_indices(self) -> List[int]:
        self._ensure_built()
        mask = ~np.all(np.isnan(self.tmat), axis=1)  # shape [n_states_total]
        return [i + 1 for i, is_nt in enumerate(mask) if is_nt]

    @property
    def nonterminal_state_names(self) -> List[str]:
        idx = self.nonterminal_state_indices  # 1-based
        return [self.state_names[i - 1] for i in idx]

    # Features: lists and combined registry
    @property
    def features(self) -> FEATURE_DICT:
        self._ensure_built()
        overlap = set(self.state_features) & set(self.baseline_features)
        if overlap:
            raise ValueError(
                f"Feature(s) defined for both baseline and state: {sorted(overlap)}"
            )
        return {**self.state_features, **self.baseline_features}

    def list_state_features(
        self,
        categories = None,
        ) -> list[str]:
        if categories is None:
            return list(self.state_features.keys())
        else:
            return [name for name, cat in self.state_features.items() if cat in categories]

    def list_baseline_features(
        self,
        categories= None,
        ) -> list[str]:
        if categories is None:
            return list(self.baseline_features.keys())
        else:
            return [name for name, cat in self.baseline_features.items() if cat in categories]

    def list_features(
        self,
        categories = None,
    ) -> list[str]:
        if categories is None:
            return list(self.features.keys())
        return [
            name for name, cat in self.features.items()
            if cat in categories
        ]

    @property
    def states(self) -> pd.DataFrame:
        self._ensure_built()
        cols = list(STATE_CORE_COLS)
        cols += [c for c in self.state_features.keys() if c in self._states.columns]
        cols = list(dict.fromkeys(cols))  # unique, preserve order
        return self._states[cols].copy()

    @property
    def baseline(self) -> pd.DataFrame:
        self._ensure_built()
        cols = ["person_id"]
        cols += [c for c in self.baseline_features.keys() if c in self._baseline.columns]
        cols = list(dict.fromkeys(cols))
        return self._baseline[cols].copy()

    @property
    def full_df(self) -> pd.DataFrame:
        self._ensure_built()
        df = self.states.merge(
            self.baseline,
            on="person_id",
            how="left",
        )
        return df.copy()

    def save(self, directory: Union[str, Path]) -> Path:
        self._ensure_built()
        return save_prep(self, directory)

    @classmethod
    def load(cls, meta_path: Union[str, Path]) -> "MultiStatePrep":
        msp = load_prep(cls, meta_path, expected_class="MultiStatePrep")
        if msp.tmat is None:
            raise ValueError("Loaded MultiStatePrep has no tmat.")
        return msp

    def subset(
        self,
        baseline_features: Optional[Sequence[str] ] = None,
        state_features: Optional[Sequence[str] ] = None,
        dataset_name: Optional[str ] = None,
        dropna: str | bool = False,
        truncate_at_first_gap: bool = True,
    ) -> "MultiStatePrep":
        from .subset import msprep_subset
        return msprep_subset(
            msp=self,
            baseline_features=baseline_features,
            state_features=state_features,
            dataset_name=dataset_name,
            dropna=dropna,
            truncate_at_first_gap=truncate_at_first_gap,
        )
            
    def get_template(self) -> "MSPTemplate":
        from .template import MSPTemplate, _get_template
        return _get_template(self)

    def to_mstate(self) -> pd.DataFrame:
        from .utils.mstate import df_to_mstate
        return df_to_mstate(self.full_df, tmat=self.tmat)


    def subset_persons(
        self,
        person_ids: Sequence[Union[int, str]],
        dataset_name: Optional[str ] = None,
    ) -> "MultiStatePrep":
        self._ensure_built()

        requested_ids = set(person_ids)
        available_ids = set(self._baseline["person_id"]) & set(self._states["person_id"])
        used_ids = requested_ids & available_ids

        if not used_ids:
            raise ValueError("subset_persons: no overlapping person_ids found.")

        missing_ids = requested_ids - available_ids

        if missing_ids:
            logger.warning(
                "subset_persons '%s': %d of %d requested person_ids are not present "
                "in both baseline and state tables and will be ignored: %s",
                dataset_name or self.dataset_name,
                len(missing_ids),
                len(requested_ids),
                sorted(missing_ids),
            )

        baseline_sub = self._baseline[self._baseline["person_id"].isin(used_ids)].copy()
        states_sub = self._states[self._states["person_id"].isin(used_ids)].copy()

        new_dataset_name = dataset_name or f"{self.dataset_name}_subset"

        # Use copies of feature dicts so metadata is not shared by reference
        return self.__class__.from_dataframes(
            dataset_name=new_dataset_name,
            baseline=baseline_sub,
            states=states_sub,
            baseline_features=dict(self.baseline_features),
            state_features=dict(self.state_features),
            state_names=self.state_names,
            event_names=self.event_names,
            tmat=self.tmat, 
        )
    
    def split(
        self,
        test_size: float = 0.2,
        val_size: Optional[float ] = 0.2,
        stratify_by_state: bool = True,
        random_state: Optional[int ] = 42,
        shuffle: bool = True, 
        ):
        from .utils.split import msp_split_by_max_state


        return msp_split_by_max_state(
            msp=self,
            test_size = test_size,
            val_size = val_size,
            stratify_by_state = stratify_by_state,
            random_state = random_state,
            shuffle = shuffle, 
        )

    @property
    def baseline_ftypes(self):
        """
        Global feature schema for baseline features.
        """
        schema = {}
        df = self.baseline  # full baseline table (one row per person)
        feature_dict = self.baseline_features  # {name: category_string}

        for name, cat in feature_dict.items():
            if name not in df.columns:
                raise KeyError(f"Feature '{name}' is registered but not present in the baseline table.")
            col = df[name]
            ftype = cat.lower()

            if ftype in {"pos", "conti", "real"}:
                # Continuous feature
                schema[name] = {
                    "type": "real" if ftype in {"real", "conti"} else "pos",
                    "nclass": 1,
                    "levels": None,   # no levels for continuous
                }

            elif ftype in {"cat", "disc", "bin", "thresh"}:
                # Categorical: define global levels
                col_non_null = col.dropna()
                levels = col_non_null.unique()
                # sort for determinism
                levels = np.sort(levels)
                # convert to Python scalars (JSON-friendly)
                levels = [lv.item() if hasattr(lv, "item") else lv for lv in levels]
                nclass = len(levels)

                schema[name] = {
                    "type": "cat",
                    "nclass": nclass,
                    "levels": levels,
                }

            elif ftype in {"ord", "ordinal"}:
                # Ordinal: define global ordered levels
                col_non_null = col.dropna()
                levels = col_non_null.unique()
                levels = np.sort(levels)
                levels = [lv.item() if hasattr(lv, "item") else lv for lv in levels]
                nclass = len(levels)

                schema[name] = {
                    "type": "ord",
                    "nclass": nclass,
                    "levels": levels,
                }

            else:
                raise ValueError(f"Unknown baseline feature type '{ftype}' for '{name}'.")

        return schema

    @property
    def state_ftypes(self):
        schema = {}
        df = self.states  # state/interval table (joined with baseline)
        feature_dict = self.state_features  # {name: category_string}

        for name, cat in feature_dict.items():
            if name not in df.columns:
                raise KeyError(f"Feature '{name}' is registered but not present in the states table.")
            col = df[name]
            ftype = cat.lower()

            if ftype in {"pos", "conti", "real"}:
                schema[name] = {
                    "type": "real" if ftype in {"real", "conti"} else "pos",
                    "nclass": 1,
                    "levels": None,
                }

            elif ftype in {"cat", "disc", "bin", "thresh"}:
                col_non_null = col.dropna()
                levels = col_non_null.unique()
                levels = np.sort(levels)
                levels = [lv.item() if hasattr(lv, "item") else lv for lv in levels]
                nclass = len(levels)

                schema[name] = {
                    "type": "cat",
                    "nclass": nclass,
                    "levels": levels,
                }

            elif ftype in {"ord", "ordinal"}:
                col_non_null = col.dropna()
                levels = col_non_null.unique()
                levels = np.sort(levels)
                levels = [lv.item() if hasattr(lv, "item") else lv for lv in levels]
                nclass = len(levels)

                schema[name] = {
                    "type": "ord",
                    "nclass": nclass,
                    "levels": levels,
                }

            else:
                raise ValueError(f"Unknown state feature type '{ftype}' for '{name}'.")

        return schema


    def same_metadata(
        self,
        other: "MultiStatePrep",
    ) -> bool:
        if not isinstance(other, MultiStatePrep):
            return False

        # --- tmat ---
        if self.tmat is None or other.tmat is None:
            if self.tmat is not None or other.tmat is not None:
                return False
        else:
            if self.tmat.shape != other.tmat.shape:
                return False
            # exact equality incl. NaNs
            if not np.array_equal(self.tmat, other.tmat, equal_nan=True):
                return False

        # --- feature registries ---
        if self.baseline_features != other.baseline_features:
            return False
        if self.state_features != other.state_features:
            return False
        return True

    def assert_same_metadata(
        self,
        other: "MultiStatePrep",
    ) -> None:
        if not self.same_metadata(other):
            raise ValueError(f"Metadata mismatch between MultiStatePrep instances ({self.dataset_name} vs {other.dataset_name}).")



    def _attach_tmat_and_next_state(
        self,
        tmat: np.ndarray,
    ) -> None:
        self._ensure_built()

        # --- tmat checks -----------------------------------------------------
        tmat = np.asarray(tmat, dtype=float)
        if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
            raise ValueError("tmat must be a square matrix [n_states, n_states].")

        vals = tmat[~np.isnan(tmat)]
        if (vals <= 0).any():
            raise ValueError("tmat must contain only positive event IDs (or NaN).")
        if not np.all(vals == np.floor(vals)):
            raise ValueError("tmat event IDs must be integers.")

        self.tmat = tmat
        n_states_total = tmat.shape[0]

        # --- check state labels vs. tmat size --------------------------------
        s_series = pd.to_numeric(self._states["state"], errors="raise")
        min_state_data = int(s_series.min())
        max_state_data = int(s_series.max())
        if min_state_data < 1 or max_state_data > n_states_total:
            raise ValueError(
                f"State labels in data must be in 1..{n_states_total}, "
                f"got min={min_state_data}, max={max_state_data}."
            )

        # --- check event codes vs. tmat --------------------------------------
        events = pd.to_numeric(self._states["event"], errors="raise").to_numpy()
        if (events < 0).any():
            raise ValueError("Negative event codes found in states['event'].")

        nonzero_events = events[events > 0]
        if nonzero_events.size > 0:
            if vals.size == 0:
                raise ValueError("tmat has no non-NaN entries but non-zero events are present.")
            max_tmat = int(vals.max())
            if nonzero_events.max() > max_tmat:
                raise ValueError(
                    f"states['event'] contains event ID {nonzero_events.max()}, "
                    f"but tmat only defines transitions up to {max_tmat}."
                )

        # --- build event -> (from_state, to_state) mapping -------------------
        event_to_dest: dict[int, tuple[int, int]] = {}
        if vals.size > 0:
            for i in range(tmat.shape[0]):
                for j in range(tmat.shape[1]):
                    ev = tmat[i, j]
                    if not np.isnan(ev) and ev > 0:
                        ev_id = int(ev)
                        if ev_id in event_to_dest:
                            raise ValueError(
                                f"Event ID {ev_id} appears multiple times in tmat; "
                                "event IDs must be globally unique."
                            )
                        event_to_dest[ev_id] = (i + 1, j + 1)  # 1-based indices

        # --- compute next_state and enforce consistency ----------------------
        s = self._states["state"].astype(int).to_numpy()
        e = self._states["event"].astype(int).to_numpy()
        next_state = np.full_like(s, fill_value=-1)

        for idx, (si, ei) in enumerate(zip(s, e)):
            if ei == 0:
                # censored / no transition
                next_state[idx] = si
            else:
                from_to = event_to_dest.get(ei)
                if from_to is None:
                    raise ValueError(f"Unknown event code {ei} in tmat mapping.")
                from_state, dest = from_to
                if si != from_state:
                    raise ValueError(
                        f"Inconsistent state/event: state={si}, but event {ei} is defined "
                        f"from state {from_state} in tmat."
                    )
                next_state[idx] = dest

        self._states["next_state"] = next_state

    def event_summary(self) -> pd.DataFrame:
        """
        Per-state summary of key event types (censored, death, next state).
        """
        self._ensure_built()
        return state_event_summary( states=self._states, state_names=self.state_names, event_names=self.event_names)
