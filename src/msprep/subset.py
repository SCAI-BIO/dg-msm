from __future__ import annotations

from typing import Sequence, Literal, Dict, Mapping
import logging

from .multistateprep import MultiStatePrep, BASELINE_CORE_COLS, STATE_CORE_COLS
from .utils.alignment import enforce_sequential_states

logger = logging.getLogger(__name__)

FeatureDict = Dict[str, str]


def msprep_subset(
    msp: MultiStatePrep,
    baseline_features: Sequence[str] | None = None,
    state_features: Sequence[str] | None = None,
    dataset_name: str | None = None,
    dropna: Literal[True, False, "baseline", "states", "both"] = False,
    truncate_at_first_gap: bool = True,
) -> MultiStatePrep:
    if dropna not in [True, False, "baseline", "states", "both"]:
        raise ValueError(f"Invalid dropna={dropna!r}.")
    msp._ensure_built()

    new_dataset_name = dataset_name or msp.dataset_name

    # Determine feature lists
    if baseline_features is None:
        baseline_features = list(msp.baseline_features.keys())
    else:
        baseline_features = list(baseline_features)

    if state_features is None:
        state_features = list(msp.state_features.keys())
    else:
        state_features = list(state_features)

    # Subset feature dictionaries according to requested columns
    def _subset_feat_dict(feat: Mapping[str, str], keys: Sequence[str]) -> FeatureDict:
        return {k: v for k, v in feat.items() if k in keys}

    baseline_features_sub = _subset_feat_dict(msp.baseline_features, baseline_features)
    state_features_sub = _subset_feat_dict(msp.state_features, state_features)

    # Build column lists and handle missing columns gracefully
    def _unique_in_order(cols: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(cols))

    requested_baseline_cols = _unique_in_order(
        list(BASELINE_CORE_COLS) + baseline_features
    )


    requested_state_cols = _unique_in_order(
        list(STATE_CORE_COLS) + state_features
    )

    missing_baseline_cols = [
        c for c in requested_baseline_cols if c not in msp._baseline.columns
    ]
    missing_state_cols = [
        c for c in requested_state_cols if c not in msp._states.columns
    ]

    if missing_baseline_cols:
        logger.warning(
            "msprep_subset: requested baseline columns not found and will be ignored: %s",
            missing_baseline_cols,
        )
    if missing_state_cols:
        logger.warning(
            "msprep_subset: requested state columns not found and will be ignored: %s",
            missing_state_cols,
        )

    baseline_cols = [
        c for c in requested_baseline_cols if c in msp._baseline.columns
    ]
    state_cols = [
        c for c in requested_state_cols if c in msp._states.columns
    ]

    baseline_sub = msp._baseline[baseline_cols].copy()
    states_sub = msp._states[state_cols].copy()

    # Initial ids relative to original MSP
    baseline_ids_initial = set(msp._baseline["person_id"].unique())
    state_ids_initial = set(msp._states["person_id"].unique())
    initial_ids = baseline_ids_initial & state_ids_initial

    # Drop rows with missing feature values, if requested
    if dropna in (True, "baseline", "both") and baseline_features:
        present_baseline_feats = [
            c for c in baseline_features if c in baseline_sub.columns
        ]
        if present_baseline_feats:
            baseline_sub = baseline_sub.dropna(subset=present_baseline_feats)

    if dropna in (True, "states", "both") and state_features:
        present_state_feats = [
            c for c in state_features if c in states_sub.columns
        ]
        if present_state_feats:
            states_sub = states_sub.dropna(subset=present_state_feats)

    # Align baseline_sub and states_sub to their common person_id set
    baseline_ids_sub = set(baseline_sub["person_id"].unique())
    state_ids_sub = set(states_sub["person_id"].unique())
    common_ids_sub = baseline_ids_sub & state_ids_sub

    baseline_sub = baseline_sub[baseline_sub["person_id"].isin(common_ids_sub)]
    states_sub = states_sub[states_sub["person_id"].isin(common_ids_sub)]

    # Optional: truncate at first gap via sequential visits
    if truncate_at_first_gap:
        if "visit" not in states_sub.columns:
            logger.warning(
                "msprep_subset: truncate_at_first_gap=True but 'visit' column not found; skipping truncation."
            )
        else:
            states_sub, baseline_sub = enforce_sequential_states(
                states=states_sub,
                baseline=baseline_sub,
                label_col="visit",
                label_name="visit",
                step=f"subset[{new_dataset_name}] enforce sequential visits",
            )

    # Final person-id alignment and logging 
    baseline_ids = set(baseline_sub["person_id"].unique())
    state_ids = set(states_sub["person_id"].unique())
    final_ids = baseline_ids & state_ids

    baseline_sub = baseline_sub[baseline_sub["person_id"].isin(final_ids)]
    states_sub = states_sub[states_sub["person_id"].isin(final_ids)]

    n_initial = len(initial_ids)
    n_final = len(final_ids)
    n_dropped = n_initial - n_final

    logger.warning(
        "Subset '%s': initial person_ids = %d, final person_ids = %d, dropped = %d",
        new_dataset_name,
        n_initial,
        n_final,
        n_dropped,
    )

    # Construct a new MultiStatePrep of the same class
    return msp.__class__.from_dataframes(
        dataset_name=new_dataset_name,
        baseline=baseline_sub,
        states=states_sub,
        baseline_features=baseline_features_sub,
        state_features=state_features_sub,
        state_names=msp.state_names,
        event_names=msp.event_names,
        tmat=msp.tmat.copy(),
    )