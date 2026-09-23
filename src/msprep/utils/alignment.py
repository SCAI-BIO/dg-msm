from __future__ import annotations

from typing import Optional, Tuple
import logging

import pandas as pd

from .log_utils import log_df_change

logger = logging.getLogger(__name__)


def align_person_tables(
    baseline: Optional[pd.DataFrame],
    states: Optional[pd.DataFrame],
    step: str = "Alignment",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    frames = {
        "baseline": baseline,
        "states": states,
    }

    # Collect person_id sets for all non-empty frames
    person_sets = {
        name: set(df["person_id"].unique())
        for name, df in frames.items()
        if df is not None and "person_id" in df.columns
    }

    if not person_sets:
        logger.debug(" %s: no tables to align.", step)
        return baseline, states

    # Common cohort
    common_ids = set.intersection(*person_sets.values())

    logger.info(
        " %s: aligning tables on common person_id set of size %d "
        "(baseline=%d, states=%d).",
        step,
        len(common_ids),
        len(person_sets.get("baseline", set())),
        len(person_sets.get("states", set())),
    )

    # Restrict each table to common_ids, with logging
    for name, df in frames.items():
        if df is None or "person_id" not in df.columns:
            continue
        before = df.copy()
        df_aligned = df[df["person_id"].isin(common_ids)].copy()
        log_df_change(f"{step}: {name}", before, df_aligned)
        frames[name] = df_aligned

    return frames["baseline"], frames["states"]




def enforce_sequential_states(
    states: Optional[pd.DataFrame],
    baseline: Optional[pd.DataFrame],
    step: str = "Alignment",
    *,
    label_col: str = "state",
    label_name: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Enforce nested person sets across ordered labels in `label_col`.

    For numeric labels l1 < l2 < ...:
        persons(label_k) ⊆ persons(label_{k-1}) for k > 1.

    This drops violating rows from `states` and re-aligns `baseline` to
    remaining person_ids.

    Parameters
    ----------
    states, baseline
        Input DataFrames (or None).
    step
        Label used in log messages to indicate calling context.
    label_col
        Column in `states` containing the ordered labels (default: "state").
    label_name
        Human-readable name for logging; defaults to `label_col`.

    Returns
    -------
    (states_enforced, baseline_synced)
    """
    if states is None or states.empty or label_col not in states.columns:
        return states, baseline

    label_name = label_name or label_col

    # Coerce labels to numeric for ordering; ignore rows with non-numeric labels
    label_num = pd.to_numeric(states[label_col], errors="coerce")
    if not label_num.notna().any():
        return states, baseline

    tmp = states.copy()
    tmp["_label_num"] = label_num

    # Unique labels in ascending order
    label_order = sorted(tmp["_label_num"].dropna().unique())
    if len(label_order) < 2:
        # Nothing to enforce if there is only one (or zero) label
        return states, baseline

    # person_id sets per label
    label_to_ids = {
        lab: set(tmp.loc[tmp["_label_num"] == lab, "person_id"])
        for lab in label_order
    }

    # Enforce nesting: persons(label_k) ⊆ persons(label_{k-1})
    for i in range(1, len(label_order)):
        prev = label_order[i - 1]
        curr = label_order[i]
        label_to_ids[curr] = label_to_ids[curr] & label_to_ids[prev]

    # Build row-level mask for states DataFrame
    before_states = states.copy()
    mask_keep = pd.Series(False, index=states.index)

    for lab in label_order:
        allowed_ids = label_to_ids[lab]
        if not allowed_ids:
            continue
        idx_lab = tmp["_label_num"] == lab
        idx_keep = idx_lab & tmp["person_id"].isin(allowed_ids)
        mask_keep |= idx_keep

    # Apply filter to states
    states_enforced = states[mask_keep].copy()
    log_df_change(
        f"{step}: states (enforce sequential {label_name} person_ids)",
        before_states,
        states_enforced,
    )

    # Re-align baseline to remaining person_ids in states
    if baseline is None or baseline.empty or "person_id" not in baseline.columns:
        return states_enforced, baseline

    before_baseline = baseline.copy()
    valid_ids = set(states_enforced["person_id"].unique())
    baseline_synced = baseline[baseline["person_id"].isin(valid_ids)].copy()
    log_df_change(
        f"{step}: baseline (sync with sequential {label_name}s)",
        before_baseline,
        baseline_synced,
    )

    return states_enforced, baseline_synced