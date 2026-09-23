import logging
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def df_to_mstate(
    df: pd.DataFrame,
    tmat: np.ndarray,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:

    logger.debug("Starting df_to_mstate with %d rows and %d columns", *df.shape)
    n_persons = df["person_id"].nunique()
    logger.debug("Number of unique persons: %d", n_persons)

    # Determine feature columns if not provided
    if feature_cols is None:
        from msprep.multistateprep import STATE_CORE_COLS
        feature_cols = [c for c in df.columns if c not in STATE_CORE_COLS]
        logger.debug("Detected feature columns (excluding core cols): %s", feature_cols)
    else:
        logger.debug("Using provided feature_cols: %s", feature_cols)

    # Ensure tmat is a proper numpy array
    tmat = np.asarray(tmat, dtype=float)
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("tmat must be square: [n_states_total, n_states_total].")
    n_states_total = tmat.shape[0]

    # Build mapping from origin state -> list of (target_state, trans_id)
    rows, cols = np.where(~np.isnan(tmat))
    trans_ids = tmat[rows, cols].astype(int)

    edges_by_state = {s: [] for s in range(1, n_states_total + 1)}
    for r, c, k in zip(rows, cols, trans_ids):
        from_state = r + 1  # 1-based
        to_state = c + 1
        edges_by_state[from_state].append((to_state, k))

    # Prepare output rows
    out_records = []

    # Iterate over each interval row and expand into (from,to) transitions
    for _, row in df.iterrows():
        pid = row["person_id"]
        s = int(row["state"])
        e = int(row["event"])
        Tstart = row["Tstart"]
        Tstop = row["Tstop"]
        duration = row["time"]

        # All outgoing transitions from this origin state
        edges = edges_by_state.get(s, [])
        if not edges:
            # No outgoing transitions, skip (terminal/no-transition states)
            continue

        for to_state, trans_id in edges:
            status = 1 if e == trans_id else 0
            rec = {
                "id": pid,
                "from": s,
                "to": to_state,
                "Tstart": Tstart,
                "Tstop": Tstop,
                "time": duration,
                "status": status,
                "trans": trans_id,
            }
            # Add feature columns
            for feat in feature_cols:
                rec[feat] = row[feat]
            out_records.append(rec)

    out = pd.DataFrame.from_records(out_records)

    # Sort and reset index
    out = out.sort_values(["id", "Tstart", "trans"]).reset_index(drop=True)

    return out