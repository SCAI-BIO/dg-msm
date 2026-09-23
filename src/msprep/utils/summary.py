from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd


def state_event_summary(
    states: pd.DataFrame,
    state_names: Optional[List[str]] = None,
    event_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    if states is None or states.empty:
        return pd.DataFrame()

    required = {"person_id", "state", "event"}
    missing = required - set(states.columns)
    if missing:
        raise KeyError(f"Missing required columns in states: {sorted(missing)}")

    df = states.copy()

    # Normalize state to integers
    state_series = pd.to_numeric(df["state"], errors="coerce")
    df = df[state_series.notna()].copy()
    df["state"] = state_series[state_series.notna()].astype(int)

    if df.empty:
        return pd.DataFrame()

    # Sorted list of states actually present (typically non-terminal states)
    states_order = sorted(df["state"].unique().tolist())

    # Total unique persons per state (any event, including censored)
    total = (
        df.groupby("state")["person_id"]
        .nunique()
        .reindex(states_order)
        .fillna(0)
        .astype(int)
        .rename("total")
    )

    # Unique persons per state with event == 0 (censored)
    censored = (
        df[df["event"] == 0]
        .groupby("state")["person_id"]
        .nunique()
        .reindex(states_order)
        .fillna(0)
        .astype(int)
        .rename("censored")
    )

    # Percentage censored
    denom = total.replace(0, np.nan)
    censored_pct = (censored / denom * 100).fillna(0).round(1).rename("censored %")

    # --- Event-specific counts (for event codes > 0) ---
    ev_df = df[df["event"] > 0].copy()
    if ev_df.empty:
        ev_counts = pd.DataFrame(index=states_order)
        ev_pct = pd.DataFrame(index=states_order)
        event_codes: List[int] = []
        event_labels: List[str] = []
    else:
        # Distinct event codes > 0
        event_codes = sorted(ev_df["event"].unique().astype(int).tolist())

        # Map codes to labels using event_names list if provided
        event_labels = []
        for e in event_codes:
            if event_names is not None and 1 <= e <= len(event_names):
                label = event_names[e - 1]
                if label is None:
                    label = str(e)
            else:
                label = str(e)
            event_labels.append(label)

        # Unique persons per (state, event)
        ev_counts = (
            ev_df.groupby(["state", "event"])["person_id"]
            .nunique()
            .unstack("event")
            .reindex(index=states_order, columns=event_codes)
            .fillna(0)
            .astype(int)
        )
        # Rename columns to labels
        ev_counts.columns = event_labels

        # Percent per state relative to total
        ev_pct = (ev_counts.div(denom, axis=0) * 100).fillna(0).round(1)
        ev_pct.columns = [f"{lbl} %" for lbl in event_labels]

    # Assemble final DataFrame
    out_parts = [total, censored, censored_pct]
    if ev_df is not None and not ev_df.empty:
        out_parts.extend([ev_counts, ev_pct])

    out = pd.concat(out_parts, axis=1)

    # Column order: total, censored, censored%, label, label%, label, label%, ...
    if ev_df is not None and not ev_df.empty:
        ordered_cols = ["total", "censored", "censored %"] + [
            x for lbl in event_labels for x in (lbl, f"{lbl} %")
        ]
        out = out.loc[:, ordered_cols]
    else:
        out = out.loc[:, ["total", "censored", "censored %"]]

    # Apply state_names if provided and long enough
    if state_names is not None:
        try:
            row_labels = [state_names[s - 1] for s in states_order]
            out.index = row_labels
        except IndexError:
            out.index = states_order
    else:
        out.index = states_order

    return out