from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd

@dataclass
class MSPTemplate:
    baseline_schema: Dict[str, Dict[str, Any]]
    states_schema: Dict[str, Dict[str, Any]]
    baseline_features: Dict[str, str]
    state_features: Dict[str, str]
    state_names: List[str]
    event_names: List[str]
    tmat: np.ndarray

def _get_template(msp) -> MSPTemplate:
    """
    Return a metadata-only template describing this MultiStatePrep instance.
    """
    baseline_view = msp.baseline     # person_id + baseline_features
    states_view   = msp.states       # core state cols + state_features

    baseline_schema = build_dtype_schema(baseline_view)
    states_schema   = build_dtype_schema(states_view)

    return MSPTemplate(
        baseline_schema=baseline_schema,
        states_schema=states_schema,
        baseline_features=dict(msp.baseline_features),
        state_features=dict(msp.state_features),
        state_names=list(msp.state_names),
        event_names=list(msp.event_names),
        tmat=msp.tmat.copy(),
    )




def build_dtype_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Extract a dtype schema from a DataFrame:
      - for each column: categorical levels/ordered or plain dtype.
    """
    schema: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_categorical_dtype(s):
            schema[col] = {
                "kind": "category",
                "categories": list(s.cat.categories),
                "ordered": bool(s.cat.ordered),
            }
        else:
            schema[col] = {
                "kind": "dtype",
                "dtype": s.dtype,
            }
    return schema


def align_dtypes_like_schema(
    df: pd.DataFrame,
    schema: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Cast columns in df to the dtypes specified in a schema
    produced by build_dtype_schema.
    """
    for col in df.columns:
        if col not in schema:
            continue
        info = schema[col]
        if info["kind"] == "category":
            df[col] = pd.Categorical(
                df[col],
                categories=info["categories"],
                ordered=info["ordered"],
            )
        else:
            try:
                df[col] = df[col].astype(info["dtype"])
            except (TypeError, ValueError):
                # If casting fails, leave as is
                pass
    return df