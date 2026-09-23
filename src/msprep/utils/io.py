from __future__ import annotations

from pathlib import Path
from typing import Union, Type, TypeVar, Optional
import json
import logging
import pandas as pd

FEATURE_DICT = dict[str, str]

logger = logging.getLogger(__name__)

TPrep = TypeVar("TPrep")


def features_to_json(feat: Optional[FEATURE_DICT]) -> dict:
    if not feat:
        return {}
    return dict(feat)


def features_from_json(feat_json: Optional[dict]) -> FEATURE_DICT:
    if not feat_json:
        return {}
    if not isinstance(feat_json, dict):
        raise ValueError(f"Invalid feature metadata: expected dict, got {type(feat_json)}")
    return {str(name): str(cat) for name, cat in feat_json.items()}


def save_prep(
    prep: object,
    directory: Union[str, Path],
) -> Path:
    """
    Save baseline, states and metadata (including features if present) to disk.

    Assumes `prep` exposes:
      * prep.dataset_name: str
      * prep.state_names: list[str] | None      
      * prep.event_names: list[str] | None
      * prep._baseline: pd.DataFrame
      * prep._states: pd.DataFrame
      * (optional) prep.baseline_features: FEATURE_DICT
      * (optional) prep.state_features: FEATURE_DICT
      * (optional) prep.tmat: np.ndarray (transition matrix)

    It writes:
      * <dataset_name>_baseline.csv
      * <dataset_name>_states.csv
      * (optional) <dataset_name>_tmat.csv
      * <dataset_name>_meta.json

    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    dataset_name = getattr(prep, "dataset_name")
    state_names = getattr(prep, "state_names", None)
    event_names = getattr(prep, "event_names", None)
    baseline_df: pd.DataFrame = getattr(prep, "_baseline")
    states_df: pd.DataFrame = getattr(prep, "_states")

    baseline_file = f"{dataset_name}_baseline.csv"
    states_file = f"{dataset_name}_states.csv"
    meta_file = f"{dataset_name}_meta.json"

    baseline_path = directory / baseline_file
    states_path = directory / states_file
    meta_path = directory / meta_file

    # Save DataFrames
    baseline_df.to_csv(baseline_path, index=False)
    states_df.to_csv(states_path, index=False)

    # Build metadata
    meta: dict = {
        "class": type(prep).__name__,
        "dataset_name": dataset_name,
        "state_names": state_names,
        "event_names": event_names,
        "baseline_csv": baseline_file,
        "states_csv": states_file,
    }

    # Optional feature dicts
    baseline_features = getattr(prep, "baseline_features", None)
    state_features = getattr(prep, "state_features", None)

    if baseline_features is not None:
        meta["baseline_features"] = features_to_json(baseline_features)
    if state_features is not None:
        meta["state_features"] = features_to_json(state_features)

    # Optional transition matrix
    tmat = getattr(prep, "tmat", None)
    if tmat is not None:
        tmat_file = f"{dataset_name}_tmat.csv"
        tmat_path = directory / tmat_file
        pd.DataFrame(tmat).to_csv(tmat_path, index=False)
        meta["tmat_csv"] = tmat_file

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta_path


def load_prep(
    cls: Type[TPrep],
    meta_path: Union[str, Path],
    expected_class: Optional[str] = None,
) -> TPrep:
    """
    Load a prep instance (e.g. MultiStatePrep) from metadata written by `save_prep`.

    Parameters
    ----------
    cls
        Target class, must expose a `from_dataframes(...)` constructor with
        signature compatible with:

        from_dataframes(
            dataset_name: str,
            baseline: pd.DataFrame,
            states: pd.DataFrame,
            baseline_features: Optional[FEATURE_DICT] = None,
            state_features: Optional[FEATURE_DICT] = None,
            *,
            state_names: Optional[list[str]] = None,
            event_names: Optional[list[str]] = None
            tmat: Optional[np.ndarray] = None,
        )

    meta_path
        Path to the JSON metadata file.
    expected_class
        If provided, the `"class"` field in metadata must match this string,
        otherwise a ValueError is raised. If None, no check is enforced.

    Returns
    -------
    instance of cls
    """
    meta_path = Path(meta_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_class = meta.get("class")
    if expected_class is not None and meta_class != expected_class:
        raise ValueError(f"Unexpected class in meta: {meta_class!r}, expected {expected_class!r}")

    directory = meta_path.parent
    dataset_name = meta.get("dataset_name", "unknown_dataset")
    state_names = meta.get("state_names", None)
    event_names = meta.get("event_names", None)
    baseline_csv = directory / meta["baseline_csv"]
    states_csv = directory / meta["states_csv"]

    baseline = pd.read_csv(baseline_csv)
    states = pd.read_csv(states_csv)

    baseline_features = features_from_json(meta.get("baseline_features", {}))
    state_features = features_from_json(meta.get("state_features", {}))

    # Optional tmat
    tmat = None
    tmat_csv = meta.get("tmat_csv", None)
    if tmat_csv is not None:
        tmat_path = directory / tmat_csv
        tmat_df = pd.read_csv(tmat_path)
        tmat = tmat_df.to_numpy()

    return cls.from_dataframes(
        dataset_name=dataset_name,
        baseline=baseline,
        states=states,
        baseline_features=baseline_features,
        state_features=state_features,
        state_names=state_names,
        event_names=event_names,
        tmat=tmat,
    )