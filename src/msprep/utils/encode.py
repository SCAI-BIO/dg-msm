from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def encode_col_for_corr(
    col: pd.Series,
    schema: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Encode a column for correlation/diagnostics.

    Differences from encode_col:
    - Missing values (and unknown categories) are encoded as NaN, not mapped
      to the first level or to 0.
    """
    ftype = str(schema["type"]).lower()
    nclass = int(schema["nclass"])
    levels = schema.get("levels", None)
    is_nan = col.isna().to_numpy()

    if ftype in ["cat", "disc", "bin", "state"]:
        if levels is None:
            raise ValueError("Categorical/state feature must have 'levels' in schema.")

        cat = pd.Categorical(col, categories=levels, ordered=True)
        codes = np.array(cat.codes, copy=True)  # -1 for NaN or category not in levels

        # Unknown categories (codes == -1 but not is_nan) – treat as missing for diagnostics
        is_unknown = (~is_nan) & (codes == -1)
        missing_mask = is_nan | is_unknown

        # Initialize all NaN
        encoded_col = np.full((len(codes), nclass), np.nan, dtype=np.float32)

        valid = ~missing_mask
        if valid.any():
            encoded_col[valid] = np.eye(nclass, dtype=np.float32)[codes[valid]]

        if ftype == "state":
            feat_type = {"type": "state", "nclass": nclass, "levels": levels}
        else:
            feat_type = {"type": "cat", "nclass": nclass, "levels": levels}

    elif ftype == "ord":
        if levels is None:
            raise ValueError("Ordinal feature must have 'levels' in schema.")

        cat = pd.Categorical(col, categories=levels, ordered=True)
        codes = np.array(cat.codes, copy=True)
        is_unknown = (~is_nan) & (codes == -1)
        missing_mask = is_nan | is_unknown

        # ordinary thermometer encoding for valid entries
        thermometer = np.zeros((len(codes), nclass + 1), dtype=np.float32)
        thermometer[:, 0] = 1.0

        valid = ~missing_mask
        if valid.any():
            thermometer[valid, 1 + codes[valid]] = -1.0
        thermometer = np.cumsum(thermometer, axis=1)
        encoded_col = thermometer[:, :-1]

        # set rows with missing/unknown to NaN
        encoded_col[missing_mask, :] = np.nan

        feat_type = {"type": "ord", "nclass": nclass, "levels": levels}

    elif ftype in ["real", "pos", "time"]:
        vals = col.to_numpy(np.float32)
        # keep NaNs as NaNs; no imputation for correlation
        encoded_col = vals[:, None]
        feat_type = {"type": ftype, "nclass": 1, "levels": None}

    else:
        raise ValueError(f"Unsupported HIVAE type '{ftype}'")

    return encoded_col, feat_type


def encode_df_for_corr(
    df: pd.DataFrame,
    ftypes: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Encode columns in df according to ftypes using encode_col_for_corr.

    Returns
    -------
    df_enc : DataFrame
        Numeric encoded features, suitable for correlation/diagnostics.

    meta : dict
        Mapping encoded_name -> {"base": original_name, "type": str, "level": Any}
    """
    encoded_parts = []
    encoded_names: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}

    for name, schema in ftypes.items():
        if name not in df.columns:
            continue
        col = df[name]
        encoded_col, feat_type = encode_col_for_corr(col, schema)
        t = feat_type["type"]
        nclass = int(feat_type["nclass"])
        levels = feat_type.get("levels", None)

        if t in {"cat", "state"}:
            level_names = [str(lv) for lv in (levels or range(nclass))]
            col_names = [f"{name}={lv}" for lv in level_names]
            for lv, enc_name in zip(level_names, col_names):
                meta[enc_name] = {"base": name, "type": t, "level": lv}

        elif t == "ord":
            level_names = [str(lv) for lv in (levels or range(nclass))]
            col_names = [f"{name}<= {lv}" for lv in level_names]
            for lv, enc_name in zip(level_names, col_names):
                meta[enc_name] = {"base": name, "type": t, "level": lv}

        elif t in {"real", "pos", "time"}:
            col_names = [name]
            meta[name] = {"base": name, "type": t, "level": None}
        else:
            continue

        encoded_parts.append(encoded_col)
        encoded_names.extend(col_names)

    if encoded_parts:
        X = np.concatenate(encoded_parts, axis=1).astype(np.float32)
        df_enc = pd.DataFrame(X, index=df.index, columns=encoded_names)
    else:
        df_enc = pd.DataFrame(index=df.index)

    return df_enc, meta