import numpy as np
import pandas as pd
import torch
from typing import Sequence, Dict, Any
import random

def set_global_seed(seed: int = 42):
    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def discrete_variables_transformation(
    data,
    feat_types: Sequence[Dict[str, Any]],
) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.asarray(data)
    N, _ = data_np.shape

    out_cols: list[np.ndarray] = []
    ind_ini = 0

    for ft in feat_types:
        ftype = str(ft["type"]).lower()
        width = int(ft["nclass"])
        levels = ft.get("levels", None)

        ind_end = ind_ini + width
        subset = data_np[:, ind_ini:ind_end]  # [N, width]

        if ftype in ("cat", "disc", "bin", "state"):
            idx = subset.argmax(axis=1)  # [N]
            if levels is not None:
                levels_arr = np.asarray(levels, dtype=object)
                vals = levels_arr[idx]  # [N]
            else:
                vals = idx.astype(np.float32)
            out_cols.append(vals.reshape(N, 1))

        elif ftype == "ord":
            idx = subset.sum(axis=1).astype(int) - 1  # [N]
            idx = np.clip(idx, 0, width - 1)
            if levels is not None:
                levels_arr = np.asarray(levels, dtype=object)
                vals = levels_arr[idx]
            else:
                vals = idx.astype(np.float32)
            out_cols.append(vals.reshape(N, 1))

        else:
            out_cols.append(subset)

        ind_ini = ind_end

    return np.concatenate(out_cols, axis=1)


def align_dtypes_like_template(
    df: pd.DataFrame,
    template_df: pd.DataFrame,
) -> pd.DataFrame:
    for col in df.columns:
        if col not in template_df.columns:
            continue

        tmpl_col = template_df[col]
        target_dtype = tmpl_col.dtype

        if pd.api.types.is_categorical_dtype(tmpl_col):
            df[col] = pd.Categorical(
                df[col],
                categories=tmpl_col.cat.categories,
                ordered=tmpl_col.cat.ordered,
            )
        else:
            try:
                df[col] = df[col].astype(target_dtype)
            except (TypeError, ValueError):
                pass

    return df
