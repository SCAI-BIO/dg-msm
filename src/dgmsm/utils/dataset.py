from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def encode_col(col: pd.Series, schema: Dict[str, Any]):
    ftype = str(schema["type"]).lower()
    nclass = int(schema["nclass"])
    levels = schema.get("levels", None)
    is_nan = col.isna().to_numpy()

    if ftype in ["cat", "disc", "bin", "state"]:
        if levels is None:
            raise ValueError("Categorical/state feature must have 'levels' in schema.")
        cat = pd.Categorical(col, categories=levels, ordered=True)
        codes = np.array(cat.codes, copy=True)  # [-1, 0, ..., nclass-1]
        codes[is_nan] = 0
        encoded_col = np.eye(nclass, dtype=np.float32)[codes.clip(min=0)]
        if ftype == "state":
            feat_type = {"type": "state", "nclass": nclass, "levels": levels}
        else:
            feat_type = {"type": "cat", "nclass": nclass, "levels": levels}

    elif ftype == "ord":
        if levels is None:
            raise ValueError("Ordinal feature must have 'levels' in schema.")
        cat = pd.Categorical(col, categories=levels, ordered=True)
        codes = np.array(cat.codes, copy=True)
        codes[is_nan] = 0

        thermometer = np.zeros((len(codes), nclass + 1), dtype=np.float32)
        thermometer[:, 0] = 1.0
        thermometer[np.arange(len(codes)), 1 + codes.clip(min=0)] = -1.0
        thermometer = np.cumsum(thermometer, axis=1)
        encoded_col = thermometer[:, :-1]
        feat_type = {"type": "ord", "nclass": nclass, "levels": levels}

    elif ftype in ["real", "pos", "time"]:
        vals = col.to_numpy(np.float32)
        vals[is_nan] = 0.0
        encoded_col = vals[:, None]
        feat_type = {"type": ftype, "nclass": 1, "levels": None}

    else:
        raise ValueError(f"Unsupported HIVAE type '{ftype}'")

    return encoded_col, feat_type


class MultiStateData(Dataset):
    def __init__(
        self,
        states_df: pd.DataFrame,
        states_ftypes: Dict[str, Dict[str, Any]], 
        baseline_df: pd.DataFrame,
        baseline_ftypes: Dict[str, Dict[str, Any]],
        n_nonterminal_states: int,
        max_path: int,
    ) -> None:
        """
        Output tensors per patient i:
            x_base[i]:   [D_base]                          concatenated one-hot/real baseline features
            m_base[i]:   [n_baseline_features]             1 if observed, 0 if missing (per feature)
            x_state[i]:  [max_path, 1+v_dim+state_x_dim]   [Tstart, state_onehot, state_covs], padded with 0
            m_state[i]:  [max_path, 2+n_user_features]     [Tstart_mask, state_mask, stae_cov_masks], 0 for padding
            labels[i]:   [max_path, 5]                     [state_id, duration, event_id, Tstart, Tstop]
        """
        self.n_nonterminal_states = int(n_nonterminal_states)
        self.max_path = int(max_path)

        # sort visits chronologically per person
        states_df = states_df.sort_values(
            ["person_id", "visit", "Tstart"]
        ).reset_index(drop=True)

        if not states_df.empty:
            max_state_in_data = int(states_df["state"].max())
            if max_state_in_data > self.n_nonterminal_states:
                raise ValueError(
                    f"states_df contains state={max_state_in_data}, "
                    f"but n_nonterminal_states={self.n_nonterminal_states}."
                )
            # check observed max path vs provided max_path
            max_len_obs = int(states_df.groupby("person_id").size().max())
            if max_len_obs > self.max_path:
                raise ValueError(
                    f"Observed maximum path length {max_len_obs} in states_df "
                    f"exceeds max_path={self.max_path}."
                )

        # baseline index 
        baseline_df = baseline_df.reset_index(drop=True)
        person_to_index = {pid: i for i, pid in enumerate(baseline_df["person_id"].values)}

        # encode baseline features 
        base_encoded_cols: List[np.ndarray] = []
        base_mask_feat_cols: List[np.ndarray] = []
        self.baseline_feat_types: List[Dict[str, Any]] = []

        N_base = baseline_df.shape[0]
        base_feature_names = list(baseline_ftypes.keys())

        for name in base_feature_names:
            if name not in baseline_df.columns:
                raise ValueError(f"Baseline feature '{name}' not in baseline_df.columns")
            col = baseline_df[name]
            encoded_col, feat_type = encode_col(col, baseline_ftypes[name])

            is_nan = col.isna().to_numpy()
            mask_feat = (~is_nan).astype(np.float32)

            base_encoded_cols.append(encoded_col)
            base_mask_feat_cols.append(mask_feat)
            self.baseline_feat_types.append(feat_type)

        if base_encoded_cols:
            base_data_all = np.concatenate(base_encoded_cols, axis=1).astype(np.float32)
        else:
            base_data_all = np.zeros((N_base, 0), dtype=np.float32)

        if base_mask_feat_cols:
            base_mask_feat_all = np.stack(base_mask_feat_cols, axis=1).astype(np.float32)
        else:
            base_mask_feat_all = np.zeros((N_base, 0), dtype=np.float32)

        self.D_base = base_data_all.shape[1]

        # encode state features (sequence over visits) 
        N_states_rows = states_df.shape[0]
        state_feature_names = list(states_ftypes.keys())

        if "state" in states_ftypes:
            raise ValueError("'state' is reserved as label and automatic covariate; do not include it in states_ftypes.")
        if "Tstart" in states_ftypes:
            raise ValueError("'Tstart' is reserved as label and automatic covariate; do not include it in states_ftypes.")

        # 0) add 'state' (one-hot) as first covariate per visit
        state_col = states_df["state"]
        state_schema = {
            "type": "state",
            "nclass": self.n_nonterminal_states,
            "levels": list(range(1, self.n_nonterminal_states + 1)),
        }
        encoded_state, feat_state = encode_col(state_col, state_schema)
        is_nan_state = state_col.isna().to_numpy()
        mask_state = (~is_nan_state).astype(np.float32)    # [N_rows]

        # 1) add 'Tstart' as second covariate per visit (continuous)
        start_col = states_df["Tstart"]
        start_schema = {
            "type": "time",
            "nclass": 1,
            "levels": None,
        }
        encoded_start, feat_tstart = encode_col(start_col, start_schema)  # [N_rows, 1]
        is_nan_start = start_col.isna().to_numpy()
        mask_start = (~is_nan_start).astype(np.float32)

        state_encoded_cols: List[np.ndarray] = [encoded_start, encoded_state]
        state_mask_feat_cols: List[np.ndarray] = [mask_start, mask_state]
        self.state_feat_types: List[Dict[str, Any]] = [feat_tstart, feat_state]

        # 2) remaining state covariates from states_ftypes (true covariates)
        for name in state_feature_names:
            if name not in states_df.columns:
                raise ValueError(f"State feature '{name}' not in states_df.columns")
            col = states_df[name]
            encoded_col, feat_type = encode_col(col, states_ftypes[name])

            is_nan = col.isna().to_numpy()
            mask_feat = (~is_nan).astype(np.float32)

            state_encoded_cols.append(encoded_col)
            state_mask_feat_cols.append(mask_feat)
            self.state_feat_types.append(feat_type)

        if state_encoded_cols:
            state_data_all = np.concatenate(state_encoded_cols, axis=1).astype(np.float32)  # [N_rows, D_state]
            D_state = state_data_all.shape[1]
        else:
            state_data_all = np.zeros((N_states_rows, 0), dtype=np.float32)
            D_state = 0

        if state_mask_feat_cols:
            state_mask_feat_all = np.stack(state_mask_feat_cols, axis=1).astype(np.float32) # [N_rows, 2 + len(states_ftypes)]
        else:
            state_mask_feat_all = np.zeros((N_states_rows, 0), dtype=np.float32)

        self.D_state = D_state

        # build per-person tensors 
        self.x_base: List[torch.Tensor] = []
        self.m_base: List[torch.Tensor] = []
        self.x_state: List[torch.Tensor] = []  # [max_path, D_state]
        self.m_state: List[torch.Tensor] = []  # [max_path, n_state_features]
        self.labels: List[torch.Tensor] = []   # [max_path, y_dim]
        self.labels_cont: List[torch.Tensor] = []

        persons = states_df["person_id"].unique().tolist()
        self.persons = persons

        label_cols = ["state", "time", "event", "Tstart", "Tstop"]
        cont_cols  = ["sojourn_time", "sojourn_frac"]
        yc_dim     = len(cont_cols)
        y_dim      = len(label_cols)
        assert len(self.state_feat_types) == state_mask_feat_all.shape[1]
        n_state_features = len(self.state_feat_types)

        for pid in persons:
            g = states_df[states_df["person_id"] == pid]
            if g.empty:
                continue

            row_idx = g.index.to_numpy()
            n_visits = len(row_idx)
            if n_visits > self.max_path:
                raise ValueError(
                    f"Person {pid} has {n_visits} visits, which exceeds max_path={self.max_path}."
                )

            try:
                base_pos = person_to_index[pid]
            except KeyError:
                raise ValueError(f"person_id {pid} in states_df but not in baseline_df")

            x_base_i = base_data_all[base_pos, :]
            m_base_i = base_mask_feat_all[base_pos, :]

            x_state_full = np.zeros((self.max_path, D_state), dtype=np.float32)
            m_state_full = np.zeros((self.max_path, n_state_features), dtype=np.float32)
            labels_full = np.zeros((self.max_path, y_dim), dtype=np.int64)
            labels_c_full = np.zeros((self.max_path, yc_dim), dtype=np.float32)

            # fill first n_visits in chronological order
            for t, idx in enumerate(row_idx):
                x_state_full[t, :] = state_data_all[idx, :]
                m_state_full[t, :] = state_mask_feat_all[idx, :]

                lab_row = g.loc[idx, label_cols]
                labels_full[t, :] = pd.to_numeric(lab_row, errors="raise").to_numpy(dtype="int64")
                labc_row = g.loc[idx, cont_cols]                                 
                labels_c_full[t, :] = labc_row.to_numpy(dtype=np.float32)        

            self.x_base.append(torch.from_numpy(x_base_i))
            self.m_base.append(torch.from_numpy(m_base_i))
            self.x_state.append(torch.from_numpy(x_state_full))
            self.m_state.append(torch.from_numpy(m_state_full))
            self.labels.append(torch.from_numpy(labels_full))
            self.labels_cont.append(torch.from_numpy(labels_c_full))
        self.n_persons = len(self.x_base)

    def __len__(self) -> int:
        return self.n_persons

    def __getitem__(self, idx: int):
        return (
            (self.x_base[idx].float(), self.m_base[idx].float()),
            (self.x_state[idx].float(), self.m_state[idx].float()),
            self.labels[idx],
        )

    def states_present(self) -> set[int]:
        """
        Set of nonterminal state ids (1-based) that occur at least once
        across all patients' visit histories.
        """
        if self.n_persons == 0:
            return set()

        # self.labels: list of [max_path, 5] tensors, col 0 = state id (0 = padding)
        all_states = torch.stack(self.labels, dim=0)[:, :, 0]   # [N, max_path]
        vals = torch.unique(all_states)
        return {int(v) for v in vals.tolist() if v != 0}

    def subset_persons(self, person_ids) -> "MultiStateData":
        """
        Return a new MultiStateData containing only the given person_ids.
        """
        pid_set = set(person_ids)
        indices = [i for i, pid in enumerate(self.persons) if pid in pid_set]

        if not indices:
            raise ValueError("subset_persons: no matching person_ids found.")

        new = object.__new__(MultiStateData)

        # Copy metadata
        new.n_nonterminal_states    = self.n_nonterminal_states
        new.max_path                = self.max_path
        new.D_base                  = self.D_base
        new.D_state                 = self.D_state
        new.baseline_feat_types     = self.baseline_feat_types
        new.state_feat_types        = self.state_feat_types

        # Slice per-patient lists
        new.persons     = [self.persons[i] for i in indices]
        new.x_base      = [self.x_base[i] for i in indices]
        new.m_base      = [self.m_base[i] for i in indices]
        new.x_state     = [self.x_state[i] for i in indices]
        new.m_state     = [self.m_state[i] for i in indices]
        new.labels      = [self.labels[i] for i in indices]
        new.labels_cont = [self.labels_cont[i] for i in indices]
        new.n_persons   = len(indices)
        return new


    def landmark_subset(
        self, 
        visit: int, # 1-based
        mask_future: bool = True,
        ) -> "MultiStateData":
        if not (1 <= visit <= self.max_path):
            raise ValueError(f"visit={visit} out of range [1, {self.max_path}].")

        # labels[:, 0] is the state column (1-indexed); 0 == padding
        indices = [
            i for i in range(self.n_persons)
            if int((self.labels[i][:, 0] != 0).sum()) >= visit
        ]

        if not indices:
            raise ValueError(
                f"landmark_subset: no eligible patients for visit {visit}."
            )

        new = object.__new__(MultiStateData)

        # Scalar / shared metadata 
        new.n_nonterminal_states = self.n_nonterminal_states
        new.max_path             = self.max_path
        new.D_base               = self.D_base
        new.D_state              = self.D_state
        new.baseline_feat_types  = self.baseline_feat_types
        new.state_feat_types     = self.state_feat_types

        # Baseline (unchanged) 
        new.persons = [self.persons[i] for i in indices]
        new.x_base  = [self.x_base[i]  for i in indices]
        new.m_base  = [self.m_base[i]  for i in indices]

        # State sequences 
        new.x_state = []
        new.m_state = []
        new.labels  = []
        new.labels_cont = [] 
        for i in indices:
            x_s = self.x_state[i].clone()
            m_s = self.m_state[i].clone()
            lb  = self.labels[i].clone()
            lc  = self.labels_cont[i].clone() 

            if mask_future:
                x_s[visit:] = 0.0
                m_s[visit:] = 0.0
                lb[visit:]  = 0
                lc[visit:]  = 0.0 

            new.x_state.append(x_s)
            new.m_state.append(m_s)
            new.labels.append(lb)
            new.labels_cont.append(lc)

        new.n_persons = len(indices)
        return new

    def state_hitting_subset(
        self,
        state: int,
        mask_future: bool = True,
    ) -> "MultiStateData":
        if not (1 <= state <= self.n_nonterminal_states):
            raise ValueError(
                f"state={state} out of range [1, {self.n_nonterminal_states}]."
            )

        indices: List[int] = []
        hit_positions: List[int] = []

        for i in range(self.n_persons):
            state_col = self.labels[i][:, 0]   # 0 = padding, real states are 1-based
            hits = (state_col == state).nonzero(as_tuple=False).squeeze(-1)
            if len(hits) == 0:
                continue
            indices.append(i)
            hit_positions.append(int(hits[0].item()))  # 0-based visit index

        if not indices:
            raise ValueError(
                f"state_hitting_subset: no patients ever visit state {state}."
            )

        new = object.__new__(MultiStateData)

        new.n_nonterminal_states = self.n_nonterminal_states
        new.max_path             = self.max_path
        new.D_base               = self.D_base
        new.D_state              = self.D_state
        new.baseline_feat_types  = self.baseline_feat_types
        new.state_feat_types     = self.state_feat_types

        new.persons = [self.persons[i] for i in indices]
        new.x_base  = [self.x_base[i]  for i in indices]
        new.m_base  = [self.m_base[i]  for i in indices]

        new.x_state = []
        new.m_state = []
        new.labels  = []
        new.labels_cont = []

        for i, hit in zip(indices, hit_positions):
            x_s = self.x_state[i].clone()
            m_s = self.m_state[i].clone()
            lb  = self.labels[i].clone()
            lc = self.labels_cont[i].clone() 

            if mask_future:
                x_s[hit + 1:] = 0.0
                m_s[hit + 1:] = 0.0
                lb[hit + 1:]  = 0
                lc[hit + 1:] = 0.0

            new.x_state.append(x_s)
            new.m_state.append(m_s)
            new.labels.append(lb)
            new.labels_cont.append(lc)

        new.n_persons = len(indices)
        new.hit_positions = hit_positions
        return new

def build_dataloader(
    states_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    batch_size: int,
    shuffle: bool = False,
    pin_memory: bool = True,
    **kwargs,
    ):
    dataset = MultiStateData(
        states_df=states_df,
        baseline_df=baseline_df,
        **kwargs,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )

    return dataloader


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype.is_floating_point:
        return torch.allclose(a, b, atol=0.0, rtol=0.0)
    return torch.equal(a, b)

def datasets_equal(a, b):
    meta_attrs = [
        "n_persons",
        "n_nonterminal_states",
        "max_path",
        "D_base",
        "D_state",
        "baseline_feat_types",
        "state_feat_types",
    ]

    for attr in meta_attrs:
        if getattr(a, attr) != getattr(b, attr):
            return False, f"Mismatch in {attr}"

    if a.persons != b.persons:
        return False, "Mismatch in persons"

    for name in ["x_base", "m_base", "x_state", "m_state", "labels", "labels_cont"]:
        xs = getattr(a, name)
        ys = getattr(b, name)

        if len(xs) != len(ys):
            return False, f"Mismatch in length of {name}"

        for i, (x, y) in enumerate(zip(xs, ys)):
            if not _tensor_equal(x, y):
                return False, f"Mismatch in {name}[{i}]"

    return True, "Datasets are equal"