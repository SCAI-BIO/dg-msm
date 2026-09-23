import torch
import torch.nn as nn
import numpy as np
import math
import logging
from typing import Sequence, Dict, Any, List, Tuple, Optional

def validate_and_count_transitions(tmat: np.ndarray) -> int:
    """
    Validate tmat and return the number of distinct global transitions.

    tmat[i, j] = k (1..n_events) if there is a transition i->j with global ID k;
    NaN otherwise.
    """
    tmat = np.asarray(tmat, dtype=float)
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Transition matrix must be square: [n_states, n_states].")

    vals = tmat[~np.isnan(tmat)]
    if vals.size == 0:
        raise ValueError("Empty transition matrix.")

    vals_int = vals.astype(int)
    if not np.all(vals == vals_int):
        raise ValueError("Non-NaN entries must be integers.")

    n = vals_int.size
    if not np.array_equal(np.sort(vals_int), np.arange(1, n + 1)):
        raise ValueError("Transitions must be numbered 1..N without gaps or duplicates.")
    return n


def compute_tmat_stats(
    tmat: np.ndarray,
    max_path: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str,int]:
    if logger is None:
        logger = logging.getLogger(__name__)

    tmat = np.asarray(tmat, dtype=float)
    nonterminal_mask = ~np.all(np.isnan(tmat), axis=1)
    nonterminal_indices = np.where(nonterminal_mask)[0]
    n_states = int(nonterminal_indices.size)
    nonterm_set = set(nonterminal_indices)

    # adjacency only among non-terminal states
    adj = {i: [] for i in nonterminal_indices}
    for i in nonterminal_indices:
        js = np.where(~np.isnan(tmat[i]))[0]
        adj[i] = [j for j in js if j in nonterm_set]

    visited: Dict[int, int] = {}
    visiting: set[int] = set()

    def dfs(u: int) -> int:
        """
        Returns longest path length (in #nonterminal visits) starting at u.
        Raises RuntimeError if a cycle is detected.
        """
        if u in visiting:
            # cycle among non-terminal states
            raise RuntimeError("Cycle detected in tmat among non-terminal states.")
        if u in visited:
            return visited[u]

        visiting.add(u)
        if not adj[u]:
            length = 1
        else:
            max_child = 0
            for v in adj[u]:
                child_len = dfs(v)
                if child_len > max_child:
                    max_child = child_len
            length = 1 + max_child
        visiting.remove(u)
        visited[u] = length
        return length

    # compute theoretical longest path and detect cycles
    try:
        longest = 0
        for u in nonterminal_indices:
            l = dfs(u)
            if l > longest:
                longest = l
        theoretical_max = longest   # finite longest path
        has_cycle = False
    except RuntimeError:
        theoretical_max = math.inf         # infinite longest path (cycle)
        has_cycle = True

    if has_cycle:
        # infinite path length -> must have max_path
        if max_path is None:
            raise ValueError(
                "tmat has cycles (infinite possible path length); "
                "provide max_path to define a cutoff."
            )
        tmat_max_path = max_path
        logger.warning(
            "Cycle detected among non-terminal states; cutting path at max_path=%d.",
            max_path,
        )
    else:
        if max_path is None:
            tmat_max_path = theoretical_max
            logger.info(
                "Longest path over non-terminal states: %d", int(tmat_max_path)
            )
        else:
            if theoretical_max > max_path:
                tmat_max_path = max_path
                logger.warning(
                    "Theoretical longest path (%d) exceeds max_path=%d; cutting at max_path.",
                    int(theoretical_max),
                    max_path,
                )
            else:
                tmat_max_path = theoretical_max
                logger.info(
                    "Theoretical longest path (%d) is within max_path=%d.",
                    int(theoretical_max),
                    max_path,
                )

    return {
        'n_nonterminal_states': n_states, 
        'max_path': tmat_max_path,
        }

class MultiStateGraph(nn.Module):
    def __init__(
        self,
        tmat,
        n_nonterminal_states: Optional[int] = None,
        max_path: Optional[int] = None,
    ):
        super().__init__()

        # Keep original tmat as numpy for the stats utils
        tmat_np = np.asarray(tmat, dtype=float)
        self.transition_matrix = tmat_np
        self.n_total_states = tmat_np.shape[0]

        # Infer / validate n_nonterminal_states and max_path
        if n_nonterminal_states is None or max_path is None:
            stats = compute_tmat_stats(tmat_np, max_path=max_path)
            if n_nonterminal_states is None:
                n_nonterminal_states = stats["n_nonterminal_states"]
            if max_path is None:
                max_path = stats["max_path"]
        else:
            stats = compute_tmat_stats(tmat_np, max_path=max_path)
            assert n_nonterminal_states == stats["n_nonterminal_states"], (
                "n_nonterminal_states mismatch"
            )

        self.n_nonterminal_states = n_nonterminal_states
        # Optional alias if you like the old name
        self.v_dim = n_nonterminal_states
        self.max_path = max_path

        # Count distinct events
        self.n_events = validate_and_count_transitions(tmat_np)

        # Build event structures
        state_event_mask = np.zeros((self.n_total_states, self.n_events), dtype=bool)
        event_from = np.empty(self.n_events, dtype=np.int64)
        event_to   = np.empty(self.n_events, dtype=np.int64)

        for i in range(self.n_total_states):
            for j in range(self.n_total_states):
                if not np.isnan(tmat_np[i, j]):
                    k = int(tmat_np[i, j])  # event index in 1..n_events
                    e = k - 1               # 0-based
                    event_from[e] = i
                    event_to[e]   = j
                    state_event_mask[i, e] = True

        # Register buffers so they follow the model to GPU/CPU etc.
        self.register_buffer("state_event_mask", torch.from_numpy(state_event_mask))
        self.register_buffer("event_from", torch.from_numpy(event_from))
        self.register_buffer("event_to", torch.from_numpy(event_to))

        # events_per_state[i]: tensor of event indices available from state i
        events_per_state: List[torch.Tensor] = []
        for i in range(self.n_total_states):
            e_idx = np.nonzero(state_event_mask[i])[0].astype(np.int64)
            e_idx_tensor = torch.from_numpy(e_idx).long()
            self.register_buffer(f"events_per_state_{i}", e_idx_tensor)
            events_per_state.append(e_idx_tensor)

        self.events_per_state = events_per_state
        
        # terminal states / events
        terminal_states = np.arange(
            self.n_nonterminal_states,
            self.n_total_states,
            dtype=np.int64,
        )
        self.register_buffer(
            "terminal_states",
            torch.from_numpy(terminal_states),
        )

        # boolean mask over events: True if event leads to a terminal state
        terminal_event_mask = (event_to >= self.n_nonterminal_states)
        self.register_buffer(
            "terminal_event_mask",
            torch.from_numpy(terminal_event_mask.astype(np.bool_)),
        )

        # indices (0-based) of terminal events
        terminal_event_idx = np.nonzero(terminal_event_mask)[0].astype(np.int64)
        self.register_buffer(
            "terminal_event_idx",
            torch.from_numpy(terminal_event_idx),
        )

    @property
    def terminal_event_ids(self) -> torch.Tensor:
        """
        1-based event IDs for events that lead to any terminal state.
        """
        # event indices in code are 0-based; labels use 1-based IDs
        return self.terminal_event_idx + 1
    
    def _apply(self, fn):
        super()._apply(fn)
        self.events_per_state = [
            getattr(self, f"events_per_state_{i}")
            for i in range(self.n_total_states)
        ]
        return self