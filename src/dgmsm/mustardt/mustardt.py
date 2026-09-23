import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple, Any
import torch.nn.functional as F

from .loss import MuStaRDTLoss
from .mlp import MLP

class MuStaRDT(nn.Module):
    def __init__(
        self,
        graph,
        input_size: int,
        n_time: int,
        f_dims: List[int],
        f_latent: int,
        dropout: float = 0.1,
        aux_weight: float = 0.0,
    ):
        super().__init__()
        self.graph = graph
        self.v_dim = graph.n_nonterminal_states
        self.n_events = graph.n_events
        self.n_time = n_time
        self.input_size = input_size
        self.lossfn = MuStaRDTLoss(aux_weight  =   aux_weight)

        # One MLP per nonterminal state
        self.msm_mlps = nn.ModuleList([
            MLP(
                in_features=input_size,
                hidden_layers=f_dims,
                out_features=f_latent,
                dropout=dropout,
            )
            for _ in range(self.v_dim)
        ])

        # per-state linear heads for p_logits (event choice)
        self.p_out = nn.ModuleList([
            nn.Linear(f_latent, self.graph.events_per_state[state_idx].numel())
            for state_idx in range(self.v_dim)
        ])

        # per-event linear heads for f_logits (time distribution)
        self.f_out = nn.ModuleList([
            nn.Linear(f_latent, self.n_time)
            for _ in range(self.n_events)
        ])

    def forward(
        self,
        state_input: torch.Tensor,  # [B, T, D]
        state_idx: torch.Tensor,    # [B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state_input.dim() != 3:
            raise ValueError(f"state_input must have shape [B, T, D], got {tuple(state_input.shape)}")
        if state_idx.dim() != 2:
            raise ValueError(f"state_idx must have shape [B, T], got {tuple(state_idx.shape)}")

        B, T, D = state_input.shape
        if D != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {D}")
        if state_idx.shape != (B, T):
            raise ValueError("state_idx must match first two dims of state_input")

        p_flat, f_flat = self.step_forward(
            input_flat=state_input.reshape(B * T, D),
            idx_flat=state_idx.reshape(B * T),
        )

        p_logits = p_flat.view(B, T, self.n_events)
        f_logits = f_flat.view(B, T, self.n_events, self.n_time)
        return p_logits, f_logits

    def compute_loss(
        self,
        p_logits: torch.Tensor,
        f_logits: torch.Tensor,
        labels: torch.Tensor,
        return_components: bool = False,
    ):
        return self.lossfn(
            p_logits=p_logits,
            f_logits=f_logits,
            labels=labels,
            return_components=return_components,
        )


    def hazard_to_pmf_with_tail(
        self,
        f_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = f_logits / temperature
        log_h = F.logsigmoid(scaled)       # [..., K]
        log_1mh = F.logsigmoid(-scaled)    # [..., K]

        cum_log_surv = torch.zeros_like(log_1mh)
        cum_log_surv[..., 1:] = torch.cumsum(log_1mh[..., :-1], dim=-1)

        log_pmf = log_h + cum_log_surv
        pmf = log_pmf.exp()                        # [..., K]
        tail = log_1mh.sum(dim=-1).exp()          # [...]

        return pmf, tail


    def hazard_to_pmf(self, f_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        pmf, _ = self.hazard_to_pmf_with_tail(f_logits, temperature=temperature)
        return pmf

    def hazard_to_survival(
        self,
        f_logits: torch.Tensor,
        temperature=1.0,
    ) -> torch.Tensor:
        """
        Convert hazard logits to survival curve S(k) = P(T > k).
        """
        log_1mh = F.logsigmoid(-f_logits/ temperature)                        # [..., K]
        cum_log_surv = torch.cumsum(log_1mh, dim=-1)             # [..., K]
        return cum_log_surv.exp()                                 # P(T > k)
       
    @staticmethod
    def renewal_survival(
        p_logits: torch.Tensor,
        f_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-visit overall survival S(k) = sum_e p(e) * S_e(k).

        Uses logistic-hazard parameterization:
            h_e(k) = sigmoid(f_logits[..., e, k])
            S_e(k) = prod_{j=0}^{k} (1 - h_e(j))
        """
        # [batch_size, max_path, E, K]
        log_1mh = F.logsigmoid(-f_logits)
        cum_log_surv = torch.cumsum(log_1mh, dim=-1)    # log S_e(k)
        S_event = cum_log_surv.exp()                     # [B, max_path, E, K]

        p = F.softmax(p_logits, dim=-1)                  # [B, max_path, E]
        survival = (p.unsqueeze(-1) * S_event).sum(dim=2)  # [B, max_path, K]
        return survival

    def dgmsm_forward(
        self,
        z_n: torch.Tensor,
        h_seq: torch.Tensor,
        t: torch.Tensor,
        state_idx: torch.Tensor,
        z_0: torch.Tensor,
        s: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = z_n.shape

        if t.dim() == 2:
            t = t.unsqueeze(-1)

        z_0_rep = z_0.unsqueeze(1).expand(B, T, -1)
        s_rep   = s.unsqueeze(1).expand(B, T, -1)

        state_input = torch.cat([z_0_rep, s_rep, z_n, h_seq, t], dim=-1)
        p_logits, f_logits = self.forward(state_input=state_input, state_idx=state_idx)
        return {"p_logits": p_logits, "f_logits": f_logits}

    @torch.no_grad()
    def transition_step(
        self,
        state_idx: torch.Tensor,
        t_curr: torch.Tensor,
        h_curr: torch.Tensor,
        z_curr: torch.Tensor,
        z_0: torch.Tensor,
        s: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bad = (state_idx < 0) | (state_idx >= self.v_dim)
        if bad.any():
            raise RuntimeError(
                f"transition_step requires nonterminal states in 0..{self.v_dim-1}, "
                f"got {state_idx[bad].tolist()}"
            )

        device = state_idx.device
        B = state_idx.size(0)

        out = self.dgmsm_step(
            z_n=z_curr,
            h_prev=h_curr,
            t_n=t_curr,
            state_idx=state_idx,
            z_0=z_0,
            s=s,
        )

        p_logits = out["p_logits"]
        f_logits = out["f_logits"]

        reachable = self.graph.state_event_mask[state_idx]
        p_logits = p_logits.masked_fill(~reachable, float("-inf"))
        p_all = F.softmax(p_logits, dim=-1)
        event_idx = torch.multinomial(p_all, 1).squeeze(-1)

        f_event = f_logits[torch.arange(B, device=device), event_idx, :]
        pmf, tail = self.hazard_to_pmf_with_tail(f_event, temperature=temperature)

        time_aug = torch.cat([pmf, tail.unsqueeze(-1)], dim=-1)
        draw = torch.multinomial(time_aug, 1).squeeze(-1)

        event_observed = draw != self.n_time
        time_idx = draw
        event_idx = event_idx.masked_fill(~event_observed, -1)

        return event_idx, time_idx, event_observed

    def dgmsm_step(
        self,
        z_n: torch.Tensor,       # [N, Z]
        h_prev: torch.Tensor,    # [N, H]   = history before current visit
        t_n: torch.Tensor,       # [N, 1] or [N]
        state_idx: torch.Tensor, # [N]
        z_0: torch.Tensor,       # [N, Z0]
        s: torch.Tensor,         # [N, S]
    ) -> Dict[str, torch.Tensor]:
        if t_n.dim() == 1:
            t_n = t_n.unsqueeze(-1)

        state_input = torch.cat(
            [z_0, s, z_n, h_prev, t_n],
            dim=-1,
        )

        if state_input.shape[-1] != self.input_size:
            raise ValueError(
                f"Constructed state_input has dim {state_input.shape[-1]}, "
                f"expected {self.input_size}"
            )

        p_logits, f_logits = self.step_forward(
            input_flat=state_input,
            idx_flat=state_idx,
        )
        return {"p_logits": p_logits, "f_logits": f_logits}

    def step_forward(
        self,
        input_flat: torch.Tensor,   # [N, input_size]
        idx_flat: torch.Tensor,     # [N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core one-step/per-row MSM forward.

        Returns
        -------
        p_logits_flat : [N, n_events]
        f_logits_flat : [N, n_events, n_time]
        """
        if input_flat.dim() != 2:
            raise ValueError(f"input_flat must have shape [N, D], got {tuple(input_flat.shape)}")
        if idx_flat.dim() != 1:
            idx_flat = idx_flat.view(-1)

        N, dim = input_flat.shape
        if dim != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {dim}")
        if idx_flat.shape[0] != N:
            raise ValueError("idx_flat and input_flat must have matching first dimension")

        bad = (idx_flat >= self.v_dim) | (idx_flat < -1)
        if bad.any():
            raise ValueError(f"state_idx contains invalid values: {idx_flat[bad].tolist()}")

        p_logits_flat = input_flat.new_full((N, self.n_events), -1e9)
        f_logits_flat = input_flat.new_full((N, self.n_events, self.n_time), -1e9)

        for s in range(self.v_dim):
            mask_s = (idx_flat == s)
            if not mask_s.any():
                continue

            idx_s_flat = mask_s.nonzero(as_tuple=False).squeeze(-1)
            inp_s = input_flat[idx_s_flat]                      # [Ns, D]
            h_state = self.msm_mlps[s](inp_s)                   # [Ns, f_latent]

            event_indices = self.graph.events_per_state[s]      # [Es]
            if event_indices.numel() == 0:
                continue

            # Event-choice logits for reachable events from state s
            p_state = self.p_out[s](h_state)                    # [Ns, Es]
            p_logits_flat[idx_s_flat.unsqueeze(1), event_indices] = p_state

            # Time logits per reachable event
            for e in event_indices.tolist():
                f_state_event = self.f_out[e](h_state)          # [Ns, n_time]
                f_logits_flat[idx_s_flat, e, :] = f_state_event

        return p_logits_flat, f_logits_flat



    def competing_risks(
        self,
        p_logits: torch.Tensor,   # [..., E]
        f_logits: torch.Tensor,   # [..., E, K]
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        p = F.softmax(p_logits, dim=-1)                              # [..., E]
        pmf_cond = self.hazard_to_pmf(f_logits, temperature)         # [..., E, K]
        joint_pmf = p.unsqueeze(-1) * pmf_cond                       # [..., E, K]
        cif = torch.cumsum(joint_pmf, dim=-1)                        # [..., E, K]
        survival = 1.0 - cif.sum(dim=-2)                             # [..., K]
        return {
            "p": p,
            "pmf_cond": pmf_cond,
            "joint_pmf": joint_pmf,
            "cif": cif,
            "survival": survival,
        }