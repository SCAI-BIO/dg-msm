import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging 
from typing import List, Dict, Any, Optional, Tuple, Union

# HIVAE modules
from .hivae.baseline_hivae import BaseHIVAE
from .hivae.state_vrnn import StateVRNN

# MuStaRDT modules
from .mustardt import MuStaRDT, EvalMuStaRDT

# helper functions
from .utils import training_loop, MultiStateGraph

class DGMSM(nn.Module):
    def __init__(
        self,
        tmat: np.ndarray, 
        z_dim: int,
        s_dim: int,
        y_dim: int,
        h_dim: int,
        base_ftypes: List[Dict[str, Any]],
        state_ftypes: List[Dict[str, Any]],

        # MuStaRDT
        n_time: int,
        f_dims: List[int],
        f_latent: int,
        dropout: float = 0.1,
        max_path: Optional[int] = None,
        n_nonterminal_states: Optional[int] = None,
        base_norm_params: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None, 
        state_norm_params: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None,            
        beta_elbo: float = 1.0,
        aux_weight:      float = 1.0,
        learn_init_state: bool = False,
        log: Optional[logging.Logger] = None,
        use_kendall: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.logger = log or logging.getLogger(__name__)
        self.n_time = n_time
        self.beta_elbo = beta_elbo
        self.learn_init_state = learn_init_state
        self.eps = 1e-8

        # Dimensions of encoded covariates 
        self.base_ftypes    = base_ftypes
        self.state_ftypes   = state_ftypes
        self.base_x_dim     = sum(int(f["nclass"]) for f in self.base_ftypes)
        self.state_x_dim    = sum(int(f["nclass"]) for f in self.state_ftypes)

        # latent dimensions
        self.s_dim = s_dim   # mixture component for the baseline
        self.z_dim = z_dim   # latent dimension (baseline / state)
        self.h_dim = h_dim   # memory dimension

        # Build graph submodule
        self.graph = MultiStateGraph(
            tmat=tmat,
            n_nonterminal_states=n_nonterminal_states,
            max_path=max_path,
        )

        self.v_dim   = self.graph.n_nonterminal_states
        self.max_path = self.graph.max_path
        self.n_events = self.graph.n_events

        # Baseline HIVAE model --------------------------------------------------------------------------------
        self.base_hivae = BaseHIVAE(
            base_x_dim=self.base_x_dim,
            base_ftypes=self.base_ftypes,
            y_dim=y_dim,
            s_dim=self.s_dim,
            z_dim=self.z_dim,
            base_norm_params=base_norm_params,
        )

        self.phi_z_0 = nn.Linear(self.z_dim, self.h_dim)


        # HIVAE-RNN for the patient trajectories-------------------------------------------------------------
        self.state_vrnn = StateVRNN(
            state_x_dim=self.state_x_dim,
            state_ftypes=self.state_ftypes,
            y_dim=y_dim,
            z_dim=self.z_dim,
            h_dim=self.h_dim,
            v_dim=self.v_dim,
            s_dim=self.s_dim,
            state_norm_params=state_norm_params,
        )
        # ------------------------------------------------------------------------------
        # Initial state
        init_state_input = self.s_dim + self.z_dim
        self.init_state = nn.Sequential(
            nn.Linear(init_state_input, 2 * init_state_input),
            nn.ReLU(),
            nn.Linear(2 * init_state_input, self.v_dim),  # logits for states 1..v
        )

        # Multi-state head (MuStaRDT) ---------------------------------------------------
        # For each observation (v_n, x_n), where
        #   - v_n is the one-hot encoded current state
        #   - x_n are the covariates observed at Tstart for this state
        # define state-specific MLPs: one per non-terminal state 

        # g(z_n, h_{n-1}, t_n, z_0, s)
        self.head_in_features = (
            self.s_dim +                                # s
            self.z_dim +                                # z_0
            self.z_dim +                                # z_n
            self.h_dim +                                # h_{n-1}
            1                                           # t_n
        )


        self.mustardt = MuStaRDT(
            graph=self.graph,
            input_size=self.head_in_features,
            n_time=self.n_time,
            f_dims=f_dims,
            f_latent=f_latent,
            dropout=dropout,
            aux_weight=aux_weight,
        )

        self.log_var_elbo = nn.Parameter(torch.zeros(()))
        self.log_var_traj = nn.Parameter(torch.zeros(()))
        self.init_state_weight = 1.0
        self.use_kendall = use_kendall

    def forward(
        self,
        x_base: torch.Tensor,  # [batch_size, base_x_dim]
        m_base: torch.Tensor,  # [batch_size, n_base_features]
        x_state: torch.Tensor, # [batch_size, max_path, dim_state_features + 2]
        m_state: torch.Tensor, # [batch_size, max_path, n_state_features + 2]
        tau: float = 1.0,
        n_generated_sample: int = 1,
    ):
        batch_size, max_path, _ = x_state.shape
        assert x_base.shape[0] == batch_size

        # Split (t_n, v_n, x_n) and masks
        t, v, x, visit_mask, m_state_feat = self._split_state_input(x_state, m_state)

        base_out = self.base_hivae(
            x_base=x_base,
            m_base=m_base,
            tau=tau,
            n_generated_sample=n_generated_sample,
        )

        base_data_norm = base_out["base_data_norm"]
        q_params_base  = base_out["q_params"]
        p_params_base  = base_out["p_params"]
        log_p_base     = base_out["log_p"]
        samples_base   = base_out["samples"]

        z_0      = samples_base["z_0"]
        s_samples = samples_base["s"]
        h0       = self.phi_z_0(z_0) # Initialize RNN state from z_0

        # Initial state p(v_1 | s, z_0)
        logits_init_state = None
        if self.learn_init_state:
            init_input = torch.cat([s_samples, z_0], dim=-1)  # [B, s_dim+z_dim]
            logits_init_state = self.init_state(init_input)   # [B, v_dim]
            
        # State block
        state_out = self.state_vrnn(
            x_state=x,
            m_state_feat=m_state_feat,
            v=v,
            t=t,
            h0=h0,
            s=s_samples,
            n_generated_sample=n_generated_sample,
        )
        state_data_norm  = state_out["state_data_norm"]
        q_params_state   = state_out["q_params"]
        p_params_state   = state_out["p_params"]
        log_p_state      = state_out["log_p"]
        samples_state    = state_out["samples"]
        h_seq            = state_out["h_seq"]
        # NOTE: h_seq[:, n, :] = h_{n-1} (memory BEFORE observing visit n)
        # This means the decoder/MuStaRDT at position n sees the memory state
        # that does NOT yet incorporate information from visit n.

        p_params = {
            "x_base":  p_params_base["x_base"],
            "z_0":     p_params_base["z_0"],
            "x_state": p_params_state["x_state"],
            "z_n":     p_params_state["z_n"],
        }
        q_params = {
            "s":   q_params_base["s"],
            "z_0": q_params_base["z_0"],
            "z_n": q_params_state["z_n"],
        }
        log_p = {**log_p_base, **log_p_state}


        # MuStaRDT -------------------------------------------------------
        z_n = samples_state["z_n"]  # [B, max_path, z_dim]


        # state index per visit: 0..v_dim-1, -1 for padding
        state_idx = v.argmax(dim=-1)                          # [B, max_path]
        state_idx = state_idx.masked_fill(~visit_mask.bool(), -1)

        # MuStaRDT head
        mustardt_out = self.mustardt.dgmsm_forward(
            z_n=z_n,
            h_seq=h_seq,
            t=t,
            state_idx=state_idx,
            z_0=z_0,
            s=s_samples,
        )
        p_logits = mustardt_out["p_logits"]
        f_logits = mustardt_out["f_logits"]

        samples = {**samples_base, **samples_state}

        return {
            "samples": samples,
            "log_p": log_p,
            "p_params": p_params,
            "q_params": q_params,
            "f_logits": f_logits,
            "p_logits": p_logits,
            "h_seq": h_seq,
            "logits_init_state": logits_init_state,
        }

    @torch.no_grad()
    def deterministic_prediction(
        self,
        x_base: torch.Tensor,
        m_base: torch.Tensor,
        x_state: torch.Tensor,   # [batch_size, max_path, dim_state_features + 1]
        m_state: torch.Tensor,   # [batch_size, max_path, n_state_features + 2]
    ) -> Dict[str, torch.Tensor]:
        was_training = self.training
        try:
            self.eval()
            batch_size, max_path, _ = x_state.shape

            # Split state and covariates
            t, v, x, visit_mask, m_state_feat = self._split_state_input(x_state, m_state)
            
            # baseline deterministic
            base_det = self.base_hivae.forward_deterministic(
                x_base=x_base,
                m_base=m_base,
            )
            samples_base_det = base_det["samples"]
            s_det   = samples_base_det["s"]
            z0_det  = samples_base_det["z_0"]
            h0     = self.phi_z_0(z0_det)

            # state deterministic
            state_data_norm, samples_state_det, h_seq_det = self.state_vrnn.forward_deterministic(
                x_state=x,
                m_state_feat=m_state_feat,
                v=v,
                t=t,
                h0=h0,
                s=s_det,
            )
            z_n_det = samples_state_det["z_n"]  # [batch_size, max_path, z_dim]
            state_idx = v.argmax(dim=-1)
            state_idx = state_idx.masked_fill(~visit_mask.bool(), -1)

            mustardt_out = self.mustardt.dgmsm_forward(
                z_n=z_n_det,
                h_seq=h_seq_det,
                t=t,
                state_idx=state_idx,
                z_0=z0_det,
                s=s_det,
            )

            return mustardt_out
        finally:
            if was_training:
                self.train()


    @torch.no_grad()
    def predict_survival_eval(
        self,
        x_base: torch.Tensor,
        m_base: torch.Tensor,
        x_state: torch.Tensor,   # [batch_size, max_path, D_state_total] (state one-hot + other)
        m_state: torch.Tensor,   # [batch_size, max_path, n_state_features+1]
        **kwargs,
    ) -> torch.Tensor:
        batch_size, max_path, _ = x_state.shape
        assert x_base.shape[0] == batch_size

        deterministic_output = self.deterministic_prediction(
            x_base=x_base,
            m_base=m_base,
            x_state=x_state,
            m_state=m_state,
        )

        p_logits = deterministic_output["p_logits"]
        f_logits = deterministic_output["f_logits"]

        return self.mustardt.renewal_survival(p_logits=p_logits, f_logits=f_logits)


    def compute_loss(
        self,
        x_base: torch.Tensor,
        m_base: torch.Tensor,
        x_state: torch.Tensor,
        m_state: torch.Tensor,
        labels: torch.Tensor,
        tau: float = 1.0,
        n_generated_sample: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            total_loss:      Scalar. Backprop target.
            neg_ELBO_loss:   Scalar. -ELBO averaged over batch.
            loss_re:         [B] Per-sample reconstruction log-likelihood (higher = better).
            KL_z0:           [B] Per-sample KL for baseline latent.
            KL_z_n:          [B] Per-sample KL for state latents (summed over visits).
            KL_s:            [B] Per-sample KL for mixture component.
            surv_loss:       Scalar. Total trajectory branch loss:
                            surv_main + init_state_weight * init_state_loss
            surv_main_loss:  Scalar. Pure MuStaRDT survival loss.
            init_state_loss: Scalar. Cross-entropy for initial state prediction.

        Note:
            - total_loss is the only backprop target.
            - w_elbo and w_surv are now Kendall precisions exp(-log_var), not reciprocal weights.
            - They do NOT sum to 1.
        """
        # forward pass
        out = self.forward(
            x_base=x_base,
            m_base=m_base,
            x_state=x_state,
            m_state=m_state,
            tau=tau,
            n_generated_sample=n_generated_sample,
        )

        q_params = out["q_params"]
        p_params = out["p_params"]
        log_p = out["log_p"]
        p_logits = out["p_logits"]
        f_logits = out["f_logits"]
        h_seq = out["h_seq"]
        samples = out["samples"]
        logits_init_state = out["logits_init_state"]

        t, v, _, visit_mask, _ = self._split_state_input(x_state, m_state)
        visit_mask = visit_mask.bool()

        state_idx = v.argmax(dim=-1)
        state_idx = state_idx.masked_fill(~visit_mask, -1)

        # ELBO branch
        loss_re_base, KL_z0, KL_s = self.base_hivae.elbo(
            log_p_x_base=log_p["log_p_x_base"],
            p_z0=p_params["z_0"],
            q_s_logits=q_params["s"],
            q_z0=q_params["z_0"],
        )

        loss_re_state, KL_z_n = self.state_vrnn.elbo(
            log_p_x_state=log_p["log_p_x_state"],
            p_zn=p_params["z_n"],
            q_zn=q_params["z_n"],
            visit_mask=visit_mask,
        )

        loss_re = loss_re_base + loss_re_state
        ELBO_per_sample = loss_re - KL_z0 - KL_z_n - KL_s
        neg_elbo = -ELBO_per_sample.mean()

        # Trajectory branch
        surv_main_loss, surv_components = self.mustardt.compute_loss(
            p_logits=p_logits,
            f_logits=f_logits,
            labels=labels,
            return_components=True,
        )


        init_state_loss = torch.tensor(0.0, device=x_base.device)
        if self.learn_init_state and logits_init_state is not None:
            init_state = labels[:, 0, 0].long()   # [B]
            init_state_idx = init_state - 1       # [B], 0..v_dim-1

            init_state_loss = F.cross_entropy(
                logits_init_state,                # [B, v_dim]
                init_state_idx,                   # [B]
                reduction="mean",
            )

        init_state_weight = getattr(self, "init_state_weight", 1.0)

        traj_loss = surv_main_loss + init_state_weight * init_state_loss
        
        if self.use_kendall:
            precision_elbo = torch.exp(-self.log_var_elbo)
            precision_traj = torch.exp(-self.log_var_traj)
            total_loss = (
                precision_elbo * (self.beta_elbo * neg_elbo) + self.log_var_elbo
                + precision_traj * traj_loss + self.log_var_traj
            )
        else:
            precision_elbo = torch.ones((), device=x_base.device)
            precision_traj = torch.ones((), device=x_base.device)
            total_loss = self.beta_elbo * neg_elbo + traj_loss

        result = {
            "total_loss": total_loss,
            "neg_ELBO_loss": neg_elbo,
            "loss_re": loss_re,
            "KL_z0": KL_z0,
            "KL_z_n": KL_z_n,
            "KL_s": KL_s,
            "surv_loss": traj_loss,
            "surv_main_loss": surv_main_loss,
            "init_state_loss": init_state_loss,
            "traj_loss": traj_loss,
            "w_elbo": precision_elbo.detach().item(),
            "w_surv": precision_traj.detach().item(),
            "w_traj": precision_traj.detach().item(),
            "log_var_elbo": self.log_var_elbo.detach().item(),
            "log_var_traj": self.log_var_traj.detach().item(),
        }
        for k, v in surv_components.items():
            result[f"surv/{k}"] = v.item()

        return result


    def eval_surv(
        self,
        dataset,
        device: Optional[Union[torch.device, str]] = None,
        censor_surv: str = "km",
        times: Optional[np.ndarray] = None,
        clamp_durations: bool = True,
        compute: bool = True,
        state_names: Optional[List[str]] = None,
        eval_mask: Optional[torch.Tensor] = None,
    ) -> EvalMuStaRDT:
        if device is None:
            device = next(self.parameters()).device

        evaluator = EvalMuStaRDT(
            model=self,
            dataset=dataset,
            device=device,
            censor_surv=censor_surv,
            clamp_durations=clamp_durations,
            logger=getattr(self, "logger", None),
            state_names=state_names,
            eval_mask=eval_mask,
        )
        if compute:
            _ = evaluator.compute(times=times)
        return evaluator
    
    def fit(
        self,
        train_loader,
        val_loader=None,
        device: Optional[torch.device] = None,
        trial=None,
        **train_kwargs: Any,
    ):
        if device is None:
            device = next(self.parameters()).device
        return training_loop(
            model=self,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            log=self.logger,
            trial=trial,
            **train_kwargs,
        )

    # Sampling from the prior distribution -----------------------------------------

    @torch.inference_mode()
    def sample(
        self, 
        N: int,
        temperature: float = 1.0,
        deterministic_generation: bool = False,
        ) -> Dict[str, torch.Tensor]:
        """
        Generate synthetic data for N patients from the full prior.
        """
        was_training = self.training
        device = next(self.parameters()).device
        try:
            self.eval()
            # 1) Sample baseline: p(s) -> p(z_0 | s) -> p(x_0 | z_0, s)
            base_prior = self.base_hivae.sample_from_prior(N, device=device)
            s_all      = base_prior["s"]
            z_0_all    = base_prior["z_0"]
            x_base     = base_prior["x_base"]
            base_norm  = base_prior["x_base_norm"]
            m_base_rep = torch.ones(N, len(self.base_ftypes), device=device)

            # 2) Sample initial state: v_1 ~ p(v_1 | s, z_0)
            state_idx = self._sample_initial_state(s_all, z_0_all)

            # 3) Initialize memory
            h_curr = self.phi_z_0(z_0_all)
            t_curr = torch.zeros(N, dtype=torch.float32, device=device)

            # 4) First visit: z_1, x_1 ~ p(· | h_0, v_1, t_1=0, s)
            first_visit = self.state_vrnn.generate_first_visit(
                state_idx=state_idx, h0=h_curr, s=s_all,
            )

            loop_out = self._run_simulation_loop(
                state_idx=state_idx,
                t_curr=t_curr,
                h_curr=h_curr,
                z_curr=first_visit["z_curr"],
                x_curr_raw=first_visit["x_curr_raw"],
                s_all=s_all,
                z_0_all=z_0_all,
                horizon=self.max_path,
                store_x_state=True,
                store_covariates=False,
                temperature=temperature,
                deterministic_generation=deterministic_generation,
                store_labels=True,
            )
            return {
                "x_base":         x_base,
                "x_state":        loop_out["x_state_out"],
                "labels":         loop_out["labels"],
                "start_time_abs": loop_out["start_time_abs"],
                "stop_time_abs":  loop_out["stop_time_abs"],
                "stop_time_rel":  loop_out["stop_time_rel"],
                "event":          loop_out["event"],
            }
        finally:
            if was_training:
                self.train()



    @torch.inference_mode()
    def simulate(
        self,
        x_base: torch.Tensor,
        m_base: torch.Tensor,
        x_state: Optional[torch.Tensor] = None,
        m_state: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        n_sim: int = 100,
        temperature: float = 1.0,
        deterministic_history: bool = True,
        deterministic_generation: bool = False,
        sim_chunk_size: int = 100,
        max_future_steps: Optional[int] = None,
        return_covariates: bool = False,
        return_labels: bool = False,
    ) -> Dict[str, Any]:

        if (x_state is None) != (m_state is None):
            raise ValueError("x_state and m_state must both be provided or both be None.")
        if n_sim <= 0:
            raise ValueError("n_sim must be >= 1.")
        if sim_chunk_size <= 0:
            raise ValueError("sim_chunk_size must be >= 1.")
        if max_future_steps is not None and not (1 <= max_future_steps <= self.max_path):
            raise ValueError(f"max_future_steps={max_future_steps} must be in [1, {self.max_path}].")

        was_training = self.training

        try:
            self.eval()

            x_base, m_base, x_state, m_state, labels, was_single = \
                self._ensure_batch(x_base, m_base, x_state, m_state, labels)

            horizon = self.max_path if max_future_steps is None else max_future_steps
            B = x_base.shape[0]

            # Cache only one history encode if deterministic.
            hist_cache = None
            if deterministic_history:
                hist_cache = self._encode_history(
                    x_base=x_base,
                    m_base=m_base,
                    x_state=x_state,
                    m_state=m_state,
                    deterministic_history=True,
                    n_sim=1,
                    tau=temperature,
                )

            start_chunks: list[torch.Tensor] = []
            stop_abs_chunks: list[torch.Tensor] = []
            stop_rel_chunks: list[torch.Tensor] = []
            event_chunks: list[torch.Tensor] = []

            if return_labels:
                labels_chunks: list[torch.Tensor] = []

            if return_covariates:
                covariates_out: Optional[list] = [[] for _ in range(B)]


            def _chunk_to_memory(hist: Dict[str, torch.Tensor], chunk_n: int) -> Dict[str, Any]:
                BN = B * chunk_n

                def flat_tensor(name: str, clone: bool = False) -> torch.Tensor:
                    t = hist[name]  # [B, S, ...], where S is 1 or chunk_n
                    if t.shape[1] == 1 and chunk_n > 1:
                        t = t.expand(B, chunk_n, *t.shape[2:])
                    t = t.reshape(BN, *t.shape[2:])
                    return t.clone() if clone else t

                def flat_scalar(name: str, clone: bool = False) -> torch.Tensor:
                    t = hist[name]  # [B, S]
                    if t.shape[1] == 1 and chunk_n > 1:
                        t = t.expand(B, chunk_n)
                    t = t.reshape(BN)
                    return t.clone() if clone else t

                return {
                    "state_idx":         flat_scalar("v_idx", clone=True),
                    "t_curr":            flat_scalar("t", clone=True),
                    "h_curr":            flat_tensor("h", clone=True),
                    "z_curr":            flat_tensor("z", clone=True),
                    "x_curr_raw":        flat_tensor("x_raw", clone=True) if return_covariates else None,
                    "s_all":             flat_tensor("s", clone=False),
                    "z_0_all":           flat_tensor("z0", clone=False),
                }

            for chunk_start in range(0, n_sim, sim_chunk_size):
                chunk_n = min(sim_chunk_size, n_sim - chunk_start)

                if deterministic_history:
                    hist_chunk = hist_cache
                else:
                    # Sample only this chunk of history draws.
                    hist_chunk = self._encode_history(
                        x_base=x_base,
                        m_base=m_base,
                        x_state=x_state,
                        m_state=m_state,
                        deterministic_history=False,
                        n_sim=chunk_n,
                        tau=temperature,
                    )

                memory = _chunk_to_memory(hist_chunk, chunk_n)

                loop_out = self._run_simulation_loop(
                    **memory,
                    horizon=horizon,
                    store_x_state=False,
                    store_covariates=return_covariates,
                    store_labels=return_labels,
                    temperature=temperature,
                    deterministic_generation=deterministic_generation,
                )

                start_chunks.append(loop_out["start_time_abs"].reshape(B, chunk_n).cpu())
                stop_abs_chunks.append(loop_out["stop_time_abs"].reshape(B, chunk_n).cpu())
                stop_rel_chunks.append(loop_out["stop_time_rel"].reshape(B, chunk_n).cpu())
                event_chunks.append(loop_out["event"].reshape(B, chunk_n).cpu())

                if return_labels:
                    labels_chunks.append(
                        loop_out["labels"].reshape(B, chunk_n, horizon, 5).cpu()
                    )

                if return_covariates:
                    cov_flat = loop_out["covariates"]
                    for b in range(B):
                        for c in range(chunk_n):
                            covariates_out[b].append(cov_flat[b * chunk_n + c])

                del memory, loop_out, hist_chunk

            result: Dict[str, Any] = {
                "start_time_abs": torch.cat(start_chunks, dim=1),
                "stop_time_abs":  torch.cat(stop_abs_chunks, dim=1),
                "stop_time_rel":  torch.cat(stop_rel_chunks, dim=1),
                "event":          torch.cat(event_chunks, dim=1),
            }

            if return_labels:
                sim_lbl = torch.cat(labels_chunks, dim=1)

                if labels is not None:
                    labels_cpu = labels.cpu()
                    B0, S, Tsim, D = sim_lbl.shape

                    if m_state is not None:
                        m_state_cpu = m_state.cpu()
                        out_lbl = torch.zeros(B0, S, self.max_path, D, dtype=sim_lbl.dtype)

                        for b in range(B0):
                            n_obs = int((m_state_cpu[b, :, 1] > 0).sum().item())
                            n_obs = min(n_obs, labels_cpu.shape[1], self.max_path)
                            n_prefix = max(n_obs - 1, 0)

                            if n_prefix > 0:
                                hist_b = labels_cpu[b, :n_prefix]
                                out_lbl[b, :, :n_prefix, :] = hist_b.unsqueeze(0).expand(S, -1, -1)

                            n_future = min(Tsim, self.max_path - n_prefix)
                            if n_future > 0:
                                out_lbl[b, :, n_prefix:n_prefix + n_future, :] = sim_lbl[b, :, :n_future, :]

                        sim_lbl = out_lbl
                    else:
                        hist_lbl = labels_cpu.unsqueeze(1).expand(-1, n_sim, -1, -1)
                        sim_lbl = torch.cat([hist_lbl, sim_lbl], dim=2)[:, :, :self.max_path, :]

                result["labels"] = sim_lbl

            if return_covariates:
                if x_state is not None:
                    hist_cov_np = x_state.cpu().numpy()
                    m_state_np = m_state.cpu().numpy()
                    for b in range(B):
                        n_obs_b = int((m_state_np[b, :, 1] > 0).sum())
                        n_prefix_b = max(n_obs_b - 1, 0)
                        hist_prefix = hist_cov_np[b, :n_prefix_b, self.v_dim + 1:]

                        covariates_out[b] = [
                            np.concatenate([hist_prefix, np.stack(path, axis=0)], axis=0)[:self.max_path]
                            for path in covariates_out[b]
                        ]
                else:
                    for b in range(B):
                        covariates_out[b] = [
                            np.stack(path, axis=0)[:self.max_path]
                            for path in covariates_out[b]
                        ]

                result["covariates"] = covariates_out

            if was_single:
                result = self._squeeze_output(result)

            return result

        finally:
            if was_training:
                self.train()


    # Internal helper functions ---------------------------------------------------------------

    @torch.no_grad()
    def _encode_history(
        self,
        x_base: torch.Tensor,
        m_base: torch.Tensor,
        x_state: Optional[torch.Tensor] = None,
        m_state: Optional[torch.Tensor] = None,
        deterministic_history: bool = True,
        n_sim: int = 1,
        tau: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode history for B patients.

        Return contract
        ---------------
        Every returned tensor has an explicit simulation axis:

        - [B, S, D] for vector-valued outputs
        - [B, S]    for scalar outputs

        where S is:
        - 1     if the quantity is shared across future simulation draws
        - n_sim if the quantity already contains per-draw posterior samples

        This matches simulate() / _chunk_to_memory(), which expands S=1 tensors
        to the requested chunk size on demand.
        """
        if (x_state is None) != (m_state is None):
            raise ValueError("x_state and m_state must both be provided or both be None.")
        if n_sim < 1:
            raise ValueError("n_sim must be >= 1.")

        batch_size = x_base.size(0)
        device = x_base.device

        # Only materialize B * n_sim encodes when we truly need independent
        # posterior samples of the history.
        stochastic_multisim = (not deterministic_history) and (n_sim > 1)

        def pack_sampled(t: torch.Tensor) -> torch.Tensor:
            """
            Quantities that are truly sampled per simulation draw:
            -> [B, n_sim, ...] if stochastic_multisim else [B, 1, ...]
            """
            if stochastic_multisim:
                return t.reshape(batch_size, n_sim, *t.shape[1:])
            return t.unsqueeze(1)

        def pack_shared(t: torch.Tensor) -> torch.Tensor:
            """
            Observation-level quantities shared across simulation draws:
            -> always [B, 1, ...]
            If they were repeated to [B*n_sim, ...], collapse them back first.
            """
            if stochastic_multisim and t.size(0) == batch_size * n_sim:
                t = t.reshape(batch_size, n_sim, *t.shape[1:])[:, 0]
            return t.unsqueeze(1)

        # Prepare encoder inputs
        if stochastic_multisim:
            x_base_in = x_base.repeat_interleave(n_sim, dim=0)
            m_base_in = m_base.repeat_interleave(n_sim, dim=0)
            x_state_in = x_state.repeat_interleave(n_sim, dim=0) if x_state is not None else None
            m_state_in = m_state.repeat_interleave(n_sim, dim=0) if m_state is not None else None
        else:
            x_base_in, m_base_in = x_base, m_base
            x_state_in, m_state_in = x_state, m_state

        # Baseline encoding
        if deterministic_history:
            base_out = self.base_hivae.forward_deterministic(
                x_base=x_base_in,
                m_base=m_base_in,
            )
        else:
            base_out = self.base_hivae(
                x_base=x_base_in,
                m_base=m_base_in,
                tau=tau,
            )

        s = base_out["samples"]["s"]        # [B*?, s_dim]
        z0 = base_out["samples"]["z_0"]     # [B*?, z_dim]
        h0 = self.phi_z_0(z0)               # [B*?, h_dim]

        # Baseline-only mode: no observed state history
        if x_state is None:
            state_idx = self._sample_initial_state(s, z0)  # [B*?]
            first_visit = self.state_vrnn.generate_first_visit(
                state_idx=state_idx,
                h0=h0,
                s=s,
            )

            return {
                "v_idx":        pack_sampled(state_idx),                  # [B,S]
                "t":            torch.zeros(batch_size, 1, device=device, dtype=h0.dtype),
                "h":            pack_sampled(h0),                         # [B,S,h_dim]
                "z":            pack_sampled(first_visit["z_curr"]),      # [B,S,z_dim]
                "x_norm":       pack_sampled(first_visit["x_curr_norm"]), # [B,S,state_x_dim]
                "x_raw":        pack_sampled(first_visit["x_curr_raw"]),  # [B,S,state_x_dim]
                "m_state_feat": pack_sampled(first_visit["m_state_feat_curr"]),
                "s":            pack_sampled(s),                          # [B,S,s_dim]
                "z0":           pack_sampled(z0),                         # [B,S,z_dim]
            }

        # History-based mode: encode observed longitudinal history
        t_inp, v, x, visit_mask, m_state_feat = self._split_state_input(x_state_in, m_state_in)
        visit_mask = visit_mask.bool()

        if not (visit_mask.sum(dim=1) > 0).all().item():
            raise ValueError("_encode_history: found patients with no real visits.")

        if deterministic_history:
            state_data_norm, samples_state, h_seq = self.state_vrnn.forward_deterministic(
                x_state=x,
                m_state_feat=m_state_feat,
                v=v,
                t=t_inp,
                h0=h0,
                s=s,
            )
            z_n = samples_state["z_n"]
        else:
            state_out = self.state_vrnn(
                x_state=x,
                m_state_feat=m_state_feat,
                v=v,
                t=t_inp,
                h0=h0,
                s=s,
            )
            state_data_norm = state_out["state_data_norm"]
            z_n = state_out["samples"]["z_n"]
            h_seq = state_out["h_seq"]

        last_idx = (visit_mask.sum(dim=1).long() - 1).clamp(min=0)   # [B*?]
        batch_idx = torch.arange(t_inp.size(0), device=device)

        # Sample-level quantities: may differ across posterior draws
        h_last = h_seq[batch_idx, last_idx]                          # [B*?, h_dim]
        z_last = z_n[batch_idx, last_idx]                            # [B*?, z_dim]
        s_last = s                                                   # [B*?, s_dim]
        z0_last = z0                                                 # [B*?, z_dim]

        # Observation-level quantities: identical across repeated draws
        x_norm_last = state_data_norm[batch_idx, last_idx]           # [B*?, state_x_dim]
        x_raw_last = x_state_in[batch_idx, last_idx, self.v_dim + 1:]# [B*?, state_x_dim]
        m_feat_last = m_state_feat[batch_idx, last_idx]              # [B*?, n_state_features]
        v_last_idx = v[batch_idx, last_idx].argmax(dim=-1)           # [B*?]
        t_last = t_inp[batch_idx, last_idx, 0]                       # [B*?]

        return {
            "v_idx":        pack_shared(v_last_idx),     # [B,1]
            "t":            pack_shared(t_last),         # [B,1]
            "h":            pack_sampled(h_last),        # [B,S,h_dim]
            "z":            pack_sampled(z_last),        # [B,S,z_dim]
            "x_norm":       pack_shared(x_norm_last),    # [B,1,state_x_dim]
            "x_raw":        pack_shared(x_raw_last),     # [B,1,state_x_dim]
            "m_state_feat": pack_shared(m_feat_last),    # [B,1,n_state_features]
            "s":            pack_sampled(s_last),        # [B,S,s_dim]
            "z0":           pack_sampled(z0_last),       # [B,S,z_dim]
        }


    def _run_simulation_loop(
        self,
        state_idx: torch.Tensor,         # [N]
        t_curr: torch.Tensor,            # [N]
        h_curr: torch.Tensor,            # [N, h_dim]
        z_curr: torch.Tensor,            # [N, z_dim]
        x_curr_raw: Optional[torch.Tensor],  # [N, state_x_dim] or None
        z_0_all: torch.Tensor,
        s_all: torch.Tensor,             # [N, s_dim]
        horizon: int,
        store_x_state: bool = False,
        store_covariates: bool = False,
        store_labels: bool = True,
        temperature: float = 1.0,
        deterministic_generation: bool = False,
    ) -> Dict[str, Any]:
        """
        Simulate forward trajectories from an already initialized current visit for up to
        `horizon` steps.

        Loop invariant at the start of each step:
            h_curr     = h_{n-1}  (memory BEFORE current visit n)
            z_curr     = z_n      (latent OF current visit n)
            x_curr_raw = x_n      (raw covariates OF current visit n) if outputs need it, else None
            state_idx  = v_n      (state at current visit n)
            t_curr     = t_n      (time of current visit n)
        """
        device = state_idx.device
        N = state_idx.size(0)

        need_x_curr_raw = store_x_state or store_covariates
        if need_x_curr_raw and x_curr_raw is None:
            raise ValueError(
                "x_curr_raw must be provided when store_x_state or store_covariates is True."
            )
        if not need_x_curr_raw:
            x_curr_raw = None

        labels = (
            torch.zeros(N, horizon, 5, dtype=torch.long, device=device)
            if store_labels else None
        )

        start_time_abs = t_curr.clone()
        stop_time_abs = t_curr.clone()
        alive = torch.ones(N, dtype=torch.bool, device=device)
        event = torch.zeros(N, dtype=torch.bool, device=device)

        x_state_out = None
        if store_x_state:
            x_state_out = torch.zeros(
                N, horizon, 1 + self.v_dim + self.state_x_dim,
                dtype=torch.float32, device=device,
            )

        covariates = None
        if store_covariates:
            assert x_curr_raw is not None
            covariates = [[x_curr_raw[i].detach().cpu().numpy()] for i in range(N)]

        for step in range(horizon):
            idx = alive.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                break

            # Snapshot current-visit info before any mutations
            v_curr = F.one_hot(state_idx[idx], num_classes=self.v_dim).float()
            t_start = t_curr[idx].clone()

            # Store visit covariates if requested
            if store_x_state:
                assert x_curr_raw is not None
                x_state_out[idx, step, :] = torch.cat(
                    [t_curr[idx].unsqueeze(-1), v_curr, x_curr_raw[idx]],
                    dim=-1,
                )

            # (a) Predict transition from current visit n
            event_idx, time_idx, event_observed = self.mustardt.transition_step(
                state_idx=state_idx[idx],
                t_curr=t_curr[idx],
                h_curr=h_curr[idx],  # h_{n-1}
                z_curr=z_curr[idx],
                z_0=z_0_all[idx],
                s=s_all[idx],
                temperature=temperature,
            )

            # 1) No-event / tail outcome -> censor and stop path
            tail_mask = ~event_observed
            if tail_mask.any():
                idx_tail = idx[tail_mask]
                dt_tail = time_idx[tail_mask].to(t_start.dtype)

                stop_time_abs[idx_tail] = t_start[tail_mask] + dt_tail
                alive[idx_tail] = False

                if store_labels:
                    labels[idx_tail, step, 0] = state_idx[idx_tail] + 1
                    labels[idx_tail, step, 1] = time_idx[tail_mask].long()
                    labels[idx_tail, step, 2] = 0
                    labels[idx_tail, step, 3] = t_start[tail_mask].long()
                    labels[idx_tail, step, 4] = (t_start[tail_mask] + dt_tail).long()

            # 2) Continue only with observed events
            obs_mask = event_observed
            if not obs_mask.any():
                if not alive.any():
                    break
                continue

            idx_obs = idx[obs_mask]
            v_curr_obs = v_curr[obs_mask]
            event_idx_obs = event_idx[obs_mask]   # 0..E-1
            time_idx_obs = time_idx[obs_mask]
            dt_obs = time_idx_obs.to(t_start.dtype)
            t_start_obs = t_start[obs_mask]

            assert ((event_idx_obs >= 0) & (event_idx_obs < self.n_events)).all()

            next_state_idx = self.graph.event_to[event_idx_obs]

            if store_labels:
                labels[idx_obs, step, 0] = state_idx[idx_obs] + 1
                labels[idx_obs, step, 1] = time_idx_obs.long()
                labels[idx_obs, step, 2] = event_idx_obs + 1
                labels[idx_obs, step, 3] = t_start_obs.long()
                labels[idx_obs, step, 4] = (t_start_obs + dt_obs).long()

            # (b) Update memory only for observed transitions
            h_curr[idx_obs] = self.state_vrnn.update_memory(
                z_n=z_curr[idx_obs],
                v_n=v_curr_obs,
                t_n=t_start_obs.unsqueeze(-1),
                h_prev=h_curr[idx_obs],
            )

            # (c) Advance time and state only for observed transitions
            t_curr[idx_obs] = t_start_obs + dt_obs
            state_idx[idx_obs] = next_state_idx

            # (d) Terminal check
            term_now = torch.isin(next_state_idx, self.graph.terminal_states)
            if term_now.any():
                idx_term = idx_obs[term_now]
                stop_time_abs[idx_term] = t_curr[idx_term]
                alive[idx_term] = False
                event[idx_term] = True

            # (e) Generate next visit's latent and covariates for non-terminal paths
            nonterm_mask = ~term_now
            if nonterm_mask.any():
                idx_nt = idx_obs[nonterm_mask]
                next_state_nt = next_state_idx[nonterm_mask]

                if need_x_curr_raw:
                    gen = self.state_vrnn.sample_state_cov(
                        h_prev=h_curr[idx_nt],
                        v_next_idx=next_state_nt,
                        t_next=t_curr[idx_nt].unsqueeze(-1),
                        s=s_all[idx_nt],
                        deterministic_generation=deterministic_generation,
                    )

                    z_curr[idx_nt] = gen["z_next"]

                    assert x_curr_raw is not None
                    x_curr_raw[idx_nt] = gen["x_next_raw"]

                    if store_covariates:
                        x_next_raw_cpu = gen["x_next_raw"].detach().cpu().numpy()
                        for j, g_idx in enumerate(idx_nt.tolist()):
                            covariates[g_idx].append(x_next_raw_cpu[j])

                else:
                    gen = self.state_vrnn.sample_z(
                        h_prev=h_curr[idx_nt],
                        v_next_idx=next_state_nt,
                        t_next=t_curr[idx_nt].unsqueeze(-1),
                        s=s_all[idx_nt],
                        deterministic_generation=deterministic_generation,
                    )
                    z_curr[idx_nt] = gen["z_next"]

            if not alive.any():
                break

        if alive.any():
            stop_time_abs[alive] = t_curr[alive]

        stop_time_rel = stop_time_abs - start_time_abs
        result = {
            "start_time_abs": start_time_abs,
            "stop_time_abs": stop_time_abs,
            "stop_time_rel": stop_time_rel,
            "event": event,
        }
        if store_labels and labels is not None:
            result["labels"] = labels
        if store_x_state:
            result["x_state_out"] = x_state_out
        if store_covariates:
            result["covariates"] = covariates

        return result


    def _sample_initial_state(self, s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """Sample v_1 ~ p(v_1 | s, z_0). Returns [N] long tensor."""
        if self.learn_init_state:
            init_input = torch.cat([s, z0], dim=-1)
            logits = self.init_state(init_input)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)
        else:
            return torch.zeros(s.size(0), dtype=torch.long, device=s.device)

    def _split_state_input(
        self,
        x_state: torch.Tensor,
        m_state: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split composite state tensors into components.

        x_state layout: [Tstart(1), state_one_hot(v_dim), covariates(state_x_dim)]
        m_state layout: [Tstart_mask, state_mask(=visit_mask), feat_1_mask, ..., feat_K_mask]
        """
        t = x_state[..., 0:1]
        v = x_state[..., 1:self.v_dim+1]
        x = x_state[..., self.v_dim+1:]
        visit_mask = m_state[..., 1]
        m_state_feat = m_state[..., 2:]
        return t, v, x, visit_mask, m_state_feat


    def _ensure_batch(
        self,
        x_base:  torch.Tensor,
        m_base:  torch.Tensor,
        x_state: Optional[torch.Tensor] = None,
        m_state: Optional[torch.Tensor] = None,
        labels:  Optional[torch.Tensor] = None,
        ):
        """Unsqueeze single-patient tensors (dim==1/2) to batch form."""
        was_single = x_base.dim() == 1
        if was_single:
            x_base = x_base.unsqueeze(0)
            m_base = m_base.unsqueeze(0)
            if x_state is not None:
                x_state = x_state.unsqueeze(0)
            if m_state is not None:
                m_state = m_state.unsqueeze(0)
            if labels is not None:
                labels = labels.unsqueeze(0)
        return x_base, m_base, x_state, m_state, labels, was_single

    def _squeeze_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove the B=1 batch dimension from simulate() output."""
        out = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
                out[k] = v.squeeze(0)
            elif k == "covariates" and v is not None and len(v) == 1:
                out[k] = v[0]   # List[n_sim][...] instead of List[1][List[n_sim][...]]
            else:
                out[k] = v
        return out


    @torch.inference_mode()
    def simulate_from_visit(
        self,
        x_base:  torch.Tensor,   # [B, D_base]
        m_base:  torch.Tensor,   # [B, n_base_feats]
        x_state: torch.Tensor,   # [B, max_path, D_state]
        m_state: torch.Tensor,   # [B, max_path, 2 + n_user_feats]
        labels:  torch.Tensor,   # [B, max_path, 5]
        visit:   int,
        **kwargs,
        ) -> Dict[str, Any]:
        x_base, m_base, x_state, m_state, labels, was_single = \
            self._ensure_batch(x_base, m_base, x_state, m_state, labels)
        B = x_base.shape[0]

        if not (1 <= visit <= self.max_path):
            raise ValueError(f"visit={visit} out of range [1, {self.max_path}].")

        n_observed = (labels[:, :, 0] != 0).sum(dim=1)   # [B]
        valid_mask = n_observed >= visit                   # [B] bool
        n_valid    = int(valid_mask.sum().item())
        n_excluded = B - n_valid

        if n_valid == 0:
            self.logger.warning(
                f"No patients have at least {visit} observed visit(s); "
                "returning empty result."
            )
            result = {"valid_mask": valid_mask}
            return self._squeeze_output(result) if was_single else result
        if n_excluded > 0:
            self.logger.info(
                f"{n_excluded}/{B} patient(s) excluded: "
                f"fewer than {visit} observed visit(s)."
            )

        x_state_lm = x_state[valid_mask].clone()
        m_state_lm = m_state[valid_mask].clone()
        x_state_lm[:, visit:] = 0.0
        m_state_lm[:, visit:] = 0.0

        labels_lm = labels[valid_mask].clone()
        labels_lm[:, visit:] = 0

        out = self.simulate(
            x_base  = x_base[valid_mask],
            m_base  = m_base[valid_mask],
            x_state = x_state_lm,
            m_state = m_state_lm,
            labels  = labels_lm,
            **kwargs,
        )

        out["valid_mask"] = valid_mask
        if was_single:
            out = self._squeeze_output(out)
        return out

    @torch.inference_mode()
    def simulate_from_state(
        self,
        x_base:  torch.Tensor,   # [B, D_base]
        m_base:  torch.Tensor,   # [B, n_base_feats]
        x_state: torch.Tensor,   # [B, max_path, D_state]
        m_state: torch.Tensor,   # [B, max_path, 2 + n_user_feats]
        labels:  torch.Tensor,   # [B, max_path, 5]
        state:   int,
        **kwargs,
        ) -> Dict[str, Any]:
        """
        Batch version of simulate_from_state.
        """

        x_base, m_base, x_state, m_state, labels, was_single = \
            self._ensure_batch(x_base, m_base, x_state, m_state, labels)
        B = x_base.shape[0]

        observed = labels[:, :, 0] != 0
        in_state = (labels[:, :, 0] == state) & observed

        valid_mask = in_state.any(dim=1)
        n_valid    = int(valid_mask.sum().item())
        n_excluded = B - n_valid

        if n_valid == 0:
            self.logger.warning(
                f"No patients have a visit in state={state}; "
                "returning empty result."
            )
            result = {"valid_mask": valid_mask}
            return self._squeeze_output(result) if was_single else result
        
        if n_excluded > 0:
            self.logger.info(
                f"{n_excluded}/{B} patient(s) excluded: never entered state={state}."
            )

        # landmark_visits: [n_valid], 0-indexed
        landmark_visits = in_state[valid_mask].long().argmax(dim=1)

        x_state_lm = x_state[valid_mask].clone()
        m_state_lm = m_state[valid_mask].clone()

        positions = torch.arange(
            self.max_path, device=x_state.device
        ).unsqueeze(0)                                          # [1, max_path]
        keep = positions <= landmark_visits.unsqueeze(1)        # [n_valid, max_path]

        x_state_lm[~keep] = 0.0
        m_state_lm[~keep] = 0.0

        # Build per-patient history labels: zero out visits beyond each patient's landmark
        labels_lm = labels[valid_mask].clone()   # [n_valid, max_path, 5]
        labels_lm[~keep] = 0                     # zero future slots

        out = self.simulate(
            x_base  = x_base[valid_mask],
            m_base  = m_base[valid_mask],
            x_state = x_state_lm,
            m_state = m_state_lm,
            labels  = labels_lm,
            **kwargs,
        )

        out["valid_mask"]      = valid_mask
        out["landmark_visits"] = landmark_visits + 1            # 1-indexed
        if was_single:
            out = self._squeeze_output(out)
        return out

