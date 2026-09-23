import os
import time
import copy
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import optuna

logger = logging.getLogger(__name__)


def _to_device(x, device_):
    return x.to(device_, non_blocking=True) if x is not None else None


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return float(x.detach().mean().cpu())
    return float(x)


def _accumulate_metric(sums: dict, counts: dict, name: str, value, weight: int):
    if value is None:
        return

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            value = float(value.detach().cpu())
            sums[name] += value * weight
            counts[name] += weight
        else:
            value = value.detach()
            sums[name] += float(value.sum().cpu())
            counts[name] += value.numel()
    else:
        value = float(value)
        sums[name] += value * weight
        counts[name] += weight


def _avg_metric(sums: dict, counts: dict, name: str):
    return sums[name] / counts[name] if counts[name] > 0 else np.nan


def training_loop(
    model: nn.Module,
    train_loader,
    val_loader=None,
    device=None,
    trial=None,
    scheduler_name: str = "ReduceLROnPlateau",
    learning_rate: float = 0.003,
    epochs: int = 1000,
    lr_decay_rate: float = 1e-5,
    patience: int = 10,
    min_delta: float = 0.001,
    weight_decay: float = 1e-5,
    eval_every: int = 1,
    tau_start: float = 1.0,
    tau_decay: float = 0.01,
    n_generated_sample: int = 1,
    max_grad_norm: Optional[float ] = 1.0,
    warmup_epochs: int = 5,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype ] = None,
    restore_best: bool = True,
    save_best_path: Optional[str ] = None,
    plot_curves: bool = True,
    plot_path: Optional[str ] = None,
    show_plot: bool = False,
    log: Optional[logging.Logger ] = None,
    verbose: bool = False,
    step_offset: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    log = log or logger

    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    model.to(device)

    # Separate Kendall uncertainty parameters so they get no weight decay
    uncertainty_names = {"log_var_elbo", "log_var_traj"}
    main_params = []
    uncertainty_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in uncertainty_names:
            uncertainty_params.append(p)
        else:
            main_params.append(p)

    param_groups = []
    if main_params:
        param_groups.append(
            {"params": main_params, "lr": learning_rate, "weight_decay": weight_decay}
        )
    if uncertainty_params:
        param_groups.append(
            {"params": uncertainty_params, "lr": learning_rate, "weight_decay": 0.0}
        )

    optimizer = AdamW(param_groups)

    # Store base LR per param group for manual warmup
    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    # Scheduler setup
    assert scheduler_name in ["ReduceLROnPlateau", "ExponentialLR", "CosineAnnealing"]

    if scheduler_name == "ExponentialLR":
        base_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=max(1.0 - lr_decay_rate, 1e-6),
        )
    elif scheduler_name == "CosineAnnealing":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=1e-5,
        )
    else:
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        )

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        if scheduler_name == "ReduceLROnPlateau":
            lr_scheduler = base_scheduler
            use_manual_warmup = True
        else:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[warmup_epochs],
            )
            use_manual_warmup = False
    else:
        lr_scheduler = base_scheduler
        use_manual_warmup = False

    amp_enabled = use_amp and isinstance(device, torch.device) and device.type == "cuda"
    if amp_enabled and amp_dtype is None:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    use_val = val_loader is not None
    best_monitor = float("inf")
    monitor_name = "surv_main_loss" # "surv_loss"    
    
    best_state = None
    best_epoch = 0
    best_path = None
    epochs_since_best = 0

    train_w_elbo_hist, train_w_surv_hist = [], []
    train_total_hist, val_total_hist, lr_hist = [], [], []
    train_neg_elbo_hist, val_neg_elbo_hist = [], []
    train_surv_hist, val_surv_hist = [], []
    train_kl_z_hist, val_kl_z_hist = [], []
    train_kl_s_hist, val_kl_s_hist = [], []

    train_surv_nll_hist = []
    train_surv_aux_raw_hist = []
    train_surv_aux_weighted_hist = []

    start_time = time.time()

    if verbose:
        log.info(
            "Starting training: epochs=%d, lr=%.3g, weight_decay=%.1e, scheduler=%s, "
            "patience=%d, min_delta=%.3g, eval_every=%d, amp=%s, "
            "max_grad_norm=%s, warmup_epochs=%d",
            epochs,
            learning_rate,
            weight_decay,
            scheduler_name,
            patience,
            min_delta,
            eval_every,
            amp_enabled,
            max_grad_norm,
            warmup_epochs,
        )

    for epoch in range(1, epochs + 1):
        if use_manual_warmup and epoch <= warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (epoch - 1) / max(warmup_epochs - 1, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * warmup_factor

        current_lr = optimizer.param_groups[0]["lr"]
        lr_hist.append(current_lr)

        tau_epoch = max(tau_start - tau_decay * (epoch - 1), 1e-3)
        model.train()

        train_sums = {
            "total": 0.0,
            "neg_elbo": 0.0,
            "surv": 0.0,
            "kl_z": 0.0,
            "kl_s": 0.0,
            "w_elbo": 0.0,
            "w_surv": 0.0,
            # MuStaRDT breakdown
            "surv_nll": 0.0,
            "surv_aux_raw": 0.0,
            "surv_aux_weighted": 0.0,
        }
        train_counts = {k: 0 for k in train_sums}

        for batch in train_loader:
            if len(batch) != 3:
                raise ValueError(
                    "Expected batch to be ((x_base, m_base), (x_state, m_state), labels). "
                    f"Got length {len(batch)}."
                )

            (x_base, m_base), (x_state, m_state), labels = batch

            x_base = _to_device(x_base, device)
            m_base = _to_device(m_base, device)
            x_state = _to_device(x_state, device)
            m_state = _to_device(m_state, device)
            labels = _to_device(labels, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                loss_out = model.compute_loss(
                    x_base=x_base,
                    m_base=m_base,
                    x_state=x_state,
                    m_state=m_state,
                    labels=labels,
                    tau=tau_epoch,
                    n_generated_sample=n_generated_sample,
                )

                total_loss = loss_out["total_loss"]
                neg_elbo = loss_out.get("neg_ELBO_loss")
                kl_z0 = loss_out.get("KL_z0")
                kl_z_n = loss_out.get("KL_z_n")
                kl_s = loss_out.get("KL_s")
                surv_loss = loss_out.get("surv_loss")
                w_elbo = loss_out.get("w_elbo")
                w_surv = loss_out.get("w_surv")

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if max_grad_norm is not None:
                    clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            batch_size = x_base.shape[0]

            _accumulate_metric(train_sums, train_counts, "total", total_loss, batch_size)
            _accumulate_metric(train_sums, train_counts, "neg_elbo", neg_elbo, batch_size)
            _accumulate_metric(train_sums, train_counts, "surv", surv_loss, batch_size)
            _accumulate_metric(train_sums, train_counts, "kl_s", kl_s, batch_size)
            _accumulate_metric(train_sums, train_counts, "w_elbo", w_elbo, batch_size)
            _accumulate_metric(train_sums, train_counts, "w_surv", w_surv, batch_size)
            _accumulate_metric(train_sums, train_counts, "surv_nll", loss_out.get("surv/nll"), batch_size)
            _accumulate_metric(train_sums, train_counts, "surv_aux_raw",loss_out.get("surv/aux_raw"), batch_size)
            _accumulate_metric(train_sums, train_counts, "surv_aux_weighted", loss_out.get("surv/aux_weighted"), batch_size)

            if kl_z0 is not None and kl_z_n is not None:
                _accumulate_metric(train_sums, train_counts, "kl_z", kl_z0 + kl_z_n, batch_size)

        avg_train_total = _avg_metric(train_sums, train_counts, "total")
        avg_train_neg_elb = _avg_metric(train_sums, train_counts, "neg_elbo")
        avg_train_surv = _avg_metric(train_sums, train_counts, "surv")
        avg_train_kl_z = _avg_metric(train_sums, train_counts, "kl_z")
        avg_train_kl_s = _avg_metric(train_sums, train_counts, "kl_s")
        avg_train_w_elbo = _avg_metric(train_sums, train_counts, "w_elbo")
        avg_train_w_surv = _avg_metric(train_sums, train_counts, "w_surv")
        avg_surv_nll          = _avg_metric(train_sums, train_counts, "surv_nll")
        avg_surv_aux_raw      = _avg_metric(train_sums, train_counts, "surv_aux_raw")
        avg_surv_aux_weighted = _avg_metric(train_sums, train_counts, "surv_aux_weighted")

        train_total_hist.append(avg_train_total)
        train_neg_elbo_hist.append(avg_train_neg_elb)
        train_surv_hist.append(avg_train_surv)
        train_kl_z_hist.append(avg_train_kl_z)
        train_kl_s_hist.append(avg_train_kl_s)
        train_w_elbo_hist.append(avg_train_w_elbo)
        train_w_surv_hist.append(avg_train_w_surv)
        train_surv_nll_hist.append(avg_surv_nll)
        train_surv_aux_raw_hist.append(avg_surv_aux_raw)
        train_surv_aux_weighted_hist.append(avg_surv_aux_weighted)

        do_eval = use_val and (epoch % eval_every == 0 or epoch == 1)

        avg_val_total = np.nan
        avg_val_neg_elb = np.nan
        avg_val_surv = np.nan
        avg_val_kl_z = np.nan
        avg_val_kl_s = np.nan

        if do_eval:
            model.eval()

            val_sums = {
                "total": 0.0,
                "neg_elbo": 0.0,
                "surv": 0.0,
                "kl_z": 0.0,
                "kl_s": 0.0,
            }
            val_counts = {k: 0 for k in val_sums}

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) != 3:
                        raise ValueError(
                            "Expected batch to be ((x_base, m_base), (x_state, m_state), labels). "
                            f"Got length {len(batch)}."
                        )

                    (x_base, m_base), (x_state, m_state), labels = batch

                    x_base = _to_device(x_base, device)
                    m_base = _to_device(m_base, device)
                    x_state = _to_device(x_state, device)
                    m_state = _to_device(m_state, device)
                    labels = _to_device(labels, device)

                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        loss_out = model.compute_loss(
                            x_base=x_base,
                            m_base=m_base,
                            x_state=x_state,
                            m_state=m_state,
                            labels=labels,
                            tau=tau_epoch,
                            n_generated_sample=n_generated_sample,
                        )

                    total_loss = loss_out["total_loss"]
                    neg_elbo = loss_out.get("neg_ELBO_loss")
                    kl_z0 = loss_out.get("KL_z0")
                    kl_z_n = loss_out.get("KL_z_n")
                    kl_s = loss_out.get("KL_s")
                    surv_loss = loss_out.get("surv_loss")
                    
                    batch_size = x_base.shape[0]

                    _accumulate_metric(val_sums, val_counts, "total", total_loss, batch_size)
                    _accumulate_metric(val_sums, val_counts, "neg_elbo", neg_elbo, batch_size)
                    _accumulate_metric(val_sums, val_counts, "surv", surv_loss, batch_size)
                    _accumulate_metric(val_sums, val_counts, "kl_s", kl_s, batch_size)

                    if kl_z0 is not None and kl_z_n is not None:
                        _accumulate_metric(val_sums, val_counts, "kl_z", kl_z0 + kl_z_n, batch_size)

            avg_val_total = _avg_metric(val_sums, val_counts, "total")
            avg_val_neg_elb = _avg_metric(val_sums, val_counts, "neg_elbo")
            avg_val_surv = _avg_metric(val_sums, val_counts, "surv")
            avg_val_kl_z = _avg_metric(val_sums, val_counts, "kl_z")
            avg_val_kl_s = _avg_metric(val_sums, val_counts, "kl_s")

            val_total_hist.append(avg_val_total)
            val_neg_elbo_hist.append(avg_val_neg_elb)
            val_surv_hist.append(avg_val_surv)
            val_kl_z_hist.append(avg_val_kl_z)
            val_kl_s_hist.append(avg_val_kl_s)

            current_monitor = avg_val_surv

            improved = np.isfinite(current_monitor) and (
                (current_monitor + min_delta) < best_monitor or not np.isfinite(best_monitor)
            )

            if improved:
                best_monitor = current_monitor
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_since_best = 0

                if save_best_path:
                    os.makedirs(os.path.dirname(save_best_path) or ".", exist_ok=True)
                    torch.save(best_state, save_best_path)
                    best_path = save_best_path
                if verbose:
                    log.info(
                        "New best %s=%.4f at epoch %d (saved=%s)",
                        monitor_name,
                        best_monitor,
                        best_epoch,
                        bool(save_best_path),
                    )
            else:
                epochs_since_best += 1
                if epochs_since_best >= patience:
                    if verbose:
                        log.info(
                            "Early stopping at epoch %d (no improvement for %d evals). "
                            "Best %s: %.4f (epoch %d)",
                            epoch,
                            epochs_since_best,
                            monitor_name,
                            best_monitor,
                            best_epoch,
                        )
                    break
        
            if trial is not None and np.isfinite(current_monitor):
                step = step_offset + epoch
                trial.report(current_monitor, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        else:
            val_total_hist.append(np.nan)
            val_neg_elbo_hist.append(np.nan)
            val_surv_hist.append(np.nan)
            val_kl_z_hist.append(np.nan)
            val_kl_s_hist.append(np.nan)

        if epoch > warmup_epochs or not use_manual_warmup:
            if scheduler_name == "ReduceLROnPlateau":
                if do_eval and np.isfinite(current_monitor):
                    lr_scheduler.step(current_monitor)
            else:
                lr_scheduler.step()

        if verbose:
            if do_eval and use_val:
                log.info(
                    "Epoch [%d/%d] - TrainTotal: %.4f  ValTotal: %.4f  "
                    "TrainSurv: %.4f  ValSurv: %.4f  LR: %.6f  tau=%.4f  "
                    "ELBO prec: %.4f  Traj prec: %.4f  | "
                    "nll=%.4f aux=%.4f(w=%.4f)",
                    epoch, epochs, avg_train_total, avg_val_total,
                    avg_train_surv, avg_val_surv, current_lr, tau_epoch,
                    avg_train_w_elbo, avg_train_w_surv,
                    avg_surv_nll, avg_surv_aux_raw, avg_surv_aux_weighted,
                )
            else:
                log.info(
                    "Epoch [%d/%d] - TrainTotal: %.4f  LR: %.6f  tau=%.4f%s",
                    epoch,
                    epochs,
                    avg_train_total,
                    current_lr,
                    tau_epoch,
                    " (no val this epoch)" if not do_eval else "",
                )

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            log.info(
                "Model weights restored from best epoch %d (%s=%.4f)",
                best_epoch,
                monitor_name,
                best_monitor,
            )

    if plot_curves:
        _plot_training(
            train_hist=train_total_hist,
            val_hist=val_total_hist,
            lr_hist=lr_hist,
            w_elbo_hist=train_w_elbo_hist,
            w_surv_hist=train_w_surv_hist,
            surv_components={
                "nll": train_surv_nll_hist,
                "aux_raw": train_surv_aux_raw_hist,
                "aux_weighted": train_surv_aux_weighted_hist,
            },
            show_plot=show_plot,
            plot_path=plot_path,
            log=log,
        )

    history: Dict[str, Any] = {
        "loss_train": np.array(train_total_hist),
        "loss_val": np.array(val_total_hist) if use_val else None,
        "neg_ELBO_train": np.array(train_neg_elbo_hist),
        "neg_ELBO_val": np.array(val_neg_elbo_hist) if use_val else None,
        "surv_loss_train": np.array(train_surv_hist),
        "surv_loss_val": np.array(val_surv_hist) if use_val else None,
        "kl_z_train": np.array(train_kl_z_hist),
        "kl_z_val": np.array(val_kl_z_hist) if use_val else None,
        "kl_s_train": np.array(train_kl_s_hist),
        "kl_s_val": np.array(val_kl_s_hist) if use_val else None,
        "lr": np.array(lr_hist),
        "best_epoch": best_epoch,
        "best_monitor_name": monitor_name,
        "best_monitor_value": best_monitor if use_val and np.isfinite(best_monitor) else None,
        "best_path": best_path,
        "w_elbo_train": np.array(train_w_elbo_hist),
        "w_surv_train": np.array(train_w_surv_hist),
        "elapsed_time_sec": time.time() - start_time,
        "surv_nll_train": np.array(train_surv_nll_hist),
        "surv_aux_raw_train": np.array(train_surv_aux_raw_hist),
        "surv_aux_weighted_train": np.array(train_surv_aux_weighted_hist),
    }

    return history


def _plot_training(
    train_hist,
    val_hist,
    lr_hist,
    w_elbo_hist=None,
    w_surv_hist=None,
    surv_components=None,
    plot_path: Optional[str] = None,
    show_plot: bool = False,
    log: Optional[logging.Logger ] = None,
) -> None:
    log = log or logger

    try:
        n = min(len(train_hist), len(val_hist), len(lr_hist))
        if n == 0:
            log.warning("Plotting skipped: empty histories.")
            return

        has_weights = (
            w_elbo_hist is not None
            and w_surv_hist is not None
            and len(w_elbo_hist) >= n
            and len(w_surv_hist) >= n
            and (
                any(np.isfinite(v) for v in w_elbo_hist[:n])
                or any(np.isfinite(v) for v in w_surv_hist[:n])
            )
        )

        # Keep only components that have at least one finite value
        comp_series = {}
        if surv_components:
            for name, series in surv_components.items():
                if series is not None and len(series) >= n and any(
                    np.isfinite(v) for v in series[:n]
                ):
                    comp_series[name] = series
        has_components = len(comp_series) > 0

        ncols = 2 + int(has_weights) + int(has_components)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
        if ncols == 1:
            axes = [axes]

        epochs_r = np.arange(1, n + 1)
        idx = 0

        # Total loss
        axes[idx].plot(epochs_r, train_hist[:n], label="Train")
        if any(np.isfinite(v) for v in val_hist[:n]):
            axes[idx].plot(epochs_r, val_hist[:n], label="Val")
        axes[idx].set_title("Total loss")
        axes[idx].set_xlabel("Epoch")
        axes[idx].grid(True)
        axes[idx].legend()
        idx += 1

        # Learning rate
        axes[idx].plot(epochs_r, lr_hist[:n])
        axes[idx].set_title("Learning rate")
        axes[idx].set_xlabel("Epoch")
        axes[idx].grid(True)
        idx += 1

        # Task precisions
        if has_weights:
            axes[idx].plot(epochs_r, w_elbo_hist[:n], label="ELBO precision")
            axes[idx].plot(epochs_r, w_surv_hist[:n], label="Trajectory precision")
            axes[idx].set_title("Learned task precisions")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_yscale("log")
            axes[idx].grid(True)
            axes[idx].legend()
            idx += 1

        # MuStaRDT loss components
        if has_components:
            for name, series in comp_series.items():
                axes[idx].plot(epochs_r, np.asarray(series[:n]), label=name)
            axes[idx].set_title("Survival loss components (train)")
            axes[idx].set_xlabel("Epoch")
            axes[idx].grid(True)
            axes[idx].legend()
            idx += 1

        fig.tight_layout()

        if plot_path:
            os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
            fig.savefig(plot_path, bbox_inches="tight")
            log.info("Training curves saved to %s", plot_path)

        if show_plot:
            plt.show()

        plt.close(fig)

    except Exception as e:
        log.warning("Plotting failed: %s", e)