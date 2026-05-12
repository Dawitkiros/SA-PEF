"""Compute gradient mismatch and theory metrics."""
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def _grad_at(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Flat gradient of CE loss on (x, y) for trainable params; model already on the right point."""
    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss()
    with torch.enable_grad():
        loss = criterion(model(x), y)
        grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            create_graph=False,
            retain_graph=False,
        )
    return torch.cat([g.reshape(-1) for g in grads]).detach()


def compute_gradient_vector(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_samples: int = 1024,
) -> torch.Tensor:
    """Single-batch gradient at the current model parameters."""
    model.eval()
    data_iter = iter(dataloader)
    x, y = next(data_iter)
    if x.shape[0] > max_samples:
        x, y = x[:max_samples], y[:max_samples]
    x, y = x.to(device), y.to(device)
    return _grad_at(model, x, y)


def compute_pre_send_metrics(
    model: nn.Module,
    device: torch.device,
    valloader,
    w_global: torch.Tensor,
    e_residual: torch.Tensor,
    alpha_preview: float,
    server_round: int,
    config: Dict,
) -> Tuple[Dict, Dict]:
    """Gradient-mismatch probe ε̂_r^(k)(α) = ‖g_r − g_r^(α,k)‖².

    Procedure:
      - model in eval mode (BN frozen, dropout disabled),
      - the SAME mini-batch S used for both gradient evaluations,
      - parameters restored after probing; no optimizer/BN buffer mutation.

    `e_residual` is the client's leftover-after-compression residual e_t.
    The eval point w_r − α·e_r^(k) equals w_r + α·e_t under this sign convention.
    """
    was_training = model.training
    model.eval()

    w_backup = torch.nn.utils.parameters_to_vector(
        [p.data for p in model.parameters() if p.requires_grad]
    ).clone()

    # Reuse the same batch for both gradient evaluations.
    x, y = next(iter(valloader))
    x, y = x.to(device), y.to(device)

    # g_r at w_r
    torch.nn.utils.vector_to_parameters(
        w_global,
        [p for p in model.parameters() if p.requires_grad],
    )
    g_wr = _grad_at(model, x, y)
    grad_norm_sq = float(g_wr.pow(2).sum().item())

    # g_r^(α,k) at w_r + α·e_t  (= w_r − α·e_r^(k) under our residual sign)
    preview = w_global + alpha_preview * e_residual
    torch.nn.utils.vector_to_parameters(
        preview,
        [p for p in model.parameters() if p.requires_grad],
    )
    g_preview = _grad_at(model, x, y)
    grad_mismatch_sq = float((g_wr - g_preview).pow(2).sum().item())

    # Restore params + zero stale grads + restore train/eval mode.
    torch.nn.utils.vector_to_parameters(
        w_backup,
        [p for p in model.parameters() if p.requires_grad],
    )
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    residual_energy = float(e_residual.pow(2).sum().item())

    from .utils import compute_delta_from_sparsity
    delta = compute_delta_from_sparsity(config["sparsify_by"])
    local_work = (
        config.get("local_steps", config["local_epochs"])
        if str(config.get("local_work_mode", "epochs")).lower() == "steps"
        else config["local_epochs"]
    )
    s_r = config["learning_rate_max"] * config.get("L_est", 1.0) * local_work
    gamma = 1.0 - 1.0 / delta if delta > 1.0 else 0.0
    rho_r = gamma * (
        2.0 * (1.0 - alpha_preview) ** 2 + 24.0 * (alpha_preview ** 2) * (s_r ** 2)
    )

    record = {
        "round": server_round,
        "grad_norm_sq": grad_norm_sq,
        "residual_energy": residual_energy,
        "grad_mismatch_sq": grad_mismatch_sq,
        "rho_r": rho_r,
        "s_r": s_r,
        "alpha_r": float(alpha_preview),
    }
    metrics = {
        "grad_norm_sq": grad_norm_sq,
        "residual_energy": residual_energy,
        "grad_mismatch_sq": grad_mismatch_sq,
        "rho_r": rho_r,
        "uplink_bits_total": 0.0,  # filled after compression
    }
    return record, metrics


def append_jsonl(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def compute_comm_bits(
    indices: "np.ndarray",
    values: "np.ndarray",
    bn_stats: Tuple = None,  # pyright: ignore[reportArgumentType]
    count: "np.ndarray" = None,  # pyright: ignore[reportArgumentType]
) -> Dict:
    """Per-round uplink payload size in MiB (kept name `bits` for back-compat)."""
    bits_update = (indices.nbytes + values.nbytes) / (1024 ** 2)
    bits_bn = 0
    if bn_stats is not None:
        bn_mu, bn_var = bn_stats
        bits_bn = (bn_mu.nbytes + bn_var.nbytes) / (1024 ** 2)
    meta_bits = 128 / (1024 ** 2)
    total = bits_update + bits_bn + meta_bits
    return {
        "uplink_bits_update": bits_update,
        "uplink_bits_bn": bits_bn,
        "uplink_bits_total": total,
    }
