"""CSER strategy: EF/top-k aggregation + periodic error-reset digest.

CSER is FedAvg + top-k EF with one extra mechanism: every `H` rounds ("reset
round") each client sends a second sparse packet containing a top-k1 digest of
its accumulated error residual. The server averages the digests into `e_bar`,
adds `e_bar` to the global model, and broadcasts `e_bar` back so each
contributing client can reconcile its residual against the consensus on the
next round.

This subclasses FedSpars and only overrides the hooks needed for the digest
channel: extra fit config, payload collection, post-aggregation assimilation,
and metric naming.
"""
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from flwr.common import Scalar

from .fedspars import FedSpars, _weighted_mean


class CSERStrategy(FedSpars):
    """FedSpars + periodic error-reset digest broadcast."""

    log_prefix = "CSER Round"

    def __init__(
        self,
        net: torch.nn.Module,
        config: Dict,
        H: int = 5,
        reset_frac: float = 0.1,
        **kwargs,
    ):
        self.H = int(H)
        self.reset_frac = float(reset_frac)
        # Sparse average of last round's e-packets, broadcast next round.
        self._last_e_bar: Optional[Tuple[List[int], List[float]]] = None
        super().__init__(net=net, config=config, **kwargs)

    def _build_save_dir(self) -> str:
        # Tag CSER knobs so H=5/rf=0.1 does not collide with H=10/rf=0.05 runs.
        base = super()._build_save_dir()
        return f"{base}_H{self.H}_rf{self.reset_frac}"

    def _extra_fit_config(self, server_round: int) -> Mapping[str, Scalar]:
        if self._last_e_bar is not None:
            idx_list, val_list = self._last_e_bar
            cfg = {
                "e_bar_idx": json.dumps([int(i) for i in idx_list]),
                "e_bar_val": json.dumps([float(v) for v in val_list]),
            }
            self._last_e_bar = None  # consumed by clients this round
            return cfg
        return {"e_bar_idx": "[]", "e_bar_val": "[]"}

    def _init_extras(self, n: int, device: torch.device) -> Dict[str, Any]:
        return {"n": n, "e_packets": []}

    def _collect_extras(self, payload, extras, weight: float, device: torch.device) -> None:
        if len(payload) >= 7:
            extras["e_packets"].append((payload[5], payload[6]))

    def _apply_extras(
        self,
        vec_global: torch.Tensor,
        extras,
        server_round: int,
    ) -> Mapping[str, Scalar]:
        e_packets = extras["e_packets"]
        n = extras["n"]
        device = vec_global.device
        reset_round = (server_round % self.H) == 0

        if reset_round and e_packets:
            e_acc = torch.zeros(n, device=device, dtype=torch.float32)
            for e_idx_np, e_val_np in e_packets:
                if e_idx_np.size == 0:
                    continue
                e_idx_t = torch.from_numpy(e_idx_np).to(device=device, dtype=torch.long)
                e_val_t = torch.from_numpy(e_val_np).to(device=device, dtype=torch.float32)
                # scatter_add in case multiple clients touch the same index
                e_acc.scatter_add_(0, e_idx_t, e_val_t)
            # Uniform mean over senders, per the V1 spec.
            e_acc /= float(len(e_packets))
            vec_global += e_acc

            nz = torch.nonzero(e_acc, as_tuple=False).squeeze(1)
            if nz.numel() > 0:
                self._last_e_bar = (
                    nz.detach().cpu().tolist(),
                    e_acc[nz].detach().cpu().tolist(),
                )
                print(
                    f"[CSER Round {server_round}] assimilated e_bar: "
                    f"nnz={nz.numel()} ||e_acc||_2={float(e_acc.norm()):.3e}"
                )
            else:
                self._last_e_bar = None
        elif reset_round:
            print(f"[CSER Round {server_round}] reset round but no e-packets received")

        return {
            "cser_reset_round": int(reset_round),
            "cser_num_packets": int(len(e_packets)),
        }

    def _aggregate_metrics(self, fit_metrics, extras_metrics):
        out: Dict[str, Scalar] = {
            "uplink_bits_total": _weighted_mean("uplink_bits_total", fit_metrics) or 0.0,
            "grad_mismatch_sq": _weighted_mean("grad_mismatch_sq", fit_metrics),
        }
        out.update(extras_metrics)
        return out
