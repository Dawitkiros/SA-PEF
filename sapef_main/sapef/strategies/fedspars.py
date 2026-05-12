"""FedSpars strategy with sparse aggregation and BN stats."""
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _weighted_mean(key: str, fit_metrics) -> Optional[float]:
    vals, weights = [], []
    for n, m in fit_metrics:
        if key in m and m[key] is not None:
            vals.append(float(m[key]))
            weights.append(n)
    if not vals:
        return None
    return float(np.average(vals, weights=weights))


class FedSpars(FedAvg):
    """Sparse top-k aggregation with BN stats. Subclasses override hooks for extras."""

    log_prefix = "Round"

    def __init__(
        self,
        net: torch.nn.Module,
        config: Dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.net = net
        self.config = config
        self.eta_g = float(config.get("global_lr", 1.0))
        self._bn_len = sum(
            m.running_mean.numel()
            for m in self.net.modules()
            if isinstance(m, torch.nn.BatchNorm2d)
        )
        self._save_dir = self._build_save_dir()
        os.makedirs(self._save_dir, exist_ok=True)

    # ---- override points -------------------------------------------------
    def _build_save_dir(self) -> str:
        c = self.config
        return (
            f"checkpoints/{c['algorithm']['name']}/"
            f"{c['comp_type']}_{c['sparsify_by']}_{c['dataset']['name']}_"
            f"{c['dataset']['partitioning']}_{c['num_clients']}_"
            f"{c['clients_per_round']}_{c['alpha_r']}_{c['learning_rate']}_"
            f"{c['seed']}"
        )

    def _init_extras(self, n: int, device: torch.device) -> Any:
        return None

    def _collect_extras(self, payload, extras, weight: float, device: torch.device) -> None:
        pass

    def _apply_extras(
        self,
        vec_global: torch.Tensor,
        extras,
        server_round: int,
    ) -> Mapping[str, Scalar]:
        return {}

    def _aggregate_metrics(
        self,
        fit_metrics: List[Tuple[int, Dict]],
        extras_metrics: Mapping[str, Scalar],
    ) -> Dict[str, Scalar]:
        out: Dict[str, Scalar] = {
            "grad_norm_sq": _weighted_mean("grad_norm_sq", fit_metrics),
            "residual_energy": _weighted_mean("residual_energy", fit_metrics),
            "grad_mismatch_sq": _weighted_mean("grad_mismatch_sq", fit_metrics),
            "rho_r": _weighted_mean("rho_r", fit_metrics),
            "uplink_bits_total": _weighted_mean("uplink_bits_total", fit_metrics),
        }
        out.update(extras_metrics)
        return out

    # ---- main flow -------------------------------------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        cfg: Dict[str, Scalar] = {"server_round": server_round}
        if self.on_fit_config_fn is not None:
            cfg.update(self.on_fit_config_fn(server_round))
        cfg.update(self._extra_fit_config(server_round))

        fit_ins = FitIns(parameters, cfg)
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        print(f"[{self.log_prefix} {server_round}] Sampled {len(clients)} clients")
        return [(client, fit_ins) for client in clients]

    def _extra_fit_config(self, server_round: int) -> Mapping[str, Scalar]:
        return {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        print(f"\n[{self.log_prefix} {server_round}] Aggregating {len(results)} client updates...")

        device = next(self.net.parameters()).device
        with torch.no_grad():
            vec_global = torch.nn.utils.parameters_to_vector(
                [p.data for p in self.net.parameters() if p.requires_grad]
            ).to(device)
            n = vec_global.numel()

            agg_update = torch.zeros_like(vec_global)
            agg_count = torch.zeros_like(vec_global)
            mu_sum = torch.zeros(self._bn_len, device=device)
            ex2_sum = torch.zeros(self._bn_len, device=device)
            total_w = 0.0
            extras = self._init_extras(n, device)

            for _, fit_res in results:
                payload = parameters_to_ndarrays(fit_res.parameters)
                idx_np, val_np, mu_np, var_np, cnt_np = payload[:5]
                w = float(cnt_np[0])

                idx_t = torch.from_numpy(idx_np).to(device=device, dtype=torch.long)
                val_t = torch.from_numpy(val_np).to(device=device, dtype=torch.float32)
                agg_update[idx_t] += val_t * w
                agg_count[idx_t] += w

                if mu_np.size > 0:
                    mu_t = torch.from_numpy(mu_np).to(device).float()
                    var_t = torch.from_numpy(var_np).to(device).float()
                    mu_sum += mu_t * w
                    ex2_sum += (var_t + mu_t.pow(2)) * w

                total_w += w
                self._collect_extras(payload, extras, w, device)

            mask = agg_count > 0
            agg_update[mask] /= agg_count[mask]
            vec_global[mask] += self.eta_g * agg_update[mask]

            extras_metrics = self._apply_extras(vec_global, extras, server_round)

            torch.nn.utils.vector_to_parameters(
                vec_global,
                [p for p in self.net.parameters() if p.requires_grad],
            )

            if total_w > 0 and self._bn_len > 0:
                mu_bar = mu_sum / total_w
                var_bar = (ex2_sum / total_w) - mu_bar.pow(2)
                var_bar.clamp_(min=1e-3)
                ptr = 0
                for m in self.net.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        c = m.running_mean.numel()
                        m.running_mean.copy_(mu_bar[ptr:ptr + c])
                        m.running_var.copy_(var_bar[ptr:ptr + c])
                        ptr += c

        torch.save(self.net.state_dict(), f"{self._save_dir}/round_{server_round}.pth")

        parameters_aggregated = ndarrays_to_parameters(
            [v.cpu().numpy() for v in self.net.state_dict().values()]
        )

        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated = self._aggregate_metrics(fit_metrics, extras_metrics)

        if self.config.get("wandb_enabled", False):
            try:
                import wandb
                wandb.log(
                    {**{k: v for k, v in metrics_aggregated.items() if v is not None}, "round": server_round},
                    step=server_round,
                )
            except Exception:
                pass

        print(f"[{self.log_prefix} {server_round}] Aggregation complete")
        for key in ("grad_norm_sq", "residual_energy", "rho_r"):
            v = metrics_aggregated.get(key)
            if v is not None:
                print(f"  {key}: {v:.6f}")

        return parameters_aggregated, metrics_aggregated
