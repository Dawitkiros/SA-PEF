import base64
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from flwr.common import (
    FitRes, Parameters, Scalar, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg 
from logging import INFO, WARNING
from flwr.common.logger import log

def _is_bn_mod(m):
    import torch.nn as nn
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))

def _flatten_ndarrays(nds: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in nds])

def _split_by_shapes(flat: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    out, off = [], 0
    for shp in shapes:
        n = int(np.prod(shp))
        out.append(flat[off:off+n].reshape(shp).astype(np.float32, copy=False))
        off += n
    return out

def _npstat(v: np.ndarray) -> str:
    if v.size == 0:
        return "empty"
    n = float(np.linalg.norm(v))
    mx = float(np.max(np.abs(v)))
    any_nan = bool(np.isnan(v).any() or np.isinf(v).any())
    return f"||·||2={n:.3e} max|·|={mx:.3e} nan={any_nan}"


def _nd_to_dense_delta(vec_len: int, nds: List[np.ndarray]) -> np.ndarray:
    """Accepts [dense] OR [idx, val] OR per-layer ndarray list."""
    if len(nds) == 0:
        raise ValueError("Empty client payload")
    if len(nds) == 1 and nds[0].ndim == 1 and nds[0].size == vec_len:
        return nds[0].astype(np.float32, copy=False)
    if len(nds) >= 2 and nds[0].dtype != np.float32 and nds[1].dtype == np.float32:
        idx = nds[0].astype(np.int64, copy=False).ravel()
        val = nds[1].astype(np.float32, copy=False).ravel()
        dense = np.zeros(vec_len, dtype=np.float32); dense[idx] = val
        return dense
    flat = _flatten_ndarrays(nds)
    if flat.size != vec_len:
        raise ValueError(f"Delta length mismatch: got {flat.size}, expected {vec_len}")
    return flat

def _nd_to_delta_and_tau(vec_len: int, nds: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Dense [delta] / [delta, tau] or sparse [idx, val] / [idx, val, tau]."""
    if len(nds) == 1 and nds[0].size == vec_len:
        return nds[0].astype(np.float32), 1.0
    if len(nds) == 2 and nds[0].size == vec_len and nds[1].size == 1:
        return nds[0].astype(np.float32), float(nds[1].astype(np.float32)[0])

    if len(nds) >= 2 and nds[0].ndim == 1:
        idx = nds[0].astype(np.int64).ravel()
        val = nds[1].astype(np.float32).ravel()
        delta = np.zeros(vec_len, dtype=np.float32)
        delta[idx] = val
        tau = 1.0
        if len(nds) >= 3 and nds[2].size == 1:
            tau = float(nds[2].astype(np.float32)[0])
        return delta, tau

    raise ValueError("Malformed client payload")


def _parse_bn_tail(nds: List[np.ndarray], expect_len: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Assume tail is [bn_mu, bn_var, count]. If absent, return (None,None,0.0)."""
    if len(nds) < 3:
        return None, None, 0.0
    bn_mu, bn_var, count = nds[-3], nds[-2], nds[-1]
    if bn_mu.size != expect_len or bn_var.size != expect_len:
        return None, None, 0.0
    c = float(np.asarray(count).ravel()[0])
    return bn_mu.astype(np.float32, copy=False), bn_var.astype(np.float32, copy=False), c



class ScaffoldLike(FedAvg):
    def __init__(self, *, net,test_loader, run_config, **fedavg_kwargs):
        super().__init__(**fedavg_kwargs)
        self.net = net
        self.test_loader = test_loader
        self.rc = run_config
        self.algo  = run_config["algorithm"]["name"].lower()
        self.eta_l = float(run_config.get("local_lr", run_config.get("learning_rate", 0.05)))
        self.eta_g = float(run_config.get("global_lr", 1.0))
        self.K     = int(run_config.get("local_steps", run_config.get("local_epochs", 1)))
        self.N     = int(run_config.get("total_num_clients", run_config.get("num_clients", 1)))
        # Trainables state
        self._shapes: List[Tuple[int, ...]] = []
        self._D: int = 0
        self._x: Optional[np.ndarray] = None
        self._c: Optional[np.ndarray] = None
        # BN mapping (order is model order)
        self._bn_sizes: List[int] = []
        self._bn_total: int = 0
        self._bn_has_any: bool = False
        self.bn_layers = []
        self.bn_sizes = []
        for m in self.net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                self.bn_layers.append(m)
                self.bn_sizes.append(m.num_features)
        self.bn_total = sum(self.bn_sizes)


    def initialize_parameters(self, client_manager):
        import torch
        # x, c live in trainables-only space.
        self._trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        trainables = [p.detach().cpu().numpy() for p in self._trainable_params]
        self._shapes = [p.shape for p in self._trainable_params]
        self._x = _flatten_ndarrays(trainables)
        self._D = int(self._x.size)
        self._c = np.zeros(self._D, dtype=np.float32)

        self._bn_sizes.clear()
        for m in self.net.modules():
            if _is_bn_mod(m) and m.track_running_stats:
                self._bn_sizes.append(int(m.running_mean.numel()))
        self._bn_total = sum(self._bn_sizes)
        self._bn_has_any = self._bn_total > 0

        # IMPORTANT: broadcast FULL state_dict, not just trainables
        full = [v.cpu().numpy() for v in self.net.state_dict().values()]
        return ndarrays_to_parameters(full)


    def _aggregate_and_apply_bn(self, mu_list, var_list, n_list):
        """Pooled BN: μ = Σ n μ / Σ n ;  σ² = Σ n (σ² + μ²)/Σ n − μ²"""
        total_n = float(np.sum(n_list))
        if total_n <= 0 or not self.bn_layers:
            return

        bn_total = self.bn_total
        mu_acc = np.zeros(bn_total, dtype=np.float32)
        m2_acc = np.zeros(bn_total, dtype=np.float32)
        for mu_i, var_i, n_i in zip(mu_list, var_list, n_list):
            if mu_i is None or var_i is None or n_i <= 0:
                continue
            mu_acc += n_i * mu_i
            m2_acc += n_i * (var_i + mu_i * mu_i)

        mu = mu_acc / max(total_n, 1e-12)
        var = m2_acc / max(total_n, 1e-12) - mu * mu
        var = np.maximum(var, 1e-8)  # guard

        off = 0
        with torch.no_grad():
            for m, c in zip(self.bn_layers, self.bn_sizes):
                m.running_mean.copy_(torch.from_numpy(mu[off:off+c]).to(m.running_mean.device))
                m.running_var.copy_(torch.from_numpy(var[off:off+c]).to(m.running_var.device))
                off += c

    def configure_fit(self, server_round, parameters, client_manager):
        pairs = super().configure_fit(server_round, parameters, client_manager)
        c_b64 = base64.b64encode(self._c.astype(np.float32).tobytes()).decode("ascii")
        for i, (client, ins) in enumerate(pairs):
            cfg = dict(ins.config)
            cfg.update({
                "server_round": server_round,
                "learning_rate": self.eta_l,
                "local_epochs": self.K,
                "global_lr": 1.0,  # kept for completeness, not used by client
                "algorithm": self.algo,
                "vec_len": self._D,
                "c_global_b64": c_b64,
                "bn_vec_len": self._bn_total,
            })
            pairs[i] = (client, FitIns(ins.parameters, cfg))
        return pairs

    def _assign_bn_buffers(self, mu: np.ndarray, var: np.ndarray) -> None:
        """Write concatenated BN stats back into self.net buffers."""
        import torch
        off = 0
        for m in self.net.modules():
            if _is_bn_mod(m) and m.track_running_stats:
                n = m.running_mean.numel()
                m.running_mean.copy_(torch.from_numpy(mu[off:off+n]).view_as(m.running_mean))
                m.running_var.copy_(torch.from_numpy(var[off:off+n]).view_as(m.running_var))
                off += n

    def _parse_client_payload(self, nds: list[np.ndarray], vec_len: int):
        """
        Parse client payload: [y_delta, c_delta, bn_mu, bn_var, count]
        Returns: y_delta, c_delta, mu, var, n_seen
        """
        mu = var = None
        n_seen = 0.0

        if len(nds) < 2:
            raise ValueError("Expected at least [y_delta, c_delta]")

        y_delta = nds[0].astype(np.float32).ravel()
        c_delta = nds[1].astype(np.float32).ravel()

        if y_delta.size != vec_len or c_delta.size != vec_len:
            raise ValueError(f"Size mismatch: y_delta={y_delta.size}, c_delta={c_delta.size}, expected={vec_len}")

        if len(nds) >= 5 and self.bn_total > 0:
            mu_arr, var_arr, n_arr = nds[2], nds[3], nds[4]
            if mu_arr.size == self.bn_total and var_arr.size == self.bn_total and n_arr.size == 1:
                mu = mu_arr.astype(np.float32).ravel()
                var = var_arr.astype(np.float32).ravel()
                n_seen = float(n_arr.astype(np.float32)[0])

        return y_delta, c_delta, mu, var, n_seen
    
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if self._x is None or self._c is None:
            raise RuntimeError("ScaffoldLike not initialized")

        # SCALLION / SCAFCOM branch (uses compressed δ̃_i payloads).
        if self.algo in ("scallion", "scafcom"):
            sum_delta_tilde = np.zeros(self._D, dtype=np.float32)
            mu_list, var_list, n_list = [], [], []
            valid_count = 0

            for _, fitres in results:
                nds = parameters_to_ndarrays(fitres.parameters)

                # Client-average aggregation: w_i = 1.0.
                w_i = 1.0

                if len(nds) == 1:
                    delta_tilde_i = nds[0].astype(np.float32)
                    mu_i = var_i = None
                    n_i = 0.0
                elif len(nds) in (2, 5):
                    idx = nds[0]
                    val = nds[1]

                    delta_tilde_i = np.zeros(self._D, dtype=np.float32)
                    if idx.size > 0:
                        idx_int = idx.astype(np.int64)
                        val_f = val.astype(np.float32)
                        delta_tilde_i[idx_int] = val_f

                    if len(nds) == 5:
                        mu_i = nds[2]
                        var_i = nds[3]
                        n_arr = nds[4]
                        n_i = float(n_arr[0]) if n_arr.size > 0 else 0.0
                    else:
                        mu_i = var_i = None
                        n_i = 0.0
                else:
                    log(
                        WARNING,
                        f"[WARN] Unexpected SCALLION/SCAFCOM payload length={len(nds)}, skipping client"
                    )
                    continue

                if not np.isfinite(delta_tilde_i).all():
                    log(WARNING, "[WARN] Non-finite δ̃_i, skipping client")
                    continue

                sum_delta_tilde += w_i * delta_tilde_i
                valid_count += 1

                if mu_i is not None and var_i is not None and n_i > 0.0:
                    mu_list.append(mu_i)
                    var_list.append(var_i)
                    n_list.append(n_i)

            if valid_count == 0:
                return ndarrays_to_parameters(_split_by_shapes(self._x, self._shapes)), {}

            m = float(valid_count)
            avg_delta_tilde = sum_delta_tilde / m

            c_old = self._c.copy()

            # Treat δ̃ as gradient-like: x^{t+1} = x^t - γ * (avg δ̃ + c^t).
            factor = self.eta_g
            self._x = self._x - factor * (avg_delta_tilde + c_old)

            # c^{t+1} = c^t + (m/N) * avg δ̃
            self._c = c_old + (valid_count / float(self.N)) * avg_delta_tilde


            print(
                f"[SRV:R{server_round}] algo={self.algo} "
                f"||avg_delta||={np.linalg.norm(avg_delta_tilde):.3e} "
                f"||x||={np.linalg.norm(self._x):.3e} "
                f"||c||={np.linalg.norm(self._c):.3e}"
            )

            new_trainables = _split_by_shapes(self._x, self._shapes)
            import torch
            with torch.no_grad():
                for p, arr in zip(self._trainable_params, new_trainables):
                    p.copy_(torch.from_numpy(arr).to(p.device))

            if mu_list:
                self._aggregate_and_apply_bn(mu_list, var_list, n_list)

            full = [v.cpu().numpy() for v in self.net.state_dict().values()]
            parameters_aggregated = ndarrays_to_parameters(full)

            metrics = {
                "avg_y_delta_l2": float(np.linalg.norm(avg_delta_tilde)),
                "avg_c_delta_l2": 0.0,
                "x_norm_l2": float(np.linalg.norm(self._x)),
                "c_norm_l2": float(np.linalg.norm(self._c)),
                "bn_seen_total": float(np.sum(n_list)) if n_list else 0.0,
            }
            return parameters_aggregated, metrics



        # Default SCAFFOLD branch (uses y_delta + c_delta).
        sum_y_delta = np.zeros(self._D, dtype=np.float32)
        sum_c_delta = np.zeros(self._D, dtype=np.float32)
        mu_list, var_list, n_list = [], [], []
        valid_count = 0

        for _, fitres in results:
            nds = parameters_to_ndarrays(fitres.parameters)
            y_delta_i, c_delta_i, mu_i, var_i, n_i = self._parse_client_payload(nds, self._D)

            if not np.isfinite(y_delta_i).all() or not np.isfinite(c_delta_i).all():
                print(f"[WARN] Non-finite updates, skipping client")
                continue

            sum_y_delta += y_delta_i
            sum_c_delta += c_delta_i
            valid_count += 1

            if mu_i is not None and var_i is not None and n_i > 0.0:
                mu_list.append(mu_i)
                var_list.append(var_i)
                n_list.append(n_i)

        if valid_count == 0:
            return ndarrays_to_parameters(_split_by_shapes(self._x, self._shapes)), {}

        m = float(valid_count)
        avg_y_delta = sum_y_delta / m
        avg_c_delta = sum_c_delta / m

        self._x = self._x + self.eta_g * avg_y_delta
        self._c = self._c + (valid_count / float(self.N)) * avg_c_delta

        print(
            f"[SRV:R{server_round}] "
            f"||avg_y_delta||={np.linalg.norm(avg_y_delta):.3e} "
            f"||avg_c_delta||={np.linalg.norm(avg_c_delta):.3e} "
            f"||x||={np.linalg.norm(self._x):.3e} "
            f"||c||={np.linalg.norm(self._c):.3e}"
        )

        new_trainables = _split_by_shapes(self._x, self._shapes)
        import torch
        with torch.no_grad():
            for p, arr in zip(self._trainable_params, new_trainables):
                p.copy_(torch.from_numpy(arr).to(p.device))

        if mu_list:
            self._aggregate_and_apply_bn(mu_list, var_list, n_list)

        full = [v.cpu().numpy() for v in self.net.state_dict().values()]
        parameters_aggregated = ndarrays_to_parameters(full)

        return parameters_aggregated, {
            "avg_y_delta_l2": float(np.linalg.norm(avg_y_delta)),
            "avg_c_delta_l2": float(np.linalg.norm(avg_c_delta)),
            "x_norm_l2": float(np.linalg.norm(self._x)),
            "c_norm_l2": float(np.linalg.norm(self._c)),
            "bn_seen_total": float(np.sum(n_list)) if n_list else 0.0,
        }


    def evaluate(self, server_round: int, parameters: Parameters):
        if self.evaluate_fn is None:
            return None
        nds = parameters_to_ndarrays(parameters)
        return self.evaluate_fn(server_round, nds, {})

