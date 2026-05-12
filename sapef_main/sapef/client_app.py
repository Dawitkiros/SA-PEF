"""Modular client architecture with inheritance."""
from abc import ABC, abstractmethod
import base64
import json
from logging import INFO
from typing import Dict, Tuple, List
import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common.typing import NDArrays
from flwr.common import Context, ArrayRecord, Array
from flwr.common.logger import log

from .compression.registry import create_compressor
from .metrics import compute_pre_send_metrics, append_jsonl, compute_comm_bits
from .models import test, train_fedavg, train_scaffold, instantiate_model, set_weights, get_weights, train_fedavg_mnist, set_weights_with_dtype_handling, train_fedavg_num_steps
from .dataset import load_partition_data, load_data
from .utils import context_to_easydict, cosine_eta_warmfloor

def _is_bn(m: torch.nn.Module) -> bool:
    return isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm))


class BaseClient(NumPyClient, ABC):
    """Base client with shared functionality."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader,
        valloader,
        device: torch.device,
        context: Context,
        config: Dict
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.context = context
        self.config = config
        self.net.to(self.device)
        self._trainable_cache: List[torch.nn.Parameter] | None = None

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        set_weights(self.net, parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}

    @abstractmethod
    def fit(self, parameters: NDArrays, config: Dict):
        pass

    def _trainable_params(self) -> List[torch.nn.Parameter]:
        if self._trainable_cache is None:
            self._trainable_cache = [p for p in self.net.parameters() if p.requires_grad]
        return self._trainable_cache

    def _flat_trainable(self) -> torch.Tensor:
        return torch.nn.utils.parameters_to_vector(
            [p.data for p in self._trainable_params()]
        ).to(self.device)

    def _set_flat_trainable(self, flat: torch.Tensor) -> None:
        torch.nn.utils.vector_to_parameters(flat, self._trainable_params())

    def _load_sparse(self, key: str, length: int) -> torch.Tensor:
        st = self.context.state
        if st.parameters_records and key in st.parameters_records:
            rec = st.parameters_records[key]
            idx = torch.from_numpy(rec["indices"].numpy()).long().to(self.device)
            val = torch.from_numpy(rec["values"].numpy()).to(self.device)
            dense = torch.zeros(length, device=self.device, dtype=torch.float32)
            dense[idx] = val
            return dense
        return torch.zeros(length, device=self.device, dtype=torch.float32)

    def _store_sparse(self, key: str, dense: torch.Tensor) -> None:
        nz = torch.nonzero(dense, as_tuple=False).squeeze(1)
        vals = dense[nz]
        rec = ArrayRecord()
        rec["indices"] = Array(nz.detach().cpu().numpy().astype(np.int32))
        rec["values"] = Array(vals.detach().cpu().numpy().astype(np.float32))
        self.context.state.parameters_records[key] = rec

    def _drop_state_key(self, key: str) -> None:
        st = self.context.state
        if st.parameters_records and key in st.parameters_records:
            try:
                del st.parameters_records[key]
            except Exception:
                rec = ArrayRecord()
                rec["indices"] = Array(np.empty((0,), dtype=np.int32))
                rec["values"] = Array(np.empty((0,), dtype=np.float32))
                st.parameters_records[key] = rec

    def _collect_bn_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        mus, vars_ = [], []
        for m in self.net.modules():
            if _is_bn(m) and m.track_running_stats:
                mus.append(m.running_mean.detach().cpu().numpy().astype(np.float32))
                vars_.append(m.running_var.detach().cpu().numpy().astype(np.float32))
        mu = np.concatenate(mus) if mus else np.array([], dtype=np.float32)
        var = np.concatenate(vars_) if vars_ else np.array([], dtype=np.float32)
        return mu, var


def _run_local_training(
    net: torch.nn.Module,
    trainloader,
    device: torch.device,
    config: Dict,
    learning_rate: float,
) -> None:
    """Run local SGD with either an epoch or fixed-step local-work budget."""
    work_mode = str(config.get("local_work_mode", "epochs")).lower()
    if work_mode == "steps":
        train_fedavg_num_steps(
            net=net,
            trainloader=trainloader,
            device=device,
            num_steps=int(config["local_steps"]),
            learning_rate=learning_rate,
            momentum=config.get("momentum", 0.0),
            weight_decay=config.get("weight_decay", 0.0),
            proximal_mu=config.get("proximal_mu", 0.0),
        )
        return

    if work_mode != "epochs":
        raise ValueError(
            f"Unknown local_work_mode={work_mode!r}; expected 'epochs' or 'steps'."
        )

    train_fedavg(
        net,
        trainloader,
        device,
        epochs=int(config["local_epochs"]),
        learning_rate=learning_rate,
        momentum=config.get("momentum", 0.0),
        weight_decay=config.get("weight_decay", 0.0),
    )


class FedProxClient(BaseClient):
    """FedProx/FedAvg client."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.straggler_schedule = self._create_straggler_schedule()
    
    def _create_straggler_schedule(self):
        num_rounds = self.config["num_server_rounds"]
        stragglers = self.config.get("stragglers_fraction", 0.0)
        return np.random.choice(
            [0, 1], 
            size=num_rounds, 
            p=[1 - stragglers, stragglers]
        )
    
    def fit(self, parameters: NDArrays, config: Dict):
        """FedProx training with straggler handling."""
        set_weights(self.net, parameters)
        
        current_round = int(config["current_round"])
        num_epochs = self.config["local_epochs"]

        is_straggler = False
        if self.straggler_schedule[current_round - 1] and num_epochs > 1:
            if self.config.get("drop_client", False):
                return get_weights(self.net), len(self.trainloader.dataset), {"is_straggler": True}
            num_epochs = np.random.randint(1, num_epochs)
            is_straggler = True

        if self.config["dataset"]["name"].lower() == "mnist_old":
            train_fedavg_mnist(
                self.net,
                self.trainloader,
                self.device,
                epochs=num_epochs,
                learning_rate=self.config["learning_rate"],
                proximal_mu=self.config.get("proximal_mu", 0.0)
            )
        else:
            _run_local_training(
                net=self.net,
                trainloader=self.trainloader,
                device=self.device,
                config=self.config,
                learning_rate=self.config["learning_rate"],
            )

        weights = get_weights(self.net)
        # Dense uplink: full float32 payload (MiB, matching key convention used elsewhere).
        uplink_bits_total = sum(int(arr.nbytes) for arr in weights) / (1024 ** 2)
        return weights, len(self.trainloader.dataset), {
            "is_straggler": is_straggler,
            "uplink_bits_total": uplink_bits_total,
        }


class SAPEFClient(BaseClient):
    """SA-PEF client with compression and error feedback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compressor = create_compressor(
            comp_type=self.config["comp_type"],
            sparsify_by=self.config["sparsify_by"]
        )
        self._pre_metrics: Dict = {}
        self._pre_record = None
        self.node_id = self.context.node_id

    def fit(self, parameters: NDArrays, config: Dict):
        set_weights(self.net, parameters)
        server_round = config["server_round"]

        lr = cosine_eta_warmfloor(
            r=server_round,
            total_R=self.config["num_server_rounds"],
            eta_max=self.config["learning_rate_max"],
            eta_min=self.config["learning_rate_min"],
            warmup_R=self.config.get("lr_warmup_rounds", 10)
        )
        alpha_r = self._get_alpha()

        flat_global = self._flat_trainable()
        e_t = self._load_sparse("residual", flat_global.numel()) if server_round > 1 else torch.zeros_like(flat_global)
        self._set_flat_trainable(flat_global + alpha_r * e_t)

        if self.config.get("track_metrics", False):
            try:
                pre_record, pre_metrics = compute_pre_send_metrics(
                    model=self.net,
                    device=self.device,
                    valloader=self.valloader,
                    w_global=flat_global.clone(),
                    e_residual=e_t.clone(),
                    alpha_preview=float(self.config.get("metric_alpha", 1.0)),
                    server_round=server_round,
                    config=self.config
                )
                self._pre_record = pre_record
                self._pre_metrics = pre_metrics
            except Exception as ex:
                log(INFO, f"[Warn] Metrics computation failed: {ex}")
                self._pre_metrics = {}

        _run_local_training(
            net=self.net,
            trainloader=self.trainloader,
            device=self.device,
            config=self.config,
            learning_rate=lr,
        )

        payload = self._compress_delta(server_round, alpha_r, flat_global)
        return payload, len(self.trainloader.dataset), self._pre_metrics or {}

    def _get_alpha(self) -> float:
        name = self.config["algorithm"]["name"]
        if name == "ef":
            return 0.0
        if name == "saef":
            return 1.0
        return self.config.get("alpha_r", 0.0)

    def _compress_delta(self, server_round, alpha_r, flat_global):
        flat_current = self._flat_trainable()
        flat_delta = flat_current - flat_global  # Δ_t

        if server_round > 1:
            # SA-PEF: add (1 - alpha_r) * previous-residual into Δ_t before compression
            e_prev = self._load_sparse("residual", flat_delta.numel())
            flat_delta = flat_delta + (1.0 - alpha_r) * e_prev

        # flat_delta == u_t in EF notation; compress on-device
        indices, values, _meta = self.compressor.compress(flat_delta.detach())

        q_delta = torch.zeros_like(flat_delta)
        if indices.size > 0:
            idx_t = torch.from_numpy(indices).long().to(self.device)
            val_t = torch.from_numpy(values).to(self.device)
            q_delta[idx_t] = val_t

        # EF residual: e_{t+1} = u_t - C(u_t)
        self._store_sparse("residual", flat_delta - q_delta)

        bn_mu, bn_var = self._collect_bn_stats()
        bn_mu_np = bn_mu.astype(np.float16) if bn_mu.size else np.array([], dtype=np.float16)
        bn_var_np = bn_var.astype(np.float16) if bn_var.size else np.array([], dtype=np.float16)
        count_np = np.array([len(self.trainloader.dataset)], dtype=np.float32)

        payload = [indices, values, bn_mu_np, bn_var_np, count_np]

        try:
            bits = compute_comm_bits(indices, values, (bn_mu_np, bn_var_np), count_np)
            if self._pre_record:
                record = dict(self._pre_record)
                record.update(bits)
                log_path = (f"logs/clients_metrics/{self.config['comp_type']}/"
                            f"metrics_client_{self.node_id}.jsonl")
                append_jsonl(log_path, record)
            if self._pre_metrics:
                self._pre_metrics["uplink_bits_total"] = bits["uplink_bits_total"]
        except Exception as ex:
            log(INFO, f"[Warn] Logging failed: {ex}")

        return payload


class CSERClient(BaseClient):
    """CSER client: EF/top-k with periodic error-reset digest.
        - Every `H` rounds sends an additional top-`k1` digest of the current
            error residual (`reset_frac`), and remembers it as `last_e_sent` for
            next-round reconciliation against the server-broadcast `e_bar`.
        - On rounds where the server broadcasts a non-empty `e_bar` and we
            had sent a digest last time, we reconcile: e_t <- e_t + (last_e_sent - e_bar)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compressor = create_compressor(
            comp_type=self.config["comp_type"],
            sparsify_by=self.config["sparsify_by"],
        )
        self.node_id = self.context.node_id
        self.H = int(self.config.get("H", 5))
        self.reset_frac = float(self.config.get("reset_frac", 0.1))

    def _parse_e_bar(self, config: Dict, n: int) -> torch.Tensor:
        """Parse broadcast e_bar from FitIns.config, return dense tensor (may be all zero)."""
        try:
            idx_json = config.get("e_bar_idx", "[]")
            val_json = config.get("e_bar_val", "[]")
            idx_list = json.loads(idx_json) if idx_json else []
            val_list = json.loads(val_json) if val_json else []
        except Exception as ex:
            log(INFO, f"[CSER] failed to parse e_bar from config: {ex}")
            return torch.zeros(n, device=self.device, dtype=torch.float32)
        if not idx_list:
            return torch.zeros(n, device=self.device, dtype=torch.float32)
        dense = torch.zeros(n, device=self.device, dtype=torch.float32)
        idx_t = torch.tensor(idx_list, dtype=torch.long, device=self.device)
        val_t = torch.tensor(val_list, dtype=torch.float32, device=self.device)
        dense[idx_t] = val_t
        return dense

    def fit(self, parameters: NDArrays, config: Dict):
        set_weights(self.net, parameters)
        flat_global = self._flat_trainable()
        n = flat_global.numel()
        server_round = int(config["server_round"])

        # Gradient-mismatch probe ||g(w_r) - g(w_r + α·e_t)||² with the
        # client's current EF residual; α is the metric probe (default 1).
        pre_metrics: Dict = {}
        if self.config.get("track_metrics", False):
            try:
                e_for_metric = self._load_sparse("residual", n)
                _, pre_metrics = compute_pre_send_metrics(
                    model=self.net,
                    device=self.device,
                    valloader=self.valloader,
                    w_global=flat_global.clone(),
                    e_residual=e_for_metric,
                    alpha_preview=float(self.config.get("metric_alpha", 1.0)),
                    server_round=server_round,
                    config=self.config,
                )
            except Exception as ex:
                log(INFO, f"[CSER Warn] mismatch metric failed: {ex}")

        # CSER reconciliation: e_t <- e_t + (last_e_sent - e_bar).
        e_bar = self._parse_e_bar(config, n)
        has_e_bar = bool(torch.any(e_bar != 0.0).item())
        has_last_sent = (
            self.context.state.parameters_records is not None
            and "last_e_sent" in self.context.state.parameters_records
        )
        if has_e_bar and has_last_sent:
            e_prev = self._load_sparse("residual", n)
            sent = self._load_sparse("last_e_sent", n)
            e_prev = e_prev + (sent - e_bar)
            self._store_sparse("residual", e_prev)
            self._drop_state_key("last_e_sent")

        lr = cosine_eta_warmfloor(
            r=server_round,
            total_R=self.config["num_server_rounds"],
            eta_max=self.config["learning_rate_max"],
            eta_min=self.config["learning_rate_min"],
            warmup_R=self.config.get("lr_warmup_rounds", 10),
        )
        _run_local_training(
            net=self.net,
            trainloader=self.trainloader,
            device=self.device,
            config=self.config,
            learning_rate=lr,
        )

        flat_curr = self._flat_trainable()
        flat_delta = flat_curr - flat_global

        # CSER: u_t = Δ_t + m_t  (EF carry); send C(u_t); memory ← u_t - C(u_t).
        e_t = self._load_sparse("residual", n)
        u_t = flat_delta + e_t

        indices, values, _meta = self.compressor.compress(u_t.detach())
        sent_dense = torch.zeros_like(u_t)
        if indices.size > 0:
            idx_t = torch.from_numpy(indices).long().to(self.device)
            val_t = torch.from_numpy(values).to(self.device)
            sent_dense[idx_t] = val_t

        e_half = u_t - sent_dense

        reset_round = (server_round % self.H) == 0
        e_idx_np = None
        e_val_np = None
        if reset_round:
            k1 = max(1, int(self.reset_frac * n))
            _, top_idx = torch.topk(torch.abs(e_half), k1)
            top_val = e_half[top_idx].clone()
            # Remove the digest from e_half in place (no full-vector clone).
            e_half[top_idx] = 0.0

            # Remember what we sent for next-round reconciliation.
            sent_sparse = torch.zeros_like(e_half)
            sent_sparse[top_idx] = top_val
            self._store_sparse("last_e_sent", sent_sparse)

            e_idx_np = top_idx.detach().cpu().numpy().astype(np.int32)
            e_val_np = top_val.detach().cpu().numpy().astype(np.float32)

        self._store_sparse("residual", e_half)

        bn_mu, bn_var = self._collect_bn_stats()
        bn_mu_np = bn_mu.astype(np.float16) if bn_mu.size > 0 else np.array([], dtype=np.float16)
        bn_var_np = bn_var.astype(np.float16) if bn_var.size > 0 else np.array([], dtype=np.float16)
        count_np = np.array([len(self.trainloader.dataset)], dtype=np.float32)

        payload = [indices, values, bn_mu_np, bn_var_np, count_np]
        if e_idx_np is not None:
            payload += [e_idx_np, e_val_np]

        # Bandwidth logging (main channel only; e-packet is counted cheap).
        metrics: Dict = {}
        try:
            bits = compute_comm_bits(indices, values, (bn_mu_np, bn_var_np), count_np)
            metrics["uplink_bits_total"] = bits["uplink_bits_total"]
            if e_idx_np is not None and e_idx_np.size > 0:
                e_bits = (e_idx_np.nbytes + e_val_np.nbytes) / (1024**2)  # type: ignore
                metrics["uplink_bits_total"] += e_bits
        except Exception as ex:
            log(INFO, f"[CSER Warn] logging failed: {ex}")

        if pre_metrics:
            for k in ("grad_norm_sq", "residual_energy", "grad_mismatch_sq", "rho_r"):
                if pre_metrics.get(k) is not None:
                    metrics[k] = pre_metrics[k]

        return payload, len(self.trainloader.dataset), metrics


def _decode_c_global_b64(s: str, device: torch.device, length: int) -> torch.Tensor:
    buf = base64.b64decode(s.encode("ascii"))
    arr = torch.frombuffer(memoryview(buf), dtype=torch.float32)
    if arr.numel() != length:
        raise RuntimeError(f"c_global length mismatch: {arr.numel()} vs {length}")
    return arr.to(device)


class ScaffoldBase(BaseClient):
    """Shared bits for SCAFFOLD/SCALLION/SCAFCOM single-uplink."""
    KEY_C = "scaffold_c"
    KEY_V = "scafcom_v"

    def _compress_delta(self, delta: torch.Tensor):
        if "comp_type" in self.config and self.config["comp_type"] != "none":
            comp = create_compressor(
                self.config["comp_type"],
                self.config.get("sparsify_by", 0.01),
            )
            idx, val, _meta = comp.compress(delta.detach())
            return [idx, val]
        return [delta.detach().cpu().numpy()]


class SCAFFOLDClient(ScaffoldBase):
    def fit(self, parameters: NDArrays, config: Dict):
        set_weights_with_dtype_handling(self.net, parameters)
        x_t = self._flat_trainable()

        D = int(config["vec_len"])
        c_g = _decode_c_global_b64(config["c_global_b64"], self.device, D)
        eta_l = cosine_eta_warmfloor(
            r=config["server_round"],
            total_R=self.config["num_server_rounds"],
            eta_max=self.config["learning_rate_max"],
            eta_min=self.config["learning_rate_min"],
            warmup_R=self.config.get("lr_warmup_rounds", 10)
        )

        c_i = self._load_sparse(self.KEY_C, D)

        K = self.config["local_steps"]

        total_steps, total_samples, scaf_loss = train_scaffold(
            net=self.net,
            trainloader=self.trainloader,
            device=self.device,
            local_steps=K,
            learning_rate=eta_l,
            momentum=0.0,
            weight_decay=self.config.get("weight_decay", 0.0),
            c_local=c_i,
            c_global=c_g,
        )

        yK = self._flat_trainable()

        coef = 1.0 / (eta_l * float(total_steps))

        y_delta = yK - x_t

        c_plus = c_i - c_g + coef * (x_t - yK)
        c_delta = c_plus - c_i

        self._store_sparse(self.KEY_C, c_plus)

        bn_mu, bn_var = self._collect_bn_stats()
        count = np.array([float(total_samples)], dtype=np.float32)

        payload = [
            y_delta.detach().cpu().numpy().astype(np.float32),
            c_delta.detach().cpu().numpy().astype(np.float32),
            bn_mu, bn_var, count,
        ]
        
        return payload, len(self.trainloader.dataset), {}

class SCAFCOMClient(ScaffoldBase):

    def fit(self, parameters: NDArrays, config: Dict):
        set_weights_with_dtype_handling(self.net, parameters)
        x_t = self._flat_trainable()

        D = int(config["vec_len"])
        c_g = _decode_c_global_b64(config["c_global_b64"], self.device, D)

        # Force beta = 1 to recover SCAFFOLD (Ci = I comes from comp_type == "none")
        beta = 0.2
        eta_l = float(config["learning_rate"])
        c_i = self._load_sparse(self.KEY_C, D)
        v_i = self._load_sparse(self.KEY_V, D)

        K = self.config["local_steps"]

        total_steps, total_samples, scaf_loss = train_scaffold(
            net=self.net,
            trainloader=self.trainloader,
            device=self.device,
            local_steps=K,
            learning_rate=eta_l,
            momentum=0.0,
            weight_decay=self.config.get("weight_decay", 0.0),
            c_local=c_i,
            c_global=c_g,
        )

        # y_i^{t,K} after local training
        yK = self._flat_trainable()

        # IMPORTANT: use total local steps (S), not epochs (K)
        S = max(int(total_steps), 1)
        denom = eta_l * float(S) + 1e-12   # avoid div-by-zero

        # v_i^{t+1} = (1-β)v_i^t + β * ( (x^t - y_i^{t,K})/(η_l S) + c_i^t - c^t )
        v_new = (1.0 - beta) * v_i + beta * ((x_t - yK) / denom + c_i - c_g)
        self._store_sparse(self.KEY_V, v_new)

        # δ_i^t = v_i^{t+1} - c_i^t
        delta = v_new - c_i

        rnd = config["server_round"]
        if rnd <= 3:
            with torch.no_grad():
                print(
                    f"[CLT R{rnd}] S={S} "
                    f"||x_t||={x_t.norm().item():.3e} "
                    f"||yK||={yK.norm().item():.3e} "
                    f"||x_t-yK||={(x_t - yK).norm().item():.3e} "
                    f"||v_old||={v_i.norm().item():.3e} "
                    f"||v_new||={v_new.norm().item():.3e} "
                    f"||c_i||={c_i.norm().item():.3e} "
                    f"||c_g||={c_g.norm().item():.3e} "
                    f"||delta||={delta.norm().item():.3e} "
                    f"η_l={eta_l:.3e}"
                )

        if self.config["comp_type"] == "none":
            delta_tilde = delta
            payload = [delta_tilde.detach().cpu().numpy()]
        else:
            comp = create_compressor(
                self.config["comp_type"],
                self.config.get("sparsify_by", 0.01),
            )
            idx, val, _ = comp.compress(delta.detach())
            bn_mu, bn_var = self._collect_bn_stats()
            count = np.array([float(total_samples)], dtype=np.float32)
            payload = [idx, val, bn_mu, bn_var, count]

            delta_tilde = torch.zeros_like(delta)
            if idx.size > 0:
                idx_t = torch.from_numpy(idx).long().to(self.device)
                val_t = torch.from_numpy(val).to(self.device)
                delta_tilde[idx_t] = val_t

        # c_i^{t+1} = c_i^t + δ̃_i^t
        c_i_new = c_i + delta_tilde
        self._store_sparse(self.KEY_C, c_i_new)

        if rnd <= 3:
            print(
                f"[CLT R{rnd}] ||c_i_new||={c_i_new.norm().item():.3e} "
                f"(Δc_i = ||δ̃||={delta_tilde.norm().item():.3e})"
            )

        return payload, len(self.trainloader.dataset), {}

    
class SCALLIONClient(ScaffoldBase):
    def fit(self, parameters: NDArrays, config: Dict):
        set_weights_with_dtype_handling(self.net, parameters)
        x_t = self._flat_trainable()
        D = int(config["vec_len"])
        alpha = float(config.get("alpha_sc", 0.1))
        c_g = _decode_c_global_b64(config["c_global_b64"], self.device, D)
        eta_l = cosine_eta_warmfloor(
            r=config["server_round"],
            total_R=self.config["num_server_rounds"],
            eta_max=self.config["learning_rate_max"],
            eta_min=self.config["learning_rate_min"],
            warmup_R=self.config.get("lr_warmup_rounds", 10)
        )
        K = int(config["local_epochs"])
        c_i = self._load_sparse(self.KEY_C, D)
        K = self.config["local_steps"]

        total_steps, total_samples, scaf_loss = train_scaffold(
            net=self.net,
            trainloader=self.trainloader,
            device=self.device,
            local_steps=K,
            learning_rate=eta_l,
            momentum=0.0,
            weight_decay=self.config.get("weight_decay", 0.0),
            c_local=c_i,
            c_global=c_g,
        )

        yK = self._flat_trainable()

        # SCALLION delta_i = alpha*((x - yK)/(eta_l K) - c_g)
        delta = alpha * ((x_t - yK) / (eta_l * K) - c_g)
        if "comp_type" in self.config:
            comp = create_compressor(
                self.config["comp_type"],
                self.config.get("sparsify_by", 0.01),
            )
            idx, val, _ = comp.compress(delta.detach())
            bn_mu, bn_var = self._collect_bn_stats()
            count = np.array([float(total_samples)], dtype=np.float32)

            payload = [
                idx, val,
                bn_mu, bn_var, count,
            ]

            # reconstruct δ̃ locally to update c_i
            delta_tilde = torch.zeros_like(delta)
            if idx.size > 0:
                idx_t = torch.from_numpy(idx).long().to(self.device)
                val_t = torch.from_numpy(val).to(self.device)
                delta_tilde[idx_t] = val_t
        else:
            payload = [delta.detach().cpu().numpy()]
            delta_tilde = delta

        c_i_new = c_i + delta_tilde
        self._store_sparse(self.KEY_C, c_i_new)

        return payload, len(self.trainloader.dataset), {}



def client_fn(context: Context):
    """Create appropriate client based on configuration."""
    configs = context_to_easydict(context)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(configs.node_config.num_partitions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = instantiate_model(config=configs.run_config)  # type: ignore
    algorithm = configs.run_config.algorithm.name # type: ignore
    
    client_classes = {
        "fedavg": FedProxClient,
        "fedprox": FedProxClient,
        "sapef": SAPEFClient,
        "saef": SAPEFClient,
        "ef": SAPEFClient,
        "cser": CSERClient,
        "scaffold": SCAFFOLDClient,
        "scallion": SCALLIONClient,
        "scafcom": SCAFCOMClient,
    }

    if configs.run_config["dataset"]["name"] == "mnist_old":
        trainloader, valloader = load_data(
            dataset_config=configs.run_config["dataset"],  # type: ignore
            partition_id=partition_id,
            num_partitions=num_partitions,
        )
    elif configs.run_config["dataset"]["name"] in (
        "cifar10", "cifar100", "mnist", "fashion_mnist", "fashion-mnist", "femnist"
    ):
        trainloader, valloader = load_partition_data(partition_id, configs.run_config) # type: ignore
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    ClientClass = client_classes.get(algorithm)
    if ClientClass is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return ClientClass(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        context=context,
        config=configs.run_config # type: ignore
    ).to_client()

app = ClientApp(client_fn=client_fn)
