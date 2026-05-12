"""Custom FedAvg strategy which drops straggler clients."""

from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgWithStragglerDrop(FedAvg):
    """Custom FedAvg which discards updates from stragglers."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ):
        """Discard all the models sent by the clients that were stragglers."""
        stragglers_mask = [res.metrics["is_straggler"] for _, res in results]
        results = [res for i, res in enumerate(results) if not stragglers_mask[i]]

        parameters, metrics = super().aggregate_fit(server_round, results, failures)

        uplink_vals = [
            (res.num_examples, float(res.metrics.get("uplink_bits_total", 0.0)))
            for _, res in results
            if "uplink_bits_total" in res.metrics
        ]
        if uplink_vals:
            total_n = sum(n for n, _ in uplink_vals)
            metrics = dict(metrics or {})
            metrics["uplink_bits_total"] = (
                sum(n * v for n, v in uplink_vals) / total_n if total_n > 0 else 0.0
            )

        return parameters, metrics
