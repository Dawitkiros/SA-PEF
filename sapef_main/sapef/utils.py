import math
from collections import defaultdict

from easydict import EasyDict


def cosine_eta_warmfloor(
    r: int,
    total_R: int,
    eta_max: float = 0.01,
    eta_min: float = 0.003,
    warmup_R: int = 10,
) -> float:
    """Cosine learning-rate schedule with linear warm-up."""
    if r <= warmup_R:
        return eta_max * (r / max(1, warmup_R))
    t = (r - warmup_R) / max(1, total_R - warmup_R)
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t))


def compute_delta_from_sparsity(sparsity: float) -> float:
    """Compression-bias parameter δ = 1 / keep-fraction for top-k / rand-k."""
    keep = max(1e-6, float(sparsity))
    return 1.0 / keep


def context_to_easydict(context):
    """Parser to generate internal config files once you are given are context.

    This is to facilitate easier sharing and better config management rather than using
    a dict.
    """
    configs_to_parse = {
        "run_config": _extract_run_configs_per_type(context.run_config),
        "node_config": _extract_run_configs_per_type(context.node_config),
    }
    return EasyDict(configs_to_parse)


def _extract_run_configs_per_type(config):
    parsed_configs = defaultdict(defaultdict)

    for key, value in config.items():
        if "." in key:
            category, name = key.split(".")
            parsed_configs[category][name.replace("-", "_")] = value
        else:
            parsed_configs[key.replace("-", "_")] = value
    return parsed_configs
