"""Compressor registry."""
from typing import Dict, Type

from .base import Compressor
from .topk import TopKCompressor
from .randk import RandKCompressor
from .quantization import DeterministicQuantizer
from .signsgd import ScaledSignCompressor


COMPRESSOR_REGISTRY: Dict[str, Type[Compressor]] = {
    "topk": TopKCompressor,
    "randk": RandKCompressor,
    "quantizer": DeterministicQuantizer,
    "scaled_sign": ScaledSignCompressor,
}

def create_compressor(comp_type: str, sparsify_by: float, **kwargs) -> Compressor: # type: ignore
    if comp_type not in COMPRESSOR_REGISTRY:
        raise ValueError(
            f"Unknown compressor: {comp_type}. "
            f"Available: {list(COMPRESSOR_REGISTRY.keys())}"
        )

    compressor_class = COMPRESSOR_REGISTRY[comp_type]

    if comp_type in ("topk", "randk", "scaled_sign"):
        return compressor_class(sparsity=sparsify_by, **kwargs)  # type: ignore
    elif comp_type == "quantizer":
        return compressor_class(**kwargs)
    else:
        raise ValueError(f"Unsupported compressor type: {comp_type}")
