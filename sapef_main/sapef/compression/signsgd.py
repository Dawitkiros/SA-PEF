"""Scaled-Sign compressor: C(x) = (||x||_1 / d) * sign(x)."""
import torch
import numpy as np
from typing import Tuple, Dict, Any

from .base import Compressor


class ScaledSignCompressor(Compressor):
    """Biased dense scaled-sign operator (M=1 group sign from Fed-EF)."""

    def __init__(self, sparsity: float = 1.0) -> None:
        # `sparsity` ignored — kept for factory compatibility.
        super().__init__()
        self.sparsity = sparsity

    def compress(
        self,
        tensor: torch.Tensor,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        original_shape = tuple(tensor.shape)
        flat = tensor.detach().view(-1)
        d = flat.numel()

        if d == 0:
            return self._empty({
                "scale": 0.0,
                "original_size": 0,
                "compressed_size": 0,
                "shape": original_shape,
                "compressor": "scaled_sign",
            })

        scale = flat.abs().sum() / d
        comp = torch.sign(flat) * scale
        idx = torch.arange(d, device=flat.device, dtype=torch.long)
        metadata = {
            "scale": scale.item(),
            "original_size": d,
            "compressed_size": d,
            "shape": original_shape,
            "compressor": "scaled_sign",
        }
        return self._pack(idx, comp, metadata)
