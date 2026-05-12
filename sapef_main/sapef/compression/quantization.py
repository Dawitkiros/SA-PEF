"""Deterministic uniform quantizer (biased)."""
import torch
import numpy as np
from typing import Tuple, Dict, Any

from .base import Compressor


class DeterministicQuantizer(Compressor):
    """Biased deterministic uniform quantizer.

    Normalize by ||·||_inf to map values into [-1, 1], deterministically round
    to `n_levels` equally spaced values, then dequantize back to float32 client-
    side. The server just sees Q(x) in float32.
    """

    def __init__(self, n_levels: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        self.n_levels = max(2, n_levels)
        self.eps = eps

    def compress(
        self,
        tensor: torch.Tensor,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        # Work on a flattened view; caller can reshape using metadata["shape"].
        original_shape = tuple(tensor.shape)
        flat = tensor.detach().view(-1)

        scale = flat.abs().max()
        if scale < self.eps:
            return self._empty({
                "scale": 0.0,
                "original_size": flat.numel(),
                "compressed_size": 0,
                "n_levels": self.n_levels,
                "shape": original_shape,
            })

        # Deterministic (biased) quantization to levels {-1, -1+2/(L-1), ..., 1}.
        s = self.n_levels - 1
        flat_quantized = torch.round(flat / scale * s) / s
        flat_reconstructed = flat_quantized * scale

        nonzero_mask = flat_reconstructed != 0
        indices = nonzero_mask.nonzero(as_tuple=False).view(-1)
        values = flat_reconstructed[nonzero_mask]

        metadata = {
            "scale": scale.item(),
            "original_size": flat.numel(),
            "compressed_size": values.numel(),
            "n_levels": self.n_levels,
            "shape": original_shape,
        }
        return self._pack(indices, values, metadata)
