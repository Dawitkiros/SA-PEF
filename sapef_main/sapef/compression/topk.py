"""Top-K compressor."""
import torch
import numpy as np
from typing import Tuple, Dict

from .base import Compressor


class TopKCompressor(Compressor):
    """Top-K sparsification by magnitude."""

    def __init__(self, sparsity: float):
        self.sparsity = sparsity

    def compress(
        self,
        tensor: torch.Tensor,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        k = max(1, int(self.sparsity * tensor.numel()))
        _, indices = torch.topk(torch.abs(tensor), k)
        values = tensor[indices]
        metadata = {
            "k": k,
            "original_size": tensor.numel(),
            "compression_ratio": tensor.numel() / k,
        }
        return self._pack(indices, values, metadata)
