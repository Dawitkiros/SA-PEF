"""Random-K compressor."""
import torch
import numpy as np
from typing import Tuple, Dict

from .base import Compressor


class RandKCompressor(Compressor):
    """Random-K sparsification."""

    def __init__(self, sparsity: float, seed: int = 0):
        self.sparsity = sparsity
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def compress(
        self,
        tensor: torch.Tensor,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        k = max(1, int(self.sparsity * tensor.numel()))
        perm = torch.randperm(tensor.numel(), device=tensor.device)
        indices = perm[:k]
        values = tensor[indices]
        metadata = {
            "k": k,
            "original_size": tensor.numel(),
            "compression_ratio": tensor.numel() / k,
        }
        return self._pack(indices, values, metadata)
