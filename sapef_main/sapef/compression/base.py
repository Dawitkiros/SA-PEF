"""Base compressor interface."""
from abc import ABC, abstractmethod
from typing import Tuple, Dict
import torch
import numpy as np


class Compressor(ABC):
    """Base class for gradient compression algorithms."""

    @abstractmethod
    def compress(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Returns (indices: int32, values: float32, metadata)."""
        pass

    def decompress(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Scatter sparse (idx, val) back into a dense float32 tensor of length `size`."""
        dense = torch.zeros(size, device=device, dtype=torch.float32)
        if indices.size == 0:
            return dense
        if indices.ndim == 0:
            indices = indices[None]
            values = values[None]
        idx_t = torch.from_numpy(indices).long().to(device)
        val_t = torch.from_numpy(values).to(device=device, dtype=torch.float32)
        dense[idx_t] = val_t
        return dense

    @staticmethod
    def _pack(
        indices: torch.Tensor,
        values: torch.Tensor,
        metadata: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        return (
            indices.detach().cpu().numpy().astype(np.int32),
            values.detach().cpu().numpy().astype(np.float32),
            metadata,
        )

    @staticmethod
    def _empty(metadata: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            metadata,
        )

    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        return original_size / max(1, compressed_size)