#!/usr/bin/env python3
"""
Spherical operations for area-weighted integration on the sphere.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import torch


@dataclass
class SphereOps:
    """
    Spherical integration utilities for training losses.
    Provides area-weighted integration using cos(latitude) weights.
    """

    lat: Union[np.ndarray, torch.Tensor]
    lon: Union[np.ndarray, torch.Tensor]

    def __post_init__(self):
        """Precompute integration weights."""
        # Convert to numpy arrays
        self.lat = np.asarray(self.lat, dtype=np.float64).reshape(-1)
        self.lon = np.asarray(self.lon, dtype=np.float64).reshape(-1)

        # Compute cos(latitude) weights
        lat_rad = np.deg2rad(self.lat)
        w_lat = np.cos(lat_rad)
        w_lat = np.clip(w_lat, 0.0, None)  # Avoid negative weights

        # Create 2D weight array [H, W]
        self._w2d_np = w_lat[:, None] * np.ones((1, self.lon.shape[0]), dtype=np.float64)

        # Cache for device/dtype combinations
        self._w_cache = {}

    @property
    def nlat(self) -> int:
        """Number of latitude points."""
        return len(self.lat)

    @property
    def nlon(self) -> int:
        """Number of longitude points."""
        return len(self.lon)

    def _get_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get cached weights for specific device and dtype."""
        key = (device, dtype)
        if key not in self._w_cache:
            self._w_cache[key] = torch.from_numpy(self._w2d_np).to(device=device, dtype=dtype)
        return self._w_cache[key]

    def integrate_grid(self, x: torch.Tensor, dimensionless: bool = True) -> torch.Tensor:
        """
        Compute area-weighted integral over the sphere.

        Args:
            x: Tensor of shape [H,W], [C,H,W], or [B,C,H,W]
            dimensionless: If True, normalize weights to sum to 1 (weighted mean)
                          If False, use unnormalized weights (weighted sum)

        Returns:
            Integrated result with spatial dimensions reduced:
            - [H,W] -> scalar
            - [C,H,W] -> [C]
            - [B,C,H,W] -> [B,C]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        if x.dim() not in (2, 3, 4):
            raise ValueError(f"Expected 2D/3D/4D tensor, got {x.dim()}D")

        # Check spatial dimensions
        H, W = x.shape[-2], x.shape[-1]
        if H != self.nlat or W != self.nlon:
            raise ValueError(f"Shape mismatch: expected ({self.nlat},{self.nlon}), got ({H},{W})")

        # Get weights in fp32 for stable reduction
        weights = self._get_weights(x.device, torch.float32)

        # Normalize weights if requested
        if dimensionless:
            weights = weights / weights.sum()

        # Cast input to fp32 for stable computation
        x_fp32 = x.to(dtype=torch.float32)

        # Compute weighted integral
        if x.dim() == 2:
            # [H,W] -> scalar
            result = (x_fp32 * weights).sum()
        elif x.dim() == 3:
            # [C,H,W] -> [C]
            result = (x_fp32 * weights).sum(dim=(-2, -1))
        else:
            # [B,C,H,W] -> [B,C]
            result = (x_fp32 * weights).sum(dim=(-2, -1))

        # Return in original dtype if it was lower precision
        if x.dtype in [torch.float16, torch.bfloat16]:
            result = result.to(dtype=x.dtype)

        return result

    def sht(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spherical harmonic transform (placeholder).
        Implement this if you need spectral losses by connecting to torch-harmonics.
        """
        raise NotImplementedError(
            "Spherical harmonic transform not implemented. "
            "Use l2 loss or wire torch-harmonics SHT for spectral losses."
        )