#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


Tensor = torch.Tensor


@dataclass
class SphereOps:
    """
    Minimal spherical utilities needed by training losses.

    - integrate_grid(x, dimensionless=True):
        Area-weighted reduction using cos(latitude) weights.
        Shapes:
          x: [H,W] -> scalar
             [C,H,W] -> [C]
             [B,C,H,W] -> [B,C]

        If dimensionless=True, returns an area-weighted *mean* (weights normalized to 1).
        If dimensionless=False, returns an area-weighted *sum* (i.e., unnormalized integral).
        (Your training uses dimensionless=True, matching the reference code.)

    - sht(x): placeholder; raise unless you wire torch-harmonics SHT.
    """
    lat: Union[np.ndarray, Tensor]
    lon: Union[np.ndarray, Tensor]

    def __post_init__(self):
        # Ensure 1D CPU numpy arrays
        self.lat = np.asarray(self.lat, dtype=np.float64).reshape(-1)
        self.lon = np.asarray(self.lon, dtype=np.float64).reshape(-1)

        # Precompute 2D latitude weights (cos(lat)) in numpy; convert per-call to tensor dtype/device
        lat_rad = np.deg2rad(self.lat)  # lat in degrees -> radians
        w_lat = np.cos(lat_rad)         # shape [H]
        w_lat = np.clip(w_lat, 0.0, None)  # avoid tiny negatives due to FP
        self._w2d_np = w_lat[:, None] * np.ones((1, self.lon.shape[0]), dtype=np.float64)  # [H,W]

        # Lazy device/dtype cache
        self._w_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}

    # --------------------------- helpers ---------------------------

    @property
    def nlat(self) -> int:
        return int(self.lat.shape[0])

    @property
    def nlon(self) -> int:
        return int(self.lon.shape[0])

    def _weights(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (device, dtype)
        wt = self._w_cache.get(key)
        if wt is None:
            wt = torch.from_numpy(self._w2d_np).to(device=device, dtype=dtype)  # [H,W]
            self._w_cache[key] = wt
        return wt

    # --------------------------- public API ---------------------------

    def integrate_grid(self, x: Tensor, dimensionless: bool = True) -> Tensor:
        """
        Area-weighted reduction across spatial dims with cos(lat) weights.

        x: [H,W], [C,H,W], or [B,C,H,W]
        Returns: scalar, [C], or [B,C], respectively.

        dimensionless=True  -> area-weighted *mean* (weights normalized to 1)
        dimensionless=False -> area-weighted *sum*  (unnormalized integral)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() not in (2, 3, 4):
            raise ValueError(f"integrate_grid expects 2D/3D/4D tensor, got shape {tuple(x.shape)}")

        H = x.shape[-2]
        W = x.shape[-1]
        if H != self.nlat or W != self.nlon:
            raise ValueError(f"Spatial shape mismatch: got HxW={H}x{W}, expected {self.nlat}x{self.nlon}")

        wt = self._weights(x.device, x.dtype)  # [H,W]

        if dimensionless:
            wt = wt / torch.clamp(wt.sum(), min=torch.finfo(wt.dtype).eps)  # normalize to sum=1

        if x.dim() == 2:
            # [H,W] -> scalar
            return (x * wt).sum()

        if x.dim() == 3:
            # [C,H,W] -> [C]
            return (x * wt).sum(dim=(-2, -1))

        # [B,C,H,W] -> [B,C]
        return (x * wt).sum(dim=(-2, -1))

    def sht(self, x: Tensor) -> Tensor:
        """
        Placeholder: spherical harmonic transform of x.
        Wire this to torch-harmonics if you want spectral loss:
          - Accept shapes [B,C,H,W] or [C,H,W], return complex coeffs.
        """
        raise NotImplementedError("SphereOps.sht is not implemented. Use l2 loss or wire torch-harmonics SHT.")
