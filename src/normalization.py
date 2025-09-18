#!/usr/bin/env python3
"""
Per-variable normalization with configurable precision.
Statistics are always computed in fp64 for stability.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
import json
import math
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class Normalizer:
    """
    Normalizes data with per-variable transformations.

    Supported modes:
    - "identity": No transformation
    - "zscore": Standard normalization
    - "log_zscore": Log transform then normalize
    - "log1p_zscore": Log(1 + x/scale) then normalize
    - "slog_zscore": Signed log transform then normalize
    """

    mode_default: str = "identity"
    mode_by_var: Dict[str, str] = field(default_factory=dict)
    stats_path: Optional[Path] = None

    # Transform parameters
    log_eps: float = 1e-6
    log1p_scale: float = 1.0
    slog1p_scale: float = 1.0
    min_std: float = 1e-3

    # Dataset layout
    var_order: Optional[List[str]] = None
    channels_per_var: Optional[Dict[str, int]] = None
    var_slices: Optional[Dict[str, Tuple[int, int]]] = None
    C_total: Optional[int] = None

    # Learned statistics
    stats: Optional[Dict[str, Dict[str, List[float]]]] = None

    def _effective_mode(self, var: str) -> str:
        """Get the normalization mode for a variable."""
        return self.mode_by_var.get(var, self.mode_default)

    def configure(self, var_order: List[str], channels_per_var: Dict[str, int]) -> None:
        """Configure the normalizer with variable layout."""
        self.var_order = list(var_order)
        self.channels_per_var = dict(channels_per_var)
        self.var_slices = {}

        c = 0
        for v in self.var_order:
            n = self.channels_per_var[v]
            self.var_slices[v] = (c, c + n)
            c += n
        self.C_total = c

    def is_ready(self) -> bool:
        """Check if normalizer has valid statistics."""
        if not all([self.stats, self.var_order, self.channels_per_var]):
            return False

        try:
            for v in self.var_order:
                st = self.stats[v]
                n = self.channels_per_var[v]
                if len(st["mean"]) != n or len(st["std"]) != n:
                    return False
            return True
        except Exception:
            return False

    def load(self) -> bool:
        """Load statistics from disk if compatible."""
        if not self.stats_path or not self.stats_path.exists():
            return False

        try:
            payload = json.loads(self.stats_path.read_text())
        except Exception as e:
            logger.warning(f"Failed to load stats: {e}")
            return False

        # Verify compatibility
        if payload.get("schema_version") != 2:
            return False
        if payload.get("var_order") != self.var_order:
            return False
        if payload.get("channels_per_var") != self.channels_per_var:
            return False
        if payload.get("mode_default") != self.mode_default:
            return False
        if payload.get("mode_by_var") != self.mode_by_var:
            return False

        stats = payload.get("stats")
        if not stats:
            return False

        # Check for valid statistics
        for v, st in stats.items():
            if any(s is None or float(s) < self.min_std for s in st.get("std", [])):
                return False

        self.stats = stats
        self.log_eps = float(payload.get("log_eps", self.log_eps))
        self.log1p_scale = float(payload.get("log1p_scale", self.log1p_scale))
        self.slog1p_scale = float(payload.get("slog1p_scale", self.slog1p_scale))

        logger.info("Loaded normalization statistics")
        return True

    def save(self) -> None:
        """Save statistics to disk."""
        if not self.stats_path:
            return

        payload = {
            "schema_version": 2,
            "mode_default": self.mode_default,
            "mode_by_var": self.mode_by_var,
            "log_eps": self.log_eps,
            "log1p_scale": self.log1p_scale,
            "slog1p_scale": self.slog1p_scale,
            "var_order": self.var_order,
            "channels_per_var": self.channels_per_var,
            "C_total": self.C_total,
            "stats": self.stats,
        }

        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_path.write_text(json.dumps(payload))
        logger.info("Saved normalization statistics")

    def fit_from_iterator(self, it: Iterable[torch.Tensor], max_batches: Optional[int] = None) -> None:
        """
        Compute per-channel mean and std from data.
        Always uses fp64 for numerical precision.
        """
        if not self.var_order or not self.var_slices:
            raise RuntimeError("Normalizer not configured")

        C = self.C_total

        # Accumulators in fp64
        sum_ = torch.zeros(C, dtype=torch.float64)
        sumsq = torch.zeros(C, dtype=torch.float64)
        count = torch.zeros(C, dtype=torch.float64)

        batches = 0
        for x in it:
            if max_batches and batches >= max_batches:
                break

            if x.shape[0] != C:
                raise ValueError(f"Expected {C} channels, got {x.shape[0]}")

            # Transform each variable to its normalization space
            for v in self.var_order:
                a, b = self.var_slices[v]
                mode = self._effective_mode(v)
                xv = x[a:b].to(dtype=torch.float64)  # [Cv, H, W]

                # Apply transform
                if mode == "identity":
                    tv = xv
                elif mode == "zscore":
                    tv = xv
                elif mode == "log_zscore":
                    tv = torch.log(torch.clamp(xv, min=self.log_eps))
                elif mode == "log1p_zscore":
                    tv = torch.log1p(torch.clamp(xv, min=0.0) / self.log1p_scale)
                elif mode == "slog_zscore":
                    tv = torch.sign(xv) * torch.log1p(torch.abs(xv) / self.slog1p_scale)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Accumulate statistics
                n = tv.shape[1] * tv.shape[2]
                sum_[a:b] += tv.sum(dim=(1, 2))
                sumsq[a:b] += (tv * tv).sum(dim=(1, 2))
                count[a:b] += n

            batches += 1

        # Compute final statistics
        mean = (sum_ / torch.clamp(count, min=1.0)).tolist()
        var = (sumsq / torch.clamp(count, min=1.0) - torch.tensor(mean) ** 2)
        var = torch.clamp(var, min=(self.min_std ** 2))
        std = [max(self.min_std, math.sqrt(v)) for v in var.tolist()]

        # Store per-variable statistics
        self.stats = {}
        for v in self.var_order:
            a, b = self.var_slices[v]
            self.stats[v] = {
                "mean": mean[a:b],
                "std": std[a:b],
            }

    def transform(self, x: torch.Tensor, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Apply normalization to tensor.
        Computation done in fp32 if output is fp16/bf16 for stability.
        """
        if output_dtype is None:
            output_dtype = x.dtype

        if not self.var_slices:
            raise RuntimeError("Normalizer not configured")

        # Check if we need statistics
        needs_stats = any(self._effective_mode(v) != "identity" for v in self.var_order)
        if needs_stats and not self.stats:
            raise RuntimeError("Statistics not available")

        # Handle batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        # Use fp32 for computation if output is half precision
        compute_dtype = torch.float32 if output_dtype in [torch.float16, torch.bfloat16] else output_dtype
        out = x.to(dtype=compute_dtype).clone()

        # Apply per-variable normalization
        for v in self.var_order:
            a, b = self.var_slices[v]
            mode = self._effective_mode(v)

            if mode == "identity":
                continue

            # Get statistics
            mu = torch.tensor(self.stats[v]["mean"], dtype=compute_dtype, device=out.device)
            sd = torch.tensor(self.stats[v]["std"], dtype=compute_dtype, device=out.device)
            mu = mu.view(1, -1, 1, 1)
            sd = sd.view(1, -1, 1, 1)
            sd = torch.clamp(sd, min=self.min_std)

            # Apply transformation
            if mode == "zscore":
                out[:, a:b] = (out[:, a:b] - mu) / sd
            elif mode == "log_zscore":
                out[:, a:b] = (torch.log(torch.clamp(out[:, a:b], min=self.log_eps)) - mu) / sd
            elif mode == "log1p_zscore":
                t = torch.log1p(torch.clamp(out[:, a:b], min=0.0) / self.log1p_scale)
                out[:, a:b] = (t - mu) / sd
            elif mode == "slog_zscore":
                t = torch.sign(out[:, a:b]) * torch.log1p(torch.abs(out[:, a:b]) / self.slog1p_scale)
                out[:, a:b] = (t - mu) / sd

        # Convert to output dtype
        out = out.to(dtype=output_dtype)

        return out.squeeze(0) if squeezed else out

    def inverse_transform(self, x: torch.Tensor, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Invert normalization.
        Computation done in fp32 if needed for stability.
        """
        if output_dtype is None:
            output_dtype = x.dtype

        if not self.var_slices or not self.stats:
            return x.to(dtype=output_dtype)

        # Handle batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        # Use fp32 for computation if needed
        compute_dtype = torch.float32 if output_dtype in [torch.float16, torch.bfloat16] else output_dtype
        out = x.to(dtype=compute_dtype).clone()

        # Invert per-variable normalization
        for v in self.var_order:
            a, b = self.var_slices[v]
            mode = self._effective_mode(v)

            if mode == "identity":
                continue

            # Get statistics
            mu = torch.tensor(self.stats[v]["mean"], dtype=compute_dtype, device=out.device)
            sd = torch.tensor(self.stats[v]["std"], dtype=compute_dtype, device=out.device)
            mu = mu.view(1, -1, 1, 1)
            sd = sd.view(1, -1, 1, 1)

            # Invert transformation
            y = out[:, a:b] * sd + mu

            if mode == "zscore":
                out[:, a:b] = y
            elif mode == "log_zscore":
                out[:, a:b] = torch.exp(y)
            elif mode == "log1p_zscore":
                out[:, a:b] = torch.expm1(y) * self.log1p_scale
            elif mode == "slog_zscore":
                out[:, a:b] = torch.sign(y) * torch.expm1(torch.abs(y)) * self.slog1p_scale

        # Convert to output dtype
        out = out.to(dtype=output_dtype)

        return out.squeeze(0) if squeezed else out