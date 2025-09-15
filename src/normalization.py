#!/usr/bin/env python3
"""
Normalization utilities with per-variable control.

Supported modes
---------------
- "identity":         no-op
- "zscore":           per-channel mean/std in linear space
- "log_zscore":       log(clamp(x, log_eps)) -> z-score     (strictly positive)
- "log1p_zscore":     log1p(x / log1p_scale) -> z-score     (>= 0; avoids hard floor)
- "slog_zscore":      sign(x) * log1p(|x| / slog1p_scale) -> z-score  (signed)

This module:
- Stores per-variable channel layout (var_order, channels_per_var, var_slices).
- Fits mean/std in the *transform space* of each mode.
- Saves/loads stats JSON (schema_version=2) but REFUSES mismatches:
  * var order/shape mismatch
  * mode_default/mode_by_var mismatch
  * any channel std < min_std (forces refit to avoid 1e6 z-scores)

Key safety knobs:
- min_std: lower bound on per-channel std when applying normalization
- log_eps: floor for legacy "log_zscore"
- log1p_scale / slog1p_scale: scale factors for log1p transforms
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
    # configuration
    mode_default: str = "identity"
    mode_by_var: Dict[str, str] = field(default_factory=dict)
    stats_path: Optional[Path] = None

    # transform parameters
    log_eps: float = 1e-6
    log1p_scale: float = 1.0
    slog1p_scale: float = 1.0
    min_std: float = 1e-3  # <<< critical: prevents divide-by-tiny

    # dataset layout (set via configure())
    var_order: Optional[List[str]] = None
    channels_per_var: Optional[Dict[str, int]] = None
    var_slices: Optional[Dict[str, Tuple[int, int]]] = None
    C_total: Optional[int] = None

    # learned stats
    stats: Optional[Dict[str, Dict[str, List[float]]]] = None

    # ----------------------------- helpers -----------------------------
    def _effective_mode(self, var: str) -> str:
        return self.mode_by_var.get(var, self.mode_default)

    def configure(self, var_order: List[str], channels_per_var: Dict[str, int]) -> None:
        """Set the stacking layout and compute channel slices per variable."""
        self.var_order = list(var_order)
        self.channels_per_var = dict(channels_per_var)
        self.var_slices = {}
        c = 0
        for v in self.var_order:
            n = int(self.channels_per_var[v])
            self.var_slices[v] = (c, c + n)
            c += n
        self.C_total = c
        logger.debug("Normalizer configured: C_total=%s, var_slices=%s", self.C_total, self.var_slices)

    def is_ready(self) -> bool:
        """Return True if stats are present and match layout."""
        if self.stats is None or self.var_order is None or self.channels_per_var is None:
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

    # ----------------------------- IO -----------------------------
    def load(self) -> bool:
        """Load stats from stats_path if compatible with current layout and modes."""
        if not self.stats_path:
            return False
        sp = Path(self.stats_path)
        if not sp.exists():
            return False
        try:
            payload = json.loads(sp.read_text())
        except Exception as e:
            logger.warning("Failed to parse stats at %s: %s", sp, e)
            return False

        if payload.get("schema_version") != 2:
            logger.warning("Stats schema mismatch (expected v2).")
            return False

        if self.var_order is None or self.channels_per_var is None or self.C_total is None:
            logger.warning("Normalizer layout not configured; cannot validate loaded stats.")
            return False

        # Strict layout checks
        if payload.get("var_order") != self.var_order:
            logger.warning("Variable order mismatch in stats; will refit.")
            return False
        if payload.get("channels_per_var") != self.channels_per_var:
            logger.warning("channels_per_var mismatch in stats; will refit.")
            return False
        if int(payload.get("C_total", -1)) != int(self.C_total):
            logger.warning("C_total mismatch in stats; will refit.")
            return False

        # Refuse mode mismatches (prevents using zscore stats with slog/log1p transforms)
        if payload.get("mode_default") != self.mode_default:
            logger.warning("mode_default mismatch in stats; will refit.")
            return False
        if payload.get("mode_by_var") != self.mode_by_var:
            logger.warning("mode_by_var mismatch in stats; will refit.")
            return False

        stats = payload.get("stats")
        if not isinstance(stats, dict):
            logger.warning("Stats missing; will refit.")
            return False

        # Refuse channels with microscopic std (forces refit)
        for v, st in stats.items():
            if any((s is None) or (float(s) < self.min_std) for s in st.get("std", [])):
                logger.warning("Stats contain std < min_std for var '%s'; will refit.", v)
                return False

        # Accept; adopt transform params from file for reproducibility
        self.stats = stats
        self.log_eps = float(payload.get("log_eps", self.log_eps))
        self.log1p_scale = float(payload.get("log1p_scale", self.log1p_scale))
        self.slog1p_scale = float(payload.get("slog1p_scale", self.slog1p_scale))
        logger.info("Loaded normalization stats from %s", sp)
        return True

    def save(self) -> None:
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
        sp = Path(self.stats_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(payload))
        logger.info("Saved normalization stats to %s", sp)

    # ----------------------------- fitting -----------------------------
    def fit_from_iterator(self, it: Iterable[torch.Tensor], max_batches: Optional[int] = None) -> None:
        """
        Fit per-channel mean/std according to each variable's mode.
        The iterator should yield tensors shaped [C,H,W] on CPU.
        """
        assert self.var_order is not None and self.channels_per_var is not None and self.var_slices is not None
        C = int(self.C_total or 0)
        if C <= 0:
            raise RuntimeError("Normalizer not configured (unknown total channels).")

        # accumulators
        sum_ = torch.zeros(C, dtype=torch.float64)
        sumsq = torch.zeros(C, dtype=torch.float64)
        count = torch.zeros(C, dtype=torch.float64)

        batches = 0
        for x in it:
            if max_batches is not None and batches >= max_batches:
                break
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if x.dim() != 3 or x.shape[0] != C:
                raise ValueError(f"Expected [C,H,W] tensor with C={C}, got {tuple(x.shape)}")

            # transform to the space in which we compute mean/std
            for v in self.var_order:
                a, b = self.var_slices[v]
                mode = self._effective_mode(v)
                xv = x[a:b].to(dtype=torch.float64)  # [Cv,H,W]

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
                    raise ValueError(f"Unknown normalization mode '{mode}'")

                # accumulate
                H, W = tv.shape[-2], tv.shape[-1]
                n = H * W
                sum_[a:b]   += tv.sum(dim=(1, 2))
                sumsq[a:b]  += (tv * tv).sum(dim=(1, 2))
                count[a:b]  += n
            batches += 1

        # finalize
        mean = (sum_ / torch.clamp(count, min=1.0)).tolist()
        var = (sumsq / torch.clamp(count, min=1.0) - torch.tensor(mean) ** 2)
        var = torch.clamp(var, min=(self.min_std ** 2)).tolist()
        std = [max(self.min_std, math.sqrt(float(v))) for v in var]

        # stash per variable subset
        stats: Dict[str, Dict[str, List[float]]] = {}
        for v in self.var_order:
            a, b = self.var_slices[v]
            stats[v] = {
                "mean": [float(m) for m in mean[a:b]],
                "std":  [float(s) for s in std[a:b]],
            }
        self.stats = stats
        logger.info("Fitted normalization stats for %d variables (%d channels).", len(self.var_order), C)

    # ----------------------------- apply -----------------------------
    def _get_mu_sd(self, v: str, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.stats is not None and v in self.stats
        mu = torch.tensor(self.stats[v]["mean"], dtype=dtype, device=device).view(1, -1, 1, 1)
        sd = torch.tensor(self.stats[v]["std"],  dtype=dtype, device=device).view(1, -1, 1, 1)
        sd = torch.clamp(sd, min=self.min_std)
        return mu, sd

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to x shaped [C,H,W] or [B,C,H,W].
        """
        if self.var_slices is None:
            raise RuntimeError("Normalizer not configured.")
        if self.stats is None:
            if any(self._effective_mode(v) in ("zscore", "log_zscore", "log1p_zscore", "slog_zscore")
                   for v in (self.var_order or [])):
                raise RuntimeError("Normalization stats not available. Call fit_from_iterator() or load().")

        if x.dim() == 3:
            x = x.unsqueeze(0)  # [1,C,H,W]
            squeezed = True
        elif x.dim() == 4:
            squeezed = False
        else:
            raise ValueError(f"Unexpected rank {x.dim()} for normalization.")

        out = x.clone()

        for v in self.var_order or []:
            a, b = self.var_slices[v]
            mode = self._effective_mode(v)
            if mode == "identity":
                continue
            mu, sd = self._get_mu_sd(v, out.dtype, out.device)

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
            else:
                raise ValueError(f"Unknown normalization mode '{mode}'")

        return out.squeeze(0) if squeezed else out

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Invert normalization on x shaped [C,H,W] or [B,C,H,W].
        """
        if self.var_slices is None:
            raise RuntimeError("Normalizer not configured.")
        if self.stats is None:
            return x  # nothing to invert

        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeezed = True
        elif x.dim() == 4:
            squeezed = False
        else:
            raise ValueError(f"Unexpected rank {x.dim()} for inverse normalization.")

        out = x.clone()
        for v in self.var_order or []:
            a, b = self.var_slices[v]
            mode = self._effective_mode(v)
            if mode == "identity":
                continue
            mu, sd = self._get_mu_sd(v, out.dtype, out.device)
            y = out[:, a:b] * sd + mu

            if mode == "zscore":
                out[:, a:b] = y
            elif mode == "log_zscore":
                out[:, a:b] = torch.exp(y)
            elif mode == "log1p_zscore":
                out[:, a:b] = torch.expm1(y) * self.log1p_scale
            elif mode == "slog_zscore":
                # y = sign(x) * log1p(|x| / s)  => x = sign(y) * (exp(|y|) - 1) * s
                out[:, a:b] = torch.sign(y) * (torch.expm1(torch.abs(y))) * self.slog1p_scale
            else:
                raise ValueError(f"Unknown normalization mode '{mode}'")

        return out.squeeze(0) if squeezed else out
