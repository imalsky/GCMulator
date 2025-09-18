#!/usr/bin/env python3
"""
sfno_model.py — Build SFNO without clobbering complex parameters.

- Accepts a `dtype` kwarg for compatibility with call sites, but DOES NOT
  pass it into `model.to(dtype=...)`. Mixed precision should be controlled
  by autocast in the training loop.
- If `dtype` is provided (string or torch.dtype), it's coerced and stored
  on the returned module as `model.preferred_autocast_dtype` for reference.
"""

from __future__ import annotations
from typing import Optional, Union
import logging
import torch
import torch.nn as nn
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator as SFNO

logger = logging.getLogger(__name__)

# ---------------- helpers ----------------
_DTYPE_ALIASES = {
    "fp32": torch.float32, "float32": torch.float32, "f32": torch.float32,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16,  "float16": torch.float16, "f16": torch.float16,
}
def _coerce_device(x: Union[str, torch.device, None]) -> torch.device:
    if isinstance(x, torch.device):
        return x
    if x is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(x)

def _coerce_dtype(x: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if x is None:
        return None
    if isinstance(x, torch.dtype):
        return x
    if isinstance(x, str):
        k = x.strip().lower()
        if k in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[k]
        raise ValueError(f"Unknown dtype string: {x!r}")
    return None

# ---------------- builder ----------------
def build_sfno(
    *,
    nlat: int,
    nlon: int,
    in_chans: int,
    out_chans: int,
    grid: str,
    grid_internal: str,
    scale_factor: int,
    embed_dim: int,
    num_layers: int,
    activation_function: str,
    encoder_layers: int,
    use_mlp: bool,
    mlp_ratio: float,
    drop_rate: float,
    drop_path_rate: float,
    normalization_layer: str,
    hard_thresholding_fraction: float,
    residual_prediction: Optional[bool],
    pos_embed: str,
    bias: bool,
    device: Union[str, torch.device, None],
    dtype: Union[str, torch.dtype, None] = None,  # accepted but NOT applied to model
) -> nn.Module:
    """
    Construct SFNO and move it to the requested device. We intentionally do not
    cast to `dtype` here to avoid discarding imaginary parts of complex weights.
    """
    device = _coerce_device(device)
    preferred_autocast = _coerce_dtype(dtype)

    if residual_prediction is None:
        residual_prediction = (out_chans == in_chans)

    model = SFNO(
        img_size=(nlat, nlon),
        grid=grid,
        grid_internal=grid_internal,
        scale_factor=scale_factor,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        num_layers=num_layers,
        activation_function=activation_function,
        encoder_layers=encoder_layers,
        use_mlp=use_mlp,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        normalization_layer=normalization_layer,
        hard_thresholding_fraction=hard_thresholding_fraction,
        residual_prediction=residual_prediction,
        pos_embed=pos_embed,
        bias=bias,
    )

    # IMPORTANT: no dtype cast here (SFNO uses complex weights/buffers)
    model = model.to(device=device)

    # Expose the intended autocast dtype for downstream reference/logging
    if preferred_autocast is not None:
        setattr(model, "preferred_autocast_dtype", preferred_autocast)

    # Robust parameter/memory logging
    try:
        param_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
        param_mb = param_bytes / 1e6
        if preferred_autocast is not None:
            logger.info(
                f"Model: {sum(p.numel() for p in model.parameters()):,} parameters "
                f"({param_mb:.1f} MB) | preferred autocast: {preferred_autocast}"
            )
        else:
            logger.info(
                f"Model: {sum(p.numel() for p in model.parameters()):,} parameters "
                f"({param_mb:.1f} MB)"
            )
    except Exception:
        logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model
