#!/usr/bin/env python3
"""
SFNO model builder that passes through the FULL torch-harmonics SFNO API.

All constructor keywords are exposed via config.py and forwarded verbatim.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

# Import the SFNO implementation provided by torch-harmonics
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator as SFNO


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
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Build SFNO with explicit image size and the full parameter set.
    """
    if residual_prediction is None:
        # Default mirrors example behavior: residuals when in/out chans match
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
    if device is not None:
        model = model.to(device)
    return model
