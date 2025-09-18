#!/usr/bin/env python3
"""
Main training entrypoint.
Loads entire dataset to GPU memory, trains model, saves checkpoints.
"""

from __future__ import annotations
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from config import (
    PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR, SNAPSHOT_GLOB,
    FILE_INDEX_START, FILE_INDEX_END, FILE_INDEX_STEP,
    VARIABLES, DROPPED_VARIABLES, LEVELS_SLICE,
    BATCH_SIZE, EPOCHS, WEIGHT_DECAY, USE_AMP,
    LR_SCHEDULER, LR_MAX, LR_MIN,
    GRID, GRID_INTERNAL, SCALE_FACTOR,
    EMBED_DIM, NUM_LAYERS, ENCODER_LAYERS,
    ACTIVATION_FUNCTION, USE_MLP, MLP_RATIO,
    DROP_RATE, DROP_PATH_RATE,
    NORM_LAYER, HARD_THRESH_F, POS_EMBED, BIAS, RESIDUAL_PREDICTION,
    NORMALIZATION_MODE, NORMALIZATION_BY_VAR, STATS_CACHE_PATH, CACHE_PROCESSED,
    LOG_LEVEL, LOG_TO_FILE, LOG_DIR, LOG_FILE, LOG_FORMAT,
    MODELS_DIR, MODEL_NAME, RUN_DIR,
    SEED_GLOBAL, SEED_TRAIN,
)
from normalization import Normalizer
from planet_model import PlanetSnapshotForecastDataset
from sfno_model import build_sfno
from train import train_loop, make_spherical_mse


# ---------------- precision selection ----------------
def _select_compute_dtype(use_amp: bool) -> Tuple[torch.dtype, str]:
    """
    Choose the autocast dtype (torch.dtype) and a label string for dataset/logs.
    We control precision with autocast in the training loop; the model is NOT cast
    to avoid clobbering complex-valued SFNO parameters.
    """
    if not use_amp:
        return torch.float32, "fp32"

    # Prefer bf16 on Ampere+ if supported; else fall back to fp16
    try:
        is_bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        is_bf16_supported = False

    if is_bf16_supported:
        return torch.bfloat16, "bf16"
    else:
        return torch.float16, "fp16"


# ---------------- logging ----------------
def _setup_logging() -> logging.Logger:
    """Configure logging to console and file."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter(LOG_FORMAT)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.addHandler(ch)

    if LOG_TO_FILE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, mode="w")
        fh.setFormatter(fmt)
        fh.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        root.addHandler(fh)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("xarray").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


# ---------------- data selection ----------------
def _select_files() -> List[Path]:
    """Select subset of snapshot files based on config indices."""
    paths = sorted(RAW_DATA_DIR.glob(SNAPSHOT_GLOB))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least 2 files in {RAW_DATA_DIR} matching '{SNAPSHOT_GLOB}'")

    sl = slice(FILE_INDEX_START or 0, FILE_INDEX_END, FILE_INDEX_STEP or 1)
    sel = paths[sl]

    if len(sel) < 2:
        raise RuntimeError(f"Selected slice yields < 2 files (got {len(sel)}).")
    return sel


def _split_indices(n: int, val_frac: float = 0.1) -> tuple[list[int], list[int]]:
    """Split dataset indices into train/validation sets."""
    val_n = max(1, int(round((n - 1) * val_frac)))
    train_n = (n - 1) - val_n
    train_idx = list(range(0, train_n))
    val_idx = list(range(train_n, n - 1))
    return train_idx, val_idx


def _copy_config_into_run_dir() -> None:
    """Copy config.py to run directory for reproducibility."""
    src_cfg = Path(__file__).resolve().parent / "config.py"
    dst_cfg = RUN_DIR / "config.py"
    try:
        shutil.copy2(src_cfg, dst_cfg)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to copy config.py: {e}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    logger = _setup_logging()

    # Repro
    torch.manual_seed(SEED_GLOBAL)
    torch.cuda.manual_seed_all(SEED_GLOBAL)
    torch.backends.cudnn.benchmark = True

    # Device
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")

    # Precision choice (autocast dtype + label for dataset/logs)
    autocast_dtype, precision_label = _select_compute_dtype(USE_AMP)
    logger.info(f"Precision: {precision_label}")

    # Dirs
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _copy_config_into_run_dir()

    # Files
    files = _select_files()
    logger.info(f"Dataset: {len(files)} files")

    # Normalizer
    normalizer = Normalizer(
        mode_default=NORMALIZATION_MODE,
        mode_by_var=dict(NORMALIZATION_BY_VAR),
        stats_path=STATS_CACHE_PATH,
    )

    # Dataset (GPU-resident)
    logger.info("Loading dataset to GPU memory...")
    ds = PlanetSnapshotForecastDataset(
        files=files,
        variables=VARIABLES,
        levels_slice=LEVELS_SLICE,
        drop_variables=DROPPED_VARIABLES,
        processed_dir=PROCESSED_DATA_DIR,
        cache_processed=CACHE_PROCESSED,
        device=device,
        normalizer=normalizer,
        dtype=precision_label,  # dataset coerces string label -> torch.dtype
    )

    # Split
    train_idx, val_idx = _split_indices(len(files))
    ds_train = Subset(ds, train_idx)
    ds_val = Subset(ds, val_idx)

    # Loaders (data already on GPU)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    # Shape
    xb0, yb0 = next(iter(train_loader))
    C, H, W = xb0.shape[-3:]
    logger.info(f"Data shape: C={C}, H={H}, W={W}")

    # Model
    model = build_sfno(
        nlat=H, nlon=W,
        in_chans=C, out_chans=C,
        grid=GRID, grid_internal=GRID_INTERNAL, scale_factor=SCALE_FACTOR,
        embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, activation_function=ACTIVATION_FUNCTION,
        encoder_layers=ENCODER_LAYERS, use_mlp=USE_MLP, mlp_ratio=MLP_RATIO,
        drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE,
        normalization_layer=NORM_LAYER, hard_thresholding_fraction=HARD_THRESH_F,
        residual_prediction=RESIDUAL_PREDICTION, pos_embed=POS_EMBED, bias=BIAS,
        device=device,
        dtype=precision_label,  # accepted for logging; not applied to model dtype
    )

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR_MIN
        )

    # Losses — restore reference behavior:
    #   train: absolute L2 (area-weighted)
    #   val  : relative L2 (area-weighted ÷ ∫target^2)
    criterion_train = make_spherical_mse(nlat=H, nlon=W, relative=False)
    criterion_val   = make_spherical_mse(nlat=H, nlon=W, relative=True)

    # Train
    logger.info(f"Starting training: {EPOCHS} epochs, batch_size={BATCH_SIZE}")
    best_val, best_ckpt, last_ckpt = train_loop(
        model=model,
        optimizer=optimizer,
        criterion=criterion_train,       # absolute L2
        val_criterion=criterion_val,     # relative L2
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        run_dir=RUN_DIR,
        logger=logger,
        epochs=EPOCHS,
        use_amp=USE_AMP,
        dtype=autocast_dtype,            # torch.dtype used by autocast
    )

    logger.info(f"Training complete. Best validation loss: {best_val:.4e}")
    logger.info(f"Checkpoints saved to: {RUN_DIR}")


if __name__ == "__main__":
    main()
