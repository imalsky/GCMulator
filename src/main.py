#!/usr/bin/env python3
"""
Main entrypoint: dataset → model → train → save best and artifacts into RUN_DIR.

Artifacts written under:
  models/<MODEL_NAME>/
    ├── train.log
    ├── metrics.csv
    ├── model_best.pt
    ├── model_last.pt
    └── config.py            (copy of the config used)
"""

from __future__ import annotations
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset

# Local imports
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
from train import train_loop


# ---------------------- logging ----------------------
def _setup_logging() -> logging.Logger:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(LOG_FORMAT)

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.addHandler(ch)

    # File (into RUN_DIR)
    if LOG_TO_FILE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, mode="w")
        fh.setFormatter(fmt)
        fh.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        logger.addHandler(fh)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("xarray").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(f"Logging to {LOG_FILE}")
    return logging.getLogger(__name__)


# ---------------------- utils ----------------------
def _select_files() -> List[Path]:
    paths = sorted(RAW_DATA_DIR.glob(SNAPSHOT_GLOB))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least 2 files in {RAW_DATA_DIR} matching '{SNAPSHOT_GLOB}'")
    sl = slice(FILE_INDEX_START or 0, FILE_INDEX_END, FILE_INDEX_STEP or 1)
    sel = paths[sl]
    if len(sel) < 2:
        raise RuntimeError(f"Selected slice yields < 2 files (got {len(sel)}).")
    return sel


def _split_indices(n: int, val_frac: float = 0.1) -> tuple[list[int], list[int]]:
    # dataset length is n-1 (pairs), but indices map into that directly
    val_n = max(1, int(round((n - 1) * val_frac)))
    train_n = (n - 1) - val_n
    train_idx = list(range(0, train_n))
    val_idx = list(range(train_n, n - 1))
    return train_idx, val_idx


def _copy_config_into_run_dir() -> None:
    src_cfg = Path(__file__).resolve().parent / "config.py"
    dst_cfg = RUN_DIR / "config.py"
    try:
        shutil.copy2(src_cfg, dst_cfg)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to copy config.py into run dir: {e}")


# ---------------------- main ----------------------
def main():
    logger = _setup_logging()

    # Repro
    torch.manual_seed(SEED_GLOBAL)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_GLOBAL)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    # Ensure run dir exists and drop a copy of the config
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _copy_config_into_run_dir()

    # Files
    files = _select_files()
    logger.info(f"Using {len(files)} files ({files[0].name} .. {files[-1].name})")

    # Normalizer
    normalizer = Normalizer(
        mode_default=NORMALIZATION_MODE,
        mode_by_var=dict(NORMALIZATION_BY_VAR),
        stats_path=STATS_CACHE_PATH,
    )

    # Dataset
    ds = PlanetSnapshotForecastDataset(
        files=files,
        variables=VARIABLES,
        levels_slice=LEVELS_SLICE,
        drop_variables=DROPPED_VARIABLES,
        processed_dir=PROCESSED_DATA_DIR,
        cache_processed=CACHE_PROCESSED,
        device=None,                  # tensors on CPU; DataLoader moves if needed
        normalizer=normalizer,
    )

    # Split train/val
    train_idx, val_idx = _split_indices(len(files))
    ds_train = Subset(ds, train_idx)
    ds_val   = Subset(ds, val_idx)

    # Loaders
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    # Build model
    # Infer C,H,W from one batch
    xb0, yb0 = next(iter(train_loader))
    C, H, W = xb0.shape[-3:]
    logger.info(f"Tensor shape: C={C}, H={H}, W={W}")

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
    )

    # Optimizer / scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)

    if LR_SCHEDULER is None:
        scheduler = None
    elif LR_SCHEDULER.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR_MIN
        )
    else:
        raise ValueError(f"Unknown LR_SCHEDULER: {LR_SCHEDULER}")

    # Train
    best_val, best_ckpt, last_ckpt = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        run_dir=RUN_DIR,
        logger=logging.getLogger(__name__),
        epochs=EPOCHS,
        use_amp=USE_AMP,
    )

    logger.info(f"Artifacts written to: {RUN_DIR}")
    logger.info(f"Best checkpoint:  {best_ckpt}")
    logger.info(f"Last checkpoint:  {last_ckpt}")


if __name__ == "__main__":
    main()
