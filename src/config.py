#!/usr/bin/env python3
"""
Global configuration: paths, dataset, normalization, model, training, logging.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict

# =========================
# Paths
# =========================
PROJECT_ROOT       = Path(__file__).resolve().parents[1]
RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Models / run directory
MODELS_DIR: Path = PROJECT_ROOT / "models"     # always "models" at repo root
MODEL_NAME: str = "trained_model"              # configurable
RUN_DIR: Path   = MODELS_DIR / MODEL_NAME      # all run artifacts live here

# Ensure base data dirs exist (model dir is created in main)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Match files like "hs94.*_latlon.nc" etc.
SNAPSHOT_GLOB = "hs94.*_latlon.nc"

# =========================
# File subset (by index)
# =========================
FILE_INDEX_START: Optional[int] = 0
FILE_INDEX_END:   Optional[int] = 500     # None -> all
FILE_INDEX_STEP:  int = 1                # positive, non-zero

# =========================
# Dataset (3D, strict dims)
# =========================
# VARIABLES:
#   "ALL" -> all data_vars (minus DROPPED_VARIABLES), or a list of names
VARIABLES = "ALL"
DROPPED_VARIABLES = ["r0", "r1", "r2"]
LEVELS_SLICE = slice(None)               # e.g. slice(0, 20)

# =========================
# Normalization
# =========================
# Modes: "identity" | "zscore" | "log_zscore" | "log1p_zscore" | "slog_zscore"
NORMALIZATION_MODE: str = "zscore"
NORMALIZATION_BY_VAR: Dict[str, str] = {
    "Kt":"log1p_zscore",
    "Kv":"log1p_zscore",
    "Teq":"zscore",
    "press":"log_zscore",
    "r0":"log_zscore",
    "r1":"log_zscore",
    "r2":"log_zscore",
    "rho":"zscore",
    "temp":"zscore",
    "theta":"zscore",
    "vel1":"zscore",
    "vel2":"zscore",
    "vel3":"slog_zscore",
    "vlat":"slog_zscore",
    "vlon":"slog_zscore"
}
STATS_CACHE_PATH: Optional[Path] = PROCESSED_DATA_DIR / "stats.pt"
CACHE_PROCESSED: bool = True

# =========================
# Model (SFNO) — ALL ARGS IN CONFIG
# These map DIRECTLY to torch_harmonics.examples.models.sfno.SFNO
# =========================
NORM_LAYER: str = "none"  # "none" or "layer_norm"

# Core geometry
GRID: str = "equiangular"
GRID_INTERNAL: str = "legendre-gauss"
SCALE_FACTOR: int = 1

# Channels are inferred from data; embed/stack depth
EMBED_DIM: int = 16
NUM_LAYERS: int = 4
ENCODER_LAYERS: int = 1

# MLP head inside each block
USE_MLP: bool = False
MLP_RATIO: float = 2.0

# Regularization
DROP_RATE: float = 0.0
DROP_PATH_RATE: float = 0.0

# Activations / normalization inside blocks
ACTIVATION_FUNCTION: str = "gelu"
HARD_THRESH_F: float = 1.0

# Positional embedding & bias
POS_EMBED: str = "learnable lat"
BIAS: bool = False

# Residual prediction:
# - If None, it defaults to (out_chans == in_chans)
RESIDUAL_PREDICTION: Optional[bool] = None

# =========================
# Training
# =========================
BATCH_SIZE: int = 4
EPOCHS: int = 100
WEIGHT_DECAY: float = 0.0

# LR scheduler (cosine annealing)
LR_SCHEDULER: Optional[str] = "cosine"  # "cosine" or None
LR_MAX: float = 5e-3
LR_MIN: float = 1e-7

USE_AMP: bool = False   # autocast

# =========================
# Repro
# =========================
SEED_GLOBAL: int = 0
SEED_TRAIN: int  = 333

# =========================
# Logging
# =========================
LOG_LEVEL = "INFO"     # "DEBUG" | "INFO" | "WARNING" | "ERROR"
LOG_TO_FILE = True
LOG_DIR = RUN_DIR                   # <— logs live inside the run directory
LOG_FILE = LOG_DIR / "train.log"    # filename inside RUN_DIR
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_EVERY_N_STEPS: int = 100
