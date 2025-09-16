#!/usr/bin/env python3
"""
Compare emulator output vs ground truth for a single sample (CPU-only).

Three panels:
  1) TRUE target (denormalized)
  2) EMULATED prediction (denormalized)
  3) FRACTIONAL ERROR = (pred - true) / (|true| + FRAC_EPS)

No CLI. Configure via GLOBALS below.
This lives in `testing/` beside `src/` and `data/`.

It:
  * Loads TWO raw snapshots: files[FILE_INDEX], files[FILE_INDEX+1]
  * Disables processed tensor caching (no .pt writes)
  * Avoids writing normalization stats
  * Loads run config from models/<MODEL_NAME>/config.py
  * FALLS BACK to repo-level data paths if the run config’s paths don’t exist
  * Builds channel mapping directly from the dataset (so indices match stacking)
"""

from __future__ import annotations
from pathlib import Path
import sys
import importlib.util
from typing import List, Tuple

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt

# ============ GLOBALS ============
MODEL_NAME          = "trained_model"      # subdir under models/
CKPT_FILENAME       = "model_best.pt"      # or "model_last.pt"
FILE_INDEX          = 50                    # raw pair: [i, i+1]
FEATURE_NAME        = "temp"               # variable name (e.g., "temp")
LEVEL_INDEX         = 39                    # vertical level index within that variable

# Colormaps / figure
CMAP_MAPS           = "twilight_shifted"   # for TRUE and PRED
CMAP_ERROR          = "RdBu_r"             # for fractional error
FIGSIZE             = (16, 5.25)
DPI                 = 200

# Color limits for maps (TRUE/PRED)
USE_PERCENTILE_CLIM = True                 # use [1,99] percentiles for vmin/vmax
VMIN                = None                 # set BOTH to override
VMAX                = None

# Color limits for fractional error
ERROR_USE_PERCENTILE_CLIM = True           # symmetric around 0 using 99th pct of |err|
ERROR_VMIN           = None                # set BOTH to override
ERROR_VMAX           = None

# Fractional error epsilon
FRAC_EPS             = 1e-6
# =================================

# Repo paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR      = PROJECT_ROOT / "src"
MODELS_DIR   = PROJECT_ROOT / "models"
RUN_DIR      = MODELS_DIR / MODEL_NAME
RUN_CONFIG   = RUN_DIR / "config.py"
CKPT_PATH    = RUN_DIR / CKPT_FILENAME
FIG_DIR      = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Project modules
from normalization import Normalizer             # type: ignore
from planet_model import PlanetSnapshotForecastDataset  # type: ignore
from sfno_model import build_sfno                # type: ignore

try:
    plt.style.use("science.mplstyle")
except Exception:
    pass


# ----------------------------- helpers -----------------------------

def _load_run_config(cfg_path: Path):
    if not cfg_path.exists():
        raise FileNotFoundError(f"Run config not found: {cfg_path}")
    spec = importlib.util.spec_from_file_location("run_config", str(cfg_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _discover_raw_files(data_dir: Path, pattern: str) -> List[Path]:
    raw = sorted(Path(data_dir).glob(pattern))
    if not raw:
        raise FileNotFoundError(f"No raw files in {data_dir} matching '{pattern}'")
    return raw


def _pick_two(files: List[Path], base_idx: int) -> List[Path]:
    """Pick exactly two consecutive files: [i, i+1]. Clamp i so i+1 exists."""
    if len(files) < 2:
        raise RuntimeError("Need at least two raw files to form a (t, t+1) pair.")
    i = int(base_idx)
    if i >= len(files) - 1:
        i = len(files) - 2
    if i < 0:
        i = 0
    return [files[i], files[i + 1]]


def _channel_map_from_dataset(ds_obj: PlanetSnapshotForecastDataset) -> List[Tuple[str, int]]:
    """
    Build channel map [(var_name, level_index), ...] from the dataset's own
    selected_variables and channels_per_var. This guarantees alignment with stacking.
    """
    mapping: List[Tuple[str, int]] = []
    for v in ds_obj.selected_variables:
        n = int(ds_obj.channels_per_var[v])
        for li in range(n):
            mapping.append((v, li))
    if not mapping:
        raise RuntimeError("Dataset has no channels after selection/level slicing.")
    return mapping


def _find_channel_index(mapping: List[Tuple[str, int]], feature_name: str, level_index: int) -> int:
    for i, (v, lev) in enumerate(mapping):
        if v == feature_name and lev == level_index:
            return i
    raise KeyError(f"Feature '{feature_name}' at level {level_index} not found. "
                   f"Available (first 20): {mapping[:20]} ... total {len(mapping)} channels")


def _denorm_single_channel(norm: Normalizer, ch_idx: int, array: np.ndarray) -> np.ndarray:
    """
    Invert normalization for a single [H,W] channel using norm.stats and mode.
    Supports: identity, zscore, log_zscore, log1p_zscore, slog_zscore.
    """
    assert array.ndim == 2, "Expect [H,W] channel array."
    if norm.var_slices is None:
        return array

    for v, (a, b) in norm.var_slices.items():
        if a <= ch_idx < b:
            local = ch_idx - a
            mode = norm._effective_mode(v)

            # If no stats, nothing to invert (plot in normalized space)
            if mode == "identity" or norm.stats is None:
                return array

            mu = float(norm.stats[v]["mean"][local])
            sd = float(norm.stats[v]["std"][local])
            sd = max(sd, 1e-12)

            if mode == "zscore":
                return array * sd + mu
            elif mode == "log_zscore":
                return np.exp(array * sd + mu)
            elif mode == "log1p_zscore":
                scale = float(getattr(norm, "log1p_scale", 1.0))
                return np.expm1(array * sd + mu) * scale
            elif mode == "slog_zscore":
                s = float(getattr(norm, "slog1p_scale", 1.0))
                y = array * sd + mu
                return np.sign(y) * np.expm1(np.abs(y)) * s
            else:
                raise ValueError(f"Unknown normalization mode '{mode}'")
    raise IndexError(f"Channel index {ch_idx} not found in normalizer var_slices.")


# ----------------------------- main -----------------------------

def main():
    # Load run config
    run_cfg = _load_run_config(RUN_CONFIG)

    # Repo-level data fallback if run_cfg points inside models/
    repo_raw_dir       = PROJECT_ROOT / "data" / "raw"
    repo_processed_dir = PROJECT_ROOT / "data" / "processed"

    raw_dir = Path(getattr(run_cfg, "RAW_DATA_DIR", repo_raw_dir))
    if not raw_dir.exists():
        raw_dir = repo_raw_dir

    processed_dir = Path(getattr(run_cfg, "PROCESSED_DATA_DIR", repo_processed_dir))
    if not processed_dir.exists():
        processed_dir = repo_processed_dir

    snapshot_glob = getattr(run_cfg, "SNAPSHOT_GLOB", "hs94.*_latlon.nc")

    device = torch.device("cpu")

    # Raw files and pair selection
    all_raw = _discover_raw_files(raw_dir, snapshot_glob)
    two_files = _pick_two(all_raw, FILE_INDEX)

    # Normalizer: mirror training modes; DO NOT save stats here
    normalizer = Normalizer(
        mode_default=getattr(run_cfg, "NORMALIZATION_MODE", "zscore"),
        mode_by_var=dict(getattr(run_cfg, "NORMALIZATION_BY_VAR", {})),
        stats_path=None,
        slog1p_scale=getattr(run_cfg, "SLOG1P_SCALE", 1.0),
    )
    setattr(normalizer, "log1p_scale", float(getattr(run_cfg, "LOG1P_SCALE", 1.0)))

    # Dataset (no cache writes)
    ds = PlanetSnapshotForecastDataset(
        files=two_files,
        variables=getattr(run_cfg, "VARIABLES", "ALL"),
        levels_slice=getattr(run_cfg, "LEVELS_SLICE", slice(None)),
        drop_variables=getattr(run_cfg, "DROPPED_VARIABLES", []),
        processed_dir=processed_dir,
        cache_processed=False,
        device=device,
        normalizer=normalizer,
    )

    # Single pair
    inp, tar = ds[0]  # [C,H,W] normalized
    C, H, W = inp.shape

    # Channel mapping from the DATASET (sorted + sliced exactly as used)
    ch_map = _channel_map_from_dataset(ds)
    ch_idx = _find_channel_index(ch_map, FEATURE_NAME, LEVEL_INDEX)
    if not (0 <= ch_idx < C):
        raise IndexError(f"Resolved channel {ch_idx} out of range 0..{C-1}")

    # Model on CPU with training hyperparams
    model = build_sfno(
        nlat=H, nlon=W,
        in_chans=C, out_chans=C,
        grid=getattr(run_cfg, "GRID", "equiangular"),
        grid_internal=getattr(run_cfg, "GRID_INTERNAL", "legendre-gauss"),
        scale_factor=getattr(run_cfg, "SCALE_FACTOR", 1),
        embed_dim=getattr(run_cfg, "EMBED_DIM", 16),
        num_layers=getattr(run_cfg, "NUM_LAYERS", 4),
        activation_function=getattr(run_cfg, "ACTIVATION_FUNCTION", "gelu"),
        encoder_layers=getattr(run_cfg, "ENCODER_LAYERS", 1),
        use_mlp=getattr(run_cfg, "USE_MLP", False),
        mlp_ratio=getattr(run_cfg, "MLP_RATIO", 2.0),
        drop_rate=getattr(run_cfg, "DROP_RATE", 0.0),
        drop_path_rate=getattr(run_cfg, "DROP_PATH_RATE", 0.0),
        normalization_layer=getattr(run_cfg, "NORM_LAYER", "none"),
        hard_thresholding_fraction=getattr(run_cfg, "HARD_THRESH_F", 1.0),
        residual_prediction=getattr(run_cfg, "RESIDUAL_PREDICTION", None),
        pos_embed=getattr(run_cfg, "POS_EMBED", "learnable lat"),
        bias=getattr(run_cfg, "BIAS", False),
        device=device,
    )

    # Load checkpoint
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={missing}, unexpected={unexpected}")

    # Forward (normalized space)
    with torch.inference_mode():
        pred = model(inp.unsqueeze(0)).squeeze(0)  # [C,H,W]

    # Denormalize the chosen channel
    true_map = _denorm_single_channel(normalizer, ch_idx, tar[ch_idx].numpy())
    pred_map = _denorm_single_channel(normalizer, ch_idx, pred[ch_idx].numpy())

    # Percent error
    err_map = 100 * (pred_map - true_map) / (np.abs(true_map) + float(FRAC_EPS))

    # Sanity prints: denormalized ranges
    print(f"[INFO] Denorm TRUE min/max: {np.nanmin(true_map):.3f} / {np.nanmax(true_map):.3f}")
    print(f"[INFO] Denorm PRED min/max: {np.nanmin(pred_map):.3f} / {np.nanmax(pred_map):.3f}")
    print(f"[INFO] Frac ERR min/max:     {np.nanmin(err_map):.3e} / {np.nanmax(err_map):.3e}")

    # Coordinates (for axes)
    with xr.open_dataset(two_files[0]) as ds_raw:
        lats = ds_raw["lat"].values
        lons = ds_raw["lon"].values

    # Titles
    title_L = f"TRUE — {FEATURE_NAME}[level={LEVEL_INDEX}] (ch {ch_idx})"
    title_C = f"PRED — {FEATURE_NAME}[level={LEVEL_INDEX}] (ch {ch_idx})"
    title_R = f"Percent ERROR"

    # Shared color limits for maps
    if VMIN is not None and VMAX is not None:
        vmin_map, vmax_map = float(VMIN), float(VMAX)
    elif USE_PERCENTILE_CLIM:
        both = np.concatenate([true_map.ravel(), pred_map.ravel()])
        vmin_map = float(np.nanpercentile(both, 1))
        vmax_map = float(np.nanpercentile(both, 99))
        if not np.isfinite(vmin_map) or not np.isfinite(vmax_map) or vmin_map == vmax_map:
            vmin_map, vmax_map = float(np.nanmin(both)), float(np.nanmax(both))
    else:
        vmin_map = float(min(np.nanmin(true_map), np.nanmin(pred_map)))
        vmax_map = float(max(np.nanmax(true_map), np.nanmax(pred_map)))

    # Color limits for fractional error (symmetric)
    if ERROR_VMIN is not None and ERROR_VMAX is not None:
        vmin_err, vmax_err = float(ERROR_VMIN), float(ERROR_VMAX)
    elif ERROR_USE_PERCENTILE_CLIM:
        m = float(np.nanpercentile(np.abs(err_map), 99))
        m = m if np.isfinite(m) and m > 0 else float(np.nanmax(np.abs(err_map)))
        if not np.isfinite(m) or m == 0:
            m = 1.0
        vmin_err, vmax_err = -m, +m
    else:
        amax = float(np.nanmax(np.abs(err_map)))
        amax = amax if np.isfinite(amax) and amax > 0 else 1.0
        vmin_err, vmax_err = -amax, +amax

    # Plot (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, constrained_layout=True, sharex=True, sharey=True)
    axL, axC, axR = axes
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]

    imL = axL.imshow(true_map, extent=extent, origin="lower", aspect="auto",
                     cmap=CMAP_MAPS, vmin=vmin_map, vmax=vmax_map)
    axL.set_title(title_L)
    axL.set_xlabel("Longitude (°)")
    axL.set_ylabel("Latitude (°)")

    imC = axC.imshow(pred_map, extent=extent, origin="lower", aspect="auto",
                     cmap=CMAP_MAPS, vmin=vmin_map, vmax=vmax_map)
    axC.set_title(title_C)
    axC.set_xlabel("Longitude (°)")

    imR = axR.imshow(err_map, extent=extent, origin="lower", aspect="auto",
                     cmap=CMAP_ERROR, vmin=vmin_err, vmax=vmax_err)
    axR.set_title(title_R)
    axR.set_xlabel("Longitude (°)")

    # Two colorbars: one for maps (TRUE/PRED), one for error
    cbar_maps = fig.colorbar(imC, ax=[axL, axC], shrink=0.9)
    cbar_maps.set_label("Value (denormalized)")

    cbar_err = fig.colorbar(imR, ax=[axR], shrink=0.9)
    cbar_err.set_label("Percent error")

    outname = f"compare_denorm_run[{MODEL_NAME}]_idx{FILE_INDEX}_{FEATURE_NAME}_lev{LEVEL_INDEX}_with_fracerr.png"
    outpath = FIG_DIR / outname
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    print(f"[FIG] saved -> {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
