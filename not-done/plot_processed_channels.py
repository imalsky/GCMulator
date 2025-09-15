#!/usr/bin/env python3
"""
Visualize processed tensors saved by the training pipeline in data/processed/.
Each processed tensor corresponds to a raw snapshot and is saved as <stem>.pt
(e.g., hs94.00000_latlon.pt). Tensors are [C,H,W] **before** normalization.

Saves figures to PROJECT_ROOT/figures/.

This script expects the current repo layout:
  <PROJECT_ROOT>/
    ├─ src/
    ├─ data/
    │   ├─ raw/
    │   └─ processed/
    └─ testing/   <-- this file lives here
"""

from __future__ import annotations
from pathlib import Path
import sys
import math
from typing import List, Tuple

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt

# -------------------- GLOBALS --------------------
FILE_INDEX         = 0           # which processed snapshot (matching raw order)
CHANNELS_TO_PLOT   = [0, 5, 10]  # channel indices to visualize (levels-as-channels)
CMAP               = "viridis"
FIGSIZE            = (14, 8)
DPI                = 200
# ------------------------------------------------

# ---------- Repo paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR      = PROJECT_ROOT / "src"
FIG_DIR      = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# so we can `from config import ...`
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------- Config ----------
from config import (  # noqa: E402
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SNAPSHOT_GLOB,
    VARIABLES, DROPPED_VARIABLES, LEVELS_SLICE,
)

# Optional style; skip if not installed
try:
    plt.style.use("../testing/science.mplstyle")
except Exception:
    pass


def _find_raw_and_processed() -> Tuple[List[Path], List[Path]]:
    raw_files = sorted(Path(RAW_DATA_DIR).glob(SNAPSHOT_GLOB))
    if not raw_files:
        raise FileNotFoundError(f"No raw files in {RAW_DATA_DIR} matching '{SNAPSHOT_GLOB}'")
    proc_files = [Path(PROCESSED_DATA_DIR) / f"{p.stem}__42a42232.pt" for p in raw_files]
    return raw_files, proc_files


def _resolve_variables(ds: xr.Dataset) -> List[str]:
    """Apply VARIABLES/DROPPED_VARIABLES like the training dataset."""
    if VARIABLES == "ALL" or VARIABLES is None:
        var_list = [v for v in ds.data_vars.keys() if v not in set(DROPPED_VARIABLES)]
    else:
        # honor explicit list; fail fast if missing
        var_list = []
        missing = []
        for v in VARIABLES:
            if v in ds.data_vars:
                if v not in set(DROPPED_VARIABLES):
                    var_list.append(v)
            else:
                missing.append(v)
        if missing:
            raise KeyError(f"Requested variables not found in file: {missing}")
    if not var_list:
        raise ValueError("After applying selection/drops, no variables remain.")
    return var_list


def _channel_map(ds: xr.Dataset, var_list: List[str]) -> List[Tuple[str, int]]:
    """
    Build a mapping from channel index -> (var_name, level_index_in_file),
    mirroring the stacking logic in the dataset (strict 3D).
    """
    mapping: List[Tuple[str, int]] = []
    for v in var_list:
        da = ds[v]
        # Strict 3D: must contain time, level, lat, lon (time can be singleton)
        dims = tuple(da.dims)
        if "level" not in dims or "lat" not in dims or "lon" not in dims:
            raise ValueError(
                f"Variable '{v}' is not strict 3D with 'level','lat','lon' dims; dims={dims}"
            )

        # Slice off time if present; then apply LEVELS_SLICE to 'level'
        if "time" in dims:
            da = da.isel(time=0)

        nlev = da.sizes["level"]
        # materialize the slice for robustness
        sl = LEVELS_SLICE if isinstance(LEVELS_SLICE, slice) else slice(None)
        indices = list(range(nlev))[sl]
        for lev_idx in indices:
            mapping.append((v, int(lev_idx)))
    if not mapping:
        raise ValueError("LEVELS_SLICE removed all levels; nothing to plot.")
    return mapping


def main():
    raw_files, proc_files = _find_raw_and_processed()

    if FILE_INDEX < 0 or FILE_INDEX >= len(proc_files):
        raise IndexError(f"FILE_INDEX out of range: 0..{len(proc_files)-1}")

    raw_path  = raw_files[FILE_INDEX]
    proc_path = proc_files[FILE_INDEX]

    if not proc_path.exists():
        raise FileNotFoundError(
            f"Processed tensor not found: {proc_path}\n"
            f"Generate caches by running your training entrypoint once."
        )

    # Load raw dataset to get coords and channel mapping
    ds = xr.open_dataset(raw_path)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        ds.close()
        raise KeyError("Expected 'lat' and 'lon' coordinates in the raw file.")

    lats = ds["lat"].values
    lons = ds["lon"].values
    var_list = _resolve_variables(ds)
    ch_map   = _channel_map(ds, var_list)  # list of (var, level_idx)
    # Load the processed tensor (pre-normalization) [C, H, W]
    x = torch.load(proc_path, map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Processed file {proc_path} did not contain a torch.Tensor")
    x = x.float()
    C, H, W = x.shape
    print(f"[INFO] Loaded: {proc_path.name}  shape=[{C},{H},{W}]")
    if C != len(ch_map):
        print(f"[WARN] Channel count mismatch: tensor C={C}, mapping={len(ch_map)}. Titles may be off.")

    # Grid layout
    k = len(CHANNELS_TO_PLOT)
    if k == 0:
        print("[INFO] No channels requested; nothing to do.")
        ds.close()
        return
    ncols = min(4, k)
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE, constrained_layout=True)
    axes = np.atleast_2d(axes)

    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))

    for i, ch in enumerate(CHANNELS_TO_PLOT):
        if ch < 0 or ch >= C:
            raise IndexError(f"Channel {ch} out of range 0..{C-1}")
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        arr = x[ch].numpy()
        # robust color range
        vmin = np.nanpercentile(arr, 1)
        vmax = np.nanpercentile(arr, 99)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = np.nanmin(arr), np.nanmax(arr)

        im = ax.imshow(
            arr,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin="lower",
            aspect="auto",
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
        )

        # Title with var/level decoding if available
        if ch < len(ch_map):
            vname, lev = ch_map[ch]
            ax.set_title(f"ch {ch} — {vname}[level={lev}]")
        else:
            ax.set_title(f"ch {ch}")

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # hide unused axes
    for j in range(k, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    stem   = Path(raw_path).stem
    ch_tag = "-".join(map(str, CHANNELS_TO_PLOT))
    outpath = FIG_DIR / f"processed_channels_{stem}_ch[{ch_tag}].png"
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    print(f"[FIG] saved -> {outpath}")
    plt.close(fig)
    ds.close()


if __name__ == "__main__":
    main()
