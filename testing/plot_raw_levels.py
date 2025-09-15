#!/usr/bin/env python3
"""
Plot multiple vertical levels (subplots) for a single variable
from one raw NetCDF snapshot and SAVE to figures/.

No CLI; tweak GLOBALS below.
"""
from __future__ import annotations
from pathlib import Path
import sys
import math

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# -------------------- GLOBALS --------------------
FILE_INDEX = 0
VARIABLE   = "temp"
LEVELS     = [0, 5, 10, 19]  # choose any valid level indices
TIME_INDEX = 0
CMAP       = "viridis"
FIGSIZE    = (14, 8)
DPI        = 200
# ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("science.mplstyle")

from config import RAW_DATA_DIR, SNAPSHOT_GLOB  # noqa: E402


def _find_files() -> list[Path]:
    files = sorted(RAW_DATA_DIR.glob(SNAPSHOT_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matching {SNAPSHOT_GLOB} in {RAW_DATA_DIR}")
    return files


def main():
    files = _find_files()
    path = files[FILE_INDEX]
    print(f"[INFO] Reading RAW snapshot: {path}")

    ds = xr.open_dataset(path)
    lats = ds["lat"].values
    lons = ds["lon"].values

    k = len(LEVELS)
    ncols = min(4, k)
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE)
    axes = np.atleast_2d(axes)

    for i, lvl in enumerate(LEVELS):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        arr = ds[VARIABLE].isel(time=TIME_INDEX, level=lvl).values
        im = ax.imshow(
            arr,
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            origin="lower",
            aspect="auto",
            cmap=CMAP,
        )
        ax.set_title(f"{VARIABLE} | level={lvl}")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # hide any unused axes
    for j in range(k, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    plt.tight_layout()

    stem = Path(path).stem
    lvl_tag = "-".join(map(str, LEVELS))
    outpath = FIG_DIR / f"raw_levels_{stem}_{VARIABLE}_lvls[{lvl_tag}]_t{TIME_INDEX}.png"
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    print(f"[FIG] saved -> {outpath}")
    plt.close(fig)
    ds.close()


if __name__ == "__main__":
    main()
