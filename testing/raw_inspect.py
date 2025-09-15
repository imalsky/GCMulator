#!/usr/bin/env python3
"""
Inspect a raw NetCDF snapshot:
- lists dims/coords/variables
- prints quick stats for a few variables
- plots one vertical level of a chosen variable and SAVES it to figures/

No CLI; tweak GLOBALS below.
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# -------------------- GLOBALS --------------------
FILE_INDEX   = 0           # which raw snapshot (by sorted order)
VARIABLE     = "temp"      # which variable to plot
LEVEL        = 0           # which vertical level index to plot
TIME_INDEX   = 0           # usually 0 (one time step per file)
MAX_VARS_LOG = 100         # how many variables to print stats for
CMAP         = "viridis"
DPI          = 200
# ------------------------------------------------

# make src importable
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


def _print_structure(ds: xr.Dataset) -> None:
    print("FILE STRUCTURE")
    print("=" * 50)
    print("Dimensions:", dict(ds.sizes))
    print("Coordinates:", list(ds.coords))
    print("Variables:", list(ds.data_vars))


def _print_basic_stats(ds: xr.Dataset, max_vars: int = 200) -> None:
    print("\nBASIC STATS (time=0):")
    for var in list(ds.data_vars.keys())[:max_vars]:
        arr = ds[var].isel(time=0)
        vmin = float(arr.min())
        vmax = float(arr.max())
        vmean = float(arr.mean())
        print(f"  - {var:>12}: shape={tuple(arr.shape)}  min={vmin:.3f}  max={vmax:.3f}  mean={vmean:.3f}")


def _plot_one_level_and_save(ds: xr.Dataset, variable: str, level: int, time_index: int = 0, cmap: str = "viridis", outpath: Path | None = None) -> None:
    if variable not in ds.data_vars:
        raise KeyError(f"Variable '{variable}' not found. Available: {list(ds.data_vars.keys())}")
    if "level" not in ds[variable].dims:
        print(f"[WARN] Variable '{variable}' has no 'level' dimension; skipping plot.")
        return

    data = ds[variable].isel(time=time_index, level=level).values
    lats = ds["lat"].values
    lons = ds["lon"].values

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(
        data,
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        origin="lower",
        aspect="auto",
        cmap=cmap,
    )
    ax.set_title(f"{variable} | level={level} | time={time_index}")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()

    if outpath is None:
        outpath = FIG_DIR / f"raw_inspect_{variable}_lvl{level}_t{time_index}.png"
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    print(f"[FIG] saved -> {outpath}")
    plt.close(fig)


def main():
    files = _find_files()
    path = files[FILE_INDEX]
    print(f"[INFO] Reading RAW snapshot: {path}")

    ds = xr.open_dataset(path)
    _print_structure(ds)
    _print_basic_stats(ds, max_vars=MAX_VARS_LOG)

    stem = Path(path).stem  # e.g., hs94.00000_latlon
    outpath = FIG_DIR / f"raw_inspect_{stem}_{VARIABLE}_lvl{LEVEL}_t{TIME_INDEX}.png"
    _plot_one_level_and_save(ds, variable=VARIABLE, level=LEVEL, time_index=TIME_INDEX, cmap=CMAP, outpath=outpath)
    ds.close()


if __name__ == "__main__":
    main()
