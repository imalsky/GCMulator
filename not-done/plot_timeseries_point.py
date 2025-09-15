#!/usr/bin/env python3
"""
Plot a time series at a single (lat, lon) for a chosen variable & level
by scanning all raw snapshots in sorted order. SAVE to figures/.

No CLI; tweak GLOBALS below.
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# -------------------- GLOBALS --------------------
VARIABLE   = "temp"   # variable to sample
LEVEL      = 0        # vertical level index to sample
LAT        = 0.0      # degrees
LON        = 0.0      # degrees
START_IDX  = 0        # first file index to include
END_IDX    = None     # None = include all to the end
MARKERS    = False
FIGSIZE    = (10, 4)
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
    files = files[START_IDX:END_IDX] if END_IDX is not None else files[START_IDX:]

    values = []
    ticks = []
    for i, path in enumerate(files):
        ds = xr.open_dataset(path)
        if VARIABLE not in ds.data_vars:
            ds.close()
            raise KeyError(f"'{VARIABLE}' not in {path.name}. Available: {list(ds.data_vars.keys())}")

        # nearest neighbor at requested lat/lon, chosen level, time=0
        field = ds[VARIABLE].isel(time=0, level=LEVEL).sel(lat=LAT, lon=LON, method="nearest")
        values.append(float(field.values))
        ticks.append(i + START_IDX)
        ds.close()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    if MARKERS:
        ax.plot(ticks, values, "-o", linewidth=1)
    else:
        ax.plot(ticks, values, linewidth=1.5)
    ax.set_title(f"{VARIABLE} @ level={LEVEL}, near (lat={LAT}, lon={LON})")
    ax.set_xlabel("snapshot index (sorted)")
    ax.set_ylabel(VARIABLE)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    end_tag = END_IDX if END_IDX is not None else "end"
    outpath = FIG_DIR / f"timeseries_{VARIABLE}_lvl{LEVEL}_lat{LAT:+.1f}_lon{LON:+.1f}_{START_IDX}-{end_tag}.png"
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    print(f"[FIG] saved -> {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
