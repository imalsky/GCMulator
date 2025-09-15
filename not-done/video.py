# testing/planet_maps.py
#!/usr/bin/env python3
"""
Planet maps & animations for HS94-style data (one time per NetCDF file).

- Reads a stack of files from data/raw matching GLOB_PATTERN (each has time=1).
- Concats them along time to create a timeline.
- Plots two variables side-by-side for selected vertical levels.
- Saves time-mean maps and per-frame PNGs to ../figures/.
- Optionally builds MP4s via ffmpeg.
- Optional orthographic globe animation (Cartopy), auto-disabled if unavailable.

Everything is controlled via GLOBALS below — no argparse.
"""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ----------------------
# GLOBAL SETTINGS
# ----------------------
PROJECT_ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR          = PROJECT_ROOT / "data" / "raw"
FIG_DIR          = PROJECT_ROOT / "figures"

# Files to read (one snapshot per file). Adjust pattern if your filenames change.
GLOB_PATTERN     = "hs94.*_latlon.nc"

# Subset the sorted file list (start:end:step)
FILE_START       = 0
FILE_END         = 120     # None = all
FILE_STEP        = 1

# Variables to plot side-by-side (must exist in the files)
VAR_LEFT         = "temp"
VAR_RIGHT        = "press"

# Vertical levels to plot (integer indices into dimension 'level')
LEVELS_TO_PLOT   = [0, 1, 3]     # change as you like

# Plot look
STYLE            = "dark_background"
CMAP             = "magma"

# Color scaling (None -> autoscale from data). You can set (vmin, vmax) per var/level if you want fixed scales.
VMIN_VMAX_LEFT   = None           # e.g., (180.0, 320.0) for temperature
VMIN_VMAX_RIGHT  = None           # e.g., (1e3, 1e5) for pressure

# Timeseries to MP4 (png → mp4), requires ffmpeg in PATH
MAKE_VIDEOS      = True
FRAMERATE_IN     = 15
FRAMERATE_OUT    = 30

# Optional: Orthographic globe animation using Cartopy (auto-disabled if not installed)
DO_ORTHOGRAPHIC  = False
ORTHO_VAR        = VAR_LEFT
ORTHO_LEVEL      = 1
ORTHO_VIEWPATH   = dict(lon_start=-80, lon_sweep=80, lat_start=20, lat_sweep=-60)  # simple pan

# ----------------------
# UTILITIES
# ----------------------

def _coord_names(ds: xr.Dataset) -> Tuple[str, str, str, str]:
    """Return the names of (time, level, lat, lon) dims in the dataset."""
    # Common aliases
    time_candidates  = ["time", "Time"]
    level_candidates = ["level", "lev", "plev"]
    lat_candidates   = ["lat", "latitude", "Latitude"]
    lon_candidates   = ["lon", "longitude", "Longitude"]
    def pick(cands):
        for c in cands:
            if c in ds.dims or c in ds.coords:
                return c
        raise KeyError(f"None of {cands} found in dataset dims/coords: {list(ds.dims)} / {list(ds.coords)}")
    t = pick(time_candidates)
    z = pick(level_candidates)
    y = pick(lat_candidates)
    x = pick(lon_candidates)
    return t, z, y, x


def _lon_ticks_and_labels(lon_vals: np.ndarray) -> Tuple[list, list]:
    """Choose reasonable lon ticks/labels whether coords are [-180,180] or [0,360]."""
    lo, hi = float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals))
    if hi <= 180.0:
        ticks = [-180, -120, -60, 0, 60, 120, 180]
    else:
        ticks = [0, 60, 120, 180, 240, 300, 360]
    return ticks, ticks


def _lat_ticks_and_labels(lat_vals: np.ndarray) -> Tuple[list, list]:
    ticks = [-90, -60, -30, 0, 30, 60, 90]
    return ticks, ticks


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ffmpeg_pngs_to_mp4(png_pattern: str, mp4_path: Path, fps_in: int, fps_out: int) -> None:
    """Build an mp4 from a PNG sequence. Requires ffmpeg in PATH."""
    try:
        # -y to overwrite; -pix_fmt yuv420p ensures wide compatibility
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps_in),
            "-i", png_pattern,
            "-r", str(fps_out),
            "-pix_fmt", "yuv420p",
            str(mp4_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[MP4] built -> {mp4_path}")
    except FileNotFoundError:
        print("[WARN] ffmpeg not found; skipping MP4 build.")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg failed: {e}. Command: {' '.join(cmd)}")


def _maybe_cartopy():
    if not DO_ORTHOGRAPHIC:
        return None, None
    try:
        import cartopy.crs as ccrs
        return True, ccrs
    except Exception:
        print("[INFO] Cartopy not available; orthographic animation disabled.")
        return False, None


# ----------------------
# DATA LOADING
# ----------------------

def load_timeseries(raw_dir: Path, pattern: str, start: int, end: Optional[int], step: int) -> xr.Dataset:
    files = sorted(raw_dir.glob(pattern))
    if end is None:
        sel = files[start::step]
    else:
        sel = files[start:end:step]
    if len(sel) == 0:
        raise FileNotFoundError(f"No files found in {raw_dir} with pattern '{pattern}' and slice {start}:{end}:{step}")
    print(f"[INFO] Reading {len(sel)} snapshots from {raw_dir} (pattern={pattern})")
    # Each file has time=1; concat along a synthetic time
    ds = xr.open_mfdataset(sel, combine="nested", concat_dim="time", parallel=False)
    return ds


# ----------------------
# PLOTTING
# ----------------------

def plot_time_mean(ds: xr.Dataset, var_left: str, var_right: str, lev_idx: int, fig_dir: Path,
                   vminmax_left=None, vminmax_right=None, cmap=CMAP, style=STYLE):
    tname, zname, yname, xname = _coord_names(ds)
    with plt.style.context(style):
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))
        lon = ds[xname].values
        lat = ds[yname].values
        # Left
        L = ds[var_left].isel({zname: lev_idx}).mean(dim=tname)  # [lat,lon]
        im0 = axes[0].contourf(lon, lat, L, levels=31, cmap=cmap,
                               vmin=None if vminmax_left is None else vminmax_left[0],
                               vmax=None if vminmax_left is None else vminmax_left[1])
        xt, xl = _lon_ticks_and_labels(lon); yt, yl = _lat_ticks_and_labels(lat)
        axes[0].set_xticks(xt); axes[0].set_xticklabels(xl)
        axes[0].set_xlabel("Longitude (deg)", weight="bold")
        axes[0].set_yticks(yt); axes[0].set_yticklabels(yl)
        axes[0].set_ylabel("Latitude (deg)", weight="bold")
        c0 = plt.colorbar(im0, ax=axes[0]); c0.set_label(f"{var_left}", weight="bold")

        # Right
        R = ds[var_right].isel({zname: lev_idx}).mean(dim=tname)
        im1 = axes[1].contourf(lon, lat, R, levels=31, cmap=cmap,
                               vmin=None if vminmax_right is None else vminmax_right[0],
                               vmax=None if vminmax_right is None else vminmax_right[1])
        axes[1].set_xticks(xt); axes[1].set_xticklabels(xl)
        axes[1].set_xlabel("Longitude (deg)", weight="bold")
        axes[1].set_yticks(yt); axes[1].set_yticklabels(yl)
        c1 = plt.colorbar(im1, ax=axes[1]); c1.set_label(f"{var_right}", weight="bold")

        plt.suptitle(f"Time-Mean {var_left} vs {var_right} @ level {lev_idx}", weight="bold")
        plt.tight_layout()
        out = fig_dir / f"time_mean_{var_left}-{var_right}_lev{lev_idx:02d}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[FIG] saved -> {out}")


def plot_frames_and_video(ds: xr.Dataset, var_left: str, var_right: str, lev_idx: int, fig_dir: Path,
                          basename: str, vminmax_left=None, vminmax_right=None,
                          cmap=CMAP, style=STYLE, make_video=MAKE_VIDEOS,
                          fps_in=FRAMERATE_IN, fps_out=FRAMERATE_OUT):
    tname, zname, yname, xname = _coord_names(ds)
    lon = ds[xname].values
    lat = ds[yname].values
    T = ds.sizes[tname]

    xt, xl = _lon_ticks_and_labels(lon); yt, yl = _lat_ticks_and_labels(lat)

    for it in range(T):
        with plt.style.context(style):
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))
            L = ds[var_left].isel({tname: it, zname: lev_idx})
            R = ds[var_right].isel({tname: it, zname: lev_idx})
            im0 = axes[0].contourf(lon, lat, L, levels=31, cmap=cmap,
                                   vmin=None if vminmax_left is None else vminmax_left[0],
                                   vmax=None if vminmax_left is None else vminmax_left[1], extend="both")
            axes[0].set_xticks(xt); axes[0].set_xticklabels(xl)
            axes[0].set_xlabel("Longitude (deg)", weight="bold")
            axes[0].set_yticks(yt); axes[0].set_yticklabels(yl)
            axes[0].set_ylabel("Latitude (deg)", weight="bold")
            c0 = plt.colorbar(im0, ax=axes[0]); c0.set_label(f"{var_left}", weight="bold")

            im1 = axes[1].contourf(lon, lat, R, levels=31, cmap=cmap,
                                   vmin=None if vminmax_right is None else vminmax_right[0],
                                   vmax=None if vminmax_right is None else vminmax_right[1], extend="both")
            axes[1].set_xticks(xt); axes[1].set_xticklabels(xl)
            axes[1].set_xlabel("Longitude (deg)", weight="bold")
            axes[1].set_yticks(yt); axes[1].set_yticklabels(yl)
            c1 = plt.colorbar(im1, ax=axes[1]); c1.set_label(f"{var_right}", weight="bold")

            ts = np.array(ds[tname].values).astype("datetime64[ns]") if np.issubdtype(ds[tname].dtype, np.datetime64) else None
            tlabel = (np.datetime_as_string(ts[it], unit="h") if ts is not None else f"t={it}")
            plt.suptitle(f"{var_left} vs {var_right} @ level {lev_idx} — {tlabel}", weight="bold")
            plt.tight_layout()

            png = fig_dir / f"{basename}_lev{lev_idx:02d}_{it:04d}.png"
            fig.savefig(png, dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"[SEQ] saved PNGs -> {fig_dir}/{basename}_lev{lev_idx:02d}_####.png")

    if make_video:
        mp4 = fig_dir / f"{basename}_lev{lev_idx:02d}.mp4"
        _ffmpeg_pngs_to_mp4(str(fig_dir / f"{basename}_lev{lev_idx:02d}_%04d.png"), mp4, fps_in, fps_out)


def plot_orthographic_series(ds: xr.Dataset, var: str, lev_idx: int, fig_dir: Path):
    ok, ccrs = _maybe_cartopy()
    if not ok:
        return
    tname, zname, yname, xname = _coord_names(ds)
    T = ds.sizes[tname]

    for it in range(T):
        # Simple camera pan along time
        frac = it / max(T - 1, 1)
        lon0 = ORTHO_VIEWPATH["lon_start"] + frac * ORTHO_VIEWPATH["lon_sweep"]
        lat0 = ORTHO_VIEWPATH["lat_start"] + frac * ORTHO_VIEWPATH["lat_sweep"]

        with plt.style.context(STYLE):
            proj = ccrs.Orthographic(lon0, lat0)
            ax = plt.subplot(1, 1, 1, projection=proj, facecolor="gray")
            ax.set_global(); ax.coastlines(linewidth=1.0)
            x_im = ds[var].isel({tname: it, zname: lev_idx}).plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=CMAP,
                vmin=None, vmax=None,
                add_colorbar=True,
                cbar_kwargs={"label": f"{var}"}
            )
            plt.title("")
            plt.tight_layout()
            png = fig_dir / f"ortho_{var}_lev{lev_idx:02d}_{it:04d}.png"
            plt.savefig(png, dpi=300, bbox_inches="tight")
            plt.close()
    print(f"[SEQ] saved PNGs -> {fig_dir}/ortho_{var}_lev{lev_idx:02d}_####.png")
    if MAKE_VIDEOS:
        mp4 = fig_dir / f"ortho_{var}_lev{lev_idx:02d}.mp4"
        _ffmpeg_pngs_to_mp4(str(fig_dir / f"ortho_{var}_lev{lev_idx:02d}_%04d.png"), mp4, FRAMERATE_IN, FRAMERATE_OUT)


# ----------------------
# MAIN
# ----------------------

def main():
    _ensure_dir(FIG_DIR)
    ds = load_timeseries(RAW_DIR, GLOB_PATTERN, FILE_START, FILE_END, FILE_STEP)

    # Quick sanity: ensure variables exist
    for v in (VAR_LEFT, VAR_RIGHT):
        if v not in ds.data_vars:
            raise KeyError(f"Variable '{v}' not found. Available: {list(ds.data_vars)}")

    # Time-mean maps and frame sequences per level
    for lev in LEVELS_TO_PLOT:
        plot_time_mean(
            ds, VAR_LEFT, VAR_RIGHT, lev, FIG_DIR,
            vminmax_left=VMIN_VMAX_LEFT, vminmax_right=VMIN_VMAX_RIGHT
        )
        plot_frames_and_video(
            ds, VAR_LEFT, VAR_RIGHT, lev, FIG_DIR,
            basename=f"{VAR_LEFT}-{VAR_RIGHT}",
            vminmax_left=VMIN_VMAX_LEFT, vminmax_right=VMIN_VMAX_RIGHT
        )

    # Optional globe animation
    if DO_ORTHOGRAPHIC:
        plot_orthographic_series(ds, ORTHO_VAR, ORTHO_LEVEL, FIG_DIR)

    # Close files
    ds.close()
    print("[DONE] All figures/videos saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
