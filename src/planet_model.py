#!/usr/bin/env python3
"""
PlanetSnapshotForecastDataset
-----------------------------
Loads time-ordered NetCDF snapshots (one time per file) and forms supervised pairs (t -> t+1).

- Select variables (or "ALL") and vertical levels; stack as channels [C,H,W]
- Drop unwanted variables via config
- Optional pre-normalization caching to data/processed for speed
  * Cache is keyed by (selected variables, levels slice) so config changes
    never collide with older cache files.
  * We verify cached tensor channel count and rebuild on mismatch.
- Per-variable normalization (identity, zscore, log_zscore, log1p_zscore, slog_zscore)
- Clear validation logging: available vs selected vs dropped, normalization plan
"""
from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr

from normalization import Normalizer
from config import LOG_EVERY_N_STEPS  # reserved for future progress pulses

logger = logging.getLogger(__name__)


def list_snapshot_files(data_dir: Path, pattern: str) -> List[Path]:
    files = sorted(data_dir.glob(pattern))
    if len(files) < 2:
        raise FileNotFoundError(f"Need at least two files in {data_dir} matching '{pattern}'")
    return files


def _open_and_stack(ds: xr.Dataset, variables: List[str], levels_slice: slice) -> np.ndarray:
    arrays = []
    for var in variables:
        if var not in ds.data_vars:
            raise KeyError(f"Variable '{var}' not found. Available: {list(ds.data_vars.keys())}")
        # Expect [time, level, lat, lon]
        arr = ds[var].isel(time=0, level=levels_slice).values.astype(np.float32)  # [L,H,W]
        if arr.ndim != 3:
            raise ValueError(f"{var} expected [level,lat,lon] after slicing, got {arr.shape}")
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)  # [C,H,W]


def _slice_triplet(sl: slice) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    return (sl.start, sl.stop, sl.step)


def _cache_signature(variables: List[str], levels_slice: slice) -> str:
    """Stable short hash for (variables, levels_slice)."""
    payload = {
        "vars": list(variables),
        "levels": _slice_triplet(levels_slice),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:8]


@dataclass
class PlanetSnapshotForecastDataset(torch.utils.data.Dataset):
    files: List[Path]
    variables: Optional[Union[List[str], str]]   # list, "ALL", or None
    levels_slice: slice
    device: Optional[torch.device]
    normalizer: Normalizer
    processed_dir: Optional[Path] = None
    cache_processed: bool = True
    drop_variables: Optional[List[str]] = None

    # runtime state
    _lat: Optional[np.ndarray] = None
    _lon: Optional[np.ndarray] = None
    _shape: Optional[Tuple[int, int, int]] = None  # (C,H,W)

    # resolved config
    selected_variables: Optional[List[str]] = None
    available_variables: Optional[List[str]] = None
    channels_per_var: Optional[Dict[str, int]] = None
    _cache_sig: Optional[str] = None  # cache signature for this selection

    def __post_init__(self):
        logger.info(
            "Initializing dataset: files=%d, variables=%s, levels=%s",
            len(self.files), self.variables, self.levels_slice
        )
        if self.processed_dir and self.cache_processed:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Processed-cache enabled at %s", self.processed_dir)

        # Always load coordinates (even if cache is hit)
        self._ensure_coords_loaded()

        # Discover all available variables (data_vars in first file)
        self._discover_variables()

        # Resolve selection: ALL/None means "use everything available" (minus drops)
        self._resolve_selected_variables()

        # Determine channels per var (levels count after slice)
        self._compute_channels_per_var()

        # Configure normalizer with final var layout
        self.normalizer.configure(self.selected_variables, self.channels_per_var)

        # Build cache signature (after variables + levels are finalized)
        self._cache_sig = _cache_signature(self.selected_variables, self.levels_slice)
        logger.debug("Cache signature for this config: %s", self._cache_sig)

        # Try loading normalization stats; fit later if needed
        loaded = self.normalizer.load()
        if loaded:
            logger.info("Normalization plan (loaded): %s", {
                v: self.normalizer._effective_mode(v) for v in self.selected_variables
            })
        else:
            logger.info("Normalization plan (will fit if needed): %s", {
                v: self.normalizer._effective_mode(v) for v in self.selected_variables
            })

        # Validation report: available vs selected vs dropped
        self._log_validation_report()

    def __len__(self) -> int:
        return len(self.files) - 1

    # -------------- coordinate helpers --------------
    def _read_lat_lon(self, ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        return ds["lat"].values, ds["lon"].values

    def _ensure_coords_loaded(self) -> None:
        if self._lat is not None and self._lon is not None:
            return
        if not self.files:
            raise RuntimeError("Dataset has no files to read coordinates from.")
        first = self.files[0]
        with xr.open_dataset(first) as ds:
            self._lat, self._lon = self._read_lat_lon(ds)
        logger.info(
            "Loaded coordinates from %s: nlat=%d, nlon=%d",
            first.name, self._lat.shape[0], self._lon.shape[0]
        )

    # -------------- variable discovery & selection --------------
    def _discover_variables(self) -> None:
        """Populate available_variables from the first file's data_vars."""
        first = self.files[0]
        with xr.open_dataset(first) as ds:
            self.available_variables = sorted(list(ds.data_vars.keys()))
        logger.info("Available variables in '%s': %s", first.name, self.available_variables)

    def _resolve_selected_variables(self) -> None:
        """Resolve the final variable list after applying 'ALL' and drops."""
        drops = set((self.drop_variables or []))
        if self.variables is None or (isinstance(self.variables, str) and self.variables.upper() == "ALL"):
            sel = [v for v in self.available_variables if v not in drops]
        else:
            # honor the given order, but filter out drops and report missing
            given = list(self.variables)
            missing = [v for v in given if v not in self.available_variables]
            if missing:
                logger.warning("Requested variables not found and will be ignored: %s", missing)
            sel = [v for v in given if (v in self.available_variables and v not in drops)]

        dropped_effective = sorted(list(set(self.available_variables) - set(sel)))
        self.selected_variables = sel
        logger.info("Selected variables (in order): %s", self.selected_variables)
        logger.info("Dropped/unused variables: %s", dropped_effective)

        if not self.selected_variables:
            raise RuntimeError("No variables selected for training after applying 'ALL' and drops.")

    def _compute_channels_per_var(self) -> None:
        """Compute number of channels per variable after applying LEVELS_SLICE."""
        first = self.files[0]
        channels: Dict[str, int] = {}
        with xr.open_dataset(first) as ds:
            for v in self.selected_variables:
                arr = ds[v].isel(time=0, level=self.levels_slice).values  # [L,H,W]
                if arr.ndim != 3:
                    raise ValueError(f"Variable '{v}' expected to yield [L,H,W]; got {arr.shape}")
                channels[v] = int(arr.shape[0])  # L after slice
        self.channels_per_var = channels
        total_C = sum(channels.values())
        logger.info("Channels per variable: %s (C_total=%d)", channels, total_C)

    def _log_validation_report(self) -> None:
        logger.info("VALIDATION REPORT ------------------------------------------------")
        logger.info("  Available variables: %s", self.available_variables)
        logger.info("  Selected variables:  %s", self.selected_variables)
        logger.info("  Dropped variables:   %s", [v for v in self.available_variables if v not in self.selected_variables])
        logger.info("  Normalization plan:  %s", {v: self.normalizer._effective_mode(v) for v in self.selected_variables})
        logger.info("  Channels/var:        %s", self.channels_per_var)
        logger.info("----------------------------------------------------------------")

    # -------------- processed caching --------------
    def _expected_C(self) -> int:
        assert self.channels_per_var is not None
        return int(sum(self.channels_per_var.values()))

    def _processed_path_for(self, raw_path: Path) -> Path:
        """
        Cache file path keyed by (variables, levels_slice).
        Example: hs94.00000_latlon__a1b2c3d4.pt
        """
        assert self.processed_dir is not None and self._cache_sig is not None
        stem = raw_path.stem  # e.g., hs94.00000_latlon
        return self.processed_dir / f"{stem}__{self._cache_sig}.pt"

    def _load_tensor_from_raw(self, path: Path) -> torch.Tensor:
        with xr.open_dataset(path) as ds:
            x = _open_and_stack(ds, self.selected_variables, self.levels_slice)  # [C,H,W]
        return torch.from_numpy(x)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        """
        Load a [C,H,W] tensor from cache if valid; otherwise read from raw and write the keyed cache.
        Also guards against stale/unkeyed cache files by checking channel count.
        """
        expected_C = self._expected_C()

        if self.processed_dir is None or not self.cache_processed:
            logger.debug("Cache disabled; loading raw %s", path.name)
            x = self._load_tensor_from_raw(path)
            if x.shape[0] != expected_C:
                raise RuntimeError(f"Loaded raw tensor channels {x.shape[0]} != expected {expected_C}")
            return x

        pth = self._processed_path_for(path)
        if pth.exists():
            x = torch.load(pth, map_location="cpu")
            if x.shape[0] == expected_C:
                logger.debug("Cache hit: %s", pth.name)
                return x
            else:
                logger.warning(
                    "Cache shape mismatch for %s (got C=%d, expected C=%d). Rebuilding from raw.",
                    pth.name, x.shape[0], expected_C
                )

        # Either cache miss or invalid cache; rebuild from raw
        x = self._load_tensor_from_raw(path)
        if x.shape[0] != expected_C:
            raise RuntimeError(f"Loaded raw tensor channels {x.shape[0]} != expected {expected_C}")
        torch.save(x, pth)
        logger.debug("Cache write: %s", pth.name)
        return x

    # -------------- normalization fitting --------------
    def _fit_normalizer_from_data(self, max_samples: int = 8) -> None:
        """
        Materialize a small, CPU-side iterator of [C,H,W] tensors from the
        *raw* files (pre-normalization) and fit per-channel mean/std in the
        correct transform domain for each variable mode.
        """
        assert self.selected_variables is not None and self.channels_per_var is not None

        # Bound the number of samples to what's available (len(self) = pairs)
        n_avail = max(0, len(self.files) - 1)
        n_take = max(1, min(int(max_samples), n_avail))

        def _iter() -> Iterable[torch.Tensor]:
            for i in range(n_take):
                # Use the "x" side of each (t -> t+1) pair
                yield self._load_tensor(self.files[i])  # [C,H,W] on CPU

        self.normalizer.fit_from_iterator(_iter(), max_batches=n_take)

    def _maybe_fit_normalizer(self):
        # If any selected variable requires stats and stats aren't ready, fit now.
        needs_fit = any(
            self.normalizer._effective_mode(v) in ("zscore", "log_zscore", "log1p_zscore", "slog_zscore")
            for v in self.selected_variables
        )
        if not needs_fit:
            return
        if self.normalizer.is_ready():
            return
        # Try load; if modes/layout mismatch or tiny std present, load() returns False.
        if self.normalizer.load():
            return

        logger.info("Fitting normalization stats from data...")
        self._fit_normalizer_from_data()
        self.normalizer.save()

    # -------------- public properties --------------
    @property
    def nlat(self) -> int:
        return int(self._lat.shape[0])

    @property
    def nlon(self) -> int:
        return int(self._lon.shape[0])

    @property
    def lat(self) -> np.ndarray:
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        return self._lon

    # -------------- dataset protocol --------------
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        x_cpu = self._load_tensor(self.files[idx])
        y_cpu = self._load_tensor(self.files[idx + 1])

        if self._shape is None:
            self._shape = tuple(x_cpu.shape)
            logger.info("First sample shape CxHxW=%s; ensuring normalization readiness.", self._shape)
            self._maybe_fit_normalizer()

        # Apply normalization
        x_cpu = self.normalizer.transform(x_cpu)
        y_cpu = self.normalizer.transform(y_cpu)

        if self.device is not None:
            return x_cpu.to(self.device), y_cpu.to(self.device)
        return x_cpu, y_cpu
