#!/usr/bin/env python3
"""
GPU-resident dataset that loads all snapshots to GPU memory at initialization.
This eliminates CPU<->GPU transfer overhead during training.

Key fixes:
- Robust coercion for `device` (str/torch.device) and `dtype` (str/torch.dtype).
- No reliance on a non-existent Normalizer.transform(output_dtype=...) arg.
- Uses configured dtype for normalization and storage.
- Safe cache signature keyed by (vars, levels_slice, dtype).
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
from config import LOG_EVERY_N_STEPS

logger = logging.getLogger(__name__)

# ---------------------- helpers: dtype/device coercion ----------------------

_DTYPE_ALIASES = {
    "fp32": torch.float32, "float32": torch.float32, "f32": torch.float32,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16,  "float16": torch.float16, "f16": torch.float16,
}

def _coerce_dtype(x: Union[str, torch.dtype, None]) -> torch.dtype:
    if isinstance(x, torch.dtype):
        return x
    if isinstance(x, str):
        key = x.strip().lower()
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
        raise ValueError(f"Unknown dtype string: {x!r}. "
                         f"Valid: {sorted(_DTYPE_ALIASES.keys())}")
    return torch.float32

def _coerce_device(x: Union[str, torch.device, None]) -> torch.device:
    if isinstance(x, torch.device):
        return x
    if isinstance(x, str):
        return torch.device(x)
    return torch.device("cpu")

# ---------------------------- IO / stacking utils ---------------------------

def _open_and_stack(ds: xr.Dataset, variables: List[str], levels_slice: slice) -> np.ndarray:
    """Stack selected variables and levels into a single array [C,H,W]."""
    arrays: List[np.ndarray] = []
    for var in variables:
        if var not in ds.data_vars:
            raise KeyError(f"Variable '{var}' not found in dataset. "
                           f"Available: {list(ds.data_vars.keys())}")
        # Expect [time, level, lat, lon] -> select first time, slice levels -> [L,H,W]
        arr = ds[var].isel(time=0, level=levels_slice).values
        if arr.ndim != 3:
            raise ValueError(f"Variable '{var}' expected [level,lat,lon] after slicing; got {arr.shape}")
        arrays.append(arr.astype(np.float32, copy=False))
    return np.concatenate(arrays, axis=0)  # [C,H,W]

def _cache_signature(variables: List[str], levels_slice: slice, dtype_name: str) -> str:
    """Short stable hash for cache keying."""
    payload = {
        "vars": list(variables),
        "levels": (levels_slice.start, levels_slice.stop, levels_slice.step),
        "dtype": dtype_name,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:8]

# ---------------------------------- Dataset ---------------------------------

@dataclass
class PlanetSnapshotForecastDataset(torch.utils.data.Dataset):
    """
    GPU-resident dataset for planet snapshots.
    Loads all snapshots, normalizes on CPU (to keep peak VRAM lower), then
    moves the normalized tensors to `device` and keeps them in GPU memory.

    Returns (x_t, x_{t+1}) pairs from the in-memory tensor.
    """
    files: List[Path]
    variables: Optional[Union[List[str], str]]
    levels_slice: slice
    device: Union[str, torch.device, None]
    normalizer: Normalizer
    dtype: Union[str, torch.dtype, None]

    processed_dir: Optional[Path] = None
    cache_processed: bool = True
    drop_variables: Optional[List[str]] = None

    # Internals
    _lat: Optional[np.ndarray] = None
    _lon: Optional[np.ndarray] = None
    _gpu_data: Optional[torch.Tensor] = None  # [N, C, H, W] on device
    selected_variables: Optional[List[str]] = None
    available_variables: Optional[List[str]] = None
    channels_per_var: Optional[Dict[str, int]] = None
    _cache_sig: Optional[str] = None

    def __post_init__(self):
        """Initialize dataset and load all data to GPU."""
        # Coerce device/dtype first so later code is type-safe
        self.device = _coerce_device(self.device)
        self.dtype  = _coerce_dtype(self.dtype)

        logger.info(f"Initializing GPU-resident dataset with {len(self.files)} files")

        if self.processed_dir and self.cache_processed:
            self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load coordinates (for plotting etc.)
        self._load_coordinates()

        # Discover & select variables, compute channels
        self._discover_variables()
        self._resolve_selected_variables()
        self._compute_channels_per_var()

        # Configure normalizer layout (var order + per-var channels)
        assert self.selected_variables is not None and self.channels_per_var is not None
        self.normalizer.configure(self.selected_variables, self.channels_per_var)

        # Build cache signature AFTER selection & levels are finalized
        dtype_str = str(self.dtype).split(".")[-1]  # e.g. "float32"
        self._cache_sig = _cache_signature(self.selected_variables, self.levels_slice, dtype_str)

        # Try load fitted stats; else fit from a small subset and save
        if not self.normalizer.load():
            logger.info("Fitting normalization statistics from data subset...")
            self._fit_normalizer()
            self.normalizer.save()
        else:
            logger.info("Loaded normalization statistics")

        # Load + normalize all snapshots, then move to GPU
        self._load_all_to_gpu()

        # Final summary
        assert self._gpu_data is not None
        n, c, h, w = self._gpu_data.shape
        bytes_total = self._gpu_data.element_size() * self._gpu_data.numel()
        logger.info(f"Dataset ready on {self.device}: N={n}, C={c}, H={h}, W={w} "
                    f"({bytes_total/1e9:.2f} GB)")

    def __len__(self) -> int:
        """Number of forecast pairs (N-1 for N snapshots)."""
        return max(0, len(self.files) - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized (input, target) pair from GPU memory."""
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        assert self._gpu_data is not None
        return self._gpu_data[idx], self._gpu_data[idx + 1]

    # ------------------------------ discovery ------------------------------

    def _load_coordinates(self) -> None:
        """Load latitude and longitude from first file."""
        if not self.files:
            raise RuntimeError("No files provided to dataset.")
        with xr.open_dataset(self.files[0]) as ds:
            self._lat = ds["lat"].values
            self._lon = ds["lon"].values

    def _discover_variables(self) -> None:
        """Discover available variables from the first file."""
        with xr.open_dataset(self.files[0]) as ds:
            self.available_variables = sorted(list(ds.data_vars.keys()))

    def _resolve_selected_variables(self) -> None:
        """Determine which variables to use based on config and drops."""
        assert self.available_variables is not None
        drops = set(self.drop_variables or [])

        if self.variables is None or (isinstance(self.variables, str) and self.variables.upper() == "ALL"):
            sel = [v for v in self.available_variables if v not in drops]
        else:
            given = list(self.variables)  # preserve caller order
            missing = [v for v in given if v not in self.available_variables]
            if missing:
                logger.warning("Requested variables not found and will be ignored: %s", missing)
            sel = [v for v in given if (v in self.available_variables and v not in drops)]

        if not sel:
            raise RuntimeError("No variables selected after applying 'ALL' / drops.")
        self.selected_variables = sel
        logger.info(f"Selected {len(sel)} variables: {sel}")

    def _compute_channels_per_var(self) -> None:
        """Compute number of channels per variable after level slicing."""
        channels: Dict[str, int] = {}
        with xr.open_dataset(self.files[0]) as ds:
            for v in self.selected_variables:
                arr = ds[v].isel(time=0, level=self.levels_slice).values
                if arr.ndim != 3:
                    raise ValueError(f"Variable '{v}' expected [level,lat,lon]; got {arr.shape}")
                channels[v] = int(arr.shape[0])
        self.channels_per_var = channels
        logger.info(f"Total channels: {sum(channels.values())}")

    # ------------------------------- caching -------------------------------

    def _processed_path_for(self, raw_path: Path) -> Path:
        """Cache file path keyed by current selection/levels/dtype."""
        assert self.processed_dir is not None and self._cache_sig is not None
        return self.processed_dir / f"{raw_path.stem}__{self._cache_sig}.pt"

    def _load_tensor(self, path: Path) -> torch.Tensor:
        """
        Load a single snapshot tensor [C,H,W] on CPU in self.dtype.
        Uses keyed cache when enabled.
        """
        # No caching -> always read raw
        if not (self.processed_dir and self.cache_processed):
            with xr.open_dataset(path) as ds:
                arr = _open_and_stack(ds, self.selected_variables, self.levels_slice)
            return torch.from_numpy(arr).to(dtype=self.dtype)

        # With caching
        cache_path = self._processed_path_for(path)
        if cache_path.exists():
            try:
                # NOTE: weights_only may not be supported for plain Tensors in all versions.
                tensor = torch.load(cache_path, map_location="cpu")
                return tensor.to(dtype=self.dtype)
            except Exception as e:
                logger.warning(f"Failed to load cache '{cache_path.name}': {e}; rebuilding.")

        with xr.open_dataset(path) as ds:
            arr = _open_and_stack(ds, self.selected_variables, self.levels_slice)
        tensor = torch.from_numpy(arr).to(dtype=self.dtype)

        try:
            torch.save(tensor, cache_path)
        except Exception as e:
            logger.warning(f"Failed to write cache '{cache_path.name}': {e}")

        return tensor

    # --------------------------- normalization fit --------------------------

    def _fit_normalizer(self) -> None:
        """Fit normalization statistics from a small subset (CPU float32)."""
        def iterator():
            k = min(8, len(self.files))
            for i in range(k):
                # Ensure fitting happens in float32 regardless of training dtype
                yield self._load_tensor(self.files[i]).to(dtype=torch.float32)

        self.normalizer.fit_from_iterator(iterator())

    # -------------------------- bulk load to device -------------------------

    def _load_all_to_gpu(self) -> None:
        """
        Load, normalize, and stack all snapshots; keep final tensor on device.

        We normalize on CPU to keep peak VRAM lower, then move the final stacked
        tensor to the target device.
        """
        all_tensors: List[torch.Tensor] = []
        n_files = len(self.files)

        for i, path in enumerate(self.files):
            # Load [C,H,W] on CPU with the configured dtype
            cpu_tensor = self._load_tensor(path)  # dtype already self.dtype

            # Apply normalization (works with [C,H,W]); stays on CPU
            normed = self.normalizer.transform(cpu_tensor)

            # Ensure dtype exactly matches the configured training dtype
            normed = normed.to(dtype=self.dtype, copy=False)
            all_tensors.append(normed)

            # Progress logging
            if LOG_EVERY_N_STEPS > 0 and ((i + 1) % LOG_EVERY_N_STEPS == 0 or (i + 1) == n_files):
                logger.info(f"  Loaded {i + 1}/{n_files} snapshots")

        # Stack on CPU first, then send one big tensor to the device
        stacked = torch.stack(all_tensors, dim=0)  # [N,C,H,W] on CPU
        self._gpu_data = stacked.to(self.device, non_blocking=True)

    # ------------------------------ properties ------------------------------

    @property
    def nlat(self) -> int:
        return int(self._lat.shape[0]) if self._lat is not None else 0

    @property
    def nlon(self) -> int:
        return int(self._lon.shape[0]) if self._lon is not None else 0

    @property
    def lat(self) -> np.ndarray:
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        return self._lon
