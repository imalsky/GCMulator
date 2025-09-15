
# Torch-Harmonics Planet Snapshot Forecasting — Project README

> **Purpose:** This document is a single, comprehensive reference for your end‑to‑end pipeline that trains a **Spherical Fourier Neural Operator (SFNO)** using **Torch‑Harmonics** on planetary simulation snapshots stored as NetCDF files. It covers the data layout, reformatting steps, model details, configuration parameters, logging, training behavior, and helper utilities (testing/plots). Keep this README alongside your repo for future runs.

---

## TL;DR

- **Data layout**: Each raw NetCDF file (`data/raw/hs94.XXXXX_latlon.nc`) is a single **time snapshot** with dims `(time=1, level, lat, lon)` and multiple variables (e.g., `temp`).
- **Dataset loader** (`PlanetSnapshotForecastDataset`): Builds **supervised pairs** `(t → t+1)`; selects variables & vertical levels; **stacks levels as channels** `[C, H, W]` (C is total selected levels across variables); **optionally caches pre‑normalized tensors** to `data/processed/<stem>.pt` to accelerate future runs.
- **Normalization**: Configurable **per-channel** (`identity` or `zscore`), with cached stats at `data/processed/normalization_stats.json`.
- **Model**: **SphericalFourierNeuralOperator** (SFNO) from `torch_harmonics.examples.models.sfno` with **scale_factor=1** to keep internal resolution == input resolution (avoids positional‑embedding shape mismatches). Uses **latitude‑aware positional embeddings** by default.
- **Loss**: **Area‑weighted** L2 using `cos(lat)` weights via `SphereOps.integrate_grid` (robust for regular lat‑lon grids).
- **Training**: Single loop with optional **AMP** (`torch.amp.autocast`) when on CUDA, epoch summaries + batch heartbeats. **Subset** of files is selectable via configuration slice.
- **Logging**: Rich logs to console and optional file (`logs/run.log`) with progress for normalization, caching, data loading, and training.
- **Testing** tools save figures under `figures/` and never plot interactively.

---

## Environment & Versions

- **Python**: 3.9 (as in your conda env)
- **Torch‑Harmonics**: 0.8.0 (verified)
- **PyTorch**: Any recent 2.x should work (AMP uses `torch.amp.autocast`)

> **Heads‑up:** In Torch‑Harmonics 0.8.0, the model class lives at `torch_harmonics.examples.models.sfno.SphericalFourierNeuralOperator`. Earlier notebooks often imported `torch_harmonics.examples.sfno` or a `*Net` variant; these are outdated. This repo uses the **current** import path and class names.

---

## Repository Layout

```
project-root/
├─ src/
│  ├─ config.py                 # all config knobs: paths, dataset subset, variables, normalization, model, logging
│  ├─ main.py                   # entry point: wires dataset → model → training; applies subset slice; sets up logging
│  ├─ planet_model.py           # dataset: builds (t→t+1) pairs; stacks [C,H,W]; caches pre-normalized tensors
│  ├─ normalization.py          # Normalizer(mode="identity"|"zscore"), JSON stats cache
│  ├─ sphere_ops.py             # spherical helpers (cos(lat) area weighting)
│  ├─ sfno_model.py             # small builder to configure & return the SFNO model
│  └─ train.py                  # training loop, losses, AMP, logging heartbeats
│
├─ data/
│  ├─ raw/                      # **input** NetCDF snapshots (one timestep per file)
│  └─ processed/                # **intermediate**: cached tensors (*.pt) + normalization_stats.json
│
├─ testing/
│  ├─ raw_inspect.py            # prints dims/vars and saves a level plot for a single snapshot
│  ├─ plot_raw_levels.py        # grid of multiple vertical levels for a single variable
│  ├─ plot_timeseries_point.py  # time series at a lat/lon across snapshots
│  └─ plot_processed_channels.py# visualize channels from cached processed tensor
│
├─ figures/                     # all figures saved by testing scripts
└─ logs/                        # training + pipeline logs (run.log)
```

---

## Raw Data: Organization & Semantics

- **Location**: `data/raw/`
- **Naming convention** (default glob): `hs94.*_latlon.nc` (e.g., `hs94.00192_latlon.nc`)
- **Each file** is one **time snapshot** with **dims**:
  - `time`: 1
  - `level`: e.g., 40
  - `lat`: e.g., 48 (−90 → +90°)
  - `lon`: e.g., 96 (−180 → +180°)
- **Variables** include (typical HS94): `temp`, `theta`, `rho`, `press`, `vel1/2/3`, `vlat`, `vlon`, `Teq`, etc.
- The code is **dataset‑agnostic**: the **only** assumptions are
  - variables you pick are present,
  - each has `(time, level, lat, lon)` dimensions,
  - `time==1` per file.

### Subsetting the File List

Control how many files are used (and at what stride) via `src/config.py`:

```python
# Apply slice to the sorted list of matched files:
FILE_INDEX_START = 0        # inclusive start
FILE_INDEX_END   = None     # exclusive end (None = to the end)
FILE_INDEX_STEP  = 1        # stride (e.g., 5 to take every 5th file)
```

The dataset forms pairs `(file[i] → file[i+1])`, so the **subset must contain ≥2 files**.

---

## Data Reformatting: From NetCDF to SFNO Input

The dataset class `PlanetSnapshotForecastDataset` (in `src/planet_model.py`) performs the following:

1. **Enumerate files**: sorted list from `RAW_DATA_DIR.glob(SNAPSHOT_GLOB)` → apply the subset slice from `config`.
2. **For each snapshot**:
   - Open the NetCDF with Xarray.
   - For each requested variable (e.g., `["temp"]`), extract **`time=0`** and the configured **`levels_slice`** (e.g., `slice(None)` for all levels).
   - Each variable yields an array `[L, H, W]` (levels × lat × lon). These are **concatenated along the channel axis** to produce a single **`[C, H, W]`** tensor, where `C = sum(levels_selected(var_i))`.
3. **Processed cache (optional)**:
   - Save **pre‑normalization** tensors to `data/processed/<stem>.pt` for faster re‑runs (`CACHE_PROCESSED=True`).
   - Reuse cached tensors on subsequent runs (cache hits are logged).
4. **Normalization** (`src/normalization.py`):
   - `mode="identity"`: no‑op.
   - `mode="zscore"`: per‑channel mean/std across the dataset. Stats are stored in `data/processed/normalization_stats.json` and reused if present.
   - Fitting is done lazily when first needed; progress is logged.
5. **Supervised pairs**:
   - `__getitem__(i)` returns `(tensor(files[i]), tensor(files[i+1]))`, both `[C, H, W]`.
   - Tensors are moved to the configured device in `__getitem__`.

**Important:** The tensors are normalized **after** loading from cache (cache stores pre‑normalized tensors). This ensures that changing normalization configs does not require rebuilding the cache.

---

## Torch‑Harmonics & SFNO: What’s What

### Library Overview (relevant parts)

- `torch_harmonics` provides **real-valued spherical harmonic transforms** and utilities for learning on the sphere.
- You will commonly use:
  - `torch_harmonics.RealSHT` and `torch_harmonics.InverseRealSHT` (forward/inverse transforms).
  - The **examples** package for ready‑to‑use models and layers:
    - `torch_harmonics.examples.models.sfno.SphericalFourierNeuralOperator`
    - Supporting layers: spectral convs, positional embeddings, etc.

### SFNO Model: Parameters & Behavior

> Import path used here:  
> `from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator as SFNO`

**Constructor args** (as implemented in v0.8.0 and used by your project):

- `img_size: Tuple[int, int]`  
  `(nlat, nlon)` of your grid. **We pass the actual data size** from the dataset.
- `grid: str = "equiangular"`  
  Input grid type. Keep `"equiangular"` for regular lat‑lon grids.
- `grid_internal: str = "legendre-gauss"`  
  Internal working grid for spectral ops.
- `scale_factor: int = 1`  
  Downsamples internal compute grid: `h = (nlat - 1) // scale_factor + 1`, `w = nlon // scale_factor`.  
  **Use 1** to match internal size to input size and avoid positional‑embedding shape mismatches.
- `in_chans: int`, `out_chans: int`  
  Channels in/out. In your flow, `in_chans == out_chans == total selected levels`.
- `embed_dim: int = 256`  
  Channel width within SFNO blocks. You use **16** for a lightweight model.
- `num_layers: int = 4`  
  Number of SFNO blocks.
- `activation_function: str = "gelu"`  
  `"relu" | "gelu" | "identity"`
- `encoder_layers: int = 1`  
  Depth of the pre‑SFNO 1×1 Conv encoder.
- `use_mlp: bool = True`  
  Enables MLP inside blocks (you often set **False**).
- `mlp_ratio: float = 2.0`  
  Hidden width = `embed_dim * mlp_ratio` when MLP is used.
- `drop_rate: float = 0.0`, `drop_path_rate: float = 0.0`  
  Standard dropout and stochastic depth rates.
- `normalization_layer: str = "none"`  
  `"layer_norm" | "instance_norm" | "none"`
- `hard_thresholding_fraction: float = 1.0`  
  Scales the maximum retained spherical modes: lower values cut high frequencies.
- `residual_prediction: bool = False`  
  If **True**, add a global residual connection (in practice: set True iff `in_chans == out_chans`).
- `pos_embed: str = "none"`  
  `"sequence" | "spectral" | "learnable lat" | "learnable latlon" | "none"`.  
  **You use `"learnable lat"`** (lat‑aware scalar embedding). This requires internal `(h, w)` to match the embedding grid → **why we keep `scale_factor=1`**.
- `bias: bool = False`  
  Bias in convs.

**Internals:**
- First, an encoder maps input `[B, C_in, H, W]` to `[B, embed_dim, H, W]` using 1×1 conv(s).
- Per‑block:
  - Transform to spectral domain via **RealSHT** (forward), apply spectral conv (`SpectralConvS2`), inverse transform, optional skip/MLP/norm/drop-path, etc.
- A decoder maps `[B, embed_dim, H, W]` back to `[B, C_out, H, W]` using 1×1 conv(s).

**Note on older APIs**: Parameters like `spectral_transform` or `operator_type` (e.g., “driscoll‑healy”) are **not** constructor args in v0.8.0’s example SFNO. The SHT‑based pipeline is chosen internally with the provided grid settings.

---

## Losses & Spherical Utilities

### `SphereOps` (src/sphere_ops.py)

- **Area weighting** on regular lat‑lon is accomplished via `cos(lat)` weights.  
  We precompute `w_lat = cos(lat_rad).clip(min=0)` and broadcast over longitude.  
  `integrate_grid(x)`: supports `[C,H,W]` or `[B,C,H,W]`, returning `[C]` or `[B,C]` sums.
- **Why**: equal angle grids have smaller cells near the poles; unweighted L2 would over‑emphasize polar regions.

### Training Loss

- `l2loss_sphere(solver, prd, tar, relative=False, squared=True)`
  - Computes area‑weighted L2 across spatial dims; with `relative=True` it’s scaled by target energy.
  - Returns scalar mean over batch.

> Spectral loss (`spectral_l2loss_sphere`) is left as a TODO. You can wire it by calling `torch_harmonics.RealSHT` in `SphereOps` if you want L2 in spectral space.

---

## Training Loop (src/train.py)

- **Device & AMP**: We auto‑detect the model’s device and use `torch.amp.autocast(device_type=..., enabled=...)` only on CUDA. AMP is configured via `config.USE_AMP`.
- **Batches**: `DataLoader` produces pairs `(inp, tar)`; both are `[B, C, H, W]` once batched.
- **Epochs**: Summary includes elapsed time, average training loss, and relative validation loss (we reuse the same loader for simplicity).
- **Heartbeats**: Every `LOG_EVERY_N_STEPS` steps, a running loss is logged.
- **nfuture**: If set >0, we repeatedly feed the model’s output back as input to train multi‑step forecasting (`prd = model(prd)` inside the inner loop).

---

## Configuration (src/config.py)

Everything is controlled here. Key groups:

### Paths & Globs

```python
PROJECT_ROOT
RAW_DATA_DIR
PROCESSED_DATA_DIR
SNAPSHOT_GLOB          # default: "hs94.*_latlon.nc"
```

### Subset Controls

```python
FILE_INDEX_START = 0
FILE_INDEX_END   = None
FILE_INDEX_STEP  = 1
```

### Variable & Levels

```python
VARIABLES    = ["temp"]
LEVELS_SLICE = slice(None)   # e.g., slice(0, 20) to limit vertical levels
```

### Normalization

```python
NORMALIZATION_MODE = "identity"          # or "zscore"
STATS_CACHE_PATH   = PROCESSED_DATA_DIR / "normalization_stats.json"
CACHE_PROCESSED    = True                # cache pre-normalized tensors
```

### Training

```python
BATCH_SIZE    = 4
EPOCHS        = 10
LEARNING_RATE = 3e-3
WEIGHT_DECAY  = 0.0
USE_AMP       = True
```

### Model (SFNO)

```python
EMBED_DIM     = 16
NUM_LAYERS    = 4
POS_EMBED     = "learnable lat"          # or "sequence" | "spectral" | "learnable latlon" | "none"
SCALE_FACTOR  = 1                        # critical to avoid pos-embed shape mismatch
ACTIVATION    = "gelu"
USE_MLP       = False
NORM_LAYER    = "none"                   # or "layer_norm" | "instance_norm"
HARD_THRESH_F = 1.0
BIAS          = False
```

### Repro & Logging

```python
SEED_GLOBAL = 0
SEED_TRAIN  = 333

LOG_LEVEL         = "INFO"
LOG_TO_FILE       = True
LOG_DIR           = PROJECT_ROOT / "logs"
LOG_FILE          = LOG_DIR / "run.log"
LOG_FORMAT        = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_EVERY_N_STEPS = 100
```

---

## Running the Pipeline

1. Activate your conda env and ensure `torch_harmonics` is installed (`0.8.0` is what you have).
2. Place your raw `.nc` snapshots into `data/raw/` (matching `SNAPSHOT_GLOB`).
3. Tweak `src/config.py` (variables, levels, subset range, model width, etc.).
4. **Run**:
   ```bash
   python src/main.py
   ```
5. Logs will appear in console **and** `logs/run.log` (if enabled).  
   Processed tensors (pre‑normalization) appear in `data/processed/` as `<stem>.pt`.  
   Normalization stats (if `zscore`) appear as `normalization_stats.json`.

---

## Testing Utilities (testing/*.py)

Tools for *offline* inspection and saved visualization under `figures/` (no interactive windows):

- **`raw_inspect.py`**  
  Prints structure and stats, saves a level image for a single snapshot.  
  Configure the GLOBALS at the top (file index, variable, level, etc.).
- **`plot_raw_levels.py`**  
  Saves a grid of multiple vertical levels for a single variable (one snapshot).
- **`plot_timeseries_point.py`**  
  Extracts and plots a time series at a chosen `(lat, lon)` for one variable/level across many snapshots.
- **`plot_processed_channels.py`**  
  Visualizes selected channels from a cached **processed** tensor, matched to the raw snapshot for extents.

> All scripts automatically create `figures/` and write PNGs there. Adjust constants at the top of each script—no CLI flags.

---

## Common Pitfalls & Gotchas

- **Import path drift**: Older tutorials do `from torch_harmonics.examples.sfno import ...` — in **0.8.0**, import **`from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator`**.
- **Positional embedding shape mismatch**: If `scale_factor > 1`, internal `(h, w)` = downsampled dims; positional embeddings (e.g., `"learnable lat"`) are created for those internal dims. Keep **`scale_factor=1`** unless you audit pos‑embed shapes.
- **File shadowing**: Naming scripts `typing.py` or `inspect.py` in your repo **shadows stdlib modules**, breaking imports (NumPy/Matplotlib use them). Avoid naming collisions.
- **Normalization order**: Cache stores **pre‑normalized** tensors. Changing normalization configs doesn’t force recache.
- **Subset size**: You need at least **two files** after slicing to form `(t → t+1)` pairs.
- **Area weights**: On regular lat‑lon, always prefer area‑weighted metrics; unweighted L2 biases toward poles.

---

## Extending the Pipeline

- **Multiple variables**: Set `VARIABLES = ["temp", "press", ...]`; the dataset will stack all selected levels from **each** variable as channels. Ensure `in_chans`/`out_chans` match the new total channel count (the builder does this automatically).
- **Vertical level subset**: Use `LEVELS_SLICE = slice(0, 20)` to reduce channels and model size.
- **Spectral loss**: Wire `SphereOps.sht` to call `torch_harmonics.RealSHT` on `[B,C,H,W]` (batched) and define a spectral L2/relative metric.
- **Multi‑step training**: Set `nfuture > 0` in `train_model` to roll out the model multiple steps per batch.
- **Validation split**: Right now we reuse the same loader for validation. Add a second dataset or sampler to split train/val windows if desired.
- **Dataloader workers**: Kept at `num_workers=0` for simplicity. Increase if I/O becomes a bottleneck and Xarray reading is thread‑safe for your backend.

---

## FAQ

- **Why cos(lat) weights?**  
  On a regular lat‑lon grid, cell areas shrink toward poles by `cos(lat)`. Area‑weighted reductions approximate integrals on the sphere fairly.

- **What does hard_thresholding_fraction do?**  
  It reduces the maximum spherical harmonic degree/order retained by the spectral convs, effectively imposing a frequency cutoff for regularization/performance.

- **Do I need CUDA?**  
  No. The code runs on CPU. AMP is automatically disabled when CUDA is not present.

- **How are channels mapped to levels/variables?**  
  Channels are concatenated in the order of `VARIABLES`, each contributing `len(LEVELS_SLICE)` channels (unless the slice bounds vary by variable). E.g., `VARIABLES=["temp"]` and `40` levels ⇒ `C=40`.

---

## Cheatsheet

- **Run training**  
  `python src/main.py`
- **Limit dataset**  
  In `src/config.py`: `FILE_INDEX_START/END/STEP`
- **Add variable**  
  `VARIABLES = ["temp", "theta"]`
- **Limit levels**  
  `LEVELS_SLICE = slice(0, 10)`
- **Switch normalization**  
  `NORMALIZATION_MODE = "zscore"`
- **Make model bigger**  
  Increase `EMBED_DIM`, `NUM_LAYERS`; optionally `USE_MLP=True`, `mlp_ratio=2.0+` (edit in builder if needed).
- **Logs**  
  `logs/run.log` (if enabled)

---

## Appendix: Minimal Code Flow

1. `main.py`  
   - set up logging → discover files → **apply subset slice** → create dataset (with normalizer & cache) → build model → train
2. `planet_model.py`  
   - `(t) → (t+1)` pairs; stack `[C,H,W]`; cache pre‑norm tensors; load/apply z‑score stats (if enabled)
3. `sfno_model.py`  
   - construct SFNO with consistent `(nlat, nlon)` and `scale_factor=1`
4. `train.py`  
   - area‑weighted L2; AMP (CUDA); epoch logs + heartbeats
5. `sphere_ops.py`  
   - `cos(lat)` area weights and reductions
6. `normalization.py`  
   - fit/load/save per‑channel stats; transform tensors

---

*End of README.*
