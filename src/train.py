#!/usr/bin/env python3
"""
Training loop with spherical losses and mixed precision support.
"""

from __future__ import annotations
import csv  # kept for parity; callers may write CSV logs
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from sphere_ops import SphereOps
from config import LOG_EVERY_N_STEPS, EPOCHS, CLIP_GRAD_NORM, CLIP_GRAD_VALUE


# ------------------- dtype helpers -------------------
_DTYPE_ALIASES = {
    "fp32": torch.float32, "float32": torch.float32, "f32": torch.float32,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16,  "float16": torch.float16, "f16": torch.float16,
}
def _coerce_dtype(x: Union[str, torch.dtype, None]) -> torch.dtype:
    if isinstance(x, torch.dtype):
        return x
    if isinstance(x, str):
        k = x.strip().lower()
        if k in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[k]
        raise ValueError(f"Unknown dtype string: {x!r}")
    return torch.float32

def _maybe_cast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t if t.dtype is dtype else t.to(dtype=dtype)


# ------------------- losses -------------------
def l2loss_sphere(
    solver: SphereOps,
    prd: torch.Tensor,
    tar: torch.Tensor,
    relative: bool = False,
    squared: bool = True,
) -> torch.Tensor:
    """
    Area-weighted L2 loss on the sphere using solver.integrate_grid.
    Expects [N, C, H, W]. Reductions accumulate in float32 for stability.

    - If relative=False: ∫ (prd - tar)^2 dA   (absolute)
    - If relative=True : ∫ (prd - tar)^2 dA / ∫ tar^2 dA   (relative)
    Channel reduction: sum over channels before batch mean (matches reference).
    """
    prd_fp32 = prd.to(dtype=torch.float32)
    tar_fp32 = tar.to(dtype=torch.float32)

    diff2 = (prd_fp32 - tar_fp32) ** 2
    # Integrate over (H, W); typical return shape: [N, C]
    num = solver.integrate_grid(diff2, dimensionless=True).sum(dim=-1)

    if relative:
        den = solver.integrate_grid(tar_fp32 ** 2, dimensionless=True).sum(dim=-1)
        num = num / torch.clamp(den, min=1e-30)

    out = num
    if not squared:
        out = torch.sqrt(torch.clamp(out, min=0.0))

    # Final batch reduction
    return out.mean()


def spectral_l2loss_sphere(
    solver: SphereOps,
    prd: torch.Tensor,
    tar: torch.Tensor,
    relative: bool = False,
    squared: bool = True,
) -> torch.Tensor:
    """
    Spectral L2 loss in spherical-harmonic space (requires SphereOps.sht()).
    """
    prd_fp32 = prd.to(dtype=torch.float32)
    tar_fp32 = tar.to(dtype=torch.float32)

    try:
        coeffs = torch.view_as_real(solver.sht(prd_fp32 - tar_fp32))  # [..., l, m, 2]
    except NotImplementedError:
        raise NotImplementedError("Spectral loss requires SphereOps.sht implementation")

    # |a_lm|^2 = Re^2 + Im^2; fold m>0 by factor 2
    coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2       # [..., l, m]
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)  # [..., l]
    num = torch.sum(norm2, dim=(-1, -2))  # sum over l (and channels if present)

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar_fp32))
        tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        den = torch.sum(tar_norm2, dim=(-1, -2))
        num = num / torch.clamp(den, min=1e-30)

    out = num
    if not squared:
        out = torch.sqrt(torch.clamp(out, min=0.0))

    return out.mean()


# ------------------- criterion builders -------------------
def make_spherical_mse(
    nlat: int,
    nlon: int,
    *,
    relative: bool = False,
    squared: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Build area-weighted MSE over an equiangular grid:
      lat ∈ [-90, 90] (inclusive), nlat points
      lon ∈ [0, 360) (uniform steps), nlon points
    """
    lat = np.linspace(-90.0, 90.0, nlat, dtype=np.float64)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
    solver = SphereOps(lat=lat, lon=lon)

    def criterion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return l2loss_sphere(solver, pred, target, relative=relative, squared=squared)
    return criterion


def make_spectral_mse(
    nlat: int,
    nlon: int,
    *,
    relative: bool = False,
    squared: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Spectral MSE criterion (requires SphereOps.sht())."""
    lat = np.linspace(-90.0, 90.0, nlat, dtype=np.float64)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
    solver = SphereOps(lat=lat, lon=lon)

    def criterion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return spectral_l2loss_sphere(solver, pred, target, relative=relative, squared=squared)
    return criterion


def train_loop(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    val_criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    scheduler: Optional[object],
    device: torch.device,
    run_dir: Path,
    logger,
    epochs: int,
    use_amp: bool,
    dtype: Union[str, torch.dtype] = torch.float32,
) -> Tuple[float, Path, Path]:
    """
    Full training loop with:
      - autocast AMP
      - NaN/Inf loss guard
      - configurable gradient clipping (norm/value) via config.py
      - optional distinct validation criterion (e.g., relative L2)
    """
    # ---- setup ----
    dtype = _coerce_dtype(dtype)
    model = model.to(device=device)
    device_type = "cuda" if getattr(device, "type", None) == "cuda" else "cpu"

    if use_amp and dtype in (torch.float16, torch.bfloat16):
        autocast_dtype = dtype
        from torch import amp  # local import to avoid global side effects
        scaler = amp.GradScaler("cuda", enabled=(device_type == "cuda" and dtype is torch.float16))
        amp_enabled = True
    else:
        autocast_dtype = torch.float32
        from torch import amp
        scaler = amp.GradScaler("cuda", enabled=False)
        amp_enabled = False

    logger.info(
        f"Train precision: autocast={autocast_dtype}, "
        f"grad_scaler={'on' if scaler.is_enabled() else 'off'}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = run_dir / "ckpt_best.pt"
    last_ckpt = run_dir / "ckpt_last.pt"

    best_val = float("inf")
    best_epoch = -1

    # ---- epochs ----
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        num_batches = 0

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = _maybe_cast(xb.to(device, non_blocking=True), dtype)
            yb = _maybe_cast(yb.to(device, non_blocking=True), dtype)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=amp_enabled):
                pred = model(xb)
                loss = criterion(pred, yb)

            # Guard against pathological batches
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss; skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                # AMP path
                scaler.scale(loss).backward()
                # Unscale so clipping sees real grads
                scaler.unscale_(optimizer)

                # --- CONFIG-DRIVEN CLIPPING ---
                if CLIP_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                if CLIP_GRAD_VALUE is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=CLIP_GRAD_VALUE)
                # --------------------------------

                scaler.step(optimizer)
                scaler.update()
            else:
                # FP32 path
                loss.backward()

                # --- CONFIG-DRIVEN CLIPPING ---
                if CLIP_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                if CLIP_GRAD_VALUE is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=CLIP_GRAD_VALUE)
                # --------------------------------

                optimizer.step()

            running_loss += loss.detach().float().item()
            num_batches += 1

            if LOG_EVERY_N_STEPS and (step % LOG_EVERY_N_STEPS == 0):
                logger.info(f"  step {step:6d} | train_loss={running_loss/num_batches: .4e}")

        train_loss = running_loss / max(1, num_batches)

        # ---- validation ----
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_batches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = _maybe_cast(xb.to(device, non_blocking=True), dtype)
                    yb = _maybe_cast(yb.to(device, non_blocking=True), dtype)
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=amp_enabled):
                        pred = model(xb)
                        vloss = (val_criterion or criterion)(pred, yb)
                    val_running += vloss.detach().float().item()
                    val_batches += 1
            val_loss = val_running / max(1, val_batches)
        else:
            val_loss = float("nan")

        # ---- scheduler ----
        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                try:
                    scheduler.step(val_loss)
                except Exception:
                    pass

        # ---- checkpoints ----
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "autocast_dtype": str(autocast_dtype),
            "input_target_dtype": str(dtype),
        }
        torch.save(state, last_ckpt)

        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            state["best_val"] = best_val
            torch.save(state, best_ckpt)

        dt = time.time() - t0
        lr = optimizer.param_groups[0].get("lr", float("nan"))
        logger.info(f"{epoch:6d} | {lr: .3e} | {train_loss: .4e} | {val_loss: .4e} | {dt:7.1f}")

    if best_epoch < 0:
        st = torch.load(last_ckpt, map_location="cpu")
        best_val = st.get("train_loss", float("nan"))
        torch.save(st, best_ckpt)

    return best_val, best_ckpt, last_ckpt
