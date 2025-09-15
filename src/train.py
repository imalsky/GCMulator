#!/usr/bin/env python3
"""
Training utilities with spherical losses and clean logging/checkpointing.

This module implements the two loss functions you referenced:

- l2loss_sphere(solver, prd, tar, relative=False, squared=True)
- spectral_l2loss_sphere(solver, prd, tar, relative=False, squared=True)

…where `solver` provides:
  - integrate_grid(x): area-weighted integral over the sphere (cos(lat) weights)
  - sht(x): spherical harmonic transform (optional; raises if not wired)

It also provides `train_loop(...)` which:
  - prints aligned epoch metrics (LR, train/val loss in scientific notation with 4 decimals)
  - tracks & saves best/last checkpoints into `run_dir`
  - writes a compact CSV of metrics in `run_dir / metrics.csv`
  - uses the modern AMP API (`torch.amp.autocast`, `torch.amp.GradScaler`)

If you want spectral loss, implement `SphereOps.sht` in `sphere_ops.py` (currently NotImplemented).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from sphere_ops import SphereOps
from config import LOG_EVERY_N_STEPS, EPOCHS


# ----------------------- Losses (from your example) -----------------------

def l2loss_sphere(solver: SphereOps, prd: torch.Tensor, tar: torch.Tensor,
                  relative: bool = False, squared: bool = True) -> torch.Tensor:
    """
    Area-weighted L2 on the sphere using cos(lat) integration.

    Expected shapes:
      prd, tar: [B,C,H,W] or [C,H,W]
    Returns a scalar tensor.
    """
    diff = prd - tar
    # integrate -> [B,C] (or [C]) then sum over channels -> [B] (or [])
    loss = solver.integrate_grid(diff ** 2, dimensionless=True)
    loss = loss.sum(dim=-1) if loss.dim() >= 2 else loss

    if relative:
        denom = solver.integrate_grid(tar ** 2, dimensionless=True)
        denom = denom.sum(dim=-1) if denom.dim() >= 2 else denom
        loss = loss / torch.clamp(denom, min=1e-30)

    if not squared:
        loss = torch.sqrt(torch.clamp(loss, min=0.0))
    # average over batch if present
    return loss.mean()


def spectral_l2loss_sphere(solver: SphereOps, prd: torch.Tensor, tar: torch.Tensor,
                           relative: bool = False, squared: bool = True) -> torch.Tensor:
    """
    Spectral L2 loss in spherical harmonic space, matching your reference.

    NOTE: Requires `solver.sht` to be implemented. If not, this will raise.
    """
    try:
        coeffs = torch.view_as_real(solver.sht(prd - tar))
    except NotImplementedError as e:
        raise NotImplementedError(
            "SphereOps.sht is not implemented. To use spectral_l2loss_sphere, "
            "wire torch-harmonics' SHT in sphere_ops.py (or switch loss_kind='l2')."
        ) from e

    # coeff power: |a_lm|^2 = Re^2 + Im^2
    coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
    # norm2 over m: m=0 term + 2*sum_{m>=1}
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    # sum over (B?, C?) and l
    loss = torch.sum(norm2, dim=(-1, -2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1, -2))
        loss = loss / torch.clamp(tar_norm2, min=1e-30)

    if not squared:
        loss = torch.sqrt(torch.clamp(loss, min=0.0))
    return loss.mean()


# ----------------------- Helpers -----------------------

def _format_sci4(x: float) -> str:
    return f"{x:.4e}"


def _current_lr(optim: torch.optim.Optimizer) -> float:
    return float(optim.param_groups[0].get("lr", 0.0))


def _unwrap_base_dataset(ds):
    """
    Unwrap torch.utils.data.Subset chains to reach the underlying dataset.
    """
    seen = set()
    cur = ds
    # Avoid infinite loop if something strange exposes .dataset self-cycles
    for _ in range(10):
        if hasattr(cur, "dataset") and cur not in seen:
            seen.add(cur)
            cur = cur.dataset
        else:
            break
    return cur


def _maybe_build_sphere_ops(train_loader: DataLoader) -> Optional[SphereOps]:
    """
    Attempt to construct SphereOps(lat, lon) from the underlying dataset.
    Returns None if not possible.
    """
    try:
        base = _unwrap_base_dataset(train_loader.dataset)
        lat = getattr(base, "lat", None)
        lon = getattr(base, "lon", None)
        if lat is None or lon is None:
            return None
        return SphereOps(lat=lat, lon=lon)
    except Exception:
        return None


# ----------------------- Training Loop -----------------------

def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    run_dir: Path,
    logger,
    epochs: int = EPOCHS,
    use_amp: bool = False,
    loss_kind: str = "l2",   # 'l2' or 'spectral_l2'
    nfuture: int = 0,        # optional autoregressive unrolls
) -> Tuple[float, Path, Path]:
    """
    Trains and checkpoints the model.

    Returns: (best_val, best_ckpt_path, last_ckpt_path)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    best_ckpt = run_dir / "model_best.pt"
    last_ckpt = run_dir / "model_last.pt"

    # Modern AMP API (fixes the deprecation warnings you saw)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    autocast_enabled = (use_amp and device.type == "cuda")

    # Construct SphereOps for spherical losses if possible
    solver = _maybe_build_sphere_ops(train_loader)
    if solver is None and loss_kind.startswith("spectral"):
        raise RuntimeError(
            "Requested spectral loss but could not construct SphereOps from dataset "
            "(lat/lon not found). Use loss_kind='l2' or expose 'lat'/'lon' on the dataset."
        )

    # CSV header
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "epoch_time_s", "lr", "loss_kind", "nfuture"])

    # Pretty header (aligned)
    header = "Epoch |    LR       |   Train Loss    |     Val Loss     |  Time/ep (s)"
    logger.info(header)
    logger.info("-" * len(header))

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        train_acc = 0.0
        n_batches = 0

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                prd = model(xb)
                for _ in range(int(nfuture)):
                    prd = model(prd)

                if solver is not None:
                    if loss_kind == "l2":
                        loss = l2loss_sphere(solver, prd, yb, relative=False)
                    elif loss_kind == "spectral_l2":
                        loss = spectral_l2loss_sphere(solver, prd, yb, relative=False)
                    else:
                        raise ValueError(f"Unknown loss_kind '{loss_kind}'")
                else:
                    # Fallback plain MSE if we couldn't build SphereOps
                    loss = torch.mean((prd - yb) ** 2)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_acc += float(loss.item())
            n_batches += 1

            if LOG_EVERY_N_STEPS and (step % LOG_EVERY_N_STEPS == 0):
                lr_dbg = _current_lr(optimizer)
                logger.debug(
                    f"[epoch {epoch:03d} step {step:05d}] lr={lr_dbg:.3e} train_loss={_format_sci4(loss.item())}"
                )

        if scheduler is not None:
            scheduler.step()

        train_loss = train_acc / max(1, n_batches)

        # Validation
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            v_acc = 0.0
            v_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    prd = model(xb)
                    for _ in range(int(nfuture)):
                        prd = model(prd)

                    if solver is not None:
                        if loss_kind == "l2":
                            vloss = l2loss_sphere(solver, prd, yb, relative=True)
                        elif loss_kind == "spectral_l2":
                            vloss = spectral_l2loss_sphere(solver, prd, yb, relative=True)
                        else:
                            raise ValueError(f"Unknown loss_kind '{loss_kind}'")
                    else:
                        vloss = torch.mean((prd - yb) ** 2)

                    v_acc += float(vloss.item())
                    v_n += 1
            val_loss = v_acc / max(1, v_n)

        epoch_time = time.perf_counter() - epoch_start

        # Log epoch line
        lr = _current_lr(optimizer)
        logger.info(
            f"{epoch:5d} | {lr:10.3e} | {_format_sci4(train_loss):>15} | "
            f"{_format_sci4(val_loss):>15} | {epoch_time:11.3f}"
        )

        # Append CSV
        with csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.8e}", f"{val_loss:.8e}", f"{epoch_time:.3f}", f"{lr:.8e}", loss_kind, int(nfuture)])

        # Checkpoints
        # Save last
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            },
            last_ckpt,
        )

        # Save best
        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "best_val": best_val,
                },
                best_ckpt,
            )

    # If no validation loader: treat last as best for convenience
    if best_epoch < 0:
        best_val = train_loss
        torch.save(torch.load(last_ckpt), best_ckpt)

    return best_val, best_ckpt, last_ckpt
