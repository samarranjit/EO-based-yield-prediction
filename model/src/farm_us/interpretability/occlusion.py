"""Occlusion sensitivity by month and by band.

Zero-out (occlude) one timestep or one band at a time and measure the change in
the masked mean prediction. Model-agnostic; runs on the dummy backbone too.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def _masked_mean_pred(model, x, tc, lc, mask) -> float:
    out = model(x, tc, lc, return_aux=False)["main"].float().cpu().numpy()
    m = mask.cpu().numpy() > 0.5
    return float(out[m].mean()) if m.any() else float(out.mean())


@torch.no_grad()
def occlusion_by_month(model, batch, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    x = batch["image"].to(device)
    tc = batch.get("temporal_coords"); lc = batch.get("location_coords")
    tc = tc.to(device) if isinstance(tc, torch.Tensor) else None
    lc = lc.to(device) if isinstance(lc, torch.Tensor) else None
    mask = batch["mask"]
    base = _masked_mean_pred(model, x, tc, lc, mask)
    T = x.shape[2]
    deltas = np.zeros(T, dtype=np.float32)
    for t in range(T):
        xo = x.clone(); xo[:, :, t] = 0.0
        deltas[t] = abs(_masked_mean_pred(model, xo, tc, lc, mask) - base)
    return deltas


@torch.no_grad()
def occlusion_by_band(model, batch, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    x = batch["image"].to(device)
    tc = batch.get("temporal_coords"); lc = batch.get("location_coords")
    tc = tc.to(device) if isinstance(tc, torch.Tensor) else None
    lc = lc.to(device) if isinstance(lc, torch.Tensor) else None
    mask = batch["mask"]
    base = _masked_mean_pred(model, x, tc, lc, mask)
    C = x.shape[1]
    deltas = np.zeros(C, dtype=np.float32)
    for c in range(C):
        xo = x.clone(); xo[:, c] = 0.0
        deltas[c] = abs(_masked_mean_pred(model, xo, tc, lc, mask) - base)
    return deltas
