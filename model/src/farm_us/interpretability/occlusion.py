"""Occlusion sensitivity by month, by band, and by soil/weather variable.

Zero-out (occlude) one timestep, band or exogenous variable at a time and
measure the change in the masked mean prediction. Model-agnostic; runs on the
dummy backbone too.

Every function passes the batch's ``soil``/``weather`` through when present, so
they work on models with models/exogenous_refiner.py enabled (which refuse to
run without those inputs).
"""

from __future__ import annotations

import numpy as np
import torch


def _inputs(batch, device: str) -> dict:
    """Image, coords and (if present) soil/weather from a batch, on ``device``."""
    out = {"x": batch["image"].to(device)}
    for key, name in (("temporal_coords", "tc"), ("location_coords", "lc"),
                      ("soil", "soil"), ("weather", "weather")):
        v = batch.get(key)
        out[name] = v.to(device) if isinstance(v, torch.Tensor) else None
    return out


@torch.no_grad()
def _masked_mean_pred(model, inp: dict, mask) -> float:
    out = model(inp["x"], inp["tc"], inp["lc"], return_aux=False,
                soil=inp["soil"], weather=inp["weather"])["main"].float().cpu().numpy()
    m = mask.cpu().numpy() > 0.5
    return float(out[m].mean()) if m.any() else float(out.mean())


@torch.no_grad()
def occlusion_by_month(model, batch, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    inp = _inputs(batch, device)
    mask = batch["mask"]
    base = _masked_mean_pred(model, inp, mask)
    T = inp["x"].shape[2]
    deltas = np.zeros(T, dtype=np.float32)
    for t in range(T):
        xo = inp["x"].clone(); xo[:, :, t] = 0.0
        deltas[t] = abs(_masked_mean_pred(model, {**inp, "x": xo}, mask) - base)
    return deltas


@torch.no_grad()
def occlusion_by_band(model, batch, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    inp = _inputs(batch, device)
    mask = batch["mask"]
    base = _masked_mean_pred(model, inp, mask)
    C = inp["x"].shape[1]
    deltas = np.zeros(C, dtype=np.float32)
    for c in range(C):
        xo = inp["x"].clone(); xo[:, c] = 0.0
        deltas[c] = abs(_masked_mean_pred(model, {**inp, "x": xo}, mask) - base)
    return deltas


@torch.no_grad()
def occlusion_by_exogenous(model, batch, soil_names=(), weather_names=(),
                           device: str = "cpu") -> dict[str, float]:
    """|change in masked mean prediction| when one soil/weather variable is occluded.

    Occluding sets the variable to 0 -- in z-space that is its TRAINING MEAN, the
    same value a missing input is imputed with, so the delta reads as "how much
    does knowing this variable's actual value move the prediction". Keys are
    ``soil:<name>`` / ``weather:<name>``; all-zero at step 0 of a zero-initialised
    branch, by construction.
    """
    model.eval().to(device)
    inp = _inputs(batch, device)
    mask = batch["mask"]
    base = _masked_mean_pred(model, inp, mask)
    out: dict[str, float] = {}
    if inp["soil"] is not None:
        for s, name in enumerate(soil_names or range(inp["soil"].shape[1])):
            so = inp["soil"].clone(); so[:, s] = 0.0
            out[f"soil:{name}"] = abs(_masked_mean_pred(model, {**inp, "soil": so}, mask) - base)
    if inp["weather"] is not None:
        for w, name in enumerate(weather_names or range(inp["weather"].shape[1])):
            wo = inp["weather"].clone(); wo[:, w] = 0.0
            out[f"weather:{name}"] = abs(_masked_mean_pred(model, {**inp, "weather": wo}, mask) - base)
    return out
