"""Spectral-band importance from the Conv3D patch-embedding weights.

Reproduces the paper's Figure 10 analysis: the magnitude of the learned patch
embedding weights per input band. Patch weights have shape
``[embed_dim, in_chans, 1, 14, 14]`` (Conv3D, shared across time). We aggregate
the absolute weight over every dimension except the band (in_chans) axis, then
normalize to relative importance.

Caveat (documented): magnitude ≠ causal importance.
"""

from __future__ import annotations

import torch

from ..config import BAND_ORDER


def _find_patch_embed_weight(model: torch.nn.Module) -> torch.Tensor:
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            return m.weight.detach()
    raise ValueError("No Conv3d patch-embedding layer found in model.")


def band_importance(model: torch.nn.Module, band_names=BAND_ORDER) -> dict[str, float]:
    w = _find_patch_embed_weight(model)  # [embed, in_chans, t, ph, pw]
    in_chans = w.shape[1]
    mag = w.abs().float()
    # aggregate over embed, time, and spatial patch dims → per-band scalar
    per_band = mag.sum(dim=(0, 2, 3, 4)).cpu().numpy()[:in_chans]
    total = per_band.sum()
    rel = per_band / total if total > 0 else per_band
    names = list(band_names)[:in_chans]
    return {n: float(v) for n, v in zip(names, rel, strict=False)}
