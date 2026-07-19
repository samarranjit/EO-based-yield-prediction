"""Masked regression losses.

All losses operate only over *valid* pixels (crop ∧ label ∧ HLS-valid). The
denominator is the number of valid pixels, never H×W. They are safe under mixed
precision, avoid divide-by-zero, and return the valid-pixel count for logging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _prep(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    # compute in fp32 for numerical stability under bf16/amp
    return pred.float(), target.float(), mask


def masked_mse(pred, target, mask):
    pred, target, mask = _prep(pred, target, mask)
    n = mask.sum()
    if n < 1:
        return pred.sum() * 0.0, n
    err = (pred - target) ** 2
    return (err * mask).sum() / n.clamp(min=1), n


def masked_mae(pred, target, mask):
    pred, target, mask = _prep(pred, target, mask)
    n = mask.sum()
    if n < 1:
        return pred.sum() * 0.0, n
    err = (pred - target).abs()
    return (err * mask).sum() / n.clamp(min=1), n


def masked_huber(pred, target, mask, delta: float = 1.0):
    pred, target, mask = _prep(pred, target, mask)
    n = mask.sum()
    if n < 1:
        return pred.sum() * 0.0, n
    err = F.huber_loss(pred, target, reduction="none", delta=delta)
    return (err * mask).sum() / n.clamp(min=1), n


def masked_heteroscedastic(pred_mean, pred_logvar, target, mask):
    """Gaussian NLL with per-pixel predicted variance (for BARC experiment 3).

    L = 0.5 * (exp(-logvar) * (y - mu)^2 + logvar)
    """
    pred_mean, target, mask = _prep(pred_mean, target, mask)
    pred_logvar = pred_logvar.float()
    n = mask.sum()
    if n < 1:
        return pred_mean.sum() * 0.0, n
    inv_var = torch.exp(-pred_logvar)
    nll = 0.5 * (inv_var * (pred_mean - target) ** 2 + pred_logvar)
    return (nll * mask).sum() / n.clamp(min=1), n


def build_loss(name: str, delta: float = 1.0):
    name = name.lower()
    if name == "mse":
        return lambda p, t, m: masked_mse(p, t, m)
    if name == "mae":
        return lambda p, t, m: masked_mae(p, t, m)
    if name == "huber":
        return lambda p, t, m: masked_huber(p, t, m, delta=delta)
    raise ValueError(f"Unknown loss {name!r}")
