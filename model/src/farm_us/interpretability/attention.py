"""Temporal attention analysis (paper Figure 9).

We capture self-attention matrices from selected transformer blocks (default 8 &
16), then reduce the spatial-temporal token attention to a ``T×T`` month→month
matrix by averaging over heads and over spatial (h,w) query/key positions.

Axis convention (documented): entry ``A[s, d]`` = attention that source month
``s`` (query) pays to target month ``d`` (key). The "incoming"/receiving score
of a month is the column sum ``A[:, d].sum()``.

Two paths:
  - :func:`temporal_from_attention` reduces a raw attention tensor you already
    captured (unit-testable, backend-agnostic).
  - :class:`AttentionCapturer` registers hooks on a real Prithvi encoder. Fused/
    flash attention must be disabled to obtain explicit weights; we document how.
"""

from __future__ import annotations

import numpy as np
import torch


def temporal_from_attention(attn: torch.Tensor, n_timesteps: int, has_cls: bool = True) -> np.ndarray:
    """Reduce ``[heads, N, N]`` (or ``[B,heads,N,N]``) attention to ``[T, T]``.

    ``N = (cls?) + T * hw``. Averages heads and batch, drops cls, then block-means
    over the spatial tokens within each timestep.
    """
    a = attn.detach().float()
    if a.dim() == 4:
        a = a.mean(0)  # avg batch
    a = a.mean(0)  # avg heads → [N, N]
    if has_cls:
        a = a[1:, 1:]
    n = a.shape[0]
    if n % n_timesteps != 0:
        raise ValueError(f"token count {n} not divisible by T={n_timesteps}")
    hw = n // n_timesteps
    a = a.reshape(n_timesteps, hw, n_timesteps, hw)
    tt = a.mean(dim=(1, 3))  # [T, T]
    return tt.cpu().numpy()


def receiving_score(tt: np.ndarray) -> np.ndarray:
    """Incoming attention per target month = column sum."""
    return tt.sum(axis=0)


class AttentionCapturer:
    """Hook the attention of selected blocks on a real Prithvi encoder.

    Usage requires the attention op to return weights. For memory-efficient
    (fused/flash) attention set the model to a non-fused path first (see
    docs/TROUBLESHOOTING.md → "interpretation mode"). Hooks are removed on exit.
    """

    def __init__(self, encoder: torch.nn.Module, block_indices: list[int]) -> None:
        self.encoder = encoder
        self.block_indices = block_indices
        self._handles: list = []
        self.captured: dict[int, torch.Tensor] = {}

    def __enter__(self):
        blocks = getattr(self.encoder, "blocks", None)
        if blocks is None:
            raise ValueError("Encoder exposes no `.blocks`; cannot hook attention.")
        for i in self.block_indices:
            attn_mod = getattr(blocks[i], "attn", blocks[i])

            def make_hook(idx):
                def hook(_m, _inp, out):
                    if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                        self.captured[idx] = out[1].detach()
                return hook

            self._handles.append(attn_mod.register_forward_hook(make_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
