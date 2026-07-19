"""Token → 2-D feature-map reshaping and transformer-block selection.

This module is intentionally free of any backbone dependency so it can be unit
tested on synthetic token tensors. It encodes two paper-critical conventions:

1. **Block indexing.** The paper extracts one-based transformer blocks
   ``{8, 16, 24, 32}`` of a 32-block encoder. Prithvi's ``forward_features``
   returns a 0-based list, so these map to indices ``{7, 15, 23, 31}`` and
   index 31 is the *final* block.
2. **Token order.** Prithvi prepends a single CLS token, then lays out patch
   tokens as ``(t h w)`` flattened. We drop CLS and rearrange to
   ``[B, D, T, gh, gw]`` so every downstream module sees an explicit,
   documented 5-D feature.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange


def one_based_to_index(one_based_blocks: list[int], depth: int) -> list[int]:
    """Map paper's 1-based block numbers to 0-based ``forward_features`` indices.

    Raises ``ValueError`` on out-of-range values (e.g. block 33 on a 32-block net).
    """
    idx: list[int] = []
    for b in one_based_blocks:
        if not (1 <= b <= depth):
            raise ValueError(
                f"Block {b} is out of range for a {depth}-block encoder "
                f"(valid one-based range 1..{depth})."
            )
        idx.append(b - 1)
    return idx


def tokens_to_feature_map(
    tokens: torch.Tensor,
    n_timesteps: int,
    has_cls_token: bool = True,
) -> torch.Tensor:
    """``[B, N, D] -> [B, D, T, gh, gw]``.

    ``N`` must equal ``(cls?) + T * gh * gw`` with a square patch grid.
    """
    b, n, d = tokens.shape
    if has_cls_token:
        tokens = tokens[:, 1:, :]
        n -= 1
    if n % n_timesteps != 0:
        raise ValueError(f"Token count {n} not divisible by T={n_timesteps}.")
    per_t = n // n_timesteps
    grid = int(round(math.sqrt(per_t)))
    if grid * grid != per_t:
        raise ValueError(f"Per-timestep token count {per_t} is not a square grid.")
    return rearrange(
        tokens, "b (t h w) d -> b d t h w", t=n_timesteps, h=grid, w=grid
    )
