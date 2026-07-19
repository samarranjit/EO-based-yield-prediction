"""Collapse the temporal axis of a Prithvi feature ``[B, D, T, gh, gw]`` to a
2-D feature ``[B, D, gh, gw]``.

The paper never states how the T tokens become a single 2-D map, so this is an
explicit, swappable module (see PAPER_REPLICATION_NOTES §7.3). All modes share
the same input/output contract so the decoder is agnostic to the choice.

Modes
-----
- ``mean``          : temporal average pooling (parameter-free ablation).
- ``attention``     : learned per-timestep softmax weighting (attention pooling).
- ``flatten_time``  : stack time into channels then 1×1-project back to D.
                      This is the *default* — it is the faithful home for the
                      paper's "treat time as channels" prose.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalFeatureReducer(nn.Module):
    def __init__(self, embed_dim: int, n_timesteps: int, mode: str = "flatten_time") -> None:
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.n_timesteps = n_timesteps
        if mode == "attention":
            self.attn = nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim // 4, 1),
                nn.GELU(),
                nn.Conv3d(embed_dim // 4, 1, 1),
            )
        elif mode == "flatten_time":
            self.proj = nn.Conv2d(embed_dim * n_timesteps, embed_dim, kernel_size=1)
        elif mode != "mean":
            raise ValueError(f"Unknown temporal reducer mode {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected [B,D,T,gh,gw], got {tuple(x.shape)}")
        b, d, t, gh, gw = x.shape
        if self.mode == "mean":
            return x.mean(dim=2)
        if self.mode == "attention":
            w = torch.softmax(self.attn(x), dim=2)  # [B,1,T,gh,gw]
            return (x * w).sum(dim=2)
        # flatten_time
        flat = x.permute(0, 2, 1, 3, 4).reshape(b, t * d, gh, gw)
        return self.proj(flat)
