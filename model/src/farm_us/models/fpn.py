"""Feature Pyramid Network fusion for a ViT-adapted UPerNet.

Because an isotropic ViT emits the *same* spatial grid (16×16) at every selected
block, the paper-faithful default performs **same-grid** top-down fusion: no
spatial rescaling, just lateral 1×1 projections + top-down additive fusion +
3×3 smoothing (see PAPER_REPLICATION_NOTES §7.5). An optional multi-scale mode
resamples levels to a synthetic pyramid before fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class FPNFusion(nn.Module):
    """Top-down FPN. ``in_channels`` is the (already projected) per-level width.

    ``levels`` are ordered shallow→deep. The deepest may already carry the PPM
    output; the module only handles lateral + top-down + smooth.
    """

    def __init__(self, in_channels: int, out_channels: int, n_levels: int, multiscale: bool = False) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.multiscale = multiscale
        self.smooth = nn.ModuleList([_conv_bn_relu(in_channels, out_channels) for _ in range(n_levels)])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(feats) == self.n_levels, f"expected {self.n_levels} levels, got {len(feats)}"
        # top-down: start from deepest
        out = [None] * self.n_levels  # type: ignore[var-annotated]
        prev = feats[-1]
        out[-1] = self.smooth[-1](prev)
        for i in range(self.n_levels - 2, -1, -1):
            up = F.interpolate(prev, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            merged = feats[i] + up
            out[i] = self.smooth[i](merged)
            prev = merged
        return out  # type: ignore[return-value]
