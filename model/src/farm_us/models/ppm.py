"""Pyramid Pooling Module (PSPNet / UPerNet).

Applied to the deepest selected feature. Pools the feature at several bin sizes,
projects each, upsamples back, and concatenates with the input, then fuses with
a 3×3 conv. Default bins ``[1, 2, 3, 6]`` (PAPER_REPLICATION_NOTES §7.7).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bins: list[int] | None = None) -> None:
        super().__init__()
        bins = bins or [1, 2, 3, 6]
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(b),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for b in bins
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(bins) * out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [x]
        for stage in self.stages:
            y = stage(x)
            feats.append(F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(feats, dim=1))
