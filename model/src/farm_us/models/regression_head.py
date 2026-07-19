"""Main convolutional regression head (Figure 3, "Main").

decoder_channels → 512 → 256 → 64 (3×3 conv + BN + ReLU + optional dropout each)
→ 1×1 projection to 1 channel → bilinear upsample to full resolution.
Output activation is linear by default (continuous regression); a non-negative
``relu`` mode is available for inference-only clipping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: tuple[int, ...] = (512, 256, 64),
        dropout: float = 0.1,
        out_size: int = 224,
        final_activation: str = "linear",
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.final_activation = final_activation
        layers: list[nn.Module] = []
        c = in_channels
        for h in hidden:
            layers += [
                nn.Conv2d(c, h, 3, padding=1, bias=False),
                nn.BatchNorm2d(h),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            c = h
        self.body = nn.Sequential(*layers)
        self.project = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, out_size: int | None = None) -> torch.Tensor:
        x = self.body(x)
        x = self.project(x)
        size = out_size or self.out_size
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        if self.final_activation == "relu":
            x = F.relu(x)
        return x
