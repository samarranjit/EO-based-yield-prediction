"""Auxiliary regression head (Figure 3, "Auxiliary").

The small UPerNet feature (256 ch) is projected to a single channel and
bilinearly upsampled to full resolution. Used only for deep supervision during
train/val (`L = L_main + 0.2 * L_aux`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels: int = 256, out_size: int = 224, final_activation: str = "linear") -> None:
        super().__init__()
        self.out_size = out_size
        self.final_activation = final_activation
        self.project = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, out_size: int | None = None) -> torch.Tensor:
        x = self.project(x)
        size = out_size or self.out_size
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        if self.final_activation == "relu":
            x = F.relu(x)
        return x
