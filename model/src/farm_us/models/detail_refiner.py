"""Sub-token detail refinement branch.

Why this exists
---------------
Prithvi tokenises 14x14 pixels, so at 30 m one token covers 420 m. RegressionHead
reduces each token to ONE value and bilinearly upsamples x14, so the main map has
only (224/14)^2 = 256 degrees of freedom: nothing finer than a token can be
expressed, however well the model is trained. On BARC this gave a near-flat
surface (predicted std 1.09 vs measured 13.56 bu/ac).

This branch reads the RAW full-resolution imagery -- which still carries the 196
native pixels per token that tokenisation averages away -- fuses it with the
head's 64-channel body features, and predicts a per-pixel residual:

    main = coarse_head_prediction + DetailRefiner(image, head_body_features)

Zero-initialised output
-----------------------
``out`` starts with zero weight and bias, so at step 0 the branch adds exactly
nothing and the model reproduces the checkpoint it was loaded from. It can only
learn to ADD detail. Earlier layers receive gradient once ``out`` takes its first
non-zero step (standard zero-conv behaviour).

No BatchNorm: fine-tuning batches are small (4 on BARC), where batch statistics
are unreliable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailRefiner(nn.Module):
    def __init__(self, in_channels: int, feat_channels: int, hidden: int = 16) -> None:
        super().__init__()
        self.pix = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden + feat_channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        # image [B,C,T,H,W]. Time is stacked into channels for THIS branch only, so
        # phenology is kept. The encoder still receives the 5-D tensor unchanged --
        # the "never reshape to [B,48,H,W]" invariant concerns the encoder input.
        b, c, t, h, w = image.shape
        img = image.reshape(b, c * t, h, w)
        feat = F.interpolate(body, size=(h, w), mode="bilinear", align_corners=False)
        return self.out(self.fuse(torch.cat([self.pix(img), feat], dim=1)))
