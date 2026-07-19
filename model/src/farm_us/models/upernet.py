"""Paper-faithful ViT-adapted UPerNet decoder.

Consumes the four selected transformer-block features (already temporally
reduced to ``[B, D_enc, gh, gw]``) and produces a single fused 2-D feature map
``[B, decoder_channels, gh, gw]`` for the regression head.

Flow (see Figure 3 / PAPER_REPLICATION_NOTES §2):
    lateral 1×1 proj per level (D_enc → decoder_channels)
    PPM on deepest level
    FPN top-down fusion (same-grid)
    multi-level fusion: concat all levels → 3×3 conv bottleneck
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn import FPNFusion, _conv_bn_relu
from .ppm import PyramidPoolingModule


class PaperFaithfulUPerNetDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_channels: int = 1024,
        n_levels: int = 4,
        ppm_bins: list[int] | None = None,
        multiscale: bool = False,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.decoder_channels = decoder_channels
        # explicit encoder_dim -> decoder_channels projection (1280 -> 1024)
        self.laterals = nn.ModuleList(
            [nn.Conv2d(encoder_dim, decoder_channels, 1) for _ in range(n_levels)]
        )
        self.ppm = PyramidPoolingModule(decoder_channels, decoder_channels, ppm_bins)
        self.fpn = FPNFusion(decoder_channels, decoder_channels, n_levels, multiscale)
        self.fuse = _conv_bn_relu(decoder_channels * n_levels, decoder_channels, k=3)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == self.n_levels
        lat = [self.laterals[i](f) for i, f in enumerate(feats)]
        lat[-1] = lat[-1] + self.ppm(lat[-1])  # PPM enriches deepest level (⊕ in fig)
        fused_levels = self.fpn(lat)
        ref_size = fused_levels[0].shape[-2:]
        aligned = [
            F.interpolate(f, size=ref_size, mode="bilinear", align_corners=False)
            for f in fused_levels
        ]
        return self.fuse(torch.cat(aligned, dim=1))


class SmallUPerNetDecoder(nn.Module):
    """Auxiliary "small UPerNet" (256 ch) attached to a single block feature."""

    def __init__(self, encoder_dim: int, channels: int = 256, ppm_bins: list[int] | None = None) -> None:
        super().__init__()
        self.lateral = nn.Conv2d(encoder_dim, channels, 1)
        self.ppm = PyramidPoolingModule(channels, channels, ppm_bins)
        self.fuse = _conv_bn_relu(channels, channels, k=3)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.lateral(feat)
        x = x + self.ppm(x)
        return self.fuse(x)
