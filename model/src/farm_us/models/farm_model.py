"""FARM-US model: Prithvi encoder → temporal reduce → UPerNet → regression heads.

Returns a dict:
    {"main": [B,1,H,W], "aux": [B,1,H,W] | None}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .auxiliary_head import AuxiliaryHead
from .prithvi_adapter import PrithviAdapter
from .regression_head import RegressionHead
from .temporal_reducer import TemporalFeatureReducer
from .upernet import PaperFaithfulUPerNetDecoder, SmallUPerNetDecoder


class FarmModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        n_timesteps: int = 8,
        chip_size: int = 224,
        in_chans: int = 6,
        use_dummy: bool = False,
        dummy_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.chip_size = chip_size
        self.n_timesteps = n_timesteps

        self.encoder = PrithviAdapter(
            backbone_id=cfg.backbone_id,
            pretrained=cfg.pretrained,
            in_chans=in_chans,
            n_timesteps=n_timesteps,
            out_blocks_one_based=cfg.out_blocks_one_based,
            finetune_mode=cfg.finetune_mode,
            gradient_checkpointing=cfg.gradient_checkpointing,
            use_dummy=use_dummy,
            dummy_embed_dim=dummy_embed_dim,
            expected_embed_dim=cfg.embed_dim,
            depth=cfg.depth,
        )
        d = self.encoder.embed_dim
        n_levels = len(cfg.out_blocks_one_based)

        # One reducer per level (independent params).
        self.reducers = nn.ModuleList(
            [TemporalFeatureReducer(d, n_timesteps, cfg.temporal_reducer) for _ in range(n_levels)]
        )
        self.decoder = PaperFaithfulUPerNetDecoder(
            encoder_dim=d,
            decoder_channels=cfg.decoder_channels,
            n_levels=n_levels,
            ppm_bins=cfg.ppm_bins,
            multiscale=cfg.multiscale_fpn,
        )
        self.head = RegressionHead(
            in_channels=cfg.decoder_channels,
            dropout=cfg.head_dropout,
            out_size=chip_size,
            final_activation=cfg.final_activation,
        )

        self.use_auxiliary = cfg.use_auxiliary
        if self.use_auxiliary:
            if cfg.aux_block_one_based not in cfg.out_blocks_one_based:
                raise ValueError(
                    f"aux_block_one_based={cfg.aux_block_one_based} must be one of "
                    f"out_blocks_one_based={cfg.out_blocks_one_based}"
                )
            self.aux_level = cfg.out_blocks_one_based.index(cfg.aux_block_one_based)
            self.aux_decoder = SmallUPerNetDecoder(d, cfg.aux_channels, cfg.ppm_bins)
            self.aux_head = AuxiliaryHead(cfg.aux_channels, chip_size, cfg.final_activation)

    def parameter_counts(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    def forward(
        self,
        x: torch.Tensor,
        temporal_coords: torch.Tensor | None = None,
        location_coords: torch.Tensor | None = None,
        return_aux: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        feats5d = self.encoder(x, temporal_coords, location_coords)  # list [B,D,T,gh,gw]
        feats2d = [self.reducers[i](f) for i, f in enumerate(feats5d)]
        fused = self.decoder(feats2d)
        main = self.head(fused, out_size=x.shape[-1])
        out: dict[str, torch.Tensor | None] = {"main": main, "aux": None}
        if self.use_auxiliary and return_aux:
            aux_feat = self.aux_decoder(feats2d[self.aux_level])
            out["aux"] = self.aux_head(aux_feat, out_size=x.shape[-1])
        return out
