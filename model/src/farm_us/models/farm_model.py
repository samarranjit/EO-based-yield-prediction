"""FARM-US model: Prithvi encoder → temporal reduce → UPerNet → regression heads.

Returns a dict:
    {"main": [B,1,H,W], "aux": [B,1,H,W] | None}

ONE optional residual branch may be added to the coarse prediction. The two
implementations are ALTERNATIVES, not layers -- exactly one may be enabled:

    main = coarse + DetailRefiner(image, body)                   if cfg.detail_refiner
    main = coarse + ExogenousRefiner(image, soil, weather, body) if cfg.exogenous_refiner
    main = coarse                                                if neither

``ExogenousRefiner`` SUPERSEDES ``DetailRefiner``: it reads the same raw
full-resolution imagery through an identical conv stack and adds soil channels
and weather conditioning on top of it. With ``soil_vars``/``weather_vars`` empty
it is structurally identical to ``DetailRefiner``. It is the branch to use for
new work.

``DetailRefiner`` is kept, unmodified, purely so the BARC ``refiner_only`` runs
(``configs/experiments/barc_refiner*.yaml``, mean pixel pearson_r2 0.465) stay
reproducible from their checkpoints.

Each has its own ``finetune_mode`` ("refiner_only" / "exo_refiner_only") for
training it alone against a frozen base. See models/detail_refiner.py and
models/exogenous_refiner.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .auxiliary_head import AuxiliaryHead
from .detail_refiner import DetailRefiner
from .exogenous_refiner import ExogenousRefiner
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
        n_soil: int = 0,
        n_weather: int = 0,
    ) -> None:
        super().__init__()
        # Checked BEFORE the 600M encoder is built, so a bad config fails in
        # milliseconds rather than after a HF Hub download.
        #
        # The two branches are alternatives, not layers. ExogenousRefiner already
        # contains DetailRefiner's entire raw-imagery residual path (same reshape,
        # same two-conv `pix` stack, same fuse-with-head-body, same zero-init
        # `out`). Enabling both would add two independent image-derived residuals
        # to the same coarse prediction: the sub-token detail signal gets
        # double-counted and its gradient split across two branches that cannot
        # see each other, so neither learns the correction the other is already
        # making.
        if cfg.detail_refiner and cfg.exogenous_refiner:
            raise ValueError(
                "detail_refiner and exogenous_refiner are mutually exclusive -- "
                "ExogenousRefiner already contains DetailRefiner's raw-imagery "
                "residual path. Use exogenous_refiner=true for new work (leave "
                "data.soil_vars/data.weather_vars empty for the image-only "
                "ablation, which is structurally identical to DetailRefiner); "
                "use detail_refiner=true only to reproduce the BARC refiner_only "
                "runs (configs/experiments/barc_refiner*.yaml)."
            )
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

        # Built LAST so the random init of every module above is unchanged.
        self.refiner: DetailRefiner | None = None
        if cfg.detail_refiner:
            self.refiner = DetailRefiner(
                in_channels=in_chans * n_timesteps,
                feat_channels=self.head.body_channels,
                hidden=cfg.refiner_hidden,
            )

        # The alternative to self.refiner (mutual exclusion enforced above), so at
        # most one of the two is ever non-None. Supersedes DetailRefiner by adding
        # soil channels and weather conditioning to the same raw-imagery path --
        # see models/exogenous_refiner.py for why the two covariates enter
        # differently (soil at pixel resolution, weather as spatial FiLM).
        self.exo_refiner: ExogenousRefiner | None = None
        if cfg.exogenous_refiner:
            self.exo_refiner = ExogenousRefiner(
                in_channels=in_chans * n_timesteps,
                feat_channels=self.head.body_channels,
                n_soil=n_soil,
                n_weather=n_weather,
                n_timesteps=n_timesteps,
                hidden=cfg.exo_refiner_hidden,
            )

        if cfg.finetune_mode == "refiner_only":
            if self.refiner is None:
                raise ValueError("finetune_mode=refiner_only requires detail_refiner=true")
            for name, p in self.named_parameters():
                p.requires_grad_(name.startswith("refiner."))
        elif cfg.finetune_mode == "exo_refiner_only":
            if self.exo_refiner is None:
                raise ValueError("finetune_mode=exo_refiner_only requires exogenous_refiner=true")
            for name, p in self.named_parameters():
                p.requires_grad_(name.startswith("exo_refiner."))

    #: finetune_mode -> the ONE child module name that stays trainable / in
    #: train() mode; every other child is frozen and eval()'d. Shared by the
    #: constructor's requires_grad_ loop (via cfg.finetune_mode directly) and
    #: by train() below -- keeping both keyed off this single mapping is what
    #: makes "add a third frozen-base branch later" a one-line change.
    _SOLO_BRANCH_MODES = {"refiner_only": "refiner", "exo_refiner_only": "exo_refiner"}

    def train(self, mode: bool = True):
        super().train(mode)
        solo = self._SOLO_BRANCH_MODES.get(self.cfg.finetune_mode)
        if mode and solo is not None:
            # requires_grad=False freezes WEIGHTS, not BatchNorm running statistics
            # or dropout. In train mode the frozen decoder/head BN stats would drift
            # toward the small fine-tuning set and the coarse prediction would stop
            # matching the loaded checkpoint. Keep everything but the one trainable
            # branch in eval mode.
            for name, module in self.named_children():
                if name != solo:
                    module.eval()
        return self

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
        soil: torch.Tensor | None = None,
        weather: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        feats5d = self.encoder(x, temporal_coords, location_coords)  # list [B,D,T,gh,gw]
        feats2d = [self.reducers[i](f) for i, f in enumerate(feats5d)]
        fused = self.decoder(feats2d)
        # With no branch enabled (the default), this reduces to exactly the
        # original two-line if/else -- same tensor ops, same order -- so plain
        # runs and their checkpoints are unaffected byte for byte. The two
        # branches are mutually exclusive (enforced in __init__), so at most one
        # of the additions below ever fires; the accumulate form is kept because
        # it needs no unreachable-branch assertion to typecheck.
        if self.refiner is None and self.exo_refiner is None:
            main = self.head(fused, out_size=x.shape[-1])
        else:
            coarse, body = self.head(fused, out_size=x.shape[-1], return_body=True)
            residual = coarse
            if self.refiner is not None:
                residual = residual + self.refiner(x, body)
            if self.exo_refiner is not None:
                residual = residual + self.exo_refiner(x, body, soil=soil, weather=weather)
            main = self.head.activate(residual)
        out: dict[str, torch.Tensor | None] = {"main": main, "aux": None}
        if self.use_auxiliary and return_aux:
            aux_feat = self.aux_decoder(feats2d[self.aux_level])
            out["aux"] = self.aux_head(aux_feat, out_size=x.shape[-1])
        return out
