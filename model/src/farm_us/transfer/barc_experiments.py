"""Three paper-inspired BARC transfer experiments (Section 4.4 analogue).

1. ``zero_shot``          : load national FARM checkpoint, no weight updates, eval.
2. ``finetune_from_farm``: init from national FARM checkpoint, fine-tune on BARC
   train years, select on BARC val year, test on held-out BARC year (LOYO).
3. ``train_from_prithvi``: init encoder from original Prithvi weights, train the
   same decoder/head on BARC (optionally heteroscedastic loss).

All three use LOYO on BARC years with strict year separation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import FarmConfig
from ..training.lightning_module import FarmLightningModule


@dataclass
class BarcExperiment:
    name: str  # zero_shot | finetune_from_farm | train_from_prithvi
    national_checkpoint: str | None = None
    freeze_encoder: bool = False
    loss: str = "mse"  # train_from_prithvi may use "huber"/heteroscedastic


def build_experiment_module(
    cfg: FarmConfig,
    exp: BarcExperiment,
    use_dummy: bool = False,
    dummy_embed_dim: int = 64,
) -> FarmLightningModule:
    cfg = _apply_experiment_overrides(cfg, exp)
    if exp.name in ("zero_shot", "finetune_from_farm"):
        if exp.national_checkpoint:
            lm = FarmLightningModule.load_from_checkpoint(  # type: ignore[attr-defined]
                exp.national_checkpoint, cfg=cfg, use_dummy=use_dummy, dummy_embed_dim=dummy_embed_dim
            )
        else:
            lm = FarmLightningModule(cfg, use_dummy=use_dummy, dummy_embed_dim=dummy_embed_dim)
    else:  # train_from_prithvi
        cfg.model.pretrained = True  # original Prithvi weights, fresh decoder/head
        lm = FarmLightningModule(cfg, use_dummy=use_dummy, dummy_embed_dim=dummy_embed_dim)
    return lm


def _apply_experiment_overrides(cfg: FarmConfig, exp: BarcExperiment) -> FarmConfig:
    if exp.name == "zero_shot":
        cfg.model.finetune_mode = "frozen"
        cfg.train.epochs = 0
    elif exp.name == "finetune_from_farm":
        cfg.model.finetune_mode = "full"
    elif exp.name == "train_from_prithvi":
        cfg.model.finetune_mode = "full"
        cfg.loss.main = exp.loss
    else:
        raise ValueError(f"Unknown BARC experiment {exp.name!r}")
    return cfg
