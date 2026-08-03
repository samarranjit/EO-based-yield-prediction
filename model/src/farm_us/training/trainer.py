"""Trainer assembly: Lightning Trainer with the memory-management ladder,
checkpointing (val-only selection), CSV/TensorBoard/optional W&B logging.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ..config import FarmConfig
from ..data.splits import FoldSpec
from .callbacks import GradNormCallback, ProvenanceCallback, ThroughputMemoryCallback


def build_trainer(
    cfg: FarmConfig,
    fold: FoldSpec,
    out_dir: str,
    norm_stats: dict,
    manifest_fp: str = "n/a",
) -> L.Trainer:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    loggers = [CSVLogger(save_dir=str(out), name="csv")]
    try:
        from lightning.pytorch.loggers import TensorBoardLogger

        loggers.append(TensorBoardLogger(save_dir=str(out), name="tb"))
    except Exception:
        pass
    if cfg.train.log_wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger

            loggers.append(WandbLogger(project="farm-us", save_dir=str(out)))
        except Exception:
            pass

    ckpt = ModelCheckpoint(
        dirpath=str(out / "checkpoints"),
        filename="farm-{epoch:03d}-{val_main_rmse_phys:.4f}",
        monitor=cfg.train.ckpt_monitor,
        mode=cfg.train.ckpt_mode,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    # save_last=True above only refreshes when this ModelCheckpoint's OWN monitored
    # metric improves that epoch -- during a long plateau (our real run went 43
    # epochs without a new best) it never fires, leaving no recoverable weights
    # past the last-best epoch even though training kept going. This one is keyed
    # on recency (monitor=None) instead, so it can't stall the same way: one
    # rolling, resumable snapshot, overwritten every N epochs. (Lightning requires
    # save_top_k in {0, 1, -1} when monitor=None; -1 would keep every periodic
    # snapshot ever made, unbounded -- 1 is the efficient choice.)
    periodic_ckpt = ModelCheckpoint(
        dirpath=str(out / "checkpoints"),
        filename="farm-periodic-{epoch:03d}",
        monitor=None,
        every_n_epochs=cfg.train.periodic_ckpt_every_n_epochs,
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    callbacks: list[L.Callback] = [
        ckpt,
        periodic_ckpt,
        LearningRateMonitor(logging_interval="epoch"),
        ProvenanceCallback(cfg, fold, out_dir, norm_stats, manifest_fp),
        ThroughputMemoryCallback(),
        GradNormCallback(),
    ]
    if cfg.train.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.ckpt_monitor, mode=cfg.train.ckpt_mode,
                patience=cfg.train.early_stopping_patience,
            )
        )

    strategy = "auto"
    if cfg.train.strategy == "ddp":
        strategy = "ddp"
    elif cfg.train.strategy == "fsdp":
        strategy = "fsdp"

    return L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=strategy,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.grad_accum,
        gradient_clip_val=cfg.train.grad_clip,
        detect_anomaly=cfg.train.detect_anomaly,
        limit_train_batches=cfg.train.limit_train_batches or 1.0,
        limit_val_batches=cfg.train.limit_val_batches or 1.0,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        default_root_dir=str(out),
    )


def effective_batch_size(cfg: FarmConfig) -> int:
    return cfg.train.batch_size * cfg.train.grad_accum * max(1, cfg.train.devices)
