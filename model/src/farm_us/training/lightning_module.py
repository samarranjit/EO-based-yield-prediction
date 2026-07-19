"""LightningModule wrapping FarmModel + masked loss + metrics + schedule."""

from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch

from ..config import FarmConfig
from ..models.farm_model import FarmModel
from .losses import build_loss
from .metrics import TargetScaler, regression_metrics


class FarmLightningModule(L.LightningModule):
    def __init__(self, cfg: FarmConfig, use_dummy: bool = False, dummy_embed_dim: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"config": cfg.to_container()})
        self.model = FarmModel(
            cfg.model,
            n_timesteps=cfg.data.n_timesteps,
            chip_size=cfg.data.chip_size,
            in_chans=len(cfg.data.band_order),
            use_dummy=use_dummy,
            dummy_embed_dim=dummy_embed_dim,
        )
        self.loss_fn = build_loss(cfg.loss.main, cfg.loss.huber_delta)
        self.aux_weight = cfg.loss.aux_weight
        self.scaler = TargetScaler(mode=cfg.norm.target_scaling)
        self._val_buf: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def set_target_scaler(self, scaler: TargetScaler) -> None:
        self.scaler = scaler

    # --- forward / step --- #
    def forward(self, batch, return_aux: bool = True):
        return self.model(
            batch["image"],
            batch.get("temporal_coords") if self.cfg.model.use_time_embed else None,
            batch.get("location_coords") if self.cfg.model.use_location_embed else None,
            return_aux=return_aux,
        )

    def _step(self, batch, stage: str):
        out = self(batch, return_aux=True)
        target, mask = batch["label"], batch["mask"]
        main_loss, n_valid = self.loss_fn(out["main"], target, mask)
        total = main_loss
        if out["aux"] is not None:
            aux_loss, _ = self.loss_fn(out["aux"], target, mask)
            total = main_loss + self.aux_weight * aux_loss
            self.log(f"{stage}/aux_loss", aux_loss, prog_bar=False, batch_size=target.shape[0])

        if n_valid < 1:
            self.log(f"{stage}/skipped_zero_valid", 1.0, batch_size=target.shape[0])

        self.log(f"{stage}/loss", total, prog_bar=True, batch_size=target.shape[0])
        self.log(f"{stage}/valid_px", n_valid.float(), batch_size=target.shape[0])
        if not torch.isfinite(total):
            self.log(f"{stage}/nan_loss", 1.0, batch_size=target.shape[0])
        return total, out

    def training_step(self, batch, _):
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, _):
        loss, out = self._step(batch, "val")
        # accumulate de-standardized preds for physical metrics
        pred = self.scaler.inverse(out["main"].detach().float().cpu().numpy())
        tgt = self.scaler.inverse(batch["label"].detach().float().cpu().numpy())
        self._val_buf.append((pred, tgt, batch["mask"].detach().cpu().numpy()))
        return loss

    def on_validation_epoch_end(self):
        if not self._val_buf:
            return
        preds = np.concatenate([b[0].ravel() for b in self._val_buf])
        tgts = np.concatenate([b[1].ravel() for b in self._val_buf])
        masks = np.concatenate([b[2].ravel() for b in self._val_buf])
        m = regression_metrics(preds, tgts, masks)
        self.log("val/main_rmse_phys", m["rmse"])
        self.log("val/main_mae_phys", m["mae"])
        self.log("val/main_r2", m["r2"])
        self.log("val/main_pearson_r", m["pearson_r"])
        self._val_buf.clear()

    # --- optim --- #
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        warmup = self.cfg.train.warmup_epochs
        total = self.cfg.train.epochs
        lr0, lrmin = self.cfg.train.lr, self.cfg.train.min_lr

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return (epoch + 1) / warmup
            prog = (epoch - warmup) / max(1, total - warmup)
            cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
            return (lrmin + (lr0 - lrmin) * cos) / lr0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
