"""Fold orchestration: compute train-only stats → train → select-by-val → evaluate.

Synthetic mode (``use_dummy=True``) exercises the entire pipeline on CPU with the
dummy backbone. Real mode wires the manifest-driven dataset (requires imagery).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import FarmConfig, save_resolved_config
from ..data.dataset import FarmDataModule
from ..data.normalization import (
    NormStats,
    find_norm_stats,
    official_prithvi_stats,
    target_scaler_from_values,
)
from ..data.splits import load_split_map, make_fold
from ..utils.logging import get_logger
from ..utils.reproducibility import seed_everything
from .lightning_module import FarmLightningModule

logger = get_logger("farm_us.run")


def resolve_fold(cfg: FarmConfig):
    smap = None
    if cfg.split.policy == "explicit_map":
        try:
            smap = load_split_map(cfg.split.split_map_path)
        except Exception:
            smap = None
    return make_fold(cfg.split.test_year, cfg.data.years, cfg.split.val_years,
                     policy=cfg.split.policy, split_map=smap)


def compute_fold_stats(cfg: FarmConfig, dm: FarmDataModule) -> NormStats:
    """Train-fold-only normalization + target scaler.

    In synthetic mode we derive stats from the synthetic *train* dataset only.
    In official mode we use fixed Prithvi band stats. Never touches val/test.
    """
    train_ds = dm.train_ds
    # target scaler from training labels only
    labels = []
    n_bands = len(cfg.data.band_order)
    from ..data.normalization import StreamingBandStats

    sbs = StreamingBandStats(n_bands)
    for i in range(len(train_ds)):
        s = train_ds[i]
        img = s["image"].numpy()  # [C,T,H,W]
        m = s["mask"].numpy()[0] > 0.5
        lab = s["label"].numpy()[0][m]
        labels.append(lab)
        sbs.update([img[b][:, m].ravel() for b in range(n_bands)])
    y = np.concatenate(labels) if labels else np.array([0.0])
    scaler = target_scaler_from_values(y, cfg.norm.target_scaling)

    if cfg.norm.mode == "official_prithvi_statistics":
        stats = official_prithvi_stats(list(range(1)), scaler)
    else:
        mean, std = sbs.finalize()
        stats = NormStats(
            band_mean=mean.tolist(), band_std=std.tolist(), target=scaler.to_dict(),
            mode=cfg.norm.mode, train_years=[2019],
        )
    return stats


def train_fold(cfg: FarmConfig, use_dummy: bool = True, dummy_embed_dim: int = 32, resume_from: str | None = None):
    import lightning as L

    seed_everything(cfg.train.seed)
    fold = resolve_fold(cfg)
    logger.info("Fold: test=%s val=%s train=%s", fold.test_year, fold.val_years, fold.train_years)

    dm = FarmDataModule(cfg, synthetic=use_dummy, n_synth=8)
    dm.setup()
    stats = compute_fold_stats(cfg, dm)
    dm.apply_norm_stats(stats)

    out_dir = Path(cfg.train.output_dir) / cfg.experiment_name / f"test{fold.test_year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, out_dir / "resolved_config.yaml")
    stats.save(out_dir / "norm_stats.json")

    lm = FarmLightningModule(cfg, use_dummy=use_dummy, dummy_embed_dim=dummy_embed_dim)
    lm.set_target_scaler(stats.target_scaler())

    from .trainer import build_trainer

    trainer = build_trainer(cfg, fold, str(out_dir), stats.__dict__, manifest_fp="synthetic")
    if isinstance(trainer, L.Trainer):
        if resume_from:
            logger.info("Resuming from checkpoint: %s", resume_from)
        trainer.fit(lm, dm.train_dataloader(), dm.val_dataloader(), ckpt_path=resume_from)
        logger.info("Best checkpoint: %s", getattr(trainer.checkpoint_callback, "best_model_path", None))
    return lm, dm, stats, out_dir


def evaluate_fold(cfg: FarmConfig, checkpoint: str | None = None, use_dummy: bool = True):
    from ..evaluation.evaluator import evaluate_loader, summarize

    fold = resolve_fold(cfg)
    run_dir = Path(cfg.train.output_dir) / cfg.experiment_name / f"test{fold.test_year}"

    dm = FarmDataModule(cfg, synthetic=use_dummy, n_synth=8)
    dm.setup()

    # Prefer the norm stats the TRAINING run saved. Recomputing them from the
    # train split costs ~1h on real data and can only reproduce (approximately)
    # what is already on disk -- and using the saved file is strictly more
    # correct, since it is bit-identical to what the checkpoint was trained
    # against. Falls back to recomputing when no saved stats exist.
    #
    # Resolution order matters: train_fold OVERWRITES run_dir/norm_stats.json on
    # every launch, so after a second experiment writes into the same fold
    # directory the run-root copy no longer describes an older checkpoint. A
    # copy archived NEXT TO that checkpoint therefore wins -- find_norm_stats
    # walks up from the checkpoint and returns the nearest match, which is also
    # what map_test_errors.py / find_extreme_predictions.py use. Keeping the two
    # consistent means "archive norm_stats.json beside the checkpoint" is a
    # complete answer, not one that only some tools honour.
    stats_path = find_norm_stats(checkpoint) if checkpoint else None
    if stats_path is None and (run_dir / "norm_stats.json").exists():
        stats_path = run_dir / "norm_stats.json"

    if stats_path is not None:
        stats = NormStats.load(stats_path)
        logger.info("Loaded saved norm stats from %s (skipping recompute)", stats_path)
    else:
        logger.info("No saved norm stats found; recomputing from the train split")
        stats = compute_fold_stats(cfg, dm)
    dm.apply_norm_stats(stats)

    if checkpoint:
        lm = FarmLightningModule.load_from_checkpoint(checkpoint, cfg=cfg, use_dummy=use_dummy, dummy_embed_dim=32)
    else:
        lm = FarmLightningModule(cfg, use_dummy=use_dummy, dummy_embed_dim=32)
    scaler = stats.target_scaler()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Evaluating on %s", device)
    df = evaluate_loader(lm.model, dm.test_dataloader(), scaler, device=device)
    # Keyed on the checkpoint's own filename, not a fixed "eval/" path -- otherwise
    # evaluating a later checkpoint (e.g. after resuming training) silently
    # overwrites every plot/CSV from a previous checkpoint's evaluation, with no
    # record anything was ever there. Re-evaluating the SAME checkpoint still
    # overwrites its own prior run, which is correct (identical inputs -> identical
    # output, nothing worth preserving twice).
    ckpt_tag = Path(checkpoint).stem if checkpoint else "no_checkpoint"
    out_dir = Path(cfg.train.output_dir) / cfg.experiment_name / f"test{fold.test_year}" / "eval" / ckpt_tag
    res = summarize(df, out_dir)
    logger.info("Eval (test year %s, checkpoint %s): %s", fold.test_year, ckpt_tag, res.get("global_pixel"))
    return res


def run_loyo(cfg: FarmConfig, use_dummy: bool = True):
    results = {}
    for test_year in cfg.data.years:
        cfg.split.test_year = test_year
        # ensure val year differs
        cfg.split.val_years = [y for y in cfg.split.val_years if y != test_year] or [
            max(y for y in cfg.data.years if y != test_year)
        ]
        try:
            train_fold(cfg, use_dummy=use_dummy)
            results[test_year] = evaluate_fold(cfg, use_dummy=use_dummy)
        except Exception as e:  # keep going across folds
            logger.error("Fold %s failed: %s", test_year, e)
            results[test_year] = {"error": str(e)}
    return results
