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


def stats_chip_indices(n_total: int, max_chips: int | None, seed: int) -> list[int]:
    """Which train-split chip indices the normalization pass should read.

    ``max_chips=None`` (the default) returns every index in order, so the full
    pass is unchanged down to iteration order. Otherwise draw a uniform sample
    WITHOUT replacement and return it sorted -- sorting costs nothing and keeps
    reads roughly sequential across the underlying rasters instead of seeking
    randomly over a multi-gigabyte tile set.
    """
    if max_chips is None or max_chips >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, size=max_chips, replace=False))


def compute_fold_stats(cfg: FarmConfig, dm: FarmDataModule) -> NormStats:
    """Train-fold-only normalization + target scaler.

    In synthetic mode we derive stats from the synthetic *train* dataset only.
    In official mode we use fixed Prithvi band stats. Never touches val/test.

    Reads every train chip unless ``norm.stats_max_chips`` is set; see that
    field for why subsampling is safe and when it is worth using.
    """
    train_ds = dm.train_ds
    # target scaler from training labels only
    labels = []
    n_bands = len(cfg.data.band_order)
    from ..data.normalization import StreamingBandStats

    n_total = len(train_ds)
    indices = stats_chip_indices(n_total, cfg.norm.stats_max_chips, cfg.norm.stats_seed)
    if len(indices) < n_total:
        logger.info(
            "Stats pass: subsampling %d/%d train chips (seed=%d)",
            len(indices), n_total, cfg.norm.stats_seed,
        )
    else:
        logger.info("Stats pass: reading all %d train chips", n_total)

    sbs = StreamingBandStats(n_bands)
    # Progress logging matters here even though it looks cosmetic: this loop is
    # single-threaded and can run for many hours, during which the process
    # otherwise emits nothing at all and is indistinguishable from a hang.
    log_every = max(1, len(indices) // 40)
    for done, i in enumerate(indices, start=1):
        s = train_ds[i]
        img = s["image"].numpy()  # [C,T,H,W]
        m = s["mask"].numpy()[0] > 0.5
        lab = s["label"].numpy()[0][m]
        labels.append(lab)
        sbs.update([img[b][:, m].ravel() for b in range(n_bands)])
        if done % log_every == 0 or done == len(indices):
            logger.info("Stats pass: %d/%d chips (%.0f%%)", done, len(indices),
                        100.0 * done / len(indices))
    y = np.concatenate(labels) if labels else np.array([0.0])
    scaler = target_scaler_from_values(y, cfg.norm.target_scaling)

    # Record the real fold train years rather than a placeholder -- this file is
    # the provenance record a reviewer reads to confirm no test year leaked in.
    train_years = list(resolve_fold(cfg).train_years)

    if cfg.norm.mode == "official_prithvi_statistics":
        stats = official_prithvi_stats(train_years, scaler)
    else:
        mean, std = sbs.finalize()
        stats = NormStats(
            band_mean=mean.tolist(), band_std=std.tolist(), target=scaler.to_dict(),
            mode=cfg.norm.mode, train_years=train_years,
            n_chips_used=len(indices), n_chips_total=n_total,
            stats_seed=cfg.norm.stats_seed if len(indices) < n_total else None,
            states=sorted(cfg.data.states),
        )
    return stats


class StatsReuseError(RuntimeError):
    """A norm_stats.json was offered for reuse that does not fit this fold."""


def load_or_compute_fold_stats(cfg: FarmConfig, dm: FarmDataModule, fold) -> NormStats:
    """Load ``norm.reuse_stats_from`` if set and valid, else run the stats pass.

    The pass is deterministic given (train split, seed, cap), so recomputing it
    after a crash or a precision change reproduces a file already on disk at a
    cost of ~1 h (4-state fold) to ~19 h (unsubsampled). Reuse must nevertheless
    be *checked*, not trusted: statistics carry the fingerprint of the years
    they were computed on, and normalising with a file that saw the test year
    would break LOYO silently and unrecoverably. We would rather spend the hour
    than train on leaked statistics, so a mismatch raises instead of warning.
    """
    path = getattr(cfg.norm, "reuse_stats_from", None)
    if not path:
        return compute_fold_stats(cfg, dm)

    p = Path(path)
    if not p.exists():
        raise StatsReuseError(
            f"norm.reuse_stats_from={path} does not exist. Remove the option to "
            f"recompute, or point it at a norm_stats.json from a matching fold."
        )

    stats = NormStats.load(p)
    want = sorted(int(y) for y in fold.train_years)
    got = sorted(int(y) for y in (stats.train_years or []))
    if got != want:
        raise StatsReuseError(
            f"Refusing to reuse {path}: it was computed on train_years={got}, but "
            f"this fold trains on {want}. Reusing it could normalise using the "
            f"test year (see docs/LOYO_PROTOCOL.md). Recompute instead."
        )
    if stats.mode != cfg.norm.mode:
        raise StatsReuseError(
            f"Refusing to reuse {path}: mode={stats.mode!r} but config asks for "
            f"{cfg.norm.mode!r}."
        )

    # A changed state list changes the train split just as surely as a changed
    # year list does, but it leaves train_years identical -- so the year check
    # alone cannot catch it. Adding a state and reusing the old file would
    # normalise with statistics that never saw the new state's imagery.
    want_states = sorted(cfg.data.states)
    if stats.states is not None:
        got_states = sorted(stats.states)
        if got_states != want_states:
            raise StatsReuseError(
                f"Refusing to reuse {path}: computed on states={got_states}, but "
                f"this run trains on {want_states}. Recompute instead."
            )
    else:
        logger.warning(
            "%s predates the `states` provenance field, so the state list CANNOT "
            "be verified -- only train_years was checked. Confirm by hand that it "
            "was computed on exactly %s before trusting this run.",
            path, want_states,
        )

    logger.warning(
        "REUSING normalization stats from %s (train_years=%s, n_chips_used=%s, "
        "seed=%s) -- statistics pass SKIPPED",
        path, got, stats.n_chips_used, stats.stats_seed,
    )
    return stats


def load_init_weights(lm: FarmLightningModule, path: str) -> None:
    """Copy weights from checkpoint ``path`` into ``lm`` (weight-only transfer init).

    Non-strict in exactly one way: tensors of a newly added DetailRefiner may be
    absent from the source checkpoint -- they keep their zero-initialised output,
    so the model starts out identical to the source. Any other missing key, or any
    unexpected key, is an architecture mismatch and raises; skipping those
    silently would train a partly random model while the logs claim it came from
    ``path``.
    """
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = lm.load_state_dict(state, strict=False)
    other = [k for k in missing if not k.startswith("model.refiner.")]
    if other or unexpected:
        raise RuntimeError(
            f"init_from={path} does not match the configured architecture -- "
            f"missing: {other[:5]}, unexpected: {list(unexpected)[:5]}"
        )
    if missing:
        logger.warning("init_from: %d DetailRefiner tensor(s) absent from %s -- starting "
                       "from the zero-initialised branch", len(missing), path)


def train_fold(cfg: FarmConfig, use_dummy: bool = True, dummy_embed_dim: int = 32,
               resume_from: str | None = None, init_from: str | None = None):
    """Train one fold. Returns ``(lm, dm, stats, out_dir, best_checkpoint_path)``.

    ``resume_from`` and ``init_from`` both take a checkpoint but mean opposite
    things, and using the wrong one silently produces a broken run:

    ``resume_from`` CONTINUES an interrupted run. Lightning restores the
        optimiser's moment buffers, the LR-scheduler position and the epoch
        counter, then carries on. Correct for "the job died at epoch 13".

    ``init_from``   TRANSFERS weights into a NEW run. Only the model weights are
        taken; the optimiser, LR schedule and epoch counter all start fresh.
        Correct for "fine-tune the Corn Belt model on BARC".

    Using ``resume_from`` for transfer is the trap: a checkpoint from epoch 18 of
    a 30-epoch cosine schedule resumes at epoch 19 with the LR already decayed to
    near ``min_lr``, so the model barely moves and the run stops almost
    immediately at ``epochs``. The source checkpoint is never written to in
    either case.
    """
    import lightning as L

    if resume_from and init_from:
        raise ValueError(
            "Pass resume_from OR init_from, not both: one continues a run "
            "(optimiser state restored), the other starts a new run from "
            "borrowed weights (optimiser fresh). See train_fold's docstring."
        )
    if init_from and cfg.model.finetune_mode == "refiner_only" and not cfg.norm.inherit_init_stats:
        raise ValueError(
            "finetune_mode=refiner_only with init_from requires norm.inherit_init_stats=true: "
            "the frozen model must be fed the normalisation it was trained with."
        )

    seed_everything(cfg.train.seed)
    fold = resolve_fold(cfg)
    logger.info("Fold: test=%s val=%s train=%s", fold.test_year, fold.val_years, fold.train_years)

    dm = FarmDataModule(cfg, synthetic=use_dummy, n_synth=8)
    dm.setup()
    if init_from and cfg.norm.inherit_init_stats:
        src = find_norm_stats(init_from)
        if src is None:
            raise FileNotFoundError(f"norm.inherit_init_stats: no norm_stats.json above {init_from}")
        # The SOURCE model's statistics, not this fold's train split -- which is
        # why the fold reuse guard in load_or_compute_fold_stats does not apply.
        stats = NormStats.load(src)
        logger.warning("INHERITED normalisation from %s (the init checkpoint's own statistics)", src)
    else:
        stats = load_or_compute_fold_stats(cfg, dm, fold)
    dm.apply_norm_stats(stats)

    out_dir = Path(cfg.train.output_dir) / cfg.experiment_name / f"test{fold.test_year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, out_dir / "resolved_config.yaml")
    stats.save(out_dir / "norm_stats.json")

    lm = FarmLightningModule(cfg, use_dummy=use_dummy, dummy_embed_dim=dummy_embed_dim)
    if init_from:
        # Weight-only transfer init: optimiser state lives in the Trainer and is
        # only restored by fit(ckpt_path=...), which is deliberately not passed.
        logger.warning(
            "INIT (weights only) from %s -- fresh optimiser, LR schedule and "
            "epoch counter; the source checkpoint is not modified", init_from,
        )
        load_init_weights(lm, init_from)
    lm.set_target_scaler(stats.target_scaler())

    from .trainer import build_trainer

    trainer = build_trainer(cfg, fold, str(out_dir), stats.__dict__, manifest_fp="synthetic")
    best_ckpt = None
    if isinstance(trainer, L.Trainer):
        if resume_from:
            logger.info("Resuming from checkpoint: %s", resume_from)
        trainer.fit(lm, dm.train_dataloader(), dm.val_dataloader(), ckpt_path=resume_from) # This is the most important line in this function, it runs the training loop using the Lightning trainer, model, and data module.
        best_ckpt = getattr(trainer.checkpoint_callback, "best_model_path", None) or None
        logger.info("Best checkpoint: %s", best_ckpt)
    return lm, dm, stats, out_dir, best_ckpt


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


def run_loyo(cfg: FarmConfig, use_dummy: bool = True, init_from: str | None = None):
    import gc

    import torch

    results = {}
    for test_year in cfg.data.years:
        cfg.split.test_year = test_year
        # ensure val year differs
        cfg.split.val_years = [y for y in cfg.split.val_years if y != test_year] or [
            max(y for y in cfg.data.years if y != test_year)
        ]
        try:
            # Record the val years the fold ACTUALLY uses. Under explicit_map the
            # split map overrides the list above inside resolve_fold, so without
            # this each fold's resolved_config.yaml names a val year never used.
            cfg.split.val_years = list(resolve_fold(cfg).val_years)
            best_ckpt = train_fold(cfg, use_dummy=use_dummy, init_from=init_from)[-1]
            # Score the fold's validation-selected checkpoint. Without one,
            # evaluate_fold would build and score a FRESH, untrained model.
            if best_ckpt is None:
                raise RuntimeError("training produced no checkpoint to evaluate")
            results[test_year] = evaluate_fold(cfg, checkpoint=best_ckpt, use_dummy=use_dummy)
        except Exception as e:  # keep going across folds
            logger.error("Fold %s failed: %s", test_year, e)
            results[test_year] = {"error": str(e)}
        finally:
            # Every fold builds a full model; release it before the next one.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    for year, res in results.items():
        logger.info("LOYO test %s: %s", year, res.get("error") or res.get("global_pixel"))
    return results
