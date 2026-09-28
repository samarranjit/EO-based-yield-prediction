# Training Guide

## Paper-replication defaults (`configs/training/paper_replication.yaml`)
seed 0 · 120 epochs · batch 8 · AdamW · wd 0.1 · LR 5e-6→1e-8 · 20-epoch linear
warm-up then cosine · dropout 0.1 · bf16-mixed · H/V-flip p=0.2 · Gaussian noise
p=0.4 · encoder fully unfrozen · MSE + 0.2·aux.

Train one fold:
```bash
uv run --extra prithvi python -m farm_us.cli train \
    --config configs/experiments/us_soybeans.yaml --real test_year=2018
```
(Drop `--real` and the extra to run the synthetic/dummy pipeline anywhere.)

## Full fine-tuning vs partial freezing
`model.finetune_mode`: `full` (paper), `frozen` (encoder frozen — fast, decoder
learns), `decoder_only` (warm-up the decoder before unfreezing). Frozen/partial
are fallbacks/ablations, not the primary experiment.

## Effective batch size & the memory ladder
600M-TL + T=8 + dense 224² output ≈ 48 GB at batch 8. Escalate in order:
1. `train.precision: bf16-mixed`
2. reduce `train.batch_size` (micro-batch)
3. `train.grad_accum` to restore the effective batch (`effective = batch × accum × devices`)
4. `model.gradient_checkpointing: true`
5. efficient attention (via the official backbone)
6. `train.strategy: ddp`
7. `train.strategy: fsdp`
8. CPU offload (last resort)

The `profile-memory` command reports physical vs effective batch and peak
alloc/reserved GB. The scientific model is **never** silently changed on OOM.

## FSDP / DDP
Set `train.strategy` and `train.devices`. The custom losses are distributed-safe
(reduction over valid pixels; zero-valid batches produce a finite 0 with grad).

## Checkpoint selection
`ModelCheckpoint(monitor=val/main_rmse_phys, mode=min, save_top_k=1)` — selection
uses **validation-fold** physical-unit RMSE only, never test data. `last.ckpt` is
also kept for resume.

## Resume vs transfer — `resume_from` and `init_from`

Both take a checkpoint and they mean **opposite** things. Using the wrong one
silently produces a broken run.

| | `resume_from` | `init_from` |
|---|---|---|
| Restores optimiser moments, LR position, epoch counter | yes | **no** |
| Use for | "the job died at epoch 13" | "fine-tune the Corn Belt model on BARC" |

```bash
# continue an interrupted run
uv run python -m farm_us.cli train --config <cfg> --real resume_from=<run>/checkpoints/last.ckpt

# NEW run from borrowed weights
uv run python -m farm_us.cli train --config <cfg> --real \
    init_from=outputs/runs/cornbelt5_soybeans_longer_time/test2024/checkpoints/farm-018-0.0000.ckpt \
    norm.inherit_init_stats=true
```

**The trap**: `resume_from` on a checkpoint from epoch 18 of a 30-epoch cosine
schedule resumes at epoch 19 with LR already decayed to near `min_lr`, so the model
barely moves and the run stops almost immediately at `epochs`. Passing both raises.
Neither modifies the source checkpoint.

Three rules for `init_from`:

1. **The architecture must match the source checkpoint exactly.** `load_init_weights`
   is strict except for missing keys prefixed `model.refiner.`; anything else
   missing or unexpected raises. `temporal_reducer: attention` vs `flatten_time`
   alone will fail the load — copy the whole `model:` block from the config the
   checkpoint was trained with. Adding a new branch? Zero-initialise its output so
   step 0 reproduces the source, and extend that prefix list rather than loosening
   the check.
2. **Set `norm.inherit_init_stats: true`** whenever the base is frozen — it reuses
   the source run's `norm_stats.json` instead of recomputing. Required with
   `finetune_mode: refiner_only` (enforced in `train_fold`).
3. **Never reuse stats across a change of `data.states`.** `norm.reuse_stats_from`'s
   guard validates `train_years` only and will happily accept a stale file.

Full resolved config, norm stats and provenance are saved per run either way.

## Refiner-only fine-tuning
`finetune_mode: refiner_only` + `detail_refiner: true` trains just the
~20,801-parameter `DetailRefiner` and freezes the 775M base (weights **and**
BatchNorm/dropout, held in `eval()`). Needs `lr 1e-2, weight_decay 0.0` — decay
cancels a zero-initialised output layer as fast as it learns. See
ARCHITECTURE.md and BARC_TRANSFER.md.

## Reproducibility
`seed_everything(0)`; deterministic cudnn; per-run `resolved_config.yaml`,
`norm_stats.json`, `provenance.json` (config, fold years, versions, git commit,
param counts, manifest fingerprint, label provenance). Normalization + target
scaling are computed from **training years only**.

## Logging
CSV + TensorBoard always; W&B if `train.log_wandb: true` and installed. Logged:
loss, aux loss, valid-pixel count, LR, grad-norm, throughput (samples/s), and
(on CUDA) peak GPU memory; NaN/Inf flags.

## Ablations (separate configs, never silent substitutions)
`ablation_no_location`, `ablation_no_aux`, `ablation_huber`,
`ablation_mean_reducer`. The 300M-TL fallback is `configs/model/farm_prithvi_300m_tl.yaml`.
