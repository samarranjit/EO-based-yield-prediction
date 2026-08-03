"""FARM-US command-line interface.

    farm-us <command> --config <yaml> [key=value ...]

Commands: inventory, build-manifest, verify-splits, leakage-audit, inspect-batch,
smoke-test-model, profile-memory, train, evaluate, run-loyo, predict-raster,
band-importance.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .data.splits import load_split_map, make_fold
from .utils.logging import get_logger

logger = get_logger("farm_us.cli")


def _common(sub):
    sub.add_argument("--config", default=None, help="YAML config path")
    sub.add_argument("--real", action="store_true",
                     help="use the real Prithvi backbone instead of the dummy one")
    sub.add_argument("overrides", nargs="*", help="key=value overrides")
    return sub


# Convenience aliases so paper-style bare overrides work, e.g. `test_year=2018`.
_ALIASES = {
    "test_year": "split.test_year",
    "val_years": "split.val_years",
    "state": "_ignore.state",
    "year": "_ignore.year",
    "checkpoint": "_ignore.checkpoint",
    "resume_from": "_ignore.resume_from",
}


def _expand_overrides(overrides: list[str]) -> list[str]:
    out = []
    for o in overrides:
        if "=" in o:
            k, v = o.split("=", 1)
            if k in _ALIASES and _ALIASES[k].startswith("_ignore"):
                continue  # consumed by the command directly, not part of config
            k = _ALIASES.get(k, k)
            out.append(f"{k}={v}")
    return out


def _cfg(args):
    return load_config(args.config, _expand_overrides(args.overrides))


def _fold(cfg):
    smap = None
    if cfg.split.policy == "explicit_map":
        try:
            smap = load_split_map(cfg.split.split_map_path)
        except Exception:
            smap = None
    return make_fold(
        cfg.split.test_year, cfg.data.years, cfg.split.val_years,
        policy=cfg.split.policy, split_map=smap,
    )


# ---- commands ---- #

def cmd_inventory(args):
    from .data.inventory import write_inventory

    inv = write_inventory(_cfg(args))
    print(json.dumps({
        "states_with_labels": sorted(inv["labels"].keys()),
        "n_missing_state_years": len(inv["missing_state_years"]),
        "imagery_present": inv["imagery_present"],
    }, indent=2))


def cmd_build_manifest(args):
    from .data.manifest import build_manifest

    df = build_manifest(_cfg(args))
    print(f"manifest rows: {len(df)} fingerprint: {df.attrs.get('manifest_fingerprint')}")


def cmd_verify_splits(args):
    cfg = _cfg(args)
    fold = _fold(cfg)
    print(json.dumps({
        "test_year": fold.test_year, "val_years": fold.val_years,
        "train_years": fold.train_years, "disjoint": True,
    }, indent=2))


def cmd_leakage_audit(args):
    cfg = _cfg(args)
    fold = _fold(cfg)
    import pandas as pd

    from .data.splits import audit_manifest_split

    try:
        df = pd.read_parquet(cfg.data.manifest_path)
        recs = df.to_dict("records")
    except Exception:
        recs = []
    res = audit_manifest_split(recs, fold)
    print(json.dumps({"ok": res["ok"], "issues": res["issues"],
                      "note": "label ridge-provenance may not be strictly LOYO"}, indent=2))


def cmd_inspect_batch(args):
    cfg = _cfg(args)
    from .data.dataset import FarmDataModule

    use_real = "--real" in sys.argv
    dm = FarmDataModule(cfg, synthetic=not use_real, n_synth=4)
    dm.setup()
    if use_real:
        from .training.run import compute_fold_stats

        stats = compute_fold_stats(cfg, dm)
        dm.apply_norm_stats(stats)
    batch = next(iter(dm.train_dataloader()))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"{k:18s} shape={tuple(v.shape)} dtype={v.dtype} "
                  f"min={float(v.float().min()):.3f} max={float(v.float().max()):.3f}")
        else:
            print(f"{k:18s} = {v}")


def cmd_smoke_test_model(args):
    import torch

    cfg = _cfg(args)
    from .models.farm_model import FarmModel

    use_dummy = "--real" not in sys.argv
    m = FarmModel(cfg.model, n_timesteps=cfg.data.n_timesteps, chip_size=cfg.data.chip_size,
                  use_dummy=use_dummy, dummy_embed_dim=32)
    bs = 2  # BatchNorm in the PPM (1x1 bin) needs >1 sample in train mode
    x = torch.randn(bs, 6, cfg.data.n_timesteps, cfg.data.chip_size, cfg.data.chip_size)
    tc = torch.zeros(bs, cfg.data.n_timesteps, 2); lc = torch.zeros(bs, 2)
    m.train()
    out = m(x, tc, lc)
    print(json.dumps({
        "backbone": "dummy" if use_dummy else cfg.model.backbone_id,
        "embed_dim": m.encoder.embed_dim,
        "out_indices_0based": m.encoder.out_indices,
        "main": list(out["main"].shape),
        "aux": list(out["aux"].shape) if out["aux"] is not None else None,
        "params": m.parameter_counts(),
    }, indent=2))


def cmd_profile_memory(args):
    import torch

    from .models.farm_model import FarmModel
    from .training.losses import masked_mse

    cfg = _cfg(args)
    use_dummy = "--real" not in sys.argv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = FarmModel(cfg.model, n_timesteps=cfg.data.n_timesteps, chip_size=cfg.data.chip_size,
                  use_dummy=use_dummy, dummy_embed_dim=32).to(dev)

    # Mirror real training exactly (lightning_module.py.configure_optimizers): a plain
    # AdamW over all trainable params. Momentum/variance buffers only get allocated on
    # the FIRST optimizer.step(), so a forward+backward-only probe (the old behavior
    # here) silently omits ~2x the trainable-param memory footprint (fp32 exp_avg +
    # exp_avg_sq) and understates peak usage by many GB for a fully-unfrozen 600M+
    # backbone. Also autocast to bf16 to match train.precision=bf16-mixed.
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    autocast_dtype = torch.bfloat16 if "bf16" in cfg.train.precision else torch.float32

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    grad_accum = max(1, cfg.train.grad_accum)
    for _ in range(grad_accum):
        x = torch.randn(cfg.train.batch_size, 6, cfg.data.n_timesteps, 224, 224, device=dev)
        tgt = torch.randn(cfg.train.batch_size, 1, 224, 224, device=dev)
        mask = torch.ones_like(tgt)
        with torch.autocast(device_type=dev, dtype=autocast_dtype, enabled=(dev == "cuda")):
            out = m(x, torch.zeros(cfg.train.batch_size, cfg.data.n_timesteps, 2, device=dev),
                    torch.zeros(cfg.train.batch_size, 2, device=dev))
            loss, _ = masked_mse(out["main"], tgt, mask)
            if out["aux"] is not None:
                aux, _ = masked_mse(out["aux"], tgt, mask); loss = loss + 0.2 * aux
            loss = loss / grad_accum
        loss.backward()
    opt.step()  # allocates AdamW's exp_avg/exp_avg_sq on first call -- this is the real peak
    opt.zero_grad(set_to_none=True)
    from .training.trainer import effective_batch_size

    info = {"device": dev, "physical_batch": cfg.train.batch_size,
            "effective_batch": effective_batch_size(cfg)}
    if dev == "cuda":
        info["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
        info["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    print(json.dumps(info, indent=2))


def cmd_train(args):
    from .training.run import train_fold

    resume_from = _kv(args.overrides, "resume_from")
    train_fold(_cfg(args), use_dummy="--real" not in sys.argv, resume_from=resume_from)


def cmd_evaluate(args):
    from .training.run import evaluate_fold

    evaluate_fold(_cfg(args), checkpoint=_kv(args.overrides, "checkpoint"),
                  use_dummy="--real" not in sys.argv)


def cmd_run_loyo(args):
    from .training.run import run_loyo

    run_loyo(_cfg(args), use_dummy="--real" not in sys.argv)


def cmd_predict_raster(args):
    print("predict-raster requires real imagery + checkpoint; see scripts/predict_raster.py")


def cmd_band_importance(args):

    from .interpretability.spectral import band_importance
    from .models.farm_model import FarmModel

    cfg = _cfg(args)
    m = FarmModel(cfg.model, n_timesteps=cfg.data.n_timesteps, use_dummy="--real" not in sys.argv, dummy_embed_dim=32)
    print(json.dumps(band_importance(m), indent=2))


def _kv(overrides, key):
    for o in overrides:
        if o.startswith(f"{key}="):
            return o.split("=", 1)[1]
    return None


COMMANDS = {
    "inventory": cmd_inventory,
    "build-manifest": cmd_build_manifest,
    "verify-splits": cmd_verify_splits,
    "leakage-audit": cmd_leakage_audit,
    "inspect-batch": cmd_inspect_batch,
    "smoke-test-model": cmd_smoke_test_model,
    "profile-memory": cmd_profile_memory,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "run-loyo": cmd_run_loyo,
    "predict-raster": cmd_predict_raster,
    "band-importance": cmd_band_importance,
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="farm-us", description="FARM-US CLI")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        _common(subs.add_parser(name))
    args = parser.parse_args(argv)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
