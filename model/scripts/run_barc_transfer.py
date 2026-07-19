#!/usr/bin/env python
"""Run one of the three BARC transfer experiments (synthetic-capable smoke).

  python scripts/run_barc_transfer.py --config configs/barc/barc_finetune.yaml \
      experiment=finetune_from_farm [national_checkpoint=/path.ckpt]
"""
import sys

from farm_us.cli import _expand_overrides, _kv
from farm_us.config import load_config
from farm_us.transfer.barc_experiments import BarcExperiment, build_experiment_module


def main(argv):
    config = None; rest = []
    it = iter(argv)
    for a in it:
        if a == "--config": config = next(it)
        else: rest.append(a)
    cfg = load_config(config, _expand_overrides(rest))
    name = _kv(rest, "experiment") or "finetune_from_farm"
    ckpt = _kv(rest, "national_checkpoint")
    use_dummy = "--real" not in argv
    lm = build_experiment_module(cfg, BarcExperiment(name=name, national_checkpoint=ckpt),
                                 use_dummy=use_dummy, dummy_embed_dim=32)
    print(f"Built BARC experiment '{name}'. params={lm.model.parameter_counts()}")
    print("Wire BARC LOYO loader (transfer.barc_dataset) + trainer to run; see docs/BARC_TRANSFER.md")


if __name__ == "__main__":
    main(sys.argv[1:])
