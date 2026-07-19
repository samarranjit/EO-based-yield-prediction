"""Custom Lightning callbacks: provenance saving, GPU-memory / throughput /
grad-norm logging, and NaN/Inf detection."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightning as L
import torch

from ..config import FarmConfig
from ..data.splits import FoldSpec
from ..utils.reproducibility import git_commit, package_versions


class ProvenanceCallback(L.Callback):
    """Write full config + fold + versions + norm stats next to the checkpoint."""

    def __init__(self, cfg: FarmConfig, fold: FoldSpec, out_dir: str, norm_stats: dict, manifest_fp: str) -> None:
        self.payload = {
            "config": cfg.to_container(),
            "fold": {"test_year": fold.test_year, "val_years": fold.val_years, "train_years": fold.train_years},
            "package_versions": package_versions(),
            "git_commit": git_commit(),
            "norm_stats": norm_stats,
            "manifest_fingerprint": manifest_fp,
            "label_provenance": "ridge-distributed county yield (external pipeline); see LOYO_PROTOCOL.md",
        }
        self.out_dir = Path(out_dir)

    def on_fit_start(self, trainer, pl_module):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pc = pl_module.model.parameter_counts()
        self.payload["parameter_counts"] = pc
        (self.out_dir / "provenance.json").write_text(json.dumps(self.payload, indent=2, default=str))


class ThroughputMemoryCallback(L.Callback):
    def __init__(self) -> None:
        self._t0 = None
        self._seen = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()
        self._seen = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._seen += batch["image"].shape[0]

    def on_train_epoch_end(self, trainer, pl_module):
        if self._t0 is None:
            return
        dt = max(1e-6, time.time() - self._t0)
        pl_module.log("train/samples_per_s", self._seen / dt)
        if torch.cuda.is_available():
            pl_module.log("train/gpu_mem_peak_gb", torch.cuda.max_memory_allocated() / 1e9)


class GradNormCallback(L.Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        total = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm(2).item()) ** 2
        pl_module.log("train/grad_norm", total**0.5)
