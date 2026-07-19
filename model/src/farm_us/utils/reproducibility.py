"""Deterministic seeding and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict
from typing import Any

import numpy as np


def seed_everything(seed: int = 0, deterministic: bool = True) -> int:
    """Seed python / numpy / torch and (optionally) force deterministic algos."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # torch optional in some tooling paths
        pass
    return seed


def git_commit(default: str = "unknown") -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return default


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("torch", "lightning", "numpy", "rasterio", "terratorch", "einops"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not-installed"
    return versions


def fingerprint(obj: Any) -> str:
    """Stable short hash of any JSON-serializable object (for manifest / config)."""
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
