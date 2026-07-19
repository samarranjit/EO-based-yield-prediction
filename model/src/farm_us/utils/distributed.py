"""Distributed helpers (safe no-ops when not in a distributed context)."""

from __future__ import annotations


def is_distributed() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def rank() -> int:
    if is_distributed():
        import torch.distributed as dist

        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    return rank() == 0
