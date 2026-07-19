"""Diagnostic plots (matplotlib, Agg backend). Saved as PNGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _ax():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def scatter_obs_pred(pred, target, out_path: str | Path, title: str = "Predicted vs Observed") -> None:
    plt = _ax()
    p, t = np.asarray(pred).ravel(), np.asarray(target).ravel()
    m = np.isfinite(p) & np.isfinite(t)
    p, t = p[m], t[m]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, s=2, alpha=0.3)
    lim = [min(t.min(), p.min()), max(t.max(), p.max())] if p.size else [0, 1]
    ax.plot(lim, lim, "g-", lw=1)
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted"); ax.set_title(title)
    _save(fig, out_path)


def residual_hist(pred, target, out_path: str | Path) -> None:
    plt = _ax()
    r = (np.asarray(pred).ravel() - np.asarray(target).ravel())
    r = r[np.isfinite(r)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(r, bins=60)
    ax.set_xlabel("Residual (pred - obs)"); ax.set_title("Residual distribution")
    _save(fig, out_path)


def residual_vs_obs(pred, target, out_path: str | Path) -> None:
    plt = _ax()
    p, t = np.asarray(pred).ravel(), np.asarray(target).ravel()
    m = np.isfinite(p) & np.isfinite(t)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(t[m], (p - t)[m], s=2, alpha=0.3)
    ax.axhline(0, color="g")
    ax.set_xlabel("Observed"); ax.set_ylabel("Residual"); ax.set_title("Residuals vs observed")
    _save(fig, out_path)


def map_image(array: np.ndarray, out_path: str | Path, title: str = "", cmap: str = "viridis") -> None:
    plt = _ax()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(array, cmap=cmap)
    fig.colorbar(im, ax=ax); ax.set_title(title); ax.axis("off")
    _save(fig, out_path)


def _save(fig, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    import matplotlib.pyplot as plt

    plt.close(fig)
