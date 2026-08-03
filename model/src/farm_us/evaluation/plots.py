"""Diagnostic plots (matplotlib, Agg backend). Saved as PNGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _ax():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def scatter_obs_pred(
    pred,
    target,
    out_path: str | Path,
    title: str = "Predicted vs Observed",
    alpha: float = 0.02,
    point_size: float = 2,
) -> None:
    """One dot per valid pixel, with a 1:1 reference line.

    ``alpha`` is deliberately very low: a full test year is ~1e6 pixels, and at
    alpha=0.3 roughly 3 overlapping points already render fully opaque, so the
    dense core becomes a flat silhouette that hides how the mass is distributed.
    At alpha=0.02 it takes ~50 overlaps to saturate, so relative density stays
    readable. Tune per dataset size: fewer points -> raise it (0.1-0.3), more
    points -> lower it. Note alpha-blending still saturates in the very densest
    region; use a hexbin/2-D histogram if you need density read off a colorbar.
    """
    plt = _ax()
    p, t = np.asarray(pred).ravel(), np.asarray(target).ravel()
    m = np.isfinite(p) & np.isfinite(t)
    p, t = p[m], t[m]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, s=point_size, alpha=alpha, linewidths=0)
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


def error_overlay_map(
    actual: np.ndarray,
    residual: np.ndarray,
    out_path: str | Path,
    title: str = "Actual yield (background) + prediction error (overlay)",
) -> None:
    """Actual yield as a background layer, prediction error (pred - actual) as
    a semi-transparent diverging overlay -- shows WHERE errors are large, in
    geographic context. Both arrays share the same grid; NaN (no data / not a
    qualifying chip) renders as fully transparent so gaps don't look like zero."""
    plt = _ax()
    import matplotlib

    fig, ax = plt.subplots(figsize=(9, 9))

    bg_cmap = matplotlib.colormaps["Greens"].copy()
    bg_cmap.set_bad(color="white")
    im_bg = ax.imshow(np.ma.masked_invalid(actual), cmap=bg_cmap)
    fig.colorbar(im_bg, ax=ax, fraction=0.045, pad=0.02, label="Actual yield (kg/ha)")

    finite_resid = residual[np.isfinite(residual)]
    v = float(np.percentile(np.abs(finite_resid), 98)) if finite_resid.size else 1.0
    fg_cmap = matplotlib.colormaps["RdBu_r"].copy()
    fg_cmap.set_bad(alpha=0)
    im_fg = ax.imshow(np.ma.masked_invalid(residual), cmap=fg_cmap, vmin=-v, vmax=v, alpha=0.65)
    fig.colorbar(im_fg, ax=ax, fraction=0.045, pad=0.08, label="Prediction error, pred - actual (kg/ha)")

    ax.set_title(title)
    ax.axis("off")
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
