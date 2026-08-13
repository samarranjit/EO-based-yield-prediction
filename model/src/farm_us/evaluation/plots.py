"""Diagnostic plots (matplotlib, Agg backend). Saved as PNGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _ax():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def auto_scatter_style(n: int) -> tuple[float, float]:
    """Pick (alpha, point_size) from the number of points being drawn.

    A fixed alpha cannot serve both regimes this project produces. A state-year
    test set is ~1e6 pixels, where alpha=0.3 saturates after ~3 overlaps and the
    dense core becomes a flat silhouette; a BARC transfer set is ~2e3 pixels,
    where that same 0.02 renders as a blank plot -- which is exactly what
    happened and cost real time to diagnose, because an empty-looking figure is
    indistinguishable from a broken pipeline.

    Scale inversely with n so ~50 overlaps still saturate, then clamp: the 0.02
    floor preserves the previous appearance for large runs, and the 0.6 ceiling
    keeps small-n plots legible without going fully opaque.
    """
    if n <= 0:
        return 0.6, 6.0
    alpha = float(np.clip(2000.0 / n, 0.02, 0.6))
    size = float(np.clip(20000.0 / n, 2.0, 8.0))
    return alpha, size


def scatter_obs_pred(
    pred,
    target,
    out_path: str | Path,
    title: str = "Predicted vs Observed",
    alpha: float | None = None,
    point_size: float | None = None,
) -> None:
    """One dot per valid pixel, with a 1:1 reference line.

    ``alpha``/``point_size`` default to None = choose from the point count via
    :func:`auto_scatter_style`. Pass explicit values to override. Note that
    alpha-blending still saturates in the very densest region; use a hexbin or
    2-D histogram if you need density read off a colorbar.

    The point count is put in the title because a sparse scatter and a failed
    run look identical otherwise -- with n on the figure, "is this empty?" is
    answerable from the image alone.
    """
    plt = _ax()
    p, t = np.asarray(pred).ravel(), np.asarray(target).ravel()
    m = np.isfinite(p) & np.isfinite(t)
    p, t = p[m], t[m]
    auto_alpha, auto_size = auto_scatter_style(p.size)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, s=point_size if point_size is not None else auto_size,
               alpha=alpha if alpha is not None else auto_alpha, linewidths=0)
    lim = [min(t.min(), p.min()), max(t.max(), p.max())] if p.size else [0, 1]
    ax.plot(lim, lim, "g-", lw=1)
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    ax.set_title(f"{title}  (n={p.size:,})")
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
