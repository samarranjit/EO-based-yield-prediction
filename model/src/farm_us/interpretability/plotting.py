"""Interpretability plots: temporal-attention heatmaps, receiving-attention
curves, and spectral band-importance bars."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import DEFAULT_TIMESTEPS


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _month_labels(t: int) -> list[str]:
    return [ts[0].title() for ts in DEFAULT_TIMESTEPS[:t]]


def attention_heatmap(tt: np.ndarray, out_path: str | Path, title: str = "Temporal attention") -> None:
    plt = _plt()
    labels = _month_labels(tt.shape[0])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(tt, cmap="RdBu_r")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Target month"); ax.set_ylabel("Source month"); ax.set_title(title)
    for i in range(tt.shape[0]):
        for j in range(tt.shape[1]):
            ax.text(j, i, f"{tt[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax)
    _save(fig, out_path)


def receiving_curve(scores: dict[str, np.ndarray], out_path: str | Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, s in scores.items():
        ax.plot(_month_labels(len(s)), s, marker="o", label=label)
    ax.set_xlabel("Month"); ax.set_ylabel("Incoming attention"); ax.legend(); ax.set_title("Temporal receiving")
    _save(fig, out_path)


def band_importance_bar(importance: dict[str, float], out_path: str | Path) -> None:
    plt = _plt()
    names = list(importance.keys()); vals = list(importance.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Relative importance"); ax.set_title("Spectral band importance")
    _save(fig, out_path)


def _save(fig, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    import matplotlib.pyplot as plt

    plt.close(fig)
