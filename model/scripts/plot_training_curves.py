#!/usr/bin/env python
"""Plot per-epoch train and validation RMSE from a run's CSV logs -- no model, no GPU.

Reads every csv/version_*/metrics.csv in order, so a run continued with resume_from
stays one curve (a later version wins for a repeated epoch). Runs from before
train/main_rmse_phys_epoch existed plot validation only.

Writes training_curves.png and training_curves.csv (the same numbers) to --run-dir.

Example:
  uv run python scripts/plot_training_curves.py \
      --run-dir outputs/runs/Cornbelt5_spatial_soil_weather/v4_main_and_soil_branch_update/test2024
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN, VAL, LR = "train/main_rmse_phys_epoch", "val/main_rmse_phys", "lr-AdamW"
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def read_epochs(run_dir: Path) -> dict[int, dict[str, float]]:
    paths = sorted((run_dir / "csv").glob("version_*/metrics.csv"),
                   key=lambda p: int(p.parent.name.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"no csv/version_*/metrics.csv under {run_dir}")
    epochs: dict[int, dict[str, float]] = {}
    last_step: dict[int, int] = {}
    lr_rows: list[tuple[int, float]] = []
    for path in paths:
        # A process killed mid-write (e.g. a reboot) can leave NUL padding in the file.
        lines = path.read_text().replace("\0", "").splitlines()
        for row in csv.DictReader(lines):
            if not (row.get("step") or "").strip():
                continue
            if row.get(LR):  # LearningRateMonitor rows carry a step but no epoch
                lr_rows.append((int(row["step"]), float(row[LR])))
            if not row.get("epoch"):
                continue
            epoch = int(float(row["epoch"]))
            last_step[epoch] = max(last_step.get(epoch, 0), int(row["step"]))
            for key in (TRAIN, VAL):
                if row.get(key):
                    epochs.setdefault(epoch, {})[key] = float(row[key])
    for step, lr in lr_rows:
        # Logged at the start of an epoch: the first epoch that ends after this step.
        epoch = min((e for e, s in last_step.items() if s > step), default=None)
        if epoch in epochs:
            epochs[epoch][LR] = lr
    return epochs


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def plot(epochs: dict[int, dict[str, float]], title: str, out: Path, show_lr: bool = True) -> None:
    if show_lr:
        fig, (ax, ax_lr) = plt.subplots(
            2, 1, figsize=(8, 5.5), sharex=True, height_ratios=[3, 1],
            facecolor=SURFACE, constrained_layout=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE, constrained_layout=True)
    for key, label, color in ((TRAIN, "Train", TRAIN_COLOR), (VAL, "Validation", VAL_COLOR)):
        xs = [e for e in sorted(epochs) if key in epochs[e]]
        if not xs:
            continue
        ys = [epochs[e][key] for e in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, clip_on=False)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK, annotation_clip=False)
    val = [e for e in epochs if VAL in epochs[e]]
    if val:
        best = min(val, key=lambda e: epochs[e][VAL])
        title += f"\nBest validation RMSE: {epochs[best][VAL]:.3f} bu/ac (epoch {best})"
        ax.plot(best, epochs[best][VAL], marker="o", markersize=16, markerfacecolor="none",
                markeredgecolor=VAL_COLOR, markeredgewidth=2, clip_on=False)
    ax.margins(x=0.1, y=0.15)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel("RMSE (bu/ac)", fontsize=9, color=MUTED)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2, frameon=False, fontsize=9,
              labelcolor=INK)
    style(ax)
    if not show_lr:
        ax.set_xlabel("Epoch", fontsize=9, color=MUTED)
        ax.xaxis.get_major_locator().set_params(integer=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return

    lr = [e for e in sorted(epochs) if LR in epochs[e]]
    ax_lr.plot(lr, [epochs[e][LR] for e in lr], color=MUTED, linewidth=2, clip_on=False)
    ax_lr.set_ylabel("Learning rate", fontsize=9, color=MUTED)
    ax_lr.set_xlabel("Epoch", fontsize=9, color=MUTED)
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_lr.yaxis.get_offset_text().set(color=MUTED, fontsize=9)
    ax_lr.xaxis.get_major_locator().set_params(integer=True)
    style(ax_lr)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot per-epoch train/val RMSE for one run.")
    ap.add_argument("--run-dir", type=Path, required=True, help="Directory containing csv/version_*/")
    ap.add_argument("--title", default=None, help="Default: the run's path below outputs/runs")
    ap.add_argument("--no-lr", action="store_true",
                    help="RMSE only, without the learning-rate panel (writes training_curves_rmse.png)")
    args = ap.parse_args()

    epochs = read_epochs(args.run_dir)
    title = args.title or "/".join(args.run_dir.resolve().parts[-3:])
    name = "training_curves_rmse.png" if args.no_lr else "training_curves.png"
    plot(epochs, title, args.run_dir / name, show_lr=not args.no_lr)
    with (args.run_dir / "training_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_rmse", "val_rmse", "lr"])
        for e in sorted(epochs):
            writer.writerow([e, *(epochs[e].get(k, "") for k in (TRAIN, VAL, LR))])
    print(f"wrote {args.run_dir / name} and training_curves.csv")


if __name__ == "__main__":
    main()
