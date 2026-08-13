#!/usr/bin/env python
"""Find corrupt blocks in the HLS composites a config will actually read.

Why this exists
---------------
A partially-written or truncated DEFLATE tile raises only when it is decoded, so
a corrupt composite looks perfectly healthy to `ls`, `gdalinfo`, and any metadata
check. It surfaces as a RasterioIOError inside a DataLoader worker, which kills
the whole training run:

    hls_soybeans_IN_2022_JUL.tif, band 1: IReadBlock failed at
    X offset 22, Y offset 52: TIFFReadEncodedTile() failed.
    ZIPDecode:Decoding error at scanline 4928

A 20-epoch run reads every chip 20 times, so one bad block anywhere is fatal --
and it fails late, after hours of work. Finding them all up front is much cheaper
than discovering them one restart at a time.

Scope: only the (state, year, month) files and only the chip windows that the
given config's manifest actually enumerates. Chips that fail the qualifying
filter are never read during training, so a bad block under one of them is
harmless and should not be reported. This makes the scan both faster than a full
file read and an exact match for what training touches.

Output: a JSON report listing every (file, row_off, col_off) that failed to
decode, plus the affected sample_ids -- which is what you need either to
re-download those files or to quarantine those chips from the manifest.

Example:
  uv run python scripts/scan_hls_corruption.py \
      --config configs/experiments/cornbelt4_soybeans.yaml \
      --workers 16 --out outputs/qc/hls_corruption_cornbelt4.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MONTHS = ("APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("outputs/qc/hls_corruption.json"))
    p.add_argument("--years", type=int, nargs="*", default=None,
                   help="Restrict to these years (default: all in the config).")
    p.add_argument("--months", nargs="*", default=list(MONTHS))
    p.add_argument("--qualifying-only", action="store_true", default=True,
                   help="Only scan windows that survive the qualifying-chip filter (default).")
    p.add_argument("--all-windows", dest="qualifying_only", action="store_false",
                   help="Scan every manifest window, including ones training never reads.")
    return p.parse_args()


def _scan_one(task):
    """Decode the requested windows of one file; return the ones that fail.

    Runs in a worker process. GDAL is not fork-safe, so the dataset is opened
    here rather than inherited -- same rule the readers follow.
    """
    path, windows = task
    import rasterio
    from rasterio.windows import Window

    bad = []
    if not os.path.exists(path):
        return path, [], f"missing: {path}"
    try:
        with rasterio.open(path) as ds:
            for (r0, c0, size) in windows:
                try:
                    ds.read(window=Window(c0, r0, size, size), boundless=True, fill_value=0)
                except Exception as exc:  # noqa: BLE001 - any decode failure counts
                    bad.append({"row_off": r0, "col_off": c0, "error": str(exc)[:200]})
    except Exception as exc:  # noqa: BLE001 - file-level open failure
        return path, [], f"open failed: {exc}"
    return path, bad, None


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import pandas as pd

    from farm_us.config import load_config
    from farm_us.data.dataset import filter_manifest_to_qualifying_chips

    cfg = load_config(args.config)
    df = pd.read_parquet(cfg.data.manifest_path)
    df = df[df["state"].isin(cfg.data.states)]
    years = args.years or list(cfg.data.years)
    df = df[df["year"].isin(years)]
    print(f"manifest rows for {cfg.data.states} {years}: {len(df):,}")

    if args.qualifying_only:
        df = filter_manifest_to_qualifying_chips(df, cfg)
        print(f"after qualifying filter: {len(df):,} chips (what training reads)")

    crop = cfg.data.crop.lower()
    root = Path(cfg.data.imagery_root)
    size = cfg.data.chip_size

    tasks = []
    index: dict[str, list] = {}
    for (state, year), g in df.groupby(["state", "year"]):
        wins = [(int(r.row_off), int(r.col_off), size) for r in g.itertuples()]
        ids = {(int(r.row_off), int(r.col_off)): r.sample_id for r in g.itertuples()}
        for mon in args.months:
            path = str(root / str(year) / f"hls_{crop}_{state}_{year}_{mon}.tif")
            tasks.append((path, wins))
            index[path] = ids
    total_reads = sum(len(w) for _, w in tasks)
    print(f"scanning {len(tasks)} files / {total_reads:,} window reads "
          f"with {args.workers} workers\n")

    report = {"config": args.config, "files_scanned": len(tasks),
              "window_reads": total_reads, "corrupt": [], "unreadable": []}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_scan_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            path, bad, err = fut.result()
            done += 1
            if err:
                report["unreadable"].append({"path": path, "error": err})
                print(f"[{done}/{len(tasks)}] UNREADABLE {Path(path).name}: {err}")
            elif bad:
                for b in bad:
                    b["sample_id"] = index.get(path, {}).get((b["row_off"], b["col_off"]))
                report["corrupt"].append({"path": path, "n_bad": len(bad), "windows": bad})
                print(f"[{done}/{len(tasks)}] CORRUPT {Path(path).name}: {len(bad)} bad window(s)")
            elif done % 10 == 0:
                el = time.time() - t0
                print(f"[{done}/{len(tasks)}] ok  ({el/60:.1f} min elapsed, "
                      f"~{el/done*(len(tasks)-done)/60:.0f} min left)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    n_bad_chips = sum(len(c["windows"]) for c in report["corrupt"])
    ids = {w["sample_id"] for c in report["corrupt"] for w in c["windows"] if w["sample_id"]}
    print(f"\n{'=' * 60}")
    print(f"files with corrupt blocks : {len(report['corrupt'])}")
    print(f"unreadable files          : {len(report['unreadable'])}")
    print(f"bad window reads          : {n_bad_chips:,} of {total_reads:,} "
          f"({100 * n_bad_chips / max(total_reads, 1):.4f}%)")
    print(f"distinct chips affected   : {len(ids):,}")
    print(f"elapsed                   : {(time.time() - t0) / 60:.1f} min")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
