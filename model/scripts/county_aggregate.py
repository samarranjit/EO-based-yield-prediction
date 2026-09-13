#!/usr/bin/env python
"""Aggregate a 30 m yield raster to county level and compare against NASS.

Answers two different questions depending on which raster you point it at:

1. "Do the distributed pseudo-labels actually preserve the county mean?"
   Point it at a LABEL raster, e.g.
     ../data_preparation/data/yield_labels/multistate_pseudo/2024/
       nass_soybeans_yield_IA_2024_pseudo_30m_soybeans_only.tif
   The distribution step is supposed to preserve each county's NASS mean by
   construction, so agreement should be near-exact. Anything else is a bug in
   distribute_county_yield_to_pixels.py, and this is the check that finds it.

2. "Do our PREDICTIONS agree with NASS at county level?"
   Point it at a prediction raster, e.g.
     outputs/predictions/<experiment>/soybeans_IA_2024_pred.tif
   This is a genuine (if weaker) accuracy claim: recovering county means is a
   lower bar than intra-field accuracy. Keep it reported separately from the
   pixel-level metrics.

IMPORTANT -- the two raster kinds are NOT directly comparable
------------------------------------------------------------
The `_pred.tif` / `_actual.tif` written by map_test_errors.py are masked to the
comparison footprint: only chips that cleared the manifest QC filter (real
imagery, min_crop_fraction inside the state boundary). A county's mean over that
SUBSET need not equal its mean over all its soybean pixels, so a label raster
will reproduce NASS far more exactly than an `_actual.tif` of the same county.
The `n_pixels` and `coverage_note` columns are there to make that visible: treat
low-pixel counties as unreliable rather than as disagreement.

Usage
-----
  uv run python scripts/county_aggregate.py \
      --raster outputs/predictions/cornbelt_5_longer_years_009/soybeans_IA_2024_pred.tif \
      --state IA --year 2024

  # check the label raster instead, and write a scatter plot
  uv run python scripts/county_aggregate.py \
      --raster ../data_preparation/data/yield_labels/multistate_pseudo/2024/nass_soybeans_yield_IA_2024_pseudo_30m_soybeans_only.tif \
      --state IA --year 2024 --plot

Memory: reads the full raster and a full county-id grid (two arrays the size of
the state raster, ~0.8 GB each for Iowa). Fine on a workstation; if that ever
becomes a problem the fix is windowed accumulation, not chunked medians.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_COUNTIES = REPO / "data_preparation/data/county_maps/selected_states_counties_2023.gpkg"
DEFAULT_NASS = REPO / "data_preparation/data/nass/nass_soybeans_county_yield_2014_2024.csv"

#: Where every county-vs-NASS comparison lands, grouped by source directory.
AGG_DIR = MODEL_DIR / "outputs/county_aggregation"

# Two-letter state -> FIPS. Mirrors STATE_FIPS in farm_us.config; duplicated here
# only so this script stays runnable without importing the training package.
STATE_FIPS = {
    "IL": "17", "IA": "19", "IN": "18", "MN": "27", "NE": "31", "MO": "29",
    "OH": "39", "SD": "46", "ND": "38", "KS": "20", "WI": "55", "MI": "26",
    "MD": "24", "DE": "10", "VA": "51", "NC": "37", "PA": "42",
}


def county_stats(raster_path: Path, state: str, counties_path: Path,
                 min_pixels: int) -> pd.DataFrame:
    """Per-county summary statistics of `raster_path` over its valid pixels."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    state_fips = STATE_FIPS.get(state.upper())
    if state_fips is None:
        sys.exit(f"Unknown state {state!r}. Known: {', '.join(sorted(STATE_FIPS))}")

    with rasterio.open(raster_path) as ds:
        values = ds.read(1)
        nodata = ds.nodata
        transform, crs, shape = ds.transform, ds.crs, (ds.height, ds.width)

    counties = gpd.read_file(counties_path)
    counties = counties[counties["STATEFP"] == state_fips].copy()
    if counties.empty:
        sys.exit(f"No counties with STATEFP={state_fips} ({state}) in {counties_path}")
    if counties.crs != crs:
        counties = counties.to_crs(crs)

    # Burn GEOID directly as the pixel value so no lookup table is needed.
    # fill=0 is safe: no real GEOID is 0.
    county_id = rasterize(
        [(geom, int(g)) for geom, g in zip(counties.geometry, counties["GEOID"], strict=True)],
        out_shape=shape, transform=transform, fill=0, dtype="int32", all_touched=False,
    )

    valid = np.isfinite(values) & (county_id > 0)
    if nodata is not None:
        valid &= values != nodata
    # Guard against a nodata value the raster failed to declare.
    valid &= values > -9000

    if not valid.any():
        sys.exit(f"No valid pixels found in {raster_path} within {state} counties. "
                 "Wrong state for this raster, or the raster is empty.")

    df = pd.DataFrame({"GEOID": county_id[valid], "v": values[valid].astype("float64")})
    del values, county_id, valid

    g = df.groupby("GEOID")["v"]
    stats = pd.DataFrame({
        "n_pixels": g.size(),
        "mean": g.mean(),
        "median": g.median(),
        "std": g.std(),
        "p10": g.quantile(0.10),
        "p90": g.quantile(0.90),
    }).reset_index()

    names = counties[["GEOID", "NAME"]].copy()
    names["GEOID"] = names["GEOID"].astype(int)
    stats = stats.merge(names.rename(columns={"NAME": "county"}), on="GEOID", how="left")

    stats["coverage_note"] = np.where(
        stats["n_pixels"] < min_pixels, f"under {min_pixels} px - unreliable", ""
    )
    return stats.sort_values("GEOID").reset_index(drop=True)


def join_nass(stats: pd.DataFrame, nass_path: Path, state: str, year: int,
              units: str) -> pd.DataFrame:
    col = {"bu_ac": "yield_bu_ac", "kg_ha": "yield_kg_ha"}[units]
    nass = pd.read_csv(nass_path)
    nass = nass[(nass["year"] == year) & (nass["state"] == state.upper())]
    if nass.empty:
        sys.exit(f"No NASS rows for {state} {year} in {nass_path}")
    nass = nass[["GEOID", col]].rename(columns={col: "nass"})
    nass["GEOID"] = nass["GEOID"].astype(int)

    merged = stats.merge(nass, on="GEOID", how="left")
    merged["diff_mean"] = merged["mean"] - merged["nass"]
    merged["pct_diff_mean"] = 100.0 * merged["diff_mean"] / merged["nass"]
    merged["diff_median"] = merged["median"] - merged["nass"]
    return merged


def agreement(merged: pd.DataFrame, min_pixels: int, stat: str = "mean") -> dict:
    """County-wise agreement. Every county weighted equally -- deliberately NOT
    pixel-weighted, so large counties cannot dominate the summary."""
    d = merged.dropna(subset=["nass"])
    d = d[d["n_pixels"] >= min_pixels]
    if d.empty:
        return {"n_counties": 0}
    x, y = d["nass"].to_numpy(), d[stat].to_numpy()
    resid = y - x
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((x - x.mean()) ** 2).sum())
    return {
        "n_counties": int(len(d)),
        "bias": float(resid.mean()),
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]) if len(d) > 1 else float("nan"),
        "max_abs_diff": float(np.abs(resid).max()),
    }


def scatter(merged: pd.DataFrame, out_png: Path, title: str, units: str,
            min_pixels: int, m: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = merged.dropna(subset=["nass"])
    d = d[d["n_pixels"] >= min_pixels]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    # NASS on the y axis, raster on x. Note the reading this implies: a point
    # ABOVE the 1:1 line is a county where NASS exceeds the raster, i.e. the
    # raster UNDER-estimates. `bias` below is still raster - NASS (the error of
    # the thing being evaluated), so a positive bias shows as points sitting
    # predominantly BELOW the line.
    ax.scatter(d["mean"], d["nass"], s=26, alpha=0.75, edgecolor="none")
    lo = float(min(d["nass"].min(), d["mean"].min()))
    hi = float(max(d["nass"].max(), d["mean"].max()))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], lw=1, ls="--", color="0.4", label="1:1")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"Raster county mean ({units})")
    ax.set_ylabel(f"NASS county yield ({units})")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", frameon=False)
    txt = (f"n = {m['n_counties']} counties\nbias = {m['bias']:+.2f}  (raster − NASS)\n"
           f"MAE = {m['mae']:.2f}\nRMSE = {m['rmse']:.2f}\nR² = {m['r2']:.3f}")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate a yield raster to counties and compare with NASS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raster", required=True, type=Path, help="yield GeoTIFF (label or prediction)")
    p.add_argument("--state", required=True, help="two-letter state code, e.g. IA")
    p.add_argument("--year", required=True, type=int, help="crop year, for the NASS join")
    p.add_argument("--counties", type=Path, default=DEFAULT_COUNTIES)
    p.add_argument("--nass", type=Path, default=DEFAULT_NASS)
    p.add_argument("--units", choices=["bu_ac", "kg_ha"], default="bu_ac",
                   help="units of the raster; the pseudo-label rasters are bu_ac")
    p.add_argument("--min-pixels", type=int, default=100,
                   help="counties with fewer valid pixels are excluded from the summary")
    p.add_argument("--out", type=Path, default=None,
                   help="explicit output CSV path (default: outputs/county_aggregation/"
                        "<source dir>/<raster stem>_county_agg.csv)")
    p.add_argument("--plot", action="store_true", help="also write a scatter PNG")
    args = p.parse_args()

    if not args.raster.exists():
        sys.exit(f"Raster not found: {args.raster}")

    # Guard against --state disagreeing with the state in the filename. This is
    # not pedantry: neighbouring states' rasters overlap along the shared border,
    # so e.g. an MN raster scored against IA counties does NOT fail -- it quietly
    # returns a handful of border counties (Winneshiek, Worth, Mitchell...) and
    # prints a confident summary over 2 counties. Silently-wrong output is worse
    # than an error, and this is easy to hit when looping over states.
    stem_states = [tok for tok in args.raster.stem.split("_") if tok in STATE_FIPS]
    if stem_states and args.state.upper() not in stem_states:
        sys.exit(
            f"State mismatch: --state {args.state.upper()} but the raster filename "
            f"says {'/'.join(stem_states)} ({args.raster.name}).\n"
            f"Neighbouring states share a border, so this would return a few "
            f"overlapping counties and a meaningless summary rather than failing.\n"
            f"Pass --state {stem_states[0]}, or rename the raster if the filename is wrong."
        )

    print(f"raster   : {args.raster}")
    print(f"state    : {args.state}   year: {args.year}   units: {args.units}")

    stats = county_stats(args.raster, args.state, args.counties, args.min_pixels)
    merged = join_nass(stats, args.nass, args.state, args.year, args.units)

    # Collected under one directory rather than written beside the source raster,
    # so these comparisons live together and never clutter (or get mistaken for)
    # the prediction GeoTIFFs. Grouped by the raster's parent directory name --
    # for predictions that is the experiment (cornbelt_5_longer_years_018, ...),
    # which matters because the file stems alone are NOT unique: every experiment
    # produces a `soybeans_IA_2024_pred.tif`, so a flat layout would have later
    # runs silently overwrite earlier ones.
    out_csv = args.out or (
        AGG_DIR / args.raster.resolve().parent.name / f"{args.raster.stem}_county_agg.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    n_matched = int(merged["nass"].notna().sum())
    n_small = int((merged["n_pixels"] < args.min_pixels).sum())
    print(f"\ncounties with raster pixels : {len(merged)}")
    print(f"  of which matched to NASS  : {n_matched}")
    print(f"  below --min-pixels ({args.min_pixels})   : {n_small}  (excluded from the summary)")

    for stat in ("mean", "median"):
        m = agreement(merged, args.min_pixels, stat)
        if not m.get("n_counties"):
            print(f"\n[{stat}] no counties left after filtering")
            continue
        print(f"\n[{stat} vs NASS]  n={m['n_counties']} counties, equally weighted")
        print(f"  bias {m['bias']:+8.3f}   MAE {m['mae']:7.3f}   RMSE {m['rmse']:7.3f}")
        print(f"  R2   {m['r2']:8.4f}   r   {m['pearson_r']:7.4f}   max|diff| {m['max_abs_diff']:.3f}")

    worst = merged.dropna(subset=["nass"]).reindex(
        merged["diff_mean"].abs().sort_values(ascending=False).index
    ).head(5)
    if not worst.empty:
        print("\nlargest county disagreements (mean - NASS):")
        for r in worst.itertuples():
            print(f"  {r.GEOID}  {str(r.county)[:18]:18s} "
                  f"raster {r.mean:7.2f}  nass {r.nass:7.2f}  diff {r.diff_mean:+7.2f} "
                  f"({int(r.n_pixels):,} px)")

    print(f"\nwrote {out_csv}")

    if args.plot:
        m = agreement(merged, args.min_pixels, "mean")
        png = out_csv.with_suffix(".png")
        scatter(merged, png, f"{args.state} {args.year} — county mean vs NASS",
                args.units, args.min_pixels, m)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
