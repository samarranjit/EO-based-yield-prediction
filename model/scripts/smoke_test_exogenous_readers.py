#!/usr/bin/env python
"""Real-data QA of full-resolution soil and nearest-sampled monthly weather maps.

Reads real POLARIS/PRISM/CDL without HLS. Reports crop-pixel coverage and spatial
variation; chip means below are QA summaries only, never the model inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import rasterio

from farm_us.config import load_config
from farm_us.data.exogenous_readers import (
    TIMESTEP_MONTHS,
    SoilReader,
    WeatherSampler,
    chip_weighted_mean,
    missing_exogenous_files,
)
from farm_us.data.raster_readers import GeotiffMonthlyReader
from farm_us.utils.geospatial import ChipWindow

PLAUSIBLE = {"clay": (0, 100), "sand": (0, 100), "silt": (0, 100), "bd": (0.3, 2.5),
             "om": (0.01, 100), "ksat": (0, 5000), "ph": (2, 11)}


def pick_windows(reader, state, year, n, chip, rng, state_geom, min_crop=0.2, tries=400):
    """Random chip windows whose IN-STATE crop fraction clears ``min_crop`` --
    the same rule data/dataset.py:_qualifying_chip_set applies to real training
    chips. Without the state mask, windows from the label raster's bounding box
    can lie wholly in a neighbouring state (e.g. Nebraska inside Iowa's extent),
    where the state-masked PRISM grids are nodata by design."""
    from rasterio.features import geometry_mask

    with rasterio.open(reader._label_path(state, year)) as src:
        h, w = src.height, src.width
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        r0 = int(rng.integers(0, (h - chip) // chip + 1)) * chip
        c0 = int(rng.integers(0, (w - chip) // chip + 1)) * chip
        g = reader.read_label_and_crop(state, year, ChipWindow(r0, c0, chip, chip))
        inside = geometry_mask([state_geom], out_shape=g.crop_mask.shape,
                               transform=g.transform, invert=True)
        if (g.crop_mask & inside).mean() >= min_crop:
            out.append((r0, c0))
    return out


def state_geometries(d, reader, ref_year):
    """Each configured state's county-union polygon, in its label raster's CRS."""
    import geopandas as gpd

    from farm_us.config import STATE_FIPS

    counties = gpd.read_file(d.counties_path)
    out = {}
    for state in d.states:
        with rasterio.open(reader._label_path(state, ref_year)) as src:
            crs = src.crs
        sub = counties[counties["STATEFP"] == STATE_FIPS[state]]
        out[state] = sub.to_crs(crs).geometry.union_all()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/cornbelt5_soil_weather.yaml")
    ap.add_argument("--chips-per-state", type=int, default=6)
    ap.add_argument("--years", type=int, nargs="+", default=[2015, 2018, 2021, 2023])
    ap.add_argument("--ref-year", type=int, default=2021)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg.data
    missing = missing_exogenous_files(d, args.years)
    print(f"pre-flight: {len(missing)} missing soil/weather files for years {args.years}")
    if missing:
        print("\n".join(f"  {p}" for p in missing[:10]))
        return

    reader, soil, weather = GeotiffMonthlyReader(d), SoilReader(d), WeatherSampler(d)
    chip = d.chip_size
    months = TIMESTEP_MONTHS[: d.n_timesteps]
    rng = np.random.default_rng(0)

    soil_vals = {v: [] for v in d.soil_vars}
    # rec[var] -> list of dicts(state, loc, year, month, mean, centroid, within_std)
    rec = {v: [] for v in d.weather_vars}
    n_chip_years = 0
    weather_seen = {v: 0 for v in d.weather_vars}
    weather_finite = {v: 0 for v in d.weather_vars}
    geoms = state_geometries(d, reader, args.ref_year)
    for state in d.states:
        wins = pick_windows(reader, state, args.ref_year, args.chips_per_state, chip, rng,
                            geoms[state])
        for loc, (r0, c0) in enumerate(wins):
            for year in args.years:
                try:
                    g = reader.read_label_and_crop(state, year, ChipWindow(r0, c0, chip, chip))
                except Exception as e:  # noqa: BLE001 -- a state-year may lack a raster
                    print(f"  skip {state} {year}: {e}")
                    continue
                crop = g.crop_mask
                if crop.sum() < 50:
                    continue
                n_chip_years += 1
                s = soil.read_chip(g.transform, g.crs, chip, chip)
                for i, v in enumerate(d.soil_vars):
                    soil_vals[v].append(s[i][crop])
                maps = weather.sample_chip(year, g.transform, g.crs, chip, chip, d.n_timesteps)
                assert maps.shape == (len(d.weather_vars), d.n_timesteps, chip, chip)
                for vi, v in enumerate(d.weather_vars):
                    for ti, m in enumerate(months):
                        f = maps[vi, ti]
                        weather_seen[v] += int(crop.sum())
                        weather_finite[v] += int((crop & np.isfinite(f)).sum())
                        vals = f[crop & np.isfinite(f)]
                        rec[v].append({
                            "state": state, "loc": (state, loc), "year": year, "month": m,
                            "mean": chip_weighted_mean(f, crop.astype(np.float32)),
                            "centroid": float(f[chip // 2, chip // 2]),
                            "within_std": float(vals.std()) if vals.size > 1 else np.nan,
                        })

    print(f"\nreal chip-years sampled: {n_chip_years} "
          f"({len(d.states)} states x up to {args.chips_per_state} locations x {len(args.years)} years)")

    print("\n--- Soil on crop pixels ---")
    print(f"{'var':<6}{'coverage':>10}{'min':>9}{'mean':>9}{'max':>9}  plausible")
    for v in d.soil_vars:
        a = np.concatenate(soil_vals[v])
        fin = a[np.isfinite(a)]
        lo, hi = PLAUSIBLE.get(v, (-np.inf, np.inf))
        ok = fin.size and lo <= fin.min() and fin.max() <= hi
        print(f"{v:<6}{fin.size / max(a.size, 1):>10.4f}{fin.min():>9.2f}{fin.mean():>9.2f}"
              f"{fin.max():>9.2f}  {'OK' if ok else '*** CHECK ***'}")

    print("\n--- Spatial weather: finite crop-pixel coverage (all months) ---")
    for v in d.weather_vars:
        print(f"{v}: {weather_finite[v] / max(weather_seen[v], 1):.4f}")
    print("\n--- Spatial weather variability (QA summaries, not model inputs) ---")
    print(f"{'var':<8}{'no-data':>8}{'within-chip':>12}{'between-chip':>13}{'between-year':>13}"
          f"{'within/btw-yr':>14}{'|centroid-mean|':>16}{'p95':>8}")
    for v in d.weather_vars:
        rows = rec[v]
        no_data = sum(not np.isfinite(r["mean"]) for r in rows)
        within = np.nanmedian([r["within_std"] for r in rows])
        # between-chip: std of chip means across locations, per (year, month)
        btw_chip = np.nanmedian([
            np.nanstd([r["mean"] for r in rows if r["year"] == y and r["month"] == m])
            for y in args.years for m in months])
        # between-year: std of one location's mean across years, per (loc, month)
        locs = {r["loc"] for r in rows}
        btw_year = np.nanmedian([
            np.nanstd([r["mean"] for r in rows if r["loc"] == L and r["month"] == m])
            for L in locs for m in months
            if sum(r["loc"] == L and r["month"] == m for r in rows) > 2])
        dc = np.abs([r["centroid"] - r["mean"] for r in rows])
        dc = dc[np.isfinite(dc)]
        print(f"{v:<8}{no_data:>8}{within:>12.3f}{btw_chip:>13.3f}{btw_year:>13.3f}"
              f"{within / btw_year:>14.3f}{np.median(dc):>16.3f}{np.percentile(dc, 95):>8.3f}")
    print("\n(medians over chips, years and Apr-Nov months; units: ppt mm/month, tmean degC,"
          " vpdmean hPa; no-data = chip-months with no finite PRISM value, expected 0)")


if __name__ == "__main__":
    main()
