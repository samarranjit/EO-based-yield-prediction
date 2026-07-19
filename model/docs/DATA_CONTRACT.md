# Data Contract

Everything the pipeline expects of the inputs. Validated by `farm_us.data`
readers, `inventory.py`, and the `inventory` / `build-manifest` CLI commands.

## Spectral band order (MANDATORY)
`[BLUE, GREEN, RED, NIR_NARROW, SWIR1, SWIR2]` — the exact order Prithvi-EO-2.0
was pretrained on. Band order is **validated against raster metadata/config, not
filenames**. Do not reorder.

## Temporal order (T = 8)
Apr, May, Jun, Jul, Aug, Sep, Oct, **Nov 1–15**. Each composite's representative
date = **midpoint of its window** (documented default; real acquisition/composite
metadata overrides when available). Prithvi metadata:
`temporal_coords[B,T,2] = (year, day-of-year)`, `location_coords[B,2] = (lat, lon)`.

## Units & scaling
- **Imagery**: HLS surface reflectance. Raw DN → reflectance via `hls_scale`
  (default 1e-4). Never treat DN as reflectance without checking; never treat
  no-data as valid 0.
- **Labels**: kg/ha (NASS bu/ac × 67.251 for soybeans). Reported de-standardized
  in kg/ha, kg/ac, and bu/ac.

## No-data semantics
- Label nodata = **−9999.0** → converted to NaN, excluded from loss/metrics.
- HLS nodata (per-band) → month marked invalid; handled by the missing-month policy.
- CDL masks are 0/1 (no nodata); crop pixels are value > 0.5.
- **Validity is never inferred from `yield == 0`** — real yields can be low/zero.

## Masks (kept explicit, combined at point of use)
`valid = crop_mask ∧ label_valid ∧ hls_valid` (optionally ∧ county membership).

## Geospatial alignment (all rasters)
- CRS **EPSG:5070**, resolution **30 m**, dtype float32.
- Label and CDL for a (state, year) share size + transform (verified for real IA 2018).
- Statewide rasters are large (e.g. IA ≈ 17795×11672) → **windowed reads only**.

## Storage patterns (adapter classes, one interface)
1. `geotiff_monthly` — one 6-band GeoTIFF per (state, year, month) *(implemented)*.
2. `state_year_stack` — one stack per (state, year) *(interface declared)*.
3. `chips_npz` — pre-extracted chips (NPZ/Zarr) *(interface declared)*.
4. `manifest_cog` — manifest of COGs *(interface declared)*.

Expected `geotiff_monthly` layout:
```
{imagery_root}/{year}/hls_{crop}_{STATE}_{YEAR}_{MON}.tif        # 6 bands, MON∈APR..NOV
{label_root}/{year}/nass_{crop}_yield_{STATE}_{YEAR}_{variant}_30m_{crop}_only.tif
{cdl_root}/cdl_{crop}_{STATE}_{YEAR}.tif
```

## Manifest fields (Parquet, one row per chip)
`sample_id, crop, year, state, state_fips, label_path, label_variant, label_units,
label_source, ridge_provenance, row_off, col_off, chip_size, stride, crs,
resolution_m, center_lat, center_lon, dates, n_valid_crop_px, valid_hls_fraction,
valid_label_fraction, split, qc_flags`. The manifest carries a **fingerprint**
(hash of crop/variant/states/years/chip/stride) in `df.attrs`.

## Thresholds (config)
`min_crop_fraction`, `min_label_fraction`, `min_month_valid_fraction`,
`max_missing_months`. Missing-month policy ∈ `{drop, zero_fill, temporal_interp
(default), nearest_valid}` — cloudy/missing months are **never silently
substituted**; the policy is explicit and recorded.
