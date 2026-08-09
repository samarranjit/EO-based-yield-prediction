# ============================================================
# Distribute County NASS Yield to Pixels using Ridge VI Weights
#
# Purpose:
#   This script converts county-level NASS soybean yield into
#   pixel-level pseudo-yield labels using vegetation-index weights.
#
# Key method:
#   Ridge is used only to learn VI importance weights.
#   We do NOT directly predict pixel yield using the Ridge intercept.
#
#   Instead:
#      1. Compute within-county VI z-scores.
#      2. Create a vegetation score using Ridge-derived VI weights.
#      3. Convert vegetation score to a bounded raw weight.
#      4. Normalize raw weights inside each county.
#      5. Multiply normalized weights by NASS county yield.
#
# This preserves the county mean:
#      mean(pixel pseudo-yield within county) = NASS county yield
#
# Inputs:
#   1) ridge_vi_coefficients_for_gee.csv
#   2) NASS county yield CSV
#   3) VI GeoTIFF
#
# Outputs:
#   1) Pixel-level pseudo-yield GeoTIFF
#   2) County diagnostics CSV
#   3) Optional comparison plot
# ============================================================

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# 1) CONFIG
# ============================================================

# Ridge coefficient output from Script 1.
COEF_CSV = Path("/home/cholab/LabMembers/Samar/EO-based-yield-prediction/data_preparation/data/ridge_vi_weight_outputs/ridge_vi_coefficients_for_gee.csv")

# NASS county soybean yield CSV.
NASS_CSV = Path(
    "/home/cholab/LabMembers/Samar/EO-based-yield-prediction/data_preparation/data/nass/nass_soybeans_county_yield_2014_2024.csv"
)

# Input VI TIFF.
#
# Supported formats:
#
# 1) County-year 4-band TIFF:
#      bands: NDVI, EVI, GCVI, NDWI
#      filename example:
#      soybeans_DE_2014_10001_Kent_vi_pixels_nasa_cmr_hls.tif
#
# 2) State-year minimal 8-band TIFF:
#      bands:
#        1 NDVI_mean
#        2 EVI_mean
#        3 GCVI_mean
#        4 NDWI_mean
#        5 min_index_obsCount
#        6 county_fips
#        7 crop_mask
#        8 valid_vi_mask
#
#      filename example:
#      soybeans_DE_2014_vi_stack_minimal.tif
DEFAULT_VI_TIF_PATH = "/home/cholab/LabMembers/Samar/EO-based-yield-prediction/data_preparation/data/county_year_wise_VI_tiff/DE/soybeans_DE_2014_10003_New_Castle_vi_pixels_nasa_cmr_hls.tif"

parser = argparse.ArgumentParser(description="Distribute county NASS yield to pixels using Ridge VI weights.")
parser.add_argument(
    "--vi-tif-path",
    type=str,
    default=DEFAULT_VI_TIF_PATH,
    help="Path to the input VI GeoTIFF.",
)


args = parser.parse_args()

VI_TIF_PATH = Path(args.vi_tif_path)
# STATE = args.state

# Year for NASS yield lookup.
# Set to None to auto-detect from TIF filename (e.g., 2014 from "soybeans_DE_2014_...")
# Set to an integer to override and use that year explicitly (e.g., YEAR_OVERRIDE = 2014)
YEAR_OVERRIDE = None

# Output directory.
OUTPUT_DIR = Path("/home/cholab/LabMembers/Samar/EO-based-yield-prediction/data_preparation/data/pixel_distributed_yield/Intermediate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Redistribution settings.
# These should match the values saved by Script 1 unless you intentionally change them.
REDIST_STRENGTH = 0.15
RAW_WEIGHT_MIN = 0.50
RAW_WEIGHT_MAX = 1.50

# Minimum number of valid VI pixels required to redistribute a county.
MIN_VALID_PIXELS_PER_COUNTY = 10

# If True:
#   Crop pixels with missing/invalid VI receive neutral weight = 1.
#   This creates a more complete crop map.
#
# If False:
#   Only valid VI pixels receive pseudo-yield.
#
# For training labels, False can be cleaner.
# For map completeness, True can be useful.
FILL_INVALID_CROP_PIXELS_WITH_COUNTY_MEAN = True

# Make comparison plot.
MAKE_PLOT = True

# Maximum image dimension shown in plot.
# Large rasters are downsampled only for visualization.
MAX_PLOT_DIMENSION = 1800

# Output nodata value.
OUT_NODATA = -9999.0


# ============================================================
# 2) HELPER FUNCTIONS
# ============================================================

FEATURES = ["NDVI_mean", "EVI_mean", "GCVI_mean", "NDWI_mean"]


def normalize_fips_value(value) -> str:
    """
    Convert FIPS/GEOID to clean 5-character string.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\.0$", "", text)
    return text.zfill(5)


def normalize_fips_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series of FIPS/GEOID values to 5-character strings.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def parse_tif_metadata(path: Path):
    """
    Parse state/year/county information from common filename patterns.

    Supported examples:
      soybeans_DE_2014_10001_Kent_vi_pixels_nasa_cmr_hls.tif
      soybeans_DE_2014_vi_stack_minimal.tif
    """
    stem = path.stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Cannot parse state/year from filename: {path.name}"
        )

    # Expected beginning:
    # soybeans_{STATE}_{YEAR}_...
    state = parts[1]
    year = int(parts[2])

    county_fips = None
    county_name = None

    # County-year format has a 5-digit FIPS after year.
    if len(parts) > 3 and re.fullmatch(r"\d{5}", parts[3]):
        county_fips = parts[3]
        county_name = parts[4] if len(parts) > 4 else "county"

    return {
        "state": state,
        "year": year,
        "county_fips": county_fips,
        "county_name": county_name,
    }


def load_weights(coef_csv: Path) -> dict:
    """
    Load normalized VI redistribution weights from Script 1 output.

    Preferred column:
      gee_weight_normalized

    Fallback:
      ridge_coef_standardized normalized by absolute sum.
    """
    coef_df = pd.read_csv(coef_csv)

    if "feature" not in coef_df.columns:
        raise ValueError("Coefficient CSV must contain a 'feature' column.")

    if "gee_weight_normalized" in coef_df.columns:
        weight_col = "gee_weight_normalized"
    elif "ridge_coef_standardized" in coef_df.columns:
        weight_col = "ridge_coef_standardized"
    else:
        raise ValueError(
            "Coefficient CSV must contain either 'gee_weight_normalized' "
            "or 'ridge_coef_standardized'."
        )

    missing = [f for f in FEATURES if f not in set(coef_df["feature"])]
    if missing:
        raise ValueError(
            "Coefficient CSV is missing required features:\n"
            + "\n".join(missing)
        )

    values = []
    for f in FEATURES:
        val = coef_df.loc[coef_df["feature"] == f, weight_col].iloc[0]
        values.append(float(val))

    values = np.array(values, dtype=float)

    # If using raw ridge coefficients, normalize them.
    if weight_col == "ridge_coef_standardized":
        denom = np.sum(np.abs(values))
        if denom == 0:
            raise RuntimeError("Cannot normalize zero Ridge coefficients.")
        values = values / denom

    return dict(zip(FEATURES, values))


def read_nass(nass_csv: Path) -> pd.DataFrame:
    """
    Load and clean NASS yield table.
    """
    nass = pd.read_csv(nass_csv)

    required = ["GEOID", "year", "yield_bu_ac"]
    missing = [c for c in required if c not in nass.columns]
    if missing:
        raise ValueError(
            "NASS CSV is missing required columns:\n" + "\n".join(missing)
        )

    nass["GEOID"] = normalize_fips_series(nass["GEOID"])
    nass["year"] = pd.to_numeric(nass["year"], errors="coerce").astype("Int64")
    nass["yield_bu_ac"] = pd.to_numeric(nass["yield_bu_ac"], errors="coerce")

    nass = nass.dropna(subset=["GEOID", "year", "yield_bu_ac"])
    nass["year"] = nass["year"].astype(int)

    keep = ["GEOID", "year", "yield_bu_ac"]
    if "state" in nass.columns:
        keep.append("state")
    if "county_name" in nass.columns:
        keep.append("county_name")

    return nass[keep].drop_duplicates(subset=["GEOID", "year"])


def get_nass_yield(nass: pd.DataFrame, county_fips: str, year: int) -> float:
    """
    Get NASS yield for one county-year.
    """
    county_fips = normalize_fips_value(county_fips)

    row = nass[
        (nass["GEOID"] == county_fips)
        & (nass["year"] == int(year))
    ]

    if len(row) == 0:
        raise KeyError(
            f"No NASS yield found for GEOID={county_fips}, year={year}"
        )

    return float(row["yield_bu_ac"].iloc[0])


def raster_band_map(src):
    """
    Determine band mapping for input VI TIFF.

    Uses band descriptions if available.
    Otherwise falls back to expected band order.
    """
    descriptions = list(src.descriptions)

    # Try reading by band descriptions first.
    desc_map = {}
    for idx, desc in enumerate(descriptions, start=1):
        if desc is not None:
            desc_map[str(desc)] = idx

    required_desc = ["NDVI_mean", "EVI_mean", "GCVI_mean", "NDWI_mean"]

    if all(b in desc_map for b in required_desc):
        band_map = {
            "NDVI_mean": desc_map["NDVI_mean"],
            "EVI_mean": desc_map["EVI_mean"],
            "GCVI_mean": desc_map["GCVI_mean"],
            "NDWI_mean": desc_map["NDWI_mean"],
        }

        if "county_fips" in desc_map:
            band_map["county_fips"] = desc_map["county_fips"]
        if "crop_mask" in desc_map:
            band_map["crop_mask"] = desc_map["crop_mask"]
        if "valid_vi_mask" in desc_map:
            band_map["valid_vi_mask"] = desc_map["valid_vi_mask"]

        return band_map

    # Fallback by band count.
    if src.count >= 8:
        # Minimal state-year TIFF from GEE.
        return {
            "NDVI_mean": 1,
            "EVI_mean": 2,
            "GCVI_mean": 3,
            "NDWI_mean": 4,
            "min_index_obsCount": 5,
            "county_fips": 6,
            "crop_mask": 7,
            "valid_vi_mask": 8,
        }

    if src.count >= 4:
        # County-year 4-band TIFF.
        return {
            "NDVI_mean": 1,
            "EVI_mean": 2,
            "GCVI_mean": 3,
            "NDWI_mean": 4,
        }

    raise ValueError(
        f"Input TIFF has {src.count} bands. Expected at least 4 bands."
    )


def read_float_band(src, band_idx: int) -> np.ndarray:
    """
    Read one raster band as float32 and convert nodata to NaN.
    """
    arr = src.read(band_idx).astype(np.float32)
    nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def county_zscore(arr: np.ndarray, mask: np.ndarray):
    """
    Compute z-score using only pixels in mask.
    """
    vals = arr[mask]

    mu = np.nanmean(vals)
    sd = np.nanstd(vals)

    if not np.isfinite(sd) or sd == 0:
        sd = 1.0

    z = (arr - mu) / sd

    return z, float(mu), float(sd)


def downsample_for_plot(arr: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    """
    Downsample large arrays for plotting only.
    """
    max_current = max(arr.shape)
    if max_current <= max_dim:
        return arr

    step = int(np.ceil(max_current / max_dim))
    return arr[::step, ::step]


# ============================================================
# 3) LOAD METADATA, WEIGHTS, AND NASS
# ============================================================

meta = parse_tif_metadata(VI_TIF_PATH)

state = meta["state"]
year = YEAR_OVERRIDE if YEAR_OVERRIDE is not None else meta["year"]
parsed_county_fips = meta["county_fips"]
parsed_county_name = meta["county_name"]

print("Parsed input TIFF metadata:")
print(f"  File        : {VI_TIF_PATH.name}")
print(f"  State       : {state}")
print(f"  Year        : {year}", end="")
if YEAR_OVERRIDE is not None:
    print(f" (overridden, auto-detected: {meta['year']})")
else:
    print(" (auto-detected from filename)")
print(f"  County FIPS : {parsed_county_fips}")
print(f"  County name : {parsed_county_name}")

weights = load_weights(COEF_CSV)

print("\nLoaded normalized VI redistribution weights:")
for f in FEATURES:
    print(f"  {f:12s}: {weights[f]: .8f}")

nass = read_nass(NASS_CSV)


# ============================================================
# 4) READ VI TIFF
# ============================================================

if not VI_TIF_PATH.exists():
    raise FileNotFoundError(f"VI TIFF file not found: {VI_TIF_PATH}")

with rasterio.open(VI_TIF_PATH) as src:
    band_map = raster_band_map(src)

    ndvi = read_float_band(src, band_map["NDVI_mean"])
    evi = read_float_band(src, band_map["EVI_mean"])
    gcvi = read_float_band(src, band_map["GCVI_mean"])
    ndwi = read_float_band(src, band_map["NDWI_mean"])

    profile = src.profile.copy()
    crs = src.crs
    transform = src.transform

    # Optional bands.
    county_arr = None
    crop_mask = None
    valid_vi_mask = None

    if "county_fips" in band_map:
        raw_county = src.read(band_map["county_fips"]).astype(np.float64)
        if src.nodata is not None:
            raw_county = np.where(raw_county == src.nodata, np.nan, raw_county)
        county_arr = np.where(np.isfinite(raw_county), np.rint(raw_county), np.nan)

    if "crop_mask" in band_map:
        crop_raw = src.read(band_map["crop_mask"])
        crop_mask = crop_raw == 1

    if "valid_vi_mask" in band_map:
        valid_raw = src.read(band_map["valid_vi_mask"])
        valid_vi_mask = valid_raw == 1

print("\nInput raster details:")
print(f"  Shape : {ndvi.shape}")
print(f"  CRS   : {crs}")
print(f"  Bands : {band_map}")


# ============================================================
# 5) BUILD COUNTY/CROP/VALID MASKS
# ============================================================

finite_vi_mask = (
    np.isfinite(ndvi)
    & np.isfinite(evi)
    & np.isfinite(gcvi)
    & np.isfinite(ndwi)
)

# If no crop mask exists, use finite VI pixels as the crop area.
if crop_mask is None:
    crop_mask = finite_vi_mask.copy()

# If no valid VI mask exists, use finite VI pixels.
if valid_vi_mask is None:
    valid_vi_mask = finite_vi_mask.copy()

# If no county_fips raster exists, this must be a county-year TIFF.
if county_arr is None:
    if parsed_county_fips is None:
        raise ValueError(
            "No county_fips raster band found and no county FIPS could be parsed "
            "from the filename."
        )

    county_arr = np.full(ndvi.shape, np.nan, dtype=np.float64)
    county_arr[crop_mask] = int(parsed_county_fips)

base_valid_mask = crop_mask & valid_vi_mask & finite_vi_mask

print(f"\nPixels:")
print(f"  Crop pixels       : {int(np.sum(crop_mask)):,}")
print(f"  Valid VI pixels   : {int(np.sum(base_valid_mask)):,}")


# ============================================================
# 6) REDISTRIBUTE YIELD COUNTY BY COUNTY
# ============================================================

pseudo_yield = np.full(ndvi.shape, np.nan, dtype=np.float32)
county_yield_uniform = np.full(ndvi.shape, np.nan, dtype=np.float32)

diagnostic_rows = []

unique_counties = np.unique(county_arr[np.isfinite(county_arr)]).astype(int)
unique_counties = sorted(unique_counties)

# Filter out dummy/nodata county values (0, 00000)
unique_counties = [c for c in unique_counties if c > 0]

print(f"\nFound {len(unique_counties)} county/county-like IDs in raster.")

for county_id in unique_counties:
    county_fips = normalize_fips_value(county_id)

    # Create mask matching this county (handles NaN safely without RuntimeWarning)
    county_mask = np.isfinite(county_arr) & (np.rint(county_arr) == county_id)

    # Pixels used to calculate county VI z-scores.
    valid_for_stats = county_mask & base_valid_mask

    n_valid = int(np.sum(valid_for_stats))
    n_crop = int(np.sum(county_mask & crop_mask))

    if n_valid < MIN_VALID_PIXELS_PER_COUNTY:
        print(
            f"Skipping county {county_fips}: only {n_valid} valid VI pixels."
        )
        continue

    try:
        nass_yield = get_nass_yield(nass, county_fips, year)
    except KeyError as e:
        print(f"Skipping county {county_fips}: {e}")
        continue

    print(
        f"Processing county {county_fips}: "
        f"NASS={nass_yield:.2f} bu/ac, "
        f"valid VI pixels={n_valid:,}, crop pixels={n_crop:,}"
    )

    # --------------------------------------------------------
    # 6.1) Within-county VI z-scores
    # --------------------------------------------------------

    z_ndvi, ndvi_mu, ndvi_sd = county_zscore(ndvi, valid_for_stats)
    z_evi, evi_mu, evi_sd = county_zscore(evi, valid_for_stats)
    z_gcvi, gcvi_mu, gcvi_sd = county_zscore(gcvi, valid_for_stats)
    z_ndwi, ndwi_mu, ndwi_sd = county_zscore(ndwi, valid_for_stats)

    # --------------------------------------------------------
    # 6.2) Vegetation score
    # --------------------------------------------------------

    veg_score = (
        weights["NDVI_mean"] * z_ndvi
        + weights["EVI_mean"] * z_evi
        + weights["GCVI_mean"] * z_gcvi
        + weights["NDWI_mean"] * z_ndwi
    )

    # --------------------------------------------------------
    # 6.3) Raw pixel weights
    # --------------------------------------------------------

    raw_weight = 1.0 + REDIST_STRENGTH * veg_score
    raw_weight = np.clip(raw_weight, RAW_WEIGHT_MIN, RAW_WEIGHT_MAX)

    # For invalid VI pixels inside the crop mask, optionally assign neutral weight.
    if FILL_INVALID_CROP_PIXELS_WITH_COUNTY_MEAN:
        output_mask = county_mask & crop_mask
        raw_weight = np.where(output_mask & np.isfinite(raw_weight), raw_weight, raw_weight)
        raw_weight[output_mask & ~valid_for_stats] = 1.0
    else:
        output_mask = valid_for_stats

    raw_weight[~output_mask] = np.nan

    # --------------------------------------------------------
    # 6.4) Normalize weights inside county
    # --------------------------------------------------------

    mean_raw_weight = np.nanmean(raw_weight[output_mask])

    if not np.isfinite(mean_raw_weight) or mean_raw_weight == 0:
        print(f"Skipping county {county_fips}: invalid mean raw weight.")
        continue

    normalized_weight = raw_weight / mean_raw_weight

    # --------------------------------------------------------
    # 6.5) Pseudo-yield
    # --------------------------------------------------------

    county_pseudo_yield = nass_yield * normalized_weight

    pseudo_yield[output_mask] = county_pseudo_yield[output_mask].astype(np.float32)
    county_yield_uniform[output_mask] = nass_yield

    out_vals = county_pseudo_yield[output_mask]

    diagnostic_rows.append({
        "state": state,
        "year": year,
        "county_fips": county_fips,
        "nass_yield_bu_ac": nass_yield,
        "crop_pixel_count": n_crop,
        "valid_vi_pixel_count": n_valid,
        "output_pixel_count": int(np.sum(output_mask)),
        "ndvi_mean_for_zscore": ndvi_mu,
        "ndvi_std_for_zscore": ndvi_sd,
        "evi_mean_for_zscore": evi_mu,
        "evi_std_for_zscore": evi_sd,
        "gcvi_mean_for_zscore": gcvi_mu,
        "gcvi_std_for_zscore": gcvi_sd,
        "ndwi_mean_for_zscore": ndwi_mu,
        "ndwi_std_for_zscore": ndwi_sd,
        "mean_raw_weight": float(mean_raw_weight),
        "pseudo_yield_mean_bu_ac": float(np.nanmean(out_vals)),
        "pseudo_yield_min_bu_ac": float(np.nanmin(out_vals)),
        "pseudo_yield_median_bu_ac": float(np.nanmedian(out_vals)),
        "pseudo_yield_max_bu_ac": float(np.nanmax(out_vals)),
        "pseudo_yield_std_bu_ac": float(np.nanstd(out_vals)),
        "mean_difference_from_nass": float(np.nanmean(out_vals) - nass_yield),
    })


# ============================================================
# 7) SAVE OUTPUTS
# ============================================================

if parsed_county_fips is not None:
    output_stem = f"soybeans_{state}_{year}_{parsed_county_fips}_{parsed_county_name}_pseudo_yield"
else:
    output_stem = f"soybeans_{state}_{year}_pseudo_yield"

OUTPUT_TIF = OUTPUT_DIR / f"{output_stem}.tif"
OUTPUT_CSV = OUTPUT_DIR / f"{output_stem}_county_diagnostics.csv"
OUTPUT_PLOT = OUTPUT_DIR / f"{output_stem}_comparison.png"

# Convert NaN to nodata for GeoTIFF output.
pseudo_yield_out = np.where(
    np.isfinite(pseudo_yield),
    pseudo_yield,
    OUT_NODATA
).astype(np.float32)

output_profile = profile.copy()
output_profile.update(
    dtype=rasterio.float32,
    count=1,
    nodata=OUT_NODATA,
    compress="lzw",
)

with rasterio.open(OUTPUT_TIF, "w", **output_profile) as dst:
    dst.write(pseudo_yield_out, 1)
    dst.set_band_description(1, "pseudo_yield_bu_ac")

diag_df = pd.DataFrame(diagnostic_rows)
diag_df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved outputs:")
print(f"  Pseudo-yield TIFF      : {OUTPUT_TIF}")
print(f"  County diagnostics CSV : {OUTPUT_CSV}")

if len(diag_df) > 0:
    print("\nCounty mean preservation check:")
    print(diag_df[[
        "county_fips",
        "nass_yield_bu_ac",
        "pseudo_yield_mean_bu_ac",
        "mean_difference_from_nass"
    ]])


# ============================================================
# 8) OPTIONAL PLOT
# ============================================================

if MAKE_PLOT:
    plot_county = county_yield_uniform.copy()
    plot_pseudo = pseudo_yield.copy()

    plot_county = downsample_for_plot(plot_county, MAX_PLOT_DIMENSION)
    plot_pseudo = downsample_for_plot(plot_pseudo, MAX_PLOT_DIMENSION)

    county_vals = plot_county[np.isfinite(plot_county)]
    pseudo_vals = plot_pseudo[np.isfinite(plot_pseudo)]

    if len(pseudo_vals) > 0:
        # Left plot: use county-uniform scale (will be flat since uniform)
        county_vmin = np.nanmin(county_vals) if len(county_vals) > 0 else 0
        county_vmax = np.nanmax(county_vals) if len(county_vals) > 0 else 1

        # Right plot: use pseudo-yield scale (shows distribution variation)
        pseudo_vmin = np.nanpercentile(pseudo_vals, 2)
        pseudo_vmax = np.nanpercentile(pseudo_vals, 98)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: County-uniform with its own scale
        im1 = axes[0].imshow(plot_county, cmap="RdYlGn", vmin=county_vmin, vmax=county_vmax)
        axes[0].set_title(
            f"County NASS Yield\n{state} {year}\nUniform within county",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_xlabel("Column")
        axes[0].set_ylabel("Row")
        cbar1 = plt.colorbar(im1, ax=axes[0], label="Yield (bu/ac)")
        cbar1.set_label(f"Yield (bu/ac)\n[{county_vmin:.2f}, {county_vmax:.2f}]")

        # Right: Distributed pseudo-yield with its own scale
        im2 = axes[1].imshow(plot_pseudo, cmap="RdYlGn", vmin=pseudo_vmin, vmax=pseudo_vmax)
        axes[1].set_title(
            f"Distributed Pseudo-Yield\n{state} {year}\nVI-weighted, county mean preserved",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_xlabel("Column")
        axes[1].set_ylabel("Row")
        cbar2 = plt.colorbar(im2, ax=axes[1], label="Yield (bu/ac)")
        cbar2.set_label(f"Yield (bu/ac)\n[{pseudo_vmin:.2f}, {pseudo_vmax:.2f}]")

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
        print(f"  Plot saved             : {OUTPUT_PLOT}")
        plt.show()
    else:
        print("No finite pseudo-yield values available for plotting.")

print("\n✓ Script complete.")