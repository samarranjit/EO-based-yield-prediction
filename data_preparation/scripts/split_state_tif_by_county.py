# ============================================================
# Split State-Wide VI TIFF into County-Wise TIFFs
#
# Purpose:
#   Takes multi-county state-wide TIF files and splits them
#   into individual county TIFFs, one per county.
#
# Input:
#   State-year TIF with county_fips band (band 6)
#   Example: soybeans_IL_2014_vi_pixels_nasa_cmr_hls.tif
#
# Output:
#   County-year TIFFs in the same folder
#   Example: soybeans_IL_2014_17001_Cook_vi_pixels_nasa_cmr_hls.tif
# ============================================================

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

# State-wide TIF folder
INPUT_FOLDER = Path("/home/cholab/LabMembers/Samar/EO-based-yield-prediction/data_preparation/data/county_year_wise_VI_tiff/NC")

# Output folder (same as input)
OUTPUT_FOLDER = INPUT_FOLDER

# County name lookup from NASS
NASS_CSV = Path("data_preparation/data/nass/nass_soybeans_county_yield_2014_2024.csv")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_county_names(nass_csv: Path) -> dict:
    """
    Load FIPS → county name mapping from NASS CSV.
    Returns dict: {GEOID: county_name}
    """
    nass = pd.read_csv(nass_csv)
    nass["GEOID"] = nass["GEOID"].astype(str).str.zfill(5)

    # Build FIPS to name mapping
    fips_to_name = {}
    for _, row in nass.iterrows():
        fips = str(int(row["GEOID"]))
        name = str(row.get("county_name", "county")).upper()
        fips_to_name[fips] = name

    return fips_to_name

# ============================================================
# MAIN PROCESSING
# ============================================================

print("Loading county name mapping...")
fips_to_name = get_county_names(NASS_CSV)

# Find all state-wide TIFs
tif_files = sorted(INPUT_FOLDER.glob("soybeans_*_vi_pixels*.tif"))

if not tif_files:
    raise FileNotFoundError(f"No state-wide TIF files found in {INPUT_FOLDER}")

print(f"Found {len(tif_files)} state-wide TIF files\n")

for tif_path in tif_files:
    print(f"Processing: {tif_path.name}")

    with rasterio.open(tif_path) as src:
        # Parse filename to get state and year
        parts = tif_path.stem.split("_")
        state = parts[1]  # IL
        year = parts[2]   # 2014

        # Read band descriptions to find county_fips band
        band_map = {}
        for idx, desc in enumerate(src.descriptions, start=1):
            if desc:
                band_map[desc] = idx

        if "county_fips" not in band_map:
            print(f"  ⚠ Skipping: No county_fips band found\n")
            continue

        county_band_idx = band_map["county_fips"]

        # Read county FIPS band
        county_arr = src.read(county_band_idx).astype(np.float64)

        # Handle nodata
        if src.nodata is not None:
            county_arr = np.where(county_arr == src.nodata, np.nan, county_arr)

        # Find unique counties
        unique_counties = np.unique(county_arr[np.isfinite(county_arr)]).astype(int)
        unique_counties = sorted([c for c in unique_counties if c > 0])

        print(f"  Found {len(unique_counties)} counties")

        # Get profile for output TIFs
        output_profile = src.profile.copy()

        # Process each county
        for county_id in tqdm(unique_counties, desc="  Splitting counties"):
            # Get county name
            county_fips = str(int(county_id)).zfill(5)
            county_name = fips_to_name.get(county_fips, "county")

            # Create county mask
            county_mask = np.isfinite(county_arr) & (np.rint(county_arr) == county_id)

            if not np.any(county_mask):
                continue

            # Find bounding box of this county
            rows, cols = np.where(county_mask)
            if len(rows) == 0:
                continue

            row_min, row_max = rows.min(), rows.max() + 1
            col_min, col_max = cols.min(), cols.max() + 1

            # Read all bands for this county's bounding box
            window = Window(col_min, row_min, col_max - col_min, row_max - row_min)

            data_list = []
            for band_idx in range(1, src.count + 1):
                band_data = src.read(band_idx, window=window)
                data_list.append(band_data)

            # Update profile for county TIF
            county_profile = output_profile.copy()
            county_profile.update(
                height=row_max - row_min,
                width=col_max - col_min,
                transform=rasterio.windows.transform(window, src.transform)
            )

            # Output filename
            output_path = OUTPUT_FOLDER / f"soybeans_{state}_{year}_{county_fips}_{county_name}_vi_pixels_nasa_cmr_hls.tif"

            # Write county TIF
            with rasterio.open(output_path, "w", **county_profile) as dst:
                for band_idx, band_data in enumerate(data_list, start=1):
                    dst.write(band_data, band_idx)
                    # Copy band descriptions
                    if band_idx <= len(src.descriptions):
                        dst.set_band_description(band_idx, src.descriptions[band_idx - 1] or f"Band {band_idx}")

        print(f"  ✓ Split {state} {year} into {len(unique_counties)} county TIFs\n")

print("✓ All state-wide TIFs split successfully!")
