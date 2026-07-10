"""
Download US county boundaries from Census Bureau TIGER/Line shapefiles (2023).
Filters to the target states defined in config and saves as a GeoPackage.

    python scripts/02_download_counties.py
"""

import sys
import io
import shutil
import zipfile
import requests
import geopandas as gpd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import COUNTIES_GPKG, STATE_FIPS, TARGET_CRS

TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"


def main():
    tmp_dir = COUNTIES_GPKG.parent / "_tiger_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading TIGER/Line county shapefile (~75 MB)…")
    resp = requests.get(TIGER_URL, timeout=180, stream=True)
    resp.raise_for_status()

    raw = b"".join(resp.iter_content(chunk_size=1 << 20))
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(tmp_dir)

    shp_path = next(tmp_dir.glob("*.shp"))
    print(f"Reading shapefile: {shp_path.name}")
    counties = gpd.read_file(shp_path)

    target_fips = set(STATE_FIPS.values())
    counties = counties[counties["STATEFP"].isin(target_fips)].copy()
    counties = counties.to_crs(TARGET_CRS)

    # Ensure GEOID is a zero-padded 5-character string
    if "GEOID" not in counties.columns:
        counties["GEOID"] = counties["STATEFP"].str.zfill(2) + counties["COUNTYFP"].str.zfill(3)

    counties.to_file(COUNTIES_GPKG, driver="GPKG")
    print(f"Saved {len(counties)} counties → {COUNTIES_GPKG}")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
