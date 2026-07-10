"""
Export USDA NASS CDL crop-presence masks via Google Earth Engine.

One GeoTIFF is exported per (state, year) — a binary mask where:
  1 = target crop pixel (per CDL_CROP_CODE in config)
  0 = non-crop pixel

Files land in Google Drive folder defined by DRIVE_FOLDER, then must be
downloaded manually (or via gdown/Drive API) into data/cdl_masks/ with
this naming convention:

    cdl_<crop>_<STATE>_<YEAR>.tif   e.g. cdl_soybeans_MD_2020.tif

Prerequisites:
    earthengine authenticate
    python scripts/03_export_cdl_masks_gee.py

Monitor task progress at:
    https://code.earthengine.google.com/tasks
"""

import sys
import ee
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CDL_MASK_DIR, CDL_CROP_CODE, CROP_NAME,
    STATE_ALPHA, STATE_FIPS, START_YEAR, END_YEAR, TARGET_CRS,
)

GEE_PROJECT = "eo-yield-prediction"   # update to your GEE project id
DRIVE_FOLDER = "cdl_masks"
EPSG_CODE = int(TARGET_CRS.split(":")[1])


def get_state_geometry(state_fips: str) -> ee.Geometry:
    states = ee.FeatureCollection("TIGER/2018/States")
    return states.filter(ee.Filter.eq("STATEFP", state_fips)).first().geometry()


def export_one(state: str, year: int) -> ee.batch.Task:
    geom = get_state_geometry(STATE_FIPS[state])

    cdl_image = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .first()
        .select("cropland")
    )

    # Binary: 1 where CDL code matches target crop
    crop_mask = cdl_image.eq(CDL_CROP_CODE).rename("crop_mask").toFloat()

    task_name = f"cdl_{CROP_NAME.lower()}_{state}_{year}"
    task = ee.batch.Export.image.toDrive(
        image=crop_mask,
        description=task_name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=task_name,
        region=geom,
        scale=30,
        crs=f"EPSG:{EPSG_CODE}",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def main():
    ee.Initialize(project=GEE_PROJECT)

    combos = [(s, y) for s in STATE_ALPHA for y in range(START_YEAR, END_YEAR + 1)]
    print(f"Submitting {len(combos)} CDL export tasks to GEE…\n")

    submitted, failed = [], []
    for state, year in combos:
        try:
            export_one(state, year)
            submitted.append((state, year))
            print(f"  submitted  {state}  {year}")
        except Exception as exc:
            failed.append((state, year, str(exc)))
            print(f"  FAILED     {state}  {year}  —  {exc}")

    print(f"\nSubmitted: {len(submitted)}   Failed: {len(failed)}")
    print(f"Monitor:   https://code.earthengine.google.com/tasks")
    print(f"\nAfter tasks complete, download .tif files from Drive folder '{DRIVE_FOLDER}' to:")
    print(f"  {CDL_MASK_DIR}")
    print(f"\nExpected filename pattern:  cdl_{CROP_NAME.lower()}_<STATE>_<YEAR>.tif")

    if failed:
        print("\nFailed tasks:")
        for state, year, err in failed:
            print(f"  {state} {year}: {err}")


if __name__ == "__main__":
    main()
