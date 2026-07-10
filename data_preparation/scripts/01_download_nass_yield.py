"""
Download county-level crop yield data from USDA NASS Quick Stats API.

Outputs one CSV to data/nass/ covering all target states and years.
Set NASS_API_KEY in the environment before running.

    export NASS_API_KEY="your_key"
    python scripts/01_download_nass_yield.py
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    NASS_CSV, CROP_NAME, STATE_ALPHA, START_YEAR, END_YEAR,
    CONVERSION_FACTOR, VALUE_COL,
)

NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"


RATE_LIMIT_DELAY = 0.5    # seconds between every request
RETRY_DELAYS     = [5, 15, 30]  # seconds to wait on 403 before each retry


def fetch_nass_yield(api_key: str, crop: str, state: str, year: int) -> pd.DataFrame:
    params = {
        "key": api_key,
        "commodity_desc": crop,
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "class_desc": "ALL CLASSES",
        "prodn_practice_desc": "ALL PRODUCTION PRACTICES",
        "agg_level_desc": "COUNTY",
        "state_alpha": state,
        "year": str(year),
        "format": "JSON",
    }
    for attempt, wait in enumerate([0] + RETRY_DELAYS):
        if wait:
            print(f"         rate-limited — waiting {wait}s then retrying…", flush=True)
            time.sleep(wait)
        resp = requests.get(NASS_API_URL, params=params, timeout=30)
        if resp.status_code == 403 and attempt < len(RETRY_DELAYS):
            continue
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return pd.DataFrame(data) if data else pd.DataFrame()
    resp.raise_for_status()  # final raise if all retries exhausted


def parse_yield(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "state_alpha": "state",
        "state_fips_code": "state_fips",
        "county_name": "county_name",
        "county_code": "county_fips",
        "year": "year",
        "Value": "yield_bu_ac",
    }
    df = df[[c for c in rename if c in df.columns]].rename(columns=rename)
    df["yield_bu_ac"] = pd.to_numeric(
        df["yield_bu_ac"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df = df.dropna(subset=["yield_bu_ac"])
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(3)
    df["GEOID"] = df["state_fips"] + df["county_fips"]
    df["year"] = df["year"].astype(int)
    df[VALUE_COL] = (df["yield_bu_ac"] * CONVERSION_FACTOR).round(4)
    # Drop duplicate GEOIDs within a year (take mean if any remain after API filter)
    df = (
        df.groupby(["state", "GEOID", "year"], as_index=False)
        .agg({"county_name": "first", "state_fips": "first", "yield_bu_ac": "mean", VALUE_COL: "mean"})
    )
    return df


def main():
    api_key = os.environ.get("NASS_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Set NASS_API_KEY environment variable before running.")

    records = []
    combos = [(s, y) for s in STATE_ALPHA for y in range(START_YEAR, END_YEAR + 1)]
    total = len(combos)

    for i, (state, year) in enumerate(combos, 1):
        print(f"[{i}/{total}]  {CROP_NAME}  {state}  {year}", flush=True)
        try:
            time.sleep(RATE_LIMIT_DELAY)
            raw = fetch_nass_yield(api_key, CROP_NAME, state, year)
            if raw.empty:
                print(f"         no data returned")
                continue
            parsed = parse_yield(raw)
            records.append(parsed)
            print(f"         {len(parsed)} counties")
        except Exception as exc:
            print(f"         WARNING: {exc}")

    if not records:
        raise RuntimeError("No records downloaded — check API key and crop name.")

    df_all = pd.concat(records, ignore_index=True)
    df_all.to_csv(NASS_CSV, index=False)
    print(f"\nSaved {len(df_all)} records → {NASS_CSV}")
    summary = df_all.groupby("year")["GEOID"].count().rename("counties")
    print(summary.to_string())


if __name__ == "__main__":
    main()
