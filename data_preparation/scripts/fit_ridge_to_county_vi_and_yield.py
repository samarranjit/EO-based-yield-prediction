# ============================================================
# Fit Ridge VI weights from GEE county VI summary CSVs + NASS
#
# Inputs:
#   1) GEE county summary CSVs:
#      soybeans_*_county_vi_summary_minimal.csv
#
#   2) NASS soybean county yield CSV:
#      nass_soybeans_county_yield_2014_2024.csv
#
# Output:
#   - Ridge coefficients
#   - Normalized GEE-ready VI weights
#   - Clean merged training table
# ============================================================

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1) CONFIG
# ============================================================

# Folder where your GEE county summary CSVs are downloaded
VI_SUMMARY_DIR = Path(
    "/Users/samarranjit/Downloads/GEE_SOYBEAN_VI_EXPORTS"
)

# Pattern for your GEE county summary files
VI_FILE_PATTERN = "*_county_vi_summary_minimal*.csv"

# NASS yield CSV path
NASS_CSV_PATH = Path(
    "/Users/samarranjit/Downloads/nass_soybeans_county_yield_2014_2024.csv"
)

# Output folder
OUTPUT_DIR = Path("./ridge_vi_weight_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Features from GEE county VI summary
FEATURES = [
    "NDVI_mean",
    "EVI_mean",
    "GCVI_mean",
    "NDWI_mean",
]

TARGET = "yield_bu_ac"

# Quality filters
MIN_VALID_PIXEL_FRACTION = 0.50
MIN_CROP_PIXEL_COUNT = 10

# Ridge alpha.
# This corresponds to lambda/regularization strength.
# alpha=1.0 is a good first default because variables are standardized.
RIDGE_ALPHA = 1.0

# If True, negative VI coefficients are clipped to zero before normalizing
# for GEE redistribution. I recommend keeping this False first.
FORCE_NONNEGATIVE_WEIGHTS = False


# ============================================================
# 2) LOAD GEE COUNTY VI SUMMARY FILES
# ============================================================

vi_files = sorted(glob.glob(str(VI_SUMMARY_DIR / VI_FILE_PATTERN)))

if len(vi_files) == 0:
    raise FileNotFoundError(
        f"No VI summary files found in:\n{VI_SUMMARY_DIR}\n"
        f"with pattern:\n{VI_FILE_PATTERN}"
    )

print(f"Found {len(vi_files)} VI summary CSV files.")

vi_dfs = []

for path in vi_files:
    df = pd.read_csv(path)

    # Keep only useful columns if present
    keep_cols = [
        "state",
        "state_fips",
        "county_fips",
        "county_name",
        "year",
        "crop",
        "season_start",
        "season_end",
        "NDVI_mean",
        "EVI_mean",
        "GCVI_mean",
        "NDWI_mean",
        "min_index_obsCount",
        "crop_pixel_count",
        "valid_pixel_count",
        "valid_pixel_fraction",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df["source_file"] = Path(path).name
    vi_dfs.append(df)

vi = pd.concat(vi_dfs, ignore_index=True)

print(f"Raw VI rows: {len(vi):,}")


# ============================================================
# 3) CLEAN VI TABLE
# ============================================================

# Standardize column types
vi["county_fips"] = vi["county_fips"].astype(str).str.replace(".0", "", regex=False)
vi["county_fips"] = vi["county_fips"].str.zfill(5)
vi["year"] = vi["year"].astype(int)

for col in FEATURES:
    vi[col] = pd.to_numeric(vi[col], errors="coerce")

if "valid_pixel_fraction" in vi.columns:
    vi["valid_pixel_fraction"] = pd.to_numeric(
        vi["valid_pixel_fraction"], errors="coerce"
    )

if "crop_pixel_count" in vi.columns:
    vi["crop_pixel_count"] = pd.to_numeric(
        vi["crop_pixel_count"], errors="coerce"
    )

# Drop rows with missing VI values
vi = vi.dropna(subset=FEATURES)

# Quality filters
if "valid_pixel_fraction" in vi.columns:
    vi = vi[vi["valid_pixel_fraction"] >= MIN_VALID_PIXEL_FRACTION]

if "crop_pixel_count" in vi.columns:
    vi = vi[vi["crop_pixel_count"] >= MIN_CROP_PIXEL_COUNT]

print(f"VI rows after cleaning/QC: {len(vi):,}")


# ============================================================
# 4) LOAD AND CLEAN NASS YIELD TABLE
# ============================================================

nass = pd.read_csv(NASS_CSV_PATH)

# Expected NASS columns:
#   state, GEOID, year, county_name, state_fips, yield_bu_ac, yield_kg_ha

if "GEOID" not in nass.columns:
    raise ValueError("NASS CSV must contain a GEOID column.")

if TARGET not in nass.columns:
    raise ValueError(f"NASS CSV must contain {TARGET} column.")

nass["GEOID"] = nass["GEOID"].astype(str).str.replace(".0", "", regex=False)
nass["GEOID"] = nass["GEOID"].str.zfill(5)
nass["year"] = nass["year"].astype(int)
nass[TARGET] = pd.to_numeric(nass[TARGET], errors="coerce")

nass = nass.dropna(subset=["GEOID", "year", TARGET])

# Keep only needed NASS columns
nass_keep = [
    "GEOID",
    "year",
    TARGET,
]

if "state" in nass.columns:
    nass_keep.append("state")

if "county_name" in nass.columns:
    nass_keep.append("county_name")

nass = nass[nass_keep].drop_duplicates(subset=["GEOID", "year"])

print(f"NASS rows: {len(nass):,}")


# ============================================================
# 5) MERGE VI SUMMARY WITH NASS YIELD
# ============================================================

df = vi.merge(
    nass,
    left_on=["county_fips", "year"],
    right_on=["GEOID", "year"],
    how="inner",
    suffixes=("", "_nass"),
)

df = df.dropna(subset=FEATURES + [TARGET])

print(f"Merged training rows: {len(df):,}")

if len(df) == 0:
    raise RuntimeError(
        "No rows after merging VI summaries with NASS yield. "
        "Check county_fips/GEOID and year formatting."
    )

print("\nTraining years:")
print(sorted(df["year"].unique()))

print("\nRows by year:")
print(df.groupby("year").size())


# ============================================================
# 6) FIT RIDGE REGRESSION
# ============================================================

X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)),
    ]
)

model.fit(X, y)

scaler = model.named_steps["scaler"]
ridge = model.named_steps["ridge"]

coef = ridge.coef_
intercept = ridge.intercept_

coef_series = pd.Series(coef, index=FEATURES)

print("\n============================================================")
print("RIDGE MODEL FITTED")
print("============================================================")
print(f"Ridge alpha: {RIDGE_ALPHA}")
print(f"Intercept: {intercept:.6f}")

print("\nRaw Ridge coefficients on standardized VI variables:")
print(coef_series)


# ============================================================
# 7) MODEL DIAGNOSTICS
# ============================================================

pred = model.predict(X)

train_r2 = r2_score(y, pred)
train_mae = mean_absolute_error(y, pred)
train_corr = np.corrcoef(y, pred)[0, 1]

print("\nTraining diagnostics:")
print(f"R²   : {train_r2:.4f}")
print(f"MAE  : {train_mae:.4f} bu/ac")
print(f"Corr : {train_corr:.4f}")

# Leave-one-year-out CV
if df["year"].nunique() > 1:
    logo = LeaveOneGroupOut()
    groups = df["year"].to_numpy()

    cv_r2 = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=logo,
        scoring="r2",
    )

    cv_mae = -cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=logo,
        scoring="neg_mean_absolute_error",
    )

    print("\nLeave-one-year-out diagnostics:")
    print(f"Median R² : {np.nanmedian(cv_r2):.4f}")
    print(f"Mean R²   : {np.nanmean(cv_r2):.4f}")
    print(f"Median MAE: {np.nanmedian(cv_mae):.4f} bu/ac")
    print(f"Mean MAE  : {np.nanmean(cv_mae):.4f} bu/ac")


# ============================================================
# 8) CONVERT COEFFICIENTS TO GEE REDISTRIBUTION WEIGHTS
# ============================================================

gee_weights = coef.copy()

if FORCE_NONNEGATIVE_WEIGHTS:
    gee_weights = np.maximum(gee_weights, 0)

denom = np.sum(np.abs(gee_weights))

if denom == 0:
    raise RuntimeError(
        "All VI coefficients became zero. Cannot normalize GEE weights."
    )

gee_weights_norm = gee_weights / denom

gee_weight_series = pd.Series(gee_weights_norm, index=FEATURES)

print("\n============================================================")
print("GEE-READY NORMALIZED VI WEIGHTS")
print("============================================================")
print(gee_weight_series)

print("\nPaste these into GEE:")
print(f"var HARDCODED_B_NDVI = {gee_weight_series['NDVI_mean']:.8f};")
print(f"var HARDCODED_B_EVI  = {gee_weight_series['EVI_mean']:.8f};")
print(f"var HARDCODED_B_GCVI = {gee_weight_series['GCVI_mean']:.8f};")
print(f"var HARDCODED_B_NDWI = {gee_weight_series['NDWI_mean']:.8f};")


# ============================================================
# 9) SAVE OUTPUTS
# ============================================================

coef_out = pd.DataFrame({
    "feature": FEATURES,
    "ridge_coef_standardized": coef,
    "gee_weight_normalized": gee_weights_norm,
    "x_mean": scaler.mean_,
    "x_std": scaler.scale_,
})

coef_out["ridge_alpha"] = RIDGE_ALPHA
coef_out["force_nonnegative_weights"] = FORCE_NONNEGATIVE_WEIGHTS

coef_path = OUTPUT_DIR / "ridge_vi_coefficients_for_gee.csv"
train_path = OUTPUT_DIR / "merged_county_vi_nass_training_table.csv"
pred_path = OUTPUT_DIR / "ridge_training_predictions.csv"

coef_out.to_csv(coef_path, index=False)
df.to_csv(train_path, index=False)

pred_df = df.copy()
pred_df["predicted_yield_bu_ac"] = pred
pred_df["residual_bu_ac"] = pred_df[TARGET] - pred_df["predicted_yield_bu_ac"]
pred_df.to_csv(pred_path, index=False)

print("\nSaved outputs:")
print(f"Coefficient table: {coef_path}")
print(f"Merged training table: {train_path}")
print(f"Training predictions: {pred_path}")