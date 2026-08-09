# ============================================================
# Fit Ridge VI Weights from GEE County VI Summary CSVs + NASS
#
# Purpose:
#   This script learns the relative importance of vegetation indices
#   for explaining county-level NASS soybean yield.
#
# Inputs:
#   1) GEE county summary CSVs:
#        soybeans_*_county_vi_summary_minimal.csv
#
#      Required columns:
#        county_fips, year,
#        NDVI_mean, EVI_mean, GCVI_mean, NDWI_mean,
#        crop_pixel_count, valid_pixel_fraction
#
#   2) NASS soybean county yield CSV:
#        nass_soybeans_county_yield_2014_2024.csv
#
#      Required columns:
#        GEOID, year, yield_bu_ac
#
# Outputs:
#   1) ridge_vi_coefficients_for_gee.csv
#      Contains raw Ridge coefficients and normalized redistribution weights.
#
#   2) merged_county_vi_nass_training_table.csv
#      Clean county-year training table used for Ridge.
#
#   3) ridge_training_predictions.csv
#      County-level predictions and residuals for diagnostics.
#
#   4) gee_weight_snippet.txt
#      Copy-paste-ready weights for GEE/Python redistribution.
#
# Important:
#   The Ridge model is NOT used directly to predict pixel yield.
#   It is used to estimate VI weights.
#   Pixel yield redistribution is done in the second script.
# ============================================================

from pathlib import Path
import glob
import json

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

# Root folder containing GEE county summary CSV files.
# This can contain year subfolders; files are found recursively.
VI_SUMMARY_DIR = Path(
    "/home/cholab/LabMembers/Samar/EO-based-yield-prediction/data_preparation/data/year_state_VI_summary"
)

# Recursively match all CSVs inside VI_SUMMARY_DIR.
VI_FILE_PATTERN = "**/*.csv"

# NASS county yield CSV.
NASS_CSV_PATH = Path(
    "/home/cholab/LabMembers/Samar/EO-based-yield-prediction/data_preparation/data/nass/nass_soybeans_county_yield_2014_2024.csv"
)

# Output folder.
OUTPUT_DIR = Path("data_preparation/outputs/ridge_vi_weight_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Vegetation-index features from county summary table.
FEATURES = [
    "NDVI_mean",
    "EVI_mean",
    "GCVI_mean",
    "NDWI_mean",
]

TARGET = "yield_bu_ac"

# Quality filters.
MIN_VALID_PIXEL_FRACTION = 0.50
MIN_CROP_PIXEL_COUNT = 10

# Ridge regularization strength.
# Because X variables are standardized, alpha=1.0 is a reasonable first default.
RIDGE_ALPHA = 1.0

# If True, negative coefficients are set to zero before creating
# redistribution weights.
#
# Recommendation:
#   Keep False first, because negative NDWI/NDVI effects can happen
#   depending on multicollinearity and moisture/stress behavior.
FORCE_NONNEGATIVE_WEIGHTS = False

# Redistribution parameters saved for the second script.
# These are not used during Ridge fitting, but they document the method.
REDIST_STRENGTH = 0.15
RAW_WEIGHT_MIN = 0.50
RAW_WEIGHT_MAX = 1.50


# ============================================================
# 2) HELPER FUNCTIONS
# ============================================================

def normalize_fips(series: pd.Series) -> pd.Series:
    """
    Convert FIPS/GEOID values to clean 5-character strings.
    Handles values like 10001, 10001.0, and '10001'.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Safely calculate Pearson correlation.
    """
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# ============================================================
# 3) LOAD GEE COUNTY VI SUMMARY FILES
# ============================================================

vi_files = sorted(
    glob.glob(str(VI_SUMMARY_DIR / VI_FILE_PATTERN), recursive=True)
)

if len(vi_files) == 0:
    raise FileNotFoundError(
        f"No VI summary CSV files found in:\n{VI_SUMMARY_DIR}\n"
        f"with pattern:\n{VI_FILE_PATTERN}"
    )

print(f"Found {len(vi_files)} VI summary CSV files.")

vi_dfs = []

for path in vi_files:
    path = Path(path)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Skipping unreadable file: {path}")
        print(f"  Error: {e}")
        continue

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
    df["source_file"] = path.name

    vi_dfs.append(df)

if len(vi_dfs) == 0:
    raise RuntimeError("No readable VI summary CSV files were found.")

vi = pd.concat(vi_dfs, ignore_index=True)

print(f"Raw VI rows: {len(vi):,}")


# ============================================================
# 4) CLEAN VI TABLE
# ============================================================

required_vi_cols = ["county_fips", "year"] + FEATURES

missing_vi_cols = [c for c in required_vi_cols if c not in vi.columns]
if missing_vi_cols:
    raise ValueError(
        "The VI summary table is missing required columns:\n"
        + "\n".join(missing_vi_cols)
    )

vi["county_fips"] = normalize_fips(vi["county_fips"])
vi["year"] = pd.to_numeric(vi["year"], errors="coerce").astype("Int64")

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

# Drop missing core values.
vi = vi.dropna(subset=["county_fips", "year"] + FEATURES)
vi["year"] = vi["year"].astype(int)

# Apply quality filters.
if "valid_pixel_fraction" in vi.columns:
    vi = vi[vi["valid_pixel_fraction"] >= MIN_VALID_PIXEL_FRACTION]

if "crop_pixel_count" in vi.columns:
    vi = vi[vi["crop_pixel_count"] >= MIN_CROP_PIXEL_COUNT]

print(f"VI rows after cleaning/QC: {len(vi):,}")


# ============================================================
# 5) LOAD AND CLEAN NASS YIELD TABLE
# ============================================================

nass = pd.read_csv(NASS_CSV_PATH)

if "GEOID" not in nass.columns:
    raise ValueError("NASS CSV must contain a GEOID column.")

if TARGET not in nass.columns:
    raise ValueError(f"NASS CSV must contain a {TARGET} column.")

nass["GEOID"] = normalize_fips(nass["GEOID"])
nass["year"] = pd.to_numeric(nass["year"], errors="coerce").astype("Int64")
nass[TARGET] = pd.to_numeric(nass[TARGET], errors="coerce")

nass = nass.dropna(subset=["GEOID", "year", TARGET])
nass["year"] = nass["year"].astype(int)

nass_keep = ["GEOID", "year", TARGET]

if "state" in nass.columns:
    nass_keep.append("state")

if "county_name" in nass.columns:
    nass_keep.append("county_name")

nass = nass[nass_keep].drop_duplicates(subset=["GEOID", "year"])

print(f"NASS rows: {len(nass):,}")


# ============================================================
# 6) MERGE VI SUMMARY WITH NASS YIELD
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
# 7) FIT RIDGE REGRESSION
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
intercept = float(ridge.intercept_)

coef_series = pd.Series(coef, index=FEATURES)

print("\n============================================================")
print("RIDGE MODEL FITTED")
print("============================================================")
print(f"Ridge alpha: {RIDGE_ALPHA}")
print(f"Intercept: {intercept:.6f}")

print("\nRaw Ridge coefficients on standardized VI variables:")
print(coef_series)


# ============================================================
# 8) MODEL DIAGNOSTICS
# ============================================================

pred = model.predict(X)

train_r2 = r2_score(y, pred)
train_mae = mean_absolute_error(y, pred)
train_corr = safe_corr(y, pred)

print("\nTraining diagnostics:")
print(f"R²   : {train_r2:.4f}")
print(f"MAE  : {train_mae:.4f} bu/ac")
print(f"Corr : {train_corr:.4f}")

# Leave-one-year-out diagnostics.
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
        error_score=np.nan,
    )

    cv_mae = -cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=logo,
        scoring="neg_mean_absolute_error",
        error_score=np.nan,
    )

    print("\nLeave-one-year-out diagnostics:")
    print(f"Median R² : {np.nanmedian(cv_r2):.4f}")
    print(f"Mean R²   : {np.nanmean(cv_r2):.4f}")
    print(f"Median MAE: {np.nanmedian(cv_mae):.4f} bu/ac")
    print(f"Mean MAE  : {np.nanmean(cv_mae):.4f} bu/ac")
else:
    cv_r2 = np.array([])
    cv_mae = np.array([])


# ============================================================
# 9) CONVERT RIDGE COEFFICIENTS TO REDISTRIBUTION WEIGHTS
# ============================================================

# These normalized weights are what we use for pseudo-yield distribution.
# The intercept is NOT used for pixel distribution.
gee_weights = coef.copy()

if FORCE_NONNEGATIVE_WEIGHTS:
    gee_weights = np.maximum(gee_weights, 0)

denom = np.sum(np.abs(gee_weights))

if denom == 0:
    raise RuntimeError(
        "All VI coefficients became zero. Cannot normalize redistribution weights."
    )

gee_weights_norm = gee_weights / denom
gee_weight_series = pd.Series(gee_weights_norm, index=FEATURES)

print("\n============================================================")
print("NORMALIZED VI REDISTRIBUTION WEIGHTS")
print("============================================================")
print(gee_weight_series)

print("\nPaste these into GEE/Python:")
print(f"var HARDCODED_B_NDVI = {gee_weight_series['NDVI_mean']:.8f};")
print(f"var HARDCODED_B_EVI  = {gee_weight_series['EVI_mean']:.8f};")
print(f"var HARDCODED_B_GCVI = {gee_weight_series['GCVI_mean']:.8f};")
print(f"var HARDCODED_B_NDWI = {gee_weight_series['NDWI_mean']:.8f};")


# ============================================================
# 10) SAVE OUTPUTS
# ============================================================

coef_out = pd.DataFrame({
    "feature": FEATURES,
    "ridge_coef_standardized": coef,
    "gee_weight_normalized": gee_weights_norm,
    "x_mean": scaler.mean_,
    "x_std": scaler.scale_,
})

coef_out["ridge_alpha"] = RIDGE_ALPHA
coef_out["ridge_intercept"] = intercept
coef_out["force_nonnegative_weights"] = FORCE_NONNEGATIVE_WEIGHTS
coef_out["redist_strength"] = REDIST_STRENGTH
coef_out["raw_weight_min"] = RAW_WEIGHT_MIN
coef_out["raw_weight_max"] = RAW_WEIGHT_MAX
coef_out["min_valid_pixel_fraction"] = MIN_VALID_PIXEL_FRACTION
coef_out["min_crop_pixel_count"] = MIN_CROP_PIXEL_COUNT

coef_path = OUTPUT_DIR / "ridge_vi_coefficients_for_gee.csv"
train_path = OUTPUT_DIR / "merged_county_vi_nass_training_table.csv"
pred_path = OUTPUT_DIR / "ridge_training_predictions.csv"
snippet_path = OUTPUT_DIR / "gee_weight_snippet.txt"
summary_json_path = OUTPUT_DIR / "ridge_model_summary.json"

coef_out.to_csv(coef_path, index=False)
df.to_csv(train_path, index=False)

pred_df = df.copy()
pred_df["predicted_yield_bu_ac"] = pred
pred_df["residual_bu_ac"] = pred_df[TARGET] - pred_df["predicted_yield_bu_ac"]
pred_df.to_csv(pred_path, index=False)

with open(snippet_path, "w") as f:
    f.write("// Ridge-derived normalized VI redistribution weights\n")
    f.write("// Use these for pseudo-yield redistribution, not direct yield prediction.\n\n")
    f.write(f"var HARDCODED_B_NDVI = {gee_weight_series['NDVI_mean']:.8f};\n")
    f.write(f"var HARDCODED_B_EVI  = {gee_weight_series['EVI_mean']:.8f};\n")
    f.write(f"var HARDCODED_B_GCVI = {gee_weight_series['GCVI_mean']:.8f};\n")
    f.write(f"var HARDCODED_B_NDWI = {gee_weight_series['NDWI_mean']:.8f};\n\n")
    f.write(f"var REDIST_STRENGTH = {REDIST_STRENGTH};\n")
    f.write(f"var RAW_WEIGHT_MIN  = {RAW_WEIGHT_MIN};\n")
    f.write(f"var RAW_WEIGHT_MAX  = {RAW_WEIGHT_MAX};\n")

summary = {
    "n_training_rows": int(len(df)),
    "years": sorted([int(v) for v in df["year"].unique()]),
    "features": FEATURES,
    "target": TARGET,
    "ridge_alpha": RIDGE_ALPHA,
    "ridge_intercept": intercept,
    "train_r2": float(train_r2),
    "train_mae_bu_ac": float(train_mae),
    "train_corr": float(train_corr) if np.isfinite(train_corr) else None,
    "normalized_weights": gee_weight_series.to_dict(),
    "redist_strength": REDIST_STRENGTH,
    "raw_weight_min": RAW_WEIGHT_MIN,
    "raw_weight_max": RAW_WEIGHT_MAX,
}

with open(summary_json_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved outputs:")
print(f"Coefficient table       : {coef_path}")
print(f"Merged training table   : {train_path}")
print(f"Training predictions    : {pred_path}")
print(f"GEE weight snippet      : {snippet_path}")
print(f"Model summary JSON      : {summary_json_path}")