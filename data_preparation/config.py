from pathlib import Path

# -----------------------------
# Project directories
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
NASS_DIR = DATA_DIR / "nass"
COUNTIES_DIR = DATA_DIR / "counties"
CDL_MASK_DIR = DATA_DIR / "cdl_masks"
YIELD_LABEL_DIR = DATA_DIR / "yield_labels"
QC_DIR = DATA_DIR / "qc"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for d in [DATA_DIR, NASS_DIR, COUNTIES_DIR, CDL_MASK_DIR, YIELD_LABEL_DIR, QC_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Study states and years
# -----------------------------
STATE_ALPHA = [
    "IL", "IA", "IN", "MN", "NE", "MO", "OH", "SD", "ND",
    "KS", "WI", "MI", 
    "MD",
    "DE",
    "VA", "NC", "PA",
]

STATE_FIPS = {
    "IL": "17", "IA": "19", "IN": "18", "MN": "27", "NE": "31",
    "MO": "29", "OH": "39", "SD": "46", "ND": "38", "KS": "20",
    "WI": "55", "MI": "26", "MD": "24",
    "DE": "10", "VA": "51",
    "NC": "37", "PA": "42",
}

START_YEAR = 2014
END_YEAR = 2024

# -----------------------------
# Crop settings  (switch CROP_NAME to change target crop)
# -----------------------------
CROP_NAME = "SOYBEANS"

CDL_CODES = {
    "SOYBEANS": 5,
    "CORN": 1,
    "WHEAT": 22,
}
CDL_CROP_CODE = CDL_CODES[CROP_NAME]

# -----------------------------
# Yield unit conversion
# -----------------------------
BU_AC_TO_KG_HA = {
    "SOYBEANS": 67.251,
    "CORN": 62.77,
    "WHEAT": 67.25,
}
CONVERSION_FACTOR = BU_AC_TO_KG_HA[CROP_NAME]

VALUE_COL = "yield_kg_ha"

NODATA = -9999.0

# -----------------------------
# Leakage protection
# -----------------------------
# Prince George's County MD excluded — used as BARC external test site
EXCLUDE_GEOIDS = {"24033"}

# -----------------------------
# Raster settings
# -----------------------------
TARGET_CRS = "EPSG:5070"
TARGET_SCALE_M = 30
COARSE_RES_M = 10_000     # intermediate resolution for bilinear label

# -----------------------------
# Derived output paths
# -----------------------------
NASS_CSV = NASS_DIR / f"nass_{CROP_NAME.lower()}_county_yield_{START_YEAR}_{END_YEAR}.csv"
COUNTIES_GPKG = COUNTIES_DIR / "selected_states_counties_2023.gpkg"
