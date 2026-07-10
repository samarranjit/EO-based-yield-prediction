# Crop Yield Label Data Preparation Pipeline

Prepares **model-ready 30 m yield-label rasters** from USDA NASS county yield data
and USDA NASS CDL crop masks.  Currently configured for **soybeans**; switch
`CROP_NAME` in `config.py` to reuse the same scripts for corn, wheat, or any
CDL-supported crop.

---

## What this pipeline is (and is not)

These rasters are **not** 30 m measured ground truth.  NASS yield is county-level
statistics.  We convert it into raster form for **weak-supervision** labelling of
the Prithvi EO model.  Two label types are produced:

| Variant | Description |
|---|---|
| `nearest` | Every crop pixel in a county receives the county's NASS yield (hard county boundaries). |
| `bilinear10km` | County yields are first rasterised at ~10 km, then bilinear-resampled to 30 m — a FARM-style smooth surface that blends across county edges. |

---

## Changing the target crop

Open `config.py` and change the single line:

```python
CROP_NAME = "SOYBEANS"   # → "CORN" or "WHEAT"
```

Everything else — CDL code, conversion factor, output filenames — updates
automatically.

---

## Setup

```bash
cd data_preparation
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Script execution

### 1 · Download NASS county yield

```bash
export NASS_API_KEY="your_key_here"   # get one at https://quickstats.nass.usda.gov/api
python scripts/01_download_nass_yield.py
```

Output: `data/nass/nass_soybeans_county_yield_2014_2024.csv`

### 2 · Download county boundaries

```bash
python scripts/02_download_counties.py
```

Downloads TIGER/Line 2023 county shapefile from the Census Bureau, filters to the
17 target states, reprojects to EPSG:5070, and saves as a GeoPackage.

Output: `data/counties/selected_states_counties_2023.gpkg`

### 3 · Export CDL masks from Google Earth Engine

```bash
earthengine authenticate          # one-time auth
python scripts/03_export_cdl_masks_gee.py
```

Submits one GEE export task per (state × year).  Tasks run asynchronously;
monitor progress at <https://code.earthengine.google.com/tasks>.

When all tasks finish, **download the `.tif` files from your Google Drive folder
`cdl_masks/`** into `data/cdl_masks/` with this naming convention:

```
cdl_soybeans_<STATE>_<YEAR>.tif     e.g.  cdl_soybeans_MD_2020.tif
```

You can use `gdown`, the Drive API, or the web UI to download.

### 4 · Create yield-label rasters

```bash
python scripts/04_make_yield_label_rasters.py
```

For every (state, year) that has both a CDL mask and NASS yield data, two
rasters are written:

```
data/yield_labels/<YEAR>/nass_soybeans_yield_<STATE>_<YEAR>_nearest_30m_soybeans_only.tif
data/yield_labels/<YEAR>/nass_soybeans_yield_<STATE>_<YEAR>_bilinear10km_30m_soybeans_only.tif
```

State–year combos with a missing CDL mask or missing NASS data are skipped with
a `SKIP` message — they do **not** cause the script to fail.

### 5 · QC check

```bash
python scripts/05_qc_check_label_rasters.py
```

Validates all label rasters: pixel counts, yield distributions, CRS, resolution,
and missing coverage.  Writes a summary CSV to `data/qc/`.

---

## Leakage protection

**Prince George's County, MD (GEOID 24033) is excluded from all labels.**
This county contains the BARC field site used as an external test set.
Edit `EXCLUDE_GEOIDS` in `config.py` to add other withheld areas.

---

## Target states

```
IL  IA  IN  MN  NE  MO  OH  SD  ND  KS  WI  MI  MD  DE  VA  NC  PA
```

Years: 2014 – 2024

---

## Yield units

Both columns are written to the NASS CSV:

| Column | Unit |
|---|---|
| `yield_bu_ac` | bu / acre (NASS native) |
| `yield_kg_ha` | kg / ha (used for raster labels) |

Conversion factors (set in `config.py`):

| Crop | Factor (bu/ac → kg/ha) |
|---|---|
| Soybeans | 67.251 |
| Corn | 62.77 |
| Wheat | 67.25 |

---

## Folder structure

```
data_preparation/
├── config.py                        ← all settings live here
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 01_download_nass_yield.py
│   ├── 02_download_counties.py
│   ├── 03_export_cdl_masks_gee.py
│   ├── 04_make_yield_label_rasters.py
│   └── 05_qc_check_label_rasters.py
├── data/
│   ├── nass/          ← NASS CSV
│   ├── counties/      ← county GeoPackage
│   ├── cdl_masks/     ← downloaded CDL GeoTIFFs
│   ├── yield_labels/  ← final output rasters (by year)
│   └── qc/            ← QC CSVs
└── outputs/
```

Data directories and large files are gitignored.
