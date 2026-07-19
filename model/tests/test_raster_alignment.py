"""Windowed-read + alignment tests on synthetic GeoTIFFs, plus an opportunistic
check against the real label/CDL rasters if they are present on disk."""

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from farm_us.utils.geospatial import tile_windows  # noqa: E402

REAL_LABEL_DIR = Path(__file__).resolve().parents[2] / "data_preparation/data/yield_labels/bilinear"
REAL_CDL_DIR = Path(__file__).resolve().parents[2] / "data_preparation/data/cdl_masks"


def _write(path, arr, nodata=-9999.0):
    t = from_origin(0, arr.shape[0] * 30, 30, 30)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32", crs="EPSG:5070", transform=t, nodata=nodata,
    ) as ds:
        ds.write(arr.astype("float32"), 1)


def test_windowed_read_matches_full(tmp_path):
    arr = np.arange(400 * 300, dtype=np.float32).reshape(400, 300)
    p = tmp_path / "r.tif"
    _write(p, arr)
    with rasterio.open(p) as ds:
        win = Window(50, 60, 224, 224)
        sub = ds.read(1, window=win)
    assert np.array_equal(sub, arr[60:284, 50:274])


def test_tile_windows_fit_raster(tmp_path):
    arr = np.zeros((500, 460), np.float32)
    p = tmp_path / "r.tif"
    _write(p, arr)
    with rasterio.open(p) as ds:
        for w in tile_windows(ds.height, ds.width, 224, 224):
            assert w.row_off + 224 <= ds.height
            assert w.col_off + 224 <= ds.width


@pytest.mark.skipif(not REAL_LABEL_DIR.exists(), reason="real label rasters not present")
def test_real_label_cdl_alignment():
    labels = list(REAL_LABEL_DIR.glob("2018/*IA*bilinear*.tif"))
    cdls = list(REAL_CDL_DIR.glob("cdl_soybeans_IA_2018.tif"))
    if not labels or not cdls:
        pytest.skip("IA 2018 rasters not present")
    with rasterio.open(labels[0]) as lab, rasterio.open(cdls[0]) as cdl:
        assert lab.crs == cdl.crs == rasterio.crs.CRS.from_epsg(5070)
        assert (lab.width, lab.height) == (cdl.width, cdl.height)
        assert lab.transform == cdl.transform
        assert lab.res == (30.0, 30.0)
