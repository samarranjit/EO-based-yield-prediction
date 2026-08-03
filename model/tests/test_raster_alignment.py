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


def _write_at(path, arr, origin_x, origin_y, nodata=-9999.0):
    """Write a raster whose origin is explicitly controlled, to build offset grids."""
    t = from_origin(origin_x, origin_y, 30, 30)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32", crs="EPSG:5070", transform=t, nodata=nodata,
    ) as ds:
        ds.write(arr.astype("float32"), 1)


def test_aligned_window_compensates_offset_grid(tmp_path):
    """Rasters offset by a whole pixel must resolve to the SAME ground.

    Mirrors the real Maryland defect: label rasters sit one 30 m pixel east of
    the CDL/HLS rasters, so reading both with the same pixel window pairs a
    label with imagery 30 m away. _aligned_window derives each raster's window
    from shared geographic bounds instead.
    """
    from farm_us.data.raster_readers import _aligned_window

    h, w = 40, 50
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ground_id = (rows * 1000 + cols).astype(np.float32)

    ref = tmp_path / "ref.tif"          # "label": origin one pixel EAST
    other = tmp_path / "other.tif"      # "cdl/hls": origin at 0
    _write_at(ref, ground_id[:, 1:], origin_x=30, origin_y=h * 30)
    _write_at(other, ground_id, origin_x=0, origin_y=h * 30)

    win = Window(3, 5, 10, 10)
    with rasterio.open(ref) as rds, rasterio.open(other) as ods:
        ref_patch = rds.read(1, window=win)
        bounds = rasterio.windows.bounds(win, rds.transform)

        naive = ods.read(1, window=win)                      # the old, buggy behaviour
        aligned = ods.read(1, window=_aligned_window(ods, bounds))

    assert np.array_equal(ref_patch, aligned), "aligned read must land on the same ground"
    assert not np.array_equal(ref_patch, naive), "same-window read should be skewed (guards the test)"


def test_aligned_window_rejects_fractional_offset(tmp_path):
    """A half-pixel offset cannot be fixed by integer windows -- fail loudly."""
    from farm_us.data.raster_readers import _aligned_window
    from farm_us.utils.logging import DataContractError

    arr = np.zeros((20, 20), np.float32)
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_at(a, arr, origin_x=0, origin_y=600)
    _write_at(b, arr, origin_x=15, origin_y=600)   # half of a 30 m pixel

    with rasterio.open(a) as ads, rasterio.open(b) as bds:
        bounds = rasterio.windows.bounds(Window(0, 0, 10, 10), ads.transform)
        with pytest.raises(DataContractError, match="fractional"):
            _aligned_window(bds, bounds)


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
