"""A corrupted HLS tile must not crash the DataLoader worker reading it.

Real incident: `hls_soybeans_MD_2015_MAY.tif` has a handful of genuinely
corrupt GDAL blocks (confirmed reproducible, same offset, by
scripts/scan_hls_corruption.py). `data.exclude_sample_ids` filters chips on
the LABEL pixel grid, but `read_chip` re-aligns each imagery read into the
IMAGERY raster's own grid (`_aligned_window`), so a neighboring, non-excluded
chip's aligned window can still hit the same bad block. That killed a
multi-hour real training run twice in a row with an uncaught
`RasterioIOError` propagating out of a `DataLoader` worker.

`read_chip` must instead treat a corrupted-tile read exactly like a missing
month: leave that month invalid (NaN / month_valid=False) and continue, so
the existing missing_month_policy handles it -- not crash, and not silently
drop the chip's other valid months either.
"""

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from farm_us.config import DataConfig  # noqa: E402
from farm_us.data.raster_readers import GeotiffMonthlyReader  # noqa: E402
from farm_us.utils.geospatial import ChipWindow  # noqa: E402

STATE, YEAR, CROP = "MD", 2015, "SOYBEANS"
H = W = 8


def _write(path: Path, arr: np.ndarray, nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = arr.shape[0] if arr.ndim == 3 else 1
    t = from_origin(0, H * 30, 30, 30)
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=count,
        dtype="float32", crs="EPSG:5070", transform=t, nodata=nodata,
    ) as ds:
        ds.write(arr.reshape(count, H, W).astype("float32"))


@pytest.fixture
def reader(tmp_path) -> GeotiffMonthlyReader:
    cfg = DataConfig(
        imagery_root=str(tmp_path / "hls"),
        label_root=str(tmp_path / "labels"),
        cdl_root=str(tmp_path / "cdl"),
        crop=CROP, label_variant="bilinear10km",
        band_order=["BLUE", "GREEN"], n_timesteps=3,  # APR, MAY, JUN -- keep it small
        chip_size=H, stride=H,
    )
    crop = cfg.crop.lower()
    _write(tmp_path / "labels" / str(YEAR) / f"nass_{crop}_yield_{STATE}_{YEAR}_bilinear10km_30m_{crop}_only.tif",
           np.full((H, W), 3000.0, np.float32))
    _write(tmp_path / "cdl" / f"cdl_{crop}_{STATE}_{YEAR}.tif", np.ones((H, W), np.float32), nodata=0)
    for mon in ("APR", "MAY", "JUN"):
        _write(tmp_path / "hls" / str(YEAR) / f"hls_{crop}_{STATE}_{YEAR}_{mon}.tif",
               np.full((2, H, W), 500.0, np.float32))
    return GeotiffMonthlyReader(cfg)


def test_all_months_readable_by_default(reader):
    """Sanity check for the fixture itself before we start breaking it."""
    win = ChipWindow(row_off=0, col_off=0, height=H, width=W)
    chip = reader.read_chip(STATE, YEAR, win)
    assert chip.month_valid.all()
    assert not np.isnan(chip.image).any()


def test_corrupted_month_is_treated_as_missing_not_a_crash(reader, monkeypatch, caplog):
    original_read = rasterio.io.DatasetReader.read

    def flaky_read(self, *args, **kwargs):
        if "MAY" in self.name:
            raise rasterio.errors.RasterioIOError(
                "hls_soybeans_MD_2015_MAY.tif, band 1: IReadBlock failed at "
                "X offset 28, Y offset 8: TIFFReadEncodedTile() failed."
            )
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(rasterio.io.DatasetReader, "read", flaky_read)

    win = ChipWindow(row_off=0, col_off=0, height=H, width=W)
    chip = reader.read_chip(STATE, YEAR, win)  # must not raise

    apr, may, jun = 0, 1, 2
    assert chip.month_valid[may].sum() == 0, "corrupted month must be marked fully invalid"
    assert np.isnan(chip.image[:, may]).all(), "corrupted month must be left as NaN, not zero-filled here"
    assert chip.month_valid[apr].all() and chip.month_valid[jun].all(), \
        "the OTHER months must still be read normally -- one bad tile must not sink the whole chip"
    assert not np.isnan(chip.image[:, apr]).any()
    assert not np.isnan(chip.image[:, jun]).any()
    assert any("Corrupted read" in r.message for r in caplog.records)


def test_real_contract_violations_still_raise(reader, monkeypatch):
    """A wrong band count is a genuine data-contract bug, not a flaky tile --
    it must still fail loudly and NOT be swallowed by the corruption guard."""
    from farm_us.utils.logging import DataContractError

    orig_count = rasterio.io.DatasetReader.count
    monkeypatch.setattr(
        rasterio.io.DatasetReader, "count",
        property(lambda self: 1 if "MAY" in self.name else orig_count.__get__(self)),
    )
    win = ChipWindow(row_off=0, col_off=0, height=H, width=W)
    with pytest.raises(DataContractError):
        reader.read_chip(STATE, YEAR, win)
