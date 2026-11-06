"""Soil/weather data-path correctness: statistics from raw values, coverage,
spatial nearest-neighbor weather, statistics preservation under transfer, no-HLS stats
pass, no cached file handles across forks, and raster inference parity.

Each test pins one specific failure mode found in review (see the docstrings).
Uses tiny GeoTIFFs written to tmp_path -- no real data or network needed.
"""

import os
import pickle

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from farm_us.config import DataConfig, FarmConfig
from farm_us.data.exogenous_readers import (
    SoilReader,
    WeatherSampler,
    build_model_inputs,
    check_exogenous_files,
    normalize_generic,
    require_exogenous_stats,
)
from farm_us.data.normalization import ExogenousStatsAccumulator, NormStats, StreamingBandStats

H = 28
T = 4
CRS = "EPSG:5070"
CHIP_TRANSFORM = from_origin(500_000, 2_000_000, 30, 30)


def _write(path, arr, transform, crs=CRS, nodata=-9999.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype="float32", crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(arr)


def _soil_weather_cfg(tmp_path, soil_vars=("clay",), weather_vars=("ppt",), years=(2020,)):
    """Soil mosaic on the chip grid (3x3 nodata block) + a 2x2-cell 'PRISM' grid
    whose left column of cells is 10 and right column 20."""
    cfg = DataConfig()
    cfg.n_timesteps = T
    cfg.soil_root = str(tmp_path / "soil")
    cfg.weather_root = str(tmp_path / "prism")
    cfg.soil_vars = list(soil_vars)
    cfg.weather_vars = list(weather_vars)
    for v in soil_vars:
        a = np.arange(H * H, dtype=np.float32).reshape(H, H) / 10.0
        a[5:8, 5:8] = -9999.0
        _write(tmp_path / "soil" / f"{v}_0-30_weighted_average.tif", a, CHIP_TRANSFORM)
    coarse = from_origin(500_000, 2_000_000, 420, 420)   # 14 px of 30 m per cell
    for v in weather_vars:
        for y in years:
            for m in (4, 5, 6, 7):
                _write(tmp_path / "prism" / v / f"prism_{v}_{y}{m:02d}.tif",
                       np.array([[10.0, 20.0], [10.0, 20.0]]), coarse)
    return cfg


# --------------------------------------------------------------------------- #
# Issue: missing values were zero-filled BEFORE statistics were computed
# --------------------------------------------------------------------------- #

def test_missing_values_are_excluded_not_counted_as_zero():
    raw = np.array([[[[10.0, np.nan, 20.0]]]], dtype=np.float32)

    # The old path: statistics computed from the NaN-filled model input.
    old = StreamingBandStats(1)
    old.update([normalize_generic(raw, [0.0], [1.0])[0]])
    assert old.finalize()[0][0] == pytest.approx(10.0)          # the bug

    acc = ExogenousStatsAccumulator(n_soil=0, n_weather=1)
    acc.update(weather_raw=raw)
    res = acc.result()["weather"]
    assert res["mean"][0] == pytest.approx(15.0)                 # the fix
    assert res["coverage"][0] == pytest.approx(2 / 3)


def test_compute_fold_stats_uses_raw_soil_with_missing_values():
    """End to end through compute_fold_stats: inject NaN into synthetic soil and
    check the statistics equal the nanmean over valid pixels, with coverage < 1."""
    from farm_us.data.dataset import FarmDataModule
    from farm_us.training.run import compute_fold_stats

    cfg = FarmConfig()
    cfg.data.soil_vars, cfg.data.weather_vars = ["a", "b"], ["w"]
    cfg.data.min_exogenous_coverage = 0.5
    dm = FarmDataModule(cfg, synthetic=True, n_synth=4)
    dm.setup()
    for item in dm.train_ds._items:
        item["soil"][0, :60, :] = np.nan              # ~27% of rows missing in channel 0

    stats = compute_fold_stats(cfg, dm)

    vals = []
    for i in range(len(dm.train_ds)):
        raw = dm.train_ds.exogenous_raw(i)
        v = raw["soil_raw"][0][raw["exo_valid"]]
        vals.append(v[np.isfinite(v)])
    assert stats.soil_mean[0] == pytest.approx(float(np.concatenate(vals).mean()), rel=1e-5)
    assert stats.soil_coverage[0] < 1.0 and stats.soil_coverage[1] == pytest.approx(1.0)
    assert stats.soil_vars == ["a", "b"] and stats.weather_vars == ["w"]
    assert dm.train_ds.return_raw_exogenous is False              # reset after the pass


def test_real_dataset_statistics_come_from_raw_values(tmp_path):
    """The same check through the PRODUCTION path: FarmChipDataset +
    GeotiffMonthlyReader on tiny GeoTIFFs, where s["soil"] really is normalized
    and NaN-filled. Statistics taken from it would count the NaN block as zeros;
    they must instead equal the nanmean of the raw values."""
    from farm_us.data.dataset import FarmChipDataset, SampleRecord
    from farm_us.data.raster_readers import GeotiffMonthlyReader
    from farm_us.training.run import compute_fold_stats

    grid = from_origin(500_000, 2_000_000, 30, 30)
    cfg = FarmConfig()
    d = cfg.data
    d.label_root, d.cdl_root, d.imagery_root = (str(tmp_path / x) for x in ("lab", "cdl", "hls"))
    d.soil_root, d.weather_root = str(tmp_path / "soil"), str(tmp_path / "prism")
    d.label_variant, d.n_timesteps, d.chip_size = "pseudo", T, H
    d.soil_vars, d.weather_vars = ["clay"], ["ppt"]
    d.min_exogenous_coverage = 0.5
    _write(tmp_path / "lab/2020/nass_soybeans_yield_IA_2020_pseudo_30m_soybeans_only.tif",
           np.full((56, 56), 40.0), grid)
    _write(tmp_path / "cdl/cdl_soybeans_IA_2020.tif", np.ones((56, 56)), grid)
    for mon in ("APR", "MAY", "JUN", "JUL"):
        _write(tmp_path / f"hls/2020/hls_soybeans_IA_2020_{mon}.tif", np.full((6, 56, 56), 1000.0), grid)
    soil = 20.0 + np.random.default_rng(0).random((56, 56)).astype(np.float32) * 10
    soil[10:30, 10:30] = -9999.0                              # nodata INSIDE crop pixels
    _write(tmp_path / "soil/clay_0-30_weighted_average.tif", soil, grid)
    for m in (4, 5, 6, 7):
        _write(tmp_path / f"prism/ppt/prism_ppt_2020{m:02d}.tif", np.full((3, 3), 50.0 + m),
               from_origin(500_000, 2_000_000, 840, 840))

    ident = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "identity", [],
                      soil_mean=[0.0], soil_std=[1.0], soil_vars=["clay"],
                      weather_mean=[0.0], weather_std=[1.0], weather_vars=["ppt"],
                      weather_layout="spatial_nearest_crop_label_v1")
    recs = [SampleRecord(f"c{i}", "IA", 2020, r, c, 42.0, -93.0)
            for i, (r, c) in enumerate([(0, 0), (0, 28), (28, 0), (28, 28)])]
    ds = FarmChipDataset(recs, cfg, GeotiffMonthlyReader(d), ident, augment=False)
    stats = compute_fold_stats(cfg, _DM(ds))

    raw = [ds.exogenous_raw(i) for i in range(len(ds))]
    vals = np.concatenate([r["soil_raw"][0][r["exo_valid"]] for r in raw])
    true_mean = float(vals[np.isfinite(vals)].mean())
    zero_filled_mean = float(np.nan_to_num(vals, nan=0.0).mean())
    assert abs(true_mean - zero_filled_mean) > 2.0        # the NaN block matters -> test not vacuous
    assert stats.soil_mean[0] == pytest.approx(true_mean, rel=1e-5)
    assert stats.soil_coverage[0] == pytest.approx(np.isfinite(vals).mean())
    assert stats.weather_mean[0] == pytest.approx(np.mean([54, 55, 56, 57]))


def test_coverage_below_threshold_raises():
    from farm_us.training.run import attach_exogenous_stats

    cfg = FarmConfig()
    cfg.data.weather_vars = ["ppt"]
    stats = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", [])
    res = {"weather": {"mean": [1.0], "std": [1.0], "coverage": [0.67]}}
    with pytest.raises(ValueError, match="coverage"):
        attach_exogenous_stats(cfg, stats, res)


# --------------------------------------------------------------------------- #
# Issue: silent identity fallback fed un-normalized soil to the model
# --------------------------------------------------------------------------- #

def test_build_model_inputs_refuses_missing_stats():
    cfg = DataConfig(); cfg.soil_vars = ["clay"]
    stats = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", [])
    with pytest.raises(ValueError, match="soil"):
        build_model_inputs(cfg, stats, {"soil": np.ones((1, 4, 4), np.float32)})


def test_require_stats_checks_names_and_order():
    cfg = DataConfig(); cfg.soil_vars = ["clay", "bd"]
    stats = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", [],
                      soil_mean=[1.0, 2.0], soil_std=[1.0, 1.0], soil_vars=["bd", "clay"])
    with pytest.raises(ValueError, match="order"):
        require_exogenous_stats(cfg, stats)


def test_apply_norm_stats_refuses_stats_without_configured_groups():
    from farm_us.data.dataset import FarmDataModule

    cfg = FarmConfig(); cfg.data.soil_vars = ["clay"]
    dm = FarmDataModule(cfg, synthetic=True, n_synth=2)
    dm.setup()
    with pytest.raises(ValueError, match="soil"):
        dm.apply_norm_stats(NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", []))


# --------------------------------------------------------------------------- #
# Issue: inherit_init_stats always recomputed (and overwrote) soil/weather stats
# --------------------------------------------------------------------------- #

class _NoReadDS:
    """A train split that fails loudly if anything reads from it."""

    def __len__(self):
        return 3

    def __getitem__(self, i):
        raise AssertionError("full __getitem__ called -- HLS would be read")

    def exogenous_raw(self, i):
        raise AssertionError("statistics were recomputed instead of preserved")


class _ExoOnlyDS(_NoReadDS):
    """exogenous_raw works; full __getitem__ (the HLS path) must never be used."""

    def exogenous_raw(self, i):
        return {"soil_raw": None, "weather_raw": np.full((1, T, H, H), 5.0, np.float32),
                "exo_valid": np.ones((H, H), bool)}


class _DM:
    def __init__(self, ds):
        self.train_ds = ds


def _cfg_with(soil=(), weather=()):
    cfg = FarmConfig()
    cfg.data.soil_vars, cfg.data.weather_vars = list(soil), list(weather)
    return cfg


def test_matching_inherited_stats_are_preserved_not_recomputed():
    from farm_us.training.run import ensure_exogenous_stats

    stats = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", [],
                      soil_mean=[3.0], soil_std=[2.0], soil_vars=["clay"],
                      weather_mean=[7.0], weather_std=[4.0], weather_vars=["ppt"],
                      weather_layout="spatial_nearest_crop_label_v1")
    out = ensure_exogenous_stats(_cfg_with(["clay"], ["ppt"]), _DM(_NoReadDS()), stats)
    assert out.soil_mean == [3.0] and out.weather_mean == [7.0]


def test_mismatched_or_unverifiable_inherited_stats_raise():
    from farm_us.training.run import ensure_exogenous_stats

    wrong = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", [],
                      soil_mean=[3.0], soil_std=[2.0], soil_vars=["sand"])
    with pytest.raises(ValueError, match="Refusing"):
        ensure_exogenous_stats(_cfg_with(["clay"]), _DM(_NoReadDS()), wrong)
    unknown = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", [],
                        soil_mean=[3.0], soil_std=[2.0])            # names not recorded
    with pytest.raises(ValueError, match="Refusing"):
        ensure_exogenous_stats(_cfg_with(["clay"]), _DM(_NoReadDS()), unknown)


def test_only_missing_groups_are_computed_and_without_hls():
    """Soil present and matching -> kept; weather absent -> computed via
    exogenous_raw only (the fake split raises if the HLS __getitem__ is used)."""
    from farm_us.training.run import ensure_exogenous_stats

    stats = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", [],
                      soil_mean=[3.0], soil_std=[2.0], soil_vars=["clay"])
    out = ensure_exogenous_stats(_cfg_with(["clay"], ["ppt"]), _DM(_ExoOnlyDS()), stats)
    assert out.soil_mean == [3.0]
    assert out.weather_mean == [pytest.approx(5.0)] and out.weather_vars == ["ppt"]


# --------------------------------------------------------------------------- #
# Issue: weather sampled at the centroid; resolution reasoning was wrong
# --------------------------------------------------------------------------- #

def test_weather_retains_source_cells_months_and_years(tmp_path):
    cfg = _soil_weather_cfg(tmp_path, weather_vars=("ppt", "tmean"), years=(2020, 2021))
    coarse = from_origin(500_000, 2_000_000, 420, 420)
    # Distinct values encode variable, year, month, and spatial position.
    for vi, v in enumerate(cfg.weather_vars):
        for y in (2020, 2021):
            for m in (4, 5, 6, 7):
                base = vi * 100 + (y - 2020) * 10 + m
                _write(tmp_path / "prism" / v / f"prism_{v}_{y}{m:02d}.tif",
                       np.array([[base, base + 1], [base + 2, -9999.0]]), coarse)
    ws = WeatherSampler(cfg)
    for y in (2020, 2021):
        maps = ws.sample_chip(y, CHIP_TRANSFORM, CRS, H, H, T)
        assert maps.shape == (2, T, H, H)
        for vi in range(2):
            for ti, m in enumerate((4, 5, 6, 7)):
                base = vi * 100 + (y - 2020) * 10 + m
                np.testing.assert_array_equal(maps[vi, ti, :14, :14], base)
                np.testing.assert_array_equal(maps[vi, ti, :14, 14:], base + 1)
                np.testing.assert_array_equal(maps[vi, ti, 14:, :14], base + 2)
                assert np.isnan(maps[vi, ti, 14:, 14:]).all()


def test_missing_weather_month_is_nan_and_refused_up_front(tmp_path):
    cfg = _soil_weather_cfg(tmp_path)
    os.remove(tmp_path / "prism" / "ppt" / "prism_ppt_202006.tif")
    w = WeatherSampler(cfg).sample_chip(2020, CHIP_TRANSFORM, CRS, H, H, T)
    assert np.isnan(w[0, 2]).all() and np.isfinite(w[0, [0, 1, 3]]).all()
    with pytest.raises(FileNotFoundError, match="prism_ppt_202006"):
        check_exogenous_files(cfg, [2020])


def test_soil_reprojection_preserves_values_and_nodata(tmp_path):
    cfg = _soil_weather_cfg(tmp_path)
    soil = SoilReader(cfg).read_chip(CHIP_TRANSFORM, CRS, H, H)[0]
    expected = np.arange(H * H, dtype=np.float32).reshape(H, H) / 10.0
    assert np.isnan(soil[6, 6])                                    # inside the nodata block
    far = np.ones((H, H), bool); far[3:10, 3:10] = False           # away from the block edge
    np.testing.assert_allclose(soil[far], expected[far], rtol=1e-5)


# --------------------------------------------------------------------------- #
# Issue: file handles opened in the main process were inherited by workers
# --------------------------------------------------------------------------- #

class _ReaderDataset(torch.utils.data.Dataset):
    def __init__(self, soil, weather):
        self.soil, self.weather = soil, weather

    def __len__(self):
        return 4

    def __getitem__(self, i):
        s = self.soil.read_chip(CHIP_TRANSFORM, CRS, H, H)
        w = self.weather.sample_chip(2020, CHIP_TRANSFORM, CRS, H, H, T)
        return torch.from_numpy(np.nan_to_num(s)), torch.from_numpy(w), os.getpid()


def test_readers_hold_no_handles_and_work_across_worker_processes(tmp_path):
    cfg = _soil_weather_cfg(tmp_path)
    soil, weather = SoilReader(cfg), WeatherSampler(cfg)
    ds = _ReaderDataset(soil, weather)
    s0, w0, _ = ds[0]                      # a main-process read, like the stats pass

    def open_datasets(obj):
        """Dataset handles held by ``obj``, including inside dict/list/tuple
        attributes -- the original bug cached them in a ``_handles`` dict."""
        found, stack = [], list(vars(obj).values())
        while stack:
            v = stack.pop()
            if isinstance(v, rasterio.io.DatasetReaderBase):
                found.append(v)
            elif isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, list | tuple | set):
                stack.extend(v)
        return found

    assert open_datasets(soil) == [] and open_datasets(weather) == []
    pickle.loads(pickle.dumps(soil)); pickle.loads(pickle.dumps(weather))   # spawn-compatible

    pids = set()
    for s, w, pid in torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2):
        assert torch.equal(s[0], s0) and torch.allclose(w[0], w0)
        pids.add(int(pid))
    assert os.getpid() not in pids and len(pids) >= 1


# --------------------------------------------------------------------------- #
# read_label_and_crop must reproduce read_chip's grid exactly
# --------------------------------------------------------------------------- #

def test_read_label_and_crop_matches_read_chip_grid(tmp_path):
    from farm_us.data.raster_readers import GeotiffMonthlyReader
    from farm_us.utils.geospatial import ChipWindow

    cfg = DataConfig()
    cfg.label_root, cfg.cdl_root, cfg.imagery_root = (str(tmp_path / d) for d in ("lab", "cdl", "hls"))
    cfg.label_variant, cfg.n_timesteps = "pseudo", 1
    lab = np.random.default_rng(0).random((60, 60)).astype(np.float32) * 50
    lab[:5] = -9999.0
    _write(tmp_path / "lab/2020/nass_soybeans_yield_IA_2020_pseudo_30m_soybeans_only.tif", lab,
           from_origin(500_000, 2_000_000, 30, 30))
    # CDL on a grid shifted by whole pixels and larger -- exercises _aligned_window
    cdl = (np.random.default_rng(1).random((70, 70)) > 0.5).astype(np.float32)
    _write(tmp_path / "cdl/cdl_soybeans_IA_2020.tif", cdl, from_origin(499_910, 2_000_060, 30, 30))
    _write(tmp_path / "hls/2020/hls_soybeans_IA_2020_APR.tif",
           np.full((6, 60, 60), 1000.0, np.float32), from_origin(500_000, 2_000_000, 30, 30))

    reader = GeotiffMonthlyReader(cfg)
    win = ChipWindow(10, 20, H, H)
    chip = reader.read_chip("IA", 2020, win)
    grid = reader.read_label_and_crop("IA", 2020, win)
    assert grid.transform == chip.transform and grid.crs == chip.crs and grid.bounds == chip.bounds
    assert np.array_equal(grid.crop_mask, chip.crop_mask)
    np.testing.assert_array_equal(grid.label, chip.label)


# --------------------------------------------------------------------------- #
# Raster inference and interpretability with the exogenous branch enabled
# --------------------------------------------------------------------------- #

def _exo_setup(tmp_path):
    from farm_us.data.raster_readers import ChipArrays
    from farm_us.models.farm_model import FarmModel

    cfg = FarmConfig()
    d = _soil_weather_cfg(tmp_path)
    cfg.data = d
    cfg.data.chip_size, cfg.data.stride = H, H
    cfg.model.exogenous_refiner = True
    model = FarmModel(cfg.model, n_timesteps=T, chip_size=H, use_dummy=True, dummy_embed_dim=16,
                      n_soil=1, n_weather=1)

    class FakeReader:
        def raster_size(self, state, year):
            return (H, H)

        def read_chip(self, state, year, win):
            return ChipArrays(image=np.random.rand(6, T, H, H).astype(np.float32),
                              month_valid=np.ones((T, H, H), bool),
                              label=np.ones((H, H), np.float32), crop_mask=np.ones((H, H), bool),
                              county_id=None, transform=CHIP_TRANSFORM, crs=CRS,
                              bounds=(0, 0, 0, 0))

    norm = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", [],
                     soil_mean=[30.0], soil_std=[20.0], soil_vars=["clay"],
                     weather_mean=[15.0], weather_std=[5.0], weather_vars=["ppt"],
                      weather_layout="spatial_nearest_crop_label_v1")
    return cfg, model, FakeReader(), norm


def test_raster_inference_runs_with_soil_and_weather(tmp_path):
    from farm_us.evaluation.inference import predict_state_year

    cfg, model, reader, norm = _exo_setup(tmp_path)
    out = predict_state_year(model, reader, cfg, norm, "IA", 2020)
    assert out["prediction"].shape == (H, H) and np.isfinite(out["prediction"]).all()


def test_raster_inference_refuses_stats_without_exogenous_groups(tmp_path):
    from farm_us.evaluation.inference import predict_state_year

    cfg, model, reader, _ = _exo_setup(tmp_path)
    bare = NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "x", [])
    with pytest.raises(ValueError, match="soil"):
        predict_state_year(model, reader, cfg, bare, "IA", 2020)


def test_occlusion_by_exogenous_variable(tmp_path):
    from torch.utils.data import default_collate

    from farm_us.data.dataset import SyntheticFarmDataset
    from farm_us.interpretability.occlusion import occlusion_by_band, occlusion_by_exogenous

    cfg, model, _, _ = _exo_setup(tmp_path)
    ds = SyntheticFarmDataset(n=2, n_timesteps=T, chip=H, n_soil=1, n_weather=1)
    batch = default_collate([ds[0], ds[1]])
    zero = occlusion_by_exogenous(model, batch, ["clay"], ["ppt"])
    assert zero == {"soil:clay": 0.0, "weather:ppt": 0.0}       # zero-init branch: exact no-op
    with torch.no_grad():
        model.exo_refiner.out.weight.fill_(0.5)
        model.exo_refiner.weather_enc.to_shift.weight.fill_(0.1)
    live = occlusion_by_exogenous(model, batch, ["clay"], ["ppt"])
    assert live["soil:clay"] > 0 and live["weather:ppt"] > 0
    assert occlusion_by_band(model, batch).shape == (6,)          # existing paths still work


def test_spatial_weather_stats_mask_and_imputation():
    # Non-crop value 1000 must not change statistics; NaN must count in coverage
    # but not in mean/std. Zero precipitation is a valid observation.
    raw = np.array([[[[0, 10, 1000]], [[20, np.nan, 1000]]]], dtype=np.float32)
    valid = np.array([[True, True, False]])
    acc = ExogenousStatsAccumulator(0, 1)
    acc.update(weather_raw=raw, valid=valid)
    result = acc.result()["weather"]
    assert result["mean"] == pytest.approx([10])
    assert result["std"] == pytest.approx([np.std([0, 10, 20])])
    assert result["coverage"] == pytest.approx([0.75])
    scaled = normalize_generic(raw, result["mean"], result["std"])
    assert scaled.shape == raw.shape
    assert scaled[0, 1, 0, 1] == 0  # mean-imputed missing cell
    assert scaled[0, 0, 0, 0] < 0   # real zero rain remains distinct


def test_legacy_weather_statistics_rejected_on_transfer_and_inference():
    from farm_us.training.run import ensure_exogenous_stats

    cfg = _cfg_with(weather=["ppt"])
    old = NormStats([0.] * 6, [1.] * 6, {"mode": "none"}, "x", [],
                    weather_mean=[15.], weather_std=[5.], weather_vars=["ppt"])
    with pytest.raises(ValueError, match="legacy"):
        ensure_exogenous_stats(cfg, _DM(_NoReadDS()), old)
    with pytest.raises(ValueError, match="legacy"):
        require_exogenous_stats(cfg.data, old)


def test_dataset_augments_weather_and_inference_uses_same_maps(tmp_path):
    from farm_us.config import AugmentConfig
    from farm_us.data.dataset import FarmChipDataset, SampleRecord
    from farm_us.evaluation.inference import _ExogenousInputs

    cfg, _, reader, norm = _exo_setup(tmp_path)
    record = SampleRecord("test", "IA", 2020, 0, 0, 42, -93)
    ds = FarmChipDataset([record], cfg, reader, norm, augment=False)
    unflipped = ds[0]
    assert unflipped["weather"].shape == (1, T, H, H)
    assert torch.all(unflipped["weather"][..., :14] == -1)
    assert torch.all(unflipped["weather"][..., 14:] == 1)
    args = _ExogenousInputs(cfg, norm).tensors(reader.read_chip("IA", 2020, None), 2020, "cpu")
    torch.testing.assert_close(args["weather"][0], unflipped["weather"])
    cfg.augment = AugmentConfig(hflip_p=1., vflip_p=1., noise_p=0.)
    aug = FarmChipDataset([record], cfg, reader, norm, augment=True)[0]
    torch.testing.assert_close(aug["weather"], unflipped["weather"].flip(-1, -2))
    torch.testing.assert_close(aug["soil"], unflipped["soil"].flip(-1, -2))
