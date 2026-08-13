"""Guards for norm.reuse_stats_from.

Reuse skips the statistics pass, so the risk is not wasted time but silently
training on statistics that saw the test year. Every test here is about the
refusal path; the happy path is one line by comparison.
"""

from __future__ import annotations

import json

import pytest

from farm_us.config import FarmConfig, load_config
from farm_us.data.normalization import NormStats
from farm_us.training.run import StatsReuseError, load_or_compute_fold_stats


class _Fold:
    def __init__(self, train_years, test_year=2024):
        self.train_years = train_years
        self.test_year = test_year
        self.val_years = [2023]


def _write(tmp_path, train_years, mode="fold_training_statistics", name="norm_stats.json"):
    p = tmp_path / name
    p.write_text(json.dumps({
        "band_mean": [0.1] * 6, "band_std": [0.2] * 6,
        "target": {"mode": "zscore", "center": 58.0, "scale": 9.4},
        "mode": mode, "train_years": list(train_years),
        "n_chips_used": 2000, "n_chips_total": 32336, "stats_seed": 0,
    }))
    return p


def _cfg(reuse=None):
    c = FarmConfig()
    c.norm.reuse_stats_from = reuse
    return c


def test_default_none_means_compute(monkeypatch):
    """Without the option, behaviour is unchanged -- the pass still runs."""
    called = {}

    def fake(cfg, dm):
        called["yes"] = True
        return NormStats([0.0] * 6, [1.0] * 6, {"mode": "none"}, "fold_training_statistics", [2020])

    monkeypatch.setattr("farm_us.training.run.compute_fold_stats", fake)
    load_or_compute_fold_stats(_cfg(None), object(), _Fold([2020]))
    assert called.get("yes"), "compute_fold_stats must still run when reuse is unset"


def test_matching_years_are_reused(tmp_path, monkeypatch):
    p = _write(tmp_path, [2020, 2021, 2022])

    def boom(cfg, dm):
        raise AssertionError("stats pass must be SKIPPED when reuse is valid")

    monkeypatch.setattr("farm_us.training.run.compute_fold_stats", boom)
    stats = load_or_compute_fold_stats(_cfg(str(p)), object(), _Fold([2020, 2021, 2022]))
    assert stats.n_chips_used == 2000
    assert stats.target["center"] == 58.0


def test_year_order_does_not_matter(tmp_path):
    p = _write(tmp_path, [2022, 2020, 2021])
    stats = load_or_compute_fold_stats(_cfg(str(p)), object(), _Fold([2020, 2021, 2022]))
    assert stats is not None


def test_refuses_when_train_years_differ(tmp_path):
    """The core leakage guard."""
    p = _write(tmp_path, [2020, 2021, 2022, 2023])
    with pytest.raises(StatsReuseError, match="train_years"):
        load_or_compute_fold_stats(_cfg(str(p)), object(), _Fold([2020, 2021, 2022]))


def test_refuses_when_stats_saw_the_test_year(tmp_path):
    """The specific disaster: stats computed on a split including 2024."""
    p = _write(tmp_path, [2020, 2021, 2022, 2024])
    with pytest.raises(StatsReuseError):
        load_or_compute_fold_stats(_cfg(str(p)), object(), _Fold([2020, 2021, 2022], test_year=2024))


def test_refuses_on_mode_mismatch(tmp_path):
    p = _write(tmp_path, [2020, 2021, 2022], mode="official_prithvi_statistics")
    with pytest.raises(StatsReuseError, match="mode"):
        load_or_compute_fold_stats(_cfg(str(p)), object(), _Fold([2020, 2021, 2022]))


def test_missing_file_raises_rather_than_silently_recomputing(tmp_path):
    """A typo'd path must not fall back to an hour-long pass unnoticed."""
    with pytest.raises(StatsReuseError, match="does not exist"):
        load_or_compute_fold_stats(_cfg(str(tmp_path / "nope.json")), object(), _Fold([2020]))


def test_shipped_configs_do_not_enable_reuse():
    for path in (
        "configs/experiments/cornbelt4_soybeans.yaml",
        "configs/experiments/multistate_soybeans.yaml",
        "configs/experiments/maryland_soybeans.yaml",
    ):
        assert load_config(path).norm.reuse_stats_from is None, path
