"""Guards for the opt-in normalization-stats subsample (norm.stats_max_chips).

The point of these tests is that the DEFAULT path must not move. Subsampling is
an optimisation for multi-hour folds; if enabling it ever became the implicit
default, every existing run's statistics would silently change meaning.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from farm_us.config import FarmConfig, load_config
from farm_us.data.normalization import NormStats
from farm_us.training.run import stats_chip_indices


def test_default_reads_every_chip_in_order():
    """None must be identity -- same chips, same order, no RNG consumed."""
    assert stats_chip_indices(1000, None, 0) == list(range(1000))


def test_cap_at_or_above_total_is_also_identity():
    assert stats_chip_indices(50, 50, 0) == list(range(50))
    assert stats_chip_indices(50, 999, 0) == list(range(50))


def test_subsample_size_uniqueness_and_order():
    idx = stats_chip_indices(10_000, 250, seed=0)
    assert len(idx) == 250
    assert len(set(idx)) == 250, "sampling must be WITHOUT replacement"
    assert idx == sorted(idx), "indices are sorted to keep raster reads sequential"
    assert all(0 <= i < 10_000 for i in idx)


def test_same_seed_reproduces_and_different_seed_differs():
    a = stats_chip_indices(10_000, 250, seed=0)
    b = stats_chip_indices(10_000, 250, seed=0)
    c = stats_chip_indices(10_000, 250, seed=1)
    assert a == b, "a seeded stats pass must be reproducible"
    assert a != c


def test_subsample_is_unbiased_enough_to_replace_a_full_pass():
    """The actual justification for the feature, stated as a test.

    Band statistics from a few hundred chips must match the full-population
    values closely enough that swapping them cannot move training. Uses the same
    order of magnitude of per-chip pixels as the real pipeline (50,176/band).
    """
    rng = np.random.default_rng(0)
    n_chips, per_chip = 4000, 5000
    population = rng.normal(loc=0.31, scale=0.14, size=(n_chips, per_chip))

    idx = stats_chip_indices(n_chips, 250, seed=0)
    sub = population[idx]

    assert abs(sub.mean() - population.mean()) < 0.002
    assert abs(sub.std() - population.std()) < 0.002


def test_norm_stats_records_subsample_provenance():
    full = NormStats(
        band_mean=[0.0] * 6, band_std=[1.0] * 6, target={"mode": "none"},
        mode="fold_training_statistics", train_years=[2020, 2021],
        n_chips_used=100, n_chips_total=100, stats_seed=None,
    )
    assert full.stats_seed is None, "a full pass records no seed"

    partial = NormStats(
        band_mean=[0.0] * 6, band_std=[1.0] * 6, target={"mode": "none"},
        mode="fold_training_statistics", train_years=[2020, 2021],
        n_chips_used=25, n_chips_total=100, stats_seed=7,
    )
    assert (partial.n_chips_used, partial.stats_seed) == (25, 7)


def test_load_tolerates_pre_existing_files_without_the_new_fields(tmp_path):
    """Checkpoints already on disk have norm_stats.json without these keys.

    NormStats.load does cls(**json), so a field lacking a default would break
    evaluation of every previously trained checkpoint.
    """
    legacy = {
        "band_mean": [0.1] * 6, "band_std": [0.2] * 6,
        "target": {"mode": "zscore", "center": 46.7, "scale": 6.3},
        "mode": "fold_training_statistics", "train_years": [2019],
    }
    p = tmp_path / "norm_stats.json"
    p.write_text(json.dumps(legacy))

    s = NormStats.load(p)
    assert s.band_mean == [0.1] * 6
    assert s.n_chips_used is None and s.stats_seed is None


def test_config_default_is_opt_in():
    assert FarmConfig().norm.stats_max_chips is None
    assert FarmConfig().norm.stats_seed == 0


@pytest.mark.parametrize("cfg_path", ["configs/experiments/cornbelt4_soybeans.yaml"])
def test_shipped_config_can_enable_it_via_override(cfg_path):
    cfg = load_config(cfg_path, ["norm.stats_max_chips=2000"])
    assert cfg.norm.stats_max_chips == 2000
    assert load_config(cfg_path).norm.stats_max_chips is None
