import pytest

from farm_us.data.splits import FoldSpec, audit_manifest_split, make_fold
from farm_us.utils.logging import LeakageError

YEARS = list(range(2014, 2025))


def test_make_fold_disjoint():
    fold = make_fold(2018, YEARS, val_years=[2017], policy="explicit_map")
    assert fold.test_year == 2018
    assert 2018 not in fold.train_years and 2018 not in fold.val_years
    assert set(fold.val_years).isdisjoint(fold.train_years)
    assert len(fold.train_years) == len(YEARS) - 2


def test_previous_year_policy():
    fold = make_fold(2020, YEARS, policy="previous_year")
    assert fold.val_years == [2019]


def test_fixed_pool_removes_test_year():
    fold = make_fold(2019, YEARS, policy="fixed_pool", val_pool=[2019, 2020])
    assert 2019 not in fold.val_years
    assert fold.val_years == [2020]


def test_missing_val_raises():
    with pytest.raises(LeakageError):
        make_fold(2018, YEARS, val_years=[], policy="explicit_map")


def test_overlap_detection():
    bad = FoldSpec(test_year=2018, val_years=[2018], train_years=[2017])
    with pytest.raises(LeakageError):
        bad.assert_disjoint()


def test_audit_flags_year_in_wrong_split():
    fold = make_fold(2018, YEARS, val_years=[2017])
    records = [
        {"sample_id": "a", "state": "IA", "year": 2016, "row_off": 0, "col_off": 0, "split": "train"},
        {"sample_id": "b", "state": "IA", "year": 2018, "row_off": 0, "col_off": 0, "split": "train"},  # wrong!
    ]
    res = audit_manifest_split(records, fold)
    assert not res["ok"]
    assert any("split" in i for i in res["issues"])


def test_audit_clean_split_ok():
    fold = make_fold(2018, YEARS, val_years=[2017])
    records = [
        {"sample_id": "a", "state": "IA", "year": 2016, "row_off": 0, "col_off": 0, "split": "train"},
        {"sample_id": "b", "state": "IA", "year": 2017, "row_off": 0, "col_off": 0, "split": "val"},
        {"sample_id": "c", "state": "IA", "year": 2018, "row_off": 0, "col_off": 0, "split": "test"},
    ]
    res = audit_manifest_split(records, fold)
    assert res["ok"], res["issues"]
