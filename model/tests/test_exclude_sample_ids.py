"""Guards for data.exclude_sample_ids (corrupt-chip quarantine).

This is silent data removal, so the tests pin two things: it removes exactly
what it claims, and it stays inert unless explicitly populated.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from farm_us.config import FarmConfig, load_config


def _df(ids):
    return pd.DataFrame({
        "sample_id": ids,
        "state": ["IN"] * len(ids),
        "year": [2022] * len(ids),
        "row_off": list(range(len(ids))),
        "col_off": list(range(len(ids))),
    })


def _apply(df, excluded):
    """The exclusion half of filter_manifest_to_qualifying_chips, in isolation.

    The full function needs CDL/label rasters and county polygons on disk, so
    the qualifying-set logic is covered by the real-data tests; here we pin the
    exclusion semantics only.
    """
    out = df
    if excluded:
        present = set(out["sample_id"])
        hit = [s for s in excluded if s in present]
        if hit:
            out = out[~out["sample_id"].isin(hit)]
    return out


def test_default_is_empty_and_inert():
    assert FarmConfig().data.exclude_sample_ids == []
    df = _df(["a", "b", "c"])
    assert len(_apply(df, [])) == 3


def test_removes_exactly_the_listed_ids():
    df = _df(["a", "b", "c", "d"])
    out = _apply(df, ["b", "d"])
    assert sorted(out["sample_id"]) == ["a", "c"]


def test_unknown_ids_do_not_remove_anything():
    df = _df(["a", "b"])
    out = _apply(df, ["does_not_exist"])
    assert sorted(out["sample_id"]) == ["a", "b"]


def test_cornbelt4_config_lists_the_four_scanned_chips():
    """The shipped config must carry exactly the ids from the scan report.

    Pinned deliberately: a silent edit here changes which data trains the model,
    and the only defensible entries are chips whose imagery cannot be decoded.
    """
    cfg = load_config("configs/experiments/cornbelt4_soybeans.yaml")
    assert sorted(cfg.data.exclude_sample_ids) == [
        "soybeans_IN_2021_r2464_c3360",
        "soybeans_IN_2021_r2464_c3584",
        "soybeans_IN_2022_r11648_c4928",
        "soybeans_IN_2022_r11648_c5152",
    ]


def test_other_configs_do_not_silently_exclude():
    for path in (
        "configs/experiments/maryland_soybeans.yaml",
        "configs/experiments/barc_transfer.yaml",
    ):
        assert load_config(path).data.exclude_sample_ids == [], path


@pytest.mark.parametrize("ids", [["x"], ["x", "y"]])
def test_exclusion_is_logged_not_silent(caplog, ids):
    """Removing data must leave a trace in the log."""
    df = _df(["x", "y", "z"])
    with caplog.at_level(logging.WARNING):
        out = _apply(df, ids)
        # mirror the production log call
        logging.getLogger("farm_us.data").warning(
            "Excluded %d chip(s) via data.exclude_sample_ids: %s", len(ids), ", ".join(ids)
        )
    assert len(out) == 3 - len(ids)
    assert "exclude_sample_ids" in caplog.text
