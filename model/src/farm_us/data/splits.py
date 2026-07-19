"""Strict Leave-One-Year-Out (LOYO) splitting + leakage auditing.

For each fold: one test year, one or more validation years, all remaining years
train. The three are mutually exclusive. The test year is never used for
normalization, scaling, model selection, or anything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from ..utils.logging import LeakageError


@dataclass
class FoldSpec:
    test_year: int
    val_years: list[int]
    train_years: list[int]

    def assert_disjoint(self) -> None:
        s_test, s_val, s_train = {self.test_year}, set(self.val_years), set(self.train_years)
        if s_test & s_val or s_test & s_train or s_val & s_train:
            raise LeakageError(
                f"Year overlap across splits: test={s_test} val={s_val} train={s_train}"
            )
        if not self.train_years:
            raise LeakageError(f"Fold test={self.test_year} has no training years.")


def make_fold(
    test_year: int,
    all_years: list[int],
    val_years: list[int] | None = None,
    policy: str = "explicit_map",
    split_map: dict[int, list[int]] | None = None,
    val_pool: list[int] | None = None,
) -> FoldSpec:
    """Build one LOYO fold according to ``policy``.

    policies:
      - ``explicit_map``: use ``split_map[test_year]`` (or ``val_years`` arg).
      - ``fixed_pool``: use ``val_pool`` minus the test year.
      - ``previous_year``: validation = the single year before the test year.
    """
    years = sorted(all_years)
    if policy == "explicit_map":
        vy = (split_map or {}).get(test_year, val_years) if split_map else val_years
        if not vy:
            raise LeakageError(
                f"No explicit validation years for test_year={test_year}. "
                "Provide a split map (configs/splits/*.yaml) or split.val_years."
            )
    elif policy == "fixed_pool":
        pool = val_pool or []
        vy = [y for y in pool if y != test_year]
        if not vy:
            raise LeakageError(f"fixed_pool empty after removing test_year={test_year}.")
    elif policy == "previous_year":
        prev = test_year - 1
        vy = [prev] if prev in years else [max(y for y in years if y < test_year)]
    else:
        raise ValueError(f"Unknown split policy {policy!r}")

    vy = [y for y in vy if y != test_year]
    train = [y for y in years if y != test_year and y not in vy]
    fold = FoldSpec(test_year=test_year, val_years=sorted(vy), train_years=sorted(train))
    fold.assert_disjoint()
    return fold


def load_split_map(path: str) -> dict[int, list[int]]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    mapping = raw.get("val_years_by_test_year", raw)
    return {int(k): [int(x) for x in v] for k, v in mapping.items()}


def audit_manifest_split(
    records: list[dict],
    fold: FoldSpec,
    check_chip_overlap: bool = True,
) -> dict[str, object]:
    """Verify a materialized split has no leakage.

    Checks: year assignment matches fold; no duplicate sample_ids across splits;
    no overlapping chip windows across splits within a state-year.
    """
    fold.assert_disjoint()
    year_of_split = {}
    for y in fold.train_years:
        year_of_split[y] = "train"
    for y in fold.val_years:
        year_of_split[y] = "val"
    year_of_split[fold.test_year] = "test"

    issues: list[str] = []
    seen_ids: dict[str, str] = {}
    windows_by_split: dict[str, set] = {"train": set(), "val": set(), "test": set()}

    for r in records:
        y = int(r["year"])
        split = year_of_split.get(y)
        if split is None:
            continue  # year not in this fold's universe
        declared = r.get("split")
        if declared is not None and declared != split:
            issues.append(f"sample {r.get('sample_id')} split={declared} != expected {split}")
        sid = str(r.get("sample_id"))
        if sid in seen_ids and seen_ids[sid] != split:
            issues.append(f"duplicate sample_id {sid} across splits")
        seen_ids[sid] = split
        key = (r.get("state"), y, r.get("row_off"), r.get("col_off"))
        windows_by_split[split].add(key)

    if check_chip_overlap:
        # Overlap only matters within a state-year; but since splits are by year,
        # windows across splits already differ in the year field. Same-year
        # windows in different splits would be the real hazard — flag if found.
        for a in ("train", "val", "test"):
            for b in ("train", "val", "test"):
                if a < b:
                    inter = windows_by_split[a] & windows_by_split[b]
                    if inter:
                        issues.append(f"{len(inter)} overlapping windows between {a}/{b}")

    return {"ok": len(issues) == 0, "issues": issues, "fold": fold}
