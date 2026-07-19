# Leave-One-Year-Out (LOYO) Protocol

## Rule
For every fold: **one** test year, **≥1** validation years, **all** remaining
years train. The three sets are mutually exclusive (`FoldSpec.assert_disjoint`).

## What the test year is NEVER used for
normalization · target scaling · model selection · early stopping ·
hyper-parameter selection · threshold selection · checkpoint selection ·
calibration · any visual inspection used to pick a model.

Normalization band stats and the target scaler are computed from **training
years only** (streaming, `data/normalization.py`; wired in `training/run.py`).

## Validation-year policies (`split.policy`)
1. `explicit_map` (default) — `configs/splits/loyo_soybeans.yaml` maps each test
   year → its validation years. Editable; the audit enforces disjointness.
2. `fixed_pool` — a fixed pool with the test year auto-removed.
3. `previous_year` — validation = the year immediately before the test year
   (forecasting style).

The shipped map uses previous-year validation (test 2018 → val 2017 → train the
other 9 years).

## Leakage audit (`leakage-audit` CLI / `audit_manifest_split`)
Checks: no year overlap across splits; declared split matches the fold; no
duplicate `sample_id` across splits; no overlapping chip windows across splits
within a state-year. Because splitting is **by year**, all chips of a (state,
year) inherit that year's split, so overlapping windows can never straddle
splits.

## Label provenance caveat (important)
The US labels are **ridge-distributed county yield**. If the ridge weights that
produced a held-out year's labels were themselves estimated using that same
year's NASS county yield, then:

> the **neural-network split is strictly LOYO**, but the **label-generation
> provenance may not be** — the held-out year's *county statistic* could have
> influenced its own pseudo-labels.

We do **not** refit or alter the ridge model or the labels. We record
`label_source = ridge_distributed_county_yield` and `ridge_provenance` in the
manifest and provenance file, and the audit surfaces this. Treat LOYO here as
evaluating **temporal transfer of the imagery→yield mapping**, complemented by
**county-level aggregation** and **BARC** (measured-label) evaluation.

## LOYO ≠ geographic generalization
LOYO holds out *years*, not *regions*. It does not prove the model generalizes to
unseen geographies, and with `-TL` **location embeddings** the model can learn
regional yield baselines (a spatial shortcut). See
[LOCATION_EMBEDDING_RISK.md](LOCATION_EMBEDDING_RISK.md) and run the mandatory
no-location ablation.
