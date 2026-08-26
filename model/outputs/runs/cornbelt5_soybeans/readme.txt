RUN - cornbelt5_soybeans
IA, IL, IN, MD, MN | 2020-2024 | train 2020-22, val 2023, test 2024 | decoder_only

config: configs/experiments/cornbelt5_soybeans.yaml
log:    ../../logs/cornbelt5/train_20260815_221757.log

STATUS: still running (started 2026-08-15 22:18). Numbers below are as of
        epoch 7 of 30. patience 10 from best epoch 5 -> can reach epoch 15.


CHANGED FROM cornbelt4_no_location
  states              4 -> 5 (added MD)   33,215 train chips (was 32,336)
  use_location_embed  false -> true
  norm stats          RECOMPUTED, not reused - adding MD changed the train split
  everything else unchanged


WHY MD
  84% of the label variance is BETWEEN counties, 16% within. Counties are what
  the model is actually predicting, so more counties = more signal. MD adds 24.


RESULT SO FAR
  best ep 5   val RMSE 5.2673   R2 0.5237   r 0.7278

  run                   best ep   val R2
  cornbelt4 batch 24        2     0.4746
  cornbelt4 batch 8         2     0.4829
  cornbelt4 no-location     2     0.4978
  cornbelt5 (this)          5     0.5237   <- best so far

  Best R2 yet and the peak moved from epoch 2 to 5. But MD and the
  regularisation both changed, so the credit isn't cleanly assigned.
  +0.026 R2 for +6% county-years is roughly linear - a 15-state run would
  probably land ~0.55-0.60, not transform things.


FILES
  test2024/checkpoints/farm-005-0.0000.ckpt   best
  test2024/csv/version_0/metrics.csv
  test2024/norm_stats.json                    2000-chip subsample, seed 0,
                                              records states + train_years


DATA CHECK
  Corruption scan over all 5 states 2020-2024: 200 files, 440,744 reads,
  0 corrupt. MD clean, test year 2024 clean.
  ../../logs/nolocation/hls_scan5_20260815_221200.log
  ../../qc/hls_corruption_cornbelt5.json


ORDER
  cornbelt4_soybeans  ->  cornbelt4_no_location  ->  [this]
