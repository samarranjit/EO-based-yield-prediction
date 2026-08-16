LOGS - cornbelt4 no-location ablation (IA, IL, IN, MN | test 2024 | decoder_only)

config:   configs/experiments/cornbelt4_soybeans.yaml
          + model.use_location_embed=false
          + experiment_name=cornbelt4_no_location
run dir:  ../../runs/cornbelt4_no_location/test2024/   (readme there)

why: Prithvi-EO-2.0-TL gets lat/lon. LOYO holds out years, not places, so the
model can learn "this location yields X" instead of reading imagery.
See docs/LOCATION_EMBEDDING_RISK.md.


train_20260814_221140.log        location embeddings OFF
  changed from ../cornbelt4/train_20260813_234744.log:
      use_location_embed  true -> false
      grad_clip           none -> 1.0
      weight_decay        0.1  -> 0.3
      warmup_epochs       5    -> 3
      grad_accum          1    -> 3   (batch 8 x 3 = effective 24)
      epochs              20   -> 30
  result:  best epoch 2, val RMSE 5.2947, R2 0.4978, r 0.7142  (13 epochs)
           slightly BETTER without location -> not a location shortcut
  caveat:  regularisation changed at the same time, so the two aren't isolated

hls_scan5_20260815_221200.log    corruption scan, 5 states, 2020-2024
  200 files / 440,744 reads -> 0 corrupt, 0 unreadable
  MD and test-year 2024 both clean
  report:  ../../qc/hls_corruption_cornbelt5.json


prev: ../cornbelt4/        next: ../cornbelt5/  (adds MD, location back on)
