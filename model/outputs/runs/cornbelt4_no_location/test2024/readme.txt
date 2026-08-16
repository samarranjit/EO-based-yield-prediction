RUN - cornbelt4_no_location   (ablation)
IA, IL, IN, MN | train 2020-22, val 2023, test 2024 | decoder_only

config: configs/experiments/cornbelt4_soybeans.yaml
        + model.use_location_embed=false
        + experiment_name=cornbelt4_no_location
log:    ../../../logs/nolocation/train_20260814_221140.log


QUESTION
  Prithvi-EO-2.0-TL is fed lat/lon. LOYO holds out years, not places, so the
  model could just learn "this location yields X" and skip the imagery.
  docs/LOCATION_EMBEDDING_RISK.md calls this ablation mandatory.


CHANGED FROM cornbelt4_soybeans (batch 8 run)
  use_location_embed  true -> false      (time embeddings stay on)
  grad_clip           none -> 1.0
  weight_decay        0.1  -> 0.3
  warmup_epochs       5    -> 3
  grad_accum          1    -> 3
  epochs              20   -> 30


RESULT
  best ep 2   val RMSE 5.2947   R2 0.4978   r 0.7142   (13 epochs)

  with location:     R2 0.4829
  without location:  R2 0.4978   <- slightly BETTER

  So location is not doing the work. Not a shortcut. The model is reading
  imagery, and it edges past a location-only lookup baseline (r 0.704).

  Caveat: regularisation changed in the same run, so location and
  regularisation are not perfectly separated.


FILES
  checkpoints/farm-002-0.0000.ckpt   best
  csv/version_0/metrics.csv
  norm_stats.json                    reused from cornbelt4 (same 4 states)


ORDER
  cornbelt4_soybeans  ->  [this]  ->  cornbelt5_soybeans
