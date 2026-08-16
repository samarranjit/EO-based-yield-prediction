RUN - cornbelt4_soybeans
IA, IL, IN, MN | 2020-2024 | train 2020-22, val 2023, test 2024 | decoder_only

config: configs/experiments/cornbelt4_soybeans.yaml
logs:   ../../logs/cornbelt4/


RESULTS
  batch 24   best ep 2   val RMSE 5.4157   R2 0.4746   csv/version_2 (archived)
  batch 8    best ep 2   val RMSE 5.3729   R2 0.4829   csv/version_3

  batch size made no difference -> ruled out as the cause of the plateau


WHERE THINGS ARE
  test2024/checkpoints/farm-002-0.0000.ckpt      best, batch 8 run
  test2024/csv/version_3/metrics.csv             batch 8 metrics
  test2024/norm_stats.json                       2000-chip subsample, seed 0
  test2024/archive/batch24_decoder_only/         the whole batch-24 run,
                                                 moved here so the batch-8 run
                                                 could not overwrite it


WHAT WE LEARNED (measured, not guessed)
  - val R2 plateaus ~0.48 from epoch 0 while train loss keeps falling 3x
  - ruled out: batch size (3x), spatial diversity (31x more locations vs
    Maryland), location embeddings (see ../cornbelt4_no_location/)
  - ruled out: label ceiling. If the model nailed every county mean the
    residual RMSE would be 2.98; it gets 5.29, so signal is still on the table
  - 84% of label variance is BETWEEN counties -> counties are the real target
  - still open: chip is 6.72 km, county is ~50 km, so each chip sees ~2% of the
    area whose mean it must predict


ORDER
  cornbelt4_soybeans  ->  cornbelt4_no_location  ->  cornbelt5_soybeans
