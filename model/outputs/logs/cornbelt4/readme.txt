LOGS - cornbelt4 (IA, IL, IN, MN | 2020-2024 | test 2024 | decoder_only)

config:   configs/experiments/cornbelt4_soybeans.yaml
run dir:  ../../runs/cornbelt4_soybeans/   (see readme there for results)


train_20260813_105608.log        batch 24
  changed: subsampled norm stats (2000 chips), precision 16-mixed, excluded
           4 corrupt chips found by the scan below
  result:  best epoch 2, val RMSE 5.4157, R2 0.4746 -> early overfit
  outputs: runs/cornbelt4_soybeans/test2024/archive/batch24_decoder_only/

train_20260813_234744.log        batch 8
  changed: batch 24 -> 8 (paper default). reused norm stats from the run above,
           so no 60 min stats pass
  result:  best epoch 2, val RMSE 5.3729, R2 0.4829 -> same, batch size ruled out
  outputs: runs/cornbelt4_soybeans/test2024/  (csv/version_3)
  note:    faster per epoch than batch 24 (7.7 vs 5.9 chips/s) - disk-bound box

hls_scan_20260813_101050.log     corruption scan, 4 states, 2020-2023
  128 files / 343,208 reads -> 2 corrupt files, 4 bad chips
  report:  ../../qc/hls_corruption_cornbelt4.json
  the 4 ids went into exclude_sample_ids in the config


next: ../nolocation/  (location embeddings off)
