LOGS - cornbelt5 (IA, IL, IN, MD, MN | 2020-2024 | test 2024 | decoder_only)

config:   configs/experiments/cornbelt5_soybeans.yaml
run dir:  ../../runs/cornbelt5_soybeans/   (readme there for results)

why MD: 84% of label variance is BETWEEN counties, so counties are what the
model is actually learning. MD adds 24 of them. Since our BARC test is for MD and the last MD only run was fairly better so why not test it right?



train_20260815_221757.log        5 states, location embeddings back ON
  changed from ../nolocation/train_20260814_221140.log:
      states              4 -> 5 (added MD)   33,215 train chips (was 32,336)
      use_location_embed  false -> true
      everything else the same (batch 8 x accum 3, grad_clip 1.0, wd 0.3,
      warmup 3, epochs 30, 16-mixed)
  norm stats: recomputed, NOT reused - train split changed when MD was added
  result:  best epoch 5, val RMSE 5.2673, R2 0.5237, r 0.7278  (best so far)
           best epoch moved 2 -> 5, but MD and the regularisation both changed


prev: ../nolocation/
