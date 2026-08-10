Third MD run, done after Augmentation_fixed overfit with the encoder unfrozen. Here we set model.finetune_mode: decoder_only, which freezes the whole Prithvi encoder and only trains the decoder and heads (194M of 826M params).

Training stopped early at epoch 34 (best val checkpoint at epoch 19). We evaluated both the best checkpoint (epoch 19) and a periodic one (epoch 29) on test year 2024 — epoch 29 actually did better on the test year despite epoch 19 being the val-selected one, so we kept both.
/val_year_tested is the log for the same decoder only model but tested on 2023 year

Check:
    - model/outputs/runs/maryland_soybeans/test2024/checkpoints/Decoder_Only
    - model/outputs/runs/maryland_soybeans/test2024/eval/Decoder_only
    - model/outputs/runs/maryland_soybeans/test2024/csv/version_6
    - model/outputs/runs/maryland_soybeans/test2024/tb/version_6
    - model/outputs/logs/Decoder_Only_MD/val_year_tested (eval against val year 2023)
