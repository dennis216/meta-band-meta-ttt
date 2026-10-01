#!/usr/bin/env bash
set -euo pipefail
r="outputs/reports/tusz_meta_ttt_v3"
p="outputs/reports/tusz_meta_ttt_v2/evaluation/future/eds_fast_mask_eds_epoch02/development_fit/probabilities.parquet"
common=(--output-root "$r" --probabilities "$p" --source-threshold 0.3025642229914664
        --high-score-threshold 0.732138168811798 --partition train
        --cohort development_fit --samples-per-group 1024
        --maximum-per-patient-group 32 --extended-controls)
for specification in "mask b0" "mask b3" "band b0" "band b1"; do
  read -r objective condition <<< "$specification"
  .venv/bin/python scripts/328_analyze_tusz_gradients_v2.py \
    --meta-checkpoint "$r/runs/development/${objective}_f/$condition/epoch_02.pt" \
    --tag "$objective" "${common[@]}" \
    > "$r/logs/mechanism_train_fit_${objective}_${condition}.log" 2>&1
done
