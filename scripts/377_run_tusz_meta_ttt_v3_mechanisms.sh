#!/usr/bin/env bash
set -euo pipefail

r="outputs/reports/tusz_meta_ttt_v3"
source_prob="$r/evaluation/future/b0_mask/development_validation/probabilities.parquet"
common=(--output-root "$r" --probabilities "$source_prob"
        --source-threshold 0.3025642229914664 --high-score-threshold 0.732138168811798
        --partition train --cohort development_validation --samples-per-group 1024
        --maximum-per-patient-group 32 --extended-controls)

.venv/bin/python scripts/328_analyze_tusz_gradients_v2.py \
  --meta-checkpoint "$r/runs/development/mask_f/b3/epoch_02.pt" \
  --method-probabilities "$r/evaluation/future/b3_mask/development_validation/probabilities.parquet" \
  --tag mask "${common[@]}" > "$r/logs/mechanism_mask_b3.log" 2>&1

for condition in b0 b1; do
  .venv/bin/python scripts/328_analyze_tusz_gradients_v2.py \
    --meta-checkpoint "$r/runs/development/band_f/$condition/epoch_02.pt" \
    --method-probabilities "$r/evaluation/future/${condition}_band/development_validation/probabilities.parquet" \
    --tag band "${common[@]}" > "$r/logs/mechanism_band_$condition.log" 2>&1
done
