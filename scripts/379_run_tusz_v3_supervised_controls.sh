#!/usr/bin/env bash
set -euo pipefail
r="outputs/reports/tusz_meta_ttt_v3"
p="$r/evaluation/future/b0_mask/development_validation/probabilities.parquet"
common=(--output-root "$r" --probabilities "$p" --source-threshold 0.3025642229914664
        --high-score-threshold 0.732138168811798 --partition train
        --cohort development_validation --samples-per-group 128
        --maximum-per-patient-group 32 --extended-controls)
.venv/bin/python scripts/328_analyze_tusz_gradients_v2.py \
  --meta-checkpoint "$r/runs/development/mask_f/b3/epoch_02.pt" \
  --method-probabilities "$r/evaluation/future/b3_mask/development_validation/probabilities.parquet" \
  --tag mask_controls "${common[@]}" > "$r/logs/controls_mask_b3.log" 2>&1
.venv/bin/python scripts/328_analyze_tusz_gradients_v2.py \
  --meta-checkpoint "$r/runs/development/band_f/b1/epoch_02.pt" \
  --method-probabilities "$r/evaluation/future/b1_band/development_validation/probabilities.parquet" \
  --tag band_controls "${common[@]}" > "$r/logs/controls_band_b1.log" 2>&1
