#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source_probabilities="outputs/reports/tusz_meta_ttt_v4/evaluation/future/a_a_mask/development_validation/probabilities.parquet"
source_threshold="0.5036728276938632"
high_threshold="0.732138168811798"

for condition in a b c; do
  .venv/bin/python scripts/328_analyze_tusz_gradients_v2.py \
    --meta-checkpoint "outputs/reports/tusz_meta_ttt_v4/runs/development/band_f/${condition}/epoch_02.pt" \
    --output-root outputs/reports/tusz_meta_ttt_v4 \
    --probabilities "$source_probabilities" \
    --method-probabilities "outputs/reports/tusz_meta_ttt_v4/evaluation/future/${condition}_${condition}_band/development_validation/probabilities.parquet" \
    --source-threshold "$source_threshold" --high-score-threshold "$high_threshold" \
    --partition train --cohort development_validation --samples-per-group 1024 \
    --maximum-per-patient-group 32 --extended-controls --tag "v4_band_${condition}" \
    > "outputs/reports/tusz_meta_ttt_v4/logs/mechanism_band_${condition}.log" 2>&1
done
