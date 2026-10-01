#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

for objective in mask band; do
  for condition in a b c; do
    score_conditions=(meta_frozen adapted)
    if [[ "$objective" == mask && "$condition" == a ]]; then
      score_conditions=(source_frozen meta_frozen adapted)
    fi
    .venv/bin/python scripts/323_evaluate_tusz_meta_ttt_v2.py \
      --meta-checkpoint "outputs/reports/tusz_meta_ttt_v4/runs/development/${objective}_f/${condition}/epoch_02.pt" \
      --output-root outputs/reports/tusz_meta_ttt_v4 \
      --partition train --cohort development_validation --calibrate \
      --conditions "${score_conditions[@]}" --tag "${condition}_${objective}" \
      > "outputs/reports/tusz_meta_ttt_v4/logs/eval_${objective}_${condition}.log" 2>&1
  done
done
