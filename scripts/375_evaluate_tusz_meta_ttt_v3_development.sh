#!/usr/bin/env bash
set -euo pipefail

root="outputs/reports/tusz_meta_ttt_v3"
evaluator="scripts/323_evaluate_tusz_meta_ttt_v2.py"

for condition in b1 b2 b3; do
  .venv/bin/python "$evaluator" \
    --meta-checkpoint "$root/runs/development/mask_f/$condition/epoch_02.pt" \
    --output-root "$root" --partition train --cohort development_validation \
    --calibrate --conditions meta_frozen adapted --tag mask \
    > "$root/logs/eval_mask_$condition.log" 2>&1
done

for condition in b0 b1 b2 b3; do
  .venv/bin/python "$evaluator" \
    --meta-checkpoint "$root/runs/development/band_f/$condition/epoch_02.pt" \
    --output-root "$root" --partition train --cohort development_validation \
    --calibrate --conditions meta_frozen adapted --tag band \
    > "$root/logs/eval_band_$condition.log" 2>&1
done
