#!/usr/bin/env bash
set -euo pipefail

# Sequential, resumable release queue.  Validation is locked before the
# corresponding test; a failed gate is recorded and does not trigger test.
PYTHON="/root/b_false_alarm_atlas/.venv/bin/python"
SCRIPT="/mnt/c/Users/User/Documents/ChatGPT/EEG_ZiquanBaoBao/metaTTT_migration_20260905/project/scripts"
OUT="/root/b_false_alarm_atlas/outputs/reports/meta-ttt-chbmit-v2-repaired"
WINDOWS="/root/b_false_alarm_atlas/manifests/windows.parquet"
FOLDS="/root/b_false_alarm_atlas/manifests/groupkfold_cv_v1"
CACHE="/mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod"
# Prefix cache is deliberately on WSL ext4.  EEG mmap inputs remain on the
# existing data volume, while millions of small prefix entries stay off NTFS.
PREFIX_CACHE="/root/b_false_alarm_atlas/cache/meta_ttt_prefix_v2"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

export PYTHONUNBUFFERED=1

run_stage() {
  local label="$1"; shift
  local log="$LOGDIR/${label}.log"
  echo "[$(date --iso-8601=seconds)] START $label" | tee -a "$LOGDIR/queue.log"
  "$@" 2>&1 | tee "$log"
  echo "[$(date --iso-8601=seconds)] DONE $label" | tee -a "$LOGDIR/queue.log"
}

run_background_stage() {
  local label="$1"; shift
  local log="$LOGDIR/${label}.log"
  echo "[$(date --iso-8601=seconds)] START $label (background)" | tee -a "$LOGDIR/queue.log"
  "$@" >"$log" 2>&1 &
  echo "$!" >"$LOGDIR/${label}.pid"
}

cleanup_fold_prefix_cache() {
  local fold="$1"
  local metadata="$OUT/runs/fold${fold}/common_classifier_seed3407/completed.json"
  local classifier_hash
  classifier_hash="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["classifier_hash"])' "$metadata")"
  if [[ ! "$classifier_hash" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Invalid classifier hash while cleaning prefix cache: $classifier_hash" >&2
    return 1
  fi
  for branch in band learned; do
    local target="$PREFIX_CACHE/$branch/$classifier_hash"
    if [[ -d "$target" ]]; then
      echo "[$(date --iso-8601=seconds)] CLEAN re-generable prefix cache $target" | tee -a "$LOGDIR/queue.log"
      rm -rf -- "$target"
    fi
  done
}

formal_args=(--seed 3407 --output-root "$OUT" --windows "$WINDOWS" --fold-root "$FOLDS" --cache-root "$CACHE" --prefix-cache "$PREFIX_CACHE" --workers 8 --device cuda)

for fold in 0 1; do
  run_stage "fold${fold}_supervised" "$PYTHON" "$SCRIPT/280_retrain_band_ttt_v2.py" --mode supervised --fold "$fold" "${formal_args[@]}"
  run_stage "fold${fold}_band_head" "$PYTHON" "$SCRIPT/280_retrain_band_ttt_v2.py" --mode band_prepare --fold "$fold" "${formal_args[@]}"

  # The two outer objectives share only the frozen classifier and prepared
  # Band head.  They are independent GPU jobs and are safe to run together;
  # each process still keeps its own episode states and never averages inner
  # gradients across branches.
  meta_pids=()
  for branch in band learned; do
    label="fold${fold}_${branch}_meta"
    run_background_stage "$label" "$PYTHON" "$SCRIPT/280_retrain_band_ttt_v2.py" --mode meta --branch "$branch" --fold "$fold" "${formal_args[@]}" --prefix-cache "$PREFIX_CACHE/$branch" --meta-steps 3000 --min-meta-steps 1000 --eval-interval 250 --meta-patience 4 --effective-episodes 8 --validation-episodes 256
    meta_pids+=("$!")
  done
  for pid in "${meta_pids[@]}"; do
    wait "$pid"
  done
  echo "[$(date --iso-8601=seconds)] DONE fold${fold}_meta_pair" | tee -a "$LOGDIR/queue.log"
  for branch in band learned; do
    run_stage "fold${fold}_${branch}_validation" "$PYTHON" "$SCRIPT/281_evaluate_retrained_band_ttt_v2.py" --fold "$fold" --branch "$branch" --split validation "${formal_args[@]}" --prefix-cache "$PREFIX_CACHE/$branch"
    allowed="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("test_allowed",False))' "$OUT/evaluations/fold${fold}/${branch}/validation/validation_selection.json")"
    if [[ "$allowed" == "True" ]]; then
      run_stage "fold${fold}_${branch}_test" "$PYTHON" "$SCRIPT/281_evaluate_retrained_band_ttt_v2.py" --fold "$fold" --branch "$branch" --split test --allow-test "${formal_args[@]}" --prefix-cache "$PREFIX_CACHE/$branch"
    else
      echo "[$(date --iso-8601=seconds)] SKIP fold${fold}_${branch}_test validation_gate_failed" | tee -a "$LOGDIR/queue.log"
    fi
  done
  # Fold-specific prefixes are re-generable and can occupy hundreds of GiB.
  # Remove only the completed fold's exact classifier-hash directories before
  # starting the next fold; checkpoints, histories, and evaluations remain.
  if [[ "$fold" == "0" ]]; then
    cleanup_fold_prefix_cache "$fold"
  fi
done

echo "[$(date --iso-8601=seconds)] QUEUE COMPLETE" | tee -a "$LOGDIR/queue.log"
