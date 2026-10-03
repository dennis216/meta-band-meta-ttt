# Reproduction and execution

## Environment

Use Python 3.11 on Linux or WSL2. Training scripts call `fcntl` and CUDA; native Windows training is not guaranteed. Release tests used the current research environment, PyTorch `2.14.0+cu130`. The CHB migration snapshot recorded `2.11.0+cu128`. Version differences can affect second-order gradients, numerical error, and events near thresholds; numerical equivalence across these versions has not been established.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# First install a CUDA wheel compatible with your driver using PyTorch's official instructions.
python -m pip install -e '.[test]'
```

Dependencies for additional historical baselines are available through `.[legacy]`; the main TUSZ path does not require installing every historical model package. `environment/tested-versions.txt` records the tested environment; it is an audit snapshot, not a portable installation lockfile.

## Relocating paths

Most modern TUSZ entry points derive the project root from their file location. Older CHB queues retain original absolute paths. The following tool creates a **new runtime copy** and replaces known paths without overwriting the repository or an existing runtime directory:

```bash
python tools/prepare_runtime.py \
  --destination /path/to/meta-ttt-runtime \
  --tusz-root /data/TUSZ_v2.0.6 \
  --chb-root /data/chbmit-1.0.0 \
  --chb-cache /data/chb-cbramod-cache
cd /path/to/meta-ttt-runtime
python -m venv .venv
source .venv/bin/activate
# After installing CUDA PyTorch:
python -m pip install -e '.[test]'
```

The runtime copy records replacements in `runtime-paths.json`. The archive is for reference, not execution. The tool relocates paths only: it does not generate patient manifests, transfer models, or change algorithms.

Relocation preserves code directories such as `src/bfa/data` and `tests/data`. It excludes root-level data/output directories, generated caches, model files, and symbolic links. Run `python tools/verify.py` after relocation; prepare EEG and weights separately as described below.

## External assets

1. Place CBraMod pretrained weights at `third_party/CBraMod/pretrained_weights/pretrained_weights.pth`. CHB entry points use `external/NeuroTTT_CBraMod/pretrained_weights/pretrained_weights.pth`.
2. Transfer validated S1 and SSL warm-start checkpoints securely from the research project, preserving their relative locations under `outputs/reports/tusz_meta_ttt_v1` and `tusz_meta_ttt_v2`. Load only trusted checkpoints; the original training entry points restore full PyTorch checkpoint objects.
3. Transfer or rebuild `records.json`, `development_split.json`, signal caches, and sidecars. This code release cannot recover missing assets. Exact reproduction requires the original splits and matching content hashes.
4. CHB requires `manifests/windows.parquet`, `recordings.parquet`, `seizures.parquet`, `groupkfold_cv_v1/fold_*.json`, and matching signal caches. `03_build_manifests.py` and `03b_build_windows.py` generate basic manifests; retain the original five-fold assignments separately.

Upstream weight location, as recorded in the original migration guide:
https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth

The recorded SHA-256 is
`0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`.
Verify it after downloading. Weights were not downloaded again during packaging.

## TUSZ workflow

To prepare new manifests and caches:

```bash
python scripts/301_prepare_tusz_meta_ttt_v1.py audit --content-hash
python scripts/301_prepare_tusz_meta_ttt_v1.py cache --partition train
python scripts/309_materialize_tusz_signal_sidecars_v1.py --help
```

For existing experiments, reuse validated S1 checkpoints and caches; migration alone does not require retraining S1. Inspect input arguments before proceeding:

```bash
python scripts/321_train_tusz_ssl_v2.py --help
python scripts/322_calibrate_tusz_inner_v2.py --help
python scripts/401_rescore_tusz_meta_ttt_v4.py --help
python scripts/410_calibrate_tusz_meta_ttt_v4.py --help
python scripts/411_train_tusz_meta_ttt_v4.py --help
```

Example v4 command, after preparing assets and calibrating loss scales:

```bash
python scripts/411_train_tusz_meta_ttt_v4.py \
  --objective band --conditions a b c \
  --loss-scale /path/to/band-loss-scale.json \
  --reference-probabilities /path/to/s1-reference.parquet \
  --patients-per-batch 1 --epochs 2 --seed 3407 \
  --output outputs/reports/tusz_meta_ttt_v4/runs/development/band_f
```

For Mask, use `--objective mask` and its independently calibrated file. Paths above are placeholders; use the files and formats produced by calibration. `configs/tusz_meta_ttt_v4/unit_loss_scale.json` contains unit coefficients and is not a substitute for calibration.

Evaluation and statistics entry points are `323`, `401`, `412`, `413`, `414`, `415`, and `416`. Shell queues contain preset directories, output names, and log paths; inspect them before running. Do not launch every `scripts/*.sh` file with a wildcard. Benchmarks and queues also start potentially long-running jobs.

## CHB-MIT workflow

```bash
export BFA_ROOT="$PWD"
export BFA_CACHE_ROOT=/data/chb-cbramod-cache
export NEUROTTT_CODE_ROOT="$PWD/external/NeuroTTT_CBraMod"
python external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_train.py --help
python external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_evaluate.py --help
python scripts/280_retrain_band_ttt_v2.py --help
python scripts/281_evaluate_retrained_band_ttt_v2.py --help
```

The early `230`-series TU transfer experiments reference historical generated manifests and remain available for traceability. They do not implement the full official-partition protocol of TUSZ v1–v4. Shared modules also retain a few historical model adapters whose weights and third-party implementations are not bundled; they are not required assets for the main Meta-TTT workflow.

## Validation scope

See [VALIDATION.md](VALIDATION.md). Release tests verify code and data-free numerical cases. They do not replace full GPU training, multiple-seed reruns, event confidence intervals, or verification of paper results.
