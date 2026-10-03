# Meta-Band / Meta-TTT

Research code for seizure detection and self-supervised test-time adaptation on **CHB-MIT** and **TUSZ**, using CBraMod representations.

This repository contains training, evaluation, mechanism analysis, and performance benchmarks for Meta-Band / Meta-TTT. It preserves the historical CHB-MIT implementation and TUSZ v1–v4. Router and idea3 experiments are outside its scope. Each version defines separate experimental conditions: do not mix data splits, checkpoints, thresholds, or temporal protocols across versions.

## Getting started

| Resource | Location |
|---|---|
| Installation, assets, and execution | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| Complete script index | [docs/SCRIPT_INDEX.md](docs/SCRIPT_INDEX.md) |
| Release validation and limitations | [docs/VALIDATION.md](docs/VALIDATION.md) |
| Source provenance and original SHA-256 hashes | [docs/source-manifest.json](docs/source-manifest.json) |
| Third-party licenses | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

```text
src/bfa/                         Shared data, preprocessing, models, scoring, and training
src/bfa/tusz_meta_ttt/            TUSZ v1 and shared components
src/bfa/tusz_meta_ttt_v2/         Joint Meta, F/C modes, batching, and parallel execution
src/bfa/tusz_meta_ttt_v3/         Alignment and damage constraints
src/bfa/tusz_meta_ttt_v4/         Paired gain, Frozen preservation, and gradient decomposition
scripts/                        Numbered training, evaluation, analysis, and benchmark entry points
external/NeuroTTT_CBraMod/       CHB-MIT GroupKFold / Meta-Band implementation
third_party/CBraMod/             CBraMod model dependency used by TUSZ
archive/chb-20260905/            Frozen CHB scripts that differ from the current versions
configs/ + protocols/           Existing configurations and protocols
tests/                          Unit, numerical, and state-semantics tests
tools/                          Runtime relocation and release verification
```

## Quick verification

Use Linux / WSL2 and Python 3.11. Install a PyTorch build compatible with your GPU driver, then run:

```bash
python -m pip install -e '.[test]'
python tools/verify.py
```

These tests require neither EEG data nor trained checkpoints. GPU training and full reproduction require datasets, CBraMod pretrained weights, S1 checkpoints, warm-start heads, and matching manifests; see the reproduction guide. Some historical scripts retain machine-specific absolute paths. `tools/prepare_runtime.py` creates a separate relocated runtime copy without changing the published source.

`tools/verify.py` runs release checks and CPU tests in the current Python environment. It returns a nonzero exit code on failure and does not start training. Run it after cloning or before committing.

## Experiment versions

| Version | Main content |
|---|---|
| CHB-MIT | GroupKFold, Band auxiliary task, historical window/record/patient adaptation experiments |
| TUSZ v1 | Data audit, supervised S0/S1, SSL, early Meta, and event evaluation |
| TUSZ v2 | Encoder / detector / SSL-head outer scopes; separately trained F/C modes; throughput optimization |
| TUSZ v3 | Band / Mask Post-BCE, alignment, and damage ablations |
| TUSZ v4 | Corrected event scoring; paired pre/post gain on the same query; Frozen preservation |

In F mode, updates from the current chunk serve future chunks. In C mode, adaptation uses the entire current chunk before retrospectively predicting that chunk. C does not have F's online timing semantics. Versions v2 onward reuse signal caches and do not establish strict sample-by-sample causality from raw EDF to alarms.

This is a code release; it includes no patient-level results or new clinical/performance claims. Historical protocols describe experimental designs, not evidence that every experiment has completed or passed. Packaging did not start training or modify the original research checkout.

## Data, weights, and licensing

Raw EEG, patient manifests, caches, model weights, per-window probabilities, logs, and emails are not distributed. Obtain TUSZ access under the provider's requirements. CBraMod license notices are retained. No additional open-source license has been selected for the original research code; public readability does not grant additional commercial or sublicensing rights.
