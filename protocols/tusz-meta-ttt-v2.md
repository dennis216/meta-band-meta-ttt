# TUSZ Meta-TTT v2 frozen implementation protocol

This version reuses the TUSZ v1 signal cache and existing supervised S1 checkpoints. The
cache was produced by the v1 causal preprocessing path, whose final resampling boundary
was later found not to be bitwise prefix invariant. Consequently, v2 makes no strict
raw-EDF-to-alarm causality claim. Adaptation compute time is treated as zero in the main
method comparison and measured separately.

Two separately trained semantics are used. In `future` mode, a chunk is predicted with the
current state and then supplies the SSL update used by the following chunk. In `current`
mode, the complete chunk supplies an SSL update before that same chunk is predicted; its
scores are retrospective and become available at the chunk end. Both modes reset at every
EDF and update only encoder blocks 10 and 11 at deployment.

The development source is the existing seed-3407 development S1. Formal runs use the
existing full-Train S1 for the matching seed. The three v2 SSL families are Band,
Temporal, and Mask. Learned scalar loss and cosine-alignment penalties are excluded.

The complete, machine-readable protocol is `configs/tusz_meta_ttt_v2.yaml`. Results and
checkpoints are written only below `outputs/reports/tusz_meta_ttt_v2`.

Implemented entry points:

- `321_train_tusz_ssl_v2.py`: two-pass SSL-head warm start and validation diagnostics.
- `325_select_tusz_ssl_v2.py`: preregistered difficulty health check, using no seizure BCE.
- `322_calibrate_tusz_inner_v2.py`: dense-support gradient calibration and automatic
  selection of the smallest relative step that passes the update checks.
- `324_benchmark_tusz_meta_ttt_v2.py`: isolated inner/meta throughput measurement.
- `320_train_tusz_meta_ttt_v2.py`: F/C joint Meta training for E, E+D, E+S, and E+D+S,
  with four-update truncation, detector warm freeze, resumable RNG/optimizer state, and
  separate gradient/change logs.
- `323_evaluate_tusz_meta_ttt_v2.py`: source-Frozen, Meta-Frozen, and Adapted inference;
  Dev calibration and fixed-threshold Eval scoring are separate command modes.
- `326_train_tusz_supervised_control_v2.py`: label-only E and E+D controls.
- `327_run_tusz_meta_development_v2.py`: dry-run queue construction by default; explicit
  `--execute` runs the 24 conditions with two-to-five epoch validation stopping.

The queue runner is deliberately inert without `--execute`. A failed SSL or inner-update
health check remains in the queue as a failed family and does not trigger a wider search.
