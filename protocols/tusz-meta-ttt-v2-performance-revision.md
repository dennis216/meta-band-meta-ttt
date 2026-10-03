# TUSZ Meta-TTT v2: performance revisions and measurements

Date: 2026-09-14. Only seed 3407 is active; seeds 17 and 42 are deferred. The formal development queue was stopped at the user's request. The measurements below are isolated performance experiments, not complete research results.

## Selection criterion

Minimize total time for the same data coverage and experimental conditions. GPU power, utilization, and memory are diagnostic measurements. The user subsequently prioritized final training speed, so a slower configuration will not be selected merely for higher power draw.

At each launch, inspect processes and consecutive NVIDIA-SMI samples before waiting for completion. Distinguish startup overhead from training time. Parallel jobs use the same seed; count patients/records separately from independent conditions.

## Measured optimizations

### Mask: 24 fixed Train-fit records, 560 inner updates per condition

| Implementation | Condition processes | Total batch wall time (s) | Aggregate updates/s | Peak memory (MiB) | Mean power (W) |
|---|---:|---:|---:|---:|---:|
| Reduced per-parameter host synchronization | 4 | 107.64 | 20.81 | 8,282 | 212.5 |
| Same | 8 | 213.28 | 21.01 | 14,886 | 216.4 |
| Same | 12 | 322.90 | 20.81 | 21,344 | 217.8 |
| Packed block updates + fixed Mask gather | 4 | 73.71 | 30.39 | 8,096 | 241.9 |
| Same | 8 | 146.48 | 30.58 | 14,657 | 249.2 |
| With frozen-prefix CUDA Graph | 4 | 62.86 | 35.64 | 8,959 | 258.6 |
| Same | 8 | 122.78 | 36.49 | 16,422 | 268.2 |

Additional processes do not resolve all bottlenecks. Packing reduces small-tensor kernels; fixed gather avoids synchronization when dynamic Boolean indexing determines output sizes; CUDA Graph reduces CPU overhead from repeatedly launching the first ten frozen encoder layers.

### Patient batching and condition ensembles on real records

These F-mode tests use 48 Train-fit records, 15 patients, four conditions (E/ED/ES/EDS), and 4,732 total updates.

| Implementation | Training time (s) | Other timing details |
|---|---:|---|
| Four processes, four patients per process | Slowest condition: 105.16 | All jobs completed in 109.28 s wall time |
| One process, four conditions sharing the prefix | 63.24 | Excludes model initialization and final output writing |
| Two processes, two conditions each | Slowest process: 69.16 | All jobs completed in 75.06 s wall time |
| One process, four conditions, exploratory TF32 high | 55.47 | Not adopted as the default numerical configuration |

Maximum absolute differences between the FP32 ensemble and independent-process final parameters were approximately 1.13e-6 for encoder/detector and 1.56e-6 for SSL heads. TF32 increased the SSL-head difference to approximately 4.08e-4. Timing alone therefore does not justify adopting it; prediction, acceptance-state, and event comparisons are still required.

On the same 48 records, Band with four conditions took 156.98 s with one patient lane per condition and 166.76 s with two. Two lanes reached 29,808 MiB and approximately 473 W steady-state mean power but were slower. Dynamic removal of completed patient lanes is being tested to avoid computing padding.

## Evidence from the real second-order path

A single-process Mask profile covering three four-step segments captured approximately 26,010 GPU kernels. Many lasted only a few microseconds, indicating substantial CPU launch and autograd scheduling overhead. Profiling itself adds overhead; the kernel-active fraction is not a measurement of production SM utilization. Nested CPU/GPU intervals must not be added, and GPU annotation percentages in raw PyTorch tables are not stage-time fractions.

A sustained second-order probe with 16 independent fast-weight lanes achieved approximately 200.87 updates/s and sampled 497–506 W. It used synthetic inputs and cached features, demonstrating batching potential rather than full EDF training throughput. Real-record results are given above.

## Preserved experiment semantics

- Reuse the original development S1, caches, and input scale; do not retrain S1.
- Keep 10-second inputs, 2-second strides, all valid dense windows, first/last chunks, and short records.
- Inner updates affect only the final two encoder blocks; outer scopes remain E/ED/ES/EDS.
- Each condition retains its own source, fast weights, heads, optimizer, and Armijo decisions. Only read-only frozen prefixes are shared.
- Take each condition's outer optimizer step only after all records of four patients finish.
- Preserve patient, class, record, and window weights; do not rebalance each chunk.
- No forced gradient alignment, learned loss, or injected random update directions.

## Explicit change to the training approximation

Experimental patient batching backpropagates every four synchronized lane ticks. A lane resets at a record boundary; a segment can contain fewer than four updates if the boundary falls within it. Carried parameter values are retained and reconnected to the encoder initialization after truncation.

This changes the truncation phase relative to independently aligning four updates within each record. Treat it as an explicit approximation revision, and do not merge old checkpoint epoch histories with the new ones. Deployment F/C ordering, record resets, support windows, and label-use rules remain unchanged.

## Validation and remaining work

Completed checks:

- Packed-block acceptance/rejection branches, finite differences, and four-step meta-gradient comparisons on real CBraMod.
- Mask gather versus Boolean indexing: loss, first-order, and second-order gradients with 3/5/7 masked positions.
- CUDA Graph prefix values, independent storage for historical outputs, and Band/Mask four-step parameters and meta-gradients; maximum difference was zero in this test.
- Condition ordering and gradient consistency for independent heads; changing one head leaves other conditions unchanged.
- The idle-lane second-order NaN fix and unaffected gradients in other lanes.
- Band/Mask padding leaves valid-support losses and gradients unchanged.
- F/C first/last windows, short records, single use of each label, and record-reset scheduling.

Before production use, still check longer records and more truncation segments, real-record C-mode comparisons, dynamic lane compression, checkpoint recovery and exception fallbacks, complete event outputs, and final runtime estimates on representative records. Entry points 352/354 are benchmarks, not production trainers.

Raw reports are under `outputs/reports/tusz_meta_ttt_v2/benchmarks/`; old and new results occupy separate directories.

## Full-training entry-point revision (2026-09-15)

The new `356_train_tusz_ensemble_v2.py` runs full development training with seed 3407. For each SSL family, one process computes eight F/C × E/ED/ES/EDS conditions with independent parameters, optimizers, losses, and histories. Mask uses two patient lanes; Band uses one. An outer step follows all records of four patients. Training starts from the original S1 and completed SSL warm-start heads; outputs go to `runs/meta/development_fast_v2_1`.

Patients are bucketed by window count, randomized within buckets, and grouped into randomly ordered groups of four. Bucketing uses no seizure labels. Bucket size is fixed at 8. The complete development set has 463 patients, 3,838 records, 710.72 hours, and 1,261,284 prediction windows. The predicted active-compute fraction for fixed two-lane scheduling improves from approximately 60% with random grouping to 92%. This is a load-balancing estimate; full-training logs determine actual speed.

The final matched Band comparison took 275.28 s before deduplication and 276.77 s afterward, completing and accepting all 9,656 updates. All eight final models and SSL heads had zero parameter differences. Deduplication reduced repeated transformations but did not improve end-to-end time in this test. After deduplication, steady-state mean power was 480.17 W and peak whole-GPU memory was 29,678 MiB. The eight-condition Mask benchmark sharing F/C support prefixes took 79.52 s at 121.42 updates/s. This covers the training computation on 48 records, not the complete dataset.

Training safeguards:

- Freeze the detector for the first 25% of outer steps in the first traversal; thereafter update detector, encoder, and SSL head according to each condition's scope.
- For each group and condition, log classification loss, windows, updates, accepted updates, inner gradient norms, pre-clipping outer gradient norms, clipping ratios, and actual parameter steps.
- Save checkpoints atomically every four outer steps. SIGTERM/SIGINT requests save after the current four-patient group finishes. Resume validates hashes for S1, SSL heads, calibration, inventory, splits, and path lists, plus bucketing, parallelism, and temporal protocols.
- Checkpoints include all mutable parameters, eight optimizers, random states, epoch, patient cursor, cumulative coverage, and diagnostics. Resume replays complete unsaved groups rather than inferring mid-record state.
- Nonfinite inner gradients reject the entire parallel inner graph, preserve pre-update parameters, and increment `nonfinite_batch_rejections`. This conservatively rejects healthy lanes too, avoiding `0*NaN` contamination in second-order backward. Nonfinite outer gradients pause training and retain the preceding complete checkpoint rather than saving corrupted weights.
- Keep FP32 and highest matrix-multiplication precision; TF32/BF16 approximations are not enabled.

`357_verify_tusz_training_resume_v2.py` compares uninterrupted execution against interruption after the first group and resume on 48 real records, checking all eight models, heads, optimizers, and coverage. Queue `358_run_tusz_fast_development_v2.py` requires this report to pass. It runs two Mask traversals, then two Band traversals; each launch samples the GPU for 30 seconds, with resource logs every 30 seconds during execution. Temporal remains excluded following its task health-check failure; old training is not resumed.

Complete event-output comparisons, internal-validation selection, and large-sample mechanism statistics remain evaluation tasks. The queue's completion file indicates only that two development traversals finished, not that the full research plan is complete.
