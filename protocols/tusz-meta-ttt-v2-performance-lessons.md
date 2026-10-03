# TUSZ Meta-TTT v2: performance findings and reusable lessons

Recorded on 2026-09-15 using one RTX 5090 (32 GB). Formal development training currently uses seed 3407. This document records engineering and throughput findings; detection gains require subsequent full-record evaluation. See the original protocol and interim measurements in [tusz-meta-ttt-v2-performance-revision.md](tusz-meta-ttt-v2-performance-revision.md).

## 1. Optimization target and measurement conventions

The target is the time required for one Meta-training traversal with identical data coverage and condition counts. Training includes patient/record reads, every valid window at a 2-second stride, SSL transforms, the frozen encoder prefix, second-order inner updates of the last two layers, future/current chunk detection BCE, and an outer step after every four patients.

Report the following separately:

- **Training-computation seconds:** from completion of model/data loading to completion of training computation; useful for operator optimization.
- **Total wall-clock seconds:** includes startup, reading, checkpoints, and writing; useful for queue budgets.
- **Aggregate inner updates/s:** summed updates across independent conditions divided by the specified time. This measures experimental-matrix throughput, not additional independent EEG exposure.
- **Whole-GPU metrics:** NVIDIA-SMI utilization, allocated memory, and power cannot be attributed to a single condition. Separate cold-start and steady-state samples.
- **Correctness:** compare window coverage, classification losses, acceptance decisions, final model/SSL-head parameters, and meta-gradients on the same records; subsequently compare probabilities and event alarms.

Power of at least 480 W was initially a diagnostic target; the user subsequently prioritized final training speed. High power neither substitutes for throughput measurement nor proves that the second-order graph uses the GPU effectively.

Short-record benchmarks take prefixes of sorted Train-fit records, which do not represent the full TUSZ patient or duration distribution. Times from different record sets, F/C combinations, or physical patient batches cannot be divided directly to infer method improvements.

## 2. Initial bottleneck: CPU dispatch of many tiny GPU operations

An early four-condition Mask test on 24 real records performed 560 updates per condition at approximately 20.81 updates/s and 212.5 W mean power. Eight independent processes reached approximately 21.01 updates/s, and twelve reached 20.81 updates/s, while peak memory rose from 8,282 MiB to 14,886 and 21,344 MiB. Additional processes duplicated models, optimizers, and prefix computation; higher GPU occupancy readings did not mean faster completion.

A real four-step Meta profile over three segments captured approximately 26,010 GPU kernels and 22,446 `cudaLaunchKernel` calls. Many kernels lasted microseconds. One Python training process nearly saturated a CPU core while aggregate usage across 32 logical cores remained low. Python/autograd scheduling and fragmented kernel launches were the main bottlenecks, not simply too few CPU workers. Profiling adds overhead: approximately 120.9 ms of active kernels within a 1,133 ms first-to-last-kernel interval must not be reported as production SM utilization or stage percentages.

Compare wall time and updates/s at matched data and condition counts first, then inspect single-core CPU load, kernel duration, memory, and power when deciding whether to enlarge computational batches. Instantaneous NVIDIA-SMI utilization alone can favor a slower configuration.

## 3. Combining small operations into larger parallel computations

### 3.1 Pack inner updates for the final two layers

The original implementation checked finite values, norms, relative steps, and Armijo candidates separately across many small parameters, causing CPU/GPU synchronization. The packed implementation in `update.py` concatenates parameters and gradients within each Transformer block, computes block norms, directions, and drift together, then reconstructs parameter views. Armijo candidate SSL losses are checked under `no_grad`, while accepted updates retain their second-order graphs. Relative-step denominators remain differentiable so scaling the SSL loss cannot arbitrarily disable or amplify adaptation.

Mask originally used dynamic Boolean indexing. Fixed-size, stably sorted indices and `gather` avoid host synchronization to determine output length. Time and spectral branches share masked positions, and loss still covers only masked elements. Comparisons passed for loss and first-/second-order gradients with 3/5/7 masked positions.

Incremental Mask measurements on the same 24 records and four independent conditions:

| Implementation | Total wall time | Aggregate updates/s | Change |
|---|---:|---:|---|
| Reduced per-parameter host synchronization | 107.64 s | 20.81 | Reference |
| Packed block updates + Mask gather | 73.71 s | 30.39 | Approximately 46% higher throughput |
| With frozen-prefix CUDA Graph | 62.86 s | 35.64 | Approximately 71% above reference |

Packing and gather were introduced together, so their individual contributions cannot be separated from the 46% improvement. With eight processes, the corresponding rates were 21.01, 30.58, and 36.49 updates/s, supporting operation consolidation over further process replication.

### 3.2 Cache and replay the frozen prefix

The model is split into patch embedding plus ten frozen encoder layers, two adaptable encoder layers, and a detector. F/C conditions can reuse frozen-prefix outputs without backward graphs. Same-shape CUDA Graphs are captured after side-stream warmup, and replay outputs are cloned into independent storage. Cloning is necessary because four-step suffix backward reads earlier prefixes, which later replays must not overwrite. Uncommon final-chunk shapes fall back to eager execution, and the graph cache is limited to four shapes to control memory.

Real CBraMod Band/Mask checks cover prefix outputs, independent storage across replays, and four-step Meta parameters and gradients. The tested maximum graph/eager difference was zero. Armijo candidate checks also reuse prefixes instead of recomputing the whole backbone.

### 3.3 Batch independent patients and outer conditions

`batched.py` represents patient fast weights along a leading lane dimension and uses `vmap` for functional computation of the final two layers and independent detector/SSL heads. Inner gradients, Armijo decisions, record resets, and drift caps are patient-specific. E/ED/ES/EDS conditions retain independent models, heads, optimizers, and checkpoints; only read-only prefixes and deterministic support views are shared.

PyTorch's native MHA inference fastpath misinterpreted `requires_grad` with BatchedTensor, and that operator lacked usable backward support. The parallel second-order path explicitly disables native MHA fastpath and flash/memory-efficient/cuDNN SDP, using verified math attention. Applying `vmap` without double-backward checks can fail during training even when forward succeeds.

Idle lanes introduce another second-order issue: computing `sqrt(norm²)` and then squaring creates a nondifferentiable intermediate at zero. The update denominator now uses the squared norm directly; diagnostic norms are detached. If any lane gradient is nonfinite, the implementation conservatively rejects the entire parallel inner batch, preserves previous parameters, and records the rejection to prevent zero-times-NaN contamination through `where`. Healthy lanes are also rejected and must be counted in reports.

On the same 48 Train-fit records, 15 patients, and four F conditions, four independent processes took 109.28 s total wall time at 43.30 updates/s. A single four-condition process took 63.24 s of training computation at approximately 74.82 updates/s. Two processes with two conditions each took 75.06 s wall time at 63.04 updates/s. Maximum FP32 parameter differences from independent execution were approximately 1.13e-6 for models and 1.56e-6 for SSL heads.

A synthetic sustained second-order probe with 16 lanes and cached prefixes reached approximately 200.87 updates/s and sampled 497–506 W. This demonstrates hardware potential but excludes EDF reads, SSL transforms, and full prefix computation, so it cannot budget full training.

### 3.4 Share identical signal computations between F and C

F updates on the current chunk and predicts a future chunk. C adapts using the entire current chunk and predicts it retrospectively. They are trained and reported separately but may share raw signal computations for the same patient/chunk. The joint scheduler caches each patient's previous, current, and next raw prefix so F's early query read does not evict the value C still needs. F skips the useless update on the final EDF chunk, but its raw signal must not be zeroed because C still updates on it.

Joint Mask over 48 records and F/C × four conditions performed 9,656 updates: 79.52 s training computation (121.42 updates/s), 85.91 s total wall time (112.40 updates/s), 20,543 MiB peak whole-GPU memory, and approximately 435 W steady-state mean power. Each condition counts each label and class weight once; F retains its first direct BCE and final query, and C retains all current queries.

The equivalent eight-condition Band run took approximately 275.28 s training computation and 281.95 s wall time at 34.25 updates/s, with 29,500 MiB peak memory and 481.84 W steady-state power. Five band views per support window make Band's memory and transform costs substantially higher than Mask's. Under the 30 GiB monitoring budget, Band therefore uses one patient lane.

## 4. Grouping, idle lanes, and memory trade-offs

After inventory filtering, development Train-fit contains 463 patients, 3,838 EDFs, 710.72 hours, and 1,261,284 prediction windows. Each condition has 83,394 F-mode or 87,160 C-mode inner updates per traversal. Overlapping support windows do not add independent EEG hours.

Patient workloads differ substantially. Bucket by window count, randomize within buckets, and shuffle four-patient groups using unlabeled length information only, not seizure type or model score. The predicted active fraction for fixed two-lane scheduling rises from approximately 0.60 with random order to 0.92 with bucket size 8; four lanes rise from approximately 0.36 to 0.85–0.86. Traversal-order seeds 3408/3409 give similar results. These are scheduling estimates, not measured speed. The formal queue uses bucket size 8 and still needs full-traversal verification.

Dynamic compression removes completed lanes instead of computing padding for later ticks. On Band with 48 records, four F conditions, and two patient lanes, compression reduced training time from 166.76 s to 144.44 s, approximately 13%, while peak memory rose from 29,808 to 30,104 MiB, closer to the 32 GB limit. One patient lane took approximately 156.98 s. Merely adding patients can therefore raise power and memory while slowing execution. Mask uses two lanes because its views are lighter.

Physical batching differs from the outer batch: process one or two patients at a time but accumulate gradients over the full four-patient group before an outer step. OOM must not silently reduce four-step truncation, support windows, or adaptable layers. Change only validated physical microbatches and record the protocol/memory impact.

## 5. Attempts without an adoptable speed improvement

- **More independent GPU processes:** 4→8→12 kept aggregate Mask throughput near 21 updates/s while memory grew nearly linearly; duplication outweighed parallel benefit.
- **Two fixed Band patient lanes:** approximately 473 W and 29.8 GiB, but slower than one lane. Dynamic compression helped, with a memory-headroom trade-off.
- **Deduplicating repeated SSL views:** eight-condition Band took 275.28 s before and 276.77 s after deduplication; all 9,656 updates were accepted, and all eight final models/heads were identical. Repeated transformations were removed, but measured end-to-end speed did not improve. Retain it as shared-computation infrastructure without claiming measured acceleration.
- **Exploratory TF32 `high`:** four F conditions fell from 63.24 s in FP32 to 55.47 s, but model differences rose to approximately 2e-5–4e-5 and SSL-head differences to 3.6e-4–4.1e-4. Event alarms are sensitive near thresholds; event equivalence remains unverified, so formal training keeps FP32/highest matmul precision.
- **Chasing >=480 W alone:** joint Band reached approximately 480–482 W but only 34–35 updates/s, far below the lower-power joint Mask run. Power does not rank different SSL workloads.
- **Unconditional prefetch, more CPU workers, or caching every SSL feature:** second-order operators and fragmented launches dominate current costs. Extra copies, caches, and workers can increase memory pressure unless read stalls are shown to dominate. Memory-mapped source caches and on-demand windows remain the default.

## 6. Meta semantics that optimization must preserve

Inner and deployment updates still affect only the final two encoder layers. E/ED/ES/EDS outer scopes optimize their specified encoder, detector, and SSL-head combinations. Dropout is disabled in Meta training and deployment. S1 initialization and cache preprocessing remain unchanged; this round does not establish strict raw-EDF-to-alarm sample-level causality.

Patient batching changes backward boundaries to four synchronized lane chunk ticks instead of four locally counted updates per record. Numeric fast weights carry across truncation and reconnect through `theta_start = theta_0 + stopgrad(theta_carry-theta_0)`; each EDF resets to its own source. Record boundaries can shorten a segment, and F's final-chunk no-op can reduce actual updates within four ticks. This is an explicit meta-gradient approximation change; old checkpoints and new training histories must not be combined.

Band views originally used band→lane→window order; shared-prefix computation requires lane→band→window. Transforms, labels, and loss normalization must be reordered together, or runnable code can assign incorrect auxiliary labels. Mask targets and spectral inputs must use the same original signal and positions; the model must not see masked spectra before reconstructing them.

## 7. Full training, resume, and validation

The full development queue [358_run_tusz_fast_development_v2.py](../scripts/358_run_tusz_fast_development_v2.py) trains Mask then Band, two traversals each, with seed 3407. Each SSL runs eight independent F/C × E/ED/ES/EDS conditions in one GPU process. Mask uses two physical patient lanes; Band uses one. Bucket size is 8, and all valid records/windows are used. The detector is frozen for the first 25% of outer steps in traversal one, then enabled where its outer scope permits. Each condition logs inner/outer gradient norms, acceptance rates, clipping ratios, actual parameter steps, class weights, and coverage.

[356_train_tusz_ensemble_v2.py](../scripts/356_train_tusz_ensemble_v2.py) atomically saves resumable checkpoints every four outer steps. They contain mutable parameters, SSL heads, optimizers, random states, patient cursors, cumulative coverage, and protocol hashes for all eight conditions. Interruption saves only at complete four-patient group boundaries; resume replays unsaved groups. On 48 real records, uninterrupted versus interrupted-after-first-group/resumed training gave zero maximum differences in all eight models, heads, and optimizers, with identical windows, updates, and class weights. `last.pt` is recovery state; per-mode/scope `epoch_XX.pt` files are evaluation checkpoints.

Each launch records 30 seconds of consecutive NVIDIA-SMI samples; the queue logs resources every 30 seconds. Full-launch checks confirmed a live process, approximately 20 GiB memory, and sampled power of 460–480 W. Completion requires verifying 463 patients, 3,838 records, and 1,261,284 prediction windows per condition per traversal. A queue marker is insufficient. Internal validation, full-background event evaluation, threshold calibration, large-sample gradient/SSL analysis, and formal confirmation require separate acceptance checks.

Before formal launch, 75 relevant tests passed, including update finite differences, Mask first-/second-order gradients, Band/Mask lane losses, F/C boundary windows and resets, independent heads, compression, bucketing, and resume. Real CBraMod four-step graphs, final parameters on 48 records, and checkpoint recovery were also compared. Deployment evaluation reads packed-update and prefix-precision settings from checkpoints. Final numerical/alarm equivalence near thresholds still requires complete evaluation.

## 8. Procedure for future optimization

1. Freeze manifests, window/label semantics, source checkpoints, and outer conditions. Select fixed real short records and inspect the full record-length distribution.
2. Separate cold start, training computation, and wall time. Measure one condition, single-core load, and launches before trying larger same-process functional batches.
3. Share read-only frozen prefixes and deterministic transforms first. Keep trainable heads, fast weights, Armijo decisions, and optimizers independent.
4. After changes to update kernels, attention backends, or CUDA Graphs, verify losses, first-/second-order gradients, acceptance, actual parameters, and events, not only maximum probability error.
5. Compare wall time with identical records, condition counts, precision, and updates. Prefer the faster configuration if increased power slows execution.
6. Test real interruption/resume and complete patient/window coverage before full training. Inspect the process, GPU samples, and first checkpoint after launch, then use long waits.

Primary raw evidence is under [benchmarks/](../outputs/reports/tusz_meta_ttt_v2/benchmarks/). Full-training throughput, GPU logs, and checkpoints are under [development_fast_v2_1/](../outputs/reports/tusz_meta_ttt_v2/runs/meta/development_fast_v2_1/). Update final speed and detection claims using complete traversals, evaluation, and fixed-threshold results.
