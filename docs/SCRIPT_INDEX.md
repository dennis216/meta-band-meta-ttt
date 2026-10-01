# Script index

保留原脚本编号。shell 队列会启动实际作业；运行前阅读脚本及所需资产说明。

| 入口 | 说明 |
|---|---|
| [02_download_chbmit.sh](../scripts/02_download_chbmit.sh) | 历史队列/运行入口 |
| [03_build_manifests.py](../scripts/03_build_manifests.py) | build manifests |
| [03b_build_windows.py](../scripts/03b_build_windows.py) | build windows |
| [110_preflight_no_duplicate_overlap.py](../scripts/110_preflight_no_duplicate_overlap.py) | Preflight guard against duplicate or silently overlapping experiments. |
| [202_cbramod_same_patient_adaptation.py](../scripts/202_cbramod_same_patient_adaptation.py) | Same-patient CBraMod adaptation upper-bound experiment. |
| [204_audit_cbramod_same_patient_adaptation.py](../scripts/204_audit_cbramod_same_patient_adaptation.py) | Audit and threshold-sweep the completed same-patient CBraMod experiment. |
| [205_audit_cbramod_prequential_calibration.py](../scripts/205_audit_cbramod_prequential_calibration.py) | Prequential, prior-record-only calibration audit for same-patient adaptation. |
| [210_joint_ttt_train.py](../scripts/210_joint_ttt_train.py) | Joint source training for the CBraMod test-time-training experiment. |
| [211_queue_joint_ttt_v1.sh](../scripts/211_queue_joint_ttt_v1.sh) | 历史队列/运行入口 |
| [212_meta_ttt_train.py](../scripts/212_meta_ttt_train.py) | First-order MT3-style meta-training for unlabeled CBraMod TTT. |
| [214_evaluate_joint_ttt.py](../scripts/214_evaluate_joint_ttt.py) | Evaluate a trained Joint- or Meta-TTT CBraMod checkpoint without target labels. |
| [216_queue_meta_ttt_v1.sh](../scripts/216_queue_meta_ttt_v1.sh) | 历史队列/运行入口 |
| [219_wait_then_start_meta_v1.sh](../scripts/219_wait_then_start_meta_v1.sh) | 历史队列/运行入口 |
| [221_queue_joint_eval_v1.sh](../scripts/221_queue_joint_eval_v1.sh) | 历史队列/运行入口 |
| [222_queue_meta_eval_v1.sh](../scripts/222_queue_meta_eval_v1.sh) | 历史队列/运行入口 |
| [223_wait_then_eval_meta_v1.sh](../scripts/223_wait_then_eval_meta_v1.sh) | 历史队列/运行入口 |
| [224_summarize_ttt_results.py](../scripts/224_summarize_ttt_results.py) | Aggregate the completed Joint-TTT, MT3-style, and label-prior runs. |
| [225_wait_then_summarize_ttt_v1.sh](../scripts/225_wait_then_summarize_ttt_v1.sh) | 历史队列/运行入口 |
| [226_audit_ttt_results.py](../scripts/226_audit_ttt_results.py) | Audit the completed method-comparison runs without recomputing metrics. |
| [227_analyze_ttt_results.py](../scripts/227_analyze_ttt_results.py) | Read-only, fold-aware analysis of the three CBraMod TTT experiments. |
| [228_write_ttt_report.py](../scripts/228_write_ttt_report.py) | Write a concise, auditable Markdown report from finished TTT analyses. |
| [230_run_tusz_joint_ttt_one_round.py](../scripts/230_run_tusz_joint_ttt_one_round.py) | One-round external TTT for Joint-TTT CBraMod on the official TUSZ Eval cohort. |
| [231_wait_then_tusz_joint_ttt_one_round.sh](../scripts/231_wait_then_tusz_joint_ttt_one_round.sh) | 历史队列/运行入口 |
| [232_validate_fast_joint_ttt_scoring.py](../scripts/232_validate_fast_joint_ttt_scoring.py) | Numerically validate and benchmark the deduplicated Joint-TTT scorer. |
| [233_parallel_joint_ttt_evaluation.py](../scripts/233_parallel_joint_ttt_evaluation.py) | Run independent Joint-TTT fold/seed evaluations concurrently on one GPU. |
| [234_audit_joint_ttt_parameter_update.py](../scripts/234_audit_joint_ttt_parameter_update.py) | Audit that one Joint-TTT update changes the frozen detector parameters. |
| [235_audit_joint_ttt_detailed.py](../scripts/235_audit_joint_ttt_detailed.py) | Read-only audit of Joint-TTT training, checkpoint changes, and evaluation. |
| [237_probe_joint_ttt_probability_drift.py](../scripts/237_probe_joint_ttt_probability_drift.py) | Read-only probe of block-wise probability drift in completed test runs. |
| [251_summarize_neurottt_eight_alarm_time.py](../scripts/251_summarize_neurottt_eight_alarm_time.py) | Summarize alarm-time occupancy for the eight non-baseline NeuroTTT conditions. |
| [264_freeze_band_ttt_v2.py](../scripts/264_freeze_band_ttt_v2.py) | Freeze the prespecified Band-TTT v2 matrix and official CHB-MIT record order. |
| [265_evaluate_band_ttt_v2.py](../scripts/265_evaluate_band_ttt_v2.py) | Causal Band-TTT v2 evaluator for the frozen 16-condition fold-0/1 matrix. |
| [266_summarize_band_ttt_v2.py](../scripts/266_summarize_band_ttt_v2.py) | Summarize completed Band-TTT v2 fold-0/1 jobs against frozen baselines. |
| [267_queue_band_ttt_v2.py](../scripts/267_queue_band_ttt_v2.py) | Resumable single-phase queue for frozen Band-TTT v2 evaluation. |
| [268_import_existing_band_ttt_v2.py](../scripts/268_import_existing_band_ttt_v2.py) | Register mathematically identical existing Window results in the final v2 release. |
| [269_wait_then_queue_band_ttt_v2.sh](../scripts/269_wait_then_queue_band_ttt_v2.sh) | 历史队列/运行入口 |
| [270_queue_band_ttt_v2_paired.py](../scripts/270_queue_band_ttt_v2_paired.py) | Resumable paired validation/test queue for frozen Band-TTT v2. |
| [271_run_band_ttt_v2_paired.sh](../scripts/271_run_band_ttt_v2_paired.sh) | 历史队列/运行入口 |
| [272_diagnose_band_gradient_alignment.py](../scripts/272_diagnose_band_gradient_alignment.py) | Diagnose Band-SSL/classification gradient alignment before retraining. |
| [280_retrain_band_ttt_v2.py](../scripts/280_retrain_band_ttt_v2.py) | Registered two-fold Band-TTT retraining release. |
| [281_evaluate_retrained_band_ttt_v2.py](../scripts/281_evaluate_retrained_band_ttt_v2.py) | Causal continuous validation/test evaluator for the repaired release. |
| [282_queue_retrained_band_ttt_v2.sh](../scripts/282_queue_retrained_band_ttt_v2.sh) | 历史队列/运行入口 |
| [283_preflight_retrained_band_ttt_v2.py](../scripts/283_preflight_retrained_band_ttt_v2.py) | Preflight checks required before the formal repaired release queue. |
| [300_preflight_tusz_meta_ttt_v1.py](../scripts/300_preflight_tusz_meta_ttt_v1.py) | Fail-closed preflight for the TUSZ Meta-TTT v1 experiment namespace. |
| [301_prepare_tusz_meta_ttt_v1.py](../scripts/301_prepare_tusz_meta_ttt_v1.py) | prepare tusz meta ttt v1 |
| [302_train_tusz_supervised_v1.py](../scripts/302_train_tusz_supervised_v1.py) | train tusz supervised v1 |
| [303_train_tusz_meta_ttt_v1.py](../scripts/303_train_tusz_meta_ttt_v1.py) | train tusz meta ttt v1 |
| [304_evaluate_tusz_meta_ttt_v1.py](../scripts/304_evaluate_tusz_meta_ttt_v1.py) | evaluate tusz meta ttt v1 |
| [305_train_tusz_ssl_head_v1.py](../scripts/305_train_tusz_ssl_head_v1.py) | train tusz ssl head v1 |
| [306_train_tusz_joint_ttt_v1.py](../scripts/306_train_tusz_joint_ttt_v1.py) | train tusz joint ttt v1 |
| [307_audit_tusz_meta_ttt_v1.py](../scripts/307_audit_tusz_meta_ttt_v1.py) | audit tusz meta ttt v1 |
| [308_analyze_tusz_gradients_v1.py](../scripts/308_analyze_tusz_gradients_v1.py) | analyze tusz gradients v1 |
| [309_materialize_tusz_signal_sidecars_v1.py](../scripts/309_materialize_tusz_signal_sidecars_v1.py) | materialize tusz signal sidecars v1 |
| [310_run_tusz_meta_short_grid_v1.py](../scripts/310_run_tusz_meta_short_grid_v1.py) | run tusz meta short grid v1 |
| [311_recalibrate_tusz_evaluation_v1.py](../scripts/311_recalibrate_tusz_evaluation_v1.py) | recalibrate tusz evaluation v1 |
| [312_run_tusz_meta_short_evaluations_v1.py](../scripts/312_run_tusz_meta_short_evaluations_v1.py) | run tusz meta short evaluations v1 |
| [313_select_tusz_meta_short_grid_v1.py](../scripts/313_select_tusz_meta_short_grid_v1.py) | select tusz meta short grid v1 |
| [314_run_tusz_meta_full_development_v1.py](../scripts/314_run_tusz_meta_full_development_v1.py) | run tusz meta full development v1 |
| [315_evaluate_tusz_ssl_heads_v1.py](../scripts/315_evaluate_tusz_ssl_heads_v1.py) | evaluate tusz ssl heads v1 |
| [316_run_tusz_meta_full_evaluations_v1.py](../scripts/316_run_tusz_meta_full_evaluations_v1.py) | run tusz meta full evaluations v1 |
| [317_analyze_tusz_carry_v1.py](../scripts/317_analyze_tusz_carry_v1.py) | analyze tusz carry v1 |
| [318_run_tusz_cosine_alignment_v1.py](../scripts/318_run_tusz_cosine_alignment_v1.py) | Run the pre-registered TUSZ cosine-alignment Meta ablation sequentially. |
| [319_run_tusz_eval_ssl_alignment_v1.py](../scripts/319_run_tusz_eval_ssl_alignment_v1.py) | Evaluate post-TTT SSL/classification alignment on Train, Dev, and Eval. |
| [320_train_tusz_meta_ttt_v2.py](../scripts/320_train_tusz_meta_ttt_v2.py) | train tusz meta ttt v2 |
| [321_train_tusz_ssl_v2.py](../scripts/321_train_tusz_ssl_v2.py) | train tusz ssl v2 |
| [322_calibrate_tusz_inner_v2.py](../scripts/322_calibrate_tusz_inner_v2.py) | calibrate tusz inner v2 |
| [323_evaluate_tusz_meta_ttt_v2.py](../scripts/323_evaluate_tusz_meta_ttt_v2.py) | evaluate tusz meta ttt v2 |
| [324_benchmark_tusz_meta_ttt_v2.py](../scripts/324_benchmark_tusz_meta_ttt_v2.py) | benchmark tusz meta ttt v2 |
| [325_select_tusz_ssl_v2.py](../scripts/325_select_tusz_ssl_v2.py) | Apply the preregistered SSL health checks without consulting seizure BCE. |
| [326_train_tusz_supervised_control_v2.py](../scripts/326_train_tusz_supervised_control_v2.py) | Continue supervised training with the v2 data weights and no inner update. |
| [327_run_tusz_meta_development_v2.py](../scripts/327_run_tusz_meta_development_v2.py) | Build or execute the 24-condition v2 development queue. |
| [328_analyze_tusz_gradients_v2.py](../scripts/328_analyze_tusz_gradients_v2.py) | Large-sample, state-replayed gradient diagnostics for v2 F and C semantics. |
| [329_package_tusz_nonmeta_ttt_v2.py](../scripts/329_package_tusz_nonmeta_ttt_v2.py) | Package S1 plus a warm-started SSL head for the shared F/C evaluator. |
| [330_bootstrap_tusz_meta_ttt_v2.py](../scripts/330_bootstrap_tusz_meta_ttt_v2.py) | Patient-paired bootstrap across one or more confirmation seeds. |
| [331_run_tusz_ssl_stage_b_v2.py](../scripts/331_run_tusz_ssl_stage_b_v2.py) | Resumable Stage-B queue: nine SSL warmups, selection, and inner calibration. |
| [332_holm_tusz_secondary_v2.py](../scripts/332_holm_tusz_secondary_v2.py) | Apply Holm correction to preregistered secondary paired comparisons. |
| [333_audit_tusz_assets_v2.py](../scripts/333_audit_tusz_assets_v2.py) | Verify reused S1 checkpoints, partition isolation, and v1 cache invariants. |
| [334_select_tusz_meta_v2.py](../scripts/334_select_tusz_meta_v2.py) | Freeze F/C winners from the completed development queue, before formal evaluation. |
| [335_evaluate_tusz_detector_v2.py](../scripts/335_evaluate_tusz_detector_v2.py) | Evaluate S1 or continued-supervision checkpoints without constructing a TTT objective. |
| [336_run_tusz_mechanisms_v2.py](../scripts/336_run_tusz_mechanisms_v2.py) | Run the three preregistered mechanism controls for each F/C winner. |
| [338_run_tusz_formal_v2.py](../scripts/338_run_tusz_formal_v2.py) | Resumable three-seed formal training, Dev calibration, and fixed Eval scoring. |
| [339_compile_tusz_meta_ttt_v2.py](../scripts/339_compile_tusz_meta_ttt_v2.py) | Compile training coverage, event results, and mechanism summaries into deliverables. |
| [340_run_tusz_meta_ttt_v2_pipeline.py](../scripts/340_run_tusz_meta_ttt_v2_pipeline.py) | Single resumable entry point for all v2 stages and final report compilation. |
| [342_verify_tusz_meta_ttt_v2_completion.py](../scripts/342_verify_tusz_meta_ttt_v2_completion.py) | Fail closed unless every scheduled v2 stage and final evidence artifact is complete. |
| [343_benchmark_tusz_parallel_v2.py](../scripts/343_benchmark_tusz_parallel_v2.py) | Measure condition-level GPU concurrency using identical bounded Meta workloads. |
| [344_run_tusz_formal_parallel_v2.py](../scripts/344_run_tusz_formal_parallel_v2.py) | Run the three independent formal seeds concurrently, then aggregate evidence. |
| [345_monitor_tusz_resources_v2.py](../scripts/345_monitor_tusz_resources_v2.py) | Sample real pipeline GPU, CPU, memory, and storage utilization. |
| [346_verify_inner_sync_v2.py](../scripts/346_verify_inner_sync_v2.py) | Paired real-model verification of fewer inner-update host synchronizations. |
| [347_profile_tusz_meta_v2.py](../scripts/347_profile_tusz_meta_v2.py) | Profile real four-update Meta segments without modifying training checkpoints. |
| [348_verify_tusz_prefix_graph_v2.py](../scripts/348_verify_tusz_prefix_graph_v2.py) | Real-model CUDA Graph prefix check, including retained four-step Meta state. |
| [349_analyze_tusz_performance_v2.py](../scripts/349_analyze_tusz_performance_v2.py) | Summarize actual benchmark throughput and CUDA kernel gaps from a trace. |
| [350_probe_tusz_batched_tail_v2.py](../scripts/350_probe_tusz_batched_tail_v2.py) | Isolated throughput/independence probe; not a production training entry point. |
| [352_benchmark_tusz_patient_lanes_v2.py](../scripts/352_benchmark_tusz_patient_lanes_v2.py) | Real-record Mask training benchmark with independent synchronized patient lanes. |
| [353_benchmark_tusz_lane_conditions_v2.py](../scripts/353_benchmark_tusz_lane_conditions_v2.py) | Measure complete-record patient-lane workloads across independent conditions. |
| [354_benchmark_tusz_condition_ensemble_v2.py](../scripts/354_benchmark_tusz_condition_ensemble_v2.py) | Complete-record benchmark: four independent conditions share one frozen prefix. |
| [355_analyze_tusz_lane_balance_v2.py](../scripts/355_analyze_tusz_lane_balance_v2.py) | Train-only workload census and label-independent grouping efficiency. |
| [356_train_tusz_ensemble_v2.py](../scripts/356_train_tusz_ensemble_v2.py) | Full development/formal training with independent F/C models and a shared prefix. |
| [357_verify_tusz_training_resume_v2.py](../scripts/357_verify_tusz_training_resume_v2.py) | Compare uninterrupted and interrupted/resumed real-data training. |
| [358_run_tusz_fast_development_v2.py](../scripts/358_run_tusz_fast_development_v2.py) | Single-GPU seed-3407 queue; start each job with observed GPU telemetry. |
| [359_compare_tusz_deduplicated_checkpoints_v2.py](../scripts/359_compare_tusz_deduplicated_checkpoints_v2.py) | Check complete-record optimizer outcomes after SSL-view deduplication. |
| [359_freeze_tusz_fast_selection_v2.py](../scripts/359_freeze_tusz_fast_selection_v2.py) | Freeze the one-seed fast development ranking before formal Dev/Eval. |
| [360_evaluate_tusz_fast_controls_v2.py](../scripts/360_evaluate_tusz_fast_controls_v2.py) | Complete one-seed internal validation of non-Meta and supervised controls. |
| [360_summarize_tusz_live_throughput_v2.py](../scripts/360_summarize_tusz_live_throughput_v2.py) | Read-only full-record throughput estimate and current patient-group workload. |
| [361_run_tusz_formal_fast_v2.py](../scripts/361_run_tusz_formal_fast_v2.py) | Resume one-seed formal fast training and two-way parallel fixed evaluation. |
| [363_run_tusz_statistics_fast_v2.py](../scripts/363_run_tusz_statistics_fast_v2.py) | Formal non-Meta controls, patient bootstrap, and large replay diagnostics. |
| [364_run_tusz_development_mechanisms_fast_v2.py](../scripts/364_run_tusz_development_mechanisms_fast_v2.py) | Six fixed one-seed development mechanism controls plus fit/val diagnostics. |
| [365_run_tusz_mechanism_companion_v2.py](../scripts/365_run_tusz_mechanism_companion_v2.py) | Launch independent mechanism training alongside detector-open training. |
| [366_compile_tusz_fast_results_v2.py](../scripts/366_compile_tusz_fast_results_v2.py) | Build auditable one-seed tables from persisted evaluations and diagnostics. |
| [367_finalize_tusz_extended_controls_v2.py](../scripts/367_finalize_tusz_extended_controls_v2.py) | Fixed supplementary probe budget, then rebuild auditable final tables. |
| [368_verify_tusz_end_to_end_v2.py](../scripts/368_verify_tusz_end_to_end_v2.py) | Small real-checkpoint acceptance probe for the formal seed-3407 evaluator. |
| [369_verify_tusz_later_truncation_v2.py](../scripts/369_verify_tusz_later_truncation_v2.py) | Check a later truncated Meta segment on the real CBraMod tail. |
| [370_verify_tusz_threshold_lock_v2.py](../scripts/370_verify_tusz_threshold_lock_v2.py) | Verify every formal seed-3407 Eval threshold equals its saved Dev choice. |
| [371_train_tusz_meta_ttt_v3.py](../scripts/371_train_tusz_meta_ttt_v3.py) | Train the four future-mode v3 outer objectives in one shared-prefix ensemble. |
| [375_evaluate_tusz_meta_ttt_v3_development.sh](../scripts/375_evaluate_tusz_meta_ttt_v3_development.sh) | 历史队列/运行入口 |
| [376_summarize_tusz_meta_ttt_v3.py](../scripts/376_summarize_tusz_meta_ttt_v3.py) | Create the fixed-budget v3 development table and exact weighted BCE audit. |
| [377_run_tusz_meta_ttt_v3_mechanisms.sh](../scripts/377_run_tusz_meta_ttt_v3_mechanisms.sh) | 历史队列/运行入口 |
| [378_decompose_tusz_v3_outer_gradients.py](../scripts/378_decompose_tusz_v3_outer_gradients.py) | Exact full/direct/through-inner outer-gradient decomposition on Train diagnostics. |
| [379_run_tusz_v3_supervised_controls.sh](../scripts/379_run_tusz_v3_supervised_controls.sh) | 历史队列/运行入口 |
| [380_audit_tusz_meta_ttt_v3_completion.py](../scripts/380_audit_tusz_meta_ttt_v3_completion.py) | audit tusz meta ttt v3 completion |
| [381_run_tusz_v3_train_fit_mechanisms.sh](../scripts/381_run_tusz_v3_train_fit_mechanisms.sh) | 历史队列/运行入口 |
| [401_rescore_tusz_meta_ttt_v4.py](../scripts/401_rescore_tusz_meta_ttt_v4.py) | rescore tusz meta ttt v4 |
| [410_calibrate_tusz_meta_ttt_v4.py](../scripts/410_calibrate_tusz_meta_ttt_v4.py) | calibrate tusz meta ttt v4 |
| [411_train_tusz_meta_ttt_v4.py](../scripts/411_train_tusz_meta_ttt_v4.py) | Train the three future-mode v4 objectives in one shared-prefix ensemble. |
| [412_evaluate_tusz_meta_ttt_v4.sh](../scripts/412_evaluate_tusz_meta_ttt_v4.sh) | 历史队列/运行入口 |
| [413_bootstrap_tusz_meta_ttt_v4.py](../scripts/413_bootstrap_tusz_meta_ttt_v4.py) | bootstrap tusz meta ttt v4 |
| [414_run_tusz_meta_ttt_v4_mechanisms.sh](../scripts/414_run_tusz_meta_ttt_v4_mechanisms.sh) | 历史队列/运行入口 |
| [415_decompose_tusz_meta_ttt_v4.py](../scripts/415_decompose_tusz_meta_ttt_v4.py) | decompose tusz meta ttt v4 |
| [416_report_tusz_meta_ttt_v4.py](../scripts/416_report_tusz_meta_ttt_v4.py) | report tusz meta ttt v4 |
| [61_run_baseline.py](../scripts/61_run_baseline.py) | run baseline |
| [64_run_conv_baseline.py](../scripts/64_run_conv_baseline.py) | run conv baseline |
