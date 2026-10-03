"""Build auditable one-seed tables from persisted evaluations and diagnostics."""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'
REPORT=OUT/'final_fast_seed3407'


def csv_file(name,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (REPORT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def required_artifacts():
    paths=[OUT/'audits/assets_v2.json', OUT/'audits/end_to_end_seed3407_v2.json',
           OUT/'audits/later_truncation_seed3407_v2.json', OUT/'audits/eval_threshold_lock_seed3407_v2.json',
           OUT/'selection/ssl_difficulty_seed3407.json',
           REPORT/'MECHANISM_REPORT.md', REPORT/'VALIDATION_AUDIT.md',
           OUT/'queues/development_fast_seed3407_v2.json',
           OUT/'benchmarks/development_parallel4_resource_sample.json',
           OUT/'benchmarks/parallel_scaling.json']
    for objective,difficulties in [('band',[.25,.5,.75]),('temporal',[2,5,10]),('mask',[3,5,7])]:
        for difficulty in difficulties:
            paths.append(OUT/'runs/ssl/development'/f'{objective}_{difficulty:g}_seed3407/epoch_02.pt')
    for objective,difficulty in [('band',.5),('mask',5)]:
        task=f'{objective}_{difficulty:g}_seed3407'
        paths.append(OUT/'runs/ssl/formal'/task/'epoch_02.pt')
        paths.append(OUT/'calibration/formal'/task/'gradient_calibration.json')
        for mode in ['future','current']:
            paths.append(OUT/'runs/nonmeta/formal_fast'/mode/task/'checkpoint.pt')
            for scope in ['e','ed','es','eds']:
                paths.append(OUT/'runs/meta/development_fast_v2_1'/objective/mode/scope/'epoch_02.pt')
                paths.append(OUT/'evaluation'/mode/f'{scope}_fast_{objective}_{scope}_epoch02/development_validation/summary.json')
    for scope in ['e','ed']:
        paths.append(OUT/'runs/supervised_controls/formal'/f'{scope}_seed3407/epoch_02.pt')
        paths.extend(OUT/'evaluation/detectors'/f'formal_{scope}_seed3407'/part/'summary.json' for part in ['dev','eval'])
    for objective,scopes in [('mask',['e','ed','es','eds']),('band',['eds'])]:
        for mode in ['future','current']:
            for scope in scopes:
                run=f'{scope}_formal_fast_{objective}_{scope}_seed3407'
                paths.extend(OUT/'evaluation'/mode/run/part/'summary.json' for part in ['dev','eval'])
                paths.append(OUT/'runs/meta/formal_fast_seed3407'/objective/mode/scope/'epoch_02.pt')
    for objective,difficulty in [('mask',5),('band',.5)]:
        for mode in ['future','current']:
            run=f'{objective}_{difficulty:g}_seed3407_formal_fast_nonmeta_{objective}_seed3407'
            paths.extend(OUT/'evaluation'/mode/run/part/'summary.json' for part in ['dev','eval'])
            for partition in ['train','dev','eval']:
                for suffix in ['', '_extended']:
                    paths.append(OUT/'mechanisms'/mode/f'eds_formal_fast_{objective}{suffix}_seed3407'/partition/'summary.json')
            for cohort in ['development_fit','development_validation']:
                paths.append(OUT/'mechanisms'/mode/f'eds_development_fast_{objective}_seed3407'/cohort/'summary.json')
    for kind in ['detector_open','fixed_sgd','stop_encoder_through_inner']:
        paths.append(OUT/'runs/meta/mechanism_fast_seed3407'/kind/'complete.json')
        for mode in ['future','current']:
            paths.append(OUT/'evaluation'/mode/f'eds_mechanism_fast_{kind}_seed3407'/'development_validation/summary.json')
    for mode in ['future','current']:
        for objective,scopes in [('mask',['e','ed','es','eds']),('band',['eds'])]:
            for scope in scopes:
                paths.append(OUT/'statistics/bootstrap'/f'formal_fast_{mode}_{objective}_{scope}_seed3407.json')
    paths.append(OUT/'queues/extended_controls_seed3407_v2.json')
    return paths


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    evaluations=[]
    for path in sorted((OUT/'evaluation').rglob('summary.json')):
        if 'formal_fast_' not in str(path) and 'fast_control_' not in str(path) and 'mechanism_fast_' not in str(path) and '/detectors/formal_' not in str(path):continue
        summary=json.loads(path.read_text())
        conditions=summary.get('conditions',{'detector':summary})
        for condition,row in conditions.items():
            metrics=row.get('metrics') or row.get('maximum_sensitivity_metrics') or {}
            evaluations.append(dict(run=path.parent.parent.name,mode=summary.get('mode','detector'),partition=path.parent.name,condition=condition,reachable=row.get('reachable',row.get('reachable_on_dev')),threshold=row.get('threshold'),**metrics,summary=str(path)))
    csv_file('event_results.csv',evaluations)
    gradients=[]
    for path in sorted((OUT/'mechanisms').rglob('summary.json')):
        if '_fast_' not in str(path):continue
        summary=json.loads(path.read_text())
        for group,values in summary['statistics'].items():
            row=dict(run=path.parent.parent.name,mode=summary['mode'],partition=summary['partition'],cohort=summary['cohort'],group=group,samples=summary['sample_counts'][group],summary=str(path))
            for metric in ['raw_gradient_cosine','actual_descent_cosine','actual_delta_bce','ssl_same_delta','ssl_independent_delta','support_to_future_cosine','future_delta_bce','ssl_future_independent_delta','median_delta_logit','replay_probability_max_error']:
                if metric in values:
                    for statistic in ['mean','median']:row[f'{metric}_{statistic}']=values[metric][statistic]
            gradients.append(row)
    csv_file('gradient_results.csv',gradients)
    bootstrap=[]
    paths=[]
    for path in sorted((OUT/'statistics/bootstrap').glob('formal_fast_*seed3407.json')):
        data=json.loads(path.read_text())
        if 'bootstrap' not in data:continue
        if '_mask_eds_' not in path.name:paths.append(path)
        for metric,value in data['bootstrap'].items():
            bootstrap.append(dict(comparison=path.stem,metric=metric,patients=data['patients'],seeds=data['seeds'],replicates=data['replicates'],mean=value['mean'],ci95_low=value['ci95'][0],ci95_high=value['ci95'][1],p=value['two_sided_bootstrap_p'],source=str(path)))
    csv_file('patient_bootstrap.csv',bootstrap)
    if paths:
        subprocess.run([sys.executable,str(ROOT/'scripts/332_holm_tusz_secondary_v2.py'),'--comparisons',*map(str,paths),'--output',str(REPORT/'holm_secondary.json')],check=True)
    artifacts=required_artifacts()
    missing=[str(path) for path in artifacts if not path.is_file()]
    statuses={}
    for name in ['formal_fast_seed3407_v2','formal_statistics_seed3407_v2','development_mechanism_fast_seed3407_v2']:
        path=OUT/'queues'/f'{name}.json'
        if not path.is_file():
            statuses[name]='missing'
            continue
        state=json.loads(path.read_text())
        if not missing and name in ['formal_fast_seed3407_v2','formal_statistics_seed3407_v2']:
            state['status']='complete'
            path.write_text(json.dumps(state,indent=2))
        statuses[name]=state.get('status')
    completion=dict(status='complete_single_seed' if not missing else 'incomplete',seed=3407,required_artifacts=len(artifacts),missing_artifacts=missing,queue_stage_labels=statuses,formal_eval_conditions=10,formal_gradient_reports=12,extended_formal_gradient_reports=12,development_mechanism_conditions=6,development_gradient_reports=8,other_seeds_deferred=True)
    (REPORT/'completion.json').write_text(json.dumps(completion,indent=2))
    text=['This round reuses S1 and existing preprocessing caches with seed 3407. S1 was not retrained; strict sample-by-sample causality from raw EDF to alarms is not claimed. F is sequential detection with idealized zero computation delay; C is retrospective detection after adaptation using the entire current chunk.','', 'Main formal Eval conditions; each threshold was independently fixed on Dev:','', '| Mode | SSL | Condition | Sensitivity | FA/hour | FA time min/hour |','|---|---|---|---:|---:|---:|']
    for row in evaluations:
        if row['partition']=='eval' and row['run'].startswith('eds_formal_fast_'):
            ssl='mask' if '_mask_' in row['run'] else 'band'
            if row.get('sensitivity') is not None:text.append(f"| {row['mode']} | {ssl} | {row['condition']} | {100*row['sensitivity']:.2f}% | {row['false_alarms_per_hour']:.4f} | {row['false_alarm_minutes_per_hour']:.4f} |")
    text+=['','Attribute adaptation gains using paired Frozen/Adapted comparisons from the same Meta checkpoint. Changes relative to the original S1 also include additional training gains; continued-supervision E/ED controls are in event_results.csv. Three-seed confirmation has not run; these results provide single-seed evidence only.','', 'Primary patient bootstrap:','', '| Mode | FA/hour ratio | 95% CI | Sensitivity difference |','|---|---:|---|---:|']
    for mode in ['future','current']:
        p=OUT/'statistics/bootstrap'/f'formal_fast_{mode}_mask_eds_seed3407.json'
        if not p.is_file():continue
        data=json.loads(p.read_text())
        if 'bootstrap' not in data:continue
        point=data['point_effects_by_seed'][0];interval=data['bootstrap']['fa_per_hour_ratio']['ci95']
        text.append(f"| {mode} | {point['fa_per_hour_ratio']:.5f} | [{interval[0]:.5f}, {interval[1]:.5f}] | {100*point['sensitivity_difference']:.3f} pp |")
    text+=['','See VALIDATION_AUDIT.md for individual engineering checks and uncovered cases; see MECHANISM_REPORT.md for mechanism and throughput analysis. Full gradient statistics are in gradient_results.csv and each summary.json. Groups with insufficient samples retain their actual counts without exceeding patient/event sampling limits. Holm-adjusted secondary comparisons are in holm_secondary.json.','',f'Single-seed experiment audit: {completion["status"]}; {len(artifacts)} required artifacts, {len(missing)} missing. Three-seed confirmation remains deferred under the current single-seed scope.']
    (REPORT/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(report=str(REPORT),evaluation_rows=len(evaluations),gradient_rows=len(gradients),bootstrap_rows=len(bootstrap),status=completion['status'],missing=len(missing))))
    if missing:raise FileNotFoundError(f'{len(missing)} required artifacts are missing; see completion.json')


if __name__=='__main__':main()
