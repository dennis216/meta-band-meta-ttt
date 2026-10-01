"""Measure complete-record patient-lane workloads across independent conditions."""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--records',type=int,default=48)
    parser.add_argument('--scopes',nargs='+',default=['e','ed','es','eds'])
    parser.add_argument('--mode',choices=['future','current'],default='future')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ensemble',action='store_true')
    parser.add_argument('--conditions-per-process',type=int,default=2)
    parser.add_argument('--objective',choices=['mask','band'],default='mask')
    parser.add_argument('--patients-per-batch',type=int,choices=[1,2,4],default=4)
    parser.add_argument('--max-memory-gib',type=float,default=28)
    parser.add_argument('--compact-lanes',action='store_true')
    parser.add_argument('--joint-modes',action='store_true')
    parser.add_argument('--deduplicate-views',action='store_true')
    parser.add_argument('--patient-bucket-size',type=int,choices=[0,8,16,32],default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    processes=[]
    handles=[]
    started=time.perf_counter()
    result_paths=[]
    jobs=[args.scopes[i:i+args.conditions_per_process] for i in range(0,len(args.scopes),args.conditions_per_process)] if args.ensemble else [[s] for s in args.scopes]
    for i,scopes in enumerate(jobs):
        scope='_'.join(scopes)
        log=(args.output/f'{scope}_{i}.log').open('w')
        handles.append(log)
        if args.ensemble:
            destination=(args.output/f'{scope}_{i}').resolve()
            command=[sys.executable,str(ROOT/'scripts/354_benchmark_tusz_condition_ensemble_v2.py'),'--records',str(args.records),'--mode',args.mode,'--objective',args.objective,'--patients-per-batch',str(args.patients_per_batch),'--scopes',*scopes,'--output',str(destination)]
            if args.compact_lanes: command.append('--compact-lanes')
            if args.joint_modes: command.append('--joint-modes')
            if args.deduplicate_views: command.append('--deduplicate-views')
            command.extend(['--patient-bucket-size',str(args.patient_bucket_size)])
            result_paths.append(destination/'report.json')
        else:
            destination=(args.output/f'{scope}_{i}.json').resolve()
            command=[sys.executable,str(ROOT/'scripts/352_benchmark_tusz_patient_lanes_v2.py'),'--records',str(args.records),'--lanes','4','--mode',args.mode,'--outer-scope',scope,'--output',str(destination)]
            result_paths.append(destination)
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        processes.append(process)
    monitored={p.pid:psutil.Process(p.pid) for p in processes}
    samples=[]
    memory_limit_hit=False
    while any(p.poll() is None for p in processes):
        values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.current.graphics,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        row=dict(zip(['gpu_util_percent','memory_util_percent','memory_mib','power_w','graphics_mhz','temperature_c'],map(float,values),strict=True))
        row['elapsed_s']=time.perf_counter()-started
        row['cpu_system_percent']=psutil.cpu_percent()
        row['cpu_process_percent']=0.
        row['rss_sum_mib']=0.
        row['disk_read_bytes']=0
        row['live_pids']=[]
        for pid,p in monitored.items():
            try:
                if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE:
                    row['live_pids'].append(pid)
                    row['cpu_process_percent']+=p.cpu_percent()
                    row['rss_sum_mib']+=p.memory_info().rss/1024**2
                    row['disk_read_bytes']+=p.io_counters().read_bytes
            except psutil.NoSuchProcess:
                pass
        samples.append(row)
        if row['memory_mib']>args.max_memory_gib*1024:
            memory_limit_hit=True
            for p in processes:
                if p.poll() is None: p.terminate()
            break
        if len(samples)==20:
            print(json.dumps({'phase':'startup_checked','samples':samples,'workload':'real records, independent outer conditions, batched patient lanes'}),flush=True)
        time.sleep(1)
    codes=[p.wait() for p in processes]
    for handle in handles: handle.close()
    elapsed=time.perf_counter()-started
    runs=[]
    if not any(codes):
        runs=[json.loads(path.read_text()) for path in result_paths]
    steady=[r for r in samples if r['elapsed_s']>=20 and len(r['live_pids'])==len(processes)]
    report=dict(status='complete' if not any(codes) and not memory_limit_hit else 'failed',exit_codes=codes,memory_limit_hit=memory_limit_hit,memory_budget_gib=args.max_memory_gib,wall_s=elapsed,runs=runs,
        aggregate_updates_per_s=sum(r['updates'] for r in runs)/elapsed if runs else None,
        peak_memory_mib=max(r['memory_mib'] for r in samples),
        mean_power_w=statistics.fmean(r['power_w'] for r in samples),
        steady_mean_power_w=statistics.fmean(r['power_w'] for r in steady) if steady else None,
        steady_fraction_at_least_480w=sum(r['power_w']>=480 for r in steady)/len(steady) if steady else None,
        samples=samples)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
    if any(codes) or memory_limit_hit: raise SystemExit(1)


if __name__=='__main__':
    main()
