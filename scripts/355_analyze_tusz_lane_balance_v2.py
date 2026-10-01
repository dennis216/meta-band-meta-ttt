"""Train-only workload census and label-independent grouping efficiency."""
import json
import time
from collections import defaultdict
from pathlib import Path

from bfa.tusz_meta_ttt.dataset import filter_inventory_records,load_cached_arrays
from bfa.tusz_meta_ttt_v2.protocol import chunk_rows
from bfa.tusz_meta_ttt_v2.grouping import patient_order,nominal_lane_efficiency

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'outputs/reports/tusz_meta_ttt_v1'
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'


def main():
    started=time.perf_counter()
    fit=set(json.loads((V1/'manifests/development_split.json').read_text())['development_fit'])
    inventory=json.loads((V1/'manifests/records.json').read_text())
    paths=filter_inventory_records([p for p in sorted((V1/'cache/train').rglob('*.npz')) if p.parts[-4] in fit],inventory,partition='train')
    workloads=defaultdict(int)
    hours=0.
    updates={'future':0,'current':0}
    support_windows={'future':0,'current':0}
    for path in paths:
        archive=load_cached_arrays(path)
        chunks=chunk_rows(archive['decision_end_s'])
        windows=len(archive['decision_end_s'])
        workloads[path.parts[-4]]+=windows
        hours+=archive['signal'].shape[-1]/200/3600
        updates['future']+=max(0,len(chunks)-1)
        updates['current']+=len(chunks)
        support_windows['future']+=sum(len(c.rows) for c in chunks[:-1])
        support_windows['current']+=windows
    grouping=[]
    for seed in [3408,3409]:
        for bucket in [0,8,16,32]:
            order=patient_order(workloads,seed=seed,bucket_size=bucket)
            grouping.append(dict(epoch_order_seed=seed,bucket_size=bucket,
                nominal_efficiency_p2=nominal_lane_efficiency(order,workloads,2),
                nominal_efficiency_p4=nominal_lane_efficiency(order,workloads,4)))
    report=dict(partition='train',cohort='development_fit',patients=len(workloads),records=len(paths),signal_hours=hours,prediction_windows=sum(workloads.values()),inner_updates_per_condition_per_epoch=updates,support_windows_per_condition_per_epoch=support_windows,grouping=grouping,elapsed_s=time.perf_counter()-started,
        limitation='Grouping uses window counts only, no seizure labels. Nominal efficiency assumes fixed lanes; it is a scheduling diagnostic, not measured throughput.')
    target=OUT/'benchmarks/development_workload_and_grouping.json'
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
