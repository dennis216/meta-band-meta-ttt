"""Read-only full-record throughput estimate and current patient-group workload."""
import json
import time
from collections import defaultdict
from pathlib import Path

from bfa.tusz_meta_ttt.dataset import filter_inventory_records,load_cached_arrays
from bfa.tusz_meta_ttt_v2.grouping import patient_order
from bfa.tusz_meta_ttt_v2.protocol import chunk_rows

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'outputs/reports/tusz_meta_ttt_v1'
RUN=ROOT/'outputs/reports/tusz_meta_ttt_v2/runs/meta/development_fast_v2_1'
contract=json.loads((RUN/'mask/contract.json').read_text())
progress=json.loads((RUN/'mask/progress.json').read_text())
rows=[json.loads(line) for line in (RUN/'logs/mask_train.log').read_text().splitlines() if line.strip()]
rows=[row for row in rows if row.get('phase')=='outer_step']
seconds=sum(row['step_s'] for row in rows)
updates=sum(sum(row['conditions']['updates']) for row in rows)
fit=set(json.loads((V1/'manifests/development_split.json').read_text())['development_fit'])
inventory=json.loads((V1/'manifests/records.json').read_text())
paths=filter_inventory_records([p for p in sorted((V1/'cache/train').rglob('*.npz')) if p.parts[-4] in fit],inventory,partition='train')
patients=defaultdict(list)
workloads=defaultdict(int)
for path in paths:
    patients[path.parts[-4]].append(path)
    workloads[path.parts[-4]]+=len(load_cached_arrays(path)['labels'])
order=patient_order(workloads,seed=contract['seed']+progress['epoch'],bucket_size=contract['patient_bucket_size'])
current=order[progress['patients_complete']:progress['patients_complete']+4]
group=[]
for patient in current:
    n_chunks=[]
    for path in patients[patient]: n_chunks.append(len(chunk_rows(load_cached_arrays(path)['decision_end_s'])))
    group.append(dict(patient=patient,records=len(n_chunks),windows=workloads[patient],all_condition_updates=4*sum(max(0,n-1)+n for n in n_chunks)))
report=dict(completed_patients=progress['patients_complete'],completed_records=progress['records_complete'],completed_group_compute_s=seconds,
    aggregate_updates_per_s=updates/seconds,current_group=group,
    group_s_at_previous_throughput=sum(p['all_condition_updates'] for p in group)/(updates/seconds),
    seconds_since_last_group=time.time()-(RUN/'mask/progress.json').stat().st_mtime,
    note='Current-group duration is an estimate using completed groups; no in-record progress is inferred.')
print(json.dumps(report,indent=2))
