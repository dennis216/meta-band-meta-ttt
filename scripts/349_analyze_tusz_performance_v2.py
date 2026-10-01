"""Summarize actual benchmark throughput and CUDA kernel gaps from a trace."""
import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    events = json.loads(args.trace.read_text())['traceEvents']
    categories = collections.Counter(e.get('cat','') for e in events)
    kernels = [e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    spans = sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
    merged=[]
    for start,end in spans:
        if merged and start<=merged[-1][1]:
            merged[-1][1]=max(end,merged[-1][1])
        else:
            merged.append([start,end])
    active=sum(end-start for start,end in merged)
    duration=spans[-1][1]-spans[0][0] if spans else 0
    top=collections.defaultdict(lambda:[0.,0])
    for event in kernels:
        top[event['name']][0]+=event['dur']
        top[event['name']][1]+=1
    launches=[e for e in events if e.get('cat')=='cuda_runtime' and e.get('ph')=='X' and 'Launch' in e['name']]
    sync=[e for e in events if e.get('cat')=='cuda_runtime' and e.get('ph')=='X' and 'Synchronize' in e['name']]
    report=dict(trace=str(args.trace), categories=dict(categories), kernel_count=len(kernels),
        kernel_union_ms=active/1000, first_to_last_kernel_ms=duration/1000,
        kernel_active_fraction=active/duration if duration else None,
        launch_calls=len(launches), launch_cpu_ms=sum(e['dur'] for e in launches)/1000,
        synchronization_calls=len(sync), synchronization_cpu_ms=sum(e['dur'] for e in sync)/1000,
        top_kernels=[dict(name=name,total_ms=values[0]/1000,calls=values[1]) for name,values in sorted(top.items(),key=lambda v:-v[1][0])[:20]],
        limitation='Profiler adds overhead; kernel-active fraction is for this isolated trace, not production SM occupancy. Overlapping CPU/GPU ranges are not additive.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='top_kernels'},indent=2))


if __name__=='__main__':
    main()
