"""Check complete-record optimizer outcomes after SSL-view deduplication."""
import json
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]/'outputs/reports/tusz_meta_ttt_v2/benchmarks'
left=ROOT/'band_joint48_p1/e_ed_es_eds_0'
right=ROOT/'band_joint48_p1_dedupe/e_ed_es_eds_0'
errors={}
for path in sorted(left.glob('*.pt')):
    a=torch.load(path,map_location='cpu',weights_only=False)
    b=torch.load(right/path.name,map_location='cpu',weights_only=False)
    errors[path.stem]={kind:max(float((value-b[kind][name]).abs().max()) for name,value in values.items()) for kind,values in a.items()}
assert len(errors)==8
assert all(v<=2e-6 for row in errors.values() for v in row.values()),errors
report=dict(status='passed',maximum_absolute_errors=errors)
(ROOT/'band_joint48_p1_dedupe/equivalence.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
