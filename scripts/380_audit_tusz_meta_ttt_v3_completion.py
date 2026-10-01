#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

r=Path('outputs/reports/tusz_meta_ttt_v3'); checks={}
for objective in ('mask','band'):
    root=r/'runs/development'/f'{objective}_f'
    checks[f'{objective}_complete']=(root/'complete.json').is_file()
    for condition in ('b0','b1','b2','b3'):
        history=json.loads((root/condition/'history.json').read_text())
        checks[f'{objective}_{condition}_epochs']=[x['epoch'] for x in history]
        checks[f'{objective}_{condition}_windows']=[x['windows'] for x in history]
        checks[f'{objective}_{condition}_checkpoints']=all((root/condition/f'epoch_{epoch:02d}.pt').is_file() for epoch in (1,2))
        checks[f'{objective}_{condition}_evaluation']=(r/'evaluation/future'/f'{condition}_{objective}'/'development_validation'/'summary.json').is_file()
checks['development_rows']=len(pd.read_csv(r/'reports/development_results.csv'))
checks['mechanism_summaries']=len(list((r/'mechanisms/future').glob('*/development_validation/summary.json')))
checks['decompositions']=all((r/'mechanisms'/name).is_file() for name in ('decomposition_mask_b3.json','decomposition_band_b1.json'))
checks['report']=(r/'reports/MECHANISM_REPORT.md').is_file()
failed={key:value for key,value in checks.items() if value is False or value == []}
(r/'reports/completion_audit.json').write_text(json.dumps({'checks':checks,'failed':failed},indent=2))
print(json.dumps({'failed':failed,'checks':len(checks)},indent=2))
