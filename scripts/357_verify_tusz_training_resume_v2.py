"""Compare uninterrupted and interrupted/resumed real-data training."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from bfa.tusz_meta_ttt_v2.launch import run_monitored

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    base=[sys.executable,str(ROOT/'scripts/356_train_tusz_ensemble_v2.py'),'--objective','mask','--epochs','1','--patients-per-batch','2','--maximum-records','48']
    for name,extra in [('continuous',[]),('resumed',['--stop-after-groups','1']),('resumed',['--resume'])]:
        command=base+['--output',str(args.output/name)]+extra
        run_monitored(command,cwd=ROOT,log=args.output/(name+('_resume' if '--resume' in extra else '')+'.log'))
    left=torch.load(args.output/'continuous/last.pt',map_location='cpu',weights_only=False)
    right=torch.load(args.output/'resumed/last.pt',map_location='cpu',weights_only=False)
    errors=[]
    def difference(a,b):
        if isinstance(a,torch.Tensor): return float((a-b).abs().max()) if a.numel() else 0.
        if isinstance(a,dict): return max((difference(a[k],b[k]) for k in a),default=0.)
        if isinstance(a,(tuple,list)): return max((difference(x,y) for x,y in zip(a,b,strict=True)),default=0.)
        if a!=b: raise AssertionError((a,b))
        return 0.
    for i,(a,b) in enumerate(zip(left['conditions'],right['conditions'],strict=True)):
        error=difference(a,b)
        errors.append(error)
        if error>2e-6: raise AssertionError(f'Condition {i} resume error {error}')
    assert left['epoch']==right['epoch']==2
    assert left['next_patient']==right['next_patient']==0
    for a,b in zip(left['histories'],right['histories'],strict=True):
        for k in ['windows','patients','records','updates','accepted_fraction','outer_steps']:
            assert a[-1][k]==b[-1][k],k
    report=dict(status='passed',condition_max_absolute_errors=errors,coverage=left['histories'][0][-1])
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
