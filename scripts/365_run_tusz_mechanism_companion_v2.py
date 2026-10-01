"""Launch independent mechanism training alongside detector-open training."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'
parser=argparse.ArgumentParser()
parser.add_argument('--kind',choices=['fixed_sgd','stop_encoder_through_inner'],required=True)
args=parser.parse_args()
calibration=json.loads((OUT/'calibration/mask_5_seed3407/gradient_calibration.json').read_text())
lr=statistics.median(calibration['selected_relative_step']*calibration['block_reference_norms'][str(b)]/calibration['block_gradient_medians'][str(b)] for b in [10,11])
extra=['--fixed-inner-lr',str(lr)] if args.kind=='fixed_sgd' else ['--stop-encoder-through-inner']
output=OUT/'runs/meta/mechanism_fast_seed3407'/args.kind
command=[sys.executable,str(ROOT/'scripts/356_train_tusz_ensemble_v2.py'),'--objective','mask','--output',str(output),'--epochs','2','--patients-per-batch','2','--patient-bucket-size','8','--stage','development','--outer-scopes','eds',*extra]
if (output/'last.pt').is_file():command.append('--resume')
log=OUT/'logs/mechanism_fast'/f'companion_{args.kind}.log'
log.parent.mkdir(parents=True,exist_ok=True)
with log.open('a') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
