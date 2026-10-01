"""Single-GPU seed-3407 queue; start each job with observed GPU telemetry."""
import argparse
import fcntl
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from bfa.tusz_meta_ttt_v2.launch import run_monitored

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT/'runs/meta/development_fast_v2_1')
    parser.add_argument('--verification',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=2)
    args=parser.parse_args()
    if json.loads(args.verification.read_text()).get('status')!='passed': raise ValueError('Real-data resume verification must pass')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(seed=3407,epochs=args.epochs,status='running',jobs=[])
        stop=threading.Event()
        def telemetry():
            with (args.output/'resources.jsonl').open('a',buffering=1) as stream:
                while not stop.is_set():
                    try:
                        value=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                        stream.write(json.dumps(dict(time=time.time(),gpu=value))+'\n')
                    except Exception as error: stream.write(json.dumps(dict(time=time.time(),error=repr(error)))+'\n')
                    stop.wait(30)
        thread=threading.Thread(target=telemetry,daemon=True)
        thread.start()
        def persist(): (args.output/'queue.json').write_text(json.dumps(state,indent=2))
        try:
            for objective,lanes in [('mask',2),('band',1)]:
                run=args.output/objective
                row=dict(kind='training',objective=objective,status='running',directory=str(run))
                state['jobs'].append(row)
                persist()
                if not (run/'complete.json').exists():
                    command=[sys.executable,str(ROOT/'scripts/356_train_tusz_ensemble_v2.py'),'--objective',objective,'--output',str(run),'--epochs',str(args.epochs),'--patients-per-batch',str(lanes),'--patient-bucket-size','8']
                    if (run/'last.pt').exists(): command.append('--resume')
                    run_monitored(command,cwd=ROOT,log=args.output/'logs'/f'{objective}_train.log')
                complete=json.loads((run/'complete.json').read_text())
                if complete.get('epochs')!=args.epochs or complete['contract']['stage']!='development': raise ValueError('Training completion contract mismatch')
                row['status']='complete'
                persist()
            state['status']='training_complete_evaluation_pending'
            persist()
            print(json.dumps(state),flush=True)
        except Exception as error:
            state.update(status='failed',error=repr(error))
            persist()
            raise
        finally:
            stop.set()
            thread.join(timeout=5)


if __name__=='__main__': main()
