"""Resume one-seed formal fast training and two-way parallel fixed evaluation."""
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'
RUN=OUT/'runs/meta/formal_fast_seed3407'
LOG=OUT/'logs/formal_fast'


def call(args,name):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/f'{name}.log').open('a') as stream:
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def evaluate(objective,mode,scope):
    checkpoint=RUN/objective/mode/scope/'epoch_02.pt'
    tag=f'formal_fast_{objective}_{scope}_seed3407'
    destination=OUT/'evaluation'/mode/f'{scope}_{tag}'
    dev=destination/'dev/summary.json'
    if not dev.is_file():
        call(['scripts/323_evaluate_tusz_meta_ttt_v2.py','--meta-checkpoint',checkpoint,'--partition','dev','--calibrate','--tag',tag],f'evaluate_dev_{objective}_{mode}_{scope}')
    if not (destination/'eval/summary.json').is_file():
        call(['scripts/323_evaluate_tusz_meta_ttt_v2.py','--meta-checkpoint',checkpoint,'--partition','eval','--thresholds',dev,'--tag',tag],f'evaluate_eval_{objective}_{mode}_{scope}')
    return dict(objective=objective,mode=mode,scope=scope,status='complete',dev=str(dev),eval=str(destination/'eval/summary.json'))


def main():
    # An existing Mask trainer owns the GPU. Its atomic completion marker is
    # checked inside this CPU-only scheduler without waking the conversation.
    while not (RUN/'mask/complete.json').is_file():
        if (RUN/'mask/failure.json').is_file():
            raise RuntimeError('Mask trainer has a failure marker; inspect before proceeding')
        time.sleep(60)
    band=RUN/'band'
    if not (band/'complete.json').is_file():
        args=['scripts/356_train_tusz_ensemble_v2.py','--objective','band','--output',band,'--epochs','2','--patients-per-batch','2','--patient-bucket-size','8','--seed','3407','--stage','formal','--outer-scopes','eds']
        if (band/'last.pt').is_file():args.append('--resume')
        call(args,'meta_band_seed3407')
    conditions=[(objective,mode,scope) for objective,scopes in [('mask',['e','ed','es','eds']),('band',['eds'])] for mode in ['future','current'] for scope in scopes]
    queue=OUT/'queues/formal_fast_seed3407_v2.json'
    queue.parent.mkdir(parents=True,exist_ok=True)
    state=dict(status='evaluation_running',seed=3407,completed=[])
    queue.write_text(json.dumps(state,indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(evaluate,*condition) for condition in conditions]
        for future in concurrent.futures.as_completed(futures):
            state['completed'].append(future.result())
            queue.write_text(json.dumps(state,indent=2))
    state['status']='meta_training_and_main_evaluation_complete_mechanisms_pending'
    queue.write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
