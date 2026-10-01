"""Fixed supplementary probe budget, then rebuild auditable final tables."""
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'
RUN=OUT/'runs/meta/formal_fast_seed3407'
LOG=OUT/'logs/extended_controls'


def probe(objective,mode,partition):
    tag=f'formal_fast_{objective}_extended_seed3407'
    summary=OUT/'mechanisms'/mode/f'eds_{tag}'/partition/'summary.json'
    if summary.is_file():return
    reference=OUT/'evaluation/future/eds_formal_fast_mask_eds_seed3407'
    command=[sys.executable,str(ROOT/'scripts/328_analyze_tusz_gradients_v2.py'),'--meta-checkpoint',str(RUN/objective/mode/'eds/epoch_02.pt'),'--probabilities',str(reference/partition/'probabilities.parquet'),'--source-threshold','0.540851937770844','--partition',partition,'--samples-per-group','128','--tag',tag,'--extended-controls']
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/f'{objective}_{mode}_{partition}.log').open('a') as stream:
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    prior=OUT/'queues/development_mechanism_fast_seed3407_v2.json'
    while True:
        state=json.loads(prior.read_text()) if prior.is_file() else {}
        if state.get('status')=='development_mechanisms_and_fit_val_statistics_complete':break
        time.sleep(300)
    queue=OUT/'queues/extended_controls_seed3407_v2.json'
    queue.write_text(json.dumps(dict(status='running',samples_per_group=128,primary_gradient_samples_per_group=1024),indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(probe,obj,mode,partition) for obj in ['mask','band'] for mode in ['future','current'] for partition in ['train','dev','eval']]
        for future in concurrent.futures.as_completed(futures):future.result()
    queue.write_text(json.dumps(dict(status='complete',reports=12,samples_per_group=128,primary_gradient_samples_per_group=1024),indent=2))
    subprocess.run([sys.executable,str(ROOT/'scripts/366_compile_tusz_fast_results_v2.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
