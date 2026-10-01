"""Formal non-Meta controls, patient bootstrap, and large replay diagnostics."""
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
SOURCE=ROOT/'outputs/reports/tusz_meta_ttt_v1/runs/supervised/formal/s1_seed3407_check0.25/best.pt'


def call(args,name):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/f'{name}.log').open('a') as stream:
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def nonmeta(objective,difficulty,mode):
    name=f'{objective}_{difficulty:g}_seed3407'
    package=OUT/'runs/nonmeta/formal_fast'/mode/name/'checkpoint.pt'
    if not package.is_file():
        call(['scripts/329_package_tusz_nonmeta_ttt_v2.py','--source',SOURCE,'--objective-checkpoint',OUT/'runs/ssl/formal'/name/'epoch_02.pt','--gradient-calibration',OUT/'calibration/formal'/name/'gradient_calibration.json','--mode',mode,'--output',package],f'package_formal_nonmeta_{objective}_{mode}')
    tag=f'formal_fast_nonmeta_{objective}_seed3407'
    destination=OUT/'evaluation'/mode/f'{name}_{tag}'
    dev=destination/'dev/summary.json'
    if not dev.is_file():
        call(['scripts/323_evaluate_tusz_meta_ttt_v2.py','--meta-checkpoint',package,'--partition','dev','--calibrate','--tag',tag],f'nonmeta_dev_{objective}_{mode}')
    if not (destination/'eval/summary.json').is_file():
        call(['scripts/323_evaluate_tusz_meta_ttt_v2.py','--meta-checkpoint',package,'--partition','eval','--thresholds',dev,'--tag',tag],f'nonmeta_eval_{objective}_{mode}')


def main():
    queue=OUT/'queues/formal_fast_seed3407_v2.json'
    while True:
        state=json.loads(queue.read_text()) if queue.is_file() else {}
        if state.get('status')=='meta_training_and_main_evaluation_complete_mechanisms_pending':break
        time.sleep(300)
    progress=OUT/'queues/formal_statistics_seed3407_v2.json'
    def status(phase):progress.write_text(json.dumps(dict(status=phase,seed=3407),indent=2))
    status('nonmeta_evaluation_running')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(nonmeta,obj,diff,mode) for obj,diff in [('mask',5),('band',.5)] for mode in ['future','current']]
        for future in concurrent.futures.as_completed(futures):future.result()
    status('bootstrap_running')
    for row in state['completed']:
        objective,mode,scope=row['objective'],row['mode'],row['scope']
        summary=json.loads(Path(row['dev']).read_text())
        output=OUT/'statistics/bootstrap'/f'formal_fast_{mode}_{objective}_{scope}_seed3407.json'
        if output.is_file():continue
        if any(summary['conditions'][name]['threshold'] is None for name in ['adapted','meta_frozen']):
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(dict(status='operating_point_unreachable',dev=row['dev']),indent=2))
            continue
        call(['scripts/330_bootstrap_tusz_meta_ttt_v2.py','--evaluations',Path(row['eval']).parent/'probabilities.parquet','--dev-summaries',row['dev'],'--output',output],f'bootstrap_{mode}_{objective}_{scope}')
    status('formal_reference_train_scores')
    reference=RUN/'mask/future/eds/epoch_02.pt'
    tag='formal_fast_mask_eds_seed3407'
    refroot=OUT/'evaluation/future'/f'eds_{tag}'
    if not (refroot/'train/probabilities.parquet').is_file():
        call(['scripts/323_evaluate_tusz_meta_ttt_v2.py','--meta-checkpoint',reference,'--partition','train','--thresholds',refroot/'dev/summary.json','--tag',tag], 'reference_train_scores')
    refsummary=json.loads((refroot/'dev/summary.json').read_text())['conditions']['source_frozen']
    threshold=refsummary['threshold']
    if threshold is None:threshold=refsummary['maximum_sensitivity_threshold']
    status('formal_large_sample_gradient_replay')
    # All conditions use the same S1 reference probabilities and deterministic
    # sample seed. Labels only enter this separate post-hoc analysis.
    for objective in ['mask','band']:
        for mode in ['future','current']:
            checkpoint=RUN/objective/mode/'eds/epoch_02.pt'
            mechanism_tag=f'formal_fast_{objective}_seed3407'
            for partition in ['train','dev','eval']:
                summary=OUT/'mechanisms'/mode/f'eds_{mechanism_tag}'/partition/'summary.json'
                if summary.is_file():continue
                verification=[]
                ownroot=OUT/'evaluation'/mode/f'eds_formal_fast_{objective}_eds_seed3407'
                if partition!='train' or (objective=='mask' and mode=='future'):
                    verification=['--method-probabilities',ownroot/partition/'probabilities.parquet']
                call(['scripts/328_analyze_tusz_gradients_v2.py','--meta-checkpoint',checkpoint,'--probabilities',refroot/partition/'probabilities.parquet','--source-threshold',str(threshold),'--partition',partition,'--samples-per-group','1024','--tag',mechanism_tag,*verification],f'gradient_{mode}_{objective}_{partition}')
    status('formal_controls_bootstrap_and_gradient_statistics_complete_mechanism_training_pending')


if __name__=='__main__':main()
