"""Six fixed one-seed development mechanism controls plus fit/val diagnostics."""
import concurrent.futures
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'
RUN=OUT/'runs/meta/mechanism_fast_seed3407'
LOG=OUT/'logs/mechanism_fast'


def call(args,name):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/f'{name}.log').open('a') as stream:
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def evaluate(kind,mode):
    tag=f'mechanism_fast_{kind}_seed3407'
    summary=OUT/'evaluation'/mode/f'eds_{tag}'/'development_validation/summary.json'
    if not summary.is_file():
        call(['scripts/323_evaluate_tusz_meta_ttt_v2.py','--meta-checkpoint',RUN/kind/mode/'eds/epoch_02.pt','--partition','train','--cohort','development_validation','--calibrate','--tag',tag],f'evaluate_{kind}_{mode}')


def main():
    prior=OUT/'queues/formal_statistics_seed3407_v2.json'
    while True:
        state=json.loads(prior.read_text()) if prior.is_file() else {}
        if state.get('status')=='formal_controls_bootstrap_and_gradient_statistics_complete_mechanism_training_pending':break
        time.sleep(300)
    queue=OUT/'queues/development_mechanism_fast_seed3407_v2.json'
    def status(phase):queue.write_text(json.dumps(dict(status=phase,seed=3407),indent=2))
    calibration=json.loads((OUT/'calibration/mask_5_seed3407/gradient_calibration.json').read_text())
    rho=calibration['selected_relative_step']
    lr=statistics.median(rho*calibration['block_reference_norms'][str(b)]/calibration['block_gradient_medians'][str(b)] for b in [10,11])
    for kind,extra in [('detector_open',['--detector-freeze-fraction','0']),('fixed_sgd',['--fixed-inner-lr',str(lr)]),('stop_encoder_through_inner',['--stop-encoder-through-inner'])]:
        status('training_'+kind)
        output=RUN/kind
        if not (output/'complete.json').is_file():
            args=['scripts/356_train_tusz_ensemble_v2.py','--objective','mask','--output',output,'--epochs','2','--patients-per-batch','2','--patient-bucket-size','8','--stage','development','--outer-scopes','eds',*extra]
            if (output/'last.pt').is_file():args.append('--resume')
            call(args,'train_'+kind)
    status('mechanism_validation')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(evaluate,kind,mode) for kind in ['detector_open','fixed_sgd','stop_encoder_through_inner'] for mode in ['future','current']]
        for future in concurrent.futures.as_completed(futures):future.result()
    status('development_reference_fit_scores')
    reference=OUT/'runs/meta/development_fast_v2_1/mask/future/eds/epoch_02.pt'
    refroot=OUT/'evaluation/future/eds_fast_mask_eds_epoch02'
    if not (refroot/'development_fit/probabilities.parquet').is_file():
        call(['scripts/323_evaluate_tusz_meta_ttt_v2.py','--meta-checkpoint',reference,'--partition','train','--cohort','development_fit','--thresholds',refroot/'development_validation/summary.json','--tag','fast_mask_eds_epoch02'],'reference_fit_scores')
    source=json.loads((refroot/'development_validation/summary.json').read_text())['conditions']['source_frozen']
    threshold=source['threshold'] if source['threshold'] is not None else source['maximum_sensitivity_threshold']
    status('development_fit_val_large_sample_gradient_replay')
    for objective in ['mask','band']:
        for mode in ['future','current']:
            checkpoint=OUT/'runs/meta/development_fast_v2_1'/objective/mode/'eds/epoch_02.pt'
            tag=f'development_fast_{objective}_seed3407'
            for cohort in ['development_fit','development_validation']:
                summary=OUT/'mechanisms'/mode/f'eds_{tag}'/cohort/'summary.json'
                if summary.is_file():continue
                call(['scripts/328_analyze_tusz_gradients_v2.py','--meta-checkpoint',checkpoint,'--probabilities',refroot/cohort/'probabilities.parquet','--source-threshold',str(threshold),'--partition','train','--cohort',cohort,'--samples-per-group','1024','--tag',tag],f'gradient_{objective}_{mode}_{cohort}')
    status('development_mechanisms_and_fit_val_statistics_complete')


if __name__=='__main__':main()
