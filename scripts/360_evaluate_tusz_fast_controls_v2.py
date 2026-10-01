"""Complete one-seed internal validation of non-Meta and supervised controls."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'
LOG=OUT/'logs/development_controls'


def call(command,name,summary):
    if summary.is_file(): return
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/f'{name}.log').open('w') as stream:
        subprocess.run([sys.executable,*map(str,command)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    for mode in ['future','current']:
        for objective,difficulty in [('band',.5),('mask',5)]:
            name=f'{objective}_{difficulty:g}_seed3407'
            package=OUT/'runs/nonmeta/development_fast'/mode/name/'checkpoint.pt'
            summary=OUT/'evaluation'/mode/f'{name}_fast_nonmeta_{objective}_epoch02'/'development_validation/summary.json'
            call(['scripts/323_evaluate_tusz_meta_ttt_v2.py','--meta-checkpoint',package,'--partition','train','--cohort','development_validation','--calibrate','--tag',f'fast_nonmeta_{objective}_epoch02'],f'nonmeta_{mode}_{objective}',summary)
    for scope in ['e','ed']:
        checkpoint=OUT/'runs/supervised_controls/development'/f'{scope}_seed3407'/'epoch_02.pt'
        summary=OUT/'evaluation/detectors'/f'fast_control_{scope}_seed3407'/'development_validation/summary.json'
        call(['scripts/335_evaluate_tusz_detector_v2.py','--checkpoint',checkpoint,'--partition','train','--cohort','development_validation','--calibrate','--tag',f'fast_control_{scope}_seed3407'],f'supervised_{scope}',summary)
    (LOG/'complete').write_text('complete\n')


if __name__=='__main__':main()
