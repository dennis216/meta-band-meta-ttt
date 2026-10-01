"""Freeze the one-seed fast development ranking before formal Dev/Eval."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reports/tusz_meta_ttt_v2'


def rank(summary):
    row = summary['conditions']['adapted']
    if row['threshold'] is None:
        metrics=row['maximum_sensitivity_metrics']
        return [2, -metrics['sensitivity'], metrics['false_alarms_per_hour'], metrics['false_alarm_minutes_per_hour']]
    metrics=row['metrics']
    return [0, metrics['false_alarms_per_hour'], metrics['false_alarm_minutes_per_hour'], -metrics['sensitivity']]


def main():
    rows=[]
    for objective,difficulty in [('mask',5),('band',.5)]:
        for mode in ['future','current']:
            for scope in ['e','ed','es','eds']:
                path=OUT/'evaluation'/mode/f'{scope}_fast_{objective}_{scope}_epoch02'/'development_validation'/'summary.json'
                summary=json.loads(path.read_text())
                rows.append(dict(status='complete',objective=objective,difficulty=difficulty,mode=mode,outer_scope=scope,best_epoch=2,best_rank=rank(summary),development_summary=str(path)))
    queue=OUT/'queues/development_fast_seed3407_v2.json'
    queue.parent.mkdir(parents=True,exist_ok=True)
    queue.write_text(json.dumps(dict(status='complete',seed=3407,runs=rows),indent=2)+'\n')
    choices={'modes':{}}
    for mode in ['future','current']:
        candidates=sorted([r for r in rows if r['mode']==mode and r['outer_scope']=='eds'],key=lambda r:r['best_rank'])
        choices['modes'][mode]=dict(first=candidates[0],second=candidates[1],all_eds_ranking=candidates)
    selection=OUT/'queues/selection_fast_seed3407_v2.json'
    selection.write_text(json.dumps(choices,indent=2)+'\n')
    print(json.dumps(dict(queue=str(queue),selection=str(selection),first={m:choices['modes'][m]['first']['objective'] for m in choices['modes']})))


if __name__=='__main__':main()
