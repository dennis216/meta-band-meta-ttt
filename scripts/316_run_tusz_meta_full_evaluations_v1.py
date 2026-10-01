#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/reports/tusz_meta_ttt_v1"
SOURCE=OUT/"runs/supervised/development/s1_seed3407_check0.25/best.pt"
SELECTED=(("band",1e-4),("temporal",1e-4),("mask",1e-6),("learned",1e-5))
FROZEN=OUT/"evaluation/band_lr1e-06/online/seed3407/development_validation/probabilities.parquet"

def main():
    state={"started_utc":datetime.now(UTC).isoformat(),"runs":[]}
    state_path=OUT/"evaluation/full_development_state.json"
    for objective,lr in SELECTED:
        checkpoint=OUT/"runs/meta/development/online"/f"{objective}_lr{lr:g}_seed3407_full/epoch_02.pt"
        destination=OUT/"evaluation"/f"{objective}_lr{lr:g}_full/online/seed3407/development_validation"
        row={"objective":objective,"inner_lr":lr,"directory":str(destination)}
        if (destination/"summary.json").is_file():
            row["status"]="existing"
        else:
            command=[sys.executable,str(ROOT/"scripts/304_evaluate_tusz_meta_ttt_v1.py"),"--source",str(SOURCE),"--objective-checkpoint",str(checkpoint),"--partition","train","--cohort","development_validation","--mode","online","--seed","3407","--calibrate","--variant-suffix","full","--frozen-probabilities",str(FROZEN)]
            log=OUT/"logs"/f"eval_full_{objective}_lr{lr:g}.log"
            with log.open("w") as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
            row.update(status="complete" if result.returncode==0 else "failed",returncode=result.returncode)
            if result.returncode:
                state["runs"].append(row); state_path.write_text(json.dumps(state,indent=2)+"\n"); raise SystemExit(result.returncode)
        state["runs"].append(row); state_path.write_text(json.dumps(state,indent=2)+"\n")
    state["completed_utc"]=datetime.now(UTC).isoformat(); state_path.write_text(json.dumps(state,indent=2)+"\n")
if __name__=="__main__":
    main()
