"""Descriptive routing screen on already-used cases; never independent confirmation."""

import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from bbnbp.io import source_manifest
from bbnbp.metrics import mcnemar


def routed_flags(first,second):
    failed=first["failed"] & second["failed"]
    logical=np.where(first["failed"],second["logical_failed"],first["logical_failed"])
    return dict(failed=failed,logical_failed=logical)


def main():
    source=Path("results/m6_fixed_case/20261109T083753363297Z-9fa92ed1")
    models={}
    for name in ("initial","tuned","final"):
        with np.load(source/f"scores-unseen-{name}.npz") as saved:
            models[name]={key:saved[key] for key in ("failed","logical_failed")}
    names={"tuned_then_initial":("tuned","initial"),"tuned_then_learned":("tuned","final"),"initial_then_learned":("initial","final")}
    output=Path("results/m6_cascade_saved_screen")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True,exist_ok=False)
    combined={name:routed_flags(models[first],models[second]) for name,(first,second) in names.items()}
    all_models={**models,**combined}
    report=dict(purpose="Posthoc routing screen on previously inspected cases, not fresh confirmation",source=str(source),
        source_hashes={name:hashlib.sha256((source/f"scores-unseen-{name}.npz").read_bytes()).hexdigest() for name in models},
        models={name:dict(shots=len(v["failed"]),fallbacks=int(v["failed"].sum()),logical_failures=int(v["logical_failed"].sum()),
                         fallback_demand=float(v["failed"].mean())) for name,v in all_models.items()},
        learned_vs_extra_fixed_pass=mcnemar(combined["tuned_then_initial"]["failed"],combined["tuned_then_learned"]["failed"]),
        scope="Inference routing uses first-pass syndrome validity only, including any first-pass wrong-logical convergence. "
              "These are reused development outcomes motivating an approved auxiliary study, with no confirmatory statistical claim. "
              "T=30 and independently tuned second-pass controls are still required.",final_test_data_used=False,m6_complete=False)
    for name,values in combined.items():np.savez_compressed(output/f"flags-{name}.npz",**values)
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,output/"executed_source.py")
    print(output)
    print(json.dumps(report["models"],indent=2))


if __name__=="__main__":main()
