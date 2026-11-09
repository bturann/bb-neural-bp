"""Validation-only longer inference for the fixed learned schedule, no retraining."""

import argparse
import hashlib
import json
import runpy
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.io import source_manifest
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default="configs/m6/cpu_extended_unroll_validation.yaml")
    args=parser.parse_args();settings=yaml.safe_load(Path(args.settings).read_text())
    confirmation=Path(settings["confirmation"])
    checkpoint=confirmation/"selected-source.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==settings["source_checkpoint_sha256"]
    c=load_config(confirmation/"config.yaml")
    d=Path("results/m6_extended_unroll_validation")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    (d/"settings.yaml").write_text(yaml.safe_dump(settings));c.save(d/"config.yaml")
    shutil.copyfile(__file__,d/"executed_source.py")
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    extension=runpy.run_path(str(confirmation/"neural_extensions_source.py"))["continued_model"]
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    torch.set_num_threads(1);set_num_threads(1);started=time.perf_counter()
    def deadline(signum,frame):raise TimeoutError("Iteration generalization diagnostic reached CPU budget")
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds"])
    try:
        records=[]
        for index,root in enumerate(map(Path,settings["long_control_sources"])):
            frozen=json.loads((root/"selection.json").read_text())
            source=Path(frozen["validation_source"])
            sampler=make_sampler(c,c.noise.p_grid[index])
            with np.load(source/"cases.npz") as saved:s,o=saved["syndromes"],saved["observables"]
            for T in settings["iterations"]:
                model=extension(checkpoint,sampler.problem,T,settings["extension"]);node=model.export_cpu()
                n=settings["exact_export_cases_per_point"]
                a,conv=model.decode_batch(s[:n]);b,bconv=node.decode_batch(s[:n])
                np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(conv,bconv)
                np.testing.assert_array_equal(model.last_iterations,node.last_iterations)
                value=decode(node,s,o,sampler.problem,c.eval.batch)
                np.testing.assert_array_equal(value["failed"],~(parity(sampler.problem.H,value["corrections"])==s).all(1))
                np.savez_compressed(d/f"values-p{index}-T{T}.npz",**value)
                ordinary=frozen["choices"][str(T)]
                records.append(dict(p=c.noise.p_grid[index],shots=len(s),T=T,learned_failures=int(value["failed"].sum()),
                    learned_logical_failures=int(value["logical_failed"].sum()),mean_iterations=float(value["iterations"].mean()),
                    ordinary_failures=ordinary["failures"],ordinary_factor=ordinary["factor"],exact_export_cases=n))
                print(f"fixed learned T={T} p={c.noise.p_grid[index]:g}: failures={records[-1]['learned_failures']}; ordinary={ordinary['failures']}",flush=True)
        r=dict(records=records,checkpoint_sha256=settings["source_checkpoint_sha256"],additional_optimizer_updates=0,
            elapsed_s=time.perf_counter()-started,scope=settings["scope"],confirmation_data_used=False,final_test_data_used=False,gpu_seconds=0,m6_complete=False)
        (d/"summary.json").write_text(json.dumps(r,indent=2)+"\n")
        print(json.dumps(r,indent=2))
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
