"""Secondary ordinary BP50/BP100 challenge, preserving the frozen primary result."""

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

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.io import source_manifest
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default="configs/m6/cpu_long_bp_challenge.yaml")
    parser.add_argument("--index",required=True,type=int,choices=(0,1))
    args=parser.parse_args();settings=yaml.safe_load(Path(args.settings).read_text())
    assert not settings["replace_primary"]
    confirmation=Path(settings["confirmation"])
    c=load_config(confirmation/"config.yaml")
    source=Path(settings["validation_sources"][args.index])
    d=Path("results/m6_long_bp_challenge")/f"p{args.index}"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    shutil.copyfile(__file__,d/"executed_source.py")
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    sampler=make_sampler(c,c.noise.p_grid[args.index])
    with np.load(source/"cases.npz") as saved:s,o=saved["syndromes"],saved["observables"]
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    torch.set_num_threads(1)
    started=time.perf_counter()
    def deadline(signum,frame):raise TimeoutError("Longer BP challenge reached its fixed CPU budget")
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds_per_point"])
    try:
        records=[]
        for iterations in settings["iterations"]:
            for factor in c.sweep.factors:
                if datetime.now(timezone.utc)>=datetime.fromisoformat(settings["deadline_utc"]):raise TimeoutError("Long-control selection deadline reached")
                value=decode(LdpcBP(sampler.problem,iterations,factor),s,o,sampler.problem,c.eval.batch)
                np.savez_compressed(d/f"validation-T{iterations}-factor{factor:g}.npz",**value)
                records.append(dict(T=iterations,factor=factor,failures=int(value["failed"].sum()),
                    logical_failures=int(value["logical_failed"].sum()),mean_iterations=float(value["iterations"].mean())))
                print(f"long BP p={c.noise.p_grid[args.index]:g} T={iterations} factor={factor:g}: {records[-1]['failures']}/{len(s)}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
        choices={str(T):min([r for r in records if r["T"]==T],key=lambda r:(r["failures"],r["logical_failures"],r["factor"])) for T in settings["iterations"]}
        selection=dict(choices=choices,grid=records,validation_source=str(source),
            validation_cases_sha256=hashlib.sha256((source/"cases.npz").read_bytes()).hexdigest(),
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),primary_unchanged=True)
        (d/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
        with np.load(confirmation/f"cases-p{args.index}.npz") as saved:s,o=saved["syndromes"],saved["observables"]
        with np.load(confirmation/f"values-p{args.index}-selected.npz") as saved:selected=saved["failed"]
        outcomes=[]
        for T,choice in choices.items():
            value=decode(LdpcBP(sampler.problem,int(T),choice["factor"]),s,o,sampler.problem,c.eval.batch)
            np.testing.assert_array_equal(value["failed"],~(parity(sampler.problem.H,value["corrections"])==s).all(1))
            np.savez_compressed(d/f"confirmation-T{T}.npz",**value)
            outcomes.append(dict(T=int(T),factor=choice["factor"],shots=len(s),failures=int(value["failed"].sum()),
                logical_failures=int(value["logical_failed"].sum()),mean_iterations=float(value["iterations"].mean()),
                secondary_nominal_comparison=mcnemar(value["failed"],selected)))
        report=dict(p=c.noise.p_grid[args.index],confirmation_directory=str(confirmation),outcomes=outcomes,
            elapsed_s=time.perf_counter()-started,scope=settings["scope"],final_test_data_used=False,gpu_seconds=0,m6_complete=False)
        (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
