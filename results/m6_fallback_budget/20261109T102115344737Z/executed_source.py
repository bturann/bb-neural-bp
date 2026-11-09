"""Measure a few fallback calls solely to budget later idle-machine experiments."""

import json
import shutil
import signal
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import yaml

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP,LdpcBPOSD
from bbnbp.evaluate import tuned_factor
from bbnbp.io import source_manifest
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    settings=yaml.safe_load(Path("configs/m6/cpu_fallback_budget.yaml").read_text())
    c=load_config(settings["base_config"])
    d=Path("results/m6_fallback_budget")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_source.py")
    print(f"Fallback budget calibration: {d}",flush=True)
    def deadline(signum,frame):raise TimeoutError("Fallback budget calibration timed out")
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds"])
    points=[]
    try:
        for i,p in enumerate(c.noise.p_grid):
            sampler=make_sampler(c,p)
            seed=c.seed+settings["seed_offset"]+i
            s,o,e=sampler.sample(settings["draw_batch"],np.random.default_rng(seed))
            first=LdpcBP(sampler.problem,15,tuned_factor(c.train.runtime.baseline_table,c,p))
            _,conv=first.decode_batch(s)
            n=settings["warmup_fallbacks"]+settings["measured_fallbacks"]
            chosen=np.flatnonzero(~conv)[:n]
            if len(chosen)!=n:raise RuntimeError("Budget pool lacks enough first-pass failures")
            osd=c.eval.osd
            decoder=LdpcBPOSD(sampler.problem,osd.bp_iters,osd.ms_scaling,osd.method,osd.order)
            times,logical,internal=[],[],[]
            for j,index in enumerate(chosen):
                start=time.perf_counter_ns();correction=decoder.decode_one(s[index]);elapsed=time.perf_counter_ns()-start
                if j>=settings["warmup_fallbacks"]:
                    times.append(elapsed/1e9)
                    logical.append(bool((parity(sampler.problem.L,correction[None,:])[0]!=o[index]).any()))
                    internal.append(decoder.last_osd_executed)
            np.savez_compressed(d/f"calibration-p{i}.npz",syndromes=s,observables=o,faults=e,chosen=chosen,
                measured_seconds=times,logical_flags=logical,internal_osd=internal)
            point=dict(p=p,seed=seed,measured=len(times),mean_s=float(np.mean(times)),max_s=float(np.max(times)),
                       internal_osd_executions=int(np.sum(internal)))
            points.append(point);print(json.dumps(point),flush=True)
        (d/"summary.json").write_text(json.dumps(dict(points=points,settings=settings,
            scope="Loaded-machine fallback budget estimate only, not an idle-machine performance comparison or E2 result.",
            final_test_data_used=False,m6_complete=False),indent=2)+"\n")
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
