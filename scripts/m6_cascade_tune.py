"""Independently tune longer BP and conditional second-pass BP for the approved cascade."""

import argparse
import csv
import hashlib
import json
import runpy
import shutil
import signal
import time
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.evaluate import tuned_factor
from bbnbp.io import git_sha,source_manifest
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def decode(decoder,s,o,problem,batch):
    hats,failed,logical,iterations=[],[],[],[]
    for start in range(0,len(s),batch):
        correction,conv=decoder.decode_batch(s[start:start+batch])
        hats.append(correction)
        failed.append(~conv)
        logical.append(~conv | (parity(problem.L,correction)!=o[start:start+batch]).any(1))
        iterations.append(decoder.last_iterations.copy())
    return dict(corrections=np.concatenate(hats),failed=np.concatenate(failed),logical_failed=np.concatenate(logical),iterations=np.concatenate(iterations))


def route(first,second,mask):
    result={key:value.copy() for key,value in first.items()}
    for key in ("corrections","failed","logical_failed"):
        result[key][mask]=second[key]
    result["iterations"][mask]+=second["iterations"]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/m6/cpu_cascade.yaml")
    parser.add_argument("--settings",default="configs/m6/cpu_cascade_controls.yaml")
    parser.add_argument("--index",type=int,required=True)
    parser.add_argument("--output-root",default="results/m6_cascade_tuning")
    args=parser.parse_args()
    c=load_config(args.config)
    settings=yaml.safe_load(Path(args.settings).read_text())
    p=c.noise.p_grid[args.index]
    seed=c.seed+c.sweep.seed_offset+args.index
    # Reuse of validation data is allowed only inside this explicit tuning family.
    forbidden={c.seed+10000+i for i in range(5)} | {c.seed+c.eval.seed_offset+i for i in range(len(c.noise.p_grid))}
    if seed in forbidden:raise ValueError("Cascade validation overlaps reserved or confirmation streams")
    d=Path(args.output_root)/f"p{args.index}"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+c.config_hash[:8])
    d.mkdir(parents=True,exist_ok=False)
    c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_source.py")
    print(f"Cascade control tuning: {d}",flush=True)
    torch.set_num_threads(settings["threads"])
    started=time.perf_counter()
    def deadline(signum,frame):raise TimeoutError("Cascade control tuning reached its CPU wall deadline")
    signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds"])
    try:
        table=Path(settings["first_pass_table"])
        original=load_config(table.with_suffix(".config.yaml"))
        helper=runpy.run_path("scripts/m6_compare_stage1.py")
        helper["validate_tuning_table"](original,table)
        assert original.code==c.code and original.seed==c.seed
        assert original.noise==replace(c.noise,p_grid=original.noise.p_grid)
        shutil.copyfile(table,d/"first_pass_scaling.csv")
        shutil.copyfile(table.with_suffix(".config.yaml"),d/"first_pass_scaling.config.yaml")
        sampler=make_sampler(c,p)
        s,o,e=sampler.sample(c.sweep.shots,np.random.default_rng(seed))
        np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
        np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
        np.savez_compressed(d/"cases.npz",syndromes=s,observables=o,faults=e)
        factor=tuned_factor(table,c,p)
        first=decode(LdpcBP(sampler.problem,15,factor),s,o,sampler.problem,c.eval.batch)
        np.savez_compressed(d/"first_pass.npz",**first)
        mask=first["failed"]
        records=[]
        for candidate in c.sweep.factors:
            if datetime.now(timezone.utc)>=datetime.fromisoformat(settings["deadline_utc"]):raise TimeoutError("User deadline reached")
            longer=decode(LdpcBP(sampler.problem,30,candidate),s,o,sampler.problem,c.eval.batch)
            second=decode(LdpcBP(sampler.problem,15,candidate),s[mask],o[mask],sampler.problem,c.eval.batch)
            cascade=route(first,second,mask)
            for name,values in (("longer",longer),("two_pass",cascade)):
                np.savez_compressed(d/f"{name}-{candidate:g}.npz",**values)
                records.append(dict(p=p,seed=seed,shots=len(s),kind=name,scaling=candidate,
                    failures=int(values["failed"].sum()),logical_failures=int(values["logical_failed"].sum()),
                    mean_iterations=float(values["iterations"].mean())))
            print(f"p={p:g} factor={candidate:g}: BP30={longer['failed'].mean():.5f}; BP15→BP15={cascade['failed'].mean():.5f}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
        chosen={kind:min([row for row in records if row["kind"]==kind],key=lambda row:(row["failures"],row["scaling"])) for kind in ("longer","two_pass")}
        with (d/"tuning.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        selection=dict(p=p,seed=seed,shots=len(s),first_factor=factor,selected=chosen,grid=records,
            elapsed_s=time.perf_counter()-started,git_sha=git_sha(),config_hash=c.config_hash,
            cases_sha256=hashlib.sha256((d/"cases.npz").read_bytes()).hexdigest(),
            scope="Independent validation tuning of two auxiliary extra-compute controls. No fresh confirmation, OSD, or latency claim.",
            final_test_data_used=False,m6_complete=False)
        (d/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
        print(json.dumps(chosen,indent=2))
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
