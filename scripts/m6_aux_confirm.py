"""One fixed-shot independent confirmation of a frozen auxiliary decoder choice."""

import argparse
import csv
import hashlib
import json
import platform
import runpy
import shutil
import signal
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.checkpoints import read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.cascade import CascadeDecoder
from bbnbp.evaluate import cpu_model_label
from bbnbp.io import append_results,git_sha,source_manifest
from bbnbp.metrics import mcnemar,wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler

HELPERS=runpy.run_path("scripts/m6_cascade_tune.py")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection",type=Path)
    parser.add_argument("--settings",default="configs/m6/cpu_aux_assessment.yaml")
    args=parser.parse_args()
    selection=json.loads((args.selection/"selection.json").read_text())
    settings=yaml.safe_load(Path(args.settings).read_text())
    if not selection["frozen_before_confirmation"]:raise ValueError("Selection was not frozen")
    if (args.selection/"confirmation_receipt.json").exists():raise ValueError("This frozen selection already requested its confirmation")
    c=load_config(args.selection/"config.yaml")
    seeds={c.seed+c.eval.seed_offset+i for i in range(len(c.noise.p_grid))}
    for previous in Path("results").rglob("*.csv"):
        with previous.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("seed") and int(row["seed"]) in seeds:raise ValueError(f"Confirmation seed already used in {previous}")
    d=Path("results/m6_aux_confirmation")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    for name in ("selection.json","models_source.py","neural_extensions_source.py","selected-source.pt","controls-p0.json","controls-p1.json"):
        shutil.copyfile(args.selection/name,d/name)
    assert hashlib.sha256((d/"selected-source.pt").read_bytes()).hexdigest()==selection["checkpoint_sha256"]
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_source.py")
    (args.selection/"confirmation_receipt.json").write_text(json.dumps(dict(directory=str(d),requested_at_utc=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")
    print(f"Independent auxiliary confirmation: {d}",flush=True)
    torch.set_num_threads(1);set_num_threads(1);torch.use_deterministic_algorithms(True)
    module=runpy.run_path(str(d/"models_source.py"))
    started=time.perf_counter()
    def deadline(signum,frame):raise TimeoutError("Auxiliary confirmation reached its CPU deadline")
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,settings["confirmation_wall_seconds"])
    points=[]
    try:
        for i,p in enumerate(c.noise.p_grid):
            sampler=make_sampler(c,p)
            seed=c.seed+c.eval.seed_offset+i
            s,o,e=sampler.sample(c.eval.max_shots,np.random.default_rng(seed))
            np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
            np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
            np.savez_compressed(d/f"cases-p{i}.npz",syndromes=s,observables=o,faults=e)
            controls=json.loads((d/f"controls-p{i}.json").read_text())
            models=module["make_decoders"](sampler.problem,d/"selected-source.pt",selection["model_design"],controls)
            for name,initial in (("selected",False),("matched_init",True)):
                reference=module["neural_model"](d/"selected-source.pt",sampler.problem,selection["model_design"],initial=initial)
                if selection["model_design"]=="cascade15":reference=CascadeDecoder(LdpcBP(sampler.problem,15,controls["cpp"]["first_factor"]),reference)
                expected,conv=reference.decode_batch(s[:64]);actual,actual_conv=models[name].decode_batch(s[:64])
                np.testing.assert_array_equal(expected,actual);np.testing.assert_array_equal(conv,actual_conv)
                np.testing.assert_array_equal(reference.last_iterations,models[name].last_iterations)
            values={}
            rows=[]
            timestamp=datetime.now(timezone.utc).isoformat()
            payload=read_checkpoint(d/"selected-source.pt")
            for name,model in models.items():
                if name=="ordinary":continue
                value=HELPERS["decode"](model,s,o,sampler.problem,c.eval.batch)
                values[name]=value
                np.savez_compressed(d/f"values-p{i}-{name}.npz",**value)
                failed,logical=int(value["failed"].sum()),int(value["logical_failed"].sum())
                lo,hi=wilson(logical,len(s))
                factor=controls["cpp"]["first_factor"]
                if name=="bp30":factor=controls["cpp"]["selected"]["longer"]["scaling"]
                elif name=="native30":factor=controls["native"]["selected"]["native30"]["scaling"]
                elif name in {"selected","matched_init"} and selection["model_design"]!="cascade15":factor=payload["model_config"]["ms_scaling"]
                rows.append(dict(run_id=d.name,git_sha=git_sha(),config_hash=c.config_hash,timestamp=timestamp,code=c.code.name,
                    noise_model=c.noise.model,rounds=c.noise.rounds,basis=c.noise.basis,p=p,decoder=name+"_only",stage1=name,
                    T=model.T,weight_sharing=payload["model_config"]["sharing"] if name in {"selected","matched_init"} else "scalar",
                    osd_method="",osd_order="",ms_scaling=factor,shots=len(s),stage1_failures=failed,osd_calls=0,osd_rate=0,
                    logical_failures=logical,ler=logical/len(s),ler_lo=lo,ler_hi=hi,ler_stage1=logical/len(s),
                    lat_mean_us="",lat_p50_us="",lat_p99_us="",lat_max_us="",lat_threads=1,
                    device=f"CPU {cpu_model_label(c.eval.cpu_model)}; {platform.platform()}",seed=seed))
                print(f"p={p:g} {name}: fallback={failed/len(s):.6f}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            append_results(d/"results.csv",rows)
            ordinary=controls["best_ordinary"]["name"]
            point=dict(p=p,seed=seed,shots=len(s),ordinary_name=ordinary,
                failures={name:int(v["failed"].sum()) for name,v in values.items()},
                wrong_logical_converged={name:int((~v["failed"] & v["logical_failed"]).sum()) for name,v in values.items()},
                primary=mcnemar(values[ordinary]["failed"],values["selected"]["failed"]),
                learning=mcnemar(values["matched_init"]["failed"],values["selected"]["failed"]))
            points.append(point)
        holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
        for family in ("primary","learning"):
            for point,value in zip(points,holm([point[family]["pvalue"] for point in points])):point[family]["holm_pvalue"]=value
        mean_selected=float(np.mean([point["failures"]["selected"]/point["shots"] for point in points]))
        mean_ordinary=float(np.mean([point["failures"][point["ordinary_name"]]/point["shots"] for point in points]))
        mean_initial=float(np.mean([point["failures"]["matched_init"]/point["shots"] for point in points]))
        benefits=sum(point["primary"]["holm_pvalue"]<selection["alpha"] and point["primary"]["first_only"]>point["primary"]["second_only"] for point in points)
        harms=sum(point["primary"]["holm_pvalue"]<selection["alpha"] and point["primary"]["first_only"]<point["primary"]["second_only"] for point in points)
        learning_benefits=sum(point["learning"]["holm_pvalue"]<selection["alpha"] and point["learning"]["first_only"]>point["learning"]["second_only"] for point in points)
        clear=(mean_ordinary-mean_selected>=selection["minimum_mean_gain"] and benefits>0 and harms==0
               and mean_initial>mean_selected and learning_benefits>0)
        report=dict(points=points,selected=selection["selected"],model_design=selection["model_design"],
            checkpoint_sha256=selection["checkpoint_sha256"],mean_selected=mean_selected,mean_ordinary=mean_ordinary,
            mean_matched_init=mean_initial,primary_benefits=benefits,primary_harms=harms,learning_benefits=learning_benefits,
            clear_gain=bool(clear),elapsed_s=time.perf_counter()-started,selection_directory=str(args.selection),
            final_test_data_used=False,gpu_seconds=0,m6_complete=False,
            scope="Fixed-shot independent auxiliary Stage1 confirmation. Two-point primary and initialization families corrected separately. No OSD, logical-equivalence or latency claim.")
        (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({key:report[key] for key in ("mean_selected","mean_ordinary","mean_matched_init","primary_benefits","primary_harms","learning_benefits","clear_gain")},indent=2))
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
