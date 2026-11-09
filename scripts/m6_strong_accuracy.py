"""Supplementary functional accuracy for the frozen strong-first-pass cascade."""

import argparse
import csv
import hashlib
import json
import runpy
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP, LdpcBPOSD
from bbnbp.io import source_manifest
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler

BASE=runpy.run_path("scripts/m6_aux_accuracy.py")
compose_logicals=BASE["compose_logicals"]
regression_upper=BASE["regression_upper"]


def make_nodes(study,problem,index):
    frozen=json.loads((study/"selection.json").read_text())
    if frozen.get("iterations")==100:
        return runpy.run_path(str(study/"executed_source.py"))["make_models"](study,problem,index)
    module=runpy.run_path(str(study/"models_source.py"))
    helper=runpy.run_path("scripts/m6_aux_latency.py")
    return helper["strong_decoders"](problem,study/"selected-source.pt",module,study,index)


def assess_point(study,d,index,settings,started):
    c=load_config(study/"config.yaml")
    frozen=json.loads((study/"selection.json").read_text())
    choice=frozen["points"][index]
    cycle100=frozen.get("iterations")==100
    if cycle100!=(settings["study_kind"]=="cycle100"):raise ValueError("Study kind does not match assessment settings")
    sampler=make_sampler(c,choice["p"])
    target=settings["accuracy_targets"][index];names=settings["accuracy_models"]
    if cycle100:
        nodes=make_nodes(study,sampler.problem,index)
        actual_names=dict(ordinary=choice["best_ordinary"],matched_init="matched_init",selected="selected")
    else:
        first=LdpcBP(sampler.problem,100,choice["first"]["factor"])
        module=runpy.run_path(str(study/"models_source.py"))
        neural={name:module["neural_model"](study/"selected-source.pt",sampler.problem,"cycle30",initial=initial).export_cpu()
                for name,initial in (("selected",False),("matched_init",True))}
        ordinary=choice["best_ordinary"]
        control=LdpcBP(sampler.problem,ordinary["T"],ordinary["factor"])
        actual_names=dict(ordinary=ordinary["name"],matched_init="strong_init",selected="strong_cascade")
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    route=runpy.run_path("scripts/m6_cascade_tune.py")["route"]
    spec=c.eval.osd;fallback=LdpcBPOSD(sampler.problem,spec.bp_iters,spec.ms_scaling,spec.method,spec.order)
    with np.load(study/f"cases-p{index}.npz") as saved:initial={key:saved[key] for key in ("syndromes","observables","faults")}
    cached={}
    for name,actual in actual_names.items():
        with np.load(study/f"values-p{index}-{actual}.npz") as saved:cached[name]={key:saved[key] for key in ("failed","logical_failed")}
    rng=np.random.default_rng(settings["accuracy_extension_seeds"][index])
    processed,executions,internal=0,0,0
    requests=np.zeros(len(names),dtype=np.int64);parts=[];batches=[]
    latest=datetime.fromisoformat(settings["accuracy_latest_finish_utc"])
    point_started=time.perf_counter();reason="fixed_target"
    def expired():
        return (time.perf_counter()-started>=settings["accuracy_wall_seconds"] or
                time.perf_counter()-point_started>=settings["accuracy_point_wall_seconds"][index] or
                datetime.now(timezone.utc)>=latest)
    while processed<target:
        if expired():reason="time_budget";break
        count=min(settings["accuracy_batch"],target-processed)
        offset=processed
        if offset<len(initial["syndromes"]):
            count=min(count,len(initial["syndromes"])-offset)
            s,o,e=(initial[key][offset:offset+count] for key in ("syndromes","observables","faults"))
            failed=np.stack([cached[name]["failed"][offset:offset+count] for name in names])
            logical=np.stack([cached[name]["logical_failed"][offset:offset+count] for name in names])
            origin="independent_confirmation_prefix"
        else:
            s,o,e=sampler.sample(count,rng)
            if cycle100:
                values={name:decode(nodes[name],s,o,sampler.problem,c.eval.batch) for name in names}
            else:
                first_values=decode(first,s,o,sampler.problem,c.eval.batch);mask=first_values["failed"]
                values={name:(route(first_values,decode(node,s[mask],o[mask],sampler.problem,c.eval.batch),mask)
                              if mask.any() else {key:value.copy() for key,value in first_values.items()})
                        for name,node in neural.items()}
                if ordinary["name"]=="bp100":values["ordinary"]=first_values
                elif ordinary["name"]=="bp130":values["ordinary"]=decode(control,s,o,sampler.problem,c.eval.batch)
                else:values["ordinary"]=(route(first_values,decode(control,s[mask],o[mask],sampler.problem,c.eval.batch),mask)
                                          if mask.any() else first_values)
            failed=np.stack([values[name]["failed"] for name in names]);logical=np.stack([values[name]["logical_failed"] for name in names])
            origin="new_extension_stream"
        np.testing.assert_array_equal(parity(sampler.problem.H,e),s);np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
        input_path=d/f"input-p{index}-offset{offset}.npz"
        np.savez_compressed(input_path,syndromes=s,observables=o,faults=e,stage1_failed=failed,stage1_logical=logical)
        fb_flags=np.zeros(len(s),dtype=bool);indices=[];corrections=[];internal_flags=[];used=0
        for j in range(len(s)):
            if expired():reason="time_budget";break
            if failed[:,j].any():
                correction=fallback.decode_one(s[j])
                np.testing.assert_array_equal(parity(sampler.problem.H,correction[None,:])[0],s[j])
                fb_flags[j]=bool((parity(sampler.problem.L,correction[None,:])[0]!=o[j]).any())
                indices.append(j);corrections.append(correction);internal_flags.append(fallback.last_osd_executed)
                executions+=1;internal+=int(fallback.last_osd_executed)
            used+=1
        if used:
            values=compose_logicals(failed[:,:used],logical[:,:used],fb_flags[:used])
            parts.append(values);requests+=failed[:,:used].sum(1);processed+=used
            fb_path=d/f"fallback-p{index}-offset{offset}.npz"
            matrix=np.stack(corrections) if corrections else np.empty((0,sampler.problem.H.shape[1]),dtype=np.uint8)
            np.savez_compressed(fb_path,source_indices=indices,corrections=matrix,internal_osd=internal_flags,
                fallback_logical=fb_flags[:used],pipeline_logical=values,used_count=used)
            batches.append(dict(input=str(input_path),fallback=str(fb_path),used=used,offset=offset,origin=origin))
            print(f"{settings['study_kind']} accuracy p={choice['p']:g}: {processed}/{target}; logical={np.concatenate(parts,axis=1).sum(1).tolist()}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
        if used<len(s):break
    if not parts:raise RuntimeError("No strong accuracy cases completed")
    flags=np.concatenate(parts,axis=1);np.savez_compressed(d/f"pipeline-flags-p{index}.npz",**dict(zip(names,flags)))
    complete=processed==target;rows=[]
    for k,name in enumerate(names):
        failures=int(flags[k].sum());lo,hi=wilson(failures,processed)
        rows.append(dict(decoder=name,p=choice["p"],shots=processed,logical_failures=failures,logical_rate=failures/processed,
            nominal_wilson_lo=lo,nominal_wilson_hi=hi,stage1_fallback_requests=int(requests[k])))
    selected=flags[names.index("selected")];pairs={}
    for name in ("ordinary","matched_init"):
        baseline=flags[names.index(name)];regressions=int((selected & ~baseline).sum())
        pairs[name]=dict(selected_only_failures=regressions,baseline_only_failures=int((~selected & baseline).sum()),
            extra_failure_probability_upper=regression_upper(regressions,processed,settings["alpha"]/4) if complete else None,
            mcnemar=mcnemar(baseline,selected))
    report=dict(p=choice["p"],shots=processed,target=target,complete=complete,stopping=reason,models=rows,pairs=pairs,
        actual_unique_fallback_executions=executions,actual_internal_osd_executions=internal,
        sum_model_fallback_requests=int(requests.sum()),elapsed_s=time.perf_counter()-point_started,batches=batches,
        scope=settings["scope"])
    (d/f"point-p{index}.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("--timings",nargs=3,type=Path,required=True)
    parser.add_argument("--settings",default="configs/m6/cpu_strong_accuracy.yaml")
    args=parser.parse_args();study=args.study
    for root in args.timings:json.loads((root/"summary.json").read_text())
    report=json.loads((study/"summary.json").read_text())
    assert hashlib.sha256((study/"selected-source.pt").read_bytes()).hexdigest()==report.get("primary_checkpoint_sha256",report.get("checkpoint_sha256"))
    if (study/"accuracy_receipt.json").exists():raise ValueError("This fixed strong accuracy study was already requested")
    settings=yaml.safe_load(Path(args.settings).read_text())
    if datetime.now(timezone.utc)>=datetime.fromisoformat(settings["accuracy_latest_finish_utc"]):raise TimeoutError("Accuracy start is beyond the finalization deadline")
    d=Path(settings.get("accuracy_output_root","results/m6_strong_accuracy"))/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,d/"executed_source.py")
    load_config(study/"config.yaml").save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    (study/"accuracy_receipt.json").write_text(json.dumps(dict(directory=str(d),
        requested_at_utc=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")
    torch.set_num_threads(1);set_num_threads(1);started=time.perf_counter()
    try:
        points=[assess_point(study,d,index,settings,started) for index in range(2)]
        rows=[row for point in points for row in point["models"]]
        with (d/"paired_accuracy.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summary=dict(purpose=settings["purpose"],confirmation_directory=str(study),points=points,
            completed=all(point["complete"] for point in points),elapsed_s=time.perf_counter()-started,
            actual_unique_fallback_executions=sum(point["actual_unique_fallback_executions"] for point in points),
            actual_internal_osd_executions=sum(point["actual_internal_osd_executions"] for point in points),
            alpha=settings["alpha"],regression_bound_family_size=4,scope=settings["scope"],
            final_test_data_used=False,gpu_seconds=0,m6_complete=False)
        (d/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise


if __name__=="__main__":main()
