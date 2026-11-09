"""Paired functional accuracy with shared fallback computation, never cached timing."""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import queue
import runpy
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads
from scipy.stats import beta

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBPOSD
from bbnbp.io import source_manifest
from bbnbp.metrics import mcnemar,wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def compose_logicals(failed,stage1_logical,fallback_logical):
    """Only syndrome failures consult fallback; valid wrong-logical outputs remain failures."""
    failed=np.asarray(failed,dtype=bool)
    logical=np.asarray(stage1_logical,dtype=bool).copy()
    fallback=np.asarray(fallback_logical,dtype=bool)
    if failed.shape!=logical.shape or failed.shape[-1]!=len(fallback):raise ValueError("Mismatched paired accuracy shapes")
    return np.where(failed,fallback[None,:],logical)


def regression_upper(k,n,alpha):
    if not 0<=k<=n or n<1 or not 0<alpha<1:raise ValueError("Invalid binomial bound inputs")
    return 1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))


def producer(confirmation,out,index,target,settings,channel,stop):
    torch.set_num_threads(1);set_num_threads(1)
    confirmation,out=Path(confirmation),Path(out)
    c=load_config(confirmation/"config.yaml")
    selection=json.loads((confirmation/"selection.json").read_text())
    controls=json.loads((confirmation/f"controls-p{index}.json").read_text())
    sampler=make_sampler(c,c.noise.p_grid[index])
    names=settings["accuracy_models"]
    b=settings["accuracy_prefetch_batch"]
    sent=0
    def send(value):
        while not stop.is_set():
            try:channel.put(value,timeout=.5);return True
            except queue.Full:pass
        return False
    try:
        with np.load(confirmation/f"cases-p{index}.npz") as saved:initial={key:saved[key] for key in ("syndromes","observables","faults")}
        cached={}
        for name in names:
            actual=controls["best_ordinary"]["name"] if name=="ordinary" else name
            with np.load(confirmation/f"values-p{index}-{actual}.npz") as saved:
                cached[name]=dict(failed=saved["failed"],logical_failed=saved["logical_failed"])
        builder=runpy.run_path(str(confirmation/"models_source.py"))["make_decoders"]
        models=builder(sampler.problem,confirmation/"selected-source.pt",selection["model_design"],controls)
        rng=np.random.default_rng(settings["accuracy_extension_seeds"][index])
        offset=0
        while offset<target and not stop.is_set():
            count=min(b,target-offset)
            if offset<len(initial["syndromes"]):
                count=min(count,len(initial["syndromes"])-offset)
                s,o,e=(initial[key][offset:offset+count] for key in ("syndromes","observables","faults"))
                failed=np.stack([cached[name]["failed"][offset:offset+count] for name in names])
                logical=np.stack([cached[name]["logical_failed"][offset:offset+count] for name in names])
                origin="independent_confirmation_prefix"
            else:
                s,o,e=sampler.sample(count,rng)
                failed,logical=[],[]
                for name in names:
                    correction,conv=models[name].decode_batch(s)
                    np.testing.assert_array_equal(conv,(parity(sampler.problem.H,correction)==s).all(1))
                    failed.append(~conv)
                    logical.append(~conv | (parity(sampler.problem.L,correction)!=o).any(1))
                failed,logical=np.stack(failed),np.stack(logical)
                origin="new_extension_stream"
            np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
            np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
            path=out/f"input-p{index}-offset{offset}.npz"
            np.savez_compressed(path,syndromes=s,observables=o,faults=e,stage1_failed=failed,stage1_logical=logical)
            if not send(dict(path=str(path),offset=offset,count=count,origin=origin)):break
            sent+=count;offset+=count
        send(None)
        (out/f"producer-p{index}.json").write_text(json.dumps(dict(sent=sent,stopped=stop.is_set(),extension_seed=settings["accuracy_extension_seeds"][index]),indent=2)+"\n")
    except BaseException as exc:
        (out/f"producer-p{index}-FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        send(dict(error=f"{type(exc).__name__}: {exc}"))


def assess_point(confirmation,out,index,settings,total_started):
    c=load_config(confirmation/"config.yaml")
    p=c.noise.p_grid[index]
    target=settings["accuracy_targets"][index]
    names=settings["accuracy_models"]
    sampler=make_sampler(c,p)
    osd=c.eval.osd
    fallback=LdpcBPOSD(sampler.problem,osd.bp_iters,osd.ms_scaling,osd.method,osd.order)
    ctx=mp.get_context("spawn");channel=ctx.Queue(maxsize=settings["accuracy_prefetch_depth"]);stop=ctx.Event()
    child=ctx.Process(target=producer,args=(str(confirmation),str(out),index,target,settings,channel,stop))
    child.start()
    processed,executions,internal=0,0,0
    request_counts=np.zeros(len(names),dtype=np.int64)
    logical_parts=[]
    batches=[]
    started=time.perf_counter()
    reason="fixed_target"
    latest=datetime.fromisoformat(settings["accuracy_latest_finish_utc"])
    try:
        while processed<target:
            if (time.perf_counter()-started>=settings["accuracy_point_wall_seconds"][index]
                    or time.perf_counter()-total_started>=settings["accuracy_wall_seconds"]
                    or datetime.now(timezone.utc)>=latest):reason="time_budget";break
            try:item=channel.get(timeout=2)
            except queue.Empty:
                if not child.is_alive():raise RuntimeError("Accuracy producer exited without a complete stream")
                continue
            if item is None:break
            if "error" in item:raise RuntimeError(item["error"])
            with np.load(item["path"]) as saved:
                s,o,failed,stage1=(saved[key] for key in ("syndromes","observables","stage1_failed","stage1_logical"))
            fb_flags=np.zeros(len(s),dtype=bool)
            indices,corrections,internal_flags=[],[],[]
            used=0
            for j in range(len(s)):
                if (time.perf_counter()-started>=settings["accuracy_point_wall_seconds"][index]
                        or time.perf_counter()-total_started>=settings["accuracy_wall_seconds"]
                        or datetime.now(timezone.utc)>=latest):reason="time_budget";break
                if failed[:,j].any():
                    correction=fallback.decode_one(s[j])
                    fb_flags[j]=bool((parity(sampler.problem.L,correction[None,:])[0]!=o[j]).any())
                    indices.append(j);corrections.append(correction);internal_flags.append(fallback.last_osd_executed)
                    executions+=1;internal+=int(fallback.last_osd_executed)
                used+=1
            if used:
                values=compose_logicals(failed[:,:used],stage1[:,:used],fb_flags[:used])
                logical_parts.append(values)
                request_counts+=failed[:,:used].sum(1)
                processed+=used
                path=out/f"fallback-p{index}-offset{item['offset']}.npz"
                matrix=np.stack(corrections) if corrections else np.empty((0,sampler.problem.H.shape[1]),dtype=np.uint8)
                np.savez_compressed(path,source_indices=indices,corrections=matrix,internal_osd=internal_flags,
                    fallback_logical=fb_flags[:used],pipeline_logical=values,used_count=used)
                batches.append(dict(input=item["path"],fallback=str(path),used=used,offset=item["offset"],origin=item["origin"]))
                print(f"accuracy p={p:g}: shots={processed}/{target}; logical={dict(zip(names,map(int,np.concatenate(logical_parts,axis=1).sum(1))))}; unique fallbacks={executions}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            if used<len(s):break
    finally:
        stop.set();child.join(timeout=10)
        if child.is_alive():child.terminate();child.join(timeout=5)
        channel.close()
    if not logical_parts:raise RuntimeError("No accuracy cases completed")
    flags=np.concatenate(logical_parts,axis=1)
    np.savez_compressed(out/f"pipeline-flags-p{index}.npz",**dict(zip(names,flags)))
    complete=processed==target
    points=[]
    for k,name in enumerate(names):
        failures=int(flags[k].sum());lo,hi=wilson(failures,processed)
        points.append(dict(decoder=name,p=p,shots=processed,logical_failures=failures,logical_rate=failures/processed,
            nominal_wilson_lo=lo,nominal_wilson_hi=hi,stage1_fallback_requests=int(request_counts[k])))
    selected=flags[names.index("selected")]
    pairs={}
    for name in ("ordinary","matched_init"):
        baseline=flags[names.index(name)]
        regressions=int((selected & ~baseline).sum())
        pairs[name]=dict(selected_only_failures=regressions,baseline_only_failures=int((~selected & baseline).sum()),
            extra_failure_probability_upper=regression_upper(regressions,processed,settings["alpha"]/4) if complete else None,
            mcnemar=mcnemar(baseline,selected))
    report=dict(p=p,shots=processed,target=target,complete=complete,stopping=reason,models=points,pairs=pairs,
        actual_unique_fallback_executions=executions,actual_internal_osd_executions=internal,
        sum_model_fallback_requests=int(request_counts.sum()),elapsed_s=time.perf_counter()-started,batches=batches,
        scope="Functional paired accuracy, sharing identical deterministic fallback results across methods. No cached timing claim. Bounds apply to completed fixed samples only; no equivalence margin is asserted.")
    (out/f"point-p{index}.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation",type=Path)
    parser.add_argument("--settings",default="configs/m6/cpu_aux_assessment.yaml")
    args=parser.parse_args()
    confirmation=args.confirmation
    if not (confirmation/"summary.json").exists():raise ValueError("Independent Stage1 confirmation must complete first")
    frozen=json.loads((confirmation/"summary.json").read_text())
    if hashlib.sha256((confirmation/"selected-source.pt").read_bytes()).hexdigest()!=frozen["checkpoint_sha256"]:
        raise ValueError("Frozen learned parameters changed")
    if (confirmation/"accuracy_receipt.json").exists():raise ValueError("This fixed accuracy study was already requested")
    settings=yaml.safe_load(Path(args.settings).read_text())
    d=Path("results/m6_aux_accuracy")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    c=load_config(confirmation/"config.yaml");c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    (d/"executed_source.py").write_bytes(Path(__file__).read_bytes())
    (confirmation/"accuracy_receipt.json").write_text(json.dumps(dict(directory=str(d),requested_at_utc=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")
    print(f"Paired auxiliary accuracy: {d}",flush=True)
    started=time.perf_counter()
    points=[]
    try:
        for index in range(len(c.noise.p_grid)):
            points.append(assess_point(confirmation,d,index,settings,started))
        rows=[row for point in points for row in point["models"]]
        with (d/"paired_accuracy.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        report=dict(purpose="paired_auxiliary_functional_accuracy",confirmation_directory=str(confirmation),points=points,
            completed=all(point["complete"] for point in points),elapsed_s=time.perf_counter()-started,
            actual_unique_fallback_executions=sum(point["actual_unique_fallback_executions"] for point in points),
            actual_internal_osd_executions=sum(point["actual_internal_osd_executions"] for point in points),
            alpha=settings["alpha"],regression_bound_family_size=4,
            scope="Auxiliary fixed-target accuracy. Per-model fallback requests are hypothetical independent pipeline calls; actual shared executions are reported separately. No performance claim uses these cached execution times. Main M6/E1 remains incomplete.",
            final_test_data_used=False,gpu_seconds=0,m6_complete=False)
        (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(dict(completed=report["completed"],elapsed_s=report["elapsed_s"],points=[dict(p=p["p"],shots=p["shots"],models=p["models"],pairs=p["pairs"]) for p in points]),indent=2),flush=True)
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise


if __name__=="__main__":main()
