"""Controlled fixed-batch test of retaining initially correct examples during learning."""

import argparse
import csv
import hashlib
import json
import platform
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
from numba import set_num_threads

from bbnbp.checkpoints import checkpoint_payload,load_model,save_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.evaluate import cpu_model_label,tuned_factor
from bbnbp.io import append_results,git_sha,source_manifest
from bbnbp.losses import ParityLoss,multi_loss
from bbnbp.metrics import mcnemar,wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference,make_model,training_batch_hash

FIXED=runpy.run_path("scripts/m6_fixed_case_diagnostic.py")
HELPERS=runpy.run_path("scripts/m6_compare_stage1.py")


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def read_cases(path):
    with np.load(path) as saved:return {key:saved[key] for key in ("syndromes","observables","faults")}


def weighted_objective(outputs,faults,objective,weights):
    mask=weights>0
    if not mask.any():raise ValueError("At least one training case must have positive weight")
    return multi_loss([p[mask] for p in outputs],faults[mask],objective,weights[mask])


def batch_plan(settings,steps):
    rng=np.random.default_rng(np.random.SeedSequence(settings["shuffle_seed_entropy"]))
    nh,ne=settings["hard_cases"],settings["easy_cases"]
    bh,be=settings["hard_per_batch"],settings["easy_per_batch"]
    if nh%bh or ne%be or nh//bh!=ne//be:raise ValueError("Balanced plan requires complete matching epochs")
    plan=[]
    while len(plan)<steps:
        h,e=rng.permutation(nh),rng.permutation(ne)+nh
        for i in range(nh//bh):
            plan.append(np.concatenate([h[i*bh:(i+1)*bh],e[i*be:(i+1)*be]]))
            if len(plan)==steps:break
    return np.stack(plan)


def group_summary(summary,values,settings):
    nh=settings["hard_cases"]
    return dict(summary,hard_correct=int((~values["logical_failed"][:nh]).sum()),
                easy_correct=int((~values["logical_failed"][nh:]).sum()))


def arm_weights(arm,settings,dtype=torch.float32):
    return torch.tensor([1.]*settings["hard_per_batch"]+[settings["easy_loss_weights"][arm]]*settings["easy_per_batch"],dtype=dtype)


def prepare(args,c,settings):
    # Reject reuse of the new sampling streams; the hard training pool is deliberate reuse.
    FIXED["seed_guard"](dict(training_pool_seed=settings["easy_seed"],unseen_seed=settings["unseen_seed"]),c)
    d=Path(args.output_root)/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+c.config_hash[:8])
    d.mkdir(parents=True,exist_ok=False)
    args.directory=d
    c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    write_json(d/"source_manifest.json",source_manifest())
    shutil.copyfile(__file__,d/"executed_source.py")
    print(f"Retention study: {d}",flush=True)
    sampler=make_sampler(c,c.noise.p_grid[0])
    model=make_model(sampler.problem,c)
    hard=read_cases(Path(settings["source_hard_bundle"])/"cases-train.npz")
    assert len(hard["syndromes"])==settings["hard_cases"]
    _,conv=model.export_cpu().decode_batch(hard["syndromes"])
    assert not conv.any()
    rng=np.random.default_rng(settings["easy_seed"])
    selected,seen,drawn=[],set(),0
    pool={key:[] for key in ("syndromes","observables","faults")}
    while len(selected)<settings["easy_cases"] and drawn<settings["easy_pool_max_shots"]:
        n=min(settings["easy_pool_batch"],settings["easy_pool_max_shots"]-drawn)
        s,o,e=sampler.sample(n,rng)
        correction,conv=model.export_cpu().decode_batch(s)
        correct=conv & (parity(sampler.problem.L,correction)==o).all(1)
        for j in np.flatnonzero(correct):
            key=s[j].tobytes()
            if key not in seen and len(selected)<settings["easy_cases"]:
                seen.add(key)
                selected.append(drawn+int(j))
        for key,value in zip(pool,(s,o,e)):pool[key].append(value)
        drawn+=n
    if len(selected)!=settings["easy_cases"]:raise RuntimeError("Could not mine unique initially correct anchors")
    pool={key:np.concatenate(value) for key,value in pool.items()}
    selected=np.asarray(selected,dtype=np.int64)
    train={key:np.concatenate([hard[key],pool[key][selected]]) for key in hard}
    s,o,e=sampler.sample(settings["unseen_cases"],np.random.default_rng(settings["unseen_seed"]))
    unseen=dict(syndromes=s,observables=o,faults=e)
    if set(row.tobytes() for row in s) & set(row.tobytes() for row in train["syndromes"]):raise ValueError("Unseen/training overlap")
    for cases in (train,unseen):
        np.testing.assert_array_equal(parity(sampler.problem.H,cases["faults"]),cases["syndromes"])
        np.testing.assert_array_equal(parity(sampler.problem.L,cases["faults"]),cases["observables"])
    np.savez_compressed(d/"easy_pool.npz",**pool,selected_indices=selected)
    np.savez_compressed(d/"cases-train.npz",**train)
    np.savez_compressed(d/"cases-unseen.npz",**unseen)
    np.savez_compressed(d/"batch_plan.npz",indices=batch_plan(settings,c.train.max_steps))
    write_json(d/"frozen_data.json",dict(frozen_before_updates=True,source_hard_bundle=settings["source_hard_bundle"],
        easy_pool_shots=drawn,easy_pool_rng_state=rng.bit_generator.state,
        hashes={name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in ("cases-train.npz","cases-unseen.npz","batch_plan.npz")}))
    table=Path(c.train.runtime.baseline_table)
    original=load_config(table.with_suffix(".config.yaml"))
    HELPERS["validate_tuning_table"](replace(c,noise=replace(c.noise,p_grid=original.noise.p_grid)),table)
    shutil.copyfile(table,d/"tuned_scaling.csv")
    shutil.copyfile(table.with_suffix(".config.yaml"),d/"tuned_scaling.config.yaml")
    tuned=LdpcBP(sampler.problem,c.model.T,tuned_factor(table,c,c.noise.p_grid[0]))
    failed,logical,iterations=_inference(tuned,s,o,sampler.problem,c.eval.batch)
    np.savez_compressed(d/"scores-tuned.npz",failed=failed,logical_failed=logical)
    objective=ParityLoss(sampler.problem,c.train.beta_logical,c.train.runtime.loss_eps,c.train.runtime.loss_floor)
    train_summary,values=FIXED["score"](model,train,objective,c,settings)
    np.savez_compressed(d/"scores-initial-train.npz",**values)
    train_summary=group_summary(train_summary,values,settings)
    unseen_summary,values=FIXED["score"](model,unseen,objective,c,settings)
    np.savez_compressed(d/"scores-initial-unseen.npz",**values)
    save_checkpoint(d/"initial.pt",checkpoint_payload(model,c,0))
    write_json(d/"initial.json",dict(train=train_summary,unseen=unseen_summary,
        tuned=dict(fallback_demand=float(failed.mean()),logical_failure_rate=float(logical.mean()),mean_iterations=iterations)))
    print(f"Frozen data ready: train hard={train_summary['hard_correct']}, easy={train_summary['easy_correct']}; unseen fallback={unseen_summary['fallback_demand']:.6f}",flush=True)


def train(args,c,settings):
    d=args.directory
    if Path(__file__).read_bytes()!=(d/"executed_source.py").read_bytes():raise ValueError("Study source changed after freezing")
    if not (d/"initial.json").exists():raise ValueError("Shared preparation must complete first")
    arm=args.arm
    out=d/arm
    out.mkdir(exist_ok=False)
    c.save(out/"config.yaml")
    write_json(out/"arm_settings.json",dict(arm=arm,easy_loss_weight=settings["easy_loss_weights"][arm]))
    sampler=make_sampler(c,c.noise.p_grid[0])
    model=load_model(d/"initial.pt",sampler.problem)
    objective=ParityLoss(sampler.problem,c.train.beta_logical,c.train.runtime.loss_eps,c.train.runtime.loss_floor)
    cases,unseen=read_cases(d/"cases-train.npz"),read_cases(d/"cases-unseen.npz")
    with np.load(d/"batch_plan.npz") as saved:plan=saved["indices"]
    optimizer=torch.optim.Adam(model.parameters(),lr=c.train.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=c.train.max_steps,eta_min=c.train.min_lr)
    weights=arm_weights(arm,settings)
    initial=json.loads((d/"initial.json").read_text())["train"]
    (out/"training_evaluation.jsonl").write_text(json.dumps(dict(step=0,**initial))+"\n")
    step,stopping=0,"max_steps"
    started=time.perf_counter()
    with (out/"training.jsonl").open("w") as log:
        for selected in plan:
            if time.perf_counter()-started>=settings["arm_wall_seconds"]-settings["finalization_reserve_seconds"]:
                stopping="training_budget"
                break
            if (datetime.fromisoformat(settings["deadline_utc"])-datetime.now(timezone.utc)).total_seconds()<settings["finalization_reserve_seconds"]:
                stopping="user_deadline"
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs=model(cases["syndromes"][selected],readout=c.train.runtime.readout)
            loss=weighted_objective(outputs,torch.from_numpy(cases["faults"][selected]),objective,weights)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite retention loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c.train.grad_clip,error_if_nonfinite=True)
            lr=optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            step+=1
            row=dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),lr=lr,
                input_sha256=training_batch_hash(c.noise.p_grid[0],cases["syndromes"][selected],cases["faults"][selected],np.ones(len(selected))),
                elapsed_s=time.perf_counter()-started)
            log.write(json.dumps(row)+"\n")
            log.flush()
            if step%c.train.runtime.log_every==0:print(f"{arm} step={step}: loss={row['loss']:.5f}; elapsed={row['elapsed_s']:.1f}s",flush=True)
            if step%c.train.val_every==0:
                summary,values=FIXED["score"](model,cases,objective,c,settings)
                summary=group_summary(summary,values,settings)
                with (out/"training_evaluation.jsonl").open("a") as stream:stream.write(json.dumps(dict(step=step,**summary))+"\n")
                np.savez_compressed(out/f"train-step{step}.npz",**values)
                save_checkpoint(out/f"step{step}.pt",checkpoint_payload(model,c,step))
                print(f"{arm} step={step}: hard correct={summary['hard_correct']}/128; easy correct={summary['easy_correct']}/128",flush=True)
    save_checkpoint(out/"last.pt",checkpoint_payload(model,c,step,training_state=dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())))
    train_summary,values=FIXED["score"](model,cases,objective,c,settings)
    train_summary=group_summary(train_summary,values,settings)
    np.savez_compressed(out/"train-final.npz",**values)
    unseen_summary,values=FIXED["score"](model,unseen,objective,c,settings)
    np.savez_compressed(out/"unseen-final.npz",**values)
    write_json(out/"summary.json",dict(arm=arm,steps=step,completed=step==c.train.max_steps,stopping=stopping,
        train=train_summary,unseen=unseen_summary,elapsed_s=time.perf_counter()-started,
        checkpoint_sha256=hashlib.sha256((out/"last.pt").read_bytes()).hexdigest(),gpu_seconds=0,final_test_data_used=False,m6_complete=False))


def compare(args,c,settings):
    d=args.directory
    if (d/"summary.json").exists() or (d/"results.csv").exists():raise ValueError("Refusing to overwrite comparison evidence")
    arms={name:json.loads((d/name/"summary.json").read_text()) for name in settings["easy_loss_weights"]}
    if not all(arm["completed"] for arm in arms.values()):raise ValueError("Both retention arms must finish their equal budgets")
    histories={name:[json.loads(line) for line in (d/name/"training.jsonl").read_text().splitlines()] for name in arms}
    assert [row["input_sha256"] for row in histories["hard_only"]]==[row["input_sha256"] for row in histories["retain_easy"]]
    arrays={}
    for name,path in (("initial",d/"scores-initial-unseen.npz"),("tuned",d/"scores-tuned.npz"),
                      ("hard_only",d/"hard_only/unseen-final.npz"),("retain_easy",d/"retain_easy/unseen-final.npz")):
        with np.load(path) as saved:arrays[name]={key:saved[key] for key in ("failed","logical_failed")}
    tests=[]
    for comparator in ("hard_only","tuned"):
        for metric in ("failed","logical_failed"):
            test=mcnemar(arrays[comparator][metric],arrays["retain_easy"][metric])
            tests.append(dict(comparator=comparator,metric=metric,gain=float(arrays[comparator][metric].mean()-arrays["retain_easy"][metric].mean()),**test))
    for test,value in zip(tests,HELPERS["holm"]([test["pvalue"] for test in tests])):test["holm_pvalue"]=value
    useful=all(test["gain"]>=settings["min_net_gain"] and test["holm_pvalue"]<settings["alpha"] for test in tests)
    transitions={}
    for name in arms:
        a,b=arrays["initial"]["logical_failed"],arrays[name]["logical_failed"]
        transitions[name]=dict(gained=int((a & ~b).sum()),lost=int((~a & b).sum()))
    rows=[]
    for name,values in arrays.items():
        n=len(values["failed"])
        failed,logical=int(values["failed"].sum()),int(values["logical_failed"].sum())
        lo,hi=wilson(logical,n)
        rows.append(dict(run_id=d.name,git_sha=git_sha(),config_hash=c.config_hash,timestamp=datetime.now(timezone.utc).isoformat(),
            code=c.code.name,noise_model=c.noise.model,rounds=c.noise.rounds,basis=c.noise.basis,p=c.noise.p_grid[0],decoder=name+"_only",
            stage1=name,T=c.model.T,weight_sharing="scalar" if name=="tuned" else c.model.sharing,osd_method="",osd_order="",
            ms_scaling=tuned_factor(d/"tuned_scaling.csv",c,c.noise.p_grid[0]) if name=="tuned" else c.model.ms_scaling,
            shots=n,stage1_failures=failed,osd_calls=0,osd_rate=0,logical_failures=logical,ler=logical/n,ler_lo=lo,ler_hi=hi,
            ler_stage1=logical/n,lat_mean_us="",lat_p50_us="",lat_p99_us="",lat_max_us="",lat_threads=1,
            device=f"CPU {cpu_model_label(c.eval.cpu_model)}; {platform.platform()}",seed=settings["unseen_seed"]))
    append_results(d/"results.csv",rows)
    write_json(d/"summary.json",dict(purpose=settings["purpose"],arms=arms,tests=tests,transitions=transitions,useful=useful,
        matched_training_batches=len(histories["hard_only"]),mean_fallback={name:float(v["failed"].mean()) for name,v in arrays.items()},
        final_test_data_used=False,gpu_seconds=0,m6_complete=False,
        scope="Controlled easy-example loss-weight ablation, fixed final checkpoints, fresh unseen data and a joint four-test Holm family. Single-p single-seed diagnostic; no OSD, latency or main-M6 claim."))
    print(json.dumps(dict(useful=useful,tests=tests,transitions=transitions),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","train","compare"))
    parser.add_argument("--config",default="configs/m6/cpu_retention.yaml")
    parser.add_argument("--settings",default="configs/m6/cpu_retention_study.yaml")
    parser.add_argument("--output-root",default="results/m6_retention")
    parser.add_argument("--directory",type=Path)
    parser.add_argument("--arm",choices=("hard_only","retain_easy"))
    args=parser.parse_args()
    c=load_config(args.config if args.mode=="prepare" else args.directory/"config.yaml")
    settings=yaml.safe_load(Path(args.settings if args.mode=="prepare" else args.directory/"settings.yaml").read_text())
    if c.train.device!="cpu" or c.noise.p_grid!=(.001,):raise ValueError("Expected the fixed-p CPU retention diagnostic")
    torch.set_num_threads(c.train.runtime.threads)
    set_num_threads(c.train.runtime.threads)
    torch.use_deterministic_algorithms(True)
    def deadline(signum,frame):raise TimeoutError("Retention diagnostic reached its CPU wall limit")
    signal.signal(signal.SIGALRM,deadline)
    limit=settings["prepare_wall_seconds"] if args.mode=="prepare" else settings["arm_wall_seconds"]
    signal.setitimer(signal.ITIMER_REAL,limit)
    try:
        {"prepare":prepare,"train":train,"compare":compare}[args.mode](args,c,settings)
    except BaseException as exc:
        if args.directory:
            parent=args.directory/args.arm if args.arm else args.directory
            if parent.exists():(parent/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
