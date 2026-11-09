"""Matched streamed-case factorial for structural/full weights and two learning rates."""

import argparse
import hashlib
import json
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

from bbnbp.checkpoints import checkpoint_payload,save_checkpoint
from bbnbp.config import load_config
from bbnbp.io import source_manifest
from bbnbp.losses import ParityLoss,multi_loss
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import make_model,training_batch_hash

TUNE=runpy.run_path("scripts/m6_cascade_tune.py")


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def validation_context(directory,c):
    result=[]
    for i in range(2):
        metadata=json.loads((directory/f"baseline-selection-p{i}.json").read_text())
        sampler=make_sampler(c,metadata["p"])
        with np.load(directory/f"validation-p{i}.npz") as saved:s,o=saved["syndromes"],saved["observables"]
        with np.load(directory/f"first-p{i}.npz") as saved:first={key:saved[key] for key in saved.files}
        if hashlib.sha256((directory/f"validation-p{i}.npz").read_bytes()).hexdigest()!=metadata["cases_sha256"]:
            raise ValueError("Validation cases no longer match independent tuning provenance")
        result.append(dict(model=make_model(sampler.problem,c),problem=sampler.problem,s=s,o=o,first=first,metadata=metadata))
    return result


def validate(model,context,out,step,export_check,label=None):
    records=[]
    for i,point in enumerate(context):
        other=point["model"]
        with torch.no_grad():
            other.w.copy_(model.w)
            other.alpha.copy_(model.alpha)
        decoder=other.export_cpu()
        mask=point["first"]["failed"]
        s,o=point["s"][mask],point["o"][mask]
        if export_check:
            n=min(64,len(s))
            expected,conv=other.decode_batch(s[:n])
            actual,actual_conv=decoder.decode_batch(s[:n])
            np.testing.assert_array_equal(expected,actual)
            np.testing.assert_array_equal(conv,actual_conv)
            np.testing.assert_array_equal(other.last_iterations,decoder.last_iterations)
        second=TUNE["decode"](decoder,s,o,point["problem"],64)
        values=TUNE["route"](point["first"],second,mask)
        np.testing.assert_array_equal(~values["failed"],(parity(point["problem"].H,values["corrections"])==point["s"]).all(1))
        np.savez_compressed(out/f"validation-{label or f'step{step}'}-p{i}.npz",**values)
        records.append(dict(p=point["metadata"]["p"],seed=point["metadata"]["seed"],shots=len(point["s"]),
            failures=int(values["failed"].sum()),logical_failures=int(values["logical_failed"].sum()),
            wrong_logical_converged=int((~values["failed"] & values["logical_failed"]).sum()),mean_iterations=float(values["iterations"].mean())))
    return dict(step=step,points=records,total_failures=sum(row["failures"] for row in records),
        total_logical_failures=sum(row["logical_failures"] for row in records))


def train(directory,arm):
    settings=yaml.safe_load((directory/"settings.yaml").read_text())
    c=load_config(directory/f"config-{arm}.yaml")
    frozen=json.loads((directory/"frozen_data.json").read_text())
    for name in ("cases-train.npz","batch-plan.npz"):
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=frozen["hashes"][name]:raise ValueError("Shared training data changed")
    out=directory/arm
    out.mkdir(exist_ok=False)
    c.save(out/"config.yaml")
    shutil.copyfile(__file__,out/"executed_source.py")
    write_json(out/"source_manifest.json",source_manifest())
    torch.set_num_threads(1)
    set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    def deadline(signum,frame):raise TimeoutError("Capacity/rate training reached its CPU wall deadline")
    signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,settings["arm_wall_seconds"])
    try:
        with np.load(directory/"cases-train.npz") as saved:s,e=saved["syndromes"],saved["faults"]
        with np.load(directory/"batch-plan.npz") as saved:plan=saved["indices"]
        sampler=make_sampler(c,c.noise.p_grid[0])
        model=make_model(sampler.problem,c)
        context=validation_context(directory,c)
        objective=ParityLoss(sampler.problem,c.train.beta_logical,c.train.runtime.loss_eps,c.train.runtime.loss_floor)
        optimizer=torch.optim.Adam(model.parameters(),lr=c.train.lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=c.train.max_steps,eta_min=c.train.min_lr)
        initial=validate(model,context,out,0,True)
        (out/"validation.jsonl").write_text(json.dumps(initial)+"\n")
        best_score,best_step=initial["total_failures"],0
        save_checkpoint(out/"best.pt",checkpoint_payload(model,c,0))
        step,stopping=0,"max_steps"
        inputs_digest=hashlib.sha256()
        print(f"{arm}: parameters={sum(p.numel() for p in model.parameters())}; initial validation failures={best_score}",flush=True)
        with (out/"training.jsonl").open("w") as log:
            for selected in plan:
                elapsed=time.perf_counter()-started
                if elapsed>=settings["arm_wall_seconds"]-settings["finalization_reserve_seconds"]:
                    stopping="training_budget";break
                remaining=(datetime.fromisoformat(settings["deadline_utc"])-datetime.now(timezone.utc)).total_seconds()
                if remaining<settings["finalization_reserve_seconds"]:
                    stopping="user_deadline";break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                outputs=model(s[selected],readout=c.train.runtime.readout)
                loss=multi_loss(outputs,torch.from_numpy(e[selected]),objective)
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite capacity/rate loss")
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c.train.grad_clip,error_if_nonfinite=True)
                lr=optimizer.param_groups[0]["lr"]
                optimizer.step();scheduler.step()
                step+=1
                digest=training_batch_hash(c.noise.p_grid[0],s[selected],e[selected],np.ones(len(selected)))
                inputs_digest.update(digest.encode())
                row=dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),lr=lr,input_sha256=digest,
                    elapsed_s=time.perf_counter()-started)
                log.write(json.dumps(row)+"\n");log.flush()
                if step%100==0:print(f"{arm} step={step} loss={row['loss']:.5f}; elapsed={row['elapsed_s']:.1f}s",flush=True)
                if step%c.train.val_every==0:
                    current=validate(model,context,out,step,False)
                    with (out/"validation.jsonl").open("a") as stream:stream.write(json.dumps(current)+"\n")
                    if current["total_failures"]<best_score:
                        best_score,best_step=current["total_failures"],step
                        save_checkpoint(out/"best.pt",checkpoint_payload(model,c,step))
                    print(f"{arm} validation step={step}: total failures={current['total_failures']}; best={best_score}@{best_step}",flush=True)
        save_checkpoint(out/"last.pt",checkpoint_payload(model,c,step,training_state=dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())))
        final=validate(model,context,out,step,True,label="final")
        write_json(out/"summary.json",dict(arm=arm,steps=step,completed=step==c.train.max_steps,stopping=stopping,
            parameters=sum(p.numel() for p in model.parameters()),best_step=best_step,best_validation_failures=best_score,
            final_validation=final,elapsed_s=time.perf_counter()-started,training_inputs_sha256=inputs_digest.hexdigest(),
            best_checkpoint_sha256=hashlib.sha256((out/"best.pt").read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256((out/"last.pt").read_bytes()).hexdigest(),confirmation_data_used=False,
            final_test_data_used=False,gpu_seconds=0,m6_complete=False))
    except BaseException as exc:
        (out/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--arm",required=True,choices=("struct_standard","struct_low","full_standard","full_low"))
    args=parser.parse_args()
    train(args.directory,args.arm)
