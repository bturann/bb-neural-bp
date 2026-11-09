"""Bounded fixed-hard-case learning test; unseen data never select a checkpoint."""

import argparse
import csv
import hashlib
import json
import runpy
import shutil
import signal
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.checkpoints import checkpoint_payload, save_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.evaluate import tuned_factor
from bbnbp.io import git_sha, source_manifest
from bbnbp.losses import ParityLoss, multi_loss
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference, make_model, training_batch_hash


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def paired_screen(initial, final, settings):
    """A local diagnostic screen, not an M6 acceptance or equivalence test."""
    tests={key:mcnemar(initial[key],final[key]) for key in ("failed","logical_failed")}
    holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    for key,value in zip(tests,holm([test["pvalue"] for test in tests.values()])):
        tests[key]["holm_pvalue"]=value
    correct_gain=float(initial["logical_failed"].mean()-final["logical_failed"].mean())
    fallback_reduction=float(initial["failed"].mean()-final["failed"].mean())
    useful=(correct_gain>=settings["unseen_min_correct_gain"]
            and fallback_reduction>=settings["unseen_min_fallback_reduction"]
            and all(test["holm_pvalue"]<settings["alpha"] for test in tests.values()))
    return dict(correct_gain=correct_gain,fallback_reduction=fallback_reduction,tests=tests,useful=bool(useful))


def classify(initial_train, final_train, unseen, settings, completed):
    correct_gain=initial_train["logical_failure_rate"]-final_train["logical_failure_rate"]
    loss_reduction=(initial_train["mean_loss"]-final_train["mean_loss"])/initial_train["mean_loss"]
    if not completed:
        decision="budget_limited"
    elif correct_gain>=settings["training_min_correct_gain"]:
        decision="learning_and_local_generalization" if unseen["useful"] else "training_learning_generalization_unconfirmed"
    elif loss_reduction>=settings["loss_min_relative_reduction"]:
        decision="loss_reduction_without_useful_training_decoding"
    else:
        decision="no_useful_learning_under_this_budget"
    return dict(decision=decision,training_correct_gain=correct_gain,training_relative_loss_reduction=loss_reduction,
                training_useful=correct_gain>=settings["training_min_correct_gain"],
                loss_reduced=loss_reduction>=settings["loss_min_relative_reduction"])


def seed_guard(settings, config):
    seeds={settings["training_pool_seed"],settings["unseen_seed"]}
    if len(seeds)!=2 or seeds & {config.seed+10000+i for i in range(5)}:
        raise ValueError("Diagnostic seeds overlap each other or reserved final data")
    checked=0
    for path in Path("results").rglob("*.csv"):
        with path.open(newline="") as stream:
            reader=csv.DictReader(stream)
            if not reader.fieldnames or "seed" not in reader.fieldnames:
                continue
            checked+=1
            for row in reader:
                if row.get("seed") and int(row["seed"]) in seeds:
                    approved=settings.get("validation_source")
                    if approved and int(row["seed"])==settings["unseen_seed"] and path.parent==Path(approved).parent:
                        continue
                    raise ValueError(f"Diagnostic seed previously used in {path}")
    for path in Path("results").rglob("provenance.json"):
        previous=json.loads(path.read_text())
        if seeds & set(previous.get("validation_seeds",[])):
            raise ValueError(f"Diagnostic seed overlaps previous selection in {path}")
    return checked


def hard_training_cases(sampler, initial, settings):
    rng=np.random.default_rng(settings["training_pool_seed"])
    pool={key:[] for key in ("syndromes","observables","faults","failed")}
    selected,seen,drawn=[],set(),0
    while len(selected)<settings["training_cases"] and drawn<settings["training_pool_max_shots"]:
        n=min(settings["training_pool_batch"],settings["training_pool_max_shots"]-drawn)
        s,o,e=sampler.sample(n,rng)
        _,conv=initial.decode_batch(s)
        for j in np.flatnonzero(~conv):
            key=s[j].tobytes()
            if key not in seen and len(selected)<settings["training_cases"]:
                seen.add(key)
                selected.append(drawn+int(j))
        for key,array in zip(pool,(s,o,e,~conv)):
            pool[key].append(array)
        drawn+=n
    if len(selected)!=settings["training_cases"]:
        raise RuntimeError("Could not obtain the configured unique initial-hard cases")
    arrays={key:np.concatenate(value) for key,value in pool.items()}
    indices=np.asarray(selected,dtype=np.int64)
    train={key:arrays[key][indices] for key in ("syndromes","observables","faults")}
    return train,arrays,indices,rng.bit_generator.state


def score(model, cases, objective, config, settings):
    s,o,e=(cases[key] for key in ("syndromes","observables","faults"))
    losses=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(s),config.train.batch):
            outputs=model(s[start:start+config.train.batch],readout=config.train.runtime.readout)
            faults=torch.from_numpy(e[start:start+config.train.batch])
            weights=torch.arange(1,model.T+1,dtype=model.w.dtype)
            per_case=(torch.stack([objective(p,faults) for p in outputs])*weights[:,None]).sum(0)/weights.sum()
            losses.append(per_case.numpy())
    decoder=model.export_cpu()
    n=min(settings["export_check_shots"],len(s))
    expected,conv=model.decode_batch(s[:n])
    actual,actual_conv=decoder.decode_batch(s[:n])
    np.testing.assert_array_equal(expected,actual)
    np.testing.assert_array_equal(conv,actual_conv)
    np.testing.assert_array_equal(model.last_iterations,decoder.last_iterations)
    failed,logical,iterations=_inference(decoder,s,o,model.problem,config.eval.batch)
    values=dict(loss=np.concatenate(losses),failed=failed,logical_failed=logical)
    summary=dict(shots=len(s),mean_loss=float(values["loss"].mean()),failures=int(failed.sum()),
                 fallback_demand=float(failed.mean()),logical_failures=int(logical.sum()),
                 logical_failure_rate=float(logical.mean()),correct_decodes=int((~logical).sum()),
                 mean_iterations=iterations,export_check_shots=n)
    return summary,values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/m6/cpu_fixed_case.yaml")
    parser.add_argument("--settings",default="configs/m6/cpu_fixed_case_diagnostic.yaml")
    parser.add_argument("--output-root",default="results/m6_fixed_case")
    args=parser.parse_args()
    c=load_config(args.config)
    settings=yaml.safe_load(Path(args.settings).read_text())
    if c.train.device!="cpu" or len(c.noise.p_grid)!=1 or c.train.loss!="parity" or settings["checkpoint_rule"]!="fixed_final_update_or_budget_stop":
        raise ValueError("Expected the single-p CPU parity diagnostic with a fixed final checkpoint")
    validation_source=settings.get("validation_source")
    if validation_source:
        metadata=json.loads(Path(validation_source).with_name("selection.json").read_text())
        if (metadata["seed"]!=settings["unseen_seed"] or metadata["p"]!=c.noise.p_grid[0]
                or metadata["shots"]!=settings["unseen_cases"]
                or hashlib.sha256(Path(validation_source).read_bytes()).hexdigest()!=metadata["cases_sha256"]):
            raise ValueError("Declared reused validation data do not match their frozen provenance")
    checked=seed_guard(settings,c)
    d=Path(args.output_root)/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+c.config_hash[:8])
    d.mkdir(parents=True,exist_ok=False)
    c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    write_json(d/"source_manifest.json",source_manifest())
    shutil.copyfile(__file__,d/"executed_source.py")
    print(f"Fixed-case diagnostic: {d}",flush=True)
    previous_threads,previous_determinism=torch.get_num_threads(),torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(c.train.runtime.threads)
    set_num_threads(c.train.runtime.threads)
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    def deadline(signum,frame):
        raise TimeoutError("Fixed-case diagnostic reached its 20-minute CPU wall limit")
    previous_handler=signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds"])
    try:
        p=c.noise.p_grid[0]
        sampler=make_sampler(c,p)
        model=make_model(sampler.problem,c)
        objective=ParityLoss(sampler.problem,c.train.beta_logical,c.train.runtime.loss_eps,c.train.runtime.loss_floor)
        mining_decoder=model.export_cpu()
        if settings.get("mining_decoder","initial_neural")=="tuned_first_pass":
            factor=tuned_factor(c.train.runtime.baseline_table,c,p)
            mining_decoder=LdpcBP(sampler.problem,c.model.T,factor)
        elif settings.get("mining_decoder","initial_neural")!="initial_neural":
            raise ValueError("Unknown fixed-case mining decoder")
        train,pool,indices,pool_state=hard_training_cases(sampler,mining_decoder,settings)
        if validation_source:
            with np.load(validation_source) as saved:s,o,e=(saved[key] for key in ("syndromes","observables","faults"))
            shutil.copyfile(Path(validation_source).with_name("selection.json"),d/"validation_source_metadata.json")
        else:
            s,o,e=sampler.sample(settings["unseen_cases"],np.random.default_rng(settings["unseen_seed"]))
        unseen=dict(syndromes=s,observables=o,faults=e)
        if set(row.tobytes() for row in train["syndromes"]) & set(row.tobytes() for row in s):
            raise ValueError("Unseen set contains a training syndrome")
        for cases in (train,unseen):
            np.testing.assert_array_equal(parity(sampler.problem.H,cases["faults"]),cases["syndromes"])
            np.testing.assert_array_equal(parity(sampler.problem.L,cases["faults"]),cases["observables"])
        np.savez_compressed(d/"training_pool.npz",**pool,selected_indices=indices)
        for name,cases in (("train",train),("unseen",unseen)):
            np.savez_compressed(d/f"cases-{name}.npz",**cases)
        write_json(d/"frozen_data.json",dict(frozen_before_updates=True,training_pool_seed=settings["training_pool_seed"],
            unseen_seed=settings["unseen_seed"],training_pool_shots=len(pool["failed"]),training_pool_rng_state=pool_state,
            prior_csv_seed_sources_checked=checked,
            sha256={name:hashlib.sha256((d/f"cases-{name}.npz").read_bytes()).hexdigest() for name in ("train","unseen")}))
        table=Path(c.train.runtime.baseline_table)
        table_config=load_config(table.with_suffix(".config.yaml"))
        helper=runpy.run_path("scripts/m6_compare_stage1.py")
        helper["validate_tuning_table"](replace(c,noise=replace(c.noise,p_grid=table_config.noise.p_grid)),table)
        shutil.copyfile(table,d/"tuned_scaling.csv")
        shutil.copyfile(table.with_suffix(".config.yaml"),d/"tuned_scaling.config.yaml")
        tuned=LdpcBP(sampler.problem,c.model.T,tuned_factor(table,c,p))
        failed,logical,iterations=_inference(tuned,s,o,sampler.problem,c.eval.batch)
        np.savez_compressed(d/"scores-unseen-tuned.npz",failed=failed,logical_failed=logical)
        tuned_summary=dict(fallback_demand=float(failed.mean()),logical_failure_rate=float(logical.mean()),mean_iterations=iterations)
        optimizer=torch.optim.Adam(model.parameters(),lr=c.train.lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=c.train.max_steps,eta_min=c.train.min_lr)
        rng=np.random.default_rng(np.random.SeedSequence(settings["shuffle_seed_entropy"]))
        initial_train,initial_values=score(model,train,objective,c,settings)
        initial_unseen,initial_unseen_values=score(model,unseen,objective,c,settings)
        np.savez_compressed(d/"scores-train-step0.npz",**initial_values)
        np.savez_compressed(d/"scores-unseen-initial.npz",**initial_unseen_values)
        save_checkpoint(d/"initial.pt",checkpoint_payload(model,c,0))
        evaluations=[dict(step=0,elapsed_s=time.perf_counter()-started,**initial_train)]
        with (d/"training_evaluation.jsonl").open("w") as stream:
            stream.write(json.dumps(evaluations[0])+"\n")
        print(f"Initial: train correct={initial_train['correct_decodes']}/{len(indices)}; unseen fallback={initial_unseen['fallback_demand']:.6f}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
        order=np.empty(0,dtype=np.int64)
        cursor,step=0,0
        batch_indices=[]
        stopping="max_steps"
        train_end=settings["wall_seconds"]-settings["finalization_reserve_seconds"]
        with (d/"training.jsonl").open("w") as log:
            while step<c.train.max_steps:
                if time.perf_counter()-started>=train_end:
                    stopping="training_budget"
                    break
                if cursor>=len(order):
                    order=rng.permutation(len(indices))
                    cursor=0
                selected=order[cursor:cursor+c.train.batch]
                cursor+=len(selected)
                batch_indices.append(selected.copy())
                model.train()
                optimizer.zero_grad(set_to_none=True)
                outputs=model(train["syndromes"][selected],readout=c.train.runtime.readout)
                loss=multi_loss(outputs,torch.from_numpy(train["faults"][selected]),objective)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite fixed-case training loss")
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c.train.grad_clip,error_if_nonfinite=True)
                used_lr=optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                step+=1
                row=dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),lr=used_lr,
                    elapsed_s=time.perf_counter()-started,
                    batch_sha256=training_batch_hash(p,train["syndromes"][selected],train["faults"][selected],np.ones(len(selected))))
                log.write(json.dumps(row,allow_nan=False)+"\n")
                log.flush()
                if step%c.train.runtime.log_every==0:
                    print(f"step={step} loss={row['loss']:.5f}; elapsed={row['elapsed_s']:.1f}s",flush=True)
                if step%c.train.val_every==0 or step==c.train.max_steps:
                    summary,values=score(model,train,objective,c,settings)
                    evaluations.append(dict(step=step,elapsed_s=time.perf_counter()-started,**summary))
                    with (d/"training_evaluation.jsonl").open("a") as stream:
                        stream.write(json.dumps(evaluations[-1])+"\n")
                    np.savez_compressed(d/f"scores-train-step{step}.npz",**values)
                    save_checkpoint(d/f"step{step}.pt",checkpoint_payload(model,c,step))
                    print(f"train step={step}: correct={summary['correct_decodes']}/{len(indices)}, mean_loss={summary['mean_loss']:.5f}",flush=True)
        np.savez_compressed(d/"batch_indices.npz",indices=np.stack(batch_indices))
        save_checkpoint(d/"last.pt",checkpoint_payload(model,c,step,training_state=dict(optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),numpy_state=rng.bit_generator.state,order=torch.from_numpy(order),cursor=cursor)))
        final_train,final_values=score(model,train,objective,c,settings)
        final_unseen,final_unseen_values=score(model,unseen,objective,c,settings)
        np.savez_compressed(d/"scores-train-final.npz",**final_values)
        np.savez_compressed(d/"scores-unseen-final.npz",**final_unseen_values)
        screen=paired_screen(initial_unseen_values,final_unseen_values,settings)
        completed=step==c.train.max_steps
        result=classify(initial_train,final_train,screen,settings,completed)
        report=dict(purpose=settings["purpose"],config_hash=c.config_hash,settings=settings,git_sha=git_sha(),
            initial_train=initial_train,final_train=final_train,initial_unseen=initial_unseen,final_unseen=final_unseen,
            tuned_unseen=tuned_summary,unseen_screen=screen,**result,steps=step,completed_update_budget=completed,stopping=stopping,
            elapsed_s=time.perf_counter()-started,gpu_seconds=0,final_test_data_used=False,m6_complete=False,
            final_checkpoint_sha256=hashlib.sha256((d/"last.pt").read_bytes()).hexdigest(),
            scope=("Reuses declared validation cases for auxiliary neural candidate selection; this is not fresh confirmation. " if validation_source else "")+
                  "Single-seed, single-p fixed-case learning diagnostic. Training cases fail the declared mining decoder; "
                  "unseen data are unfiltered and evaluated only initially/finally. Fixed final checkpoint, no unseen selection. "
                  "Thresholds are practical diagnostic screens, not M6 gates or evidence of equal two-stage LER/latency.")
        write_json(d/"summary.json",report)
        print(json.dumps({key:report[key] for key in ("decision","training_correct_gain","training_relative_loss_reduction","unseen_screen","steps","elapsed_s")},indent=2),flush=True)
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_handler)
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


if __name__=="__main__":
    main()
