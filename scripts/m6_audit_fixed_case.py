"""Reproduce fixed-case data, optimization inputs, checkpoint scores and decision."""

import argparse
import csv
import hashlib
import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from bbnbp.checkpoints import load_model,read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.evaluate import tuned_factor
from bbnbp.losses import ParityLoss,multi_loss
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference,make_model,training_batch_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    d=args.directory
    if (d/"audit.json").exists():
        parser.error("Refusing to overwrite audit evidence")
    r=json.loads((d/"summary.json").read_text())
    settings=r["settings"]
    c=load_config(d/"config.yaml")
    source=json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==source["scripts/m6_fixed_case_diagnostic.py"]
    for path,digest in source.items():
        if path.startswith("src/"):
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    helpers=runpy.run_path(str(d/"executed_source.py"))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sampler=make_sampler(c,c.noise.p_grid[0])
    initial=make_model(sampler.problem,c)
    train,pool,indices,state=helpers["hard_training_cases"](sampler,initial.export_cpu(),settings)
    with np.load(d/"training_pool.npz") as saved:
        for key,value in pool.items():np.testing.assert_array_equal(value,saved[key])
        np.testing.assert_array_equal(indices,saved["selected_indices"])
    frozen=json.loads((d/"frozen_data.json").read_text())
    assert state==frozen["training_pool_rng_state"]
    assert frozen["frozen_before_updates"] and pool["failed"][indices].all()
    np.testing.assert_array_equal(parity(sampler.problem.H,pool["faults"]),pool["syndromes"])
    np.testing.assert_array_equal(parity(sampler.problem.L,pool["faults"]),pool["observables"])
    s,o,e=sampler.sample(settings["unseen_cases"],np.random.default_rng(settings["unseen_seed"]))
    unseen=dict(syndromes=s,observables=o,faults=e)
    for name,cases in (("train",train),("unseen",unseen)):
        assert hashlib.sha256((d/f"cases-{name}.npz").read_bytes()).hexdigest()==frozen["sha256"][name]
        with np.load(d/f"cases-{name}.npz") as saved:
            for key,value in cases.items():np.testing.assert_array_equal(value,saved[key])
        np.testing.assert_array_equal(parity(sampler.problem.H,cases["faults"]),cases["syndromes"])
        np.testing.assert_array_equal(parity(sampler.problem.L,cases["faults"]),cases["observables"])
    assert not set(row.tobytes() for row in s) & set(row.tobytes() for row in train["syndromes"])
    table=d/"tuned_scaling.csv"
    original=load_config(table.with_suffix(".config.yaml"))
    runpy.run_path("scripts/m6_compare_stage1.py")["validate_tuning_table"](replace(c,noise=replace(c.noise,p_grid=original.noise.p_grid)),table)
    tuned=LdpcBP(sampler.problem,c.model.T,tuned_factor(table,c,c.noise.p_grid[0]))
    failed,logical,iterations=_inference(tuned,s,o,sampler.problem,c.eval.batch)
    with np.load(d/"scores-unseen-tuned.npz") as saved:
        np.testing.assert_array_equal(failed,saved["failed"])
        np.testing.assert_array_equal(logical,saved["logical_failed"])
    assert r["tuned_unseen"]==dict(fallback_demand=float(failed.mean()),logical_failure_rate=float(logical.mean()),mean_iterations=iterations)
    final_payload=read_checkpoint(d/"last.pt")
    assert hashlib.sha256((d/"last.pt").read_bytes()).hexdigest()==r["final_checkpoint_sha256"]
    assert final_payload["step"]==r["steps"] and final_payload["config_hash"]==c.config_hash==r["config_hash"]
    rows=[json.loads(line) for line in (d/"training.jsonl").read_text().splitlines()]
    rng=np.random.default_rng(np.random.SeedSequence(settings["shuffle_seed_entropy"]))
    order=np.empty(0,dtype=np.int64)
    cursor=0
    with np.load(d/"batch_indices.npz") as saved:
        assert len(saved["indices"])==len(rows)==r["steps"]
        for i,row in enumerate(rows):
            if cursor>=len(order):order,cursor=rng.permutation(len(indices)),0
            selected=order[cursor:cursor+c.train.batch]
            cursor+=len(selected)
            np.testing.assert_array_equal(selected,saved["indices"][i])
            assert row["step"]==i+1
            assert row["batch_sha256"]==training_batch_hash(c.noise.p_grid[0],train["syndromes"][selected],train["faults"][selected],np.ones(len(selected)))
            lr=c.train.min_lr+(c.train.lr-c.train.min_lr)*(1+np.cos(np.pi*i/c.train.max_steps))/2
            np.testing.assert_allclose(row["lr"],lr,rtol=1e-12)
            assert np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"])
    assert rng.bit_generator.state==final_payload["training_state"]["numpy_state"]
    np.testing.assert_array_equal(order,final_payload["training_state"]["order"].numpy())
    assert cursor==final_payload["training_state"]["cursor"]
    objective=ParityLoss(sampler.problem,c.train.beta_logical,c.train.runtime.loss_eps,c.train.runtime.loss_floor)
    first=np.random.default_rng(np.random.SeedSequence(settings["shuffle_seed_entropy"])).permutation(len(indices))[:c.train.batch]
    with torch.no_grad():
        first_loss=multi_loss(initial(train["syndromes"][first]),torch.from_numpy(train["faults"][first]),objective)
    assert float(first_loss)==rows[0]["loss"]
    checked=0
    exports=0
    outcomes={}
    for phase,path in (("initial",d/"initial.pt"),("final",d/"last.pt")):
        model=load_model(path,sampler.problem)
        for name,cases in (("train",train),("unseen",unseen)):
            summary,values=helpers["score"](model,cases,objective,c,settings)
            assert summary==r[f"{phase}_{name}"]
            filename=f"scores-train-{'step0' if phase=='initial' else 'final'}.npz" if name=="train" else f"scores-unseen-{phase}.npz"
            with np.load(d/filename) as saved:
                for key,value in values.items():np.testing.assert_array_equal(value,saved[key])
            outcomes[phase,name]=values
            checked+=len(cases["syndromes"])
            exports+=summary["export_check_shots"]
    evaluations=[json.loads(line) for line in (d/"training_evaluation.jsonl").read_text().splitlines()]
    for row in evaluations[1:]:
        model=load_model(d/f"step{row['step']}.pt",sampler.problem)
        summary,values=helpers["score"](model,train,objective,c,settings)
        assert all(row[key]==value for key,value in summary.items())
        with np.load(d/f"scores-train-step{row['step']}.npz") as saved:
            for key,value in values.items():np.testing.assert_array_equal(value,saved[key])
        checked+=len(indices)
        exports+=summary["export_check_shots"]
    screen=helpers["paired_screen"](outcomes["initial","unseen"],outcomes["final","unseen"],settings)
    assert screen==r["unseen_screen"]
    transitions=json.loads((d/"unseen_transitions.json").read_text())
    a,b=outcomes["initial","unseen"],outcomes["final","unseen"]
    assert transitions["initial_hard_now_correct"]==int((a["failed"] & ~b["logical_failed"]).sum())
    assert transitions["correct_decodes_gained"]==screen["tests"]["logical_failed"]["first_only"]
    assert transitions["correct_decodes_lost"]==screen["tests"]["logical_failed"]["second_only"]
    assert transitions["correct_decodes_gained"]-transitions["correct_decodes_lost"]==r["final_unseen"]["correct_decodes"]-r["initial_unseen"]["correct_decodes"]
    result=helpers["classify"](r["initial_train"],r["final_train"],screen,settings,r["completed_update_budget"])
    assert all(r[key]==value for key,value in result.items())
    assert r["completed_update_budget"]==(r["steps"]==c.train.max_steps)
    assert r["elapsed_s"]<settings["wall_seconds"] and not r["final_test_data_used"] and not r["m6_complete"]
    with (d/"results.csv").open(newline="") as stream:csv_rows=list(csv.DictReader(stream))
    assert len(csv_rows)==5
    for row in csv_rows:
        name=row["stage1"]
        if name=="bp_tuned_unseen":filename="scores-unseen-tuned.npz"
        elif name=="nbp_initial_fixed_train":filename="scores-train-step0.npz"
        elif name=="nbp_final_fixed_train":filename="scores-train-final.npz"
        else:filename=f"scores-unseen-{'initial' if 'initial' in name else 'final'}.npz"
        with np.load(d/filename) as values:
            assert int(row["shots"])==len(values["failed"])
            assert int(row["stage1_failures"])==int(values["failed"].sum())
            assert int(row["logical_failures"])==int(values["logical_failed"].sum())
            assert float(row["ler"])==float(values["logical_failed"].mean())
        assert row["config_hash"]==c.config_hash and int(row["osd_calls"])==float(row["osd_rate"])==0
        assert int(row["seed"])==settings["training_pool_seed" if "fixed_train" in name else "unseen_seed"]
    audit=dict(checks_passed=True,regenerated_pool_cases=len(pool["failed"]),regenerated_train_cases=len(indices),
        regenerated_unseen_cases=len(s),checked_training_updates=len(rows),first_training_loss_reproduced=True,
        replayed_model_cases=checked,exact_export_shots=exports,checkpoint_evaluations_reproduced=len(evaluations),
        checked_csv_rows=len(csv_rows),
        frozen_data_hashes_verified=True,training_only_protocol_verified=True,decision_reproduced=r["decision"],
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    helpers["write_json"](d/"audit.json",audit)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
