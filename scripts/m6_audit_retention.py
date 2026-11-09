"""Replay frozen retention data, paired optimization inputs and final outcomes."""

import argparse
import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch
import yaml

from bbnbp.checkpoints import load_model
from bbnbp.config import load_config
from bbnbp.losses import ParityLoss
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import make_model,training_batch_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    d=args.directory
    if (d/"audit.json").exists():parser.error("Refusing to overwrite audit evidence")
    r=json.loads((d/"summary.json").read_text())
    settings=yaml.safe_load((d/"settings.yaml").read_text())
    c=load_config(d/"config.yaml")
    helper=runpy.run_path(str(d/"executed_source.py"))
    old=Path(settings["source_hard_bundle"])/"executed_source.py"
    score=runpy.run_path(str(old))["score"]
    manifest=json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_retention.py"]
    assert hashlib.sha256(old.read_bytes()).hexdigest()==manifest["scripts/m6_fixed_case_diagnostic.py"]
    for path,digest in manifest.items():
        if path.startswith("src/"):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sampler=make_sampler(c,c.noise.p_grid[0])
    initial=make_model(sampler.problem,c)
    frozen=json.loads((d/"frozen_data.json").read_text())
    for name,digest in frozen["hashes"].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest
    rng=np.random.default_rng(settings["easy_seed"])
    parts={key:[] for key in ("syndromes","observables","faults")}
    for start in range(0,frozen["easy_pool_shots"],settings["easy_pool_batch"]):
        values=sampler.sample(min(settings["easy_pool_batch"],frozen["easy_pool_shots"]-start),rng)
        for key,value in zip(parts,values):parts[key].append(value)
    pool={key:np.concatenate(value) for key,value in parts.items()}
    correction,conv=initial.export_cpu().decode_batch(pool["syndromes"])
    correct=conv & (parity(sampler.problem.L,correction)==pool["observables"]).all(1)
    selected,seen=[],set()
    for j in np.flatnonzero(correct):
        key=pool["syndromes"][j].tobytes()
        if key not in seen and len(selected)<settings["easy_cases"]:seen.add(key);selected.append(int(j))
    assert rng.bit_generator.state==frozen["easy_pool_rng_state"]
    with np.load(d/"easy_pool.npz") as saved:
        np.testing.assert_array_equal(saved["selected_indices"],selected)
        for key,value in pool.items():np.testing.assert_array_equal(saved[key],value)
    hard=helper["read_cases"](Path(settings["source_hard_bundle"])/"cases-train.npz")
    train={key:np.concatenate([hard[key],pool[key][selected]]) for key in hard}
    s,o,e=sampler.sample(settings["unseen_cases"],np.random.default_rng(settings["unseen_seed"]))
    unseen=dict(syndromes=s,observables=o,faults=e)
    for name,cases in (("train",train),("unseen",unseen)):
        with np.load(d/f"cases-{name}.npz") as saved:
            for key,value in cases.items():np.testing.assert_array_equal(value,saved[key])
        np.testing.assert_array_equal(parity(sampler.problem.H,cases["faults"]),cases["syndromes"])
        np.testing.assert_array_equal(parity(sampler.problem.L,cases["faults"]),cases["observables"])
    plan=helper["batch_plan"](settings,c.train.max_steps)
    with np.load(d/"batch_plan.npz") as saved:np.testing.assert_array_equal(plan,saved["indices"])
    objective=ParityLoss(sampler.problem,c.train.beta_logical,c.train.runtime.loss_eps,c.train.runtime.loss_floor)
    all_values={}
    replayed,exports=0,0
    initial_report=json.loads((d/"initial.json").read_text())
    for name,cases in (("train",train),("unseen",unseen)):
        summary,values=score(initial,cases,objective,c,settings)
        with np.load(d/f"scores-initial-{name}.npz") as saved:
            for key,value in values.items():np.testing.assert_array_equal(value,saved[key])
        expected=helper["group_summary"](summary,values,settings) if name=="train" else summary
        assert initial_report[name]==expected
        replayed+=len(cases["syndromes"]);exports+=summary["export_check_shots"]
    for arm in settings["easy_loss_weights"]:
        records=[json.loads(line) for line in (d/arm/"training.jsonl").read_text().splitlines()]
        assert len(records)==c.train.max_steps
        for row,selected in zip(records,plan):
            assert row["input_sha256"]==training_batch_hash(c.noise.p_grid[0],train["syndromes"][selected],train["faults"][selected],np.ones(len(selected)))
            assert np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"])
            index=row["step"]-1
            expected_lr=c.train.min_lr+(c.train.lr-c.train.min_lr)*(1+np.cos(np.pi*index/c.train.max_steps))/2
            np.testing.assert_allclose(row["lr"],expected_lr,rtol=1e-12)
        with torch.no_grad():
            outputs=initial(train["syndromes"][plan[0]])
            loss=helper["weighted_objective"](outputs,torch.from_numpy(train["faults"][plan[0]]),objective,helper["arm_weights"](arm,settings))
        assert float(loss)==records[0]["loss"]
        assert hashlib.sha256((d/arm/"last.pt").read_bytes()).hexdigest()==r["arms"][arm]["checkpoint_sha256"]
        model=load_model(d/arm/"last.pt",sampler.problem)
        for name,cases in (("train",train),("unseen",unseen)):
            summary,values=score(model,cases,objective,c,settings)
            expected=helper["group_summary"](summary,values,settings) if name=="train" else summary
            assert r["arms"][arm][name]==expected
            with np.load(d/arm/f"{name}-final.npz") as saved:
                for key,value in values.items():np.testing.assert_array_equal(value,saved[key])
            if name=="unseen":all_values[arm]=values
            replayed+=len(cases["syndromes"]);exports+=summary["export_check_shots"]
        for row in map(json.loads,(d/arm/"training_evaluation.jsonl").read_text().splitlines()):
            if not row["step"]:continue
            with np.load(d/arm/f"train-step{row['step']}.npz") as saved:
                assert float(saved["loss"].mean())==row["mean_loss"]
                assert int((~saved["logical_failed"][:settings["hard_cases"]]).sum())==row["hard_correct"]
                assert int((~saved["logical_failed"][settings["hard_cases"]:]).sum())==row["easy_correct"]
    with np.load(d/"scores-tuned.npz") as saved:all_values["tuned"]={key:saved[key] for key in ("failed","logical_failed")}
    tests=[]
    for comparator in ("hard_only","tuned"):
        for metric in ("failed","logical_failed"):
            result=mcnemar(all_values[comparator][metric],all_values["retain_easy"][metric])
            tests.append(dict(comparator=comparator,metric=metric,gain=float(all_values[comparator][metric].mean()-all_values["retain_easy"][metric].mean()),**result))
    adjusted=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]([test["pvalue"] for test in tests])
    for test,value in zip(tests,adjusted):test["holm_pvalue"]=value
    assert tests==r["tests"]
    assert r["useful"]==all(test["gain"]>=settings["min_net_gain"] and test["holm_pvalue"]<settings["alpha"] for test in tests)
    with (d/"results.csv").open(newline="") as stream:csv_rows=list(csv.DictReader(stream))
    assert len(csv_rows)==4 and all(row["config_hash"]==c.config_hash and int(row["seed"])==settings["unseen_seed"] for row in csv_rows)
    result=dict(checks_passed=True,matched_training_batches=c.train.max_steps,regenerated_pool_cases=len(pool["syndromes"]),
        regenerated_train_cases=len(train["syndromes"]),regenerated_unseen_cases=len(s),replayed_model_cases=replayed,
        exact_export_shots=exports,first_weighted_losses_reproduced=True,checkpoint_training_score_files_checked=16,
        statistical_tests_reproduced=4,csv_rows_checked=4,final_test_data_used=False,m6_complete=False,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    helper["write_json"](d/"audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
