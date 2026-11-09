"""Audit one frozen training pool and completed matched product/sine arms."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.checkpoints import load_model, read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.losses import ParityLoss, multi_loss
from bbnbp.loss_sine import SineParityLoss
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import make_model, training_batch_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--sine-root",type=Path)
    args=parser.parse_args();d=args.directory
    if (d/"audit.json").exists():raise ValueError("Audit already exists")
    torch.set_num_threads(1);set_num_threads(1)
    settings=yaml.safe_load((d/"settings.yaml").read_text())
    frozen=json.loads((d/"frozen_data.json").read_text())
    for name,digest in frozen["hashes"].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest
    configs={arm:load_config(d/f"config-{arm}.yaml") for arm in settings["arms"]}
    c=next(iter(configs.values()))
    sampler=make_sampler(c,c.noise.p_grid[0])
    mine=runpy.run_path("scripts/m6_fixed_case_diagnostic.py")["hard_training_cases"]
    decoder=LdpcBP(sampler.problem,c.model.T,frozen["first_pass_factor"])
    train,pool,indices,state=mine(sampler,decoder,dict(training_pool_seed=settings["pool_seed"],
        training_cases=settings["training_cases"],training_pool_batch=settings["pool_batch"],
        training_pool_max_shots=settings["pool_max_shots"]))
    assert state==frozen["mining_rng_state"] and frozen["frozen_before_updates"]
    for filename,values in (("cases-train.npz",train),("mining-pool.npz",dict(**pool,selected_indices=indices))):
        with np.load(d/filename) as saved:
            for key,value in values.items():np.testing.assert_array_equal(saved[key],value)
    shuffle=np.random.default_rng(np.random.SeedSequence(settings["shuffle_entropy"]))
    plan=shuffle.permutation(len(indices)).reshape(-1,c.train.batch)
    with np.load(d/"batch-plan.npz") as saved:np.testing.assert_array_equal(plan,saved["indices"])
    assert shuffle.bit_generator.state==frozen["shuffle_rng_state"]
    for start in range(0,len(pool["syndromes"]),512):
        sl=slice(start,start+512)
        np.testing.assert_array_equal(parity(sampler.problem.H,pool["faults"][sl]),pool["syndromes"][sl])
        np.testing.assert_array_equal(parity(sampler.problem.L,pool["faults"][sl]),pool["observables"][sl])
    expected_hashes=[training_batch_hash(c.noise.p_grid[0],train["syndromes"][idx],train["faults"][idx],np.ones(len(idx))) for idx in plan]
    digest=hashlib.sha256()
    for value in expected_hashes:digest.update(value.encode())
    contexts=[]
    for index in range(2):
        meta=json.loads((d/f"baseline-selection-p{index}.json").read_text())
        other=make_sampler(c,meta["p"])
        s,o,e=other.sample(meta["shots"],np.random.default_rng(meta["seed"]))
        with np.load(d/f"validation-p{index}.npz") as saved:
            for key,value in (("syndromes",s),("observables",o),("faults",e)):
                np.testing.assert_array_equal(saved[key],value)
        contexts.append((other.problem,s,o))
    roots=[d/arm for arm in settings["arms"]]
    if args.sine_root:roots.extend(args.sine_root/arm for arm in ("struct_low","full_low"))
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    reports=[]
    for root in roots:
        r=json.loads((root/"summary.json").read_text());config=load_config(root/"config.yaml")
        assert r["completed"] and r["steps"]==len(plan)==config.train.max_steps
        assert r["training_inputs_sha256"]==digest.hexdigest()
        manifest=json.loads((root/"source_manifest.json").read_text())
        driver="scripts/m6_capacity_sine_train.py" if "objective_name" in r else "scripts/m6_capacity_train.py"
        assert hashlib.sha256((root/"executed_source.py").read_bytes()).hexdigest()==manifest[driver]
        for path,value in manifest.items():
            if path.startswith("src/"):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,path
        logs=[json.loads(line) for line in (root/"training.jsonl").read_text().splitlines()]
        assert len(logs)==len(plan)
        for i,(row,expected) in enumerate(zip(logs,expected_hashes)):
            assert row["step"]==i+1 and row["input_sha256"]==expected
            lr=config.train.min_lr+(config.train.lr-config.train.min_lr)*(1+np.cos(np.pi*i/config.train.max_steps))/2
            np.testing.assert_allclose(row["lr"],lr,rtol=1e-12)
            assert np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"])
        initial=make_model(sampler.problem,config)
        objective=(SineParityLoss(sampler.problem,config.train.beta_logical) if "objective_name" in r
                   else ParityLoss(sampler.problem,config.train.beta_logical,config.train.runtime.loss_eps,config.train.runtime.loss_floor))
        first=plan[0]
        with torch.no_grad():loss=multi_loss(initial(train["syndromes"][first]),torch.from_numpy(train["faults"][first]),objective)
        assert float(loss)==logs[0]["loss"],(root,float(loss),logs[0]["loss"])
        history=[json.loads(line) for line in (root/"validation.jsonl").read_text().splitlines()]
        best=min(history,key=lambda row:(row["total_failures"],row["step"]))
        assert (r["best_step"],r["best_validation_failures"])==(best["step"],best["total_failures"])
        outcomes=0
        for record in history+[r["final_validation"]]:
            label="final" if record is r["final_validation"] else f"step{record['step']}"
            for index,point in enumerate(record["points"]):
                problem,s,o=contexts[index]
                with np.load(root/f"validation-{label}-p{index}.npz") as saved:
                    failed=~(parity(problem.H,saved["corrections"])==s).all(1)
                    logical=failed | (parity(problem.L,saved["corrections"])!=o).any(1)
                    np.testing.assert_array_equal(saved["failed"],failed);np.testing.assert_array_equal(saved["logical_failed"],logical)
                    assert (point["failures"],point["logical_failures"])==(int(failed.sum()),int(logical.sum()))
                    assert point["mean_iterations"]==float(saved["iterations"].mean())
                    outcomes+=len(s)
            assert record["total_failures"]==sum(point["failures"] for point in record["points"])
        replays=0
        for name,key,step in (("best.pt","best_checkpoint_sha256",r["best_step"]),("last.pt","checkpoint_sha256",r["steps"])):
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==r[key]
            assert read_checkpoint(root/name)["step"]==step
            for index,(problem,s,o) in enumerate(contexts):
                # Replay neural second pass on the corresponding first-pass failures.
                with np.load(d/f"first-p{index}.npz") as first_saved:mask=first_saved["failed"]
                selected=np.flatnonzero(mask)[:64]
                model=load_model(root/name,problem)
                expected,conv=model.decode_batch(s[selected])
                actual,aconv=model.export_cpu().decode_batch(s[selected])
                np.testing.assert_array_equal(expected,actual);np.testing.assert_array_equal(conv,aconv)
                label=f"step{step}" if name=="best.pt" else "final"
                with np.load(root/f"validation-{label}-p{index}.npz") as saved:
                    np.testing.assert_array_equal(saved["corrections"][selected],actual)
                replays+=len(selected)
        reports.append(dict(directory=str(root),updates=len(logs),first_loss_reproduced=True,
            raw_validation_outcomes_verified=outcomes,exact_replayed_neural_cases=replays,
            best_selection_reproduced=True,checkpoint_hashes_verified=True))
        print(f"Audited {root}: {len(logs)} updates, {outcomes} raw validation outcomes",flush=True)
    audit=dict(checks_passed=True,regenerated_mining_cases=len(pool["failed"]),unique_selected_cases=len(indices),
        regenerated_validation_cases=2048,matching_batch_plan=True,training_inputs_sha256=digest.hexdigest(),arms=reports,
        scope="Input, schedule, first-loss, raw outcomes, checkpoint/export and selection audit; does not replay every optimizer update.",
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
