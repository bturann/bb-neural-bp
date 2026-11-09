"""Freeze a shared large first-pass-failure stream for the capacity/rate factorial."""

import argparse
import csv
import hashlib
import json
import runpy
import shutil
import signal
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import yaml

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.evaluate import tuned_factor
from bbnbp.io import source_manifest
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default="configs/m6/cpu_capacity_study.yaml")
    parser.add_argument("--output-root",default="results/m6_capacity")
    args=parser.parse_args()
    settings=yaml.safe_load(Path(args.settings).read_text())
    for path in Path("results").rglob("*.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("seed") and int(row["seed"])==settings["pool_seed"]:raise ValueError(f"Training pool seed already used in {path}")
    d=Path(args.output_root)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_prepare_source.py")
    for arm,path in settings["arms"].items():load_config(path).save(d/f"config-{arm}.yaml")
    print(f"Capacity shared data: {d}",flush=True)
    def deadline(signum,frame):raise TimeoutError("Capacity-data preparation reached its CPU wall deadline")
    signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,settings["prepare_wall_seconds"])
    try:
        c=load_config(next(iter(settings["arms"].values())))
        sampler=make_sampler(c,c.noise.p_grid[0])
        factor=tuned_factor(settings["first_pass_table"],c,c.noise.p_grid[0])
        decoder=LdpcBP(sampler.problem,c.model.T,factor)
        class ProgressDecoder:
            drawn=0
            failures=0
            def decode_batch(self,s):
                result=decoder.decode_batch(s)
                self.drawn+=len(s)
                self.failures+=int((~result[1]).sum())
                if self.drawn%4096==0:print(f"Mining: drawn={self.drawn}; first-pass failures={self.failures}",flush=True)
                return result
        mine=runpy.run_path("scripts/m6_fixed_case_diagnostic.py")["hard_training_cases"]
        protocol=dict(training_pool_seed=settings["pool_seed"],training_cases=settings["training_cases"],
            training_pool_batch=settings["pool_batch"],training_pool_max_shots=settings["pool_max_shots"])
        train,pool,indices,state=mine(sampler,ProgressDecoder(),protocol)
        np.testing.assert_array_equal(parity(sampler.problem.H,train["faults"]),train["syndromes"])
        np.testing.assert_array_equal(parity(sampler.problem.L,train["faults"]),train["observables"])
        np.savez_compressed(d/"cases-train.npz",**train)
        np.savez_compressed(d/"mining-pool.npz",**pool,selected_indices=indices)
        rng=np.random.default_rng(np.random.SeedSequence(settings["shuffle_entropy"]))
        plan=rng.permutation(len(indices)).reshape(-1,c.train.batch)
        if len(plan)!=c.train.max_steps:raise ValueError("Training pool must supply exactly one shared pass")
        np.savez_compressed(d/"batch-plan.npz",indices=plan)
        for i,root in enumerate(map(Path,settings["validation_bundles"])):
            for source,target in (("cases.npz",f"validation-p{i}.npz"),("first_pass.npz",f"first-p{i}.npz"),("selection.json",f"baseline-selection-p{i}.json")):
                shutil.copyfile(root/source,d/target)
        frozen=dict(frozen_before_updates=True,training_pool_seed=settings["pool_seed"],drawn=len(pool["failed"]),
            selected=len(indices),first_pass_factor=factor,mining_rng_state=state,shuffle_rng_state=rng.bit_generator.state,
            hashes={name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in ("cases-train.npz","mining-pool.npz","batch-plan.npz")},
            validation_used_for_selection_only=True,confirmation_data_used=False,final_test_data_used=False)
        (d/"frozen_data.json").write_text(json.dumps(frozen,indent=2)+"\n")
        print(f"Frozen {len(indices)} first-pass failures from {len(pool['failed'])} sampled cases",flush=True)
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
