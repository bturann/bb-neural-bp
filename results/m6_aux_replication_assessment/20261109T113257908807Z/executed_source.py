"""Assess precommitted final-step training replications on fixed auxiliary cases."""

import argparse
import hashlib
import json
import runpy
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.checkpoints import read_checkpoint
from bbnbp.config import load_config
from bbnbp.io import source_manifest
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation",type=Path)
    parser.add_argument("replications",nargs=2,type=Path)
    parser.add_argument("--plan",default="configs/m6/cpu_aux_replication_plan.yaml")
    args=parser.parse_args()
    plan=yaml.safe_load(Path(args.plan).read_text())
    confirmation=args.confirmation
    primary=json.loads((confirmation/"summary.json").read_text())
    assert plan["primary_selection"]==primary["selection_directory"]
    receipt=json.loads((Path(primary["selection_directory"])/"confirmation_receipt.json").read_text())
    assert datetime.fromisoformat(plan["frozen_at_utc"])<datetime.fromisoformat(receipt["requested_at_utc"])
    if plan["select_primary_on_replications"]:raise ValueError("Primary choice must remain frozen")
    for path,digest in plan["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    for root,spec in zip(args.replications,plan["replications"]):
        train=root/"struct_standard"
        summary=json.loads((train/"summary.json").read_text())
        c=load_config(train/"config.yaml")
        assert summary["completed"] and summary["steps"]==spec["expected_step"] and c.seed==spec["seed"]
        assert yaml.safe_load((root/"settings.yaml").read_text())["pool_seed"]==spec["pool_seed"]
        assert not (train/"confirmation_receipt.json").exists(),"Replication was already assessed"
        assert read_checkpoint(train/spec["source_checkpoint"])["step"]==spec["expected_step"]
    d=Path("results/m6_aux_replication_assessment")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.plan,d/"plan.yaml")
    shutil.copyfile(__file__,d/"executed_source.py")
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    c=load_config(confirmation/"config.yaml");c.save(d/"config.yaml")
    torch.set_num_threads(1);set_num_threads(1)
    helper=runpy.run_path("scripts/m6_cascade_tune.py")
    builder=runpy.run_path(str(confirmation/"models_source.py"))
    reports=[]
    try:
        for root,spec in zip(args.replications,plan["replications"]):
            source=root/"struct_standard"/spec["source_checkpoint"]
            dest=d/f"seed{spec['seed']}";dest.mkdir()
            shutil.copyfile(source,dest/"last.pt")
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            (root/"struct_standard"/"confirmation_receipt.json").write_text(json.dumps(dict(directory=str(dest),
                requested_at_utc=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")
            points=[]
            for index,p in enumerate(c.noise.p_grid):
                sampler=make_sampler(c,p)
                with np.load(confirmation/f"cases-p{index}.npz") as saved:s,o=saved["syndromes"],saved["observables"]
                reference=builder["neural_model"](source,sampler.problem,spec["inference_design"])
                node=reference.export_cpu()
                a,conv=reference.decode_batch(s[:64]);b,bconv=node.decode_batch(s[:64])
                np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(conv,bconv)
                np.testing.assert_array_equal(reference.last_iterations,node.last_iterations)
                value=helper["decode"](node,s,o,sampler.problem,c.eval.batch)
                np.testing.assert_array_equal(value["failed"],~(parity(sampler.problem.H,value["corrections"])==s).all(1))
                np.savez_compressed(dest/f"values-p{index}.npz",**value)
                ordinary=primary["points"][index]["ordinary_name"]
                comparisons={}
                for name,actual in (("ordinary",ordinary),("matched_init","matched_init")):
                    with np.load(confirmation/f"values-p{index}-{actual}.npz") as saved:baseline=saved["failed"]
                    comparisons[name]=mcnemar(baseline,value["failed"])
                points.append(dict(p=p,shots=len(s),failures=int(value["failed"].sum()),
                    logical_failures=int(value["logical_failed"].sum()),comparisons=comparisons,exact_export_shots=64))
                print(f"replication seed={spec['seed']} p={p:g}: {points[-1]['failures']}/{len(s)} Stage1 failures",flush=True)
            reports.append(dict(training_directory=str(root),training_seed=spec["seed"],pool_seed=spec["pool_seed"],
                checkpoint_sha256=digest,step=spec["expected_step"],design=spec["inference_design"],points=points,
                mean_fallback_rate=float(np.mean([point["failures"]/point["shots"] for point in points]))))
        holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
        points=[point for report in reports for point in report["points"]]
        for family in ("ordinary","matched_init"):
            corrected=holm([point["comparisons"][family]["pvalue"] for point in points])
            for point,value in zip(points,corrected):point["comparisons"][family]["holm_pvalue"]=value
        report=dict(replications=reports,primary_checkpoint_unchanged=hashlib.sha256((confirmation/"selected-source.pt").read_bytes()).hexdigest()==primary["checkpoint_sha256"],
            primary_confirmation=str(confirmation),scope="Precommitted method replications using final step, no checkpoint selection. Same fixed confirmation cases across seeds; do not count them as independent extra shots. Stage1 robustness only, separate four-test families over added replications/noise points. Main M6 incomplete.",
            final_test_data_used=False,gpu_seconds=0,m6_complete=False)
        (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise


if __name__=="__main__":main()
