"""Audit fixed-final-step replication outcomes and their separate paired families."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();d=args.directory
    if (d/"audit.json").exists():raise ValueError("Audit already exists")
    r=json.loads((d/"summary.json").read_text())
    source=Path(r["primary_confirmation"]);primary=json.loads((source/"summary.json").read_text())
    c=load_config(d/"config.yaml")
    manifest=json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_aux_replicate.py"]
    assert hashlib.sha256((source/"selected-source.pt").read_bytes()).hexdigest()==primary["checkpoint_sha256"]
    builder=runpy.run_path(str(source/"models_source.py"))
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    torch.set_num_threads(1);set_num_threads(1)
    tests=[];verified=0;replayed=0
    for replication in r["replications"]:
        root=d/f"seed{replication['training_seed']}"
        checkpoint=root/"last.pt"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==replication["checkpoint_sha256"]
        assert checkpoint.read_bytes()==(Path(replication["training_directory"])/"struct_standard"/"last.pt").read_bytes()
        mean=[]
        for index,point in enumerate(replication["points"]):
            sampler=make_sampler(c,point["p"])
            with np.load(source/f"cases-p{index}.npz") as saved:s,o=saved["syndromes"],saved["observables"]
            with np.load(root/f"values-p{index}.npz") as saved:value={key:saved[key] for key in saved.files}
            failed=~(parity(sampler.problem.H,value["corrections"])==s).all(1)
            logical=failed | (parity(sampler.problem.L,value["corrections"])!=o).any(1)
            np.testing.assert_array_equal(failed,value["failed"]);np.testing.assert_array_equal(logical,value["logical_failed"])
            assert (point["shots"],point["failures"],point["logical_failures"])==(len(s),int(failed.sum()),int(logical.sum()))
            model=builder["neural_model"](checkpoint,sampler.problem,replication["design"])
            a,conv=model.decode_batch(s[:64]);node=model.export_cpu();b,bconv=node.decode_batch(s[:64])
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(conv,bconv)
            actual=decode(node,s[:64],o[:64],sampler.problem,64)
            for key,array in actual.items():np.testing.assert_array_equal(array,value[key][:64])
            comparisons={}
            for name,key in (("ordinary",primary["points"][index]["ordinary_name"]),("matched_init","matched_init")):
                with np.load(source/f"values-p{index}-{key}.npz") as saved:baseline=saved["failed"]
                comparisons[name]=mcnemar(baseline,failed)
            tests.append((point,comparisons));mean.append(float(failed.mean()));verified+=len(s);replayed+=64
        assert replication["mean_fallback_rate"]==float(np.mean(mean))
    holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    for family in ("ordinary","matched_init"):
        for (point,actual),pvalue in zip(tests,holm([actual[family]["pvalue"] for _,actual in tests])):
            actual[family]["holm_pvalue"]=pvalue
            assert point["comparisons"][family]==actual[family]
    audit=dict(checks_passed=True,raw_replication_outcomes_verified=verified,
        replayed_exact_neural_export_cases=replayed,separate_four_test_families_reproduced=True,
        fixed_final_checkpoint_hashes_verified=True,primary_unchanged=True,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
