"""Reproduce strong-first-pass controls, fresh cases, routing and paired families."""

import argparse
import hashlib
import json
import runpy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.cascade import CascadeDecoder
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def verify_values(path,problem,s,o):
    with np.load(path) as saved:value={key:saved[key] for key in saved.files}
    failed=~(parity(problem.H,value["corrections"])==s).all(1)
    logical=failed | (parity(problem.L,value["corrections"])!=o).any(1)
    np.testing.assert_array_equal(failed,value["failed"]);np.testing.assert_array_equal(logical,value["logical_failed"])
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();d=args.directory
    if (d/"audit.json").exists():raise ValueError("Audit already exists")
    r=json.loads((d/"summary.json").read_text())
    frozen=json.loads((d/"selection.json").read_text())
    settings=yaml.safe_load((d/"settings.yaml").read_text())
    receipt=json.loads((d/"confirmation_receipt.json").read_text())
    assert datetime.fromisoformat(frozen["frozen_at_utc"])<datetime.fromisoformat(receipt["started_at_utc"])
    assert receipt["seeds"]==settings["confirmation_seeds"] and receipt["shots_per_point"]==settings["shots"]
    assert hashlib.sha256((d/"selected-source.pt").read_bytes()).hexdigest()==frozen["checkpoint_sha256"]==r["primary_checkpoint_sha256"]
    manifest=json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_strong_cascade.py"]
    for path,digest in manifest.items():
        if path.startswith("src/"):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    c=load_config(d/"config.yaml")
    module=runpy.run_path(str(d/"models_source.py"))
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    torch.set_num_threads(1);set_num_threads(1)
    checked,replayed,exports=0,0,0
    tests=[]
    for index,(choice,point) in enumerate(zip(frozen["points"],r["points"])):
        sampler=make_sampler(c,point["p"])
        source=Path(choice["validation_source"])
        assert hashlib.sha256((source/"cases.npz").read_bytes()).hexdigest()==choice["validation_cases_sha256"]
        long=json.loads((Path(frozen["long_controls"][index])/"selection.json").read_text())
        assert choice["first"]==long["choices"][str(settings["first_iterations"]) ]
        with np.load(source/"cases.npz") as saved:s,o=saved["syndromes"],saved["observables"]
        for row in choice["grid"]:
            name=row["name"]
            path=d/f"validation-p{index}-bp100.npz" if name=="bp100" else d/f"validation-p{index}-{name}-{row['factor']:g}.npz"
            value=verify_values(path,sampler.problem,s,o)
            assert row["failures"]==int(value["failed"].sum()) and row["logical_failures"]==int(value["logical_failed"].sum())
            checked+=len(s)
        for name,spec in choice["choices"].items():
            assert spec==min([row for row in choice["grid"] if row["name"]==name],key=lambda row:(row["failures"],row["logical_failures"],row["factor"]))
        assert choice["best_ordinary"]==min(choice["choices"].values(),key=lambda row:(row["failures"],row["logical_failures"],row["total_T"],row["name"]))
        first=lambda:LdpcBP(sampler.problem,100,choice["first"]["factor"])
        models={"bp100":first(),"bp130":LdpcBP(sampler.problem,130,choice["choices"]["bp130"]["factor"]),
            "two_bp":CascadeDecoder(first(),LdpcBP(sampler.problem,30,choice["choices"]["two_bp"]["factor"]))}
        for name,initial in (("strong_cascade",False),("strong_init",True)):
            model=module["neural_model"](d/"selected-source.pt",sampler.problem,"cycle30",initial=initial)
            a,conv=model.decode_batch(s[:64]);node=model.export_cpu();b,bconv=node.decode_batch(s[:64])
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(conv,bconv);exports+=64
            models[name]=CascadeDecoder(first(),node)
            value=verify_values(d/f"validation-p{index}-{name}.npz",sampler.problem,s,o)
            assert int(value["failed"].sum())==choice["learned_validation_failures" if name=="strong_cascade" else "initialized_validation_failures"]
            actual=decode(models[name],s[:64],o[:64],sampler.problem,64)
            for key,array in actual.items():np.testing.assert_array_equal(array,value[key][:64])
            checked+=len(s);replayed+=64
        s,o,e=sampler.sample(settings["shots"],np.random.default_rng(settings["confirmation_seeds"][index]))
        with np.load(d/f"cases-p{index}.npz") as saved:
            for key,array in (("syndromes",s),("observables",o),("faults",e)):np.testing.assert_array_equal(saved[key],array)
        np.testing.assert_array_equal(parity(sampler.problem.H,e),s);np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
        values={}
        for name,node in models.items():
            value=verify_values(d/f"values-p{index}-{name}.npz",sampler.problem,s,o)
            assert int(value["failed"].sum())==point["failures"][name]
            assert int((~value["failed"] & value["logical_failed"]).sum())==point["wrong_logical_converged"][name]
            actual=decode(node,s[:64],o[:64],sampler.problem,64)
            for key,array in actual.items():np.testing.assert_array_equal(array,value[key][:64])
            checked+=len(s);replayed+=64;values[name]=value
        tests.append(dict(primary=mcnemar(values[point["best_ordinary"]]["failed"],values["strong_cascade"]["failed"]),
            learning=mcnemar(values["strong_init"]["failed"],values["strong_cascade"]["failed"])))
    holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    for family in ("primary","learning"):
        for point,value in zip(tests,holm([point[family]["pvalue"] for point in tests])):point[family]["holm_pvalue"]=value
    for point,actual in zip(r["points"],tests):
        for family in ("primary","learning"):assert point[family]==actual[family]
    audit=dict(checks_passed=True,regenerated_fresh_cases=2*settings["shots"],raw_outcomes_verified=checked,
        routed_pipeline_cases_replayed=replayed,exact_neural_exports=exports,
        control_selection_and_pair_families_reproduced=True,learned_parameters_unchanged=True,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
