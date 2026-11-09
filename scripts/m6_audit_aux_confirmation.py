"""Replay frozen auxiliary confirmation data, outcomes, paired statistics and gate."""

import argparse
import csv
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
    report=json.loads((d/"summary.json").read_text())
    selection=json.loads((d/"selection.json").read_text())
    assert hashlib.sha256((d/"selected-source.pt").read_bytes()).hexdigest()==report["checkpoint_sha256"]==selection["checkpoint_sha256"]
    manifest=json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_aux_confirm.py"]
    for path,digest in manifest.items():
        if path.startswith("src/"):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    c=load_config(d/"config.yaml")
    builder=runpy.run_path(str(d/"models_source.py"))
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    torch.set_num_threads(1);set_num_threads(1)
    outcome_count,replays=0,0
    reconstructed=[]
    with (d/"results.csv").open(newline="") as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==14
    for index,point in enumerate(report["points"]):
        sampler=make_sampler(c,point["p"])
        s,o,e=sampler.sample(c.eval.max_shots,np.random.default_rng(c.seed+c.eval.seed_offset+index))
        with np.load(d/f"cases-p{index}.npz") as saved:
            for key,actual in (("syndromes",s),("observables",o),("faults",e)):
                np.testing.assert_array_equal(saved[key],actual)
        np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
        np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
        controls=json.loads((d/f"controls-p{index}.json").read_text())
        models=builder["make_decoders"](sampler.problem,d/"selected-source.pt",selection["model_design"],controls)
        values={}
        for name,node in models.items():
            if name=="ordinary":continue
            with np.load(d/f"values-p{index}-{name}.npz") as saved:value={key:saved[key] for key in saved.files}
            failed=~(parity(sampler.problem.H,value["corrections"])==s).all(1)
            logical=failed | (parity(sampler.problem.L,value["corrections"])!=o).any(1)
            np.testing.assert_array_equal(value["failed"],failed)
            np.testing.assert_array_equal(value["logical_failed"],logical)
            assert int(failed.sum())==point["failures"][name]
            assert int((~failed & logical).sum())==point["wrong_logical_converged"][name]
            subset=decode(node,s[:64],o[:64],sampler.problem,64)
            for key,item in subset.items():np.testing.assert_array_equal(item,value[key][:64])
            row=next(row for row in rows if row["stage1"]==name and float(row["p"])==point["p"])
            assert int(row["shots"])==len(s) and int(row["stage1_failures"])==int(failed.sum())
            assert int(row["logical_failures"])==int(logical.sum()) and row["config_hash"]==c.config_hash
            assert int(row["osd_calls"])==float(row["osd_rate"])==0
            outcome_count+=len(s);replays+=64;values[name]=failed
        reconstructed.append(dict(primary=mcnemar(values[point["ordinary_name"]],values["selected"]),
            learning=mcnemar(values["matched_init"],values["selected"])))
    for family in ("primary","learning"):
        for point,value in zip(reconstructed,holm([point[family]["pvalue"] for point in reconstructed])):
            point[family]["holm_pvalue"]=value
    for original,actual in zip(report["points"],reconstructed):
        for family in ("primary","learning"):assert original[family]==actual[family]
    selected=np.mean([p["failures"]["selected"]/p["shots"] for p in report["points"]])
    ordinary=np.mean([p["failures"][p["ordinary_name"]]/p["shots"] for p in report["points"]])
    initial=np.mean([p["failures"]["matched_init"]/p["shots"] for p in report["points"]])
    assert (selected,ordinary,initial)==(report["mean_selected"],report["mean_ordinary"],report["mean_matched_init"])
    benefits=sum(p["primary"]["holm_pvalue"]<selection["alpha"] and p["primary"]["first_only"]>p["primary"]["second_only"] for p in report["points"])
    harms=sum(p["primary"]["holm_pvalue"]<selection["alpha"] and p["primary"]["first_only"]<p["primary"]["second_only"] for p in report["points"])
    learning=sum(p["learning"]["holm_pvalue"]<selection["alpha"] and p["learning"]["first_only"]>p["learning"]["second_only"] for p in report["points"])
    clear=bool(ordinary-selected>=selection["minimum_mean_gain"] and benefits>0 and harms==0 and initial>selected and learning>0)
    assert clear==report["clear_gain"] and (benefits,harms,learning)==(report["primary_benefits"],report["primary_harms"],report["learning_benefits"])
    audit=dict(checks_passed=True,regenerated_independent_cases=2*c.eval.max_shots,
        raw_outcomes_verified=outcome_count,replayed_decoder_cases=replays,checked_csv_rows=len(rows),
        frozen_checkpoint_verified=True,paired_families_and_gate_reproduced=True,clear_gain=clear,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
