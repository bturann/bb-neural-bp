"""Audit fresh cycle100 data, controls, source checkpoints and all paired families."""

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
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def check_value(path,problem,s,o):
    with np.load(path) as saved:value={k:saved[k] for k in saved.files}
    failed=~(parity(problem.H,value["corrections"])==s).all(1)
    logical=failed | (parity(problem.L,value["corrections"])!=o).any(1)
    np.testing.assert_array_equal(value["failed"],failed);np.testing.assert_array_equal(value["logical_failed"],logical)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();d=args.directory
    if (d/"audit.json").exists():raise ValueError("Audit already exists")
    r=json.loads((d/"summary.json").read_text())
    frozen=json.loads((d/"selection.json").read_text())
    receipt=json.loads((d/"confirmation_receipt.json").read_text())
    settings=yaml.safe_load((d/"settings.yaml").read_text())
    assert datetime.fromisoformat(frozen["frozen_at_utc"])<datetime.fromisoformat(receipt["started_at_utc"])
    assert receipt["seeds"]==settings["confirmation_seeds"] and receipt["shots_per_point"]==settings["shots"]
    assert frozen["iterations"]==100 and frozen["extension"]=="cycle" and frozen["additional_optimizer_updates"]==0
    for name,digest in frozen["checkpoints"].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest
    assert frozen["checkpoints"]["selected-source.pt"]==r["checkpoint_sha256"]==settings["source_checkpoint_sha256"]
    for name,path in settings["replication_sources"].items():assert (d/f"{name}-source.pt").read_bytes()==Path(path).read_bytes()
    source=Path(settings["source_confirmation"])
    assert (d/"neural_extensions_source.py").read_bytes()==(source/"neural_extensions_source.py").read_bytes()
    manifest=json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_cycle100_study.py"]
    for path,digest in manifest.items():
        if path.startswith("src/"):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    module=runpy.run_path(str(d/"executed_source.py"))
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    c=load_config(d/"config.yaml");torch.set_num_threads(1);set_num_threads(1)
    checked,replayed,exports=0,0,0;reconstructed=[]
    for index,(choice,point) in enumerate(zip(frozen["points"],r["points"])):
        sampler=make_sampler(c,point["p"])
        validation=Path(choice["validation_source"])
        assert hashlib.sha256((validation/"cases.npz").read_bytes()).hexdigest()==choice["validation_cases_sha256"]
        long=json.loads((Path(settings["long_control_sources"][index])/"selection.json").read_text())
        assert choice["cpp"]==long["choices"]["100"]
        with np.load(validation/"cases.npz") as saved:s,o=saved["syndromes"],saved["observables"]
        assert sorted(row["factor"] for row in choice["native_grid"])==sorted(c.sweep.factors)
        for row in choice["native_grid"]:
            value=check_value(d/f"validation-native-p{index}-factor{row['factor']:g}.npz",sampler.problem,s,o)
            assert row["failures"]==int(value["failed"].sum()) and row["logical_failures"]==int(value["logical_failed"].sum())
            node=NeuralMinSum(sampler.problem,100,"scalar",row["factor"],20.,dtype=torch.float32).export_cpu()
            actual=decode(node,s[:16],o[:16],sampler.problem,16)
            for key,array in actual.items():np.testing.assert_array_equal(array,value[key][:16])
            checked+=len(s);replayed+=16
        assert choice["native"]==min(choice["native_grid"],key=lambda row:(row["failures"],row["logical_failures"],row["factor"]))
        best=min((("cpp100",choice["cpp"]),("native100",choice["native"])),key=lambda pair:(pair[1]["failures"],pair[1]["logical_failures"],pair[0]))[0]
        assert best==choice["best_ordinary"]==point["ordinary_name"]
        s,o,e=sampler.sample(settings["shots"],np.random.default_rng(settings["confirmation_seeds"][index]))
        with np.load(d/f"cases-p{index}.npz") as saved:
            for key,array in (("syndromes",s),("observables",o),("faults",e)):np.testing.assert_array_equal(saved[key],array)
        np.testing.assert_array_equal(parity(sampler.problem.H,e),s);np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
        nodes=module["make_models"](d,sampler.problem,index);values={}
        for name,node in nodes.items():
            if name=="ordinary":continue
            value=check_value(d/f"values-p{index}-{name}.npz",sampler.problem,s,o)
            assert int(value["failed"].sum())==point["failures"][name]
            assert int((~value["failed"] & value["logical_failed"]).sum())==point["wrong_logical_converged"][name]
            actual=decode(node,s[:64],o[:64],sampler.problem,64)
            for key,array in actual.items():np.testing.assert_array_equal(array,value[key][:64])
            if name in {"selected","matched_init","seed1235","seed1236"}:
                model=module["neural"](d,sampler.problem,name);hat,conv=model.decode_batch(s[:64])
                np.testing.assert_array_equal(hat,actual["corrections"]);np.testing.assert_array_equal(~conv,actual["failed"])
                exports+=64
            checked+=len(s);replayed+=64;values[name]=value
        comparisons={name:dict(primary=mcnemar(values[best]["failed"],values[name]["failed"]),
            learning=mcnemar(values["matched_init"]["failed"],values[name]["failed"])) for name in ("selected","seed1235","seed1236")}
        reconstructed.append(comparisons)
    holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    for family in ("primary","learning"):
        for names in (("selected",),("seed1235","seed1236")):
            entries=[point[name][family] for point in reconstructed for name in names]
            for entry,value in zip(entries,holm([entry["pvalue"] for entry in entries])):entry["holm_pvalue"]=value
    for original,actual in zip(r["points"],reconstructed):assert original["comparisons"]==actual
    means={name:float(np.mean([point["failures"][name]/point["shots"] for point in r["points"]])) for name in r["mean_fallback"]}
    ordinary=float(np.mean([p["failures"][p["ordinary_name"]]/p["shots"] for p in r["points"]]))
    assert means==r["mean_fallback"] and ordinary==r["mean_ordinary"]
    benefits=sum(p["selected"]["primary"]["holm_pvalue"]<settings["alpha"] and p["selected"]["primary"]["first_only"]>p["selected"]["primary"]["second_only"] for p in reconstructed)
    harms=sum(p["selected"]["primary"]["holm_pvalue"]<settings["alpha"] and p["selected"]["primary"]["first_only"]<p["selected"]["primary"]["second_only"] for p in reconstructed)
    learning=sum(p["selected"]["learning"]["holm_pvalue"]<settings["alpha"] and p["selected"]["learning"]["first_only"]>p["selected"]["learning"]["second_only"] for p in reconstructed)
    clear=bool(ordinary-means["selected"]>=settings["minimum_mean_gain"] and benefits>0 and harms==0 and means["matched_init"]>means["selected"] and learning>0)
    assert (benefits,harms,learning,clear)==(r["primary_benefits"],r["primary_harms"],r["learning_benefits"],r["clear_gain"])
    audit=dict(checks_passed=True,regenerated_fresh_cases=2*settings["shots"],raw_outcomes_verified=checked,
        replayed_decoder_cases=replayed,exact_neural_export_cases=exports,
        all_checkpoint_hashes_and_control_choices_verified=True,all_paired_families_and_screen_reproduced=True,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
