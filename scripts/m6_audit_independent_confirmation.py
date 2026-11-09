"""Audit fixed-candidate provenance, independent cases, outcomes and confirmation rules."""

import argparse
import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch

from bbnbp.checkpoints import load_model, read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output-name",default="audit.json")
    args=parser.parse_args()
    d=args.directory
    if (d/args.output_name).exists():
        parser.error("Refusing to overwrite audit evidence")
    c=load_config(d/"config.yaml")
    report=json.loads((d/"summary.json").read_text())
    settings=report["settings"]
    preferences=json.loads((d/"selection_record.json").read_text())
    assert (d/"selection_record.json").read_bytes()==Path(settings["preferences"]).read_bytes()
    manifest=json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_independent_confirmation.py"]
    with (d/"results.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    with (d/"tuned_scaling.csv").open(newline="") as stream:
        tuning=list(csv.DictReader(stream))
    assert len(rows)==20 and all(row["config_hash"]==c.config_hash for row in rows)
    helpers=runpy.run_path("scripts/m6_independent_confirmation.py")
    payloads={arm:read_checkpoint(d/f"checkpoint-{arm}.pt") for arm in ("parity","bitwise")}
    helpers["check_seed_separation"](c,preferences,payloads,tuning)
    new_seeds={point["seed"] for point in report["points"]}
    prior_csvs=0
    for previous in Path("results").rglob("*.csv"):
        if d in previous.parents:
            continue
        with previous.open(newline="") as stream:
            reader=csv.DictReader(stream)
            if not reader.fieldnames or "seed" not in reader.fieldnames:
                continue
            prior_csvs+=1
            for row in reader:
                if row.get("seed"):
                    assert int(row["seed"]) not in new_seeds, f"Previously used confirmation seed in {previous}"
    table_helpers=runpy.run_path("scripts/m6_compare_stage1.py")
    table_helpers["validate_tuning_table"](c,d/"tuned_scaling.csv")
    for arm,meta in report["frozen_candidates"].items():
        assert hashlib.sha256(Path(meta["path"]).read_bytes()).hexdigest()==meta["sha256"]
        assert (d/f"checkpoint-{arm}.pt").read_bytes()==Path(meta["path"]).read_bytes()
        assert meta["sha256"]==preferences["candidates"][arm]["checkpoint"]["sha256"]
    torch.set_num_threads(1)
    labels=("bp_tuned","nbp_initial","nbp_parity","nbp_bitwise")
    by_model={name:[] for name in labels}
    regenerated,checked,replayed,exports=0,0,0,0
    primary,secondary=[],[]
    for i,point in enumerate(report["points"]):
        assert point["seed"]==c.seed+c.eval.seed_offset+i
        sampler=make_sampler(c,point["p"])
        s,o,e=sampler.sample(c.eval.max_shots,np.random.default_rng(point["seed"]))
        np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
        np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
        del e
        with np.load(d/f"cases-p{i}.npz") as saved:
            np.testing.assert_array_equal(saved["syndromes"],s)
            np.testing.assert_array_equal(saved["observables"],o)
        regenerated+=len(s)
        models={f"nbp_{arm}":load_model(d/f"checkpoint-{arm}.pt",sampler.problem) for arm in payloads}
        models["nbp_initial"]=NeuralMinSum(sampler.problem,**payloads["parity"]["model_config"],dtype=models["nbp_parity"].w.dtype)
        decoders={"bp_tuned":LdpcBP(sampler.problem,c.model.T,point["tuned_factor"])}
        n=min(settings["export_check_shots"],len(s))
        for name,model in models.items():
            compiled=model.export_cpu()
            expected,conv=model.decode_batch(s[:n])
            actual,actual_conv=compiled.decode_batch(s[:n])
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(actual_conv,conv)
            np.testing.assert_array_equal(model.last_iterations,compiled.last_iterations)
            decoders[name]=compiled
            exports+=n
        with np.load(d/f"failed-p{i}.npz") as flags,np.load(d/f"logical-p{i}.npz") as logical:
            for name in labels:
                row=next(row for row in rows if row["stage1"]==name and float(row["p"])==point["p"])
                assert int(row["seed"])==point["seed"] and int(row["shots"])==len(s)
                assert int(row["stage1_failures"])==int(flags[name].sum())
                assert int(row["logical_failures"])==int(logical[name].sum())
                assert int(row["osd_calls"])==float(row["osd_rate"])==0
                assert float(flags[name].mean())==point["models"][name]["fallback_demand"]
                np.testing.assert_allclose(wilson(int(logical[name].sum()),len(s)),[float(row["ler_lo"]),float(row["ler_hi"])])
                failed,logical_subset,_=_inference(decoders[name],s[:n],o[:n],sampler.problem,n)
                np.testing.assert_array_equal(failed,flags[name][:n])
                np.testing.assert_array_equal(logical_subset,logical[name][:n])
                checked+=len(s)
                replayed+=n
                by_model[name].append(flags[name])
            test=mcnemar(flags["bp_tuned"],flags["nbp_bitwise"])
            assert all(test[k]==point["primary"][k] for k in test)
            primary.append(test["pvalue"])
            for name in ("nbp_parity","nbp_initial"):
                test=mcnemar(flags[name],flags["nbp_bitwise"])
                assert all(test[k]==point["secondary"][name][k] for k in test)
                secondary.append(test["pvalue"])
    np.testing.assert_allclose(table_helpers["holm"](primary),[p["primary"]["holm_pvalue"] for p in report["points"]])
    np.testing.assert_allclose(table_helpers["holm"](secondary),[p["secondary"][name]["holm_pvalue"] for p in report["points"] for name in ("nbp_parity","nbp_initial")])
    means={name:float(np.mean([a.mean() for a in parts])) for name,parts in by_model.items()}
    assert means==report["mean_fallback"]
    delta=[a.astype(np.int8)-b.astype(np.int8) for a,b in zip(by_model["nbp_bitwise"],by_model["bp_tuned"])]
    boot=runpy.run_path("scripts/m6_validation_stability.py")["paired_bootstrap"](delta,
        np.random.default_rng(np.random.SeedSequence(settings["bootstrap_seed_entropy"])),settings["bootstrap_replicates"],settings["confidence_level"])
    assert boot==report["paired_grid_mean_bootstrap"]
    benefit=[p for p in report["points"] if p["models"]["nbp_bitwise"]["fallback_demand"]<p["models"]["bp_tuned"]["fallback_demand"] and p["primary"]["holm_pvalue"]<settings["alpha"]]
    harm=[p for p in report["points"] if p["models"]["nbp_bitwise"]["fallback_demand"]>p["models"]["bp_tuned"]["fallback_demand"] and p["primary"]["holm_pvalue"]<settings["alpha"]]
    assert report["clear_gain"]==(means["nbp_bitwise"]<means["bp_tuned"] and bool(benefit) and not harm)
    result=dict(checks_passed=True,independent_seed_guard_verified=True,prior_csv_seed_sources_checked=prior_csvs,frozen_candidate_hashes_unchanged=True,
        regenerated_cases=regenerated,checked_decoder_flags=checked,replayed_subset_decodes=replayed,
        exact_export_shots=exports,checked_csv_rows=len(rows),primary_family_tests=5,secondary_family_tests=10,
        bootstrap_and_decision_rule_reproduced=True,final_test_data_used=False,m6_complete=False)
    (d/args.output_name).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
