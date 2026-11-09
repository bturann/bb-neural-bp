"""Audit matched training, larger validation and the fresh training-readout ablation."""

import argparse
import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch

from bbnbp.checkpoints import load_model,read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.metrics import mcnemar,wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference,validation_sets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    d=args.directory
    if (d/"audit.json").exists():
        parser.error("Refusing to overwrite audit evidence")
    report=json.loads((d/"summary.json").read_text())
    c=load_config(d/"config.yaml")
    arm_loader=runpy.run_path("scripts/m6_loss_ablation.py")["training_arm"]
    arms={name:arm_loader(Path(path)) for name,path in report["training_roots"].items()}
    controls=runpy.run_path("scripts/m6_readout_ablation.py")["verify_readout_control"](arms["full"],arms["frozen"])
    assert controls==report["controls"]
    torch.set_num_threads(1)
    manifest=json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_readout_ablation.py"]
    with (d/"results.csv").open(newline="") as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==20 and all(row["config_hash"]==c.config_hash for row in rows)
    new_seeds={point["seed"] for point in report["points"]}
    prior_csvs=0
    for previous in Path("results").rglob("*.csv"):
        if Path("results/m6_cpu_readout") in previous.parents:
            continue
        with previous.open(newline="") as stream:
            reader=csv.DictReader(stream)
            if not reader.fieldnames or "seed" not in reader.fieldnames:
                continue
            prior_csvs+=1
            for row in reader:
                if row.get("seed"):
                    assert int(row["seed"]) not in new_seeds,previous
    completed_partial_rows=0
    if report.get("completion_of"):
        previous=Path(report["completion_of"])
        assert "TimeoutError" in (previous/"FAILURE.txt").read_text()
        assert load_config(previous/"config.yaml")==c
        with (previous/"results.csv").open(newline="") as stream:
            for old in csv.DictReader(stream):
                new=next(row for row in rows if row["stage1"]==old["stage1"] and row["p"]==old["p"])
                for key in old:
                    if key not in {"run_id","git_sha","timestamp"}:
                        assert old[key]==new[key],key
                completed_partial_rows+=1
        for path in previous.glob("*-p*.npz"):
            with np.load(path) as old,np.load(d/path.name) as new:
                assert old.files==new.files
                for key in old.files:np.testing.assert_array_equal(old[key],new[key])
    validation_checked=0
    parameter_delta={}
    for name,arm in arms.items():
        training_manifest=json.loads((arm["run"]/"source_manifest.json").read_text())
        assert all(manifest[path]==digest for path,digest in training_manifest.items() if path.startswith("src/"))
        selections=[json.loads(line) for line in (arm["run"]/"validation.jsonl").read_text().splitlines()]
        assert min(selections,key=lambda row:row["mean_osd_rate"])["step"]==arm["payload"]["step"]
        meta=report["checkpoints"][name]
        assert hashlib.sha256(arm["checkpoint"].read_bytes()).hexdigest()==meta["sha256"]
        assert (d/f"checkpoint-{name}.pt").read_bytes()==arm["checkpoint"].read_bytes()
        samplers={p:make_sampler(arm["config"],p) for p in arm["config"].noise.p_grid}
        cases=validation_sets(arm["config"],samplers)
        with np.load(arm["run"]/"best_validation_flags.npz") as flags:
            for i,p in enumerate(arm["config"].noise.p_grid):
                s,o,_=cases[p]
                model=load_model(arm["checkpoint"],samplers[p].problem).export_cpu()
                failed,logical,_=_inference(model,s,o,samplers[p].problem,arm["config"].train.runtime.val_batch)
                np.testing.assert_array_equal(failed,flags[f"nbp-p{i}"])
                np.testing.assert_array_equal(logical,flags[f"logical-p{i}"])
                validation_checked+=len(s)
    final={name:read_checkpoint(a["checkpoint"].parent/"last.pt") for name,a in arms.items()}
    for key in ("w","alpha"):
        parameter_delta[key]=float((final["full"]["parameters"][key]-final["frozen"]["parameters"][key]).abs().max())
    labels=("bp_tuned","nbp_initial","nbp_full","nbp_frozen")
    means={name:[] for name in labels}
    regenerated,checked,replayed,exports=0,0,0,0
    primary,secondary=[],[]
    for i,point in enumerate(report["points"]):
        sampler=make_sampler(c,point["p"])
        s,o,e=sampler.sample(c.eval.max_shots,np.random.default_rng(point["seed"]))
        np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
        np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
        del e
        with np.load(d/f"cases-p{i}.npz") as saved:
            np.testing.assert_array_equal(s,saved["syndromes"])
            np.testing.assert_array_equal(o,saved["observables"])
        regenerated+=len(s)
        models={f"nbp_{name}":load_model(arm["checkpoint"],sampler.problem) for name,arm in arms.items()}
        payload=arms["full"]["payload"]
        models["nbp_initial"]=NeuralMinSum(sampler.problem,**payload["model_config"],dtype=models["nbp_full"].w.dtype)
        decoders={"bp_tuned":LdpcBP(sampler.problem,c.model.T,point["tuned_factor"])}
        n=min(report["settings"]["export_check_shots"],len(s))
        for name,model in models.items():
            compiled=model.export_cpu()
            expected,conv=model.decode_batch(s[:n])
            actual,actual_conv=compiled.decode_batch(s[:n])
            np.testing.assert_array_equal(expected,actual)
            np.testing.assert_array_equal(conv,actual_conv)
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
                a,b,_=_inference(decoders[name],s[:n],o[:n],sampler.problem,n)
                np.testing.assert_array_equal(a,flags[name][:n])
                np.testing.assert_array_equal(b,logical[name][:n])
                means[name].append(float(flags[name].mean()))
                checked+=len(s)
                replayed+=n
            test=mcnemar(flags["nbp_full"],flags["nbp_frozen"])
            assert all(test[k]==point["primary"][k] for k in test)
            primary.append(test["pvalue"])
            for name in ("bp_tuned","nbp_initial"):
                test=mcnemar(flags[name],flags["nbp_frozen"])
                assert all(test[k]==point["secondary"][name][k] for k in test)
                secondary.append(test["pvalue"])
    adjust=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    np.testing.assert_allclose(adjust(primary),[p["primary"]["holm_pvalue"] for p in report["points"]])
    np.testing.assert_allclose(adjust(secondary),[p["secondary"][name]["holm_pvalue"] for p in report["points"] for name in ("bp_tuned","nbp_initial")])
    assert {name:float(np.mean(values)) for name,values in means.items()}==report["mean_fallback"]
    result=dict(checks_passed=True,matched_training_batches=controls["matched_training_batches"],
        validation_cases_replayed=validation_checked,regenerated_comparison_cases=regenerated,
        checked_decoder_flags=checked,subset_decodes_replayed=replayed,exact_export_shots=exports,
        checked_csv_rows=len(rows),final_parameter_max_difference=parameter_delta,
        prior_csv_seed_sources_checked=prior_csvs,completed_partial_rows_reproduced=completed_partial_rows,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statistical_families_verified=[5,10],final_test_data_used=False,m6_complete=False)
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
