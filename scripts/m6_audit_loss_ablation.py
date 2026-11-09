"""Audit batch-matched training and fresh, paired parity-versus-BCE outcomes."""

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
from bbnbp.train import _inference, validation_sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    d = args.directory
    output = d/"audit.json"
    if output.exists():
        parser.error("Refusing to overwrite audit evidence")
    c = load_config(d/"config.yaml")
    torch.set_num_threads(1)
    summary = json.loads((d/"summary.json").read_text())
    helpers = runpy.run_path("scripts/m6_loss_ablation.py")
    arms = {name: helpers["training_arm"](Path(path)) for name,path in summary["training_roots"].items()}
    assert helpers["verify_control"](arms["parity"], arms["bitwise"]) == summary["controls"]
    manifest = json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest() == manifest["scripts/m6_loss_ablation.py"]
    with (d/"results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 25
    validation_replayed = 0
    for name, arm in arms.items():
        assert hashlib.sha256(arm["checkpoint"].read_bytes()).hexdigest() == summary["checkpoints"][name]["sha256"]
        assert (d/f"checkpoint-{name}.pt").read_bytes() == arm["checkpoint"].read_bytes()
        samplers = {p: make_sampler(arm["config"],p) for p in arm["config"].noise.p_grid}
        validation = validation_sets(arm["config"], samplers)
        with np.load(arm["run"]/"best_validation_flags.npz") as saved:
            for i,p in enumerate(arm["config"].noise.p_grid):
                s,o,_ = validation[p]
                model = load_model(arm["checkpoint"],samplers[p].problem).export_cpu()
                failed,logical,_ = _inference(model,s,o,samplers[p].problem,arm["config"].train.runtime.val_batch)
                np.testing.assert_array_equal(failed,saved[f"nbp-p{i}"])
                np.testing.assert_array_equal(logical,saved[f"logical-p{i}"])
                validation_replayed += len(s)
    labels = ("bp","bp_tuned","nbp_initial","nbp_parity","nbp_bitwise")
    means = {name:[] for name in labels}
    primary, secondary = [], {"vs_tuned":[],"vs_initial":[]}
    regenerated, checked, replayed, exported = 0,0,0,0
    for i,point in enumerate(summary["points"]):
        sampler = make_sampler(c,point["p"])
        s,o,errors = sampler.sample(c.eval.max_shots,np.random.default_rng(point["seed"]))
        np.testing.assert_array_equal(parity(sampler.problem.H,errors),s)
        np.testing.assert_array_equal(parity(sampler.problem.L,errors),o)
        del errors
        with np.load(d/f"cases-p{i}.npz") as saved:
            np.testing.assert_array_equal(saved["syndromes"],s)
            np.testing.assert_array_equal(saved["observables"],o)
        regenerated += len(s)
        models = {f"nbp_{name}":load_model(d/f"checkpoint-{name}.pt",sampler.problem) for name in arms}
        payload = read_checkpoint(d/"checkpoint-parity.pt")
        models["nbp_initial"] = NeuralMinSum(sampler.problem,**payload["model_config"],dtype=models["nbp_parity"].w.dtype)
        decoders = {"bp":LdpcBP(sampler.problem,c.model.T,c.model.ms_scaling),
                    "bp_tuned":LdpcBP(sampler.problem,c.model.T,point["tuned_factor"])}
        n = min(c.eval.batch,len(s))
        for name,model in models.items():
            compiled = model.export_cpu()
            expected,conv = model.decode_batch(s[:n])
            actual,actual_conv = compiled.decode_batch(s[:n])
            np.testing.assert_array_equal(expected,actual)
            np.testing.assert_array_equal(conv,actual_conv)
            np.testing.assert_array_equal(model.last_iterations,compiled.last_iterations)
            decoders[name] = compiled
            exported += n
        with np.load(d/f"failed-p{i}.npz") as flags,np.load(d/f"logical-p{i}.npz") as logical:
            for name in labels:
                row = next(row for row in rows if row["stage1"]==name and float(row["p"])==point["p"])
                assert int(row["shots"])==len(s)==len(flags[name])
                assert int(row["stage1_failures"])==int(flags[name].sum())
                assert int(row["logical_failures"])==int(logical[name].sum())
                assert int(row["osd_calls"])==float(row["osd_rate"])==0
                rate = float(flags[name].mean())
                assert rate==point["decoders"][name]["fallback_demand"]
                means[name].append(rate)
                np.testing.assert_allclose(wilson(int(logical[name].sum()),len(s)),[float(row["ler_lo"]),float(row["ler_hi"])])
                a,b,_ = _inference(decoders[name],s[:n],o[:n],sampler.problem,n)
                np.testing.assert_array_equal(a,flags[name][:n])
                np.testing.assert_array_equal(b,logical[name][:n])
                checked += len(s)
                replayed += n
            test = mcnemar(flags["nbp_parity"],flags["nbp_bitwise"])
            assert all(test[k]==point["parity_vs_bitwise"][k] for k in test)
            primary.append(test["pvalue"])
            for family,baseline in (("vs_tuned","bp_tuned"),("vs_initial","nbp_initial")):
                for name in ("nbp_parity","nbp_bitwise"):
                    test = mcnemar(flags[baseline],flags[name])
                    assert all(test[k]==point[family][name][k] for k in test)
                    secondary[family].append(test["pvalue"])
    adjust = runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    np.testing.assert_allclose(adjust(primary),[p["parity_vs_bitwise"]["holm_pvalue"] for p in summary["points"]])
    for family,values in secondary.items():
        np.testing.assert_allclose(adjust(values),[p[family][name]["holm_pvalue"] for p in summary["points"] for name in ("nbp_parity","nbp_bitwise")])
    for name,values in means.items():
        assert float(np.mean(values))==summary["mean_fallback"][name]
    report = dict(checks_passed=True,matched_training_batches=summary["controls"]["matched_training_batches"],
                  validation_cases_replayed=validation_replayed,regenerated_comparison_cases=regenerated,
                  checked_decoder_flags=checked,independently_replayed_decodes=replayed,exact_export_shots=exported,
                  checked_csv_rows=len(rows),statistical_families_verified=[5,10,10],final_test_data_used=False,m6_complete=False)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
