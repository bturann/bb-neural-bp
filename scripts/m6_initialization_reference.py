"""Separate learned improvement from a stronger untrained neural starting point."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bbnbp.checkpoints import read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.metrics import mcnemar, wilson
from bbnbp.io import append_results
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    directory = args.comparison
    output = directory/"initialization_reference.json"
    if output.exists():
        parser.error("Refusing to overwrite reference evidence")
    config = load_config(directory/"config.yaml")
    payload = read_checkpoint(directory/"checkpoint.pt")
    torch.set_num_threads(config.eval.threads)
    summary = json.loads((directory/"summary.json").read_text())
    assert hashlib.sha256((directory/"checkpoint.pt").read_bytes()).hexdigest() == summary["checkpoint_sha256"]
    with (directory/"results.csv").open(newline="") as stream:
        originals = list(csv.DictReader(stream))
    rows, csv_rows = [], []
    for index, point in enumerate(summary["points"]):
        sampler = make_sampler(config, point["p"])
        dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}[payload["dtype"]]
        initial = NeuralMinSum(sampler.problem, **payload["model_config"], dtype=dtype)
        compiled = initial.export_cpu()
        with np.load(directory/f"cases-p{index}.npz") as cases:
            s, o = cases["syndromes"], cases["observables"]
        n = min(config.eval.batch, len(s))
        expected, expected_conv = initial.decode_batch(s[:n])
        actual, actual_conv = compiled.decode_batch(s[:n])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_conv, expected_conv)
        np.testing.assert_array_equal(compiled.last_iterations, initial.last_iterations)
        failed, logical, iterations = _inference(compiled, s, o, sampler.problem, config.eval.batch)
        row = dict(next(row for row in originals if row["stage1"] == "nbp" and float(row["p"]) == point["p"]))
        lo, hi = wilson(int(logical.sum()), len(s))
        row.update(timestamp=datetime.now(timezone.utc).isoformat(), decoder="nbp_initial_only", stage1="nbp_initial",
                   stage1_failures=int(failed.sum()), logical_failures=int(logical.sum()), ler=float(logical.mean()),
                   ler_lo=lo, ler_hi=hi, ler_stage1=float(logical.mean()), ms_scaling=payload["model_config"]["ms_scaling"])
        csv_rows.append(row)
        with np.load(directory/f"stage1-failed-p{index}.npz") as selected:
            trained, tuned = selected["nbp"], selected["bp_tuned"]
            rows.append(dict(p=point["p"], shots=len(s), initial_fallback=float(failed.mean()),
                             selected_fallback=float(trained.mean()), tuned_fallback=float(tuned.mean()),
                             learned_vs_initial=mcnemar(failed, trained), initial_vs_tuned=mcnemar(tuned, failed),
                             initial_mean_iterations=iterations, exact_export_shots=n))
        np.savez_compressed(directory/f"initialization-p{index}.npz", failed=failed, logical=logical)
        print(f"p={point['p']:g}: initial={rows[-1]['initial_fallback']:.6f}, selected={rows[-1]['selected_fallback']:.6f}", flush=True)
    import runpy
    adjust = runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    adjusted = adjust([row["learned_vs_initial"]["pvalue"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["learned_vs_initial"]["holm_pvalue"] = value
    means = {name: float(np.mean([row[f"{name}_fallback"] for row in rows])) for name in ("initial", "selected", "tuned")}
    report = dict(purpose="matched_initialization_reference_on_fresh_development_cases", rows=rows,
                  checkpoint_sha256=summary["checkpoint_sha256"],
                  initialization=payload["model_config"], selected_step=payload["step"], mean_fallback=means,
                  learning_improvement_percentage_points=100*(means["initial"]-means["selected"]),
                  learning_signal=means["selected"] < means["initial"] and any(
                      row["selected_fallback"] < row["initial_fallback"] and row["learned_vs_initial"]["holm_pvalue"] < 0.05
                      for row in rows),
                  final_test_data_used=False, m6_complete=False,
                  scope="Single-seed development initialization diagnostic; fresh cases remain separate from "
                        "checkpoint selection. Initialization also sets the frozen hard-mining model. "
                        "This is not a paired training-data isolation of initialization alone.")
    append_results(directory/"initialization_results.csv", csv_rows)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("selected_step", "mean_fallback", "learning_improvement_percentage_points", "learning_signal")}, indent=2))


if __name__ == "__main__":
    main()
