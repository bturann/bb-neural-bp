"""Verify fresh-data pilot-check artifacts and independently replay decoder subsets."""

import argparse
import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch

from bbnbp.checkpoints import load_model
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    output = args.directory/"audit.json"
    if output.exists():
        parser.error("Refusing to overwrite audit evidence")
    c = load_config(args.directory/"config.yaml")
    torch.set_num_threads(c.eval.threads)
    directory = args.directory/"comparison"
    summary = json.loads((directory/"summary.json").read_text())
    status = json.loads((args.directory/"status.json").read_text())
    assert status["complete"] and not status["final_test_data_used"]
    checkpoint = Path(c.eval.checkpoints[0])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == summary["checkpoint_sha256"]
    assert (directory/"checkpoint.pt").read_bytes() == checkpoint.read_bytes()
    source = directory/"executed_compare_source.py"
    manifest = json.loads((directory/"source_manifest.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["scripts/m6_compare_stage1.py"]
    with (directory/"results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (args.directory/"tuned_scaling.csv").open(newline="") as stream:
        tuning = list(csv.DictReader(stream))
    assert len(tuning) == 55 and len(rows) == 15
    checked_flags, regenerated_cases, replayed_decodes, torch_numba = 0, 0, 0, 0
    means = {name: [] for name in ("bp", "bp_tuned", "nbp")}
    pvalues = []
    for i, point in enumerate(summary["points"]):
        p = point["p"]
        candidates = [row for row in tuning if float(row["p"]) == p]
        best = min(candidates, key=lambda row: (int(row["nonconverged"]), float(row["ler_stage1"]), float(row["ms_scaling"])))
        assert best["selected"] == "True" and float(best["ms_scaling"]) == point["tuned_factor"]
        sampler = make_sampler(c, p)
        s, o, errors = sampler.sample(c.eval.max_shots, np.random.default_rng(point["seed"]))
        np.testing.assert_array_equal(parity(sampler.problem.H, errors), s)
        np.testing.assert_array_equal(parity(sampler.problem.L, errors), o)
        del errors
        with np.load(directory/f"cases-p{i}.npz") as cases:
            np.testing.assert_array_equal(cases["syndromes"], s)
            np.testing.assert_array_equal(cases["observables"], o)
        assert hashlib.sha256(s.tobytes()).hexdigest() == point["syndrome_sha256"]
        regenerated_cases += len(s)
        n = min(c.eval.batch, len(s))
        model = load_model(checkpoint, sampler.problem)
        compiled = model.export_cpu()
        expected, conv = model.decode_batch(s[:n])
        actual, actual_conv = compiled.decode_batch(s[:n])
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(conv, actual_conv)
        np.testing.assert_array_equal(model.last_iterations, compiled.last_iterations)
        torch_numba += n
        decoders = {"bp": LdpcBP(sampler.problem, c.model.T, c.model.ms_scaling),
                    "bp_tuned": LdpcBP(sampler.problem, c.model.T, point["tuned_factor"]), "nbp": compiled}
        with np.load(directory/f"stage1-failed-p{i}.npz") as flags, np.load(directory/f"logical-stage1-p{i}.npz") as logical:
            for name, decoder in decoders.items():
                row = next(row for row in rows if float(row["p"]) == p and row["stage1"] == name)
                assert len(flags[name]) == int(row["shots"]) == c.eval.max_shots
                assert int(flags[name].sum()) == int(row["stage1_failures"])
                assert int(logical[name].sum()) == int(row["logical_failures"])
                assert float(row["osd_rate"]) == int(row["osd_calls"]) == 0
                rate = float(flags[name].mean())
                assert rate == point["decoders"][name]["fallback_demand"]
                means[name].append(rate)
                np.testing.assert_allclose(wilson(int(logical[name].sum()), len(s)), [float(row["ler_lo"]), float(row["ler_hi"])])
                failed, replay_logical, _ = _inference(decoder, s[:n], o[:n], sampler.problem, n)
                np.testing.assert_array_equal(failed, flags[name][:n])
                np.testing.assert_array_equal(replay_logical, logical[name][:n])
                checked_flags += len(s)
                replayed_decodes += n
            paired = mcnemar(flags["bp_tuned"], flags["nbp"])
            for key, value in paired.items():
                assert value == point["paired_vs_tuned"][key]
            pvalues.append(paired["pvalue"])
    adjust = runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    np.testing.assert_allclose(adjust(pvalues), [p["paired_vs_tuned"]["holm_pvalue"] for p in summary["points"]])
    for name, values in means.items():
        assert float(np.mean(values)) == summary["mean_fallback_demand"][name]
    report = dict(checks_passed=True, actual_tuning_rows=len(tuning), checked_result_rows=len(rows),
                  regenerated_comparison_cases=regenerated_cases, checked_decoder_flags=checked_flags,
                  independently_replayed_decodes=replayed_decodes, torch_numba_exact_shots=torch_numba,
                  checkpoint_sha256=summary["checkpoint_sha256"], statistical_summary_verified=True,
                  final_test_data_used=False, m6_complete=False,
                  optimization_screen_passed=summary["supports_cpu_optimization"])
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
