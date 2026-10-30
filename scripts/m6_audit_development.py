"""Audit retained M6 development artifacts; this cannot establish M6 acceptance."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bbnbp.checkpoints import load_model
from bbnbp.config import load_config
from bbnbp.metrics import wilson
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference, validation_sets


def only_run(directory):
    runs = [p for p in directory.iterdir() if p.is_dir()]
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one development run in {directory}")
    return runs[0]


def main():
    torch.set_num_threads(1)
    root = Path("results/m6_development")
    destination = root/"audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    training = only_run(root/"training")
    report = json.loads((training/"m5_gate.json").read_text())
    assert report["purpose"] == "smoke" and not report["passed"]
    checkpoint = Path(report["checkpoint_directory"])/"best.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["best_checkpoint_sha256"]
    config = load_config(training/"config.yaml")
    samplers = {p: make_sampler(config, p) for p in config.noise.p_grid}
    validation = validation_sets(config, samplers)
    verified_validation = 0
    with np.load(training/"best_validation_flags.npz") as flags:
        for index, row in enumerate(report["rows"]):
            p = row["p"]
            problem = samplers[p].problem
            model = load_model(checkpoint, problem).export_cpu()
            s, o, _ = validation[p]
            failed, logical, _ = _inference(model, s, o, problem, config.train.runtime.val_batch)
            np.testing.assert_array_equal(failed, flags[f"nbp-p{index}"])
            np.testing.assert_array_equal(logical, flags[f"logical-p{index}"])
            assert int(failed.sum()) == row["nonconverged"]
            verified_validation += len(s)
    checked_rows, checked_timings = 0, 0
    evaluations = {}
    for category in ("evaluation", "latency"):
        directory = only_run(root/category)
        evaluations[category] = str(directory)
        c = load_config(directory/"config.yaml")
        assert c.eval.purpose == "development"
        with (directory/"results.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        for index, p in enumerate(c.noise.p_grid):
            with np.load(directory/f"paired-p{index}.npz") as calls, np.load(directory/f"logical-p{index}.npz") as logical:
                for row in (r for r in rows if float(r["p"]) == p):
                    name = row["stage1"]
                    shots, failures = int(row["shots"]), int(row["logical_failures"])
                    assert len(calls[name]) == len(logical[name]) == shots
                    assert int(calls[name].sum()) == int(row["osd_calls"])
                    assert int(logical[name].sum()) == failures
                    np.testing.assert_allclose([float(row["ler_lo"]), float(row["ler_hi"])], wilson(failures, shots))
                    checked_rows += 1
                    if c.eval.latency_shots:
                        with np.load(directory/f"latency-components-p{index}-{name}.npz") as timing:
                            assert np.isfinite(timing["total_us"]).all()
                            assert (timing["overhead_us"] >= 0).all()
                            np.testing.assert_allclose(timing["total_us"], timing["stage1_us"]+timing["osd_us"]+timing["overhead_us"])
                            assert (timing["osd_us"][~timing["osd_used"]] == 0).all()
                            checked_timings += len(timing["total_us"])
    export = json.loads((root/"export_verification.json").read_text())
    assert export["passed"] and export["shots"] == 10000
    assert export["checkpoint_sha256"] == report["best_checkpoint_sha256"]
    assert export["hashes"]["torch"] == export["hashes"]["numba"]
    summary = dict(purpose="development_artifact_audit", checks_passed=True,
                   m6_complete=False, final_test_data_used=False,
                   checkpoint_sha256=report["best_checkpoint_sha256"],
                   validation_flags_reproduced=verified_validation,
                   evaluation_rows_checked=checked_rows, timing_samples_checked=checked_timings,
                   trained_circuit_export_shots=export["shots"], evaluations=evaluations,
                   caveat="Tiny development accuracy samples; latency ran alongside other development "
                          "work. These measurements support mechanics checks only, not performance claims.")
    destination.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
