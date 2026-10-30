"""Run the development circuit baseline and write the M3 gate's measured evidence."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bbnbp.config import load_config
from bbnbp.evaluate import run
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler

ROOT = Path(__file__).resolve().parents[1]


def check_noise_grid(config):
    checks = []
    first_h, first_l = None, None
    for i, p in enumerate(config.noise.p_grid):
        sampler = make_sampler(config, p)
        problem = sampler.problem
        if first_h is None:
            first_h, first_l = problem.H, problem.L
        if problem.H.shape != first_h.shape or (problem.H != first_h).nnz or (problem.L != first_l).nnz:
            raise AssertionError("DEM H/L structure changed across the noise grid")
        entropy = [config.seed, config.eval.seed_offset, i, 3]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        for _ in range(20):
            s, o, e = sampler.sample(500, rng)
            if not np.array_equal(s, parity(problem.H, e)) or not np.array_equal(o, parity(problem.L, e)):
                raise AssertionError("Circuit sampler violates its exact fault-label identities")
        checks.append(dict(p=p, exact_label_shots=10000, seed_entropy=entropy,
                           H_shape=problem.H.shape, L_shape=problem.L.shape,
                           H_nnz=problem.H.nnz, L_nnz=problem.L.nnz,
                           max_check_degree=int(np.diff(problem.H.indptr).max()),
                           priors_min=float(problem.priors.min()), priors_max=float(problem.priors.max())))
    return checks


def validate(config, output_root):
    if config.noise.model != "circuit_level" or config.eval.purpose != "development":
        raise ValueError("M3 acceptance uses an explicit circuit-level development configuration")
    output = run(config, ["bp"], None, output_root)
    with (output/"results.csv").open() as f:
        rows = sorted(csv.DictReader(f), key=lambda row: float(row["p"]))
    checks = check_noise_grid(config)
    monotonic = all(float(a["ler"]) <= float(b["ler"]) for a, b in zip(rows, rows[1:]))
    low = rows[0]
    has_low_point = float(low["p"]) == 0.001
    low_rate = has_low_point and float(low["ler_hi"]) < 0.01
    passed = monotonic and low_rate and all(check["L_shape"][0] == 12 for check in checks)
    report = dict(passed=passed, config_hash=config.config_hash,
                  noise_grid_checks=checks, matrices_identical_across_p=True,
                  monotonic_ler=monotonic, low_p=low["p"], low_p_ler=low["ler"],
                  low_p_ler_hi=low["ler_hi"], low_p_upper_below_one_percent=low_rate,
                  low_p_rule="95% Wilson upper bound at p=0.001 < 0.01",
                  osd_method=config.eval.osd.method, osd_order=config.eval.osd.order,
                  purpose="M3 development baseline; not the final E1 experiment")
    (output/"m3_gate.json").write_text(json.dumps(report, indent=2)+"\n")
    if not passed:
        with (ROOT/"NOTES.md").open("a") as f:
            f.write(f"\nM3 gate failed for `{output}`: monotonic={monotonic}, "
                    f"low-p LER={low['ler']} (Wilson upper {low['ler_hi']}). "
                    "Stop at the M3 gate and review this measured evidence.\n")
    print(f"M3 gate: {'PASS' if passed else 'FAIL'}; evidence: {output}", flush=True)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cl_bb72_m3.yaml")
    parser.add_argument("--output-root", default="results/m3")
    args = parser.parse_args()
    raise SystemExit(0 if validate(load_config(args.config), args.output_root) else 1)
