"""Validate the stronger baseline on matched shots, then the remaining noise grid.

The predeclared low-p checkpoints are an exploratory development gate, not an
anytime-valid statistical test or evidence of LER equivalence between decoders.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from bbnbp.config import load_config
from bbnbp.evaluate import run
from bbnbp.metrics import wilson
from m3_validate_circuit import check_noise_grid

ROOT = Path(__file__).resolve().parents[1]


def read_rows(directory):
    with (Path(directory)/"results.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    shots = sum(int(row["shots"]) for row in rows)
    failures = sum(int(row["logical_failures"]) for row in rows)
    lo, hi = wilson(failures, shots)
    return dict(shots=shots, logical_failures=failures, ler=failures/shots,
                ler_lo=lo, ler_hi=hi,
                fallback_calls=sum(int(row["osd_calls"]) for row in rows),
                source_runs=[row["run_id"] for row in rows])


def validate_plan(plan):
    reference = ROOT/plan["reference_run"]
    old = load_config(reference/"config.yaml")
    full = load_config(ROOT/plan["full_config"])
    expected = replace(old, eval=replace(old.eval, osd=replace(old.eval.osd, method="osd_cs", order=7)))
    if full != expected or full.eval.purpose != "development":
        raise ValueError("Full baseline must change only OSD method/order from the reference")
    offset = 0
    for path in plan["low_p_segments"]:
        config = load_config(ROOT/path)
        expected = replace(full, noise=replace(full.noise, p_grid=(plan["low_p"],)),
                           eval=replace(full.eval, max_shots=config.eval.max_shots,
                                        stop_failures=config.eval.max_shots+1, sample_offset=offset))
        if config != expected:
            raise ValueError("Low-p segments must be contiguous with the original seed, batch, and circuit")
        offset += config.eval.max_shots
    tail = load_config(ROOT/plan["tail_config"])
    expected = replace(full, noise=replace(full.noise, p_grid=full.noise.p_grid[1:]),
                       eval=replace(full.eval, seed_offset=full.eval.seed_offset+1))
    if tail != expected or full.noise.p_grid[0] != plan["low_p"]:
        raise ValueError("Tail must preserve reference per-p seeds and cover the remaining noise grid")
    return reference, full, tail


def validate(plan_path, output_root):
    plan = yaml.safe_load(Path(plan_path).read_text())
    reference, full, tail = validate_plan(plan)
    bundle = Path(output_root)/datetime.now(timezone.utc).strftime("order7-%Y%m%dT%H%M%S%fZ")
    bundle.mkdir(parents=True, exist_ok=False)
    (bundle/"validation_plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=True))
    low_rows, checkpoints = [], []
    report = dict(reference_run=str(reference), purpose="M3 stronger-baseline development gate",
                  threshold=plan["max_ler"], checkpoints=checkpoints,
                  statistical_scope="exploratory Wilson checks; not anytime-valid inference")
    try:
        for index, path in enumerate(plan["low_p_segments"]):
            config = load_config(ROOT/path)
            print(f"Low-p segment: offset={config.eval.sample_offset}, shots={config.eval.max_shots}", flush=True)
            output = run(config, ["bp"], None, bundle, progress=True)
            low_rows.extend(read_rows(output))
            if index == 0:
                for artifact in ("circuit-p0.stim", "dem-p0.dem"):
                    if (output/artifact).read_bytes() != (reference/artifact).read_bytes():
                        raise AssertionError(f"Pilot changed the reference {artifact}")
                with np.load(output/"paired-p0.npz") as new, np.load(reference/"paired-p0.npz") as old:
                    np.testing.assert_array_equal(new["bp"], old["bp"])
                reference_row = read_rows(reference)[0]
                if int(reference_row["shots"]) != int(low_rows[0]["shots"]) or reference_row["seed"] != low_rows[0]["seed"]:
                    raise AssertionError("Pilot is not paired with the full 2048-shot reference")
                report["reference_pairing"] = dict(
                    shots=int(low_rows[0]["shots"]), seed=int(reference_row["seed"]),
                    original_logical_failures=int(reference_row["logical_failures"]),
                    stage1_flags_identical=True,
                    dem_sha256=hashlib.sha256((output/"dem-p0.dem").read_bytes()).hexdigest())
            point = summarize(low_rows)
            checkpoints.append(point)
            report["low_p_summary"] = point
            (bundle/"progress.json").write_text(json.dumps(report, indent=2)+"\n")
            print(f"Checkpoint: {point['logical_failures']}/{point['shots']}, "
                  f"Wilson [{point['ler_lo']:.6f}, {point['ler_hi']:.6f}]", flush=True)
            if point["ler_hi"] < plan["max_ler"]:
                report["low_p_status"] = "pass"
                break
            if point["ler_lo"] >= plan["max_ler"]:
                report["low_p_status"] = "fail"
                break
        else:
            report["low_p_status"] = "inconclusive_at_cap"

        if report["low_p_status"] == "pass":
            output = run(tail, ["bp"], None, bundle, progress=True)
            tail_rows = sorted(read_rows(output), key=lambda row: float(row["p"]))
            curve = [dict(p=plan["low_p"], **report["low_p_summary"])]
            curve.extend(dict(p=float(row["p"]), **summarize([row])) for row in tail_rows)
            report["curve"] = curve
            report["monotonic_ler"] = all(a["ler"] <= b["ler"] for a, b in zip(curve, curve[1:]))
            report["noise_grid_checks"] = check_noise_grid(full)
            report["passed"] = report["monotonic_ler"]
        else:
            report["passed"] = False
            report["remaining_grid"] = "not run because the low-p gate did not pass"
        (bundle/"m3_gate.json").write_text(json.dumps(report, indent=2)+"\n")
        with (ROOT/"NOTES.md").open("a") as f:
            f.write(f"\nM3 order-7 validation {'passed' if report['passed'] else 'not accepted'}: "
                    f"`{bundle}`. Low-p summary: {report['low_p_summary']}.\n")
        print(f"M3 order-7 gate: {'PASS' if report['passed'] else 'NOT ACCEPTED'}; {bundle}", flush=True)
        return report["passed"]
    except BaseException as exc:
        (bundle/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        with (ROOT/"NOTES.md").open("a") as f:
            f.write(f"\nM3 order-7 execution failed in `{bundle}`: {type(exc).__name__}: {exc}.\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/m3_order7_validation.yaml")
    parser.add_argument("--output-root", default="results/m3")
    args = parser.parse_args()
    raise SystemExit(0 if validate(args.plan, args.output_root) else 1)
