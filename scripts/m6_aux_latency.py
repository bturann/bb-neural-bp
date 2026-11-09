"""Bounded, paired CPU pilot with fresh uncached fallback calls for every method."""

import argparse
import hashlib
import json
import platform
import runpy
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP, LdpcBPOSD
from bbnbp.decoders.pipeline import TwoStageDecoder
from bbnbp.evaluate import cpu_model_label
from bbnbp.io import source_manifest
from bbnbp.metrics import latency_stats
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def paired_bootstrap(first, second, rng, repeats):
    """Resample paired cases; report selected-minus-baseline differences."""
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or first.shape != second.shape or not len(first):
        raise ValueError("Require nonempty paired timings")
    differences = np.empty((repeats, 2))
    for i in range(repeats):
        indices = rng.integers(0, len(first), len(first))
        a, b = first[indices], second[indices]
        differences[i] = b.mean()-a.mean(), np.percentile(b, 99)-np.percentile(a, 99)
    return {name: dict(difference_us=float(actual), nominal_95_bootstrap_lo_us=float(lo),
                      nominal_95_bootstrap_hi_us=float(hi))
            for name, actual, (lo, hi) in zip(
                ("mean", "p99"),
                (second.mean()-first.mean(), np.percentile(second, 99)-np.percentile(first, 99)),
                np.percentile(differences, [2.5, 97.5], axis=0).T)}


def measure_point(confirmation, out, index, settings, started, long_controls=None):
    c = load_config(confirmation/"config.yaml")
    sampler = make_sampler(c, c.noise.p_grid[index])
    names = settings["latency_models"]
    selection = json.loads((confirmation/"selection.json").read_text())
    controls = json.loads((confirmation/f"controls-p{index}.json").read_text())
    builder = runpy.run_path(str(confirmation/"models_source.py"))["make_decoders"]
    nodes = builder(sampler.problem, confirmation/"selected-source.pt", selection["model_design"], controls)
    if long_controls:
        root=long_controls[index]
        challenge=json.loads((root/"summary.json").read_text())
        if challenge["p"]!=c.noise.p_grid[index]:raise ValueError("Long-control noise point mismatch")
        frozen=json.loads((root/"selection.json").read_text())
        for key,choice in frozen["choices"].items():
            nodes[f"bp{key}"]=LdpcBP(sampler.problem,int(key),choice["factor"])
    spec = c.eval.osd
    # Separate fallback objects and actual calls, including when inputs are identical.
    models = {name: TwoStageDecoder(nodes[name], LdpcBPOSD(
        sampler.problem, spec.bp_iters, spec.ms_scaling, spec.method, spec.order)) for name in names}
    warmup, target = settings["latency_warmup"], settings["latency_shots"]
    seed = c.seed+settings["latency_seed_offset"]+index
    s, o, e = sampler.sample(target+warmup, np.random.default_rng(seed))
    np.testing.assert_array_equal(parity(sampler.problem.H, e), s)
    np.testing.assert_array_equal(parity(sampler.problem.L, e), o)
    order_rng = np.random.default_rng(np.random.SeedSequence(settings["latency_order_entropy"]+[index]))
    order = np.stack([order_rng.permutation(len(names)) for _ in range(len(s))])
    np.savez_compressed(out/f"cases-p{index}.npz", syndromes=s, observables=o, faults=e, order=order)
    latest = datetime.fromisoformat(settings["finalization_utc"])
    records, complete_cases, stop_reason = [], 0, "fixed_target"
    actual_calls = 0
    with (out/f"raw-p{index}.jsonl").open("w") as stream:
        for i, syndrome in enumerate(s):
            case_records = []
            for k in order[i]:
                if (time.perf_counter()-started >= settings["latency_wall_seconds"]
                        or datetime.now(timezone.utc) >= latest):
                    stop_reason = "time_budget"
                    break
                name = names[k]
                begin = time.perf_counter_ns()
                result = models[name].decode_timed_one(syndrome)
                elapsed = (time.perf_counter_ns()-begin)/1000
                used = bool(result["osd_used"][0])
                actual_calls += int(used)
                row = dict(case=i, warmup=i<warmup, method=name, order_position=len(case_records),
                    total_us=elapsed, stage1_us=result["stage1_ns"]/1000, osd_us=result["osd_ns"]/1000,
                    overhead_us=elapsed-(result["stage1_ns"]+result["osd_ns"])/1000,
                    fallback_called=used, internal_osd_executed=bool(result["osd_executed"][0]),
                    stage1_failed=not bool(result["stage1_converged"][0]),
                    stage1_logical=bool(used or (parity(sampler.problem.L, result["stage1_e_hat"])[0]!=o[i]).any()),
                    logical=bool((parity(sampler.problem.L, result["e_hat"])[0]!=o[i]).any()),
                    syndrome_valid=bool((parity(sampler.problem.H, result["e_hat"])[0]==syndrome).all()),
                    iterations=int(models[name].stage1.last_iterations[0]))
                stream.write(json.dumps(row)+"\n"); stream.flush()
                case_records.append(row)
            records.extend(case_records)
            if len(case_records)!=len(names): break
            complete_cases += 1
            if complete_cases%32==0:
                print(f"latency p={c.noise.p_grid[index]:g}: paired cases={complete_cases}/{len(s)}; wall={time.perf_counter()-started:.1f}s", flush=True)
    measured = max(0, complete_cases-warmup)
    rows, arrays = [], {}
    for name in names:
        kept = [row for row in records if row["method"]==name and warmup<=row["case"]<complete_cases]
        if not kept: continue
        values = np.array([row["total_us"] for row in kept])
        arrays[name] = values
        rows.append(dict(method=name, shots=len(values), **latency_stats(values),
            mean_stage1_us=float(np.mean([row["stage1_us"] for row in kept])),
            mean_fallback_us=float(np.mean([row["osd_us"] for row in kept])),
            fallback_calls=sum(row["fallback_called"] for row in kept),
            internal_osd_executions=sum(row["internal_osd_executed"] for row in kept),
            logical_failures=sum(row["logical"] for row in kept)))
    report = dict(p=c.noise.p_grid[index], seed=seed, measured_paired_cases=measured,
        completed=measured==target, warmup_complete=complete_cases>=warmup,
        stopping=stop_reason, all_actual_fallback_calls_including_warmup_and_partial=actual_calls,
        complete_cases_including_warmup=complete_cases, models=rows)
    # Bootstrap is deliberately postponed until all timed measurements finish.
    (out/f"point-p{index}.json").write_text(json.dumps(report, indent=2)+"\n")
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation", type=Path)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--replications", nargs=2, type=Path, required=True)
    parser.add_argument("--settings", default="configs/m6/cpu_aux_assessment.yaml")
    parser.add_argument("--long-controls", nargs=2, type=Path)
    args = parser.parse_args()
    # Prevent our known CPU jobs from overlapping timing; external system load is uncontrolled.
    for path in [args.accuracy, *[root/"struct_standard" for root in args.replications]]:
        summary = json.loads((path/"summary.json").read_text())
        if path != args.accuracy and not summary["completed"]:
            raise ValueError(f"Replication incomplete: {path}")
    confirmation = args.confirmation
    summary = json.loads((confirmation/"summary.json").read_text())
    if hashlib.sha256((confirmation/"selected-source.pt").read_bytes()).hexdigest()!=summary["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    settings = yaml.safe_load(Path(args.settings).read_text())
    receipt_name=settings.get("latency_receipt_name","latency_receipt.json")
    if Path(receipt_name).name!=receipt_name:raise ValueError("Receipt must be a filename")
    if (confirmation/receipt_name).exists(): raise ValueError("This latency pilot was already requested")
    d = Path(settings.get("latency_output_root","results/m6_aux_latency"))/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, d/"executed_source.py")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    (d/"machine.json").write_text(json.dumps(dict(cpu=cpu_model_label(), platform=platform.platform(),
        threads=1, known_training_and_accuracy_jobs_finished=True,
        other_system_load="uncontrolled", confirmation=str(confirmation), accuracy=str(args.accuracy),
        replications=list(map(str,args.replications)),long_controls=list(map(str,args.long_controls or []))), indent=2)+"\n")
    (confirmation/receipt_name).write_text(json.dumps(dict(directory=str(d),
        requested_at_utc=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")
    torch.set_num_threads(1); set_num_threads(1)
    print(f"Uncached paired latency pilot: {d}", flush=True)
    started = time.perf_counter()
    try:
        points, arrays = [], []
        for index in range(2):
            point, values = measure_point(confirmation,d,index,settings,started,args.long_controls)
            points.append(point); arrays.append(values)
        measurement_wall = time.perf_counter()-started
        for index, (point, values) in enumerate(zip(points, arrays)):
            if "selected" not in values: continue
            point["paired_nominal_bootstrap"] = {
                name: paired_bootstrap(values[name],values["selected"],np.random.default_rng(
                    np.random.SeedSequence(settings["latency_bootstrap_entropy"]+[index,k])),
                    settings["latency_bootstrap_repeats"])
                for k,name in enumerate(settings["latency_models"]) if name!="selected" and name in values}
        report = dict(points=points, completed=all(point["completed"] for point in points),
            measurement_wall_s=measurement_wall, elapsed_s=time.perf_counter()-started,
            scope=settings.get("latency_scope","Auxiliary single-thread paired CPU pilot, fresh uncached fallback calls, randomized per-case method order. Nominal bootstrap intervals are descriptive, not multiplicity-corrected or main E2 acceptance; 512 samples weakly characterize p99."),
            final_test_data_used=False, gpu_seconds=0, m6_complete=False)
        (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n"); raise


if __name__=="__main__": main()
