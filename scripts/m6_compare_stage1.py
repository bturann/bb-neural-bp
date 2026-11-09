"""Compare a frozen pilot checkpoint with independently tuned BP on fresh dev shots."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from numba import set_num_threads

from bbnbp.checkpoints import load_model, read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.evaluate import cpu_model_label, tuned_factor
from bbnbp.io import append_results, git_sha, source_manifest
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(pvalues))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(pvalues)-rank)*pvalues[index])
        adjusted[index] = min(1.0, previous)
    return adjusted.tolist()


def validate_tuning_table(config, path):
    """Accept compatible earlier tuning without rewriting its original provenance."""
    path = Path(path)
    tuning_config = load_config(path.with_suffix(".config.yaml"))
    if (tuning_config.code != config.code or tuning_config.noise != config.noise
            or tuning_config.sweep != config.sweep or tuning_config.seed != config.seed):
        raise ValueError("Tuning and comparison problem/sampling configurations differ")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(p, T, factor) for p in config.noise.p_grid for T in config.sweep.iterations
                for factor in config.sweep.factors}
    actual = {(float(row["p"]), int(row["T"]), float(row["ms_scaling"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Tuning table does not contain the complete configured factor grid")
    for row in rows:
        index = config.noise.p_grid.index(float(row["p"]))
        if (int(row["seed"]) != config.seed+config.sweep.seed_offset+index
                or int(row["shots"]) != config.sweep.shots or row["config_hash"] != tuning_config.config_hash):
            raise ValueError("Actual tuning seeds, shot counts or configuration differ from their source provenance")
    return tuning_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuning-table", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = Path(config.eval.checkpoints[0])
    payload = read_checkpoint(checkpoint)
    if payload["model_config"]["T"] != config.model.T:
        raise ValueError("Checkpoint and baseline iteration budgets differ")
    training = payload["config"]
    runtime = training["train"]["runtime"]
    validation_seeds = {training["seed"]+runtime["validation_seed_offset"]+i
                        for i in range(len(training["noise"]["p_grid"]))}
    tuning_seeds = {config.seed+config.sweep.seed_offset+i for i in range(len(config.noise.p_grid))}
    comparison_seeds = {config.seed+config.eval.seed_offset+i for i in range(len(config.noise.p_grid))}
    final_seeds = {config.seed+10000+i for i in range(len(config.noise.p_grid))}
    regions = [validation_seeds, tuning_seeds, comparison_seeds, final_seeds]
    if any(regions[i] & regions[j] for i in range(len(regions)) for j in range(i)):
        raise ValueError("Tuning, checkpoint validation, comparison and final streams must be disjoint")
    if config.eval.purpose != "development" or config.model.T != config.eval.stage1_iters:
        raise ValueError("Expected a matched-budget development comparison")
    validate_tuning_table(config, args.tuning_table)
    torch.set_num_threads(config.eval.threads)
    set_num_threads(config.eval.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    config.save(args.output/"config.yaml")
    shutil.copyfile(args.tuning_table, args.output/"tuned_scaling.csv")
    shutil.copyfile(Path(args.tuning_table).with_suffix(".config.yaml"), args.output/"tuned_scaling.config.yaml")
    shutil.copyfile(__file__, args.output/"executed_compare_source.py")
    shutil.copyfile(checkpoint, args.output/"checkpoint.pt")
    (args.output/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    timestamp = datetime.now(timezone.utc)
    sha = git_sha()
    rows, comparison = [], []
    started = time.perf_counter()
    try:
        for index, p in enumerate(config.noise.p_grid):
            sampler = make_sampler(config, p)
            seed = config.seed+config.eval.seed_offset+index
            s, observations, errors = sampler.sample(config.eval.max_shots, np.random.default_rng(seed))
            del errors
            model = load_model(checkpoint, sampler.problem)
            compiled = model.export_cpu()
            verification_shots = min(config.eval.batch, len(s))
            expected, expected_converged = model.decode_batch(s[:verification_shots])
            actual, actual_converged = compiled.decode_batch(s[:verification_shots])
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_converged, expected_converged)
            np.testing.assert_array_equal(compiled.last_iterations, model.last_iterations)
            factor = tuned_factor(args.tuning_table, config, p)
            decoders = {"bp": LdpcBP(sampler.problem, config.model.T, config.model.ms_scaling),
                        "bp_tuned": LdpcBP(sampler.problem, config.model.T, factor),
                        "nbp": compiled}
            flags, logical_flags, point_rows = {}, {}, {}
            for name, decoder in decoders.items():
                failed, logical, iterations = _inference(decoder, s, observations, sampler.problem, config.eval.batch)
                shots, failures = len(s), int(logical.sum())
                lo, hi = wilson(failures, shots)
                row = dict(run_id=args.output.parent.name, git_sha=sha, config_hash=config.config_hash,
                           timestamp=timestamp.isoformat(), code=config.code.name, noise_model=config.noise.model,
                           rounds=config.noise.rounds, basis=config.noise.basis, p=p,
                           decoder=name+"_only", stage1=name, T=config.model.T,
                           weight_sharing=model.sharing if name == "nbp" else "scalar",
                           osd_method="", osd_order="",
                           ms_scaling=payload["model_config"]["ms_scaling"] if name == "nbp" else factor if name == "bp_tuned" else config.model.ms_scaling,
                           shots=shots, stage1_failures=int(failed.sum()), osd_calls=0, osd_rate=0,
                           logical_failures=failures, ler=failures/shots, ler_lo=lo, ler_hi=hi,
                           ler_stage1=failures/shots, lat_mean_us="", lat_p50_us="", lat_p99_us="", lat_max_us="",
                           lat_threads=1, device=f"CPU {cpu_model_label(config.eval.cpu_model)}; {platform.platform()}", seed=seed)
                rows.append(row)
                point_rows[name] = dict(failures=int(failed.sum()), fallback_demand=float(failed.mean()),
                                        fallback_wilson=list(wilson(int(failed.sum()), shots)),
                                        mean_iterations=iterations)
                flags[name], logical_flags[name] = failed, logical
                print(f"p={p:g} {name}: fallback demand={failed.mean():.6f}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
            append_results(args.output/"results.csv", rows[-3:])
            np.savez_compressed(args.output/f"cases-p{index}.npz", syndromes=s, observables=observations)
            np.savez_compressed(args.output/f"stage1-failed-p{index}.npz", **flags)
            np.savez_compressed(args.output/f"logical-stage1-p{index}.npz", **logical_flags)
            comparison.append(dict(p=p, seed=seed, shots=len(s), tuned_factor=factor, decoders=point_rows,
                                   torch_numba_checked_shots=verification_shots,
                                   syndrome_sha256=hashlib.sha256(s.tobytes()).hexdigest(),
                                   paired_vs_tuned=mcnemar(flags["bp_tuned"], flags["nbp"])))
        adjusted = holm([point["paired_vs_tuned"]["pvalue"] for point in comparison])
        for point, value in zip(comparison, adjusted):
            point["paired_vs_tuned"]["holm_pvalue"] = value
        means = {name: float(np.mean([point["decoders"][name]["fallback_demand"] for point in comparison]))
                 for name in ("bp", "bp_tuned", "nbp")}
        improvements = [point for point in comparison if point["decoders"]["nbp"]["fallback_demand"] < point["decoders"]["bp_tuned"]["fallback_demand"]]
        regressions = [point for point in comparison if point["decoders"]["nbp"]["fallback_demand"] > point["decoders"]["bp_tuned"]["fallback_demand"]]
        evidence = any(point["paired_vs_tuned"]["holm_pvalue"] < 0.05 for point in improvements)
        report = dict(purpose="independent_stage1_development_comparison", elapsed_s=time.perf_counter()-started,
                      checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      training_seed=training["seed"], checkpoint_step=payload["step"], points=comparison,
                      mean_fallback_demand=means, strict_improvement_points=len(improvements), strict_regression_points=len(regressions),
                      absolute_reduction_percentage_points=100*(means["bp_tuned"]-means["nbp"]),
                      supports_cpu_optimization=means["nbp"] < means["bp_tuned"] and evidence,
                      decision_rule="Predeclared development screen: lower grid-mean fallback demand and at least "
                                    f"one improvement with paired McNemar Holm-adjusted p<.05 across {len(comparison)} points; report all regressions.",
                      final_test_data_used=False, m6_complete=False,
                      scope=f"Frozen single-seed checkpoint; {len(config.sweep.factors)}-factor T={config.model.T} tuning on independent "
                            f"{config.sweep.shots} shots/p; fresh paired {config.eval.max_shots} shots/p. "
                            "Stage-1 demand only. CSV osd_rate=0 because no stage-2 calls "
                            "are made; fallback demand=stage1_failures/shots. No two-stage LER or latency claim.")
        (args.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({key: report[key] for key in ("mean_fallback_demand", "strict_improvement_points", "strict_regression_points", "supports_cpu_optimization")}, indent=2), flush=True)
    except BaseException as exc:
        (args.output/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
