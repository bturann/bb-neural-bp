"""Paired baseline evaluation with shared shots and failure-count stopping."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from .config import load_config
from .decoders.bp_ldpc import LdpcBP, LdpcBPOSD
from .decoders.pipeline import TwoStageDecoder
from .io import append_results, git_sha, source_manifest
from .metrics import latency_stats, mcnemar, wilson
from .noise.base import parity
from .noise.factory import make_sampler


def tuned_factor(table, config, p):
    with Path(table).open(newline="") as f:
        rows = list(csv.DictReader(f))
    # Check the data actually used for tuning, including unselected candidates.
    # The caller may supply a table produced from another configuration.
    tuning_seeds = {int(row["seed"]) for row in rows}
    evaluation_seeds = {config.seed+config.eval.seed_offset+i
                        for i in range(len(config.noise.p_grid))}
    overlap = tuning_seeds & evaluation_seeds
    if overlap:
        raise ValueError(f"Tuning and evaluation seeds overlap: {sorted(overlap)}")
    candidates = [r for r in rows if r["code"] == config.code.name
                  and r["noise_model"] == config.noise.model and r["basis"] == config.noise.basis
                  and r["rounds"] == ("" if config.noise.rounds is None else str(config.noise.rounds))
                  and float(r["p"]) == p and int(r["T"]) == config.eval.stage1_iters
                  and r["selected"].lower() == "true"]
    if len(candidates) != 1:
        raise ValueError(f"Expected one validation-tuned factor for p={p}, T={config.eval.stage1_iters}")
    row = candidates[0]
    return float(row["ms_scaling"])


def make_decoder(name, problem, config, p, tuning_table):
    if name not in {"bp", "bp_tuned", "bp_only", "bp_tuned_only"}:
        raise ValueError(f"Decoder {name!r} is not implemented at M2")
    alpha = tuned_factor(tuning_table, config, p) if "tuned" in name else config.model.ms_scaling
    stage1 = LdpcBP(problem, config.eval.stage1_iters, alpha)
    stage1.name = "bp_tuned" if "tuned" in name else "bp"
    if name.endswith("_only"):
        return stage1, stage1, alpha
    settings = config.eval.osd
    osd = LdpcBPOSD(problem, settings.bp_iters, settings.ms_scaling,
                   settings.method, settings.order)
    return TwoStageDecoder(stage1, osd), stage1, alpha


def evaluate_p(sampler, decoders, rng, *, stop_failures, max_shots, batch, progress=None):
    """Stop at a shared batch boundary when every decoder reaches its failure target.

    All decoders see every shot. Overshoot is bounded by one batch for the last
    decoder to reach the target; faster-failing decoders keep running to retain pairing.
    """
    if min(stop_failures, max_shots, batch) < 1:
        raise ValueError("Stopping and batch counts must be positive")
    counts = {name: dict(logical=0, stage1_fail=0, stage1_logical=0, calls=0, executed=0,
                         iterations=0, used=[], logical_flags=[]) for name in decoders}
    shots = 0
    while shots < max_shots and not all(v["logical"] >= stop_failures for v in counts.values()):
        n = min(batch, max_shots-shots)
        s, obs, _ = sampler.sample(n, rng)
        for name, (decoder, stage1, _) in decoders.items():
            if isinstance(decoder, TwoStageDecoder):
                result = decoder.decode_batch(s)
                out, first = result["e_hat"], result["stage1_e_hat"]
                conv, used = result["stage1_converged"], result["osd_used"]
                counts[name]["executed"] += int(result["osd_executed"].sum())
            else:
                out, conv = decoder.decode_batch(s)
                first = out
                used = np.zeros(n, dtype=bool)
            logical = (parity(sampler.problem.L, out) != obs).any(axis=1)
            first_logical = (parity(sampler.problem.L, first) != obs).any(axis=1) | ~conv
            if not isinstance(decoder, TwoStageDecoder):
                logical |= ~conv
            record = counts[name]
            record["logical"] += int(logical.sum())
            record["stage1_fail"] += int((~conv).sum())
            record["stage1_logical"] += int(first_logical.sum())
            record["calls"] += int(used.sum())
            record["iterations"] += int(stage1.last_iterations.sum())
            record["used"].append(used)
            record["logical_flags"].append(logical)
        shots += n
        if progress is not None:
            progress(shots, counts)
    for record in counts.values():
        record["used"] = np.concatenate(record["used"])
        record["logical_flags"] = np.concatenate(record["logical_flags"])
        record["shots"] = shots
    return counts


def measure_latency(sampler, decoder, rng, shots, warmup):
    s, _, _ = sampler.sample(shots+warmup, rng)
    timings = np.empty(shots)
    for i, row in enumerate(s):
        start = time.perf_counter_ns()
        decoder.decode_batch(row[None, :])
        elapsed = (time.perf_counter_ns()-start)/1000
        if i >= warmup:
            timings[i-warmup] = elapsed
    return timings


def run(config, names, tuning_table, output_root, *, progress=False):
    config.validate()
    if not names or len(names) != len(set(names)):
        raise ValueError("Specify unique decoder names")
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + config.config_hash[:8]
    outdir = Path(output_root)/run_id
    outdir.mkdir(parents=True, exist_ok=False)
    config.save(outdir/"config.yaml")
    (outdir/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    if any("tuned" in name for name in names):
        shutil.copyfile(tuning_table, outdir/"tuned_scaling.csv")
    sha = git_sha()
    rows, details = [], {}
    try:
        for i, p in enumerate(config.noise.p_grid):
            sampler = make_sampler(config, p)
            if config.noise.model == "circuit_level":
                (outdir/f"circuit-p{i}.stim").write_text(str(sampler.circuit)+"\n")
                (outdir/f"dem-p{i}.dem").write_text(str(sampler.dem)+"\n")
                (outdir/f"problem-p{i}.json").write_text(json.dumps(sampler.problem.meta, indent=2)+"\n")
            decoders = {name: make_decoder(name, sampler.problem, config, p, tuning_table) for name in names}
            seed = config.seed+config.eval.seed_offset+i
            rng = np.random.default_rng(seed)
            # Advance the sampling stream without repeating expensive decoding.
            # Keeping the original batch size preserves Stim sub-seeds exactly.
            for _ in range(config.eval.sample_offset//config.eval.batch):
                sampler.sample(config.eval.batch, rng)
            started = time.perf_counter()

            def report(shots, counts):
                summary = ", ".join(f"{name}: {record['logical']} failures, {record['calls']} fallback calls"
                                    for name, record in counts.items())
                print(f"p={p:g} shots={shots}: {summary}; elapsed={time.perf_counter()-started:.1f}s", flush=True)

            counts = evaluate_p(sampler, decoders, rng,
                                stop_failures=config.eval.stop_failures,
                                max_shots=config.eval.max_shots, batch=config.eval.batch,
                                progress=report if progress else None)
            paired, logical_flags = {}, {}
            for name, (decoder, _, alpha) in decoders.items():
                count = counts[name]
                shots = count["shots"]
                lo, hi = wilson(count["logical"], shots)
                latency = dict.fromkeys(["lat_mean_us", "lat_p50_us", "lat_p99_us", "lat_max_us"], "")
                if config.eval.latency_shots:
                    # Separate stream: timing does not consume the accuracy stream.
                    timing_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
                    times = measure_latency(sampler, decoder, timing_rng,
                                            config.eval.latency_shots, config.eval.latency_warmup)
                    np.save(outdir/f"latency-p{i}-{name}.npy", times, allow_pickle=False)
                    latency = latency_stats(times)
                fallback = isinstance(decoder, TwoStageDecoder)
                row = dict(run_id=run_id, git_sha=sha, config_hash=config.config_hash,
                           timestamp=timestamp.isoformat(), code=config.code.name,
                           noise_model=config.noise.model, rounds=config.noise.rounds,
                           basis=config.noise.basis, p=p,
                           decoder=decoder.name if fallback else decoder.name+"_only",
                           stage1=decoder.stage1.name if fallback else decoder.name,
                           T=config.eval.stage1_iters, weight_sharing="scalar",
                           osd_method=config.eval.osd.method if fallback else "",
                           osd_order=config.eval.osd.order if fallback else "", ms_scaling=alpha,
                           shots=shots, stage1_failures=count["stage1_fail"],
                           osd_calls=count["calls"], osd_rate=count["calls"]/shots,
                           logical_failures=count["logical"], ler=count["logical"]/shots,
                           ler_lo=lo, ler_hi=hi, ler_stage1=count["stage1_logical"]/shots,
                           **latency, lat_threads=config.eval.threads,
                           device=f"CPU {platform.machine()} {platform.processor()}", seed=seed)
                rows.append(row)
                paired[name] = count["used"]
                logical_flags[name] = count["logical_flags"]
                details[f"p{i}:{name}"] = dict(osd_executions=count["executed"],
                                               mean_stage1_iterations=count["iterations"]/shots,
                                               reached_failure_target=count["logical"] >= config.eval.stop_failures)
                print(f"p={p:g} {row['decoder']}: shots={shots}, "
                      f"fallback={row['osd_rate']:.4f}, LER={row['ler']:.5f}", flush=True)
            np.savez_compressed(outdir/f"paired-p{i}.npz", **paired)
            np.savez_compressed(outdir/f"logical-p{i}.npz", **logical_flags)
            for a, b in combinations(names, 2):
                if isinstance(decoders[a][0], TwoStageDecoder) and isinstance(decoders[b][0], TwoStageDecoder):
                    details[f"p{i}:{a}_vs_{b}"] = mcnemar(paired[a], paired[b])
        append_results(outdir/"results.csv", rows)
        details.update(purpose=config.eval.purpose, git_sha=sha,
                       sample_offset=config.eval.sample_offset,
                       stopping="shared batch boundary; all decoders reach target or max_shots",
                       osd_rate_definition="stage-2 calls/shots; internal OSD executions in this file")
        (outdir/"diagnostics.json").write_text(json.dumps(details, indent=2)+"\n")
    except Exception as exc:
        (outdir/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    print(f"Saved {outdir}", flush=True)
    return outdir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--decoders", default="bp,bp_tuned,bp_only,bp_tuned_only")
    parser.add_argument("--tuning-table", default="results/tuned_scaling.csv")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--progress", action="store_true", help="Print a progress record after each batch")
    args = parser.parse_args()
    run(load_config(args.config), args.decoders.split(","), args.tuning_table, args.output_root,
        progress=args.progress)
