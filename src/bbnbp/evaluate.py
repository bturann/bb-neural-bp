"""Paired baseline evaluation with shared shots and failure-count stopping."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import time
import hashlib
import subprocess
from dataclasses import replace
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
from .checkpoints import load_model, read_checkpoint


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
    neural = name == "nbp" or name.startswith("nbp_")
    if neural:
        index = int(name.split("_")[-1]) if name.startswith("nbp_") else 0
        if index >= len(config.eval.checkpoints):
            raise ValueError("Neural evaluation requires a checkpoint for every named model")
        path = config.eval.checkpoints[index]
        model = load_model(path, problem)
        if model.T != config.eval.stage1_iters:
            raise ValueError("Checkpoint iteration budget differs from the baseline budget")
        stage1 = model.export_cpu()
        stage1.name = name
        stage1.sharing = model.sharing
        stage1.checkpoint_path = path
        stage1.training_seed = read_checkpoint(path)["config"]["seed"]
        alpha = config.model.ms_scaling
    else:
        if name not in {"bp", "bp_tuned", "bp_only", "bp_tuned_only"}:
            raise ValueError(f"Unknown decoder {name!r}")
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
    s, observations, _ = sampler.sample(shots+warmup, rng)
    timings = np.empty(shots)
    stage1_times, osd_times, used = np.zeros(shots), np.zeros(shots), np.zeros(shots, dtype=bool)
    failed, logical, first_logical = np.zeros(shots, dtype=bool), np.zeros(shots, dtype=bool), np.zeros(shots, dtype=bool)
    executed, iterations = np.zeros(shots, dtype=bool), np.zeros(shots, dtype=np.int64)
    for i, row in enumerate(s):
        start = time.perf_counter_ns()
        result = decoder.decode_timed_one(row) if isinstance(decoder, TwoStageDecoder) else decoder.decode_batch(row[None, :])
        elapsed = (time.perf_counter_ns()-start)/1000
        if i >= warmup:
            j = i-warmup
            timings[j] = elapsed
            if isinstance(decoder, TwoStageDecoder):
                stage1_times[j] = result["stage1_ns"]/1000
                osd_times[j] = result["osd_ns"]/1000
                used[j] = bool(result["osd_used"][0])
                failed[j] = not bool(result["stage1_converged"][0])
                executed[j] = result["osd_executed"][0]
                correction, first = result["e_hat"], result["stage1_e_hat"]
                iterations[j] = decoder.stage1.last_iterations[0]
            else:
                stage1_times[j] = elapsed
                correction, conv = result
                first = correction
                failed[j] = not bool(conv[0])
                iterations[j] = decoder.last_iterations[0]
            logical[j] = bool((parity(sampler.problem.L, correction)[0] != observations[i]).any())
            if not isinstance(decoder, TwoStageDecoder):
                logical[j] |= failed[j]
            first_logical[j] = failed[j] or bool((parity(sampler.problem.L, first)[0] != observations[i]).any())
    return dict(total_us=timings, stage1_us=stage1_times, osd_us=osd_times,
                overhead_us=timings-stage1_times-osd_times, osd_used=used,
                stage1_failed=failed, logical_flags=logical, stage1_logical_flags=first_logical,
                osd_executed=executed, iterations=iterations,
                syndrome_sha256=np.array(hashlib.sha256(s[warmup:].tobytes()).hexdigest()))


def cpu_model_label(override=None):
    if override:
        return override
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    try:
        return subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return f"{platform.machine()} (CPU brand unavailable)"


def run(config, names, tuning_table, output_root, *, progress=False):
    config.validate()
    import torch
    from numba import set_num_threads
    torch.set_num_threads(config.eval.threads)
    set_num_threads(config.eval.threads)
    if "nbp" in names and len(config.eval.checkpoints) > 1:
        names = [label for name in names for label in
                 ([f"nbp_{i}" for i in range(len(config.eval.checkpoints))] if name == "nbp" else [name])]
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
    checkpoint_manifest = []
    for index, path in enumerate(config.eval.checkpoints):
        payload = read_checkpoint(path)
        training = payload["config"]
        runtime = training["train"].get("runtime")
        if runtime:
            validation_seeds = {training["seed"]+runtime["validation_seed_offset"]+i
                                for i in range(len(training["noise"]["p_grid"]))}
            test_seeds = {config.seed+config.eval.seed_offset+i for i in range(len(config.noise.p_grid))}
            if validation_seeds & test_seeds:
                raise ValueError("Checkpoint validation seeds overlap evaluation seeds")
        shutil.copyfile(path, outdir/f"checkpoint-{index}.pt")
        checkpoint_manifest.append(dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                        training_seed=payload["config"]["seed"], step=payload["step"]))
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

            measured = {}
            if config.eval.latency_only:
                counts = {}
                for name, (decoder, _, _) in decoders.items():
                    samples = measure_latency(sampler, decoder,
                                              np.random.default_rng(np.random.SeedSequence([seed, 1])),
                                              config.eval.latency_shots, config.eval.latency_warmup)
                    measured[name] = samples
                    counts[name] = dict(shots=config.eval.latency_shots, logical=int(samples["logical_flags"].sum()),
                                        stage1_fail=int(samples["stage1_failed"].sum()),
                                        stage1_logical=int(samples["stage1_logical_flags"].sum()),
                                        calls=int(samples["osd_used"].sum()), executed=int(samples["osd_executed"].sum()),
                                        iterations=int(samples["iterations"].sum()), used=samples["osd_used"],
                                        logical_flags=samples["logical_flags"])
            else:
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
                    samples = measured.get(name)
                    if samples is None:
                        samples = measure_latency(sampler, decoder, timing_rng,
                                                  config.eval.latency_shots, config.eval.latency_warmup)
                    np.save(outdir/f"latency-p{i}-{name}.npy", samples["total_us"], allow_pickle=False)
                    np.savez_compressed(outdir/f"latency-components-p{i}-{name}.npz", **samples)
                    latency = latency_stats(samples["total_us"])
                    details[f"latency-p{i}:{name}"] = dict(
                        stage1_mean_us=float(samples["stage1_us"].mean()),
                        osd_mean_us=float(samples["osd_us"].mean()),
                        overhead_mean_us=float(samples["overhead_us"].mean()),
                        osd_calls=int(samples["osd_used"].sum()),
                        warmup=config.eval.latency_warmup, shots=config.eval.latency_shots,
                        learned_backend="numba" if name.startswith("nbp") else "ldpc C++")
                fallback = isinstance(decoder, TwoStageDecoder)
                row = dict(run_id=run_id, git_sha=sha, config_hash=config.config_hash,
                           timestamp=timestamp.isoformat(), code=config.code.name,
                           noise_model=config.noise.model, rounds=config.noise.rounds,
                           basis=config.noise.basis, p=p,
                           decoder=decoder.name if fallback else decoder.name+"_only",
                           stage1=decoder.stage1.name if fallback else decoder.name,
                           T=config.eval.stage1_iters, weight_sharing=getattr(decoders[name][1], "sharing", "scalar"),
                           osd_method=config.eval.osd.method if fallback else "",
                           osd_order=config.eval.osd.order if fallback else "", ms_scaling=alpha,
                           shots=shots, stage1_failures=count["stage1_fail"],
                           osd_calls=count["calls"], osd_rate=count["calls"]/shots,
                           logical_failures=count["logical"], ler=count["logical"]/shots,
                           ler_lo=lo, ler_hi=hi, ler_stage1=count["stage1_logical"]/shots,
                           **latency, lat_threads=config.eval.threads,
                           device=f"CPU {cpu_model_label(config.eval.cpu_model)}; {platform.platform()}", seed=seed)
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
                       checkpoints=checkpoint_manifest,
                       cpu_model=cpu_model_label(config.eval.cpu_model),
                       torch_threads=torch.get_num_threads(),
                       gpu_throughput="not measured in this CPU evaluation",
                       sample_offset=config.eval.sample_offset,
                       stopping="fixed measured latency shots after warmup" if config.eval.latency_only else
                                "shared batch boundary; all decoders reach target or max_shots",
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
    parser.add_argument("--checkpoint", action="append", default=[], help="Repeat for multiple trained seeds")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config = replace(config, eval=replace(config.eval, checkpoints=tuple(args.checkpoint)))
    run(config, args.decoders.split(","), args.tuning_table, args.output_root,
        progress=args.progress)
