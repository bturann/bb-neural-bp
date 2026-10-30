"""Streamed neural BP training with fixed validation, honest gates, and portable checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .checkpoints import checkpoint_payload, read_checkpoint, restore_parameters, save_checkpoint, load_model
from .config import load_config
from .decoders.bp_ldpc import LdpcBP
from .decoders.neural_bp import NeuralMinSum
from .evaluate import tuned_factor
from .events import ScalarEventWriter
from .io import git_sha, source_manifest
from .losses import ParityLoss, bitwise_bce, multi_loss
from .metrics import mcnemar
from .noise.base import IndependentFaultSampler, parity
from .noise.factory import make_sampler

ROOT = Path(__file__).resolve().parents[2]


def make_model(problem, config):
    return NeuralMinSum(problem, config.model.T, config.model.sharing,
                         config.model.ms_scaling, config.model.msg_clip)


def training_probability(config, step, rng):
    curriculum = config.train.curriculum
    if step < curriculum.steps:
        fraction = step/max(1, curriculum.steps-1)
        p = curriculum.p_start+(curriculum.p_end-curriculum.p_start)*fraction
        if config.noise.model == "circuit_level":
            return min(config.noise.p_grid, key=lambda candidate: abs(candidate-p))
        return float(p)
    return float(rng.choice(config.noise.p_grid))


def hard_batch(sampler, rng, config, frozen, llr):
    target = config.train.batch
    mode = config.train.hard_mining
    parts, drawn, hard_count = [], 0, 0
    draws = config.train.mining_max_draws if mode == "filter" else 1
    for _ in range(draws):
        s, _, e = sampler.sample(target, rng)
        drawn += len(s)
        hard = np.zeros(len(s), dtype=bool)
        if mode != "off":
            with torch.inference_mode():
                frozen(s, llr=llr, early_stop=True)
                hard = ~frozen.last_converged.cpu().numpy().copy()
        hard_count += int(hard.sum())
        if mode == "filter":
            parts.append((s[hard], e[hard]))
            if sum(len(a) for a, _ in parts) >= target:
                break
        else:
            weights = np.where(hard, config.train.hard_weight, 1.0) if mode == "upweight" else np.ones(len(s))
            return s, e, weights, dict(drawn=drawn, hard=hard_count, capped=False)
    kept = min(target, sum(len(a) for a, _ in parts))
    if kept == 0:
        raise RuntimeError("Hard-example filter exhausted its configured draws without any BP failures")
    s = np.concatenate([a for a, _ in parts])[:kept]
    e = np.concatenate([b for _, b in parts])[:kept]
    return s, e, np.ones(kept), dict(drawn=drawn, hard=hard_count, capped=kept < target)


def _inference(decoder, syndromes, observables, problem, batch):
    failed, logical, iterations = [], [], 0
    for start in range(0, len(syndromes), batch):
        s = syndromes[start:start+batch]
        correction, conv = decoder.decode_batch(s)
        failed.append(~conv)
        logical.append(~conv | (parity(problem.L, correction) != observables[start:start+batch]).any(1))
        iterations += int(decoder.last_iterations.sum())
    return np.concatenate(failed), np.concatenate(logical), iterations/len(syndromes)


def validation_sets(config, samplers):
    result = {}
    for i, p in enumerate(config.noise.p_grid):
        sampler = samplers[p]
        seed = config.seed+config.train.runtime.validation_seed_offset+i
        rng = np.random.default_rng(seed)
        syndromes, observables = [], []
        for start in range(0, config.train.val_shots, config.train.runtime.val_batch):
            n = min(config.train.runtime.val_batch, config.train.val_shots-start)
            s, o, _ = sampler.sample(n, rng)
            syndromes.append(s)
            observables.append(o)
        result[p] = (np.concatenate(syndromes), np.concatenate(observables), seed)
    return result


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def _plot_history(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (directory/"training.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (directory/"validation.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    if rows:
        axes[0].plot([r["step"] for r in rows], [r["loss"] for r in rows], alpha=0.5, linewidth=0.6)
    axes[0].set(xlabel="Optimizer step", ylabel="Streamed multi-loss")
    axes[1].plot([r["step"] for r in validation], [r["mean_osd_rate"] for r in validation], label="Neural BP")
    if validation:
        axes[1].axhline(validation[0]["mean_tuned_rate"], label="Tuned BP", linestyle="--", color="black")
    axes[1].set(xlabel="Optimizer step", ylabel="Validation fallback demand")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(directory/"loss_curves.png", dpi=180)
    fig.savefig(directory/"loss_curves.pdf")
    plt.close(fig)


def run(config, output_root="results/m5", checkpoint_root="checkpoints", *, resume=None, record_notes=True):
    config.validate()
    runtime = config.train.runtime
    if runtime is None:
        raise ValueError("Training requires an explicit train.runtime configuration")
    if config.eval.stage1_iters != config.model.T:
        raise ValueError("M5 validation must use the same T for neural and tuned BP")
    device = torch.device(config.train.device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Requested training device is unavailable or unsupported: {device}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+config.config_hash[:8]
    directory, checkpoint_dir = Path(output_root)/run_id, Path(checkpoint_root)/run_id
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for parent in (directory, checkpoint_dir):
        config.save(parent/"config.yaml")
    _write_json(directory/"source_manifest.json", source_manifest())
    provenance = dict(git_sha=git_sha(), config_hash=config.config_hash,
                      purpose=runtime.purpose, checkpoint_directory=str(checkpoint_dir),
                      validation_seeds=[config.seed+runtime.validation_seed_offset+i for i in range(len(config.noise.p_grid))],
                      training_entropy=[config.seed, runtime.training_stream], final_test_data_used=False,
                      validation_metric="Stage-1 nonconvergence; OSD is not called during training validation")
    _write_json(directory/"provenance.json", provenance)
    previous_threads, previous_determinism = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(runtime.threads)
    torch.use_deterministic_algorithms(True)
    events = ScalarEventWriter(directory/"events") if runtime.tensorboard else None
    started = time.perf_counter()
    try:
        table = Path(runtime.baseline_table)
        with table.open(newline="") as f:
            tuning_rows = list(csv.DictReader(f))
        if {int(row["seed"]) for row in tuning_rows} & set(provenance["validation_seeds"]):
            raise ValueError("Tuning data and training validation data use overlapping RNG streams")
        shutil.copyfile(table, directory/"tuned_scaling.csv")
        samplers = {p: make_sampler(config, p) for p in config.noise.p_grid}
        template = samplers[config.noise.p_grid[0]].problem
        for sampler in samplers.values():
            other = sampler.problem
            if other.H.shape != template.H.shape or (other.H != template.H).nnz or (other.L != template.L).nnz:
                raise ValueError("H/L structure must be identical across the training p grid")
        validation = validation_sets(config, samplers)
        baseline, baseline_arrays = {}, {}
        for i, p in enumerate(config.noise.p_grid):
            problem = samplers[p].problem
            s, o, seed = validation[p]
            factor = tuned_factor(table, config, p)
            result = {}
            for name, alpha in (("bp", config.model.ms_scaling), ("bp_tuned", factor)):
                decoder = LdpcBP(problem, config.model.T, alpha)
                failed, logical, iterations = _inference(decoder, s, o, problem, runtime.val_batch)
                result[name] = dict(nonconverged=int(failed.sum()), osd_rate=float(failed.mean()),
                                    logical_failures=int(logical.sum()), ler_stage1=float(logical.mean()),
                                    ms_scaling=alpha, mean_iterations=iterations)
                baseline_arrays[f"{name}-p{i}"] = failed
            baseline[p] = result
            print(f"validation baseline p={p:g}: tuned fallback={result['bp_tuned']['osd_rate']:.5f}", flush=True)
        _write_json(directory/"baseline_validation.json", {str(p): values for p, values in baseline.items()})
        np.savez_compressed(directory/"baseline_flags.npz", **baseline_arrays)
        model = make_model(template, config).to(device)
        frozen = make_model(template, config).to(device).eval().requires_grad_(False)
        objective = ParityLoss(template, config.train.beta_logical, runtime.loss_eps, runtime.loss_floor).to(device)
        per_shot_loss = objective if config.train.loss == "parity" else lambda p, e: bitwise_bce(p, e, runtime.loss_eps)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.max_steps, eta_min=config.train.min_lr)
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, runtime.training_stream]))
        torch_rng = torch.Generator(device=device).manual_seed(config.seed+runtime.torch_seed_offset)
        step, best_score, best_step, stale = 0, float("inf"), 0, 0
        if resume is not None:
            payload = read_checkpoint(resume)
            if payload["config_hash"] != config.config_hash:
                raise ValueError("Resume requires the exact resolved training configuration")
            restore_parameters(model, payload)
            state = payload["training_state"]
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            rng.bit_generator.state = state["numpy_state"]
            torch_rng.set_state(state["torch_state"])
            step, best_score, best_step, stale = payload["step"], state["best_score"], state["best_step"], state["stale"]
            shutil.copyfile(Path(resume).parent/"best.pt", checkpoint_dir/"best.pt")

        val_models = {p: make_model(samplers[p].problem, config) for p in config.noise.p_grid}
        history = []

        def state():
            return dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        numpy_state=rng.bit_generator.state, torch_state=torch_rng.get_state(),
                        best_score=best_score, best_step=best_step, stale=stale)

        def validate_current(at_step):
            rows, flags = [], {}
            for i, p in enumerate(config.noise.p_grid):
                candidate = val_models[p]
                with torch.no_grad():
                    candidate.w.copy_(model.w.detach().cpu())
                    candidate.alpha.copy_(model.alpha.detach().cpu())
                decoder = candidate.export_cpu()
                s, o, seed = validation[p]
                failed, logical, iterations = _inference(decoder, s, o, samplers[p].problem, runtime.val_batch)
                flags[f"nbp-p{i}"] = failed
                flags[f"logical-p{i}"] = logical
                rows.append(dict(p=p, shots=len(s), seed=seed, nonconverged=int(failed.sum()),
                                 osd_rate=float(failed.mean()), ler_stage1=float(logical.mean()),
                                 logical_failures=int(logical.sum()), mean_iterations=iterations,
                                 tuned=baseline[p]["bp_tuned"], bp=baseline[p]["bp"],
                                 paired_vs_tuned=mcnemar(baseline_arrays[f"bp_tuned-p{i}"], failed)))
            summary = dict(step=at_step, rows=rows, mean_osd_rate=float(np.mean([r["osd_rate"] for r in rows])),
                           mean_tuned_rate=float(np.mean([r["tuned"]["osd_rate"] for r in rows])))
            history.append(summary)
            with (directory/"validation.jsonl").open("a") as f:
                f.write(json.dumps(summary, allow_nan=False)+"\n")
            if events:
                events.scalar("validation/mean_osd_rate", summary["mean_osd_rate"], at_step)
                for row in rows:
                    events.scalar(f"validation/p{row['p']:g}/osd_rate", row["osd_rate"], at_step)
            print(f"step={at_step} validation mean fallback={summary['mean_osd_rate']:.6f}; "
                  f"tuned={summary['mean_tuned_rate']:.6f}", flush=True)
            return summary, flags

        initial, flags = validate_current(step)
        if initial["mean_osd_rate"] < best_score:
            best_score, best_step = initial["mean_osd_rate"], step
            save_checkpoint(checkpoint_dir/"best.pt", checkpoint_payload(model, config, step, training_state=state()))
            _write_json(directory/"best_validation.json", initial)
            np.savez_compressed(directory/"best_validation_flags.npz", **flags)
        last_summary = initial
        stopping = "max_steps"
        with (directory/"training.jsonl").open("a") as training_log:
            while step < config.train.max_steps:
                if device.type != "cpu" and time.perf_counter()-started >= runtime.max_gpu_hours*3600:
                    stopping = "gpu_budget"
                    break
                p = training_probability(config, step, rng)
                if config.noise.model == "code_capacity":
                    problem = replace(template, priors=np.full(template.H.shape[1], p),
                                      name=f"{config.code.name}-train-p{p:g}", meta={**template.meta, "p": p})
                    sampler = IndependentFaultSampler(problem)
                else:
                    sampler, problem = samplers[p], samplers[p].problem
                llr = torch.tensor(problem.llr, dtype=model.w.dtype, device=device)
                s, e, weights, mining = hard_batch(sampler, rng, config, frozen, llr)
                if mining["capped"]:
                    print(f"warning: hard mining capped at {config.train.mining_max_draws} draws; kept {len(s)} shots", flush=True)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(torch.from_numpy(s).to(device), llr=llr)
                faults = torch.from_numpy(e).to(device)
                weight_tensor = torch.tensor(weights, dtype=model.w.dtype, device=device)
                loss = multi_loss(outputs, faults, per_shot_loss, weight_tensor)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss is not finite")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip, error_if_nonfinite=True)
                used_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                step += 1
                record = dict(step=step, p=p, loss=float(loss.detach()), grad_norm=float(norm), lr=used_lr,
                              effective_batch=len(s), **mining, elapsed_s=time.perf_counter()-started)
                training_log.write(json.dumps(record, allow_nan=False)+"\n")
                training_log.flush()
                if step % runtime.log_every == 0:
                    print(f"step={step} p={p:.5f} loss={record['loss']:.5f} elapsed={record['elapsed_s']:.1f}s", flush=True)
                    if events:
                        events.scalar("train/loss", record["loss"], step)
                        events.scalar("train/lr", used_lr, step)
                        events.scalar("train/grad_norm", record["grad_norm"], step)
                if step % config.train.val_every == 0 or step == config.train.max_steps:
                    last_summary, flags = validate_current(step)
                    if last_summary["mean_osd_rate"] < best_score:
                        best_score, best_step, stale = last_summary["mean_osd_rate"], step, 0
                        save_checkpoint(checkpoint_dir/"best.pt", checkpoint_payload(model, config, step, training_state=state()))
                        _write_json(directory/"best_validation.json", last_summary)
                        np.savez_compressed(directory/"best_validation_flags.npz", **flags)
                    else:
                        stale += 1
                    save_checkpoint(checkpoint_dir/"last.pt", checkpoint_payload(model, config, step, training_state=state()))
                    if stale >= config.train.patience:
                        stopping = "validation_patience"
                        break
        save_checkpoint(checkpoint_dir/"last.pt", checkpoint_payload(model, config, step, training_state=state()))
        best = json.loads((directory/"best_validation.json").read_text()) if (directory/"best_validation.json").exists() else None
        if best is None:
            restored = load_model(checkpoint_dir/"best.pt", template)
            with torch.no_grad():
                model.w.copy_(restored.w.to(device)); model.alpha.copy_(restored.alpha.to(device))
            best, flags = validate_current(best_step)
            _write_json(directory/"best_validation.json", best)
            np.savez_compressed(directory/"best_validation_flags.npz", **flags)
        wins = sum(row["nonconverged"] < row["tuned"]["nonconverged"] for row in best["rows"])
        m3 = json.loads(Path(runtime.m3_gate_report).read_text())
        accepted = runtime.purpose == "m5_gate" and wins >= runtime.gate_min_points
        report = dict(passed=accepted, purpose=runtime.purpose, wins=wins, required_wins=runtime.gate_min_points,
                      best_step=best_step, best_mean_osd_rate=best_score, steps_completed=step,
                      stopping=stopping, elapsed_s=time.perf_counter()-started, m3_passed=bool(m3["passed"]),
                      eligible_for_m6=accepted and bool(m3["passed"]), final_test_data_used=False,
                      rows=best["rows"], checkpoint_directory=str(checkpoint_dir),
                      best_checkpoint_sha256=hashlib.sha256((checkpoint_dir/"best.pt").read_bytes()).hexdigest())
        _write_json(directory/"m5_gate.json", report)
        _plot_history(directory)
        if record_notes and runtime.purpose == "m5_gate":
            with (ROOT/"NOTES.md").open("a") as f:
                f.write(f"\nM5 {'passed' if accepted else 'gate failed'}: `{directory}`; "
                        f"wins {wins}/{len(best['rows'])}, best step {best_step}, "
                        f"mean fallback {best_score}. Final-test data untouched.\n")
        print(f"M5 {'PASS' if accepted else 'NOT ACCEPTED'}: wins={wins}/{len(best['rows'])}, "
              f"best_step={best_step}; {directory}", flush=True)
        return directory
    except BaseException as exc:
        (directory/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        if record_notes:
            with (ROOT/"NOTES.md").open("a") as f:
                f.write(f"\nM5 execution failed in `{directory}`: {type(exc).__name__}: {exc}.\n")
        raise
    finally:
        if events:
            events.close()
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="results/m5")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--resume")
    args = parser.parse_args()
    result = run(load_config(args.config), args.output_root, args.checkpoint_root, resume=args.resume)
    gate = json.loads((result/"m5_gate.json").read_text())
    raise SystemExit(0 if gate["passed"] or gate["purpose"] == "smoke" else 1)
