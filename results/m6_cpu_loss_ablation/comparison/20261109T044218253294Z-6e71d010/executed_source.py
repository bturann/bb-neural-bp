"""Compare two data-matched loss arms after freezing validation-selected checkpoints."""

import argparse
import csv
import hashlib
import json
import platform
import runpy
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.checkpoints import load_model, read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.evaluate import cpu_model_label, tuned_factor
from bbnbp.io import append_results, git_sha, source_manifest
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference

ROOT = Path(__file__).resolve().parents[1]


def training_arm(directory):
    runs = list((directory/"training").iterdir())
    if len(runs) != 1:
        raise ValueError("Expected exactly one training run per arm")
    run = runs[0]
    status = json.loads((directory/"pilot_status.json").read_text())
    if status["returncode"] or status["stopping"] != "child_completed":
        raise ValueError("Both training arms must complete before comparison")
    config = load_config(run/"config.yaml")
    history = [json.loads(line) for line in (run/"training.jsonl").read_text().splitlines()]
    report = json.loads((run/"m5_gate.json").read_text())
    checkpoint = Path(report["checkpoint_directory"])/"best.pt"
    if len(history) != config.train.max_steps or history[-1]["step"] != config.train.max_steps:
        raise ValueError("Controlled arms must complete their full equal update budgets")
    return dict(directory=directory, run=run, config=config, history=history,
                report=report, checkpoint=checkpoint, payload=read_checkpoint(checkpoint))


def verify_control(parity, bitwise):
    a, b = parity["config"], bitwise["config"]
    if a.train.loss != "parity" or b.train.loss != "bitwise" or replace(b, train=replace(b.train, loss="parity")) != a:
        raise ValueError("Controlled configurations may differ only in train.loss")
    keys = ("step", "p", "batch_sha256", "drawn", "hard", "effective_batch", "capped")
    first = [{key: row[key] for key in keys} for row in parity["history"]]
    second = [{key: row[key] for key in keys} for row in bitwise["history"]]
    if first != second:
        raise ValueError("Training inputs or sampling metadata differ between losses")
    last_a = read_checkpoint(parity["checkpoint"].parent/"last.pt")["training_state"]
    last_b = read_checkpoint(bitwise["checkpoint"].parent/"last.pt")["training_state"]
    if last_a["numpy_state"] != last_b["numpy_state"] or not torch.equal(last_a["torch_state"], last_b["torch_state"]):
        raise ValueError("Ending generator states differ between arms")
    with np.load(parity["run"]/"baseline_flags.npz") as x, np.load(bitwise["run"]/"baseline_flags.npz") as y:
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key])
    initial = []
    for arm in (parity, bitwise):
        initial.append(json.loads((arm["run"]/"validation.jsonl").read_text().splitlines()[0]))
    if initial[0] != initial[1]:
        raise ValueError("Initial models or validation samples differ")
    return dict(configurations_differ_only_in_loss=True, matched_training_batches=len(first),
                training_inputs_sha256=hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest(),
                ending_generators_match=True, baseline_validation_matches=True, initial_validation_matches=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--bitwise", type=Path, required=True)
    parser.add_argument("--settings", default="configs/m6/cpu_loss_ablation_budget.yaml")
    parser.add_argument("--output-root", type=Path, default=Path("results/m6_cpu_loss_ablation/comparison"))
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.settings).read_text())
    parity, bitwise = training_arm(args.parity), training_arm(args.bitwise)
    controls = verify_control(parity, bitwise)
    c = load_config("configs/m6/cpu_pilot_check.yaml")
    c = replace(c, eval=replace(c.eval, seed_offset=settings["evaluation_seed_offset"],
                max_shots=settings["evaluation_shots"], batch=settings["evaluation_batch"],
                stop_failures=settings["evaluation_shots"]+1,
                checkpoints=(str(parity["checkpoint"]), str(bitwise["checkpoint"])))).validate()
    training_seeds = {parity["config"].seed+parity["config"].train.runtime.validation_seed_offset+i for i in range(len(c.noise.p_grid))}
    regions = [training_seeds, {c.seed+c.sweep.seed_offset+i for i in range(len(c.noise.p_grid))},
               {c.seed+c.eval.seed_offset+i for i in range(len(c.noise.p_grid))}, {c.seed+10000+i for i in range(len(c.noise.p_grid))}]
    if any(regions[i] & regions[j] for i in range(len(regions)) for j in range(i)):
        raise ValueError("Ablation selection, tuning, comparison and final streams overlap")
    helpers = runpy.run_path(str(ROOT/"scripts/m6_compare_stage1.py"))
    table = Path(parity["config"].train.runtime.baseline_table)
    helpers["validate_tuning_table"](c, table)
    torch.set_num_threads(1)
    set_num_threads(1)
    timestamp = datetime.now(timezone.utc)
    output = args.output_root/(timestamp.strftime("%Y%m%dT%H%M%S%fZ")+"-"+c.config_hash[:8])
    output.mkdir(parents=True, exist_ok=False)
    c.save(output/"config.yaml")
    (output/"settings.yaml").write_text(yaml.safe_dump(settings))
    (output/"controls.json").write_text(json.dumps(controls, indent=2)+"\n")
    (output/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    shutil.copyfile(__file__, output/"executed_source.py")
    shutil.copyfile(table, output/"tuned_scaling.csv")
    shutil.copyfile(table.with_suffix(".config.yaml"), output/"tuned_scaling.config.yaml")
    for label, arm in (("parity", parity), ("bitwise", bitwise)):
        arm["config"].save(output/f"training-{label}.yaml")
        shutil.copyfile(arm["checkpoint"], output/f"checkpoint-{label}.pt")
    begin = time.perf_counter()
    starts = [datetime.strptime(arm["directory"].name.split("-")[0], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc) for arm in (parity, bitwise)]
    prior_elapsed = (timestamp-min(starts)).total_seconds()
    limit = min(settings["comparison_wall_seconds"], settings["wall_seconds"]-prior_elapsed)
    if limit <= 0:
        raise TimeoutError("The controlled study's total wall budget is exhausted")
    labels = ("bp", "bp_tuned", "nbp_initial", "nbp_parity", "nbp_bitwise")
    points, rows = [], []
    print(f"Ablation comparison: {output}; remaining comparison budget {limit:.1f}s", flush=True)
    try:
        for index, p in enumerate(c.noise.p_grid):
            sampler = make_sampler(c, p)
            seed = c.seed+c.eval.seed_offset+index
            s, o, errors = sampler.sample(c.eval.max_shots, np.random.default_rng(seed))
            del errors
            factor = tuned_factor(table, c, p)
            models = {"nbp_parity": load_model(parity["checkpoint"], sampler.problem),
                      "nbp_bitwise": load_model(bitwise["checkpoint"], sampler.problem)}
            models["nbp_initial"] = NeuralMinSum(sampler.problem, **parity["payload"]["model_config"], dtype=models["nbp_parity"].w.dtype)
            decoders = {"bp": LdpcBP(sampler.problem, c.model.T, c.model.ms_scaling),
                        "bp_tuned": LdpcBP(sampler.problem, c.model.T, factor)}
            for name, model in models.items():
                compiled = model.export_cpu()
                n = min(c.eval.batch, len(s))
                expected, conv = model.decode_batch(s[:n])
                actual, actual_conv = compiled.decode_batch(s[:n])
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual_conv, conv)
                np.testing.assert_array_equal(compiled.last_iterations, model.last_iterations)
                decoders[name] = compiled
            flags, logical_flags, point = {}, {}, dict(p=p, seed=seed, shots=len(s), tuned_factor=factor, decoders={})
            for name in labels:
                if time.perf_counter()-begin > limit:
                    raise TimeoutError("Ablation comparison reached its configured wall budget")
                failed, logical, iterations = _inference(decoders[name], s, o, sampler.problem, c.eval.batch)
                lo, hi = wilson(int(logical.sum()), len(s))
                row = dict(run_id=output.name, git_sha=git_sha(), config_hash=c.config_hash, timestamp=timestamp.isoformat(),
                           code=c.code.name, noise_model=c.noise.model, rounds=c.noise.rounds, basis=c.noise.basis, p=p,
                           decoder=name+"_only", stage1=name, T=c.model.T, weight_sharing=models[name].sharing if name.startswith("nbp") else "scalar",
                           osd_method="", osd_order="", ms_scaling=parity["payload"]["model_config"]["ms_scaling"] if name.startswith("nbp") else factor if name=="bp_tuned" else c.model.ms_scaling,
                           shots=len(s), stage1_failures=int(failed.sum()), osd_calls=0, osd_rate=0,
                           logical_failures=int(logical.sum()), ler=float(logical.mean()), ler_lo=lo, ler_hi=hi, ler_stage1=float(logical.mean()),
                           lat_mean_us="", lat_p50_us="", lat_p99_us="", lat_max_us="", lat_threads=1,
                           device=f"CPU {cpu_model_label(c.eval.cpu_model)}; {platform.platform()}", seed=seed)
                rows.append(row)
                flags[name], logical_flags[name] = failed, logical
                point["decoders"][name] = dict(fallback_demand=float(failed.mean()), failures=int(failed.sum()), mean_iterations=iterations)
                print(f"p={p:g} {name}: fallback={failed.mean():.6f}; elapsed={time.perf_counter()-begin:.1f}s", flush=True)
            append_results(output/"results.csv", rows[-len(labels):])
            np.savez_compressed(output/f"cases-p{index}.npz", syndromes=s, observables=o)
            np.savez_compressed(output/f"failed-p{index}.npz", **flags)
            np.savez_compressed(output/f"logical-p{index}.npz", **logical_flags)
            point["parity_vs_bitwise"] = mcnemar(flags["nbp_parity"], flags["nbp_bitwise"])
            point["vs_tuned"] = {name: mcnemar(flags["bp_tuned"], flags[name]) for name in ("nbp_parity", "nbp_bitwise")}
            point["vs_initial"] = {name: mcnemar(flags["nbp_initial"], flags[name]) for name in ("nbp_parity", "nbp_bitwise")}
            points.append(point)
        primary = helpers["holm"]([p["parity_vs_bitwise"]["pvalue"] for p in points])
        for point, value in zip(points, primary):
            point["parity_vs_bitwise"]["holm_pvalue"] = value
        for family in ("vs_tuned", "vs_initial"):
            pairs = [p[family][name] for p in points for name in ("nbp_parity", "nbp_bitwise")]
            for pair, adjusted in zip(pairs, helpers["holm"]([pair["pvalue"] for pair in pairs])):
                pair["holm_pvalue"] = adjusted
        means = {name: float(np.mean([p["decoders"][name]["fallback_demand"] for p in points])) for name in labels}
        report = dict(purpose=settings["purpose"], controls=controls, points=points, mean_fallback=means,
                      elapsed_s=time.perf_counter()-begin, total_study_elapsed_s=prior_elapsed+time.perf_counter()-begin,
                      parity_improvement_percentage_points=100*(means["nbp_bitwise"]-means["nbp_parity"]),
                      significant_primary_points=sum(value < settings["primary_alpha"] for value in primary),
                      checkpoints={label: dict(path=str(arm["checkpoint"]), step=arm["payload"]["step"],
                                   sha256=hashlib.sha256(arm["checkpoint"].read_bytes()).hexdigest()) for label,arm in (("parity",parity),("bitwise",bitwise))},
                      training_roots=dict(parity=str(args.parity),bitwise=str(args.bitwise)),
                      final_test_data_used=False, gpu_seconds=0, m6_complete=False,
                      scope="Single-seed, controlled development loss ablation. Primary Holm family: five parity/BCE comparisons. "
                            "Each secondary family corrects both losses across five p values (ten tests). "
                            "CSV osd_rate is zero: fallback demand is stage1_failures/shots; no OSD, two-stage LER, or latency claim.")
        (output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({key:report[key] for key in ("mean_fallback","parity_improvement_percentage_points","significant_primary_points","total_study_elapsed_s")},indent=2))
    except BaseException as exc:
        (output/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
