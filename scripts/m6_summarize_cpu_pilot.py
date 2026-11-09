"""Replay CPU pilot validation and summarize learning and measured update costs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bbnbp.checkpoints import load_model, read_checkpoint
from bbnbp.config import load_config
from bbnbp.metrics import mcnemar
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference, make_model, validation_sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.directory/"pilot_summary.json"
    if output.exists():
        parser.error("Refusing to overwrite pilot evidence")
    status = json.loads((args.directory/"pilot_status.json").read_text())
    runs = list((args.directory/"training").iterdir())
    if len(runs) != 1:
        parser.error("Expected exactly one training run")
    run = runs[0]
    config = load_config(run/"config.yaml")
    torch.set_num_threads(config.train.runtime.threads)
    history = [json.loads(line) for line in (run/"training.jsonl").read_text().splitlines()]
    validations = [json.loads(line) for line in (run/"validation.jsonl").read_text().splitlines()]
    provenance = json.loads((run/"provenance.json").read_text())
    checkpoint = Path(provenance["checkpoint_directory"])/"best.pt"
    with (run/"tuned_scaling.csv").open(newline="") as stream:
        tuning_factors = sorted({float(row["ms_scaling"]) for row in csv.DictReader(stream)})
    selected_step = read_checkpoint(checkpoint)["step"]
    samplers = {p: make_sampler(config, p) for p in config.noise.p_grid}
    sets = validation_sets(config, samplers)
    rows, replay = [], {}
    with np.load(run/"best_validation_flags.npz") as saved, np.load(run/"baseline_flags.npz") as baseline:
        for i, p in enumerate(config.noise.p_grid):
            problem = samplers[p].problem
            s, o, seed = sets[p]
            initial = make_model(problem, config).export_cpu()
            selected = load_model(checkpoint, problem).export_cpu()
            first, _, _ = _inference(initial, s, o, problem, config.train.runtime.val_batch)
            chosen, logical, _ = _inference(selected, s, o, problem, config.train.runtime.val_batch)
            np.testing.assert_array_equal(chosen, saved[f"nbp-p{i}"])
            np.testing.assert_array_equal(logical, saved[f"logical-p{i}"])
            rows.append(dict(p=p, shots=len(s), seed=seed,
                             initial_nonconverged=int(first.sum()), selected_nonconverged=int(chosen.sum()),
                             initial_osd_rate=float(first.mean()), selected_osd_rate=float(chosen.mean()),
                             development_tuned_osd_rate=float(baseline[f"bp_tuned-p{i}"].mean()),
                             exploratory_paired_vs_initial=mcnemar(first, chosen)))
            replay[f"initial-p{i}"], replay[f"selected-p{i}"] = first, chosen
    intervals = [current["elapsed_s"]-previous["elapsed_s"]
                 for previous, current in zip(history, history[1:])
                 if current["step"] == previous["step"]+1 and previous["step"] % config.train.val_every != 0]
    initial_mean = float(np.mean([row["initial_osd_rate"] for row in rows]))
    best_mean = float(np.mean([row["selected_osd_rate"] for row in rows]))
    np.testing.assert_allclose(initial_mean, validations[0]["mean_osd_rate"], atol=0, rtol=0)
    selected_record = next(row for row in validations if row["step"] == selected_step)
    np.testing.assert_allclose(best_mean, selected_record["mean_osd_rate"], atol=0, rtol=0)
    summary = dict(purpose="cpu_development_learning_pilot", status=status,
                   steps_completed=history[-1]["step"] if history else 0,
                   selected_step=selected_step, train_batch=config.train.batch,
                   parameter_count=sum(value.numel() for value in read_checkpoint(checkpoint)["parameters"].values()),
                   finite_logged_losses=all(np.isfinite(row["loss"]) for row in history),
                   initial_mean_osd_rate=initial_mean, selected_mean_osd_rate=best_mean,
                   last_mean_osd_rate=validations[-1]["mean_osd_rate"],
                   development_tuned_mean_osd_rate=float(np.mean([row["development_tuned_osd_rate"] for row in rows])),
                   tuning_factor_count=len(tuning_factors),
                   absolute_reduction_percentage_points=100*(initial_mean-best_mean),
                   improved_p_points_vs_initial=sum(row["selected_osd_rate"] < row["initial_osd_rate"] for row in rows),
                   improved_p_points_vs_development_tuned=sum(row["selected_osd_rate"] < row["development_tuned_osd_rate"] for row in rows),
                   timing=dict(update_intervals=len(intervals),
                               median_update_s=float(np.median(intervals)) if intervals else None,
                               p10_update_s=float(np.percentile(intervals, 10)) if intervals else None,
                               p90_update_s=float(np.percentile(intervals, 90)) if intervals else None,
                               definition="Consecutive logged update intervals, excluding intervals across validation; "
                                          "includes mining, sampling, optimization and cProfile overhead; batch 8 only"),
                   checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   validated_shots_replayed=sum(row["shots"] for row in rows), rows=rows,
                   uncertainty="Exploratory single-seed pilot; repeated validation selects this checkpoint. "
                               f"Paired p-values are post-selection diagnostics. Comparator uses a {len(tuning_factors)}-factor "
                               "development sweep. No main E1/E2 or final-test claim.")
    if not (args.directory/"replayed_validation.npz").exists():
        np.savez_compressed(args.directory/"replayed_validation.npz", **replay)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
