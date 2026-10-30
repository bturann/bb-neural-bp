"""Summarize independent training seeds without pooling their paired test shots."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def seed_summary(rows, checkpoints):
    seeds = [int(checkpoint["training_seed"]) for checkpoint in checkpoints]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Main summary requires three distinct training seeds")
    summary = []
    for p in sorted({float(row["p"]) for row in rows}):
        selected = [row for row in rows if float(row["p"]) == p]
        models = []
        for index, seed in enumerate(seeds):
            matches = [row for row in selected if row["stage1"] == f"nbp_{index}"]
            if len(matches) != 1:
                raise ValueError(f"Expected one result for p={p}, training seed {seed}")
            row = matches[0]
            models.append(dict(training_seed=seed, osd_rate=float(row["osd_rate"]),
                               shots=int(row["shots"]), logical_failures=int(row["logical_failures"]),
                               ler=float(row["ler"]), ler_lo=float(row["ler_lo"]), ler_hi=float(row["ler_hi"])))
        if len({model["shots"] for model in models}) != 1:
            raise ValueError("Training-seed models must use paired shots at each p")
        rates = [model["osd_rate"] for model in models]
        summary.append(dict(p=p, osd_rate_mean=float(np.mean(rates)),
                            osd_rate_std=float(np.std(rates, ddof=1)), models=models))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1", type=Path, required=True)
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a summary")
    with (args.e1/"results.csv").open(newline="") as f:
        e1 = list(csv.DictReader(f))
    with (args.e2/"results.csv").open(newline="") as f:
        e2 = list(csv.DictReader(f))
    d1 = json.loads((args.e1/"diagnostics.json").read_text())
    d2 = json.loads((args.e2/"diagnostics.json").read_text())
    if d1["purpose"] != "final" or d2["purpose"] != "final":
        parser.error("Main summary requires final evaluations")
    identities = lambda d: sorted((c["sha256"], c["training_seed"]) for c in d["checkpoints"])
    if identities(d1) != identities(d2):
        parser.error("E1 and E2 must use the same three selected checkpoints")
    if {float(row["p"]) for row in e1} != {0.001, 0.002, 0.003, 0.004, 0.005}:
        parser.error("E1 grid is incomplete")
    if {float(row["p"]) for row in e2} != {0.001, 0.003}:
        parser.error("E2 grid is incomplete")
    expected_names = {"bp", "bp_tuned", "nbp_0", "nbp_1", "nbp_2"}
    for rows in (e1, e2):
        for p in {row["p"] for row in rows}:
            selected = [row for row in rows if row["p"] == p]
            if len(selected) != 5 or {row["stage1"] for row in selected} != expected_names:
                parser.error("Each p requires two baselines and three neural model rows")
            if len({int(row["shots"]) for row in selected}) != 1:
                parser.error("All decoders must use paired shots at each p")
            if any(row["code"] != "bb72" or row["rounds"] != "6"
                   or row["noise_model"] != "circuit_level" or row["lat_threads"] != "1"
                   or row["T"] != "15" or row["osd_order"] != "7" for row in selected):
                parser.error("Rows do not match the main M6 experiment")
    if d1["cpu_model"] != d2["cpu_model"]:
        parser.error("E1/E2 machine labels differ")
    for index in range(2):
        for name in ("bp", "bp_tuned", "nbp_0", "nbp_1", "nbp_2"):
            latency = d2[f"latency-p{index}:{name}"]
            if latency["warmup"] != 1000 or latency["shots"] != 20000:
                parser.error("E2 requires 1,000 warmup and 20,000 measured shots")
    report = dict(e1_directory=str(args.e1), e2_directory=str(args.e2),
                  checkpoints=d1["checkpoints"], e1_neural=seed_summary(e1, d1["checkpoints"]),
                  e1_rows=e1, e2_rows=e2,
                  uncertainty="Sample standard deviation across three independently trained models. "
                              "LER and Wilson intervals remain separate per model; the same test shots "
                              "are never pooled as independent observations across models.",
                  scope="E1/E2 summary only; M6 acceptance also requires the training and throughput evidence.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
