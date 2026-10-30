"""Tune BP scaling on validation shots, independently for each p and T budget."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.io import git_sha, source_manifest
from bbnbp.noise.code_capacity import CodeCapacitySampler
from bbnbp.noise.base import parity


def run(config, output):
    if config.noise.model != "code_capacity":
        raise ValueError("This M2 sweep currently supports code_capacity only")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite tuning evidence: {output}")
    rows = []
    sha = git_sha()
    for i, p in enumerate(config.noise.p_grid):
        sampler = CodeCapacitySampler(config.code.name, p, config.noise.basis)
        seed = config.seed + config.sweep.seed_offset + i
        s, obs, _ = sampler.sample(config.sweep.shots, np.random.default_rng(seed))
        for T in config.sweep.iterations:
            candidates = []
            for factor in config.sweep.factors:
                decoder = LdpcBP(sampler.problem, T, factor)
                e_hat, conv = decoder.decode_batch(s)
                failures = (~conv | (parity(sampler.problem.L, e_hat) != obs).any(axis=1))
                candidates.append(dict(code=config.code.name, noise_model=config.noise.model,
                                       rounds=config.noise.rounds, basis=config.noise.basis,
                                       p=p, T=T, ms_scaling=factor, shots=len(s),
                                       nonconverged=int((~conv).sum()),
                                       ler_stage1=float(failures.mean()), seed=seed,
                                       config_hash=config.config_hash, git_sha=sha))
            best = min(candidates, key=lambda r: (r["nonconverged"], r["ler_stage1"], r["ms_scaling"]))
            for row in candidates:
                row["selected"] = row is best
            rows.extend(candidates)
            print(f"p={p:g} T={T}: alpha={best['ms_scaling']:g}, "
                  f"nonconvergence={best['nonconverged']/len(s):.4f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config.save(output.with_suffix(".config.yaml"))
    output.with_suffix(".meta.json").write_text(json.dumps(
        dict(timestamp=datetime.now(timezone.utc).isoformat(), purpose="validation_tuning",
             source_manifest=source_manifest(),
             selection="min nonconvergence, then min stage1 LER, then lower factor"), indent=2)+"\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/tuned_scaling.csv")
    args = parser.parse_args()
    run(load_config(args.config), args.output)
