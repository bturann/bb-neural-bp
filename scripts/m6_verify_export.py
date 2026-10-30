"""Verify a trained circuit checkpoint's compiled CPU export on 10,000 dev shots."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from bbnbp.checkpoints import load_model
from bbnbp.config import load_config
from bbnbp.io import git_sha, source_manifest
from bbnbp.noise.factory import make_sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/m6/e1.yaml")
    parser.add_argument("--settings", default="configs/m6/verify_export.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Refusing to overwrite verification evidence")
    settings = yaml.safe_load(Path(args.settings).read_text())
    config = load_config(args.config)
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    sampler = make_sampler(config, settings["p"])
    model = load_model(args.checkpoint, sampler.problem)
    compiled = model.export_cpu()
    rng = np.random.default_rng(np.random.SeedSequence(settings["seed_entropy"]))
    hashes = {name: hashlib.sha256() for name in ("syndromes", "torch", "numba")}
    shots, converged = 0, 0
    while shots < settings["shots"]:
        count = min(settings["batch"], settings["shots"]-shots)
        s, _, errors = sampler.sample(count, rng)
        del errors
        expected, expected_conv = model.decode_batch(s)
        actual, actual_conv = compiled.decode_batch(s)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_conv, expected_conv)
        np.testing.assert_array_equal(compiled.last_iterations, model.last_iterations)
        hashes["syndromes"].update(s.tobytes())
        for name, correction in (("torch", expected), ("numba", actual)):
            hashes[name].update(correction.tobytes())
        converged += int(actual_conv.sum())
        shots += count
        if shots % 1024 == 0:
            print(f"Verified {shots}/{settings['shots']} shots", flush=True)
    report = dict(passed=True, settings=settings, config=config.to_dict(), shots=shots,
                  converged=converged, git_sha=git_sha(), source_manifest=source_manifest(),
                  checkpoint=args.checkpoint,
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  hashes={name: digest.hexdigest() for name, digest in hashes.items()},
                  checked="Exact corrections, convergence flags and first-convergence iteration counts",
                  final_test_data_used=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Exact match on {shots} trained circuit-level shots: {output}")


if __name__ == "__main__":
    main()
