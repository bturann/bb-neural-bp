"""Secondary PyTorch batched GPU throughput, separate from primary CPU latency."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
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
    parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--config", default="configs/m6/e1.yaml")
    parser.add_argument("--benchmark-config", default="configs/m6/throughput.yaml")
    parser.add_argument("--output-root", default="results/m6/throughput")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = yaml.safe_load(Path(args.benchmark_config).read_text())
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    available = torch.cuda.is_available() if args.device == "cuda" else torch.backends.mps.is_available()
    if not available:
        parser.error(f"GPU backend unavailable: {args.device}")
    synchronize = torch.cuda.synchronize if args.device == "cuda" else torch.mps.synchronize
    directory = Path(args.output_root)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    config.save(directory/"config.yaml")
    (directory/"benchmark.yaml").write_text(yaml.safe_dump(settings))
    (directory/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    rows = []
    try:
        for i, p in enumerate(settings["p_grid"]):
            sampler = make_sampler(config, p)
            batch, shots = settings["batch"], settings["shots"]
            rng = np.random.default_rng(np.random.SeedSequence([*settings["seed_entropy"], i]))
            warmup, _, _ = sampler.sample(batch*settings["warmup_batches"], rng)
            syndromes, _, _ = sampler.sample(shots, rng)
            # Device transfer and sampling are outside this secondary model-only timer.
            warmup = torch.from_numpy(warmup).to(args.device)
            inputs = torch.from_numpy(syndromes).to(args.device)
            for checkpoint in args.checkpoint:
                model = load_model(checkpoint, sampler.problem, device=args.device)
                with torch.inference_mode():
                    for start in range(0, len(warmup), batch):
                        model(warmup[start:start+batch], early_stop=True)
                    synchronize()
                    begin = time.perf_counter()
                    for start in range(0, shots, batch):
                        model(inputs[start:start+batch], early_stop=True)
                    synchronize()
                    elapsed = time.perf_counter()-begin
                rows.append(dict(p=p, checkpoint=checkpoint,
                                 checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                                 shots=shots, batch=batch, elapsed_s=elapsed, shots_per_s=shots/elapsed))
        report = dict(purpose=settings["purpose"], device=args.device,
                      device_name=torch.cuda.get_device_name() if args.device == "cuda" else "Apple MPS",
                      git_sha=git_sha(), rows=rows,
                      methodology="Batched Stage-1 PyTorch inference with first-convergence stopping; "
                                  "device-resident inputs, no OSD; synchronized before and after timing; "
                                  "includes model host synchronizations. Not primary per-shot latency.")
        (directory/"throughput.json").write_text(json.dumps(report, indent=2)+"\n")
    except BaseException as exc:
        (directory/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    print(directory)


if __name__ == "__main__":
    main()
