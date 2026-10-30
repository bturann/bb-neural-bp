"""Measure deterministic GPU training mechanics without touching final-test streams."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from bbnbp.config import load_config
from bbnbp.io import git_sha, source_manifest
from bbnbp.losses import ParityLoss, multi_loss
from bbnbp.noise.factory import make_sampler
from bbnbp.train import make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "mps"), required=True)
    parser.add_argument("--config", default="configs/m6/train_mps_1234.yaml")
    parser.add_argument("--settings", default="configs/m6/preflight.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        parser.error("Preflight receipt already exists; select a new output")
    settings = yaml.safe_load(Path(args.settings).read_text())
    config = load_config(args.config)
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    synchronize = torch.cuda.synchronize if args.device == "cuda" else torch.mps.synchronize
    sampler = make_sampler(config, settings["p"])
    rng = np.random.default_rng(np.random.SeedSequence(settings["seed_entropy"]))
    rows = []
    for batch in settings["batch_sizes"]:
        s, _, faults = sampler.sample(batch, rng)
        model = make_model(sampler.problem, config).to(args.device)
        objective = ParityLoss(sampler.problem, config.train.beta_logical)
        gradients, elapsed, losses = [], [], []
        for _ in range(settings["steps_per_batch"]):
            model.zero_grad(set_to_none=True)
            synchronize()
            start = time.perf_counter()
            outputs = model(torch.from_numpy(s).to(args.device))
            loss = multi_loss([p.cpu() for p in outputs], torch.from_numpy(faults), objective)
            loss.backward()
            synchronize()
            elapsed.append(time.perf_counter()-start)
            losses.append(float(loss.detach()))
            gradients.append(torch.cat([p.grad.cpu().flatten() for p in model.parameters()]))
        repeated = all(torch.equal(gradients[0], grad) for grad in gradients[1:])
        if not repeated:
            raise AssertionError("Repeated deterministic gradients differ")
        rows.append(dict(batch=batch, elapsed_s=elapsed, losses=losses,
                         warm_seconds_per_backward=float(np.mean(elapsed[settings["warmup_steps"]:])),
                         repeated_gradients_bit_identical=repeated,
                         finite_gradients=bool(torch.isfinite(gradients[0]).all())))
    report = dict(purpose="development_preflight", device=args.device, git_sha=git_sha(),
                  config=config.to_dict(), settings=settings, results=rows,
                  source_manifest=source_manifest(),
                  methodology="Repeated forward+CPU parity loss+backward on identical batches; "
                              "no optimizer, mining or validation; not an end-to-end training estimate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
