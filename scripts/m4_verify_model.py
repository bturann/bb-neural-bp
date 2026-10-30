"""Verify configured CPU neural/Numba inference readiness without training a model."""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bbnbp.config import load_config
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.io import git_sha, source_manifest
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def verify(config, output_root):
    torch.set_num_threads(config.eval.threads)
    sampler = make_sampler(config, config.noise.p_grid[0])
    model = NeuralMinSum(sampler.problem, config.model.T, config.model.sharing,
                          config.model.ms_scaling, config.model.msg_clip)
    entropy = [config.seed, 4]
    s, o, e = sampler.sample(config.eval.batch, np.random.default_rng(np.random.SeedSequence(entropy)))
    expected, conv = model.decode_batch(s)
    compiled = model.export_cpu()
    actual, actual_conv = compiled.decode_batch(s)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_conv, conv)
    np.testing.assert_array_equal(compiled.last_iterations, model.last_iterations)
    np.testing.assert_array_equal(conv, (parity(sampler.problem.H, expected) == s).all(1))
    report = dict(trained=False, purpose="M4 inference readiness smoke; not E1 performance",
                  code=config.code.name, noise_model=config.noise.model, rounds=config.noise.rounds,
                  p=config.noise.p_grid[0], shots=len(s), seed_entropy=entropy,
                  n_checks=sampler.problem.H.shape[0], n_variables=sampler.problem.H.shape[1],
                  n_edges=sampler.problem.H.nnz, max_check_degree=int(model.chk_mask.sum(1).max()),
                  T=model.T, sharing=model.sharing, msg_clip=model.msg_clip,
                  weight_groups=model.w.shape[1], n_parameters=sum(x.numel() for x in model.parameters()),
                  message_tensor_bytes=len(s)*model.chk_idx.numel()*model.w.element_size(),
                  converged_shots=int(conv.sum()), torch_numba_exact_match=True,
                  dtype=str(model.w.dtype), platform=platform.platform(),
                  config_hash=config.config_hash, git_sha=git_sha(),
                  temporal_anchor="earliest affected detector round; not physical gate time" if model.sharing == "struct" else None)
    directory = Path(output_root)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    config.save(directory/"config.yaml")
    (directory/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    (directory/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    np.savez_compressed(directory/"inference.npz", syndromes=s, observables=o, faults=e,
                        corrections=actual, converged=actual_conv, iterations=compiled.last_iterations)
    print(json.dumps(report, indent=2), flush=True)
    print(f"M4 readiness evidence: {directory}", flush=True)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cl_bb72_m3_order7.yaml")
    parser.add_argument("--output-root", default="results/m4")
    args = parser.parse_args()
    verify(load_config(args.config), args.output_root)
