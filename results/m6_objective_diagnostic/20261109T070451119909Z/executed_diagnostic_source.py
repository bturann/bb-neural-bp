"""Inspect soft objectives, full-unroll behavior, and deployed first-convergence decisions."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import rankdata

from bbnbp.checkpoints import load_model, read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.io import git_sha, source_manifest
from bbnbp.losses import ParityLoss, bitwise_bce
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def failure_auc(losses, failures):
    """Ranking AUC: higher loss should identify failed Stage-1 decisions."""
    labels = np.asarray(failures, dtype=bool)
    positives = int(labels.sum())
    negatives = len(labels)-positives
    if not positives or not negatives:
        return None
    return float((rankdata(losses)[labels].sum()-positives*(positives+1)/2)/(positives*negatives))


def first_convergence(history, zero_syndrome):
    history = np.asarray(history, dtype=bool)
    first = np.full(history.shape[1], len(history)+1, dtype=np.int64)
    any_match = history.any(0)
    first[any_match] = history.argmax(0)[any_match]+1
    first[np.asarray(zero_syndrome, dtype=bool)] = 0
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default="configs/m6/cpu_objective_diagnostic.yaml")
    parser.add_argument("--output-root", default="results/m6_objective_diagnostic")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.settings).read_text())
    source = Path(settings["source"])
    config = load_config(source/"config.yaml")
    study = json.loads((source/"summary.json").read_text())
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    directory = Path(args.output_root)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"settings.yaml").write_text(yaml.safe_dump(settings))
    (directory/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    config.save(directory/"source_config.yaml")
    paths = {}
    for arm in ("parity", "bitwise"):
        best = Path(study["checkpoints"][arm]["path"])
        assert hashlib.sha256(best.read_bytes()).hexdigest() == study["checkpoints"][arm]["sha256"]
        paths[f"{arm}_selected"], paths[f"{arm}_last"] = best, best.parent/"last.pt"
    checkpoints = {name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             step=read_checkpoint(path)["step"]) for name,path in paths.items()}
    initial_payload = read_checkpoint(paths["parity_selected"])
    started, rows = time.perf_counter(), []
    print(f"Objective diagnostic bundle: {directory}", flush=True)
    try:
        for index, point in enumerate(study["points"]):
            sampler = make_sampler(config, point["p"])
            s_all, o_all, e_all = sampler.sample(config.eval.max_shots, np.random.default_rng(point["seed"]))
            with np.load(source/f"cases-p{index}.npz") as saved:
                np.testing.assert_array_equal(s_all, saved["syndromes"])
                np.testing.assert_array_equal(o_all, saved["observables"])
            count = min(settings["shots_per_p"], len(s_all))
            s, o, e = s_all[:count], o_all[:count], e_all[:count]
            models = {"initial": NeuralMinSum(sampler.problem, **initial_payload["model_config"])}
            models.update({name: load_model(path, sampler.problem) for name,path in paths.items()})
            arrays = {}
            for name, model in models.items():
                objective = ParityLoss(sampler.problem, config.train.beta_logical)
                values = {key: [] for key in ("parity_loss", "bitwise_loss", "final_parity_loss", "final_bitwise_loss",
                           "first_iteration", "converged", "last_converged", "first_logical_failed", "last_logical_failed",
                           "clipped_fraction", "lost_after_first")}
                gradient = None
                for start in range(0, count, settings["batch"]):
                    if time.perf_counter()-started > settings["wall_seconds"]:
                        raise TimeoutError("Objective diagnostic reached its CPU wall budget")
                    stop = min(count, start+settings["batch"])
                    syndromes, faults = s[start:stop], torch.from_numpy(e[start:stop])
                    with torch.no_grad():
                        outputs = model(syndromes)
                        decisions = [hard.cpu().numpy().astype(np.uint8) for hard in model.last_hard_decisions]
                        convergence = np.stack([(parity(sampler.problem.H, hard)==syndromes).all(1) for hard in decisions])
                        first = first_convergence(convergence, ~syndromes.any(1))
                        deployed = first <= model.T
                        correction = decisions[-1].copy()
                        correction[first==0] = 0
                        for t, hard in enumerate(decisions, 1):
                            correction[first==t] = hard[first==t]
                        first_logical = (~deployed | (parity(sampler.problem.L, correction)!=o[start:stop]).any(1))
                        last_logical = (~convergence[-1] | (parity(sampler.problem.L, decisions[-1])!=o[start:stop]).any(1))
                        parity_values = [objective(p, faults) for p in outputs]
                        bitwise_values = [bitwise_bce(p, faults) for p in outputs]
                        weights = torch.arange(1, model.T+1, dtype=outputs[0].dtype)
                        p_loss = (torch.stack(parity_values)*weights[:,None]).sum(0)/weights.sum()
                        b_loss = (torch.stack(bitwise_values)*weights[:,None]).sum(0)/weights.sum()
                        clipping = torch.stack([((p <= objective.eps)|(p >= 1-objective.eps)).float().mean(1) for p in outputs]).mean(0)
                        lost = ((~convergence) & (np.arange(1, model.T+1)[:,None] >= first[None,:]) & deployed[None,:]).any(0)
                    compiled = model.export_cpu()
                    expected, conv = compiled.decode_batch(syndromes)
                    np.testing.assert_array_equal(conv, deployed)
                    np.testing.assert_array_equal(expected, correction)
                    np.testing.assert_array_equal(compiled.last_iterations, np.where(deployed, first, model.T))
                    record = dict(parity_loss=p_loss.numpy(), bitwise_loss=b_loss.numpy(),
                                  final_parity_loss=parity_values[-1].numpy(), final_bitwise_loss=bitwise_values[-1].numpy(),
                                  first_iteration=first, converged=deployed, last_converged=convergence[-1],
                                  first_logical_failed=first_logical, last_logical_failed=last_logical,
                                  clipped_fraction=clipping.numpy(), lost_after_first=lost)
                    for key,value in record.items():
                        values[key].append(value)
                    if start == 0:
                        n = min(settings["gradient_probe_shots"], len(faults))
                        probe = outputs[-1][:n].detach().clone().requires_grad_()
                        pg, = torch.autograd.grad(objective(probe, faults[:n]).sum(), probe)
                        bg, = torch.autograd.grad(bitwise_bce(probe, faults[:n]).sum(), probe)
                        gradient = dict(shots=n, parity_zero_fraction=float((pg==0).float().mean()),
                                        bitwise_zero_fraction=float((bg==0).float().mean()),
                                        finite=bool(torch.isfinite(pg).all() and torch.isfinite(bg).all()))
                combined = {key:np.concatenate(value) for key,value in values.items()}
                arrays[name] = combined
                np.savez_compressed(directory/f"scores-p{index}-{name}.npz", **combined)
                failed = ~combined["converged"]
                row = dict(p=point["p"], model=name, shots=count, converged=int((~failed).sum()),
                           fallback_demand=float(failed.mean()), final_iteration_fallback=float((~combined["last_converged"]).mean()),
                           lost_by_final=int((combined["converged"] & ~combined["last_converged"]).sum()),
                           lost_at_any_later_iteration=int(combined["lost_after_first"].sum()),
                           mean_parity_loss=float(combined["parity_loss"].mean()), mean_bitwise_loss=float(combined["bitwise_loss"].mean()),
                           parity_failure_auc=failure_auc(combined["parity_loss"], failed),
                           bitwise_failure_auc=failure_auc(combined["bitwise_loss"], failed),
                           mean_clipped_probability_fraction=float(combined["clipped_fraction"].mean()), gradient_probe=gradient)
                rows.append(row)
                print(f"p={point['p']:g} {name}: fallback={row['fallback_demand']:.4f}, lost_by_final={row['lost_by_final']}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
            for row in rows[-len(models):]:
                name = row["model"]
                if name == "initial":
                    continue
                objective_name = "parity_loss" if name.startswith("parity") else "bitwise_loss"
                initial, current = arrays["initial"], arrays[name]
                lower_loss = current[objective_name] < initial[objective_name]
                regressed = initial["converged"] & ~current["converged"]
                improved = ~initial["converged"] & current["converged"]
                row.update(loss_down_cases=int(lower_loss.sum()), loss_down_but_convergence_regressed=int((lower_loss & regressed).sum()),
                           convergence_gained=int(improved.sum()), convergence_lost=int(regressed.sum()),
                           objective_change=float((current[objective_name]-initial[objective_name]).mean()))
        report = dict(purpose=settings["purpose"], settings=settings, checkpoints=checkpoints, rows=rows,
                      elapsed_s=time.perf_counter()-started, git_sha=git_sha(), final_test_data_used=False, gpu_seconds=0,
                      scope="Descriptive reuse of deterministic prefixes of saved development cases; no retraining, "
                            "model selection or independent performance claim. AUC is within each p; null when one class is absent. "
                            "Training uses every unrolled iteration; deployment freezes at first convergence as specified.")
        (directory/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(f"Saved objective diagnostic: {directory}", flush=True)
    except BaseException as exc:
        (directory/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
