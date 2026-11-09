"""Measure ranking stability among frozen checkpoints on larger fixed dev validation."""

import argparse
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
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.evaluate import cpu_model_label
from bbnbp.io import append_results, git_sha, source_manifest
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference


def paired_bootstrap(differences, rng, replicates, confidence):
    """Empirical paired bootstrap within each noise stratum via category counts."""
    draws = np.zeros(replicates)
    for values in differences:
        values = np.asarray(values, dtype=np.int8)
        counts = np.bincount(values+1, minlength=3)
        sample = rng.multinomial(len(values), counts/len(values), size=replicates)
        draws += (sample[:,2]-sample[:,0])/len(values)/len(differences)
    tail = (1-confidence)/2
    return dict(low=float(np.quantile(draws,tail)), high=float(np.quantile(draws,1-tail)),
                fraction_last_lower=float((draws<0).mean()), fraction_selected_lower=float((draws>0).mean()))


def block_preferences(selected, last, block_size):
    if any(len(values)%block_size for values in selected+last):
        raise ValueError("Validation blocks must divide every noise stratum exactly")
    n_blocks = len(selected[0])//block_size
    choices = []
    for block in range(n_blocks):
        part = slice(block*block_size,(block+1)*block_size)
        a = float(np.mean([values[part].mean() for values in selected]))
        b = float(np.mean([values[part].mean() for values in last]))
        choices.append(dict(block=block, selected_rate=a,last_rate=b,
                            preference="last" if b<a else "selected" if a<b else "tie"))
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default="configs/m6/cpu_validation_stability.yaml")
    parser.add_argument("--output-root",default="results/m6_validation_stability")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.settings).read_text())
    source = Path(settings["source"])
    original = json.loads((source/"summary.json").read_text())
    c = load_config(source/"config.yaml")
    source_config = c
    if (settings["shots_per_p"] % settings["small_block_shots"]
            or settings["tie_policy"] != "keep_original_selected"):
        raise ValueError("Validation blocks must divide the pool and ties must keep the original selection")
    torch.set_num_threads(settings["threads"])
    set_num_threads(settings["threads"])
    candidate_paths = {}
    checkpoint_info = {}
    old_validation_seeds = set()
    for arm in ("parity","bitwise"):
        best = Path(original["checkpoints"][arm]["path"])
        assert hashlib.sha256(best.read_bytes()).hexdigest()==original["checkpoints"][arm]["sha256"]
        for state,path in (("selected",best),("last",best.parent/"last.pt")):
            name=f"{arm}_{state}"
            candidate_paths[name]=path
            payload=read_checkpoint(path)
            checkpoint_info[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=payload["step"])
            runtime=payload["config"]["train"]["runtime"]
            old_validation_seeds.update(payload["config"]["seed"]+runtime["validation_seed_offset"]+i for i in range(len(c.noise.p_grid)))
    initial_payload=read_checkpoint(candidate_paths["parity_selected"])
    c=replace(c,seed=settings["seed"],
              model=replace(c.model,**initial_payload["model_config"]),
              eval=replace(c.eval,seed_offset=settings["validation_seed_offset"],
                           max_shots=settings["shots_per_p"],batch=settings["batch"],
                           stop_failures=settings["shots_per_p"]+1,
                           checkpoints=tuple(str(path) for path in candidate_paths.values()))).validate()
    seeds={settings["seed"]+settings["validation_seed_offset"]+i for i in range(len(c.noise.p_grid))}
    prohibited=old_validation_seeds | {point["seed"] for point in original["points"]} | {c.seed+10000+i for i in range(len(c.noise.p_grid))} | {c.seed+c.sweep.seed_offset+i for i in range(len(c.noise.p_grid))}
    if seeds & prohibited:
        raise ValueError("Larger validation seeds overlap training selection, tuning, comparison or final seeds")
    stamp=datetime.now(timezone.utc)
    directory=Path(args.output_root)/stamp.strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True,exist_ok=False)
    (directory/"settings.yaml").write_text(yaml.safe_dump(settings))
    source_config.save(directory/"source_config.yaml")
    c.save(directory/"config.yaml")
    (directory/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,directory/"executed_source.py")
    (directory/"checkpoint_manifest.json").write_text(json.dumps(checkpoint_info,indent=2)+"\n")
    started,points= time.perf_counter(),[]
    by_model={name:[] for name in ("initial",*candidate_paths)}
    dtype={"torch.float32":torch.float32,"torch.float64":torch.float64}
    print(f"Larger fixed validation bundle: {directory}",flush=True)
    try:
        for index,p in enumerate(c.noise.p_grid):
            sampler=make_sampler(c,p)
            seed=settings["seed"]+settings["validation_seed_offset"]+index
            s,o,e=sampler.sample(settings["shots_per_p"],np.random.default_rng(seed))
            del e
            np.savez_compressed(directory/f"cases-p{index}.npz",syndromes=s,observables=o)
            models={"initial":NeuralMinSum(sampler.problem,**initial_payload["model_config"],dtype=dtype[initial_payload["dtype"]])}
            models.update({name:load_model(path,sampler.problem) for name,path in candidate_paths.items()})
            flags,logical_flags,results={},{},{}
            rows=[]
            for name,model in models.items():
                if time.perf_counter()-started>settings["wall_seconds"]:
                    raise TimeoutError("Larger validation reached its configured CPU wall budget")
                compiled=model.export_cpu()
                n=min(settings["export_check_shots"],len(s))
                expected,conv=model.decode_batch(s[:n])
                actual,actual_conv=compiled.decode_batch(s[:n])
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(actual_conv,conv)
                np.testing.assert_array_equal(compiled.last_iterations,model.last_iterations)
                failed,logical,iterations=_inference(compiled,s,o,sampler.problem,settings["batch"])
                by_model[name].append(failed)
                flags[name],logical_flags[name]=failed,logical
                results[name]=dict(fallback_demand=float(failed.mean()),failures=int(failed.sum()),mean_iterations=iterations)
                lo,hi=wilson(int(logical.sum()),len(s))
                rows.append(dict(run_id=directory.name,git_sha=git_sha(),config_hash=c.config_hash,timestamp=stamp.isoformat(),
                    code=c.code.name,noise_model=c.noise.model,rounds=c.noise.rounds,basis=c.noise.basis,p=p,
                    decoder=f"nbp_{name}_only",stage1=f"nbp_{name}",T=model.T,weight_sharing=model.sharing,
                    osd_method="",osd_order="",ms_scaling=initial_payload["model_config"]["ms_scaling"],shots=len(s),
                    stage1_failures=int(failed.sum()),osd_calls=0,osd_rate=0,logical_failures=int(logical.sum()),
                    ler=float(logical.mean()),ler_lo=lo,ler_hi=hi,ler_stage1=float(logical.mean()),lat_mean_us="",lat_p50_us="",lat_p99_us="",lat_max_us="",lat_threads=1,
                    device=f"CPU {cpu_model_label(c.eval.cpu_model)}; {platform.platform()}",seed=seed))
                print(f"p={p:g} {name}: fallback={failed.mean():.6f}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            append_results(directory/"results.csv",rows)
            np.savez_compressed(directory/f"failed-p{index}.npz",**flags)
            np.savez_compressed(directory/f"logical-p{index}.npz",**logical_flags)
            points.append(dict(p=p,seed=seed,shots=len(s),models=results,
                paired={arm:mcnemar(flags[f"{arm}_selected"],flags[f"{arm}_last"]) for arm in ("parity","bitwise")}))
        helpers=runpy.run_path("scripts/m6_compare_stage1.py")
        pairs=[p["paired"][arm] for p in points for arm in ("parity","bitwise")]
        for pair,value in zip(pairs,helpers["holm"]([p["pvalue"] for p in pairs])):
            pair["holm_pvalue"]=value
        means={name:float(np.mean([values.mean() for values in parts])) for name,parts in by_model.items()}
        rng=np.random.default_rng(np.random.SeedSequence(settings["bootstrap_seed_entropy"]))
        rankings={}
        for arm in ("parity","bitwise"):
            selected,last=by_model[f"{arm}_selected"],by_model[f"{arm}_last"]
            differences=[b.astype(np.int8)-a.astype(np.int8) for a,b in zip(selected,last)]
            rankings[arm]=dict(selected_mean=means[f"{arm}_selected"],last_mean=means[f"{arm}_last"],
                last_minus_selected=float(np.mean([d.mean() for d in differences])),
                validation_preference="last" if means[f"{arm}_last"]<means[f"{arm}_selected"] else "selected",
                small_blocks=block_preferences(selected,last,settings["small_block_shots"]),
                paired_bootstrap=paired_bootstrap(differences,rng,settings["bootstrap_replicates"],settings["confidence_level"]))
        report=dict(purpose=settings["purpose"],settings=settings,checkpoints=checkpoint_info,points=points,
            mean_fallback=means,rankings=rankings,elapsed_s=time.perf_counter()-started,git_sha=git_sha(),
            final_test_data_used=False,gpu_seconds=0,original_selected_checkpoints_changed=False,
            scope="New fixed development validation among available frozen selected/final candidates only. "
                  "Each 256-shot/p block is a disjoint subset of the larger validation set, not an independent assessment of its winner. "
                  "Paired bootstrap intervals are descriptive and conditional on fixed checkpoints. Bootstrap fractions are not posterior probabilities. "
                  "Validation preference is not an independent performance result; no retraining or original checkpoint replacement.")
        (directory/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(json.dumps(dict(mean_fallback=means,rankings=rankings),indent=2),flush=True)
    except BaseException as exc:
        (directory/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__=="__main__":
    main()
