"""Confirm frozen development candidates without reusing selection or final-test cases."""

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


def check_seed_separation(config, preferences, payloads, tuning_rows):
    seeds={config.seed+config.eval.seed_offset+i for i in range(len(config.noise.p_grid))}
    excluded=set(preferences["validation_seeds"]) | {int(row["seed"]) for row in tuning_rows}
    excluded.update(config.seed+10000+i for i in range(len(config.noise.p_grid)))
    for payload in payloads.values():
        c=payload["config"]
        runtime=c["train"]["runtime"]
        excluded.update(c["seed"]+runtime["validation_seed_offset"]+i for i in range(len(c["noise"]["p_grid"])))
    if seeds & excluded:
        raise ValueError("Confirmation seeds overlap checkpoint selection, larger validation, tuning or final-test data")
    return sorted(seeds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default="configs/m6/cpu_independent_confirmation.yaml")
    parser.add_argument("--output-root",default="results/m6_independent_confirmation")
    args=parser.parse_args()
    settings=yaml.safe_load(Path(args.settings).read_text())
    preferences=json.loads(Path(settings["preferences"]).read_text())
    paths={arm:Path(preferences["candidates"][arm]["checkpoint"]["path"]) for arm in ("parity","bitwise")}
    payloads={arm:read_checkpoint(path) for arm,path in paths.items()}
    for arm,path in paths.items():
        expected=preferences["candidates"][arm]["checkpoint"]
        if payloads[arm]["step"]!=500 or hashlib.sha256(path.read_bytes()).hexdigest()!=expected["sha256"]:
            raise ValueError("Candidate differs from the frozen step-500 validation preference")
    if payloads["parity"]["model_config"]!=payloads["bitwise"]["model_config"]:
        raise ValueError("Frozen candidates must share the same initial architecture")
    c=load_config(settings["template"])
    c=replace(c,seed=settings["seed"],eval=replace(c.eval,seed_offset=settings["seed_offset"],
        max_shots=settings["shots_per_p"],batch=settings["batch"],stop_failures=settings["shots_per_p"]+1,
        checkpoints=tuple(str(path) for path in paths.values()))).validate()
    with Path(settings["tuning_table"]).open(newline="") as stream:
        tuning_rows=list(csv.DictReader(stream))
    check_seed_separation(c,preferences,payloads,tuning_rows)
    funcs=runpy.run_path("scripts/m6_compare_stage1.py")
    funcs["validate_tuning_table"](c,settings["tuning_table"])
    torch.set_num_threads(settings["threads"])
    set_num_threads(settings["threads"])
    stamp=datetime.now(timezone.utc)
    directory=Path(args.output_root)/(stamp.strftime("%Y%m%dT%H%M%S%fZ")+"-"+c.config_hash[:8])
    directory.mkdir(parents=True,exist_ok=False)
    c.save(directory/"config.yaml")
    (directory/"settings.yaml").write_text(yaml.safe_dump(settings))
    (directory/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,directory/"executed_source.py")
    shutil.copyfile(settings["preferences"],directory/"selection_record.json")
    shutil.copyfile(settings["tuning_table"],directory/"tuned_scaling.csv")
    shutil.copyfile(Path(settings["tuning_table"]).with_suffix(".config.yaml"),directory/"tuned_scaling.config.yaml")
    frozen={arm:dict(path=str(path),step=payloads[arm]["step"],sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for arm,path in paths.items()}
    for arm,path in paths.items():
        shutil.copyfile(path,directory/f"checkpoint-{arm}.pt")
    (directory/"frozen_candidates.json").write_text(json.dumps(frozen,indent=2)+"\n")
    print(f"Independent development confirmation: {directory}",flush=True)
    started,points=time.perf_counter(),[]
    flags_by_model={name:[] for name in ("bp_tuned","nbp_initial","nbp_parity","nbp_bitwise")}
    try:
        for i,p in enumerate(c.noise.p_grid):
            sampler=make_sampler(c,p)
            seed=c.seed+c.eval.seed_offset+i
            s,o,e=sampler.sample(c.eval.max_shots,np.random.default_rng(seed))
            del e
            factor=tuned_factor(settings["tuning_table"],c,p)
            models={f"nbp_{arm}":load_model(path,sampler.problem) for arm,path in paths.items()}
            dtype={"torch.float32":torch.float32,"torch.float64":torch.float64}[payloads["parity"]["dtype"]]
            models["nbp_initial"]=NeuralMinSum(sampler.problem,**payloads["parity"]["model_config"],dtype=dtype)
            decoders={"bp_tuned":LdpcBP(sampler.problem,c.model.T,factor)}
            for name,model in models.items():
                compiled=model.export_cpu()
                n=min(settings["export_check_shots"],len(s))
                expected,conv=model.decode_batch(s[:n])
                actual,actual_conv=compiled.decode_batch(s[:n])
                np.testing.assert_array_equal(expected,actual)
                np.testing.assert_array_equal(conv,actual_conv)
                np.testing.assert_array_equal(model.last_iterations,compiled.last_iterations)
                decoders[name]=compiled
            rows,flags,logical_flags=[],{},{}
            point=dict(p=p,seed=seed,shots=len(s),tuned_factor=factor,models={})
            for name in flags_by_model:
                if time.perf_counter()-started>settings["wall_seconds"]:
                    raise TimeoutError("Independent confirmation reached its CPU wall budget")
                failed,logical,iterations=_inference(decoders[name],s,o,sampler.problem,c.eval.batch)
                flags[name],logical_flags[name]=failed,logical
                flags_by_model[name].append(failed)
                point["models"][name]=dict(fallback_demand=float(failed.mean()),failures=int(failed.sum()),mean_iterations=iterations)
                lo,hi=wilson(int(logical.sum()),len(s))
                rows.append(dict(run_id=directory.name,git_sha=git_sha(),config_hash=c.config_hash,timestamp=stamp.isoformat(),
                    code=c.code.name,noise_model=c.noise.model,rounds=c.noise.rounds,basis=c.noise.basis,p=p,
                    decoder=name+"_only",stage1=name,T=c.model.T,weight_sharing=models[name].sharing if name in models else "scalar",
                    osd_method="",osd_order="",ms_scaling=payloads["parity"]["model_config"]["ms_scaling"] if name in models else factor,
                    shots=len(s),stage1_failures=int(failed.sum()),osd_calls=0,osd_rate=0,logical_failures=int(logical.sum()),
                    ler=float(logical.mean()),ler_lo=lo,ler_hi=hi,ler_stage1=float(logical.mean()),
                    lat_mean_us="",lat_p50_us="",lat_p99_us="",lat_max_us="",lat_threads=1,
                    device=f"CPU {cpu_model_label(c.eval.cpu_model)}; {platform.platform()}",seed=seed))
                print(f"p={p:g} {name}: fallback={failed.mean():.6f}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            append_results(directory/"results.csv",rows)
            np.savez_compressed(directory/f"cases-p{i}.npz",syndromes=s,observables=o)
            np.savez_compressed(directory/f"failed-p{i}.npz",**flags)
            np.savez_compressed(directory/f"logical-p{i}.npz",**logical_flags)
            point["primary"]=mcnemar(flags["bp_tuned"],flags["nbp_bitwise"])
            point["secondary"]={name:mcnemar(flags[name],flags["nbp_bitwise"]) for name in ("nbp_parity","nbp_initial")}
            points.append(point)
        for point,value in zip(points,funcs["holm"]([p["primary"]["pvalue"] for p in points])):
            point["primary"]["holm_pvalue"]=value
        secondary=[p["secondary"][name] for p in points for name in ("nbp_parity","nbp_initial")]
        for pair,value in zip(secondary,funcs["holm"]([p["pvalue"] for p in secondary])):
            pair["holm_pvalue"]=value
        means={name:float(np.mean([v.mean() for v in parts])) for name,parts in flags_by_model.items()}
        benefit=[p for p in points if p["models"]["nbp_bitwise"]["fallback_demand"]<p["models"]["bp_tuned"]["fallback_demand"] and p["primary"]["holm_pvalue"]<settings["alpha"]]
        harm=[p for p in points if p["models"]["nbp_bitwise"]["fallback_demand"]>p["models"]["bp_tuned"]["fallback_demand"] and p["primary"]["holm_pvalue"]<settings["alpha"]]
        differences=[a.astype(np.int8)-b.astype(np.int8) for a,b in zip(flags_by_model["nbp_bitwise"],flags_by_model["bp_tuned"])]
        bootstrap=runpy.run_path("scripts/m6_validation_stability.py")["paired_bootstrap"](differences,
            np.random.default_rng(np.random.SeedSequence(settings["bootstrap_seed_entropy"])),settings["bootstrap_replicates"],settings["confidence_level"])
        report=dict(purpose=settings["purpose"],settings=settings,frozen_candidates=frozen,points=points,mean_fallback=means,
            bce_minus_tuned_percentage_points=100*(means["nbp_bitwise"]-means["bp_tuned"]),paired_grid_mean_bootstrap=bootstrap,
            primary_significant_benefits=len(benefit),primary_significant_regressions=len(harm),
            clear_gain=means["nbp_bitwise"]<means["bp_tuned"] and bool(benefit) and not harm,
            elapsed_s=time.perf_counter()-started,git_sha=git_sha(),final_test_data_used=False,gpu_seconds=0,m6_complete=False,
            scope="Fixed-candidate, independent development confirmation after larger-validation selection. Primary: BCE versus tuned BP, Holm over five p values. "
                  "Secondary: BCE versus parity and untrained reference, Holm over ten tests. Bootstrap interval is descriptive. "
                  "No OSD calls or latency measurement; fallback demand=stage1_failures/shots; standalone LER counts nonconvergence. "
                  "Single training seed; main final-test data remain reserved.")
        (directory/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(json.dumps({key:report[key] for key in ("mean_fallback","bce_minus_tuned_percentage_points","clear_gain")},indent=2))
    except BaseException as exc:
        (directory/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__=="__main__":
    main()
