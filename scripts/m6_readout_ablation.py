"""Fresh-data comparison of matched full-unroll and first-convergence training arms."""

import argparse
import hashlib
import json
import platform
import runpy
import shutil
import time
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.checkpoints import load_model,read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.evaluate import cpu_model_label,tuned_factor
from bbnbp.io import append_results,git_sha,source_manifest
from bbnbp.metrics import mcnemar,wilson
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference


def verify_readout_control(full,frozen):
    a,b=full["config"],frozen["config"]
    if a.train.runtime.readout!="full_unroll" or b.train.runtime.readout!="first_converged":
        raise ValueError("Expected full-unroll and first-converged training arms")
    normalized=replace(b,train=replace(b.train,runtime=replace(b.train.runtime,readout="full_unroll")))
    if a!=normalized:
        raise ValueError("Controlled configurations may differ only in training readout")
    common=runpy.run_path("scripts/m6_loss_ablation.py")["verify_training_inputs"](full,frozen)
    return dict(configurations_differ_only_in_readout=True,**common)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full",type=Path,required=True)
    parser.add_argument("--frozen",type=Path,required=True)
    parser.add_argument("--settings",default="configs/m6/cpu_readout_study.yaml")
    parser.add_argument("--output-root",default="results/m6_cpu_readout/comparison")
    args=parser.parse_args()
    settings=yaml.safe_load(Path(args.settings).read_text())
    arm_loader=runpy.run_path("scripts/m6_loss_ablation.py")["training_arm"]
    arms={"full":arm_loader(args.full),"frozen":arm_loader(args.frozen)}
    controls=verify_readout_control(arms["full"],arms["frozen"])
    c=load_config("configs/m6/cpu_pilot_check.yaml")
    c=replace(c,eval=replace(c.eval,seed_offset=settings["evaluation_seed_offset"],max_shots=settings["evaluation_shots"],
        batch=settings["evaluation_batch"],stop_failures=settings["evaluation_shots"]+1,
        checkpoints=tuple(str(a["checkpoint"]) for a in arms.values()))).validate()
    completion=settings.get("completion_of")
    if completion:
        previous=Path(completion)
        if not (previous/"FAILURE.txt").exists() or "TimeoutError" not in (previous/"FAILURE.txt").read_text() or (previous/"summary.json").exists():
            raise ValueError("Completion requires a recorded, incomplete timeout")
        if load_config(previous/"config.yaml")!=c:
            raise ValueError("Completion must retain the exact original evaluation config")
        old_settings=yaml.safe_load((previous/"settings.yaml").read_text())
        for key in ("evaluation_seed_offset","evaluation_shots","evaluation_batch","threads","export_check_shots","alpha"):
            if settings[key]!=old_settings[key]:
                raise ValueError(f"Completion changed fixed study setting: {key}")
        for name,arm in arms.items():
            if (previous/f"checkpoint-{name}.pt").read_bytes()!=arm["checkpoint"].read_bytes():
                raise ValueError("Completion changed a frozen candidate")
    seeds={c.seed+c.eval.seed_offset+i for i in range(len(c.noise.p_grid))}
    selection={arms["full"]["config"].seed+arms["full"]["config"].train.runtime.validation_seed_offset+i for i in range(len(c.noise.p_grid))}
    if seeds & (selection | {c.seed+10000+i for i in range(len(c.noise.p_grid))} | {c.seed+c.sweep.seed_offset+i for i in range(len(c.noise.p_grid))}):
        raise ValueError("Fresh readout comparison overlaps selection, tuning or final seeds")
    helpers=runpy.run_path("scripts/m6_compare_stage1.py")
    table=Path(arms["full"]["config"].train.runtime.baseline_table)
    helpers["validate_tuning_table"](c,table)
    torch.set_num_threads(settings["threads"])
    set_num_threads(settings["threads"])
    stamp=datetime.now(timezone.utc)
    d=Path(args.output_root)/(stamp.strftime("%Y%m%dT%H%M%S%fZ")+"-"+c.config_hash[:8])
    d.mkdir(parents=True,exist_ok=False)
    c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    (d/"controls.json").write_text(json.dumps(controls,indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_source.py")
    shutil.copyfile(table,d/"tuned_scaling.csv")
    shutil.copyfile(table.with_suffix(".config.yaml"),d/"tuned_scaling.config.yaml")
    for name,arm in arms.items():
        shutil.copyfile(arm["checkpoint"],d/f"checkpoint-{name}.pt")
        arm["config"].save(d/f"training-{name}.yaml")
    starts=[datetime.strptime(a["directory"].name.split("-")[0],"%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc) for a in arms.values()]
    prior=(stamp-min(starts)).total_seconds()
    limit=min(settings["comparison_wall_seconds"],settings["wall_seconds"] if completion else settings["wall_seconds"]-prior)
    if limit<=0:
        raise TimeoutError("Readout study exhausted its overall wall budget")
    print(f"Readout comparison: {d}; remaining budget {limit:.1f}s",flush=True)
    started,points=time.perf_counter(),[]
    labels=("bp_tuned","nbp_initial","nbp_full","nbp_frozen")
    try:
        for i,p in enumerate(c.noise.p_grid):
            sampler=make_sampler(c,p)
            seed=c.seed+c.eval.seed_offset+i
            s,o,e=sampler.sample(c.eval.max_shots,np.random.default_rng(seed))
            del e
            factor=tuned_factor(table,c,p)
            models={f"nbp_{name}":load_model(a["checkpoint"],sampler.problem) for name,a in arms.items()}
            models["nbp_initial"]=NeuralMinSum(sampler.problem,**arms["full"]["payload"]["model_config"],dtype=models["nbp_full"].w.dtype)
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
            flags,logical_flags,rows={},{},[]
            point=dict(p=p,seed=seed,shots=len(s),tuned_factor=factor,models={})
            for name in labels:
                if time.perf_counter()-started>limit:
                    raise TimeoutError("Readout study reached its comparison wall budget")
                failed,logical,iterations=_inference(decoders[name],s,o,sampler.problem,c.eval.batch)
                flags[name],logical_flags[name]=failed,logical
                point["models"][name]=dict(fallback_demand=float(failed.mean()),failures=int(failed.sum()),mean_iterations=iterations)
                lo,hi=wilson(int(logical.sum()),len(s))
                rows.append(dict(run_id=d.name,git_sha=git_sha(),config_hash=c.config_hash,timestamp=stamp.isoformat(),code=c.code.name,
                    noise_model=c.noise.model,rounds=c.noise.rounds,basis=c.noise.basis,p=p,decoder=name+"_only",stage1=name,T=c.model.T,
                    weight_sharing=models[name].sharing if name in models else "scalar",osd_method="",osd_order="",
                    ms_scaling=arms["full"]["payload"]["model_config"]["ms_scaling"] if name in models else factor,shots=len(s),
                    stage1_failures=int(failed.sum()),osd_calls=0,osd_rate=0,logical_failures=int(logical.sum()),ler=float(logical.mean()),
                    ler_lo=lo,ler_hi=hi,ler_stage1=float(logical.mean()),lat_mean_us="",lat_p50_us="",lat_p99_us="",lat_max_us="",lat_threads=1,
                    device=f"CPU {cpu_model_label(c.eval.cpu_model)}; {platform.platform()}",seed=seed))
                print(f"p={p:g} {name}: fallback={failed.mean():.6f}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            append_results(d/"results.csv",rows)
            np.savez_compressed(d/f"cases-p{i}.npz",syndromes=s,observables=o)
            np.savez_compressed(d/f"failed-p{i}.npz",**flags)
            np.savez_compressed(d/f"logical-p{i}.npz",**logical_flags)
            point["primary"]=mcnemar(flags["nbp_full"],flags["nbp_frozen"])
            point["secondary"]={name:mcnemar(flags[name],flags["nbp_frozen"]) for name in ("bp_tuned","nbp_initial")}
            points.append(point)
        for point,value in zip(points,helpers["holm"]([p["primary"]["pvalue"] for p in points])):
            point["primary"]["holm_pvalue"]=value
        pairs=[p["secondary"][name] for p in points for name in ("bp_tuned","nbp_initial")]
        for pair,value in zip(pairs,helpers["holm"]([p["pvalue"] for p in pairs])):
            pair["holm_pvalue"]=value
        means={name:float(np.mean([p["models"][name]["fallback_demand"] for p in points])) for name in labels}
        checkpoints={name:dict(path=str(a["checkpoint"]),step=a["payload"]["step"],sha256=hashlib.sha256(a["checkpoint"].read_bytes()).hexdigest()) for name,a in arms.items()}
        report=dict(purpose=settings["purpose"],settings=settings,controls=controls,points=points,mean_fallback=means,checkpoints=checkpoints,
            training_roots=dict(full=str(args.full),frozen=str(args.frozen)),
            frozen_minus_full_percentage_points=100*(means["nbp_frozen"]-means["nbp_full"]),
            significant_primary_points=sum(p["primary"]["holm_pvalue"]<settings["alpha"] for p in points),
            elapsed_s=time.perf_counter()-started,total_study_elapsed_s=prior+time.perf_counter()-started,
            completion_of=completion,budget_scope="comparison_only" if completion else "overall_study",
            final_test_data_used=False,gpu_seconds=0,m6_complete=False,
            scope="Single-seed controlled development training-readout ablation. Both use unchanged parity/logical loss and identical actual batches. "
                  "Primary Holm family: frozen/full at five p values; secondary family: frozen/tuned and frozen/initial across ten tests. "
                  "Canonical first-convergence inference is identical for both arms. No OSD, two-stage LER or latency claim.")
        (d/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(json.dumps({key:report[key] for key in ("mean_fallback","frozen_minus_full_percentage_points","significant_primary_points","total_study_elapsed_s")},indent=2))
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__=="__main__":
    main()
