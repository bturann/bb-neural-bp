"""Frozen learned30 after strong BP100, with BP130 and ordinary two-pass controls."""

import argparse
import hashlib
import json
import runpy
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.io import source_manifest
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("long_controls",nargs=2,type=Path)
    parser.add_argument("--settings",default="configs/m6/cpu_strong_cascade.yaml")
    args=parser.parse_args();settings=yaml.safe_load(Path(args.settings).read_text())
    confirmation=Path(settings["primary_confirmation"])
    primary=json.loads((confirmation/"summary.json").read_text())
    c=load_config(confirmation/"config.yaml")
    for root in args.long_controls:
        if not (root/"summary.json").exists():raise ValueError("Both longer-control challenges must complete first")
    if (confirmation/"strong_cascade_receipt.json").exists():raise ValueError("Strong cascade study was already requested")
    d=Path("results/m6_strong_cascade")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    (d/"settings.yaml").write_text(yaml.safe_dump(settings));c.save(d/"config.yaml")
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_source.py")
    for name in ("selected-source.pt","models_source.py","neural_extensions_source.py"):
        shutil.copyfile(confirmation/name,d/name)
    assert hashlib.sha256((d/"selected-source.pt").read_bytes()).hexdigest()==primary["checkpoint_sha256"]
    (confirmation/"strong_cascade_receipt.json").write_text(json.dumps(dict(directory=str(d),
        requested_at_utc=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")
    torch.set_num_threads(1);set_num_threads(1)
    helper=runpy.run_path("scripts/m6_cascade_tune.py");decode,route=helper["decode"],helper["route"]
    builder=runpy.run_path(str(d/"models_source.py"))
    started=time.perf_counter()
    def deadline(signum,frame):raise TimeoutError("Strong cascade study reached its CPU budget")
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds"])
    try:
        selections=[]
        for index,p in enumerate(c.noise.p_grid):
            sampler=make_sampler(c,p)
            source=Path(settings["validation_sources"][index])
            long=json.loads((args.long_controls[index]/"selection.json").read_text())
            assert json.loads((args.long_controls[index]/"summary.json").read_text())["p"]==p
            assert long["validation_cases_sha256"]==hashlib.sha256((source/"cases.npz").read_bytes()).hexdigest()
            first_choice=long["choices"][str(settings["first_iterations"])]
            with np.load(source/"cases.npz") as saved:s,o=saved["syndromes"],saved["observables"]
            first=decode(LdpcBP(sampler.problem,settings["first_iterations"],first_choice["factor"]),s,o,sampler.problem,c.eval.batch)
            np.savez_compressed(d/f"validation-p{index}-bp100.npz",**first)
            mask=first["failed"]
            records=[dict(name="bp100",T=settings["first_iterations"],total_T=settings["first_iterations"],factor=first_choice["factor"],
                failures=int(mask.sum()),logical_failures=int(first["logical_failed"].sum()))]
            for factor in c.sweep.factors:
                if datetime.now(timezone.utc)>=datetime.fromisoformat(settings["selection_deadline_utc"]):raise TimeoutError("Strong cascade selection deadline reached")
                longer=decode(LdpcBP(sampler.problem,settings["long_iterations"],factor),s,o,sampler.problem,c.eval.batch)
                second=decode(LdpcBP(sampler.problem,settings["second_iterations"],factor),s[mask],o[mask],sampler.problem,c.eval.batch)
                two=route(first,second,mask)
                for name,value,T in (("bp130",longer,settings["long_iterations"]),("two_bp",two,settings["second_iterations"])):
                    np.savez_compressed(d/f"validation-p{index}-{name}-{factor:g}.npz",**value)
                    records.append(dict(name=name,T=T,total_T=settings["long_iterations"],factor=factor,failures=int(value["failed"].sum()),logical_failures=int(value["logical_failed"].sum())))
                print(f"strong cascade tuning p={p:g} factor={factor:g}: long={longer['failed'].sum()}, two={two['failed'].sum()}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            neural={}
            for name,initial in (("strong_cascade",False),("strong_init",True)):
                model=builder["neural_model"](d/"selected-source.pt",sampler.problem,"cycle30",initial=initial)
                node=model.export_cpu()
                a,conv=model.decode_batch(s[mask][:64]);b,bconv=node.decode_batch(s[mask][:64])
                np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(conv,bconv)
                neural[name]=route(first,decode(node,s[mask],o[mask],sampler.problem,c.eval.batch),mask)
                np.savez_compressed(d/f"validation-p{index}-{name}.npz",**neural[name])
            choices={name:min([row for row in records if row["name"]==name],key=lambda row:(row["failures"],row["logical_failures"],row["factor"])) for name in ("bp100","bp130","two_bp")}
            best=min(choices.values(),key=lambda row:(row["failures"],row["logical_failures"],row["total_T"],row["name"]))
            selections.append(dict(p=p,first=first_choice,choices=choices,best_ordinary=best,
                grid=records,validation_source=str(source),validation_cases_sha256=hashlib.sha256((source/"cases.npz").read_bytes()).hexdigest(),
                learned_validation_failures=int(neural["strong_cascade"]["failed"].sum()),
                initialized_validation_failures=int(neural["strong_init"]["failed"].sum())))
        frozen=dict(points=selections,checkpoint_sha256=primary["checkpoint_sha256"],
            source_confirmation=str(confirmation),long_controls=list(map(str,args.long_controls)),
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),model_design="bp100_then_cycle30",additional_optimizer_updates=0)
        (d/"selection.json").write_text(json.dumps(frozen,indent=2)+"\n")
        # Nothing below this point selects a model, factor, pipeline or sample size.
        (d/"confirmation_receipt.json").write_text(json.dumps(dict(
            started_at_utc=datetime.now(timezone.utc).isoformat(),seeds=settings["confirmation_seeds"],
            shots_per_point=settings["shots"]),indent=2)+"\n")
        points=[]
        for index,choice in enumerate(selections):
            if datetime.now(timezone.utc)>=datetime.fromisoformat(settings["deadline_utc"]):raise TimeoutError("Strong cascade assessment deadline reached")
            sampler=make_sampler(c,choice["p"])
            s,o,e=sampler.sample(settings["shots"],np.random.default_rng(settings["confirmation_seeds"][index]))
            np.testing.assert_array_equal(parity(sampler.problem.H,e),s);np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
            np.savez_compressed(d/f"cases-p{index}.npz",syndromes=s,observables=o,faults=e)
            first=decode(LdpcBP(sampler.problem,settings["first_iterations"],choice["first"]["factor"]),s,o,sampler.problem,c.eval.batch)
            values={"bp100":first};mask=first["failed"]
            for name,initial in (("strong_cascade",False),("strong_init",True)):
                node=builder["neural_model"](d/"selected-source.pt",sampler.problem,"cycle30",initial=initial).export_cpu()
                values[name]=route(first,decode(node,s[mask],o[mask],sampler.problem,c.eval.batch),mask)
            for name in ("bp130","two_bp"):
                spec=choice["choices"][name]
                if name=="bp130":values[name]=decode(LdpcBP(sampler.problem,settings["long_iterations"],spec["factor"]),s,o,sampler.problem,c.eval.batch)
                else:values[name]=route(first,decode(LdpcBP(sampler.problem,settings["second_iterations"],spec["factor"]),s[mask],o[mask],sampler.problem,c.eval.batch),mask)
            for name,value in values.items():
                np.testing.assert_array_equal(value["failed"],~(parity(sampler.problem.H,value["corrections"])==s).all(1))
                np.savez_compressed(d/f"values-p{index}-{name}.npz",**value)
            best=choice["best_ordinary"]["name"]
            points.append(dict(p=choice["p"],seed=settings["confirmation_seeds"][index],shots=len(s),best_ordinary=best,
                failures={name:int(value["failed"].sum()) for name,value in values.items()},
                wrong_logical_converged={name:int((~value["failed"] & value["logical_failed"]).sum()) for name,value in values.items()},
                primary=mcnemar(values[best]["failed"],values["strong_cascade"]["failed"]),
                learning=mcnemar(values["strong_init"]["failed"],values["strong_cascade"]["failed"])))
            print(f"strong cascade fresh p={choice['p']:g}: {points[-1]['failures']}",flush=True)
        holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
        for family in ("primary","learning"):
            for point,value in zip(points,holm([point[family]["pvalue"] for point in points])):point[family]["holm_pvalue"]=value
        report=dict(points=points,selection_directory=str(d),primary_checkpoint_sha256=primary["checkpoint_sha256"],
            elapsed_s=time.perf_counter()-started,scope=settings["scope"],final_test_data_used=False,gpu_seconds=0,m6_complete=False)
        (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
