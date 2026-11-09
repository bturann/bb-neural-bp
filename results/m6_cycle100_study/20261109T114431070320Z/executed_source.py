"""Fresh equal-iteration assessment of fixed cycle100 weights and replications."""

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

from bbnbp.checkpoints import read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.io import source_manifest
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def neural(study,problem,name):
    checkpoint=study/("selected-source.pt" if name in {"selected","matched_init"} else f"{name}-source.pt")
    if name=="matched_init":
        payload=read_checkpoint(checkpoint);kwargs=dict(payload["model_config"],T=100)
        return NeuralMinSum(problem,**kwargs,dtype={"torch.float32":torch.float32,"torch.float64":torch.float64}[payload["dtype"]]).eval()
    helper=runpy.run_path(str(study/"neural_extensions_source.py"))["continued_model"]
    return helper(checkpoint,problem,100,"cycle")


def make_models(study,problem,index):
    frozen=json.loads((study/"selection.json").read_text())
    point=frozen["points"][index]
    result={"cpp100":LdpcBP(problem,100,point["cpp"]["factor"]),
        "native100":NeuralMinSum(problem,100,"scalar",point["native"]["factor"],20.,dtype=torch.float32).export_cpu()}
    for name in ("selected","matched_init","seed1235","seed1236"):
        result[name]=neural(study,problem,name).export_cpu()
    result["ordinary"]=result[point["best_ordinary"]]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default="configs/m6/cpu_cycle100_study.yaml")
    args=parser.parse_args();settings=yaml.safe_load(Path(args.settings).read_text())
    assert settings["iterations"]==100 and settings["extension"]=="cycle"
    source=Path(settings["source_confirmation"])
    assert hashlib.sha256((source/"selected-source.pt").read_bytes()).hexdigest()==settings["source_checkpoint_sha256"]
    if (source/"cycle100_receipt.json").exists():raise ValueError("Cycle100 study already requested")
    for path in map(Path,settings["replication_sources"].values()):
        assert read_checkpoint(path)["step"]==4000
        assert json.loads((path.parent/"summary.json").read_text())["completed"]
    d=Path("results/m6_cycle100_study")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    c=load_config(source/"config.yaml");c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_source.py")
    shutil.copyfile(source/"selected-source.pt",d/"selected-source.pt")
    shutil.copyfile(source/"neural_extensions_source.py",d/"neural_extensions_source.py")
    for name,path in settings["replication_sources"].items():shutil.copyfile(path,d/f"{name}-source.pt")
    (source/"cycle100_receipt.json").write_text(json.dumps(dict(directory=str(d),requested_at_utc=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")
    torch.set_num_threads(1);set_num_threads(1);started=time.perf_counter()
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    def deadline(signum,frame):raise TimeoutError("Cycle100 study reached its CPU budget")
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds"])
    try:
        selections=[]
        for index,root in enumerate(map(Path,settings["long_control_sources"])):
            long=json.loads((root/"selection.json").read_text())
            validation=Path(long["validation_source"])
            sampler=make_sampler(c,c.noise.p_grid[index])
            with np.load(validation/"cases.npz") as saved:s,o=saved["syndromes"],saved["observables"]
            grid=[]
            for factor in c.sweep.factors:
                node=NeuralMinSum(sampler.problem,100,"scalar",factor,20.,dtype=torch.float32).export_cpu()
                value=decode(node,s,o,sampler.problem,c.eval.batch)
                np.savez_compressed(d/f"validation-native-p{index}-factor{factor:g}.npz",**value)
                grid.append(dict(factor=factor,failures=int(value["failed"].sum()),logical_failures=int(value["logical_failed"].sum())))
                print(f"native100 tuning p={c.noise.p_grid[index]:g} factor={factor:g}: {grid[-1]['failures']}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            native=min(grid,key=lambda row:(row["failures"],row["logical_failures"],row["factor"]))
            cpp=long["choices"]["100"]
            best=min((("cpp100",cpp),("native100",native)),key=lambda pair:(pair[1]["failures"],pair[1]["logical_failures"],pair[0]))[0]
            selections.append(dict(p=c.noise.p_grid[index],cpp=cpp,native=native,native_grid=grid,best_ordinary=best,
                validation_source=str(validation),validation_cases_sha256=hashlib.sha256((validation/"cases.npz").read_bytes()).hexdigest()))
        frozen=dict(points=selections,iterations=100,extension="cycle",additional_optimizer_updates=0,
            checkpoints={name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in ("selected-source.pt","seed1235-source.pt","seed1236-source.pt")},
            frozen_at_utc=datetime.now(timezone.utc).isoformat())
        (d/"selection.json").write_text(json.dumps(frozen,indent=2)+"\n")
        (d/"confirmation_receipt.json").write_text(json.dumps(dict(started_at_utc=datetime.now(timezone.utc).isoformat(),seeds=settings["confirmation_seeds"],shots_per_point=settings["shots"]),indent=2)+"\n")
        points=[]
        for index,choice in enumerate(selections):
            if datetime.now(timezone.utc)>=datetime.fromisoformat(settings["deadline_utc"]):raise TimeoutError("Cycle100 assessment deadline reached")
            sampler=make_sampler(c,choice["p"])
            s,o,e=sampler.sample(settings["shots"],np.random.default_rng(settings["confirmation_seeds"][index]))
            np.testing.assert_array_equal(parity(sampler.problem.H,e),s);np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
            np.savez_compressed(d/f"cases-p{index}.npz",syndromes=s,observables=o,faults=e)
            nodes=make_models(d,sampler.problem,index);values={}
            for name,node in nodes.items():
                if name=="ordinary":continue
                if name in {"selected","matched_init","seed1235","seed1236"}:
                    model=neural(d,sampler.problem,name);n=settings["exact_export_cases"]
                    a,conv=model.decode_batch(s[:n]);b,bconv=node.decode_batch(s[:n])
                    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(conv,bconv)
                    np.testing.assert_array_equal(model.last_iterations,node.last_iterations)
                value=decode(node,s,o,sampler.problem,c.eval.batch)
                np.testing.assert_array_equal(value["failed"],~(parity(sampler.problem.H,value["corrections"])==s).all(1))
                np.savez_compressed(d/f"values-p{index}-{name}.npz",**value);values[name]=value
                print(f"cycle100 fresh p={choice['p']:g} {name}: {value['failed'].sum()}/{len(s)}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            ordinary=choice["best_ordinary"]
            comparisons={name:dict(primary=mcnemar(values[ordinary]["failed"],values[name]["failed"]),
                learning=mcnemar(values["matched_init"]["failed"],values[name]["failed"])) for name in ("selected","seed1235","seed1236")}
            points.append(dict(p=choice["p"],seed=settings["confirmation_seeds"][index],shots=len(s),ordinary_name=ordinary,
                failures={name:int(value["failed"].sum()) for name,value in values.items()},
                wrong_logical_converged={name:int((~value["failed"] & value["logical_failed"]).sum()) for name,value in values.items()},comparisons=comparisons))
        holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
        for family in ("primary","learning"):
            for names in (("selected",),("seed1235","seed1236")):
                entries=[point["comparisons"][name][family] for point in points for name in names]
                for entry,value in zip(entries,holm([entry["pvalue"] for entry in entries])):entry["holm_pvalue"]=value
        means={name:float(np.mean([point["failures"][name]/point["shots"] for point in points])) for name in points[0]["failures"]}
        mean_ordinary=float(np.mean([point["failures"][point["ordinary_name"]]/point["shots"] for point in points]))
        benefits=sum(p["comparisons"]["selected"]["primary"]["holm_pvalue"]<settings["alpha"] and p["comparisons"]["selected"]["primary"]["first_only"]>p["comparisons"]["selected"]["primary"]["second_only"] for p in points)
        harms=sum(p["comparisons"]["selected"]["primary"]["holm_pvalue"]<settings["alpha"] and p["comparisons"]["selected"]["primary"]["first_only"]<p["comparisons"]["selected"]["primary"]["second_only"] for p in points)
        learn=sum(p["comparisons"]["selected"]["learning"]["holm_pvalue"]<settings["alpha"] and p["comparisons"]["selected"]["learning"]["first_only"]>p["comparisons"]["selected"]["learning"]["second_only"] for p in points)
        clear=bool(mean_ordinary-means["selected"]>=settings["minimum_mean_gain"] and benefits>0 and harms==0 and means["matched_init"]>means["selected"] and learn>0)
        report=dict(points=points,mean_fallback=means,mean_ordinary=mean_ordinary,clear_gain=clear,
            primary_benefits=benefits,primary_harms=harms,learning_benefits=learn,checkpoint_sha256=settings["source_checkpoint_sha256"],
            elapsed_s=time.perf_counter()-started,scope=settings["scope"],final_test_data_used=False,gpu_seconds=0,m6_complete=False)
        (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
