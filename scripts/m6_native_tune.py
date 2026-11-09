"""Tune untrained clipped min-sum controls using the same native kernel as learned BP."""

import argparse
import hashlib
import json
import runpy
import shutil
import signal
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.io import source_manifest
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler

HELPERS=runpy.run_path("scripts/m6_cascade_tune.py")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default="configs/m6/cpu_native_controls.yaml")
    parser.add_argument("--index",required=True,type=int)
    args=parser.parse_args()
    settings=yaml.safe_load(Path(args.settings).read_text())
    root=Path(settings["validation_bundles"][args.index])
    c=load_config(root/"config.yaml")
    selection=json.loads((root/"selection.json").read_text())
    if hashlib.sha256((root/"cases.npz").read_bytes()).hexdigest()!=selection["cases_sha256"]:raise ValueError("Changed validation cases")
    d=Path("results/m6_native_tuning")/f"p{args.index}"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    c.save(d/"config.yaml")
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_source.py")
    torch.set_num_threads(settings["threads"]);set_num_threads(settings["threads"])
    sampler=make_sampler(c,selection["p"])
    with np.load(root/"cases.npz") as saved:s,o=saved["syndromes"],saved["observables"]
    with np.load(root/"first_pass.npz") as saved:first={key:saved[key] for key in saved.files}
    mask=first["failed"]
    print(f"Native min-sum controls: {d}",flush=True)
    started=time.perf_counter()
    def deadline(signum,frame):raise TimeoutError("Native-control tuning reached its CPU wall deadline")
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds"])
    records=[]
    try:
        for factor in c.sweep.factors:
            for kind,iterations in (("native30",settings["longer_iterations"]),("native_second",settings["second_iterations"])):
                model=NeuralMinSum(sampler.problem,iterations,settings["sharing"],factor,settings["message_clip"],dtype=torch.float32)
                decoder=model.export_cpu()
                if kind=="native30":values=HELPERS["decode"](decoder,s,o,sampler.problem,c.eval.batch)
                else:
                    second=HELPERS["decode"](decoder,s[mask],o[mask],sampler.problem,c.eval.batch)
                    values=HELPERS["route"](first,second,mask)
                np.testing.assert_array_equal(~values["failed"],(parity(sampler.problem.H,values["corrections"])==s).all(1))
                np.savez_compressed(d/f"{kind}-{factor:g}.npz",**values)
                records.append(dict(kind=kind,scaling=factor,failures=int(values["failed"].sum()),
                    logical_failures=int(values["logical_failed"].sum()),mean_iterations=float(values["iterations"].mean())))
            print(f"p={selection['p']:g} factor={factor:g}: native30={records[-2]['failures']}/{len(s)}; native second={records[-1]['failures']}/{len(s)}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
        chosen={kind:min([row for row in records if row["kind"]==kind],key=lambda row:(row["failures"],row["scaling"])) for kind in ("native30","native_second")}
        for kind,row in chosen.items():
            count=settings["longer_iterations"] if kind=="native30" else settings["second_iterations"]
            model=NeuralMinSum(sampler.problem,count,settings["sharing"],row["scaling"],settings["message_clip"],dtype=torch.float32)
            sample=s if kind=="native30" else s[mask]
            n=min(settings["export_check_shots"],len(sample))
            expected,conv=model.decode_batch(sample[:n]);actual,actual_conv=model.export_cpu().decode_batch(sample[:n])
            np.testing.assert_array_equal(expected,actual);np.testing.assert_array_equal(conv,actual_conv)
        report=dict(purpose=settings["purpose"],validation_source=str(root),cases_sha256=selection["cases_sha256"],
            p=selection["p"],seed=selection["seed"],shots=len(s),first_factor=selection["first_factor"],selected=chosen,grid=records,
            elapsed_s=time.perf_counter()-started,optimizer_updates=0,confirmation_data_used=False,final_test_data_used=False,m6_complete=False)
        (d/"selection.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(chosen,indent=2))
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
