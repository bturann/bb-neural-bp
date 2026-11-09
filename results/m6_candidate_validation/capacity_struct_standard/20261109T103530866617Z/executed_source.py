"""Validate a frozen learned candidate family without sampling confirmation cases."""

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

from bbnbp.checkpoints import load_model,read_checkpoint
from bbnbp.config import load_config
from bbnbp.io import source_manifest
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler

TUNE=runpy.run_path("scripts/m6_cascade_tune.py")
EXTEND=runpy.run_path("scripts/m6_neural_extensions.py")["continued_model"]


def ready_candidate(spec):
    path=Path(spec["checkpoint"])
    report=json.loads((path.parent/"summary.json").read_text())
    if not report[spec["complete_key"]]:raise ValueError("Candidate training did not complete")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=report[spec["hash_key"]]:raise ValueError("Candidate checkpoint hash changed")
    payload=read_checkpoint(path)
    expected_step=report[spec["expected_step_key"]] if spec.get("expected_step_key") else spec["expected_step"]
    if payload["step"]!=expected_step or payload["model_config"]["T"]!=15:
        raise ValueError("Unexpected candidate training budget or unroll")
    return path,payload,digest


def exact_export(model,s,n):
    n=min(n,len(s))
    expected,conv=model.decode_batch(s[:n])
    decoder=model.export_cpu()
    actual,actual_conv=decoder.decode_batch(s[:n])
    np.testing.assert_array_equal(expected,actual)
    np.testing.assert_array_equal(conv,actual_conv)
    np.testing.assert_array_equal(model.last_iterations,decoder.last_iterations)
    return decoder,n


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default="configs/m6/cpu_cascade_candidates.yaml")
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--output-root",default="results/m6_candidate_validation")
    args=parser.parse_args()
    settings=yaml.safe_load(Path(args.settings).read_text())
    path,payload,digest=ready_candidate(settings["candidates"][args.candidate])
    d=Path(args.output_root)/args.candidate/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_source.py")
    shutil.copyfile(path,d/"source-checkpoint.pt")
    torch.set_num_threads(settings["threads"])
    set_num_threads(settings["threads"])
    print(f"Candidate validation: {d}",flush=True)
    def deadline(signum,frame):raise TimeoutError("Neural candidate validation reached its CPU wall deadline")
    signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds"])
    started=time.perf_counter()
    records=[]
    try:
        for index,root in enumerate(map(Path,settings["tuning_bundles"])):
            c=load_config(root/"config.yaml")
            metadata=json.loads((root/"selection.json").read_text())
            if hashlib.sha256((root/"cases.npz").read_bytes()).hexdigest()!=metadata["cases_sha256"]:
                raise ValueError("Validation cases changed after baseline tuning")
            c.save(d/f"validation-config-p{index}.yaml")
            sampler=make_sampler(c,metadata["p"])
            with np.load(root/"cases.npz") as saved:s,o=(saved[key] for key in ("syndromes","observables"))
            with np.load(root/"first_pass.npz") as saved:first={key:saved[key] for key in saved.files}
            mask=first["failed"]
            for design in settings["designs"]:
                if design=="cascade15":
                    model=load_model(path,sampler.problem)
                    decoder,n=exact_export(model,s[mask],settings["export_check_shots"])
                    second=TUNE["decode"](decoder,s[mask],o[mask],sampler.problem,c.eval.batch)
                    values=TUNE["route"](first,second,mask)
                else:
                    mode={"cycle30":"cycle","last30":"last"}[design]
                    model=EXTEND(path,sampler.problem,30,mode)
                    decoder,n=exact_export(model,s,settings["export_check_shots"])
                    values=TUNE["decode"](decoder,s,o,sampler.problem,c.eval.batch)
                np.testing.assert_array_equal(~values["failed"],(parity(sampler.problem.H,values["corrections"])==s).all(1))
                np.testing.assert_array_equal(values["logical_failed"],values["failed"] | (parity(sampler.problem.L,values["corrections"])!=o).any(1))
                np.savez_compressed(d/f"{design}-p{index}.npz",**values)
                record=dict(candidate=args.candidate,design=design,p=metadata["p"],seed=metadata["seed"],shots=len(s),
                    failures=int(values["failed"].sum()),logical_failures=int(values["logical_failed"].sum()),
                    wrong_logical_converged=int((~values["failed"] & values["logical_failed"]).sum()),
                    mean_iterations=float(values["iterations"].mean()),exact_export_shots=n)
                records.append(record)
                print(f"{args.candidate} {design} p={metadata['p']:g}: fallback={record['failures']/len(s):.6f}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
        report=dict(purpose=settings["purpose"],candidate=args.candidate,source_checkpoint=str(path),source_sha256=digest,
            source_step=payload["step"],records=records,elapsed_s=time.perf_counter()-started,
            confirmation_data_used=False,final_test_data_used=False,m6_complete=False,
            scope="Reuses independent validation only. Continued models perform zero additional optimization; source trained unroll is T=15. "
                  "All inference choices must be frozen before reserved confirmation is sampled.")
        (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
