"""Ten-thousand-case exact Torch/Numba acceptance for frozen cycle100 inference."""

import argparse
import hashlib
import json
import runpy
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.io import source_manifest
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("--settings",default="configs/m6/cpu_cycle100_export_acceptance.yaml")
    args=parser.parse_args();study=args.study
    summary=json.loads((study/"summary.json").read_text())
    frozen=json.loads((study/"selection.json").read_text())
    settings=yaml.safe_load(Path(args.settings).read_text())
    assert hashlib.sha256((study/"selected-source.pt").read_bytes()).hexdigest()==summary["checkpoint_sha256"]
    d=Path("results/m6_cycle100_export_acceptance")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    (d/"executed_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    c=load_config(study/"config.yaml");c.save(d/"config.yaml")
    module=runpy.run_path(str(study/"executed_source.py"))
    torch.set_num_threads(1);set_num_threads(1);started=time.perf_counter()
    def deadline(signum,frame):raise TimeoutError("Cycle100 exact acceptance reached CPU budget")
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,settings["wall_seconds"])
    try:
        points=[]
        for index,point in enumerate(frozen["points"]):
            with np.load(study/f"cases-p{index}.npz") as saved:fresh=saved["syndromes"]
            with np.load(Path(point["validation_source"])/"cases.npz") as saved:validation=saved["syndromes"]
            needed=settings["cases_per_point"]-len(fresh)
            assert 0<=needed<=len(validation)
            s=np.concatenate([fresh,validation[:needed]])
            np.savez_compressed(d/f"tested-syndromes-p{index}.npz",syndromes=s)
            sampler=make_sampler(c,point["p"])
            model=module["neural"](study,sampler.problem,"selected");node=model.export_cpu()
            checked=0
            for start in range(0,len(s),settings["batch"]):
                inputs=s[start:start+settings["batch"]]
                expected,conv=model.decode_batch(inputs);actual,actual_conv=node.decode_batch(inputs)
                np.testing.assert_array_equal(expected,actual);np.testing.assert_array_equal(conv,actual_conv)
                np.testing.assert_array_equal(model.last_iterations,node.last_iterations)
                checked+=len(inputs)
                if checked%1024==0:print(f"cycle100 exact export p={point['p']:g}: {checked}/{len(s)}",flush=True)
            points.append(dict(p=point["p"],cases=checked,syndrome_sha256=hashlib.sha256(s.tobytes()).hexdigest()))
        report=dict(checks_passed=True,study=str(study),points=points,total_exact_cases=sum(p["cases"] for p in points),
            checkpoint_sha256=summary["checkpoint_sha256"],elapsed_s=time.perf_counter()-started,scope=settings["scope"],m6_complete=False)
        (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    except BaseException as exc:
        (d/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n");raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
