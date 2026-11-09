"""Check cached functional composition against fresh actual pipelines on known cases."""

import argparse
import hashlib
import json
import runpy
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBPOSD
from bbnbp.decoders.pipeline import TwoStageDecoder
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("--settings",required=True)
    args=parser.parse_args()
    json.loads((args.study/"summary.json").read_text())
    settings=yaml.safe_load(Path(args.settings).read_text())
    settings.update(accuracy_targets=[8,8],accuracy_wall_seconds=120,accuracy_point_wall_seconds=[60,60])
    d=Path("results/m6_accuracy_preflight")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    (d/"settings.yaml").write_text(yaml.safe_dump(settings))
    (d/"executed_preflight_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"functional_driver_source.py").write_bytes(Path("scripts/m6_strong_accuracy.py").read_bytes())
    helper=runpy.run_path("scripts/m6_strong_accuracy.py")
    c=load_config(args.study/"config.yaml");torch.set_num_threads(1);set_num_threads(1)
    checked=0;started=time.perf_counter()
    for index in range(2):
        point=helper["assess_point"](args.study,d,index,settings,started)
        assert point["complete"]
        sampler=make_sampler(c,c.noise.p_grid[index])
        with np.load(args.study/f"cases-p{index}.npz") as saved:s,o=saved["syndromes"][:8],saved["observables"][:8]
        nodes=helper["make_nodes"](args.study,sampler.problem,index)
        spec=c.eval.osd
        with np.load(d/f"pipeline-flags-p{index}.npz") as saved:flags={k:saved[k] for k in saved.files}
        for name in settings["accuracy_models"]:
            decoder=TwoStageDecoder(nodes[name],LdpcBPOSD(sampler.problem,spec.bp_iters,spec.ms_scaling,spec.method,spec.order))
            actual=decoder.decode_batch(s)
            np.testing.assert_array_equal(parity(sampler.problem.H,actual["e_hat"]),s)
            np.testing.assert_array_equal((parity(sampler.problem.L,actual["e_hat"])!=o).any(1),flags[name])
            checked+=len(s)
    report=dict(checks_passed=True,study=str(args.study),study_kind=settings["study_kind"],
        actual_pipeline_method_cases=checked,elapsed_s=time.perf_counter()-started,
        scope="Integration preflight on already saved cases, not a new independent research result or timing comparison. Fresh uncached actual pipeline outputs match shared functional composition. No model or control selection.",
        functional_driver_sha256=hashlib.sha256((d/"functional_driver_source.py").read_bytes()).hexdigest())
    (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
