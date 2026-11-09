"""Freeze one validation-selected learned pipeline and its ordinary controls."""

import argparse
import hashlib
import json
import runpy
import shutil
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.io import source_manifest
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default="configs/m6/cpu_cascade_candidates_v2.yaml")
    parser.add_argument("--output-root",default="results/m6_aux_selection")
    args=parser.parse_args()
    settings=yaml.safe_load(Path(args.settings).read_text())
    ready=runpy.run_path("scripts/m6_validate_candidates.py")["ready_candidate"]
    options=[]
    for name,spec in settings["candidates"].items():
        _,_,digest=ready(spec)
        paths=list((Path("results/m6_candidate_validation")/name).glob("*/summary.json"))
        if len(paths)!=1:raise ValueError(f"Expected exactly one completed validation for {name}")
        report=json.loads(paths[0].read_text())
        assert report["source_sha256"]==digest and not report["confirmation_data_used"]
        for design in settings["designs"]:
            rows=[row for row in report["records"] if row["design"]==design]
            if len(rows)!=len(settings["tuning_bundles"]):raise ValueError("Incomplete candidate validation grid")
            options.append(dict(candidate=name,design=design,failures=sum(row["failures"] for row in rows),
                logical_failures=sum(row["logical_failures"] for row in rows),source_sha256=digest,
                source_checkpoint=spec["checkpoint"],source_step=report["source_step"],validation_directory=str(paths[0].parent),points=rows))
    selected=min(options,key=lambda row:(row["failures"],row["candidate"],row["design"]))
    d=Path(args.output_root)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    c=load_config(Path(settings["tuning_bundles"][0])/"config.yaml")
    c.save(d/"config.yaml")
    (d/"candidate_settings.yaml").write_text(yaml.safe_dump(settings))
    for source,target in (("scripts/m6_aux_models.py","models_source.py"),("scripts/m6_neural_extensions.py","neural_extensions_source.py"),(__file__,"executed_source.py")):
        shutil.copyfile(source,d/target)
    shutil.copyfile(Path(selected["validation_directory"])/"source-checkpoint.pt",d/"selected-source.pt")
    assert hashlib.sha256((d/"selected-source.pt").read_bytes()).hexdigest()==selected["source_sha256"]
    torch.set_num_threads(1);set_num_threads(1)
    builder=runpy.run_path(str(d/"models_source.py"))["make_decoders"]
    controls=[]
    native_roots=["results/m6_native_tuning/p0/20261109T094715558247Z","results/m6_native_tuning/p1/20261109T094715558379Z"]
    prefix_checks=0
    for i,(cpp_root,native_root) in enumerate(zip(map(Path,settings["tuning_bundles"]),map(Path,native_roots))):
        cpp=json.loads((cpp_root/"selection.json").read_text())
        native=json.loads((native_root/"selection.json").read_text())
        choices=[dict(name="bp30",**cpp["selected"]["longer"]),dict(name="two_bp",**cpp["selected"]["two_pass"]),
            dict(name="native30",**native["selected"]["native30"]),dict(name="native_second",**native["selected"]["native_second"])]
        ordinary=min(choices,key=lambda row:(row["failures"],row["name"]))
        point=dict(cpp=cpp,native=native,best_ordinary=ordinary,cpp_directory=str(cpp_root),native_directory=str(native_root))
        controls.append(point)
        (d/f"controls-p{i}.json").write_text(json.dumps(point,indent=2)+"\n")
        sampler=make_sampler(c,cpp["p"])
        with np.load(cpp_root/"cases.npz") as saved:s,o=saved["syndromes"][:64],saved["observables"][:64]
        models=builder(sampler.problem,d/"selected-source.pt",selected["design"],point)
        expected_files=dict(selected=Path(selected["validation_directory"])/f"{selected['design']}-p{i}.npz",
            bp30=cpp_root/f"longer-{cpp['selected']['longer']['scaling']:g}.npz",
            two_bp=cpp_root/f"two_pass-{cpp['selected']['two_pass']['scaling']:g}.npz",
            native30=native_root/f"native30-{native['selected']['native30']['scaling']:g}.npz",
            native_second=native_root/f"native_second-{native['selected']['native_second']['scaling']:g}.npz",
            bp15=cpp_root/"first_pass.npz")
        for name,path in expected_files.items():
            correction,conv=models[name].decode_batch(s)
            with np.load(path) as expected:
                np.testing.assert_array_equal(correction,expected["corrections"][:len(s)])
                np.testing.assert_array_equal(~conv,expected["failed"][:len(s)])
                np.testing.assert_array_equal(models[name].last_iterations,expected["iterations"][:len(s)])
                logical=~conv | (parity(sampler.problem.L,correction)!=o).any(1)
                np.testing.assert_array_equal(logical,expected["logical_failed"][:len(s)])
            prefix_checks+=len(s)
    protocol=dict(purpose="frozen_auxiliary_pipeline_confirmation",selected=selected,options=options,
        ordinary_controls=[dict(point["best_ordinary"],p=point["cpp"]["p"]) for point in controls],
        checkpoint_sha256=selected["source_sha256"],model_design=selected["design"],maximum_stage1_iterations=30,
        prefix_builder_checks=prefix_checks,frozen_before_confirmation=True,confirmation_data_used=False,
        confirmation_seeds=[c.seed+c.eval.seed_offset+i for i in range(len(c.noise.p_grid))],confirmation_shots=c.eval.max_shots,
        primary_family="Selected learned versus validation-best ordinary control; two p values, Holm adjustment",
        learning_family="Selected learned versus architecture-matched initialization; two p values, Holm adjustment",
        clear_gain_rule="Mean reduction >= .01 versus ordinary, at least one corrected primary benefit, no corrected primary harm; learning also requires lower mean and a corrected benefit versus initialization.",
        minimum_mean_gain=.01,alpha=.05,final_test_data_used=False,m6_complete=False,
        scope="Auxiliary single-source-model study; validation selected among all listed choices. No main M6 completion or logical-equivalence/latency claim.")
    (d/"selection.json").write_text(json.dumps(protocol,indent=2)+"\n")
    (d/"pipeline_spec.yaml").write_text(yaml.safe_dump(protocol))
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    print(d)
    print(json.dumps({key:protocol[key] for key in ("selected","ordinary_controls","prefix_builder_checks")},indent=2))


if __name__=="__main__":main()
