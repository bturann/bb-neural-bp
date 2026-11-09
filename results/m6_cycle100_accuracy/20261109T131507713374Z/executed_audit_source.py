"""Reproduce shared-fallback functional accuracy from seeded inputs and raw outputs."""

import argparse
import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBPOSD
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();d=args.directory
    if (d/"audit.json").exists():raise ValueError("Audit already exists")
    r=json.loads((d/"summary.json").read_text())
    settings=yaml.safe_load((d/"settings.yaml").read_text())
    confirmation=Path(r["confirmation_directory"])
    primary=json.loads((confirmation/"summary.json").read_text())
    strong="primary_checkpoint_sha256" in primary
    cycle100=isinstance(primary.get("mean_fallback"),dict)
    supplementary=strong or cycle100
    assert hashlib.sha256((confirmation/"selected-source.pt").read_bytes()).hexdigest()==primary["primary_checkpoint_sha256" if strong else "checkpoint_sha256"]
    manifest=json.loads((d/"source_manifest.json").read_text())
    driver="scripts/m6_strong_accuracy.py" if supplementary else "scripts/m6_aux_accuracy.py"
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest[driver]
    for path,digest in manifest.items():
        if path.startswith("src/"):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    helper=runpy.run_path(str(d/"executed_source.py"))
    if supplementary:
        for path in ("scripts/m6_aux_accuracy.py","scripts/m6_aux_latency.py"):
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest[path]
    builder=None if supplementary else runpy.run_path(str(confirmation/"models_source.py"))["make_decoders"]
    c=load_config(d/"config.yaml");names=settings["accuracy_models"]
    torch.set_num_threads(1);set_num_threads(1)
    checked,stage1_replays,fallback_replays,actual_count=0,0,0,0
    rows=[]
    for index,point in enumerate(r["points"]):
        sampler=make_sampler(c,point["p"])
        spec=c.eval.osd
        fallback=LdpcBPOSD(sampler.problem,spec.bp_iters,spec.ms_scaling,spec.method,spec.order)
        if supplementary:
            choice=json.loads((confirmation/"selection.json").read_text())["points"][index]
            actual_names=(dict(ordinary=choice["best_ordinary"],matched_init="matched_init",selected="selected") if cycle100 else
                          dict(ordinary=choice["best_ordinary"]["name"],matched_init="strong_init",selected="strong_cascade"))
            nodes=helper["make_nodes"](confirmation,sampler.problem,index)
        else:
            controls=json.loads((confirmation/f"controls-p{index}.json").read_text())
            nodes=builder(sampler.problem,confirmation/"selected-source.pt",primary["model_design"],controls)
            actual_names={name:controls["best_ordinary"]["name"] if name=="ordinary" else name for name in names}
        with np.load(confirmation/f"cases-p{index}.npz") as saved:initial={k:saved[k] for k in ("syndromes","observables","faults")}
        cached={}
        for name in names:
            actual=actual_names[name]
            with np.load(confirmation/f"values-p{index}-{actual}.npz") as saved:
                cached[name]={k:saved[k] for k in ("failed","logical_failed")}
        rng=np.random.default_rng(settings["accuracy_extension_seeds"][index])
        flags=[];requests=np.zeros(len(names),dtype=np.int64)
        processed,executions,internal,replayed_here=0,0,0,0
        for batch in point["batches"]:
            assert batch["offset"]==processed
            with np.load(batch["input"]) as saved:values={k:saved[k] for k in saved.files}
            s,o,e=(values[k] for k in ("syndromes","observables","faults"))
            used=batch["used"]
            if processed<len(initial["syndromes"]):
                assert batch["origin"]=="independent_confirmation_prefix"
                for key,array in (("syndromes",s),("observables",o),("faults",e)):
                    np.testing.assert_array_equal(array,initial[key][processed:processed+len(s)])
                np.testing.assert_array_equal(values["stage1_failed"],np.stack([cached[name]["failed"][processed:processed+len(s)] for name in names]))
                np.testing.assert_array_equal(values["stage1_logical"],np.stack([cached[name]["logical_failed"][processed:processed+len(s)] for name in names]))
            else:
                assert batch["origin"]=="new_extension_stream"
                generated=sampler.sample(len(s),rng)
                for a,b in zip((s,o,e),generated):np.testing.assert_array_equal(a,b)
                for k,name in enumerate(names):
                    correction,conv=nodes[name].decode_batch(s)
                    actual_conv=(parity(sampler.problem.H,correction)==s).all(1)
                    logical=~actual_conv | (parity(sampler.problem.L,correction)!=o).any(1)
                    np.testing.assert_array_equal(conv,actual_conv)
                    np.testing.assert_array_equal(~conv,values["stage1_failed"][k])
                    np.testing.assert_array_equal(logical,values["stage1_logical"][k])
                    stage1_replays+=len(s)
            np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
            np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
            failed=values["stage1_failed"][:,:used]
            with np.load(batch["fallback"]) as saved:fb={k:saved[k] for k in saved.files}
            indices=np.asarray(fb["source_indices"],dtype=np.int64)
            np.testing.assert_array_equal(indices,np.flatnonzero(failed.any(0)))
            assert int(fb["used_count"])==used and len(fb["corrections"])==len(indices)
            logical=np.zeros(used,dtype=bool)
            if len(indices):
                np.testing.assert_array_equal(parity(sampler.problem.H,fb["corrections"]),s[indices])
                logical[indices]=(parity(sampler.problem.L,fb["corrections"])!=o[indices]).any(1)
            np.testing.assert_array_equal(logical,fb["fallback_logical"])
            composed=helper["compose_logicals"](failed,values["stage1_logical"][:,:used],logical)
            np.testing.assert_array_equal(composed,fb["pipeline_logical"])
            for position in range(min(8-replayed_here,len(indices))):
                actual=fallback.decode_one(s[indices[position]])
                np.testing.assert_array_equal(actual,fb["corrections"][position])
                assert fallback.last_osd_executed==fb["internal_osd"][position]
                fallback_replays+=1;replayed_here+=1
            flags.append(composed);requests+=failed.sum(1)
            processed+=used;executions+=len(indices);internal+=int(fb["internal_osd"].sum())
            checked+=used
        flags=np.concatenate(flags,axis=1)
        with np.load(d/f"pipeline-flags-p{index}.npz") as saved:
            for k,name in enumerate(names):np.testing.assert_array_equal(saved[name],flags[k])
        assert processed==point["shots"] and point["complete"]==(processed==point["target"])
        assert executions==point["actual_unique_fallback_executions"]
        assert internal==point["actual_internal_osd_executions"]
        assert int(requests.sum())==point["sum_model_fallback_requests"]
        for k,row in enumerate(point["models"]):
            failures=int(flags[k].sum());lo,hi=wilson(failures,processed)
            assert row==dict(decoder=names[k],p=point["p"],shots=processed,logical_failures=failures,
                logical_rate=failures/processed,nominal_wilson_lo=lo,nominal_wilson_hi=hi,
                stage1_fallback_requests=int(requests[k]))
            rows.append(row)
        selected=flags[names.index("selected")]
        for name,pair in point["pairs"].items():
            baseline=flags[names.index(name)]
            regressions=int((selected & ~baseline).sum())
            assert pair==dict(selected_only_failures=regressions,baseline_only_failures=int((~selected & baseline).sum()),
                extra_failure_probability_upper=helper["regression_upper"](regressions,processed,settings["alpha"]/4) if point["complete"] else None,
                mcnemar=mcnemar(baseline,selected))
        actual_count+=executions
        print(f"Audited full accuracy p={point['p']:g}: {processed} cases and {executions} shared fallback outputs",flush=True)
    assert actual_count==r["actual_unique_fallback_executions"]
    assert r["completed"]==all(p["complete"] for p in r["points"])
    with (d/"paired_accuracy.csv").open(newline="") as stream:csv_rows=list(csv.DictReader(stream))
    assert csv_rows==[{key:str(value) for key,value in row.items()} for row in rows]
    audit=dict(checks_passed=True,paired_cases_verified=checked,extension_stage1_cases_redecoded=stage1_replays,
        shared_fallback_outputs_verified=actual_count,uncached_fallback_replays=fallback_replays,
        custom_csv_rows_verified=len(rows),paired_statistics_and_joint_bounds_reproduced=True,
        scope="Functional correctness and accounting only; cached accuracy execution times are not latency evidence.",
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
