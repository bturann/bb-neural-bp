"""Audit randomized timing inputs, pairing, accounting, statistics and sampled outputs."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch
import yaml
from numba import set_num_threads

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP, LdpcBPOSD
from bbnbp.metrics import latency_stats
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();d=args.directory
    if (d/"audit.json").exists():raise ValueError("Audit already exists")
    r=json.loads((d/"summary.json").read_text())
    settings=yaml.safe_load((d/"settings.yaml").read_text())
    machine=json.loads((d/"machine.json").read_text())
    confirmation=Path(machine["confirmation"])
    primary=json.loads((confirmation/"summary.json").read_text())
    assert hashlib.sha256((confirmation/"selected-source.pt").read_bytes()).hexdigest()==primary["checkpoint_sha256"]
    manifest=json.loads((d/"source_manifest.json").read_text())
    for path,digest in json.loads((d/"frozen_input_hashes.json").read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    assert r["frozen_inputs_unchanged"]
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_aux_latency.py"]
    assert machine["threads"]==1 and machine["known_training_and_accuracy_jobs_finished"]
    helper=runpy.run_path(str(d/"executed_source.py"))
    builder_module=runpy.run_path(str(confirmation/"models_source.py"))
    builder=builder_module["make_decoders"]
    c=load_config(confirmation/"config.yaml");names=settings["latency_models"]
    torch.set_num_threads(1);set_num_threads(1)
    checked,sampled,actual_calls=0,0,0
    for index,point in enumerate(r["points"]):
        sampler=make_sampler(c,point["p"])
        seed=c.seed+settings["latency_seed_offset"]+index
        s,o,e=sampler.sample(settings["latency_warmup"]+settings["latency_shots"],np.random.default_rng(seed))
        order_rng=np.random.default_rng(np.random.SeedSequence(settings["latency_order_entropy"]+[index]))
        order=np.stack([order_rng.permutation(len(names)) for _ in range(len(s))])
        with np.load(d/f"cases-p{index}.npz") as saved:
            for key,array in (("syndromes",s),("observables",o),("faults",e),("order",order)):
                np.testing.assert_array_equal(saved[key],array)
        np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
        np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
        raw=[json.loads(line) for line in (d/f"raw-p{index}.jsonl").read_text().splitlines()]
        assert len(raw)<=len(s)*len(names)
        for k,row in enumerate(raw):
            case,position=divmod(k,len(names))
            assert row["case"]==case and row["order_position"]==position
            assert row["method"]==names[order[case,position]]
            assert row["warmup"]==(case<settings["latency_warmup"])
            assert row["fallback_called"]==row["stage1_failed"]
            assert not row["internal_osd_executed"] or row["fallback_called"]
            assert row["syndrome_valid"]
            for key in ("total_us","stage1_us","osd_us","overhead_us"):
                assert np.isfinite(row[key]) and row[key]>=0
            np.testing.assert_allclose(row["total_us"],row["stage1_us"]+row["osd_us"]+row["overhead_us"],rtol=1e-14,atol=1e-8)
            if not row["fallback_called"]:assert row["osd_us"]==0
        complete=len(raw)//len(names)
        assert point["complete_cases_including_warmup"]==complete
        measured=max(0,complete-settings["latency_warmup"])
        assert measured==point["measured_paired_cases"] and point["completed"]==(measured==settings["latency_shots"])
        calls=sum(row["fallback_called"] for row in raw)
        assert calls==point["all_actual_fallback_calls_including_warmup_and_partial"]
        actual_calls+=calls
        arrays={}
        for summary in point["models"]:
            name=summary["method"]
            kept=[row for row in raw if row["method"]==name and settings["latency_warmup"]<=row["case"]<complete]
            values=np.array([row["total_us"] for row in kept]);arrays[name]=values
            assert len(values)==measured==summary["shots"]
            for key,value in latency_stats(values).items():assert summary[key]==value
            for key,source in (("mean_stage1_us","stage1_us"),("mean_fallback_us","osd_us")):
                assert summary[key]==float(np.mean([row[source] for row in kept]))
            for key,source in (("fallback_calls","fallback_called"),("internal_osd_executions","internal_osd_executed"),("logical_failures","logical")):
                assert summary[key]==sum(row[source] for row in kept)
        if arrays:
            for k,name in enumerate(names):
                if name=="selected" or name not in arrays:continue
                actual=helper["paired_bootstrap"](arrays[name],arrays["selected"],np.random.default_rng(
                    np.random.SeedSequence(settings["latency_bootstrap_entropy"]+[index,k])),settings["latency_bootstrap_repeats"])
                assert point["paired_nominal_bootstrap"][name]==actual
            for k,(baseline,candidate) in enumerate(settings.get("latency_additional_contrasts",[])):
                if baseline not in arrays or candidate not in arrays:continue
                actual=helper["paired_bootstrap"](arrays[baseline],arrays[candidate],np.random.default_rng(
                    np.random.SeedSequence(settings["latency_bootstrap_entropy"]+[index,len(names)+k])),settings["latency_bootstrap_repeats"])
                assert point["additional_paired_contrasts"][f"{candidate}_vs_{baseline}"]==dict(baseline=baseline,candidate=candidate,statistics=actual)
        controls=json.loads((confirmation/f"controls-p{index}.json").read_text())
        nodes=builder(sampler.problem,confirmation/"selected-source.pt",primary["model_design"],controls)
        if machine["long_controls"]:
            choices=json.loads((Path(machine["long_controls"][index])/"selection.json").read_text())["choices"]
            for T,choice in choices.items():nodes[f"bp{T}"]=LdpcBP(sampler.problem,int(T),choice["factor"])
        if machine.get("strong_study"):
            nodes=helper["strong_decoders"](sampler.problem,confirmation/"selected-source.pt",builder_module,Path(machine["strong_study"]),index)
        if machine.get("cycle100_study"):
            nodes.update(helper["extended_decoders"](sampler.problem,Path(machine["cycle100_study"]),index))
        spec=c.eval.osd
        fallback=LdpcBPOSD(sampler.problem,spec.bp_iters,spec.ms_scaling,spec.method,spec.order)
        start=settings["latency_warmup"];end=min(complete,start+32)
        if end>start:
            syndromes=s[start:end]
            hats={};failed={}
            for name in names:
                hats[name],conv=nodes[name].decode_batch(syndromes)
                failed[name]=~conv
                np.testing.assert_array_equal(conv,(parity(sampler.problem.H,hats[name])==syndromes).all(1))
            for j,case in enumerate(range(start,end)):
                fb=fallback.decode_one(s[case]) if any(failed[name][j] for name in names) else None
                for name in names:
                    row=next(row for row in raw if row["case"]==case and row["method"]==name)
                    used=bool(failed[name][j]);hat=fb if used else hats[name][j]
                    assert row["fallback_called"]==used
                    assert row["logical"]==bool((parity(sampler.problem.L,hat[None,:])[0]!=o[case]).any())
                    assert row["stage1_logical"]==bool(used or (parity(sampler.problem.L,hats[name][j:j+1])[0]!=o[case]).any())
                    sampled+=1
        checked+=len(raw)
        print(f"Audited timing p={point['p']:g}: {measured} paired measured cases, {len(raw)} raw method calls",flush=True)
    assert r["completed"]==all(point["completed"] for point in r["points"])
    audit=dict(checks_passed=True,raw_method_calls_verified=checked,actual_uncached_fallback_calls_accounted=actual_calls,
        functional_pipeline_outputs_replayed=sampled,randomized_seeded_case_orders_reproduced=True,
        warmup_and_partial_pair_exclusion_verified=True,statistics_and_paired_bootstrap_reproduced=True,
        scope="Verifies saved timing/accounting and sampled functional outputs; uncontrolled external system load and pilot p99 uncertainty remain limitations.",
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
