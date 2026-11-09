"""Audit every saved control outcome and independently replay selected/prefix decodes."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch
import yaml

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.evaluate import tuned_factor
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler

HELPERS=runpy.run_path("scripts/m6_cascade_tune.py")


def read_values(path):
    with np.load(path) as saved:return {key:saved[key] for key in saved.files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp",type=Path,required=True)
    parser.add_argument("--native",type=Path,required=True)
    args=parser.parse_args()
    cpp,native=args.cpp,args.native
    if (cpp/"combined_control_audit.json").exists():parser.error("Refusing to overwrite audit evidence")
    c=load_config(cpp/"config.yaml")
    reports={"cpp":json.loads((cpp/"selection.json").read_text()),"native":json.loads((native/"selection.json").read_text())}
    assert reports["native"]["validation_source"]==str(cpp)
    p,seed=reports["cpp"]["p"],reports["cpp"]["seed"]
    assert seed==c.seed+c.sweep.seed_offset+c.noise.p_grid.index(p)
    table=cpp/"first_pass_scaling.csv"
    original=load_config(table.with_suffix(".config.yaml"))
    runpy.run_path("scripts/m6_compare_stage1.py")["validate_tuning_table"](original,table)
    assert reports["cpp"]["first_factor"]==tuned_factor(table,c,p)
    torch.set_num_threads(1)
    sampler=make_sampler(c,p)
    s,o,e=sampler.sample(c.sweep.shots,np.random.default_rng(seed))
    assert hashlib.sha256((cpp/"cases.npz").read_bytes()).hexdigest()==reports["cpp"]["cases_sha256"]==reports["native"]["cases_sha256"]
    with np.load(cpp/"cases.npz") as saved:
        for key,value in (("syndromes",s),("observables",o),("faults",e)):np.testing.assert_array_equal(value,saved[key])
    np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
    np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
    first=HELPERS["decode"](LdpcBP(sampler.problem,15,reports["cpp"]["first_factor"]),s,o,sampler.problem,c.eval.batch)
    saved_first=read_values(cpp/"first_pass.npz")
    for key,value in first.items():np.testing.assert_array_equal(value,saved_first[key])
    total_flags,replayed,grid_count=0,len(s),0
    for backend,root in (("cpp",cpp),("native",native)):
        r=reports[backend]
        settings=yaml.safe_load((root/"settings.yaml").read_text())
        entry="scripts/m6_cascade_tune.py" if backend=="cpp" else "scripts/m6_native_tune.py"
        manifest=json.loads((root/"source_manifest.json").read_text())
        assert hashlib.sha256((root/"executed_source.py").read_bytes()).hexdigest()==manifest[entry]
        kinds=("longer","two_pass") if backend=="cpp" else ("native30","native_second")
        assert len(r["grid"])==2*len(c.sweep.factors)
        for kind in kinds:
            rows=[row for row in r["grid"] if row["kind"]==kind]
            assert {row["scaling"] for row in rows}==set(c.sweep.factors)
            assert min(rows,key=lambda row:(row["failures"],row["scaling"]))==r["selected"][kind]
        for row in r["grid"]:
            kind,factor=row["kind"],row["scaling"]
            values=read_values(root/f"{kind}-{factor:g}.npz")
            failed=(parity(sampler.problem.H,values["corrections"])!=s).any(1)
            logical=failed | (parity(sampler.problem.L,values["corrections"])!=o).any(1)
            np.testing.assert_array_equal(values["failed"],failed)
            np.testing.assert_array_equal(values["logical_failed"],logical)
            assert int(failed.sum())==row["failures"] and int(logical.sum())==row["logical_failures"]
            assert float(values["iterations"].mean())==row["mean_iterations"]
            assert (values["iterations"]<=30).all()
            selected=row==r["selected"][kind]
            n=len(s) if selected else min(64,len(s))
            single=kind in {"longer","native30"}
            iterations=30 if single else 15
            if backend=="cpp":decoder=LdpcBP(sampler.problem,iterations,factor)
            else:decoder=NeuralMinSum(sampler.problem,iterations,settings["sharing"],factor,settings["message_clip"],dtype=torch.float32).export_cpu()
            if single:actual=HELPERS["decode"](decoder,s[:n],o[:n],sampler.problem,c.eval.batch)
            else:
                initial={key:value[:n] for key,value in first.items()}
                mask=initial["failed"]
                second=HELPERS["decode"](decoder,s[:n][mask],o[:n][mask],sampler.problem,c.eval.batch)
                actual=HELPERS["route"](initial,second,mask)
            for key,value in actual.items():np.testing.assert_array_equal(value,values[key][:n])
            replayed+=n;total_flags+=len(s);grid_count+=1
    audit=dict(checks_passed=True,p=p,seed=seed,regenerated_cases=len(s),checked_control_rows=grid_count,
        checked_full_decoder_outcomes=total_flags,replayed_model_cases=replayed,first_pass_replayed=True,
        all_selections_reproduced=True,confirmation_data_used=False,final_test_data_used=False,m6_complete=False,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (cpp/"combined_control_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (cpp/"combined_control_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
