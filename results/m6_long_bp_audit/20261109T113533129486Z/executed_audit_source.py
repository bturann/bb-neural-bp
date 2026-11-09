"""Audit both secondary BP iteration-budget challenges and their joint statistics."""

import argparse
import hashlib
import json
import runpy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.metrics import mcnemar
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",nargs=2,type=Path)
    args=parser.parse_args()
    for root in args.directories:
        if (root/"audit_receipt.json").exists():raise ValueError("Challenge already audited")
        if not (root/"summary.json").exists():raise ValueError("Challenge must complete before audit")
    d=Path("results/m6_long_bp_audit")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    decode=runpy.run_path("scripts/m6_cascade_tune.py")["decode"]
    pairs=[];verified=0;replayed=0
    for index,root in enumerate(args.directories):
        r=json.loads((root/"summary.json").read_text())
        frozen=json.loads((root/"selection.json").read_text())
        source=Path(frozen["validation_source"])
        manifest=json.loads((root/"source_manifest.json").read_text())
        assert hashlib.sha256((root/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_long_bp_challenge.py"]
        assert hashlib.sha256((source/"cases.npz").read_bytes()).hexdigest()==frozen["validation_cases_sha256"]
        c=load_config(root/"config.yaml");sampler=make_sampler(c,r["p"])
        with np.load(source/"cases.npz") as saved:s,o=saved["syndromes"],saved["observables"]
        for row in frozen["grid"]:
            with np.load(root/f"validation-T{row['T']}-factor{row['factor']:g}.npz") as saved:
                failed=~(parity(sampler.problem.H,saved["corrections"])==s).all(1)
                logical=failed | (parity(sampler.problem.L,saved["corrections"])!=o).any(1)
                np.testing.assert_array_equal(saved["failed"],failed);np.testing.assert_array_equal(saved["logical_failed"],logical)
                assert row["failures"]==int(failed.sum()) and row["logical_failures"]==int(logical.sum())
                assert row["mean_iterations"]==float(saved["iterations"].mean())
                value=decode(LdpcBP(sampler.problem,row["T"],row["factor"]),s[:16],o[:16],sampler.problem,16)
                for key,array in value.items():np.testing.assert_array_equal(array,saved[key][:16])
            verified+=len(s);replayed+=16
        for T,choice in frozen["choices"].items():
            assert choice==min([row for row in frozen["grid"] if row["T"]==int(T)],
                key=lambda row:(row["failures"],row["logical_failures"],row["factor"]))
        confirmation=Path(r["confirmation_directory"])
        with np.load(confirmation/f"cases-p{index}.npz") as saved:s,o=saved["syndromes"],saved["observables"]
        with np.load(confirmation/f"values-p{index}-selected.npz") as saved:selected=saved["failed"]
        for row in r["outcomes"]:
            choice=frozen["choices"][str(row["T"])]
            assert row["factor"]==choice["factor"]
            with np.load(root/f"confirmation-T{row['T']}.npz") as saved:
                failed=~(parity(sampler.problem.H,saved["corrections"])==s).all(1)
                logical=failed | (parity(sampler.problem.L,saved["corrections"])!=o).any(1)
                np.testing.assert_array_equal(saved["failed"],failed);np.testing.assert_array_equal(saved["logical_failed"],logical)
                assert (row["shots"],row["failures"],row["logical_failures"])==(len(s),int(failed.sum()),int(logical.sum()))
                pair=mcnemar(failed,selected)
                assert row["secondary_nominal_comparison"]==pair
                pairs.append(dict(p=r["p"],T=row["T"],**pair))
                value=decode(LdpcBP(sampler.problem,row["T"],row["factor"]),s[:64],o[:64],sampler.problem,64)
                for key,array in value.items():np.testing.assert_array_equal(array,saved[key][:64])
            verified+=len(s);replayed+=64
        (root/"audit_receipt.json").write_text(json.dumps(dict(directory=str(d)),indent=2)+"\n")
        print(f"Audited long-control challenge at p={r['p']:g}",flush=True)
    corrected=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]([row["pvalue"] for row in pairs])
    for row,value in zip(pairs,corrected):row["secondary_holm_pvalue"]=value
    audit=dict(checks_passed=True,directories=list(map(str,args.directories)),raw_outcomes_verified=verified,
        replayed_decoder_cases=replayed,secondary_four_test_family=pairs,
        scope="Secondary posthoc stronger-iteration-budget challenge. No primary replacement or LER/latency claim.",
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
