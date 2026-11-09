"""Reproduce posthoc oracle opportunity from existing BP validation outcomes only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();d=args.directory
    if (d/"audit.json").exists():raise ValueError("Audit already exists")
    original=json.loads((d/"summary.json").read_text())
    longer=json.loads((d/"longer_upper_bound.json").read_text())
    checked=0
    for index,source in enumerate(map(Path,original["validation_sources"])):
        selection=json.loads((source/"selection.json").read_text())
        with np.load(source/"first_pass.npz") as saved:first={k:saved[k] for k in saved.files}
        mask=first["failed"]
        factors=np.array(sorted(row["scaling"] for row in selection["grid"] if row["kind"]=="longer"))
        for kind,prefix in (("two_pass",""),("longer","longer-")):
            converged,correct=[],[]
            for factor in factors:
                filename=(f"two_pass-{factor:g}.npz" if kind=="two_pass" else f"longer-{factor:g}.npz")
                with np.load(source/filename) as saved:
                    converged.append(~saved["failed"]);correct.append(~saved["logical_failed"])
            converged,correct=np.stack(converged,axis=1),np.stack(correct,axis=1)
            with np.load(d/f"{prefix}validation-labels-p{index}.npz") as saved:
                np.testing.assert_array_equal(saved["factors"],factors)
                expected_c=converged[mask] if kind=="two_pass" else converged
                expected_l=correct[mask] if kind=="two_pass" else correct
                np.testing.assert_array_equal(saved["converged"],expected_c)
                np.testing.assert_array_equal(saved["correct"],expected_l)
                if kind=="two_pass":np.testing.assert_array_equal(saved["source_indices"],np.flatnonzero(mask))
            if kind=="two_pass":
                point=original["points"][index]
                assert point["oracle_union_failures"]==int((~converged.any(1)).sum())
                assert point["oracle_correct_rescues"]==int(correct[mask].any(1).sum())
                assert point["best_constant_two_pass_failures"]==int((~converged).sum(0).min())
                assert point["bp30_failures"]==selection["selected"]["longer"]["failures"]
            else:
                assert longer[index]["oracle_failures"]==int((~converged.any(1)).sum())
                assert longer[index]["oracle_correct_failures"]==int((~correct.any(1)).sum())
                assert longer[index]["best_fixed_bp30_failures"]==int((~converged).sum(0).min())
            checked+=converged.size
    audit=dict(checks_passed=True,reconstructed_validation_factor_outcomes=checked,
        scope="Oracle union only, not an implementable decoder or performance result. Uses saved, already audited validation outcomes; no new decoding or training.",
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_audit_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
