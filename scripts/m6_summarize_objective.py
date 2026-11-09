"""Audit stored objective scores and summarize paired diagnostic observations."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args = parser.parse_args()
    d = args.directory
    output = d/"alignment_summary.json"
    if output.exists():
        parser.error("Refusing to overwrite summary evidence")
    source = json.loads((d/"summary.json").read_text())
    manifest = json.loads((d/"source_manifest.json").read_text())
    code = Path("scripts/m6_objective_diagnostic.py")
    assert hashlib.sha256(code.read_bytes()).hexdigest()==manifest[str(code)]
    (d/"executed_diagnostic_source.py").write_bytes(code.read_bytes())
    helpers = runpy.run_path(str(code))
    names = list(dict.fromkeys(row["model"] for row in source["rows"]))
    pooled = {name:{} for name in names}
    checked = 0
    for index,p in enumerate(sorted({row["p"] for row in source["rows"]})):
        for name in names:
            row = next(row for row in source["rows"] if row["p"]==p and row["model"]==name)
            with np.load(d/f"scores-p{index}-{name}.npz") as values:
                arrays = {key:values[key] for key in values.files}
            failed = ~arrays["converged"]
            assert row["shots"]==len(failed)
            assert row["fallback_demand"]==float(failed.mean())
            assert row["lost_by_final"]==int((arrays["converged"] & ~arrays["last_converged"]).sum())
            assert row["lost_at_any_later_iteration"]==int(arrays["lost_after_first"].sum())
            for objective in ("parity","bitwise"):
                loss = arrays[f"{objective}_loss"]
                assert np.isfinite(loss).all()
                assert row[f"mean_{objective}_loss"]==float(loss.mean())
                assert row[f"{objective}_failure_auc"]==helpers["failure_auc"](loss,failed)
            for key,array in arrays.items():
                pooled[name].setdefault(key,[]).append(array)
            checked += len(failed)
    data = {name:{key:np.concatenate(parts) for key,parts in values.items()} for name,values in pooled.items()}
    summaries = {}
    for name,values in data.items():
        successes = values["converged"]
        summary = dict(shots=len(successes),first_converged=int(successes.sum()),
                       final_converged=int(values["last_converged"].sum()),
                       lost_by_final=int((successes & ~values["last_converged"]).sum()),
                       lost_at_any_later_iteration=int(values["lost_after_first"].sum()),
                       first_stage_logical_failures=int(values["first_logical_failed"].sum()),
                       mean_parity_loss=float(values["parity_loss"].mean()),mean_bitwise_loss=float(values["bitwise_loss"].mean()),
                       mean_clipped_probability_fraction=float(values["clipped_fraction"].mean()))
        if name!="initial":
            initial=data["initial"]
            objective="parity_loss" if name.startswith("parity") else "bitwise_loss"
            down=values[objective]<initial[objective]
            regression=initial["converged"] & ~values["converged"]
            correct_regression=regression & ~initial["first_logical_failed"]
            summary.update(loss_down_cases=int(down.sum()),
                           convergence_gained=int((~initial["converged"] & successes).sum()),
                           convergence_lost=int(regression.sum()),
                           loss_down_but_convergence_lost=int((down & regression).sum()),
                           loss_down_but_correct_initial_decoding_lost=int((down & correct_regression).sum()),
                           mean_training_objective_change=float((values[objective]-initial[objective]).mean()))
        summaries[name]=summary
    report=dict(purpose="descriptive_objective_alignment_summary",models=summaries,
                audit=dict(checks_passed=True,checked_model_shots=checked,
                           exact_trace_vs_numba_checks=checked,per_p_auc_recomputed=True),
                raw_source_sha256=hashlib.sha256((d/"summary.json").read_bytes()).hexdigest(),
                summary_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                final_test_data_used=False,
                caveats="Same 640 saved development cases per model; no new holdout or causal claim. "
                        "Different p values are not pooled for AUC. Gradients refer to final probabilities, "
                        "not parameter gradients; clipping can be appropriate for confidently correct variables.")
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
