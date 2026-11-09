"""Review existing final versus selected checkpoints on all saved development cases."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from bbnbp.checkpoints import load_model
from bbnbp.config import load_config
from bbnbp.io import git_sha, source_manifest
from bbnbp.metrics import mcnemar
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default="configs/m6/cpu_checkpoint_review.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite review evidence")
    settings = yaml.safe_load(Path(args.settings).read_text())
    source = Path(settings["source"])
    c = load_config(source/"config.yaml")
    study = json.loads((source/"summary.json").read_text())
    torch.set_num_threads(settings["threads"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"settings.yaml").write_text(yaml.safe_dump(settings))
    (args.output/"source_manifest.json").write_text(json.dumps(source_manifest(), indent=2)+"\n")
    started, rows = time.perf_counter(), []
    try:
        for i,point in enumerate(study["points"]):
            sampler = make_sampler(c, point["p"])
            with np.load(source/f"cases-p{i}.npz") as cases:
                s,o = cases["syndromes"],cases["observables"]
            saved_flags = {}
            for arm in ("parity", "bitwise"):
                if time.perf_counter()-started > settings["wall_seconds"]:
                    raise TimeoutError("Checkpoint review reached its CPU wall budget")
                best = Path(study["checkpoints"][arm]["path"])
                assert hashlib.sha256(best.read_bytes()).hexdigest() == study["checkpoints"][arm]["sha256"]
                last = best.parent/"last.pt"
                model = load_model(last,sampler.problem)
                compiled = model.export_cpu()
                n = min(settings["export_check_shots"],len(s))
                expected,conv = model.decode_batch(s[:n])
                actual,actual_conv = compiled.decode_batch(s[:n])
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(actual_conv,conv)
                np.testing.assert_array_equal(compiled.last_iterations,model.last_iterations)
                failed,logical,iterations = _inference(compiled,s,o,sampler.problem,c.eval.batch)
                with np.load(source/f"failed-p{i}.npz") as flags:
                    selected = flags[f"nbp_{arm}"]
                    rows.append(dict(p=point["p"],arm=arm,shots=len(s),selected_fallback=float(selected.mean()),
                                     last_fallback=float(failed.mean()),paired_last_vs_selected=mcnemar(selected,failed),
                                     last_checkpoint_sha256=hashlib.sha256(last.read_bytes()).hexdigest(),
                                     export_checked_shots=n,mean_last_iterations=iterations))
                saved_flags[f"{arm}_last_failed"] = failed
                saved_flags[f"{arm}_last_logical_failed"] = logical
                print(f"p={point['p']:g} {arm}: selected={rows[-1]['selected_fallback']:.6f}, last={rows[-1]['last_fallback']:.6f}",flush=True)
            np.savez_compressed(args.output/f"last-flags-p{i}.npz",**saved_flags)
        means = {arm:{key:float(np.mean([row[f"{key}_fallback"] for row in rows if row["arm"]==arm]))
                      for key in ("selected","last")} for arm in ("parity","bitwise")}
        report = dict(purpose=settings["purpose"],settings=settings,rows=rows,means=means,
                      elapsed_s=time.perf_counter()-started,git_sha=git_sha(),final_test_data_used=False,
                      selected_checkpoints_changed=False,
                      scope="Retrospective reuse of saved development cases; final checkpoints were saved before this analysis. "
                            "No new checkpoint selection, training, final-test or confirmatory performance claim.")
        (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(means,indent=2))
    except BaseException as exc:
        (args.output/"FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
