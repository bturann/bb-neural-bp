"""Audit fresh fixed validation cases, config provenance, rankings and bootstrap."""

import argparse
import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch

from bbnbp.checkpoints import load_model, read_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler
from bbnbp.train import _inference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    d=args.directory
    if (d/"audit.json").exists():
        parser.error("Refusing to overwrite audit evidence")
    report=json.loads((d/"summary.json").read_text())
    settings=report["settings"]
    c=load_config(d/"config.yaml")
    assert c.eval.seed_offset==settings["validation_seed_offset"] and c.eval.max_shots==settings["shots_per_p"]
    assert c.eval.batch==settings["batch"] and c.seed==settings["seed"]
    with (d/"results.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==25 and all(row["config_hash"]==c.config_hash for row in rows)
    manifest=json.loads((d/"source_manifest.json").read_text())
    assert hashlib.sha256((d/"executed_source.py").read_bytes()).hexdigest()==manifest["scripts/m6_validation_stability.py"]
    paths={name:Path(meta["path"]) for name,meta in report["checkpoints"].items()}
    for name,path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==report["checkpoints"][name]["sha256"]
    assert set(c.eval.checkpoints)=={str(path) for path in paths.values()}
    torch.set_num_threads(1)
    payload=read_checkpoint(paths["parity_selected"])
    labels=("initial",*paths)
    by_model={name:[] for name in labels}
    regenerated,checked,replayed,exports=0,0,0,0
    for i,point in enumerate(report["points"]):
        assert point["seed"]==c.seed+c.eval.seed_offset+i
        sampler=make_sampler(c,point["p"])
        s,o,e=sampler.sample(c.eval.max_shots,np.random.default_rng(point["seed"]))
        np.testing.assert_array_equal(parity(sampler.problem.H,e),s)
        np.testing.assert_array_equal(parity(sampler.problem.L,e),o)
        del e
        with np.load(d/f"cases-p{i}.npz") as saved:
            np.testing.assert_array_equal(saved["syndromes"],s)
            np.testing.assert_array_equal(saved["observables"],o)
        regenerated+=len(s)
        models={"initial":NeuralMinSum(sampler.problem,**payload["model_config"])}
        models.update({name:load_model(path,sampler.problem) for name,path in paths.items()})
        with np.load(d/f"failed-p{i}.npz") as flags,np.load(d/f"logical-p{i}.npz") as logical:
            for name,model in models.items():
                row=next(row for row in rows if row["stage1"]==f"nbp_{name}" and float(row["p"])==point["p"])
                assert int(row["seed"])==point["seed"] and int(row["shots"])==len(s)
                assert int(row["stage1_failures"])==int(flags[name].sum())
                assert int(row["logical_failures"])==int(logical[name].sum())
                assert int(row["osd_calls"])==float(row["osd_rate"])==0
                assert float(flags[name].mean())==point["models"][name]["fallback_demand"]
                np.testing.assert_allclose(wilson(int(logical[name].sum()),len(s)),[float(row["ler_lo"]),float(row["ler_hi"])])
                n=min(settings["export_check_shots"],len(s))
                expected,conv=model.decode_batch(s[:n])
                compiled=model.export_cpu()
                actual,actual_conv=compiled.decode_batch(s[:n])
                np.testing.assert_array_equal(expected,actual)
                np.testing.assert_array_equal(conv,actual_conv)
                np.testing.assert_array_equal(model.last_iterations,compiled.last_iterations)
                failed,logical_subset,_=_inference(compiled,s[:n],o[:n],sampler.problem,n)
                np.testing.assert_array_equal(failed,flags[name][:n])
                np.testing.assert_array_equal(logical_subset,logical[name][:n])
                checked+=len(s)
                replayed+=n
                exports+=n
                by_model[name].append(flags[name])
            for arm in ("parity","bitwise"):
                test=mcnemar(flags[f"{arm}_selected"],flags[f"{arm}_last"])
                assert all(value==point["paired"][arm][key] for key,value in test.items())
    funcs=runpy.run_path("scripts/m6_validation_stability.py")
    holm=runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    pairs=[p["paired"][arm] for p in report["points"] for arm in ("parity","bitwise")]
    np.testing.assert_allclose(holm([p["pvalue"] for p in pairs]),[p["holm_pvalue"] for p in pairs])
    rng=np.random.default_rng(np.random.SeedSequence(settings["bootstrap_seed_entropy"]))
    for arm in ("parity","bitwise"):
        a,b=by_model[f"{arm}_selected"],by_model[f"{arm}_last"]
        assert funcs["block_preferences"](a,b,settings["small_block_shots"])==report["rankings"][arm]["small_blocks"]
        delta=[y.astype(np.int8)-x.astype(np.int8) for x,y in zip(a,b)]
        boot=funcs["paired_bootstrap"](delta,rng,settings["bootstrap_replicates"],settings["confidence_level"])
        assert boot==report["rankings"][arm]["paired_bootstrap"]
    for name,values in by_model.items():
        assert float(np.mean([v.mean() for v in values]))==report["mean_fallback"][name]
    result=dict(checks_passed=True,resolved_config_hash_verified=c.config_hash,regenerated_cases=regenerated,
        checked_decoder_flags=checked,replayed_subset_decodes=replayed,exact_export_shots=exports,checked_csv_rows=len(rows),
        bootstrap_reproduced=True,disjoint_blocks_reproduced=True,checkpoint_hashes_unchanged=True,
        final_test_data_used=False,independent_performance_claim=False)
    (d/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
