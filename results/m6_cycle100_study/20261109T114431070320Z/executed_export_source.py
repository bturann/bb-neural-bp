"""Export completed strong/cycle100 raw Stage1 outcomes to the standard CSV schema."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bbnbp.checkpoints import read_checkpoint
from bbnbp.config import load_config
from bbnbp.evaluate import cpu_model_label
from bbnbp.io import append_results, git_sha
from bbnbp.metrics import wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();d=args.directory
    if (d/"results.csv").exists():raise ValueError("CSV already exported")
    r=json.loads((d/"summary.json").read_text());frozen=json.loads((d/"selection.json").read_text())
    cycle=frozen.get("iterations")==100
    c=load_config(d/"config.yaml");payload=read_checkpoint(d/"selected-source.pt")
    rows=[];verified=0;timestamp=datetime.now(timezone.utc).isoformat()
    for index,(point,choice) in enumerate(zip(r["points"],frozen["points"])):
        sampler=make_sampler(c,point["p"])
        with np.load(d/f"cases-p{index}.npz") as saved:s,o=saved["syndromes"],saved["observables"]
        for name,count in point["failures"].items():
            with np.load(d/f"values-p{index}-{name}.npz") as saved:
                failed=~(parity(sampler.problem.H,saved["corrections"])==s).all(1)
                logical=failed | (parity(sampler.problem.L,saved["corrections"])!=o).any(1)
                np.testing.assert_array_equal(failed,saved["failed"]);np.testing.assert_array_equal(logical,saved["logical_failed"])
                assert int(failed.sum())==count
            neural=name in {"selected","matched_init","seed1235","seed1236","strong_cascade","strong_init"}
            T=100 if cycle or name=="bp100" else 130
            factor=payload["model_config"]["ms_scaling"]
            if cycle and name in {"cpp100","native100"}:factor=choice["cpp" if name=="cpp100" else "native"]["factor"]
            elif not cycle and name in {"bp100","bp130"}:factor=choice["first"]["factor"] if name=="bp100" else choice["choices"]["bp130"]["factor"]
            elif not cycle:factor=choice["first"]["factor"]
            n=len(s);failures=int(logical.sum());lo,hi=wilson(failures,n)
            rows.append(dict(run_id=d.name,git_sha=git_sha(),config_hash=c.config_hash,timestamp=timestamp,
                code=c.code.name,noise_model=c.noise.model,rounds=c.noise.rounds,basis=c.noise.basis,p=point["p"],
                decoder=name+"_only",stage1=name,T=T,weight_sharing=payload["model_config"]["sharing"] if neural else "scalar",
                osd_method="",osd_order="",ms_scaling=factor,shots=n,stage1_failures=count,osd_calls=0,osd_rate=0,
                logical_failures=failures,ler=failures/n,ler_lo=lo,ler_hi=hi,ler_stage1=failures/n,
                lat_mean_us="",lat_p50_us="",lat_p99_us="",lat_max_us="",lat_threads=1,
                device=f"CPU {cpu_model_label(c.eval.cpu_model)}; {platform.platform()}",seed=point["seed"]))
            verified+=n
    append_results(d/"results.csv",rows)
    export=dict(rows=len(rows),raw_outcomes_reverified=verified,csv_sha256=hashlib.sha256((d/"results.csv").read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((d/"summary.json").read_bytes()).hexdigest(),
        scope="Postprocessed standalone Stage1 outcomes. osd_calls/rate are zero; stage1_failures/shots is potential fallback demand. T is inference maximum, source training remains T15. Base config plus frozen design/settings resolves the auxiliary experiment.",
        export_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/"executed_export_source.py").write_bytes(Path(__file__).read_bytes())
    (d/"export_manifest.json").write_text(json.dumps(export,indent=2)+"\n")
    print(json.dumps(export,indent=2))


if __name__=="__main__":main()
