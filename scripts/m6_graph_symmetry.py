"""Read-only graph diagnostics for exact parallel-fault symmetry and tied weights."""

import hashlib
import json
import shutil
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from bbnbp.config import load_config
from bbnbp.io import source_manifest
from bbnbp.noise.factory import make_sampler
from bbnbp.train import make_model


def main():
    c=load_config("configs/m6/cpu_fixed_case.yaml")
    sampler=make_sampler(c,c.noise.p_grid[0])
    problem=sampler.problem
    h,l=problem.H.tocsc(),problem.L.tocsc()
    groups=defaultdict(list)
    full_groups=defaultdict(list)
    prior_groups=defaultdict(list)
    for j in range(h.shape[1]):
        hk=tuple(h.indices[h.indptr[j]:h.indptr[j+1]])
        lk=tuple(l.indices[l.indptr[j]:l.indptr[j+1]])
        groups[hk].append(j)
        full_groups[hk,lk].append(j)
        prior_groups[hk,lk,float(problem.priors[j])].append(j)
    model=make_model(problem,c)
    paired=[indices for indices in groups.values() if len(indices)>1]
    report=dict(purpose="Read-only exact-column structural diagnostics; no decoder or experiment matrix change",p=c.noise.p_grid[0],
        fault_columns=h.shape[1],detectors=h.shape[0],edges=h.nnz,parameters=sum(p.numel() for p in model.parameters()),
        unique_detector_columns=len(groups),unique_detector_observable_columns=len(full_groups),
        duplicate_detector_groups=len(paired),faults_in_duplicate_detector_groups=sum(map(len,paired)),
        redundant_identical_detector_observable_columns=h.shape[1]-len(full_groups),
        redundant_identical_columns_and_priors=h.shape[1]-len(prior_groups),
        detector_column_multiplicities={str(k):v for k,v in sorted(Counter(map(len,groups.values())).items())},
        variable_degrees={str(k):v for k,v in sorted(Counter(np.diff(h.indptr)).items())},
        zero_detector_columns=len(groups.get((),[])),
        scope="An exact equality check, not proof of absent higher-order graph symmetries or a cause of learning failure.",
        final_test_data_used=False,m6_complete=False)
    d=Path("results/m6_graph_symmetry")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d.mkdir(parents=True,exist_ok=False)
    c.save(d/"config.yaml")
    (d/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (d/"source_manifest.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")
    shutil.copyfile(__file__,d/"executed_source.py")
    print(d)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
