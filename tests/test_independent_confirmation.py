import csv
import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from bbnbp.checkpoints import read_checkpoint
from bbnbp.config import load_config


def test_independent_confirmation_rejects_larger_selection_pool_reuse():
    check=runpy.run_path("scripts/m6_independent_confirmation.py")["check_seed_separation"]
    preferences=json.loads(Path("results/m6_validation_stability/20261109T073249552750Z/validation_preferences.json").read_text())
    payloads={arm:read_checkpoint(entry["checkpoint"]["path"]) for arm,entry in preferences["candidates"].items()}
    with open("results/m6_cpu_pilot_check/20261109T040246608374Z-cb99df2f/tuned_scaling.csv",newline="") as stream:
        rows=list(csv.DictReader(stream))
    c=load_config("configs/m6/cpu_pilot_check.yaml")
    reused=replace(c,eval=replace(c.eval,seed_offset=25000))
    with pytest.raises(ValueError,match="larger validation"):
        check(reused,preferences,payloads,rows)
    fresh=replace(c,eval=replace(c.eval,seed_offset=27000))
    assert check(fresh,preferences,payloads,rows)==[28234,28235,28236,28237,28238]
