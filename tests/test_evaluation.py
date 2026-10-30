import csv
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bbnbp.config import load_config
from bbnbp.evaluate import run, tuned_factor
from bbnbp.io import RESULT_COLUMNS, append_results

ROOT = Path(__file__).resolve().parents[1]


def test_small_tuning_and_evaluation_roundtrip(tmp_path):
    spec = importlib.util.spec_from_file_location("sweep", ROOT/"scripts/sweep_scaling_factor.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    config = load_config(ROOT/"configs/cc_bb72_e0.yaml")
    config = replace(config, noise=replace(config.noise, p_grid=(0.06,)),
                     sweep=replace(config.sweep, factors=(0.5, 0.75), iterations=(5,), shots=50),
                     eval=replace(config.eval, stage1_iters=5, max_shots=40, batch=20,
                                  stop_failures=5)).validate()
    table = tmp_path/"tuning.csv"
    rows = sweep.run(config, table)
    assert len(rows) == 2 and sum(r["selected"] for r in rows) == 1
    assert tuned_factor(table, config, 0.06) in (0.5, 0.75)
    with pytest.raises(FileExistsError):
        sweep.run(config, table)
    output = run(config, ["bp", "bp_tuned", "bp_only"], table, tmp_path/"results")
    assert (output/"config.yaml").exists() and (output/"paired-p0.npz").exists()
    assert (output/"tuned_scaling.csv").read_bytes() == table.read_bytes()
    diagnostics = json.loads((output/"diagnostics.json").read_text())
    assert diagnostics["purpose"] == "development"
    with (output/"results.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        assert reader.fieldnames == RESULT_COLUMNS
    assert len(records) == 3 and len({row["shots"] for row in records}) == 1
    assert all(int(r["shots"]) <= 40 for r in records)
    for row in records[:2]:
        assert int(row["osd_calls"]) == int(row["stage1_failures"])


def test_result_schema_enforced(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        append_results(tmp_path/"wrong.csv", [{"ler": 0}])
