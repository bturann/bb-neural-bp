import csv
import importlib.util
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np

from bbnbp.config import load_config, CIRCUIT_POLICY
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
    with np.load(output/"logical-p0.npz") as flags:
        for name, row in zip(["bp", "bp_tuned", "bp_only"], records):
            assert len(flags[name]) == int(row["shots"])
            assert int(flags[name].sum()) == int(row["logical_failures"])


def test_result_schema_enforced(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        append_results(tmp_path/"wrong.csv", [{"ler": 0}])


@pytest.mark.parametrize("purpose", ["development", "final"])
def test_actual_tuning_seeds_cannot_overlap_evaluation(tmp_path, purpose):
    config = load_config(ROOT/"configs/cc_bb72_e0.yaml")
    config = replace(config, eval=replace(config.eval, purpose=purpose))
    seed = config.seed+config.eval.seed_offset
    row = dict(code="bb72", noise_model="code_capacity", basis="Z", rounds="", p=0.02,
               T=config.eval.stage1_iters, ms_scaling=0.95, selected=True, seed=1)
    other = dict(row, selected=False, seed=seed)
    table = tmp_path/"overlap.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows([row, other])
    with pytest.raises(ValueError, match="seeds overlap"):
        tuned_factor(table, config, 0.02)


def test_experiment_command_can_run_twice(tmp_path):
    config = load_config(ROOT/"configs/cc_bb72_e0.yaml")
    config = replace(config, noise=replace(config.noise, p_grid=(0.06,)),
                     sweep=replace(config.sweep, factors=(0.5,), iterations=(5,), shots=10),
                     eval=replace(config.eval, stage1_iters=5, max_shots=10, batch=10,
                                  stop_failures=1)).validate()
    path = tmp_path/"config.yaml"
    config.save(path)
    command = ["bash", str(ROOT/"scripts/run_experiment.sh"), "--config", str(path),
               "--output-root", str(tmp_path/"runs")]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    first = {str(p): p.read_bytes() for p in (tmp_path/"runs").rglob("*") if p.is_file()}
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert len(list((tmp_path/"runs").glob("e0-*"))) == 2
    assert len(list((tmp_path/"runs").glob("e0-*/*/results.csv"))) == 2
    assert all(Path(p).read_bytes() == data for p, data in first.items())


def test_circuit_evaluation_records_actual_circuit_and_dem(tmp_path):
    config = load_config(ROOT/"configs/cc_bb72_e0.yaml")
    config = replace(config, noise=replace(config.noise, model="circuit_level", rounds=1,
                                           p_grid=(0.001,), circuit=CIRCUIT_POLICY),
                     eval=replace(config.eval, max_shots=4, batch=4, stop_failures=1,
                                  osd=replace(config.eval.osd, method="osd0", order=0))).validate()
    output = run(config, ["bp"], None, tmp_path/"results")
    assert (output/"circuit-p0.stim").exists()
    assert (output/"dem-p0.dem").exists()
    meta = json.loads((output/"problem-p0.json").read_text())
    assert meta["rounds"] == 1 and meta["n_detectors"] == 72
    assert meta["circuit_policy"]["noisy_final_meas"] is True
    assert meta["n_faults"] > 0


def test_segmented_evaluation_reproduces_continuous_sample_stream(tmp_path):
    config = load_config(ROOT/"configs/cc_bb72_e0.yaml")
    config = replace(config, noise=replace(config.noise, p_grid=(0.06,)),
                     eval=replace(config.eval, max_shots=40, batch=10, stop_failures=100)).validate()
    whole = run(config, ["bp"], None, tmp_path/"whole")
    parts = []
    for offset in (0, 20):
        part = replace(config, eval=replace(config.eval, max_shots=20, sample_offset=offset)).validate()
        parts.append(run(part, ["bp"], None, tmp_path/"parts"))
    for artifact in ("paired-p0.npz", "logical-p0.npz"):
        with np.load(whole/artifact) as data:
            expected = data["bp"].copy()
        arrays = []
        for directory in parts:
            with np.load(directory/artifact) as data:
                arrays.append(data["bp"].copy())
        np.testing.assert_array_equal(expected, np.concatenate(arrays))
