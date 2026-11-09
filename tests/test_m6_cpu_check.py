import os
import runpy
import subprocess
import sys
import shutil
import csv
from dataclasses import replace

import pytest

from bbnbp.config import load_config


def test_holm_adjustment_restores_original_order_and_enforces_monotonicity():
    adjust = runpy.run_path("scripts/m6_compare_stage1.py")["holm"]
    assert adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    assert adjust([0.9, 0.8]) == [1.0, 1.0]
    assert adjust([0, 0.001, 1]) == pytest.approx([0, 0.002, 1])


def test_fresh_comparison_rejects_checkpoint_selection_seed_reuse(tmp_path):
    config = load_config("configs/m6/cpu_pilot_check.yaml")
    config = replace(config, eval=replace(config.eval, seed_offset=12000))
    path = tmp_path/"overlap.yaml"
    config.save(path)
    output = tmp_path/"comparison"
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path/"mpl"), BBNBP_CACHE_DIR=str(tmp_path/"cache"))
    result = subprocess.run([sys.executable, "scripts/m6_compare_stage1.py", "--config", str(path),
                             "--tuning-table", str(tmp_path/"unused.csv"), "--output", str(output)],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "must be disjoint" in result.stderr
    assert not output.exists()


def test_reusing_tuning_preserves_source_hash_and_rejects_incompatible_noise():
    validate = runpy.run_path("scripts/m6_compare_stage1.py")["validate_tuning_table"]
    config = load_config("configs/m6/cpu_pilot_check.yaml")
    changed = replace(config, eval=replace(config.eval, seed_offset=19000))
    table = "results/m6_cpu_pilot_check/20261109T040246608374Z-cb99df2f/tuned_scaling.csv"
    assert changed.config_hash != config.config_hash
    assert validate(changed, table).config_hash == config.config_hash
    with pytest.raises(ValueError, match="configurations differ"):
        validate(replace(changed, noise=replace(changed.noise, basis="X")), table)


def test_reused_tuning_requires_actual_row_provenance(tmp_path):
    validate = runpy.run_path("scripts/m6_compare_stage1.py")["validate_tuning_table"]
    config = load_config("configs/m6/cpu_pilot_check.yaml")
    source = "results/m6_cpu_pilot_check/20261109T040246608374Z-cb99df2f/tuned_scaling"
    with open(source+".csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["config_hash"] = "wrong"
    with (tmp_path/"table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(source+".config.yaml", tmp_path/"table.config.yaml")
    with pytest.raises(ValueError, match="source provenance"):
        validate(config, tmp_path/"table.csv")
