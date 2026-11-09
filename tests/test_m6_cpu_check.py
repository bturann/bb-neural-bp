import os
import runpy
import subprocess
import sys
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
