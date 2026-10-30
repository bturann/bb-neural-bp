import csv
import json
from dataclasses import replace

import numpy as np
import torch
import pytest

from bbnbp.checkpoints import checkpoint_payload, save_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.evaluate import run
from bbnbp.noise.code_capacity import CodeCapacitySampler
from bbnbp.m6_report import seed_summary


@pytest.mark.parametrize("latency_only", [False, True])
def test_multiple_models_share_shots_and_latency_components(tmp_path, latency_only):
    config = load_config("configs/cc_bb72_e0.yaml")
    sampler = CodeCapacitySampler("bb72", 0.06)
    paths = []
    for seed in (1234, 1235, 1236):
        model = NeuralMinSum(sampler.problem, 3, "orbit")
        c = replace(config, seed=seed, model=replace(config.model, T=3))
        path = tmp_path/f"{seed}.pt"
        save_checkpoint(path, checkpoint_payload(model, c, 1))
        paths.append(str(path))
    config = replace(config, noise=replace(config.noise, p_grid=(0.06,)),
                     eval=replace(config.eval, stage1_iters=3, max_shots=12, batch=4,
                                  stop_failures=10, latency_shots=6, latency_warmup=2,
                                  checkpoints=tuple(paths), cpu_model="test CPU", latency_only=latency_only))
    output = run(config, ["bp", "nbp"], None, tmp_path/"results")
    with (output/"results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and len({r["shots"] for r in rows}) == 1
    assert all(int(r["shots"]) == (6 if latency_only else 12) for r in rows)
    assert {r["weight_sharing"] for r in rows} == {"scalar", "orbit"}
    assert all("test CPU" in r["device"] for r in rows)
    details = json.loads((output/"diagnostics.json").read_text())
    assert {r["training_seed"] for r in details["checkpoints"]} == {1234, 1235, 1236}
    summary = seed_summary(rows, details["checkpoints"])
    assert summary[0]["osd_rate_std"] == 0
    assert len(summary[0]["models"]) == 3
    assert all(model["shots"] == int(rows[0]["shots"]) for model in summary[0]["models"])
    syndrome_hashes = set()
    for name in ("bp", "nbp_0", "nbp_1", "nbp_2"):
        with np.load(output/f"latency-components-p0-{name}.npz") as samples:
            assert len(samples["total_us"]) == 6
            assert np.all(samples["overhead_us"] >= 0)
            np.testing.assert_allclose(samples["total_us"], samples["stage1_us"]+samples["osd_us"]+samples["overhead_us"])
            assert np.all(samples["osd_us"][~samples["osd_used"]] == 0)
            syndrome_hashes.add(str(samples["syndrome_sha256"]))
            if latency_only:
                row = next(r for r in rows if r["stage1"] == name)
                assert int(row["logical_failures"]) == int(samples["logical_flags"].sum())
    assert len(syndrome_hashes) == 1
    with np.load(output/"logical-p0.npz") as values:
        np.testing.assert_array_equal(values["nbp_0"], values["nbp_1"])
        np.testing.assert_array_equal(values["nbp_0"], values["nbp_2"])


def test_training_seed_spread_uses_sample_standard_deviation():
    checkpoints = [{"training_seed": seed} for seed in (1234, 1235, 1236)]
    rows = [dict(p=0.003, stage1=f"nbp_{i}", osd_rate=rate, shots=100,
                 logical_failures=2, ler=0.02, ler_lo=0.005, ler_hi=0.07)
            for i, rate in enumerate((0.1, 0.2, 0.3))]
    summary = seed_summary(rows, checkpoints)
    assert summary[0]["osd_rate_mean"] == pytest.approx(0.2)
    assert summary[0]["osd_rate_std"] == pytest.approx(0.1)
    assert [m["shots"] for m in summary[0]["models"]] == [100, 100, 100]
