import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from bbnbp.checkpoints import read_checkpoint
from bbnbp.config import TrainingRuntimeConfig, load_config
from bbnbp.train import hard_batch, run, training_probability
from bbnbp.noise.code_capacity import CodeCapacitySampler


def tiny_config(tmp_path):
    c = load_config("configs/base.yaml")
    table = tmp_path/"tuning.csv"
    rows = [dict(code="bb72", noise_model="code_capacity", rounds="", basis="Z", p=p,
                 T=3, selected=True, ms_scaling=0.75, seed=2234+i) for i, p in enumerate((0.04, 0.06))]
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    runtime = TrainingRuntimeConfig(threads=1, val_batch=16, validation_seed_offset=500,
                                   training_stream=701, torch_seed_offset=100, log_every=2,
                                   loss_eps=1e-7, loss_floor=1e-12, gate_min_points=1,
                                   tensorboard=True, max_gpu_hours=4.0, baseline_table=str(table),
                                   purpose="smoke", m3_gate_report="results/m3/order7-20261030T212238643481Z/m3_gate.json")
    return replace(c, noise=replace(c.noise, p_grid=(0.04, 0.06)), model=replace(c.model, T=3),
                   train=replace(c.train, batch=8, max_steps=4, val_every=2, val_shots=32,
                                 hard_mining="off", patience=10, runtime=runtime,
                                 curriculum=replace(c.train.curriculum, steps=2)),
                   eval=replace(c.eval, stage1_iters=3)).validate()


def test_curriculum_then_grid_is_explicitly_seeded(tmp_path):
    config = tiny_config(tmp_path)
    rng = np.random.default_rng(81)
    assert training_probability(config, 0, rng) == 0.08
    assert training_probability(config, 1, rng) == pytest.approx(0.03)
    assert training_probability(config, 2, rng) in config.noise.p_grid


@pytest.mark.parametrize("mode", ["off", "filter", "upweight"])
def test_hard_mining_modes(tmp_path, mode):
    config = tiny_config(tmp_path)
    config = replace(config, train=replace(config.train, hard_mining=mode, mining_max_draws=3))
    sampler = CodeCapacitySampler("bb72", 0.08)

    class Frozen:
        def __call__(self, s, llr, early_stop):
            self.last_converged = torch.tensor(np.arange(len(s)) % 2 == 0)

    s, e, weights, stats = hard_batch(sampler, np.random.default_rng(42), config, Frozen(), None)
    assert s.shape == (8, 36) and e.shape == (8, 72)
    if mode == "filter":
        assert stats["drawn"] == 16 and not stats["capped"]
    elif mode == "upweight":
        np.testing.assert_array_equal(weights, [1, 10]*4)


def test_training_resume_matches_uninterrupted_steps(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    whole = run(config, tmp_path/"whole", tmp_path/"whole_ckpt", record_notes=False)
    original = torch.optim.Adam.step
    calls = 0

    def interrupt(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt("simulated interruption")
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(config, tmp_path/"interrupted", tmp_path/"interrupted_ckpt", record_notes=False)
    monkeypatch.setattr(torch.optim.Adam, "step", original)
    saved = next((tmp_path/"interrupted_ckpt").glob("*/last.pt"))
    assert read_checkpoint(saved)["step"] == 2
    resumed = run(config, tmp_path/"resumed", tmp_path/"resumed_ckpt", resume=saved, record_notes=False)
    whole_meta = json.loads((whole/"m5_gate.json").read_text())
    resumed_meta = json.loads((resumed/"m5_gate.json").read_text())
    first = read_checkpoint(Path(whole_meta["checkpoint_directory"])/"last.pt")
    second = read_checkpoint(Path(resumed_meta["checkpoint_directory"])/"last.pt")
    for key in ("w", "alpha"):
        torch.testing.assert_close(first["parameters"][key], second["parameters"][key], rtol=0, atol=0)
    assert first["training_state"]["numpy_state"] == second["training_state"]["numpy_state"]
    assert (resumed/"loss_curves.png").exists() and (resumed/"loss_curves.pdf").exists()


def test_m6_training_reports_completion_without_claiming_m5_acceptance(tmp_path):
    config = tiny_config(tmp_path)
    config = replace(config, train=replace(config.train, max_steps=1,
                     runtime=replace(config.train.runtime, purpose="m6_train", validation_backend="torch")))
    output = run(config, tmp_path/"results", tmp_path/"checkpoints", record_notes=False)
    report = json.loads((output/"training_summary.json").read_text())
    assert report["training_complete"] and report["steps_completed"] == 1
    assert "passed" not in report and "eligible_for_m6" not in report
    assert not (output/"m5_gate.json").exists()
