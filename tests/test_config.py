from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from bbnbp.config import load_config, CIRCUIT_POLICY
from dataclasses import replace

BASE = Path(__file__).resolve().parents[1]/"configs/base.yaml"


@pytest.mark.parametrize("section", ["eval", "sweep"])
@pytest.mark.parametrize("offset", [True, False, 1000.5, -1])
def test_seed_offset_requires_nonnegative_integer(section, offset):
    config = load_config(BASE)
    updated = replace(config, **{section: replace(getattr(config, section), seed_offset=offset)})
    with pytest.raises(ValueError, match="Seed offsets"):
        updated.validate()


def test_historical_config_hash_is_preserved():
    path = BASE.parents[0].parent/"results/20261030T191424331467Z-0c364ef1/config.yaml"
    assert load_config(path).config_hash == "0c364ef1c90ff244cd0f53277ac4602a3984025c"


def test_circuit_policy_is_explicit(tmp_path):
    config = load_config(BASE)
    config = replace(config, noise=replace(config.noise, model="circuit_level", rounds=6))
    with pytest.raises(ValueError, match="explicitly"):
        config.validate()
    config = replace(config, noise=replace(config.noise, circuit=CIRCUIT_POLICY)).validate()
    config.save(tmp_path/"circuit.yaml")
    assert load_config(tmp_path/"circuit.yaml") == config


@pytest.mark.parametrize("offset", [-1, True, 0.5, 1])
def test_sample_offset_is_whole_sampling_batches(offset):
    config = load_config(BASE)
    with pytest.raises(ValueError, match="Sample offset"):
        replace(config, eval=replace(config.eval, sample_offset=offset)).validate()


def test_config_roundtrip_and_hash(tmp_path):
    config = load_config(BASE)
    config.save(tmp_path/"resolved.yaml")
    restored = load_config(tmp_path/"resolved.yaml")
    assert restored == config and restored.config_hash == config.config_hash
    with pytest.raises(FrozenInstanceError):
        config.seed = 3


@pytest.mark.parametrize("mutation", ["missing", "unknown", "bad_p", "bad_seed", "yaml_off"])
def test_invalid_config_rejected(tmp_path, mutation):
    raw = yaml.safe_load(BASE.read_text())
    if mutation == "missing":
        del raw["eval"]["stop_failures"]
    elif mutation == "unknown":
        raw["noise"]["surprise"] = True
    elif mutation == "bad_p":
        raw["noise"]["p_grid"] = [0.5]
    elif mutation == "bad_seed":
        raw["seed"] = True
    else:
        raw["train"]["hard_mining"] = False
    path = tmp_path/"invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)
