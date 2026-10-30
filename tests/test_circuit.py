"""Circuit/DEM acceptance checks, with exact labels rather than inferred faults."""

import numpy as np
import pytest
import stim
from scipy.stats import binom

from bbnbp.decoders.bp_ldpc import LdpcBP, LdpcBPOSD
from bbnbp.decoders.pipeline import TwoStageDecoder
from bbnbp.noise.base import parity
from bbnbp.noise.circuit_level import CircuitLevelSampler, build_memory_circuit, dem_to_problem


def test_dem_duplicates_separators_repeats_and_observable_only_faults():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D0 L0
        error(0.03) L1
        repeat 2 {
            error(0.04) D0 ^ D1 L0
            shift_detectors 1
        }
        detector D2
    """)
    problem = dem_to_problem(dem, {"name": "toy"})
    assert problem.H.shape == (5, 5) and problem.L.shape == (2, 5)
    np.testing.assert_array_equal(problem.H[:, 0].toarray(), problem.H[:, 1].toarray())
    np.testing.assert_array_equal(problem.priors, [0.1, 0.2, 0.03, 0.04, 0.04])
    assert problem.H[:, 2].nnz == 0 and problem.L[:, 2].nnz == 1
    det, obs, errors = dem.compile_sampler(seed=123).sample(1000, return_errors=True)
    np.testing.assert_array_equal(det, parity(problem.H, errors.astype(np.uint8)))
    np.testing.assert_array_equal(obs, parity(problem.L, errors.astype(np.uint8)))
    cancelled = dem_to_problem(stim.DetectorErrorModel("error(0.1) D0 ^ D0 L0"), {})
    assert cancelled.H.nnz == 0 and cancelled.L.nnz == 1


def test_reject_effectless_error_column():
    with pytest.raises(ValueError, match="no detector or observable effect"):
        dem_to_problem(stim.DetectorErrorModel("error(0.1) D0 ^ D0"), {})


@pytest.mark.slow
def test_six_round_exact_fault_labels_and_priors():
    sampler = CircuitLevelSampler("bb72", 6, 0.003)
    problem = sampler.problem
    assert problem.H.shape[0] == sampler.dem.num_detectors == 432
    assert problem.L.shape[0] == sampler.dem.num_observables == 12
    assert problem.H.shape[1] == sampler.dem.num_errors
    support = np.asarray(problem.H.sum(0)+problem.L.sum(0)).ravel()
    assert (support > 0).all()
    rng = np.random.default_rng(973)
    counts = np.zeros(problem.H.shape[1], dtype=np.int64)
    for _ in range(20):
        s, o, e = sampler.sample(500, rng)
        assert s.dtype == o.dtype == e.dtype == np.uint8
        assert s.shape == (500, 432) and o.shape == (500, 12)
        np.testing.assert_array_equal(s, parity(problem.H, e))
        np.testing.assert_array_equal(o, parity(problem.L, e))
        counts += e.sum(0).astype(np.int64)
    # Simultaneous binomial bounds handle many rare columns, where Gaussian
    # per-column 5-sigma bounds would produce false failures by multiplicity.
    alpha = 0.01/len(counts)
    lo, hi = binom.interval(1-alpha, 10000, problem.priors)
    assert np.all((counts >= lo) & (counts <= hi))
    expected = 10000*problem.priors.sum()
    variance = 10000*(problem.priors*(1-problem.priors)).sum()
    assert abs(counts.sum()-expected) < 5*np.sqrt(variance)


def test_seeded_circuit_samples_and_noise_grid_structure():
    sampler = CircuitLevelSampler("bb72", 6, 0.001)
    first = sampler.sample(100, np.random.default_rng(91))
    second = sampler.sample(100, np.random.default_rng(91))
    other = sampler.sample(100, np.random.default_rng(92))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[2], other[2])
    high = CircuitLevelSampler("bb72", 6, 0.005)
    assert sampler.problem.H.shape == high.problem.H.shape
    assert (sampler.problem.H != high.problem.H).nnz == 0
    assert (sampler.problem.L != high.problem.L).nnz == 0
    assert np.all(high.problem.priors > sampler.problem.priors)
    assert len(sampler.problem.meta["detector_rounds"]) == sampler.dem.num_detectors


def test_noise_placement_and_total_round_count():
    p = 0.001
    circuit = build_memory_circuit("bb72", 6, p)
    flat = circuit.flattened()
    mrs = [i for i in flat if i.name == "MR"]
    assert len(mrs) == 6
    assert all(len(i.targets_copy()) == 72 for i in mrs)
    for i in flat:
        if i.name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}:
            assert i.gate_args_copy() == [p]
    instructions = list(flat)
    for index, inst in enumerate(instructions):
        if inst.name in {"H", "CX", "R", "RX"}:
            expected = {"H": "DEPOLARIZE1", "CX": "DEPOLARIZE2", "R": "X_ERROR", "RX": "Z_ERROR"}[inst.name]
            after = instructions[index+1]
            assert after.name == expected
            if inst.name == "H":
                # Stim coalesces the equal-p gate and idle DEPOLARIZE1 channels.
                assert set(inst.targets_copy()) <= set(after.targets_copy())
            else:
                assert after.targets_copy() == inst.targets_copy()
        if inst.name in {"M", "MX", "MR"}:
            before = instructions[index-1]
            assert before.name == ("Z_ERROR" if inst.name == "MX" else "X_ERROR")
            assert before.targets_copy() == inst.targets_copy()
            if inst.name == "MR":
                after = instructions[index+1]
                assert after.name == "X_ERROR" and after.targets_copy() == inst.targets_copy()
    # Final data measurements have noise even though QUITS defaults to noiseless.
    final_measure = max(i for i, inst in enumerate(instructions) if inst.name == "M")
    before = instructions[final_measure-1]
    assert before.name == "X_ERROR" and len(before.targets_copy()) == 72


def test_zero_noise_circuit_is_deterministic():
    circuit = build_memory_circuit("bb72", 6, 0.0)
    det, obs = circuit.compile_detector_sampler(seed=19).sample(1000, separate_observables=True)
    assert not det.any() and not obs.any()


@pytest.mark.slow
def test_very_low_noise_decoded_logicals():
    sampler = CircuitLevelSampler("bb72", 6, 1e-6)
    problem = sampler.problem
    decoder = TwoStageDecoder(LdpcBP(problem, 50, 0.625),
                               LdpcBPOSD(problem, 50, 0.625, "osd_cs", 7))
    rng = np.random.default_rng(1971)
    failures = 0
    for _ in range(20):
        s, o, _ = sampler.sample(500, rng)
        result = decoder.decode_batch(s)
        np.testing.assert_array_equal(parity(problem.H, result["e_hat"]), s)
        failures += int((parity(problem.L, result["e_hat"]) != o).any(1).sum())
    assert failures == 0


@pytest.mark.parametrize("code,rounds,basis", [("bb72", 2, "X"), ("bb144", 1, "Z")])
def test_other_circuit_basis_and_size(code, rounds, basis):
    sampler = CircuitLevelSampler(code, rounds, 0.001, basis)
    s, o, e = sampler.sample(100, np.random.default_rng(88))
    assert o.shape == (100, 12)
    np.testing.assert_array_equal(s, parity(sampler.problem.H, e))
    np.testing.assert_array_equal(o, parity(sampler.problem.L, e))
