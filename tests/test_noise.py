import numpy as np
import pytest

from bbnbp.noise.base import DecodingProblem, parity
from bbnbp.noise.code_capacity import CodeCapacitySampler


@pytest.mark.parametrize("code,basis", [("bb72", "Z"), ("bb72", "X"), ("bb144", "Z")])
def test_code_capacity_sampling(code, basis):
    sampler = CodeCapacitySampler(code, 0.06, basis)
    problem = sampler.problem
    s, o, e = sampler.sample(1000, np.random.default_rng(81))
    assert s.shape == (1000, problem.H.shape[0])
    assert o.shape == (1000, 12) and e.shape == (1000, problem.H.shape[1])
    assert s.dtype == o.dtype == e.dtype == np.uint8
    np.testing.assert_array_equal(s, parity(problem.H, e))
    np.testing.assert_array_equal(o, parity(problem.L, e))
    sigma = np.sqrt(problem.priors*(1-problem.priors)/len(e))
    assert np.all(np.abs(e.mean(0)-problem.priors) < 5*sigma)


def test_sampling_explicit_seed_and_empty_batch():
    sampler = CodeCapacitySampler("bb72", 0.05)
    first = sampler.sample(100, np.random.default_rng(42))
    second = sampler.sample(100, np.random.default_rng(42))
    third = sampler.sample(100, np.random.default_rng(43))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[2], third[2])
    assert sampler.sample(0, np.random.default_rng(0))[2].shape == (0, 72)
    with pytest.raises(TypeError):
        sampler.sample(10, None)


@pytest.mark.parametrize("p", [0, 0.5, -0.1, np.nan])
def test_invalid_prior_rejected(p):
    with pytest.raises(ValueError, match="priors"):
        DecodingProblem("bad", np.eye(2), np.eye(2), np.full(2, p), {})
