import numpy as np
import pytest

from ref_impl import minsum_bp
from bbnbp.decoders.bp_ldpc import LdpcBP, LdpcBPOSD
from bbnbp.decoders.pipeline import TwoStageDecoder
from bbnbp.noise.base import parity
from bbnbp.noise.code_capacity import CodeCapacitySampler


def test_bp_convergence_flag():
    sampler = CodeCapacitySampler("bb72", 0.06)
    s, _, _ = sampler.sample(1000, np.random.default_rng(2))
    bp = LdpcBP(sampler.problem, 50, 0.625)
    e_hat, conv = bp.decode_batch(s)
    np.testing.assert_array_equal(conv, (parity(sampler.problem.H, e_hat) == s).all(1))
    assert bp._checked == 1000
    assert np.all(bp.last_iterations <= 50)


@pytest.mark.slow
@pytest.mark.parametrize("p,expected", [(0.04, 0.087), (0.06, 0.214)])
def test_bp_vs_reference_and_sanity_band(p, expected):
    sampler = CodeCapacitySampler("bb72", p)
    s, _, _ = sampler.sample(5000, np.random.default_rng(2))
    _, actual = LdpcBP(sampler.problem, 50, 0.625).decode_batch(s)
    _, reference = minsum_bp(sampler.problem.H.toarray(), s, sampler.problem.llr,
                             max_iter=50, alpha=0.625)
    a, b = (~actual).mean(), (~reference).mean()
    pooled = (a+b)/2
    sigma = np.sqrt(2*pooled*(1-pooled)/len(s))
    assert abs(a-b) <= 3*sigma
    assert abs(a-expected) <= expected*0.30


def test_two_stage_and_reuse():
    sampler = CodeCapacitySampler("bb72", 0.08)
    problem = sampler.problem
    s, _, _ = sampler.sample(1000, np.random.default_rng(3))
    pipeline = TwoStageDecoder(LdpcBP(problem, 15, 0.625),
                               LdpcBPOSD(problem, 50, 0.625, "osd_cs", 7))
    result = pipeline.decode_batch(s)
    np.testing.assert_array_equal(parity(problem.H, result["e_hat"]), s)
    np.testing.assert_array_equal(result["osd_used"], ~result["stage1_converged"])
    assert np.all(~result["osd_executed"] | result["osd_used"])
    zero = pipeline.decode_batch(np.zeros((1, problem.H.shape[0]), dtype=np.uint8))
    assert not zero["e_hat"].any() and not zero["osd_used"].any()
    assert pipeline.stage1.last_iterations[0] == 0


def test_invalid_syndrome_rejected():
    sampler = CodeCapacitySampler("bb72", 0.05)
    bp = LdpcBP(sampler.problem, 15, 0.625)
    with pytest.raises(ValueError, match="syndromes"):
        bp.decode_batch(np.full((1, 36), 2))
