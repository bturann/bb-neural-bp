import numpy as np
import pytest
import torch

from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.decoders.bp_ldpc import LdpcBPOSD
from bbnbp.decoders.pipeline import TwoStageDecoder
from bbnbp.noise.base import DecodingProblem, parity
from bbnbp.noise.code_capacity import CodeCapacitySampler
from bbnbp.noise.circuit_level import CircuitLevelSampler
from reference_trace import reference_trace


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("p", [0.03, 0.06])
def test_initialization_matches_reference_every_iteration(p):
    sampler = CodeCapacitySampler("bb72", p)
    s, _, _ = sampler.sample(500, np.random.default_rng(571))
    model = NeuralMinSum(sampler.problem, 15, "scalar", dtype=torch.float64)
    with torch.no_grad():
        probabilities = model(torch.from_numpy(s))
    expected = reference_trace(sampler.problem.H.toarray(), s, sampler.problem.llr, 15)
    assert len(probabilities) == len(expected) == 15
    for actual, wanted in zip(model.last_hard_decisions, expected):
        np.testing.assert_array_equal(actual.numpy(), wanted)


def test_dummy_zero_check_changes_nothing():
    sampler = CodeCapacitySampler("bb72", 0.06)
    problem = sampler.problem
    augmented = DecodingProblem("padded", np.vstack([problem.H.toarray(), np.zeros((1, 72), dtype=np.uint8)]),
                                 problem.L, problem.priors, problem.meta)
    a = NeuralMinSum(problem, 15, "scalar", dtype=torch.float64)
    b = NeuralMinSum(augmented, 15, "scalar", dtype=torch.float64)
    s, _, _ = sampler.sample(50, np.random.default_rng(22))
    with torch.no_grad():
        first = a(s)
        second = b(np.pad(s, ((0, 0), (0, 1))))
    for x, y in zip(first, second):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_degree_one_zero_messages_and_padding():
    problem = DecodingProblem("toy", [[1, 0, 0], [1, 1, 1], [0, 0, 0]], [[0, 0, 1]],
                               [0.1, 0.1, 0.1], {})
    s = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    model = NeuralMinSum(problem, 5, "scalar", dtype=torch.float64)
    with torch.no_grad():
        values = model(s, llr=torch.zeros(3, dtype=torch.float64))
    assert all(torch.isfinite(v).all() for v in values)
    expected = reference_trace(problem.H.toarray(), s, np.zeros(3), 5, msg_clip=20)
    for actual, wanted in zip(model.last_hard_decisions, expected):
        np.testing.assert_array_equal(actual.numpy(), wanted)


@pytest.mark.parametrize("sharing", ["scalar", "orbit", "full"])
def test_all_parameters_have_finite_nonzero_gradients(sharing):
    sampler = CodeCapacitySampler("bb72", 0.08)
    s, _, _ = sampler.sample(128, np.random.default_rng(988))
    model = NeuralMinSum(sampler.problem, 4, sharing)
    values = model(torch.from_numpy(s))
    coefficients = torch.linspace(0.2, 1.0, 72)
    loss = sum((p*coefficients).mean() for p in values)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert (parameter.grad != 0).all()


def test_first_convergence_freezes_each_sample():
    sampler = CodeCapacitySampler("bb72", 0.06)
    s, _, _ = sampler.sample(300, np.random.default_rng(93))
    s[0] = 0
    model = NeuralMinSum(sampler.problem, 20, "scalar", dtype=torch.float64)
    with torch.no_grad():
        model(s)
    histories = [h.numpy() for h in model.last_hard_decisions]
    expected = histories[-1].astype(np.uint8).copy()
    conv = ~(s.any(1))
    expected[conv] = 0
    iterations = np.where(conv, 0, 20)
    for t, hard in enumerate(histories):
        now = (parity(sampler.problem.H, hard.astype(np.uint8)) == s).all(1)
        first = now & ~conv
        expected[first] = hard[first]
        iterations[first] = t+1
        conv |= now
    actual, actual_conv = model.decode_batch(s)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_conv, conv)
    np.testing.assert_array_equal(model.last_iterations, iterations)
    assert model.last_iterations[0] == 0


@pytest.mark.slow
def test_compiled_cpu_matches_trained_like_weights_on_ten_thousand_shots():
    sampler = CodeCapacitySampler("bb72", 0.06)
    model = NeuralMinSum(sampler.problem, 15, "orbit", dtype=torch.float64)
    generator = torch.Generator().manual_seed(880)
    with torch.no_grad():
        model.w.copy_(0.3+0.7*torch.rand(model.w.shape, generator=generator, dtype=torch.float64))
        model.alpha.copy_(0.8+0.4*torch.rand(model.alpha.shape, generator=generator, dtype=torch.float64))
    compiled = model.export_cpu()
    rng = np.random.default_rng(673)
    for _ in range(20):
        s, _, _ = sampler.sample(500, rng)
        expected, conv = model.decode_batch(s)
        actual, actual_conv = compiled.decode_batch(s)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_conv, conv)
        np.testing.assert_array_equal(compiled.last_iterations, model.last_iterations)


@pytest.mark.parametrize("sharing", ["scalar", "full"])
def test_float32_cpu_export_and_snapshot(sharing):
    sampler = CodeCapacitySampler("bb72", 0.06)
    model = NeuralMinSum(sampler.problem, 10, sharing)
    generator = torch.Generator().manual_seed(33)
    with torch.no_grad():
        model.w.copy_(0.3+0.7*torch.rand(model.w.shape, generator=generator))
    s, _, _ = sampler.sample(1000, np.random.default_rng(31))
    expected, conv = model.decode_batch(s)
    compiled = model.export_cpu()
    actual, actual_conv = compiled.decode_batch(s)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_conv, conv)
    with torch.no_grad():
        model.w.fill_(0)
    unchanged, _ = compiled.decode_batch(s)
    np.testing.assert_array_equal(unchanged, actual)


def test_circuit_structural_model_and_cpu_export():
    sampler = CircuitLevelSampler("bb72", 2, 0.001)
    model = NeuralMinSum(sampler.problem, 4, "struct", dtype=torch.float64)
    s, _, _ = sampler.sample(8, np.random.default_rng(871))
    expected, conv = model.decode_batch(s)
    compiled = model.export_cpu()
    actual, actual_conv = compiled.decode_batch(s)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_conv, conv)


def test_six_cycle_float32_trained_like_structural_weights():
    sampler = CircuitLevelSampler("bb72", 6, 0.003)
    model = NeuralMinSum(sampler.problem, 15, "struct")
    generator = torch.Generator().manual_seed(731)
    with torch.no_grad():
        model.w.copy_(0.3+0.7*torch.rand(model.w.shape, generator=generator))
        model.alpha.copy_(0.8+0.4*torch.rand(model.alpha.shape, generator=generator))
    s, _, _ = sampler.sample(64, np.random.default_rng(734))
    expected, conv = model.decode_batch(s)
    compiled = model.export_cpu()
    actual, actual_conv = compiled.decode_batch(s)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_conv, conv)
    np.testing.assert_array_equal(compiled.last_iterations, model.last_iterations)


@pytest.mark.parametrize("compiled", [False, True])
def test_neural_decoder_in_two_stage_pipeline(compiled):
    sampler = CodeCapacitySampler("bb72", 0.06)
    model = NeuralMinSum(sampler.problem, 15, "orbit")
    stage1 = model.export_cpu() if compiled else model
    pipeline = TwoStageDecoder(stage1, LdpcBPOSD(sampler.problem, 50, 0.625, "osd_cs", 7))
    s, _, _ = sampler.sample(100, np.random.default_rng(321))
    result = pipeline.decode_batch(s)
    np.testing.assert_array_equal(parity(sampler.problem.H, result["e_hat"]), s)
    np.testing.assert_array_equal(result["osd_used"], ~result["stage1_converged"])


def test_empty_and_zero_syndrome_batches():
    sampler = CodeCapacitySampler("bb72", 0.06)
    model = NeuralMinSum(sampler.problem, 5, "scalar")
    out, conv = model.decode_batch(np.zeros((0, 36), dtype=np.uint8))
    assert out.shape == (0, 72) and conv.shape == (0,)
    out, conv = model.decode_batch(np.zeros((3, 36), dtype=np.uint8))
    assert not out.any() and conv.all() and not model.last_iterations.any()
