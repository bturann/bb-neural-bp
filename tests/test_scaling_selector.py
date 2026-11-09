import runpy

import numpy as np
import pytest
import torch

from bbnbp.decoders.scaling_selector import DensePredictor, ScalingSelector, choose_factors, syndrome_features
from bbnbp.noise.base import DecodingProblem
from bbnbp.decoders.bp_ldpc import LdpcBP


def test_selector_routes_each_case_once_and_preserves_order(monkeypatch):
    calls = []
    class Decoder:
        def __init__(self, problem, iterations, factor):
            self.factor = factor
        def decode_batch(self, s):
            calls.extend((int(row[0]), self.factor) for row in s)
            self.last_iterations = np.full(len(s), int(self.factor*10))
            return s.copy(), np.ones(len(s), bool)
    class Predictor:
        def scores(self, s, p):
            assert p == .001
            return np.column_stack((1-s[:, 0], s[:, 0]))
    monkeypatch.setattr('bbnbp.decoders.scaling_selector.LdpcBP', Decoder)
    problem = DecodingProblem('toy', [[1]], [[1]], [.1], {})
    decoder = ScalingSelector(problem, .001, [.5, 1.], 0, 0., Predictor())
    s = np.array([[1], [0], [1]], np.uint8)
    hat, conv = decoder.decode_batch(s)
    np.testing.assert_array_equal(hat, s)
    np.testing.assert_array_equal(decoder.last_choices, [1, 0, 1])
    np.testing.assert_array_equal(decoder.last_iterations, [10, 5, 10])
    assert sorted(calls) == [(0, .5), (1, 1.), (1, 1.)]
    assert conv.all()


def test_margins_ties_and_invalid_features():
    scores = np.array([[.6, .6], [.6, .65], [.5, .8]])
    np.testing.assert_array_equal(choose_factors(scores, 0, .1), [0, 0, 1])
    s = np.array([[0, 1], [1, 1]], np.uint8)
    f = syndrome_features(s, .003)
    np.testing.assert_array_equal(f[:, :2], s)
    np.testing.assert_array_equal(f[:, 2:], [[.5, 1], [1, 1]])
    with pytest.raises(ValueError):
        syndrome_features([[2]], .003)
    with pytest.raises(ValueError):
        choose_factors([[np.nan, .1]], 0, 0)


def test_explicit_generator_and_numpy_export(tmp_path):
    Predictor = runpy.run_path('scripts/m6_selector_train.py')['Predictor']
    state = torch.random.get_rng_state().clone()
    first = Predictor([4, 8, 3, 2], torch.Generator().manual_seed(1234))
    second = Predictor([4, 8, 3, 2], torch.Generator().manual_seed(1234))
    assert torch.equal(torch.random.get_rng_state(), state)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
    s = np.array([[0, 1], [1, 0], [1, 1]], np.uint8)
    mean, scale = np.zeros(4, np.float32), np.ones(4, np.float32)
    path = tmp_path/'model.npz'
    first.save(path, mean, scale)
    expected = torch.sigmoid(first(torch.from_numpy(syndrome_features(s, .001)))).detach().numpy()
    np.testing.assert_allclose(DensePredictor(path).scores(s, .001), expected, atol=1e-7)


def test_actual_bp_matches_individual_selected_factor_and_empty_batch():
    problem = DecodingProblem('toy', [[1, 1, 0], [0, 1, 1]], [[1, 0, 0]], [.1, .12, .08], {})
    factors = [.5, .75, 1.]
    class Predictor:
        def scores(self, s, p):
            result = np.zeros((len(s), 3))
            result[np.arange(len(s)), s.sum(1).astype(int) % 3] = 1
            return result
    decoder = ScalingSelector(problem, .001, factors, 0, 0., Predictor())
    s = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.uint8)
    hat, conv = decoder.decode_batch(s)
    iterations = decoder.last_iterations.copy()
    for row, choice in enumerate(decoder.last_choices):
        reference = LdpcBP(problem, 30, factors[choice])
        expected, flags = reference.decode_batch(s[row:row+1])
        np.testing.assert_array_equal(hat[row:row+1], expected)
        assert conv[row] == flags[0] and iterations[row] == reference.last_iterations[0]
    empty, flags = decoder.decode_batch(s[:0])
    assert empty.shape == (0, 3) and flags.shape == (0,) and decoder.last_iterations.shape == (0,)
