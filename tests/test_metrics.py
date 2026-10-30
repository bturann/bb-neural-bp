import numpy as np
import pytest

from bbnbp.evaluate import evaluate_p
from bbnbp.metrics import latency_stats, mcnemar, wilson
from bbnbp.noise.base import DecodingProblem


def test_wilson_known_values():
    assert wilson(0, 100) == pytest.approx((0.0, 0.0369935), abs=1e-6)
    assert wilson(100, 100) == pytest.approx((0.9630065, 1.0), abs=1e-6)
    assert wilson(50, 100) == pytest.approx((0.403832, 0.596168), abs=1e-6)


def test_latency_and_paired_counts():
    assert latency_stats([1, 2, 3, 4])["lat_p50_us"] == 2.5
    result = mcnemar([1, 1, 0, 0], [0, 1, 1, 0])
    assert result == dict(first_only=1, second_only=1, pvalue=1.0)
    assert mcnemar([1], [1])["pvalue"] == 1.0


def test_failure_count_stopping_keeps_pairing():
    class Sampler:
        problem = DecodingProblem("mock", [[1]], [[1]], [0.1], {})

        def sample(self, shots, rng):
            # Every second shot has a logical error for the all-zero decoder.
            e = (np.arange(shots) % 2).astype(np.uint8)[:, None]
            return e.copy(), e.copy(), e

    class Decoder:
        def decode_batch(self, s):
            self.last_iterations = np.zeros(len(s), dtype=int)
            return np.zeros_like(s), ~s[:, 0].astype(bool)

    a, b = Decoder(), Decoder()
    result = evaluate_p(Sampler(), {"a": (a, a, 1), "b": (b, b, 1)},
                        np.random.default_rng(1), stop_failures=5, max_shots=100, batch=4)
    assert result["a"]["shots"] == result["b"]["shots"] == 12
    assert result["a"]["logical"] == 6
    limited = evaluate_p(Sampler(), {"a": (a, a, 1)}, np.random.default_rng(1),
                         stop_failures=100, max_shots=7, batch=4)
    assert limited["a"]["shots"] == 7
