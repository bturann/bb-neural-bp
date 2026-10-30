import numpy as np
from ldpc import BpDecoder, BpOsdDecoder

from ..noise.base import DecodingProblem, parity
from .base import validate_syndromes


def _kwargs(problem, T, alpha):
    if type(T) is not int or T < 1 or not 0 < alpha <= 1:
        raise ValueError("T must be positive and scaling must be in (0, 1]")
    return dict(error_channel=problem.priors.tolist(), max_iter=T, bp_method="minimum_sum",
                ms_scaling_factor=alpha, schedule="parallel", omp_thread_count=1,
                input_vector_type="syndrome", random_schedule_seed=0)


class LdpcBP:
    name = "bp"

    def __init__(self, problem: DecodingProblem, T: int, alpha: float):
        self.problem = problem
        self.T, self.alpha = T, alpha
        self.decoder = BpDecoder(problem.H, **_kwargs(problem, T, alpha))
        self._checked = 0
        self.last_iterations = np.empty(0, dtype=np.int64)

    def decode_batch(self, s):
        s = validate_syndromes(s, self.problem.H.shape[0])
        out = np.empty((len(s), self.problem.H.shape[1]), dtype=np.uint8)
        flags = np.empty(len(s), dtype=bool)
        self.last_iterations = np.empty(len(s), dtype=np.int64)
        for i, row in enumerate(s):
            out[i] = self.decoder.decode(row)
            flags[i] = self.decoder.converge
            # ldpc can retain a previous iteration count for a zero syndrome.
            self.last_iterations[i] = self.decoder.iter if row.any() else 0
        converged = (parity(self.problem.H, out) == s).all(axis=1)
        count = min(len(s), max(0, 1000-self._checked))
        if not np.array_equal(flags[:count], converged[:count]):
            raise AssertionError("ldpc converge attribute disagrees with the syndrome check")
        self._checked += count
        return out, converged


class LdpcBPOSD:
    name = "bp_osd"

    def __init__(self, problem: DecodingProblem, T: int, alpha: float, method: str, order: int):
        if method not in {"osd0", "osd_cs"} or type(order) is not int or order < 0:
            raise ValueError("Invalid OSD method or order")
        if method == "osd0" and order != 0:
            raise ValueError("osd0 requires order zero")
        self.problem = problem
        self.decoder = BpOsdDecoder(problem.H, **_kwargs(problem, T, alpha),
                                    osd_method=method, osd_order=order)
        self.last_osd_executed = False

    def decode_one(self, s_row):
        row = validate_syndromes(np.asarray(s_row)[None, :], self.problem.H.shape[0])[0]
        out = np.array(self.decoder.decode(row), dtype=np.uint8, copy=True)
        self.last_osd_executed = not bool(self.decoder.converge)
        if not np.array_equal(parity(self.problem.H, out[None, :])[0], row):
            raise RuntimeError("BP-OSD returned a correction that fails the syndrome check")
        return out
