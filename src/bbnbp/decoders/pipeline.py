import numpy as np
import time

from ..noise.base import parity
from .base import validate_syndromes


class TwoStageDecoder:
    def __init__(self, stage1, osd):
        self.stage1, self.osd = stage1, osd
        self.problem = osd.problem
        self.name = f"{stage1.name}+osd"
        if hasattr(stage1, "problem"):
            other = stage1.problem
            if other.H.shape != self.problem.H.shape or (other.H != self.problem.H).nnz:
                raise ValueError("Both stages must decode the same parity-check matrix")
            if not np.array_equal(other.priors, self.problem.priors):
                raise ValueError("Both stages must use the same fault priors")

    def decode_batch(self, s):
        return self._decode(s)

    def decode_timed_one(self, s_row):
        return self._decode(np.asarray(s_row)[None, :], timed=True)

    def _decode(self, s, timed=False):
        s = validate_syndromes(s, self.problem.H.shape[0])
        start = time.perf_counter_ns() if timed else 0
        stage1_hat, converged = self.stage1.decode_batch(s)
        stage1_ns = time.perf_counter_ns()-start if timed else 0
        actual = (parity(self.problem.H, stage1_hat) == s).all(axis=1)
        if not np.array_equal(actual, converged):
            raise AssertionError("Stage-1 convergence flags are incorrect")
        out = stage1_hat.copy()
        used = ~converged
        executed = np.zeros(len(s), dtype=bool)
        osd_ns = 0
        for i in np.flatnonzero(used):
            start = time.perf_counter_ns() if timed else 0
            out[i] = self.osd.decode_one(s[i])
            if timed:
                osd_ns += time.perf_counter_ns()-start
            executed[i] = self.osd.last_osd_executed
        result = dict(e_hat=out, stage1_e_hat=stage1_hat, osd_used=used,
                      stage1_converged=converged, osd_executed=executed)
        if timed:
            result["stage1_ns"], result["osd_ns"] = stage1_ns, osd_ns
        return result
