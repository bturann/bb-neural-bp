import numpy as np

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
        s = validate_syndromes(s, self.problem.H.shape[0])
        stage1_hat, converged = self.stage1.decode_batch(s)
        actual = (parity(self.problem.H, stage1_hat) == s).all(axis=1)
        if not np.array_equal(actual, converged):
            raise AssertionError("Stage-1 convergence flags are incorrect")
        out = stage1_hat.copy()
        used = ~converged
        executed = np.zeros(len(s), dtype=bool)
        for i in np.flatnonzero(used):
            out[i] = self.osd.decode_one(s[i])
            executed[i] = self.osd.last_osd_executed
        return dict(e_hat=out, stage1_e_hat=stage1_hat, osd_used=used,
                    stage1_converged=converged, osd_executed=executed)
