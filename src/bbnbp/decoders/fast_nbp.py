"""Numba CPU inference with the same weights, clipping, and stopping as PyTorch."""

import numpy as np
from numba import njit

from .base import validate_syndromes


@njit(cache=True)
def _decode(indptr, indices, n, syndromes, llr, weights, alpha, edge_map, clip):
    shots = len(syndromes)
    out = np.zeros((shots, n), dtype=np.uint8)
    converged = np.zeros(shots, dtype=np.bool_)
    iterations = np.full(shots, len(alpha), dtype=np.int64)
    for b in range(shots):
        if not np.any(syndromes[b]):
            converged[b] = True
            iterations[b] = 0
            continue
        messages = np.zeros(len(indices), dtype=llr.dtype)
        total = np.zeros(n, dtype=llr.dtype)
        for t in range(len(alpha)):
            total[:] = 0
            for edge in range(len(indices)):
                total[indices[edge]] += messages[edge]
            for c in range(len(indptr)-1):
                start, end = indptr[c], indptr[c+1]
                sign_product = 1
                minimum, second = 1e18, 1e18
                for edge in range(start, end):
                    v = indices[edge]
                    mu = (alpha[t]*llr[v]+total[v])-messages[edge]
                    if mu < 0:
                        sign_product = -sign_product
                    mag = abs(mu)
                    if mag < minimum:
                        second, minimum = minimum, mag
                    elif mag < second:
                        second = mag
                syndrome_sign = 1-2*int(syndromes[b, c])
                for edge in range(start, end):
                    v = indices[edge]
                    mu = (alpha[t]*llr[v]+total[v])-messages[edge]
                    sign = -1 if mu < 0 else 1
                    magnitude = second if abs(mu) == minimum else minimum
                    message = weights[t, edge_map[edge]]*syndrome_sign*sign_product*sign*magnitude
                    messages[edge] = min(clip, max(-clip, message))
            total[:] = 0
            for edge in range(len(indices)):
                total[indices[edge]] += messages[edge]
            for v in range(n):
                out[b, v] = llr[v]+total[v] < 0
            matches = True
            for c in range(len(indptr)-1):
                parity = 0
                for edge in range(indptr[c], indptr[c+1]):
                    parity ^= int(out[b, indices[edge]])
                if parity != syndromes[b, c]:
                    matches = False
                    break
            if matches:
                converged[b] = True
                iterations[b] = t+1
                break
    return out, converged, iterations


class FastNeuralBP:
    name = "nbp"

    def __init__(self, model):
        self.problem = model.problem
        self.T = model.T
        self.msg_clip = model.msg_clip
        self.weights = np.ascontiguousarray(model.w.detach().cpu().numpy())
        self.alpha = np.ascontiguousarray(model.alpha.detach().cpu().numpy())
        self.llr = np.array(model.llr.detach().cpu().numpy(), dtype=self.weights.dtype, order="C", copy=True)
        self.edge_map = np.ascontiguousarray(model.w_map[model.chk_mask].detach().cpu().numpy())
        if not np.isfinite(self.weights).all() or not np.isfinite(self.alpha).all():
            raise ValueError("Cannot export non-finite decoder parameters")
        # Export is a snapshot, independent of subsequent model parameter updates.
        self.weights = self.weights.copy()
        self.alpha = self.alpha.copy()
        self.last_iterations = np.empty(0, dtype=np.int64)

    def decode_batch(self, s):
        s = validate_syndromes(s, self.problem.H.shape[0])
        out, converged, self.last_iterations = _decode(
            self.problem.H.indptr, self.problem.H.indices, self.problem.H.shape[1],
            s, self.llr, self.weights, self.alpha, self.edge_map, self.msg_clip)
        return out, converged
