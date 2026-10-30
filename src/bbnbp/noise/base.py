from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp


def binary_csr(matrix) -> sp.csr_matrix:
    out = sp.csr_matrix(matrix, copy=True)
    if not np.isin(out.data, [0, 1]).all():
        raise ValueError("Matrix must contain only binary entries")
    out = out.astype(np.uint8)
    out.sum_duplicates()
    out.data %= 2
    out.eliminate_zeros()
    out.sort_indices()
    return out


@dataclass(frozen=True)
class DecodingProblem:
    name: str
    H: sp.csr_matrix
    L: sp.csr_matrix
    priors: np.ndarray
    meta: dict

    def __post_init__(self):
        h, logical = binary_csr(self.H), binary_csr(self.L)
        p = np.array(self.priors, dtype=np.float64, copy=True)
        if logical.shape[1] != h.shape[1] or p.shape != (h.shape[1],):
            raise ValueError("H, L, and priors must agree on fault count")
        if not np.all(np.isfinite(p) & (p > 0) & (p < 0.5)):
            raise ValueError("All priors must be finite and in (0, 0.5)")
        p.setflags(write=False)
        object.__setattr__(self, "H", h)
        object.__setattr__(self, "L", logical)
        object.__setattr__(self, "priors", p)
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def llr(self):
        return np.log1p(-self.priors) - np.log(self.priors)


class Sampler(Protocol):
    problem: DecodingProblem

    def sample(self, shots: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return syndrome, observable flips, and faults as uint8 [shots, width]."""


def validate_shots(shots: int, rng: np.random.Generator):
    if type(shots) is not int or shots < 0:
        raise ValueError("shots must be a nonnegative integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("An explicit numpy.random.Generator is required")


def parity(matrix, faults):
    return np.asarray((matrix @ faults.T).T % 2, dtype=np.uint8)
