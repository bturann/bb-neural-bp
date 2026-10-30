import numpy as np

from ..codes import bb_matrices, logicals
from .base import DecodingProblem, parity, validate_shots


class CodeCapacitySampler:
    def __init__(self, code: str, p: float, basis: str = "Z"):
        if basis not in {"X", "Z"}:
            raise ValueError("basis must be X or Z")
        hx, hz = bb_matrices(code)
        lx, lz = logicals(code)
        h, logical = (hx, lx) if basis == "Z" else (hz, lz)
        self.problem = DecodingProblem(
            name=f"{code}-cc-{basis}-p{p:g}", H=h, L=logical,
            priors=np.full(h.shape[1], p, dtype=np.float64),
            meta=dict(code=code, noise_model="code_capacity", rounds=None, basis=basis,
                      p=float(p), n_detectors=h.shape[0]),
        )

    def sample(self, shots: int, rng: np.random.Generator):
        validate_shots(shots, rng)
        e = (rng.random((shots, self.problem.H.shape[1])) < self.problem.priors).astype(np.uint8)
        return parity(self.problem.H, e), parity(self.problem.L, e), e
