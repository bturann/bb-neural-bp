"""Verified BB construction; shifts follow the handoff reference convention."""

import numpy as np

def _S(n): return np.roll(np.eye(n, dtype=np.uint8), 1, axis=1)


def bb_code(l, m, a, b):
    x = np.kron(_S(l), np.eye(m, dtype=np.uint8)); y = np.kron(np.eye(l, dtype=np.uint8), _S(m))
    P = lambda t, k: np.linalg.matrix_power(x if t == 'x' else y, k)
    A = sum(P(*t) for t in a) % 2; B = sum(P(*t) for t in b) % 2
    return (np.hstack([A, B]) % 2).astype(np.uint8), (np.hstack([B.T, A.T]) % 2).astype(np.uint8)


POLY = ([('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)])

CODES = {"bb72": (6, 6, POLY), "bb144": (12, 6, POLY)}

