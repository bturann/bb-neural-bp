"""GF(2) reference routines copied from the supplied, verified implementation."""

import numpy as np

def gf2_rref(M):
    """Row-reduce a binary matrix. Returns (R, pivot_cols)."""
    R = (np.array(M, dtype=np.uint8) % 2).copy()
    rows, cols = R.shape
    piv = []; r = 0
    for c in range(cols):
        if r == rows: break
        nz = np.nonzero(R[r:, c])[0]
        if nz.size == 0: continue
        p = r + nz[0]
        if p != r: R[[r, p]] = R[[p, r]]
        others = np.nonzero(R[:, c])[0]; others = others[others != r]
        R[others] ^= R[r]
        piv.append(c); r += 1
    return R[:r], piv


def gf2_rank(M): return len(gf2_rref(M)[1])


def gf2_nullspace(M):
    """Basis (rows) of {x : M x = 0 mod 2}."""
    R, piv = gf2_rref(M); n = np.array(M).shape[1]
    free = [c for c in range(n) if c not in set(piv)]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8); v[f] = 1
        for i, pc in enumerate(piv):
            if R[i, f]: v[pc] = 1
        basis.append(v)
    return np.array(basis, dtype=np.uint8).reshape(-1, n)


def logical_ops(H_X, H_Z):
    """Rows of L_X: X-type logicals = ker(H_Z) independent of rowspace(H_X).
    L_X detects Z errors: Z-error e is a logical failure iff L_X @ (e + e_hat) != 0 mod 2
    when decoding Z errors with H_X. (Symmetric for the other basis.)"""
    ker = gf2_nullspace(H_Z)
    base = H_X.copy(); r0 = gf2_rank(base); out = []
    for v in ker:
        cand = np.vstack([base, v])
        if gf2_rank(cand) > r0:
            base = cand; r0 += 1; out.append(v)
    return np.array(out, dtype=np.uint8)


