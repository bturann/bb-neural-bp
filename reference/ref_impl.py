import numpy as np

# ---------- GF(2) linear algebra ----------
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

# ---------- BB codes (verified) ----------
def _S(n): return np.roll(np.eye(n, dtype=np.uint8), 1, axis=1)
def bb_code(l, m, a, b):
    x = np.kron(_S(l), np.eye(m, dtype=np.uint8)); y = np.kron(np.eye(l, dtype=np.uint8), _S(m))
    P = lambda t, k: np.linalg.matrix_power(x if t == 'x' else y, k)
    A = sum(P(*t) for t in a) % 2; B = sum(P(*t) for t in b) % 2
    return (np.hstack([A, B]) % 2).astype(np.uint8), (np.hstack([B.T, A.T]) % 2).astype(np.uint8)

POLY = ([('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)])
CODES = {"bb72": (6, 6, POLY), "bb144": (12, 6, POLY)}

# ---------- batched min-sum BP reference (numpy) ----------
def minsum_bp(H, s, llr, max_iter=50, alpha=0.625):
    """H: [C,N] uint8; s: [B,C]; llr: [N] prior LLRs. Returns (e_hat [B,N], converged [B])."""
    C, N = H.shape; Bsz = s.shape[0]
    ci, vi = np.nonzero(H)                      # edges
    E = ci.size
    m_cv = np.zeros((Bsz, E))
    sign_s = 1 - 2 * s[:, ci].astype(float)     # (-1)^{s_c} per edge
    e_hat = np.zeros((Bsz, N), dtype=np.uint8); conv = np.zeros(Bsz, bool)
    for _ in range(max_iter):
        tot = np.zeros((Bsz, N)); np.add.at(tot, (slice(None), vi), m_cv)
        post = llr + tot
        mu = post[:, vi] - m_cv                 # var->check extrinsic
        sg = np.where(mu < 0, -1.0, 1.0); mag = np.abs(mu)
        # per-check sign product & two smallest magnitudes
        logneg = (sg < 0).astype(int)
        par = np.zeros((Bsz, C), int); np.add.at(par, (slice(None), ci), logneg)
        sprod_c = 1 - 2 * (par % 2)
        big = 1e18
        min1 = np.full((Bsz, C), big); np.minimum.at(min1, (slice(None), ci), mag)
        is_min = mag <= min1[:, ci]
        mag2 = np.where(is_min, big, mag)
        min2 = np.full((Bsz, C), big); np.minimum.at(min2, (slice(None), ci), mag2)
        # handle ties: if two edges share min1, min2 should equal min1
        cnt = np.zeros((Bsz, C), int); np.add.at(cnt, (slice(None), ci), is_min.astype(int))
        min2 = np.where(cnt > 1, min1, min2)
        ext = np.where(is_min, min2[:, ci], min1[:, ci])
        m_cv = alpha * sign_s * sprod_c[:, ci] * sg * ext
        tot = np.zeros((Bsz, N)); np.add.at(tot, (slice(None), vi), m_cv)
        e_hat = ((llr + tot) < 0).astype(np.uint8)
        conv = ((e_hat @ H.T) % 2 == s).all(1)
        if conv.all(): break
    return e_hat, conv
