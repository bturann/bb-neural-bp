"""Bivariate bicycle matrices, logical bases, and translation edge orbits."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._construction import CODES, bb_code
from .gf2 import gf2_rank, logical_ops


def bb_matrices(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name not in CODES:
        raise ValueError(f"Unknown BB code {name!r}; choose from {tuple(CODES)}")
    l, m, poly = CODES[name]
    return bb_code(l, m, *poly)


def logicals(name: str) -> tuple[np.ndarray, np.ndarray]:
    hx, hz = bb_matrices(name)
    cache = Path(os.environ.get("BBNBP_CACHE_DIR", Path.home() / ".cache" / "bbnbp"))
    path = cache / f"{name}-logicals-v1.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if not (np.array_equal(saved["hx"], hx) and np.array_equal(saved["hz"], hz)):
                raise ValueError(f"Cached logicals at {path} use different parity checks")
            lx, lz = saved["lx"].copy(), saved["lz"].copy()
    else:
        lx, lz = logical_ops(hx, hz), logical_ops(hz, hx)
        _validate_logicals(hx, hz, lx, lz)
        cache.mkdir(parents=True, exist_ok=True)
        # Atomic replacement prevents partially written caches from being consumed.
        import tempfile
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            np.savez_compressed(temp_path, hx=hx, hz=hz, lx=lx, lz=lz)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    _validate_logicals(hx, hz, lx, lz)
    return lx, lz


def _validate_logicals(hx, hz, lx, lz):
    k = hx.shape[1] - gf2_rank(hx) - gf2_rank(hz)
    if lx.shape != (k, hx.shape[1]) or lz.shape != lx.shape:
        raise ValueError("Logical basis has incorrect dimensions")
    if np.any((hz @ lx.T) % 2) or np.any((hx @ lz.T) % 2):
        raise ValueError("Logical operators do not commute with opposite-basis checks")
    if gf2_rank((lx @ lz.T) % 2) != k:
        raise ValueError("Logical X/Z pairing is not full rank")


def shift_permutation(l: int, m: int, a: int, b: int) -> np.ndarray:
    if l <= 0 or m <= 0:
        raise ValueError("Shift dimensions must be positive")
    i, j = np.divmod(np.arange(l*m), m)
    block = ((i+a) % l)*m + (j+b) % m
    return np.concatenate([block, block+l*m])


def edge_orbits(H: np.ndarray, l: int, m: int) -> np.ndarray:
    h = np.asarray(H, dtype=np.uint8)
    if h.shape != (l*m, 2*l*m) or not np.isin(h, [0, 1]).all():
        raise ValueError("Expected a binary (l*m, 2*l*m) parity-check matrix")
    ci, vi = np.nonzero(h)
    edge_lookup = {(int(c), int(v)): e for e, (c, v) in enumerate(zip(ci, vi))}
    parent = np.arange(len(ci))

    def find(e):
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return int(e)

    for a in range(l):
        for b in range(m):
            pv = shift_permutation(l, m, a, b)
            pc = pv[:l*m]
            if not np.array_equal(h[np.ix_(pc, pv)], h):
                raise ValueError(f"Shift {(a, b)} does not preserve the check matrix")
            for e, (c, v) in enumerate(zip(ci, vi)):
                other = edge_lookup[(int(pc[c]), int(pv[v]))]
                first, second = find(e), find(other)
                parent[max(first, second)] = min(first, second)
    roots = np.array([find(e) for e in range(len(ci))], dtype=np.int64)
    return np.unique(roots, return_inverse=True)[1]
