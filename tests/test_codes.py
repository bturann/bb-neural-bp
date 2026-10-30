from pathlib import Path

import numpy as np
import pytest

from bbnbp.codes import bb_matrices, edge_orbits, logicals, shift_permutation
from bbnbp._construction import CODES
from bbnbp.gf2 import gf2_rank, gf2_nullspace


@pytest.mark.parametrize("name,n", [("bb72", 72), ("bb144", 144)])
def test_code_and_logical_parameters(name, n):
    hx, hz = bb_matrices(name)
    assert hx.shape == hz.shape == (n//2, n)
    assert hx.dtype == hz.dtype == np.uint8
    assert not ((hx @ hz.T) % 2).any()
    assert n - gf2_rank(hx) - gf2_rank(hz) == 12
    for h in (hx, hz):
        assert np.all(h.sum(1) == 6)
        assert np.all(h.sum(0) == 3)
    lx, lz = logicals(name)
    assert lx.shape == lz.shape == (12, n)
    assert not ((hz @ lx.T) % 2).any()
    assert not ((hx @ lz.T) % 2).any()
    assert gf2_rank((lx @ lz.T) % 2) == 12
    assert gf2_rank(np.vstack([hx, lx])) == gf2_rank(hx) + 12
    assert gf2_rank(np.vstack([hz, lz])) == gf2_rank(hz) + 12
    cached = logicals(name)
    for a, b in zip((lx, lz), cached):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", ["bb72", "bb144"])
def test_shifts_and_edge_orbits(name):
    l, m, _ = CODES[name]
    for h in bb_matrices(name):
        ci, vi = np.nonzero(h)
        orbits = edge_orbits(h, l, m)
        lookup = {(int(c), int(v)): e for e, (c, v) in enumerate(zip(ci, vi))}
        assert len(np.unique(orbits)) == 6
        assert np.all(np.bincount(orbits) == l*m)
        for a in range(l):
            for b in range(m):
                pv = shift_permutation(l, m, a, b)
                pc = pv[:l*m]
                np.testing.assert_array_equal(h[np.ix_(pc, pv)], h)
                perm_edges = [lookup[(int(pc[c]), int(pv[v]))] for c, v in zip(ci, vi)]
                np.testing.assert_array_equal(orbits, orbits[perm_edges])


def test_reference_files_unchanged():
    root = Path(__file__).resolve().parents[1]
    # When the original handoff is available, enforce byte identity.
    for name in ("ref_impl.py", "test_ref.py"):
        original = root.parent / "reference" / name
        if original.exists():
            assert (root / "reference" / name).read_bytes() == original.read_bytes()


def test_gf2_nullspace_rank_nullity():
    rng = np.random.default_rng(821)
    for shape in ((0, 5), (3, 7), (7, 3), (5, 5)):
        h = rng.integers(0, 2, size=shape, dtype=np.uint8)
        kernel = gf2_nullspace(h)
        assert kernel.shape == (shape[1] - gf2_rank(h), shape[1])
        assert not ((h @ kernel.T) % 2).any()


def test_reject_unknown_code_and_broken_symmetry():
    with pytest.raises(ValueError, match="Unknown"):
        bb_matrices("bb999")
    h, _ = bb_matrices("bb72")
    h[0, 0] ^= 1
    with pytest.raises(ValueError, match="does not preserve"):
        edge_orbits(h, 6, 6)
