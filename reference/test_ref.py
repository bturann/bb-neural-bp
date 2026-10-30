import numpy as np
from ref_impl import *

def test_bb_params():
    for name, k_exp in [("bb72", 12), ("bb144", 12)]:
        l, m, poly = CODES[name]; HX, HZ = bb_code(l, m, *poly)
        n = HX.shape[1]
        assert not ((HX.astype(int) @ HZ.T.astype(int)) % 2).any()
        assert n - gf2_rank(HX) - gf2_rank(HZ) == k_exp
        assert set(HX.sum(1)) == {6} and set(HZ.sum(1)) == {6}

def test_logicals():
    HX, HZ = bb_code(*CODES["bb72"][:2], *CODES["bb72"][2])
    LX = logical_ops(HX, HZ); LZ = logical_ops(HZ, HX)
    assert LX.shape == (12, 72) and LZ.shape == (12, 72)
    assert not ((HZ.astype(int) @ LX.T) % 2).any()
    assert gf2_rank(((LX.astype(int) @ LZ.T) % 2)) == 12   # symplectic pairing full rank

def test_bp_low_noise():
    rng = np.random.default_rng(1)
    HX, HZ = bb_code(*CODES["bb72"][:2], *CODES["bb72"][2]); LX = logical_ops(HX, HZ)
    p = 0.01; B = 2000
    e = (rng.random((B, 72)) < p).astype(np.uint8); s = (e @ HX.T) % 2
    llr = np.full(72, np.log((1 - p) / p))
    eh, conv = minsum_bp(HX, s, llr, max_iter=50)
    fail = (((e ^ eh).astype(int) @ LX.T) % 2).any(1) | ~conv
    print("conv rate", conv.mean(), "fail rate", fail.mean())
    assert conv.mean() > 0.9
