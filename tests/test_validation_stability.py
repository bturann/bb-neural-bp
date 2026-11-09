import runpy

import numpy as np
import pytest


def test_paired_bootstrap_preserves_pairs_and_equal_stratum_weights():
    helper=runpy.run_path("scripts/m6_validation_stability.py")["paired_bootstrap"]
    result=helper([np.ones(1,dtype=int),np.zeros(3,dtype=int)],np.random.default_rng(55),200,0.95)
    assert result["low"]==result["high"]==0.5
    assert result["fraction_selected_lower"]==1
    tied=helper([np.zeros(8,dtype=int)],np.random.default_rng(55),200,0.95)
    assert tied["low"]==tied["high"]==0
    assert tied["fraction_selected_lower"]==tied["fraction_last_lower"]==0


def test_disjoint_small_blocks_expose_opposite_rankings_and_ties():
    helper=runpy.run_path("scripts/m6_validation_stability.py")["block_preferences"]
    selected=[np.array([0,0,1,1,0,1],dtype=bool)]
    last=[np.array([1,1,0,0,0,1],dtype=bool)]
    assert [r["preference"] for r in helper(selected,last,2)]==["selected","last","tie"]
    with pytest.raises(ValueError,match="divide"):
        helper(selected,last,4)
