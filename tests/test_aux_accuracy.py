import runpy

import numpy as np
from scipy.stats import binom

HELPERS=runpy.run_path("scripts/m6_aux_accuracy.py")


def test_shared_fallback_composition_does_not_hide_valid_wrong_logical_outputs():
    failed=np.array([[False,True,True],[True,False,True]])
    stage1=np.array([[True,True,True],[True,False,True]])
    fallback=np.array([False,False,True])
    actual=HELPERS["compose_logicals"](failed,stage1,fallback)
    np.testing.assert_array_equal(actual,[[True,False,True],[False,False,True]])


def test_one_sided_regression_bound_matches_binomial_tail():
    upper=HELPERS["regression_upper"]
    np.testing.assert_allclose(upper(0,100,.025),1-.025**.01)
    assert upper(100,100,.025)==1.
    p=upper(3,200,.0125)
    np.testing.assert_allclose(binom.cdf(3,200,p),.0125,rtol=1e-10)
