import runpy

import numpy as np


def test_first_convergence_retains_early_success_and_iteration_zero():
    helper = runpy.run_path("scripts/m6_objective_diagnostic.py")["first_convergence"]
    history = np.array([[False, False, False, True],
                        [True, False, False, False],
                        [False, False, False, True]])
    actual = helper(history, [False, False, True, False])
    np.testing.assert_array_equal(actual, [2, 4, 0, 1])


def test_failure_auc_ties_direction_and_missing_class():
    helper = runpy.run_path("scripts/m6_objective_diagnostic.py")["failure_auc"]
    assert helper([1, 2, 3, 4], [False, False, True, True]) == 1
    assert helper([4, 3, 2, 1], [False, False, True, True]) == 0
    assert helper([1, 1, 1, 1], [False, False, True, True]) == 0.5
    assert helper([1, 2], [True, True]) is None
