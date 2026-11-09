import runpy

import numpy as np

ROUTE=runpy.run_path("scripts/m6_cascade_tune.py")["route"]
FLAGS=runpy.run_path("scripts/m6_cascade_saved_screen.py")["routed_flags"]


def test_cascade_keeps_first_valid_output_even_if_wrong_logical():
    first=dict(corrections=np.array([[1,0],[0,0],[1,1]],dtype=np.uint8),
        failed=np.array([False,True,True]),logical_failed=np.array([True,True,True]),iterations=np.array([2,15,15]))
    second=dict(corrections=np.array([[0,1],[0,0]],dtype=np.uint8),failed=np.array([False,True]),
        logical_failed=np.array([False,True]),iterations=np.array([3,15]))
    result=ROUTE(first,second,first["failed"])
    np.testing.assert_array_equal(result["corrections"],[[1,0],[0,1],[0,0]])
    np.testing.assert_array_equal(result["logical_failed"],[True,False,True])
    np.testing.assert_array_equal(result["iterations"],[2,18,30])
    np.testing.assert_array_equal(first["corrections"],[[1,0],[0,0],[1,1]])
    full_second=dict(failed=np.array([False,False,True]),logical_failed=np.array([False,False,True]))
    flags=FLAGS(first,full_second)
    np.testing.assert_array_equal(flags["logical_failed"],result["logical_failed"])
    np.testing.assert_array_equal(flags["failed"],result["failed"])
