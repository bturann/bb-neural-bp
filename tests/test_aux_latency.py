import runpy

import numpy as np


def test_paired_bootstrap_retains_constant_casewise_difference():
    helper=runpy.run_path("scripts/m6_aux_latency.py")["paired_bootstrap"]
    a=np.array([1.,5.,10.,30.,40.])
    result=helper(a,a+7,np.random.default_rng(916),100)
    for metric in result.values():
        np.testing.assert_allclose(list(metric.values()),[7.,7.,7.],atol=1e-12)
