import runpy

import numpy as np
from scipy.stats import binom, binomtest


module = runpy.run_path('scripts/m6_plan_cycle100.py')


def test_quantile_ranks_have_declared_binomial_coverage_and_unbounded_tail():
    for n in (1, 256, 1000, 20000):
        for q in (.5, .99):
            result = module['quantile_ranks'](n, q, .003125)
            r, t = result['lower_rank'], result['upper_rank']
            assert binom.cdf(r-1, n, q) <= .003125+1e-12
            assert binom.sf(t-1, n, q) <= .003125+1e-12
            assert result['binomial_coverage_lower_bound'] >= 1-.00625-1e-12
    assert not module['quantile_ranks'](256, .99, .003125)['finite_upper']


def test_paired_risk_bound_matches_exact_binomial_marginals_and_edges():
    for plus, minus, n in ((0, 0, 256), (2, 6, 8192), (9, 13, 1024), (10, 0, 10), (0, 10, 10)):
        upper = binomtest(plus, n, alternative='less').proportion_ci(confidence_level=.99375).high
        lower = binomtest(minus, n, alternative='greater').proportion_ci(confidence_level=.99375).low
        np.testing.assert_allclose(module['paired_risk_upper'](plus, minus, n, .00625), upper-lower, atol=1e-10)


def test_mean_planning_does_not_invent_power_for_slower_direction():
    size = module['mean_sample_size']
    assert size(200, -5, .00625, .8) is None
    assert size(200, 0, .00625, .8) is None
    assert size(200, 5, .00625, .8) > size(200, 10, .00625, .8)
