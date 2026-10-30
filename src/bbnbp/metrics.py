import math

import numpy as np
from scipy.stats import binomtest


def wilson(k: int, n: int) -> tuple[float, float]:
    if not 0 <= k <= n or n < 1:
        raise ValueError("Require 0 <= k <= n and n > 0")
    z = 1.959963984540054
    p = k/n
    denominator = 1 + z*z/n
    midpoint = (p + z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return max(0.0, midpoint-half), min(1.0, midpoint+half)


def latency_stats(times_us):
    times = np.asarray(times_us, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times) & (times >= 0)):
        raise ValueError("Latency samples must be nonempty, finite, and nonnegative")
    return dict(lat_mean_us=float(times.mean()), lat_p50_us=float(np.percentile(times, 50)),
                lat_p99_us=float(np.percentile(times, 99)), lat_max_us=float(times.max()))


def mcnemar(first, second):
    a, b = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("Paired flags must be one-dimensional and have equal shapes")
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(first_only=n10, second_only=n01,
                pvalue=float(binomtest(n10, n10+n01).pvalue) if n10+n01 else 1.0)
