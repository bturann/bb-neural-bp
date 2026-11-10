# Frozen learned100: CPU evaluation planning

No new training, decoding, timing or final-test data were collected. The selector/cascade branches remain closed. The next comparison keeps the existing learned100 weights and validation-frozen controls.

## Runtime options

Three methods per case: learned100, the already validation-selected ordinary100 implementation (CPP at p=.001, native at p=.003), and BP130. All use the original fallback. Each option includes 1,000 untimed warmup cases per noise point. Method means use the maximum across available historical blocks. The 25% cushion is an engineering allowance, not a confidence interval or a guarantee.

| Measured cases / p | Estimated CPU hours | With 25% cushion | Expected observations beyond p99 / method |
|---:|---:|---:|---:|
| 1,000 | 0.68 | 0.85 | 10 |
| 2,000 | 1.02 | 1.27 | 20 |
| 5,000 | 2.04 | 2.55 | 50 |
| 10,000 | 3.74 | 4.67 | 100 |
| 20,000 | 7.13 | 8.91 | 200 |

The original 20,000-measured/1,000-warmup E2 regime remains unchanged. A smaller option is an auxiliary decision study and cannot complete E2.

## Mean-latency sensitivity

Normal-approximation paired-mean planning uses pilot standard deviations. A 5% reduction is an explicit hypothetical design alternative, not an accepted practical threshold or a prediction of the true effect. The design uses the largest available within-block SD and the smallest available baseline mean. Between-block drift, pilot uncertainty and heavy tails can increase the required sample.

| p | Baseline | 80% power n for hypothetical 5% gain | Observed pilot mean differences, ms |
|---|---|---:|---|
| 0.001 | ordinary100 | 29,464 | longer: -8.39; strong: -14.60 |
| 0.001 | bp130 | 35,653 | strong: +4.71 |
| 0.003 | ordinary100 | 2,812 | longer: -42.63 |
| 0.003 | bp130 | 2,125 | strong: -11.17 |

A larger sample does not reverse a slower true model. The low-noise BP130 pilot favors BP130 on mean latency; no finite superiority sample size is supplied for that observed direction. Other apparent gains are provisional pilot estimates.

## p99 cannot be promised from the pilot

Use exact binomial order-statistic bounds for the candidate upper p99 and each baseline lower p99. At the planned simultaneous confidence level, 256 cases have no finite distribution-free upper p99 bound. Resampling that finite sample cannot reveal an unseen tail. The rank table below describes precision in probability rank, not milliseconds or power to detect a p99 gain. No reliable p99-power forecast is claimed.

| n | Lower rank | Upper rank | Finite upper bound |
|---:|---:|---:|---|
| 256 | 248 | 257 | False |
| 1,000 | 980 | 998 | True |
| 2,000 | 1967 | 1992 | True |
| 5,000 | 4930 | 4969 | True |
| 10,000 | 9872 | 9927 | True |
| 20,000 | 19760 | 19838 | True |

These bounds follow the binomial/order-statistic construction in [NIST TN 2119, section 5.3](https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2119.pdf).

## Logical accuracy requires a prospective tolerance

No acceptable excess logical-error probability has been chosen. Therefore the proposed study reports risk-difference bounds and cannot declare equivalence or noninferiority merely from overlapping intervals.

Let q+ be the probability that only learned100 fails and q- the probability that only the baseline fails. An upper bound on the excess LER is CP-upper(q+) minus CP-lower(q-), with the error budget split over four contrasts and both tails. The marginal bounds use the exact binomial method documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html). This union bound is conservative and does not assume independence between paired outcomes.

| p | Baseline | Pilot cases | Learned-only / baseline-only errors | Simultaneous excess-LER upper bound, pp |
|---|---|---:|---:|---:|
| 0.001 | ordinary100 | 8192 | 2/6 | 0.0901 |
| 0.001 | bp130 | 256 | 0/0 | 1.9630 |
| 0.003 | ordinary100 | 1024 | 9/13 | 1.3432 |
| 0.003 | bp130 | 256 | 3/1 | 4.1073 |

The simulation CSV evaluates specified tolerance scenarios under pilot rates, equal-risk rates, and an upper-discordance sensitivity case. These are hypothetical planning draws, not extra observed error events. A zero-discordance pilot is not assumed to establish zero risk. Monte Carlo error is recorded separately; uncertainty in the assumed rates is larger and is not removed by extra simulations.

## Concrete prospective decision rules

- Fixed checkpoint, iteration rule, noise points and control choices; no further tuning or case exclusion.
- Draw a new stream only after the chosen sample size, budget and accuracy tolerance are frozen. Proposed seeds 100234/100235 are unused; main 11234–11238 stay reserved.
- Time full pipelines uncached, batch1, single thread, randomized method order per case; finish other compute jobs first.
- Four mean and four p99 contrasts share latency alpha=.05 using Bonferroni allocations. Mean upper bounds use paired standard errors; they are asymptotic. p99 upper-difference bounds use exact marginal ranks.
- Four paired excess-LER upper bounds form a separate accuracy family. A safety claim requires a prospectively chosen tolerance; null is unresolved.
- A full benefit requires accuracy within that tolerance plus both mean and p99 superiority against both controls at both rates. A partial result is reported as partial.
- No outcome-dependent extension or early success. A wall-cap failure yields an incomplete study, not a smaller successful fixed sample.

The ready proposal is 2,000 measured cases/p, 1,000 warmups/p and a 90-minute cap. This is a bounded auxiliary decision study, not a promise of decisive p99/accuracy evidence. The full E2 option is much longer. Neither benchmark has been launched. The original matrix, main final tests and completed evidence remain intact.

## Reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/m6_plan_cycle100.py
```

All input hashes, resolved settings, source, tables and labeled simulation results accompany this report.
