"""Read-only planning from saved paired CPU timings and logical-error outcomes."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import beta, binom, norm

from bbnbp.io import source_manifest


def mean_sample_size(sd, reduction, alpha, power):
    if sd < 0 or not 0 < alpha < .5 or not .5 < power < 1:
        raise ValueError('Invalid design parameters')
    if reduction <= 0:
        return None
    return max(2, math.ceil(((norm.ppf(1-alpha)+norm.ppf(power))*sd/reduction)**2))


def quantile_ranks(n, q, tail):
    """One-based order statistics; 0/n+1 denote unbounded sample endpoints."""
    if n < 1 or not 0 < q < 1 or not 0 < tail < .5:
        raise ValueError('Invalid quantile design')
    r = int(binom.ppf(tail, n, q))
    t = int(binom.ppf(1-tail, n, q))+1
    coverage = float(binom.cdf(min(t-1, n), n, q)-binom.cdf(r-1, n, q))
    return dict(lower_rank=r, upper_rank=t, finite_upper=t <= n,
                binomial_coverage_lower_bound=coverage)


def paired_risk_upper(k_plus, k_minus, n, tail):
    """Union of exact marginal binomial bounds gives an upper risk-difference bound."""
    plus, minus = np.asarray(k_plus), np.asarray(k_minus)
    upper = np.where(plus == n, 1., beta.ppf(1-tail, plus+1, np.maximum(n-plus, 1)))
    lower = np.where(minus == 0, 0., beta.ppf(tail, np.maximum(minus, 1), n-minus+1))
    return upper-lower


def read_block(root, index):
    rows = [json.loads(line) for line in (root/f'raw-p{index}.jsonl').read_text().splitlines()]
    methods = sorted({row['method'] for row in rows})
    result = {}
    for method in methods:
        kept = sorted((row for row in rows if row['method'] == method and not row['warmup']), key=lambda row: row['case'])
        assert len(kept) == 256 and len({row['case'] for row in kept}) == 256
        assert all(row['syndrome_valid'] for row in kept)
        result[method] = dict(cases=np.array([row['case'] for row in kept]),
            ms=np.array([row['total_us'] for row in kept])/1000,
            logical=np.array([row['logical'] for row in kept], bool))
    return result


def run_analysis(settings):
    cycle = Path(settings['cycle100_study'])
    frozen = json.loads((cycle/'selection.json').read_text())
    accuracy = json.loads((Path(settings['accuracy'])/'summary.json').read_text())
    alpha = settings['latency_family_alpha']/settings['latency_family_size']
    qtail = alpha/2
    risk_tail = settings['accuracy_family_alpha']/settings['accuracy_family_size']/2
    blocks = {(block['name'], i): read_block(Path(block['directory']), i)
              for block in settings['timing_blocks'] for i in range(2)}
    contrasts, accuracy_sources, runtime_points = [], [], []
    for i, p in enumerate(settings['noise_points']):
        ordinary = frozen['points'][i]['best_ordinary']
        baseline_method = 'bp100' if ordinary == 'cpp100' else 'native100'
        runtime_means = {}
        for method, aliases in [('learned100', ['learned100']), ('ordinary100', [baseline_method]), ('bp130', ['ordinary'])]:
            values = []
            for block in settings['timing_blocks']:
                name = block['name']
                if method == 'bp130' and name != 'strong':
                    continue
                data = blocks[name, i]
                if aliases[0] in data:
                    values.append(float(data[aliases[0]]['ms'].mean()))
            runtime_means[method] = max(values)
        runtime_points.append(dict(p=p, ordinary_backend=ordinary, conservative_method_mean_ms=runtime_means,
                                   ms_per_three_method_case=sum(runtime_means.values())))
        for baseline, available in [('ordinary100', [b['name'] for b in settings['timing_blocks'] if baseline_method in blocks[b['name'], i]]),
                                    ('bp130', ['strong'])]:
            evidence = []
            for name in available:
                data = blocks[name, i]
                key = baseline_method if baseline == 'ordinary100' else 'ordinary'
                candidate, control = data['learned100'], data[key]
                np.testing.assert_array_equal(candidate['cases'], control['cases'])
                diff = candidate['ms']-control['ms']
                mean, sd = float(diff.mean()), float(diff.std(ddof=1))
                reduction = -mean
                ranks = quantile_ranks(len(diff), settings['quantile'], qtail)
                evidence.append(dict(block=name, cases=len(diff), candidate_mean_ms=float(candidate['ms'].mean()),
                    baseline_mean_ms=float(control['ms'].mean()), candidate_minus_baseline_ms=mean,
                    paired_sd_ms=sd, observed_effect_n80=mean_sample_size(sd, reduction, alpha, settings['target_power']),
                    empirical_p99_difference_ms=float(np.percentile(candidate['ms'], 99)-np.percentile(control['ms'], 99)),
                    exact_p99_ranks=ranks))
            max_sd = max(row['paired_sd_ms'] for row in evidence)
            base_mean = min(row['baseline_mean_ms'] for row in evidence)
            target = settings['design_relative_mean_reduction']*base_mean
            design_n = mean_sample_size(max_sd, target, alpha, settings['target_power'])
            powers = {str(n): float(norm.cdf(math.sqrt(n)*target/max_sd-norm.ppf(1-alpha)))
                      for n in settings['measured_cases_per_point']}
            contrasts.append(dict(p=p, baseline=baseline, evidence=evidence, worst_block_sd_ms=max_sd,
                                  five_percent_design_effect_ms=target, five_percent_design_n80=design_n,
                                  approximate_design_power=powers))
            if baseline == 'ordinary100':
                point = accuracy['points'][i]
                counts = point['pairs']['ordinary']
                kp, km, n = counts['selected_only_failures'], counts['baseline_only_failures'], point['shots']
                source = settings['accuracy']
            else:
                data = blocks['strong', i]
                a, b = data['learned100']['logical'], data['ordinary']['logical']
                kp, km, n = int((a & ~b).sum()), int((~a & b).sum()), len(a)
                source = next(block['directory'] for block in settings['timing_blocks'] if block['name'] == 'strong')
            discordance = (kp+km)/n
            upper_discordance = float(beta.ppf(.95, kp+km+1, n-kp-km)) if kp+km < n else 1.
            accuracy_sources.append(dict(p=p, baseline=baseline, source=source, cases=n,
                candidate_only_errors=kp, baseline_only_errors=km,
                observed_risk_difference=(kp-km)/n,
                simultaneous_risk_difference_upper=float(paired_risk_upper(kp, km, n, risk_tail)),
                scenarios=dict(pilot_point=[kp/n, km/n],
                    equal_risk_same_discordance=[discordance/2]*2,
                    equal_risk_upper95_discordance=[upper_discordance/2]*2)))
    runtime = []
    seconds_per_case = sum(point['ms_per_three_method_case'] for point in runtime_points)/1000
    for n in settings['measured_cases_per_point']:
        seconds = (n+settings['warmup_cases_per_point'])*seconds_per_case
        runtime.append(dict(cases_per_point=n, warmup_per_point=settings['warmup_cases_per_point'],
                            estimated_hours=seconds/3600, cushioned_hours=seconds*settings['runtime_cushion']/3600,
                            expected_tail_cases_per_method=n*(1-settings['quantile']),
                            quantile_ranks=quantile_ranks(n, settings['quantile'], qtail)))
    simulations = []
    for j, point in enumerate(accuracy_sources):
        for k, (scenario, rates) in enumerate(point['scenarios'].items()):
            for index, n in enumerate(settings['accuracy_case_grid']):
                rng = np.random.default_rng(np.random.SeedSequence(settings['simulation_entropy']+[j, k, index]))
                counts = rng.multinomial(n, [*rates, 1-sum(rates)], size=settings['accuracy_simulation_repetitions'])
                bounds = paired_risk_upper(counts[:, 0], counts[:, 1], n, risk_tail)
                for margin in settings['accuracy_margins_probability']:
                    chance = float((bounds < margin).mean())
                    simulations.append(dict(p=point['p'], baseline=point['baseline'], scenario=scenario, cases=n,
                        margin_probability=margin, probability_bound_passes=chance,
                        monte_carlo_se=math.sqrt(chance*(1-chance)/len(bounds))))
    return dict(runtime_points=runtime_points, runtime_options=runtime, mean_contrasts=contrasts,
                accuracy_sources=accuracy_sources, accuracy_scenario_simulations=simulations,
                mean_test_alpha=alpha, p99_marginal_tail_alpha=qtail, accuracy_marginal_tail_alpha=risk_tail,
                new_decoder_cases=0, new_training_updates=0, final_test_data_used=False, benchmark_launched=False,
                scope='Planning calculations and labeled hypothetical simulations from reused evidence; not independent confirmation or main M6 completion')


def write_report(out, result, settings):
    lines = ['# Frozen learned100: CPU evaluation planning', '',
        'No new training, decoding, timing or final-test data were collected. The selector/cascade branches remain closed. '
        'The next comparison keeps the existing learned100 weights and validation-frozen controls.', '',
        '## Runtime options', '',
        'Three methods per case: learned100, the already validation-selected ordinary100 implementation '
        '(CPP at p=.001, native at p=.003), and BP130. All use the original fallback. '
        'Each option includes 1,000 untimed warmup cases per noise point. Method means use the maximum across available historical blocks. '
        'The 25% cushion is an engineering allowance, not a confidence interval or a guarantee.', '',
        '| Measured cases / p | Estimated CPU hours | With 25% cushion | Expected observations beyond p99 / method |',
        '|---:|---:|---:|---:|']
    for row in result['runtime_options']:
        lines.append(f'| {row["cases_per_point"]:,} | {row["estimated_hours"]:.2f} | {row["cushioned_hours"]:.2f} | {row["expected_tail_cases_per_method"]:.0f} |')
    lines += ['', 'The original 20,000-measured/1,000-warmup E2 regime remains unchanged. A smaller option is an auxiliary decision study and cannot complete E2.', '',
        '## Mean-latency sensitivity', '',
        'Normal-approximation paired-mean planning uses pilot standard deviations. A 5% reduction is an explicit hypothetical design alternative, '
        'not an accepted practical threshold or a prediction of the true effect. The design uses the largest available within-block SD and '
        'the smallest available baseline mean. Between-block drift, pilot uncertainty and heavy tails can increase the required sample.', '',
        '| p | Baseline | 80% power n for hypothetical 5% gain | Observed pilot mean differences, ms |', '|---|---|---:|---|']
    for row in result['mean_contrasts']:
        diffs = '; '.join(f'{e["block"]}: {e["candidate_minus_baseline_ms"]:+.2f}' for e in row['evidence'])
        lines.append(f'| {row["p"]:g} | {row["baseline"]} | {row["five_percent_design_n80"]:,} | {diffs} |')
    lines += ['', 'A larger sample does not reverse a slower true model. The low-noise BP130 pilot favors BP130 on mean latency; '
        'no finite superiority sample size is supplied for that observed direction. Other apparent gains are provisional pilot estimates.', '',
        '## p99 cannot be promised from the pilot', '',
        'Use exact binomial order-statistic bounds for the candidate upper p99 and each baseline lower p99. '
        'At the planned simultaneous confidence level, 256 cases have no finite distribution-free upper p99 bound. '
        'Resampling that finite sample cannot reveal an unseen tail. The rank table below describes precision in probability rank, '
        'not milliseconds or power to detect a p99 gain. No reliable p99-power forecast is claimed.', '',
        '| n | Lower rank | Upper rank | Finite upper bound |', '|---:|---:|---:|---|']
    for n in [256]+settings['measured_cases_per_point']:
        row = quantile_ranks(n, settings['quantile'], result['p99_marginal_tail_alpha'])
        lines.append(f'| {n:,} | {row["lower_rank"]} | {row["upper_rank"]} | {row["finite_upper"]} |')
    lines += ['', 'These bounds follow the binomial/order-statistic construction in '
        '[NIST TN 2119, section 5.3](https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2119.pdf).', '',
        '## Logical accuracy requires a prospective tolerance', '',
        'No acceptable excess logical-error probability has been chosen. Therefore the proposed study reports risk-difference bounds '
        'and cannot declare equivalence or noninferiority merely from overlapping intervals.', '',
        'Let q+ be the probability that only learned100 fails and q- the probability that only the baseline fails. '
        'An upper bound on the excess LER is CP-upper(q+) minus CP-lower(q-), with the error budget split over four contrasts and both tails. '
        'The marginal bounds use the exact binomial method documented by '
        '[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html). '
        'This union bound is conservative and does not assume independence between paired outcomes.', '',
        '| p | Baseline | Pilot cases | Learned-only / baseline-only errors | Simultaneous excess-LER upper bound, pp |', '|---|---|---:|---:|---:|']
    for row in result['accuracy_sources']:
        lines.append(f'| {row["p"]:g} | {row["baseline"]} | {row["cases"]} | {row["candidate_only_errors"]}/{row["baseline_only_errors"]} | {100*row["simultaneous_risk_difference_upper"]:.4f} |')
    lines += ['', 'The simulation CSV evaluates specified tolerance scenarios under pilot rates, equal-risk rates, and an upper-discordance sensitivity case. '
        'These are hypothetical planning draws, not extra observed error events. A zero-discordance pilot is not assumed to establish zero risk. '
        'Monte Carlo error is recorded separately; uncertainty in the assumed rates is larger and is not removed by extra simulations.', '',
        '## Concrete prospective decision rules', '',
        '- Fixed checkpoint, iteration rule, noise points and control choices; no further tuning or case exclusion.',
        '- Draw a new stream only after the chosen sample size, budget and accuracy tolerance are frozen. Proposed seeds 100234/100235 are unused; main 11234–11238 stay reserved.',
        '- Time full pipelines uncached, batch1, single thread, randomized method order per case; finish other compute jobs first.',
        '- Four mean and four p99 contrasts share latency alpha=.05 using Bonferroni allocations. Mean upper bounds use paired standard errors; they are asymptotic. p99 upper-difference bounds use exact marginal ranks.',
        '- Four paired excess-LER upper bounds form a separate accuracy family. A safety claim requires a prospectively chosen tolerance; null is unresolved.',
        '- A full benefit requires accuracy within that tolerance plus both mean and p99 superiority against both controls at both rates. A partial result is reported as partial.',
        '- No outcome-dependent extension or early success. A wall-cap failure yields an incomplete study, not a smaller successful fixed sample.', '',
        'The ready proposal is 2,000 measured cases/p, 1,000 warmups/p and a 90-minute cap. This is a bounded auxiliary decision study, '
        'not a promise of decisive p99/accuracy evidence. The full E2 option is much longer. Neither benchmark has been launched. '
        'The original matrix, main final tests and completed evidence remain intact.', '',
        '## Reproduction', '', '```sh',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/m6_plan_cycle100.py', '```', '',
        'All input hashes, resolved settings, source, tables and labeled simulation results accompany this report.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default='configs/m6/cpu_cycle100_planning.yaml')
    parser.add_argument('--output-root', default='results/m6_cycle100_planning')
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.settings).read_text())
    assert settings['launch_benchmark'] is False and settings['prospective_accuracy_margin'] is None
    out = Path(args.output_root)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    (out/'settings.yaml').write_text(yaml.safe_dump(settings))
    shutil.copyfile(__file__, out/'executed_source.py')
    (out/'source_manifest.json').write_text(json.dumps(source_manifest(), indent=2)+'\n')
    paths = []
    for block in settings['timing_blocks']:
        root = Path(block['directory'])
        assert json.loads((root/'audit.json').read_text())['checks_passed']
        paths += [root/name for name in ('audit.json', 'summary.json', 'raw-p0.jsonl', 'raw-p1.jsonl')]
    root = Path(settings['accuracy'])
    assert json.loads((root/'audit.json').read_text())['checks_passed']
    paths += [root/'summary.json', root/'audit.json', Path(settings['cycle100_study'])/'selection.json',
              Path(settings['strong_study'])/'selection.json', Path(settings['cycle100_study'])/'selected-source.pt']
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (out/'input_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    try:
        result = run_analysis(settings)
        (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        for name, rows in [('runtime', result['runtime_options']), ('accuracy_scenarios', result['accuracy_scenario_simulations'])]:
            with (out/f'{name}.csv').open('w', newline='') as stream:
                flattened = [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in rows]
                writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
                writer.writeheader(); writer.writerows(flattened)
        write_report(out, result, settings)
        for path, expected in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
        print(out/'report.md')
    except BaseException as exc:
        (out/'FAILURE.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise


if __name__ == '__main__':
    main()
