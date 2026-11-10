"""Verify planning arithmetic against raw data and reproduce hypothetical simulations."""

import argparse
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binom, binomtest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    d = args.directory
    if (d/'audit.json').exists():
        raise ValueError('Audit already exists')
    settings = yaml.safe_load((d/'settings.yaml').read_text())
    result = json.loads((d/'analysis.json').read_text())
    hashes = json.loads((d/'input_hashes.json').read_text())
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    assert (d/'decision-plan.yaml').read_bytes() == Path(settings['decision_plan']).read_bytes()
    independently_derived_means = []
    for i in range(2):
        methods = {'learned100': [], 'ordinary100': [], 'bp130': []}
        for block in settings['timing_blocks']:
            rows = [json.loads(line) for line in (Path(block['directory'])/f'raw-p{i}.jsonl').read_text().splitlines()]
            mapping = dict(learned100='learned100', ordinary100='bp100' if i == 0 else 'native100')
            if block['name'] == 'strong':
                mapping['bp130'] = 'ordinary'
            for alias, method in mapping.items():
                values = [row['total_us']/1000 for row in rows if not row['warmup'] and row['method'] == method]
                if values:
                    assert len(values) == 256
                    methods[alias].append(sum(values)/len(values))
        expected = {method: max(values) for method, values in methods.items()}
        for key, value in expected.items():
            np.testing.assert_allclose(result['runtime_points'][i]['conservative_method_mean_ms'][key], value)
        independently_derived_means.append(sum(expected.values())/1000)
    for row in result['runtime_options']:
        hours = (row['cases_per_point']+settings['warmup_cases_per_point'])*sum(independently_derived_means)/3600
        np.testing.assert_allclose(row['estimated_hours'], hours)
        np.testing.assert_allclose(row['cushioned_hours'], hours*settings['runtime_cushion'])
        ranks = row['quantile_ranks']
        n, q = row['cases_per_point'], settings['quantile']
        tail = result['p99_marginal_tail_alpha']
        assert binom.cdf(ranks['lower_rank']-1, n, q) <= tail+1e-12
        assert binom.sf(ranks['upper_rank']-1, n, q) <= tail+1e-12
    for point in result['accuracy_sources']:
        level = 1-result['accuracy_marginal_tail_alpha']
        upper = binomtest(point['candidate_only_errors'], point['cases'], alternative='less').proportion_ci(confidence_level=level).high
        lower = binomtest(point['baseline_only_errors'], point['cases'], alternative='greater').proportion_ci(confidence_level=level).low
        np.testing.assert_allclose(point['simultaneous_risk_difference_upper'], upper-lower, atol=1e-10)
    module = runpy.run_path(str(d/'executed_source.py'))
    regenerated = module['run_analysis'](settings)
    assert regenerated == result
    assert result['benchmark_launched'] is False and result['new_decoder_cases'] == result['new_training_updates'] == 0
    for grid in result['mean_contrasts']:
        if grid['p'] == .001 and grid['baseline'] == 'bp130':
            assert grid['evidence'][0]['observed_effect_n80'] is None
    audit = dict(checks_passed=True, input_hashes_verified=len(hashes), runtime_options_verified=len(result['runtime_options']),
                 exact_accuracy_bounds_verified=len(result['accuracy_sources']),
                 hypothetical_simulation_rows_reproduced=len(result['accuracy_scenario_simulations']),
                 all_analysis_fields_reproduced=True, new_decoding_or_training=False,
                 audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    shutil.copyfile(__file__, d/'executed_audit_source.py')
    (d/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    paths = [d/name for name in ('analysis.json', 'settings.yaml', 'decision-plan.yaml', 'report.md',
                                'runtime.csv', 'accuracy_scenarios.csv', 'executed_source.py', 'audit.json')]
    (d/'output_hashes.json').write_text(json.dumps({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
