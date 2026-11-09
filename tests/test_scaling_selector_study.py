"""Small actual-decoder CLI integration on code capacity, separate from research data."""

import json
import os
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.noise.factory import make_sampler


def test_selector_study_freezes_then_confirms_and_audits(tmp_path):
    config = load_config('configs/m6/cpu_cascade.yaml')
    config = replace(config, noise=replace(config.noise, model='code_capacity', rounds=None, circuit=None, p_grid=(.02, .04)))
    base = tmp_path/'base.yaml'
    config.save(base)
    settings = yaml.safe_load(Path('configs/m6/cpu_scaling_selector.yaml').read_text())
    settings.update(base_config=str(base), training_cases_per_point=128,
                    epochs=5, validate_every=5, confirmation_shots=16,
                    confirmation_seeds=[96234, 96235])
    sampler = make_sampler(config, .02)
    s, o, e = sampler.sample(512, np.random.default_rng(settings['train_seeds'][0]))
    pool = tmp_path/'pool.npz'
    np.savez_compressed(pool, syndromes=s, observables=o, faults=e)
    settings['unfiltered_pool'] = str(pool)
    opportunity = tmp_path/'opportunity'
    opportunity.mkdir()
    settings['opportunity'] = str(opportunity)
    validation, long_controls, strong = [], [], []
    decode = runpy.run_path('scripts/m6_cascade_tune.py')['decode']
    for i, p in enumerate(config.noise.p_grid):
        root = tmp_path/f'val{i}'
        root.mkdir()
        sampler = make_sampler(config, p)
        s, o, e = sampler.sample(64, np.random.default_rng(42234+i))
        np.savez_compressed(root/'cases.npz', syndromes=s, observables=o, faults=e)
        conv, correct, records = [], [], []
        for factor in config.sweep.factors:
            values = decode(LdpcBP(sampler.problem, 30, factor), s, o, sampler.problem, 64)
            conv.append(~values['failed'])
            correct.append(~values['logical_failed'])
            records.append(dict(scaling=factor, failures=int(values['failed'].sum())))
        best = min(records, key=lambda row: (row['failures'], row['scaling']))
        (root/'selection.json').write_text(json.dumps(dict(first_factor=1., selected=dict(longer=best, two_pass=dict(scaling=1.)))))
        np.savez_compressed(opportunity/f'longer-validation-labels-p{i}.npz',
                            factors=config.sweep.factors, converged=np.stack(conv, 1), correct=np.stack(correct, 1))
        long = tmp_path/f'long{i}'
        long.mkdir()
        (long/'selection.json').write_text(json.dumps(dict(choices={'100': dict(factor=1.)})))
        strong.append(dict(choices=dict(bp130=dict(factor=1.))))
        validation.append(str(root))
        long_controls.append(str(long))
    strong_root = tmp_path/'strong'
    strong_root.mkdir()
    (strong_root/'selection.json').write_text(json.dumps(dict(points=strong)))
    settings.update(validation_sources=validation, long_controls=long_controls, strong_study=str(strong_root))
    settings_path = tmp_path/'settings.yaml'
    settings_path.write_text(yaml.safe_dump(settings))
    runtime = tmp_path/'runtime.yaml'
    runtime.write_text(yaml.safe_dump(dict(workers=2, wall_seconds=120, audit_cases_per_factor=16, audit_entropy=[1234, 1103])))
    d = tmp_path/'study'
    env = dict(os.environ, OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    def run(script, *args):
        result = subprocess.run([sys.executable, f'scripts/{script}.py', *map(str, args)], env=env,
                                text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout+result.stderr
    for i in (0, 1):
        run('m6_selector_labels', '--settings', settings_path, '--output', d, '--index', i, '--prepare-only')
    run('m6_selector_label_resume', d, '--runtime', runtime)
    run('m6_audit_selector', d, '--phase', 'labels')
    run('m6_selector_train', d)
    assert not (d/'confirmation').exists()
    selection = json.loads((d/'selection.json').read_text())
    assert selection['confirmation_data_used'] is False
    run('m6_selector_confirm', d)
    run('m6_audit_selector', d, '--phase', 'confirmation')
    summary = json.loads((d/'confirmation/summary.json').read_text())
    assert [point['shots'] for point in summary['points']] == [16, 16]
    assert len(summary['primary_tests']) == 4
    assert json.loads((d/'audit-confirmation.json').read_text())['checks_passed']
