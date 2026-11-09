"""Fresh fixed-target confirmation of the frozen auxiliary factor selector."""

import argparse
import hashlib
import inspect
import json
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.cascade import CascadeDecoder
from bbnbp.decoders.scaling_selector import CountPredictor, DensePredictor, ScalingSelector
from bbnbp.io import source_manifest
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def models(d, problem, frozen, index):
    if hashlib.sha256(Path(inspect.getfile(ScalingSelector)).read_bytes()).hexdigest() != hashlib.sha256((d/'selector_source.py').read_bytes()).hexdigest():
        raise ValueError('Selector implementation changed since selection')
    point = frozen['points'][index]
    result = {}
    for name, cls in (('neural', DensePredictor), ('count', CountPredictor)):
        spec = point[name]
        if hashlib.sha256((d/spec['file']).read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Frozen selector model changed')
        result[name] = ScalingSelector(problem, point['p'], frozen['factors'], point['fixed_index'],
                                       spec['margin'], cls(d/spec['file']), frozen['iterations'])
    result['fixed30'] = LdpcBP(problem, 30, point['fixed_factor'])
    controls = point['controls']
    result['two30'] = CascadeDecoder(LdpcBP(problem, 15, controls['two30']['first_factor']),
                                     LdpcBP(problem, 15, controls['two30']['second_factor']))
    for name, iterations in (('bp100', 100), ('bp130', 130)):
        result[name] = LdpcBP(problem, iterations, controls[name]['factor'])
    return result


def holm(values):
    order = np.argsort(values)
    result = np.empty(len(values))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(values)-rank)*values[index]))
        result[index] = previous
    return result.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    d = args.directory
    frozen = json.loads((d/'selection.json').read_text())
    settings = yaml.safe_load((d/'settings.yaml').read_text())
    c = load_config(d/'config.yaml')
    out = d/'confirmation'
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out/'executed_source.py')
    (out/'source_manifest.json').write_text(json.dumps(source_manifest(), indent=2)+'\n')
    for path, expected in frozen['input_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {path}')
    if hashlib.sha256(Path('src/bbnbp/decoders/scaling_selector.py').read_bytes()).hexdigest() != hashlib.sha256((d/'selector_source.py').read_bytes()).hexdigest():
        raise ValueError('Selector implementation changed since selection')
    (out/'pre_sampling.json').write_text(json.dumps(dict(selection_sha256=hashlib.sha256((d/'selection.json').read_bytes()).hexdigest(),
        started_at_utc=datetime.now(timezone.utc).isoformat(), seeds=frozen['confirmation_seeds'],
        selection_frozen_before_sampling=True), indent=2)+'\n')
    started = time.perf_counter()
    def timeout(signum, frame):
        raise TimeoutError('Selector confirmation reached its fixed CPU wall cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, settings['confirmation_wall_seconds'])
    try:
        points, tests = [], []
        for i, point in enumerate(frozen['points']):
            sampler = make_sampler(c, point['p'])
            seed = frozen['confirmation_seeds'][i]
            if seed in settings['train_seeds'] or seed in (42234, 42235) or 11234 <= seed <= 11238:
                raise ValueError('Confirmation overlaps training, validation or reserved final data')
            s, o, e = sampler.sample(frozen['confirmation_shots'], np.random.default_rng(seed))
            np.testing.assert_array_equal(parity(sampler.problem.H, e), s)
            np.testing.assert_array_equal(parity(sampler.problem.L, e), o)
            np.savez_compressed(out/f'cases-p{i}.npz', syndromes=s, observables=o, faults=e)
            outcomes, flags = {}, {}
            for name, decoder in models(d, sampler.problem, frozen, i).items():
                corrections, failed, logical, iterations, choices = [], [], [], [], []
                for start in range(0, len(s), c.eval.batch):
                    stop = min(len(s), start+c.eval.batch)
                    hat, conv = decoder.decode_batch(s[start:stop])
                    np.testing.assert_array_equal(conv, (parity(sampler.problem.H, hat) == s[start:stop]).all(1))
                    corrections.append(hat)
                    failed.append(~conv)
                    logical.append(~conv | (parity(sampler.problem.L, hat) != o[start:stop]).any(1))
                    iterations.append(decoder.last_iterations.copy())
                    if isinstance(decoder, ScalingSelector):
                        choices.append(decoder.last_choices.copy())
                values = dict(corrections=np.concatenate(corrections), failed=np.concatenate(failed),
                              logical_failed=np.concatenate(logical), iterations=np.concatenate(iterations))
                if choices:
                    values['choices'] = np.concatenate(choices)
                np.savez_compressed(out/f'values-p{i}-{name}.npz', **values)
                n, k = len(s), int(values['failed'].sum())
                outcomes[name] = dict(failures=k, fallback_rate=k/n, fallback_wilson=wilson(k, n),
                                      stage1_logical_failures=int(values['logical_failed'].sum()),
                                      mean_iterations=float(values['iterations'].mean()))
                flags[name] = values['failed']
                print(f'p={point["p"]:g} {name}: {k}/{n} failures; elapsed={time.perf_counter()-started:.1f}s', flush=True)
            for name in ('fixed30', 'count'):
                tests.append(dict(index=i, baseline=name, neural_failures=outcomes['neural']['failures'],
                                  baseline_failures=outcomes[name]['failures'], **mcnemar(flags[name], flags['neural'])))
            points.append(dict(p=point['p'], seed=seed, shots=len(s), outcomes=outcomes,
                               cases_sha256=hashlib.sha256((out/f'cases-p{i}.npz').read_bytes()).hexdigest()))
        for test, adjusted in zip(tests, holm([test['pvalue'] for test in tests])):
            test['holm_pvalue'] = adjusted
            test['clear_gain'] = test['neural_failures'] < test['baseline_failures'] and adjusted < settings['primary_alpha']
        # Gate is predeclared; do not change it after viewing fresh results.
        followup = all(point['outcomes']['neural']['failures'] < point['outcomes']['fixed30']['failures'] for point in points)
        result = dict(points=points, primary_tests=tests, followup_accuracy_and_timing=followup,
                      selection_sha256=hashlib.sha256((d/'selection.json').read_bytes()).hexdigest(),
                      elapsed_s=time.perf_counter()-started, final_test_data_used=False, m6_complete=False,
                      scope='Auxiliary fixed-target Stage1 confirmation; no full-pipeline LER or latency claim')
        (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        (out/'FAILURE.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
