"""Replay conditional functional outputs or raw timing statistics for the selector."""

import argparse
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import yaml

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBPOSD
from bbnbp.decoders.pipeline import TwoStageDecoder
from bbnbp.metrics import latency_stats, mcnemar, wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--phase', required=True, choices=('accuracy', 'latency'))
    args = parser.parse_args()
    d, phase = args.directory, args.phase
    out = d/phase
    target = d/f'audit-{phase}.json'
    if target.exists():
        raise ValueError('Audit already exists')
    settings = yaml.safe_load((d/'followup-settings.yaml').read_text())
    c = load_config(d/'config.yaml')
    frozen = json.loads((d/'selection.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    assert summary['selection_sha256'] == hashlib.sha256((d/'selection.json').read_bytes()).hexdigest()
    assert json.loads((d/'confirmation/summary.json').read_text())['followup_accuracy_and_timing']
    module = runpy.run_path(str(d/'confirmation/executed_source.py'))
    totals = dict(checked_functional_model_cases=0, raw_timed_calls=0, uncached_replayed_calls=0)
    for i, p in enumerate(c.noise.p_grid):
        sampler = make_sampler(c, p)
        spec = c.eval.osd
        point = summary['points'][i]
        nodes = module['models'](d, sampler.problem, frozen, i)
        if phase == 'accuracy':
            n = settings['accuracy_shots'][i]
            with np.load(d/'confirmation'/f'cases-p{i}.npz') as saved:
                s, o = saved['syndromes'][:n], saved['observables'][:n]
            with np.load(out/f'fallback-p{i}.npz') as saved:
                indices, fallback = saved['indices'], saved['corrections']
            np.testing.assert_array_equal(parity(sampler.problem.H, fallback), s[indices])
            union = np.zeros(n, bool)
            location = np.full(n, -1, np.int64)
            location[indices] = np.arange(len(indices))
            errors = {}
            for name in settings['models']:
                with np.load(d/'confirmation'/f'values-p{i}-{name}.npz') as saved:
                    first, mask = saved['corrections'][:n], saved['failed'][:n]
                expected = first.copy()
                assert (location[mask] >= 0).all()
                expected[mask] = fallback[location[mask]]
                with np.load(out/f'values-p{i}-{name}.npz') as saved:
                    np.testing.assert_array_equal(saved['corrections'], expected)
                    np.testing.assert_array_equal(saved['fallback_used'], mask)
                    logical = (parity(sampler.problem.L, expected) != o).any(1)
                    np.testing.assert_array_equal(saved['logical_failed'], logical)
                np.testing.assert_array_equal(parity(sampler.problem.H, expected), s)
                item = point['outcomes'][name]
                assert item['logical_failures'] == int(logical.sum()) and item['fallback_requests'] == int(mask.sum())
                np.testing.assert_allclose(item['ler_wilson'], wilson(int(logical.sum()), n))
                errors[name] = logical
                union |= mask
                totals['checked_functional_model_cases'] += n
            np.testing.assert_array_equal(np.flatnonzero(union), indices)
            for name in settings['models']:
                if name != 'neural':
                    assert point['paired_nominal_tests'][name] == mcnemar(errors[name], errors['neural'])
            rng = np.random.default_rng(np.random.SeedSequence(settings['audit_entropy']+[i]))
            replay = rng.choice(len(indices), min(settings['audit_cases_per_point'], len(indices)), replace=False)
            decoder = LdpcBPOSD(sampler.problem, spec.bp_iters, spec.ms_scaling, spec.method, spec.order)
            for j in replay:
                np.testing.assert_array_equal(decoder.decode_one(s[indices[j]]), fallback[j])
                totals['uncached_replayed_calls'] += 1
        else:
            names = settings['models']
            with np.load(out/f'cases-p{i}.npz') as saved:
                s, o, e, order = (saved[key] for key in ('syndromes', 'observables', 'faults', 'order'))
            generated = sampler.sample(settings['timing_warmups']+settings['timing_shots'], np.random.default_rng(settings['timing_seeds'][i]))
            for actual, expected in zip((s, o, e), generated):
                np.testing.assert_array_equal(actual, expected)
            rng = np.random.default_rng(np.random.SeedSequence(settings['order_entropy']+[i]))
            np.testing.assert_array_equal(order, np.stack([rng.permutation(len(names)) for _ in s]))
            records = [json.loads(line) for line in (out/f'raw-p{i}.jsonl').read_text().splitlines()]
            assert len(records) == len(s)*len(names)
            for j, row in enumerate(records):
                case, position = divmod(j, len(names))
                assert row['case'] == case and row['order_position'] == position
                assert row['method'] == names[order[case, position]]
                assert row['warmup'] == (case < settings['timing_warmups']) and row['syndrome_valid']
                assert row['total_us'] >= row['stage1_us']+row['fallback_us'] >= 0
            totals['raw_timed_calls'] += len(records)
            arrays = {}
            for k, name in enumerate(names):
                kept = [row for row in records if row['method'] == name and not row['warmup']]
                arrays[name] = np.array([row['total_us'] for row in kept])
                assert len(kept) == settings['timing_shots']
                for key, value in latency_stats(arrays[name]).items():
                    np.testing.assert_allclose(point['outcomes'][name][key], value)
                assert point['outcomes'][name]['fallback_calls'] == sum(row['fallback_called'] for row in kept)
                assert point['outcomes'][name]['logical_failures'] == sum(row['logical_failed'] for row in kept)
                rng = np.random.default_rng(np.random.SeedSequence(settings['audit_entropy']+[i, k]))
                replay = rng.choice(len(s), settings['audit_cases_per_point'], replace=False)
                decoder = TwoStageDecoder(nodes[name], LdpcBPOSD(sampler.problem, spec.bp_iters, spec.ms_scaling, spec.method, spec.order))
                for case in replay:
                    result = decoder.decode_batch(s[case:case+1])
                    row = records[case*len(names)+int(np.flatnonzero(order[case] == k)[0])]
                    assert row['correction_sha256'] == hashlib.sha256(result['e_hat'].tobytes()).hexdigest()
                    assert row['fallback_called'] == bool(result['osd_used'][0])
                    assert row['internal_osd_executed'] == bool(result['osd_executed'][0])
                    assert row['logical_failed'] == bool((parity(sampler.problem.L, result['e_hat'])[0] != o[case]).any())
                    totals['uncached_replayed_calls'] += 1
            bootstrap = runpy.run_path(str(out/'bootstrap_source.py'))['paired_bootstrap']
            for k, name in enumerate(names):
                if name == 'neural':
                    continue
                actual = bootstrap(arrays[name], arrays['neural'], np.random.default_rng(np.random.SeedSequence(settings['bootstrap_entropy']+[i, k])), settings['bootstrap_repeats'])
                assert actual == point['paired_nominal_bootstrap'][name]
    result = dict(checks_passed=True, phase=phase, **totals,
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target.write_text(json.dumps(result, indent=2)+'\n')
    shutil.copyfile(__file__, d/f'executed_audit_{phase}_source.py')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
