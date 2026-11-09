"""Regenerate selector inputs and replay actual labels, routing and fresh comparisons."""

import argparse
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.scaling_selector import CountPredictor, DensePredictor, choose_factors, syndrome_features
from bbnbp.metrics import mcnemar, wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--phase', choices=('labels', 'confirmation'), required=True)
    args = parser.parse_args()
    d = args.directory
    target = d/f'audit-{args.phase}.json'
    if target.exists():
        raise ValueError('Audit already exists')
    settings = yaml.safe_load((d/'p0/settings.yaml').read_text())
    runtime = yaml.safe_load((d/'label_runtime.yaml').read_text())
    c = load_config(d/'p0/config.yaml')
    torch.set_num_threads(1)
    counts = dict(regenerated_training_cases=0, replayed_training_factor_outcomes=0,
                  regenerated_confirmation_cases=0, replayed_confirmation_model_cases=0,
                  checked_validation_choices=0, reconstructed_validation_candidates=0,
                  torch_numpy_and_single_batch_choice_cases=0)
    if args.phase == 'labels':
        for i, p in enumerate(c.noise.p_grid):
            root = d/f'p{i}'
            receipt = json.loads((root/'summary.json').read_text())
            assert sha(root/'cases.npz') == receipt['cases_sha256']
            assert sha(root/'labels.npz') == receipt['labels_sha256']
            sampler = make_sampler(c, p)
            with np.load(root/'cases.npz') as saved:
                cases = {key: saved[key] for key in saved.files}
            rng = np.random.default_rng(settings['train_seeds'][i])
            n = len(cases['syndromes'])
            for start in range(0, n, settings['pool_batch']):
                # Point 0 is a prefix of the original full-512 mining sampler calls.
                batch = settings['pool_batch'] if i == 0 else min(settings['pool_batch'], n-start)
                generated = sampler.sample(batch, rng)
                take = min(batch, n-start)
                for key, expected in zip(('syndromes', 'observables', 'faults'), generated):
                    np.testing.assert_array_equal(cases[key][start:start+take], expected[:take])
                counts['regenerated_training_cases'] += take
            np.testing.assert_array_equal(parity(sampler.problem.H, cases['faults']), cases['syndromes'])
            np.testing.assert_array_equal(parity(sampler.problem.L, cases['faults']), cases['observables'])
            with np.load(root/'labels.npz') as saved:
                labels = {key: saved[key] for key in saved.files}
            for j, factor in enumerate(labels['factors']):
                record = receipt['completed'][j]
                assert record['factor'] == factor and sha(root/record['file']) == record['sha256']
                with np.load(root/record['file']) as saved:
                    for key in ('converged', 'correct', 'iterations'):
                        np.testing.assert_array_equal(saved[key], labels[key][:, j])
                rng = np.random.default_rng(np.random.SeedSequence(runtime['audit_entropy']+[i, j]))
                indices = np.concatenate(([0, n-1], rng.choice(n-2, runtime['audit_cases_per_factor']-2, replace=False)+1))
                decoder = LdpcBP(sampler.problem, settings['iterations'], float(factor))
                hat, conv = decoder.decode_batch(cases['syndromes'][indices])
                np.testing.assert_array_equal(conv, (parity(sampler.problem.H, hat) == cases['syndromes'][indices]).all(1))
                correct = conv & (parity(sampler.problem.L, hat) == cases['observables'][indices]).all(1)
                np.testing.assert_array_equal(labels['converged'][indices, j], conv)
                np.testing.assert_array_equal(labels['correct'][indices, j], correct)
                np.testing.assert_array_equal(labels['iterations'][indices, j], decoder.last_iterations)
                counts['replayed_training_factor_outcomes'] += len(indices)
            assert not (labels['correct'] & ~labels['converged']).any()
            print(f'p{i} regenerated {n} training cases and replayed all factors', flush=True)
    else:
        assert json.loads((d/'audit-labels.json').read_text())['checks_passed']
        frozen = json.loads((d/'selection.json').read_text())
        summary = json.loads((d/'confirmation/summary.json').read_text())
        pre = json.loads((d/'confirmation/pre_sampling.json').read_text())
        assert summary['selection_sha256'] == pre['selection_sha256'] == sha(d/'selection.json')
        assert frozen['frozen_at_utc'] < pre['started_at_utc']
        for path, expected in frozen['input_hashes'].items():
            assert sha(path) == expected, path
        module = runpy.run_path(str(d/'confirmation/executed_source.py'))
        training_module = runpy.run_path(str(d/'executed_train_source.py'))
        grid = json.loads((d/'validation_grid.json').read_text())
        tests = []
        for i, point in enumerate(frozen['points']):
            sampler = make_sampler(c, point['p'])
            with np.load(d/'confirmation'/f'cases-p{i}.npz') as saved:
                cases = {key: saved[key] for key in saved.files}
            generated = sampler.sample(frozen['confirmation_shots'], np.random.default_rng(frozen['confirmation_seeds'][i]))
            for key, expected in zip(('syndromes', 'observables', 'faults'), generated):
                np.testing.assert_array_equal(cases[key], expected)
            counts['regenerated_confirmation_cases'] += len(generated[0])
            source = Path(settings['validation_sources'][i])
            with np.load(source/'cases.npz') as saved:
                vs = saved['syndromes']
            with np.load(Path(settings['opportunity'])/f'longer-validation-labels-p{i}.npz') as saved:
                vl = {key: saved[key] for key in saved.files}
            # Reconstruct every validation option and the deterministic selection rule.
            for filename in sorted({row['file'] for row in grid if row['index'] == i}):
                scores = DensePredictor(d/filename).scores(vs, point['p'])
                for row in [row for row in grid if row['index'] == i and row['file'] == filename]:
                    actual = training_module['outcome'](scores, vl, point['fixed_index'], row['margin'])
                    for key, value in actual.items():
                        assert row[key] == value
                    counts['reconstructed_validation_candidates'] += 1
            best = min((row for row in grid if row['index'] == i),
                       key=lambda row: (row['logical_failures'], row['failures'], row['arch_index'], row['epoch'], -row['margin']))
            for key, value in best.items():
                assert point['neural'][key] == value
            count_scores = CountPredictor(d/point['count']['file']).scores(vs, point['p'])
            count_grid = [training_module['outcome'](count_scores, vl, point['fixed_index'], margin) for margin in settings['margins']]
            assert count_grid == point['count_grid']
            best_count = min(count_grid, key=lambda row: (row['logical_failures'], row['failures'], -row['margin']))
            for key, value in best_count.items():
                assert point['count'][key] == value
            flags = {}
            for name, decoder in module['models'](d, sampler.problem, frozen, i).items():
                if name in ('neural', 'count'):
                    choice = decoder.choices(vs)
                    rows = np.arange(len(vs))
                    assert int((~vl['converged'][rows, choice]).sum()) == point[name]['failures']
                    assert int((~vl['correct'][rows, choice]).sum()) == point[name]['logical_failures']
                    counts['checked_validation_choices'] += len(vs)
                with np.load(d/'confirmation'/f'values-p{i}-{name}.npz') as saved:
                    values = {key: saved[key] for key in saved.files}
                s, o = cases['syndromes'], cases['observables']
                conv = (parity(sampler.problem.H, values['corrections']) == s).all(1)
                logical = ~conv | (parity(sampler.problem.L, values['corrections']) != o).any(1)
                np.testing.assert_array_equal(values['failed'], ~conv)
                np.testing.assert_array_equal(values['logical_failed'], logical)
                expected = summary['points'][i]['outcomes'][name]
                assert expected['failures'] == int((~conv).sum())
                assert expected['stage1_logical_failures'] == int(logical.sum())
                np.testing.assert_allclose(expected['fallback_wilson'], wilson(int((~conv).sum()), len(s)))
                rng = np.random.default_rng(np.random.SeedSequence(runtime['audit_entropy']+[20, i]))
                indices = rng.choice(len(s), runtime['audit_cases_per_factor'], replace=False)
                hat, replay_conv = decoder.decode_batch(s[indices])
                np.testing.assert_array_equal(hat, values['corrections'][indices])
                np.testing.assert_array_equal(replay_conv, conv[indices])
                np.testing.assert_array_equal(decoder.last_iterations, values['iterations'][indices])
                if name in ('neural', 'count'):
                    np.testing.assert_array_equal(decoder.choices(s), values['choices'])
                if name == 'neural':
                    predictor = decoder.predictor
                    x = torch.from_numpy((syndrome_features(s, point['p'])-predictor.mean)/predictor.scale)
                    for layer, (w, b) in enumerate(zip(predictor.weights, predictor.biases)):
                        x = x @ torch.from_numpy(w)+torch.from_numpy(b)
                        if layer+1 < len(predictor.weights):
                            x = torch.relu(x)
                    scores = torch.sigmoid(x.clamp(-80, 80)).numpy()
                    np.testing.assert_allclose(predictor.scores(s, point['p']), scores, atol=1e-6, rtol=1e-5)
                    np.testing.assert_array_equal(choose_factors(scores, point['fixed_index'], point[name]['margin']), values['choices'])
                    one_at_a_time = np.concatenate([decoder.choices(row[None, :]) for row in s])
                    np.testing.assert_array_equal(one_at_a_time, values['choices'])
                    counts['torch_numpy_and_single_batch_choice_cases'] += len(s)
                counts['replayed_confirmation_model_cases'] += len(indices)
                flags[name] = values['failed']
            for name in ('fixed30', 'count'):
                tests.append(mcnemar(flags[name], flags['neural']))
        adjusted = module['holm']([test['pvalue'] for test in tests])
        for expected, actual, adj in zip(summary['primary_tests'], tests, adjusted):
            for key, value in actual.items():
                assert expected[key] == value
            assert expected['holm_pvalue'] == adj
            assert expected['clear_gain'] == (expected['neural_failures'] < expected['baseline_failures'] and adj < settings['primary_alpha'])
        assert summary['followup_accuracy_and_timing'] == all(point['outcomes']['neural']['failures'] < point['outcomes']['fixed30']['failures'] for point in summary['points'])
    result = dict(checks_passed=True, phase=args.phase, **counts,
                  audit_source_sha256=sha(__file__), final_test_data_used=False)
    shutil.copyfile(__file__, d/f'executed_audit_{args.phase}_source.py')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
