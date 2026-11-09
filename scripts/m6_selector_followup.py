"""Conditional functional accuracy and uncached, randomized single-thread CPU pilot."""

import argparse
import hashlib
import json
import os
import platform
import runpy
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBPOSD
from bbnbp.decoders.pipeline import TwoStageDecoder
from bbnbp.io import source_manifest
from bbnbp.metrics import latency_stats, mcnemar, wilson
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accuracy(d, out, c, frozen, settings, module):
    points = []
    spec = c.eval.osd
    for i, p in enumerate(c.noise.p_grid):
        sampler = make_sampler(c, p)
        n = settings['accuracy_shots'][i]
        with np.load(d/'confirmation'/f'cases-p{i}.npz') as saved:
            s, o = saved['syndromes'][:n], saved['observables'][:n]
        stage1, union = {}, np.zeros(n, bool)
        for name in settings['models']:
            with np.load(d/'confirmation'/f'values-p{i}-{name}.npz') as saved:
                stage1[name] = dict(corrections=saved['corrections'][:n], failed=saved['failed'][:n])
            union |= stage1[name]['failed']
        fallback = LdpcBPOSD(sampler.problem, spec.bp_iters, spec.ms_scaling, spec.method, spec.order)
        indices = np.flatnonzero(union)
        hats, executed = [], []
        for count, row in enumerate(indices):
            hats.append(fallback.decode_one(s[row]))
            executed.append(fallback.last_osd_executed)
            if (count+1) % 128 == 0:
                print(f'accuracy p{i}: actual shared fallback {count+1}/{len(indices)}', flush=True)
        hats = np.asarray(hats, np.uint8).reshape(len(indices), sampler.problem.H.shape[1])
        np.savez_compressed(out/f'fallback-p{i}.npz', indices=indices, corrections=hats, executed=executed)
        location = np.full(n, -1, np.int64)
        location[indices] = np.arange(len(indices))
        outcomes, errors = {}, {}
        for name, first in stage1.items():
            hat, mask = first['corrections'].copy(), first['failed']
            hat[mask] = hats[location[mask]]
            np.testing.assert_array_equal(parity(sampler.problem.H, hat), s)
            logical = (parity(sampler.problem.L, hat) != o).any(1)
            np.savez_compressed(out/f'values-p{i}-{name}.npz', corrections=hat, logical_failed=logical, fallback_used=mask)
            k = int(logical.sum())
            outcomes[name] = dict(logical_failures=k, ler=k/n, ler_wilson=wilson(k, n), fallback_requests=int(mask.sum()))
            errors[name] = logical
        points.append(dict(p=p, shots=n, outcomes=outcomes, actual_shared_fallback_calls=len(indices),
                           paired_nominal_tests={name: mcnemar(errors[name], errors['neural']) for name in settings['models'] if name != 'neural'}))
    return dict(points=points, scope='Functional accuracy only. Shared deterministic fallback outputs are reused; no latency measurement. Error-rate equivalence is not established.')


def latency(d, out, c, frozen, settings, module):
    spec, names = c.eval.osd, settings['models']
    all_arrays, points = [], []
    for i, p in enumerate(c.noise.p_grid):
        sampler = make_sampler(c, p)
        nodes = module['models'](d, sampler.problem, frozen, i)
        decoders = {name: TwoStageDecoder(nodes[name], LdpcBPOSD(sampler.problem, spec.bp_iters, spec.ms_scaling, spec.method, spec.order)) for name in names}
        warm, target = settings['timing_warmups'], settings['timing_shots']
        s, o, e = sampler.sample(warm+target, np.random.default_rng(settings['timing_seeds'][i]))
        np.testing.assert_array_equal(parity(sampler.problem.H, e), s)
        np.testing.assert_array_equal(parity(sampler.problem.L, e), o)
        rng = np.random.default_rng(np.random.SeedSequence(settings['order_entropy']+[i]))
        order = np.stack([rng.permutation(len(names)) for _ in s])
        np.savez_compressed(out/f'cases-p{i}.npz', syndromes=s, observables=o, faults=e, order=order)
        records = []
        with (out/f'raw-p{i}.jsonl').open('w') as stream:
            for case, syndrome in enumerate(s):
                for position, index in enumerate(order[case]):
                    name = names[index]
                    started = time.perf_counter_ns()
                    result = decoders[name].decode_timed_one(syndrome)
                    elapsed = (time.perf_counter_ns()-started)/1000
                    row = dict(case=case, warmup=case < warm, method=name, order_position=position,
                               total_us=elapsed, stage1_us=result['stage1_ns']/1000, fallback_us=result['osd_ns']/1000,
                               fallback_called=bool(result['osd_used'][0]), internal_osd_executed=bool(result['osd_executed'][0]),
                               logical_failed=bool((parity(sampler.problem.L, result['e_hat'])[0] != o[case]).any()),
                               syndrome_valid=bool((parity(sampler.problem.H, result['e_hat'])[0] == syndrome).all()),
                               correction_sha256=hashlib.sha256(result['e_hat'].tobytes()).hexdigest())
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    records.append(row)
                if (case+1) % 32 == 0:
                    print(f'timing p{i}: {case+1}/{len(s)} paired cases', flush=True)
        arrays, outcomes = {}, {}
        for name in names:
            kept = [row for row in records if not row['warmup'] and row['method'] == name]
            arrays[name] = np.array([row['total_us'] for row in kept])
            assert len(kept) == target and all(row['syndrome_valid'] for row in kept)
            outcomes[name] = dict(**latency_stats(arrays[name]), fallback_calls=sum(row['fallback_called'] for row in kept),
                                  logical_failures=sum(row['logical_failed'] for row in kept))
        points.append(dict(p=p, seed=settings['timing_seeds'][i], measured_cases=target, warmups=warm, outcomes=outcomes))
        all_arrays.append(arrays)
    # Bootstrap after all timed calls, so statistics never compete with measurement.
    bootstrap = runpy.run_path('scripts/m6_aux_latency.py')['paired_bootstrap']
    for i, (point, arrays) in enumerate(zip(points, all_arrays)):
        point['paired_nominal_bootstrap'] = {name: bootstrap(arrays[name], arrays['neural'],
            np.random.default_rng(np.random.SeedSequence(settings['bootstrap_entropy']+[i, k])), settings['bootstrap_repeats'])
            for k, name in enumerate(names) if name != 'neural'}
    return dict(points=points, scope='Auxiliary uncached paired CPU pilot. Predictor included in stage1 cost. Nominal bootstrap intervals are descriptive; no main E2 or p99 superiority claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--phase', required=True, choices=('accuracy', 'latency'))
    args = parser.parse_args()
    d = args.directory
    summary = json.loads((d/'confirmation/summary.json').read_text())
    if not summary['followup_accuracy_and_timing']:
        raise ValueError('Predeclared follow-up gate did not pass')
    assert json.loads((d/'audit-confirmation.json').read_text())['checks_passed']
    if args.phase == 'latency':
        json.loads((d/'accuracy/summary.json').read_text())
    frozen = json.loads((d/'selection.json').read_text())
    for path, expected in frozen['input_hashes'].items():
        assert sha(path) == expected, path
    settings = yaml.safe_load((d/'followup-settings.yaml').read_text())
    c = load_config(d/'config.yaml')
    out = d/args.phase
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out/'executed_source.py')
    shutil.copyfile('scripts/m6_aux_latency.py', out/'bootstrap_source.py')
    (out/'source_manifest.json').write_text(json.dumps(source_manifest(), indent=2)+'\n')
    (out/'machine.json').write_text(json.dumps(dict(cpu=c.eval.cpu_model, platform=platform.platform(), threads=1,
        environment={name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        other_system_load='uncontrolled', known_labeling_and_training_jobs_finished=True), indent=2)+'\n')
    torch.set_num_threads(1)
    module = runpy.run_path(str(d/'confirmation/executed_source.py'))
    started = time.perf_counter()
    def timeout(signum, frame):
        raise TimeoutError(f'Selector {args.phase} reached its fixed CPU wall cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, settings[f'{"timing" if args.phase == "latency" else "accuracy"}_wall_seconds'])
    try:
        result = (accuracy if args.phase == 'accuracy' else latency)(d, out, c, frozen, settings, module)
        result.update(elapsed_s=time.perf_counter()-started, selection_sha256=sha(d/'selection.json'),
                      final_test_data_used=False, m6_complete=False)
        (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        (out/'FAILURE.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
