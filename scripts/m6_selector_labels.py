"""Actual BP30 factor outcomes on unfiltered, independently seeded training cases."""

import argparse
import hashlib
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
from bbnbp.io import source_manifest
from bbnbp.noise.base import parity
from bbnbp.noise.factory import make_sampler


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default='configs/m6/cpu_scaling_selector.yaml')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--index', required=True, type=int, choices=(0, 1))
    parser.add_argument('--prepare-only', action='store_true', help='Freeze inputs without decoding factors; use parallel resume next')
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.settings).read_text())
    c = load_config(settings['base_config'])
    d = args.output / f'p{args.index}'
    d.mkdir(parents=True, exist_ok=False)
    c.save(d / 'config.yaml')
    (d / 'settings.yaml').write_text(yaml.safe_dump(settings))
    shutil.copyfile(__file__, d / 'executed_source.py')
    (d / 'source_manifest.json').write_text(json.dumps(source_manifest(), indent=2)+'\n')
    print(f'Selector labels: {d}', flush=True)
    started = time.perf_counter()
    def timeout(signum, frame):
        raise TimeoutError('Selector labeling reached its fixed CPU wall cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, settings['label_wall_seconds'])
    try:
        sampler = make_sampler(c, c.noise.p_grid[args.index])
        if hasattr(sampler, 'dem'):
            (d / 'problem.dem').write_text(str(sampler.dem)+'\n')
            (d / 'circuit.stim').write_text(str(sampler.circuit)+'\n')
        n = settings['training_cases_per_point']
        seed = settings['train_seeds'][args.index]
        if seed in settings['confirmation_seeds'] or 11234 <= seed <= 11238:
            raise ValueError('Training overlaps reserved seeds')
        if args.index == 0:
            source = Path(settings['unfiltered_pool'])
            with np.load(source) as pool:
                cases = {key: pool[key][:n].copy() for key in ('syndromes', 'observables', 'faults')}
            origin = dict(source=str(source), sha256=sha(source), prefix=n, unfiltered=True, seed=seed)
        else:
            rng = np.random.default_rng(seed)
            parts = [sampler.sample(min(settings['pool_batch'], n-start), rng)
                     for start in range(0, n, settings['pool_batch'])]
            cases = dict(zip(('syndromes', 'observables', 'faults'),
                             (np.concatenate([part[k] for part in parts]) for k in range(3))))
            origin = dict(seed=seed, batch=settings['pool_batch'], rng_state=rng.bit_generator.state, unfiltered=True)
        s, o, e = (cases[k] for k in ('syndromes', 'observables', 'faults'))
        if len(s) != n:
            raise ValueError('Wrong unfiltered case count')
        np.testing.assert_array_equal(parity(sampler.problem.H, e), s)
        np.testing.assert_array_equal(parity(sampler.problem.L, e), o)
        np.savez_compressed(d / 'cases.npz', **cases)
        (d / 'origin.json').write_text(json.dumps(origin, indent=2)+'\n')
        (d / 'prepared.json').write_text(json.dumps(dict(cases_sha256=sha(d/'cases.npz'), shots=n,
            prepared_at_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
        if args.prepare_only:
            return
        factors = np.asarray(c.sweep.factors)
        shape = (n, len(factors))
        converged, correct = np.zeros(shape, bool), np.zeros(shape, bool)
        iterations = np.zeros(shape, np.int64)
        records = []
        # Save every factor independently: completed outcomes survive any later failure.
        for j, factor in enumerate(factors):
            decoder = LdpcBP(sampler.problem, settings['iterations'], float(factor))
            hashes = []
            for start in range(0, n, settings['label_batch']):
                stop = min(n, start+settings['label_batch'])
                hat, conv = decoder.decode_batch(s[start:stop])
                converged[start:stop, j] = conv
                correct[start:stop, j] = conv & (parity(sampler.problem.L, hat) == o[start:stop]).all(1)
                iterations[start:stop, j] = decoder.last_iterations
                hashes.append(hashlib.sha256(hat.tobytes()).hexdigest())
                if stop % 8192 == 0:
                    print(f'p{args.index} factor={factor:g} cases={stop}/{n}; elapsed={time.perf_counter()-started:.1f}s', flush=True)
            filename = f'factor-{factor:g}.npz'
            np.savez_compressed(d / filename, converged=converged[:, j], correct=correct[:, j], iterations=iterations[:, j])
            records.append(dict(factor=float(factor), file=filename, sha256=sha(d / filename), correction_batch_hashes=hashes))
            (d / 'progress.json').write_text(json.dumps(dict(completed=records, elapsed_s=time.perf_counter()-started), indent=2)+'\n')
            print(f'p{args.index} factor={factor:g}: failures={int((~converged[:,j]).sum())}/{n}; elapsed={time.perf_counter()-started:.1f}s', flush=True)
        np.savez_compressed(d / 'labels.npz', factors=factors, converged=converged, correct=correct, iterations=iterations)
        receipt = dict(p=c.noise.p_grid[args.index], seed=seed, shots=n, cases_sha256=sha(d/'cases.npz'),
                       labels_sha256=sha(d/'labels.npz'), completed=records,
                       elapsed_s=time.perf_counter()-started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                       training_only=True, final_test_data_used=False, gpu_seconds=0)
        (d / 'summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
    except BaseException as exc:
        (d / 'FAILURE.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
