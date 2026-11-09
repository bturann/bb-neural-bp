"""Resume preserved factor labeling with bounded single-thread CPU process parallelism."""

import argparse
import hashlib
import json
import multiprocessing
import shutil
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def label(task):
    root, index, factor = task
    d = Path(root)/f'p{index}'
    settings = yaml.safe_load((d/'settings.yaml').read_text())
    c = load_config(d/'config.yaml')
    sampler = make_sampler(c, c.noise.p_grid[index])
    with np.load(d/'cases.npz') as saved:
        s, o = saved['syndromes'], saved['observables']
    decoder = LdpcBP(sampler.problem, settings['iterations'], factor)
    conv, correct, iterations, hashes = [], [], [], []
    started = time.perf_counter()
    for start in range(0, len(s), settings['label_batch']):
        stop = min(len(s), start+settings['label_batch'])
        hat, flags = decoder.decode_batch(s[start:stop])
        conv.append(flags)
        correct.append(flags & (parity(sampler.problem.L, hat) == o[start:stop]).all(1))
        iterations.append(decoder.last_iterations.copy())
        hashes.append(hashlib.sha256(hat.tobytes()).hexdigest())
    filename = f'factor-{factor:g}.npz'
    temporary = d/f'pending-{factor:g}.npz'
    np.savez_compressed(temporary, converged=np.concatenate(conv), correct=np.concatenate(correct), iterations=np.concatenate(iterations))
    temporary.rename(d/filename)
    record = dict(factor=factor, file=filename, sha256=sha(d/filename), correction_batch_hashes=hashes,
                  elapsed_s=time.perf_counter()-started, source='executed_label_resume_source.py')
    (d/f'receipt-{factor:g}.json').write_text(json.dumps(record, indent=2)+'\n')
    return index, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--runtime', default='configs/m6/cpu_selector_label_runtime.yaml')
    args = parser.parse_args()
    root = args.directory
    runtime = yaml.safe_load(Path(args.runtime).read_text())
    (root/'label_runtime.yaml').write_text(yaml.safe_dump(runtime))
    shutil.copyfile(__file__, root/'executed_label_resume_source.py')
    (root/'resume_source_manifest.json').write_text(json.dumps(source_manifest(), indent=2)+'\n')
    records, tasks = {}, []
    started = time.perf_counter()
    for index in (0, 1):
        d = root/f'p{index}'
        c = load_config(d/'config.yaml')
        completed = json.loads((d/'progress.json').read_text())['completed'] if (d/'progress.json').exists() else []
        records[index] = {record['factor']: record for record in completed}
        for factor in c.sweep.factors:
            receipt = d/f'receipt-{factor:g}.json'
            if receipt.exists():
                records[index][factor] = json.loads(receipt.read_text())
            if factor in records[index]:
                record = records[index][factor]
                if sha(d/record['file']) != record['sha256']:
                    raise ValueError('Preserved factor labels changed')
            else:
                tasks.append((str(root), index, float(factor)))
    # Interleave points, avoiding a long tail dominated by the higher-noise point.
    tasks.sort(key=lambda task: (task[2], task[1]))
    pool = ProcessPoolExecutor(max_workers=runtime['workers'], mp_context=multiprocessing.get_context('spawn'))
    def timeout(signum, frame):
        raise TimeoutError('Parallel selector labeling reached its fixed CPU wall cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, runtime['wall_seconds'])
    try:
        futures = [pool.submit(label, task) for task in tasks]
        for future in as_completed(futures):
            index, record = future.result()
            records[index][record['factor']] = record
            print(f'p{index} factor={record["factor"]:g} saved; completed={len(records[0])+len(records[1])}/22; elapsed={time.perf_counter()-started:.1f}s', flush=True)
        pool.shutdown(wait=True)
        for index in (0, 1):
            d = root/f'p{index}'
            c = load_config(d/'config.yaml')
            settings = yaml.safe_load((d/'settings.yaml').read_text())
            columns = []
            for factor in c.sweep.factors:
                record = records[index][factor]
                with np.load(d/record['file']) as saved:
                    columns.append({key: saved[key] for key in saved.files})
            np.savez_compressed(d/'labels.npz', factors=np.asarray(c.sweep.factors),
                                **{key: np.stack([col[key] for col in columns], axis=1) for key in columns[0]})
            receipt = dict(p=c.noise.p_grid[index], seed=settings['train_seeds'][index], shots=settings['training_cases_per_point'],
                           cases_sha256=sha(d/'cases.npz'), labels_sha256=sha(d/'labels.npz'),
                           completed=[records[index][factor] for factor in c.sweep.factors],
                           elapsed_resume_s=time.perf_counter()-started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                           training_only=True, final_test_data_used=False, gpu_seconds=0, runtime_file='../label_runtime.yaml')
            (d/'summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
    except BaseException as exc:
        (root/'LABEL_RESUME_FAILURE.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        # A deadline must also stop workers rather than leave unbounded background jobs.
        for process in list(pool._processes.values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
