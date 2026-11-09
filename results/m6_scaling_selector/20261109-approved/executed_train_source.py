"""Fit auxiliary CPU predictors and freeze validation-only choices before confirmation."""

import argparse
import hashlib
import json
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from bbnbp.config import load_config
from bbnbp.decoders.scaling_selector import DensePredictor, CountPredictor, choose_factors, syndrome_features
from bbnbp.io import source_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Predictor(nn.Module):
    """Every randomized parameter receives the explicit protocol generator."""
    def __init__(self, dimensions, generator):
        super().__init__()
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        for a, b in zip(dimensions[:-1], dimensions[1:]):
            w = torch.empty(a, b)
            w.uniform_(-np.sqrt(6/(a+b)), np.sqrt(6/(a+b)), generator=generator)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(b)))

    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w+b
            if i+1 < len(self.weights):
                x = torch.relu(x)
        return x

    def save(self, path, mean, scale):
        values = dict(mean=mean, scale=scale, layers=np.asarray(len(self.weights)))
        values.update({f'w{i}': w.detach().numpy() for i, w in enumerate(self.weights)})
        values.update({f'b{i}': b.detach().numpy() for i, b in enumerate(self.biases)})
        np.savez_compressed(path, **values)


def outcome(scores, labels, fixed, margin):
    choices = choose_factors(scores, fixed, margin)
    rows = np.arange(len(choices))
    return dict(failures=int((~labels['converged'][rows, choices]).sum()),
                logical_failures=int((~labels['correct'][rows, choices]).sum()),
                switched=int((choices != fixed).sum()), margin=float(margin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    d = args.directory
    if (d/'selection.json').exists():
        raise ValueError('Selection is already frozen')
    settings = yaml.safe_load((d/'p0/settings.yaml').read_text())
    if yaml.safe_load((d/'p1/settings.yaml').read_text()) != settings:
        raise ValueError('Label workers used different protocols')
    c = load_config(settings['base_config'])
    c.save(d/'config.yaml')
    (d/'settings.yaml').write_text(yaml.safe_dump(settings))
    shutil.copyfile(__file__, d/'executed_train_source.py')
    shutil.copyfile('src/bbnbp/decoders/scaling_selector.py', d/'selector_source.py')
    (d/'training_source_manifest.json').write_text(json.dumps(source_manifest(), indent=2)+'\n')
    torch.set_num_threads(settings['threads'])
    started = time.perf_counter()
    def timeout(signum, frame):
        raise TimeoutError('Selector training reached its fixed CPU wall cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, settings['training_wall_seconds'])
    try:
        training, targets, validation, vlabels, fixed, factors = [], [], [], [], [], None
        for i, p in enumerate(c.noise.p_grid):
            receipt = json.loads((d/f'p{i}/summary.json').read_text())
            for filename, key in (('cases.npz', 'cases_sha256'), ('labels.npz', 'labels_sha256')):
                if sha(d/f'p{i}'/filename) != receipt[key]:
                    raise ValueError('Training inputs changed after labeling')
            with np.load(d/f'p{i}/cases.npz') as saved:
                s = saved['syndromes']
            with np.load(d/f'p{i}/labels.npz') as saved:
                labels = {k: saved[k] for k in saved.files}
            if factors is not None:
                np.testing.assert_array_equal(factors, labels['factors'])
            factors = labels['factors']
            training.append(syndrome_features(s, p))
            targets.append(labels['correct'].astype(np.float32))
            source = Path(settings['validation_sources'][i])
            with np.load(source/'cases.npz') as saved:
                vs = saved['syndromes']
            validation.append((vs, p))
            with np.load(Path(settings['opportunity'])/f'longer-validation-labels-p{i}.npz') as saved:
                vlabels.append({k: saved[k] for k in saved.files})
            np.testing.assert_array_equal(factors, vlabels[-1]['factors'])
            selection = json.loads((source/'selection.json').read_text())
            factor = selection['selected']['longer']['scaling']
            fixed.append(int(np.flatnonzero(np.isclose(factors, factor))[0]))
            counts = s.sum(1)
            edges = np.unique(np.quantile(counts, np.arange(1, settings['count_bins'])/settings['count_bins']))
            bins = np.searchsorted(edges, counts, side='right')
            prior = settings['count_prior_cases']
            rates = np.stack([(labels['correct'][bins == j].sum(0)+prior*labels['correct'].mean(0)) /
                              ((bins == j).sum()+prior) for j in range(len(edges)+1)])
            np.savez_compressed(d/f'count-p{i}.npz', edges=edges, rates=rates)
        x = np.concatenate(training)
        mean, scale = x.mean(0), np.maximum(x.std(0), .1)
        x = torch.from_numpy((x-mean)/scale)
        y = torch.from_numpy(np.concatenate(targets))
        points = []
        for i, (s, p) in enumerate(validation):
            scores = CountPredictor(d/f'count-p{i}.npz').scores(s, p)
            records = [outcome(scores, vlabels[i], fixed[i], margin) for margin in settings['margins']]
            best = min(records, key=lambda row: (row['logical_failures'], row['failures'], -row['margin']))
            source_selection = json.loads((Path(settings['validation_sources'][i])/'selection.json').read_text())
            long_selection = json.loads((Path(settings['long_controls'][i])/'selection.json').read_text())
            strong_selection = json.loads((Path(settings['strong_study'])/'selection.json').read_text())['points'][i]
            controls = dict(two30=dict(first_factor=source_selection['first_factor'],
                                      second_factor=source_selection['selected']['two_pass']['scaling']),
                            bp100=long_selection['choices']['100'],
                            bp130=strong_selection['choices']['bp130'])
            points.append(dict(p=p, fixed_index=fixed[i], fixed_factor=float(factors[fixed[i]]), controls=controls,
                               count=dict(file=f'count-p{i}.npz', sha256=sha(d/f'count-p{i}.npz'), **best),
                               count_grid=records))
        records = []
        for arch_index, architecture in enumerate(settings['architectures']):
            generator = torch.Generator().manual_seed(int(np.random.SeedSequence(settings['torch_entropy']+[arch_index]).generate_state(1)[0]))
            model = Predictor([x.shape[1]]+(settings['hidden'] if architecture == 'mlp' else [])+[len(factors)], generator)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
            rng = np.random.default_rng(np.random.SeedSequence(settings['shuffle_entropy']+[arch_index]))
            for epoch in range(1, settings['epochs']+1):
                order = rng.permutation(len(x))
                total = 0.
                for start in range(0, len(x), settings['batch']):
                    indices = torch.from_numpy(order[start:start+settings['batch']])
                    loss = nn.functional.binary_cross_entropy_with_logits(model(x[indices]), y[indices])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite selector training loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())*len(indices)
                if epoch % settings['validate_every']:
                    continue
                filename = f'{architecture}-epoch{epoch}.npz'
                model.save(d/filename, mean, scale)
                predictor = DensePredictor(d/filename)
                for i, (s, p) in enumerate(validation):
                    scores = predictor.scores(s, p)
                    for margin in settings['margins']:
                        records.append(dict(index=i, architecture=architecture, arch_index=arch_index, epoch=epoch,
                                            file=filename, train_bce=total/len(x),
                                            **outcome(scores, vlabels[i], fixed[i], margin)))
                print(f'{architecture} epoch={epoch} train_bce={total/len(x):.6f}; elapsed={time.perf_counter()-started:.1f}s', flush=True)
        for i, point in enumerate(points):
            best = min((row for row in records if row['index'] == i),
                       key=lambda row: (row['logical_failures'], row['failures'], row['arch_index'], row['epoch'], -row['margin']))
            point['neural'] = dict(**best, sha256=sha(d/best['file']))
        (d/'validation_grid.json').write_text(json.dumps(records, indent=2)+'\n')
        followup_settings = Path('configs/m6/cpu_selector_followup.yaml')
        shutil.copyfile(followup_settings, d/'followup-settings.yaml')
        hashed = [d/'settings.yaml', d/'config.yaml', d/'selector_source.py', d/'executed_train_source.py', d/'validation_grid.json',
                  d/'followup-settings.yaml']
        for i in range(len(points)):
            hashed += [d/f'p{i}/cases.npz', d/f'p{i}/labels.npz', Path(settings['validation_sources'][i])/'cases.npz']
            hashed += [d/points[i][name]['file'] for name in ('count', 'neural')]
            hashed += [Path(settings['validation_sources'][i])/'selection.json',
                       Path(settings['long_controls'][i])/'selection.json']
        hashed += [Path(settings['strong_study'])/'selection.json']
        frozen = dict(factors=factors.tolist(), iterations=settings['iterations'], points=points,
                      frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                      input_hashes={str(path): sha(path) for path in hashed},
                      confirmation_seeds=settings['confirmation_seeds'], confirmation_shots=settings['confirmation_shots'],
                      primary_family='Holm: neural vs fixed30 and count rule, paired fallback failures, two noise rates (4 tests)',
                      inference_inputs=['syndrome', 'known_p'], elapsed_s=time.perf_counter()-started,
                      confirmation_data_used=False, final_test_data_used=False, m6_complete=False)
        (d/'selection.json').write_text(json.dumps(frozen, indent=2)+'\n')
        print(json.dumps(points, indent=2), flush=True)
    except BaseException as exc:
        (d/'TRAIN_FAILURE.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
