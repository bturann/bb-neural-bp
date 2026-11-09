"""Generate an evidence-linked auxiliary selector report from completed, audited runs."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    d = args.directory
    out = d/'report'
    out.mkdir(exist_ok=False)
    frozen = json.loads((d/'selection.json').read_text())
    summary = json.loads((d/'confirmation/summary.json').read_text())
    train_counts = [json.loads((d/f'p{i}/summary.json').read_text())['shots'] for i in range(len(frozen['points']))]
    total_training = sum(train_counts)
    for phase in ('labels', 'confirmation'):
        assert json.loads((d/f'audit-{phase}.json').read_text())['checks_passed']
    if summary['followup_accuracy_and_timing']:
        for phase in ('accuracy', 'latency'):
            json.loads((d/phase/'summary.json').read_text())
            assert json.loads((d/f'audit-{phase}.json').read_text())['checks_passed']
    grid = json.loads((d/'validation_grid.json').read_text())
    names = list(summary['points'][0]['outcomes'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, name in enumerate(names):
        rates, errors = [], [[], []]
        for point in summary['points']:
            item = point['outcomes'][name]
            value = 100*item['fallback_rate']
            rates.append(value)
            errors[0].append(value-100*item['fallback_wilson'][0])
            errors[1].append(100*item['fallback_wilson'][1]-value)
        axes[0].bar(np.arange(2)+(j-(len(names)-1)/2)*.12, rates, .12, yerr=errors, label=name, capsize=2)
    axes[0].set_xticks([0, 1], ['p=.001', 'p=.003'])
    axes[0].set_ylabel('Stage1 fallback requests (%)')
    axes[0].set_title('Fresh paired confirmation')
    axes[0].legend(fontsize=8)
    for arch in ('linear', 'mlp'):
        curve = {row['epoch']: row['train_bce'] for row in grid if row['architecture'] == arch}
        epochs = sorted(curve)
        axes[1].plot(epochs, [curve[epoch] for epoch in epochs], label=arch)
    axes[1].set_xlabel('Training epoch')
    axes[1].set_ylabel('Multilabel training BCE')
    axes[1].set_title(f'{total_training:,} unfiltered training cases')
    axes[1].legend()
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'selector-study.{suffix}', dpi=180)
    plt.close(fig)
    fixed = [test for test in summary['primary_tests'] if test['baseline'] == 'fixed30']
    cheap = [test for test in summary['primary_tests'] if test['baseline'] == 'count']
    fixed_gain = all(test['clear_gain'] for test in fixed)
    cheap_gain = all(test['clear_gain'] for test in cheap)
    lines = ['# Approved CPU scaling-selector study', '',
             ('The selector established a corrected fallback-demand gain over fixed BP30 at both noise rates.' if fixed_gain else
              'The selector did not establish a corrected fallback-demand gain over fixed BP30 at both noise rates.'),
             ('It also beat the count rule at both rates.' if cheap_gain else 'Superiority over the cheap count rule at both rates was not established.'), '',
             'This is an auxiliary model-class experiment. Original learned-message evidence remains intact; main M6/E1/E2 and M7 are incomplete.', '',
             '## Protocol', '',
             f'{train_counts} unfiltered training cases by point; {len(frozen["factors"])} factors were executed with actual BP30 decoders ({total_training*len(frozen["factors"]):,} case-factor outcomes). '
             'Logical-success multilabel BCE trained linear and 64/32 ReLU predictors. Inputs were syndrome bits, normalized count and known p. '
             'No true faults, observables or decoder outcomes enter inference. Every case receives exactly one selected BP30 pass.', '',
             'Architecture, checkpoint and switching margin were chosen on reused 42234/42235 validation cases, then hashed and frozen before '
             f'sampling fresh confirmation sets with seeds {frozen["confirmation_seeds"]}. Validation performance is development evidence. The oracle union is not an achieved decoder.', '',
             '## Frozen choices', '', '| p | Predictor | Epoch | Margin | Fixed factor | Count margin |',
             '|---|---|---:|---:|---:|---:|']
    for point in frozen['points']:
        spec = point['neural']
        lines.append(f'| {point["p"]:g} | {spec["architecture"]} | {spec["epoch"]} | {spec["margin"]:g} | {point["fixed_factor"]:g} | {point["count"]["margin"]:g} |')
    lines += ['', '## Fresh Stage1 outcomes', '', '| p | Method | Fallback requests / shots | Rate | Stage1 logical failures* |',
              '|---|---|---:|---:|---:|']
    for point in summary['points']:
        for name, values in point['outcomes'].items():
            lines.append(f'| {point["p"]:g} | {name} | {values["failures"]}/{point["shots"]} | {100*values["fallback_rate"]:.3f}% | {values["stage1_logical_failures"]} |')
    lines += ['', '*Stage1 logical failures include nonconvergence. They are not full-pipeline logical error rates.', '',
              '| p | Baseline | Baseline fails / neural succeeds | Neural fails / baseline succeeds | Holm p | Clear gain |',
              '|---|---|---:|---:|---:|---|']
    for test in summary['primary_tests']:
        lines.append(f'| {summary["points"][test["index"]]["p"]:g} | {test["baseline"]} | {test["first_only"]} | {test["second_only"]} | {test["holm_pvalue"]:.6g} | {test["clear_gain"]} |')
    lines += ['', 'The four neural-vs-fixed/count paired tests form one Holm family. Stronger BP100/BP130 and two-pass comparisons are descriptive challenges.', '',
              '![Fresh results and training loss](selector-study.png)', '', '## Follow-up and limits', '']
    if not summary['followup_accuracy_and_timing']:
        lines += ['The predeclared follow-up gate failed: the selector did not reduce observed fallback requests versus fixed BP30 at both points. '
                  'No selector full-pipeline accuracy or latency sample was collected. No speed, p99 or LER-equivalence claim follows.']
    else:
        lines += ['The predeclared directional follow-up gate passed. Accuracy uses fixed confirmation prefixes; timing uses independent 92234/92235 '
                  'cases with uncached fallback calls and randomized method order. Predictor overhead is included. These are pilot samples, '
                  'with descriptive intervals; they do not establish p99 superiority or LER equivalence.', '',
                  '| p | Method | Logical failures / shots | Mean CPU ms | p99 CPU ms |', '|---|---|---:|---:|---:|']
        accuracy = json.loads((d/'accuracy/summary.json').read_text())
        timing = json.loads((d/'latency/summary.json').read_text())
        for acc, lat in zip(accuracy['points'], timing['points']):
            for name, val in acc['outcomes'].items():
                t = lat['outcomes'][name]
                lines.append(f'| {acc["p"]:g} | {name} | {val["logical_failures"]}/{acc["shots"]} | {t["lat_mean_us"]/1000:.3f} | {t["lat_p99_us"]/1000:.3f} |')
    lines += ['', 'The main final-test seeds 11234–11238 were not used. All work used the pinned local CPU environment, with no new dependency or GPU run.', '',
              '## Evidence and reproduction', '',
              '- [Frozen selection](../selection.json)', '- [Fresh paired results](../confirmation/summary.json)',
              '- [Training-label audit](../audit-labels.json)', '- [Confirmation audit](../audit-confirmation.json)',
              '- [Protocol](../settings.yaml)', '- [Training/validation grid](../validation_grid.json)', '',
              'The first two sequential labeling workers were deliberately interrupted after saving factor .50 to resume the identical computation '
              'in six CPU processes. Their KeyboardInterrupt receipts, executed source and completed factor outputs are retained. '
              'The label audit regenerates every training input and independently replays 64 cases per factor and p; it does not claim an exhaustive second decoding of all 704,000 outcomes.', '',
              '```sh', f'.venv/bin/python scripts/m6_audit_selector.py {d} --phase labels',
              f'.venv/bin/python scripts/m6_audit_selector.py {d} --phase confirmation', '```', '',
              'Audits refuse to overwrite receipts; use a copied bundle with the relevant audit receipt removed for replay. '
              'Complete rerun instructions are in `docs/m6_scaling_selector.md`.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    paths = [d/'selection.json', d/'confirmation/summary.json', d/'audit-labels.json', d/'audit-confirmation.json',
             out/'report.md', out/'selector-study.png', out/'selector-study.pdf']
    (out/'input_hashes.json').write_text(json.dumps({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, indent=2)+'\n')
    shutil.copyfile(__file__, out/'executed_source.py')
    print(out/'report.md')


if __name__ == '__main__':
    main()
