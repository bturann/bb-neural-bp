# Learned BP for bivariate bicycle codes

**Status: exploratory research archive; development paused after feasibility review.**

This repository studies a small neural min-sum belief-propagation (BP) first stage
with BP-OSD fallback for bivariate bicycle quantum error-correcting codes.
Modest first-stage improvements were observed. An end-to-end advantage in latency
and preserved logical accuracy over strong controls was **not established**.
This is a research prototype, not a validated commercial decoder or a completed
lab-to-market product. The original main M6/E1/E2 and M7 experiments are incomplete.

Start with the [project assessment](docs/project_viability_assessment.md) and
[archive guide](docs/archive_guide.md). The guide maps claims to their evidence
and distinguishes completed studies from proposals that were never run.

## Results and limits

On independent BB72 circuit-level cases with a 100-iteration cap, the frozen
learned model requested fallback less often than the validation-selected ordinary
control:

| Physical noise p | Ordinary100 fallback requests | Learned100 fallback requests |
|---|---:|---:|
| .001 | 495 / 4,096 (12.08%) | 443 / 4,096 (10.82%) |
| .003 | 2,654 / 4,096 (64.79%) | 2,518 / 4,096 (61.47%) |

Both primary paired differences passed their declared two-point Holm correction.
All three training seeds improved in direction; the additional low-noise seed
comparisons were not individually significant after their declared correction.
These are first-stage request counts, not speedups or full-pipeline error rates.

- The [audited CPU report](results/m6_aux_report/20261109T135141505262Z/report.md)
  contains paired comparisons, logical errors, actual timing measurements,
  uncertainty intervals and negative controls. Timing pilots used 256 measured
  cases per noise point; p99 superiority and LER equivalence remain unproven.
- The BP100-to-learned30 cascade lost to its stronger ordinary control.
- The [CPU scaling-selector study](results/m6_scaling_selector/20261109-approved/report/report.md)
  found no improvement over fixed BP30 on fresh cases at either noise rate.
  Its predeclared gate stopped additional selector accuracy/timing work.
- The [sample-size/runtime analysis](results/m6_cycle100_planning/20261110T015309813879Z/report.md)
  is planning from saved data. Its hypothetical simulations are not new decoder
  observations. Neither proposed follow-up benchmark was launched.
- BB144 scaling and the full original experiment matrix were not demonstrated.
  Main final-test streams remain reserved and unused in the reported studies.

Positive, negative and inconclusive outcomes are retained. No experiment is
queued by this archive. [NOTES.md](NOTES.md) records development decisions,
failed attempts, protocol deviations, gates and verification results.

## Install and verify

Use Python 3.11 and the pinned environment from the repository root:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -c environment.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q
```

The full suite includes slow acceptance checks and small software-test fixtures.
Tests verify implementation behavior; they do not establish the research claim.
Dependencies are installed separately; no runtime environment is vendored.

For a functional eight-case example using the tracked frozen checkpoint, follow
[the cycle100 CPU inference guide](docs/m6_frozen_cycle100.md). The example uses
saved development cases and is not a new benchmark. The checkpoint holds the
original 15-step parameters: use the saved builder to obtain the tested
100-iteration rule. No new training or GPU is needed for that example.

## Repository map

| Path | Contents |
|---|---|
| `src/bbnbp/` | Code construction, noise adapters, BP/neural decoders, training and evaluation |
| `tests/`, `reference/` | Software checks and immutable handoff reference code |
| `configs/` | Experiment settings, including unexecuted plans clearly identified in the archive guide |
| `scripts/` | Experiment, analysis, audit and figure-generation commands |
| `results/` | Saved configurations, source snapshots/hashes, model copies, cases, per-shot outcomes, reports and plots |
| `docs/` | Assessment, evidence guide, inference examples and historical instructions |
| `checkpoints/` | Ignored local training output; reviewed frozen model copies are tracked under `results/` |

The recorded CPU pilots used an Apple M4 Pro, one decoding thread and fresh
uncached fallback calls for timing. Shared fallback outputs in functional accuracy
studies were used only for outcome assessment, never as timing evidence. Absolute
runtime depends on hardware and system load.

The repository retains the full evidence history. Raw provenance can contain
original local paths or host labels; see the archive guide for portability limits.
Some early training checkpoints and the original planning PDF were local-only.
The final frozen inference checkpoint and primary evidence bundles are included.

## Reproduction and historical scope

[Historical setup and experiment commands](docs/experiment_history.md) preserve
the original development instructions. Running them explicitly may launch
substantial computation and creates new result bundles. Reusing recorded seeds
reproduces the same streams; it is not independent confirmation.

The [implementation handoff](docs/implementation_spec.md) remains unchanged as a
historical specification. Its aspirational claims and proposed matrix describe
the plan, not completed results. See the assessment and evidence reports for
what actually ran. Source snapshots and configs in each result bundle document
its execution state; current documentation changes do not rewrite that history.

## License and citation

Project code and associated materials: [MIT](LICENSE). Dependencies retain their
own licenses. [ATTRIBUTION.md](ATTRIBUTION.md) identifies the reference-code origin,
direct dependencies, AI-assisted development and relevant research methods.
[CITATION.cff](CITATION.cff) supplies software citation metadata. Cite the original
methods as applicable, alongside this archive.
