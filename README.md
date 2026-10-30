# Learned BP for bivariate bicycle codes

Research implementation of a neural min-sum first stage with BP-OSD fallback.
The target is fewer fallback calls at preserved logical error performance, measured
against ordinary and tuned-scaling BP. The neural model is implemented; training
and learned-performance results are still pending.

## Setup

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -c environment.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/m0_verify_apis.py
.venv/bin/python -m pytest -q
```

See `NOTES.md` for milestone status, API findings, and recorded deviations.
The unchanged handoff is preserved in `docs/implementation_spec.md`.

M0–M2 are complete: environment/API verification, both BB code constructions and
their symmetries, code-capacity sampling, BP/BP-OSD baselines, scaling tuning, and
paired E0 evaluation. M3 passed with the specified OSD-CS order-7 baseline; the
earlier OSD-0 failed development check remains recorded. M4's neural model and matching
compiled CPU inference passed their acceptance checks. The full suite currently has
84 passing tests, including the slow checks. See `NOTES.md` for evidence.

## Code-capacity baseline experiment (E0)

```sh
bash scripts/run_experiment.sh
```

This runs the full scaling sweep on separate validation samples, then evaluates
ordinary and tuned BP, both alone and with BP-OSD fallback, on paired development
shots. It writes the sweep, its config, and a timestamped evaluation directory under
`results/e0-<timestamp>/`. Each invocation creates a fresh bundle, including a new
tuning table, so the command can be run again without overwriting evidence. Use
`--config <path> --output-root <directory>` to customize the run. Final-test seeds are
reserved for later milestones; E0 uses a separate development seed range.

Every evaluation includes the resolved YAML, exact-schema CSV, paired per-shot
fallback flags, and diagnostic counts of actual internal OSD executions. `osd_rate`
is actual stage-2 calls divided by shots. For standalone BP it is zero; its potential
fallback demand is `stage1_failures / shots`. Standalone LER counts nonconvergence
as failure. With several decoders, stopping occurs at a shared batch boundary when
all reach the logical-failure target, or at the configured maximum shot count.

Latency is disabled in E0. The primary latency comparison, including compiled neural
inference, belongs to M6. Neural training is not yet implemented.

Set `BBNBP_CACHE_DIR` to control logical-operator caching (default `~/.cache/bbnbp`).
The experiment script uses a repository-local cache so it also works in a restricted
workspace. The supplied `reference/` files are immutable copies of the handoff.

## Circuit-level development check (M3)

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 .venv/bin/python scripts/m3_validate_circuit.py
```

This builds six total syndrome-extraction cycles using QUITS, with noisy preparation,
final measurements, gates, and idle qubits, and both bases of detectors. It preserves
the order of every DEM fault column. Circuit-level `basis` denotes the stored and
measured logical basis; code-capacity `basis` denotes the error basis.

Each invocation writes a new directory under `results/m3/`, including the actual
circuits, DEMs, resolved configuration, CSV, metadata, and gate report. The M3 config
uses the cheaper OSD-0 decoder for development; the base experiment retains OSD-CS
order 7. Verification samples use a separate recorded RNG stream.

The first development run had 20/2048 logical failures at p=0.001 (0.98%, Wilson
95% interval 0.63%–1.50%) and failed the low-error gate. The command returns nonzero
on this gate failure and records it. These OSD-0 results make no claim about neural performance.

The authorized stronger-baseline check uses the same circuit and sample streams:

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 .venv/bin/python -u scripts/m3_validate_order7.py
```

Its predeclared plan starts with 2,048 paired low-p shots, extends to 4,096 and
8,192 only if needed, and runs the rest of the grid only after the low-p check
passes. Each segment, per-shot logical outcomes, pairing checks, and gate decision
are saved in a new `results/m3/order7-<timestamp>/` bundle. These are exploratory
development confidence checks; final statistical claims require the later experiments.

The stronger baseline passed at 2,048 shots: 8 logical failures at p=0.001 (0.39%,
95% Wilson interval 0.20%–0.77%), with identical circuit, DEM, and stage-1 fallback
flags to the OSD-0 reference. The full grid is monotonic and exact labels were
verified on 50,000 samples. Evidence is in
`results/m3/order7-20261030T212238643481Z/m3_gate.json`.

## Neural decoder readiness (M4)

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/m4_verify_model.py
```

`NeuralMinSum` supports scalar, exact code-capacity edge-orbit, structural, and full
edge weights. It initializes to min-sum weights, returns differentiable probability
tensors for every unrolled iteration, and freezes corrections at first convergence
during inference. `model.export_cpu()` creates an independent Numba decoder snapshot
with the same stopping and clipping rules.

The configured circuit model has 1,455 parameters (96 structural message groups,
15 iterations). Circuit temporal features use earliest affected detector rounds,
not unique physical gate times. CPU tests cover exact initialization decisions,
gradients, padding, edge cases, and trained-like PyTorch/Numba agreement on 10,000
shots. The readiness command saves an explicitly **untrained** smoke result with
configuration, source hashes, and per-shot corrections. M5 training is next.
