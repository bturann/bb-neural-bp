# Learned BP for bivariate bicycle codes

Research implementation of a neural min-sum first stage with BP-OSD fallback.
The target is fewer fallback calls at preserved logical error performance, measured
against ordinary and tuned-scaling BP. No learned-decoder results exist yet.

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
paired E0 evaluation. The full suite currently has 37 passing tests, including the
slow acceptance checks. Circuit-level integration is the next milestone (M3).

## Code-capacity baseline experiment (E0)

```sh
bash scripts/run_experiment.sh
```

This runs the full scaling sweep on separate validation samples, then evaluates
ordinary and tuned BP, both alone and with BP-OSD fallback, on paired development
shots. It writes the sweep, its config, and a timestamped evaluation directory under
`results/`. Existing tuning evidence is never overwritten. Final-test seeds are
reserved for later milestones; E0 uses a separate development seed range.

Every evaluation includes the resolved YAML, exact-schema CSV, paired per-shot
fallback flags, and diagnostic counts of actual internal OSD executions. `osd_rate`
is actual stage-2 calls divided by shots. For standalone BP it is zero; its potential
fallback demand is `stage1_failures / shots`. Standalone LER counts nonconvergence
as failure. With several decoders, stopping occurs at a shared batch boundary when
all reach the logical-failure target, or at the configured maximum shot count.

Latency is disabled in E0. The primary latency comparison, including compiled neural
inference, belongs to M6. Neural training and circuit-level evaluation are not yet
implemented.

Set `BBNBP_CACHE_DIR` to control logical-operator caching (default `~/.cache/bbnbp`).
The experiment script uses a repository-local cache so it also works in a restricted
workspace. The supplied `reference/` files are immutable copies of the handoff.
