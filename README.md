# Learned BP for bivariate bicycle codes

Research implementation of a neural min-sum first stage with BP-OSD fallback.
The target is fewer fallback calls at preserved logical error performance, measured
against ordinary and tuned-scaling BP. M0–M5 acceptance checks are complete;
circuit-level neural results and the final performance claim remain pending.

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
compiled CPU inference passed their acceptance checks. See `NOTES.md` for the
latest full-suite test count and evidence, including the slow checks.

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
inference, belongs to M6.

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
configuration, source hashes, and per-shot corrections.

## Streamed training and validation (M5)

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -u -m bbnbp.train --config configs/cc_bb72_m5_cpu.yaml
```

The CPU acceptance run streams fresh samples, filters the initial BP's hard cases,
and trains orbit-tied min-sum with the parity multi-loss. Validation uses 20,000
fixed, paired shots per p and the independently tuned scaling table at the same T.
The gate requires improvement at four of seven p values. Final-test seeds are not
used. The process returns nonzero if an acceptance run fails its gate.

Results, scalar event logs, JSONL history, and loss curves are saved under
`results/m5/<run_id>/`; `best.pt`, `last.pt`, and their config are saved under
`checkpoints/<run_id>/`. To resume an interrupted run with the exact same config,
add `--resume checkpoints/<run_id>/last.pt`. Parameter-only restoration preserves
the evaluation problem's current prior probabilities. Checkpoints include generator,
optimizer, and scheduler state. The TensorBoard scalar format is written directly;
no TensorBoard package is required to train. A reader/viewer is not installed here.

`configs/cc_bb72_m5_smoke.yaml` is a short, separately labelled mechanics check.
It cannot establish M5 acceptance.

The completed acceptance run selected step 1,000 and stopped at step 3,000. On the
fixed 20k-shot-per-p validation grid, Neural BP beat tuned BP at four of seven noise
rates (p=0.02–0.05); higher-noise points were worse. Mean fallback demand was
15.184% vs 15.286%. This passes M5's screening rule, with a modest mean improvement.
The final test data have not been used and equal two-stage LER/latency is untested.

Evidence: `results/m5/20261030T220645511485Z-3a224cbf/m5_gate.json` and
`checkpoint_replay.json`. The selected weights are
`checkpoints/20261030T220645511485Z-3a224cbf/best.pt`; use `last.pt` only to continue
the final training state. An artifact audit reproduced every saved outcome on all
140,000 validation shots using the selected weights. M6 is in progress.

## Circuit training and paired evaluation (M6, in progress)

The main configs preserve bb72, six rounds, structural sharing, T=15, three training
seeds (1234–1236), and OSD-CS order 7. Training uses a 5,000-step high-to-low
curriculum, then a uniform p-grid mixture, batch 32, at most 20,000 updates, and
20,000 fixed validation shots per p. Selection uses the canonical compiled CPU
decoder; eight validations without improvement stop training. M6 has not passed.

Run the independent full baseline sweep first, then one training job per seed:

```sh
bash scripts/run_experiment.sh m6-tune
bash scripts/run_experiment.sh m6-train --device cuda --seed 1234
bash scripts/run_experiment.sh m6-train --device cuda --seed 1235
bash scripts/run_experiment.sh m6-train --device cuda --seed 1236
```

Use `--device mps` for local Apple GPU jobs. Deterministic gather-and-sum reductions
avoid unsupported MPS scatter backward operations; the parity loss runs on CPU
with differentiable transfers. No determinism checks are disabled. The Slurm array
template is `scripts/m6_gpu.slurm`; set the site's partition/account and install the
pinned environment before submitting it. No cluster jobs have been submitted here.
Each training command has a four-hour wall-clock deadline and tracks cumulative
GPU budget across resumes. A budget-stopped run remains incomplete and cannot be
fed to E1/E2 through the launcher. Resume adds `--resume <run>/last.pt`.

At the measured local MPS rate, all 20,000 optimizer updates would require at least
9.2 hours per seed before mining and validation. Early stopping may finish sooner;
otherwise a faster accelerator or explicit approval for a larger per-run budget
is required. The four-hour cap is not an experimental success criterion.

After all three training reports are complete, use the selected checkpoints:

```sh
bash scripts/run_experiment.sh e1 --checkpoint <seed1234>/best.pt --checkpoint <seed1235>/best.pt --checkpoint <seed1236>/best.pt
bash scripts/run_experiment.sh e2 --checkpoint <seed1234>/best.pt --checkpoint <seed1235>/best.pt --checkpoint <seed1236>/best.pt
.venv/bin/python -m bbnbp.m6_report --e1 <E1-output-directory> --e2 <E2-output-directory> --output results/m6/summary.json
```

E1 pairs all five decoders on the same held-out shots until every decoder reaches
100 logical failures or 10 million shots. E2 uses 1,000 warmups and 20,000 measured
shots at each of p=0.001 and 0.003, single thread, batch one, compiled Numba versus
ldpc C++. Run it on an otherwise idle machine. It saves total, Stage-1, fallback,
and overhead timing samples, with mean/p50/p99/max in the CSV. Fallback time includes
the fresh BP+OSD call. Timing samples use a separate deterministic sampling stream;
warmups do not enter reported counts. GPU throughput is a separate Stage-1 regime:

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/m6_throughput.py --device cuda --checkpoint <seed1234>/best.pt --checkpoint <seed1235>/best.pt --checkpoint <seed1236>/best.pt
```

The summary reports mean and sample standard deviation of fallback rate across
training seeds. LER counts and Wilson intervals remain separate per model because
their shared test shots cannot be pooled as independent observations.

Local development evidence is under `results/m6_development/`; preflight receipts
are under `results/m6_preflight/`. These short checks use distinct development
seeds and establish mechanics only. They are not the three main models or final
accuracy/latency evidence.

## Bounded CPU learning pilot

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -u scripts/m6_cpu_pilot.py
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/m6_summarize_cpu_pilot.py <printed-pilot-bundle-directory>
```

This development pilot uses the full six-round circuit and T=15 structural model,
batch 8, up to 500 updates, a 125-update curriculum, and 256 fixed validation shots
at each p. It stops the training process after ten minutes, allowing a short cleanup
interval. Logs, selected/continuation checkpoints, loss curves, source hashes and a
CPU profile are retained. The summary replays the selected checkpoint and compares
it with the initial model and the coarse development tuning table. Profiled timing
includes instrumentation overhead; batch-8 costs do not estimate batch-32 costs.
This pilot uses separate development streams and cannot establish M6 acceptance.

## Independent check of the CPU pilot

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -u scripts/m6_check_cpu_pilot.py
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/m6_audit_cpu_check.py <printed-check-bundle-directory>
```

This freezes the selected pilot checkpoint, tunes all 11 BP scaling factors at T=15
on 1,024 new shots per p, then compares all three Stage-1 decoders on 2,048 different
paired shots per p. The study has a 15-minute wall-clock cap. Raw cases, per-shot
nonconvergence/logical flags, configs, source hashes, the tuning table, and a copy of
the checkpoint are retained. CSV `osd_calls` and `osd_rate` are zero because this
check makes no Stage-2 calls; potential fallback demand is `stage1_failures/shots`.

The completed independent check found mean neural fallback demand of 88.574% versus
88.750% for tuned BP. Neural BP improved at three p values, worsened at one, and
tied at one. None of the five paired differences passed Holm correction at .05.
This did not meet the predeclared screen for proceeding with CPU optimization.
The earlier pilot demonstrated learning relative to initialization; a clear
advantage over the stronger tuned baseline remains unestablished.

## Stronger-initialization diagnostic

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -u scripts/m6_cpu_pilot.py --config configs/m6/cpu_strong_init.yaml --output-root results/m6_cpu_strong_init --checkpoint-root checkpoints/m6_cpu_strong_init
```

The .95-initialized CPU variant uses fresh training/validation streams and the
existing 11-factor tuning table. Its matched untrained reference distinguishes
the starting point from learned changes. The completed 500-update run selected
step 300. On a new 10,240-shot paired comparison, fallback demand was 88.398% for
the selected model, 89.072% for its untrained .95 reference, and 88.936% for tuned BP.
All five observed differences versus tuned BP favored the selected model, but none
passed Holm correction. The learning contrast also did not pass corrected .05.
These are development findings; main configs and final-test seeds are unchanged.

`scripts/m6_check_cpu_pilot.py --reuse-tuning <table.csv>` validates the original
table's resolved-config sidecar and row hashes, so independent evaluation configs
can reuse valid tuning without changing its provenance. The strong-initialization
check budget also enables `scripts/m6_initialization_reference.py`. The recorded
check config points to the saved selected checkpoint; a new training invocation
creates a new checkpoint path that must be set in its comparison config.

## Controlled parity-versus-BCE loss pilot

Run the two training commands concurrently, each on one CPU thread:

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -u scripts/m6_cpu_pilot.py --config configs/m6/cpu_loss_parity.yaml --output-root results/m6_cpu_loss_ablation/parity --checkpoint-root checkpoints/m6_cpu_loss_ablation/parity &
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -u scripts/m6_cpu_pilot.py --config configs/m6/cpu_loss_bitwise.yaml --output-root results/m6_cpu_loss_ablation/bitwise --checkpoint-root checkpoints/m6_cpu_loss_ablation/bitwise &
wait
```

Pass the printed bundle paths to the controlled comparison and audit:

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/m6_loss_ablation.py --parity <parity-bundle> --bitwise <bitwise-bundle>
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/m6_audit_loss_ablation.py <comparison-bundle>
```

Both arms use the same .95 initialization, 500-update budget, frozen mining model,
training stream, and checkpoint-selection cases. Per-update input hashes and ending
generator states must match before evaluation. Fresh cases are shared by ordinary
BP, tuned BP, matched untrained neural BP, and both selected learned models. The
primary paired loss contrast uses Holm correction over five p values; secondary
families each correct both losses across all five values. Main final-test data are
reserved. The complete study is bounded to 15 minutes, with per-stage limits.

The completed pilot found fallback demand of 88.350% for parity and 88.525% for BCE,
versus 88.467% for matched initialization and 88.672% for tuned BP. No primary or
secondary point passed corrected .05. This is insufficient evidence to favor a
different loss or claim useful extra learning; the planned parity default remains.

## Objective-alignment and checkpoint review

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/m6_objective_diagnostic.py
.venv/bin/python scripts/m6_summarize_objective.py <printed-diagnostic-bundle>
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/m6_checkpoint_review.py --output <new-review-directory>
```

These tools reuse saved development cases for diagnosis. They measure both losses,
first-versus-final convergence, probability clipping, and final-output gradients
for initialization, selected checkpoints and final checkpoints. Failure-ranking
AUC is computed within each p only. The first-convergence trace must match Numba.
The separate review measures final checkpoints on all saved comparison cases and
preserves the original checkpoint selections. Neither tool trains or uses final tests.

The check found an informative loss ranking and correct first-convergence inference,
alongside non-monotonic changes in loss and hard decoding. The retrospective BCE
checkpoint ranking reversed between the tiny selection set and the larger saved
comparison. A larger fixed development validation set is the next stability check;
the planned main runs already use 20k shots per p.

## Larger fixed validation for checkpoint stability

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/m6_validation_stability.py
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/m6_audit_validation_stability.py <validation-bundle>
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/m6_plot_validation_stability.py <validation-bundle>
```

This compares existing selected/final checkpoints on 2,048 fixed new cases per p,
eight times the pilot selection pool. It does not retrain or overwrite weights.
Eight disjoint smaller blocks and a seeded, paired bootstrap describe how rankings
change with sampling. Original and resolved configs are saved separately, and the
CSV hash identifies the resolved sampling and candidate configuration.

The corrected completed run favored BCE step 500 over its old step-50 selection:
88.916% versus 89.609% fallback demand, with a descriptive paired interval excluding
zero. Parity's 0.303-percentage-point difference remained uncertain, and its small
blocks split 4–4. `validation_preferences.json` retains candidate hashes and the
validation cases used for those preferences. They require independent confirmation;
the original selected checkpoints and main experiment choices are unchanged.
