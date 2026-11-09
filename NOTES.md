# Research and implementation log

## Scope

The governing handoff is `../02_Coding_Agent_Implementation_Spec.md` (2026-10-30).
No research claims have been established. Results must come from executed code.

## M0 — environment and API verification (complete)

- The supplied `codebase/` was empty and was not a Git repository.
- System Python is 3.12; the specification requires Python 3.11. Create an isolated
  Python 3.11 environment rather than modify system Python.
- Sandboxed `uv` initially failed accessing its default cache, then panicked in the
  macOS system-configuration library with a writable cache. Retried environment
  setup through the execution tool's approved escalation path.
- Only specification-listed direct dependencies are requested. Dependencies brought
  in by those packages are transitive dependencies, not new research dependencies.
- Sources checked: https://github.com/mkangquantum/quits and
  https://software.roffe.eu/ldpc/ldpc/bposd_decoder.html . Installed API smoke tests
  below take precedence over documentation sketches.
- Installed QUITS 1.2.0 from its published distribution, rather than a floating Git
  branch. The installed project metadata and BB builder match the upstream project.
- `requirements.in` preserves the allowed direct dependency bounds;
  `requirements.txt` pins the resolved direct dependencies and `environment.lock.txt`
  pins all 37 resolved distributions for this macOS ARM environment.
- Python 3.11.14 editable package installation succeeded. The executable API audit
  passed, including a QUITS bb72 smoke circuit with 12 observables. Full six-round
  circuit and noise-model verification is still required at M3.
- Validation before the M0 commit: `python -m pytest -q`: **37 passed** (49.50 s).
  This includes the prepared M1/M2 tests, but does not establish M2 experiment gates.
- Reference files and code/baseline modules were prepared while the package downloads
  ran (15 minutes). Milestone acceptance and commits remain sequential.

## Issues to resolve at the relevant milestone

- The reference min-sum implementation does not clip messages. Initialization
  equivalence must distinguish the unclipped reference from configured clipping.
  M4 verified exact hard decisions at the configured T=15 with clip=20; larger
  iteration budgets require their own checks before use.
- Reference batch convergence differs from first-convergence freezing in inference.
  M4 tests explicitly distinguish per-iteration forward outputs from frozen inference.
- Overlapping LER confidence intervals alone do not establish statistical equivalence.
- The M3 low-noise test is stochastic; a nonzero noise rate does not guarantee zero
  detection events. Do not turn an expected low failure rate into a deterministic claim.

<!-- M0 API REPORT START -->
## M0 executed API report

Platform: macOS-26.6.2-arm64-arm-64bit
Python: 3.11.14

### Installed versions

- numpy==2.4.6
- scipy==1.17.1
- stim==1.16.0
- ldpc==2.4.1
- torch==2.15.0
- numba==0.68.1
- pyyaml==6.0.3
- pandas==3.1.0
- matplotlib==3.11.2
- pytest==9.2.0
- tqdm==4.70.1
- quits==1.2.0

### BpDecoder

```text
Installed stub: def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float]=None, error_channel: Optional[Union[np.ndarray, List[float]]]=None, max_iter: Optional[int]=0, bp_method: Optional[str]='minimum_sum', ms_scaling_factor: Optional[Union[float, int]]=1.0, schedule: Optional[str]='parallel', omp_thread_count: Optional[int]=1, random_schedule_seed: Optional[int]=0, serial_schedule_order: Optional[List[int]]=None, input_vector_type: str='auto', random_serial_schedule: bool=False, **kwargs):
```
Constructor smoke test passed with {'error_channel': [0.03, 0.03, 0.03], 'max_iter': 15, 'bp_method': 'minimum_sum', 'ms_scaling_factor': 0.625, 'schedule': 'parallel', 'omp_thread_count': 1}.
Convergence attribute: `converge` (True); iterations: `2`.

decode: `(input_vector: 'np.ndarray') -> 'np.ndarray'`

### BpOsdDecoder

```text
Installed stub: def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float]=None, error_channel: Optional[List[float]]=None, max_iter: Optional[int]=0, bp_method: Optional[str]='minimum_sum', ms_scaling_factor: Optional[Union[float, int]]=1.0, schedule: Optional[str]='parallel', omp_thread_count: Optional[int]=1, random_schedule_seed: Optional[int]=0, serial_schedule_order: Optional[List[int]]=None, osd_method: Union[str, int, float]=0, osd_order: int=0, input_vector_type: str='syndrome', **kwargs):
```
Constructor smoke test passed with {'error_channel': [0.03, 0.03, 0.03], 'max_iter': 15, 'bp_method': 'minimum_sum', 'ms_scaling_factor': 0.625, 'schedule': 'parallel', 'omp_thread_count': 1, 'osd_method': 'osd_cs', 'osd_order': 2}.
Convergence attribute: `converge` (True); iterations: `2`.

decode: `(syndrome: 'np.ndarray') -> 'np.ndarray'`

### External soft information for OSD

`BpOsdDecoder.decode` accepts a syndrome only. Its public API exposes posterior
`log_prob_ratios` as output and `update_channel_probs` for input channel priors,
but no documented standalone OSD method taking external posterior LLRs.
Stage 2 will reuse a separate BpOsdDecoder with fixed problem priors; it runs its
own BP and invokes OSD internally only if needed. Report stage-2 calls separately
from claims about actual internal OSD executions.

### Stim DEM sampler

```text
sample(self: stim._stim_polyfill.CompiledDemSampler, shots: int, *, bit_packed: bool = False, return_errors: bool = False, recorded_errors_to_replay: object = None) -> object
```
`return_errors=True` passed; tuple order is detectors, observables, errors.

### QUITS BB circuit entry point

`quits.qldpc_code.BbCode(...).build_circuit(strategy='custom', ...)`

```text
(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows)
(strategy='custom', error_model=None, num_rounds=0, basis='Z', circuit_build_options=None, **opts)
(idle_error: Union[float, Tuple[float, float, float]] = 0.0, sqgate_error: Union[float, Tuple[float, float, float]] = 0.0, tqgate_error: Union[float, Tuple[float, ...]] = 0.0, spam_error: float = 0.0) -> None
```
bb72 one-round smoke circuit: 108 detectors, 12 observables, 792 error instructions.
Full noise placement and six-round acceptance remain M3 work.

Numba compiled CPU smoke test and PyTorch autograd smoke test passed.

**M0 acceptance: PASS.**
<!-- M0 API REPORT END -->

## M1 — codes and reference (acceptance passed)

- Copied both supplied reference files byte-for-byte. Their tests pass unchanged.
- Copied the GF(2) and BB construction routines from the reference. Both codes have
  k=12, commuting CSS checks, check weight 6, variable degree 3, and full-rank
  logical X/Z pairings. Tests verify both bases and the independent logical rows.
- All l*m simultaneous check/qubit translations preserve both parity-check matrices.
  Union-find identifies **6 edge orbits** for each matrix: 36 edges per orbit for
  bb72 and 72 per orbit for bb144.
- Logical operators are cached with matrix validation and atomic writes. The public
  default remains `~/.cache/bbnbp`; `BBNBP_CACHE_DIR` permits a writable workspace
  cache. Tests use isolated temporary caches.
- Additional tests cover GF(2) rank/nullity and rejection of broken symmetries.
- Validation before the M1 commit: full suite **37 passed** (6.95 s), including
  the slow-marked statistical acceptance tests.

## M2 — code-capacity data and baselines (acceptance passed)

- Implemented strict frozen configuration sections, canonical SHA-1 config hashes,
  resolved YAML output, binary sparse decoding problems, explicitly seeded samplers,
  reusable ldpc adapters, and the two-stage pipeline. Convergence flags are checked
  against the actual syndrome. The tests cover X/Z bookkeeping and both code sizes.
- Added Wilson intervals, shared-shot multi-decoder stopping, exact paired McNemar
  tests, schema-checked CSV writing, source checksums, and source Git provenance.
- Full test suite: **37 passed** (7.01 s), including T2.x/T3.x, both reference-rate
  calibration points, and slow-marked tests. Reference nonconvergence agreement is
  within the requested 3-sigma band and the p=0.04/0.06 sanity rates are within 30%.
- Validation sweep executed: `results/tuned_scaling.csv`, **231 configurations**
  (7 p values × 3 iteration budgets × 11 factors), 5,000 paired validation shots
  per p. Selection is independent for each (p,T); ties prefer lower stage-1 LER,
  then the lower factor. All candidates, selected flags, configs, and source hashes
  are saved. No final-test seeds were used.
- E0 executed: `results/20261030T191424331467Z-0c364ef1/`. It contains **28 rows**
  (four decoders × seven p values), seven per-shot pairing archives, a resolved
  config, a copy of the tuning table, diagnostics, and source checksums. Shared
  shot counts total 18,432 per decoder across the p grid. Every decoder reached
  100 logical failures at every p; the configured 20,000-shot development cap was
  not reached. Integrity checks verified counts, CI bounds, and schema.
- Example development result at p=0.04, T=50, 2,048 shared shots: BP+OSD fallback
  153/2048 = 7.47%; tuned BP+OSD fallback 77/2048 = 3.76%. LERs are 171/2048 and
  164/2048 respectively, with Wilson intervals in the CSV. This establishes a
  strong tuned baseline; it is not evidence for neural improvement or equivalence.
- Runs were made from the M1 commit plus the pending M2 source files, so the
  CSVs correctly identify a dirty source tree. Exact source hashes accompany them;
  results were not altered to insert the later commit SHA.

### Explicit M2 conventions and deviations

- E0 uses development evaluation seeds 3234–3240, validation tuning seeds 2234–2240,
  and a 20k shot cap. The final seed region beginning at seed+10,000 is untouched.
- Paired stopping keeps all decoders running on the same samples until all reach
  their target, or the shared shot cap. Final-batch overshoot is documented.
- The CSV's `osd_calls` and `osd_rate` count stage-2 invocations as specified. Actual
  internal OSD executions are separate diagnostics. Standalone BP makes no stage-2
  calls; its potential fallback demand is `stage1_failures/shots`, and its LER
  counts nonconvergence as failure.
- Tuning uses validation shots only and picks a factor separately for each fixed T;
  evaluation never silently changes the iteration budget to obtain a better rate.
- E0 disables latency sampling. Fair compiled neural-vs-BP timing and timing splits
  remain M6 work, alongside the learned decoder. No GPU training has been launched.

## Next milestone after the original M2 checkpoint

M3: implement the QUITS adapter with explicit noise placement and round semantics,
DEM conversion preserving fault-column order, exact fault-label sampling, and the
six-round circuit-level acceptance tests. The M0 smoke circuit alone does not pass M3.

## M2 audit corrections (2026-10-30)

- Reproduced the default experiment command's failure on the already committed
  tuning CSV. The command now creates a timestamped E0 bundle with its own tuning
  table and evaluation output. A regression invokes it twice and checks that all
  first-run files retain identical bytes.
- Reproduced acceptance of overlapping tuning/development RNG streams by supplying
  a tuning table from another configuration. Evaluation now checks every saved
  tuning seed, including unselected candidates, against all evaluation seeds for
  both development and final runs. The original E0 streams remain disjoint.
- Seed offsets must now be nonnegative integers; floats and booleans are rejected
  before any experiment starts.
- Regression and full acceptance suite: **48 passed** (11.82 s). No existing results
  or supplied reference files were edited.

## M3 implementation and development checks (gate not passed)

- QUITS 1.2.0's `num_rounds` means cycles after its zeroth cycle. The adapter maps
  six total extraction cycles to `num_rounds=5`, sets noisy zeroth preparation and
  noisy final data measurement explicitly, and retains detectors from both bases.
- Noise is uniform at rate p: DEPOLARIZE1 after H and on idle qubits,
  DEPOLARIZE2 after CX, basis-appropriate preparation/measurement flips. This is
  QUITS' SPAM convention, not SI1000. Its parity-check matrices match the reference.
- Exact-label smoke checks passed on 1,000 bb72 shots: H has shape (432, 16164),
  L has shape (12, 16164), and every fault instruction remains a separate column.
- QUITS supplies no detector coordinates for this builder. Detector round and basis
  metadata is derived from its verified order: boundary block, paired Z/X blocks
  per subsequent extraction cycle, final boundary block.
- Performance probes on 128 identical p=0.001 shots: BP+OSD-CS order 7 took 49.21 s;
  BP+OSD-0 took 5.09 s. Both had 0 logical failures and 91 stage-2 calls in that
  small probe. At p=0.005 the order-7 probe took 90.51 s and had 76 failures on
  128 shots. These are development wall times, not fair per-shot latency results.
- The M3 development configuration explicitly uses OSD-0, 20 target failures and a
  5,000-shot cap to check the circuit pipeline at manageable CPU cost. The main
  experiment's OSD-CS order-7 setting is unchanged in the base configuration.
- The first strengthened noise-placement test failed because it assumed separate
  gate/idle DEPOLARIZE1 instructions. Stim coalesces equal-rate instructions. The
  assertion now checks coverage of the gate targets; circuit construction was not
  changed to satisfy the test.
- T4.3 is interpreted physically: p=0 must yield no detectors/observables, and
  decoded logical failures at p=1e-6 are checked on 10,000 shots. Nonzero p cannot
  guarantee zero raw detector events or raw logical-observable flips.
- Rare-column empirical-rate checks use simultaneous exact binomial intervals
  (99% family coverage) plus a pooled 5-sigma check, avoiding spurious failures
  from Gaussian approximations across thousands of low-probability columns.

M3 gate failed for `results/m3/20261030T210408544859Z-ae88f467`: monotonic=True, low-p LER=0.009765625 (Wilson upper 0.015036263050956185). Stop at the M3 gate and review this measured evidence.

### M3 outcome

- Full unit/integration/reference suite: **60 passed** (17.32 s), including slow
  tests, exact identities on 10,000 six-cycle samples, zero-noise detectors and
  logicals, and zero decoded logical failures on 10,000 p=1e-6 samples.
- The development runner separately checked exact H/L fault labels on **10,000
  shots at each of five p values** and identical H/L structure throughout the grid.
  bb72 has 91,116 Tanner edges, maximum check degree 242, 432 detectors, and 16,164
  DEM fault columns under the recorded circuit policy.
- The executed curve is saved under `results/m3/20261030T210408544859Z-ae88f467/`,
  with the resolved config, source checksums, five circuits, five DEMs, per-shot
  fallback flags, problem metadata, diagnostics, and `m3_gate.json`.
- The OSD-0 development curve is monotonic. At p=0.001 it has **20/2048 logical
  failures**: LER 0.009765625, Wilson interval [0.0063306211, 0.0150362631]. This
  does not meet the operationalized low-noise criterion (95% upper bound < 0.01),
  and the point estimate is not well below 0.01 either.
- **M3 is not accepted.** Successful code/label tests do not override this failed
  research gate. No M4 model or neural training was started after the failure.
- The stronger, specified OSD-CS order-7 full curve is still pending. These OSD-0
  results do not demonstrate that the stronger baseline or research hypothesis
  fails. The next diagnostic for review is that stronger baseline under the same
  recorded circuit policy; do not silently weaken the noise or change the gate.
- No experiments were dropped and no final-test seeds were touched. All previous
  E0 results and reference sources remain unchanged. The source tree was dirty
  while M3 ran, as identified in its CSV and source manifest.

## M3 stronger-baseline validation plan (authorized continuation)

- The human authorized running the specified OSD-CS order-7 baseline after review
  of the failed OSD-0 gate. Only the OSD method/order changes; circuit policy, BP
  settings, p grid, and per-p RNG streams stay matched to the original run.
- Predeclared plan: `configs/m3_order7_validation.yaml`. First decode the same 2,048
  p=0.001 shots (seed 4234, batch 128). If the Wilson interval straddles 0.01,
  extend the same stream to 4,096, then 8,192 total shots, without decoding earlier
  shots again. Stop earlier if the interval is wholly below or above the threshold.
- These repeated Wilson checks are an exploratory development gate, not an
  anytime-valid test, an equivalence claim, or final paper evidence. All intermediate
  outcomes and the predeclared cap are retained.
- If the low-p check passes, evaluate the remaining four p values using original
  seeds 4235–4238. Then recheck exact fault labels across all five points and the
  monotonic curve. A failed/inconclusive low-p gate does not authorize M4.
- Evaluation now saves per-shot logical outcomes and emits optional batch progress.
  `sample_offset` skips whole sampling batches without decoding them. A regression
  verifies that concatenated segments reproduce continuous per-shot fallback and
  logical arrays. Historical zero-offset config hashes remain unchanged.
- Validation infrastructure and existing suite: **65 passed** (16.08 s).

M3 order-7 validation passed: `results/m3/order7-20261030T212238643481Z`. Low-p summary: {'shots': 2048, 'logical_failures': 8, 'ler': 0.00390625, 'ler_lo': 0.0019806786976977143, 'ler_hi': 0.007689395340136621, 'fallback_calls': 1450, 'source_runs': ['20261030T212238645326Z-0ffbb368']}.

### M3 stronger-baseline outcome — acceptance passed

- The specified OSD-CS order-7 baseline passed at the first, predeclared checkpoint:
  **8/2048** failures at p=0.001, LER **0.00390625**, Wilson interval
  **[0.0019806787, 0.0076893953]**. The extensions were not used.
- Both circuit and DEM files are byte-identical to the OSD-0 reference, and all
  2,048 stage-1 fallback flags are identical (1,450 calls). The sampling seed and
  batch size are unchanged. The order-0 reference had 20 failures on these shots.
- Remaining grid, in increasing p order: 20/384 failures at 0.002, 27/128 at 0.003,
  60/128 at 0.004, and 68/128 at 0.005. LER is monotonic. These development points
  retain their Wilson intervals and do not constitute the final E1 experiment.
- Exact H/L labels reverified on 10,000 independent verification samples per p,
  totaling 50,000. Matrix structure is unchanged across p.
- `results/m3/order7-20261030T212238643481Z/m3_gate.json` records **passed=true**;
  raw segment CSVs/configs, logical outcomes, fallback flags, circuits, DEMs,
  source hashes, and the predeclared plan are retained. Both runs identify the
  clean source revision 840db5b.
- The earlier OSD-0 failure remains intact. Acceptance uses the stronger baseline
  originally specified, with the same noise policy and low-p threshold.

## M4 — neural min-sum model (acceptance passed)

- Implemented padded Tanner tensors, deterministic CSR edge order, scalar/orbit/full
  maps, and structural maps using check degree, variable degree, and temporal offset.
  Padding is masked and degree-zero/one checks are handled explicitly.
- For circuit DEMs the temporal anchor is the earliest detector round affected by
  the mechanism. A DEM mechanism can represent multiple physical fault locations,
  so this is an explicit reproducible structural feature, not a claimed physical
  gate time or exact translation orbit. The source of detector round labels is
  the verified QUITS instruction layout recorded at M3.
- `NeuralMinSum` unrolls T flooding updates, with learnable per-iteration edge-group
  weights and prior weights. Initialization is w=0.625, alpha=1, with clip=20. The
  posterior uses the unweighted prior. Default computation is float32.
- Inference freezes each shot at its first matching syndrome; zero syndromes return
  zero corrections at iteration zero. Training forward returns all T probability
  tensors. Hard decisions use posterior LLR signs directly, avoiding sigmoid
  rounding around probability 0.5.
- T5.1: default clipped initialization at T=15 matches the immutable NumPy reference
  hard decisions **at every iteration**, on 500 p=0.03 and 500 p=0.06 code-capacity
  shots, in float64. A runtime copy adds the trace hook; reference files are unchanged.
- T5.2/T5.3: dummy-zero-row invariance, finite/nonzero gradients for every parameter
  under scalar/orbit/full tying, and degree-one/zero-message cases pass.
- T5.4: Numba inference snapshots reproduce PyTorch corrections, convergence flags,
  and iteration counts on **10,000** code-capacity shots with trained-like random
  weights. Float32 scalar/full exports, circuit structural exports, and per-shot
  freezing are also tested. A six-cycle float32 structural model with randomized
  weights and heterogeneous circuit priors passes an additional 64-shot equivalence check.
- Both neural and compiled stages work in the shared BP-OSD pipeline. Weight exports
  are independent snapshots, and both provide the Decoder protocol.
- Configured six-cycle readiness run: `results/m4/20261030T214824647527Z/`. T=15,
  structural tying yields **96 groups and 1,455 parameters**. The padded message
  tensor is 53,526,528 bytes at the configured 128-shot evaluation batch. PyTorch
  and Numba matched exactly on all 128 untrained smoke shots; 29 converged.
- Full acceptance/reference suite: **84 passed** (31.09 s), including slow checks.
  All work was CPU-only, and no training or learned-performance claim is made.
- Next is M5: parity/BCE losses, streamed training, validation/checkpoint selection,
  and the code-capacity gate against tuned BP. Checkpoint restoration must preserve
  the evaluation problem's current p-specific prior LLRs, rather than overwrite
  them with a training p's saved buffer.

## M5 implementation and predeclared CPU run (in progress)

- Implemented log-space degeneracy-aware parity multi-loss and the bitwise BCE
  ablation. The parity-product backward pass analytically handles exact zero factors
  (probability 0.5), preserving gradients that a naive clamped log implementation
  would discard. Float64 gradcheck and degenerate-equivalence tests pass.
- Added streamed sampling, high-to-low p curriculum, off/filter/upweight mining,
  Adam, cosine decay, gradient clipping, fixed paired validation, best/last parameter
  checkpoints, and deterministic resume. Evaluation priors are constructed from the
  evaluation problem and never replaced by a training p's checkpoint buffer.
- Checkpoints contain optimizer/scheduler and explicit NumPy/Torch generator states.
  A simulated interruption after two updates resumes to bit-identical learned
  parameters and RNG state relative to an uninterrupted four-update test.
- No CUDA/MPS accelerator is available. The acceptance config is
  `configs/cc_bb72_m5_cpu.yaml`: bb72 code capacity, orbit sharing, T=15, batch 256,
  at most 4,000 updates, a 1,000-update p=0.08→0.03 curriculum then a uniform p-grid
  mixture, filtered hard examples (50-draw cap), lr 1e-3→1e-5, and clip norm 1.
- Predeclared selection/gate: validate every 250 updates on 20,000 fixed shots at
  each of seven p values; select lowest mean Stage-1 nonconvergence; stop after eight
  validations without improvement. M5 passes only with lower nonconvergence than
  tuned BP at at least four points. Both use T=15 and see the same validation shots.
- Validation seeds 1734–1740 are separate from the tuning table's 2234–2240 and
  final-test region. Training uses explicit SeedSequence entropy [1234, 701].
- TensorBoard itself is not installed and is outside the handoff's direct dependency
  list. A standard-library writer emits the scalar Event/TFRecord format without
  adding dependencies; CRC32C and wire framing are tested against the published
  TensorFlow schema. Actual TensorBoard-reader integration has not been run here.
  JSONL training/validation records and rendered PNG/PDF curves are also saved.
- Sources for the scalar event format:
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/core/util/event.proto ,
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/core/framework/summary.proto ,
  https://www.tensorflow.org/tutorials/load_data/tfrecord .
- Full mechanics/reference suite: **97 passed** (21.68 s). The separately labelled
  50-update smoke run at `results/m5_smoke/20261030T220420918152Z-68bce5e1/` completed
  in 6.73 s, wrote finite losses/checkpoints/events/curves, and did not pass a research
  gate (purpose=smoke). Its curves were visually inspected. The main run is pending.

M5 passed: `results/m5/20261030T220645511485Z-3a224cbf`; wins 4/7, best step 1000, mean fallback 0.15184285714285714. Final-test data untouched.

### M5 outcome — acceptance passed

- The predeclared CPU run stopped at step **3,000** after eight validations without
  improvement; the selected checkpoint is step **1,000**, not the final weights.
  Wall time was **849.11 s** (14.15 min), with zero GPU use.
- On 20,000 fixed validation shots per p, selected model vs tuned BP fallback demand:

  | p | Neural BP | Tuned BP | Neural failures / shots |
  |---|---|---|---|
  | 0.02 | 0.665% | 0.965% | 133 / 20000 |
  | 0.03 | 2.630% | 3.510% | 526 / 20000 |
  | 0.04 | 6.390% | 7.205% | 1278 / 20000 |
  | 0.05 | 12.030% | 12.585% | 2406 / 20000 |
  | 0.06 | 19.940% | 19.665% | 3988 / 20000 |
  | 0.07 | 27.815% | 27.405% | 5563 / 20000 |
  | 0.08 | 36.820% | 35.670% | 7364 / 20000 |

- **4/7 strict wins** satisfy the stated M5 gate. Mean fallback demand is
  **0.1518428571 vs 0.1528642857** (a modest 0.102 percentage-point reduction).
  Higher-noise points are worse; this is not a uniform improvement claim.
- The 105-parameter code-capacity orbit model is saved at
  `checkpoints/20261030T220645511485Z-3a224cbf/best.pt`; `last.pt` preserves the
  step-3000 continuation state. Configs, event logs, complete training/validation
  history, per-shot flags, paired comparisons, and PNG/PDF loss curves are retained
  under `results/m5/20261030T220645511485Z-3a224cbf/`.
- The run streamed **9,249,280** candidate shots and used **745,060** selected hard
  shots for optimization. **281** batches hit the configured 50-draw cap; their
  actual smaller effective batch sizes are recorded. No easy samples were silently
  substituted and no extra training runs were selected after seeing this outcome.
- Independent artifact audit: reloaded `best.pt` separately at all seven p values,
  verified each current prior vector, and reproduced **all 140,000** saved per-shot
  convergence/logical outcomes and mean iteration counts exactly. Checkpoint SHA-256:
  `90a98815b7f4794ea701f78e6d0cef2e1f48886772ca3d72ac4de3152a120197`.
- The gate report and replay audit agree on best step, all wins, stopping reason,
  and source revision 803a391. Existing M3/reference/E0 evidence is unchanged.
- Full suite before the outcome commit: **97 passed** (21.12 s). Main loss curves
  were visually inspected and are legible.
- Statistical scope: the same validation set selected the checkpoint across repeated
  validations. Paired p-values are exploratory diagnostics, not post-selection
  confirmatory inference. Final-test seeds have not been touched. No claim of equal
  two-stage LER, circuit-level neural improvement, or latency speedup is made yet.
- M3 and M5 are now both passed, so the go/no-go permits M6. Next work is
  circuit-level training with the main three seeds, followed by held-out paired
  BP/tuned-BP/NBP+OSD accuracy and compiled CPU latency evaluation.

## M6 implementation and development checks — in progress

- Corrected the earlier accelerator discovery: inside the restricted process,
  `torch.backends.mps.is_available()` returned false. An approved process outside
  that restriction detects MPS successfully. Hardware is Apple M4 Pro, 12 logical
  CPUs, 24 GiB RAM; CUDA is unavailable. The historical M5 CPU run is unchanged.
- The first deterministic MPS backward probe failed because
  `index_put_with_accumulate_mps` lacks a deterministic implementation. Added
  fixed inverse-index gather/sum autograd operations for MPS, with their gradients
  checked against CPU gather/index-add. Determinism remains enabled. MPS priors
  use float32 because it cannot hold float64 buffers; canonical CPU inference
  retains the original prior precision. CPU/CUDA message reductions are unchanged.
- The parity loss runs on CPU for MPS, with differentiable transfers. All draws
  still use explicit configured generators. No new dependencies were added.
- Receipts in `results/m6_preflight/` measure six-round bb72, structural T=15:
  CPU batch-32 forward/loss/backward was 2.253 s; MPS warmed batch-32 was 1.658 s,
  and batch-8 was 0.437 s. Repeated MPS gradients were bit-identical and finite at
  both batch sizes. This excludes optimizer, mining and validation overhead.
  A nominal 20k-update run is therefore at least 9.2 local GPU-hours per seed at
  batch 32; an early-stopped run could finish sooner. Three full CPU runs project
  about 38 hours of forward/loss/backward alone. These are estimates, not outcomes.
- Batch 32 limits memory on the 16,164-variable circuit problem (CPU peak RSS
  about 1.87 GB in the batch-32 probe); the handoff's example GPU batch 2,048 is
  unsuitable for this local memory budget. The research question and E1/E2 matrix
  have not changed.
- Main CUDA/MPS configs are prepared for seeds 1234–1236, a 5k-step curriculum,
  at most 20k updates, 20k fixed validation shots per p, and canonical Numba CPU
  checkpoint selection. The three seeds share validation sets 1734–1738 and have
  independent training streams. Eight validations without improvement stop a run.
  The full 11-factor, T=15/50/100 independent scaling sweep is configured separately.
- `scripts/m6.py` and the Slurm array template prepare explicit stages. Each training
  launch has a four-hour external deadline, with cumulative GPU budget on resume.
  Budget exhaustion reports incomplete training; it cannot masquerade as success.
  E1/E2 launchers require three selected main checkpoints and completed reports.
  The real development checkpoint was rejected by this guard. No cluster jobs or
  full main runs have been launched; no spending above four GPU-hours is authorized.
- A configuration-generation check initially rejected the two-point E2 config
  because an irrelevant inherited M5 four-point training gate was present. Fixed
  evaluation-only configs to omit training runtime. All experiment configs load.
- Development tuning uses only three factors, T=15, 128 shots/p, seeds 9234–9238.
  It is stored under `results/m6_development/` and will never substitute for the
  main full sweep. The 15-step MPS mechanics run completed in 20.80 s and selected
  step 10: `training/20261030T231030592263Z-5d4b322e/`. All losses were finite and
  its plotted curves were visually checked. Selected validation fallback was
  0.925 versus the coarsely tuned baseline's 0.890625, with zero strict wins on
  five 64-shot points. This short run makes no circuit-improvement claim.
- Selected development checkpoint SHA-256:
  `ad665b148baed57f8f5921208f2d73e4b06d0483384af0c72b3ca6e1e029e37e`.
  Both selected and continuation checkpoints retain portable parameters/RNG state.
- Development accuracy exercises all five p values, 32 shared shots/p, three
  decoders, and order-7 fallback. Development latency exercises p=.001/.003,
  four warmups plus 16 measured shots per decoder, including exact stage/overhead
  accounting. It ran alongside other development work and is unsuitable for
  performance claims. GPU throughput was separately exercised on 64 shots/p.
- Main E1 keeps paired failure stopping (100 failures/all models, max 10M shots).
  E2 keeps 1,000 warmups and 20k measured shots/p, batch one and single CPU thread,
  comparing compiled Numba with ldpc C++. Its fixed timing sample supplies its
  own counts, avoiding a duplicate failure-stopping decode. All models see the
  same latency syndromes from an independent tagged RNG stream; warmups are excluded.
  Fallback timing includes fresh BP plus OSD. Raw component samples are retained.
- The main summary computes mean and sample standard deviation across three
  independently trained models. Per-model LER/Wilson intervals remain separate;
  shared test shots are never pooled as independent samples across model seeds.
- M6 remains **incomplete**: full independent tuning, three main training runs,
  held-out E1 and full E2, and main secondary throughput are still required.
  Final-test seeds remain untouched. Development checks do not satisfy the M6 gate.
- Trained-checkpoint export verification passed on **10,000 circuit-level shots**
  at p=.003: exact corrections, convergence flags, and first-convergence iterations
  agree between canonical CPU PyTorch and Numba. Independent audit reproduced all
  **320** saved selected-checkpoint validation outcomes and checked **21** CSV rows
  and **96** raw latency measurements. See `results/m6_development/audit.json` and
  `export_verification.json`. The source receipts identify the pre-commit dirty
  revision and exact file hashes used for each check; prior results were not edited.
- Full suite before the M6 development commit: **103 passed** (33.04 s), including
  the slow checks. Shell entry points pass `bash -n`; the source whitespace check
  is clean. Retained CSV artifacts use Python csv.writer's CRLF line endings,
  which Git's global whitespace check flags. Their executed bytes were preserved.

## CPU learning pilot — November 8, 2026 (started)

- User authorized a CPU pilot after discussing the main run's compute cost.
  `configs/m6/cpu_pilot.yaml` retains six-round bb72, the five circuit p values,
  structural T=15 and parity loss. It uses batch 8, up to 500 updates, a 125-update
  curriculum, validation every 50 updates on 256 shots/p, and patience 8.
- The existing coarse three-factor development tuning table is a development
  comparator only. The main full baseline sweep and experiment matrix are unchanged.
  Training entropy is [1234, 704]; fresh fixed validation seeds are 13234–13238,
  disjoint from tuning, previous development validation and final-test seeds.
- `scripts/m6_cpu_pilot.py` limits the child to 600 wall-clock seconds, then sends
  an interrupt with a 15-second cleanup allowance and retains checkpoints and
  logs. The resolved config, budget, source hashes, console output, cProfile data
  and final exit status are retained. Timings include profiler overhead. GPU use
  is zero; no final-test data or two-stage performance experiment is involved.
- Pilot bundle: `results/m6_cpu_pilot/20261109T024448577043Z-832e804d/`.
  This is a development learning/profiling run, not M6 acceptance.

### CPU pilot outcome — completed

- Completed all **500 updates** on one CPU thread in **373.94 s** (6 min 14 s),
  including imports, setup, validation, plotting and cProfile instrumentation.
  GPU use was zero. The wall-clock limit did not trigger. All logged losses were
  finite; selected and final checkpoints are both at step 500.
- On 256 fixed validation shots at each of five p values, mean fallback demand
  decreased from **94.84375% to 88.828125%** (6.015625 percentage points).
  The coarse tuned development comparator was **90.78125%**. Selected neural
  weights had strictly lower fallback at four p values and tied at p=.005.
  The initial model improved at the same four points. No uniform improvement
  or final held-out performance claim is made.

  | p | Initial neural | Selected neural | Coarse tuned BP |
  |---|---|---|---|
  | .001 | 78.515625% | 64.0625% | 64.84375% |
  | .002 | 96.484375% | 85.546875% | 91.015625% |
  | .003 | 99.609375% | 96.09375% | 98.828125% |
  | .004 | 99.609375% | 98.4375% | 99.21875% |
  | .005 | 100% | 100% | 100% |

- Independently reloaded the selected checkpoint with each p's current priors,
  reproduced **all 1,280** saved selected validation outcomes, and recomputed the
  initialized model on the identical shots. Replayed flags and the derived summary
  are retained as `replayed_validation.npz` and `pilot_summary.json` in the bundle.
- The median batch-8 update interval excluding validation boundaries was **0.562 s**
  (10th–90th percentile 0.547–0.660 s). This includes sampling, mining, optimization
  and profiler overhead. It is not an estimate for the main batch-32 experiment.
- The profile recorded 75.91 s in multi-loss evaluation and 110.42 s in backward
  propagation, together about half the elapsed run. All neural forward calls,
  including frozen mining, took 105.59 s; validation inference took 69.32 s.
  Nested profile entries must not be added as disjoint totals. These observations
  identify useful targets for later CPU optimization; no optimization was applied.
- Selected checkpoint SHA-256:
  `2861bf998133848ebea62f97732037ac60f81571569ec9cd9f249212d6f15770`.
  PNG/PDF curves were visually inspected. The loss varies with the sampled p;
  the fixed validation curve, rather than raw loss alone, establishes pilot progress.
- This is a single-seed, repeatedly selected development result against a coarse
  tuning table. Paired p-values in the summary are exploratory after selection.
  Main full baseline tuning, three-seed training, final accuracy and latency remain
  outstanding. Main configs and final-test seeds were not changed or consumed.
- The trainer's legacy smoke output says `M5 NOT ACCEPTED` and writes `m5_gate.json`
  because smoke runs cannot establish an acceptance gate. This does not reverse
  the earlier M5 pass and does not indicate a CPU execution failure.
- The run sampled 4,856 candidate shots and optimized on 4,000 selected hard shots;
  no mining batch exhausted its draw cap. Full suite before the pilot checkpoint
  commit: **103 passed** (31.84 s). Research implementation and reference files
  are unchanged by this pilot.

## Independent CPU pilot check — November 8, 2026 (started)

- User authorized the stronger-baseline/fresh-data check. The step-500 CPU pilot
  checkpoint is frozen before this check; its SHA-256 is
  `2861bf998133848ebea62f97732037ac60f81571569ec9cd9f249212d6f15770`.
  No retraining or additional checkpoint selection is allowed on the new cases.
- `configs/m6/cpu_pilot_check.yaml` tunes all 11 factors .50, .55, …, 1.00 at T=15,
  on 1,024 new shots/p (seeds 15234–15238). It then compares ordinary BP,
  independently tuned BP and the frozen Numba neural model on 2,048 paired fresh
  shots/p (seeds 16234–16238). Both regions are separate from checkpoint-selection
  validation (13234–13238), previous tuning, and reserved final-test seeds.
- This is a Stage-1 fallback-demand check. No OSD calls are made. Exact-schema
  CSVs label decoders `*_only`, set actual `osd_calls` and `osd_rate` to zero, and
  report potential fallback demand as `stage1_failures / shots` in the summary.
  Nonconvergence counts as failure in standalone logical-error statistics.
- The predeclared development screen for proceeding to CPU optimization is a
  strictly lower grid-mean fallback demand and at least one paired improvement
  with McNemar Holm-adjusted p<.05 across five points. All pointwise regressions
  will be reported. This screen cannot establish the main research claim.
- The study has a 900-second wall-clock cap, plus a short interrupt/kill cleanup
  allowance. Source hashes, configs, tuning rows, paired flags, sampled syndromes
  and observables, checkpoint identity, failures and exit status are retained.
  Main experiments and final-test data remain untouched.
- Bundle: `results/m6_cpu_pilot_check/20261109T040246608374Z-cb99df2f/`.

### Independent check outcome — completed, optimization screen not met

- Completed both phases in **516.27 s** (8 min 36 s), using CPU only. Tuning took
  346.78 s; fresh comparison plus process overhead took another 169.44 s. The
  wall-clock cap did not trigger. All 55 tuning candidates and 15 evaluation rows
  are saved. There were 5,120 unique tuning shots and 10,240 unique comparison
  shots, with the latter shared by the three decoders.
- The selected factors by increasing p were **1.00, 1.00, .95, .75, .85**.
  Fresh mean fallback demand was **93.69140625% ordinary BP**, **88.75% tuned BP**,
  and **88.57421875% neural BP**. The neural/tuned gap is **0.17578125 percentage
  points**, or only 18 fewer potential fallback calls across 10,240 paired cases.

  | p | Tuned BP | Frozen neural | Neural minus tuned (percentage points) |
  |---|---|---|---|
  | .001 | 59.375% | 60.302734375% | +0.927734375 |
  | .002 | 87.6953125% | 87.255859375% | -0.439453125 |
  | .003 | 97.412109375% | 96.435546875% | -0.9765625 |
  | .004 | 99.31640625% | 98.92578125% | -0.390625 |
  | .005 | 99.951171875% | 99.951171875% | 0 |

- Neural BP is strictly better at three points, worse at one and tied at one.
  The smallest raw paired McNemar p-value is .04221 at p=.003, but its Holm-adjusted
  value is **.21105**; no point passes corrected .05. The predeclared screen
  therefore returned `supports_cpu_optimization=false`. CPU optimization and
  additional training were not launched. This is insufficient evidence of an
  advantage, not evidence that a true effect must be zero.
- The previous small validation result was optimistic relative to this fresh,
  more thoroughly tuned comparison. Learning relative to initialization remains
  demonstrated, while a useful improvement over the strong baseline is unproven.
  No main M6 completion, equal two-stage LER, or latency-speedup claim is made.
- The comparison process loaded its source before an extra export-verification
  check was added to the tool. The exact executed source is retained as
  `comparison/executed_compare_source.py`, recovered by reversing only that added
  check and verified byte-for-byte against the recorded SHA-256
  `81eb2d7a7608b058fd2fdca81fccfeed31e1dea23440045e2553aaf285237da7`.
  PyTorch/Numba subset agreement is checked separately by the retained audit.
  No experimental outcome or existing result artifact was edited.
- Independent artifact audit regenerated **all 10,240** comparison cases from their
  seeds and checked the true-fault H/L parity labels. It verified **30,720** decoder
  flag counts, all 55 tuning candidates' selections, all 15 CSV rows, paired tests
  and adjusted p-values. It independently replayed 960 subset decodes and checked
  exact PyTorch/Numba corrections, convergence and stopping iterations on 320 shots.
  See the bundle's `audit.json`; all checks passed and the frozen checkpoint hash
  is unchanged. The full suite passed **105 tests** (32.16 s), including seed-reuse
  rejection and Holm-adjustment checks. M6 remains incomplete.

## Stronger-initialization CPU diagnostic — November 8, 2026 (started)

- User requested continuation after the fresh-data check found no clear advantage.
  Inspection of the selected .625-initialized weights showed mean .626685 and
  range .548504–.771213, while independently tuned BP selected .95–1.00 at the
  lower/middle p values. Hypothesis: the short run's starting point limits learning.
  This is a hypothesis, not a demonstrated mechanism.
- `configs/m6/cpu_strong_init.yaml` uses structural weights initialized to .95,
  still T=15, six rounds, batch 8, parity loss, up to 500 CPU updates and a 600-second
  training budget. Validation uses new seeds 18234–18238; training entropy is
  [1234, 707]. The existing full 11-factor tuning table is reused unchanged.
  Main experiment configs and reference implementations are unchanged.
- The frozen hard-mining model follows the same .95 initialization, as specified
  by the trainer. This variant is not a causal isolation with identical training
  samples to the earlier .625 run. A matched untrained .95 reference will separate
  initialization gains from additional learning on the new comparison cases.
- New comparison seeds are 20234–20238, disjoint from both pilots' selection sets,
  tuning, the earlier independent comparison, and reserved final-test seeds.
  Selected weights will be frozen before generating these cases. Ordinary BP
  retains .625 in that comparison; tuned BP reuses the independent full sweep.
- The tuning-table validator now accepts compatible previously executed tuning
  through its original resolved-config sidecar and row hashes, rather than requiring
  that an evaluation with a different checkpoint/seed have the same config hash.
  Actual tuning seeds/counts and all factor-grid rows remain checked. New studies
  retain their executed comparison source before decoding.
- Training bundle: `results/m6_cpu_strong_init/20261109T041850480159Z-dee36314/`.
  Training and fresh-data checks remain development diagnostics, not main M6 results.

### Stronger-initialization diagnostic outcome — completed

- The CPU run completed 500 updates in **380.98 s** (6 min 21 s), with finite losses
  and zero GPU use. It sampled 5,472 candidates, optimized on 4,000 hard shots,
  and hit no mining caps. Median batch-8 update time excluding validation was
  .583 s. Selected checkpoint was **step 300**, not the step-500 continuation.
  Validation fallback was 87.734375% initially, 86.953125% selected, and 87.734375%
  finally, versus 88.046875% for independently tuned BP. Selection remains exploratory.
- All 1,280 selected validation outcomes replayed exactly. The selected checkpoint
  SHA-256 is `e0187b03bce6970b5a7b20562f85b62baeded967522431bcb4254caeaa1e2bd2`.
  Curves were visually inspected; validation fluctuates and does not establish
  steady improvement. The source initializer and frozen mining model both use .95.
- The fresh check reused the original 55-row tuning table unchanged, verified its
  resolved config and original hashes, and finished comparison plus the matched
  untrained reference in **223.90 s**. No training or selection used these cases.
  Bundle: `results/m6_cpu_strong_init_check/20261109T042526482211Z-594f5da1/`.

  | p | Tuned BP | Untrained .95 neural | Selected neural |
  |---|---|---|---|
  | .001 | 61.376953125% | 62.158203125% | 59.814453125% |
  | .002 | 86.9140625% | 87.255859375% | 86.5234375% |
  | .003 | 96.826171875% | 96.826171875% | 96.630859375% |
  | .004 | 99.658203125% | 99.169921875% | 99.169921875% |
  | .005 | 99.90234375% | 99.951171875% | 99.853515625% |

- Grid means were **88.935546875% tuned BP**, **89.072265625% untrained reference**,
  and **88.3984375% selected neural**. Selected neural had 55 fewer potential fallback
  calls than tuned BP and 69 fewer than its untrained reference across 10,240 paired
  cases. All five observed neural/tuned differences favored neural; four favored
  learned weights over the initial model and one tied. These are small observed gains.
- Neither five-point comparison passed Holm-adjusted .05: the smallest adjusted
  p-value versus tuned BP was **.2069473**, and versus the matched initial model
  **.2035642**. The optimization and learning screens remain unmet. No CPU optimization
  or main configuration replacement was performed. The variant is retained as a
  development result, not as a proven improvement. Different training streams and
  mining sets prevent causal attribution to initialization alone across pilots.
- The legacy generic pilot summary used `coarse_tuned` field names even though this
  run used the full 11-factor table. The summary tool now reports the actual factor
  count with generic development-tuning labels. `pilot_summary_full_tuning.json`
  recomputes the same outcomes with corrected labels; the original artifact was
  preserved. Main comparison results use the correct original tuning provenance.
- Decoder equations, parity loss, main experiment matrix and reserved final-test
  seeds are unchanged. M6 remains incomplete. A short controlled loss-function
  ablation is a more informative next diagnostic than assuming longer training
  or faster computation will resolve the weak advantage.
- Independent audit regenerated all 10,240 fresh cases and checked their H/L
  true-fault labels, 30,720 main decoder flags, another 10,240 untrained-reference
  flags, all 55 tuning selections, all 15 main CSV rows, means and adjusted tests.
  It independently replayed 960 main subset decodes plus 320 initialization subset
  decodes; exact PyTorch/Numba agreement was verified on 320 selected-model shots.
  All audit checks passed, and the frozen checkpoint hash is unchanged.
- Full suite before this development checkpoint commit: **107 passed** (32.10 s),
  including compatible tuning reuse and rejection of mismatched row provenance.
  Source whitespace checks are clean; executed CSV line endings remain preserved.

## Controlled CPU loss ablation — November 8, 2026 (started)

- User authorized parity-versus-bitwise BCE after the initialization diagnostic.
  `configs/m6/cpu_loss_parity.yaml` and `cpu_loss_bitwise.yaml` differ only in
  `train.loss`. Both use .95 initialization, structural T=15, six-round bb72,
  batch 8, Adam/cosine, the same frozen hard-mining model, and 500 updates.
  Patience 11 guarantees the equal 500-update budget with ten post-update validations.
- Training entropy is [1234, 708]. Shared fresh checkpoint-selection seeds are
  22234–22238; new paired comparison seeds are 24234–24238. Both regions are
  separate from prior tuning, prior selection/comparison cases and final-test seeds.
- The trainer now records a SHA-256 digest of each actual p/syndrome/fault/weight
  batch. Before comparing models, all 500 batch hashes, sampling metadata and
  ending generator states must match. This logging changes no decoder or loss math.
- Each CPU arm has a 600-second cap and uses one thread. They run concurrently;
  their wall times are not an isolated speed comparison. The study targets a
  900-second total envelope. No GPU is used and no final-test data are sampled.
- The primary comparison is selected parity versus selected BCE on 2,048 fresh
  paired shots/p, with exact McNemar tests and Holm correction over the five points.
  Secondary contrasts versus tuned BP and matched untrained .95 each use correction
  over both losses × five p values. All outcomes, including regressions and any
  selection of an untrained step-zero checkpoint, will be reported.
- This is a development subset of the specified E3 loss ablation. It does not
  complete E3/M7 or the pending three-seed E1 and E2 experiments.

### Controlled loss-ablation outcome — completed

- Both arms completed all 500 updates with finite losses. Their configs differ
  only in loss; **all 500 actual batch hashes and sampling metadata match**, as
  do ending NumPy/Torch generator states, initial validation results and baseline
  flags. Each sampled 5,496 candidates and optimized on 4,000 hard shots, with no
  mining caps. Shared training-input digest:
  `443081b06d9608c128db43b614997034b2b94aeeb6e6346077694451dcade70f`.
- The parity arm selected step **250**; BCE selected **50**. Shared initial
  validation fallback was 88.28125%; selected parity was 87.734375%, BCE 88.203125%.
  Individual profiled wall times were 381.23 s and 244.42 s, respectively. Because
  they ran concurrently, these do not establish an isolated training speedup.
- The five-model fresh-data comparison completed in 298.6 s; total study time from
  the first training launch was **698.77 s** (11 min 39 s), below its 900-second cap.
  Bundle: `results/m6_cpu_loss_ablation/comparison/20261109T044218253294Z-6e71d010/`.

  | Decoder | Mean potential fallback demand |
  |---|---|
  | Ordinary BP | 93.59375% |
  | Independently tuned BP | 88.671875% |
  | Matched untrained .95 neural | 88.466796875% |
  | Selected parity neural | 88.349609375% |
  | Selected BCE neural | 88.525390625% |

- Parity's mean was 0.17578125 percentage points lower than BCE (18 fewer potential
  calls/10,240 paired cases), but **no primary point passed Holm-adjusted .05**;
  all five adjusted primary p-values were 1.0. Parity improved at three points,
  worsened at one, and tied at one relative to BCE. This does not establish equality
  of the objectives or a reliable winner.
- Relative to matched initialization, parity had only 12 fewer potential calls and
  BCE six more. Smallest adjusted learning p-values were .64036 (parity) and 1.0
  (BCE). Neither objective showed convincing learning gains in this short run.
  Neither passed the secondary tuned-BP comparison family either (smallest adjusted
  p-values .67180 and .86627). Parity remains the planned default; no main config
  was changed and no large run was launched based on these exploratory differences.
- Selected checkpoint SHA-256 values:
  parity `eb3914e68c1ca479bde054e40f96fff75fb947d9855a8707b640a90752443c0b`;
  BCE `7f0514de98e4474a3d0ae500e33250a9a51c22ecc38902a2713202a85cb6cc82`.
  Selected and continuation checkpoints, configs, raw cases/flags, controlled-input
  proof, event logs, profiles and PNG/PDF learning curves are retained. Both sets
  of curves were visually inspected; raw loss magnitudes are on different scales
  and must not be compared directly between objectives.
- Independent audit rechecked the 500 matched batches, replayed all **2,560** selected
  validation cases, regenerated **10,240** comparison cases with true H/L parity
  labels, and checked **51,200** decoder flags and all **25** exact-schema CSV rows.
  It independently replayed 1,600 subset decodes, verified exact CPU PyTorch/Numba
  corrections/convergence/iterations on 960 shots, and recomputed all three
  statistical families (5/10/10 tests). All checks passed.
- Full suite before the development checkpoint commit: **108 passed** (33.25 s),
  including the controlled-input match and deliberate batch-mismatch rejection.
  Final-test seeds, decoder equations, loss definitions and reference files remain
  unchanged. This is a development E3 subset; M6/M7 remain incomplete.

## Objective-alignment check — November 9, 2026 (started)

- User requested the check following the controlled loss ablation. This is a
  descriptive analysis of the first 128 saved development cases at each p,
  not a new holdout experiment or a new training run. It compares the matched
  initial model, both validation-selected models, and both step-500 continuations.
- `configs/m6/cpu_objective_diagnostic.yaml` fixes batch 8, one CPU thread, a
  300-second budget and eight-shot final-output gradient probes per p/model.
  True faults are regenerated from the original explicit comparison seeds and
  saved syndromes/observables must match exactly before scoring.
- Both parity and bitwise multi-losses are evaluated on the same cases for each
  model. Per-p failure-ranking AUC is reported only when both success/failure
  classes exist. No pooled cross-p correlation is interpreted as learning evidence.
- Full-unroll hard decisions are compared with canonical first-convergence
  deployment, including iteration-zero success, any later loss of convergence,
  final-iteration failure, and logical flags. Every traced deployed correction,
  convergence flag and stopping iteration must match the Numba export exactly.
- The implementation follows the handoff: training uses all unrolled iterations,
  while inference freezes at first convergence. This intended difference is a
  candidate source of objective/metric misalignment, not an identified decoder bug.
  No production loss, stopping rule, main config or reference implementation is changed.

### Objective and checkpoint check outcome — completed

- The objective diagnostic completed in **113.3 s** on CPU, evaluating five model
  states on the same 640 saved cases (128/p). It retained 3,200 per-model shot scores
  and verified every traced first-convergence correction, flag and iteration count
  against Numba exactly. No numerical decoder mismatch was found in these checks.
  Bundle: `results/m6_objective_diagnostic/20261109T070451119909Z/`.
- Parity multi-loss ranked failed versus successful cases well within p where
  both classes existed (roughly .95–.99 AUC at the lower p values). Higher-p estimates
  sometimes have only one or two successes, or no AUC when all shots fail. Thus the
  loss is informative about individual failures, but that does not make a reduction
  in mean loss a reliable guarantee of improved hard decoding.
- On these 640 cases, the initial model decoded 73 shots; selected parity 78;
  final parity 76; selected BCE 71; final BCE 85. These prefix results are diagnostic,
  not independent performance estimates. Selected parity lowered its own loss on
  434 cases but made two previously correct initial decodes nonconvergent despite
  lower loss. Selected BCE did so on four cases (472 loss decreases). This directly
  illustrates non-monotonic loss/decision changes without proving a loss-function bug.
- Of selected parity's 78 first-converged shots, **seven** were no longer converged
  at iteration 15; nine lost convergence at some later iteration. Deployment correctly
  preserves their first successful corrections. The full-unroll training objective
  still scores later outputs, as the handoff requires. This intentional difference
  is a candidate for a controlled ablation, not an implementation error to silently fix.
- Roughly 73% of probabilities were at the configured clamp boundary averaged over
  iterations and shots. Final-output direct probability-gradient probes were sparse
  (e.g. about 98% exactly zero at p=.001); all were finite. This can reflect correctly
  confident variables. It does not establish that clipping causes weak parameter
  learning, and no clamp or gradient behavior was changed.
- A retrospective review then evaluated both existing step-500 continuations on
  all 10,240 saved development cases, with the selected models unchanged. Results:

  | Loss | Original validation-selected checkpoint | Existing final checkpoint |
  |---|---|---|
  | Parity | 88.349609375% | 88.466796875% |
  | BCE | 88.525390625% | 88.3203125% |

- BCE's selected step 50 beat step 500 on the 1,280-shot selection set by only one
  case (88.203125% vs 88.28125%), while step 500 had 21 fewer potential fallback
  calls on the saved development comparison. This demonstrates a ranking reversal
  in this run, not proof that the final checkpoint is generally better. It was not
  substituted into the earlier experiment or selected as a new winner after viewing
  those cases. Review output is under `full_checkpoint_review/` in the diagnostic bundle.
- The main training configs already specify 20,000 validation shots/p. The small
  256-shot/p pilot shortcut is noisy relative to the tiny observed differences.
  Before further hyperparameter changes, a larger fixed development selection set
  is the most direct next stability check. No new training or final-test use occurred.
- The full-checkpoint review completed in **87.97 s**, with exact Torch/Numba
  subset checks on 640 final-model shots. All 20,480 final-checkpoint flags and
  the selected-model hashes were independently rechecked from the saved files.
  All per-p selected/final McNemar p-values were above .05 even before correction;
  observed ranking reversals are descriptive, not significant performance claims.
- Aggregate-score audit recomputed all 3,200 model-shot summaries and per-p AUCs;
  all checks passed. Full suite before this diagnostic checkpoint commit:
  **110 passed** (69.79 s), including first-convergence/iteration-zero and ranking
  AUC edge-case checks. Source whitespace checks are clean. M6/M7 remain incomplete.

## Larger fixed development validation — November 9, 2026 (started)

- User authorized a larger fixed validation set before further training changes.
  `configs/m6/cpu_validation_stability.yaml` fixes 2,048 cases/p, eight times the
  earlier 256/p selection pool, for 10,240 new development cases total. New seeds
  26234–26238 are disjoint from training-selection, tuning, comparison and final
  regions. No training, reweighting or numerical decoder change is involved.
- Candidate weights are frozen: parity step 250/500, BCE step 50/500, plus matched
  untrained .95 as a reference. Rankings concern only these available checkpoints,
  not intermediate checkpoints that were not retained. Original selected files
  remain unchanged; any new preference is a development validation recommendation.
- The study retains paired flags and exact-schema CSVs, makes CPU Torch/Numba
  subset checks, and compares eight disjoint 256-shot/p blocks with the whole pool.
  A fixed-seed 2,000-replicate paired bootstrap resamples the {-1,0,1} checkpoint
  differences within each p, giving descriptive conditional intervals for the
  grid-mean last-minus-selected difference. Bootstrap frequencies are not posterior
  probabilities, and blocks are subsets of the larger pool, not independent tests
  of a winner selected from that pool. Per-p paired tests use Holm over ten tests.
- The initial attempt at `results/m6_validation_stability/20261109T073007704368Z/`
  was intentionally interrupted after a metadata bug was noticed: CSV config hashes
  referred to the source evaluation configuration rather than resolved new sampling
  settings. Its partial artifacts and failure trace are preserved. SIGINT during a
  Numba call surfaced as a SystemError; this is the deliberate interruption, not a
  decoder finding. No protocol or weights were changed in response to its outcomes.
- The corrected runner saves both the source config and a resolved actual config
  with the new seed range, shot count, model initializer and candidate paths. The
  same predeclared cases are rerun in a new bundle with no overwrites:
  `results/m6_validation_stability/20261109T073249552750Z/`. One CPU thread, 600-second
  cap, zero GPU use. This is validation, not independent performance or M6 completion.

### Larger-validation outcome — completed

- The corrected study completed in **228.13 s** (3 min 48 s) on one CPU thread,
  using 10,240 fixed new validation cases and five frozen model states. No training
  or numerical decoder changes were made. The resolved run config hash is
  `30b4f27b6fd29fdb2aeaff100772436b5fc8331e`; CSV hashes/seeds/counts were audited
  against that actual config rather than the separately saved source config.

  | Loss | Original selected checkpoint | Existing step-500 checkpoint | Last minus selected |
  |---|---|---|---|
  | Parity | 89.541015625% | 89.23828125% | -0.302734375 pp |
  | BCE | 89.609375% | 88.916015625% | -0.693359375 pp |

- Matched initialization had 89.560546875% fallback demand. These are selection-set
  measurements, not estimates from independent performance data. They must not be
  compared directly with rates from earlier pools as though the cases were paired.
- For BCE, the descriptive 95% paired bootstrap interval for last-minus-selected
  was **[-1.22070, -0.17578] percentage points**. Six of eight disjoint small blocks
  favored the final checkpoint, one favored the original selection and one tied.
  For parity, the interval was **[-0.79126, +0.16602] pp**, and blocks split **4–4**.
  BCE step 500 is the stronger within-loss validation candidate; parity remains
  unresolved at this sample size. These are conditional, per-pair bootstrap
  observations, not joint confirmatory inference or posterior probabilities.
- No individual p comparison passed Holm correction over the ten tests (smallest
  adjusted p=.16184). A grid-mean bootstrap interval and per-p corrected tests
  address different quantities and are not interchangeable claims.
- `validation_preferences.json` records lower-mean candidates, their immutable
  hashes, the selection seed range and interval uncertainty for future independent
  confirmation. It reproduces exactly from the saved summary. Original `best.pt`
  files remain unchanged, and no main default loss or main checkpoint is replaced.
- The PNG/PDF ranking plots were generated from saved observations and visually
  inspected. They show the original small-set ranking reversal and fluctuations
  across eight disjoint 256-shot/p blocks within the larger validation pool.
- Independent audit regenerated all 10,240 cases and true H/L labels; checked
  51,200 decoder flags, all 25 CSV rows and resolved provenance; replayed 1,600
  subset decodes with exact PyTorch/Numba corrections, flags and iterations; and
  reproduced the seeded bootstrap, block preferences and paired statistics.
  All checkpoint hashes and all audit checks passed.
- Full suite before the development checkpoint commit: **112 passed** (34.79 s),
  including paired-bootstrap stratum weights/ties and block-ranking edge cases.
  Final-test seeds remain untouched. Independent confirmation of any preferred
  checkpoint and the full M6/M7 studies are still pending.
- The complete first-noise-point cases and all five decoder/logical outcome arrays
  from the intentionally interrupted attempt match the corrected rerun exactly.
  The provenance correction did not alter those numerical observations.

## Frozen-candidate independent confirmation — November 9, 2026 (started)

- User authorized a fresh confirmation of the larger-validation candidates.
  `configs/m6/cpu_independent_confirmation.yaml` freezes BCE step 500, parity step
  500, matched untrained .95, and the independently tuned T=15 BP baseline before
  sampling. Candidate hashes must match `validation_preferences.json` exactly.
- New seeds **28234–28238** provide 2,048 shared fresh development cases/p
  (10,240 total), separate from tuning, all previous selection/comparison regions
  and reserved final-test seeds. The larger-selection seeds 26234–26238 are
  explicitly excluded by the guard in addition to checkpoint validation and tuning.
- The primary contrast is BCE versus tuned BP, exact paired McNemar with Holm
  correction over five p values. Secondary BCE/parity and BCE/untrained contrasts
  are corrected jointly over ten tests. Before outcomes, the clear-gain screen is:
  lower grid-mean demand, at least one corrected primary benefit, and no corrected
  primary regression. Otherwise confirmation is inconclusive under this screen.
- A seeded paired stratified bootstrap gives a descriptive interval for grid-mean
  BCE-minus-tuned demand. It does not replace the predeclared primary screen or
  establish logical-error equality. Four Stage-1 decoders only; no OSD or latency
  experiment. One CPU thread, 600-second cap, no training or changes to main configs.

### Independent confirmation outcome — completed, clear gain not confirmed

- Completed in **196.45 s** (3 min 16 s), on CPU only. All four candidate definitions
  were frozen before sampling; hashes still match the larger-validation preference
  record. No new selection, retuning or retraining was performed on these cases.
  Bundle: `results/m6_independent_confirmation/20261109T074900968718Z-e7ad7a97/`.

  | Decoder | Mean potential fallback demand |
  |---|---|
  | 11-factor T=15 tuned BP | 88.7890625% |
  | Matched untrained .95 | 88.7890625% |
  | Frozen parity step 500 | 88.642578125% |
  | Frozen BCE step 500 | 88.916015625% |

- BCE had **13 more** potential fallback calls than tuned BP across 10,240 paired
  cases: +0.126953125 percentage points. Its descriptive 95% paired bootstrap
  interval for BCE-minus-tuned was **[-0.45923, +0.71289] pp**, containing zero.
  The generic bootstrap helper calls its sign frequencies `fraction_last_lower`
  and `fraction_selected_lower`; in this contrast those mean BCE-lower and
  tuned-BP-lower, respectively, and are not posterior probabilities.
- No primary comparison passed Holm-adjusted .05 (smallest adjusted p=.08450 at
  p=.004). There were zero corrected benefits and zero corrected regressions;
  `clear_gain=false` under the unchanged predeclared rule. BCE was slightly worse
  at the two lowest p values and slightly better at the other three. The secondary
  comparison family also had no corrected differences (minimum adjusted p=.19211).
- Parity's 15 fewer potential calls than tuned BP are a descriptive secondary
  observation, not a new primary result or a reason to choose a new winner on
  these confirmation cases. The validation preference for BCE step 500 within its
  checkpoint pair did not establish superiority over the independently tuned
  baseline on new data. This pilot is inconclusive, not proof of equal decoders.
- No three-seed or longer run was launched from this outcome. Existing defaults,
  checkpoint selections and the reserved main final-test pool remain unchanged.
  The next research decision concerns training/convergence alignment or study
  design, rather than assuming the current pilot has a confirmed performance gain.
- Independent audit regenerated all **10,240** confirmation cases and H/L labels,
  checked **40,960** decoder flags and all **20** CSV rows, independently replayed
  1,280 subset decodes, and verified exact PyTorch/Numba outputs/convergence/iterations
  on 960 neural shots. Candidate hashes, original selection metadata, all paired
  tests, bootstrap values and the predeclared decision rule reproduced exactly.
- An extended audit additionally checked the new seed IDs against **29 prior CSV
  seed sources** and found no collisions; both original and extended audit files
  are retained. No prior result was overwritten. Full suite before this development
  checkpoint commit: **113 passed** (34.44 s), including rejection of larger-selection
  data reuse. Source whitespace checks are clean. Full M6/M7 remain incomplete.

## Controlled training-readout experiment — November 9, 2026 (started)

- User authorized testing inference-aligned training. Added an explicit optional
  `train.runtime.readout` setting: default `full_unroll` preserves historical config
  hashes and behavior; `first_converged` freezes each shot's soft training readout
  after its first syndrome-valid decision. Message updates continue internally,
  and the same T=15 weighted parity/logical loss still scores every returned output.
- Frozen probabilities retain their gradient path. Syndrome-valid wrong-logical
  outputs are frozen too, matching actual inference, and remain penalized by the
  logical loss. There is no oracle-dependent stop rule, discarded logical term,
  inference change or NumPy/Numba decoder change. This is an authorized development
  objective ablation; main experiment configs remain at the default readout.
- Targeted model/config/loss/reference checks passed (44 tests) before training.
  They verify soft prediction freezing, agreement with canonical first-convergence
  corrections/iterations, finite gradients and a wrong-logical toy that retains
  both its logical penalty and its first-iteration gradient.
- Both parity-loss arms use .95 initialization, 400 updates, batch 8, a 100-update
  curriculum, and the same frozen mining model. The 400-update budget leaves time
  for the larger selection sets and fresh comparison under a 900-second study cap.
  Validation every 100 updates uses 2,048 fixed cases/p; patience 5 guarantees the
  equal 400-update budget. Each arm has a 660-second wall limit, one CPU thread,
  and runs concurrently with the other arm. GPU use is zero.
- Shared training entropy is [1234, 713]. Selection seeds **30234–30238** and fresh
  comparison seeds **32234–32238** are new, separate from all prior development,
  tuning and final regions. Inputs, ending RNG states and initial validation must
  match exactly; the only configuration difference may be the readout setting.
- The primary paired contrast is frozen-readout versus full-unroll, Holm over five
  p values. Secondary frozen/tuned and frozen/untrained contrasts form a ten-test
  family. All cases and model choices are fixed before fresh outcomes are sampled.
  No OSD, two-stage LER, latency or main-M6 completion claim will follow from this pilot.
- Training bundles: `results/m6_cpu_readout/full_unroll/20261109T080813488970Z-65bd777f/`
  and `results/m6_cpu_readout/first_converged/20261109T080814638217Z-818b5db1/`.

### Readout comparison timeout and fixed completion

- Both training arms completed all 400 updates: 582.47 s for full-unroll and
  589.31 s for first-converged, including profiler overhead. Their actual batch
  hashes, mining metadata, ending generator states, initial validation and baseline
  flags matched. Both selected step 200; frozen step 300 tied its step-200 score
  and did not replace the earlier selection. Selection demand was 88.544921875%
  for full-unroll and 88.30078125% for frozen readout (descriptive selection scores).
- The first comparison began 757.94 s after the training start because of a delay
  between stages, leaving 142.06 s of the original 900-second wall allowance.
  Its decoder-boundary budget check stopped after 148.6 s, before completing
  p=.004; only the first three complete points (12 CSV rows) were saved. The
  failed bundle and original executed source remain untouched at
  `results/m6_cpu_readout/comparison/20261109T082051430751Z-9ecb5bb7/`.
  The overall 15-minute target was therefore missed; it is not reported as met.
- A separately capped 240-second CPU comparison completion was launched with
  `configs/m6/cpu_readout_completion.yaml`. It repeats the exact fixed candidate
  bytes, resolved evaluation config, seeds, shot counts and statistical families.
  Preflight rejects any changes to these choices and requires the recorded timeout.
  There is no retraining, selection on comparison outcomes, extra sample, or claim
  that repeated cases are an independent confirmation. The original 900-second
  settings file is unchanged. Completion timing is explicitly comparison-only;
  total elapsed time still includes the delay and stopped attempt.

### Controlled readout outcome — completed, no confirmed advantage

- The fixed comparison completion finished in **197.84 s** (CPU only), within its
  separate 240-second allowance. Total elapsed time since the first training start
  was **1,128.65 s** (18 min 49 s), including the delay and stopped attempt.
  Completed bundle:
  `results/m6_cpu_readout/comparison/20261109T082344302341Z-9ecb5bb7/`.

  | Decoder | Mean potential fallback demand |
  |---|---|
  | Independent T=15 tuned BP | 88.955078125% |
  | Matched untrained .95 | 88.8671875% |
  | Full-unroll selected step 200 | 88.73046875% |
  | First-converged selected step 200 | 88.69140625% |

- Frozen readout had **four fewer** potential fallback calls than full-unroll
  across 10,240 paired development cases, a difference of **-0.0390625 percentage
  points**. Its differences versus full-unroll favored frozen readout at two p
  values and full-unroll at three. All five primary Holm-adjusted p values were
  **1.0**. All ten secondary adjusted p values were also **1.0**; the 27 fewer
  potential calls than tuned BP do not establish a performance advantage.
- The lower frozen selection score did not translate into a clear fresh-data gain.
  This single-seed, short-training ablation is inconclusive; it does not prove
  equivalence or justify promoting the optional readout into the main experiment.
  Main parity/full-unroll settings and the reserved final-test seeds remain intact.
  M6/M7 are incomplete; two-stage LER equality and latency have not been measured
  by this study. No GPU, OSD or external compute was used.
- Selected checkpoint SHA-256 values are
  `9af773b51d99dd55ecaba1918df3333280e780ff1450ca4d3e54936482f820c9`
  (full-unroll) and
  `8e5c230c3d13503cadf728d69327666f123025a03e4b4dd48ba4e9b9f7cc3588`
  (first-converged). Both selected and final checkpoints are retained with configs.
- Full suite: **117 passed** (25.28 s), including readout gradient/logical edge
  cases and historical config hash preservation. Source whitespace checks passed.
  Generated validation/fresh-comparison plots are saved as PNG and PDF. A second
  `readout_comparison_spaced` rendering adds annotation room; the original is retained.
- Independent audit passed: **20,480** selected-validation cases replayed,
  **10,240** comparison cases and H/L labels regenerated, **40,960** decoder flags
  and **20** CSV rows checked, **1,280** subset decodes replayed, and **960** neural
  shots matched exactly between PyTorch and Numba (outputs/convergence/iterations).
  All **12** completed rows and saved case/flag arrays from the timed-out attempt
  reproduced exactly. Both statistical families reproduced, selection was verified
  as the earliest minimum validation score, and training/evaluation core source
  hashes matched. A search across **31** prior CSV seed sources found no collisions.
  Final parameter differences (max |delta w|=.0422532, |delta alpha|=.0272748)
  confirm the readout treatment affected optimization despite the small outcome.

## Fixed-hard-case learning diagnostic — November 9, 2026 (predeclared)

- User authorized a bounded test of whether the existing circuit model can learn
  a small fixed dataset before further hyperparameter pilots or main training.
  This is a development diagnostic, not a replacement for the main experiment
  matrix. No core decoder, loss, main configuration, or reference file is changed.
- `configs/m6/cpu_fixed_case.yaml` resolves BB72, six rounds, p=.001, structural
  T=15, .95 initialization, the existing parity/logical full-unroll multi-loss,
  Adam 1e-3 with cosine decay to 1e-5, gradient clip 1, batch 8, and 1,200 updates.
  The fixed p and repeated cases intentionally replace curriculum/streaming only
  inside this diagnostic. One CPU thread; no GPU or OSD. Resolved config hash:
  `9fa92ed15632f63a6f80dcd3d699e2f84257dde4`.
- Train on the first 128 unique syndromes where the initial .95 model fails.
  Pool seed **34234**, draw batch 256 with a 4,096-shot maximum. Preserve the full
  sampled pool, selected indices, true faults, syndromes and observable labels.
  Shuffle each epoch with explicit entropy **[1234, 911]**. Whole-set training loss
  and canonical first-convergence decoding are checked every 100 updates.
- Freeze 2,048 independent, unfiltered development cases with seed **36234** before
  any update. Reject train/unseen syndrome overlap and validate all H/L labels.
  Evaluate this set only initially and at the fixed final update (or budget stop).
  It never selects a checkpoint or changes training. The independently tuned BP
  baseline is descriptive context, using its existing table with original provenance.
- `cpu_fixed_case_diagnostic.yaml` fixes a **1,200-second hard wall deadline** from
  diagnostic setup onward, including final evaluation. Reserve the last 120 s for
  finalization; stop new training updates by 1,080 s. A budget-truncated run is
  explicitly `budget_limited`, not successful completion of the update protocol.
- Practical screens fixed before outcomes: useful training gain requires at least
  10 percentage points more logically correct Stage-1 decodes on the initial-hard
  set (at least 13/128). A relative whole-set loss drop of at least 10% without that
  decoding gain is labelled loss/decoding disconnection. Useful local generalization
  requires at least 2 pp gains in both correct decoding and fallback reduction on
  unseen cases, with both paired McNemar contrasts passing joint two-test Holm .05.
  These thresholds guide the next decision; they are not M6 acceptance criteria,
  proof of a mechanism, or evidence of equal two-stage LER/latency.
- If training decoding improves but the unseen screen does not pass, report
  generalization unconfirmed. If neither meaningful loss nor training-decoding gain
  occurs, report no useful learning under this budget. Main runs wait for evidence;
  no further variant will be launched automatically from an inconclusive result.

### Fixed-case outcome — local learning, no net unseen benefit

- Completed all **1,200 updates** (75 shuffled passes over the 128 cases) in
  **783.23 s** / 13 min 3 s, including setup and final scoring, within the
  predeclared 1,200-second CPU cap. No timeout, GPU, OSD or final-test use occurred.
  Bundle: `results/m6_fixed_case/20261109T083753363297Z-9fa92ed1/`.
- Whole-set training multi-loss decreased from **56.7148743 to 48.0201035**,
  a **15.3307%** reduction. Logically correct training decodes increased from
  **0/128 to 30/128** (23.4375%), passing the practical training-learning screen.
  The model still failed on **98/128** repeated cases; it did not memorize or solve
  the whole training set. Training success fluctuated between intermediate checks;
  the final step was fixed in advance and no earlier checkpoint was substituted.

  | Unfiltered unseen decoder | Nonconverged / 2,048 | Fallback demand | Logically correct Stage-1 decodes |
  |---|---|---|---|
  | Matched initial .95 | 1,295 | 63.232421875% | 753 |
  | Fixed step-1,200 model | 1,301 | 63.525390625% | 747 |
  | Independently tuned BP | 1,249 | 60.986328125% | 798 |

- On unseen data, final training produced **six more** potential fallback calls
  and **six fewer** correct decodes than initialization: **+0.29296875 pp** demand.
  Both paired contrasts had discordant counts **232 gained / 238 lost**, exact
  McNemar p=**.8176302604**, and joint two-test Holm p=**1.0**. The unseen gain
  screen failed. This does not demonstrate a statistically significant regression
  or equivalence; it establishes no useful net benefit in this diagnostic.
  Tuned BP's 52 fewer nonconvergences than the final model are descriptive context;
  no additional confirmatory test versus tuned BP was introduced after outcomes.
  Tuned BP had one converged wrong-logical case; both neural states had none.
- A descriptive decomposition of the fixed unseen flags is more informative than
  calling this simple memorization: **232/1,295** initially nonconvergent cases
  became logically correct (**17.9151%**), while **238/753** initially correct
  cases became failures (**31.6069%**). Thus some learned rescues transfer to
  unseen cases, but regressions cancel those gains. This is an observed stability
  trade-off, not an isolated causal explanation. The machine decision
  `training_learning_generalization_unconfirmed` refers to the failed *net-gain*
  screen, not absence of any transferable casewise changes.
- Hard-only training omits initially successful examples. Loss of those successes
  is compatible with that sampling policy, but this single-arm diagnostic cannot
  establish that mining caused it, or rule out objective/optimizer/model effects.
  It does not justify buying more compute or promising that longer training fixes
  the problem. The useful next hypothesis, if the user continues, is preserving
  initially correct decodes while learning hard cases (a controlled hard-only versus
  mixed/upweighted comparison), rather than another arbitrary loss/readout tweak.
  No follow-up run is launched automatically. Main M6/M7 remain incomplete and
  their experiment matrix and configs are intact.
- All cases were frozen before updates, all 1,200 minibatch index lists and input
  hashes are saved, and all 100-update training checkpoints plus initial/final
  models are retained. Final checkpoint SHA-256:
  `1d3f6b6e6b4f51d30353ef749ed5f61ce0818f7edfe2d2e895c51a3495c464ac`.
  The reporting tool generated five exact-schema CSV rows, a PNG/PDF figure, and
  the descriptive transition decomposition without changing experimental outcomes.
- Final full suite: **120 passed** (25.44 s). The first targeted command referenced
  the nonexistent `test_losses.py` and ran zero tests; the corrected targeted
  diagnostic/readout suite passed all six checks before training. Source whitespace
  checks passed. The figure was rendered and visually inspected.
- Independent audit passed: regenerated the full 256-shot mining pool, the exact
  128 training and 2,048 unseen cases and their H/L labels; verified all 1,200
  shuffled batch lists/input hashes, ending RNG/order state, the first training
  loss and learning-rate schedule; replayed **5,888** model/case scores and all
  **13** training checkpoint evaluations; verified **1,024** exact Torch/Numba
  corrections/convergence/iteration results and all five CSV rows. Initial/final
  losses, paired tests, transition counts, checkpoint hashes and the predeclared
  decision reproduced. The source snapshot and audit source are retained.

## Continued research window and controlled retention test — November 9, 2026

- User extended authorized CPU research through **09:00 America/New_York**
  (14:00 UTC). User also explicitly approved the auxiliary BP→NBP→OSD study with
  longer-BP and two-BP-pass controls; `docs/m6_cascade_proposal.md` records the
  approved scope. Original E1/E2 and other main experiments remain intact.
- Within the original single-pass pipeline, test the specific retention hypothesis:
  the same 128 hard training cases plus 128 initially logically correct anchors.
  Both arms receive identical 16-shot forward batches (eight hard/eight easy),
  initialized models, 800 updates, Adam/cosine schedule, clipping, and all seeds.
  The control gives easy cases zero loss weight; retention gives hard and easy
  cases equal positive weight. Thus both see the same hard examples and differ
  only in whether easy-example loss contributes to gradients.
- Config `cpu_retention.yaml`, hash `dda8532f05365dd03849a69bb06dff3be9009831`.
  This deliberately reuses the audited 128-case hard training set. New easy-pool
  seed **38234** and unseen seed **40234** are separate from earlier selection,
  comparisons, tuning and final tests. Freeze 2,048 unfiltered unseen cases and
  the entire shared shuffled batch plan before either arm trains. Shuffle entropy
  is **[1234, 913]**; no unseen checkpoint selection or early stopping.
- Preparation has a 180-second CPU cap; each arm has a 1,200-second cap with
  120 seconds reserved for finalization. Fixed final checkpoints only; both equal
  800-update budgets must complete before a comparison can be called complete.
  Stop training to honor the user's 14:00 UTC deadline if it becomes limiting.
- Primary practical screen: retention must reduce fallback and standalone logical
  failure by at least 2 pp versus *both* hard-only control and independently tuned
  BP, with all four paired tests passing one joint Holm .05 family. Training hard
  rescue/easy retention and initial-success gains/losses are descriptive diagnostics.
  No OSD/latency/three-seed main claim follows from this single-p, single-seed test.
- Targeted checks verify that zero-weight easy cases contribute no gradient and
  that both groups receive complete, seeded matching epochs. Main/reference code
  is unchanged. The cascade study is separate and will include extra-compute
  controls before attributing any reduction to learning.

### Approved cascade control tuning (predeclared)

- User explicitly approved the separate cascade study and then extended work to
  09:00 New York time. The saved-case screen is labelled posthoc: ordinary tuned
  BP15→initial .95 BP15 had 1,019/2,048 nonconvergences (49.756%), while tuned
  BP15→learned NBP15 had 1,039 (50.732%). Both improve over single-pass tuned BP's
  1,249 (60.986%), but this screen gives no learning advantage over a fixed extra
  pass. This motivates strong controls; it is not independent confirmation.
- `cpu_cascade.yaml`, hash `277fb140c4406bd9fffb5d5da99aefbeea06272f`, freezes
  p=.001/.003, 1,024 independent validation cases/p with seeds **42234–42235**.
  First-pass BP15 uses the original independently tuned table with its provenance.
  Tune single-pass BP30 and conditional second-pass BP15 on the 11-factor .50–1.00
  grid. All selected controls minimize whole-pipeline nonconvergence, breaking ties
  toward the lower factor. Preserve raw corrections, logical flags and iterations.
- Each validation point has a 1,200-second CPU limit. Confirmation uses a separately
  reserved 4,096-case/p stream, seeds **44234–44235**; it is not sampled during
  control tuning or neural candidate selection. All inference routing must use
  syndrome validity, never logical labels. The routing test explicitly retains
  first-pass syndrome-valid wrong-logical outputs and accounts for both passes.
  No OSD/LER-equivalence/latency claim follows from this Stage-1 control tuning.

### Conditional second-pass training and selection discipline

- The approved cascade will route only first-pass failures to its learned decoder.
  Train two independent data/shuffle streams on **1,024 unique BP15-first-pass
  failures** at p=.001, rather than failures of a different .95 neural initializer.
  Training pool seeds **48234/48235**, shuffle entropies **[1234,917]/[1234,918]**.
  The neural initializer, structural T=15, parity loss, batch 8, Adam/cosine,
  1,200-update budget and clipping remain fixed. Training includes the second
  decoder's initially correct cases naturally present among first-pass failures.
- The independent cascade validation pool (seed 42234) is explicitly reused for
  candidate scoring, with its original cases hash/selection metadata required.
  It is *validation*, not a new independent confirmation. The normal fixed-case
  runner still rejects accidental seed reuse; a narrow tested exception permits
  this declared source directory only. The original diagnostic's executed source
  remains preserved, and its numerical model/loss helpers are unchanged.
- Config hashes: `e959d9874a26cdcb8fac8c7c7f5ca1773b5af8e2` and
  `d450d95ab06ffcdc2e61edcd2820b8236cecf541`. Each CPU run has the same 1,200 s
  hard deadline with 120 s reserved for final scoring, training checks every 300
  updates, and fixed final checkpoint. No candidate is chosen on confirmation data.
- The original fixed-case candidate, retention candidates and conditioned candidates
  are development choices. All model/pipeline choices must be frozen on validation
  before the reserved confirmation seeds 44234–44235 are sampled. Main final seeds
  remain untouched. Candidate selection should finish by 07:30 New York time,
  leaving the final window for independent outcomes, actual latency and verification.
- An additional guard test caught an absolute/relative path comparison error in
  the declared-validation exception. The first two conditional jobs were launched
  before that test result was inspected; both were interrupted during initial
  scoring, before any optimizer update. Their frozen pools, source snapshots and
  `FAILURE.txt` files are retained. Canonical path comparison fixed the guard;
  all seven targeted conditional/retention/routing checks then passed before the
  corrected jobs were launched. Sampling parameters and all numerical training
  functions stayed fixed; the original fixed-case run's executed source is intact.

### Retention result and validation-only continuation catalogue

- Both retention arms completed 800 matched batches. Hard-only final training had
  24/128 hard and 102/128 easy cases correct; retention had 26/128 hard and 93/128
  easy correct. On fresh seed 40234, demand was **61.71875% hard-only**, **61.91406%
  retention**, **61.76758% tuned BP**, and **61.86523% initial**. All four corrected
  paired p values were 1.0; `useful=false`. Merely adding equal-weight easy losses
  did not establish the proposed preservation mechanism. Audit is running.
- Independent control tuning chose BP30 factors .95 at p=.001 and 1.0 at p=.003;
  validation demand was **39.453125%** and **90.917969%**. Conditional BP15 second
  factors .90/.75 gave **45.117188%/94.628906%**. These are selection scores, not
  held-out confirmation. Longer ordinary BP is a substantially stronger control.
- `cpu_cascade_candidates.yaml` fixes five learned source candidates (the completed
  fixed-case, both retention, and both conditioned streams). Each is checked in
  three validation-only designs: BP15→NBP15, single NBP30 with its trained 15-step
  schedule cycled, and single NBP30 repeating its last trained parameters after
  step 15. Continuations involve **zero additional optimizer updates**; they are
  explicit E5-style inference checks, not models trained at T=30. All use at most
  30 Stage-1 iterations, matching the stronger BP controls' maximum budget.
- All 15 options are developmental validation choices; selection minimizes total
  validation nonconvergence with deterministic name/design tie breaking. The
  reserved confirmation pools remain unsampled. Source checkpoint hashes, complete
  training budgets and exact Torch/Numba exports must verify before scoring.
- Added an auxiliary `CascadeDecoder` implementation with explicit per-pass routing
  and iteration accounting. Tests retain first-pass syndrome-valid wrong-logical
  outputs, skip the second pass after success, and reject mismatched priors/flags.
  Continued-model tests preserve the trained prefix and verify the exact explicit
  parameter schedules plus Torch/Numba identity. Three targeted tests passed.
- Read-only structural diagnostics found **16,164 unique detector columns out of
  16,164 faults**, with no exact parallel/duplicate columns or zero-detector columns.
  A simple duplicate-column merging explanation is therefore unsupported; this
  does not rule out higher-order graph symmetry or other capacity limitations.

### Matched capacity / learning-rate factorial (predeclared)

- Existing E3 already includes structural versus full edge weights. The read-only
  graph check found no exact duplicate columns, while the small structurally tied
  models showed limited learning/stability. Test capacity and optimization separately
  with a **2×2 factorial**: structural/full weights and initial Adam lr 1e-3/1e-4.
  Both schedules decay by the same factor 100; all other settings are matched.
  This is an auxiliary development test, not a change to main configs or a claim
  that graph symmetry or learning rate has been established as the cause.
- Freeze **32,000 unique first-pass BP15 failures**, at p=.001, using new pool
  seed **52234**, 512-shot generation batches, and at most 65,536 draws. A single
  shared permutation with entropy **[1234,921]** supplies exactly 4,000 eight-shot
  batches, one use of each selected case. This increases training diversity without
  choosing training cases using any confirmation outcomes.
- All four arms start at .95 min-sum, alpha=1, T=15, with the same parity/logical
  full-unroll objective, clip=1, priors, and fault labels. Only the documented
  sharing and learning-rate factors differ. Structural has 1,455 parameters; full
  has 1,366,755. Matching batch hashes must verify across all completed arms.
- Preparation cap is 600 s; each CPU-only arm has a 3,600 s cap with 120 s reserved
  for finalization, plus the 14:00 UTC user deadline. Validation uses only the
  existing 42234–42235 pools, every 500 updates. Select the earliest minimum total
  BP15→NBP15 validation nonconvergence, saving best and last parameters. Complete
  all 4,000 updates before treating an arm as complete. Confirmation remains
  reserved until all final model/pipeline choices are frozen.
- Published neural-BP work motivates checking weight flexibility and cautious
  initialization, but its code/noise/model differ from this circuit experiment;
  it does not establish that either factor will help here. See
  [Liu and Poulin, 2019](https://arxiv.org/html/1811.07835v2), which discusses
  learned weights and symmetry-breaking patterns in its studied code families.

### Auxiliary sine parity surrogate (predeclared before its runs)

- The product-of-probabilities objective can have very small direct probability
  gradients on high-degree uncertain rows. Test a separate absolute-sine parity
  surrogate adapted from Eq. 8 of the cited Liu/Poulin paper on the same circuit
  H/L constraints. It is an adaptation, not a replication of their architecture
  or noise model. Both objectives retain the logical weight beta=5 and the same
  rising per-iteration multi-loss. The original objective/defaults remain intact.
- Add exactly two auxiliary arms, structural/full at lr=1e-4, matched to the
  corresponding product-loss controls on the identical 32,000-case pool, 4,000
  batches, optimizer schedule, initial parameters, validation pools and time caps.
  The only numerical treatment is the parity surrogate; result directories and
  source snapshots are separate. No probability-product floor is used by the sine
  surrogate. Its constructor and settings record that explicitly.
- The alternate driver is derived from the capacity driver, base SHA-256
  `8eafe106501b6af0c1cca5c8ef9caab2bacc218dd491284b06ca7e31a8477318`, with the
  objective factory and output/protocol metadata changed. Its complete executed
  source is saved per run. Core config hashes intentionally match the matched
  low-rate controls; a full experiment hash additionally includes the surrogate
  settings. Primary experiment configs and ongoing product-loss code are unmodified.
- Two tests passed before launch: degeneracy/logical-penalty checks, float64
  gradcheck, and a high-degree toy showing finite non-vanishing sine probability
  gradients where the product gradient is tiny. That toy does not establish the
  cause of the circuit learning failure or predict useful decoding improvements.
  All new model selection remains validation-only; confirmation is still reserved.

### Frozen auxiliary choice and independent assessment protocol

- All four capacity/rate and two sine arms completed 4,000 updates. The expanded
  validation catalogue has 11 source checkpoints × 3 inference designs. Its minimum
  was **structural, lr=1e-3, 4,000 updates, cycle the learned 15-step schedule to
  30 inference iterations**: 372/1,024 nonconvergences at p=.001 (36.328125%) and
  927/1,024 at p=.003 (90.527344%). Tuned BP30 had 404/1,024 and 931/1,024.
  These remain model-selection observations; the winner among many options can
  be optimistic. Larger parameter count, lower lr and sine loss did not win.
- Frozen bundle `results/m6_aux_selection/20261109T104322214172Z/` copies the
  selected source checkpoint (SHA-256
  `48591637735919bb3e12febe8793eca7f9a81e8e23584dfe53a07d9f6b098c3d`), the
  builder/extension source and all ordinary control choices. **768** prefix
  builder/correction/flag/iteration checks matched saved validation outputs.
- Before confirmation sampling, two method replications were precommitted at
  **10:50:18 UTC**, using training seeds 1235/1236, new pool seeds **68234/69234**,
  and shuffle entropies **[1235,921]/[1236,921]**. The final 4,000-update checkpoint
  and cycle30 rule are fixed for both, irrespective of primary confirmation.
  Their source/data configs and code hashes are in `cpu_aux_replication_plan.yaml`.
  These are auxiliary method replications, not official main M6 training runs.
- Fixed Stage1 confirmation: **4,096 cases/p**, seeds **44234–44235**, primary
  selected learned versus validation-best ordinary control, Holm over two p values;
  architecture-matched initialization is a separate two-test learning family.
  The predeclared screen requires at least 1 pp lower grid mean, at least one
  corrected primary benefit, no corrected primary harm, and a corrected learning
  benefit plus lower mean versus initialization. No model replacement after testing.
- Regardless of that screen, the fixed selected/ordinary/initialization pipelines
  receive auxiliary paired two-stage accuracy assessment, targeting **16,384 cases
  at p=.001 and 1,024 at p=.003**. The initial confirmation pool is reused without
  selection; new extension streams are **64234/64235**. Compute the common deterministic
  OSD result once for cases requested by any pipeline, then compose each actual
  Stage1 decision with that result. Report **per-model fallback requests separately
  from actual unique executions** in a custom accuracy table; this is an evaluation
  optimization, not a deployed cache or timing improvement. No fabricated physical
  call counts enter the standard results schema.
- Accuracy cap 6,300 s, point budgets 5,700/600 s. Any incomplete fixed target is
  labelled budget-limited. For completed fixed samples, report logical rates and
  joint one-sided bounds on learned-only logical failures versus ordinary and
  initialization (four bounds, alpha/4). No equivalence margin or claim is invented.
- Actual idle-machine timing uses **fresh uncached fallback calls**, 512 measured
  cases plus 64 warmups/p, seeds **66234–66235**, for BP15, the ordinary control,
  initialization and selected model. Randomize model order per paired case with
  entropy **[1234,965,p-index]**. Cap 2,400 s and stop by 13:50 UTC for finalization.
  This is explicitly a pilot, below main E2's 20k/1k regime; p99 uncertainty must
  be shown rather than declaring the main latency criterion passed.
- A loaded-machine budget calibration (not a performance comparison) measured
  16 fallback calls/p after two warmups: means .473 s at p=.001 and .569 s at
  p=.003. It explains the need for shared accuracy work and bounded timing samples.
  All main final-test pools and M6/M7 completion status remain unchanged.

### Auxiliary verification and stronger-control challenge (11:17 UTC)

- The first fixed independent Stage1 confirmation passed its declared screen:
  1,640 versus 1,830 failures at p=.001, and 3,671 versus 3,680 at p=.003,
  each on 4,096 paired cases. Only the low-p primary difference is significant
  after the two-point correction. Matched initialization had 1,803/3,751 failures;
  both learned-versus-initialization points passed their separate correction.
  This is a Stage1 result, not a complete LER/latency or M6 result.
- Its audit regenerated 8,192 cases, verified 57,344 raw outcomes, replayed 896
  decoder cases and reproduced the two paired families and screen. Full suite:
  **134 passed in 30.70 s**. Full-pipeline accuracy and precommitted training-seed
  replications remain in progress. Timing starts only after known CPU jobs finish.
- Add a clearly **secondary, posthoc** challenge using ordinary BP at T=50/100,
  which are already in the original handoff baseline sweep. Tune all eleven
  factors on existing validation, freeze those choices, then assess the already
  fixed auxiliary confirmation cases. This was proposed after seeing the T30
  comparison and cannot replace it or acquire a predeclared-primary label.
  Source/settings/results are separate; the selected learned model is unchanged.
- The read-only oracle-opportunity audit initially used .50-style filenames while
  the saved sweep uses .5. Preserve that failed audit source/error; the corrected
  replay reproduced 45,056 saved validation factor outcomes. No selector has been
  trained; its proposed future confirmation seeds are now 63234–63235 because
  44234–44235 were used by the separate learned-message study.
- Latency bootstrap settings are frozen before timing: 5,000 paired case
  resamples, entropy [1234,967,p-index,baseline-index], nominal descriptive 95%
  intervals for selected-minus-baseline mean and p99. They are not a main E2 gate.
- Before inspecting T50/T100 outcomes, also specify a separate secondary latency
  pilot for both longer controls, frozen learned30 and ordinary30. New seeds
  72234/72235, 512 measured + 64 warmups, randomized order [1234,968,p-index],
  5,000 paired bootstrap resamples [1234,969,p-index,baseline-index]. Run only
  after original timing and known compute jobs finish, cap 1,800 s and stop by
  13:50 UTC; budget-limited samples cannot pass main E2. This assesses whether
  extra ordinary iterations are practically preferable, without selecting a new
  learned checkpoint or changing the original experiment matrix.

### Completed auxiliary diagnostics and provenance audits

- Retention study `results/m6_retention/20261109T090301820313Z-dda8532f`:
  adding initially correct anchors to the loss did not help. On 2,048 fresh
  cases, fallback demand was 61.719% hard-only and 61.914% retention, versus
  61.768% tuned BP; all four corrected tests were inconclusive. Its audit passed
  800 paired input hashes, 6,912 model cases, 384 exact exports and four CSV rows.
- Both completed 1,024-case conditional-training streams received 1,200 updates.
  Neither met its useful-learning screen. Each audit regenerated its mining and
  reused-validation sets, checked every update hash/schedule and first loss,
  replayed 8,192 model cases and 512 exact export cases, and checked five CSV rows.
  The earlier pre-update interrupted attempts remain recorded separately.
- The six-arm capacity/rate/sine audit regenerated **52,736 mining cases** and
  **32,000 unique selected failures**, reproduced the shuffle and all input hashes,
  and checked all **24,000** recorded updates and their learning-rate schedules.
  Initial losses reproduced exactly for every arm. It verified **122,880** raw
  validation outcomes, replayed **1,536** neural/export cases, and reproduced
  earliest-minimum checkpoint selection and checkpoint hashes. It does not claim
  to replay every optimizer update. Evidence:
  `results/m6_capacity/20261109T093911371390Z/audit.json`.
- The structural/high-rate arm won validation; full weight flexibility, lower
  learning rate and the sine surrogate did not improve the selected result.
  Those negative arms and their complete checkpoints/source are retained.
