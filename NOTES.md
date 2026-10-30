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
