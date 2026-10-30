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
- Reference batch convergence differs from first-convergence freezing in inference.
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

## Next milestone

M3: implement the QUITS adapter with explicit noise placement and round semantics,
DEM conversion preserving fault-column order, exact fault-label sampling, and the
six-round circuit-level acceptance tests. The M0 smoke circuit alone does not pass M3.
