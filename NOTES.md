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
