# Implementation Spec: Learned BP Decoder for Bivariate Bicycle Codes

**Audience:** an autonomous coding agent (and the student supervising it).
**Companion documents:** `01_Research_Plan_Learned_BP_Decoder.pdf` (research context, motivation and schedule) and `reference/` (tested reference code, see Appendix A).
**Version:** 2026-10-30.

---

## 0. Read this first: rules for the agent

1. **Do not fabricate results.** Every number in `results/` must come from code that ran. If something fails, write the failure to `NOTES.md` and stop at that milestone's gate. Do not paper over it.
2. **Verify third-party APIs before using them.** The APIs of `stim`, `ldpc` and `quits` in this spec are written from documentation knowledge and were **not executed** while writing it. Milestone M0 exists to check them. Where this spec and the real API disagree, the real API wins. Record the difference in `NOTES.md` and keep the adapter functions' signatures the same.
3. **Reference code is ground truth.** Files in `reference/` were executed and their tests pass (Appendix A). Copy them into the repo unchanged at M1. The PyTorch model must match `reference/ref_impl.py::minsum_bp` exactly at initialization (test T5.1).
4. **Work milestone by milestone** (Section 7). A milestone is done only when all of its acceptance tests pass. Commit at the end of each milestone, using the message format `M<k>: <summary>`.
5. **Determinism.** Every random draw goes through an explicit `numpy.random.Generator` or a `torch.Generator` seeded from the config. Never use global random state.
6. **No silent defaults.** Every experiment parameter lives in a YAML config, and the resolved config is saved next to every result.
7. **Ask the human** before (a) changing the research question, (b) dropping an experiment, (c) adding a dependency not listed in Section 2, or (d) spending more than 4 GPU-hours on a single run.

---

## 1. What we are building

**Research question.** Can a small *neural belief-propagation* decoder (min-sum BP unrolled for T iterations, with learned message weights) solve more syndromes on its own than standard min-sum BP on bivariate bicycle (BB) codes? If so, the expensive OSD fallback runs less often, at the same logical error rate.

**System under test: a two-stage decoder.**

```
syndrome s ──► Stage 1: BP or NeuralBP ──► ê ── H·ê == s ? ──yes──► output ê
                                                    │no
                                                    ▼
                                    Stage 2: OSD (ldpc BpOsdDecoder) ──► output ê
```

**Headline metrics** (computed for every decoder configuration):

| Metric | Definition |
|---|---|
| `osd_rate` | Fraction of shots where Stage 1 output fails the syndrome check, so OSD runs |
| `ler` | Logical error rate: fraction of shots where L·(e ⊕ ê) ≠ 0 for any logical row, with 95% Wilson interval |
| `ler_stage1` | LER if OSD were never called (non-convergence counts as failure) |
| Latency | Per-shot decode time: mean, p50, p99, max (µs), CPU single thread |

**Target claim.** At equal LER (within confidence intervals), NeuralBP+OSD has a materially lower `osd_rate` and lower p99 latency than (a) BP+OSD and (b) BP with a *tuned single scaling factor* +OSD.

**Codes**

| Name | (l, m) | A | B | [[n, k, d]] |
|---|---|---|---|---|
| `bb72` | (6, 6) | x³ + y + y² | y³ + x + x² | [[72, 12, 6]] |
| `bb144` | (12, 6) | x³ + y + y² | y³ + x + x² | [[144, 12, 12]] |

`bb72` is the main code. `bb144` is the scaling experiment (E4).

**Noise models (in order of implementation)**

1. `code_capacity`: i.i.d. bit flips with probability p on data qubits, perfect syndrome. Decode Z errors with H_X (the X basis is symmetric). Used for debugging and as a secondary result.
2. `phenomenological`: data flips per round plus measurement flips, built directly from H_X (Appendix C). This is the fallback if circuit-level generation stalls.
3. `circuit_level`: a full syndrome-extraction circuit from QUITS, uniform depolarizing noise p, converted to a detector error model (DEM). This is the main result.

All three are presented to the decoders through **one interface**: a check matrix `H` (detectors × faults), an observable matrix `L` (observables × faults), and a prior vector `priors` (one probability per fault).

---

## 2. Environment

- Python 3.11, Linux or macOS. A GPU is optional for M0–M4 and recommended for training (M5+). The CPU-only path must work end to end at small scale.
- Use a virtual environment (`uv venv` or `python -m venv`). Pin versions in `requirements.txt` after M0 resolves them.

```
numpy>=1.26
scipy>=1.11
stim>=1.13
ldpc>=2.0          # v2 API: BpDecoder, BpOsdDecoder (verify in M0)
torch>=2.2
numba>=0.59        # fast CPU inference for fair latency (M6)
pyyaml
pandas
matplotlib
pytest
tqdm
# QUITS: install from https://github.com/mkangquantum/quits (pip install git+... or vendored, see M3)
```

`numpy`, `scipy`, `pyyaml`, `pandas`, `matplotlib` and `pytest` were confirmed working with the reference code. The others are verified in M0.

---

## 3. Repository layout

```
bb-neural-bp/
├── README.md                 # how to install, run tests, reproduce each figure
├── NOTES.md                  # agent's running log: API differences, decisions, failures
├── requirements.txt
├── pyproject.toml            # package name: bbnbp
├── configs/
│   ├── base.yaml
│   ├── cc_bb72_*.yaml        # code-capacity experiments
│   ├── cl_bb72_*.yaml        # circuit-level experiments
│   └── cl_bb144_*.yaml
├── reference/                # copied verbatim from the handoff (Appendix A); never edited
│   ├── ref_impl.py
│   └── test_ref.py
├── src/bbnbp/
│   ├── __init__.py
│   ├── config.py             # dataclasses + YAML load/save, config hashing
│   ├── gf2.py                # GF(2) linear algebra (from reference)
│   ├── codes.py              # BB code construction, logicals, shift automorphisms
│   ├── noise/
│   │   ├── base.py           # DecodingProblem dataclass + Sampler protocol
│   │   ├── code_capacity.py
│   │   ├── phenomenological.py
│   │   └── circuit_level.py  # QUITS adapter + DEM -> (H, L, priors) + sampler
│   ├── graph.py              # Tanner graph tensors: padded check index, masks, weight maps
│   ├── decoders/
│   │   ├── base.py           # Decoder protocol
│   │   ├── bp_ldpc.py        # ldpc BpDecoder / BpOsdDecoder adapters
│   │   ├── neural_bp.py      # PyTorch NeuralMinSum
│   │   ├── fast_nbp.py       # numba weighted min-sum for CPU inference (latency)
│   │   └── pipeline.py       # TwoStageDecoder(stage1, osd)
│   ├── losses.py             # degeneracy-aware parity loss
│   ├── train.py              # training loop (CLI: python -m bbnbp.train --config ...)
│   ├── evaluate.py           # evaluation (CLI: python -m bbnbp.evaluate --config ...)
│   ├── metrics.py            # Wilson CI, latency stats, failure-count stopping
│   └── io.py                 # results CSV writer, checkpoint save/load
├── scripts/
│   ├── m0_verify_apis.py
│   ├── sweep_scaling_factor.py
│   ├── run_experiment.sh
│   └── make_figures.py
├── tests/                    # pytest; see Section 8
├── results/                  # one CSV per experiment + resolved config; never hand-edited
├── checkpoints/
└── figures/
```

---

## 4. Conventions

**Dtypes and shapes.** Binary arrays use `np.uint8`, booleans use `bool`. Dimension names: `B` = batch, `C` = checks or detectors, `N` = faults or variables, `K` = logical observables, `T` = BP iterations, `E` = edges (nonzeros of H), `D` = maximum check degree.

| Object | Shape | Type |
|---|---|---|
| `H` | [C, N] | `scipy.sparse.csr_matrix` (uint8); dense allowed for N ≤ 2,000 |
| `L` | [K, N] | same |
| `priors` | [N] | float64, in (0, 0.5) |
| `llr` | [N] | float, = log((1 − p)/p) |
| syndrome `s` | [B, C] | uint8 |
| true faults `e` | [B, N] | uint8 |
| observable flips `o` | [B, K] | uint8, = L·e mod 2 |

**The `DecodingProblem` dataclass** (`noise/base.py`):

```python
@dataclass(frozen=True)
class DecodingProblem:
    name: str                 # e.g. "bb72-cl-r6-p0.003"
    H: sp.csr_matrix          # [C, N]
    L: sp.csr_matrix          # [K, N]
    priors: np.ndarray        # [N]
    meta: dict                # code, noise_model, rounds, p, basis, n_detectors, ...

class Sampler(Protocol):
    problem: DecodingProblem
    def sample(self, shots: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (s [B,C] uint8, o [B,K] uint8, e [B,N] uint8). Must satisfy s == H e and o == L e (mod 2)."""
```

**The Decoder protocol** (`decoders/base.py`):

```python
class Decoder(Protocol):
    name: str
    def decode_batch(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (e_hat [B,N] uint8, converged [B] bool), where converged = (H e_hat == s)."""
```

**Results CSV schema** (one row per decoder × p × code). Every evaluation writes exactly these columns:

```
run_id, git_sha, config_hash, timestamp, code, noise_model, rounds, basis, p,
decoder, stage1, T, weight_sharing, osd_method, osd_order, ms_scaling,
shots, stage1_failures, osd_calls, osd_rate,
logical_failures, ler, ler_lo, ler_hi, ler_stage1,
lat_mean_us, lat_p50_us, lat_p99_us, lat_max_us, lat_threads, device, seed
```

**Config:** YAML with nested sections `code`, `noise`, `model`, `train`, `eval`, `seed`. `config.py` loads it into frozen dataclasses, computes `config_hash` (sha1 of the canonical JSON) and saves the resolved config to `results/<run_id>/config.yaml`.

---

## 5. Module specifications

### 5.1 `gf2.py` and `codes.py` (M1)

Copy `gf2_rref`, `gf2_rank`, `gf2_nullspace`, `bb_code` and `logical_ops` from `reference/ref_impl.py`. Add the following.

```python
def bb_matrices(name: str) -> tuple[np.ndarray, np.ndarray]:         # "bb72" | "bb144" -> (H_X, H_Z)
def logicals(name: str) -> tuple[np.ndarray, np.ndarray]:            # (L_X, L_Z), cached to ~/.cache/bbnbp/
def shift_permutation(l: int, m: int, a: int, b: int) -> np.ndarray:
    """Permutation of the n = 2lm qubits under the automorphism x^a y^b.
    Left-block qubit i*m + j -> ((i+a) % l)*m + (j+b) % m; the right block (offset lm) uses the same rule.
    The same map applies to X-check and Z-check indices (each block has lm checks)."""
def edge_orbits(H: np.ndarray, l: int, m: int) -> np.ndarray:
    """For each edge (c, v) in np.nonzero(H) order, an orbit id under all l*m shifts.
    Two edges share an id iff some shift maps (c, v) to (c', v'). Use union-find over all shifts."""
```

**Note on `edge_orbits`.** The shift (a, b) must map H to itself under simultaneous row and column permutation. Assert `H[perm_c][:, perm_v] == H` for every shift before computing orbits. For `bb72`, expect a small number of orbits (on the order of the check weight, 6), not one per edge. Record the actual counts in `NOTES.md`.

### 5.2 Noise models and samplers (M2, M3)

**`code_capacity.py`.** Problem: `H = H_X`, `L = L_X`, `priors = p` everywhere. The sampler draws `e ~ Bernoulli(p)` i.i.d. and returns `s = H e`, `o = L e`.

**`phenomenological.py`** (Appendix C). This is the fallback space-time model.

**`circuit_level.py`.** This splits into three functions.

```python
def build_memory_circuit(code: str, rounds: int, p: float, basis: str = "Z") -> "stim.Circuit":
    """Adapter around QUITS. Uniform depolarizing noise p after every 1- and 2-qubit gate,
    on resets and on measurements (SI1000 is NOT used; keep it uniform for comparability).
    Memory experiment: prepare logical |0...0> (basis Z), run `rounds` syndrome cycles, measure data."""

def dem_to_problem(dem: "stim.DetectorErrorModel", meta: dict) -> DecodingProblem:
    """One fault column per `error(p)` instruction of dem.flattened(), in instruction order.
    For each instruction: targets with t.is_relative_detector_id() -> rows of H (t.val);
    t.is_logical_observable_id() -> rows of L (t.val); ignore t.is_separator().
    priors[j] = inst.args_copy()[0]. Do NOT merge duplicate columns (that would break the
    correspondence with DemSampler's error vector). Assert C == dem.num_detectors, K == dem.num_observables."""

class CircuitLevelSampler:
    """Wraps dem.compile_sampler(seed=...). sample(shots) calls
    sampler.sample(shots, return_errors=True) -> (det, obs, err), all bool arrays;
    cast to uint8. Seed each call from the passed numpy Generator (rng.integers(2**63))."""
```

**DEM decomposition.** Call `circuit.detector_error_model(decompose_errors=False)`. BP does not need graphlike decomposition, and decomposition can fail on non-matching codes.

**QUITS integration.** Inspect the repo's README and examples to find the BB-code circuit builder. Wrap it so the rest of the code never imports QUITS directly. If QUITS cannot produce a valid circuit by the end of M3, follow the fallback in Section 7 (M3 gate).

### 5.3 Baseline decoders (M2)

`decoders/bp_ldpc.py` defines:

```python
class LdpcBP(Decoder):      # ldpc.BpDecoder, bp_method="minimum_sum", ms_scaling_factor=alpha, max_iter=T
class LdpcBPOSD:            # ldpc.BpOsdDecoder, osd_method in {"osd0", "osd_cs"}, osd_order
    def decode_one(self, s_row: np.ndarray) -> np.ndarray
```

- Construct each decoder once per `DecodingProblem`, passing `error_channel=priors`, and reuse it across shots.
- `ldpc` decodes one syndrome at a time, so `decode_batch` loops in Python. Convergence for BP comes from the decoder's `converge` attribute; **also** recompute `H ê == s` yourself and assert they agree on the first 1,000 shots.
- Check in M0 whether `ldpc` exposes an OSD entry point that accepts external soft information (posterior LLRs). Record the result in `NOTES.md`. If none exists, Stage 2 always runs a fresh `BpOsdDecoder` on the syndrome (its own BP followed by OSD). This is acceptable: the headline metric is the OSD-call rate, which does not depend on what seeds OSD.

`decoders/pipeline.py`:

```python
class TwoStageDecoder:
    def __init__(self, stage1: Decoder, osd: LdpcBPOSD): ...
    def decode_batch(self, s) -> dict:
        """Returns {"e_hat": [B,N], "osd_used": [B] bool, "stage1_converged": [B] bool}."""
```

### 5.4 Tanner graph tensors (M4)

`graph.py`:

```python
@dataclass
class TannerTensors:
    chk_idx: torch.LongTensor   # [C, D] variable index in each check slot (pad with 0)
    chk_mask: torch.BoolTensor  # [C, D] True where the slot is a real edge
    edge_ids: torch.LongTensor  # [C, D] global edge id (np.nonzero(H) order), -1 on padding
    n_vars: int

def build_tanner(H) -> TannerTensors
def weight_map(tt: TannerTensors, sharing: str, orbits: np.ndarray | None) -> torch.LongTensor:
    """[C, D] index into the weight vector of each iteration.
    sharing="scalar"   -> all zeros (one weight per iteration)
    sharing="orbit"    -> orbit id per edge (code capacity: from codes.edge_orbits)
    sharing="struct"   -> id of the tuple (check degree, var degree, detector round - fault round)
                          for circuit-level DEMs, where exact automorphism orbits are not available
    sharing="full"     -> edge id
    Padding slots point to index 0, and their messages are masked anyway."""
```

For circuit-level `struct` tying, the round of a detector and of a fault come from the circuit's detector coordinates (`circuit.get_detector_coordinates()`; QUITS should set a time coordinate) or from the index layout. Document how you derived it.

### 5.5 Neural model (M4)

`decoders/neural_bp.py` contains `NeuralMinSum(nn.Module)`. This is flooding-schedule min-sum with learnable weights. The update equations:

```
μ_{v→c}^t  = α_t · λ_v + Σ_{c'∋v, c'≠c} m_{c'→v}^{t-1}
m_{c→v}^t  = w^t_{map(c,v)} · (−1)^{s_c} · Π_{v'≠v} sign(μ_{v'→c}^t) · min_{v'≠v} |μ_{v'→c}^t|
Λ_v^t      = λ_v + Σ_{c∋v} m_{c→v}^t                 # note: posterior uses unweighted prior
p̂_v^t      = σ(−Λ_v^t)
```

- **Parameters:** `w ∈ R^{T × n_w}`, initialized to `ms_scaling` (default 0.625). `alpha ∈ R^{T}`, initialized to 1.0. Optionally add a learnable per-iteration **damping** later; it is off by default.
- **Initialization identity.** With `sharing="scalar"`, w = α_ms and α = 1, the model must reproduce `reference.minsum_bp(H, s, llr, max_iter=T, alpha=α_ms)` hard decisions exactly at every iteration (test T5.1). The reference uses `μ = (llr + Σ m) − m_{c→v}`, i.e. prior weight 1. Match it.
- **Implementation pattern** (from the plan; dense over padded checks, no graph library):
  - Gather variable posteriors into check slots: `mu = Lam[:, chk_idx] - m` → [B, C, D]. Put +∞ (use 1e9) in padding slots.
  - Sign product per check × own sign (division-free): `sprod = prod(sgn, -1, keepdim) * sgn`.
  - Two smallest magnitudes via `topk(2, largest=False)`. The extrinsic minimum is `min2` where `mag == min1`, else `min1`. Ties come out right automatically, since both tied slots receive min1 = min2. Checks of degree 1 must get min2 = +∞ before scaling, and their messages are then clamped (below).
  - Scatter back to variables with `index_add_` over `chk_idx[chk_mask]`.
  - **Numerics:** clamp messages to ±`msg_clip` (default 20.0) after each update. Compute in float32; run the exactness test in float64.
- `forward(s, llr, early_stop=False) -> list[Tensor[B, N]]` returns per-iteration error probabilities (for the multi-loss). With `early_stop=True` (inference only), stop when every syndrome in the batch is satisfied, and track per-sample convergence.
- `decode_batch(s) -> (e_hat, converged)` wraps the model to satisfy the `Decoder` protocol. A shot counts as converged at the first iteration whose hard decision satisfies the syndrome. Its `e_hat` is frozen at that iteration, matching how `ldpc` stops.

### 5.6 Loss (M5)

`losses.py` implements the degeneracy-aware parity loss. Given per-iteration probabilities p̂ [B, N] and true faults e [B, N]:

```
q = p̂ + e − 2·p̂·e                       # P(residual fault v is 1) = P(ê_v ≠ e_v)
q = clamp(q, 1e-7, 1 − 1e-7)
For each row r of R ∈ {H, L} (sparse):
   S_r   = Σ_{v∈r} log|1 − 2 q_v|        # via sparse matmul: R @ log|1-2q|ᵀ
   sgn_r = (−1)^{#{v∈r : q_v > 0.5}}     # parity of negatives, via R @ 1[q>0.5] mod 2
   P_even_r = (1 + sgn_r · exp(S_r)) / 2
   loss_r = −log(clamp(P_even_r, 1e-12))
L_t = mean_B [ Σ_{r∈H} loss_r + β · Σ_{r∈L} loss_r ]
Loss = Σ_t γ_t L_t / Σ_t γ_t,  γ_t = (t+1)/T   (weights rising toward the last iteration)
```

Default β = 5.0. Use `torch.sparse` or a precomputed padded index for R. Also implement a baseline `bitwise_bce(p̂, e)` for ablation E3.

**Test T6.1:** for random q with only 0/1 entries, the loss is ≈0 when e ⊕ ê satisfies all rows, and large otherwise. **Test T6.2:** gradcheck in float64 on a 10-variable toy H.

### 5.7 Training (M5)

`python -m bbnbp.train --config configs/<x>.yaml` does the following.

1. Build the problem and samplers. Training data is **streamed**: each step draws a fresh batch, and nothing is stored.
2. **Curriculum** over `p`: linear schedule from `p_start` (e.g. 0.06 for code capacity, 0.005 for circuit level) to `p_end` over `curriculum_steps`, then sample p uniformly from the training grid. For circuit level, p changes the priors, so pre-build one `DecodingProblem` per p on the grid and pick per batch. H's structure is identical across p; only `priors` changes, which you should assert.
3. **Hard-example mining** (`train.hard_mining: {off, filter, upweight}`). Run a frozen `NeuralMinSum` at its initial weights (that is, plain min-sum, on the same device, batched) on the batch. `filter`: keep only the shots where plain BP fails to converge, refilling until the batch is full (cap at 50 draws, then log a warning). `upweight`: weight failed shots by `hard_weight` (default 10) in the loss.
4. **Optimizer:** Adam, lr 1e-3, cosine decay to 1e-5 over `max_steps`, gradient clip 1.0, batch 2,048 (GPU) or 256 (CPU).
5. **Validation** every `val_every` steps on a **fixed** validation set (seeded, 200k shots per p on the grid; 20k for CPU dev runs). The metric is the Stage-1 non-convergence rate (= `osd_rate`), plus `ler_stage1`. Keep the best checkpoint by mean `osd_rate` across p; stop early after `patience` validations without improvement.
6. Log to TensorBoard (or W&B if configured). Save `checkpoints/<run_id>/{best.pt, last.pt, config.yaml}`.

**Default training config (`configs/base.yaml`):**

```yaml
seed: 1234
code: {name: bb72}
noise: {model: code_capacity, rounds: null, basis: Z, p_grid: [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]}
model: {T: 15, sharing: orbit, ms_scaling: 0.625, msg_clip: 20.0}
train:
  batch: 2048
  max_steps: 20000
  lr: 1.0e-3
  loss: parity          # parity | bitwise
  beta_logical: 5.0
  hard_mining: filter   # off | filter | upweight
  hard_weight: 10
  curriculum: {p_start: 0.08, p_end: 0.03, steps: 5000}
  val_every: 500
  patience: 8
eval:
  stop_failures: 100
  max_shots: 10000000
  osd: {method: osd_cs, order: 7}
  stage1_iters: 15      # same T at inference unless the experiment says otherwise
  latency_shots: 20000
```

For circuit level, `p_grid` defaults to `[0.001, 0.002, 0.003, 0.004, 0.005]`, rounds = 6 for bb72 (and rounds = 12 for bb144 if memory allows; see risks), and the curriculum runs from 0.005 to 0.002.

### 5.8 Evaluation (M6)

`python -m bbnbp.evaluate --config ... --decoders bp,bp_tuned,nbp --checkpoint ...`:

- **Test sets** are seeded separately from validation (`seed + 10_000 + i_p`).
- **Failure-count stopping:** draw batches until `logical_failures ≥ stop_failures` or `shots ≥ max_shots`. Report Wilson 95% intervals (`metrics.wilson(k, n)`).
- **Paired evaluation:** all decoders see the **same syndromes** at each p, which lowers the variance of comparisons. Keep the per-shot `osd_used` arrays for a McNemar test of the OSD-call difference (in `metrics.py`).
- **Latency methodology** (E2):
  - Stage 1 for NeuralBP runs through `fast_nbp.py` (numba, the learned weights exported from the checkpoint), **not** PyTorch: Python overhead per shot would make the comparison meaningless. Test T7.2 asserts numba output == PyTorch output on 10k shots.
  - Single thread: `NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`. Batch size 1. Do 1,000 warm-up shots, then time `latency_shots` shots with `time.perf_counter_ns()` around the full two-stage decode of each shot.
  - Report mean, p50, p99 and max, plus the time split between Stage 1 and OSD. Report machine info (CPU model) in the CSV `device` column.
  - Also report, as a secondary metric, batched GPU throughput (shots/s) of the PyTorch model, clearly labelled as a different regime.
- Write results with `io.append_results(csv_path, rows)` using the schema in Section 4.

### 5.9 Scaling-factor sweep baseline (M2/M6)

`scripts/sweep_scaling_factor.py`: for each p, run `LdpcBP` with `ms_scaling_factor ∈ {0.5, 0.55, …, 1.0}` and T ∈ {15, 50, 100}. Pick the factor that minimizes non-convergence at that p (ties → lower `ler_stage1`). Save `results/tuned_scaling.csv`. The `bp_tuned` decoder reads this table. **This is the baseline the paper must beat.**

### 5.10 Figures (M8)

`scripts/make_figures.py` reads only `results/*.csv` and writes PDF and PNG to `figures/`:

- **Fig. 2** (E1): two panels vs p (log-log). Left: `osd_rate` for BP, BP-tuned and NBP, each +OSD. Right: `ler` with Wilson error bars.
- **Fig. 3** (E2): latency CDF per decoder, log-x, with p99 marked.
- **Table 2** (E3): ablation grid → LaTeX table (`figures/table_ablation.tex`).
- **Fig. 4** (E4): Fig. 2 repeated for bb144.
- **Appendix figure** (E5): generalization heatmap, train-p × test-p → `osd_rate`.

---

## 6. Experiment matrix

| ID | Code | Noise | Decoders | Varying | Output |
|---|---|---|---|---|---|
| E0 | bb72 | code capacity | BP, BP-tuned, BP+OSD | p grid | sanity / Appendix |
| E1 | bb72 | circuit level, r = 6 | BP+OSD, BP-tuned+OSD, NBP(orbit/struct)+OSD | p grid | Fig. 2 |
| E2 | bb72 | circuit level | same as E1 | p ∈ {0.001, 0.003} | Fig. 3 |
| E3 | bb72 | code capacity **and** circuit level | NBP variants | sharing ∈ {scalar, orbit/struct, full}; loss ∈ {parity, bitwise}; mining ∈ {off, filter, upweight}; T ∈ {5, 10, 15, 25} | Table 2 |
| E4 | bb144 | circuit level, r = 6 (12 if feasible) | as E1 | p grid | Fig. 4 |
| E5 | bb72 | circuit level | NBP | train p ∈ {0.002, 0.004}, test on full grid; test T ∈ {15, 30, 50} | Appendix |

Each experiment gets one config file and one line in `scripts/run_experiment.sh`. Use 3 training seeds for the main NBP configuration in E1 and report mean ± std of `osd_rate`; ablations use 1 seed.

---

## 7. Milestones, acceptance criteria and gates

| M | Goal | Acceptance criteria (all must pass) | Target |
|---|---|---|---|
| **M0** | Environment + API verification | `scripts/m0_verify_apis.py` prints and saves to `NOTES.md`: installed versions; `ldpc.BpDecoder`/`BpOsdDecoder` constructor signatures and the convergence attribute name; whether OSD accepts external LLRs; `stim.DetectorErrorModel.compile_sampler().sample` signature incl. `return_errors`; the QUITS entry point for BB circuits. | Day 1 |
| **M1** | Codes + reference | `reference/test_ref.py` passes unchanged; T1.x pass (bb72 and bb144: k = 12, H_X H_Zᵀ = 0, weight 6; logical counts 12; shift automorphisms preserve H; orbit counts logged). | Day 2 |
| **M2** | Code-capacity data + baselines | T2.x, T3.x pass; E0 CSV produced; `LdpcBP` non-convergence within ±30% (relative) of the reference sanity table (Appendix B) at p = 0.04, 0.06 with the same α and T = 50; scaling sweep done. | Day 4 |
| **M3** | Circuit-level problem | T4.x pass: circuit builds; DEM has 12 observables; `s == H e` and `o == L e` on 10k sampled shots; BP+OSD LER curve on bb72 decreases monotonically as p decreases; reasonable magnitude (LER at p = 0.001 well below 1e-2). | Day 7 |
| **M4** | Neural model | T5.x pass, including **exact match to reference min-sum at initialization** (float64) and numba == torch. | Day 9 |
| **M5** | Training works | On code capacity: trained NBP (orbit sharing) has lower validation `osd_rate` than BP-tuned at ≥ 4 of 7 p values; loss curves saved. | Day 12 |
| **GATE** | Go / no-go (end of week 2) | Continue to M6 iff M3 passed **and** M5 passed (or NBP within 5% of BP-tuned with a clear downward trend). Otherwise stop and write the status to `NOTES.md` for the human (fallback project). | Day 14 |
| **M6** | Circuit-level training + eval | E1 complete for bb72 with 3 seeds; E2 latency complete; all CSVs + configs saved. | Day 24 |
| **M7** | Ablations + scaling | E3 complete; E4 attempted (OK to report OOM/infeasible with evidence); E5 if time. | Day 32 |
| **M8** | Figures + reproducibility | `make_figures.py` regenerates every figure from CSVs; `README.md` lists one command per figure; fresh-clone test: `pip install -e . && pytest` passes. | Day 35 |

**M3 fallback.** If QUITS cannot produce a valid bb72 circuit by day 7, implement `phenomenological.py` (Appendix C), run E1–E3 on it, and log the reason in `NOTES.md`. Do not write a circuit generator from scratch without asking the human.

---

## 8. Tests (pytest, `tests/`)

| ID | File | Checks |
|---|---|---|
| T1.1 | test_codes.py | bb72/bb144: shape, k = 12, H_X·H_Zᵀ = 0, all row and column weights = 6 / 3 respectively. Compute and assert column weights (each qubit is in 3 X-checks). |
| T1.2 | test_codes.py | `logicals`: 12 rows each; H_Z·L_Xᵀ = 0; L_X·L_Zᵀ has GF(2) rank 12; L_X rows not in rowspace(H_X) |
| T1.3 | test_codes.py | Every shift permutation maps H_X to itself (rows and columns permuted); orbit ids are consistent (edges in one orbit have equal check and variable degree) |
| T2.1 | test_noise.py | Each sampler: shapes, dtypes, `s == H e`, `o == L e` (mod 2) on 1k shots; empirical fault rate ≈ priors (within 5σ) |
| T2.2 | test_noise.py | Same seed → identical samples; different seed → different samples |
| T3.1 | test_baselines.py | `LdpcBP` convergence flag == recomputed `H ê == s` on 1k shots |
| T3.2 | test_baselines.py | `LdpcBP` vs `reference.minsum_bp` at the same α, T: non-convergence rates within 3σ on 5k shots (exact equality not required: scheduling may differ) |
| T3.3 | test_baselines.py | `TwoStageDecoder` output always satisfies the syndrome (OSD guarantees it) |
| T4.1 | test_circuit.py | DEM conversion: C = num_detectors, K = 12, N = number of error instructions; every column has ≥ 1 detector or observable |
| T4.2 | test_circuit.py | Circuit-level sampler satisfies T2.1 |
| T4.3 | test_circuit.py | p → 0: no detection events; zero logical flips on 10k shots at p = 1e-6 |
| T5.1 | test_neural.py | **Exactness:** float64 NeuralMinSum at init (scalar sharing) == `reference.minsum_bp` hard decisions per iteration, on 500 code-capacity shots at p ∈ {0.03, 0.06}. Expose a `per_iter=True` hook in the reference if needed (copy, do not edit the original). |
| T5.2 | test_neural.py | Padding: adding a dummy zero row to H changes nothing |
| T5.3 | test_neural.py | Gradients are finite and nonzero for every parameter after one backward pass |
| T5.4 | test_neural.py | `fast_nbp` (numba) == PyTorch decisions on 10k shots for trained-like random weights in [0.3, 1.0] |
| T6.1 | test_loss.py | Parity loss ≈ 0 for perfect (degenerate-equivalent) corrections; large for logical errors |
| T6.2 | test_loss.py | `torch.autograd.gradcheck` on a toy problem |
| T7.1 | test_metrics.py | Wilson interval against known values (k = 0, n = 100 → upper ≈ 0.037) |
| T7.2 | test_metrics.py | Failure-count stopping halts at the right shot count on a mocked decoder |

Run `pytest -q` before every commit. Mark tests slower than 30 s with `@pytest.mark.slow`, and run them at each milestone end.

---

## 9. Known pitfalls

1. **Basis bookkeeping.** In code capacity, decode Z errors with H_X and check logical failure with L_X (logicals in ker H_Z). Mixing bases gives LER ≈ 0.5 or 0. Test T1.2 guards this.
2. **Detector ↔ check confusion.** In circuit-level DEMs, H rows are detectors (syndrome *differences* between rounds), not stabilizers. Never reuse H_X there.
3. **Duplicate DEM columns.** Do not merge them (Section 5.2). If BP struggles because identical columns split probability mass, add a *separate*, explicitly mapped merged problem as an ablation.
4. **Min-sum ties and degree-1 checks.** Handled by top-2 with +∞ padding (Section 5.5); covered by T5.1.
5. **Float32 overflow.** LLRs at p = 1e-4 are about 9.2. Sums can grow, so clamp messages (`msg_clip`).
6. **Unfair latency.** Compare compiled code to compiled code (numba vs ldpc's C++), single thread, same machine. PyTorch per-shot timing must never appear in the latency figure.
7. **Test-set leakage.** Validation and test seeds must be disjoint, and test sets are only touched in M6/M7 final runs.
8. **Early stopping vs fixed T.** Training unrolls exactly T iterations; inference stops at the first converged iteration. Report the mean number of iterations used per decoder.
9. **QUITS noise model defaults.** Confirm noise is applied to idles or not, and document it. Keep it identical across all decoders.
10. **Memory for bb144 circuit level.** N (faults) can reach tens of thousands. Estimate `B × C × D × 4 bytes` before launching, and reduce the batch or use 6 rounds if needed.

---

## 10. Deliverables checklist

- [ ] Repo with the layout of Section 3, installable via `pip install -e .`
- [ ] `pytest -q` green (slow tests included at final commit)
- [ ] `NOTES.md` containing API findings (M0), orbit counts, every deviation from this spec, and every failed attempt
- [ ] `results/` CSVs for E0–E3 (E4/E5 if done) + resolved configs
- [ ] `figures/` regenerated by one command
- [ ] `README.md` with install, test and "reproduce Fig. N" commands, plus hardware used and wall-clock times
- [ ] Trained checkpoints for the main E1 models (3 seeds)

---

## Appendix A: verified reference code

Copy this to `reference/ref_impl.py` and `reference/test_ref.py` unchanged (both files are also provided alongside this spec). Run with numpy 2.x: **3 tests passed**.

```python
# reference/ref_impl.py
import numpy as np

# ---------- GF(2) linear algebra ----------
def gf2_rref(M):
    """Row-reduce a binary matrix. Returns (R, pivot_cols)."""
    R = (np.array(M, dtype=np.uint8) % 2).copy()
    rows, cols = R.shape
    piv = []; r = 0
    for c in range(cols):
        if r == rows: break
        nz = np.nonzero(R[r:, c])[0]
        if nz.size == 0: continue
        p = r + nz[0]
        if p != r: R[[r, p]] = R[[p, r]]
        others = np.nonzero(R[:, c])[0]; others = others[others != r]
        R[others] ^= R[r]
        piv.append(c); r += 1
    return R[:r], piv

def gf2_rank(M): return len(gf2_rref(M)[1])

def gf2_nullspace(M):
    """Basis (rows) of {x : M x = 0 mod 2}."""
    R, piv = gf2_rref(M); n = np.array(M).shape[1]
    free = [c for c in range(n) if c not in set(piv)]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8); v[f] = 1
        for i, pc in enumerate(piv):
            if R[i, f]: v[pc] = 1
        basis.append(v)
    return np.array(basis, dtype=np.uint8).reshape(-1, n)

def logical_ops(H_X, H_Z):
    """Rows of L_X: X-type logicals = ker(H_Z) independent of rowspace(H_X).
    L_X detects Z errors: Z-error e is a logical failure iff L_X @ (e + e_hat) != 0 mod 2
    when decoding Z errors with H_X. (Symmetric for the other basis.)"""
    ker = gf2_nullspace(H_Z)
    base = H_X.copy(); r0 = gf2_rank(base); out = []
    for v in ker:
        cand = np.vstack([base, v])
        if gf2_rank(cand) > r0:
            base = cand; r0 += 1; out.append(v)
    return np.array(out, dtype=np.uint8)

# ---------- BB codes (verified) ----------
def _S(n): return np.roll(np.eye(n, dtype=np.uint8), 1, axis=1)
def bb_code(l, m, a, b):
    x = np.kron(_S(l), np.eye(m, dtype=np.uint8)); y = np.kron(np.eye(l, dtype=np.uint8), _S(m))
    P = lambda t, k: np.linalg.matrix_power(x if t == 'x' else y, k)
    A = sum(P(*t) for t in a) % 2; B = sum(P(*t) for t in b) % 2
    return (np.hstack([A, B]) % 2).astype(np.uint8), (np.hstack([B.T, A.T]) % 2).astype(np.uint8)

POLY = ([('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)])
CODES = {"bb72": (6, 6, POLY), "bb144": (12, 6, POLY)}

# ---------- batched min-sum BP reference (numpy) ----------
def minsum_bp(H, s, llr, max_iter=50, alpha=0.625):
    """H: [C,N] uint8; s: [B,C]; llr: [N] prior LLRs. Returns (e_hat [B,N], converged [B])."""
    C, N = H.shape; Bsz = s.shape[0]
    ci, vi = np.nonzero(H)                      # edges
    E = ci.size
    m_cv = np.zeros((Bsz, E))
    sign_s = 1 - 2 * s[:, ci].astype(float)     # (-1)^{s_c} per edge
    e_hat = np.zeros((Bsz, N), dtype=np.uint8); conv = np.zeros(Bsz, bool)
    for _ in range(max_iter):
        tot = np.zeros((Bsz, N)); np.add.at(tot, (slice(None), vi), m_cv)
        post = llr + tot
        mu = post[:, vi] - m_cv                 # var->check extrinsic
        sg = np.where(mu < 0, -1.0, 1.0); mag = np.abs(mu)
        # per-check sign product & two smallest magnitudes
        logneg = (sg < 0).astype(int)
        par = np.zeros((Bsz, C), int); np.add.at(par, (slice(None), ci), logneg)
        sprod_c = 1 - 2 * (par % 2)
        big = 1e18
        min1 = np.full((Bsz, C), big); np.minimum.at(min1, (slice(None), ci), mag)
        is_min = mag <= min1[:, ci]
        mag2 = np.where(is_min, big, mag)
        min2 = np.full((Bsz, C), big); np.minimum.at(min2, (slice(None), ci), mag2)
        # handle ties: if two edges share min1, min2 should equal min1
        cnt = np.zeros((Bsz, C), int); np.add.at(cnt, (slice(None), ci), is_min.astype(int))
        min2 = np.where(cnt > 1, min1, min2)
        ext = np.where(is_min, min2[:, ci], min1[:, ci])
        m_cv = alpha * sign_s * sprod_c[:, ci] * sg * ext
        tot = np.zeros((Bsz, N)); np.add.at(tot, (slice(None), vi), m_cv)
        e_hat = ((llr + tot) < 0).astype(np.uint8)
        conv = ((e_hat @ H.T) % 2 == s).all(1)
        if conv.all(): break
    return e_hat, conv
```

**Note for T5.1.** `minsum_bp` stops the whole batch only when *all* shots converge, and it reports the final iteration's decision for every shot. For per-iteration exactness, compare against a copy with a `per_iter` hook that returns `e_hat` after each iteration.

---

## Appendix B: sanity numbers (measured with the reference code)

These are code-capacity numbers for `bb72`, Z errors decoded with H_X, min-sum with α = 0.625, 50 iterations, 4,000 shots per point, numpy seed 2. Use them as a tolerance band for M2 (expect ±30% relative from sampling noise and scheduling differences, not an exact match).

| p | BP non-convergence (= OSD-call rate) | Logical failure among converged shots |
|---|---|---|
| 0.02 | 0.013 | 0.0040 |
| 0.04 | 0.087 | 0.0283 |
| 0.06 | 0.214 | 0.0843 |
| 0.08 | 0.379 | 0.1467 |

At p = 0.01 with 2,000 shots: convergence 0.999, total failure 0.001.

---

## Appendix C: phenomenological noise model (fallback)

This models R noisy syndrome rounds followed by one perfect round, for Z errors and X-type checks H_X (C₀ = 36 checks, n = 72 qubits for bb72).

- **Faults:** data flips d_{t,v} for t = 1..R, v = 1..n, with probability p; measurement flips f_{t,c} for t = 1..R, c = 1..C₀, with probability q (default q = p).
- **Measured syndrome** at round t: σ_t = H_X·(Σ_{τ≤t} d_τ) + f_t. The final round t = R+1 is perfect: σ_{R+1} = H_X·(Σ_{τ≤R} d_τ).
- **Detectors:** D_t = σ_t ⊕ σ_{t−1}, with σ_0 = 0, for t = 1..R+1. This gives C = (R+1)·C₀ detectors.
- **Space-time check matrix** H_st (C × N with N = R·n + R·C₀). Column d_{t,v} has support H_X[:, v] in detector block t. Column f_{t,c} has a 1 at (block t, check c) and at (block t+1, check c).
- **Observables:** L_st = [L_X, L_X, …, L_X (R copies), 0 (for measurement faults)].
- **Priors:** p for data columns, q for measurement columns.

Build this with `scipy.sparse.bmat`. Test it with T2.1 and against the sampler: draw d and f, compute D directly from the definitions above, and check D == H_st·[d; f].

---

## Appendix D: glossary

| Term | Meaning |
|---|---|
| BB code | Bivariate bicycle code, a qLDPC code from Bravyi et al. (2024) |
| BP / min-sum | Belief propagation; min-sum is its log-domain approximation |
| OSD | Ordered statistics decoding; post-processing for BP failures (Gaussian elimination) |
| DEM | Detector error model (Stim): faults × detectors × observables with probabilities |
| Detector | A parity of measurements that is deterministic without noise |
| Code capacity | Noise only on data qubits, perfect syndrome measurement |
| LER | Logical error rate |
| Orbit tying | Sharing one learned weight across all edges related by a code automorphism |
