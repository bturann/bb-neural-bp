"""QUITS memory circuits and a fault-column-preserving Stim DEM adapter.

`rounds` counts total extraction cycles, including the noisy zeroth cycle.
For circuit-level noise, `basis` is the stored/measured logical basis; exact
observable labels come from the DEM rather than code-capacity basis conventions.
"""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version

import numpy as np
import scipy.sparse as sp
import stim

from .._construction import CODES
from ..codes import bb_matrices
from ..config import CIRCUIT_POLICY
from .base import DecodingProblem, validate_shots


def build_memory_circuit(code: str, rounds: int, p: float, basis: str = "Z") -> stim.Circuit:
    if code not in CODES:
        raise ValueError(f"Unknown BB code {code!r}")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer, including the zeroth cycle")
    if not np.isfinite(p) or not 0 <= p < 0.5:
        raise ValueError("Circuit noise p must be in [0, 0.5)")
    if basis not in {"X", "Z"}:
        raise ValueError("Memory basis must be X or Z")
    # All QUITS imports used by the experiment pipeline stay behind this adapter.
    from quits import CircuitBuildOptions
    from quits.noise import ErrorModel
    from quits.qldpc_code import BbCode

    l, m, _ = CODES[code]
    bb = BbCode(l=l, m=m, A_x_pows=[3], A_y_pows=[1, 2],
                B_x_pows=[1, 2], B_y_pows=[3])
    hx, hz = bb_matrices(code)
    if not np.array_equal(bb.hx, hx) or not np.array_equal(bb.hz, hz):
        raise RuntimeError("QUITS parity checks disagree with the reference convention")
    policy = CIRCUIT_POLICY
    circuit = bb.build_circuit(
        strategy=policy.strategy, num_rounds=rounds-1, basis=basis,
        error_model=ErrorModel(idle_error=float(p), sqgate_error=float(p),
                               tqgate_error=float(p), spam_error=float(p)),
        circuit_build_options=CircuitBuildOptions(
            get_all_detectors=policy.get_all_detectors,
            noisy_zeroth_round=policy.noisy_zeroth_round,
            noisy_final_meas=policy.noisy_final_meas),
    )
    if circuit.num_observables != 12:
        raise RuntimeError("BB memory experiment must expose 12 logical observables")
    return circuit


def dem_to_problem(dem: stim.DetectorErrorModel, meta: dict) -> DecodingProblem:
    """Keep every flattened error instruction as one column, in sampling order."""
    flat = dem.flattened()
    c_rows, c_cols, l_rows, l_cols, priors = [], [], [], [], []
    for inst in flat:
        if inst.type != "error":
            continue
        column = len(priors)
        detectors, observables = set(), set()
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                support = detectors
            elif target.is_logical_observable_id():
                support = observables
            elif target.is_separator():
                continue
            else:
                raise ValueError(f"Unexpected target in DEM fault column {column}: {target}")
            # Repeated targets across separator groups cancel by XOR.
            if target.val in support:
                support.remove(target.val)
            else:
                support.add(target.val)
        if not detectors and not observables:
            raise ValueError(f"DEM fault column {column} has no detector or observable effect")
        for row in sorted(detectors):
            c_rows.append(row)
            c_cols.append(column)
        for row in sorted(observables):
            l_rows.append(row)
            l_cols.append(column)
        priors.append(inst.args_copy()[0])
    n = len(priors)
    if n != flat.num_errors:
        raise AssertionError("Flattened DEM error count disagrees with fault column count")
    h = sp.csr_matrix((np.ones(len(c_rows), dtype=np.uint8), (c_rows, c_cols)),
                      shape=(dem.num_detectors, n))
    logical = sp.csr_matrix((np.ones(len(l_rows), dtype=np.uint8), (l_rows, l_cols)),
                            shape=(dem.num_observables, n))
    return DecodingProblem(meta.get("name", "stim-dem"), h, logical,
                           np.asarray(priors, dtype=np.float64), meta)


class CircuitLevelSampler:
    def __init__(self, code: str, rounds: int, p: float, basis: str = "Z"):
        self.circuit = build_memory_circuit(code, rounds, p, basis)
        self.dem = self.circuit.detector_error_model(
            decompose_errors=CIRCUIT_POLICY.decompose_errors).flattened()
        l, m, _ = CODES[code]
        checks = l*m
        detector_rounds = [0]*checks
        detector_bases = [basis]*checks
        for t in range(1, rounds):
            detector_rounds.extend([t]*(2*checks))
            # QUITS custom builder emits Z-check differences before X-check differences.
            detector_bases.extend(["Z"]*checks+["X"]*checks)
        detector_rounds.extend([rounds]*checks)
        detector_bases.extend([basis]*checks)
        if len(detector_rounds) != self.dem.num_detectors:
            raise RuntimeError("QUITS detector layout changed; round metadata must be reverified")
        meta = dict(name=f"{code}-cl-r{rounds}-{basis}-p{p:g}", code=code,
                    noise_model="circuit_level", rounds=rounds, basis=basis, p=float(p),
                    n_detectors=self.dem.num_detectors, n_faults=self.dem.num_errors,
                    circuit_policy=asdict(CIRCUIT_POLICY), quits_version=version("quits"),
                    stim_version=version("stim"), detector_rounds=detector_rounds,
                    detector_bases=detector_bases,
                    round_metadata_source="QUITS custom detector instruction order; no coordinates",
                    rounds_semantics="total extraction cycles; QUITS num_rounds=rounds-1",
                    gate_noise="DEPOLARIZE1(p), DEPOLARIZE2(p); idle DEPOLARIZE1(p)",
                    spam_noise="basis-appropriate flip(p) after reset and before measurement")
        self.problem = dem_to_problem(self.dem, meta)

    def sample(self, shots: int, rng: np.random.Generator):
        validate_shots(shots, rng)
        seed = int(rng.integers(2**63))
        det, obs, errors = self.dem.compile_sampler(seed=seed).sample(shots, return_errors=True)
        return det.astype(np.uint8), obs.astype(np.uint8), errors.astype(np.uint8)
