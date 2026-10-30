"""Execute M0 checks and replace the generated API report in NOTES.md."""

from __future__ import annotations

import importlib.metadata
import ast
import inspect
import platform
from pathlib import Path

import numpy as np
import stim
import torch
from ldpc import BpDecoder, BpOsdDecoder
from numba import njit
from quits.noise import ErrorModel
from quits.qldpc_code import BbCode

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- M0 API REPORT START -->"
END = "<!-- M0 API REPORT END -->"


def describe(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        # ldpc ships type stubs alongside its Cython extension classes.
        if inspect.isclass(obj):
            stub = Path(inspect.getfile(obj)).parent / "__init__.pyi"
            if stub.exists():
                for node in ast.walk(ast.parse(stub.read_text())):
                    if isinstance(node, ast.ClassDef) and node.name == obj.__name__:
                        for method in node.body:
                            if isinstance(method, ast.FunctionDef) and method.name == "__cinit__":
                                return "Installed stub: " + ast.unparse(method).split("\n")[0]
        # pybind methods publish their signatures on the first docstring line.
        return (inspect.getdoc(obj) or "No signature exposed").split("\n\n")[0]


def verify():
    lines = ["## M0 executed API report", "", f"Platform: {platform.platform()}",
             f"Python: {platform.python_version()}", "", "### Installed versions", ""]
    for name in ("numpy", "scipy", "stim", "ldpc", "torch", "numba", "pyyaml",
                 "pandas", "matplotlib", "pytest", "tqdm", "quits"):
        lines.append(f"- {name}=={importlib.metadata.version(name)}")
    h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    syndrome = np.array([1, 0], dtype=np.uint8)
    for cls in (BpDecoder, BpOsdDecoder):
        kwargs = dict(error_channel=[0.03]*3, max_iter=15, bp_method="minimum_sum",
                      ms_scaling_factor=0.625, schedule="parallel", omp_thread_count=1)
        if cls is BpOsdDecoder:
            kwargs.update(osd_method="osd_cs", osd_order=2)
        decoder = cls(h, **kwargs)
        correction = decoder.decode(syndrome).copy()
        assert np.array_equal((h @ correction) % 2, syndrome)
        assert isinstance(decoder.converge, (bool, np.bool_))
        lines += ["", f"### {cls.__name__}", "", "```text", describe(cls), "```",
                  f"Constructor smoke test passed with {kwargs}.",
                  f"Convergence attribute: `converge` ({decoder.converge}); iterations: `{decoder.iter}`.",
                  "", "decode: `" + describe(decoder.decode) + "`"]
    lines += ["", "### External soft information for OSD", "",
              "`BpOsdDecoder.decode` accepts a syndrome only. Its public API exposes posterior",
              "`log_prob_ratios` as output and `update_channel_probs` for input channel priors,",
              "but no documented standalone OSD method taking external posterior LLRs.",
              "Stage 2 will reuse a separate BpOsdDecoder with fixed problem priors; it runs its",
              "own BP and invokes OSD internally only if needed. Report stage-2 calls separately",
              "from claims about actual internal OSD executions."]
    dem = stim.DetectorErrorModel("error(0.03) D0 L0\nerror(0.02) D0")
    sampler = dem.compile_sampler(seed=123)
    det, obs, err = sampler.sample(shots=1000, return_errors=True)
    assert det.shape == (1000, 1) and obs.shape == (1000, 1) and err.shape == (1000, 2)
    assert np.array_equal(det[:, 0], err[:, 0] ^ err[:, 1])
    assert np.array_equal(obs[:, 0], err[:, 0])
    lines += ["", "### Stim DEM sampler", "", "```text", describe(sampler.sample), "```",
              "`return_errors=True` passed; tuple order is detectors, observables, errors."]
    bb = BbCode(l=6, m=6, A_x_pows=[3], A_y_pows=[1, 2],
                B_x_pows=[1, 2], B_y_pows=[3])
    circuit = bb.build_circuit(strategy="custom", num_rounds=1, basis="Z",
                               error_model=ErrorModel(idle_error=0.001, sqgate_error=0.001,
                                                      tqgate_error=0.001, spam_error=0.001))
    assert isinstance(circuit, stim.Circuit)
    check_dem = circuit.detector_error_model(decompose_errors=False)
    assert check_dem.num_observables == 12
    lines += ["", "### QUITS BB circuit entry point", "",
              "`quits.qldpc_code.BbCode(...).build_circuit(strategy='custom', ...)`",
              "", "```text", describe(BbCode), describe(bb.build_circuit),
              describe(ErrorModel), "```",
              f"bb72 one-round smoke circuit: {check_dem.num_detectors} detectors, "
              f"{check_dem.num_observables} observables, {check_dem.num_errors} error instructions.",
              "Full noise placement and six-round acceptance remain M3 work."]

    @njit
    def add_one(x):
        return x + 1

    assert add_one(3) == 4
    x = torch.tensor([1.0], requires_grad=True)
    (x*x).sum().backward()
    assert x.grad.item() == 2.0
    lines += ["", "Numba compiled CPU smoke test and PyTorch autograd smoke test passed.",
              "", "**M0 acceptance: PASS.**", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        report = verify()
    except Exception as exc:
        with (ROOT / "NOTES.md").open("a") as file:
            file.write(f"\nM0 verification failed: `{type(exc).__name__}: {exc}`.\n")
        raise
    notes = (ROOT / "NOTES.md").read_text()
    if START in notes:
        before, rest = notes.split(START, 1)
        _, after = rest.split(END, 1)
        notes = before + START + "\n" + report + END + after
    else:
        notes += "\n" + START + "\n" + report + END + "\n"
    (ROOT / "NOTES.md").write_text(notes)
    print(report)
