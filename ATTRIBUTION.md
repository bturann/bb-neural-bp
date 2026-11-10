# Attribution and licenses

The project's own code and associated materials are provided under the
[MIT license](LICENSE). Dependency source and binaries remain under their own
licenses; the root license does not relicense them. Dependencies are installed
from the pinned requirements, rather than vendored in this repository.

## Reference implementation and development

`reference/ref_impl.py` and `reference/test_ref.py` are unchanged copies of the
verified implementation supplied with the original project handoff. The GF(2)
routines in `src/bbnbp/gf2.py` and corresponding code constructions originate from
that handoff. The preserved [implementation specification](docs/implementation_spec.md)
records the source and matching requirements. This repository does not claim
invention of those reference routines or of the cited decoding algorithms.

Implementation, experiment orchestration and analysis were AI-assisted. Research
claims are limited to the saved runs and audits; an automated software check is
not evidence of a scientific or commercial advantage.

## Direct dependencies

This is an attribution overview, not a replacement for the distributions' full
license notices. Versions are the pinned environment used for this archive. Where
a distribution bundles additional components, consult its complete license files.

| Dependency | Version | Declared license / notice | Source |
|---|---|---|---|
| NumPy | 2.4.6 | BSD-3-Clause and bundled-component licenses | [NumPy](https://github.com/numpy/numpy) |
| SciPy | 1.17.1 | BSD-3-Clause and bundled-component notices | [SciPy](https://github.com/scipy/scipy) |
| Stim | 1.16.0 | Apache-2.0 | [Stim](https://github.com/quantumlib/Stim) |
| ldpc | 2.4.1 | MIT | [LDPC](https://github.com/quantumgizmos/ldpc) |
| PyTorch | 2.15.0 | Mixed BSD, Apache, MIT and other bundled-component licenses | [PyTorch](https://github.com/pytorch/pytorch) |
| Numba | 0.68.1 | BSD-family license and bundled-component notices | [Numba](https://github.com/numba/numba) |
| PyYAML | 6.0.3 | MIT | [PyYAML](https://github.com/yaml/pyyaml) |
| pandas | 3.1.0 | BSD-3-Clause | [pandas](https://github.com/pandas-dev/pandas) |
| Matplotlib | 3.11.2 | Matplotlib license, based on the PSF license | [Matplotlib](https://github.com/matplotlib/matplotlib) |
| pytest | 9.2.0 | MIT | [pytest](https://github.com/pytest-dev/pytest) |
| tqdm | 4.70.1 | MPL-2.0 and MIT | [tqdm](https://github.com/tqdm/tqdm) |
| QUITS | 1.2.0 | MIT | [QUITS](https://github.com/mkangquantum/quits) |

The checked installed metadata identifies Joschka Roffe as the ldpc copyright
holder and Mingyu Kang as the QUITS copyright holder. Their license notices and
those of other dependency authors remain applicable to those packages.

## Methods and related work

Please cite the methods relevant to the particular experiment, in addition to
this archive. Related-work references do not imply that this project replicated
their published results or benchmarked their implementations.

- Bravyi et al., *High-threshold and low-overhead fault-tolerant quantum memory*
  (2024): [BB codes and circuit context](https://arxiv.org/abs/2308.07915).
- Roffe et al., *Decoding across the quantum low-density parity-check code
  landscape* (2020): [BP-OSD methods](https://doi.org/10.1103/PhysRevResearch.2.043423).
- Liu and Poulin, *Neural Belief-Propagation Decoders for Quantum Error-Correcting
  Codes* (2019): [quantum neural BP and degeneracy-aware training](https://arxiv.org/abs/1811.07835).
- Nachmani et al., *Deep Learning Methods for Improved Decoding of Linear Codes*
  (2018): [learned message passing](https://doi.org/10.1109/JSTSP.2017.2788405).
- Kang et al., *QUITS: A modular Qldpc code circUIT Simulator*:
  [circuit simulator and citation information](https://pypi.org/project/quits/1.2.0/).
- Blue et al., *Machine Learning Decoding of Circuit-Level Noise for Bivariate
  Bicycle Codes* (2026): [related circuit-level ML work](https://quantum-journal.org/papers/q-2026-06-30-2149/).
- Mueller et al., *Improved belief propagation is sufficient for real-time decoding
  of quantum memory*: [Relay-BP](https://arxiv.org/abs/2506.01779).
- Hillmann et al., *Localized statistics decoding for quantum low-density
  parity-check codes*: [LSD](https://arxiv.org/abs/2406.18655).

See the [feasibility assessment](docs/project_viability_assessment.md) for the
scope and limitations of the literature review.
