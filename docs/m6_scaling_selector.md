# Auxiliary CPU scaling selector

The user approved this separate model-class study on November 9 after the earlier
09:00 handover. It preserves the original learned-message studies and experiment
matrix. Main M6/E1/E2 and M7 remain incomplete. The protocol and wall caps are in
`configs/m6/cpu_scaling_selector.yaml`; CPU scheduling is in
`configs/m6/cpu_selector_label_runtime.yaml` and conditional follow-up settings in
`configs/m6/cpu_selector_followup.yaml`.

Each prediction uses syndrome bits, normalized syndrome count and known p. A
linear or 64/32 ReLU predictor estimates logical success for eleven BP30 factors.
Only one selected factor runs for a case. Ground-truth faults, observables and
factor outcomes are training/assessment labels; they are absent from inference.
NumPy evaluates the saved dense network, so timing includes feature extraction,
prediction, routing and the chosen C++ BP pass.

Training uses 32,000 unfiltered cases per p. Validation chooses architecture,
epoch and switching margin separately at each p by Stage1 logical failures, then
fallback count, then simpler architecture/earlier epoch/higher margin. The count
rule uses the same labels with ten count quantiles and 32-case prior shrinkage.
The independently tuned fixed BP30 factor is retained whenever predicted gain
does not exceed the selected margin. All choices and hashes are frozen before
the 4,096-case confirmation streams at 63234/63235. Those streams are now allocated
to this study; reusing them is reproduction, not new independent evidence.

Use the pinned Python 3.11 environment. These commands reproduce the same inputs
and protocol under a **new** output path; existing receipts are not overwritten:

```sh
export BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
selector_run=results/m6_scaling_selector/reproduction
.venv/bin/python scripts/m6_selector_labels.py --output "$selector_run" --index 0 --prepare-only
.venv/bin/python scripts/m6_selector_labels.py --output "$selector_run" --index 1 --prepare-only
.venv/bin/python scripts/m6_selector_label_resume.py "$selector_run"
.venv/bin/python scripts/m6_audit_selector.py "$selector_run" --phase labels
.venv/bin/python scripts/m6_selector_train.py "$selector_run"
.venv/bin/python scripts/m6_selector_confirm.py "$selector_run"
.venv/bin/python scripts/m6_audit_selector.py "$selector_run" --phase confirmation
```

Inspect `confirmation/summary.json`. Follow-up is allowed only when its frozen
`followup_accuracy_and_timing` gate is true; both follow-up commands enforce it:

```sh
.venv/bin/python scripts/m6_selector_followup.py "$selector_run" --phase accuracy
.venv/bin/python scripts/m6_audit_selector_followup.py "$selector_run" --phase accuracy
.venv/bin/python scripts/m6_selector_followup.py "$selector_run" --phase latency
.venv/bin/python scripts/m6_audit_selector_followup.py "$selector_run" --phase latency
```

Known labeling, training and accuracy jobs must finish before latency. Accuracy
reuses deterministic shared fallback corrections for functional assessment only;
timing makes an independent uncached fallback call for every method request.
The 256 measured +64 warmup cases per p are a pilot, with descriptive bootstrap
intervals. They cannot establish the main E2/p99 claim or LER equivalence.

After the required phases and audits, generate the report and standalone figures:

```sh
.venv/bin/python scripts/m6_selector_report.py "$selector_run"
```

For inference, load the frozen confirmation builder, supply the same circuit-level
problem at a supported p, and select its `neural` model:

```python
import json
import runpy
from pathlib import Path
import numpy as np
from bbnbp.config import load_config
from bbnbp.noise.factory import make_sampler

study = Path("results/m6_scaling_selector/20261109-approved")
config = load_config(study / "config.yaml")
frozen = json.loads((study / "selection.json").read_text())
index = 0
problem = make_sampler(config, frozen["points"][index]["p"]).problem
builder = runpy.run_path(str(study / "confirmation/executed_source.py"))
decoder = builder["models"](study, problem, frozen, index)["neural"]
with np.load(study / "p0/cases.npz") as saved:
    syndromes = saved["syndromes"][:8]
# These training examples only demonstrate the API; they are not new evidence.
corrections, converged = decoder.decode_batch(syndromes)
```

Connect `decoder` to the existing `TwoStageDecoder` and configured BP50/OSD-CS7
fallback for functional pipeline inference. Supported noise points, factor order,
standardization, predictor hash and switching margins all belong to the frozen
bundle. Do not infer generalization to other codes, rates or iteration budgets.

The saved eight-case full-pipeline example returned an 8×16,164 correction
matrix with four fallback requests. Its receipt is `inference-example.json` in
the approved bundle. It demonstrates functionality, not independent accuracy.
