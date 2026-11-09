# Using the frozen cycle100 CPU decoder

The auxiliary study uses a model trained for 15 iterations and cycles those
learned weights through up to 100 inference iterations. The checkpoint stores
the original 15-step parameters. Use the saved builder to obtain the tested
100-step rule. Restoring that checkpoint directly gives the 15-step model.

The study covers bb72 circuit-level noise, six extraction cycles, at p=.001 and
p=.003. The main experiment matrix and final performance claim remain pending.
See the [independent study](../results/m6_cycle100_study/20261109T114431070320Z/summary.json)
and [exact 10,000-case implementation check](../results/m6_cycle100_export_acceptance/20261109T122411320866Z/summary.json).

From the repository root, this example decodes eight already saved development
syndromes. It is a functional example, not a timing experiment or new test set.

```sh
BBNBP_CACHE_DIR=.cache/bbnbp MPLCONFIGDIR=.cache/matplotlib OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python - <<'PY'
from pathlib import Path
import runpy
import numpy as np

from bbnbp.config import load_config
from bbnbp.decoders.bp_ldpc import LdpcBPOSD
from bbnbp.decoders.pipeline import TwoStageDecoder
from bbnbp.noise.factory import make_sampler

study = Path("results/m6_cycle100_study/20261109T114431070320Z")
config = load_config(study / "config.yaml")
problem = make_sampler(config, 0.001).problem
builder = runpy.run_path(str(study / "executed_source.py"))
stage1 = builder["neural"](study, problem, "selected").export_cpu()
spec = config.eval.osd
fallback = LdpcBPOSD(problem, spec.bp_iters, spec.ms_scaling, spec.method, spec.order)
decoder = TwoStageDecoder(stage1, fallback)

with np.load(study / "cases-p0.npz") as saved:
    result = decoder.decode_batch(saved["syndromes"][:8])
print({"fault_corrections_shape": result["e_hat"].shape,
       "fallback_calls": int(result["osd_used"].sum())})
PY
```

Inputs have shape `[shots, detectors]`. `e_hat` contains corrections in the DEM
fault space; multiplying it by `problem.L` gives predicted observable flips.
The first stage accepts the first syndrome-valid correction. Remaining cases
invoke a fresh BP50/OSD-CS7 fallback using the saved configuration.

The saved study and source checkpoint are immutable evidence. Fresh experiments
need separate declared seeds and result bundles; the original main final-test
seeds remain reserved.
