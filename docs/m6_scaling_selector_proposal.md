# Proposed auxiliary neural scaling selector

Status: completed and independently audited after explicit user approval on
2026-11-09. Fresh confirmation did not establish a benefit; see the
[completed report](../results/m6_scaling_selector/20261109-approved/report/report.md).
The original learned-message experiments
and approved cascade controls stay intact.

The existing validation results show substantial complementarity between the
eleven ordinary BP30 factors .50–1.00. An oracle union has 164 versus 404 failures
at p=.001, and 838 versus 931 at p=.003. Those numbers are posthoc family upper
bounds on reused validation cases, not an implementable decoder or confirmation.

Proposed model: a small CPU neural network predicts a BP scaling factor from the
syndrome and the known noise rate, then runs exactly one BP decode with at most
30 iterations. It never receives true faults, observable labels, or per-case
decoder outcomes at inference. Network computation is included in timed cost.

Training uses 32,000 unfiltered cases per p (.001 and .003), separate from
validation/confirmation. For p=.001, reuse the first 32,000 unfiltered cases from
the existing independently sampled mining pool, not its selected hard subset.
Generate a new p=.003 pool with seed 56234. Execute all eleven BP30 factors on
these training cases to produce actual convergence/logical-success labels.
Preserve case hashes, labels, source code, and all failed attempts.

Use a multilabel success-prediction objective. Candidate inputs are syndrome bits,
known p, and static graph-derived local pair/count features; no additional BP
decode is hidden in feature extraction. Compare a small linear predictor and a
two-hidden-layer predictor. A conservative validation-selected switching margin
can retain the fixed tuned factor when predicted gains are uncertain.

Controls include the independently tuned fixed factor, a cheap syndrome-count
binning rule trained on the same outcomes, and the existing longer/two-pass BP
controls. Any selector must beat the strong fixed control on new cases and retain
logical accuracy while accounting for inference overhead. An oracle bound never
counts as an achieved result.

Use the existing 42234–42235 validation pools for all development choices. Freeze
the model, preprocessing, margins and fallback policy before sampling the reserved
63234–63235 confirmation cases. The originally proposed 44234–44235 stream has
since been used for the separate, frozen learned-message confirmation; it can no
longer serve as fresh confirmation for a future selector. Reuse of validation does not constitute independent
confirmation. Main final-test pools stay reserved. All computation is local CPU,
under `configs/m6/cpu_scaling_selector.yaml`. The earlier 09:00 New York window
ended before this approval; it does not apply to the newly authorized study.
Each labeling worker has a two-hour wall cap, training/selection a 30-minute cap,
and fresh confirmation a 30-minute cap. Intermediate chunks and failures are
preserved. No GPU or paid service is used.

The executable protocol fixes joint training over both rates, syndrome bits,
normalized syndrome count and known p as features. Linear and 64/32-hidden-unit
ReLU predictors use multilabel BCE on logical success (nonconvergence is failure).
Validation chooses architecture, checkpoint and switching margin by Stage1 logical
failures, then fallback count, then simpler architecture/earlier checkpoint/margin.
The fixed factor still uses independent validation convergence tuning. A separate
ten-quantile syndrome-count rule uses the same training labels with 32-case prior
shrinkage. No learned message weights or previous models are modified.

Confirmation uses 4,096 cases per p, sampled once per point with seeds 63234/63235,
after saving all choices and checkpoint hashes. Paired fallback comparisons of the
neural selector against fixed BP30 and the count rule (two rates, four contrasts)
form one Holm-corrected family. BP100/BP130 and two-pass controls are descriptive
stronger-compute challenges. Full pipeline accuracy and uncached CPU timing follow
only if the frozen selector improves fallback demand over fixed BP30 at both
points; otherwise retain and report the negative result without a latency claim.
Neither this study nor a small timing pilot completes main M6/E2.
