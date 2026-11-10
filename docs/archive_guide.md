# Research archive guide

Development is paused following feasibility review. The archive preserves an
exploratory research prototype and its evidence. It establishes neither a
commercially validated decoder nor the original complete performance claim.
The user's lab-to-market course project is moving to a replacement direction.

## Reading order

1. [Project viability assessment](project_viability_assessment.md): what worked,
   what remains uncertain, relevant prior work and the recommendation to hold
   further expansion.
2. [Audited CPU assessment](../results/m6_aux_report/20261109T135141505262Z/report.md):
   completed first-stage, accuracy and timing results with uncertainty and controls.
3. [Negative scaling-selector study](../results/m6_scaling_selector/20261109-approved/report/report.md).
4. [Read-only evaluation planning](../results/m6_cycle100_planning/20261110T015309813879Z/report.md).
5. [Development log](../NOTES.md) and [historical commands](experiment_history.md)
   for earlier attempts and exact execution context.

## Evidence index

| Evidence | Primary bundle | Interpretation |
|---|---|---|
| Frozen learned100 Stage1 comparison | [Summary](../results/m6_cycle100_study/20261109T114431070320Z/summary.json), [audit](../results/m6_cycle100_study/20261109T114431070320Z/audit.json) | Independent first-stage gain at two BB72 noise rates; no system-level claim by itself. |
| Torch/CPU numerical acceptance | [Summary](../results/m6_cycle100_export_acceptance/20261109T122411320866Z/summary.json) | 10,000 matching cases; numerical acceptance, not extra performance samples. |
| Learned100 functional accuracy | [Summary](../results/m6_cycle100_accuracy/20261109T131507713374Z/summary.json), [audit](../results/m6_cycle100_accuracy/20261109T131507713374Z/audit.json) | Shared deterministic fallback outputs assess logical outcomes only; no timing inference from reuse. |
| Longer-BP CPU timing pilot | [Summary](../results/m6_long_bp_latency/20261109T124523000569Z/summary.json), [audit](../results/m6_long_bp_latency/20261109T124523000569Z/audit.json) | 256 measured cases/p, random per-case order, uncached fallback; descriptive pilot intervals. |
| Strong-control CPU timing pilot | [Summary](../results/m6_strong_cascade_latency/20261109T130204474636Z/summary.json), [audit](../results/m6_strong_cascade_latency/20261109T130204474636Z/audit.json) | Includes direct learned100 and BP130; their practical superiority comparison remains inconclusive. |
| BP100-to-learned30 cascade | [Summary](../results/m6_strong_cascade/20261109T113533689212Z/summary.json), [audit](../results/m6_strong_cascade/20261109T113533689212Z/audit.json) | Negative against the validation-selected stronger ordinary control. |
| Scaling selector | [Confirmation](../results/m6_scaling_selector/20261109-approved/confirmation/summary.json), [audit](../results/m6_scaling_selector/20261109-approved/audit-confirmation.json) | Negative; conditional accuracy/timing phases were not launched. |
| Evaluation cost/precision planning | [Analysis](../results/m6_cycle100_planning/20261110T015309813879Z/analysis.json), [audit](../results/m6_cycle100_planning/20261110T015309813879Z/audit.json) | Reused data and labeled hypothetical simulations; no new experiment. |
| Viability review derivations | [Evidence](../results/project_assessment/20261110-review/evidence.json) | Reused Stage1 counts and empirical tail checks; mixture explanation is an explicit approximation. |

The learned100 source checkpoint is
`results/m6_cycle100_study/20261109T114431070320Z/selected-source.pt`, SHA-256
`48591637735919bb3e12febe8793eca7f9a81e8e23584dfe53a07d9f6b098c3d`.
The [inference guide](m6_frozen_cycle100.md) uses the frozen builder and this tracked
model. Related training-seed snapshots are in the same bundle.

## Completed, omitted and proposed work

- Main M0-M5 software/acceptance gates passed. This is not completion of the main
  circuit-level performance experiments.
- Main M6/E1/E2 and M7 remain incomplete. A small pilot or the existence of a
  configuration file does not mark a milestone complete.
- The optional focused timing extension in
  `configs/m6/cpu_cycle100_pair_latency.yaml` was omitted for its original schedule;
  neither its extra sample nor the planned pooled sample exists.
- `configs/m6/cpu_cycle100_decision_plan.yaml` is prepared, not run. Its launch
  flag is false, and its accuracy acceptance tolerance remains unset.
- The scaling selector failed its predeclared gate. Its optional follow-up code
  is preserved, but no selector accuracy/timing dataset was collected.
- Main final-test seeds 11234-11238 remain reserved and unused in reported studies.

## Reproduction and portability

The maintained entry points are the root installation/test commands and the
frozen inference guide. Run commands from the repository root. Historical
experiment commands are explicit reruns, which create new bundles and can be
expensive; nothing runs automatically on checkout.

Raw result files, per-run configs, failed attempts and executed-source snapshots
are historical evidence. They are retained without editing their numerical
contents or rewriting their hashes. Some contain machine-specific paths, labels,
old checkpoint locations or expired deadlines. These document the original run;
they do not imply portability of every historical command or that a deadline is
current. Frozen source copies should be interpreted in their original repository
context, not as standalone packages.

`checkpoints/` is ignored local training output. Some early original checkpoints
are therefore absent from a fresh clone. Reviewed final model copies and saved
evaluation inputs are present under `results/`. Follow the guide's tracked model
path rather than an old absolute path from a log.

The original local 14-page planning PDF is not redistributed. Its path/hash in
review provenance is an external historical input; it is not required to read
the public evidence. The [implementation handoff](implementation_spec.md) is
included. The original viability memo is preserved byte-for-byte as
`results/project_assessment/20261110-review/assessment_at_review.txt`, matching the
historical `assessment_sha256` in `review_verification.json`; the public-facing
assessment adds course-status and portability clarifications without changing
experimental findings.

Source manifests describe execution-time source, so a later documentation or
licensing commit can differ from those historical hashes. Model and data hashes
remain independently checkable. Do not interpret historical proposals or oracle
upper bounds as achieved results, combine shared cases as independent samples,
or infer logical-error equivalence from overlapping intervals.

## Reuse

See [MIT licensing and attribution](../ATTRIBUTION.md) and
[citation metadata](../CITATION.cff). The reusable components include the noise
adapter, checked inference implementations, experiment configuration and audit
patterns. Any new scientific claim needs its own relevant controls and data.
