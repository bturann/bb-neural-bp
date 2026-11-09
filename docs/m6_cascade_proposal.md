# Proposed auxiliary retention/cascade study

Status: prepared for human agreement; not an alteration of the main matrix.

The fixed-case diagnostic rescued 232 initially failing unseen cases but lost 238
initial successes. A separate auxiliary pipeline could retain every syndrome-valid
first-pass output and run a learned decoder only on first-pass failures, before
the existing OSD-CS7 fallback. This is a new multi-pass experimental arm; the
original single-pass E1/E2 configurations would remain intact.

No observable label or true fault may control inference. First-pass syndrome
validity is the only gate, including syndrome-valid wrong-logical outputs.

Proposed comparisons, with all choices frozen before fresh evaluation:

1. Independently tuned BP at T=15 followed by OSD-CS7 (existing baseline).
2. Independently tuned BP at T=30 followed by OSD-CS7 (extra-iteration control).
3. Tuned BP15, then a second BP15 with scaling chosen on separate validation
   failures, then OSD-CS7 (extra-pass/scaling-diversity control).
4. The same first BP15, then the fixed learned NBP15, then OSD-CS7.

Reuse the original T=15 tuning only with its provenance checks. Tune T=30 and the
second BP factor on fresh validation data, separate from confirmation and reserved
final seeds. Correct paired tests across the predeclared comparisons and p values.
Any observed advantage must survive the extra-iteration and extra-pass controls;
fewer calls than single-pass BP alone does not establish a learning benefit.

Start with bounded CPU Stage-1 checks at p=.001 and .003. Preserve exact corrections,
logical flags, per-pass routing and source/config provenance. Only if the screen
supports continuing, measure actual two-stage logical performance and end-to-end
CPU latency including both passes and OSD. A pilot cannot replace three-seed main
training or establish rare-event logical equivalence.

Existing saved outcomes can motivate this proposal but cannot serve as its fresh
confirmation. No claim is made from an oracle union of decoder successes.
