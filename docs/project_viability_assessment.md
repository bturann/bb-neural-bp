# QuantAI project viability assessment

Assessment date: 9 November 2026, New York / 10 November 2026 UTC.
Scope: review of the original plan, saved experimental evidence and primary
literature. No training, decoding experiment or timing benchmark was launched.

Course decision after this review: the lab-to-market deliverable must demonstrate
a successful, useful product. The current research result does not meet that
requirement. The recommendation under that constraint is to switch projects;
this repository preserves the exploratory work as a research archive.

## Recommendation

Do not continue expanding or scaling the current approach unchanged. Hold the
prepared 90-minute and full timing runs while the contribution and baseline
requirements are reconsidered. This is a recommendation, not a claim that neural
BP is impossible or a silent cancellation of the original experiment matrix.

The project has a credible, modest first-stage learning result and a reusable
research implementation. It has not established the original system claim:
materially lower mean and p99 latency at preserved logical accuracy on both BB72
and BB144. More compute would reduce statistical uncertainty; it would not by
itself resolve the baseline, mechanism or novelty gaps.

The implementation and bounded empirical report remain reusable technical
assets. They are not a successful lab-to-market course product. As a competitive
new decoder paper with the original headline, current
evidence is insufficient and my confidence is low. I cannot assign a defensible
numerical probability of publication or future experimental success.

## What the evidence actually establishes

| Question | Finding | Assessment |
|---|---|---|
| Does the implementation function? | Recorded full suite: 142 passing tests; frozen neural/CPU export acceptance matched 10,000 cases; saved-data audits passed. | Good implementation evidence, not proof of research value. |
| Do learned weights help Stage1 at a fixed 100-iteration cap? | At p=.001, 443/4,096 fallback requests versus ordinary100's 495; at p=.003, 2,518 versus 2,654. Primary paired comparisons passed their declared two-point correction. | Yes, for this frozen method, code and two rates. |
| Is that a large effect? | Absolute reductions are 1.270 and 3.320 percentage points; relative reductions in requests are 10.51% and 5.12%. | Modest; these are not relative speedups or LER improvements. |
| Is it robust to training seed? | Three seeds improved in direction at both rates; the two additional low-noise comparisons were not individually significant after their declared correction. | Encouraging, with limited low-noise strength. |
| Does it beat a stronger ordinary decoder in end-to-end CPU cost? | Direct learned100 minus BP130 pilot mean: +4.711 ms at low noise, -11.166 ms at high noise; both nominal paired intervals include zero. | Not established. Do not use the failed cascade comparison as a paired direct-learned100 result. |
| Is logical accuracy preserved? | Learned/ordinary100 full-pipeline failures: 20/8,192 versus 24/8,192, and 177/1,024 versus 181/1,024. No prospective accuracy margin was selected. | No demonstrated accuracy penalty, but no equivalence/noninferiority result. |
| Is p99 improved? | Timing pilots contain 256 measured cases per point and comparison intervals remain inconclusive. | Not established. |
| Does it scale to BB144 or the complete noise grid? | Original main experiments remain incomplete. | Unknown; small parameter count does not establish scaling. |

Evidence: [audited CPU assessment](../results/m6_aux_report/20261109T135141505262Z/report.md),
[independent Stage1 results](../results/m6_cycle100_study/20261109T114431070320Z/summary.json),
[planning report](../results/m6_cycle100_planning/20261110T015309813879Z/report.md).

The auxiliary selector failed fresh confirmation at both rates. The strong
BP100-to-learned30 cascade also failed its stronger ordinary comparison. These
negative results close those tested branches; they do not logically disprove all
learned-message approaches.

## Why the practical headline is at risk

### 1. The strongest relevant ordinary baseline is unresolved

The original local research plan lists ordinary BP iteration caps of 30, 100 and 1,000
among its baseline settings. The recent auxiliary challenges reach 130. We have
not established an ordinary-BP latency-versus-accuracy frontier over iteration
budget, scaling and implementation. Increasing a cap can save enough expensive
fallback calls to offset additional message passing; it can also fail to pay off.
It must be measured rather than assumed either way.

The LSD paper explicitly discusses BP+OSD results using 10,000 BP iterations and
why longer BP can reduce total runtime by avoiding global post-processing. Those
numbers are not a recommended cap for our setup, and published timing values are
not directly comparable to ours. They demonstrate why 100/130 is not automatically
the correct practical stopping point. [Hillmann et al.](https://arxiv.org/html/2406.18655v2)

### 2. Fallback demand is an incomplete proxy for p99

If fast and slow paths are well separated and share a common fallback-time
distribution, an illustrative mixture model is

`F_total(t) ~= 1 - q + q F_fallback(t)`

where q is the probability of fallback. Then overall p99 corresponds approximately
to fallback percentile `1 - .01/q`. Reducing q does not remove the slow path from
p99 when q remains well above 1%. The exact effect depends on which cases are
rescued, the costs of those cases, and the first-stage cost.

For the independent Stage1 request rates, that illustrative conditional percentile
moves from about 91.73% to 90.75% at low noise, and 98.46% to 98.37% at high noise.
These are derived illustrations, not predicted latency improvements. In every
inspected learned100 timing block/noise point, all three observations in the
empirical slowest 1% invoked fallback. There is therefore a concrete reason the
tail claim is harder than the request-count claim.

### 3. There is no prospective definition of preserved accuracy

A syndrome-valid output can still be logically wrong; retaining OSD only for
nonconvergence does not guarantee that the learned method preserves accuracy.
The implementation correctly checks logical failures in evaluation. Similar
observed LERs and overlapping intervals do not establish an acceptable upper
bound on extra logical-error probability. A scientifically justified tolerance,
or a different clearly stated accuracy objective, is needed before a confirmatory
acceptance claim. A tolerance must not be chosen after seeing what the current
bounds happen to allow.

### 4. Scope and exploration have outgrown the decisive question

The source model is a 15-step network trained on a filtered low-noise pool, then
cycled to 100 inference iterations. This is a useful iteration-generalization
finding. It is still narrower than the original main training and scaling matrix.
Structural sharing, a learned scalar schedule and implementation differences
must be separated before attributing a systems advantage to a specific mechanism.

Many branches used the same validation pools. Fresh confirmation streams and
frozen choices are strengths, but each study's correction covers its declared
family, not every adaptive choice made throughout the project. Preserve the
exploratory history and use untouched final data only for a well-justified final
claim. Do not add seed counts as if the same confirmation cases were new data.

## Novelty and competition

This is a focused primary-source review, not an exhaustive priority search. It
does not establish that our exact circuit-level structural-sharing/cycling scheme
has already been published. It does establish that the broad premise needs a
more specific contribution statement.

- [Liu and Poulin, PRL 2019](https://arxiv.org/abs/1811.07835): quantum neural BP
  trained with a degeneracy-aware objective is established prior work. The
  general idea of learning BP weights is not our novelty.
- [Gong, Harvard thesis 2025](https://dash.harvard.edu/items/78dcef0b-782d-43e9-bf9b-a630fc7c8057):
  the institutional abstract explicitly examines NBP+OSD on BB codes and discusses
  difficulty obtaining a BB threshold. This is relevant overlap; the abstract
  alone does not establish a duplicate circuit-level experiment or a matching
  latency methodology.
- [Blue et al., Quantum 2026](https://quantum-journal.org/papers/q-2026-06-30-2149/):
  learned circuit-level BB decoding and runtime evaluation already exist. Its
  transformer, hardware and OSD benchmarking choices differ from ours, so its
  reported ratios must not be transplanted as direct comparisons.
- [Mueller et al., Relay-BP](https://arxiv.org/abs/2506.01779): a BP-based circuit
  decoder reports strong accuracy improvements on BB codes with a structure
  designed for hardware implementation. We have not benchmarked it in our setup.
- [Hillmann et al., LSD](https://arxiv.org/abs/2406.18655): a localized alternative
  targets the cost of global post-processing. A benefit specific to a slow global
  OSD fallback may shrink under a different post-processor; that remains untested.
- [Fregona et al., October 2026 preprint](https://arxiv.org/abs/2610.30305):
  code-capacity BB failure mechanisms include symmetry and numerical issues.
  Its scope differs from our circuit-level task; it reinforces that baseline
  behavior deserves investigation before a learning-based explanation is assumed.

A tiny CPU-trained model could still be a useful contribution if it demonstrates
a distinctive resource/accuracy tradeoff. Small size alone, fewer requests against
one capped baseline, or merely applying NBP to another graph is not sufficient
evidence of that contribution. A negative result can support a good technical
report, but publication is not automatic.

## Compute is not the main scientific blocker

The recorded source training already completed on CPU in roughly 36 minutes.
A GPU could accelerate further training; it would not establish a new research
gap, improve a fixed checkpoint's statistical effect, or settle the CPU fallback
tail. Nothing in the current evidence warrants a hardware purchase for this
hypothesis.

The three-method 20,000-case timing option is estimated at 7.13 hours, or 8.91
with its planning cushion. The 2,000-case option is about 1.27 cushioned hours.
Those costs do not complete all missing main experiments. More samples may show
a benefit, no benefit or a disadvantage. They are not a path that guarantees the
original headline.

## What I recommend doing with the work

1. Hold both proposed timing runs and all new architecture branches now. Keep the
   frozen checkpoint, complete evidence, negative outcomes and untouched final
   streams. No experiment configuration has been changed by this assessment.
2. Consolidate the narrow defensible result into a technical report: learned
   weights modestly reduce Stage1 fallback demand at an equal iteration cap in
   this BB72 setup; practical superiority, strict accuracy preservation and
   BB144 scaling remain unresolved. State those limitations in the title/abstract
   rather than presenting the original headline as nearly proven.
3. Before any renewed experimental effort, require a specific contribution beyond
   existing work and a falsifiable strong-baseline comparison. Establish the
   ordinary iteration/runtime tradeoff and relevance of a contemporary baseline
   on matching noise/circuit conventions before spending hours on tail precision.
4. Continue toward the original paper only if that review reveals a plausible
   unoccupied tradeoff and a fixed acceptance rule with a justified accuracy
   tolerance. Otherwise close or explicitly rescope the project. Do not make the
   automatic fallback project a new promise of success without assessing it too.

For an engineering portfolio or a reproducibility study, the implementation and
measured limits remain useful. They are not presented as a completed lab-to-market
product. For a novel competitive decoder paper, I recommend
a research-direction reset before further compute. This recommendation does not
discard the existing matrix or claim impossibility; it recognizes that its
current strongest evidence supports a smaller conclusion.

I should have prioritized this baseline/novelty review earlier. Repeated successful
software checks and small exploratory screens were too easily treated as forward
progress toward the systems claim. They establish different things. The next
decision should be based on the value of the remaining hypothesis, not the amount
of code or elapsed effort already invested.

## Review provenance

The original 14-page research plan was read with its headings, experiment matrix,
baseline caps, risks and reading list; its first page was rendered for inspection.
The original PDF and implementation specification remain unchanged. The PDF
proposes a narrower circuit-level gap than the broad NBP idea, which is why this
assessment does not label the project a proven duplicate.

The original local planning PDF is not bundled in this archive. Its historical
path/hash remains in the raw review provenance; readers do not need that file to
inspect the experimental results. The public repository preserves the separate
[implementation handoff](implementation_spec.md). The review's baseline-cap
statement describes the local plan; it is not a claim that the implementation
handoff specifies the same sweep.

[Derived evidence and input hashes](../results/project_assessment/20261110-review/evidence.json)
record the Stage1 effect sizes and tail checks. The mixture explanation is an
explicit approximation. Literature claims are attributed to their authors and
are not treated as results reproduced by this project. No new experimental cases,
training, benchmark, dependency or final-test stream was used in the assessment.
