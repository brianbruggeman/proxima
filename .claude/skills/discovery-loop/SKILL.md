---
name: discovery-loop
description: Empirical discovery for problems with NO known answer — when you're inventing something nobody has the answer to yet (a new architecture, a new retrieval rule, a sparse-network LLM) and there is no paper, RFC vector, or reference value to reproduce. Replaces the missing oracle with a comparative objective on a held-out set; truth becomes a measured delta against a fairly-tuned baseline that survives ablation, variance, and a scaling ladder — not a hand-derived expected output. Pre-register a falsifiable hypothesis, build the minimal flagged change, measure on held-out, ablate to attribute the gain, keep or roll back, record every result including negatives. Use when developing something whose correct output is unknown, when the objective is a metric not a target, when the danger is fooling yourself (weak baseline, small-scale mirage, eval leakage, metric-hacking). Triggers on "we don't have the answer yet", "discover", "does X actually help", "empirical", "ablate", "beat the baseline", "is this a real win". NOT for reproducing a known answer (use algorithm-development), contested-but-reasoned design (research-rigor), or a known-incumbent perf primitive (disciplined-component).
---

# discovery-loop

`algorithm-development` and `algorithm-rigor` are **verification** — they assume an *oracle* (a paper, an RFC vector, a reference impl, a benchmark answer key) and prove a procedure reproduces it. They cannot front genuinely novel work, because the defining property of novel work is that **the answer does not exist yet**. You can't hand-derive the expected output of a layer in a model you haven't trained; the point of training is to *discover* the weights.

This skill is the front end for that case. It replaces the missing oracle with a **comparative objective on a held-out set**, and replaces "matches the reference value" with "a measured delta against a fairly-tuned baseline that survives ablation, variance, and scale."

The hard part is never getting a number to go up. It is **not fooling yourself** — the whole discipline below exists for that one purpose.

## When to use

- developing something whose correct output is genuinely unknown (new architecture, new scoring rule, new training objective, sparse-network LLM)
- the success signal is a *metric on a distribution*, not a *target output on an input*
- most ideas in this space fail, and you need to tell a real win from noise
- the failure mode you fear is self-deception: weak baseline, dev-set overfit, eval leakage, cherry-picked checkpoint, theoretical-not-real gains

Do NOT use for:
- reproducing a known answer (RFC port, paper algorithm with a worked example) — use `algorithm-development`
- a contested design settled by *reasoning*, not experiment — use `research-rigor`
- a new perf primitive competing with a *known incumbent* on a *known workload* — use `disciplined-component` (it is the bench-driven sibling of this skill; this skill is for when even the objective is uncertain)
- gathering evidence / surveying prior art — use `research-flow` (run it first; see Composition)

## The oracle problem

Verification asks: "does the output equal the known answer?" Discovery has no known answer, so it asks a weaker but still falsifiable question: **"does this change beat a fairly-tuned baseline on held-out data, by a margin that survives ablation and holds across scale?"**

Three substitutions make that question honest:

- **Baseline replaces answer key.** The unit of truth is `metric(new) − metric(baseline)` on data neither one was tuned on. A delta against an *undertuned* baseline is the canonical lie — the baseline gets equal tuning budget or there is no comparison.
- **Held-out replaces the input.** You measure across a distribution, on a slice you did not tune on. Tuning and reporting on the same slice manufactures confidence the substrate can't back.
- **Trend replaces the point.** A single measurement is noise. The claim is a *consistent* delta across seeds (where affordable) and across a **scaling ladder** — the trend is the evidence, the point is an anecdote.

## The loop

### 1. Pre-register the hypothesis

State a **falsifiable** hypothesis with a **kill criterion**, BEFORE measuring. Pre-registration is what stops you retrofitting the story to whatever the number did.

```
Hypothesis: a sparse network reaches dense-equivalent held-out loss at ≤50% of
            the active FLOPs/token, and the advantage does NOT shrink as scale grows.
Axis claimed: iso-active-FLOP (NOT iso-param, NOT iso-memory).
Kill criterion: if the FLOP-matched advantage shrinks monotonically across the
            scaling ladder, the idea is dead — sparse helps small, dies large.
Baseline: a dense model given EQUAL tuning budget (LR sweep, warmup, data order).
Held-out: corpus slice + downstream evals never seen during tuning.
```

If you cannot state a kill criterion, you cannot run a discovery loop — you'll declare victory no matter what happens.

### 2. Minimal build, flagged, baseline preserved

Smallest change that tests the hypothesis, behind a default-off flag, with the baseline path intact and runnable in the same harness. Big builds entangle confounds; you won't be able to attribute the result.

### 3. Measure on held-out

Run new and baseline through the *same* harness on the *held-out* slice. Report the metric you pre-registered, on data neither model was tuned on. Measure **real** cost (wall-clock, actual FLOPs, memory), never theoretical — "fewer FLOPs on paper" is not a result.

### 4. Ablate to attribute

Turn the new mechanism off and re-measure. If the metric doesn't move, the gain was never yours — it was another knob, the harness, or noise. Ablate each *confounding* sub-part too (e.g. an auxiliary loss the new mechanism needed): is the win from the idea, or from the regularization the idea dragged in?

### 5. Variance / scale gate

- **Variance**: re-run with different seeds / data order. If the delta is inside the noise band (CoV, or the spread of baseline-vs-baseline runs), it is not a win.
- **Scale**: re-measure across a ladder of sizes. A delta that shrinks or inverts as scale grows is a **small-scale mirage** — the single most common way architecture work fools itself. When runs are too expensive for many seeds (LLM-scale), scaling-ladder consistency *replaces* seed count as the noise defense.

### 6. Keep or roll back — record either way

- Win that survives 4 + 5 → keep; log the row; proceed to the handoff below.
- Anything else → roll back; **log the negative**. A recorded negative ("sparse routing was FLOP-neutral above 1B params") is worth more than silence — it stops you and future-you from re-running a dead path. Negatives are the majority result in discovery; a log with no negatives is hiding them.

## Worked example — a sparse-network LLM

| Loop step | Applied to "build an LLM with a sparse network" |
|---|---|
| 1. Hypothesis | "Sparse net matches dense held-out loss at ≤50% active FLOPs/token; advantage holds across scale." Kill: advantage shrinks monotonically up the ladder. Axis: **iso-active-FLOP, declared and fixed.** |
| 2. Build | Sparse FFN/attention behind a flag; the dense path is the same harness with sparsity off. |
| 3. Measure | Held-out loss + downstream evals; **real** wall-clock and measured FLOPs, not the spreadsheet FLOP count (unstructured sparsity often buys zero on real hardware). |
| 4. Ablate | Sparsity off → collapses to dense (sanity). Load-balancing aux loss off → is the gain from sparsity or from that loss's regularization? |
| 5. Gate | 2 seeds where affordable; **scaling ladder** at 100M / 350M / 1B / 3B — the trend is the evidence, because a 100M win routinely dies at 3B. |
| 6. Record | Keep only if the iso-FLOP advantage is flat-or-growing up the ladder. Otherwise log the negative with the scale at which it died. |

The traps this example exists to defeat — each is a real, named way sparse-LLM work has fooled itself:

- **Weak baseline** — sparse model hand-tuned, dense model undertuned. Equal tuning budget or no claim.
- **Axis-switching** — winning iso-FLOP, then quoting the iso-param number when memory is challenged. Declare one axis; hold it.
- **Small-scale mirage** — the headline result; the ladder is the only defense.
- **Theoretical FLOPs** — measure real latency; sparse kernels are often slower than their FLOP count promises.
- **Aux-loss confound** — attribute the gain by ablating the regularizer the mechanism needed.
- **Eval leakage** — web-scale pretraining contaminates benchmarks; the held-out must be genuinely held out.

## Anti-self-deception (the core)

Everything above reduces to defeating these. If a discovery claim lands without clearing them, it is a hypothesis wearing a result's clothes:

- **Dev/test leakage** — tuned and reported on overlapping data. Hold the test slice out and never touch it during iteration.
- **Weak baseline** — the comparison only looks good because the other side wasn't tried hard. Equal budget, same harness.
- **Metric-hacking** — optimizing the proxy metric while the real objective stagnates. Carry at least one downstream/real-task metric the proxy can't game.
- **Checkpoint cherry-picking** — reporting the best checkpoint on the eval you report. Fix the selection rule (e.g. best held-out loss) before looking at the test metric.
- **Noise-as-signal** — a delta inside the run-to-run band. Establish the band (baseline vs baseline) before believing any delta.
- **Confound smuggling** — the new mechanism quietly brought a second change (more params, a new loss, different data order). Ablate until the *only* difference is the hypothesis.
- **Scale denial** — a small-scale win asserted to hold at scale without the ladder.

## Rigor mode (the tournament shape that fits discovery)

There is deliberately no `discovery-rigor` peer to `research-rigor` / `algorithm-rigor`. Those work by a judge panel deciding which artifact is best *by reading it*. Discovery's arbiter is **measurement against the world**, not reasoning — a panel asked "which architecture will win" is just more opinions, the exact thing this skill exists to escape. If a panel could settle it by argument, it was a `research-rigor` problem misfiled, not a discovery. The experiment is the tournament; the held-out metric is the judge.

Two tournament-shaped escalations ARE legitimate, because both keep measurement as the arbiter — the panel never predicts the outcome. Engage them when the win is high-stakes (a load-bearing architecture decision, an expensive run, a result that will gate downstream work):

- **Hypothesis-slate judging (front of the loop).** Fan out N hypothesis-generators; a fresh panel judges their *experiment design* — falsifiability, presence of a kill criterion, diversity, orthogonality, cost-to-test — to choose **which to run**. The winner is still decided by the experiments afterward, never by the panel. This picks what to spend compute on; it does not pick the answer.
- **Adversarial kill-panel (end of the loop).** This is the real "rigor" escalation and the fan-out version of the solo Anti-self-deception passes. Once the loop produces a candidate win, spawn a fresh red-team — each agent's sole job is to **refute** the measured claim: find the leak, the confound, the undertuned baseline, the metric-hack, the scale at which it dies. Each returns refuted / not-refuted with the specific hole. The claim is believed only if it survives a majority. Default each refuter to "refuted unless I can't find a hole" — the burden is on the win, not the red-team.

Both are external judges in the loop pointed at *killing a measured claim*, not at *predicting one*. That is the only sense in which discovery has a "rigor."

## Composition

```
research-flow       → what's already known / prior art (run FIRST)
research-rigor      → settle a contested design fork by reasoning (token-choice vs expert-choice routing)
DISCOVERY-LOOP      → find what empirically works when no oracle exists      ← here
  └─ inside it, verification-shaped sub-pieces still have oracles:
algorithm-development → the router top-k kernel / gather-scatter / sparse mask:
                        numerical parity vs a dense reference within fp tolerance
algorithm-rigor      → if the winning formulation is itself contested, tournament the bundles
disciplined-component → if the win is a perf primitive vs a known incumbent, bench-gate it
```

Discovery-loop **manufactures the oracle** the verification skills assume. The moment an experiment proves "input *X* should produce *Y*," *Y* is a now-known answer — hand it to `algorithm-development` to pin as a worked example + regression that locks the win in permanently. A discovery win that is never converted into a verification test will silently rot the next time the harness changes.

## Relationship to disciplined-component

`disciplined-component` and `discovery-loop` are the **same discipline pointed at different epistemic targets**, and this skill **inherits the former's mechanical substrate** — do not reinvent it:

- shared substrate: default-off feature flag (the firewall), a versioned discipline log with one row per tweak, **rollbacks and negatives recorded not buried**, baseline parity, CoV/variance gates, home-turf / frequency-weighted reads, delegate-the-grind-to-sonnet (principle 6), and — for any slot-0 perf artifact — the `AGENTS.md` hot-path gates.
- where they differ:

| | `disciplined-component` | `discovery-loop` |
|---|---|---|
| objective | known (throughput / latency / allocs) | uncertain — a metric standing in for an unknown best |
| baseline | a **named incumbent exists** (flume, dashmap, tokio mpsc) | may have **no incumbent** — you construct a fairly-tuned reference |
| question | "do I beat them on their home turf?" | "does this even work / is the win real?" |
| mode | optimization (perf *verification*) | capability *discovery* |

`discovery-loop` **adds** what an uncertain objective demands: a pre-registered hypothesis + kill criterion, a held-out slice, ablation-for-attribution, and a scaling ladder. It **relaxes** one disciplined-component gate: "must beat a *named* incumbent" becomes "must beat a *fairly-tuned* baseline you may have to build," because at the frontier the incumbent sometimes doesn't exist yet.

Handoff: when a discovered win is *also* a perf primitive competing with a known incumbent, hand it to `disciplined-component` for the full 13-point gate; its algorithmic core to `algorithm-development`; a contested formulation to `algorithm-rigor`.

## Discipline log (principles 7, 16)

Every experiment is a row, win or loss: hypothesis id, the one change, baseline + axis, held-out metric Δ, ablation result, variance band, ladder points, real cost, KEEP/ROLLBACK. A row must be **mechanically re-runnable from its logged config + seed alone** — "it worked when I ran it Tuesday" is not a result (principle 16). Negatives are rows too (principle 15 — no quiet punting of a dead path).

## guiding-principles alignment

- **Principle 1** (RISC reuse first): before inventing a new mechanism, audit whether a known one already does it. Discovery is for what doesn't exist yet — not for re-deriving what a published architecture or an existing primitive already provides. The hypothesis names why the existing answer is insufficient.
- **Principle 6** (sonnet subagents aggressively): the grind — building harness arms, running the scaling ladder, parsing metrics — is delegated to sonnet subagents; the main thread keeps the hypothesis, the gate, the log, and the keep/rollback judgment. Same division as `disciplined-component`.
- **Principle 13** (skill-budget): `discovery-loop` is itself a per-component skill engagement, registered in principle 13's table. It runs BEFORE the implementation slice for any component whose correct output is unknown, and its stable win then engages `/algorithm-development` to lock the result.
- **Principle 14** adapts: with no primary source for an answer, the *protocol* is the authority — the named baseline, the fixed held-out slice, the pre-registered metric and selection rule. A claim whose baseline/held-out isn't named scores zero, exactly as an uncited expected output would.
- **Principle 9** (real data) is load-bearing: the held-out distribution must be what a production caller actually hands the system, or the metric measures plumbing.
- **Principle 7** (discipline over momentum) and **15** (no punt): negatives are recorded, not buried; a dead path is logged with the scale/condition at which it died.
- **Principle 16**: each row re-provable from config + seed, not dev memory.
- **Principles 3 / 11 / 12** bind only when the discovered artifact is itself a perf primitive — they engage via the `disciplined-component` handoff, not on the discovery loop itself.

## The gate — seven questions with answers

Not a "look harder" pass. Each is a question about an artifact or a measurement
that either exists or does not; answer each once, in the report. Every one of
these is a way you can fool yourself with a real number, which is why the gate
is questions rather than an instruction to be careful.

- **Was the hypothesis pre-registered with a kill criterion, BEFORE the number
  came in?** A criterion written afterward makes the result a story.
- **Did the baseline get *equal* tuning budget in the *same* harness?** An
  undertuned baseline invalidates the delta entirely.
- **Is the test slice genuinely held out** — never tuned on, never inspected
  during iteration?
- **Does an ablation attribute the gain to the hypothesis alone,** with every
  confound removed (extra params, aux loss, data order)?
- **Is the delta outside the run-to-run noise band, and does it hold across the
  scaling ladder** rather than at one convenient size?
- **Was the real cost measured** — wall-clock, actual FLOPs, memory — rather
  than the theoretical figure?
- **Is the win converted into a verification test** (`algorithm-development`) so
  it cannot silently regress?

Any "no" means this is not a discovery. It is a hypothesis with a suggestive
number, and it gets recorded as one.

## Output shape

```markdown
## <Idea>: <what it claims, in one sentence>

### Hypothesis (pre-registered)
- claim + falsifiable kill criterion
- axis claimed (iso-FLOP / iso-param / iso-wall-clock / …) — fixed
- baseline (with equal-budget note) + held-out slice + metric + selection rule

### Result
- held-out Δ vs baseline (the number)
- ablation: gain attributed to <hypothesis> alone? (confounds removed: …)
- variance band + ladder points (the trend, not the point)
- real cost measured (wall-clock / FLOPs / memory)
- verdict: KEEP / ROLLBACK

### If KEEP — handoff
- worked example locked via algorithm-development at <test>
- contested sub-forks routed to research-rigor / kernels to algorithm-development

### If ROLLBACK — recorded negative
- what died, and the scale/condition at which it died (so it isn't re-run)
```

A number that went up is not a discovery. A number that went up against a fairly-tuned baseline, on held-out data, attributed by ablation, holding across scale, measured in real cost, and converted into a regression test — that is a discovery. Everything between the two is self-deception with a chart.
