---
name: algorithm-rigor
description: Self-play tournament for vetting a non-mechanical algorithm — incumbent Algorithm A authored via the algorithm-development discipline (worked example, pseudocode, walk-through, code-mapping, test), parallel algorithm-critique, independent Algorithm B (its own worked example AND its own formulation, blind to A), synthesis AB, judged by a fresh 3-agent Borda panel on algorithm-specific axes (proof ordering, worked-example coverage, walk-through fidelity, primary-source faithfulness, code-to-pseudocode mapping, test gating, correctness, minimality). Use when the algorithm is contested, has multiple plausible formulations, or a wrong rule is expensive, AND a single algorithm-development pass is unlikely to land it — scoring rules, graph walks, ranking fusion, RFC pseudocode ports, integer-arithmetic algorithms, novel state machines. Triggers on "vet this algorithm", "algorithm-rigor", "rigorous algorithm", "tournament the algorithm", "Mac's pipeline for algorithms", "best of N algorithms", "prove this algorithm". NOT for a single uncontested algorithm (use algorithm-development directly), evidence gathering (research-flow), plans (plan-rigor), or specs (spec-rigor).
---

# Algorithm Rigor

Sibling of `research-rigor`, `plan-rigor`, and `spec-rigor`, specialized for non-mechanical algorithms — scoring rules, graph walks, ranking fusion, RFC pseudocode ports, integer-arithmetic algorithms, novel state machines. Same tournament shape; the judged candidate is an **algorithm-development artifact bundle**, and the judge axes are algorithm-specific.

An algorithm is not a plan and not a spec. It is a *procedure that must reproduce a known answer on a concrete input*. The thing being judged is whether the bundle — worked example, pseudocode, walk-through, code-mapping, test — proves the procedure correct, not whether it "looks right."

Composes with `algorithm-development`: that skill is the methodology each author follows to produce one bundle. This skill runs the methodology N-ways in competition when one pass isn't trusted to land the right formulation. Per guiding-principles **principle 13**, a non-mechanical algorithm must engage `algorithm-development` before the implementation slice; when the algorithm is also *contested*, this skill is that engagement, tournamented.

Composes with `research-rigor` upstream: for a **frontier** algorithm, the contested question isn't "which formulation reproduces a known answer" — it's "what is the right rule at all, and is there even a ground-truth answer to reproduce?" That is `research-rigor`'s job, not this skill's. The two run in sequence:

1. **`research-rigor` settles the frontier** — what the algorithm should compute, what the state of the art is, which primary source (paper / RFC / reference impl) is ground truth, what the expected output even is on a representative input. Its resolution becomes the **primary source + task brief** this skill consumes.
2. **`algorithm-rigor` proves the chosen rule** — tournaments competing *bundles* that each reproduce that now-settled answer by hand.

Without step 1, the worked examples in step 2 have nothing authoritative to derive their expected outputs from (principle 14), and "primary source" collapses to memory. The tell that you need `research-rigor` first: authors disagree not on *how* to compute the answer but on *what the answer is*. When you reach for `algorithm-rigor` on a frontier problem and can't name the primary source, stop and run `research-rigor` first.

## When to use

- the algorithm has multiple plausible formulations (different traversal direction, different decomposition, different scoring rule) and the wrong one is expensive
- a single `algorithm-development` pass is unlikely to land the right formulation, or you don't yet trust the one you have
- the worked example itself is contested — reasonable authors would pick *different* concrete inputs to prove it, and the choice of example changes which bugs get caught
- explicit invocation

Do NOT use for:
- a single uncontested algorithm with one natural formulation — use `algorithm-development` directly
- mechanical bugs (off-by-one, missing `?`, swapped variable) — fix directly; there's no second-author angle
- evidence gathering or codebase surveys — use `research-flow`
- implementation plans — use `plan-rigor`
- formal specs / axiom systems — use `spec-rigor`
- the answer doesn't exist yet — no oracle to reproduce (new architecture, sparse-network LLM, an empirical retrieval rule) — use `discovery-loop` to find it first, then this skill to vet the formulation

If there is no genuine second formulation and no second reasonable choice of worked example, the algorithm isn't contested — abort and use `algorithm-development` directly.

## Cost warning and role dial

Same as siblings: 6–8x baseline. Algorithm authoring is heavier than general-purpose because every author must produce a full bundle (worked example derived by hand, pseudocode, walk-through, code site, test) and load the primary source. Per-agent token cost runs at or above spec-rigor. Confirm before invoking.

Set `model` and `effort` per role — that is the cost lever, not agent count:

| Role | model | effort |
|---|---|---|
| author A / author B / synthesizer | `opus` | `xhigh` |
| critique | `opus` | `high` |
| judge ×3 | `opus` | `medium` |

Authors run a step above the sibling skills: a hand walk-through that must land an exact expected output is the one place where more reasoning reliably buys correctness. Judges are scoring against eight stated axes and hold at `medium`.

**Do not add agents beyond the roles listed.** In particular, do not spawn an agent to re-check an author's walk-through — that is what the critique axis "walk-through fidelity" and the judge panel are for, and a stacked verifier spends real tokens to reach the same verdict.

## Primary source is mandatory (principle 14)

Before round 0, fix the **primary source** for expected outputs and put it in every author and judge brief: the RFC URL + section, NIST CAVP vector ID, the paper + equation number, the incumbent crate whose output is ground truth, etc. Per guiding-principles principle 14, expected outputs come from a named primary source — never from memory, never from a draft. An algorithm whose worked example sources its expected output from recollection cannot win this tournament regardless of how clean it looks.

## Loop

### Round 0 — produce incumbent Algorithm A

Spawn one `Agent` with `subagent_type=general-purpose`, fresh context. The brief MUST include:

- the full `algorithm-development` methodology (worked example → pseudocode → walk-through → code-mapping → test) as the required output shape
- the primary source for expected outputs
- the task: what the algorithm must compute, at which boundary, what's in and out of scope
- the relevant current code site / data shape (load it; don't make the author infer per principle 6)

Returned bundle is incumbent **A**. If A is missing any of the five artifacts — concrete worked example, pseudocode, hand walk-through, code-step mapping, encoded test — reject and respawn. A bundle without a hand walk-through that lands the exact expected output is not an incumbent; it's a guess.

### Round k — tournament

**Phase 1 (parallel) — critique and alternative algorithm:**

In a single message:

- `critique_agent` — fresh `general-purpose`. Input: task + bundle A + primary source. Output: written critique scoring these axes explicitly:
  - **proof ordering** — does the worked example precede and constrain the code, or was the example reverse-derived to match the code (rationalization)?
  - **worked-example coverage** — does the example exercise the hard/contested case, each branch, each transition? concrete instances, or ellipses / "similarly for the others"?
  - **walk-through fidelity** — does the hand-trace produce the EXACT expected output with no skipped or "assumed" step?
  - **primary-source faithfulness** — is the expected output sourced from the named primary source (not memory)? does it match the incumbent byte-for-byte / value-for-value (principle 14)?
  - **correctness** — does the procedure's logic actually reproduce the reference behavior, or only the single example by luck?
  - **code-to-pseudocode mapping** — is every named pseudocode step identifiable in the code? are deviations (batching, caching, early-exit) accompanied by an equivalence argument?
  - **test gating** — does the test use the EXACT worked-example inputs and assert the EXACT output? would it fail under an off-by-one, wrong-direction-walk, swapped-argument, or skipped-filter bug, or is it tautological?
  - **minimality** — is this the simplest procedure that reproduces the example, or is there gratuitous machinery?
  Critique only — do not propose fixes.
- `author_B` — fresh `general-purpose`. Input: task + primary source ONLY (blind to A). Output: an independent bundle **B** — B picks its **own** worked example AND its **own** formulation.

Blindness of B is mandatory and is the whole point. B's value is two-fold: a *different concrete worked example* (a case A's example never hit) and a *different procedure* (different decomposition or traversal direction). If B sees A, it edits A's example and A's steps instead of choosing its own — and the second worked example is where the regression you didn't know about gets caught.

**Phase 2 — synthesis:**

Spawn `synthesizer_agent` — fresh `general-purpose`. Input: task, A, B, critique, primary source. Output: merged bundle **synthesis_AB**.

Synthesis for algorithms has a specific shape — you cannot union two procedures:

- **union the worked examples** — keep BOTH A's and B's concrete examples as the test suite when they exercise different cases; each is a regression lock. Drop a duplicate only when one example strictly subsumes the other.
- **choose one formulation** — pick a single procedure. Where A and B disagree on a step (direction, filter, scoring rule), justify the choice in one line against the primary source.
- **re-walk** — re-run the chosen procedure by hand against *every* retained worked example and show it lands each exact expected output. Do not paste A's or B's walk-through unchanged; a merged procedure must be re-traced.
- **re-map and re-test** — every named step traces to the code; every retained worked example is an encoded test asserting its exact output.

**Phase 3 (parallel) — judge panel:**

Three judges in one message (fresh `general-purpose`, distinct prompts). Each judge sees:

- the task
- A, B, synthesis_AB (labelled `Algorithm-1` / `Algorithm-2` / `Algorithm-3` in randomized order per judge)
- the critique (auxiliary, not a candidate)
- the primary source (for faithfulness scoring — without it, that axis is fiction)

Each judge ranks `[first, second, third]` and scores each bundle on the eight axes above with one-line justification per axis. Ranking feeds Borda; per-axis scores feed the trail.

**Aggregate (Borda count):**

- 1st → 2, 2nd → 1, 3rd → 0
- Sum across 3 judges per candidate
- Winner = highest total. Tie-break by first-place vote count, then by sum of per-axis scores across judges.

Winner becomes new incumbent A.

### Convergence

- Same candidate wins two consecutive rounds → stop. Emit it.
- Hard cap: 4 rounds. After that, emit final winner + "no convergence" flag.

## Output

- **Algorithm**: the final incumbent bundle (worked example(s), pseudocode, walk-through, code-mapping, test)
- **Provenance**: round produced, whether stable
- **Worked-example suite**: every concrete example retained, each with its primary-source-derived expected output, mapped to its encoded test
- **Risk register**: weakest per-axis scores (axes where even the winner scored low — e.g. a coverage gap a future example should close)
- **Trail** (optional): per-round critique + Borda totals + per-axis judge scores

## Rules

- author A, B, and synthesizer all use `subagent_type=general-purpose` (no `Algorithm` subagent exists; compensate by loading the algorithm-development methodology + primary source + code site into every author brief)
- every author bundle MUST contain all five artifacts (worked example, pseudocode, walk-through, code-mapping, test) — reject and respawn any bundle missing one; these are the algorithm-development contract
- the walk-through must land the EXACT expected output by hand — a bundle whose walk hand-waves a step does not enter or remain in the tournament
- expected outputs trace to the named primary source, not memory (principle 14) — a judge cannot score faithfulness without the source in context
- B must be blind to A — non-negotiable; B picks its own worked example and its own formulation
- synthesis unions the worked examples (tests) and chooses ONE procedure; it re-walks every retained example rather than pasting prior walk-throughs
- judges are `general-purpose`; randomize bundle order per judge; neutral labels (`Algorithm-1/2/3`)
- Borda ties: first-place count, then per-axis sum; never orchestrator preference
- cap at 4 rounds
- principle 15 applies to the winner: ship the correct procedure, not a stub that passes only the one example — if the winner can't reproduce the reference beyond its own example, it hasn't converged

## Common failure modes

- **Reverse-derived worked example** — an author writes the code first, then picks an input whose expected output it already produces. Tautology. Mitigation: critique axis "proof ordering" must flag any example that only exercises the happy path the code was written for; the contested/hard case must be in the example.
- **Walk-through hand-waves** — the trace says "and similarly the filter rejects the rest" instead of showing each step. Mitigation: critique axis "walk-through fidelity" rejects ellipses; the walk must land the exact value.
- **Expected output from memory** — the bundle states a tag / score / rank as expected without a primary-source citation. Mitigation: faithfulness axis requires the RFC section / vector ID / equation; an uncited expected output scores zero on that axis (principle 14).
- **B reinvents A** — B's worked example and formulation are isomorphic to A's. Symptom: the second example catches nothing A's didn't. The algorithm isn't contested; abort and use `algorithm-development` directly.
- **Synthesis unions procedures** — synthesizer concatenates A's and B's steps into one Frankenstein procedure instead of choosing. Mitigation: synthesizer brief requires one formulation, one justification per conflicting step; it unions *examples*, never *steps*.
- **Synthesis drops a worked example** — synthesizer keeps only one example and loses the case the other locked in. Mitigation: synthesis must retain every example that exercises a distinct case; dropping one requires a strict-subsumption argument.
- **Tautological test wins on ranking** — a bundle ranks first overall but its test only asserts what the code does, not what the primary source says. Mitigation: surface the "test gating" per-axis score in the output even when it doesn't change the winner; a winner weak on that axis ships with a flagged risk.
- **Code drifted from pseudocode** — the winning bundle's code no longer maps step-for-step to its pseudocode (a deviation with no equivalence argument). Mitigation: "code-to-pseudocode mapping" axis requires each step be pointable-to in the code; unjustified deviations fail it even if the test passes.
