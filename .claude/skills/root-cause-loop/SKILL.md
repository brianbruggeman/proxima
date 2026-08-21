---
name: root-cause-loop
description: Instrument-and-loop debugging methodology for hard bugs where the cause isn't obvious from a quick read. Add observability, reproduce, observe, narrow, repeat — until the root cause is proven, not guessed. Use when the user asks to find a root cause, when a bug is intermittent or non-obvious, when initial inspection didn't yield a cause, or when the user says "instrument", "narrow it down", "find the root cause", "stop guessing".
---

# Root-Cause Loop

Find the cause by making the system tell you, not by guessing. Add observability, reproduce, observe, narrow, repeat.

This is the recursive owner of principle 19's evidence ladder: it repeats an atomic proof down the causal chain until every link is data-backed. For a single retrieval link, the atomic proof is a `/deep-dive` toggle via the retrieval-forensic agent — delegate the per-case dig, do not grind it inline. root-cause-loop is the loop that chains those proofs; deep-dive is one edge of the chain.

## When to use

- the bug isn't obvious from reading the code
- initial inspection produced hypotheses but no proof
- intermittent / state-dependent / concurrency-flavored failures
- the user explicitly says "instrument", "find root cause", "stop guessing"

Do NOT use for one-line typos or trivial null checks. Read and fix.

## Loop

### 1. Frame

Write down (in conversation):
- expected behavior
- actual behavior
- what's already known / ruled out
- the smallest reproduction available

If reproduction is missing, get one before instrumenting. Without repro, the loop is blind.

### 2. Hypothesize

List 2–4 candidate causes. For each, ask: **what observation would falsify it?** That observation is what you instrument for.

If you only have one hypothesis, you don't have a hypothesis — you have a guess. Generate alternatives.

### 3. Instrument

Add observability at decision points along the suspect code path:
- structured log lines at branches, returns, retries, locks acquired/released
- assertions for invariants you believe hold
- timing if latency-related
- state dumps before/after suspect mutations

Logs must distinguish hypotheses. If two candidates produce the same log output, the instrumentation is useless — refine before running.

Instrument generously; remove later. Lowercase, terse, causal log messages per project rules.

### 4. Reproduce

Run the failing case. Capture the log output. If the bug doesn't reproduce, the instrumentation may have changed timing — note it, retry, consider whether the observer effect is itself a clue.

### 5. Observe

Read the logs against the hypotheses:
- which hypothesis does the evidence support?
- which does it falsify?
- is there evidence neither hypothesis predicted? that's a new candidate

If multiple hypotheses survive, the instrumentation wasn't discriminating enough. Go back to step 3.

### 6. Narrow — descend the causal chain, prove every link

Take the surviving hypothesis. It is one link, not the whole why: the symptom is caused by X *because* of Y *because* of Z. Add finer-grained instrumentation at the NEXT link down and re-run — or, when a link is a discrete site, prove it `/deep-dive`-style by toggling exactly that site (on/off, same repro, delta as predicted). Repeat down the chain.

This is principle 19's ladder in motion: the loop terminates only when every link from symptom to ground is individually backed by an observation you took — not merely when you have reached a single line. A single line with an unproven chain above it is still a shallow why. Never accept a link backed by the system's own computed output (a score/rank restating itself); ground it in the substrate underneath — the data, the edges, the traversal.

### 7. Prove

State the root cause as the full chain: *"when X happens, Y, because Z"* — and every link (X→Y, Y→Z) is backed by a specific observation you took (log line, counter delta, toggle result, payload/edge you read), not by the system's own output restating itself (principle 19). If any link rests on "presumably", or on a score the mechanism emitted about itself, you have a measurement, not a result — go get the datum for that link. If you can't cite evidence for a link, you haven't found it — you've guessed again.

### 8. Fix

Only after the cause is proven. Address the root cause, not the symptom. Verify the fix by re-running with instrumentation still in place — the bad log line should disappear.

### 9. Clean up

Remove instrumentation that was diagnostic-only. Keep instrumentation that has lasting observability value (per project's logging rules).

## The gate — two questions with answers

Not a re-read of your own conclusion. Both are questions about the evidence,
and the second is the one that actually catches things:

- **Does the proposed cause explain the observed behaviour in full**, including
  the edge cases that worked *correctly*? A cause that only explains the failure
  is usually a correlate.
- **What assumption am I making that the logs do not actually verify — and
  could the same evidence support a different cause?** Name the competing
  hypothesis explicitly. If you cannot name one, you have not looked for one.

Either answer shaking the conclusion sends you back to step 6 for more
instrumentation. That is more evidence, not more thinking about the evidence
you have.

## Anti-patterns

- "let me try this fix and see" — that's guessing, not diagnosing
- removing instrumentation before the cause is proven
- accepting the first plausible explanation
- narrowing past the symptom into surrounding code without evidence the surrounding code is involved
- changing the code under test while debugging — fix one variable at a time

## Output

- **Root cause**: one sentence, the *why*
- **Evidence**: log lines and code paths that prove it
- **Falsified candidates**: hypotheses ruled out and how
- **Fix**: change + rationale
- **Risk**: what else might be affected
