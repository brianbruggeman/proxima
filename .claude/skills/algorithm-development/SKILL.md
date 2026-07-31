---
name: algorithm-development
description: Paper-first algorithm development. Derive a worked example by hand BEFORE writing code, prove the algorithm reproduces the paper on that example, implement faithfully, then encode the example as a test. Use when adding or modifying an algorithm (scoring rule, graph walk, ranking path, expansion logic, retrieval fusion, lemma rule, anything where "looks right" isn't enough). The worked example is the spec AND the test. Triggers when the user says "show me on paper", "walk through it", "derive the algorithm", "prove it works", "before writing code", "show me the algorithm". OUT OF SCOPE: typo fixes, boilerplate refactors, dependency bumps, plumbing-only changes that don't introduce new logic, anything where the change is mechanical and doesn't shift behavior.
---

# algorithm-development

Code-first asks: "does this look right?" Paper-first asks: "for THIS specific input, what's the exact expected output, and does my code produce it?"

The first is rationalization. The second is proof.

## When to use

- A new scoring path, ranker, fusion rule, or expansion algorithm
- Modifying graph traversal (depth, direction, filter predicate)
- A bug fix in an algorithm where the bug isn't mechanical (off-by-one is mechanical; "wrong sense expanded" is algorithmic)
- The user has said: "show me on paper", "walk through it", "prove it works", "derive the algorithm", "before writing code"
- Any change where someone could plausibly ask "does it work?" and "yes I think so" isn't a valid answer

If the algorithm is ALSO contested — multiple plausible formulations, or a wrong rule is expensive enough to justify the spend — escalate to `algorithm-rigor`. It runs this same discipline N-ways in a judged tournament: this skill produces one worked-example bundle; `algorithm-rigor` tournaments competing bundles and Borda-judges them, and the winner's worked example becomes the locked test.

## When NOT to use

- Typo and one-character fixes
- Renaming variables, moving files, formatting passes
- Adding a missing `?` or unwrapping a Result — mechanical bugs
- Bumping dependency versions
- Adding logging or instrumentation only
- Refactors that preserve behavior by construction (extract function, inline variable)
- Per-case data fixes where the algorithm itself isn't changing (add a JSONL row, fix a typo in vocab) — use `reverse-engineer` instead, the data IS the proof there
- The correct output is unknown because you're inventing something with no oracle (a new architecture, a sparse-network LLM, a retrieval rule nobody has the answer to yet) — use `discovery-loop`. This skill is verification; it needs an answer to reproduce. `discovery-loop` manufactures that answer empirically, and you return here to lock the discovered win as a worked example + test.

## The loop

### 1. Worked example (paper)

Pick ONE concrete input. State it explicitly. Hand-derive the expected output. No machine, no execution — the paper trace is independent evidence.

State three things:
- **Inputs** at the boundary of the algorithm (query tokens, source slot, edge tuples that exist at corpus state, etc.)
- **State** that the algorithm reads from (graph edges, vocab rows, scoring constants — quoted concretely, not abstractly)
- **Expected output** — derived by hand from inputs + state. Exact values.

For substrate work, the example usually shapes like:

```
Inputs:
  query word: "own"
  query word's other-context tokens: ["musical", "instrument", "currently"]

State at corpus time T:
  polysemy.jsonl row for "own": {senses: [
    {pos: "verb",      synonyms: ["possess", "have"]},
    {pos: "adjective", synonyms: ["personal"]}
  ]}
  → load_polysemy at slotgen.rs:962 creates groups:
    group_V = (own/verb/"verb sense"), members [own, possess, have]
    group_A = (own/adjective/"adjective sense"), members [own, personal]
  → seed.rs writes (own, rel_V[type=wk.polysemy], group_V)
                    (own, rel_A[type=wk.polysemy], group_A)
                    and reverse edges for possess, have, personal

Expected output of expansion(own, …):
  expanded_weighted ⊇ {"possess" ≥ 0.5, "have" ≥ 0.5, "personal" ≥ 0.5}
  Reason: with wk.polysemy in hierarchical_types, the 0.5 floor at expansion.rs:596 fires
          for each polysemy-sense sibling.
```

If you can't derive the expected output by hand from inputs + state, you don't understand the algorithm well enough to implement it. Stop and re-read until you do. (If you can't derive it because there is no answer to derive *from* yet — no paper, reference impl, or benchmark key exists — you are in discovery, not verification: switch to `discovery-loop`, which replaces the missing oracle with a held-out objective, and return here once it has produced a known answer to lock.)

### 2. Algorithm (pseudocode)

Write the algorithm in plain prose or pseudocode. Don't optimize. Don't worry about zero-copy or batching. Describe the LOGIC, not the implementation.

For each step, name:
- inputs at that step
- the specific operation (named lookup, named filter, named reduction)
- outputs

The pseudocode should be readable by someone unfamiliar with the codebase. If you find yourself reaching for an existing function name as a shortcut, expand it inline at least once — the reader needs to see what it does, not what it's called.

```
expand(query_word, other_query_words, db, wk):
  groups <- find_relations(query_word, [wk.synonym, wk.hyponym, wk.hypernym, wk.meronym, wk.metonym, wk.polysemy, wk.collocation, wk.antonym, wk.morpheme])
  expanded <- {}
  for (rel_slot, group_idx) in groups:
    rel_type <- resolve_type(rel_slot)
    siblings <- find_relations_rev(group_idx, rel_types)
    for (sib_rel, sib_word) in siblings:
      if sib_word == query_word: skip
      if pos_filter rejects (rel_slot's pos disjoint from query_word's pos): skip
      sib_groups <- find_relations(sib_word, all_rel_types).map(.group_idx)
      context_score <- sum over other_query_words of jaccard(sib_groups, other_word's groups)
      if rel_type in [wk.hypernym, wk.hyponym, wk.meronym, wk.metonym, wk.polysemy]:    ← change here
        context_score <- max(context_score, 0.5)
      if context_score > 0:
        expanded[sib_word] <- max(expanded[sib_word], context_score)
  return expanded
```

### 3. Walk-through (paper × algorithm)

Run the pseudocode against the worked example by hand. Show your work — every step's input and output, named in terms of the example.

```
Walk for expand("own", ["musical", "instrument", "currently"], db, wk):

Step 1: find_relations("own", rel_types)
  → [(rel_V, group_V), (rel_A, group_A), (rel_syn, group_syn_own), ...]
  rel_V has type wk.polysemy
  rel_A has type wk.polysemy

Step 2: for (rel_V, group_V):
  rel_type = wk.polysemy
  siblings of group_V = [(rel_x, "possess"), (rel_y, "have")]
  pos_filter: rel_V.pos = own's pos = {JJ, VB}; query "own" pos_set = {JJ, VB}. NOT disjoint. pass.
  For sibling "possess":
    sib_groups = possess's all groups (some set S_possess)
    context_score = jaccard(S_possess, "musical"'s groups) + jaccard(S_possess, "instrument"'s groups) + ...
       = small number ε
    rel_type wk.polysemy IS in hierarchical_types ← change applies here
    context_score ← max(ε, 0.5) = 0.5
    expanded["possess"] = 0.5
  For sibling "have":
    same logic → expanded["have"] = 0.5

Step 3: for (rel_A, group_A):
  siblings = [("personal")]
  → expanded["personal"] = 0.5

Result: expanded ⊇ {"possess": 0.5, "have": 0.5, "personal": 0.5} ✓
       matches paper's expected output.
```

If the walk produces the expected output: GOOD. Proceed.

If it doesn't match the paper: the ALGORITHM is wrong (don't blame the paper). Fix the pseudocode and re-walk. If you blame the paper, re-derive it from scratch with fresh state.

Forbidden: "the algorithm probably does X" without walking it. The walk IS the proof.

### 4. Code

Implement the pseudocode in the target language. Stay structurally faithful — every named step in the pseudocode should appear as identifiable lines in the code (a function, a block, a named variable).

If the code deviates from the pseudocode for performance (zero-copy, SIMD, batch), name the deviation and demonstrate the result is equivalent. Common honest deviations:

- batching: pseudocode loops; code calls `add_relation_batch_grouped`. Equivalent if the batch contents match.
- caching: pseudocode recomputes; code memoizes. Equivalent if cache invalidation is correct.
- early-exit: pseudocode iterates fully; code breaks when threshold reached. Equivalent if the threshold is sound.

If the deviation can't be justified by an equivalence argument, the code is broken even if the tests pass.

### 5. Test

Encode the worked example as a unit test. Inputs = exactly the example's inputs. Expected output = exactly the paper's derived output.

The test:
- uses the SAME inputs as the worked example, not a "similar" case
- asserts the EXACT expected output, not an approximation or shape
- has a docstring or comment that points back to the paper proof (commit message, design doc, this skill's writeup)

```rust
#[test]
fn polysemy_siblings_get_hierarchical_bypass_for_own_verb_sense() {
    // Worked example from <commit>: "own" → polysemy verb sense → ["possess", "have"]
    // each should land in expanded_weighted at score ≥ 0.5 via hierarchical_types
    // expansion at scoring/expansion.rs:498.
    let (db, wk) = fixture_with_polysemy_row("own",
        [("verb", ["possess", "have"]), ("adjective", ["personal"])]);
    let expanded = expand("own", &["musical", "instrument", "currently"], &db, &wk);
    assert!(expanded.get("possess").copied().unwrap_or(0.0) >= 0.5);
    assert!(expanded.get("have").copied().unwrap_or(0.0) >= 0.5);
    assert!(expanded.get("personal").copied().unwrap_or(0.0) >= 0.5);
}
```

If you can't write a unit test for the worked example (inputs require a full corpus, algorithm is too entangled with other systems): the algorithm is too coupled. Refactor for isolation. The test gates the rest of the discipline.

## Anti-patterns

- **Code first, then "deriving" the paper to match.** Tautology. The paper must precede the code or it's rationalization.
- **Vague pseudocode.** "Find the relevant edges, score them, return top N" is a description, not an algorithm. Algorithm steps name specific lookups, specific data structures, specific orderings.
- **Skipping the walk-through.** "The algorithm clearly works" without a hand-trace is a guess. The walk catches off-by-one, wrong-direction, missing-case errors that eyeballed pseudocode hides.
- **"Similar" tests.** The test must encode the EXACT worked example, not a "case like it." Similar cases pass for the wrong reasons and miss the regression the worked example was supposed to lock in.
- **Tautological tests.** Tests that assert what the code currently does, not what the paper says it should do. If the test was written after the code with no reference to the paper, it's tautological.
- **Algorithm steps "implicit" in the code.** Every named step in the pseudocode must appear identifiably in the code. If a reviewer can't point to "this is step 3," the code drifted.
- **"See paper" with no paper.** The paper proof must live somewhere durable: commit message, design doc, test docstring. "See the napkin I scribbled on" doesn't survive code review or future-you.

## The gate — six questions with answers

This is not a "read it again and look harder" pass. Each of these is a question
about an artifact that either exists or does not, and you answer it once, in the
report. "I critiqued it twice" is unfalsifiable and self-certifies; these are
checkable.

- **Was the paper proof written BEFORE the code?** If the order was reversed,
  the proof is rationalization. The artifact dates tell on you.
- **Does the walk against the worked example land the EXACT expected output,
  with no hand-waving?** A step you skipped or "assumed" means you do not yet
  understand the algorithm. Re-walk it — that is doing the work, not
  re-verifying it.
- **Does the code map step-by-step to the algorithm?** Every named step appears
  in the code as a function, block, or variable. Steps that are "implicit" mean
  the code drifted.
- **Does the test use the EXACT worked-example inputs and assert the EXACT
  expected output?** Approximations, "similar" inputs, or shape-only assertions
  invalidate it.
- **Would the test fail under an off-by-one, wrong-direction walk, swapped
  arguments, or a skipped filter?** If not, it is tautological and gates
  nothing.
- **Is the paper linked from the test?** The test docstring or commit body
  references the proof, so the next reader re-derives the expected output
  without asking you. Without the link the test is opaque even when correct.

## Output shape

```markdown
## <Algorithm name>: <what it does in one sentence>

### Worked example (paper)

Inputs:
- <named input 1: concrete value>
- <named input 2: concrete value>

State at time T:
- <data row / edge tuple / config value, concretely quoted>
- <...>

Expected output (derived by hand):
- <exact value with derivation>
- <...>

### Algorithm (pseudocode)

```
function name(inputs):
  step 1: <named operation>
  step 2: <named operation>
  ...
  return <output>
```

### Walk-through (paper × algorithm)

Step 1: with input <X>, lookup produces <Y>, ...
Step 2: with <Y>, compute <Z>, ...
Step 3: <Z> matches expected output ✓

### Code site

`path/to/file.ext:NN-MM` — algorithm implementation.
- step 1 → lines NN-NN  (function/block name)
- step 2 → lines NN-NN  (function/block name)
- step 3 → lines NN-NN  (function/block name)

Deviations from pseudocode (if any) and their equivalence arguments:
- <e.g. "batch-write at line MM aggregates the per-step writes from steps 4-6; equivalent because the batch's contents are exactly the union of those steps' outputs">

### Test

`path/to/test.ext::test_name` — encodes the worked example.
- inputs: <same as paper>
- assertion: <expected output matches paper's derived value>
- docstring references: <commit hash / design doc / skill output>
```

A paper proof without a walk-through is a guess. A walk-through without a code-step-to-pseudocode mapping is a drift. A test that doesn't use the worked example's exact inputs is tautological. All four artifacts (paper, walk, code-mapping, test) MUST line up or the algorithm isn't proven, it's hoped.
