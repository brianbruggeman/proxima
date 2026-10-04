# sketch 14: action speculation over a non-token entry at H11 (generic entry)

status: paper test. Read at main 4b4be6cf with `git show main:<path>`. Nothing was built or run. `interop` =
`proxima-model-interop/src`; `FSM` = `interop/serving_fsm.rs`. `FT` = the draft under
`/Users/brianbruggeman/repos/slot-0/proxima-fsm-techniques/proxima-tensor/specs/fsm-techniques/` (read-only, one
opinion, not main).

Technique as the draft states it (`FT SPEC.md:79`, row T10, AOSpec): action and observation forks on copy-on-write
environment snapshots; a fork is reused iff its action is equal AND its environment version is equal; 8 action forks,
5 observation branches. I did not read arXiv:2608.00881; everything about the technique below is the draft's one-line
description, and the sketch models only the accept decision, not forking.

## ground facts on main (what exists, read)

- The only FSM is `ServingState<Cache>`: generic over the cache, with the entry hard-coded as `u32` in every variant:
  `Prefill { positions: Vec<u32> }`, `Decode { last: u32 }`, `Verify { draft: Vec<u32> }`, `Accept { next: u32 }`,
  `Rollback { to: u32 }` (`FSM:48-74`). It is `pub(crate)` (`FSM:48`), its module is private and std-gated
  (`interop/lib.rs:69-70`), and it carries `#![allow(dead_code)]` with the note that its own test is its only caller
  (`FSM:36-38`). Outside the crate: `git grep -n "ServingState\|serving_fsm::" main -- proxima-model-interop proxima-core
  proxima-tensor` finds only `serving_fsm.rs` itself and doc mentions plus one test in `generate/serving_backend.rs`.
- `accept` compares drafted and predicted entries with `==` through `take_while(..).count()` and then picks the
  resume cache by that count (`FSM:201-231`); `Verify.snapshot` is cloned by `enter_verify` (`FSM:163-167`) and never
  read by any transition (`accept` destructures `Verify { draft, .. }`, `FSM:192,201`).
- The proposer the FSM ships is `draft_prompt_lookup(history: &[u32], ngram, max_k)`: most recent earlier occurrence of
  the last `ngram` entries, then up to `max_k` followers (`FSM:97-112`). The live loop uses a different drafter,
  llama.cpp's n-gram family, a closed enum `Drafter` over `&[u32]` (`interop/generate/drafter.rs:42-48`,
  `ngram_simple_draft` at `proxima-tokenizer/src/draft/ngram_simple.rs:127-132`). Two n-gram drafters on main.
- The evaluation boundary is token-typed too: `ServingBackend::evaluate(&mut self, positions: &[u32], cache)`
  (`interop/generate/serving_backend.rs`, `pub(super)`, `#![allow(dead_code)]`).
- The live token loop does not use the FSM; its drafter set is a closed enum and a bitset
  (`serving.rs:272-273` `SpeculativeTypeSet(u16)`, `drafter.rs:123-176`).
- No action, environment or fork code exists on main (the draft's own check, `FT SPEC.md:399`; I did not re-run it).

## shape chosen, and the contested decision

Shape: configuration and ZERO pipes, with one hook change (reachability). The consumer owns the environment; proxima
owns the transitions and the two small pure functions; the sketch's glue is an interning table in the consumer.

Contested decision: does the FSM need a generic entry (the draft's slice 1, `FT tasks/01-fsm-generic-entry.md`, motivated
by audit row A4: "an action log cannot instantiate it", `FT SPEC.md:93`)? Answered by writing the call site both
ways, on a history of actions:
- generic entry: `ServingState::<Fields, Environment>::start(history_actions, environment)`, with a comparator
  argument added to `draft_prompt_lookup` and to `accept` because propose and accept need different equalities
  (section 3, control);
- interned: `ServingState::<Environment>::start(history.map(|action| vocabulary.id(action, fields)), environment)`,
  `draft_prompt_lookup(&ids, ngram, max_k)` unchanged, `accept(&row_ids, row_environments)` unchanged.
Both reach the same behaviour; the second needs no change to the FSM. By the second gate (what can a caller do that
they could not before?) the generic entry adds nothing to capability: a typed log instead of an id table. Chosen: the
interned form. A4 as the draft states it ("cannot instantiate") is false on this reading; the true statement is
"cannot name it" (G1).

## 1. the configuration

```toml
[speculation.propose]
kind = "ngram"
size_n = 2
size_m = 2
same_on = ["key", "delta"]

[speculation.accept]
same_on = ["key", "delta", "version"]
```

This section belongs to the consumer's agent loop; proxima does not read it (reason in section 5, G2). Default
reproduces today: with no section, the FSM is instantiated with token ids and `==`, which is `FSM:97-112` and
`FSM:202-206` exactly. The token-side configuration that exists is `speculative` (the n-gram family set,
`serving.rs:381-`, `interop/speculative_settings.rs`); the draft's new `SpeculationSettings` differs from it by three
letters and would live in `proxima-core` behind `config` (`FT SPEC.md:306-309`, `proxima-core/Cargo.toml:65`, whose
`config` pulls the `registry` feature). Two sections named almost alike is defect D2.

## 2. the glue (consumer side, not a proxima pipe)

```rust
pub type Fields = BTreeMap<String, String>;

pub struct ActionVocabulary { ids: BTreeMap<Vec<Option<String>>, u32> }

impl ActionVocabulary {
    pub fn id(&mut self, entry: &Fields, fields: &[String]) -> Option<u32> {
        let key: Vec<Option<String>> = fields.iter().map(|field| entry.get(field).cloned()).collect();
        let next = u32::try_from(self.ids.len()).ok()?;
        Some(*self.ids.entry(key).or_insert(next))
    }
}
```

An action tokenizer: the H3 shape (`In` = a structured entry, `Out` = an id) for a non-text modality, with ids
assigned in first-seen order. A hash is not used: two different actions colliding would be accepted as equal, a
wrong accept. The key keeps a missing field distinct from an empty value (`Option`). Two vocabularies are kept, one per
configured `same_on` list. About 8 lines. As a `Pipe` it would need `&mut self` for first-seen assignment and `Pipe::call`
takes `&self` (`interop/generate/residency_caches.rs:3151`, the one `Pipe` impl in this crate), so it is a plain struct.

## 3. worked example (hand-derived, not executed)

Action `(key, delta, version)`, where `version` is the environment version the action is decided against. History of
five committed actions: `(0,1,0) (1,1,1) (2,1,2) (0,1,3) (1,1,4)`. The environment is now at version 5.

Propose, `same_on = [key, delta]`: ids `[0, 1, 2, 0, 1]` (`(0,1)` is id 0, `(1,1)` is 1, `(2,1)` is 2).
`draft_prompt_lookup(&[0,1,2,0,1], 2, 2)`: needle `[0,1]`; `search_end = 5 - 2 = 3`; starts 2, 1, 0 tried in that
order: `[2,0]` no, `[1,2]` no, `[0,1]` yes at start 0; `match_end = 2`; `available = 3`; `take = min(3, 2) = 2`; the
draft is `history[2..4] = [2, 0]`, i.e. actions `(2,1)` and `(0,1)` (`FSM:97-112`).

The consumer re-stamps the draft with the versions of the forks that will run it: `(2,1,5)` and `(0,1,6)`.
Verification rows (the true policy evaluated at each fork state, supplied by the consumer): row 0 `(2,1,5)`, row 1
`(0,1,7)`. Version 7 stands for an environment advanced by something outside the fork chain (how that arises is the
technique's business, unread).

Accept, `same_on = [key, delta, version]`, vocabulary assigned in encounter order: draft `[(2,1,5), (0,1,6)]` is ids
`[0, 1]`; rows `[(2,1,5), (0,1,7)]` are ids `[0, 2]`. `accept(&[0, 2], row_environments)`: `take_while` matches row 0,
fails row 1, `accepted = 1`; since `1 < draft.len() = 2` the result is `Rollback { snapshot: row_environments[1],
to: 2 }` (`FSM:208-229`: `placement_index = accepted`, `to = row_tokens[accepted]`). One draft action kept; resume from
the true policy's own action at version 7.

With `same_on = [key, delta]` for accept: draft ids `[A, B]`, row ids `[A, B]`, `accepted = 2 = draft.len()`, result
`Accept { n: 2, next: draft[1], cache: row_environments[1] }`: the fork executed against version 6 is reused though
the environment is at 7. That is the stale reuse the version field exists to refuse.

Control that must fail: propose with `same_on = [key, delta, version]`. History ids are all distinct (`[0,1,2,3,4]`);
needle `[3, 4]` has no earlier occurrence; `draft_prompt_lookup` returns an empty draft and speculation never
starts. So propose and accept need different equalities over one entry type, which is why a single `PartialEq` on a
generic `Entry` cannot replace the two vocabularies, and why the draft's generic entry would need a comparator
argument on both functions.

## 4. cache-key binding (cross-cutting rule, checked against `generate/prompt_cache_key.rs`)

No KV row is cached by this sketch: entries are actions and the "cache" is the consumer's environment. Nothing is added
to `ServingConfig`, so `CacheKey::of`'s exhaustive destructure is unaffected (`speculative: _` with the reason that
draft width reaches the key through `ring_slack_rows`, `prompt_cache_key.rs:175-177`). If the true policy is a proxima
model, its prompt cache stays keyed as before. The binding rule appears in miniature in the consumer's domain: the
fork's environment version is the key component that makes a reused fork correct, which is exactly the version field
in the accept equality above. The rule's wording in SPEC.md (an H9 or H13 key binds the producing configuration) is
about KV rows; here the analogue is the consumer's, outside what proxima can enforce.

## 5. HOOK GAPS (every place the sketch edits core)

G1. H11 reachability. Missing input: a consumer can name the FSM. `mod serving_fsm` is private and std-gated
(`interop/lib.rs:69-70`), the type is `pub(crate)` (`FSM:48`), and the module is dead code by attribute
(`FSM:36-38`). Smallest change: make `ServingState`, `ServingFsmError` and `draft_prompt_lookup` public. The file
imports only `alloc::vec::Vec` and `thiserror` (`FSM:40-42`), so it needs `alloc`, not `std`; whether it should also move
to a lower crate (`proxima-core` has `alloc` and `config` features, `Cargo.toml:65`) is a placement question, not a
capability one. Note what a consumer would import: about 300 lines of which the sketch uses `enter_verify`, `accept`,
`resume` and `rollback` (about 60 lines); the decisions are small and the value is a shared transition table.

G2. H11 configuration. Missing input: nothing in proxima reads `same_on`. The H11 slot's configured list is the
drafter bitset over a closed enum (`serving.rs:272-273`, `drafter.rs:42-48`), so a non-token proposer is not a member
and cannot become one by configuration. With the interned design the proposer is called directly by the consumer, so
the list is not consulted. Smallest change if the SPEC wants a proxima-owned list: none proposed; a variant per
entry type would be the "one type per technique" pattern the draft's own R20 forbids (`FT SPEC.md:300`).

G3. `Verify.snapshot` is a clone nothing reads (`FSM:163-167`; accept ignores it, `FSM:192,201`). For a copy-on-write
environment the clone is cheap; for any other cache it is a full copy per verify. Smallest change: delete the field,
or read it in the zero-accept path. Not required by this sketch.

G4. `accept` verifies `draft.len()` rows and the live loop evaluates `1 + draft.len()` and emits a bonus token
(sketch 8, GAP-3, a claim carried from that sketch and not re-read here). An action consumer using the FSM as written gets one row fewer
per verify than the live loop's accounting; whether that matters for actions is the consumer's choice, but the two
drivers of one transition table disagree.

Limits, not changes:
- L1. Forking, version bumps, and the true policy's evaluation are the consumer's; proxima never sees them, so this sketch
  tests only the accept decision and the id-space argument, not the technique.
- L2. The interning table is per run; ids are not stable across runs, so a persisted macro library (T12) would need a
  durable vocabulary, which this sketch does not write.

## 6. is any of it a pipe? each candidate, against the two gates

- Action tokenizer as `Pipe<In = Fields, Out = u32>`: needs `&mut self`; with interior mutability it would be a
  pipe, and the call sites `vocabulary.id(&entry, &fields)` and `vocabulary.call(entry).await` are the same lookup
  plus a future around a map insert. Relocation; not minted.
- Propose as a pipe (`In = history`, `Out = draft`): the convention at `ngram_simple.rs:115-125` wraps a drafter
  function in a pipe; the call site both ways is the same function call. Not minted.
- Accept policy as a pipe (`In = (draft, rows)`, `Out = count`): it is `take_while(..).count()` at `FSM:202-206`.
  Not minted.
- Generic `Entry` type parameter: written both ways in the contested decision; capability unchanged.
Therefore zero pipes and zero new types in proxima.

## 7. designs abandoned

- Generic `Entry` on `ServingState`, with a comparator argument on propose and accept (the draft's slice 1): abandoned
  for interning; same capability, no change to the FSM.
- Hashing the selected fields into a `u32`: abandoned, a collision is a wrong accept.
- A `SpeculationSettings` section in `proxima-core` with `accept = equality{fields, fork_budget}` (`FT SPEC.md:172`):
  abandoned for the consumer's own section; nothing in proxima reads it (G2).
- Making the live loop action-aware: abandoned; it is token-typed throughout (`decode.rs` drafting uses `DrafterSet` over
  `&[u32]`, `drafter.rs:198-`).

## defects found in passing

- D1. Two n-gram drafters: `draft_prompt_lookup` (`FSM:97-112`, scans most recent start first, no `sampled` input) and
  `ngram_simple_draft` (`ngram_simple.rs:127-170`, llama.cpp parity port with `sampled`, scan from `cur_len - size_n - 1`
  down to 1). Different tie-breaks and thresholds; the first is dead code.
- D2. `speculative` (token n-gram set, exists) and the draft's `speculation` (generic accept grammar) are two sections one
  edit apart in a name.
- D3. The FSM's own module doc says wiring it onto `qwen35moe` is blocked on an `#[ignore]`d parity test
  (`FSM:25-34`); the draft's slice 1 moves this file and re-exports it, and does not address that block.
- D4. `ServingBackend` (`generate/serving_backend.rs`) is a one-method trait with `&mut self`, token-typed, dead; the
  sketch needs neither it nor a replacement.

## verdict

fits after 1 named hook change (G1, make the FSM nameable from outside the crate); G3 and G4 are recorded, not
required. Configuration plus zero pipes plus a consumer-side id table, and the draft's generic-entry slice is not
needed for this technique. Not claimed: that any accept decision improves an agent loop, that interning matches the
draft's typed entry in every case (a typed entry could carry values that a vocabulary id hides from a downstream
reader), anything about forking, or any speed.
