# sketch 8: rectified sparse attention (arXiv:2506.04108) at H11 (enter, commit), H12 (read), H13 (seal summaries)

status: paper test. Read at main 4b4be6cf (`git show main:<path>`). Nothing was built or run, except one
`awk` arithmetic check noted in section 2. `interop` = `proxima-model-interop/src`. `FT` = the draft in
`/Users/brianbruggeman/repos/slot-0/proxima-fsm-techniques/proxima-tensor/specs/fsm-techniques/` (read-only;
treated as one opinion, not as main).

Technique as the draft states it (`FT SPEC.md:70`, row T1): block-sparse decode; each block is scored
against its key min/max, the top-n blocks plus the local block are read; every f tokens a dense re-encode
overwrites the last f KV rows; published defaults b=16, p=0.9, n_min=16, n_local=1, f=32. I did not re-read
the paper. Unverified by me: whether the paper selects per head or per KV group. The draft pools over every
query head and both rotated planes (`FT tasks/06-read-sets.md` "Decided in this file"); I carry that choice and
label it the draft's.

Ground fact first: none of this exists on main. `git grep -n -i -E "rectif|block_sparse|keep_ratio|sealed|key_minmax|read_set|attention\.read|ReadSpec" main -- proxima-model-interop/src proxima-model-interop/tests proxima-tensor/src omega/src proxima-core/src`
(excluding the novel `examples/data/war_and_peace.txt`, whose hits are prose) returns five lines, all unrelated:
`interop/error.rs:405` (a Mixtral tensor name), `proxima-tensor/src/cpu/arena.rs:526` and
`cpu/tests.rs:5063` (the word "sealed" in prose), `cpu/tests.rs:12603,12669` (two test names about a static
block-sparse matmul). So "configuration" below is four new keys with no consumer; every consumer is a gap.

## shape chosen, and the one contested decision

Shape: four config keys, ZERO pipes, three small pure functions in core, and three compiled primitives (the `Block` read lowering, a selection kernel,
per-block summary storage). Each
candidate for "the one ~40-line pipe" fails a gate (section 6), so the honest answer to "config plus at most one
pipe" is config plus zero pipes plus the hook changes in section 7.

Contested decision: is the rectify pass its own FSM state or hook, or a `Verify` with unconditional accept?
Chosen: `Verify` with accept-all and commit-by-rewind, which needs no new state and no new cache primitive; and
the draft's three core functions for it (`FT tasks/07-rectify.md` cards 7.1-7.3) collapse into one,
`rectify_range`, because the three compute one fact (section 3).

## 1. mechanism to hook map

| mechanism | hook | on main |
|---|---|---|
| enter every f steps | H11 enter | inline condition, not a rule: `interop/generate/decode.rs:3642-3678` |
| proposal = last f committed entries | H11 propose | drafts come only from closed `Drafter` enum (`interop/generate/drafter.rs:42-48`, 5 n-gram variants) |
| accept all f rows | H11 accept | `ServingState::accept` is token-chain equality only (`interop/serving_fsm.rs:186-237`); FSM is `#![allow(dead_code)]` (:38) |
| overwrite last f KV rows | H11 commit | live commit is append then `LayerCache::truncate` to `cached_len_before_step + emitted.len()` (`decode.rs:5803-5821`, `residency_caches.rs:155-162`) |
| per-block key min/max | H13 seal | no seal concept; `LayerCache` is three flat `Vec<f32>` (`residency_caches.rs:112-119`) |
| top-n blocks + local + tail | H12 read | none; cache-side visibility mask is padding (+ window) only (`proxima-tensor/src/spec/lfm2_single_range_cached.rs:328-406`) |

## 2. the config

```toml
[kv]
block_tokens = 16

[kv.seal]
horizon_rows = 32
summaries = ["key_minmax"]

[attention.read]
kind = "block"
keep_ratio_milli = 100
min_blocks = 16
local_blocks = 1

[decode.rectify]
every = 32
```

Values are the published defaults (b=16 -> `kv.block_tokens`; p=0.9 sparsity -> keep 10% -> `keep_ratio_milli =
100`; n_min=16; n_local=1; f=32). Key names follow the draft (`FT SPEC.md` R4, R8a, R10a). `keep_ratio` is an
integer milli, the draft's choice. Its stated reason (card 6.1 of `FT tasks/06`) has two halves: the algebra has no
ceil, and "a float ratio drifts at `ceil(0.1 * 30)`". I checked only the second: `awk 'BEGIN{printf "%.20f", 0.1*30}'`
prints `3.00000000000000000000` (f64), and `0.1*n` for n = 10, 20, .. 200 is an exact integer, so that example does
not drift. Integer milli is still right for a
reason that does not depend on that example: the value must round-trip through env text and compare equal inside
a cache key.

Consumers on main for these four keys: none (grep above). `kv.block_tokens` also collides with an existing key:
`prompt_cache.block_tokens` defaults to 64 (`interop/prompt_cache_settings.rs:66-68`) and sizes the prefix trie
(`interop/generate/prefix_trie.rs:6-19`). The draft's "one block size" rule (`FT SPEC.md:153-157`, refusal row
`BlockTokensConflict` :372) therefore forces the prompt cache to 16 as well, or leaves ReSA at b=64, not the
published 16. A coupling the user did not ask for; carried as a finding, not resolved here.

## 3. the three pure functions (tier 1, no alloc), and what the draft's seven collapse into

```rust
pub const fn rectify_range(step: u64, committed: usize, every: u32) -> Option<core::ops::Range<usize>> {
    let rows = every as usize;
    if every == 0 || step == 0 || step % (every as u64) != 0 || rows > committed {
        return None;
    }
    Some(committed - rows..committed)
}

pub fn fold_key_bounds(rows: &[f32], width: usize, low: &mut [f32], high: &mut [f32]) -> bool {
    if width == 0 || rows.len() % width != 0 || low.len() != width || high.len() != width {
        return false;
    }
    low.fill(f32::INFINITY);
    high.fill(f32::NEG_INFINITY);
    for row in rows.chunks_exact(width) {
        for ((value, floor), ceiling) in row.iter().zip(low.iter_mut()).zip(high.iter_mut()) {
            *floor = floor.min(*value);
            *ceiling = ceiling.max(*value);
        }
    }
    true
}

pub fn block_score(query: &[f32], low: &[f32], high: &[f32]) -> Option<f32> {
    (query.len() == low.len() && low.len() == high.len())
        .then(|| query.iter().zip(low).zip(high).map(|((q, l), h)| (q * h).max(q * l)).sum())
}
```

`rectify_range` replaces the draft's `choose_proposer` rectify arm, `rectify_proposal`, and
`rectify_overwrite_range` (`FT tasks/07` cards 7.1-7.3): the proposal is `&committed[range.clone()]`, the
overwrite range is `range`, and the enter rule is `range.is_some()`. Card 7.3's check `start + count ==
committed_rows` is a tautology for a range produced this way. Three functions existed to make one fact
usable; the defect was the fact being split. The draft also makes the enter order configurable
(`[Rectify, Drafter]` vs `[Drafter, Rectify]`, card 7.2); I fix it as rectify-first, not configurable: a
drafter that wins the coincident step silently skips a rectification, which breaks the one invariant ReSA
relies on (no row older than f steps is left unrectified).

`block_score` is the draft's card 6.3 sum, `sum_d max(q_d * kmax_d, q_d * kmin_d)`. These are not pipes
(section 6). Top-n selection among the sealed non-local blocks is the draft's `top_fraction` decision, not
repeated here.

## 4. worked example (doubles as the test; hand-derived, not executed)

Read decision, head dimension 2, b = 16, 4 sealed blocks (rows 0..63), 5 unsealed tail rows (69 rows),
`keep_ratio_milli = 500`, `min_blocks = 1`, `local_blocks = 1`. Pooled query q = (2, -1); per-dimension
key bounds (min, max): block 0 ((0,0),(1,1)), block 1 ((-2,-1),(0,3)), block 2 ((1,-3),(2,-1)), block 3
((-1,-1),(1,1)).
- Scores `block_score`: block 0 = max(2*1, 2*0) + max(-1*1, -1*0) = 2 + 0 = 2; block 1 = max(0, -4) + max(-3, 1)
  = 0 + 1 = 1; block 2 = max(4, 2) + max(1, 3) = 4 + 3 = 7; block 3 = max(2, -2) + max(-1, 1) = 2 + 1 = 3.
- Local block = the most recent sealed block = block 3. Non-local = blocks 0..2, so nonlocal = 3 and
  n = min(3, max(1, ceil(0.5 * 3) = 2)) = 2. Top 2 of scores [2, 1, 7] = blocks 2 and 0.
- Attended = blocks {0, 2, 3} + all 5 tail rows = 3 * 16 + 5 = 53 of 69 rows; block 1 (rows 16..31) is skipped.
  (The draft's example reaches 53 as well but with different scores, `FT tasks/06` "shared worked values".)

Enter and range: `every = 32`, prompt P = 4096, `committed = P + step`. Steps 1..=100 give `Some` exactly at
steps 32, 64, 96: `rectify_range(32, 4128, 32) = Some(4096..4128)`, `rectify_range(64, 4160, 32) =
Some(4128..4160)`, `rectify_range(96, 4192, 32) = Some(4160..4192)`; step 31 gives `None`; `every = 0` gives
`None` for all steps; `rows > committed` gives `None`.

Seal fold: rows `[[1.0, -2.0], [0.5, 3.0], [-1.0, 0.0]]` with width 2 give low `[-1.0, -2.0]`, high
`[1.0, 3.0]`. A real-data version of this test folds one sealed block of K rows read from the
`LayerCache` of a real checkpoint (principle 9); the corpus is the repo's
`proxima-model-interop/examples/data/war_and_peace.txt`.

## 5. HOOK GAPS (every place the sketch edits core)

GAP-1. H11 enter. Missing input: a rule. Entering Verify is the inline condition
`speculative_enabled && next_ids.len() == 1 && cached_len > 0 && self.speculative_verify_program.is_some()`
(`decode.rs:3642-3646`), and `speculative_enabled` itself comes from `drafter_set` and
`rings_cover_speculation` (`decode.rs:3535-3536`). Smallest generic change: call `rectify_range` at the top of the
step closure and treat a `Some` as a Verify entry that takes precedence over drafting.

GAP-2. H11 Verify availability. `speculative_verify_program` is `Some` only for gemma4: the trait default is
`Ok(None)` (`interop/architecture.rs:327-335`), the doc says "every architecture this crate ships except gemma4" (`architecture.rs:317-322`), the one override is
`interop/gemma4/bind.rs:843`, and the loop checks it (`decode.rs:3645`). On qwen3, qwen2 and openchat (three of the four parity checkpoints) there is no
verify program, so no rectify pass exists. Owned by architecture-as-data R3; listed because ReSA cannot run
on those models without it.

GAP-3. H11 accept. Missing input: unconditional accept, and a way to resume. `accept` returns `Accept` only if
the drafted tokens equal the verifier's row tokens (`serving_fsm.rs:201-232`), which a teacher-forced re-encode
of committed tokens does not guarantee. And `enter_verify` discards `last` (`serving_fsm.rs:162-166`; `Verify`
has no `last` field, :59-63), but a rectify pass must return to `Decode { last }` with the original, not-yet-fed
`last`, whereas `accept` resumes from `draft[n-1]` (:221-225). Also: the FSM verifies `draft.len()` rows, row 0 following `last` (`serving_fsm.rs:174-181,193-195`), while the live
loop evaluates `1 + draft.len()` rows and emits a correction or bonus token (`decode.rs:3680-3684,5773-5780`); the
FSM has diverged from the loop it is meant to drive. Smallest change: `Verify` carries `last`; one transition `accept_all` that resumes from it.

GAP-4. H11 commit. Missing input: rewind before the pass, no token out. The live accept path keeps
`cached_len_before_step + emitted.len()` rows and returns one token per closure call (`decode.rs:5808-5829`).
Rectify needs `cached_len := range.start` before the evaluation, `keep_positions := range.end`, and no emitted
token. No new primitive is needed for the rows: `LayerCache::truncate(range.start, ..)` followed by the verify's
own append already produces an in-place overwrite (`residency_caches.rs:155-162`; speculative verify rewinds the
same way). This is where I part from the draft: its `overwrite_rows` plus `RewindIntoSealed`
(`FT tasks/07` card 7.4 step 2) duplicates truncate-then-append. Two real limits: `truncate` is a no-op on ring
layers (`residency_caches.rs:156-158`; the rewind there is `cached_len` going back, doc :146-148), and the
device-resident path (gemma4 on Metal, `decode.rs:3550-3563`) sizes its step buffer as `draft_limit + 1` rows
(`decode.rs:3565`), which must become `max(draft_limit + 1, every)`.

GAP-5. H11 rewind capacity. Missing input: `every` in the slack arithmetic. Ring slack is
`max(speculative_draft_limit, prompt_cache.rewind_slack_rows())` (`interop/generate/kv_ring.rs:243-250`),
`rewind_slack_rows` is `0` when the prompt cache is off (`serving.rs:604-610`), exactness is checked by
`rings_cover_speculation(.., draft_limit)` (`kv_ring.rs:274-284`), and the value is a `CacheKey` field
(`prompt_cache_key.rs:67-70,197`). Smallest change: `every` becomes a third term in all three.

GAP-6. H12 read, descriptor. Missing input: a read field. `LayerAttentionConfig` has no read parameter
(`proxima-tensor/src/spec/attention_forward.rs:411-445`) and `ServingConfig` has none (grep above). Smallest
change: `read: ReadSpec` with default `Dense` (a lowering byte-identical to today, per architecture-as-data R7),
variants `Dense | Block { keep_ratio_milli, min_blocks, local_blocks }`. It must also enter the cache key: the
exhaustive destructure of `ServingConfig` in `prompt_cache_key.rs:89-180` forces the decision, and a cached
prefix produced under block read is not interchangeable with a dense one.

GAP-7. H12 read, graph. Missing input: a data-dependent visibility term. The selection depends on the layer's own
query, which exists only inside the graph, so a host-side gather of selected rows is ruled out (section 8). The
cache-side mask is built as `Maximum(is_padding, too_old)` with `-inf` (`lfm2_single_range_cached.rs:376-405`);
an unwindowed layer returns `is_padding` early (:383-385). Smallest change: one optional visibility operand,
`Maximum`-ed in by the same convention as the window term. Two engines, two situations:
- two-range (gemma4): `causal_mask_cached_windowed` is the one function to extend.
- single-range (qwen3, qwen2, openchat; the draft's long-context harness runs gemma4 and qwen3-8B, `FT SPEC.md` acceptance-criteria preamble): the cached block is NEVER
  masked, no exclusion mask exists in its program, and the source's own doc says threading a mask there "would
  mean rewriting its cache algebra, not adding a parameter" (`proxima-tensor/src/spec/descriptor.rs:21-32`).
  A read set on those models is an engine change, not a hook. The draft treats it as prerequisite DAD C1-C3
  (`FT tasks/06` "Prerequisites"), owned by architecture-as-data and not on main.
No new `Op` variant is needed: block bounds are `Reduce` with `Maximum`/`Minimum` and `NegativeInfinity` /
`PositiveInfinity` init (`proxima-tensor/src/op.rs:76-94,140-148`), scoring is `Multiply` plus `Maximum` plus
`Reduce Add`. The draft's first refutation condition (a sixth `Op`) is not triggered.

GAP-8. H7 specialize, selection. Missing input: a top-k. There is no top-k op; the in-algebra form is rank-count,
`Reduce Add` over `Greater`, O(M^2) in block count (`FT SPEC.md` audit row A16). At 128K context and b = 16,
M = 8192, so 8192^2 = 67,108,864 comparisons per layer per decode step. That is arithmetic, not a measurement;
whether it is acceptable is unmeasured. Smallest change: the draft's selection-kernel specialization
(`FT SPEC.md` R9), admitted by the shape of the rank-count expression.

GAP-9. H7 specialize, fusion. The fused cached-attention matcher unwraps exactly the padding `Select`, or the
padding-plus-window form, and drops the mask node on the strength of its runtime `cached_len` bound
(`proxima-tensor/src/bind/dead_code_cached_attention.rs:442-466, 576-616, 871-906`). Any other predicate leaves
the cached score as a `Select`, which fails the next `Multiply` match and declines the fusion
(`:908-918`, debug stage `cached_scale_shape`). So an extra visibility term fails closed (correct, unfused),
not silently dropped; the technique's speed benefit does not survive until the matcher takes a visibility
operand. Reading, not measured; I did not run the matcher on a block-read graph.

GAP-10. H13 seal, storage. Missing input: per-block summary storage and a seal point. `LayerCache` has no block
structure (`residency_caches.rs:112-119`); the per-layer leaf set is the closed enums `LayerCacheNames` (:735-755),
`LayerCacheState` (:644), `DeclaredCacheKind` (:767), `LayerPadRowWidths` (:956). Summaries need leaves `k_min`
and `k_max` per K plane (two planes on `Attention`, three on `DenseAttention`) in all four. Smallest change: a
summary leaf pair added to each K-bearing variant, fed from the fold at block fill.

GAP-11. H13 seal, device residency. On the device-resident path the host never sees K rows ("the host touches
neither", `interop/generate/device_kv.rs:11-13`), so the host `fold_key_bounds` cannot run there. Smallest
change: the same fold as a seal-time `Reduce` evaluation over the placed buffer slice at block fill. That is a
second implementation of the same math (host fold, device reduce); one of them is the compensator. Not
resolved here: it needs a measurement of whether the host path is needed at all on the CPU engine.

GAP-12. Config validation. The draft's reachability matrix has `RectifyNeedsSparseRead` and `RectifyNeedsRewind`
(`FT SPEC.md:363-364`) but no row for `decode.rectify.every > kv.seal.horizon_rows`, and card 7.4 declines to add
one because "the typed runtime refusal already names it" (`FT tasks/07:31-32`). A runtime refusal means the
config loads and validates and fails at step `every`, after `every` tokens were generated. Add the row. Same for
`every > ring slack` (GAP-5) and `every > device step rows` (GAP-4).

## 6. is any of it a pipe? each candidate, against the two gates

- Seal summary as `Pipe<In = rows, Out = summary>`: the call site is `fold_key_bounds(rows, width, &mut low,
  &mut high)` before and after; a `Pipe` impl adds a future and a host type for the same two lines, and
  `kv.seal.summaries` is a closed set (`content_key`, `key_minmax`), so `enum + match`. Relocation; not minted.
- Enter/commit as pipes: `rectify_range` is a pure function over three integers; a pipe would add an async
  boundary inside the step closure for a modulo.
- Read as a user-supplied pipe: not possible. A pipe cannot be named in config, and a read variant is a graph
  builder run at lowering time. Config-as-composition requires a variant to be data; `Block { .. }` is a
  compiled variant, so it is a core edit by definition (GAP-6/7).
Therefore: zero pipes; the three sketched items are pure functions in core (the draft's R23b placement rule).

## 7. minimal hook surface the technique needs

1. H11: one pure function (`rectify_range`), a `Verify` that carries `last` and an `accept_all`, and commit by
   rewind-then-append. Nothing else on the FSM. (GAP-1, 3, 4, 5)
2. H12: one optional visibility operand in the cache-side mask, one `ReadSpec` field defaulting to `Dense`, and
   summary leaves as graph inputs. (GAP-6, 7)
3. H13: summary storage in the cache-leaf enums and one fold at block fill. (GAP-10, 11)
4. Performance only, not correctness: a selection kernel and a fusion-matcher visibility operand. (GAP-8, 9)
5. Validation rows. (GAP-12)
Prerequisites owned elsewhere: verify programs for every family (architecture-as-data R3, GAP-2); cache-side mask
unification across engines (DAD C1-C3, GAP-7).

## 8. designs abandoned

- Host-side read set: gather the selected blocks' rows into the padded KV leaf, which `KvPadScratch::fill`
  already builds each step (`residency_caches.rs:278-310`), and set `cached_len` to the gathered count with zero
  graph change. Abandoned: block scores depend on the layer's own query, which only the graph holds
  (`FT tasks/06` "Binding"); on the device-resident path the leaf is an input placement over one contiguous
  buffer, `(NodeId, &PlacedBuffer, usize)` (`device_kv.rs:39-43`), with no multi-range form.
- Rectify as its own FSM state: abandoned for `Verify` plus `accept_all`.
- `overwrite_rows` + `RewindIntoSealed` (draft card 7.4): abandoned for truncate-then-append.
- A configurable enter order: abandoned (GAP-1 rationale).

## verdict

fits after 12 named hook changes (GAP-1 to GAP-12); two of them are owned by other specs (GAP-2 by
architecture-as-data R3, the single-range half of GAP-7 by DAD C1-C3), and GAP-8, GAP-9 are speed-only. Not a
pipe-and-config fit: the technique introduces a compiled read primitive and cache storage, so "zero core edits"
is false, and the "one ~40-line pipe" is zero pipes. Not claimed: that block read at keep ratio 1.0 reproduces
dense ids, that rectify recovers dense quality, or any speed; none was run.
