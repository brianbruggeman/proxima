# sketch 9: CacheBlend (H9 assemble, H12 read, N9 position; touches H13)

status: paper test. Read at main 4b4be6cf (`git show main:<path>`). Nothing was built or run; every
"hand-derived" value is derived from cited lines, not executed. Paths relative to the proxima repo root;
`interop` = `proxima-model-interop/src`, `tensor` = `proxima-tensor/src`.

Technique, as this sketch uses it (primary source not re-read for this sketch; `frontier.md` in this
directory is the SPEC's own record of the recent sources): chunks of retrieved text are prefilled ONCE,
alone, and their KV rows reused in a later prompt at new positions; because each chunk was computed
without seeing the others, a small fraction of tokens (chosen by how much their KV differs) is recomputed
in the full context so cross-chunk attention is repaired. Two inputs are therefore needed: moving rows to
new positions (rotate the keys), and recomputing a SPARSE subset of positions against the assembled cache.

## shape chosen, and the contested decision

Shape: H9 assemble gets a configured section `[prefill.assemble]` naming where chunks come from and what
fraction to recompute; the selection of which positions to recompute is one pure function, written below
(about 20 lines); the position move is the chunk-shift code that already exists.

Contested decision: build the sparse recompute (CacheBlend as published), or the contiguous variant the
existing code can almost express? Both are written down; neither is chosen as "the" answer.
- Variant B (blend): recompute the top `r` per mille of tokens by KV deviation, at arbitrary positions.
  Needs hook changes GAP-2 and GAP-3 (section 6), which are structural.
- Variant S (seam): recompute a fixed number of tokens after each chunk boundary, as an ordinary
  contiguous gap prefill. Needs GAP-1 and GAP-6 only. It is an approximation of B's repair (it fixes the
  tokens nearest each seam, not the tokens whose KV drifted most), and whether it is good enough is
  unmeasured.
The sketch's job is to find which hook inputs are missing, so both are traced.

Second gate, call site both ways, for an `Assemble: Pipe<In = Request, Out = StartingCache>` type: before,
`self.prompt_cache_lookup(&prompt_ids, &key, &widths, waited, serving_config)`
(`interop/generate/prompt_cache.rs:1047-1054`); after, `assemble.call(request).await`. Same operands, same
result, so it is a relocation and is not minted.

## 1. ground: what exists today

A great deal of the mechanism exists, and the SPEC's catalog row for H9 ("prefix reuse") understates it.

Prompt cache (H9): a per-model LRU of `PrefixState` entries keyed by token ids and a `CacheKey`
(`interop/generate/prompt_cache.rs:1-31, 359-383`). Lookup is `take_best_shifting` (:644-686): the trie
(`PrefixTrie`, `prefix_trie.rs`) names ONE entry sharing the longest prefix (`best_candidate`, :696-722);
that entry is rewound to the shared prefix (`resume_at`, :745-760). Entry fields: `state`, `key`,
`checkpoints`, `restored_at`, `prewarmed`, `branch_base`, `bloom`, `moved` (:359-382). There is no field
saying how an entry's rows were produced other than `prewarmed` (a prewarm) and `moved` (this request's
lifted chunks).

Chunk shift (N9, partial): `chunk_shift.rs` moves runs of rows to new positions and re-rotates the cached
keys. A cached key is `R(p) k`, rotations compose, so the key at `p + d` is `R(d)(R(p) k)`
(`chunk_shift.rs:21-27`); `rotate_rows` applies the per-pair `(cos, sin)` of the delta
(:173-201); `delta_rotations` builds those tables per layer from the bound program, negative deltas
rotating the other way (:384-452). `plan_runs` finds runs of at least `min_len` tokens by rolling hash,
"each at a stored position at or past where the previous one ended" (:98-154). Gaps between runs are
prefilled; each moved run is written when the prefill reaches it (`apply_moved`, :331-351;
`prefill_through_stops`, `prompt_cache.rs:1128-1183`). Turned on by
`prompt_cache.cache_reuse_min > 0` and `ring_rewind_slack > 0` (`prompt_cache.rs:1059`); the shipped
default is `cache_reuse_min: 0` (`interop/serving.rs:571`).

What it does not do, stated by the code itself: "A moved chunk's rows were computed under the entry's
older context, so layers past the first full-attention layer see keys that differ from a fresh prefill ...
llama-server's reuse has the same property" (`chunk_shift.rs:31-35`), and an entry built from a shift
"carries those rows into later requests" (`serving.rs:509-513`). That is exactly the drift CacheBlend
repairs. Nothing repairs it today.

Content index (H9 input, unused): each entry keeps a bloom filter over its blocks hashed on content alone
(`block_bloom.rs:1-9`), and `PromptCache::bloom_candidates` returns `(entry stamp, prompt block index)`
pairs (`prompt_cache.rs:550-559`). The only caller outside tests logs the counts
(`prompt_cache.rs:1073-1091`). `block_bloom.rs:8-9` says "moving the rows (spec R5) is not built", which
`chunk_shift.rs` contradicts.

Restrictions on what can move (`chunk_shift.rs:285-293`): every layer must be `Attention` with an
unrotated ring, or `SharedFromLayer`; `DenseAttention` and `Ssm` layers refuse. So recurrent hybrids
(qwen3.5's GDN layers) cannot be blended at all: their state is a function of the whole prefix, not of
positions. gemma4 E2B qualifies.

KV geometry on gemma4 E2B (the checkpoint a blend would run on first), from the fixtures
(`proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/`): 35 layers, 20 shared-KV, so 15
cache-owning layers: 12 sliding (window 512) and 3 full (`swa_layers.txt`, full at 4, 9, 14 among own-KV).
A sliding layer's ring has capacity `window + slack` (`interop/generate/kv_ring.rs:44-59`); slack is the
larger of the speculative verify width and `ring_rewind_slack`, default 256 (`serving.rs:488-491, 568`),
so capacity is 512 + 256 = 768 rows. A ring layer returns at most `window` live rows on a read
(`kv_ring.rs:67-70`).

The program side: a step evaluates a contiguous block of new positions. `StepInputContext` carries
`new_start` and `new_count` (`interop/architecture.rs:522-537`); the RoPE table is built by
`build_position_inputs(new_ids, start_position, ..)` with `position = start_position + offset`
(`interop/generate/residency_caches.rs:1206-1225`); the attention mask is built from an iota and the
scalar `cached_len`: `query_absolute = query_index + cached_len`
(`tensor/spec/primitives.rs:1406-1433`). New rows are appended at the end of a layer's cache
(`LayerCache::append`; `append_at` for a ring, `kv_ring.rs:117-120`).

## 2. the config a user would write

Real data for sizing: the repo's own measurement, "one entry holds 69.2 MB after a 2,063-token request",
made of "18.9 MB of ring rows that never grow, plus the three full-attention layers' rows at 12,288 bytes
per token" (`serving.rs:475-481`). 768 rows x 12 ring layers x 2,048 bytes = 18,874,368 bytes, which is
the 18.9 MB figure and confirms the capacity of 768 used here.

```toml
[prompt_cache]
byte_budget = 2147483648
max_entries = 96
ring_rewind_slack = 256
cache_reuse_min = 64

[prefill.assemble]
source = "chunks"
chunk_tokens = 500
recompute = "deviation"
recompute_ratio_milli = 150
seam_tokens = 0

[attention.read]
kind = "dense"

[kv.entry]
provenance = "blended"
```

Variant S is the same file with `recompute = "seam"` and `seam_tokens = 32`.

Defaults equal today: with no `[prefill.assemble]` the cache behaves as shipped (`source = "prefix"`,
`cache_reuse_min = 0`, no recompute). The `[prompt_cache]` keys are the existing
`PromptCacheSettings` fields (`interop/prompt_cache_settings.rs:20-77`) with two values changed
(`max_entries` 4 to 96, `cache_reuse_min` 0 to 64); `[prefill.assemble]` and `[kv.entry]` are new.

## 3. field map: every key to a consumer, or a gap

| key | consumer | status |
|---|---|---|
| `prompt_cache.byte_budget`, `max_entries`, `ring_rewind_slack`, `cache_reuse_min` | `PromptCacheConfig` (`serving.rs:472-557`) | EXISTS |
| `prefill.assemble.source = "chunks"` | none: lookup names one entry by longest prefix (`prompt_cache.rs:696-722`) | GAP-1 |
| `prefill.assemble.chunk_tokens` | none; must satisfy `chunk_tokens + 1 <= window + slack` (768) or sliding rows are lost | GAP-4 |
| `prefill.assemble.recompute = "deviation"`, `recompute_ratio_milli` | none; needs sparse positions through the program | GAP-2, GAP-3 |
| `prefill.assemble.recompute = "seam"`, `seam_tokens` | none: `plan_runs` output goes straight to `lift_chunks` (`chunk_shift.rs:474-491`) | GAP-6 |
| `attention.read.kind = "dense"` | the cached-attention ops read the whole live extent | EXISTS (H12 default) |
| `kv.entry.provenance` | none: `CacheEntry` has no provenance field (`prompt_cache.rs:359-383`) | GAP-5 |

## 4. can the decision be a pure function in core plus a pipe?

Decision half: yes, for the part that is a decision. Which positions to recompute is a pure function of a
deviation vector and a ratio. It is a top-`k` selection with a stable tie-break, the same shape as the
sampler's `apply_top_k` (`proxima-tokenizer/src/sample.rs:209-219`). It needs no model access: the
deviation values are an input. About 20 lines, no comments:

```rust
pub fn select_recompute(deviation: &[f32], ratio_milli: u32, positions: &mut Vec<u32>) {
    let keep = (deviation.len() * ratio_milli as usize)
        .div_ceil(1000)
        .min(deviation.len());
    positions.clear();
    positions.extend(0..deviation.len() as u32);
    let order = |left: &u32, right: &u32| {
        deviation[*right as usize]
            .total_cmp(&deviation[*left as usize])
            .then(left.cmp(right))
    };
    if keep < positions.len() {
        positions.select_nth_unstable_by(keep, order);
        positions.truncate(keep);
    }
    positions.sort_unstable();
}
```

Second gate: before, there is no call site; the nearest existing code is `plan_runs` (a pure function
returning `Vec<ChunkRun>`) and `apply_top_k`. A `Select: Pipe` wrapper would be the same line as the free
function, so none is minted.

Placement half: the existing seams are `Lift` (a `&mut dyn FnMut(&CacheEntry, usize) -> Vec<MovedRun>`
borrowed closure passed into `take_best_shifting`, `prompt_cache.rs:531`) and `prefill_through_stops`.
The deviation values come from the model: layer roots give each attention layer's rotated-key node
(`chunk_shift.rs:357-382` reads them off `Qwen35LayerRoots::Attention`), and the decode loop reads
evaluated node values by id (`decode.rs:5742`), so a fresh-versus-cached key difference is readable
without a new tap (plausible; not exercised). What is NOT there is a way to run a subset of positions
(section 6).

## 5. worked example (doubles as the test)

Setup, chosen to fit the geometry above. gemma4 E2B, default ring slack. A prompt of: a system prefix of
20 tokens (BOS included), chunk X of 500 tokens, chunk Y of 500 tokens, a question of 30 tokens: 1,050
tokens. X and Y were each prefilled alone earlier, as their own entries (`chunk_tokens = 500 <= 768 - 1`,
so every sliding-layer row of each entry is still in its ring, by `ring_rows_live`'s condition
`first_needed + capacity >= stored_len`, `chunk_shift.rs:160-165`: `old_end = 501`, `first_needed = 501 - min(500, 512) = 1`,
`1 + 768 >= 501`). Each chunk entry starts with BOS (the tokenizer prepends it), so a chunk's tokens sit
at stored positions 1 to 500.

Moves (hand-derived from `chunk_shift.rs:486-488`, `delta = new_start - old_start`): X lands at
`new_start = 20`, so `delta = 20 - 1 = +19`; Y lands at `new_start = 520`, so `delta = 520 - 1 = +519`.
Each is one `delta_rotations` call per run, applied to the keys of the 15 cache-owning layers
(`rotations[layer]` is `None` for the 20 shared layers, `chunk_shift.rs:390-451`).

Token accounting. Without reuse: 1,050 positions prefilled, 20 of them the shared system prefix reused by
longest-common-prefix, so 1,030 forwarded. Variant B at `recompute_ratio_milli = 150`: the recompute set
is `ceil(1000 * 150 / 1000) = 150` of the 1,000 chunk tokens, plus the 30 question tokens, so 180
forwarded. 850 of 1,030 forward positions skipped. This counts positions, not time; no timing was taken.
Variant S at `seam_tokens = 32`: the first 32 tokens after each of the two boundaries are gap tokens, 64,
plus 30: 94 forwarded.

Memory, hand-derived from the repo's own measurements (`serving.rs:475-481`): one chunk entry is the ring
(18,874,368 bytes) plus 500 tokens x 12,288 bytes = 6,144,000 bytes, so about 25.0 MB. At the 2 GiB
default budget that is `2,147,483,648 / 25,018,368 = 85.8`, so 85 entries; the shipped `max_entries = 4`
(`serving.rs:567`) is the binding limit by a factor of 21, which is why the TOML raises it.

Unit test for the one function written above (exact values, hand-derived): `select_recompute([0.1, 0.9,
0.3, 0.9, 0.0], 400)`: `keep = ceil(5 * 400 / 1000) = 2`; the two largest are index 1 and 3 (both 0.9; the
tie goes to the lower index first, and both survive), so positions `[1, 3]`. With ratio 500: `keep = 3`,
positions `[1, 2, 3]` (0.3 at index 2 is third). With ratio 0: `[]`. With ratio 1000: `[0, 1, 2, 3, 4]`.

Config parity (P4): the `[prompt_cache]` table by the existing settings loader; the new sections not built.

## 6. HOOK GAPS (stage, missing input, smallest generic change)

GAP-1. H9 assemble. Missing input: more than one source entry. Lookup names one entry by longest prefix
and lifts runs only from it (`prompt_cache.rs:652-659, 755`; `lift_chunks(entry, ..)`,
`chunk_shift.rs:459`). A blend assembles chunks that live in different entries, in a request-given order,
and `plan_runs` also requires runs at increasing stored positions (`chunk_shift.rs:98-100`). Smallest
change: `ChunkRun` (`chunk_shift.rs:47-52`, `{old_start, new_start, len}`) gains `source: u64`, the entry
stamp; the assembly plan becomes a function of the request's chunk list instead of a hash search, and the
existing bloom candidates (`prompt_cache.rs:550-559`) are the content-addressed way to find the source
when the request gives text, not ids. Call site both ways: `lift_chunks(entry, ..)` versus
`lift_chunks(cache, plan, ..)`; they differ (many sources), so a signature change.

GAP-2. N9 position / H6. Missing input: positions are a contiguous range, not a list. A sparse recompute
needs the RoPE table, the mask and the cache write to take arbitrary positions. Today: RoPE from
`start_position + offset` (`residency_caches.rs:1225`); mask from `iota + cached_len`
(`primitives.rs:1428-1433`); cache write is `append` at the end or `append_at(start, block)` for one
contiguous block (`kv_ring.rs:117-120`). Smallest change: `new_positions: &[u32]` replaces
`(new_start, new_count)` in `StepInputContext` and `build_position_inputs`; the mask is built from a
positions input leaf instead of `Iota` plus the scalar; and a `write_rows_at(positions, rows)` on
`LayerCache` that overwrites in place. For the existing contiguous case the list is `start..start+count`,
so the default reproduces today (the repo's digests, `arch_data_digest_`, would show it).

GAP-3. H6 / N2 / N3. Missing input: a layer-boundary hand-off, which blend needs in its published form
(all tokens through the first layers, then only the selected subset through the rest). The descriptor's
per-layer residual roots are populated only by the `SingleRange` engine and are EMPTY for `TwoRange`,
which is gemma4's engine (`tensor/spec/descriptor.rs:230-252` doc; gemma4 is `TwoRange`,
`gguf_descriptor.rs:120-124`; digest line `bind.residual_roots=0`). So for E2B there is no hidden state at
a layer boundary to resume from. Smallest change: expose the per-layer residual node on `TwoRange` and
accept a program split at a layer index (`split_layer_program` already exists as "the shared cut
semantics" for routed models, per sketch 3, `execution.rs:179-190`; not re-read here). Status: plausible.
Variant S does not need this.

GAP-4. H13 / ring. Missing input: a bound between chunk size and ring capacity. Sliding layers keep
`window + slack` rows (`kv_ring.rs:44-59`); a chunk entry longer than that has lost the leading sliding
rows, so a recompute inside it cannot read its own window, and `ring_rows_live` correctly refuses the run
(`chunk_shift.rs:160-165`). The constraint `chunk_tokens + 1 <= window + slack` is a configuration
validity rule with no home: the window is a descriptor fact and `apply_serving_config` was not checked for reading it.
Smallest change: `Validate` row on `[prefill.assemble]` against the descriptor's `mask_window` and
`ring_rewind_slack`. Read from code; not measured.

GAP-5. H9 / H13. Missing input: provenance. `CacheEntry` records `prewarmed` and the request's `moved`
runs but nothing that survives storing, so a blended or shifted entry is later offered to an exact-prefix
request as if it were fresh (`serving.rs:509-513` says so for shift). The SPEC's cross-cutting rule
(cache key binds the producing configuration) is not met. Smallest change: `provenance` on `CacheEntry`
(`Fresh`, `Shifted`, `Blended { ratio_milli }`), compared in `best_candidate`'s key filter
(`prompt_cache.rs:715`) so a request that demands exactness refuses non-fresh entries. Call site both
ways: `entry.key == *key` versus `entry.key == *key && entry.provenance.admits(request)`; they differ.
`CacheKey::of` destructures `ServingConfig` with no `..` (`prompt_cache_key.rs:89-180`), so a new config
field forces the decision at compile time; the new `[prefill.assemble]` fields must be named there.

GAP-6. H9. Missing input: a trim between planning and lifting, for variant S. `lift_chunks` takes
`plan_runs`'s output whole (`chunk_shift.rs:474-491`). Smallest change: after `plan_runs`, advance each
run's `new_start`/`old_start` by `seam_tokens` and shorten `len`, dropping runs that vanish; the skipped
tokens fall into the existing gap prefill. A pure function over `Vec<ChunkRun>`, about 10 lines.

## 7. what is not a pipe, and why

- `select_recompute`: a pure function over a slice; the second gate shows the wrapper is the same line.
- The cache lookup and the lift: stateful structure behind a lock, called once per request.
- The position move: `rotate_rows`, a numeric kernel over rows.
- The program split (GAP-3): graph surgery at lowering time, outside the runtime pipe form (sketch 3, same
  statement for `partition_at`).

## 8. designs abandoned

- Doing blend entirely inside a new `Blend` type holding its own chunk store: the prompt cache already
  stores entries, indexes blocks, and moves rows; a second store would duplicate all three.
- Approximating deviation by token-id statistics: the deviation is a property of rows, so it would be a
  guess; only the row difference is data.
- Treating blended output as exact: serving.rs already documents that a shifted entry's ids "can differ" as
  they do under llama-server; the oracle for a blend is therefore not llama's ids, and the feature
  must stay default off and labelled, as `cache_reuse_min = 0` is.
- Putting `ratio` in the cache key as a float: the key compares by equality; per-mille integers (the
  repo's own convention, `min_similarity_milli`, `follow_up_temperature_milli`) keep it exact.

## defects found in passing

- `block_bloom.rs:8-9` says row moving "is not built"; `chunk_shift.rs` builds it. The bloom candidates
  are computed on every lookup and only logged (`prompt_cache.rs:1073-1091`): a content index with no
  consumer.
- `std::sync::Mutex<PromptCache>` (`interop/generate/load_model.rs:1014`) and `Mutex<ExpertSlab>`
  (:1005) are bare std mutexes; the repo rule names `proxima_lock::Mutex`, whose absence the code's own
  doc records (:995-997). Confirmed here: `git grep proxima-lock` over the workspace tomls returns nothing
  and no `proxima-lock` directory exists at the repo root. The rule cannot be satisfied as written; the
  justification in the doc (a synchronous lock held across no await) is the rule's own tier-3 case.

## verdict

fits after 6 named hook changes: GAP-1 (multi-entry source runs), GAP-2 (positions as an input list through
RoPE, mask and cache write), GAP-3 (layer-boundary hand-off on `TwoRange`), GAP-4 (chunk-size versus ring
capacity validity), GAP-5 (entry provenance in the lookup), GAP-6 (seam trim). Variant S needs GAP-1, 4, 5,
6 (four); variant B needs all six. Not claimed: that either variant preserves answer quality, that the
token counts translate to time, or that the deviation is readable as described; none of it was run.
