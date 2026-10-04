# sketch 11: cartridges, load path only, at H9 (assemble) with N8 (import and export)

status: paper test. Read at main 4b4be6cf with `git show main:<path>`. Nothing was built or run. `interop` =
`proxima-model-interop/src`; `PC` = `interop/generate/prompt_cache.rs`; `CS` = `interop/generate/chunk_shift.rs`.
`FT` = the draft under `/Users/brianbruggeman/repos/slot-0/proxima-fsm-techniques/proxima-tensor/specs/fsm-techniques/`
(read-only, one opinion, not main).

Technique as the draft states it (`FT SPEC.md:71`, row T2): a trained KV prefix Z of p rows per layer, loaded into the
prefix slots; cartridges compose by concatenation; training is self-study plus context distillation. Training is out
of scope here; this sketch is the load path. I did not read arXiv:2506.06266.

## ground facts on main (what exists, read)

- H9 today is one function: `run_decode_loop_through_cache` looks the prompt up and falls back to an empty entry:
  `found.unwrap_or_else(|| CacheEntry::empty(key))` (`PC:1248`), then `prefill_through_runs` (`PC:1256-1266`).
  The shifting arm is enabled by `cache_reuse_min > 0 && ring_rewind_slack > 0` (`PC:1059`). So the default
  assemble list is `["prefix"]`, or `["prefix", "shift"]` when that condition holds.
- The assembled-prefill structure already exists: `CacheEntry.moved: Vec<MovedRun>` (`PC:381-383`) and
  `prefill_through_runs`, which prefills the gap before each run, writes the run with `apply_moved`, and continues
  (`CS:538-583`). A `MovedRun` is `{run: ChunkRun{old_start, new_start, len}, ids, layers: Vec<LayerRows>}`
  (`CS:45-71`). A cartridge is one `MovedRun` with `new_start = 0` and `len = p`.
- Re-rotating keys for another position already exists: `delta_rotations(serving_config, delta)` returns the per-layer
  `(cos, sin)` for a position delta from the program's own rope leaves (`CS:390-453`), and `rotate_rows` applies it
  (`CS:173-198`). Loading a second cartridge at offset p1 is the same rotation with `delta = p1`.
- The seeded entry point `generate_from_prefix(&PrefixState, suffix, ..)` is public (`interop/generate/decode.rs:2221`)
  but its result is never cached: the cached path requires `seed.is_none()` (`PC:1221`), and `PrefixState`'s fields are
  `pub(super)` (`interop/generate/residency_caches.rs:707-711`), so no caller outside `generate` can build one from rows.
- The Metal single-range fast path cannot take a seed: callers that need a `PrefixState` pass `force_two_range`, and the
  placed-KV path returns an empty state (`decode.rs:3047-3058` for the doc and the `!force_two_range` condition, `:3076-3090` for the empty state). So a cartridge load excludes the
  device-resident single-range engine.
- Cache rows are two rotated planes plus values per attention layer: `LayerCache {k_even, k_odd, v, ring}`
  (`residency_caches.rs:112-119`); other layer kinds are `DenseAttention`, `Ssm`, `SharedFromLayer`
  (`residency_caches.rs:644-655`).
- No cartridge format, loader or digest exists: `git grep -n -i -E "cartridge|prefix_slots"` over interop and
  `proxima-tensor/src` finds nothing outside this spec set (the draft's own search, `FT SPEC.md:399`, expects 0; I did
  not re-run it). No descriptor or weights digest function exists (grep in sketch 10).

## shape chosen, and the contested decision

Shape: configuration, one list element kind (`load`), ZERO pipes, and core changes at one hook. The two candidate
pipes (an id-prefix rewrite and a file loader) each fail a gate (section 6).

Contested decision: how does a cartridge identify itself to the cache? The cached entry stores ids and the trie
matches ids, with row index equal to id index (`PC:1271` slices `ids[resumed_at..]`; `CS:299-339` extends ids by the
run's ids). A cartridge has p rows and no tokens. Options: (a) p placeholder ids prepended to every prompt;
(b) a separate per-entry "assembled prefix" field. Chosen (a), because everything downstream (trie, lcp, rewind,
checkpoint positions, `entry_is_reusable`) already works on ids, and (b) would touch every one of them. The cost of (a)
is that placeholder ids enter the repetition-penalty window and the drafters' history (`decode.rs:3024-3028` builds
`token_history` from `seed.ids`); unmeasured here. Because placeholder ids are not unique per cartridge, two
cartridges would share entries; the key must bind the cartridge (section 4).

## 1. the configuration

```toml
[[prefill.assemble]]
kind = "prefix"

[[prefill.assemble]]
kind = "load"
cartridge = "/models/cartridges/legal-v3.cart"
```

Order is the same walk as the drafter set: the first stage that yields a starting entry wins
(`interop/generate/drafter.rs:188-197` for the precedent). `prefix` first means a follow-up request reuses the entry
a previous request stored, which already begins with the cartridge rows; `load` runs only on a miss and replaces
`CacheEntry::empty(key)` at `PC:1248`. A second `load` stage with its own path concatenates (section 3).

Default reproduces today: an absent `prefill.assemble` is `[prefix]`, plus `[shift]` under the `PC:1059` condition,
which is today's behaviour.

## 2. the pure function and what it is

The new decision is placement arithmetic for a list of loads: stage i's cartridge starts at the sum of the previous
cartridges' lengths.

```rust
pub fn load_offsets(lengths: &[usize]) -> impl Iterator<Item = usize> + '_ {
    lengths.iter().scan(0, |next, &length| { let at = *next; *next += length; Some(at) })
}
```

That is a prefix sum. Written both ways, `let starts = prefix_sums(lengths)` and the loop that applies each run at
`new_start = starts[i]` are the same lines, so it is a helper, not a type. Each load becomes
`MovedRun { run: ChunkRun { old_start: 0, new_start: start, len }, .. }` with keys rotated by
`delta_rotations(config, start as isize)`; the shift path builds the same struct at `CS:484-489`.

## 3. worked example (hand-derived, not executed)

Two cartridges: A with p1 = 128 rows, B with p2 = 256 rows, both trained at positions starting at 0. Prompt of 40
tokens.
- `load_offsets([128, 256])` yields 0, 128. Total prefix rows 384.
- A: `new_start = 0`, delta 0, no rotation. B: `new_start = 128`, `delta = 128`; B's keys are rotated by the angle for
  position 128 (`CS:21-27`: a key stored at p is `R(p) k`, so one more rotation by the delta gives `R(p + d) k`).
- Starting ids: 128 + 256 placeholder ids, then the 40 prompt ids, so `ids.len() = 424` and prefill covers rows
  384..424.
- A second request with a 55-token prompt sharing the same cartridges: `prefix` matches `ids[..384 + lcp]` of the stored
  entry (`entry_is_reusable`, `PC:60-67`), with `lcp` in the cartridge-prefixed id space; only the new tokens
  prefill.
- Control that must fail: the same prompt under cartridge B alone (`p = 256`) must not hit the entry built under A+B.
  With placeholder ids identical, only the key separates them (section 4), so this is a test of the binding rule,
  not of the loader.

## 4. cache-key binding (cross-cutting rule, checked against `generate/prompt_cache_key.rs`)

This is the sketch where the rule bites. Rows in an assembled entry are not what the ids would compute; they are
trained rows. `CacheKey` has no field for it (`prompt_cache_key.rs:31-78`, fifteen fields, none about the prefix), and
`CacheKey::of` takes extras as parameters (`ring_slack_rows`, `ring_write_offset`, `prompt_cache_key.rs:82-88`). So:
- add the assembled-prefix identity as a parameter of `CacheKey::of`: the ordered list of (cartridge content digest,
  start). It must be a content digest, not the path, because a path can be overwritten under a running server; two
  entries built from one path before and after an overwrite must not match.
- `ServingConfig` cannot carry the list: it is `Copy` (`serving.rs:719`) and `prefill.assemble` is a list. The
  existing precedent for list-valued cache configuration is a setter into `PromptCache`
  (`interop/generate/prewarm.rs:141`, `prewarm_follow_up.rs:36`), outside the key. The cartridge identity then enters
  the key by parameter, since it is derived at load time from file content.
- model identity: `model_path: _` (`prompt_cache_key.rs:91`) rests on weights being fixed per `LoadedModel`
  (`:14-18`). A cartridge file crosses processes, so its header must name the model it was trained on and the loader
  must refuse a mismatch (draft R13b, `FT SPEC.md:267`). I found no digest of weights or descriptor to compare
  against; the sketch needs one chosen. Structural identity (layer count and `LayerPadRowWidths`,
  `residency_caches.rs:956`) detects a wrong shape but not a different fine-tune of the same shape: stated limit.
- N9 position scheme: the cartridge's keys are rotated for positions 0..p; the cache key already carries the rope
  scaling that produced them (`prompt_cache_key.rs:35-37`), so a cartridge trained under one scaling must record it
  and be refused under another.

## 5. HOOK GAPS (every place the sketch edits core)

G1. H9 stage list. Missing input: a list. The three arms (prefix lookup, shift, empty fallback) are inline at
`PC:1059`, `PC:1241-1248`. Smallest change: read `prefill.assemble` once, in the same function, to pick which of
the arms run and in what order; the default list is today's code path.

G2. H9 load, fresh entry shape. `CacheEntry::empty` has `layer_caches: Vec::new()` (`PC:399-409`), and
`append_moved` returns an error when the run names a layer the state does not hold: `layer_caches.get_mut(rows.layer)`
is `None` (`CS:305-306`). So the first run on a fresh entry cannot be written. Smallest change: build the per-layer
empty states from the declared kinds before applying the first run. I read `declared_layer_cache_names_and_widths`
(`decode.rs:1452-1475`) but not the constructor the decode loop uses to grow `layer_caches`; I did not find one that
can be called from outside the loop. Unread beyond that.

G3. H9 load, id space. The prompt `ids` must be prefixed by the cartridge's placeholder ids before the lookup and the
prefill (`PC:1244-1247` look up `ids`; `PC:1271` indexes `ids[resumed_at..]`). Smallest change: one rewrite of `ids`
at the top of the cacheable branch (`PC:1220`), applied only when a `load` stage is configured. The tokenizer adds BOS
for a fresh prompt (`decode.rs:2957-2965` passes `wants_bos` only when `seed` is none); whether the cartridge rows
already include the BOS row is a property of how it was trained, which this sketch does not read: a guess.

G4. N8 file format and loader. Missing input: everything. A `MovedRun` is built from file bytes; `LayerRows` and
`MovedRun` are `pub(super)` (`ring_checkpoint.rs:28`, `CS:67`), so the loader lives in `generate`. Same file format as
sketch 10's G4: one format for entry spill and cartridge, which is the draft's own observation (R13a, `FT SPEC.md:266`).
Not about 40 lines.

G5. Cache key. The assembled-prefix identity parameter and the model identity check (section 4).

G6. `cacheable` requires `seed.is_none()` (`PC:1221`); loading through `generate_from_prefix` would bypass the cache
entirely. The sketch avoids that route; it is recorded because it is the route a caller outside the crate would
take today, and it works only without caching.

Limits, not changes:
- L1. Attention-only. `rows_are_movable` is false for `DenseAttention` and `Ssm` layers (`CS:285-294`), so no
  cartridge on the qwen35 hybrid family through this path; whether an `Ssm` state snapshot is a cartridge at all is a
  question for the technique, unread.
- L2. Ring layers hold only the last window of a loaded run (`CS:308-313`) and the entry is marked restored at the
  run's end, so rewinds to before p need a checkpoint (`CS:341-355`). A cartridge is always a prefix, so nothing
  legitimately rewinds below p.
- L3. The device-resident single-range Metal path is excluded (ground facts).

## 6. is any of it a pipe? each candidate, against the two gates

- Id-prefix rewrite as `Pipe<In = Vec<u32>, Out = Vec<u32>>`: the call site is `ids.splice(0..0, placeholders)`
  before, and `rewrite.call(ids).await` after; same work, plus a future. Relocation; not minted.
- Loader as a source pipe (`In = ()`, `Out = MovedRun`): passes the pipe question; the second gate says the call site
  `load_cartridge(path, config)` versus `source.call(()).await` is the same I/O, and the body is G4 regardless.
- Stage selection as a pipe: a list walk with first-non-empty-wins, which `DrafterSet::draft` already is as an enum
  walk (`drafter.rs:198-`); the config selects members of a closed set, so `enum + match`.
Therefore zero pipes.

## 7. designs abandoned

- A per-entry "assembled prefix" field instead of placeholder ids (option (b)): abandoned, it touches the trie, lcp,
  rewind and checkpoint code, all of which key on ids.
- Loading through `generate_from_prefix` (public today): abandoned, it bypasses the cache (G6) and needs a
  `PrefixState` constructor that does not exist outside `generate`.
- Path as the cache-key component: abandoned for content digest (overwrite hazard, section 4).
- The draft's `load{block keys or cartridge path}` as one variant (`FT SPEC.md:210`): split, block keys are sketch
  10's cold records and a cartridge is a file; one list element each.

## defects found in passing

- D1. `append_moved` truncates a full layer to `run.new_start` before appending (`CS:317`); on a fresh state the
  `truncate` is a no-op, but on a state that already holds more rows than `new_start` it silently discards the
  excess instead of refusing. Fine for the shift path (the state is at `new_start`, checked at `CS:301-303`), but a
  load stage composes it with other stages, so the precondition must stay checked.
- D2. The draft's R13c says cartridges concatenate "with K re-rotated by `chunk_shift::rotate_rows`" (`FT SPEC.md:270`);
  `rotate_rows` is a private function in `chunk_shift.rs` (`CS:173`) and needs the per-layer table that
  `delta_rotations` builds (`CS:390`), so the working entry point is `delta_rotations`, not `rotate_rows`.
- D3. `CacheEntry::store` refuses an entry with empty `layer_caches` (`PC:957`), correct for a fresh empty entry; a
  cartridge-only entry (p rows, no prompt) passes only after G2 gives it layers.

## verdict

fits after 5 named hook changes (G1 to G5); G6 is a recorded route, not a change. Not a pipe-and-config fit: the
loader, the format and the key parameter are core, and the Metal single-range engine and the hybrid layer kinds are
outside it. Not claimed: that rows loaded from a file produce the same ids as the cartridge's own training run (the
draft's oracle, `FT SPEC.md:579`, needs a trainer and a model), or any speed.
