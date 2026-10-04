# sketch 10: tiered chunk cache at H13 (seal, evict, demote) with N7 (seal codec)

status: paper test. Read at main 4b4be6cf with `git show main:<path>`. Nothing was built or run. `interop` =
`proxima-model-interop/src`; `PC` = `interop/generate/prompt_cache.rs`. `FT` = the draft under
`/Users/brianbruggeman/repos/slot-0/proxima-fsm-techniques/proxima-tensor/specs/fsm-techniques/` (read-only, one
opinion, not main). Technique as the draft states it (`FT SPEC.md:82`, row T13): hash-keyed chunks across device,
host, disk and remote tiers, asynchronous offload, LRU eviction, chunk size 256. I did not read the LMCache paper.

## ground facts on main (what exists, read)

- The cache is a map of whole conversations, not chunks: `PromptCache.entries: BTreeMap<u64, CacheEntry>`
  (`PC:539-548`), one `PrefixState` per entry (`PC:359-361`), indexed by a block trie that stores no rows, only names
  an owning entry and resolves its ids through a caller closure (`interop/generate/prefix_trie.rs:1-26`, `PC:533-535`).
- Host memory only. `kv_offload=true` is refused at admission (`interop/serving.rs:1325-1333`).
- Eviction is inline: `eviction_victim` picks the first entry with `branch_base` set, else the lowest stamp
  (`PC:820-826`); it is called when the trie is full (`PC:792-795`) and when the entry count or byte budget is
  exceeded (`PC:966-968`). The loser is dropped (`drop_entry`, `PC:806-818`).
- Index capacity equals `max_entries` (`PC:921`, `PC:933-936`; `prefix_trie.rs:127-136`, default 4,
  `interop/prompt_cache_settings.rs:26-28`).
- No KV byte ever reaches a file: `git grep -n -E "File::create|std::fs::|write_all|fs::write" main --
  proxima-model-interop/src/generate` returns only debug dumps (`interop/generate/decode.rs:1057-1266`) and test
  fixtures. No serde in `generate/` either (grep returned zero hits outside a test reading JSON).
- Row-level block extraction exists only for attention layers: `LayerRows` (`interop/generate/ring_checkpoint.rs:28`),
  `RingCheckpoint::capture` returns `None` on any `DenseAttention` or `Ssm` layer (`ring_checkpoint.rs:75`), and
  `rows_are_movable` is false for the same two (`interop/generate/chunk_shift.rs:285-294`).
- `proxima-core` on main has a `config` feature (`proxima-core/Cargo.toml:65`) but no module of model-serving
  decisions (file list: arch, arena, batch, buffer, ring, time, and others). Pure serving decisions on main live as
  `pub(super)` functions in interop: `entry_is_reusable` (`PC:60-67`), `draft_limit_for_step`
  (`interop/generate/drafter.rs:107-116`). So "decision in proxima-core" has no precedent to follow; carried as a
  catalog defect (D5).

## shape chosen, and the contested decision

Shape: configuration, one pure function (`eviction_victim` over an ordered rule list), and ZERO pipes of at most
40 lines. The disk write is a sink (`Pipe` with `Out = ()`, the `place` form), but its body is a file format for
`PrefixState`, `RingCheckpoint` and `CacheKey`, which is core by size and by field visibility (section 5, G4).

Contested decision: is the tier unit a block (the draft: seal, key, summary, tier all at `block_tokens`,
`FT SPEC.md:153-157`) or an entry? Chosen: the entry, which is already the unit of storage, rewind, byte
accounting and checkpointing. Reason read from main: an entry at rest is immutable by construction, because a
request takes the entry out of the cache to mutate it and stores it back (`PC:23-26`); so a seal point and a
`RewindIntoSealed` refusal protect nothing at that grain. A block grain would also be unavailable on hybrid models
(Ssm and DenseAttention layers cannot be cut into rows, see ground facts). Whole-entry demotion works for every
layer kind because `CacheEntry::byte_len` already counts all four `LayerCacheState` variants (`PC:339-353`).

## 1. the configuration

```toml
[prompt_cache]
byte_budget = 2147483648
max_entries = 4
block_tokens = 64

[[kv.tiers]]
kind = "host"

[[kv.tiers]]
kind = "disk"
path = "/Volumes/scratch/proxima-kv"
bytes = 68719476736
max_entries = 4096

[kv]
policy = ["branch", "oldest"]

[kv.seal]
codec = "f32"
```

Default reproduces today: an absent `kv.tiers` is `[host]`; the host tier's size is `prompt_cache.byte_budget` and
its count is `prompt_cache.max_entries`, not repeated under `kv`; `policy = ["branch", "oldest"]` is exactly
`PC:820-826`; `codec = "f32"` is the identity (rows are `Vec<f32>`, `PC:339-353`).

`kv.block_tokens` is not introduced: the key already exists as `prompt_cache.block_tokens` (default 64,
`prompt_cache_settings.rs:66-68`). The draft's separate `kv.block_tokens` would give one fact two names.

## 2. the pure function, and what the draft's four collapse into

The draft names four decisions (`FT tasks/05-tiers.md` header: `eviction_victim`, `demotion_target`, `lookup_tier`,
`initial_tier`). Writing each as the expression it is:

| draft function | the expression on main's shapes | verdict |
|---|---|---|
| `demotion_target(tiers, from)` | `tiers.get(from + 1)` | one list index; not a function |
| `initial_tier` | `tiers.first()` | one list index |
| `lookup_tier` | the tier recorded on the entry that the trie named | a field read |
| `eviction_victim(policy, ..)` | the only real decision | kept |

```rust
pub enum Rule { Branch, Oldest }

pub fn eviction_victim(entries: impl Iterator<Item = (u64, bool)> + Clone, rules: &[Rule]) -> Option<u64> {
    rules.iter().find_map(|rule| match rule {
        Rule::Branch => entries.clone().find(|&(_, is_branch)| is_branch),
        Rule::Oldest => entries.clone().next(),
    }).map(|(stamp, _)| stamp)
}
```

The input is the entries in ascending stamp order, which is `BTreeMap` iteration order (`PC:821-825` relies on it).
`rules = [Branch, Oldest]` reproduces `PC:820-826` line for line.

## 3. worked example (hand-derived, not executed)

Resident entries by stamp, with `branch_base` set on 5 and 7: `[(3, false), (5, true), (7, true), (9, false)]`.
- `rules = [Branch, Oldest]`: `Branch` finds stamp 5 first in ascending order, so the victim is 5.
- `rules = [Oldest]`: the victim is 3.
- no branch present, `[(3, false), (9, false)]`, `rules = [Branch, Oldest]`: `Branch` finds nothing, `Oldest` gives 3.
- empty iterator: `None` (matches `PC:966-968`'s `?` returning `None` from `store`).

Demotion trace with `tiers = [host, disk]` and victim 5: the victim is not dropped; its entry is encoded and handed to
the disk sink, and a record `{stamp 5, ids, path, bytes, key}` stays resident so the trie can still resolve
`ids_of(5)`. A later prompt sharing a block-aligned prefix with ids of 5 reaches stamp 5 through the trie; the
lookup finds the tier is `disk`, reads the file outside the cache lock (G5), decodes, and re-stores it as a host
entry. Real-data test (principle 9): the corpus is `proxima-model-interop/examples/data/war_and_peace.txt`; the
oracle for "a restored entry is the entry" is byte equality of every layer `Vec<f32>` plus equal ids on a real
gemma4 `PrefixState`, which needs a model load and was not run.

## 4. cache-key binding (cross-cutting rule, checked against `generate/prompt_cache_key.rs`)

- Pure moves between tiers with `codec = "f32"`: rows are byte-identical, so nothing about "what a cached row
  means" changes; no key field is needed. This is plausible, not proven (needs the byte-equality test above).
- `codec != "f32"` changes what a restored row is. A restored lossy entry is not interchangeable with a freshly
  computed one, so the key must carry the codec. `CacheKey::of` destructures `ServingConfig` with no `..`
  (`prompt_cache_key.rs:89-180`), so a codec added to `ServingConfig` would not compile until classified. But
  `ServingConfig` is `#[derive(Debug, Clone, Copy, PartialEq)]` (`serving.rs:719`), so the `[[kv.tiers]]` list
  cannot be a field of it. The precedent for list-valued cache configuration on main is a setter that stores into
  `PromptCache`: `set_prewarm_suffix(&[u32])` (`interop/generate/prewarm.rs:141`) and `set_follow_up_closing`
  (`interop/generate/prewarm_follow_up.rs:36`, state at `PC:548-549`). Chosen: tiers by setter (outside the key,
  correct because placement does not change rows), codec as an extra parameter to `CacheKey::of` the way
  `ring_slack_rows` and `ring_write_offset` already are (`prompt_cache_key.rs:82-88`).
- A disk file outlives the process, and the key does not say which weights made the rows: `model_path: _` with the
  stated reason that weights are fixed for the life of a `LoadedModel` (`prompt_cache_key.rs:14-18`, `:91`). A tier
  shared across processes breaks that assumption. The file header must carry a model identity. I found no digest
  function for weights or descriptors on main (`git grep -n -i -E "fn [a-z_]*(digest|fingerprint)"` over interop and
  `proxima-tensor/src/spec` returned only `block_bloom.rs:26 content_hashes`, which hashes token ids). Whether the
  GGUF tensor directory is enough to identify weights: unread, a guess.
- N7 and the quantized-KV refusal: admission refuses `kv_cache_key_quant` and `kv_cache_value_quant` other than F32
  because the quantized read path does not work (`serving.rs:1270-1295`). A seal codec that encodes on demotion and
  decodes back to f32 on promotion never reaches that read path, so the refusal is not in its way. An f32 to Q8_0
  row encoder: not found by `git grep -n -E "fn (quantize|encode|pack)_?q8_0"`; the nearest is `quantize_row_q8k`
  (`proxima-tensor/src/cpu/gemm_q8k.rs:824`), a q8_K activation quantizer. Whether it suits KV row widths: unread.

## 5. HOOK GAPS (every place the sketch edits core)

G1. H13 evict, decision. Missing input: none; the rule is inline at `PC:820-826`. Smallest change: move the body to
`eviction_victim(entries, rules)` (section 2), called from `PC:792-795` and `PC:966-968`. Today's behaviour is the
default rule list.

G2. H13 place, sink slot. Missing input: somewhere for the victim to go. Both eviction sites drop the loser
(`PC:794`, `PC:968`). Smallest change: pass the victim to the next tier's sink when `kv.tiers` has one, before
`drop_entry`. With `tiers = [host]` there is no next tier and the path is today's drop.

G3. H13 index. Missing input: an index larger than the resident set, and entries that are only ids. The trie's
capacity is `max_entries` (`PC:921`); `ids_in` reads `entry.state.ids` from `entries` (`PC:533-535`), and
`CacheEntry.state` is not optional (`PC:360`). A demoted entry cannot stay in `entries`. Smallest change: a resident
record of `{ids, tier, path, bytes, key}` that `ids_in` can also resolve, and an index capacity separate from the
resident count. Call site both ways for a new "cold entry" type: today `entries.get(&stamp)` gives a `CacheEntry`;
with a second map `cold: BTreeMap<u64, ColdRecord>` consulted by `ids_in`, the lookup is the same line, so the
record is data inside the cache, not a new public type.

G4. H13 and N7 serialization. Missing input: any encoding of `PrefixState`, `RingCheckpoint` and `CacheKey`. The
fields are `pub(super)` (`interop/generate/residency_caches.rs:707-711`; `ring_checkpoint.rs:36-39`), so the writer
and reader must live inside `generate`. This is the one place the "about 40 lines" budget does not hold: a header
(format version, model identity, key fields, ids, per-layer plane lengths) plus planes, and the codec hook.

G5. H13 promote, lock discipline. The lookup holds the cache mutex while it takes an entry
(`PC:1060-1065`; the module's own doc says the lock is held only to take an entry out, `PC:23-26`). Reading a
disk file under that lock would stall every request and prewarm. Smallest change: `take_best_shifting` reports a
cold hit (stamp and path) instead of reading, the caller reads outside the lock, stores the entry, and retries
the lookup.

G6. Cache key. Two edits: the codec parameter to `CacheKey::of`, and a model identity in the file header (section 4).

Verdict support: G1 to G6; G7 below is a limit, not a change.

G7 (limit). On hybrid models only whole entries tier, never blocks, because `RingCheckpoint::capture` and
`rows_are_movable` refuse `Ssm` and `DenseAttention` (`ring_checkpoint.rs:75`, `chunk_shift.rs:285-294`). The
sketch's grain (whole entry) is the one that works for them; a block grain would need those two layer kinds to
become row-addressable, which is an engine change.

## 6. is any of it a pipe? each candidate, against the two gates

- Disk sink as `Pipe<In = (stamp, &CacheEntry), Out = ()>`: it passes the pipe question (a sink is the `place` form),
  so the second question decides. Call site both ways: `demote(stamp, &entry)` inside `store` versus
  `sink.call((stamp, &entry))`; the two lines are the same work with a future added around a synchronous file
  write. Relocation, not minted. The body is G4, which no pipe wrapper removes.
- `eviction_victim`: a pure function over an iterator; a pipe adds an async boundary inside `store`, which holds
  the lock.
- Tier choice as a pipe: `tiers.get(from + 1)`; nothing to wrap.
Therefore zero pipes.

## 7. designs abandoned

- Block grain (the draft's seal, per-block key and summary, `RewindIntoSealed`): abandoned for entry grain
  (section "shape chosen").
- `policy = lru | lfu | fifo` as the draft's closed enum: abandoned. Today's rule is not LRU; it evicts unused
  follow-up branches first (`PC:820-826`). `lfu` and `fifo` need inputs `CacheEntry` does not carry: a hit count
  and a creation stamp (fields at `PC:359-383`; the stamp is reissued on every store, `PC:960-961`, so lowest stamp is
  least recently stored, not oldest). An ordered rule list reproduces the default and leaves new rules to new
  entry fields.
- Host tier size duplicated under `kv`: abandoned for reading `prompt_cache.byte_budget`.

## defects found in passing

- D1. `kv.tiers[host].bytes` and `kv.block_tokens` in the draft duplicate `prompt_cache.byte_budget` and
  `prompt_cache.block_tokens`; one fact, two keys, and the draft's own refusal row `BlockTokensConflict` exists only
  because of it (sketch 8 recorded the same collision).
- D2. The draft's `policy` default `lru` does not reproduce today's victim rule (above).
- D3. `byte_len` counts `Vec` capacity, not length (`PC:332-355`), so the byte budget over-counts doubled
  allocations; a disk tier budgeted in encoded bytes cannot reuse that number.
- D4. `MissReason` has no tier-aware variant; a cold hit that failed to read would report `Empty` or
  `NoCommonPrefix` (`PC:104-139`).
- D5. SPEC.md places each stage's decision "in proxima-core", but `proxima-core/src` has no serving-decision module
  and the working precedents are `pub(super)` functions in interop. Either the SPEC names the crate that will hold
  them, or it follows the precedent.

## verdict

fits after 6 named hook changes (G1 to G6). Not a pipe-and-config fit at the storage edge: G4 is a file format and
G3 is index plumbing, both in core. Not claimed: that a restored entry decodes to the same ids as a resident one
(unrun), that disk promotion beats a re-prefill (unmeasured), or any speed.
