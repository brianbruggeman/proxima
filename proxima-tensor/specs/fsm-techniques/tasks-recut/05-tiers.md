# slice 5 cards, re-cut: host and disk tiers, eviction rules, restore from disk

anchors read at main a7c08c4c (full sha a7c08c4c95836f112742f0e30e4ddd672087cda0), with
`git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`. Paths are under
`proxima-model-interop/src/` unless a card says otherwise. The line hints in this file were re-read at that sha;
an executor still re-locates every anchor by its symbol before editing. Every card runs with
`CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_<n>` and removes it when done. Anything a card tells the
executor to write (code, doc comments, test names, error strings, commit) is English with no slice, stage,
card, AC, R or FT id.

Governing text: `proxima-tensor/specs/pipeline-as-data/SPEC.md` hook H13 (seal, evict, demote), sketch
`sketches/10-tiered-chunk-cache.md` (gap list G1 to G6) and sketch 11 G4 (one file format for spill and
cartridge). Owner direction for this file: build the hook so the policy is a configured list and the tier a
slot; the default reproduces today's cache byte for byte; a technique appears only as a proof test driving
the hook. Here the hook is the `ColdTier` slot in the library (FT5.10, FT5.24, FT5.25, FT5.11); the technique is the disk
tier, which lives in test code written against that slot: its file writer (FT5.4), its content key scheme (FT5.6) and
the tier itself (FT5.23). The proofs drive it (FT5.12 to FT5.16). The only file code in `src/` is the reader of the
block file format (FT5.5), which the cartridge loader in the cartridge slice calls from production code.
Model direction: dense and MoE proofs use gemma4 (E2B is dense, 26B is MoE); the second MoE proof is granite.

## public items per card

The size rule says a card must not add more than one public item that another card consumes. A public item is
one reachable from the crate root: a `pub` type, trait, free function or method, re-exported or in a public module.
The card format file counts nothing else as public, so a `pub(super)` or private helper is not a public item; this
file still counts every helper in the line estimate of the card that adds it. How a card is checked:
- an error variant belongs to the function that returns it;
- the fields, variants and provided methods of a type or trait the same card adds belong to that type or trait;
- the types that appear only in the signature of the function a card adds belong to that function;
- test code is not counted.
Where a second name was needed, it has its own card (FT5.17 to FT5.25 exist for that reason). The one crate-internal
name that later cards in this file write in code is `PromptCache::set_cold` (FT5.23 and the cache tests need a cache
with a tier and no loaded model); `ColdSlot` stays private to `prompt_cache.rs` and no later card names it.

## id map (previous cut -> the 16-card cut)

| previous | 16-card cut | verdict |
|---|---|---|
| FT5.1 | FT5.7 (rule list) and FT5.12 (proof) | recut |
| FT5.2 | none | dropped |
| FT5.3 | FT5.1 | kept (format amended, see below) |
| FT5.4 | FT5.2 | kept (format amended) |
| FT5.5 | FT5.3 | kept |
| FT5.6 | FT5.4 | kept |
| FT5.7 | FT5.5 | kept |
| FT5.8 | FT5.6 | kept |
| FT5.9 | FT5.10 (spill), FT5.11 (restore), FT5.12 (proof) | recut |
| FT5.10 | FT5.12 | recut |
| FT5.11 | FT5.13 | kept (retargeted to the entry codec) |
| FT5.12 | FT5.9 (restore a state) and FT5.11 (restore a cold entry) | recut |
| FT5.14 | FT5.8 (encode a state) and FT5.10 (spill at eviction) | recut |
| FT5.15 | FT5.14, FT5.15, FT5.16 (one card per checkpoint) | recut |

Counts for that mapping: 16 cards; 7 kept, 6 recut, 1 dropped.

## id map (16-card cut -> this file)

An audit of the 16-card cut found: the disk tier built into the library as a concrete type with a setter; FT5.10
over the size rule; cards adding more than one public item; two cards whose validation loaded more than one model;
a `needs` line that named no card id; a false 26B fixture count; a stale lock anchor; counts that depended on cards
missing from `needs`; calls not named; a missing import. A later audit found FT5.10 still over the size rule, the
disk tier's file writer and key scheme (FT5.4, FT5.6) written into `src/` ahead of their proof, and a lint failure
in the new `block_file.rs` and `prefix_state_file.rs` test modules. Ids that other slice files cite keep their
meaning (FT5.1, FT5.2, FT5.7, FT5.10, FT5.11, FT5.12 to FT5.16); the cards that had to be split got new ids FT5.17 to
FT5.25, so a `needs` line can point at a higher number.

| id | what the card is now | against the 16-card cut |
|---|---|---|
| FT5.1 | block file header types and `encoded_len` | recut: `encode_block` moved to FT5.17 |
| FT5.17 | `encode_block` | new |
| FT5.2 | `decode_block` and the borrowed view | recut: plane reads moved to FT5.18; needs FT5.17 |
| FT5.18 | `BlockFileView::plane_f32` | new |
| FT5.3 | `BlockFileView::require_digest` | recut: `model_digest` moved to FT5.19 |
| FT5.19 | `model_digest` | new |
| FT5.4 | `write_block_file`, a test support file | recut: moved out of `src/` into test code; needs FT5.17 and FT5.5 |
| FT5.5 | `MappedBlockFile` and `InteropError::BlockFileIo` | recut: its test writes the file with `std::fs::write` and not `write_block_file`; needs FT5.2 and FT5.18 |
| FT5.6 | `chained_keys`, a test support file | recut: moved out of `src/` into test code |
| FT5.7 | `EvictionRule`, `rule_victim` and the default list | recut: `set_eviction_rules` moved to FT5.20 |
| FT5.20 | `LoadedModel::set_eviction_rules` | new |
| FT5.8 | `PrefixState::to_block_file` | kept; needs spelled out |
| FT5.9 | `PrefixState::from_block_file` | kept; needs FT5.8 |
| FT5.10 | the `ColdTier` slot and the demote that keeps an evicted entry cold | recut: no disk code in the library; the cold limits moved to FT5.24 and FT5.25 |
| FT5.24 | size the prefix index for cold entries and bound the cold entries by count | new |
| FT5.25 | bound the cold entries by bytes | new |
| FT5.11 | the promote half of the `ColdTier` slot | recut: labels moved to FT5.21 and FT5.22; lock type and `thaw` fixed; needs FT5.25 |
| FT5.21 | `CachePath::Tier` | new |
| FT5.22 | `MissReason::TierRestoreFailed` | new |
| FT5.23 | the disk tier, written against the slot, and its file tests | new |
| FT5.12 | trace host to disk to host through the disk tier | recut: now drives FT5.23, no non-test change |
| FT5.13 | a real gemma4 E2B state keeps its bytes across tiers | recut: one model load, one command |
| FT5.14 | follow-up from a disk-restored prefix, gemma4 E2B | recut: calls named, drives FT5.23 |
| FT5.15 | the same, gemma4 26B | recut: fixture counts from main |
| FT5.16 | the same, granite MoE | recut: `needs` names the granite cards; one test per run |

Counts: 25 cards in this file; 2 kept (FT5.8, FT5.9), 14 recut (FT5.1, FT5.2, FT5.3, FT5.4, FT5.5, FT5.6, FT5.7,
FT5.10, FT5.11, FT5.12, FT5.13, FT5.14, FT5.15, FT5.16), 9 new (FT5.17 to FT5.25), 0 dropped.
The size exception the card format file lists for FT5.9 described an earlier card; FT5.9 here is one method on an
existing type in one file and needs no exception.

Size check, from the change lists (non-test lines are an estimate from the listed items; files are source files
touched, not counting the test file):

| card | source files | new non-test lines | public item |
|---|---|---|---|
| FT5.1 | 2 | 35 | `encoded_len` |
| FT5.17 | 2 | 35 | `encode_block` |
| FT5.2 | 1 | 60 | `decode_block` |
| FT5.18 | 1 | 30 | `plane_f32` |
| FT5.3 | 2 | 12 | `require_digest` |
| FT5.19 | 1 | 6 | `model_digest` |
| FT5.4 | 0 (two new test-only files) | 0 | none (test code) |
| FT5.5 | 2 | 24 | `MappedBlockFile` |
| FT5.6 | 0 (two new test-only files) | 0 | none (test code) |
| FT5.7 | 3 | 20 | `EvictionRule` |
| FT5.20 | 1 | 14 | `set_eviction_rules` |
| FT5.8 | 2 | 85 | `to_block_file` |
| FT5.9 | 1 | 40 | `from_block_file` |
| FT5.10 | 3 | 84 | `ColdTier` |
| FT5.24 | 1 | 25 | none (crate-internal) |
| FT5.25 | 1 | 12 | none (crate-internal) |
| FT5.11 | 1 | 70 | `ColdTier::promote` |
| FT5.21 | 1 | 15 | `CachePath::Tier` |
| FT5.22 | 1 | 10 | `MissReason::TierRestoreFailed` |
| FT5.23 | 2 (one `mod` line, one new test-only file) | 0 | none (test code) |
| FT5.12 to FT5.16 | 0 | 0 | none (test code) |

## dropped

- previous FT5.2 (demotion target, lookup tier, initial tier, and a `Tier` enum): each is one expression on the
  configured list (`tiers.get(from + 1)`, `tiers.first()`, a field on the entry the prefix index named), so a
  function adds nothing a caller can do; the enum duplicates the list element.

## what changed from the previous cut, and why (all read on main at a7c08c4c)

- Grain is the whole cache entry, not a block. An entry at rest is immutable because a request takes it out of
  the cache to mutate it and stores it back (`generate/prompt_cache.rs` module doc, ~line 23). A block grain
  would also exclude sliding-window layers, whose rows live in a ring. The previous `BlockTiers`, seal-time
  block encoding and block restore are replaced by the entry spill and restore below.
- The decision lives where the precedent lives. proxima-core has no serving-decision module on main; the working
  precedents are `pub(super)` functions in interop (`generate/prompt_cache.rs::entry_is_reusable` ~line 59).
  The eviction rule function therefore sits in `generate/prompt_cache.rs`.
- Eviction is today's rule, not LRU: an unused follow-up branch goes first, then the lowest stamp
  (`generate/prompt_cache.rs::PromptCache::eviction_victim` ~line 819). A closed `Lru/Lfu/Fifo` enum does not
  reproduce it. The rule list `[Branch, Oldest]` does. `Lfu` and `Fifo` need a hit count and a creation order
  that `CacheEntry` does not carry (the stamp is reissued on every store, so the lowest stamp is the least
  recently stored). A rule that reads a new entry field arrives with that field, so none is built here.
- `ServingConfig` is `Copy` (`serving.rs`), so a list cannot be a field of it. List-valued cache configuration on
  main is a setter (`LoadedModel::set_prewarm_suffix`, `generate/prewarm.rs` ~line 140). The rule list and the
  tier are setters outside the cache key, which is correct because where a row rests does not change the row.
- The tier is a hook plus a proof. The library holds a slot (`ColdTier`: demote an evicted entry, promote a cold
  one, discard one that left the cache) and the cache bookkeeping around it: a cold entry keeps its stamp and ids,
  its rows are released, the prefix index still names it, and a cold hit is read back outside the cache lock. The
  library holds no file code beyond the block file format and its mapped reader. The disk tier is a struct in
  `proxima-model-interop/tests/support/disk_tier.rs` that implements the slot with the crate's public API only,
  so it also shows an outside consumer can write a tier. With no tier installed the cache is today's cache.
- The lock on main is `proxima_primitives::sync::blocking::Mutex`, imported at `generate/mod.rs` line 70 and in
  scope in `generate/prompt_cache.rs` through `use super::*`; `lock()` returns the guard directly (no
  `PoisonError`, no `Result`; see `LoadedModel::prompt_cache_bytes` ~line 1007).
- Test models: gemma4 E2B (dense, sliding-window and shared-KV layers), gemma4 26B (MoE), granite MoE. The
  previous oracle card used qwen2, openchat and qwen3; none remain. Oracles are the vendored llama ids in
  `tests/fixtures/llama-parity/<name>/llama_ids.json`; nothing here queries Ollama or llama.

## block file format (amended; shared by spill, blocks and cartridges)

All integers little-endian, floats `f32` little-endian. The previous cut had one row count for the whole file.
An entry needs one per layer (full layers hold every position, a ring layer holds its allocated ring rows, a
shared-KV layer holds none), and a ring layer needs its geometry to be rebuilt without the model. So the layer
table carries rows and ring geometry per layer.

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `b"PXKVBLK1"` |
| 8 | 4 | version `u32` = 1 |
| 12 | 16 | `descriptor_digest` |
| 28 | 8 | `content_key` |
| 36 | 8 | `base_position` (absolute position of the first row; 0 for an entry or a cartridge) |
| 44 | 4 | `layer_count` |
| 48 | 24 each | per layer: `rows`, `k_even_row_bytes`, `k_odd_row_bytes`, `v_row_bytes`, `ring_window`, `ring_capacity` (all `u32`; a layer with no rows is all zero; `ring_window` 0 means not a ring) |

Header length is `48 + 24 * layer_count`. The payload follows: for each layer in order, the `k_even` plane
(`rows * k_even_row_bytes` bytes), the `k_odd` plane, the `v` plane. Payload length must equal the sum over
layers of `rows * (k_even_row_bytes + k_odd_row_bytes + v_row_bytes)`; any other length is malformed. No
checksum: the header digest and the exact-length check refuse a wrong-model or truncated file.

## worked trace of the tiered cache (used by the demote, restore and proof cards)

Entry ids (each entry holds 4 ids and 4 rows per layer; none shares a first token with another):
A = [818, 5279, 529, 7001], B = [2063, 10779, 78113, 236769], C = [194618, 6081, 86460, 699],
D = [3459, 1883, 7733, 1156], E = [5001, 5002, 5003, 5004].
Hot limit 2 entries, cold limit 2 entries (the arguments the tier is installed with), default rules
`[Branch, Oldest]`, stamps issued 0, 1, 2, ... per store. A cold entry keeps its stamp and its ids; its rows are
held by the tier, which the disk tier writes to `<stamp as 16 hex digits>.pxkv` under its directory.

| step | operation | hot stamps after | cold stamps after | why |
|---|---|---|---|---|
| 1 | store A | 0 | none | stamp 0 |
| 2 | store B | 0, 1 | none | stamp 1 |
| 3 | store C | 1, 2 | 0 | 3 hot entries exceed 2; the victim is the lowest stamp, 0 (A), no branch entry exists; it is demoted |
| 4 | restore the cold entry that A + [9] reaches | 2, 3 | 1 | A's cold entry (stamp 0) and its file are discarded; the restored A is stored as stamp 3; 3 hot entries exceed 2; victim stamp 1 (B) is demoted |
| 5 | store D | 3, 4 | 1, 2 | stamp 4; victim stamp 2 (C) is demoted |
| 6 | store E | 4, 5 | 2, 3 | stamp 5; victim stamp 3 (the restored A) is demoted; 3 cold entries exceed 2; the lowest cold stamp, 1 (B), is discarded |

After step 6: entries 2, 3, 4, 5; cold flags true, true, false, false; the directory holds exactly
`0000000000000002.pxkv` and `0000000000000003.pxkv`. After step 4 the hot entry stamp 3 encodes to bytes equal
to the encoding of the original A.

Rule list check: hot limit 2, store A (stamp 0), a follow-up branch of A (stamp 1, `branch_base` set), then C
(stamp 2). Default rules: the victim is stamp 1 (the branch); cold {1}, hot {0, 2}. Rules `[Oldest]`: the victim is
stamp 0; cold {0}, hot {1, 2}.

## designs abandoned (what each constraint ruled out)

- A concrete disk tier in the library (a `DiskTier` type, a spill file owner and a `LoadedModel::set_disk_tier`
  setter, the previous cut of this file): abandoned for the `ColdTier` slot. The technique has to appear as a proof
  driving a hook, and a setter that switches a concrete technique on is the technique built into the library. What
  the constraint changed: the library holds only the slot and the cache bookkeeping; the file naming, the content
  key and the directory are the proof's.
- A trait and a `LoadedModel` setter as two public items: abandoned for one trait whose provided `install` method
  is the installer, so one card adds one public item.
- `write_block_file` and `chained_keys` as library functions (the previous cut of FT5.4 and FT5.6): abandoned for test
  support files. The call site written both ways is the same line (`write_block_file(dir, key, &bytes)`) whether the
  function lives in `src/` or in `tests/support/`, and no code in `src/` calls either one, so the library copy
  bought nothing a caller could do and put the disk tier's crash rule and file naming ahead of its proof. What the
  constraint changed: `MappedBlockFile` stays in `src/` because it is the one file function with a production caller
  (the cartridge loader opens a block file through it, `tasks-recut/14-cartridges.md` at the `open_cartridge` card), and the writer and the
  key scheme belong to the tier that owns the directory.
- One card for the whole slot and its limits (the previous cut of FT5.10, about 150 non-test lines by the
  count of its change list): abandoned for the slot and the demote (FT5.10), the prefix index sizing and the count bound
  (FT5.24) and the byte bound (FT5.25). What the constraint changed: each card adds the one installer argument whose
  field it reads (`max_entries` in FT5.24, `byte_budget` in FT5.25), so no card carries a field nothing reads.
- The slot as a pipe: a demote runs inside `PromptCache::store` under the synchronous cache lock, and a pipe's
  call is asynchronous, so the call site written both ways is `tier.demote(..)` against a future that would need a
  runtime inside the lock. A promote is read outside the lock as a plain call for the same reason.
- Block-grain tiers with seal points, a `BlockTiers` store and per-block restore: abandoned for the entry, which
  is already the unit of storage, rewind and byte accounting, and which carries ring layers.
- A closed policy enum `Lru | Lfu | Fifo`: abandoned for the ordered rule list that reproduces today's rule.
- A second map of cold records of a new type: abandoned for the existing `CacheEntry` with its rows released and
  an owned handle on the tier's copy; the trie, the id lookup and the capacity count then need no second source.
- A `Tier` enum and three tier functions: dropped (call site both ways is the same line).
- Reading the file inside the cache lock: abandoned for a ticket taken under the lock, a read outside it, and a
  store under the lock after it.
- A codec parameter on the cache key: not built. With the identity codec a restored row equals the stored row
  (the trace and the real-state card assert it byte for byte), and a parameter with exactly one value fails the
  call-site test. The first non-identity codec arrives with the technique that needs it and adds the parameter then.

## cards

### 5.1 block file header types and their encoded length

- id: FT5.1
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `lib.rs`: where root modules are declared (`mod bind;` ~line 29);
  - the "block file format" section of this file;
  - `generate/chunk_shift.rs::rotate_rows (~line 173)`: a pure function over `&mut [f32]` planes, for style.
- change:
  1. `proxima-model-interop/src/block_file.rs` (new): private `const HEADER_FIXED_BYTES: usize = 48;` and
     `const LAYER_RECORD_BYTES: usize = 24;` (the format section's numbers), then
     - `#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)] pub struct BlockFileLayer { pub rows: u32, pub k_even_row_bytes: u32, pub k_odd_row_bytes: u32, pub v_row_bytes: u32, pub ring_window: u32, pub ring_capacity: u32 }`;
     - `#[derive(Debug, Clone, PartialEq, Eq)] pub struct BlockFileHeader { pub descriptor_digest: [u8; 16], pub content_key: u64, pub base_position: u64, pub layers: Vec<BlockFileLayer> }`;
     - `pub fn encoded_len(header: &BlockFileHeader) -> usize`: `HEADER_FIXED_BYTES + LAYER_RECORD_BYTES * layers.len()` plus,
       per layer, `rows * (k_even_row_bytes + k_odd_row_bytes + v_row_bytes)`, every multiply and add done with
       `saturating_mul` and `saturating_add` on `usize` (each `u32` widened with `as usize`). One doc line says it saturates
       and why that is safe: a header whose size does not fit memory can have no planes to match it.
     One doc line per public field with a real value (for example `rows: 57`, `ring_window: 512`). The types are the
     argument of `encoded_len` and of the encoder and decoder that follow; this card adds no other function.
  2. `proxima-model-interop/src/lib.rs`: declare `#[cfg(feature = "std")] pub mod block_file;` after `mod bind;`.
     A public root module is deliberate: every item is public API and the tests are its use, so no dead code.
- test: declare the unit-test module at the end of `block_file.rs`:
  `#[cfg(test)] #[allow(clippy::unwrap_used, clippy::expect_used)] mod tests { use super::*; .. }`. The workspace denies
  `unwrap_used` and `expect_used` (workspace `Cargo.toml`, `[workspace.lints.clippy]`) and `clippy.toml` grants no
  test exemption, so each in-crate test module that calls fallible functions carries this allow, as
  `generate/kv_ring.rs` (~line 295) and `generate/prompt_cache.rs` (~line 1295) do on main. The later cards put
  their tests in this module and call every `Result`-returning function with `.expect("<what failed>")`; none of
  the tests returns `Result`. Tests (names start `blockfile_encoded_len_`) with shared helpers
  `ramp(count: usize, start: f32) -> Vec<f32>` (`start + index as f32 * 0.5`) and `fixture_header() -> BlockFileHeader`.
  The fixture is 3 layers: layer 0 full `{rows: 4, 8, 8, 4, 0, 0}`, layer 1 ring `{rows: 3, 8, 8, 4, ring_window: 2, ring_capacity: 3}`,
  layer 2 absent (default); digest `[0x11; 16]`, content key `0x0102030405060708`, base position 64.
  - `blockfile_encoded_len_counts_header_table_and_planes`: `encoded_len(&fixture_header()) == 260` (48 + 72 for three
    layer records = 120, plus 80 for layer 0 and 60 for layer 1); a header with no layers gives 48; a header whose only
    layer is `{rows: 1, 4, 4, 4, 0, 0}` gives 48 + 24 + 12 = 84;
  - `blockfile_encoded_len_saturates_on_an_oversized_header`: a header whose only layer is
    `{rows: u32::MAX, k_even_row_bytes: u32::MAX, k_odd_row_bytes: u32::MAX, v_row_bytes: u32::MAX, 0, 0}` gives `usize::MAX`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_1 cargo nextest run -p proxima-model-interop --features std -E 'test(/blockfile_encoded_len_/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/block_file.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): describe a kv block file header and its length`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add an encoder, a decoder, a digest or file io (later cards); add an error variant; touch `generate/`.
- gpu: none

### 5.17 block file bytes: encode (pure, in memory)

- id: FT5.17
- needs: FT5.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `block_file.rs` (FT5.1): `BlockFileHeader`, `BlockFileLayer`, `encoded_len`, and the test helpers `ramp` and `fixture_header`;
  - `error.rs::InteropError (~line 13)`: variant style; `SidecarIo (~line 52)` is the io-carrying style;
  - the "block file format" section of this file.
- change:
  1. `proxima-model-interop/src/error.rs`: add the variant `BlockFileMalformed { reason: &'static str }` with
     `#[error("kv block file is malformed: {reason}")]`.
  2. `proxima-model-interop/src/block_file.rs`: private `const BLOCK_FILE_MAGIC: [u8; 8] = *b"PXKVBLK1";` and
     `const BLOCK_FILE_VERSION: u32 = 1;` (the decoder in the next block file card reads the same two), then
     `pub fn encode_block(header: &BlockFileHeader, planes: &[&[f32]], out: &mut Vec<u8>) -> Result<(), InteropError>`.
     `planes` is the flat list `[k_even_0, k_odd_0, v_0, k_even_1, ...]`, three per layer, an empty slice for an
     absent plane. Refusals, exact strings: `"plane count is not 3 per layer"` when `planes.len() != 3 * layers.len()`;
     `"plane length disagrees with the header"` when any plane's `len() * 4` differs from `rows * row_bytes` of its
     layer (compare with `checked_mul`, so an oversized header refuses instead of overflowing). Both checks run before
     `out` is touched. It then clears `out`, calls `reserve(encoded_len(header))` once, writes the header then every
     `f32` through `to_le_bytes`. No `std::io`, no file system.
- test: add in the `tests` module of `block_file.rs` (declared in FT5.1 with `#[allow(clippy::unwrap_used, clippy::expect_used)]`;
  every successful `encode_block` call in these tests is `.expect("encode the fixture")`, no test returns `Result`)
  tests (names start `blockfile_encode_`) with the helpers `words_le(&[u32]) -> Vec<u8>`
  and `fixture_planes() -> Vec<Vec<f32>>`, over the FT5.1 fixture. Planes: layer 0 `ramp(8, 0.0)`, `ramp(8, 100.0)`,
  `ramp(4, 200.0)`; layer 1 `ramp(6, 300.0)`, `ramp(6, 400.0)`, `ramp(3, 500.0)`; layer 2 three empties.
  - `blockfile_encode_header_bytes`: asserts `out[0..8] == *b"PXKVBLK1"`, `out[8..12] == [1, 0, 0, 0]`,
    `out[12..28] == [0x11; 16]`, `out[28..36] == [8, 7, 6, 5, 4, 3, 2, 1]`, `out[36..44] == [64, 0, 0, 0, 0, 0, 0, 0]`,
    `out[44..48] == [3, 0, 0, 0]`, `out[48..72] == words_le(&[4, 8, 8, 4, 0, 0])`,
    `out[72..96] == words_le(&[3, 8, 8, 4, 2, 3])`, `out[96..120] == [0u8; 24]`;
  - `blockfile_encode_payload_layout`: `out.len() == 260` and `encoded_len(&header) == 260` (120 header + 80 + 60);
    `out[120..124] == 0.0f32.to_le_bytes()`, `out[152..156] == 100.0f32.to_le_bytes()` (layer 0 `k_odd` starts after 32
    bytes), `out[200..204] == 300.0f32.to_le_bytes()` (layer 1 starts after layer 0's 80 bytes),
    `out[256..260] == 501.0f32.to_le_bytes()`; encoding a second time into the same `out` leaves `out.len() == 260`;
  - `blockfile_encode_refuses_malformed_planes`: one `f32` removed from plane 1 gives
    `Err(InteropError::BlockFileMalformed { reason: "plane length disagrees with the header" })` and 8 planes instead
    of 9 gives `"plane count is not 3 per layer"`; after either refusal a previously filled `out` still holds its old
    bytes. `InteropError` has no `PartialEq`, so assert with `matches!`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_17 cargo nextest run -p proxima-model-interop --features std -E 'test(/blockfile_encode_/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/error.rs`, `proxima-model-interop/src/block_file.rs`
- commit: `feat(interop): encode kv rows into block file bytes`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add decode, file io or a digest (later cards); touch `generate/`.
- gpu: none

### 5.2 block file bytes: decode (pure, borrowed view)

- id: FT5.2
- needs: FT5.17
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `block_file.rs` (FT5.1, FT5.17): the types, `encoded_len`, the private constants and the test helpers `fixture_header`, `fixture_planes`, `words_le`;
  - the "block file format" section of this file;
  - `error.rs::BlockFileMalformed` (FT5.17).
- change:
  1. `proxima-model-interop/src/block_file.rs`: add `pub struct BlockFileView<'bytes> { pub header: BlockFileHeader, pub payload: &'bytes [u8] }`
     (one doc line each: the parsed header; the bytes after the layer table, borrowed from the input) and
     `pub fn decode_block(bytes: &[u8]) -> Result<BlockFileView<'_>, InteropError>`. Refusals, all
     `BlockFileMalformed { reason }`, exact strings, checked in this order: `"shorter than the fixed header"`
     (fewer than 48 bytes), `"bad magic"`, `"unsupported version"` (version other than 1), `"truncated layer table"`
     (the table needs `layer_count * 24` bytes; use `checked_mul` and `checked_add` so a huge count refuses instead of
     overflowing), `"payload length disagrees with the header"` (`bytes.len() != encoded_len(&header)`; the
     saturating length cannot equal a real length). The header's `layers` vector is the only allocation (cold path);
     the payload stays a borrow of `bytes`.
- test: add in the `tests` module of `block_file.rs` (it carries the lint allow from FT5.1; `encode_block` and the
  successful `decode_block` calls are `.expect(..)` with a message naming what failed, no test returns `Result`)
  tests (names start `blockfile_decode_`), encoding with the FT5.17 fixture:
  - `blockfile_decode_reads_the_header_back`: encode the fixture, decode it; `view.header == fixture_header()` and
    `view.payload.len() == 140` (260 minus the 120 header bytes);
  - one refusal test each, asserting the exact `reason` with `matches!`: `blockfile_decode_refuses_short` (47 zero
    bytes: `"shorter than the fixed header"`), `blockfile_decode_refuses_bad_magic` (byte 0 changed to `b'X'`),
    `blockfile_decode_refuses_bad_version` (bytes 8..12 set to 2),
    `blockfile_decode_refuses_truncated_layer_table` (`&bytes[..100]`: the fixture needs 120),
    `blockfile_decode_refuses_wrong_payload_length` (`&bytes[..bytes.len() - 4]` and, separately, one extra byte
    appended).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_2 cargo nextest run -p proxima-model-interop --features std -E 'test(/blockfile_decode_/)'`
- expect: `6 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/block_file.rs`
- commit: `feat(interop): decode block file bytes as a borrowed view`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add plane reads, file io or digest checks; copy payload bytes.
- gpu: none

### 5.18 read a plane out of a decoded block file

- id: FT5.18
- needs: FT5.2
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `block_file.rs::BlockFileView` and `decode_block` (FT5.2): the borrowed payload this card reads;
  - the "block file format" section of this file: the plane order inside a layer.
- change:
  1. `proxima-model-interop/src/block_file.rs`: private `fn plane(&self, layer: usize, plane: usize) -> Option<&[u8]>` on
     `BlockFileView`: the bytes of `plane` (0 `k_even`, 1 `k_odd`, 2 `v`) of `layer`, an empty slice for an absent
     plane, `None` when `layer` is out of range or `plane > 2`. Offsets come from prefix sums of the layer table at
     call time; nothing is stored (decode already proved the payload length, so the sums stay inside it).
  2. same file: `pub fn plane_f32(&self, layer: usize, plane: usize, out: &mut Vec<f32>) -> Option<()>` on `BlockFileView`:
     clears `out`, appends the plane through `f32::from_le_bytes`; `None` when `plane` returns `None` or the plane's
     byte length is not a multiple of 4.
- test: add in the `tests` module of `block_file.rs` (it carries the lint allow from FT5.1; every `encode_block` and
  `decode_block` call is `.expect(..)` with a message naming what failed, no test returns `Result`)
  tests (names start `blockfile_plane_`):
  - `blockfile_plane_round_trip_bytes_identical`: encode the FT5.17 fixture, decode it; every plane of every layer read
    through `plane_f32` equals the fixture bit for bit (`to_bits`); `view.plane(0, 0).map(<[u8]>::len) == Some(32)`;
    `view.plane(2, 0) == Some(&[][..])`; `view.plane(3, 0) == None`; `view.plane(0, 3) == None`; re-encoding the decoded
    header and the planes read back gives bytes equal to the first encoding;
  - `blockfile_plane_f32_refuses_a_plane_that_is_not_whole_floats`: encode a one-layer block `{rows: 1, 4, 4, 4, 0, 0}`,
    patch the layer table's `k_even_row_bytes` (bytes 52..56) to 6 and append 2 zero bytes so `decode_block` still
    accepts the length; `plane_f32(0, 0, &mut out)` is `None`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_18 cargo nextest run -p proxima-model-interop --features std -E 'test(/blockfile_plane_/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/block_file.rs`
- commit: `feat(interop): read block file planes as floats`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: make `plane` public; copy the payload; add file io.
- gpu: none

### 5.3 refuse block files written for another model

- id: FT5.3
- needs: FT5.17, FT5.2
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `block_file.rs` (FT5.1, FT5.17, FT5.2): `BlockFileView::header`, `descriptor_digest`, the fixture helpers;
  - `error.rs::BlockFileMalformed` (FT5.17): the variant this one sits beside.
- change:
  1. `proxima-model-interop/src/error.rs`: add `BlockFileDigestMismatch { expected: [u8; 16], found: [u8; 16] }`
     with `#[error("kv block file was written for another model: expected digest {expected:02x?}, found {found:02x?}")]`.
  2. `proxima-model-interop/src/block_file.rs`: `BlockFileView::require_digest(&self, expected: [u8; 16]) -> Result<(), InteropError>`:
     `Err(BlockFileDigestMismatch { expected, found: self.header.descriptor_digest })` when they differ.
- test: add `blockfile_digest_mismatch_refused` in the `tests` module of `block_file.rs` (it carries the lint allow from
  FT5.1; `encode_block` and `decode_block` are called with `.expect(..)`, the test returns no `Result`): encode the fixture with digest `[0x11; 16]`;
  `require_digest([0x22; 16])` matches `Err(InteropError::BlockFileDigestMismatch { expected, found })` with
  `expected == [0x22; 16]` and `found == [0x11; 16]`; `require_digest([0x11; 16])` is `Ok(())`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_3 cargo nextest run -p proxima-model-interop --features std -E 'test(/blockfile_digest_mismatch_refused/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/error.rs`, `proxima-model-interop/src/block_file.rs`
- commit: `feat(interop): refuse block files written for another model`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: compute a digest here (next card); hash anything per block; add a dependency.
- gpu: none

### 5.19 digest of the bound program

- id: FT5.19
- needs: FT5.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `Cargo.toml (~lines 45, 339)`: the `std` feature already lists `dep:xxhash-rust`, with `xxh3` enabled on the dependency;
  - `tests/arch_data_baseline.rs::ops_digest (~line 142)`: the precedent for digesting a bound program through
    its `Debug` text (test-only there);
  - `generate/block_bloom.rs (~line 14)`: the `xxhash_rust::xxh3` import style in this crate;
  - `proxima-tensor/src/spec/primitives.rs::input_leaf (~line 723)`: how the test builds a program.
- change:
  1. `proxima-model-interop/src/block_file.rs`: `pub fn model_digest(program: &[proxima_tensor::op::Op]) -> [u8; 16]` =
     `xxhash_rust::xxh3::xxh3_128(format!("{program:?}").as_bytes()).to_le_bytes()`. One-line why: no production digest
     of weights or descriptor exists on main (`git grep -n -i -E "fn [a-z_]*(digest|fingerprint)" main -- proxima-model-interop/src proxima-tensor/src/spec`
     prints one line, a test helper at `proxima-tensor/src/spec/tests.rs:17474`); the bound program fixes the layout
     the rows were computed for. It allocates the whole debug text, so a caller digests once per model load and never
     per block.
- test: add `blockfile_model_digest_separates_programs` in the `tests` module of `block_file.rs` (it carries the lint
  allow from FT5.1; any `Result` the program builder returns is `.expect(..)`, the test returns no `Result`): two programs, each built with
  `proxima_tensor::spec::input_leaf(&mut program, DType::Float32, vec![Extent::Static(4)], name)` for names `"a"` and
  `"b"` (`DType` and `Extent` from `proxima_tensor`); the digests differ; the same program digested twice is equal.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_19 cargo nextest run -p proxima-model-interop --features std -E 'test(/blockfile_model_digest_separates_programs/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/block_file.rs`
- commit: `feat(interop): digest the bound program a block file was written for`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a dependency; add a `LoadedModel` accessor (a later slice reads the bound program from inside the crate).
- gpu: none

### 5.5 read a block file through a memory map

- id: FT5.5
- needs: FT5.2, FT5.18
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `Cargo.toml (~lines 49, 298, 314, 357)`: `std` already enables the optional `memmap2`; `tempfile.workspace = true` (~line 357) is a dev-dependency, used by the test;
  - `expert_sidecar.rs::MappedExpertSidecar::from_file (~line 551)`: the production precedent for `unsafe { memmap2::MmapOptions::new().map(..) }` and its `SAFETY:` note;
  - `error.rs::SidecarIo (~line 52)`: the io-carrying variant style; this card adds one beside it;
  - `block_file.rs` (FT5.2, FT5.18): `decode_block`, `plane_f32`, and the `tests` module with its helpers `fixture_header`, `fixture_planes`, `ramp`.
- change:
  1. `proxima-model-interop/src/error.rs`: add `#[cfg(feature = "std")] BlockFileIo { path: std::path::PathBuf, source: std::io::Error }`
     with `#[error("kv block file io at {path:?}: {source}")]`.
  2. `proxima-model-interop/src/block_file.rs`: `pub struct MappedBlockFile { map: memmap2::Mmap }` with
     `pub fn open(path: &std::path::Path) -> Result<Self, InteropError>` (open the file, then `unsafe { Mmap::map(&file) }`
     with a one-line lowercase comment stating why it is sound: a block file is written once, renamed into place and
     never modified in place; io errors wrapped in `BlockFileIo { path, source }` naming the path that failed) and
     `pub fn view(&self) -> Result<BlockFileView<'_>, InteropError>` that is `decode_block(&self.map)`. The doc names
     `memmap2::Mmap` and `decode_block` as the primitives it composes. One-line why `memmap2` and not `proxima-storage`:
     `memmap2` is already in `std`, while `proxima-storage` sits behind its own non-default feature. One-line why this
     reader is library code and the writer is not: reading a file in the block file format is the format's own job and
     the cartridge loader reads such files from production code, while writing one belongs to whichever tier owns
     the directory and the crash rules.
- test: add `blockfile_read_maps_what_was_written` in the `tests` module of `block_file.rs` (it carries the lint allow
  from FT5.1; `encode_block`, `open` and `view` are called with `.expect(..)` naming what failed, the test returns no
  `Result`): `tempfile::tempdir()`, encode the fixture (`fixture_header`, `fixture_planes`), write it with
  `std::fs::write(&path, &bytes)`, `MappedBlockFile::open(&path)`; `view()` yields `header == fixture_header()` and
  every plane bit equal to the fixture through `plane_f32`; `open` of a missing path matches
  `Err(InteropError::BlockFileIo { .. })`; a file rewritten as its first 60 bytes (`std::fs::write(&path, &bytes[..60])`)
  gives `Err(InteropError::BlockFileMalformed { reason: "truncated layer table" })` from a fresh `open` then `view()`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_5 cargo nextest run -p proxima-model-interop --features std -E 'test(/blockfile_read_maps_what_was_written/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/error.rs`, `proxima-model-interop/src/block_file.rs`
- commit: `feat(interop): read block files through a memory map`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: use `proxima-storage`; add a second `unsafe` block; add a writer or any other file code to `src/`.
- gpu: none

### 5.4 write a block file atomically, as disk tier test support

- id: FT5.4
- needs: FT5.17, FT5.5
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `block_file.rs` (FT5.17): `encode_block`, `BlockFileHeader`, `BlockFileLayer`; the unit-test helpers there are private to that module, so this card writes its own small fixture;
  - `error.rs::BlockFileIo` (FT5.5): the variant the writer returns;
  - `tests/support/mod.rs`: the existing shared test support directory; the new file this card adds there is not declared in `mod.rs`;
  - `tests/arch_data_baseline.rs` (first lines): how a test file starts (`#![cfg(feature = "std")]`).
- change:
  1. `proxima-model-interop/tests/support/block_write.rs` (new, test code, written against the crate's public API only;
     it names `InteropError` through `super` so the same file compiles in an integration test and under the library's
     unit tests): `use super::InteropError;` and `pub fn write_block_file(directory: &std::path::Path, file_key: u64, encoded: &[u8]) -> Result<std::path::PathBuf, InteropError>`.
     The final path is `directory/{file_key:016x}.pxkv`. It writes `encoded` to `{final}.tmp` with `std::fs::write`,
     then `std::fs::rename` onto the final path (atomic on one volume, so a crash leaves no half file under the final
     name) and returns the final path. Every `io::Error` is wrapped in `InteropError::BlockFileIo { path, source }`
     naming the path that failed. The directory must already exist; the caller creates it once. Why this is test code
     and not library code: the file naming, the crash rule and the directory belong to the tier that owns them (the
     disk tier of the later card), not to the block file format.
  2. `proxima-model-interop/tests/block_file_write.rs` (new): `#![cfg(feature = "std")]`,
     `#![allow(clippy::unwrap_used, clippy::expect_used)]` (the workspace denies both lints and an integration test
     has no module to inherit an allow from), `use proxima_model_interop::block_file::{BlockFileHeader, BlockFileLayer, encode_block};`,
     `use proxima_model_interop::InteropError;`, `#[path = "support/block_write.rs"] mod block_write;`,
     `use block_write::write_block_file;`, and a helper `fn fixture_bytes() -> Vec<u8>`: encode with `encode_block`
     (`.expect("encode the fixture")`) the header `BlockFileHeader { descriptor_digest: [0x11; 16], content_key: 0x0102030405060708, base_position: 0, layers: vec![BlockFileLayer { rows: 2, k_even_row_bytes: 8, k_odd_row_bytes: 8, v_row_bytes: 4, ring_window: 0, ring_capacity: 0 }] }`
     over the planes `k_even` `[0.0, 0.5, 1.0, 1.5]`, `k_odd` `[100.0, 100.5, 101.0, 101.5]`, `v` `[200.0, 200.5]`.
- test: add `blockfile_write_creates_the_final_name_only` in `block_file_write.rs`: `tempfile::tempdir()`,
  `write_block_file(dir.path(), 0x0102030405060708, &bytes)` returns a path whose file name is `0102030405060708.pxkv`;
  `std::fs::read` of it equals `bytes`; the directory lists exactly 1 entry (no `.tmp` left);
  `write_block_file(&dir.path().join("absent"), 1, &bytes)` matches `Err(InteropError::BlockFileIo { ref path, .. })`
  with `path.ends_with("0000000000000001.pxkv.tmp")`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_4 cargo nextest run -p proxima-model-interop --features std --test block_file_write -E 'test(/blockfile_write_creates_the_final_name_only/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/support/block_write.rs`, `proxima-model-interop/tests/block_file_write.rs`
- commit: `test(interop): write block files atomically for the disk tier`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add anything under `src/`; declare `block_write` in `tests/support/mod.rs` (the capability matrix would see unused items); create directories inside the function.
- gpu: none

### 5.6 chained block content keys, as disk tier test support

- id: FT5.6
- needs: FT5.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/block_bloom.rs::content_hashes (~line 26)` and `::block_hash (~line 18)`: the existing per-block hash,
    deliberately independent of earlier blocks (chunk reuse needs that);
  - `tests/support/mod.rs`: the existing shared test support directory; the new file this card adds there is not declared in `mod.rs`;
  - `Cargo.toml (~lines 45, 339)`: `xxhash-rust` with `xxh3` is already in `std`, so a test file can use it.
- change:
  1. `proxima-model-interop/tests/support/block_keys.rs` (new, test code): `pub fn chained_keys(ids: &[u32], block_tokens: usize, seed: u64) -> Vec<u64>`.
     `key[0]` is `xxhash_rust::xxh3::xxh3_64_with_seed` of the little-endian bytes of block 0 under `seed`; `key[i]` is `xxh3_64_with_seed`
     of `key[i-1].to_le_bytes()` followed by the little-endian bytes of block `i`, under `seed`; whole blocks only;
     `block_tokens == 0` gives an empty vector. One allocation per call, never per token. One-line why a second key
     beside `content_hashes`: a hit must prove the whole prefix equal, so the key depends on every earlier block,
     while `content_hashes` stays position independent for chunk shifting. The seed is where the producing
     configuration enters the key: callers pass a digest of the cache key so rows computed under another
     configuration never share a key. Why this is test code: the key scheme is the disk tier's file naming policy,
     not the block file format.
  2. `proxima-model-interop/tests/block_file_keys.rs` (new): `#![cfg(feature = "std")]`,
     `#![allow(clippy::unwrap_used, clippy::expect_used)]`, `#[path = "support/block_keys.rs"] mod block_keys;`,
     `use block_keys::chained_keys;`.
- test: add in `block_file_keys.rs`:
  - `blockfile_chained_key_depends_on_every_earlier_block`: ids `0..16`, `block_tokens = 4`, seed 0 gives 4 keys
    `keys`; with `ids[0]` changed to 999 all 4 keys differ from `keys` at their index; with `ids[5]` changed to 999
    key 0 is equal and keys 1, 2, 3 all differ; 7 ids with `block_tokens = 4` gives length 1; `block_tokens = 0`
    gives an empty vector; 3 ids with `block_tokens = 4` gives an empty vector;
  - `blockfile_chained_key_is_bound_to_the_seed`: the same ids under seeds 0 and 1 differ at every index; the same
    seed twice is equal.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_6 cargo nextest run -p proxima-model-interop --features std --test block_file_keys -E 'test(/blockfile_chained_key/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/support/block_keys.rs`, `proxima-model-interop/tests/block_file_keys.rs`
- commit: `test(interop): chain block content keys for the disk tier`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add anything under `src/`; change `content_hashes`; declare `block_keys` in `tests/support/mod.rs`.
- gpu: none

### 5.7 evict by an ordered rule list

- id: FT5.7
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prompt_cache.rs::PromptCache::eviction_victim (~line 819)`: today's rule, unused follow-up branch
    first and then the lowest stamp; the body moves out of the method;
  - `generate/prompt_cache.rs::PromptCache::index_ids (~line 786)` and `::store (~line 942)`: the two call sites of
    `eviction_victim` (~lines 791 and 966); they keep calling the method;
  - `generate/mod.rs (~line 265)` and `lib.rs (~line 115)`: the two `pub use` lists that gain `EvictionRule`;
  - `generate/prompt_cache.rs::tests::unused_branches_are_evicted_before_entries_a_request_produced (~line 1636)`
    and `branch_entry (~line 1586)`: the test style and helper to reuse.
- change:
  1. `proxima-model-interop/src/generate/prompt_cache.rs`: add `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum EvictionRule { Branch, Oldest }`
     (docs: `Branch` selects the first entry that is an unused follow-up branch; `Oldest` selects the entry with the
     lowest stamp and always selects one when any entry exists) and a private
     `fn rule_victim(candidates: impl Iterator<Item = (u64, bool)> + Clone, rules: &[EvictionRule]) -> Option<u64>`
     whose candidates are `(stamp, is_branch)` in ascending stamp order:
     `rules.iter().find_map(|rule| match rule { Branch => candidates.clone().find(|&(_, is_branch)| is_branch), Oldest => candidates.clone().next() }).map(|(stamp, _)| stamp)`.
  2. same file: `PromptCache` gains `eviction_rules: Vec<EvictionRule>`, initialised in `new()` to
     `vec![EvictionRule::Branch, EvictionRule::Oldest]` (this reproduces today's rule line for line). The body of
     `PromptCache::eviction_victim` becomes `rule_victim(self.entries.iter().map(|(stamp, entry)| (*stamp, entry.branch_base.is_some())), &self.eviction_rules)`.
  3. `proxima-model-interop/src/generate/mod.rs` (~line 265) add `EvictionRule` to
     `pub use prompt_cache::{CachePath, CacheReport, MissReason};` and `proxima-model-interop/src/lib.rs` (~line 115)
     add `EvictionRule` to the `pub use generate::{ .. }` list.
- test: record `N` first with `cargo nextest run -p proxima-model-interop --features std -E 'test(/generate::prompt_cache::tests::/)'`. Then add in
  `prompt_cache.rs` tests `eviction_rules_default_reproduces_branch_then_oldest`: candidates `[(3, false), (5, true), (7, true), (9, false)]`
  with `[Branch, Oldest]` give `Some(5)`; with `[Oldest]` give `Some(3)`; `[(3, false), (9, false)]` with
  `[Branch, Oldest]` give `Some(3)`; an empty iterator gives `None`; `[(3, false), (9, false)]` with `[Branch]`
  gives `None` (why the setter in the next rule card refuses a list that does not end with `Oldest`). The existing
  `unused_branches_are_evicted_before_entries_a_request_produced` and
  `the_least_recently_used_entry_is_evicted_past_max_entries` run unchanged and prove the default list is today's rule.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_7 cargo nextest run -p proxima-model-interop --features std -E 'test(/eviction_rules_/)'`; then the `generate::prompt_cache::tests::` filter again.
- expect: `1 passed`; then `N + 1 passed` from the `generate::prompt_cache::tests::` filter (`N` recorded before the edit).
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`, `proxima-model-interop/src/generate/mod.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): choose eviction victims from an ordered rule list`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a setter (next card); add a rule that reads a field `CacheEntry` lacks; add a config or serde derive; add a trait; change which entry the default evicts.
- gpu: none

### 5.20 set the eviction rule list on a loaded model

- id: FT5.20
- needs: FT5.7
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prewarm.rs::LoadedModel::set_prewarm_suffix (~line 140)`: the precedent for a list-valued setter on
    `LoadedModel` that locks `prompt_cache`;
  - `generate/prompt_cache.rs::LoadedModel::prompt_cache_bytes (~line 1007)`: where the new method sits, and the
    lock call shape (`self.prompt_cache.lock()`, a guard, no `Result`);
  - `generate/prompt_cache.rs::EvictionRule` and `PromptCache::eviction_rules` (FT5.7);
  - `error.rs::InteropError::UnsupportedServingConfig`: the refusal variant.
- change:
  1. `proxima-model-interop/src/generate/prompt_cache.rs`: add `pub(super) fn set_eviction_rules(&mut self, rules: &[EvictionRule]) -> Result<(), InteropError>`
     on `PromptCache`: refuses with
     `InteropError::UnsupportedServingConfig("eviction rules must end with the oldest rule so a full cache always has a victim".into())`
     unless `rules.last() == Some(&EvictionRule::Oldest)`; otherwise stores the list. In `impl LoadedModel<'_>` add
     `pub fn set_eviction_rules(&self, rules: &[EvictionRule]) -> Result<(), InteropError>` locking `self.prompt_cache`
     the way `prompt_cache_bytes` does and calling the setter; doc it as a setter and not a `PromptCacheConfig` field
     because that config is `Copy`, and point at `PromptCache::eviction_victim`'s default list.
- test: record `N` first as in the previous rule card. Then add in `prompt_cache.rs` tests (names start `eviction_rules_`):
  - `eviction_rules_setter_requires_oldest_last`: on a fresh `PromptCache`, `set_eviction_rules(&[])` and
    `set_eviction_rules(&[Branch])` match `Err(InteropError::UnsupportedServingConfig(_))`;
    `set_eviction_rules(&[Oldest])` and `&[Branch, Oldest]` are `Ok(())`;
  - `eviction_rules_choose_which_entry_a_full_cache_gives_up`: `max_entries: 3` over `enabled_config()`; store
    `state_with_ids(&[1, 1, 1])`, `branch_entry(&[1, 1, 1, 5], 3)`, `state_with_ids(&[2, 2, 2])`; with the default rules
    storing `[3, 3, 3]` evicts the branch, so `take_best` of `[1, 1, 1, 5, 9]` has `lcp == 3`; on a second cache
    with `set_eviction_rules(&[EvictionRule::Oldest])` and the same stores, `take_best` of `[1, 1, 1, 5, 9]` has
    `lcp == 4` (the branch survives, the base was the oldest and is gone).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_20 cargo nextest run -p proxima-model-interop --features std -E 'test(/eviction_rules_setter_|eviction_rules_choose_/)'`; then the `generate::prompt_cache::tests::` filter again.
- expect: `2 passed`; then `N + 2 passed` from the `generate::prompt_cache::tests::` filter (`N` recorded before the edit).
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`
- commit: `feat(interop): set the eviction rule list on a loaded model`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a config field or a serde derive; accept a list that does not end with `Oldest`.
- gpu: none

### 5.8 encode a prefix state into block file bytes

- id: FT5.8
- needs: FT5.1, FT5.17, FT5.2, FT5.18
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/residency_caches.rs::PrefixState (~line 707)`, `::LayerCacheState (~line 644)` and `::LayerCache (~line 112)`:
    the state and the three layer shapes; a full layer holds every position, a ring layer holds its allocated ring
    rows, a shared-KV layer holds nothing;
  - `generate/kv_ring.rs::KvRing (~line 33)` and `LayerCache::ring (~line 91)`: the ring geometry and constructor;
  - `generate/ring_checkpoint.rs::RingCheckpoint::capture (~line 64)`: it refuses dense-attention and recurrent
    layers, which this card also refuses;
  - `block_file.rs` (FT5.1, FT5.17): `encode_block`, `BlockFileHeader`, `BlockFileLayer`.
- change:
  1. `proxima-model-interop/src/generate/prefix_state_file.rs` (new): `impl PrefixState { pub fn to_block_file(&self, descriptor_digest: [u8; 16], content_key: u64, out: &mut Vec<u8>) -> Result<(), InteropError> }`.
     It builds a `BlockFileHeader` (`base_position` 0) with one `BlockFileLayer` per layer and the flat plane list,
     then calls `encode_block`. Per layer:
     - `SharedFromLayer`: the default (all zero) record and three empty planes;
     - `Attention` with `ring == None` (a full layer): `rows = cached_len`; row bytes are `4 * plane_len / cached_len`
       for `k_even`, `k_odd` and `v` respectively; planes are the three vectors as they are, zero copy;
     - `Attention` with a ring: `rows = k_even.len() / ring.even_odd_row`; row bytes `4 * even_odd_row`, `4 * even_odd_row`,
       `4 * v_row`; `ring_window` and `ring_capacity` from the geometry; planes are the whole ring vectors as they are
       (every ring slot, including the slack rows a rewind may still read, so the restore reproduces the ring exactly);
     - any `u32` conversion that fails refuses instead of truncating.
     Refusals, all `InteropError::BlockFileMalformed { reason }`, exact strings: `"nothing cached"` when `cached_len == 0`;
     `"layer kind has no row planes"` for `DenseAttention` and `Ssm`; `"ring is displaced"` when a ring's
     `write_offset != 0`; `"layer holds no rows"` when an `Attention` layer's `k_even` is empty;
     `"layer rows disagree with the cached length"` when a full layer's plane length is not a multiple of
     `cached_len` or `k_odd` and `k_even` differ in length; `"layer rows disagree with the ring geometry"` when a
     ring layer's plane lengths are not whole multiples of its row widths or differ between `k_even` and `k_odd`;
     `"layer dimension exceeds u32"` for a failed conversion. Public, so it is the crate's API for spilling a
     state (the doc names `encode_block` as the primitive it composes and says why a wrapper exists: the planes
     are `pub(super)` fields no outside caller can read).
  2. `proxima-model-interop/src/generate/mod.rs`: declare `mod prefix_state_file;` after `mod block_bloom;` (~line 222).
- test: declare the unit-test module at the end of `prefix_state_file.rs`:
  `#[cfg(test)] #[allow(clippy::unwrap_used, clippy::expect_used)] mod tests { use super::*; .. }` (the workspace denies
  `unwrap_used` and `expect_used` and `clippy.toml` grants no test exemption; the allow is the one the other in-crate
  test modules carry, for example `generate/kv_ring.rs` ~line 295). Every `to_block_file`, `encode_block` and
  `decode_block` call that is meant to succeed is `.expect("<what failed>")`; no test returns `Result`. Tests (names
  start `prefix_state_file_`). Helper `three_layer_state()`: ids
  `[2, 818, 5279, 529]`, `cached_len = 4`; layer 0 a full `LayerCache` with widths 2, 2, 1 and values `ramp` style
  (`k_even` `0.0, 0.5, ..`, `k_odd` from 100.0, `v` from 200.0); layer 1 `LayerCache::ring(KvRing::new(2, 1, 2, 1, 0), 4)`
  then `append_at(0, ..)` of four rows of the same shape (the ring ends with 3 rows allocated); layer 2
  `LayerCacheState::SharedFromLayer`.
  - `prefix_state_file_encodes_full_ring_and_shared_layers`: `to_block_file([0x33; 16], 0xABCD, &mut out)` is `Ok`;
    `decode_block(&out)` has `header.layers == [BlockFileLayer { rows: 4, k_even_row_bytes: 8, k_odd_row_bytes: 8, v_row_bytes: 4, ring_window: 0, ring_capacity: 0 }, BlockFileLayer { rows: 3, k_even_row_bytes: 8, k_odd_row_bytes: 8, v_row_bytes: 4, ring_window: 2, ring_capacity: 3 }, BlockFileLayer::default()]`,
    `descriptor_digest == [0x33; 16]`, `content_key == 0xABCD`, `base_position == 0`; `plane_f32(0, 0)` equals layer 0's
    `k_even` bit for bit, `plane_f32(1, 2)` equals the ring layer's `v` vector bit for bit (slot order, not
    chronological), `plane_f32(2, 1)` is `Some(())` with an empty output;
  - `prefix_state_file_refuses_layers_without_row_planes`: a state whose layers are
    `[Ssm(SsmLayerCache::new(2, 2))]` gives `BlockFileMalformed { reason: "layer kind has no row planes" }`;
  - `prefix_state_file_refuses_a_displaced_ring`: the ring built with `KvRing::new(2, 1, 2, 1, 1)` gives `"ring is displaced"`;
  - `prefix_state_file_refuses_an_empty_state`: `cached_len = 0` gives `"nothing cached"`.
  Assert refusals with `matches!` (`InteropError` has no `PartialEq`).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_8 cargo nextest run -p proxima-model-interop --features std -E 'test(/prefix_state_file_/)'`
- expect: `4 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prefix_state_file.rs`, `proxima-model-interop/src/generate/mod.rs`
- commit: `feat(interop): encode a prefix state into block file bytes`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a restore (next card); encode recurrent or dense-attention layers; clone the planes.
- gpu: none

### 5.9 restore a prefix state from block file bytes

- id: FT5.9
- needs: FT5.8
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prefix_state_file.rs` (FT5.8): the encoder and the `three_layer_state()` helper;
  - `generate/kv_ring.rs::KvRing::new (~line 45)`: `(window, slack, even_odd_row, v_row, write_offset)`; capacity is
    `window + slack`;
  - `block_file.rs::BlockFileView::plane_f32` (FT5.18);
  - `generate/residency_caches.rs::LayerCache (~line 112)`: the fields built directly here.
- change:
  1. `proxima-model-interop/src/generate/prefix_state_file.rs`: `impl PrefixState { pub fn from_block_file(ids: Vec<u32>, cached_len: usize, view: &BlockFileView<'_>) -> Result<PrefixState, InteropError> }`.
     Refusals, `BlockFileMalformed { reason }`, exact strings: `"cached length exceeds ids"` when
     `cached_len > ids.len()`; `"plane is not whole f32 values"` when `plane_f32` returns `None` for a layer that has
     rows; `"ring window exceeds ring capacity"` when `ring_window > ring_capacity`. Per layer record: `rows == 0`
     gives `LayerCacheState::SharedFromLayer`; otherwise the three planes are read through `plane_f32` into fresh
     vectors and the layer is `LayerCacheState::Attention(LayerCache { k_even, k_odd, v, ring })` with `ring` `None`
     when `ring_window == 0` and `Some(KvRing::new(window, capacity - window, k_even_row_bytes / 4, v_row_bytes / 4, 0))`
     otherwise. The result is `PrefixState { ids, layer_caches, cached_len }`.
- test: add in the `tests` module of `prefix_state_file.rs` (declared in FT5.8 with the lint allow; every `to_block_file`,
  `encode_block`, `decode_block` and `from_block_file` call meant to succeed is `.expect("<what failed>")`; no test
  returns `Result`):
  - `prefix_state_file_round_trips_every_layer_bit_for_bit`: encode `three_layer_state()`, decode, restore with the
    state's own ids and `cached_len`; layer 0 vectors equal the source bit for bit; layer 1 vectors equal the
    source bit for bit and `ring_geometry() == Some(&KvRing::new(2, 1, 2, 1, 0))`; layer 2 is `SharedFromLayer`;
    `cached_len` and `ids` equal; encoding the restored state with the same digest and key gives bytes identical
    to the first encoding;
  - `prefix_state_file_restore_refuses_a_ring_window_over_its_capacity`: a header built by hand with
    `ring_window: 4, ring_capacity: 3` and valid planes through `encode_block` gives `"ring window exceeds ring capacity"`;
  - `prefix_state_file_restore_refuses_cached_length_over_ids`: `cached_len = 5` with 4 ids gives `"cached length exceeds ids"`;
  - `prefix_state_file_restore_refuses_a_plane_that_is_not_whole_floats`: encode a one-layer block with `rows: 1` and
    4-byte rows, then patch the layer table's `k_even_row_bytes` to 6 and append 2 zero bytes so `decode_block`
    still accepts the length; the restore gives `"plane is not whole f32 values"`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_9 cargo nextest run -p proxima-model-interop --features std -E 'test(/prefix_state_file_/)'`
- expect: `8 passed` (4 from the encode card, 4 here)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prefix_state_file.rs`
- commit: `feat(interop): restore a prefix state from block file bytes`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: reconstruct ring checkpoints; add a model parameter.
- gpu: none

### 5.10 hand evicted prompt cache entries to a cold tier slot

- id: FT5.10
- needs: FT5.7
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prompt_cache.rs::PromptCache::store (~line 942)`: the budget loop `while self.entries.len() > max_entries || self.stored_bytes() > budget`;
    `::drop_entry (~line 805)` and `::eviction_victim (~line 819)` (FT5.7);
  - `generate/prompt_cache.rs::CacheEntry (~line 358)` and `::new (~line 385)`: the one constructor;
  - `generate/prompt_cache.rs::PromptCache::best_candidate (~line 695)`: the per-entry filter;
  - `generate/load_model.rs::LoadedModel::prompt_cache (~line 1008)`: `pub(super) prompt_cache: Mutex<super::PromptCache>`.
- change:
  1. `proxima-model-interop/src/generate/prompt_cache.rs` (all items below; line 37 becomes
     `use proxima_telemetry::{debug, error, warn};`, because the eviction path logs a warning and only `debug` and
     `error` are imported there):
     - `pub trait ColdTier: Send + Sync` with two required methods and one provided method:
       `fn demote(&self, stamp: u64, ids: &[u32], state: &PrefixState) -> Result<u64, InteropError>` (the tier takes a
       copy of an entry the cache is evicting and returns how many bytes it now holds for it; `Err` means it could not
       take it);
       `fn discard(&self, stamp: u64) -> Result<(), InteropError>` (the entry left the cache; the tier frees its copy);
       and `fn install(self: Arc<Self>, model: &LoadedModel<'_>) where Self: Sized + 'static`,
       whose body is `model.prompt_cache.lock().set_cold(self)`. Doc, English: names `PromptCache::store` as the caller
       of `demote`, names `PrefixState::to_block_file` and `PrefixState::from_block_file` as the primitives a tier
       composes with its own file code (plain backticks, not intra-doc links, so the doc does not depend on cards that
       may not have landed), says why the slot is a trait and not a pipe (a demote runs inside the synchronous cache
       store under the cache lock, and a pipe's call is asynchronous), says `install` is a method here and not a
       `LoadedModel` setter because a tier and the way to install it travel together and not a `PromptCacheConfig`
       field because that config is `Copy`, and says a tier must not call back into the cache because `demote` and
       `discard` run under its lock;
     - `struct ColdSlot { tier: Arc<dyn ColdTier> }` (private) with a one-line why for `Arc<dyn ColdTier>`: an open set
       of tiers behind one slot on the non-generic model type;
     - `struct ColdHold { tier: Arc<dyn ColdTier>, stamp: u64 }` with `impl Drop` calling
       `self.tier.discard(self.stamp)` and, on `Err(error)`, `error!(cache_stamp = self.stamp, tier_error = %error, "cold prompt cache entry could not be removed from its tier")`;
       owning the handle by value means every way an entry leaves the cache (drop, clear, trim, restore, index rebuild)
       frees the tier's copy with no explicit cleanup;
     - `ColdSlot::demote(&self, stamp: u64, entry: &mut CacheEntry) -> Result<(), InteropError>`: `let bytes = self.tier.demote(stamp, &entry.state.ids, &entry.state)?;`
       then `debug!(cache_stamp = stamp, tier_bytes = bytes, "prompt cache entry moved to its cold tier")`, then, only
       after the tier took it, release the rows: `entry.state.layer_caches = Vec::new()`, `entry.checkpoints = Vec::new()`,
       `entry.moved = Vec::new()`, `entry.prewarmed = None`, `entry.branch_base = None`, and set
       `entry.cold = Some(ColdHold { tier: Arc::clone(&self.tier), stamp })`. The ids and `cached_len` stay, so the
       prefix index keeps resolving the entry. On `Err` the entry is untouched;
     - `CacheEntry` gains `cold: Option<ColdHold>` (private; `None` in `new`) and
       `const fn is_cold(&self) -> bool { self.cold.is_some() }`;
     - `PromptCache` gains `cold: Option<ColdSlot>` (private; `None` in `new()`) and
       `pub(super) fn set_cold(&mut self, tier: Arc<dyn ColdTier>)`, whose production caller is `ColdTier::install`;
     - `fn hot_count(&self) -> usize` (entries that are not cold). `store`'s loop condition uses `hot_count()` in place
       of `entries.len()`, evicts through `fn evict(&mut self, stamp: u64)`, and `store` returns `Some(self.hot_count())`;
     - `eviction_victim` considers only entries that are not cold (its `rule_victim` candidates come from the filtered
       iterator);
     - `fn evict(&mut self, stamp: u64)`: with a tier and a present entry, call `slot.demote(stamp, entry)` (read the two
       fields `self.cold` and `self.entries` directly so the borrows stay disjoint); `Ok` keeps the entry (now cold);
       `Err(error)` logs
       `warn!(cache_stamp = stamp, tier_error = %error, "prompt cache entry dropped, the cold tier could not take it")`
       and falls back to `drop_entry(stamp)`. Without a tier `evict` is `drop_entry`, which is today's behaviour;
     - `best_candidate`'s per-entry filter also requires `!entry.is_cold()`, so a cold entry is never offered to the
       in-memory path;
     - test-only accessors under `#[cfg(test)]`, beside `index_byte_len`: `pub(super) fn held(&self) -> Vec<(u64, bool)>`
       (stamp and cold flag of every entry, ascending by stamp) and `pub(super) fn entry(&self, stamp: u64) -> Option<&CacheEntry>`.
  2. `proxima-model-interop/src/generate/mod.rs` (~line 265): add `ColdTier` to `pub use prompt_cache::{..}`.
  3. `proxima-model-interop/src/lib.rs` (~line 115): add `ColdTier` to the `pub use generate::{ .. }` list.
  Not in this card: the cold entry limits (the next two cards). Until they land the cold entries are bounded only by
  the prefix index capacity, which this card leaves as it is.
- test: record `N` first with `cargo nextest run -p proxima-model-interop --features std -E 'test(/generate::prompt_cache::tests::/)'`. Then add in
  `prompt_cache.rs` tests (names start `tier_demote_`; that module carries the lint allow already, `.expect(..)` is fine). Helpers: consts `TRACE_A`, `TRACE_B`, `TRACE_C` (the
  entries A to C of the worked trace); `RecordingTier { demoted: Mutex<Vec<u64>>, discarded: Mutex<Vec<u64>>, refuses: bool }`
  implementing `ColdTier` (`demote` returns `Err(InteropError::UnsupportedServingConfig("recording tier is full".into()))` when
  `refuses`, otherwise records the stamp and returns `Ok(0)`; `discard` records the stamp), with
  `Mutex` the in-scope `proxima_primitives::sync::blocking::Mutex`; and a function `cache_with_tier(tier: &Arc<RecordingTier>) -> PromptCache` building a cache with
  `cache.set_cold(Arc::clone(tier))` (the `Arc<RecordingTier>` coerces to the argument) that tests store into with
  `PromptCacheConfig { max_entries: 2, ..enabled_config() }`. Entries come from `state_with_ids`.
  - `tier_demote_evicted_entry_is_kept_cold_and_handed_to_the_tier`: store A, B, C: `held()` is
    `[(0, true), (1, false), (2, false)]`; entry 0 has `state.layer_caches.is_empty()`, `checkpoints.is_empty()` and
    `state.ids == TRACE_A`; `tier.demoted == [0]` and `tier.discarded` is empty; `take_best` of `TRACE_A + [9]` is a miss
    (`report.path == CachePath::Miss`, `reused_tokens == 0`) and entry 0 is still in `held()`;
  - `tier_demote_discards_a_cold_entry_when_it_leaves_the_cache`: after the stores above `clear()` leaves
    `tier.discarded == [0]`; a second cache with a second tier and one cold entry, dropped with `drop(cache)`, leaves that
    tier's `discarded == [0]`;
  - `tier_demote_drops_an_entry_the_tier_refuses`: `refuses: true`, store A, B, C: `held()` is `[(1, false), (2, false)]`,
    `tier.demoted` and `tier.discarded` are empty.
  The existing `the_least_recently_used_entry_is_evicted_past_max_entries` runs unchanged with no tier set, which is the
  proof that the default is today's cache.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_10 cargo nextest run -p proxima-model-interop --features std --lib -E 'test(/tier_demote_/)'`; then the `generate::prompt_cache::tests::` filter again.
- expect: `3 passed`; then `N + 3 passed` from the `generate::prompt_cache::tests::` filter (`N` recorded before the edit).
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`, `proxima-model-interop/src/generate/mod.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): hand evicted prompt cache entries to a cold tier slot`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: write file code in `src/` (the disk tier is test code, FT5.23); read a tier in this card (FT5.11); add a codec; add a limit on the cold entries (FT5.24, FT5.25); change behaviour when no tier is installed; add `Drop` to `PromptCache`.
- gpu: none

### 5.24 size the prefix index for cold entries and bound the cold entries by count

- id: FT5.24
- needs: FT5.10
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prompt_cache.rs::PromptCache::reconfigure (~line 919)`: the entry capacity handed to the prefix index;
    `::index_ids (~line 786)`: the eviction taken when the prefix index is full;
  - `generate/prompt_cache.rs::ColdSlot`, `ColdHold`, `PromptCache::set_cold`, `PromptCache::evict` and `ColdTier::install` (FT5.10);
  - `generate/prompt_cache.rs::tests::RecordingTier` and `cache_with_tier` (FT5.10): the test tier and cache builder this card extends.
- change:
  1. `proxima-model-interop/src/generate/prompt_cache.rs`:
     - `ColdSlot` gains `max_entries: usize`; `set_cold` becomes `set_cold(&mut self, tier: Arc<dyn ColdTier>, max_entries: usize)`;
       `ColdTier::install` becomes `install(self: Arc<Self>, model: &LoadedModel<'_>, max_entries: usize)` and its doc gains the
       sentence that `max_entries` bounds the cold entries the cache keeps (past it the oldest cold entry is discarded);
     - `reconfigure`: the entry capacity handed to the prefix index becomes `config.max_entries as usize` plus
       `self.cold.as_ref().map_or(0, |slot| slot.max_entries)`, because cold entries stay indexed;
     - `fn cold_victim(&self) -> Option<u64>` (the lowest stamp among cold entries, `BTreeMap` order) and `index_ids`'s full
       branch takes `self.cold_victim().or_else(|| self.eviction_victim())` and plainly drops it (a demoted entry would
       not free an index slot);
     - `fn trim_cold(&mut self)`: while the cold entries' count exceeds `slot.max_entries`, `drop_entry(cold_victim)`; `evict`
       calls it after a successful demote.
- test: record `N` first as in the previous card. Update `cache_with_tier` to `cache_with_tier(tier: &Arc<RecordingTier>, max_entries: usize) -> PromptCache`
  (the three tests of the previous card pass 6) and add consts `TRACE_D` and `TRACE_E` (entries D and E of the worked trace).
  Add in `prompt_cache.rs` tests (names start `tier_demote_`):
  - `tier_demote_trims_cold_entries_past_the_tier_limit`: tier limit 1, store A, B, C, D (stamps 0 to 3): `held()` is
    `[(1, true), (2, false), (3, false)]`, `tier.demoted == [0, 1]` and `tier.discarded == [0]`; a second cache with tier limit 6 after
    storing A, B, C has `cache.index.entry_capacity() == 8` (2 hot plus 6 cold; the default is 4);
  - `tier_demote_trims_cold_entries_when_the_hot_limit_shrinks`: tier limit 1; store A, B, C, D under `PromptCacheConfig { max_entries: 4, ..enabled_config() }`
    (nothing is evicted, the index holds 5), then store E under `max_entries: 2`: `held()` is `[(2, true), (3, false), (4, false)]`,
    `tier.demoted == [0, 1, 2]` and `tier.discarded == [0, 1]` (the index never fills, so only the trim keeps the cold entries at 1).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_24 cargo nextest run -p proxima-model-interop --features std --lib -E 'test(/tier_demote_/)'`; then the `generate::prompt_cache::tests::` filter again.
- expect: `5 passed` (3 from the previous card, 2 here); then `N + 2 passed` from the `generate::prompt_cache::tests::` filter (`N` recorded before the edit).
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`
- commit: `feat(interop): bound cold prompt cache entries by count`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a byte limit (next card); add a config field or a serde derive; change behaviour when no tier is installed.
- gpu: none

### 5.25 bound the cold entries by bytes

- id: FT5.25
- needs: FT5.24
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prompt_cache.rs::ColdSlot`, `ColdHold`, `ColdSlot::demote`, `PromptCache::trim_cold`, `PromptCache::set_cold` and `ColdTier::install` (FT5.10, FT5.24);
  - `generate/prompt_cache.rs::tests::RecordingTier` and `cache_with_tier` (FT5.10, FT5.24): the test tier and cache builder this card extends.
- change:
  1. `proxima-model-interop/src/generate/prompt_cache.rs`: `ColdSlot` gains `byte_budget: u64`; `ColdHold` gains `bytes: u64`, which
     `ColdSlot::demote` sets from the value the tier returned; `set_cold` becomes
     `set_cold(&mut self, tier: Arc<dyn ColdTier>, max_entries: usize, byte_budget: u64)`; `ColdTier::install` becomes
     `install(self: Arc<Self>, model: &LoadedModel<'_>, max_entries: usize, byte_budget: u64)` and its doc says `byte_budget` bounds the sum of
     the bytes `demote` reported (past it the oldest cold entry is discarded); `trim_cold` also loops while the cold entries'
     `ColdHold::bytes` sum exceeds `slot.byte_budget`.
- test: record `N` first as in the previous card. `RecordingTier` gains `bytes_per_entry: u64` (its `demote` returns `Ok(bytes_per_entry)`;
  the earlier tests build it with 0) and `cache_with_tier` gains `byte_budget: u64` (the earlier tests pass `1 << 30`). Add in `prompt_cache.rs` tests:
  - `tier_demote_trims_cold_entries_past_the_tier_byte_budget`: `bytes_per_entry: 100`, byte budget 150, tier limit 8, store
    A, B, C, D: `held()` is `[(1, true), (2, false), (3, false)]` and `tier.discarded == [0]` (two cold entries hold 200 bytes).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_25 cargo nextest run -p proxima-model-interop --features std --lib -E 'test(/tier_demote_/)'`; then the `generate::prompt_cache::tests::` filter again.
- expect: `6 passed` (5 from the earlier cards, 1 here); then `N + 1 passed` from the `generate::prompt_cache::tests::` filter (`N` recorded before the edit).
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`
- commit: `feat(interop): bound cold prompt cache entries by bytes`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a config field or a serde derive; change behaviour when no tier is installed.
- gpu: none

### 5.11 restore cold entries outside the cache lock

- id: FT5.11
- needs: FT5.25
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prompt_cache.rs::LoadedModel::prompt_cache_lookup (~line 1043)`: takes the cache lock, calls
    `take_best_shifting`, patches the report, releases the lock; the only production lookup;
  - `generate/prompt_cache.rs::PromptCache::best_candidate (~line 695)`: the trie walk with a per-entry closure;
  - `generate/prompt_cache.rs::ColdTier`, `ColdSlot`, `ColdHold` (FT5.10, FT5.24, FT5.25);
  - `generate/prompt_cache.rs::tests::RecordingTier` and `cache_with_tier` (FT5.10, FT5.24, FT5.25): the test tier this card gives a promote.
- change:
  1. `proxima-model-interop/src/generate/prompt_cache.rs`:
     - `ColdTier` gains the required method
       `fn promote(&self, stamp: u64, ids: &[u32], cached_len: usize) -> Result<PrefixState, InteropError>` (the tier
       rebuilds the entry it holds for `stamp` from the ids and cached length the cache kept; `Err` means it cannot);
     - `pub(super) struct ColdRead { pub(super) stamp: u64, ids: Vec<u32>, cached_len: usize, tier: Arc<dyn ColdTier> }`
       with `fn read(&self) -> Result<PrefixState, InteropError>` calling `self.tier.promote(self.stamp, &self.ids, self.cached_len)`.
       The ticket exists because the cache lock must not be held across the tier read;
     - `best_candidate` takes a `cold: bool` and filters entries with `entry.is_cold() == cold` (replacing FT5.10's
       `!entry.is_cold()`); its one existing caller passes `false`;
     - `PromptCache::cold_read(&self, prompt_ids: &[u32], key: &CacheKey, min_similarity_milli: u32) -> Option<ColdRead>`:
       `None` without a tier; otherwise the best cold candidate `(stamp, cold_lcp)` and the best in-memory
       candidate's `lcp` (0 when none); a ticket only when `cold_lcp` is strictly greater, with the ids cloned from the
       cold entry and the tier cloned from the slot;
     - `PromptCache::thaw(&mut self, stamp: u64, restored: Result<PrefixState, InteropError>, key: &CacheKey, config: &PromptCacheConfig) -> Result<bool, InteropError>`:
       `drop_entry(stamp)` first (the cold entry and the tier's copy leave the cache whatever the read returned); when
       that finds no entry (another request restored or trimmed it meanwhile) return `Ok(false)` and keep nothing; then
       `let state = restored?`, then `self.store(CacheEntry::new(state, *key), config)` (ignore its `Option`: an entry
       larger than the whole budget is simply not kept), then `Ok(true)`;
     - `pub(super) fn thaw_best_cold(cache: &Mutex<PromptCache>, prompt_ids: &[u32], key: &CacheKey, config: &PromptCacheConfig) -> Result<bool, InteropError>`
       (`Mutex` is the in-scope `proxima_primitives::sync::blocking::Mutex`; `lock()` returns the guard directly):
       lock, `cold_read(prompt_ids, key, config.min_similarity_milli)`, unlock; `Ok(false)` when there is no ticket; otherwise
       `ticket.read()` outside any lock; on `Err(error)` emit
       `error!(cache_stamp = ticket.stamp, tier_error = %error, "prompt cache tier entry could not be read back, dropped")`;
       lock again and return `thaw(ticket.stamp, restored, key, config)`;
     - `prompt_cache_lookup`: before `let mut cache = self.prompt_cache.lock();` call
       `let _restored = thaw_best_cold(&self.prompt_cache, prompt_ids, key, &config);` (the outcome is logged inside; the
       report label in a later card reads it).
- test: record `N` first as in the previous card. Update `RecordingTier` (FT5.10, FT5.25) with `fail_promote: bool` and a
  `Mutex<BTreeMap<u64, PrefixState>>` of the states it was given (`demote` stores `state.branch()`); its `promote` returns the
  stored state (`branch()` again) or, when `fail_promote`, `Err(InteropError::UnsupportedServingConfig("recording tier cannot read".into()))`.
  Add in `prompt_cache.rs` tests (names start `tier_restore_`), with the trace entries A to D from FT5.10 and FT5.24 and a `Mutex<PromptCache>`
  (no rewind happens on an extend, so `shared_widths()` is enough):
  - `tier_restore_cold_entry_is_served_after_reading_outside_the_lock`: store A, B, C (A cold, tier limit 2);
    `thaw_best_cold(&cache, A + [9], ..)` is `Ok(true)`; `tier.discarded == [0]` (A's cold copy is gone) and `tier.demoted == [0, 1]`
    (B was demoted to make room); `held()` is `[(1, true), (2, false), (3, false)]` (the restored A is stamp 3); then `take_best` of
    `A + [9]` returns `Some(entry)` with `entry.state.ids == A`, `report.path == CachePath::Extend` and `report.reused_tokens == 4`,
    and `held()` is `[(1, true), (2, false)]`;
  - `tier_restore_answers_false_when_no_cold_entry_matches`: an unrelated prompt `[1, 2, 3]` gives `Ok(false)` and `tier.discarded` is empty;
  - `tier_restore_drops_a_cold_entry_whose_tier_read_fails`: `fail_promote: true`; `thaw_best_cold` matches
    `Err(InteropError::UnsupportedServingConfig(_))`; entry 0 is no longer in `held()` and `tier.discarded == [0]`;
  - `tier_restore_keeps_nothing_when_another_request_restored_the_entry_first`: take the ticket with `cold_read`, drop entry 0
    with `drop_entry(0)`, then `thaw(0, Ok(state), ..)` is `Ok(false)` and `held()` has no new entry;
  - `tier_restore_prefers_the_longer_prefix_between_cold_and_hot`: store A, B, C (A cold), then store
    `[818, 5279, 99, 99]` (stamp 3, hot): the prompt `A + [9]` shares 4 ids with cold A and 2 with the hot entry, so
    `thaw_best_cold` is `Ok(true)`; in a second cache holding cold A and a hot `[818, 5279, 529, 7001, 9, 8]` the
    same prompt gives `Ok(false)` (the hot prefix is longer).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_11 cargo nextest run -p proxima-model-interop --features std --lib -E 'test(/tier_restore_/)'`; then the `generate::prompt_cache::tests::` filter again.
- expect: `5 passed`; then `N + 5 passed` from the `generate::prompt_cache::tests::` filter (`N` recorded before the edit).
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`
- commit: `feat(interop): restore cold prompt cache entries outside the lock`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: hold the cache lock while the tier reads; label the report (next cards); restore from a prewarm or a follow-up draft lookup (they stay in-memory only); change a request's result when no tier is installed.
- gpu: none

### 5.21 name the cold tier in the cache report

- id: FT5.21
- needs: FT5.11
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prompt_cache.rs::CachePath (~line 70)` and `::as_str (~line 92)`: the enum that gains a variant and its label;
  - `generate/prompt_cache.rs::LoadedModel::prompt_cache_lookup (~line 1043)`: where `_restored` was bound (FT5.11) and where
    `report.prewarm_wait = waited` is set;
  - `generate/prompt_cache.rs::CacheReport (~line 160)`: the fields the label leaves alone.
- change:
  1. `proxima-model-interop/src/generate/prompt_cache.rs`: `CachePath` gains `Tier` (label `"tier"`; the entry that served
     the request came back from a cold tier, and `lcp` and `reused_tokens` say how much of it was used); a free
     `fn relabel_after_tier(report: CacheReport, restored: &Result<bool, InteropError>) -> CacheReport`: `Ok(true)` with
     `report.miss == None` returns `CacheReport { path: CachePath::Tier, ..report }`; every other combination returns the
     report unchanged; in `prompt_cache_lookup` rename `_restored` to `restored` and, after `take_best_shifting` and before
     `report.prewarm_wait = waited`, apply `report = relabel_after_tier(report, &restored);` so the `last_report` it stores and
     the telemetry fields (`cache_path`) carry the tier outcome.
- test: add in `prompt_cache.rs` tests (names start `tier_label_`), the hit report built as
  `CacheReport { path: CachePath::Extend, miss: None, reused_tokens: 4, ..CacheReport::miss(5, MissReason::Empty) }`:
  - `tier_label_names_the_tier_for_a_restored_hit`: `relabel_after_tier(hit, &Ok(true)).path == CachePath::Tier` and
    `CachePath::Tier.as_str() == "tier"`;
  - `tier_label_leaves_other_outcomes_alone`: `&Ok(false)` leaves the hit `Extend`; a miss report with `&Ok(true)` keeps
    `path == CachePath::Miss` and its miss reason; the hit with `&Err(InteropError::UnsupportedServingConfig("x".into()))` is unchanged.
  The label reaching a real request is asserted by the oracle cards (`report.path == CachePath::Tier`).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_21 cargo nextest run -p proxima-model-interop --features std --lib -E 'test(/tier_label_/)'`; then the `generate::prompt_cache::tests::` filter (`N` recorded before the edit).
- expect: `2 passed`; then `N + 2 passed` from the `generate::prompt_cache::tests::` filter.
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`
- commit: `feat(interop): name the cold tier in the prompt cache report`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a miss reason (next card); relabel a report whose lookup missed.
- gpu: none

### 5.22 name a failed cold tier read as a miss reason

- id: FT5.22
- needs: FT5.21
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prompt_cache.rs::MissReason (~line 105)` and `::as_str (~line 143)`: the enum that gains a variant and its label;
  - `generate/prompt_cache.rs::relabel_after_tier` (FT5.21): the function that gains an arm;
  - `generate/prompt_cache.rs::thaw_best_cold` (FT5.11): where the read error is logged and returned.
- change:
  1. `proxima-model-interop/src/generate/prompt_cache.rs`: `MissReason` gains `TierRestoreFailed` (label
     `"tier_restore_failed"`; the best match was a cold entry whose tier read failed, so the request prefilled in full);
     `relabel_after_tier` gains the arm `Err(_)` with `report.miss.is_some()` returning `CacheReport { miss: Some(MissReason::TierRestoreFailed), ..report }`;
     every other combination stays as it was.
- test: add in `prompt_cache.rs` tests (names start `tier_failure_`), with the hit and miss reports of the previous card:
  - `tier_failure_restore_error_names_the_miss`: a miss report with `&Err(InteropError::UnsupportedServingConfig("x".into()))`
    has `miss == Some(MissReason::TierRestoreFailed)` and `path == CachePath::Miss`; `MissReason::TierRestoreFailed.as_str() == "tier_restore_failed"`;
  - `tier_failure_restore_error_leaves_a_hit_alone`: the hit report with the same `Err` is unchanged.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_22 cargo nextest run -p proxima-model-interop --features std --lib -E 'test(/tier_failure_/)'`; then the `generate::prompt_cache::tests::` filter (`N` recorded before the edit).
- expect: `2 passed`; then `N + 2 passed` from the `generate::prompt_cache::tests::` filter.
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`
- commit: `feat(interop): name a failed cold tier read as a miss reason`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change what a failed read does to the cache (the entry is dropped and the request prefills, as before this card).
- gpu: none

### 5.23 a disk tier written against the cold tier slot

- id: FT5.23
- needs: FT5.3, FT5.4, FT5.5, FT5.6, FT5.9, FT5.11
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `generate/prompt_cache.rs::ColdTier` and `PromptCache::set_cold` (FT5.10, FT5.24, FT5.25, FT5.11): the slot, its three methods and the installer the tests use (no loaded model exists in a unit test);
  - `block_file.rs::MappedBlockFile` and `BlockFileView::require_digest` (FT5.5, FT5.3), `tests/support/block_write.rs::write_block_file` (FT5.4), `tests/support/block_keys.rs::chained_keys` (FT5.6) and
    `generate/prefix_state_file.rs::PrefixState::{to_block_file, from_block_file}` (FT5.8, FT5.9);
  - `generate/block_index_real_model_tests.rs::entry_of (~line 72)`: the precedent for building a `CacheEntry` and a `CacheKey` in an
    in-crate test (`CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0)`);
  - `tests/support/mod.rs`: the existing shared test support directory this card adds a file to (the new file is not declared there).
- change:
  1. `proxima-model-interop/tests/support/disk_tier.rs` (new): written against the crate's public API only, so an outside
     consumer could write the same. It names its imports through `super`, so the same file compiles under the library's
     unit tests and under an integration test: `use super::{ColdTier, InteropError, MappedBlockFile, PrefixState, chained_keys, write_block_file};`
     plus `std::path::{Path, PathBuf}`. Contents:
     - `pub struct DiskTier { directory: PathBuf, descriptor_digest: [u8; 16] }` with
       `pub fn open(directory: &Path, descriptor_digest: [u8; 16]) -> Result<Self, InteropError>` (`std::fs::create_dir_all`,
       error wrapped in `InteropError::BlockFileIo { path, source }`) and a private `fn path_of(&self, stamp: u64) -> PathBuf`
       (`directory/{stamp:016x}.pxkv`, the name `write_block_file` gives file key `stamp`);
     - `impl ColdTier for DiskTier`: `demote` takes the content key from `chained_keys(ids, ids.len(), 0)` (its first element; an entry
       with no ids refuses with `BlockFileMalformed { reason: "entry has no ids" }`), encodes with
       `state.to_block_file(self.descriptor_digest, content_key, &mut bytes)?`, writes with `write_block_file(&self.directory, stamp, &bytes)?`,
       and returns the byte count; `promote` opens `path_of(stamp)` with `MappedBlockFile::open`, takes `view()`, calls
       `view.require_digest(self.descriptor_digest)?` and returns `PrefixState::from_block_file(ids.to_vec(), cached_len, &view)`;
       `discard` removes `path_of(stamp)` with `std::fs::remove_file`, treats `NotFound` as `Ok(())` and wraps any other error in `BlockFileIo`;
     - `pub fn spilled_names(directory: &Path) -> Vec<String>` (the sorted file names in the directory).
  2. `proxima-model-interop/src/generate/disk_tier_tests.rs` (new, `#![allow(clippy::expect_used)]`): imports
     `use super::*;`, `use crate::block_file::MappedBlockFile;`, `use crate::{ColdTier, InteropError, PrefixState};`,
     `use super::prompt_cache::{CacheEntry, PromptCache};` and declares the three test support files the disk tier reads its names from through `super`:
     `#[path = "../../tests/support/block_write.rs"] mod block_write;`, `#[path = "../../tests/support/block_keys.rs"] mod block_keys;` and
     `#[path = "../../tests/support/disk_tier.rs"] mod disk_tier;` (a path attribute inside a file that is not a `mod.rs` is relative to the
     directory holding it, here `src/generate/`), then `use block_write::write_block_file;` and `use block_keys::chained_keys;`
    . Helpers: `base_key()` and `enabled_config()` as in `prompt_cache.rs` tests, consts
     `TRACE_A` to `TRACE_E` (the five entries of the worked trace), `entry_with_rows(ids: &[u32]) -> CacheEntry` (a full layer with widths 2, 2, 1 whose
     row `i` is `[id * 0.5, id * 0.5 + 0.25]`, `[id * 0.5 + 0.5, id * 0.5 + 0.75]`, `[id]` for `id = ids[i]` as `f32`; a ring layer
     `LayerCache::ring(KvRing::new(2, 1, 2, 1, 0), ids.len())` with `append_at(0, ..)` of the same rows; a `SharedFromLayer` layer;
     `cached_len = ids.len()`; key `base_key()`), and `fn disk_cache(directory: &Path, max_entries: usize) -> PromptCache` installing
     `Arc::new(DiskTier::open(directory, [0x22; 16]).expect(..))` through `set_cold(tier, max_entries, 1 << 30)`
     (with `use disk_tier::{DiskTier, spilled_names};` at the top of the file).
  3. `proxima-model-interop/src/generate/mod.rs`: declare `#[cfg(test)] mod disk_tier_tests;` beside `mod chunked_prefill_tests;` (~line 231).
- test: add in `disk_tier_tests.rs` (names start `tier_disk_`), each with `tempfile::tempdir()`, the worked-trace entries and
  `PromptCacheConfig { max_entries: 2, ..enabled_config() }`:
  - `tier_disk_demote_writes_the_entry_and_keeps_it_cold`: clone A's state with `PrefixState::branch` as `original`, then store A, B, C (tier limit 2):
    `held()` is `[(0, true), (1, false), (2, false)]`; entry 0 has `state.layer_caches.is_empty()`, `checkpoints.is_empty()` and `state.ids == TRACE_A`; the
    directory holds exactly `0000000000000000.pxkv`; with `key` read from
    `MappedBlockFile::open(path).view().header.content_key`, `std::fs::read(path)` equals the output of
    `original.to_block_file([0x22; 16], key, &mut expected)`; `take_best` of `A + [9]` is a miss with `reused_tokens == 0` and entry 0 is still in `held()`;
  - `tier_disk_files_leave_with_their_entries`: after the stores above, `clear()` leaves the directory empty; a second cache with one cold entry,
    dropped with `drop(cache)`, leaves its directory empty;
  - `tier_disk_trims_cold_files_past_the_limit`: tier limit 1, store A, B, C, D (stamps 0 to 3): `held()` is `[(1, true), (2, false), (3, false)]` and the directory
    holds exactly `0000000000000001.pxkv`;
  - `tier_disk_drops_an_entry_whose_layers_have_no_row_planes`: an entry whose layers are `[Ssm(SsmLayerCache::new(2, 2))]` stored first, then two more:
    it is not in `held()` and the directory is empty (the encoder refused, so the slot dropped the entry as it did before any tier existed).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_23 cargo nextest run -p proxima-model-interop --features std --lib -E 'test(/tier_disk_/)'`
- expect: `4 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/support/disk_tier.rs`, `proxima-model-interop/src/generate/disk_tier_tests.rs`, `proxima-model-interop/src/generate/mod.rs`
- commit: `test(interop): write a disk tier against the cold tier slot`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add non-test code to `src/` beyond the one `mod` line; declare `disk_tier` in `tests/support/mod.rs` (the capability matrix would see unused items); add a dependency.
- gpu: none

### 5.12 trace host to disk to host moves through the cache

- id: FT5.12
- needs: FT5.20, FT5.23
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - the "worked trace of the tiered cache" section of this file: the six steps and the rule list check, with the
    expected stamps and file names;
  - `generate/disk_tier_tests.rs` (FT5.23): the helpers `entry_with_rows`, `disk_cache`, the consts `TRACE_A` to `TRACE_E`, and `disk_tier::spilled_names`;
  - `generate/prompt_cache.rs::thaw_best_cold` (FT5.11, `pub(super)`) and `PromptCache::set_eviction_rules` (FT5.20).
- change: none to non-test code. The technique this proves is a host-then-disk cache with promotion on a hit, and a
  changed eviction policy, expressed only by the slot, the disk tier written against it, and the rule list.
- test: add in `disk_tier_tests.rs` tests, importing `thaw_best_cold` from `super::prompt_cache` and `EvictionRule` from `crate`:
  - `tier_round_trip_entry_survives_host_to_disk_to_host`: run steps 1 to 6 of the worked trace on one `Mutex<PromptCache>` built
    by `disk_cache(dir, 2)` (step 4 is `thaw_best_cold(&cache, &A_plus_nine, &base_key(), &config)`, which is `Ok(true)`),
    asserting `held()` and the directory after every step against the table. Clone A's state with `PrefixState::branch` before step 1. After
    step 4 the hot entry stamp 3 encodes (digest `[0x22; 16]`, key 7) to bytes equal to the encoding of the cloned original A. After step 6:
    `held()` is `[(2, true), (3, true), (4, false), (5, false)]`, the directory is `["0000000000000002.pxkv", "0000000000000003.pxkv"]`;
    `take_best` of `B + [9]` is a miss, and `thaw_best_cold` of `B + [9]` is `Ok(false)`;
  - `tier_policy_oldest_rule_list_moves_a_different_entry_to_disk`: the rule list check of the worked trace, with a branch entry built from
    `entry_with_rows(&[818, 5279, 529, 7001, 9, 9])` and `branch_base = Some(4)`. Default rules: after the third store entry 1 is cold and entries 0 and 2 are
    hot; with `set_eviction_rules(&[EvictionRule::Oldest])` on a second cache: entry 0 is cold and entries 1 and 2 are hot.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_12 cargo nextest run -p proxima-model-interop --features std --lib -E 'test(/tier_round_trip_|tier_policy_/)'`
- expect: `2 passed`, with `tier_round_trip_entry_survives_host_to_disk_to_host` and `tier_policy_oldest_rule_list_moves_a_different_entry_to_disk` named in the output
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/generate/disk_tier_tests.rs`
- commit: `test(interop): trace host to disk to host moves through the cache`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add non-test code; name another test `tier_round_trip_` or `tier_policy_` in the library tests (the count is part of the slice exit).
- gpu: none

### 5.13 a real gemma4 state keeps its bytes across tiers

- id: FT5.13
- needs: FT5.4, FT5.5, FT5.9
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `tests/arch_data_baseline.rs::Checkpoint (~line 41)`, `::open (~line 112)`, `GEMMA4_E2B (~line 54)` and `::llama_parity (~line 668)`: the
    checkpoint struct, the mapping, and the load calls to copy (a test file cannot import another test file);
  - `generate/decode.rs::LoadedModel::prefill_prefix (~line 2163)`: `fn prefill_prefix(&self, prompt: &str, serving_config: &ServingConfig) -> Result<PrefixState, InteropError>`
    returns a real `PrefixState` on the CPU path with the prompt cache off;
  - `tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json`: three records with `prompt`, `prompt_ids`, `generated_ids` from llama.cpp (prompt id
    counts 6, 8 and 57); the record used here is the one with the most prompt ids (57, the "Rivers carry silt ..." paragraph);
  - `block_file.rs::MappedBlockFile` (FT5.5), `tests/support/block_write.rs::write_block_file` (FT5.4) and `PrefixState::{to_block_file, from_block_file}` (FT5.8, FT5.9);
- change:
  1. `proxima-model-interop/tests/tier_real_state.rs` (new, `#![cfg(feature = "std")]`, `#![allow(clippy::unwrap_used, clippy::expect_used)]`): a checkpoint
     with env `PROXIMA_ARCH_GEMMA4_E2B_GGUF` and default path
     `/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`
     that fails, never skips, when the file is absent (an `assert!` on `Path::exists` naming the path and the env var, then `File::open` and
     `unsafe { memmap2::Mmap::map(&file) }`, as `Checkpoint::open` does); a `std::sync::OnceLock<(Vec<u32>, usize, Vec<u8>)>` holding
     `(prompt_ids, cached_len, encoded_bytes)` for the whole process so the model loads once. Its initialiser:
     read the fixture with `serde_json`, take the record with the most prompt ids (`prompt: &str`, `prompt_ids`);
     `let parsed = proxima_gguf::parse_complete(file_bytes)`; `let model = LoadedModel::load(&parsed, file_bytes)`;
     `let state = model.prefill_prefix(prompt, &ServingConfig { prompt_cache: PromptCacheConfig::off(), ..ServingConfig::default() })`;
     `let mut bytes = Vec::new(); state.to_block_file(PROOF_DIGEST, 7, &mut bytes)`, with
     `const PROOF_DIGEST: [u8; 16] = [0x7e; 16];` (the digest only has to match between write and read inside one process; binding a file to a model is
     the cartridge path). `cached_len` is `state.len()`. Premise, asserted in the initialiser: `state.len() == prompt_ids.len()`; if not, stop and report
     both numbers. Imports: `proxima_model_interop::{LoadedModel, PrefixState, PromptCacheConfig, ServingConfig}` and
     `proxima_model_interop::block_file::{MappedBlockFile, decode_block}`, `proxima_model_interop::InteropError`, then `#[path = "support/block_write.rs"] mod block_write;` and `use block_write::write_block_file;` (the support file names `InteropError` through `super`, so the root imports it).
- test: add in `tests/tier_real_state.rs`:
  - `tier_real_state_device_to_host`: `decode_block(&bytes)`, restore with `PrefixState::from_block_file(prompt_ids.clone(), cached_len, &view)`, encode the
    restored state again with the same digest and key: the bytes are identical and `restored.len() == cached_len`; the header layers of the real state hold
    exactly 3 full layers (`rows == cached_len`, `ring_window == 0`), 12 ring layers (`ring_window == 512`, `ring_capacity >= 512`) and 20 absent layers
    (`rows == 0`), which are the counts the gemma4 E2B figures in `PromptCacheConfig::byte_budget`'s doc give (`serving.rs ~lines 474 to 503`); if the
    real state differs, the premise is false: stop and report the counts;
  - `tier_real_state_host_to_disk`: `write_block_file(<tempdir>, 7, &bytes)` returns `0000000000000007.pxkv` and `std::fs::read` of the path equals `bytes`;
  - `tier_real_state_disk_to_host`: write, `MappedBlockFile::open(&path)`, `view()`, restore, re-encode: identical bytes.
- validate: run one process so the model loads once: `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_13 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_13 cargo test -p proxima-model-interop --features std --test tier_real_state -- --test-threads 1 > /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_13/run.log 2>&1; grep -E "^test result|^test tier_real_state_" /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_13/run.log`
- expect: `test result: ok. 3 passed` and the three test names printed as `ok`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/tier_real_state.rs`
- commit: `test(interop): keep a real gemma4 state identical across tiers`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: run it beside another model-loading process; run it a second time under nextest (that starts one process per test and loads the model three times); add a fourth test in this file; use any qwen checkpoint.
- gpu: one run, waiting for a quiet box (CPU forward, no Metal, one model load; run the machine-safety peer check first)

### 5.14 follow-up served from a disk-restored prefix equals llama, gemma4 E2B

- id: FT5.14
- needs: FT5.13, FT5.21, FT5.23
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `tests/tier_real_state.rs` (FT5.13): the checkpoint struct and mapping to copy;
  - `tests/arch_data_baseline.rs::llama_parity (~line 668)`: the load calls and the call
    `model.generate_from_ids(&ids, max_tokens, &config, &mut |_event| ControlFlow::Continue(()))` (`generate/decode.rs::generate_from_ids ~line 2277`, returning
    `Result<(Vec<u32>, String, bool), InteropError>`); the comparison rule here is that the generated ids equal the vendored ones over the shorter length;
  - `tests/support/disk_tier.rs` (FT5.23): `DiskTier::open(&Path, [u8; 16])` and `spilled_names(&Path)`; `generate/prompt_cache.rs::CacheReport`, `CachePath::Tier` (FT5.21) and `ColdTier::install` (FT5.10);
  - `tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json`: three records (prompt ids 6, 8 and 57; generated ids 3, 32 and 1).
- change:
  1. `proxima-model-interop/tests/tier_disk_followup_oracle.rs` (new, `#![cfg(feature = "std")]`, `#![allow(clippy::unwrap_used, clippy::expect_used)]`): imports at the file root
     `use core::ops::ControlFlow;`, `use std::sync::Arc;`, `use proxima_model_interop::block_file::MappedBlockFile;` and `use proxima_model_interop::{CachePath, CacheReport, ColdTier, InteropError, LoadedModel, PrefixState, PromptCacheConfig, ServingConfig};`
     then `#[path = "support/block_write.rs"] mod block_write;`, `#[path = "support/block_keys.rs"] mod block_keys;`, `use block_write::write_block_file;`, `use block_keys::chained_keys;` and
     `#[path = "support/disk_tier.rs"] mod disk_tier;` (the disk tier reads those names through `super`); a `struct Checkpoint { name: &'static str, env: &'static str, path: &'static str }`
     with the E2B constant (`name: "gemma4_e2b"`, env `PROXIMA_ARCH_GEMMA4_E2B_GGUF`, the path of FT5.13) and an `open()` that fails, never skips, as in FT5.13; `const PROOF_DIGEST: [u8; 16] = [0x7e; 16];`;
     and a helper `fn follow_up_from_disk(checkpoint: &Checkpoint) -> FollowUp`, where `FollowUp` is a test-local struct of
     `{ report: CacheReport, restored_ids: Vec<u32>, control_ids: Vec<u32>, files_after_eviction: Vec<String>, case_prompt_len: usize, split: usize, vendored_ids: Vec<u32> }`. The helper:
     1. reads `tests/fixtures/llama-parity/<name>/llama_ids.json` with `serde_json` into records of `prompt_ids` and `generated_ids`; the case is the first record with at least 8 prompt ids and at
        least 8 generated ids; the evicting record is the first record that is not the case; `split = case.prompt_ids.len() * 5 / 8`; `max_tokens = case.generated_ids.len()` (the recorded count, not a constant);
     2. asserts the premise `shared_prefix(evicting.prompt_ids, case.prompt_ids[..split]) * 1000 <= 400 * evicting.prompt_ids.len()` (an inline `zip` and `take_while` count), so the evicting prompt cannot reuse the stored entry; if it
        fails the premise is false: stop and report both id lists;
     3. maps the checkpoint, `proxima_gguf::parse_complete(file_bytes)`, `LoadedModel::load(&parsed, file_bytes)`, opens `let directory = tempfile::tempdir()`, and installs the tier with
        `Arc::new(disk_tier::DiskTier::open(directory.path(), PROOF_DIGEST).expect(..)).install(&model, 8, 1 << 30)`;
     4. builds `cached = ServingConfig { prompt_cache: PromptCacheConfig { max_entries: 1, min_similarity_milli: 400, ..PromptCacheConfig::standard() }, ..ServingConfig::default() }` and
        `uncached = ServingConfig { prompt_cache: PromptCacheConfig::off(), ..ServingConfig::default() }`, and serves with
        `model.generate_from_ids(ids, max_tokens, config, &mut |_event| ControlFlow::Continue(())).expect(..).0`:
        (a) `case.prompt_ids[..split]` with `max_tokens = 1` and `cached` (one entry, hot);
        (b) `evicting.prompt_ids` with `max_tokens = 1` and `cached`: it shares too little with the stored entry, so it misses, is stored, and `max_entries = 1` pushes the first entry to the tier; record
            `files_after_eviction = disk_tier::spilled_names(directory.path())`;
        (c) `case.prompt_ids` with `max_tokens` and `cached`, then `model.last_prompt_cache_report().expect(..)`;
        (d) positive control: `case.prompt_ids` with `max_tokens` and `uncached`.
- test: add `tier_disk_followup_oracle_gemma4_e2b` asserting: (1) `report.path == CachePath::Tier`, the discriminator: a run that silently re-prefilled would pass (4) alone;
  (2) `report.reused_tokens >= split` and `report.prefilled_tokens == case_prompt_len - report.reused_tokens`; (3) `files_after_eviction` has exactly 1 name ending `.pxkv`;
  (4) `restored_ids` equals `vendored_ids` over the shorter length and `restored_ids.len() > 0`; (5) `control_ids` equals `vendored_ids` over the shorter length, so the fixture is the right one;
  (6) `restored_ids == control_ids` exactly (a tier restore changes no id). For this checkpoint the case is the second record (8 prompt ids, 32 generated ids, `split = 5`) and the
  evicting record is the first (6 prompt ids; it shares only its first id with the case).
- validate: `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_14 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_14 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'binary(tier_disk_followup_oracle) & test(/tier_disk_followup_oracle_gemma4_e2b/)' > /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_14/run.log 2>&1; grep -E "Summary|PASS|FAIL" /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_14/run.log`
- expect: `1 passed`, with `tier_disk_followup_oracle_gemma4_e2b` named in the output
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/tier_disk_followup_oracle.rs`
- commit: `test(interop): serve follow-ups from disk-restored gemma4 prefixes`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: query Ollama or llama; use any qwen or openchat checkpoint; add a second model load in this card; run beside another model-loading process; declare `disk_tier` in `tests/support/mod.rs`.
- gpu: one run, waiting for a quiet box (CPU forward, one model load; run the machine-safety peer check first)

### 5.15 follow-up served from a disk-restored prefix equals llama, gemma4 26B

- id: FT5.15
- needs: FT5.14
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `tests/tier_disk_followup_oracle.rs` (FT5.14): the helper and the E2B test to mirror;
  - `tests/arch_data_baseline.rs::GEMMA4_26B (~line 48)`: the 26B checkpoint constants (env `PROXIMA_ARCH_GEMMA4_26B_GGUF`);
  - `tests/fixtures/llama-parity/gemma4_26b/llama_ids.json`: three chat-templated records on main at a7c08c4c, with prompt id counts 18, 23 and 22 and generated id counts 9, 12 and 12.
    The case rule of the E2B card selects the first record (18 prompt ids, 9 generated ids, `split = 11`); the evicting record is the second (23 prompt ids), which shares its first 4 ids
    (`2, 105, 2364, 107`, the chat template's opening) with the stored entry: 4000 is at most 400 times 23, so the premise assertion in the helper holds.
- change:
  1. `proxima-model-interop/tests/tier_disk_followup_oracle.rs`: a `GEMMA4_26B` `Checkpoint` constant (`name: "gemma4_26b"`, env `PROXIMA_ARCH_GEMMA4_26B_GGUF`, path
     `/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129`, the entry for `batiai/gemma4-26b:latest` in
     `tests/fixtures/llama-parity/checkpoints.toml`) and the test below, using the existing helper unchanged (its `max_tokens` is the recorded generated count, 9 here).
- test: add `tier_disk_followup_oracle_gemma4_26b` with the six assertions of the E2B test, on the MoE checkpoint (its layers add expert routing; the cache rows are the same three kinds).
- validate: `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_15 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_15 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'binary(tier_disk_followup_oracle) & test(/tier_disk_followup_oracle_gemma4_26b/)' > /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_15/run.log 2>&1; grep -E "Summary|PASS|FAIL" /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_15/run.log`
- expect: `1 passed`, with `tier_disk_followup_oracle_gemma4_26b` named in the output
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/tier_disk_followup_oracle.rs`
- commit: `test(interop): serve follow-ups from disk-restored gemma4 moe prefixes`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: run it beside another model-loading process (13.3 GB mapped); add a second load; change the helper.
- gpu: one run, waiting for a quiet box (CPU forward, one model load of 13.3 GB; run the machine-safety peer check first)

### 5.16 follow-up served from a disk-restored prefix equals the recorded oracle, granite MoE

- id: FT5.16
- needs: FT5.14, FT0.31, FT0.41, FT0.43
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `tests/tier_disk_followup_oracle.rs` (FT5.14, FT5.15): the helper and the two tests to mirror;
  - `tests/fixtures/llama-parity/checkpoints.toml`: the `granite_moe` row (path `/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`,
    env `PROXIMA_ARCH_GRANITE_MOE_GGUF`); neither it nor the fixture below is on main at a7c08c4c, FT0.31 adds the row and FT0.41 adds the fixture, and FT0.43 proves proxima's tokens equal the recorded ones;
  - `tests/fixtures/llama-parity/granite_moe/llama_ids.json`: the recorded oracle, 3 records with prompt id counts 5, 6 and 72 and 32 generated ids each (the counts FT0.41 measured); the case rule
    selects the third record (72 prompt ids, `split = 45`) and the evicting record is the first (5 prompt ids). Premise commands, run first: `test -f proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json && echo present`
    prints `present` and `git grep -c 'name = "granite_moe"' -- proxima-model-interop/tests/fixtures/llama-parity/checkpoints.toml` prints `1`; if either is absent the premise is false: stop and report. The oracle was recorded once; this card never queries Ollama.
- change:
  1. `proxima-model-interop/tests/tier_disk_followup_oracle.rs`: a `GRANITE_MOE` `Checkpoint` constant (`name: "granite_moe"`, the env and path above) and the test below, using the
     existing helper unchanged. The header of the checkpoint declares no sliding window, so no ring layer is expected in its entries (unmeasured; the test does not depend on it).
- test: add `tier_disk_followup_oracle_granite_moe` with the six assertions of the E2B test.
- validate: `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_16 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft5_16 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'binary(tier_disk_followup_oracle) & test(/tier_disk_followup_oracle_granite_moe/)' > /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_16/run.log 2>&1; grep -E "Summary|PASS|FAIL" /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft5_16/run.log`
- expect: `1 passed`, with `tier_disk_followup_oracle_granite_moe` named in the output
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/tier_disk_followup_oracle.rs`
- commit: `test(interop): serve follow-ups from disk-restored granite prefixes`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a granite-specific code path in `src/`; re-record the oracle; run beside another model-loading process; change the helper.
- gpu: one run, waiting for a quiet box (CPU forward, one model load of 1.4 GB; run the machine-safety peer check first)

## spec drift

1. The decision functions sit in interop, not proxima-core: main has no serving-decision module in proxima-core,
   and the precedents are `pub(super)` functions in interop. The previous cut required `proxima-core/src/kv_decision.rs`;
   nothing in this file depends on it.
2. The mapped read uses `memmap2` (already in `std`), not `proxima-storage` (own non-default feature).
3. Grain is the entry. The block-level `BlockTiers` of the previous cut is gone; ring layers are stored as their whole
   ring (every slot, slack rows included) so a restored ring answers every rewind the original answered.
4. Dense-attention and recurrent layers are refused by `to_block_file` ("layer kind has no row planes") and the
   evicted entry is dropped as today. Those layer kinds exist only in the qwen35 family, which is out of scope.
5. The descriptor digest helper is `xxh3_128` of the `Debug` text of the bound program (FT5.19); no production digest of
   weights or descriptor exists on main. A same-shape fine-tune would share it. The disk tier proof uses a fixed digest
   because its files are per process, so a cross-process tier needs a weight digest and a way to read the bound program
   from outside the crate (a later slice adds the accessor).
6. Disk files belong to the process: the cold entries are in memory, the tier's copy is freed when an entry leaves the
   cache, and a restart does not reuse a previous run's files.
7. The seal codec (KV rows to encoded rows) is not built; see "designs abandoned".
8. Acceptance counts: the library tests print `tier_demote_` 6, `tier_restore_` 5, `tier_label_` 2, `tier_failure_` 2,
   `tier_disk_` 4, `tier_round_trip_` 1 and `tier_policy_` 1; the integration binary `tier_real_state` prints 3 in one process
   (one model load); `tier_disk_followup_oracle_` is 3 tests (gemma4 E2B, gemma4 26B, granite MoE), each run alone by its own
   card, where the previous cut counted 4 checkpoints including qwen2, openchat and qwen3. The cross-slice count that
   mentioned 4 of 7 llama parity runs is not touched by this file.
9. Unrun and unmeasured: that a restored entry decodes to the same ids as a resident one is asserted only by the
   three oracle cards at the vendored prompts (and against an uncached control run in the same test); whether a disk restore is
   faster than a re-prefill is not measured.
10. Lfu and fifo rules are not built (no hit count or creation order on `CacheEntry`); the rule list is where they
    land when the field does.
11. `demote` and `discard` run under the cache lock (a demote is called from inside `PromptCache::store`), so a slow tier
    stalls other lookups for the length of its write. Only the read of a cold entry happens outside the lock. Unmeasured.
12. Two requests that both take a ticket for one cold entry restore it once: the second `thaw` finds the entry gone and
    keeps nothing (asserted by `tier_restore_keeps_nothing_when_another_request_restored_the_entry_first`).

## slice exit

- `cargo nextest run -p proxima-model-interop --features std -E 'test(/blockfile_/)'` prints `19 passed`
  (2 encoded length, 3 encode, 6 decode, 2 plane, 1 digest check, 1 model digest, 1 read in the library tests; 1 write in `block_file_write` and 2 chained keys in `block_file_keys`, both integration binaries the same filter reaches);
  `test(/prefix_state_file_/)` prints `8 passed`; `test(/eviction_rules_/)` prints `3 passed`.
- `cargo nextest run -p proxima-model-interop --features std --lib -E 'test(/tier_demote_/)'` prints `6 passed`; `test(/tier_restore_/)` prints `5 passed`;
  `test(/tier_label_/)` prints `2 passed`; `test(/tier_failure_/)` prints `2 passed`; `test(/tier_disk_/)` prints `4 passed`; `test(/tier_round_trip_|tier_policy_/)` prints `2 passed`.
- `cargo test -p proxima-model-interop --features std --test tier_real_state -- --test-threads 1` prints `3 passed` (one model load).
- Each oracle test is run by its own card, alone, with `1 passed`: `tier_disk_followup_oracle_gemma4_e2b`, `tier_disk_followup_oracle_gemma4_26b`, `tier_disk_followup_oracle_granite_moe`.
- `cargo clippy -p proxima-model-interop --features std,metal --all-targets` is clean.
- Premises that stop an executor when false: the gemma4 E2B layer counts in FT5.13 (3 full, 12 ring, 20 absent) and its `state.len() == prompt_ids.len()`; the evicting-record premise assertion in
  the oracle helper (FT5.14); the granite checkpoint row and recorded oracle in FT5.16.
- Not cuttable further: FT5.8 is the largest card (about 85 non-test lines, one public item); FT5.10 (about 84, the slot, its installer and the demote path) cannot lose the
  installer or the demote without leaving an unused item, and its limits already sit in FT5.24 and FT5.25; FT5.8 and FT5.9 stay two cards because each adds one public item
  another card consumes. Every line count in this file is an estimate derived from the change lists, not a measurement.
