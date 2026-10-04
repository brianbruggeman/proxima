# sketch 7: n-gram speculation, the control (H11 step; reads H14 and H3)

status: paper test. Read at main 4b4be6cf (`git show main:<path>`). Nothing was built or run; every
"hand-derived" value is derived from cited lines, not executed. Paths relative to the proxima repo root;
`interop` = `proxima-model-interop/src`, `tok` = `proxima-tokenizer/src`.

Purpose of a control: speculation is the one stage the SPEC says composes from configuration today
(SPEC "how much composes today", row H11). If this sketch needs hook changes, the catalog is wrong about
what "exists today" means, and that is the finding. If it needs none, the method can pass something.

## shape chosen, and the contested decision

Shape: the whole stage is ONE config section, `[speculative]`, whose keys already exist
(`SpeculativeSettings`, `interop/speculative_settings.rs:253-338`). No pipe is written. The verify half is
not a new stage: it is H14's own `select_decoded_token` run on each verify row.

Contested decision: should the drafter list become a `Pipe`-typed slot (`Drafter: Pipe<In = DraftQuery,
Out = Vec<u32>>`) composed from config? Chosen: no. Second gate, call site both ways. Before
(`interop/generate/decode.rs:3669-3674`): `drafter_set.draft(&token_history[..history_len], next_ids[0],
step_draft_limit, &mut speculative_draft)`. After: `drafter.call(query)` with `query` holding the same
four values. Same arguments, same effect, so it is a relocation and is not minted. Two further reasons it
fails, both already on record: `Pipe::call` takes `&self` (`proxima-primitives/src/pipe/primitives.rs:101`)
while the drafters mutate their own tables (`ngram_map_begin`, `ngram_mod_accept`, `drafter.rs:65-96`);
and a config-length chain cannot be `and_then` (`proxima-primitives/src/pipe/ext.rs:48-54`, type fixed at
compile time). The existing shape, a closed enum walked in order, is the right one.

## 1. ground: what a user writes today

The loader exists and round-trips TOML and env: the test at `interop/speculative_settings.rs:398-454`
builds `SpeculativeSettings` three ways (builder, `conflaguration::from_file` on a TOML with root keys
`speculative_types = "ngram-simple,ngram-map-k"`, env `PROXIMA_SPECULATIVE_TYPES`) and asserts equality.
It lowers to the `Copy` struct the decode loop reads through `as_speculative_config`
(`speculative_settings.rs:346-381`), which `ServingConfig.speculative` carries (`interop/serving.rs:1077`).
Default is ON with `ngram-simple` (`speculative_settings.rs:262-264`; `serving.rs:458-462`); `none` turns
it off (`serving.rs:412`).

The live loop, in order (`interop/generate/decode.rs`):
1. `DrafterSet::build(&serving_config.speculative, context_length)` once per call (:3529-3530), then
   `begin(&token_history)` trains `ngram-map` and `ngram-mod` over the prompt (:3531; `drafter.rs:65-71`).
2. Each decode step with `next_ids.len() == 1`, `cached_len > 0`, and `speculative_verify_program.is_some()`
   (:3642-3646): `draft_limit_for_step` (:3655; `drafter.rs:107-116`: context room minus 2, budget room
   minus 1), then `drafter_set.draft` walks the enabled drafters in llama's fixed priority order and the
   first non-empty draft wins (`drafter.rs:198-218`).
3. One forward over `[sampled, draft...]` through the all-positions program (:3680-3687, :3775-3800).
4. Verify (:5741-5830): row `r` is selected by `select_decoded_token(_step + r, row, &token_history,
   repeat_window, token_override, sample_config, &mut rng)` (:5758-5766); the loop breaks at the first row
   whose selection differs from `speculative_draft[r]` (:5769-5771); `accepted = emitted.len() - 1`
   (:5780); `drafter_set.accept(accepted as u16)` (:5801); attention caches `truncate` to
   `cached_len_before_step + emitted.len()` (:5808-5821); extra emitted ids queue in `pending` (:5822-5824).

Because verify uses the same selection as plain decode (`decode.rs:1307-1318` doc), any H14 or N1 stage
composes with speculation without a second implementation. That is the property worth protecting.

## 2. the config a user would write

Real data: gemma4 E2B is the one checkpoint with a verify program (gap 4 below). Its header
(`proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/gguf_kv.txt:12-13`): `block_count = 35`,
`context_length = 131072`. A user who wants two drafters in llama's priority order, with tighter sizes,
writes the following. Every key is an existing `SpeculativeSettings` field; the section header is the only
addition (today the keys sit at the TOML root, `speculative_settings.rs:414-431`).

```toml
[speculative]
speculative_types = "ngram-simple,ngram-map-k"
ngram_simple_size_n = 12
ngram_simple_size_m = 48
ngram_simple_min_hits = 1
ngram_map_k_size_n = 12
ngram_map_k_size_m = 48
ngram_map_k_min_hits = 1
```

Defaults equal today: this block with only `speculative_types = "ngram-simple"` is the shipped default
(sizes 12/48/1, `serving.rs:332-341`).

## 3. field map: every key to a consumer, or a gap

| key | consumer | status |
|---|---|---|
| `speculative_types` | `SpeculativeTypeNameSet` -> `SpeculativeTypeSet` bitset (`serving.rs:273`) -> `DrafterSet::build` iterates `iter_priority_order` (`drafter.rs:134`) | EXISTS (order fixed, no repeats: GAP-2) |
| `ngram_simple_size_n`, `_size_m` | `NgramSimpleConfig { size_n, size_m }` (`drafter.rs:136-139`) | EXISTS |
| `ngram_simple_min_hits` | none: `NgramSimpleConfig` has two fields, `size_n` and `size_m` (`tok/draft/ngram_simple.rs:47-52`), so the value has nowhere to go | GAP-3 |
| `ngram_map_k_*`, `ngram_map_k4v_*` | `NgramMapConfig` (`drafter.rs:140-157`) | EXISTS |
| `ngram_mod_n_match`, `_n_max`, `_n_min` | `NgramModConfig` (`drafter.rs:158-162`); `n_max` also sizes ring slack (`generate/kv_ring.rs:233`) | EXISTS |
| `n_max`, `n_min`, `p_min` | none: grep for `speculative.n_max`, `.n_min`, `.p_min`, `config.n_max` over `interop/src` finds only the loader and its tests; the per-step cap is `draft_limit_for_step` alone (`decode.rs:3655`) | GAP-3 |
| `ngram_cache_lookup_static`, `_dynamic` | `NgramCacheState::new(max_context_len)` takes neither (`drafter.rs:163-165`) | GAP-3 (git grep `ngram_cache_lookup` over `interop/src` finds only `serving.rs` and the loader) |
| (draft-model types `draft-*`) | parse, then rejected by `apply_serving_config` (`serving.rs:160-170`) | out of scope: needs a second model |

## 4. can the decision be a pure function in core plus a pipe?

Decision half: yes, and it is already pure, but it lives in `proxima-tokenizer`, not `proxima-core`.
`ngram_simple_draft`, `ngram_map_draft`, `ngram_mod_draft`, `ngram_cache_state_draft` are free functions
over caller-owned buffers (`tok/draft/mod.rs`, imported `drafter.rs:31-35`; module is `no_std + alloc`,
`tok/lib.rs:43-45`). The catalog column "decision in proxima-core" is wrong for H11 and H14 (sampler,
`tok/sample.rs`): both decisions live in the tokenizer crate. Not a defect in the code; a defect in the
SPEC row. Recorded below.

Placement half: the step loop is inline code in a closure (`decode.rs:3600-3830`), not a pipe and not the
FSM. The FSM exists (`interop/serving_fsm.rs`, whose module doc claims its transitions replace
`generate.rs`'s branches) but `#![allow(dead_code)]` sits at `serving_fsm.rs:38` and the module is declared
at `interop/lib.rs:70`; `generate/serving_backend.rs:57` carries the same allow. The SPEC says this
(status table row "the serving FSM drives the live loop: false today"); this sketch adds the mechanism:
the FSM's `Verify`/`accept` are the same operation as `decode.rs:5741-5830`, written twice.

## 5. worked example (doubles as the test)

The repo's own fixture, `drafter.rs:255-289`. Config: types = `ngram-simple` + `ngram-map-k`, both with
`size_n = 3, size_m = 3, min_hits = 1`. History `[0, 1, 2, 3, 4, 5, 9, 9, 1, 2]`, sampled `3`.
The trailing pattern is the last two history ids plus sampled: `[1, 2, 3]`. It recurs at index 1; the
ids after it are `[4, 5, 9]`, so `ngram-simple` drafts `[4, 5, 9]` (the test's own comment, :272-276, says
the same). `ngram-simple` is first in priority (`serving.rs:237-238`), so it wins and `active_type()` is
`NgramSimple` (asserted :284-287). Hand-derived from the cited test; not executed.

Verify trace, hand-derived from `decode.rs:5757-5780`: draft `[4, 5, 9]` is forwarded as 4 rows
(`sampled`, 4, 5, 9). Suppose greedy selection gives `[4, 5, 7, ...]` on rows 0..2. Row 0 selects 4 and
matches draft[0]=4; row 1 selects 5 and matches draft[1]=5; row 2 selects 7, does not match draft[2]=9,
break. `emitted = [4, 5, 7]`, `accepted = 2`, caches truncate to `cached_len_before_step + 3`, ids 5 and 7
queue in `pending`. `drafter_set.accept(2)` reaches only the winning drafter (`drafter.rs:223-229`).

Config parity (P4): the TOML above loads to a value equal to
`SpeculativeSettings::builder().speculative_types(simple+map_k).build()`; the test at
`speculative_settings.rs:398-454` is that assertion for a different parameter set. The section header is
not covered; the loader's `#[settings(prefix = "PROXIMA_SPECULATIVE")]` (:254) is the env side only.

## 6. HOOK GAPS (stage, missing input, smallest generic change)

GAP-1. H11 step. Missing input: the stage is not a slot. The decode loop open-codes draft, forward,
verify and rewind (`decode.rs:3642-3830`, `:5741-5830`); the FSM that would own it is dead code
(`serving_fsm.rs:38`). Smallest change: drive the loop through the FSM's `enter_verify`/`accept`, so the
draft call is the one place a drafter list plugs in. Evidence this is real and not style: the rewind
(`cache.truncate(keep_positions, ..)`, :5808-5821) acts on attention layers only, so a hybrid or
recurrent checkpoint cannot use this path (the FSM's `snapshot` is the shape that would, `serving_fsm.rs`
`Verify { snapshot }`).

GAP-2. H11 step. Missing input: the drafter list is a bitset over a closed enum (`SpeculativeTypeSet(u16)`,
`serving.rs:273`; `Drafter` enum, `drafter.rs:42-48`; the priority table is written twice,
`serving.rs:237-248` and `speculative_settings.rs:118-129`, the second noting it "mirrors ... one to one").
So the order is fixed, a type cannot appear twice (two `ngram-simple` with different sizes), and a new
drafter edits the enum and four `match` sites (`drafter.rs` x3, `kv_ring.rs`). Smallest change: an ordered borrowed slice of
`{ kind, params }` entries in `SpeculativeConfig`, the same shape as
`weight_precision: &'model [WeightPrecisionRule<'model>]` (`serving.rs:107-110`), which keeps
`ServingConfig` `Copy` (the load-bearing reason given at `speculative_settings.rs:10-13`). Call site both
ways: before `SpeculativeTypeSet::single(NgramSimple).insert(NgramMapK)` with params in sibling fields
(`serving.rs:393-396`); after `&[Entry::NgramSimple(p), Entry::NgramMapK(q)]`. These differ (order and
repeats become expressible), so this is a data-shape change, not a new behaviour type. New drafter kinds
still need a Rust variant: that is the closed-set case and is correct (box-free rule). Status: plausible;
no technique in this batch requires it.

GAP-3. H11 step. Missing input: config fields with no consumer. `speculative.n_max`, `n_min`, `p_min`
(`serving.rs:386-392`, `speculative_settings.rs:269-277`), `ngram_simple_min_hits` (:288-290,
`drafter.rs:136-139`), and `ngram_cache_lookup_static/_dynamic` (:331-337; `drafter.rs:163-165`). Each
parses, validates and does nothing. Smallest change: wire `n_max` into `DrafterSet::draft`'s cap
(`min(step_draft_limit, n_max)`; llama's `dp.n_max`, the comment at `drafter.rs:194-197` already names
it) and delete `ngram_simple_min_hits` (`NgramSimpleConfig` has no such field) and the cache-path keys until a
drafter consumes them. Deleting is the Directive-compliant repair for a field nothing reads.

GAP-4. H11 / H6. Missing input: the default-on speculation is silently off for six of the seven
checkpoints. `speculative_verify_program` defaults to `Ok(None)` (`interop/architecture.rs:327-335`) and
only gemma4 overrides it (`gemma4/bind.rs:843`); the loop checks `.is_some()` (`decode.rs:3645`), and the
`ServingConfig.speculative` doc says this field is "the sole gate for whether speculation runs"
(`serving.rs:1074-1076`). Config says on; behaviour is off, with no log at the decision point. Smallest
change: `apply_serving_config` (or load) emits one debug event stating "speculation enabled by config,
unavailable: architecture has no verify program", and the `Verify` program becomes a descriptor-derived
variant (H6) rather than a per-architecture override. Status: the silent-off is read from code
(`:3645`); which of the seven checkpoints have a verify program was checked only for the override site
(`git grep speculative_verify_program` over `interop/src` shows one override, `gemma4/bind.rs:838-852`).

GAP-5. SPEC catalog. The column "decision in proxima-core" is wrong for H11 and H14: both decisions are
in `proxima-tokenizer` (`tok/draft/`, `tok/sample.rs`). Smallest change: fix the catalog column to name
the crate that owns each decision, or move the functions; moving is not needed for any technique here.

## 7. what is not a pipe, and why

- `DrafterSet`: a stateful walk over a closed enum, `&mut self`; `Pipe::call(&self)` forbids it without
  interior mutability (section "contested decision").
- `Drafter::{begin, draft, accept}`: three methods of one object, not an `In -> Out` step. `begin` and
  `accept` are state updates around the draft.
- The verify loop: it is H14 applied per row. A pipe over (row, draft) would be a second selector.

## 8. designs abandoned

- `Drafter: Pipe` composed with `and_then`: identical call site; static chain; `&self` versus mutation.
- `Box<dyn Drafter>` list: ruled out by the box-free default; the kinds are a closed set.
- A `trait DraftPolicy` with per-technique impls: the five existing kinds share one shape and one walk;
  a trait would be a blanket-impl-under-a-new-name.
- Moving `DrafterSet` into core: it is `pub(crate)` in interop and its only dependency is
  `proxima_tokenizer::draft`, so the decision half is already as low as it can usefully go.

## defects found in passing

- Dead config fields (GAP-3); stale docs: `serving.rs:266-271` and `:371-375` say the decode loop wires
  one set member and that a `Drafter` enum "slice 9 adds" the rest, while `drafter.rs` wires five.
- Duplicated priority table (`serving.rs:237-248`, `speculative_settings.rs:118-129`).
- `serving_fsm.rs` and `serving_backend.rs` carry `#![allow(dead_code)]` (`:38`, `:57`) over a whole
  module; the repo rule is "avoid allow(...)".
- `debug!("speculative_pending_pop")` and the verify event are the only decision-point logs; the
  unavailable-verify case (GAP-4) logs nothing.

## verdict

As configuration, the control passes: the section above loads and reproduces today's behaviour through
existing keys, with no core edit. As a hook, it fails the invariant "every stage is a slot whose default
reproduces today": fits after 5 named hook changes: GAP-1 (live loop through the FSM), GAP-2 (ordered
drafter list as data), GAP-3 (wire or delete five dead keys), GAP-4 (silent-off for architectures without
a verify program), GAP-5 (catalog column). Not claimed: that any of this preserves token ids or speed;
none of it was run.
