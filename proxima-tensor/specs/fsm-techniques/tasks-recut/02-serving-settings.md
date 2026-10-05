# slice 2 (re-cut): serving settings as the configuration surface for the hooks (cards FT2.1 - FT2.24)

anchors read at main a7c08c4c (full sha a7c08c4c95836f112742f0e30e4ddd672087cda0) with `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`. An earlier read at main e4cf9beb found every source line below unchanged: `git diff --stat e4cf9beb a7c08c4c -- proxima-model-interop proxima-tensor/src omega/src` lists only `proxima-model-interop/tests/arch_data_baseline.rs` and a llama-parity fixture json. Every `path::symbol (~line N)` below is the line at that sha; the executor re-locates by symbol. Paths are relative to the proxima repo root. `interop/` means `proxima-model-interop/src/`.

Rules: `../CARDS.md` (binding). Governing text: `proxima-tensor/specs/pipeline-as-data/SPEC.md` (hook catalog, composition section, structural findings), `research.md` (CC-1 to CC-12), sketches 03, 08, 10, 11, 12 and 13 (gap lists). Owner direction this re-cut obeys: the hooks are built so techniques can be vetted later; every stage is a slot whose contents are a configured list; its default reproduces today byte for byte; a technique is never library code.

Crate for every card: `proxima-model-interop`. Features for every settings card: `std,conflaguration` (`std` already pulls `conflaguration`, `bon`, `toml` and `proxima-tensor/config`: `proxima-model-interop/Cargo.toml` `std = [..]`, ~line 37). Validate command prefix for every card: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_<card number, dots as underscores>` (for example `ft_2_15a`), removed when the card is done.

Test models (owner, 2026-10-04): gemma4 for dense or MoE, granite for MoE. No qwen of any kind appears in any test, fixture or validation of this file. Fixtures here are the vendored gemma4 headers (`proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/gguf_kv.txt`, dense; `gemma4_26b/gguf_kv.txt`, MoE). Granite has no vendored header on main (`git grep -n -i granite main -- proxima-model-interop proxima-tensor/src` prints nothing at a7c08c4c); it arrives as a family profile plus descriptor from the slice-0 granite card. Exactly one card, FT2.20, loads a model (the gemma4 E2B checkpoint, on the CPU path, one run, waiting for a quiet box); every other card needs neither a model load nor the GPU.

Counts: 32 cards. 9 are kept from the previous cut (re-anchored; FT2.4 and FT2.8 keep their ids and hand their extra public types to fresh cards), 23 are new (10 replace a recut card under its old id, 13 carry fresh ids FT2.3, FT2.22 to FT2.27 and FT2.29 to FT2.34). Previous cut: 21 cards; 9 kept, 10 recut, 2 dropped. An earlier draft of this re-cut had 30 cards, with an FT2.28 that added a `Tier` enum; it is gone (see "dropped").

## old to new id map

Sibling files cite the old ids; this table is the key. An id that stays keeps its role.

| old | new | verdict |
|---|---|---|
| FT2.1 | FT2.1 | recut: only the `AttentionConfig` field of the read spec on `ServingConfig`, and the read spec joins the prompt-cache key; the read spec type itself is FT2.32. The kv and decode sections, `SealSummary`, `EvictionPolicy`, `Allocation` and `ReadSpec::Sampled` are gone |
| FT2.2 | FT2.2 | recut: only the `PrefillConfig` field with the assemble list on `ServingConfig`; the step type itself (`Prefix`, `Shift`) is FT2.33. `Load`, `Blend`, `Sleep`, `Compact` and the whole cascade grammar are gone: a variant lands with the card that binds its arm |
| none | FT2.32 | new: the `ReadSpec` enum alone (`Dense`, `Operand`), split out of FT2.1 so each card adds one public type |
| none | FT2.33 | new: the `AssembleStep` enum alone (`Prefix`, `Shift`), split out of FT2.2 so each card adds one public type |
| none | FT2.34 | new: nests the attention section and carries the refusal of an unbound read (the refusal enum, `refusals()` and `Validate`), so `operand` is never loadable from settings without being refused in the same commit |
| none | FT2.22 | new: the eviction rule and the idle step become loadable from text (their types come from the tiers and idle-jobs slices) |
| FT2.3 | FT2.3 | new under an id the previous cut never used: the `CacheType` mirror enum alone, so the settings skeleton adds one public type |
| FT2.4 | FT2.4 | kept; fixtures move to gemma4; `CacheType` moves out to FT2.3 |
| FT2.5 | FT2.5 | kept; the metal math mirror has three variants (main grew `Fast`); the dispatch mirror and its field move to FT2.31 |
| FT2.6 | FT2.6 | recut: `weight_precision` list (one enum, `WeightPrecisionRuleSettings`), `gdn_prefill_backend`, `gpu_correctness_fallback`. The eleven placement fields move to FT2.23 |
| none | FT2.23 | new: the placement section wrapper that nests the two placement leaves (experts and budgets) into the settings; the leaves are FT2.29 and FT2.30 |
| FT2.7 | FT2.7 | kept |
| FT2.8 | FT2.8 | kept; narrowed to the admission level, in `levels.rs`; the phase level is FT2.25, the expert residency level FT2.26, `speculative` and `prompt_cache` FT2.27 |
| FT2.9 | FT2.9 | recut: kv section is the one block size and the eviction rule list; the tier list and the `Tier` enum are gone (see "dropped") and the shared three-way helper moved to FT2.27 |
| none | FT2.25 | new: the phase level |
| none | FT2.26 | new: the expert residency level |
| none | FT2.27 | new: `speculative` and `prompt_cache` nested; carries the shared three-way helper |
| none | FT2.29 | new: the expert placement leaf alone |
| none | FT2.30 | new: the placement budget leaf alone |
| none | FT2.31 | new: the `DispatchTypeName` mirror enum and the `dispatch_type` field, split out of FT2.5 so each card adds one public type |
| FT2.10 | FT2.10 | recut: the attention section leaf loads `Dense` or `Operand` on its own; it is not reachable from `ServingSettings` until FT2.34 nests it |
| FT2.11 | none | dropped, see below |
| FT2.12 | FT2.12 | recut: prefill section loads the assemble list (`Prefix`, `Shift`) |
| FT2.13 | FT2.13 | recut: schedule section loads the idle step list |
| FT2.14 | none | dropped, see below |
| FT2.15 | FT2.15 | kept; field count 58 (60 with the metal backend), union fixture on gemma4 values |
| FT2.15a | FT2.15a | kept; lands before FT2.9 (needs FT2.27), so the kv lowering never overrides an explicit prompt cache block size |
| FT2.16 | FT2.16 | recut: only the block-size conflict row; the refusal enum, `refusals()` and `Validate` come from FT2.34 |
| FT2.17 | FT2.17 | recut: read rows only (attention layers, two-range cache); the unbound read row moved to FT2.34; the rectify rows are gone; fixtures are gemma4-shaped descriptors |
| FT2.18 | FT2.18 | recut: assemble, shift and eviction-list rows; the tier-list row is gone with the tier list |
| FT2.19 | FT2.19 | recut: idle-step rows (worker, entry room) |
| none | FT2.24 | new: placement rows |
| FT2.20 | FT2.20 | kept, premise fixed: the guard is shared, not deleted; one model-loading test asserts the refusal on the uncached path |
| FT2.21 | FT2.21 | kept; 14 rows, 21 tests |

## dropped

- FT2.11 (decode section: `rectify.every`, `samples`; the triage verdict was recut, and the recut resolves to the sibling cards named here, so no card remains): `samples` is not needed (sketch 13 G2: N samples are N calls with `seed + index`) and `rectify.every` is a technique knob. The hook changes sketch 08 assigns to the step stage already have cards in the sibling re-cuts: the enter rule list and its config home are FT7.3 and FT7.4, the accept rule is FT1.10 and FT1.11, the replay resume is FT7.2, the rewind capacity is FT7.5, the replay-versus-seal row is FT7.7. A decode section here would duplicate them, so this file builds none. Required amendments to those cards are listed under "contracts with sibling files".
- FT2.28 (the `Tier` enum, `Host | Disk { path, bytes, max_entries }`) and the `kv.tiers` list, with the `TiersNeedHostThenDisk` row of FT2.18 (all inside this file; no sibling card cites FT2.28): the tiers slice (`tasks-recut/05-tiers.md`) drops the `Tier` enum as a duplicate of the list element (its "dropped" section) and builds the library as a `ColdTier` slot (FT5.10) with the disk tier as test support (`tests/support/disk_tier.rs`, FT5.23). No setter takes a disk tier from settings, so a `Tier::Disk` value in settings would be a technique's grammar (a directory, a byte cap, an entry cap) with no hook arm behind it, which this file refuses to build (the rule under "Designs abandoned"). The prompt cache setters that FT5.20 and FT10 add are the only list-valued setters settings feeds, and the tier is not one of them. If the owner wants a tier list in settings, a library lowering card from the `ColdTier` slot must exist first.
- FT2.14 (cascade section): sketch 13 finds the conformal cascade fits as is with no hook change (one pure settle function plus a pipe of about 38 lines in the consuming crate). A cascade section would build the technique grammar (`RouterSpec::Cluster`, five judge kinds) in the library, and three of the judges read per-token values `TokenEvent` does not carry. It becomes a worked example in the cascade slice. Its rows (`JudgeNeedsReadout`, `CascadeTooManyTiers`) go with it.

## what this re-cut changes, in English

- The settings type is the home of every list. `ServingConfig` is `Copy` (`interop/serving.rs::ServingConfig`, derive ~line 719), so a list cannot be one of its fields as an owned container. Two rules decide where a list lives:
  - A list the live request reads from its `ServingConfig` rides in it as a borrowed slice, `&'model [Elem]` (the existing precedent is `weight_precision: &'model [WeightPrecisionRule<'model>]`). Only the assemble list is such a list.
  - A list the prompt cache reads through a setter (eviction rules and idle steps; sketches 10 and 12 and the tiers and idle-jobs slices) lives only in `ServingSettings`. The caller passes it to the setter: `model.set_eviction_rules(&settings.kv.eviction)`. Call site both ways: a `ServingConfig.kv.eviction` field read by the cache against that line; the cache would need the same setter anyway, so the field adds nothing a caller can do. No such field is built.
- The cache key sees exactly one new fact: the read spec (`Dense` or `Operand`), because rows computed under a skipped read are not the rows a dense read computes (sketch 08 gap on the read field). Lists that only choose where rows come from or rest do not enter the key; a stage that changes what a row means adds its own identity to the key when it lands (sketch 11 section 4).
- Every default reproduces today: the empty assemble list is today's lookup (prefix reuse, plus shifted-chunk reuse when `prompt_cache.cache_reuse_min` and `ring_rewind_slack` are above 0), the eviction list `[branch, oldest]` is today's victim rule, an empty idle list is today's two-step sequence.
- Each list element is closed grammar in settings. A stage the library does not define is written in Rust against the hook (the hook cards own that proof); settings name only the stages the library defines.
- Validation rows exist only where a loaded configuration would otherwise be silently ignored or fail later at run time, and each row is a pure function of settings (or of settings plus a model descriptor).

Designs abandoned (each constraint changed the design):
- `ReadSpec::Block { keep_ratio, min_blocks, local_blocks }` as config (sketch 08 gap on the read field, and the triage verdict for this slice). The read-hook cards implement the hook as a function argument and select it with a unit variant; block selection is test code there. A `Block` value would be parameters of a technique no library code runs, so it would need a permanent refusal. CC-3 (a summary granularity separate from the block size) therefore has no config field here: the summary granularity is the summarizer's argument, and the one block size stays `kv.block_tokens`. If the owner wants a `Block` config, a library lowering card must exist first.
- `EvictionPolicy lru | lfu | fifo` (does not reproduce today's rule: unused follow-up branches go first, `interop/generate/prompt_cache.rs::PromptCache::eviction_victim` ~line 820; `lfu` and `fifo` need entry fields `CacheEntry` lacks).
- `AssembleStep::Load { path }` in the settings grammar. A `Load` value is a cartridge-loading configuration with no library arm behind it (the assemble slice's `kind_of` binds only `Prefix` and `Shift`), which is a technique's grammar with no hook. The variant lands, with its payload, in the card that binds its arm (see "contracts with sibling files").
- A `PatternKind` enum beside a `WeightPrecisionRuleSettings` struct. One internally tagged enum, `WeightPrecisionRuleSettings`, carries the same text form (`pattern_kind = "suffix"`, `pattern`, `target`) with one public type instead of two.
- `Allocation::Reserve`, `SealSummary`, `kv.seal` and a second host-tier size (the host tier is `prompt_cache.byte_budget`).
- A `ServingConfig.kv` or `ServingConfig.schedule` section (above).
- `kv.tiers` with a `Tier` enum (`Host`, `Disk { path, bytes, max_entries }`): see "dropped". What the constraint changed: the library holds the `ColdTier` slot and settings hold no disk-tier grammar.
- Loading `attention.read = "operand"` in settings in a commit that precedes its refusal (the previous draft nested the attention section at FT2.10 and refused it at FT2.17). What the constraint changed: the section is cut into a leaf (FT2.10, unreachable from `ServingSettings`) and a nesting card (FT2.34) that carries the refusal, so no commit loads an `Operand` read that nothing runs or refuses.
- Writing `kv.block_tokens` over an explicit `prompt_cache.block_tokens` until a later card refuses the conflict (the previous draft). What the constraint changed: FT2.15a lands first and the lowering takes the explicit value when one is set (D7), so no commit overwrites a value the user wrote.
- A `ReadSpec` and an `AttentionConfig` in one card, and an `AssembleStep` and a `PrefillConfig` in one card (the previous draft of FT2.1 and FT2.2). What the constraint changed: one new public type per card (D16), so each pair is a leaf card and a field card.
- An `execution` enum replacing the four expert booleans (sketch 03 gap on the four booleans). `decode.rs` shows `pre_gather`, `monolithic_all_low` and `monolithic_high_mmap` are separate requests read at separate sites (`generate/decode.rs` ~lines 3118 to 3140 and `qwen35moe_monolithic_all_low_enabled` ~line 7264); this file cannot prove they are one mode, so it adds rows for the inert combinations that the code does show instead.

## contracts with sibling files (amendments the owner must apply; this file does not edit them)

Each premise below is read from the sibling card text in `tasks-recut/`. Until it is applied, the named sibling card stops on its false premise.

- FT7.3, FT7.4, FT11.3: they assume `DecodeConfig`, `ServingConfig.decode`, `DecodeSettings` and `ServingSettings.decode` exist from this slice (with a `samples` field). They do not. FT7.3 must create `DecodeConfig<'model> { enter: &'model [EnterRule] }` (plus whatever FT11.3 adds), the `ServingConfig.decode` field, the six full-literal sites in `interop/serving.rs` and the two exhaustive destructures (`decode: _` lines in `interop/generate/prompt_cache_key.rs` and `interop/generate/resident_plans.rs`). FT7.4 must create `DecodeSettings` and the nested `ServingSettings.decode` field with its lowering line in `as_serving_config`. FT11.3 must drop its reference to `samples`.
- FT7.7: it reads `self.kv.seal.horizon_rows` (FT2.9 of the previous cut). This file has no `kv.seal`; the one home of the horizon is `prompt_cache.seal_horizon_rows` (the seal card, FT4.2). FT7.7 must read `self.prompt_cache.seal_horizon_rows` and its test must set that field.
- FT6.9 to FT6.14: they assume `ReadSpec` is `Dense` plus one unit variant `Operand`, `Default` is `Dense`, and `PlanIdentity::of` carries `attention: _` until FT6.10 binds it. FT2.32 defines the first two and `PlanIdentity::of` carries `attention: _` after FT2.1; no amendment.
- FT5.7, FT5.10, FT10.4: they define `EvictionRule`, `IdleStep` and `DraftStep`. FT2.22 adds the serde derives to those definitions and keeps one definition each. No amendment.
- FT8.4 (`08-assembled-prefill.md`, the `assemble_kind_refuses_a_step_with_no_bound_pipe` test and the refusing arm of `kind_of`): the test builds `AssembleStep::Load { block_keys: Vec::new(), cartridge_path: None }`. FT2.2 defines exactly `Prefix` and `Shift`, so that variant does not exist and the arm over "every other variant" would be an unreachable pattern. The card's own fallback applies ("if it declares none, delete the refusing arm and this test and report"); the amendment makes that the plan: `kind_of` is an exhaustive `match` of the two variants, with no refusal arm and no refusal test, and the `has no bound pipe` text returns with the first variant the grammar gains.
- The first FT14 card (`14-cartridges.md`, the premise at its "FT2 (all cards)" line): it premises `AssembleStep` with `Load { block_keys: Vec<String>, cartridge_path: Option<String> }` and `Blend { .. }`, and its premise command `git grep -n "cartridge_path" -- proxima-model-interop/src/serving_grammar.rs` prints nothing after FT2.2, so FT14 stops on it. Amendment: the FT14 card that adds the `Load` arm of `kind_of` also adds `Load { block_keys: Vec<String>, cartridge_path: Option<String> }` to `AssembleStep` in `serving_grammar.rs` (the enum already derives serde with `#[serde(tag = "kind")]`, and derives only `Clone`, not `Copy`, so adding a payload variant breaks no caller), adds a `[[prefill.assemble]]` case for it to the `serving_settings_prefill_assemble_variants` test (FT2.12), and the arm that runs it, all in that one commit. A `Blend` variant arrives the same way with the card that binds its arm. A settings variant with no arm behind it is a technique's grammar with no hook, which this file refuses to build.
- FT7.4 and FT11.3 cite `round_trip::assert_three_ways` as FT2.9's helper. It is defined by FT2.27, which FT2.9 needs, so every card that needs FT2.9 has it. No amendment beyond reading the citation as FT2.27.
- FT6.11 (`06-read-sets.md`, the decode step binds the read rule's output): FT2.34 adds `ServingRefusal::ReadHookNotBound`, which refuses `attention.read = operand` in settings while no decode arm reads it. FT6.11 binds that arm, so its commit also deletes the `ReadHookNotBound` variant, its `check_read` method and call in `refusals()`, and the `serving_settings_refuses_read_hook_not_bound` test; rewrites the expected lists in the two `serving_settings_refuses_read_needs_` tests to drop the leading `ReadHookNotBound`; and drops row 14 from `serving_settings_refusal_field_paths` (14 rows become 13, 21 tests become 20), the exit alternations and the `21 passed` lines in this file. The variant, `check_read` and the hook-not-bound test live in FT2.34's files (`refusal.rs`, `refusals.rs`); the two `serving_settings_refuses_read_needs_` tests live in FT2.17's. Until that commit lands, settings that load `read = "operand"` are refused by `validate()`, so no configuration runs a dense read under an operand cache key.

## preconditions (an executor that finds one false stops and reports)

- `ServingConfig` has 58 fields at a7c08c4c (56 unconditional; `math_mode` and `dispatch_type` under `#[cfg(all(feature = "metal", target_os = "macos"))]`). Check: `git show main:proxima-model-interop/src/serving.rs | sed -n 719,1090p | grep -c "^    pub "` prints 58. FT2.1 and FT2.2 raise it to 60.
- `ServingConfig` has six full struct literals with no `..`, all in `interop/serving.rs`: `impl Default for ServingConfig<'static>` (~line 1136), `tests::fully_supported_config_applies_without_error` (~line 1567) and the four `via_full_literal` literals (~lines 1744, 1846, 1938, 2020). Every other literal in the repo spreads `..ServingConfig::default()`. The compiler lists the missing sites (`E0063`); a seventh site in another file is added the same way and reported.
- The profiles module exists (`interop/profiles/mod.rs::family_profile`, `pub mod profiles` under `std`, `lib.rs` ~line 66). The previous cut's precondition on the architecture-as-data config slice is not needed and is dropped.
- Pattern verified in a scratch crate before these cards were written (conflaguration at the pinned rev, `bon`, `serde`, `toml`, `temp-env`; scratch removed): `#[derive(Settings)]` with `resolve_with = "from_json"` on `Vec<enum>` and on `Option<enum>` (`default`), `default_str` carrying a JSON list, a nested section with no struct-level prefix two levels deep (env `PARENT_CHILD_GRANDCHILD_FIELD`), unattributed `Option<u32>` and `Option<u64>` fields (unset env is `None`), `#[serde(default)]` on every section struct, TOML tables, and `Settings` without a `Validate` impl.

## decisions made in these cards (each with its one-line why)

- D1. List-valued sections follow the two-rule split above. Why: `ServingConfig` must stay `Copy`; a borrowed slice is `Copy`, and `weight_precision` is the precedent.
- D2. `ServingSettings::as_serving_config<'a>(&'a self, weight_precision: &'a [WeightPrecisionRule<'a>]) -> ServingConfig<'a>` and `weight_precision_rules(&self) -> Vec<WeightPrecisionRule<'_>>`. Why: a `&'a [Rule<'a>]` cannot borrow from `&self` (self-referential); a `Vec` returned by a method needs no new type.
- D3. Structured fields (lists, tagged enums, `numeric_policy`, `rope_scaling`) load from env as one JSON value (`from_json`); unit-variant enums load from env as a bare word (`from_name`). TOML is native tables. Why: `serde_json` is a dependency already (`Cargo.toml` `serde_json.workspace = true`), and `interop/speculative_settings.rs` already uses one textual form per setting through a `resolve_with` function. Each section file imports the two functions with `use super::{from_json, from_name};` and names them bare in `resolve_with`.
- D4. `context_length: ContextLength` loads as two scalars: `context_length: u32` (0 is `Native`) and `context_extrapolate: bool`. Why: `ContextLength` has no serde and `0` is the crate's sentinel for unbounded (`dense_weights_budget_bytes`, `max_concurrent_requests`).
- D5. Foreign enums without serde are mirrored the way `SpeculativeTypeName` mirrors `SpeculativeType` (`interop/speculative_settings.rs`, ~line 36): `CacheType` (FT2.3) for `GgmlType`, `MathModeName` (FT2.5) for `omega::MathMode`, `DispatchTypeName` (FT2.31) for `omega::DispatchType`. `GdnPrefillBackend` and `RopeScaling` (both in this crate, small `Copy` enums) get a serde derive instead. Why: a mirror earns its type only when the original is foreign or a 35-variant `#[non_exhaustive]` wire enum.
- D6. `speculative` and `prompt_cache` nest with `override_prefix`, so `PROXIMA_SPECULATIVE_*` and `PROXIMA_PROMPT_CACHE_*` keep working (`PROXIMA_PROMPT_CACHE_BYTE_BUDGET=0` is the documented off switch). Every other field is `PROXIMA_SERVING_<SECTION>_<FIELD>`.
- D7. The one block size is `kv.block_tokens` (default 64). `PromptCacheSettings.block_tokens` becomes `Option<u32>` (FT2.15a); standalone lowering takes `PromptCacheConfig::standard().block_tokens` (64) for `None`, so existing users are unchanged. Under `ServingSettings`, the lowering takes the explicit `prompt_cache.block_tokens` when one is set and `kv.block_tokens` otherwise (`Some(x)` wins, `None` takes `kv.block_tokens`), so an explicit value is never overwritten. `Some(x)` with `x != kv.block_tokens` is refused with `BlockTokensConflict` (FT2.16), because the kv value is then shadowed. Why: no second chunk size may exist, and an `Option` is the only way to tell explicit from default. FT2.15a (the `Option`) lands before FT2.9 (the first reader), so no commit writes `kv.block_tokens` over a value the user set; FT2.16 refuses the shadowing rather than repairing an override.
- D8. "the prewarm worker is configured" means `prompt_cache.byte_budget > 0 && prompt_cache.max_entries > 0 && prompt_cache.prewarm_chunk_tokens > 0` (`interop/serving.rs::PromptCacheConfig::is_enabled` ~line 596 and the `prewarm_chunk_tokens` doc: `0` is one chunk a request cannot preempt).
- D9. Descriptor rows live in `refusals_for(&ModelDescriptor)`, not in `validate()`, because `validate()` sees no model. Fixtures are shape-built descriptors, `mistral_descriptor_from_shape(.., &family_profile("gemma4")?)` over the first values of the vendored gemma4 headers, with `cache_strategy` and `layers[i].kind` set by field assignment (`ModelDescriptor` fields are public). Why: no model load, and `gemma4_descriptor_from_gguf` needs a parsed GGUF.
- D10. The phase-schedule refusal is one function, `interop/serving.rs::check_phase_schedule`, called by `apply_serving_config` and by the decode loop entry in the place where the inline block stands today. Why: at a7c08c4c the uncached path reaches that block before any `apply_serving_config` call (`interop/generate/prompt_cache.rs::run_decode_loop_through_cache` ~line 1221 returns into the loop before its own apply call at ~line 1237), so deleting the block would move the refusal later, not keep it.
- D11. Every section type has `impl Default` as `Self::builder().build()`, and `ServingSettings` and every section struct carry struct-level `#[serde(default)]`, so a TOML file may name only the sections it changes.
- D12. Test-name prefixes: `serving_settings_` is reserved for the 21 tests of the final count (FT2.21 lists them); helper tests use `serving_grammar_`, `serving_section_`, `serving_scalars_`, `text_form_` and `cache_key_`; the one model-loading test uses `decode_entry_`. Why: the exit count is an exact alternation, so sibling tests that also say `serving_settings_` cannot change it, and a model-loading test must not match the model-free filters.
- D13. No doctest anywhere under `interop/serving_settings.rs` or `interop/serving_settings/`.
- D14. Text written into the repository is English that describes behaviour. It never contains slice, stage, card, requirement or acceptance ids, or pointers into the spec files.
- D15. Visibility, so no card leaves it to the executor. Every section struct and enum is `pub`, and so is every field of a section struct. A lowering method that only the parent module `serving_settings.rs` calls (`as_admission_schedule`, `as_phase_schedule`, `as_expert_residency_schedule`, `as_prefill_config`) is `pub(super)`: a private method in a child module cannot be called from its parent. A lowering method a test or a caller outside the crate needs is `pub` (`CacheType::as_ggml`, `ServingSettings::as_serving_config`, `ServingSettings::weight_precision_rules`). Each `pub(super)` method has its first production caller in the same card, so none trips `dead_code`.
- D16. One new public type per card. A section that nests a leaf type is cut into the leaf card (the leaf is `pub`, re-exported, and its test is its use) and then the card that nests it. Why: the size rule allows at most one new public item that another card consumes, and a leaf with a public test is not dead code.

## card index (32 cards, in landing order)

Landing order across slices: FT2.32 to FT2.15a (the first 14 rows) need no card of another slice and land first. FT2.22 needs `EvictionRule` (FT5.7) and `DraftStep`/`IdleStep` (FT10.4), which are absent from main, so slices 5 and 10 run between FT2.15a and FT2.22: FT5.7 with the cards it needs, and FT10.4 with the cards it needs (FT10.1 to FT10.3), land after FT2.15a, then FT2.22 and every later row here (FT2.9, FT2.10, FT2.34, FT2.12, FT2.13 and onward, through FT2.21) land after them. FT2.18 also needs FT5.20, so it lands after FT5.20. None of those slice 5 and slice 10 cards cites a card of this file. The slice-level order in `../TASKS.md` states the same split.

| card | title | stages | files |
|---|---|---|---|
| FT2.32 | the `ReadSpec` enum | read | `serving_grammar.rs`, `lib.rs` |
| FT2.1 | the attention config with the read spec on `ServingConfig` | read | `serving.rs`, `lib.rs`, `generate/prompt_cache_key.rs`, `generate/resident_plans.rs` |
| FT2.33 | the `AssembleStep` enum | assemble | `serving_grammar.rs`, `lib.rs` |
| FT2.2 | the prefill config with the assemble list on `ServingConfig` | assemble | `serving.rs`, `lib.rs`, `generate/prompt_cache_key.rs`, `generate/resident_plans.rs` |
| FT2.3 | the `CacheType` mirror enum | settings | `serving_settings.rs`, `lib.rs` |
| FT2.4 | `ServingSettings` skeleton and the first 15 fields | settings | `serving_settings.rs`, `lib.rs`, `rope_scaling.rs` |
| FT2.5 | sampling, math mode mirror, numeric policy | sample | `serving_settings.rs` |
| FT2.31 | the `DispatchTypeName` mirror and `dispatch_type` | sample | `serving_settings.rs` |
| FT2.6 | weight precision, gdn backend, correctness fallback | bind | `serving_settings.rs`, `serving.rs`, `lib.rs` |
| FT2.7 | the ten runtime switches | specialize | `serving_settings.rs` |
| FT2.8 | admission level | schedule | `serving_settings.rs`, `serving_settings/levels.rs`, `lib.rs` |
| FT2.25 | phase level | step | `serving_settings.rs`, `serving_settings/levels.rs`, `lib.rs` |
| FT2.26 | expert residency level | schedule | `serving_settings.rs`, `serving_settings/levels.rs`, `lib.rs` |
| FT2.27 | speculative and prompt cache nested, and the shared three-way helper | schedule, seal and tier | `serving_settings.rs` |
| FT2.15a | `PromptCacheSettings.block_tokens` becomes `Option<u32>` | seal and tier | `prompt_cache_settings.rs` |
| FT2.22 | eviction rules and idle steps loadable from text | seal and tier, schedule | `generate/prompt_cache.rs`, `generate/prewarm_follow_up.rs` |
| FT2.9 | kv section: block size, eviction rules | seal and tier | `serving_settings/kv.rs`, `serving_settings.rs`, `lib.rs` |
| FT2.10 | attention section leaf | read | `serving_settings/attention.rs`, `serving_settings.rs`, `lib.rs` |
| FT2.34 | nest the attention section; refusal enum, `Validate`, unbound read row | read | `serving_settings/refusal.rs`, `serving_settings/refusals.rs`, `serving_settings.rs` |
| FT2.12 | prefill section | assemble | `serving_settings/prefill.rs`, `serving_settings.rs`, `lib.rs` |
| FT2.13 | schedule section | schedule | `serving_settings/schedule.rs`, `serving_settings.rs`, `lib.rs` |
| FT2.29 | expert placement leaf | place | `serving_settings/placement.rs`, `serving_settings.rs`, `lib.rs` |
| FT2.30 | placement budget leaf | place | `serving_settings/placement.rs`, `serving_settings.rs`, `lib.rs` |
| FT2.23 | placement section nesting the two leaves | place | `serving_settings/placement.rs`, `serving_settings.rs`, `lib.rs` |
| FT2.15 | whole-surface round trip and default parity | settings | `serving_settings.rs` |
| FT2.16 | block-size conflict row | settings | `serving_settings/refusal.rs`, `serving_settings/refusals.rs` |
| FT2.17 | read rows (attention layers, two-range cache) | read | same two files |
| FT2.18 | assemble, shift and eviction rows | assemble, seal and tier | same two files |
| FT2.19 | idle-step rows (worker, entry room) | schedule | same two files |
| FT2.24 | placement rows | place | same two files |
| FT2.20 | phase interleave row, one shared refusal | step | `serving_settings/refusal.rs`, `serving_settings/refusals.rs`, `serving.rs`, `generate/decode.rs`, and the test file `tests/phase_schedule_refusal.rs` |
| FT2.21 | refusal field paths and the exit count | settings | `serving_settings/refusals.rs` |

Every `ServingConfig` field and the card that lowers it (58 existing fields plus 2 new = 60):

| ServingConfig field | settings field | card |
|---|---|---|
| model_path | model_path (String) | FT2.4 |
| context_length | context_length (u32), context_extrapolate (bool) | FT2.4 |
| rope_scaling | rope_scaling (Option<RopeScaling>) | FT2.4 |
| parallel_sequences | parallel_sequences | FT2.4 |
| kv_cache_key_quant, kv_cache_value_quant | same names (CacheType) | FT2.4 |
| flash_attention, batch_size, ubatch_size, gpu_layers | same names | FT2.4 |
| gpu_memory_fit, gpu_memory_limit_bytes, kv_offload, multimodal_projector, reasoning_budget | same names | FT2.4 |
| temperature, top_k, top_p, min_p, repeat_last_n, repeat_penalty, frequency_penalty, presence_penalty, seed | same names | FT2.5 |
| kv_bucket_tokens, exact_activations, numeric_policy | same names | FT2.5 |
| math_mode (under metal and macos) | math_mode (MathModeName), same cfg | FT2.5 |
| dispatch_type (under metal and macos) | dispatch_type (DispatchTypeName), same cfg | FT2.31 |
| weight_precision | weight_precision (Vec<WeightPrecisionRuleSettings>) | FT2.6 |
| gdn_prefill_backend, gpu_correctness_fallback | same names | FT2.6 |
| qwen35moe_pre_gather, qwen35moe_persistent_cuts, qwen35moe_residency_budget_bytes, qwen35moe_expert_prefetch, qwen35moe_monolithic_all_low, qwen35moe_layer_window, qwen35moe_monolithic_high_mmap | placement.experts.{pre_gather, persistent_cuts, residency_budget_bytes, prefetch, monolithic_all_low, layer_window, monolithic_high_mmap} | FT2.23 |
| dense_weights_budget_bytes, expert_weights_budget_bytes, activations_budget_bytes, kv_cache_budget_bytes | placement.budget.{dense_weights_bytes, expert_weights_bytes, activations_bytes, kv_cache_bytes} | FT2.23 |
| prefill_one_evaluation, prefill_chunk_positions, cached_attention_fusion, gated_delta_net_fusion, moe_topk_fusion, plan_time_constants, plan_refit, command_buffer_chunks, max_command_buffers_per_token, overlap_transfer_compute | same names | FT2.7 |
| admission_schedule | nested struct of the same name | FT2.8 |
| phase_schedule | nested struct of the same name | FT2.25 |
| expert_residency_schedule | nested struct of the same name | FT2.26 |
| speculative, prompt_cache | nested `SpeculativeSettings`, `PromptCacheSettings` | FT2.27 |
| attention (new) | attention.read | FT2.34 (the section is FT2.10) |
| prefill (new) | prefill.assemble | FT2.12 |
| none (settings only) | kv.{block_tokens, eviction}; schedule.idle | FT2.9, FT2.13 |

Count check: 15 (FT2.4) + 13 (FT2.5) + 1 (FT2.31) + 3 (FT2.6) + 11 (FT2.23) + 10 (FT2.7) + 1 (FT2.8) + 1 (FT2.25) + 1 (FT2.26) + 2 (FT2.27) = 58, plus `attention` and `prefill` = 60. `kv.block_tokens` lowers into `prompt_cache.block_tokens`.

## cards

### 2.32 Add the `ReadSpec` enum

- id: FT2.32
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std for the serde test; the edit is alloc-tier)
- read first:
  - `interop/rope_scaling.rs::RopeScaling` (~line 41): the house shape of a small `Copy` config enum;
  - `interop/lib.rs` (~lines 70 and 146): the `mod serving;` line and the `pub use rope_scaling::RopeScaling;` line this card sits beside;
  - `interop/speculative_settings.rs::SpeculativeTypeName` (~line 36): the serde derive style on an enum.
- change (everything written into source is plain English, no ids, no spec pointers):
  1. `proxima-model-interop/src/serving_grammar.rs` (new, ungated; `use serde::{Deserialize, Serialize};`): `#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)] #[serde(rename_all = "snake_case")] pub enum ReadSpec { #[default] Dense, Operand }`. Docs: `Dense` "every cached row is read, today's behaviour"; `Operand` "each attention layer declares a per-row skip operand that is filled every decode step from the read rule the loaded model carries, so the rule decides which cached rows a step reads; until the decode step binds that rule, a request that sets it reads every row".
  2. `proxima-model-interop/src/lib.rs`: add `mod serving_grammar;` (ungated) beside `mod serving;` (~line 70); `pub use serving_grammar::ReadSpec;` beside `pub use rope_scaling::RopeScaling;` (~line 146).
- test: add `serving_grammar_read_spec_round_trips_json` in `serving_grammar.rs` `tests` (`serde_json`): `"dense"` equals `ReadSpec::Dense` and `"operand"` equals `ReadSpec::Operand`; `serde_json::to_string` of each returns the same quoted word; `ReadSpec::default() == ReadSpec::Dense`; sad: `"block"` and `"Dense"` are `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_32 cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_grammar_read_spec_round_trips_json/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --no-default-features`
- stage: `proxima-model-interop/src/serving_grammar.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add the attention read spec values`
- done when: the expect line printed, clippy and the check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a variant beyond `Dense` and `Operand`; add `AttentionConfig` or any other type (FT2.1); put a float or a list in `ReadSpec`; touch `serving.rs`.
- gpu: none

### 2.1 Add the attention config with the read spec to the serving config

- id: FT2.1
- needs: FT2.32
- budget: 20 min
- crate(s): proxima-model-interop (features: std for the keyed-type tests; the edit is alloc-tier)
- read first:
  - `interop/serving.rs::ServingConfig` (~line 720, last field `prompt_cache` ~line 1082) and `impl Default for ServingConfig<'static>` (~line 1136); the six full literals are listed in the preconditions;
  - `interop/generate/prompt_cache_key.rs::CacheKey` (~line 32) and `::CacheKey::of` (~line 82): the key struct and the exhaustive destructure of `ServingConfig`, whose module doc says a field is either in the key or named `_` with the reason;
  - `interop/generate/resident_plans.rs::PlanIdentity::of` (~line 80): the second exhaustive destructure;
  - `interop/serving_grammar.rs::ReadSpec` (FT2.32): the type the new field holds.
- change (everything written into source is plain English, no ids, no spec pointers):
  1. `proxima-model-interop/src/serving.rs`: `use crate::serving_grammar::ReadSpec;`, add `#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)] pub struct AttentionConfig { pub read: ReadSpec }` (field doc: "which cached rows a decode step reads"), append `pub attention: AttentionConfig` to `ServingConfig` after `prompt_cache`, `attention: AttentionConfig::default()` to `impl Default` and to the other five full literals. Module doc (~line 17, the paragraph that starts "No `serde`/`toml`/`bon`/`clap`/`conflaguration`"): append "The section grammar in `serving_grammar.rs` derives serde; the `Copy` structs in this module do not."
  2. `interop/generate/prompt_cache_key.rs`: import `AttentionConfig` and `crate::serving_grammar::ReadSpec`; `CacheKey` gains `pub(super) read: ReadSpec` (doc: "which cached rows a decode step reads; rows computed under a skipped read are not the rows a dense read computes"); `CacheKey::of` destructures `attention: AttentionConfig { read },` and stores it.
  3. `interop/generate/resident_plans.rs::PlanIdentity::of`: add `attention: _,` with the one-line comment `// a skipped read has no lowering yet, so every resident plan is a dense plan`.
  4. `proxima-model-interop/src/lib.rs`: add `AttentionConfig` to the `pub use serving::{..}` list (~line 149).
- test: two tests.
  - `serving_section_attention_defaults_to_the_dense_read` in `serving.rs` `tests`: `ServingConfig::default().attention == AttentionConfig { read: ReadSpec::Dense }`.
  - `cache_key_separates_read_specs` in a new `#[cfg(test)] mod tests` of `prompt_cache_key.rs`: with `dense = ServingConfig::default()` and `operand = ServingConfig { attention: AttentionConfig { read: ReadSpec::Operand }, ..dense }`, `CacheKey::of(&dense, false, RopeScaling::None, 0, 0) != CacheKey::of(&operand, false, RopeScaling::None, 0, 0)` and `CacheKey::of(&dense, ..) == CacheKey::of(&ServingConfig::default(), ..)`.
- validate: run `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_1 cargo nextest run -p proxima-model-interop --features std -E 'test(/^serving::tests::/)'` before editing and again after editing, then `cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_section_attention_defaults|cache_key_separates_read_specs/)'`
- expect: before: `27 passed`; after: `28 passed`; then `2 passed`. The 27 is derived by reading `serving.rs` `mod tests` on main (22 plain `#[test]` functions, plus 2 `#[proxima::test]` functions that expand to one test per `#[case]` row, 2 + 3 = 5); it was not run. If the before run prints another number, stop and report it.
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --no-default-features`; `cargo check -p proxima-model-interop --features std,metal --all-targets` (compiles every `ServingConfig { .. }` literal in examples and benches)
- stage: `proxima-model-interop/src/serving.rs`, `proxima-model-interop/src/lib.rs`, `proxima-model-interop/src/generate/prompt_cache_key.rs`, `proxima-model-interop/src/generate/resident_plans.rs`
- commit: `feat(interop): add the attention read spec to the serving config`
- done when: the expect lines printed, clippy and both checks clean, `git diff --cached --stat` equals the stage list (four files, two of them one-line mechanical edits: a new config field must reach every exhaustive destructure in one commit), and the commit landed with that message
- do not: add a variant to `ReadSpec`; add a second public type (`AttentionConfig` is the only one); add the prefill section; derive serde on any struct in `serving.rs`; touch `decode.rs`; touch `serving_grammar.rs`.
- gpu: none

### 2.33 Add the `AssembleStep` enum

- id: FT2.33
- needs: FT2.32
- budget: 20 min
- crate(s): proxima-model-interop (features: std for the serde test; the edit is alloc-tier)
- read first:
  - `interop/serving_grammar.rs::ReadSpec` (FT2.32): the file this card extends, same derive style;
  - `interop/generate/prompt_cache.rs::run_decode_loop_through_cache` (~line 1196, `shifting` at ~line 1055 inside `prompt_cache_lookup`): the two stages the lookup runs today, `cacheable` at ~line 1213, the shift gate `config.cache_reuse_min > 0 && config.ring_rewind_slack > 0`;
  - `interop/lib.rs` (~line 146): the `pub use serving_grammar::ReadSpec;` line this card extends.
- change (English, no ids):
  1. `serving_grammar.rs`: `#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)] #[serde(tag = "kind", rename_all = "snake_case")] pub enum AssembleStep { Prefix, Shift }` (`Clone` and not `Copy`: a later variant may carry a payload). Docs: `Prefix` "start from the stored entry that shares the longest prefix with the prompt"; `Shift` "also reuse stored chunks that moved position; needs the prompt cache's `cache_reuse_min` and `ring_rewind_slack` above 0". These are the two stages the lookup in `prompt_cache.rs` runs today; the enum names no stage the library does not run.
  2. `lib.rs`: `pub use serving_grammar::{AssembleStep, ReadSpec};`.
- test: add `serving_grammar_assemble_steps_round_trip_json` in `serving_grammar.rs` `tests`: `{"kind":"prefix"}` equals `AssembleStep::Prefix`; `{"kind":"shift"}` equals `Shift`; `[{"kind":"shift"},{"kind":"prefix"}]` parses to `vec![Shift, Prefix]` (order kept); each re-serializes to a value that parses back equal; sad: `{"kind":"load","path":"/models/cartridges/legal-v3.cart"}`, `{"kind":"blend"}` and `{"kind":"Prefix"}` are `Err` (a stage the library does not run is not in the grammar).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_33 cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_grammar_assemble_steps_round_trip_json/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --no-default-features`
- stage: `proxima-model-interop/src/serving_grammar.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add the assemble stage values`
- done when: the expect line printed, clippy and the check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a `Load` or `Blend` variant, or any variant a library arm does not run today (a variant lands in the card that binds its arm); add `PrefillConfig` (FT2.2); touch `serving.rs`.
- gpu: none

### 2.2 Add the prefill config with the assemble list to the serving config

- id: FT2.2
- needs: FT2.1, FT2.33
- budget: 20 min
- crate(s): proxima-model-interop (features: std for the keyed-type tests; the edit is alloc-tier)
- read first:
  - `interop/serving_grammar.rs::AssembleStep` (FT2.33): the element type of the list;
  - `interop/serving.rs::ServingConfig`, `::AttentionConfig` and `impl Default for ServingConfig<'static>` (FT2.1 extended them);
  - `interop/generate/prompt_cache_key.rs::CacheKey::of` and `generate/resident_plans.rs::PlanIdentity::of`: the two destructures;
  - `interop/serving.rs::ServingConfig::weight_precision` (~line 760): the existing borrowed-slice field, `&'model [WeightPrecisionRule<'model>]`.
- change (English, no ids):
  1. `serving.rs`: `use crate::serving_grammar::AssembleStep;`, add `#[derive(Debug, Clone, Copy, PartialEq)] pub struct PrefillConfig<'model> { pub assemble: &'model [AssembleStep] }` with `impl Default for PrefillConfig<'static> { fn default() -> Self { Self { assemble: &[] } } }`. Field doc: "ordered stages that build the starting cache of a request; empty is today's behaviour: prefix reuse, plus shifted-chunk reuse when the prompt cache's `cache_reuse_min` and `ring_rewind_slack` are above 0". Append `pub prefill: PrefillConfig<'model>` after `attention`; extend `impl Default` and the other five full literals with `prefill: PrefillConfig::default()`.
  2. `generate/prompt_cache_key.rs::CacheKey::of`: add `prefill: _,` with `// stages choose where rows come from; a stage that changes what a row means adds its identity to the key`. `generate/resident_plans.rs::PlanIdentity::of`: add `prefill: _,`.
  3. `lib.rs`: add `PrefillConfig` to the `pub use serving::{..}` list.
- test: two tests.
  - `serving_section_prefill_defaults_to_todays_assemble` in `serving.rs` `tests`: `ServingConfig::default().prefill.assemble.is_empty()`.
  - `cache_key_ignores_the_assemble_list` in `prompt_cache_key.rs` `tests`: with `steps = [AssembleStep::Prefix, AssembleStep::Shift]` and `config = ServingConfig { prefill: PrefillConfig { assemble: &steps }, ..ServingConfig::default() }`, `CacheKey::of(&config, false, RopeScaling::None, 0, 0) == CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0)`.
- validate: run `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_2 cargo nextest run -p proxima-model-interop --features std -E 'test(/^serving::tests::/)'` before editing and again after editing, then `cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_section_prefill_defaults|cache_key_ignores_the_assemble_list/)'`
- expect: before: `28 passed`; after: `29 passed`; then `2 passed`. The 28 is derived: the 27 tests of `serving::tests` on main plus the one test the previous card added; it was not run. If the before run prints another number, stop and report it.
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --no-default-features`; `cargo check -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/serving.rs`, `proxima-model-interop/src/lib.rs`, `proxima-model-interop/src/generate/prompt_cache_key.rs`, `proxima-model-interop/src/generate/resident_plans.rs`
- commit: `feat(interop): add the assemble stage list to the serving config`
- done when: the expect lines printed, clippy and both checks clean, `git diff --cached --stat` equals the stage list (four files, same reason as the previous card), and the commit landed with that message
- do not: add a second public type (`PrefillConfig` is the only one); add a variant to `AssembleStep`; make the decode loop read the list (the assemble slice does); touch `decode.rs` or `serving_grammar.rs`.
- gpu: none

### 2.3 Add the `CacheType` mirror enum for the kv cache quant settings

- id: FT2.3
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/speculative_settings.rs::SpeculativeTypeName` (~line 36): a serde mirror of a foreign enum with explicit `#[serde(rename = ..)]` per variant;
  - `proxima-gguf/src/types.rs::GgmlType` (~line 109): the enum mirrored; premise: it has the variants `F32`, `F16`, `Bf16`, `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Iq4Nl` and derives `Debug, Clone, Copy, PartialEq, Eq`; if one is missing, stop and report;
  - `interop/serving.rs::ServingConfig::kv_cache_key_quant` (~line 738): the field this type will lower into (llama `--cache-type-k`).
- change:
  1. `proxima-model-interop/src/serving_settings.rs` (new). Imports `serde::{Deserialize, Serialize}` and `proxima_gguf::types::GgmlType`. Add `#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)] pub enum CacheType { F32, F16, Bf16, Q80, Q40, Q41, Q50, Q51, Iq4Nl }` (no underscores in variant names, so no lint allowance), each with an explicit `#[serde(rename = "...")]`: `"f32"`, `"f16"`, `"bf16"`, `"q8_0"`, `"q4_0"`, `"q4_1"`, `"q5_0"`, `"q5_1"`, `"iq4_nl"`; and `impl CacheType { pub const fn as_ggml(self) -> GgmlType }` mapping to `GgmlType::{F32, F16, Bf16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Iq4Nl}`. Type doc: "the values llama accepts for `--cache-type-k` and `--cache-type-v`, spelled as llama spells them; `as_ggml` is the lowering to the type the serving config holds". Mirrored, not derived on `GgmlType`, because that enum is foreign and `#[non_exhaustive]` (D5).
  2. `proxima-model-interop/src/lib.rs`: add `#[cfg(feature = "std")] mod serving_settings;` beside the `speculative_settings` line (~line 76) and `#[cfg(feature = "std")] pub use serving_settings::CacheType;` beside ~line 157.
- test: add `serving_section_cache_type_round_trips_json` in `serving_settings.rs` `tests` (`serde_json`). A table of nine rows `(word, variant, ggml)`: `("f32", F32, GgmlType::F32)`, `("f16", F16, GgmlType::F16)`, `("bf16", Bf16, GgmlType::Bf16)`, `("q8_0", Q80, GgmlType::Q8_0)`, `("q4_0", Q40, GgmlType::Q4_0)`, `("q4_1", Q41, GgmlType::Q4_1)`, `("q5_0", Q50, GgmlType::Q5_0)`, `("q5_1", Q51, GgmlType::Q5_1)`, `("iq4_nl", Iq4Nl, GgmlType::Iq4Nl)`. Assert the table length is 9 first; for each row the quoted word parses to the variant, the variant serializes back to the same quoted word, and `variant.as_ggml() == ggml`. Sad: `"q9_9"`, `"Q8_0"` and `"q8"` are `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_3 cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_section_cache_type_round_trips_json/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --no-default-features`
- stage: `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add the kv cache type setting values`
- done when: the expect line printed, clippy and the check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add `ServingSettings` or any other type; derive serde on `GgmlType`; add a quant llama does not accept for the kv cache; add a doctest.
- gpu: none

### 2.4 Create `ServingSettings` with its first 15 fields

- id: FT2.4
- needs: FT2.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/speculative_settings.rs::SpeculativeSettings` (~line 256) and `::as_speculative_config` (~line 346): the derive and lowering pattern;
  - `interop/prompt_cache_settings.rs::PromptCacheSettings` (~line 20): the `Builder, Deserialize, Serialize, Settings` derive order;
  - `interop/serving.rs::ServingConfig` fields `model_path` through `reasoning_budget` (~lines 725 to 790) and `DEFAULT_MODEL_PATH` (~line 622), `DEFAULT_GPU_LAYERS` (~line 145), `GPU_LAYERS_ALL` (~line 139);
  - `interop/serving_settings.rs::CacheType` (FT2.3): the file this card extends and the type its two quant fields use.
- change:
  1. `interop/rope_scaling.rs::RopeScaling` (~line 41): add `serde::Serialize, serde::Deserialize` to its derive list and `#[serde(tag = "kind", rename_all = "snake_case")]` (D5).
  2. `proxima-model-interop/src/serving_settings.rs` (created by FT2.3; it already holds `CacheType`). Add the imports as in `speculative_settings.rs` plus `crate::serving::{ContextLength, DEFAULT_GPU_LAYERS, DEFAULT_MODEL_PATH, ServingConfig, WeightPrecisionRule}` and `crate::RopeScaling`. Add:
     - `fn from_json<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> { serde_json::from_str(raw) }`;
     - `fn from_name<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> { serde_json::from_value(serde_json::Value::String(raw.to_owned())) }`;
     - `ServingSettings` with `#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]`, `#[settings(prefix = "PROXIMA_SERVING")]`, `#[builder(derive(Clone, Debug))]`, `#[serde(default)]`, and `impl Default for ServingSettings { fn default() -> Self { Self::builder().build() } }`. Fields in this order, each with a `#[setting(..)]` and a `#[builder(default = ..)]` carrying the same value:

       | field | type | default | setting attribute |
       |---|---|---|---|
       | model_path | String | `DEFAULT_MODEL_PATH.to_owned()` | `default = "<the DEFAULT_MODEL_PATH literal>"` |
       | context_length | u32 | 0 | `default = 0` |
       | context_extrapolate | bool | false | `default = false` |
       | rope_scaling | Option<RopeScaling> | None | `resolve_with = "from_json", default` |
       | parallel_sequences | u32 | 1 | `default = 1` |
       | kv_cache_key_quant | CacheType | F32 | `resolve_with = "from_name", default_str = "f32"` |
       | kv_cache_value_quant | CacheType | F32 | `resolve_with = "from_name", default_str = "f32"` |
       | flash_attention | bool | false | `default = false` |
       | batch_size | u32 | 32 | `default = 32` |
       | ubatch_size | u32 | 32 | `default = 32` |
       | gpu_layers | i32 | `DEFAULT_GPU_LAYERS` | two `cfg_attr` lines, below |
       | gpu_memory_fit | bool | true | `default = true` |
       | gpu_memory_limit_bytes | Option<u64> | None | none (an unattributed `Option` resolves to `None`) |
       | kv_offload | bool | false | `default = false` |
       | multimodal_projector | bool | false | `default = false` |
       | reasoning_budget | i32 | 0 | `default = 0` |

       `#[setting(default = ..)]` cannot read a const, so `gpu_layers` carries `#[cfg_attr(feature = "metal", setting(default = -1))]` and `#[cfg_attr(not(feature = "metal"), setting(default = 0))]` plus `#[builder(default = DEFAULT_GPU_LAYERS)]`; the default-parity test (FT2.15) pins the two together. Every field has a one-line doc naming its `ServingConfig` field and llama flag (copy the flag from the `ServingConfig` doc); `serde` renames nothing.
     - `impl ServingSettings`: `#[must_use] pub fn as_serving_config<'a>(&'a self, weight_precision: &'a [WeightPrecisionRule<'a>]) -> ServingConfig<'a>` returning `ServingConfig { model_path: &self.model_path, context_length: self.lowered_context_length(), rope_scaling: self.rope_scaling, parallel_sequences: self.parallel_sequences, kv_cache_key_quant: self.kv_cache_key_quant.as_ggml(), kv_cache_value_quant: self.kv_cache_value_quant.as_ggml(), flash_attention: self.flash_attention, batch_size: self.batch_size, ubatch_size: self.ubatch_size, gpu_layers: self.gpu_layers, gpu_memory_fit: self.gpu_memory_fit, gpu_memory_limit_bytes: self.gpu_memory_limit_bytes, kv_offload: self.kv_offload, multimodal_projector: self.multimodal_projector, reasoning_budget: self.reasoning_budget, weight_precision, ..ServingConfig::default() }` and `const fn lowered_context_length(&self) -> ContextLength` = `match (self.context_length, self.context_extrapolate) { (0, _) => ContextLength::Native, (length, false) => ContextLength::Within(length), (length, true) => ContextLength::Extrapolate(length) }`.
  3. `proxima-model-interop/src/lib.rs`: extend the `pub use serving_settings::CacheType;` line FT2.3 added to `pub use serving_settings::{CacheType, ServingSettings};`.
- test: add `serving_scalars_first_fifteen_lower_and_round_trip` in `serving_settings.rs` `tests` (`temp-env` and `tempfile` are dev-dependencies). Arrange TOML (top-level keys first, the table last): `model_path = "/models/gemma4-e2b-it-qat.gguf"`, `context_length = 32768`, `context_extrapolate = true`, `parallel_sequences = 2`, `kv_cache_key_quant = "q8_0"`, `kv_cache_value_quant = "q4_0"`, `flash_attention = true`, `batch_size = 512`, `ubatch_size = 128`, `gpu_layers = 99`, `gpu_memory_fit = false`, `gpu_memory_limit_bytes = 17179869184`, `kv_offload = true`, `multimodal_projector = true`, `reasoning_budget = 1024`, then `[rope_scaling]` with `kind = "yarn"`, `factor = 4.0`, `original_context = 32768`, `extrapolation_factor = 1.0`, `attention_factor = 1.1386294`, `beta_fast = 32.0`, `beta_slow = 1.0`. Act: `conflaguration::from_toml_str::<ServingSettings>` (a), the builder with the same values (b), and `temp_env::with_vars` over `PROXIMA_SERVING_MODEL_PATH`, `PROXIMA_SERVING_CONTEXT_LENGTH=32768`, `PROXIMA_SERVING_CONTEXT_EXTRAPOLATE=true`, `PROXIMA_SERVING_PARALLEL_SEQUENCES=2`, `PROXIMA_SERVING_KV_CACHE_KEY_QUANT=q8_0`, `PROXIMA_SERVING_KV_CACHE_VALUE_QUANT=q4_0`, `PROXIMA_SERVING_FLASH_ATTENTION=true`, `PROXIMA_SERVING_BATCH_SIZE=512`, `PROXIMA_SERVING_UBATCH_SIZE=128`, `PROXIMA_SERVING_GPU_LAYERS=99`, `PROXIMA_SERVING_GPU_MEMORY_FIT=false`, `PROXIMA_SERVING_GPU_MEMORY_LIMIT_BYTES=17179869184`, `PROXIMA_SERVING_KV_OFFLOAD=true`, `PROXIMA_SERVING_MULTIMODAL_PROJECTOR=true`, `PROXIMA_SERVING_REASONING_BUDGET=1024`, `PROXIMA_SERVING_ROPE_SCALING={"kind":"yarn","factor":4.0,"original_context":32768,"extrapolation_factor":1.0,"attention_factor":1.1386294,"beta_fast":32.0,"beta_slow":1.0}`, then `ServingSettings::from_env()` (c). Assert `a == b` and `c == b`. Assert on `b.as_serving_config(&[])`: `context_length == ContextLength::Extrapolate(32768)`, `kv_cache_key_quant == GgmlType::Q8_0`, `kv_cache_value_quant == GgmlType::Q4_0`, `gpu_layers == 99`, `gpu_memory_limit_bytes == Some(17_179_869_184)`, `rope_scaling == Some(RopeScaling::Yarn { factor: 4.0, original_context: 32768, extrapolation_factor: 1.0, attention_factor: 1.1386294, beta_fast: 32.0, beta_slow: 1.0 })`, `model_path == "/models/gemma4-e2b-it-qat.gguf"`. Sad path in the same test: TOML `kv_cache_key_quant = "q9_9"` is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_4 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_first_fifteen_lower_and_round_trip/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo nextest run -p proxima-model-interop --features std -E 'test(/rope_scaling::/)'` prints `8 passed` (derived by reading `rope_scaling.rs` `mod tests` on main: 4 `#[proxima::test]` cases, `rope_scaling_override_replaces_the_gguf_value`, 2 `yarn_inv_freq_` tests and `yarn_attention_factor_is_point_one_ln_factor_plus_one`; this card adds none; not run)
- stage: `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`, `proxima-model-interop/src/rope_scaling.rs`
- commit: `feat(interop): add serving settings with the llama flag fields`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add fields from FT2.5 onward; define a second `CacheType`; add a doctest; use a `serving_settings_` test prefix; add a `Validate` impl.
- gpu: none

### 2.5 Add the sampling fields, `kv_bucket_tokens`, the backend enums and `numeric_policy`

- id: FT2.5
- needs: FT2.4
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving.rs::ServingConfig` fields `temperature` through `exact_activations` (~lines 790 to 895) and their defaults in `impl Default` (~lines 1153 to 1177);
  - `omega/src/metal/pipeline_buffers_upload.rs::MathMode` (~line 34) and `::DispatchType` (~line 149): premise: `MathMode` has exactly `Safe`, `Relaxed`, `Fast`; `DispatchType` has exactly `Serial`, `Concurrent`. If either has another variant, stop and report;
  - `proxima-tensor/src/numeric.rs::NumericPolicy` (~line 24: `#[non_exhaustive]`, six public bool fields, serde under feature `config`) and `::llama_relaxed` (~line 81) and `::bit_exact`;
  - `interop/serving_settings.rs` (FT2.4).
- change: `proxima-model-interop/src/serving_settings.rs` only.
  1. Add, under `#[cfg(all(feature = "metal", target_os = "macos"))]`, beside `CacheType`, `#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)] #[serde(rename_all = "snake_case")] pub enum MathModeName { Safe, Relaxed, Fast }` with `const fn as_math_mode(self) -> omega::MathMode`. This is the one new public type of the card; the dispatch mirror is FT2.31.
  2. Append these fields to `ServingSettings`, in this order, same attribute rule as FT2.4:

     | field | type | default | setting attribute |
     |---|---|---|---|
     | temperature | f32 | 0.0 | `default = 0.0` |
     | top_k | i32 | 0 | `default = 0` |
     | top_p | f32 | 1.0 | `default = 1.0` |
     | min_p | f32 | 0.0 | `default = 0.0` |
     | repeat_last_n | i32 | 64 | `default = 64` |
     | repeat_penalty | f32 | 1.0 | `default = 1.0` |
     | frequency_penalty | f32 | 0.0 | `default = 0.0` |
     | presence_penalty | f32 | 0.0 | `default = 0.0` |
     | seed | u64 | 0 | `default = 0` |
     | kv_bucket_tokens | usize | 32 | `default = 32` |
     | math_mode (cfg metal and macos) | MathModeName | Relaxed | `resolve_with = "from_name", default_str = "relaxed"` |
     | numeric_policy | NumericPolicy | `NumericPolicy::llama_relaxed()` | `resolve_with = "from_json", default_str = "{\"contraction\":true,\"reassociation\":true,\"nan_assumptions\":false,\"signed_zero\":false,\"approx_functions\":false,\"epilogue_sources\":false}"` |
     | exact_activations | bool | true | `default = true` |

     The `math_mode` field carries the same `#[cfg(all(feature = "metal", target_os = "macos"))]` as the `ServingConfig` field it lowers.
  3. Extend the `as_serving_config` literal with `temperature`, `top_k`, `top_p`, `min_p`, `repeat_last_n`, `repeat_penalty`, `frequency_penalty`, `presence_penalty`, `seed`, `kv_bucket_tokens`, `numeric_policy: self.numeric_policy`, `exact_activations`, and under the cfg `math_mode: self.math_mode.as_math_mode()`.
- test: add `serving_scalars_sampling_and_policy_lower_and_round_trip` in `serving_settings.rs` `tests`. Arrange TOML: `temperature = 0.7`, `top_k = 40`, `top_p = 0.95`, `min_p = 0.05`, `repeat_last_n = 128`, `repeat_penalty = 1.1`, `frequency_penalty = 0.2`, `presence_penalty = 0.3`, `seed = 424242`, `kv_bucket_tokens = 64`, `exact_activations = false`, then `[numeric_policy]` with `contraction = true`, `reassociation = false`, `nan_assumptions = true`, `signed_zero = true`, `approx_functions = true`, `epilogue_sources = true`. The builder's policy is made from `NumericPolicy::bit_exact()` by assigning those six fields (the type is `#[non_exhaustive]`, so a struct literal is not allowed from this crate). Compare `from_toml_str`, the builder and `from_env` (`PROXIMA_SERVING_TEMPERATURE=0.7`, `PROXIMA_SERVING_TOP_K=40`, `PROXIMA_SERVING_TOP_P=0.95`, `PROXIMA_SERVING_MIN_P=0.05`, `PROXIMA_SERVING_REPEAT_LAST_N=128`, `PROXIMA_SERVING_REPEAT_PENALTY=1.1`, `PROXIMA_SERVING_FREQUENCY_PENALTY=0.2`, `PROXIMA_SERVING_PRESENCE_PENALTY=0.3`, `PROXIMA_SERVING_SEED=424242`, `PROXIMA_SERVING_KV_BUCKET_TOKENS=64`, `PROXIMA_SERVING_EXACT_ACTIVATIONS=false`, `PROXIMA_SERVING_NUMERIC_POLICY={"contraction":true,"reassociation":false,"nan_assumptions":true,"signed_zero":true,"approx_functions":true,"epilogue_sources":true}`): all three equal. Assert on the lowered config: `temperature == 0.7`, `top_k == 40`, `seed == 424242`, `kv_bucket_tokens == 64`, `exact_activations == false`, `numeric_policy.nan_assumptions == true`. Under `cfg(all(feature = "metal", target_os = "macos"))` the same TOML adds `math_mode = "fast"` (env `PROXIMA_SERVING_MATH_MODE=fast`) and asserts `omega::MathMode::Fast`. A second assertion on an unset env: `ServingSettings::from_env()` with every key above cleared has `numeric_policy == NumericPolicy::llama_relaxed()` and `kv_bucket_tokens == 32`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_5 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'`
- expect: `2 passed` (FT2.4's test and this one)
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo check -p proxima-model-interop --features std,metal --all-targets` (compiles the cfg fields)
- stage: `proxima-model-interop/src/serving_settings.rs`
- commit: `feat(interop): add sampling, backend and numeric policy settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add fields beyond the 13 listed; add `DispatchTypeName` or the `dispatch_type` field (FT2.31); add an `unwrap` outside tests; touch `serving.rs`.
- gpu: none

### 2.31 Add the `DispatchTypeName` mirror enum and the `dispatch_type` field

- id: FT2.31
- needs: FT2.5
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration,metal; every item here is under `#[cfg(all(feature = "metal", target_os = "macos"))]`)
- read first:
  - `omega/src/metal/pipeline_buffers_upload.rs::DispatchType` (~line 149): premise: exactly `Serial` and `Concurrent`. If it has another variant, stop and report;
  - `interop/serving.rs::ServingConfig::dispatch_type` (cfg metal and macos) and its default in `impl Default for ServingConfig<'static>`: the field this lowers into, and the default `Serial`;
  - `interop/serving_settings.rs::MathModeName` and the `math_mode` field (FT2.5): the shape to copy, one line each.
- change: `proxima-model-interop/src/serving_settings.rs` only.
  1. Beside `MathModeName`, under the cfg: `#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)] #[serde(rename_all = "snake_case")] pub enum DispatchTypeName { Serial, Concurrent }` with `const fn as_dispatch_type(self) -> omega::DispatchType`.
  2. Append to `ServingSettings`, under the cfg, `dispatch_type: DispatchTypeName` with `#[setting(resolve_with = "from_name", default_str = "serial")]` and `#[builder(default = DispatchTypeName::Serial)]`; extend the `as_serving_config` literal, under the cfg, with `dispatch_type: self.dispatch_type.as_dispatch_type()`.
- test: add `serving_scalars_dispatch_type_lowers_and_round_trips` in `serving_settings.rs` `tests`, under the same cfg. TOML `dispatch_type = "concurrent"`, env `PROXIMA_SERVING_DISPATCH_TYPE=concurrent`, builder `DispatchTypeName::Concurrent`: all three equal; `settings.as_serving_config(&[]).dispatch_type == omega::DispatchType::Concurrent`. `ServingSettings::default().as_serving_config(&[]).dispatch_type == omega::DispatchType::Serial`. Sad: `dispatch_type = "parallel"` is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_31 cargo nextest run -p proxima-model-interop --features std,conflaguration,metal -E 'test(/serving_scalars_dispatch_type_lowers_and_round_trips/)'`
- expect: `1 passed` (the test exists only under the metal cfg, so the non-metal filters of later cards do not count it)
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration,metal --all-targets`; `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings.rs`
- commit: `feat(interop): add the dispatch type serving setting`
- done when: the expect line printed, both clippy lines clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a second type; add a field other than `dispatch_type`; touch `serving.rs` or `omega`; run a GPU workload (the test lowers an enum and loads no model).
- gpu: none

### 2.6 Add `weight_precision`, the gdn backend and the correctness fallback

- id: FT2.6
- needs: FT2.31
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving.rs::WeightPrecisionRule` (~line 107) and `::NamePattern` (~line 65): the lowering target;
  - `interop/serving.rs::GdnPrefillBackend` (~line 155, `pub enum { Cpu, Mlx }`): gets a serde derive;
  - `interop/serving.rs::ServingConfig` fields `weight_precision`, `gdn_prefill_backend`, `gpu_correctness_fallback` (~lines 909, 918, 959);
  - `interop/serving_settings.rs::ServingSettings::as_serving_config` (FT2.4 passes `&[]`).
- change:
  1. `interop/serving.rs`: add `serde::Serialize, serde::Deserialize` to the derive of `GdnPrefillBackend` and `#[serde(rename_all = "snake_case")]`.
  2. `interop/serving_settings.rs`: add the one new public type, `#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)] #[serde(tag = "pattern_kind", rename_all = "snake_case")] pub enum WeightPrecisionRuleSettings { Exact { pattern: String, target: CacheType }, Prefix { pattern: String, target: CacheType }, Suffix { pattern: String, target: CacheType } }` (the tag word is the text form's `pattern_kind`; `target` reuses `CacheType`: the targets an encoder exists for are a subset of that vocabulary, and `InteropError::UnsupportedWeightPrecisionTarget` refuses the rest at bind time); and `pub fn weight_precision_rules(&self) -> Vec<WeightPrecisionRule<'_>>` on `ServingSettings`: one `WeightPrecisionRule { pattern: NamePattern::Exact(pattern) / NamePattern::Prefix(pattern) / NamePattern::Suffix(pattern), target: target.as_ggml() }` per entry, by an exhaustive `match` on the variant, in list order.
  3. Append fields to `ServingSettings`, in this order:

     | field | type | default | setting attribute |
     |---|---|---|---|
     | weight_precision | Vec<WeightPrecisionRuleSettings> | empty | `resolve_with = "from_json", default_str = "[]"` |
     | gdn_prefill_backend | GdnPrefillBackend | Cpu | `resolve_with = "from_name", default_str = "cpu"` |
     | gpu_correctness_fallback | bool | false | `default = false` |

     and extend the `as_serving_config` literal with `gdn_prefill_backend` and `gpu_correctness_fallback` (`weight_precision` is the argument already passed).
  4. `lib.rs`: extend the `pub use serving_settings::{..}` line with `WeightPrecisionRuleSettings`.
- test: add `serving_scalars_weight_precision_and_fallbacks_lower_and_round_trip` in `serving_settings.rs` `tests`. TOML (top-level keys first): `gdn_prefill_backend = "mlx"`, `gpu_correctness_fallback = true`, then two tables `[[weight_precision]]` with `pattern_kind = "suffix"`, `pattern = "ffn_down_exps.weight"`, `target = "q4_0"` and `pattern_kind = "exact"`, `pattern = "token_embd.weight"`, `target = "q8_0"` (tensor names read from `interop/gemma4/bind.rs` ~lines 57 to 130). Equal across TOML, builder and env (`PROXIMA_SERVING_WEIGHT_PRECISION=[{"pattern_kind":"suffix","pattern":"ffn_down_exps.weight","target":"q4_0"},{"pattern_kind":"exact","pattern":"token_embd.weight","target":"q8_0"}]`, `PROXIMA_SERVING_GDN_PREFILL_BACKEND=mlx`, `PROXIMA_SERVING_GPU_CORRECTNESS_FALLBACK=true`). Assert on `let rules = settings.weight_precision_rules(); let config = settings.as_serving_config(&rules);`: `config.weight_precision == [WeightPrecisionRule { pattern: NamePattern::Suffix("ffn_down_exps.weight"), target: GgmlType::Q4_0 }, WeightPrecisionRule { pattern: NamePattern::Exact("token_embd.weight"), target: GgmlType::Q8_0 }]`, `config.gdn_prefill_backend == GdnPrefillBackend::Mlx`, `config.gpu_correctness_fallback == true`. Sad: `pattern_kind = "glob"` is `Err`, and an `exact` rule with no `pattern` key is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_6 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo check -p proxima-model-interop --no-default-features`
- stage: `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/serving.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add weight precision and fallback serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `WeightPrecisionRule` or `NamePattern`; add a `Cow`; add a glob pattern kind; add a second public type beside `WeightPrecisionRuleSettings`; add the expert placement fields (FT2.23).
- gpu: none

### 2.7 Add the ten runtime switches

- id: FT2.7
- needs: FT2.6
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving.rs::ServingConfig` fields `prefill_one_evaluation` through `overlap_transfer_compute` (~lines 970 to 1062);
  - `interop/serving.rs` `impl Default for ServingConfig<'static>` (~lines 1193 to 1202): the defaults;
  - `interop/serving_settings.rs` (FT2.6).
- change: `proxima-model-interop/src/serving_settings.rs` only. Append to `ServingSettings` and to the `as_serving_config` literal:

  | field | type | default | setting attribute |
  |---|---|---|---|
  | prefill_one_evaluation | bool | false | `default = false` |
  | prefill_chunk_positions | usize | 0 | `default = 0` |
  | cached_attention_fusion | bool | true | `default = true` |
  | gated_delta_net_fusion | bool | true | `default = true` |
  | moe_topk_fusion | bool | true | `default = true` |
  | plan_time_constants | bool | true | `default = true` |
  | plan_refit | bool | true | `default = true` |
  | command_buffer_chunks | u32 | 1 | `default = 1` |
  | max_command_buffers_per_token | usize | 0 | `default = 0` |
  | overlap_transfer_compute | bool | false | `default = false` |
- test: add `serving_scalars_runtime_switches_lower_and_round_trip` in `serving_settings.rs` `tests`. TOML sets every field to the opposite of its default: `prefill_one_evaluation = true`, `prefill_chunk_positions = 256`, `cached_attention_fusion = false`, `gated_delta_net_fusion = false`, `moe_topk_fusion = false`, `plan_time_constants = false`, `plan_refit = false`, `command_buffer_chunks = 8`, `max_command_buffers_per_token = 12`, `overlap_transfer_compute = true`. Equal across TOML, builder and env (`PROXIMA_SERVING_PREFILL_ONE_EVALUATION=true` and the same pattern for the others). Assert the lowered config has each value; assert `ServingSettings::default().as_serving_config(&[])` equals `ServingConfig::default()` on these ten fields individually (not whole-struct equality: that is FT2.15).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_7 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'`
- expect: `4 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings.rs`
- commit: `feat(interop): add the runtime switches to serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add fields beyond these ten.
- gpu: none

### 2.8 Add the admission level as a nested setting

- id: FT2.8
- needs: FT2.7
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving.rs::AdmissionSchedule` (~line 630): field `max_concurrent_requests: usize`, default 0 (the check is off);
  - `interop/serving.rs::apply_serving_config` (~line 1250): the one site that reads it;
  - `interop/speculative_settings.rs::SpeculativeSettings` (~line 256): a section struct whose derives this card copies.
- change:
  1. `proxima-model-interop/src/serving_settings/levels.rs` (new; `mod levels;` and `pub use levels::AdmissionScheduleSettings;` in `serving_settings.rs`; `use crate::serving::AdmissionSchedule;`): `#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)] #[builder(derive(Clone, Debug))] #[serde(default)] pub struct AdmissionScheduleSettings { pub max_concurrent_requests: usize }` with `#[setting(default = 0)]` and `#[builder(default = 0)]` on the field, doc "hard ceiling on `parallel_sequences`; 0 turns the check off", `impl Default for AdmissionScheduleSettings { fn default() -> Self { Self::builder().build() } }` and `pub(super) fn as_admission_schedule(&self) -> AdmissionSchedule { AdmissionSchedule { max_concurrent_requests: self.max_concurrent_requests } }` (`pub(super)`: the parent module calls it, D15).
  2. `serving_settings.rs`: append `#[setting(nested)] #[builder(default)] pub admission_schedule: AdmissionScheduleSettings` to `ServingSettings` and `admission_schedule: self.admission_schedule.as_admission_schedule()` to the `as_serving_config` literal.
  3. `lib.rs`: extend the `pub use serving_settings::{..}` line with `AdmissionScheduleSettings`.
  Env key: `PROXIMA_SERVING_ADMISSION_SCHEDULE_MAX_CONCURRENT_REQUESTS`.
- test: add `serving_scalars_admission_level_round_trips` in `serving_settings.rs` `tests`. TOML `[admission_schedule]` with `max_concurrent_requests = 4`; the builder `ServingSettings::builder().admission_schedule(AdmissionScheduleSettings::builder().max_concurrent_requests(4).build()).build()`; env `PROXIMA_SERVING_ADMISSION_SCHEDULE_MAX_CONCURRENT_REQUESTS=4` (`temp_env::with_vars`, then `ServingSettings::from_env()`): all three equal. Assert the lowered config `as_serving_config(&[]).admission_schedule.max_concurrent_requests == 4`, and `ServingSettings::default().as_serving_config(&[]).admission_schedule == ServingConfig::default().admission_schedule`. Sad: TOML `max_concurrent_requests = -1` is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_8 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'`
- expect: `5 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/serving_settings/levels.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): nest the admission level in serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add the phase or expert residency level (FT2.25, FT2.26); change `AdmissionSchedule`.
- gpu: none

### 2.25 Add the phase level as a nested setting

- id: FT2.25
- needs: FT2.8
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving.rs::PhaseSchedule` (~line 642): field `prefill_before_decode: bool`, default `true`;
  - `interop/serving_settings/levels.rs::AdmissionScheduleSettings` (FT2.8): the section struct to copy;
  - `interop/serving_settings.rs::ServingSettings::as_serving_config`: the literal to extend.
- change:
  1. `interop/serving_settings/levels.rs`: add `use crate::serving::PhaseSchedule;` and `pub struct PhaseScheduleSettings { pub prefill_before_decode: bool }` with the derives of `AdmissionScheduleSettings`, `#[setting(default = true)]` and `#[builder(default = true)]` on the field, doc "`true` finishes a sequence's prefill before its decode loop starts; `false` asks to interleave them across sequences", `impl Default` = `Self::builder().build()` and `pub(super) fn as_phase_schedule(&self) -> PhaseSchedule { PhaseSchedule { prefill_before_decode: self.prefill_before_decode } }`.
  2. `serving_settings.rs`: extend `pub use levels::AdmissionScheduleSettings;` to `pub use levels::{AdmissionScheduleSettings, PhaseScheduleSettings};`; append `#[setting(nested)] #[builder(default)] pub phase_schedule: PhaseScheduleSettings` and `phase_schedule: self.phase_schedule.as_phase_schedule()` in the literal.
  3. `lib.rs`: extend the `pub use serving_settings::{..}` line with `PhaseScheduleSettings`.
  Env key: `PROXIMA_SERVING_PHASE_SCHEDULE_PREFILL_BEFORE_DECODE`.
- test: add `serving_scalars_phase_level_round_trips` in `serving_settings.rs` `tests`. TOML `[phase_schedule]` with `prefill_before_decode = false` (the opposite of the default); the builder with `.prefill_before_decode(false)`; env `PROXIMA_SERVING_PHASE_SCHEDULE_PREFILL_BEFORE_DECODE=false`: all three equal. Assert `as_serving_config(&[]).phase_schedule.prefill_before_decode == false` and `ServingSettings::default().as_serving_config(&[]).phase_schedule == ServingConfig::default().phase_schedule`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_25 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'`
- expect: `6 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/serving_settings/levels.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): nest the phase level in serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a refusal for `false` (FT2.20); change `PhaseSchedule`; add the expert residency level.
- gpu: none

### 2.26 Add the expert residency level as a nested setting

- id: FT2.26
- needs: FT2.25
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving.rs::ExpertResidencySchedule` (~line 664): field `per_layer_budget_bytes: u64`, default 0 (the per-layer cap is off);
  - `interop/serving_settings/levels.rs::PhaseScheduleSettings` (FT2.25): the section struct to copy;
  - `interop/serving_settings.rs::ServingSettings::as_serving_config`: the literal to extend.
- change:
  1. `interop/serving_settings/levels.rs`: add `use crate::serving::ExpertResidencySchedule;` and `pub struct ExpertResidencyScheduleSettings { pub per_layer_budget_bytes: u64 }` with the same derives, `#[setting(default = 0)]` and `#[builder(default = 0)]`, doc "byte budget for each layer's own resident experts; 0 turns the per-layer cap off", `impl Default` = `Self::builder().build()` and `pub(super) fn as_expert_residency_schedule(&self) -> ExpertResidencySchedule { ExpertResidencySchedule { per_layer_budget_bytes: self.per_layer_budget_bytes } }`.
  2. `serving_settings.rs`: extend the `pub use levels::{..}` line with `ExpertResidencyScheduleSettings`; append `#[setting(nested)] #[builder(default)] pub expert_residency_schedule: ExpertResidencyScheduleSettings` and `expert_residency_schedule: self.expert_residency_schedule.as_expert_residency_schedule()` in the literal.
  3. `lib.rs`: extend the `pub use serving_settings::{..}` line with `ExpertResidencyScheduleSettings`.
  Env key: `PROXIMA_SERVING_EXPERT_RESIDENCY_SCHEDULE_PER_LAYER_BUDGET_BYTES`.
- test: add `serving_scalars_expert_residency_level_round_trips` in `serving_settings.rs` `tests`. TOML `[expert_residency_schedule]` with `per_layer_budget_bytes = 268435456`; the builder with the same; env `PROXIMA_SERVING_EXPERT_RESIDENCY_SCHEDULE_PER_LAYER_BUDGET_BYTES=268435456`: all three equal. Assert `as_serving_config(&[]).expert_residency_schedule.per_layer_budget_bytes == 268_435_456` and the default lowers equal to `ServingConfig::default().expert_residency_schedule`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_26 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'`
- expect: `7 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/serving_settings/levels.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): nest the expert residency level in serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `ExpertResidencySchedule`; add a refusal.
- gpu: none

### 2.27 Nest `speculative` and `prompt_cache`, and add the shared three-way helper

- id: FT2.27
- needs: FT2.26
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/speculative_settings.rs::SpeculativeSettings::as_speculative_config` (~line 346) and `interop/prompt_cache_settings.rs::PromptCacheSettings::as_prompt_cache_config` (~line 82): the two existing section types and their lowerings (no new type is added);
  - the conflaguration README "Nested configuration" (`~/.cargo/git/checkouts/conflaguration-*/*/README.md`): `#[setting(nested, override_prefix = "...")]` takes an absolute prefix;
  - `interop/prompt_cache_settings.rs::tests::prompt_cache_builder_matches_toml_and_env_loaders` (~line 138): the three-way comparison this card turns into a shared helper.
- change: `proxima-model-interop/src/serving_settings.rs` only.
  1. Append to `ServingSettings` `#[setting(nested, override_prefix = "PROXIMA_SPECULATIVE")] #[builder(default = SpeculativeSettings::builder().build())] pub speculative: SpeculativeSettings` and `#[setting(nested, override_prefix = "PROXIMA_PROMPT_CACHE")] #[builder(default = PromptCacheSettings::builder().build())] pub prompt_cache: PromptCacheSettings`; extend the `as_serving_config` literal with `speculative: self.speculative.as_speculative_config()` and `prompt_cache: self.prompt_cache.as_prompt_cache_config()`.
  2. Add `#[cfg(test)] mod round_trip` (module path `serving_settings::round_trip`) holding `pub(super) fn assert_three_ways(toml_text: &str, env_pairs: &[(&str, &str)], via_builder: &ServingSettings)`: parse `toml_text` with `conflaguration::from_toml_str::<ServingSettings>`; run `temp_env::with_vars` over every `PROXIMA_SERVING_*`, `PROXIMA_SPECULATIVE_*` and `PROXIMA_PROMPT_CACHE_*` variable currently set (each cleared with `None`) chained with `env_pairs` (each `Some(value)`), calling `ServingSettings::from_env()`; assert `parsed_toml == *via_builder` and `parsed_env == *via_builder` with a message naming which loader differs. Its first caller is this card's test; later cards reuse it.
  Env keys: `PROXIMA_SPECULATIVE_*` and `PROXIMA_PROMPT_CACHE_*` (their own prefixes, D6).
- test: add `serving_scalars_speculative_and_prompt_cache_round_trip` in `serving_settings.rs` `tests`, one `assert_three_ways` call. TOML: `[prompt_cache]` with the values of `interop/prompt_cache_settings.rs::tests::prompt_cache_builder_matches_toml_and_env_loaders` (~line 138): `byte_budget = 1073741824`, `max_entries = 8`, `ring_rewind_slack = 512`, `checkpoint_interval = 1024`, `max_checkpoints = 4`, `cache_reuse_min = 64`, `prewarm_chunk_tokens = 128`, `follow_up_branches = 3`, `follow_up_max_tokens = 64`, `follow_up_temperature_milli = 700`, `min_similarity_milli = 250`, `block_tokens = 32`, `bloom_bits_per_entry = 8192`, `bloom_hashes = 6`, and a full `[speculative]` table with the values of `interop/speculative_settings.rs::tests::speculative_config_builder_matches_loader` (~line 399): `speculative_types = "ngram-simple,ngram-map-k"`, `n_max = 3`, `n_min = 0`, `p_min = 0.0`, `ngram_simple_size_n = 16`, `ngram_simple_size_m = 32`, `ngram_simple_min_hits = 2`, `ngram_map_k_size_n = 12`, `ngram_map_k_size_m = 48`, `ngram_map_k_min_hits = 1`, `ngram_map_k4v_size_n = 12`, `ngram_map_k4v_size_m = 48`, `ngram_map_k4v_min_hits = 1`, `ngram_mod_n_match = 24`, `ngram_mod_n_max = 64`, `ngram_mod_n_min = 48`. The builder side is `ServingSettings::builder().prompt_cache(PromptCacheSettings::builder()..build()).speculative(SpeculativeSettings::builder()..build()).build()` with the same values. Env pairs: `PROXIMA_PROMPT_CACHE_BYTE_BUDGET=1073741824`, one pair for each other `prompt_cache` key under `PROXIMA_PROMPT_CACHE_` (for example `PROXIMA_PROMPT_CACHE_MAX_ENTRIES=8`), `PROXIMA_SPECULATIVE_TYPES=ngram-simple,ngram-map-k`, and one pair for each other `speculative` key under `PROXIMA_SPECULATIVE_`. `assert_three_ways` checks TOML, env and builder equal. Then assert the lowered config `as_serving_config(&[])`: `prompt_cache.byte_budget == 1_073_741_824` and `speculative.ngram_simple.size_n == 16`. Assert `PROXIMA_PROMPT_CACHE_BYTE_BUDGET=0` alone (`temp_env::with_vars`, then `ServingSettings::from_env()`) gives `!lowered.prompt_cache.is_enabled()` (the documented off switch survives). Sad: TOML `[prompt_cache]` with `byte_budget = "large"` is `Err` from `conflaguration::from_toml_str::<ServingSettings>`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_27 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'`
- expect: `8 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings.rs`
- commit: `feat(interop): nest speculative and prompt cache in serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `SpeculativeSettings` or `PromptCacheSettings`; delete their own env prefixes; add a type.
- gpu: none

### 2.15a Make `PromptCacheSettings.block_tokens` an `Option<u32>`

- id: FT2.15a
- needs: FT2.27
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/prompt_cache_settings.rs::PromptCacheSettings` (~line 20), field `block_tokens` (~lines 65 to 68) and `as_prompt_cache_config` (~line 82);
  - `interop/prompt_cache_settings.rs::tests::prompt_cache_builder_matches_toml_and_env_loaders` (~line 138): sets `.block_tokens(32)`, TOML `block_tokens = 32`, env `PROXIMA_PROMPT_CACHE_BLOCK_TOKENS=32`;
  - `interop/serving.rs::PromptCacheConfig::standard` (~line 564): `block_tokens` default 64;
  - `interop/serving_settings.rs::ServingSettings::as_serving_config` (FT2.27): reaches the prompt cache section only through `as_prompt_cache_config()` and never reads the settings field. The kv section (FT2.9) is the first reader of the `Option`, and this card lands before it so the kv lowering can tell an explicit block size from the default.
- change:
  1. `prompt_cache_settings.rs`: change the field to `pub block_tokens: Option<u32>` with no `#[setting(default = ..)]` and no `#[builder(default = ..)]` (an unattributed `Option` resolves to `None` when the env var is unset and parses `PROXIMA_PROMPT_CACHE_BLOCK_TOKENS` when set; the bon setter still takes a `u32`). Update its doc: "`None` takes the owning block size (`kv.block_tokens` under `ServingSettings`, 64 standalone); `Some` is an explicit choice and wins".
  2. `as_prompt_cache_config`: `block_tokens: match self.block_tokens { Some(value) => value, None => PromptCacheConfig::standard().block_tokens }` (a `match`, because `Option::unwrap_or` is not usable in a `const fn`).
  No other edit. Premise: nothing else reads the field, so the type change compiles with the one file staged: `git grep -n "self.prompt_cache.block_tokens" -- proxima-model-interop/src | wc -l` prints `0`. If it prints more, or the crate fails to compile in a file other than `prompt_cache_settings.rs`, the executor stops and reports the file and line.
- test: add `serving_scalars_prompt_cache_block_tokens_is_optional` in `prompt_cache_settings.rs` `tests`: with `PROXIMA_PROMPT_CACHE_BLOCK_TOKENS` unset, `PromptCacheSettings::from_env()` has `block_tokens == None` and `as_prompt_cache_config().block_tokens == 64`; with it set to `128` (`temp_env::with_vars`), `block_tokens == Some(128)` and the lowered value is `128`; TOML `block_tokens = 128` loads to `Some(128)` and TOML without the key loads to `None`; the builder `.block_tokens(128)` equals the TOML result. This card edits none of the existing tests in that file; they are named in the filter below, so a test another card adds to the same file does not change the count.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_15a cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/^prompt_cache_settings::tests::(prompt_cache_builder_matches_toml_and_env_loaders|default_settings_lower_to_the_standard_config|a_zero_byte_budget_from_the_environment_turns_the_cache_off|serving_scalars_prompt_cache_block_tokens_is_optional)$/)'`
- expect: `4 passed` (the 3 existing tests named in the filter plus the new one; the count is the same whether or not the seal-horizon card has landed)
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'` prints `9 passed` (the 8 of the previous settings card plus the new one)
- stage: `proxima-model-interop/src/prompt_cache_settings.rs`
- commit: `refactor(interop): make the prompt cache block size optional`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add the kv section (FT2.9); add the refusal (FT2.16); change `PromptCacheConfig` in `serving.rs`.
- gpu: none

### 2.22 Make eviction rules and idle steps loadable from text

- id: FT2.22
- needs: FT2.15a, FT5.7, FT10.4
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prompt_cache.rs::EvictionRule` (added by FT5.7, `pub enum EvictionRule { Branch, Oldest }`, re-exported from the crate root);
  - `interop/generate/prewarm_follow_up.rs::IdleStep` and `::DraftStep` (added by FT10.1 to FT10.4: `DraftStep { branches: u32, max_tokens: u32, temperature_milli: u32, lead: Vec<u32>, keep: bool }`, `IdleStep { Prewarm, Draft(DraftStep) }`, both `pub` and re-exported);
  - `interop/speculative_settings.rs::SpeculativeTypeName` (~line 36): the serde derive style on a mirror enum.
- change:
  1. `interop/generate/prompt_cache.rs`: derive `serde::Serialize, serde::Deserialize` on `EvictionRule` and add `#[serde(rename_all = "snake_case")]` (import `serde::{Deserialize, Serialize}` at the top if the file does not already).
  2. `interop/generate/prewarm_follow_up.rs`: derive `Serialize, Deserialize` on `DraftStep` with `#[serde(default)]` on `lead` and on `keep` (the other three fields are required); derive them on `IdleStep` with `#[serde(tag = "kind", rename_all = "snake_case")]`.
  Why a text form here and not a mirror type: the element types are the hook's own types (one definition each), so a settings mirror would be a second grammar.
- test: two tests, both through `serde_json`.
  - `text_form_eviction_rules_round_trip_json` in `prompt_cache.rs` `tests`: `["branch","oldest"]` parses to `vec![EvictionRule::Branch, EvictionRule::Oldest]`, `["oldest"]` to `vec![EvictionRule::Oldest]`, and each re-serialized value parses back equal; sad: `["mru"]` and `["Branch"]` are `Err`.
  - `text_form_idle_steps_round_trip_json` in `prewarm_follow_up.rs` `tests`: `[{"kind":"prewarm"},{"kind":"draft","branches":5,"max_tokens":256,"temperature_milli":800,"lead":[818,5279],"keep":true}]` parses to `vec![IdleStep::Prewarm, IdleStep::Draft(DraftStep { branches: 5, max_tokens: 256, temperature_milli: 800, lead: vec![818, 5279], keep: true })]` and parses back equal after re-serializing; `{"kind":"draft","branches":3,"max_tokens":48,"temperature_milli":800}` parses to a `DraftStep` with `lead` empty and `keep` false; sad: `{"kind":"draft","max_tokens":48}` and `{"kind":"sleep"}` are `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_22 cargo nextest run -p proxima-model-interop --features std -E 'test(/text_form_/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --no-default-features` (the new serde derives compile at the alloc floor, where `serde` is an unconditional dependency)
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`, `proxima-model-interop/src/generate/prewarm_follow_up.rs`
- commit: `feat(interop): load eviction rules and idle steps from text`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a type; change any variant, field or default of the two hook types; add a `Validate` impl.
- gpu: none

### 2.9 Add the kv section: the one block size and the eviction rule list

- id: FT2.9
- needs: FT2.15a, FT2.22
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - decision D7 above; `interop/generate/prompt_cache.rs::EvictionRule` (FT5.7: `Branch`, `Oldest`; the default list `[Branch, Oldest]` is today's victim rule) and `::PromptCache::eviction_victim` (~line 820);
  - `interop/serving.rs::PromptCacheConfig::block_tokens` (~line 550) and `interop/prompt_cache_settings.rs::PromptCacheSettings` field attribute style (its `block_tokens` is an `Option<u32>` after FT2.15a);
  - `interop/serving_settings.rs::round_trip::assert_three_ways` and `::ServingSettings::as_serving_config` (FT2.27): the helper the test calls and the lowering this card extends.
- change:
  1. `proxima-model-interop/src/serving_settings/kv.rs` (new; `mod kv;` plus `pub use kv::KvSettings;` in `serving_settings.rs`; `use super::from_json;`, `use crate::EvictionRule;` and the section derive imports). Add the one new public type, `KvSettings`, with `#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]`, `#[builder(derive(Clone, Debug))]`, `#[serde(default)]`, `impl Default` = `Self::builder().build()`:
     - `block_tokens: u32 = 64` (`#[setting(default = 64)]`), doc: "the one block size: the prefix index block, the content key block and the tier chunk; the prompt cache takes it unless the prompt cache section sets its own block size, which then wins";
     - `eviction: Vec<EvictionRule>` default `[Branch, Oldest]` (`resolve_with = "from_json", default_str = "[\"branch\",\"oldest\"]"`, builder `vec![EvictionRule::Branch, EvictionRule::Oldest]`), doc: "rules tried in order to pick the entry a full cache gives up".
  2. `serving_settings.rs`: append `#[setting(nested)] #[builder(default)] pub kv: KvSettings`; in `as_serving_config` set `prompt_cache: PromptCacheConfig { block_tokens: match self.prompt_cache.block_tokens { Some(explicit) => explicit, None => self.kv.block_tokens }, ..self.prompt_cache.as_prompt_cache_config() }` (D7: an explicit prompt cache block size is never overwritten). `kv.eviction` is not lowered: the caller passes it to the prompt-cache setter.
  3. `lib.rs`: extend the `pub use serving_settings::{..}` line with `KvSettings`.
  Env keys: `PROXIMA_SERVING_KV_BLOCK_TOKENS`, `PROXIMA_SERVING_KV_EVICTION` (JSON list).
- test: add `serving_settings_kv_variants` in `serving_settings/kv.rs` `tests`. Two `assert_three_ways` calls.
  1. TOML `[kv]` `block_tokens = 256` and `eviction = ["oldest"]`; env `PROXIMA_SERVING_KV_BLOCK_TOKENS=256`, `PROXIMA_SERVING_KV_EVICTION=["oldest"]`; builder `ServingSettings::builder().kv(KvSettings::builder().block_tokens(256).eviction(vec![EvictionRule::Oldest]).build()).build()`.
  2. TOML `[kv]` `block_tokens = 16`; env `PROXIMA_SERVING_KV_BLOCK_TOKENS=16`; builder sets the same (the eviction list stays the default).
  Then assert: for variant 1, `settings.kv.eviction == vec![EvictionRule::Oldest]` and `settings.as_serving_config(&[]).prompt_cache.block_tokens == 256`; for variant 2, `settings.kv.eviction == vec![EvictionRule::Branch, EvictionRule::Oldest]` and the lowered block size is 16; `KvSettings::default() == KvSettings { block_tokens: 64, eviction: vec![EvictionRule::Branch, EvictionRule::Oldest] }`; `ServingSettings::default().as_serving_config(&[]).prompt_cache.block_tokens == 64`. Explicit wins: `ServingSettings::builder().kv(KvSettings::builder().block_tokens(256).build()).prompt_cache(PromptCacheSettings::builder().block_tokens(128).build()).build().as_serving_config(&[]).prompt_cache.block_tokens == 128`, and the same with `.block_tokens(256)` on the prompt cache section lowers to 256. Sad: TOML `eviction = ["mru"]` is `Err`; `[kv]` `block_tokens = "wide"` is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_9 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_kv_variants/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'` still `9 passed`
- stage: `proxima-model-interop/src/serving_settings/kv.rs`, `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add the kv section to serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a refusal (FT2.16 on); add a tier list or a `Tier` enum (see "dropped"); add a `kv.seal` or a summary field; add a host tier size; define a second three-way helper; overwrite an explicit prompt cache block size; add a doctest.
- gpu: none

### 2.10 Add the attention settings section

- id: FT2.10
- needs: FT2.9, FT2.32
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving_grammar.rs::ReadSpec` (FT2.32): the type the section holds;
  - `interop/serving_settings/kv.rs::KvSettings` (FT2.9): the section derives and `impl Default` to copy;
  - `interop/prompt_cache_settings.rs::PromptCacheSettings` (~line 20): a public section struct with the same `Builder, Deserialize, Serialize, Settings` derives and `#[builder(derive(Clone, Debug))]`: the section derive shape this card copies (the `KvSettings` read above shows the `Default` to pair with it).
- change:
  1. `proxima-model-interop/src/serving_settings/attention.rs` (new; `mod attention;` plus `pub use attention::AttentionSettings;` in `serving_settings.rs`; `use super::from_name;` and `use crate::serving_grammar::ReadSpec;`): `AttentionSettings { read: ReadSpec }` with the section derives of `KvSettings` and `impl Default` = `Self::builder().build()`; `read` default `ReadSpec::Dense` (`#[setting(resolve_with = "from_name", default_str = "dense")]`, `#[builder(default = ReadSpec::Dense)]`), doc "which cached rows a decode step reads". No lowering method here: the card that nests the section lowers its one field. This card does not nest the section into `ServingSettings`, so no configuration can load `read = "operand"` through the serving settings until the card that also refuses it (FT2.34).
  2. `lib.rs`: extend the `pub use serving_settings::{..}` line with `AttentionSettings`.
- test: add `serving_section_attention_loads_from_toml_and_builder` in `serving_settings/attention.rs` `tests`: TOML (top-level key, no table) `read = "operand"` through `conflaguration::from_toml_str::<AttentionSettings>` equals `AttentionSettings::builder().read(ReadSpec::Operand).build()`; TOML `read = "dense"` equals `AttentionSettings::default()`; `AttentionSettings::default().read == ReadSpec::Dense` and `AttentionSettings::default() == AttentionSettings::builder().build()`. Sad: `read = "block"` and `read = "Operand"` are `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_10 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_section_attention_loads_from_toml_and_builder/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/attention.rs`, `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add the attention settings section`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a field to `ServingSettings` (FT2.34); add a lowering method; add a refusal; add a field to `AttentionSettings`.
- gpu: none

### 2.34 Nest the attention section and refuse an unbound read

- id: FT2.34
- needs: FT2.10, FT2.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `~/.cargo/git/checkouts/conflaguration-*/*/src/lib.rs`: `ValidationMessage` (~line 248, public `path` and `message`), `Error::Validation { errors }` (~line 295, the enum is `#[non_exhaustive]`), `Validate` (~line 376): the string-based error `Validate` must return;
  - `interop/speculative_settings.rs` `InvalidSpeculativeType` (~line 87): the `thiserror` style used in this crate;
  - `interop/serving_settings.rs::ServingSettings`, `::as_serving_config` (FT2.9) and `::round_trip::assert_three_ways` (FT2.27);
  - `interop/serving_settings/attention.rs::AttentionSettings` (FT2.10) and `interop/serving.rs::AttentionConfig` (FT2.1): the section this card nests and the config it lowers into.
- change:
  1. `proxima-model-interop/src/serving_settings/refusal.rs` (new; `mod refusal; mod refusals;` plus `pub use refusal::ServingRefusal;` in `serving_settings.rs`; `pub use serving_settings::ServingRefusal` in `lib.rs` is added by FT2.21): `#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)] pub enum ServingRefusal { #[error("attention.read operand is refused: no read hook is bound to the decode step yet; use dense")] ReadHookNotBound }` (later cards add the other variants with lowercase `#[error]` text) and `pub const fn field_path(&self) -> &'static str` returning `"attention.read"`. `ReadHookNotBound` exists because `Operand` loads from text while nothing in the library reads it yet (FT2.1 binds it only into the cache key): without the row, an `Operand` configuration runs a dense read under a different cache key and nothing says so. It lands in the same commit that makes `read` loadable through the serving settings. The card that binds the decode arm for `Operand` removes this variant (see "contracts with sibling files").
  2. `proxima-model-interop/src/serving_settings/refusals.rs` (new): `impl ServingSettings { pub fn refusals(&self) -> Vec<ServingRefusal> { let mut found = Vec::new(); self.check_read(&mut found); found } fn check_read(&self, found: &mut Vec<ServingRefusal>) { .. } }`, where `check_read` pushes `ReadHookNotBound` when `self.attention.read != ReadSpec::Dense`; and `impl conflaguration::Validate for ServingSettings { fn validate(&self) -> conflaguration::Result<()> { let refusals = self.refusals(); if refusals.is_empty() { return Ok(()); } Err(conflaguration::Error::Validation { errors: refusals.iter().map(|refusal| conflaguration::ValidationMessage::new(refusal.field_path(), refusal.to_string())).collect() }) } }`. A manual impl, not a derive (the derive cascades into nested sections that carry their own `Validate`).
  3. `serving_settings.rs`: the two `mod` lines and the `pub use` of item 1; append `#[setting(nested)] #[builder(default)] pub attention: AttentionSettings`; in `as_serving_config` set `attention: AttentionConfig { read: self.attention.read }` (import `crate::serving::AttentionConfig`; the lowering is one copied field, so no method is added).
  Env key: `PROXIMA_SERVING_ATTENTION_READ` (a bare word).
- test: two tests.
  - `serving_settings_attention_read_variants` in `serving_settings.rs` `tests`, two `assert_three_ways` calls: dense (TOML `[attention]` `read = "dense"`, env `PROXIMA_SERVING_ATTENTION_READ=dense`, builder `ReadSpec::Dense`, equal to `ServingSettings::default()`) and operand (TOML `read = "operand"`, env `PROXIMA_SERVING_ATTENTION_READ=operand`, builder `ReadSpec::Operand`). Assert `ServingSettings::default().as_serving_config(&[]).attention.read == ReadSpec::Dense` and, for the operand case, `.attention.read == ReadSpec::Operand`. Sad: `read = "block"` is `Err`.
  - `serving_settings_refuses_read_hook_not_bound` in `refusals.rs` `tests`: `attention.read = ReadSpec::Operand`: `refusals() == vec![ReadHookNotBound]` and `validate()` is `Err(conflaguration::Error::Validation { errors })` (matched with `let .. else`) with `errors.len() == 1` and `errors[0].path == "attention.read"`; `ReadSpec::Dense`: `refusals()` is empty and `validate()` is `Ok`; `ServingSettings::default().validate().is_ok()` (happy path).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_34 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_attention_read_variants|serving_settings_refuses_read_hook_not_bound/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_scalars_/)'` still `9 passed`
- stage: `proxima-model-interop/src/serving_settings/refusal.rs`, `proxima-model-interop/src/serving_settings/refusals.rs`, `proxima-model-interop/src/serving_settings.rs`
- commit: `feat(interop): nest the attention section and refuse an unbound read`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add variants beyond `ReadHookNotBound`; derive `Validate`; put a model-dependent row here (FT2.17); add the block-size row (FT2.16); edit `lib.rs`; add a lowering method for the attention section.
- gpu: none

### 2.12 Add the prefill section

- id: FT2.12
- needs: FT2.34, FT2.2
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving_grammar.rs::AssembleStep` and `interop/serving.rs::PrefillConfig` (FT2.2);
  - `interop/serving_settings/attention.rs` (FT2.10) and `serving_settings.rs::round_trip::assert_three_ways` (FT2.27).
- change:
  1. `proxima-model-interop/src/serving_settings/prefill.rs` (new; `mod prefill;` plus `pub use prefill::PrefillSettings;`; `use super::from_json;`): `PrefillSettings { assemble: Vec<AssembleStep> }` with the section derives, `#[setting(resolve_with = "from_json", default_str = "[]")]`, builder default empty; doc "ordered stages that build the starting cache of a request; empty is today's behaviour"; `pub(super) fn as_prefill_config(&self) -> PrefillConfig<'_> { PrefillConfig { assemble: &self.assemble } }` (`pub(super)`: the parent module calls it, D15).
  2. `serving_settings.rs`: append `#[setting(nested)] #[builder(default)] pub prefill: PrefillSettings`; set `prefill: self.prefill.as_prefill_config()`.
  3. `lib.rs`: extend the `pub use serving_settings::{..}` line with `PrefillSettings`.
  Env key: `PROXIMA_SERVING_PREFILL_ASSEMBLE` (JSON list).
- test: add `serving_settings_prefill_assemble_variants` in `serving_settings/prefill.rs` `tests`, three `assert_three_ways` calls: `[prefix]`; `[prefix, shift]`; `[shift, prefix]`, each with TOML `[[prefill.assemble]]` tables (`kind = "prefix"`, `kind = "shift"`, in list order) and env `PROXIMA_SERVING_PREFILL_ASSEMBLE=[{"kind":"prefix"},{"kind":"shift"}]` (and the matching text for the other two lists). Assert for the second case `as_serving_config(&[]).prefill.assemble == [AssembleStep::Prefix, AssembleStep::Shift]`; for the third case `as_serving_config(&[]).prefill.assemble == [AssembleStep::Shift, AssembleStep::Prefix]` (order is kept); `ServingSettings::default().prefill.assemble.is_empty()`. Sad: `{"kind":"blend"}` and `{"kind":"load","path":"/models/cartridges/legal-v3.cart"}` are `Err` (no library stage runs them).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_12 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_prefill_assemble_variants/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/prefill.rs`, `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add the prefill section to serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a refusal (FT2.18); run any assemble stage; add a `Load` or `Blend` case.
- gpu: none

### 2.13 Add the schedule section

- id: FT2.13
- needs: FT2.12, FT2.22
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/generate/prewarm_follow_up.rs::IdleStep` and `::DraftStep` (FT10.1 to FT10.4; serde from FT2.22): the element grammar; the empty list means today's two-step sequence (prewarm, then drafting from the `follow_up_*` fields of `prompt_cache`);
  - sketch 12 (`pipeline-as-data/sketches/12-sleep-time-prewarm.md`) section 1: `[[schedule.idle]]` with `kind = "prewarm"` or `kind = "draft"` and its own parameters;
  - `interop/serving_settings/prefill.rs` (FT2.12).
- change:
  1. `proxima-model-interop/src/serving_settings/schedule.rs` (new; `mod schedule;` plus `pub use schedule::ScheduleSettings;`; `use super::from_json;` and `use crate::IdleStep;`): `ScheduleSettings { idle: Vec<IdleStep> }` with the section derives, `#[setting(resolve_with = "from_json", default_str = "[]")]`, builder default empty; doc "ordered jobs the prewarm worker runs for each queued end-of-answer job; empty is today's behaviour". Nothing is lowered into `ServingConfig`: the caller passes `settings.schedule.idle` to the idle-schedule setter.
  2. `serving_settings.rs`: append `#[setting(nested)] #[builder(default)] pub schedule: ScheduleSettings`.
  3. `lib.rs`: extend the `pub use serving_settings::{..}` line with `ScheduleSettings`.
  Env key: `PROXIMA_SERVING_SCHEDULE_IDLE` (JSON list).
- test: add `serving_settings_schedule_idle_variants` in `serving_settings/schedule.rs` `tests`: `assert_three_ways` with TOML `[[schedule.idle]]` `kind = "prewarm"`, `[[schedule.idle]]` `kind = "draft"`, `branches = 5`, `max_tokens = 256`, `temperature_milli = 800`, `lead = [818, 5279]`, `keep = true`, and `[[schedule.idle]]` `kind = "draft"`, `branches = 2`, `max_tokens = 48`, `temperature_milli = 700`; env `PROXIMA_SERVING_SCHEDULE_IDLE=[{"kind":"prewarm"},{"kind":"draft","branches":5,"max_tokens":256,"temperature_milli":800,"lead":[818,5279],"keep":true},{"kind":"draft","branches":2,"max_tokens":48,"temperature_milli":700}]`; builder sets the same three steps. Assert `settings.schedule.idle[2] == IdleStep::Draft(DraftStep { branches: 2, max_tokens: 48, temperature_milli: 700, lead: Vec::new(), keep: false })` (omitted `lead` and `keep` take their defaults) and `ServingSettings::default().schedule.idle.is_empty()`. Sad: `{"kind":"draft","max_tokens":48}` (no `branches`) is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_13 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_schedule_idle_variants/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/schedule.rs`, `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add the schedule section to serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a refusal (FT2.19); add a job kind; add a `ServingConfig` field.
- gpu: none

### 2.29 Add the expert placement settings section

- id: FT2.29
- needs: FT2.13
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving.rs::ServingConfig` fields `qwen35moe_pre_gather` through `qwen35moe_monolithic_high_mmap` (~lines 910 to 960): names, docs and the defaults in `impl Default` (~lines 1178 to 1192);
  - sketch 03 (`pipeline-as-data/sketches/03-moe-offload.md`) section 2: the shape `[placement.experts]`, no family name in any key;
  - `interop/serving_settings/kv.rs::KvSettings` (FT2.9): the section derives and `impl Default` to copy.
- change:
  1. `proxima-model-interop/src/serving_settings/placement.rs` (new; `mod placement;` plus `pub use placement::ExpertPlacementSettings;` in `serving_settings.rs`): `ExpertPlacementSettings` with the section derives of `KvSettings` and `impl Default` = `Self::builder().build()`. Fields (all `pub`), each `#[setting(default = ..)]` with the same `#[builder(default = ..)]` and a one-line doc taken from the matching `ServingConfig` field: `pre_gather: bool = false`, `persistent_cuts: bool = false`, `residency_budget_bytes: u64 = 0`, `prefetch: bool = false`, `monolithic_all_low: bool = false`, `layer_window: usize = 1`, `monolithic_high_mmap: bool = false`. This card does not nest the section into `ServingSettings`; the section is public, loads through `conflaguration::from_toml_str` and the builder, and the card that nests it (FT2.23) tests the env keys.
  2. `lib.rs`: extend the `pub use serving_settings::{..}` line with `ExpertPlacementSettings`.
- test: add `serving_section_expert_placement_loads_from_toml_and_builder` in `serving_settings/placement.rs` `tests`. TOML (top-level keys, no table): `pre_gather = true`, `persistent_cuts = true`, `residency_budget_bytes = 8589934592`, `prefetch = true`, `monolithic_all_low = true`, `layer_window = 2`, `monolithic_high_mmap = true`; `conflaguration::from_toml_str::<ExpertPlacementSettings>` equals the builder with the same seven values. Assert `ExpertPlacementSettings::default() == ExpertPlacementSettings::builder().build()` and its `layer_window == 1` and the six other fields are `false` or `0`. Sad: `layer_window = -1` is `Err`; `pre_gather = "yes"` is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_29 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_section_expert_placement_loads_from_toml_and_builder/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/placement.rs`, `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add the expert placement settings section`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a field to `ServingSettings` (FT2.23); add the budget fields (FT2.30); add an enum that merges the three modes; add a refusal (FT2.24); rename a `ServingConfig` field.
- gpu: none

### 2.30 Add the placement budget settings section

- id: FT2.30
- needs: FT2.29
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving.rs::ServingConfig` fields `dense_weights_budget_bytes`, `expert_weights_budget_bytes`, `activations_budget_bytes`, `kv_cache_budget_bytes` (~lines 925 to 960): `0` is unbounded; their defaults in `impl Default` (~lines 1178 to 1192);
  - sketch 03 (`pipeline-as-data/sketches/03-moe-offload.md`) section 2: the shape `[placement.budget]`;
  - `interop/serving_settings/placement.rs::ExpertPlacementSettings` (FT2.29): the section to copy.
- change:
  1. `interop/serving_settings/placement.rs`: add `PlacementBudgetSettings` with the same derives and `impl Default` = `Self::builder().build()`. Fields (all `pub`, each `u64`, each `#[setting(default = 0)]` with `#[builder(default = 0)]`, `0` is unbounded, doc from the matching `ServingConfig` field): `dense_weights_bytes`, `expert_weights_bytes`, `activations_bytes`, `kv_cache_bytes`.
  2. `serving_settings.rs`: extend `pub use placement::ExpertPlacementSettings;` to `pub use placement::{ExpertPlacementSettings, PlacementBudgetSettings};`. `lib.rs`: extend the `pub use serving_settings::{..}` line with `PlacementBudgetSettings`.
- test: add `serving_section_placement_budget_loads_from_toml_and_builder` in `serving_settings/placement.rs` `tests`. TOML top-level keys `dense_weights_bytes = 1073741824`, `expert_weights_bytes = 2147483648`, `activations_bytes = 536870912`, `kv_cache_bytes = 4294967296`; `conflaguration::from_toml_str::<PlacementBudgetSettings>` equals the builder with the same four values. Assert `PlacementBudgetSettings::default() == PlacementBudgetSettings::builder().build()` and all four default to `0`. Sad: `kv_cache_bytes = -1` is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_30 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_section_placement_budget_loads_from_toml_and_builder/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/placement.rs`, `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): add the placement budget settings section`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a field to `ServingSettings` (FT2.23); add a read cache limit; rename a `ServingConfig` field.
- gpu: none

### 2.23 Nest the placement section in the serving settings

- id: FT2.23
- needs: FT2.30
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving.rs::ServingConfig` fields `qwen35moe_pre_gather` through `qwen35moe_monolithic_high_mmap`, and the four budgets (~lines 910 to 960): the eleven fields this card lowers;
  - `interop/serving_settings/placement.rs::ExpertPlacementSettings` and `::PlacementBudgetSettings` (FT2.29, FT2.30): the two leaves this card nests;
  - `interop/serving_settings/levels.rs` (FT2.8): nested section structs with no struct-level prefix.
- change:
  1. `interop/serving_settings/placement.rs`: add the one new public type, `PlacementSettings { pub experts: ExpertPlacementSettings, pub budget: PlacementBudgetSettings }` with the section derives, `#[setting(nested)] #[builder(default)]` on each field, `impl Default` = `Self::builder().build()`; extend `pub use placement::{..}` in `serving_settings.rs` with `PlacementSettings` and the `pub use serving_settings::{..}` line in `lib.rs` with `PlacementSettings`.
  2. `serving_settings.rs`: append `#[setting(nested)] #[builder(default)] pub placement: PlacementSettings`; extend the `as_serving_config` literal: `qwen35moe_pre_gather: self.placement.experts.pre_gather`, `qwen35moe_persistent_cuts: ..persistent_cuts`, `qwen35moe_residency_budget_bytes: ..residency_budget_bytes`, `qwen35moe_expert_prefetch: ..prefetch`, `qwen35moe_monolithic_all_low: ..monolithic_all_low`, `qwen35moe_layer_window: ..layer_window`, `qwen35moe_monolithic_high_mmap: ..monolithic_high_mmap`, `dense_weights_budget_bytes: self.placement.budget.dense_weights_bytes`, `expert_weights_budget_bytes: ..expert_weights_bytes`, `activations_budget_bytes: ..activations_bytes`, `kv_cache_budget_bytes: ..kv_cache_bytes`.
  Env keys: `PROXIMA_SERVING_PLACEMENT_EXPERTS_<FIELD>` and `PROXIMA_SERVING_PLACEMENT_BUDGET_<FIELD>`.
- test: add `serving_settings_placement_variants` in `serving_settings/placement.rs` `tests`: `assert_three_ways` with TOML `[placement.experts]` `pre_gather = true`, `persistent_cuts = true`, `residency_budget_bytes = 8589934592`, `prefetch = true`, `monolithic_all_low = true`, `layer_window = 2`, `monolithic_high_mmap = true`, and `[placement.budget]` `dense_weights_bytes = 1073741824`, `expert_weights_bytes = 2147483648`, `activations_bytes = 536870912`, `kv_cache_bytes = 4294967296`; env `PROXIMA_SERVING_PLACEMENT_EXPERTS_PRE_GATHER=true` and the same pattern for every key above; builder sets the same through `PlacementSettings::builder().experts(..).budget(..)`. Assert on the lowered config: `qwen35moe_pre_gather`, `qwen35moe_persistent_cuts`, `qwen35moe_expert_prefetch`, `qwen35moe_monolithic_all_low`, `qwen35moe_monolithic_high_mmap` are `true`, `qwen35moe_residency_budget_bytes == 8_589_934_592`, `qwen35moe_layer_window == 2`, `dense_weights_budget_bytes == 1_073_741_824`, `expert_weights_budget_bytes == 2_147_483_648`, `activations_budget_bytes == 536_870_912`, `kv_cache_budget_bytes == 4_294_967_296`; and `ServingSettings::default().as_serving_config(&[])` equals `ServingConfig::default()` on these eleven fields individually. Sad: `layer_window = -1` is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_23 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_placement_variants/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_(kv|attention_read|prefill_assemble|schedule_idle|placement)_variants/)'` prints `5 passed`
- stage: `proxima-model-interop/src/serving_settings/placement.rs`, `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(interop): nest the placement section in serving settings`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a type other than `PlacementSettings`; add an enum that merges the three modes; add a refusal (FT2.24); rename a `ServingConfig` field; add a field for the read cache limit.
- gpu: none

### 2.15 Whole-surface round trip and default parity

- id: FT2.15
- needs: FT2.23
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving_settings.rs::ServingSettings::as_serving_config`: it still ends with `..ServingConfig::default()`; this card removes it;
  - `interop/serving.rs::impl Default for ServingConfig<'static>` (~line 1136): the oracle for default parity;
  - the `serving_scalars_*` tests and the section `*_variants` tests (FT2.4 to FT2.27, FT2.31, FT2.9 to FT2.13, FT2.34, FT2.23): the values and env pairs the union reuses.
- change: `proxima-model-interop/src/serving_settings.rs` only.
  1. In `as_serving_config`, delete `..ServingConfig::default()`. The literal then names all 60 fields (compiler-checked: a missed field is `E0063`); `math_mode` and `dispatch_type` stay under their cfg. If a field is missing, add it by name from the lowering table at the top of this file; do not restore the base.
  2. No other edit. Premise: every `ServingSettings` field default already equals the matching `ServingConfig::default()` value, because each earlier card copied its defaults from `impl Default for ServingConfig<'static>` (FT2.7, FT2.8, FT2.25, FT2.26 and FT2.23 also assert theirs); this card's default-parity test is the first whole-struct comparison. If `serving_settings_default_parity` fails, the executor stops and reports the field the assertion names (it prints both values); it does not change a default in this card, because the card that added the field owns that default and the stage list below is one file.
- test: add two tests in `serving_settings.rs` `tests`:
  - `serving_settings_default_parity`: `let settings = ServingSettings::default(); let rules = settings.weight_precision_rules(); assert_eq!(settings.as_serving_config(&rules), ServingConfig::default());` and `assert_eq!(settings, ServingSettings::builder().build())`, and with every `PROXIMA_SERVING_*`, `PROXIMA_SPECULATIVE_*` and `PROXIMA_PROMPT_CACHE_*` variable cleared through `temp_env::with_vars`, `ServingSettings::from_env()` equals `ServingSettings::default()` (the three default sources agree).
  - `serving_settings_round_trip`: every one of the `ServingConfig` fields at a non-default value, through `assert_three_ways`. The TOML is the union of the test TOMLs of FT2.4 to FT2.7, FT2.31, FT2.8, FT2.25, FT2.26 and FT2.27 and of the section tests FT2.9, FT2.34, FT2.12, FT2.13 and FT2.23, with their values (all top-level keys first, then every table in card order; the `[prompt_cache]` table and its env pairs omit `block_tokens`, which `[kv] block_tokens = 256` owns, FT2.9 variant 1; the `[phase_schedule]` table keeps `prefill_before_decode = false` from FT2.25, the opposite of its default), `[attention] read = "operand"`, `[[prefill.assemble]]` prefix, shift (the second case of FT2.12), `[[schedule.idle]]` the three steps of FT2.13, and `[kv]` `block_tokens = 256` with `eviction = ["oldest"]` of FT2.9 variant 1. Env pairs are the union of those tests' env pairs. After the three-way equality, assert on `settings.as_serving_config(&rules)` that no field equals `ServingConfig::default()`'s value for that field: a `[(&str, bool)]` table in the test, one row per field in declaration order, each `bool` = `lowered.<field> != default.<field>`, all `true`:
    `model_path, context_length, rope_scaling, parallel_sequences, kv_cache_key_quant, kv_cache_value_quant, flash_attention, batch_size, ubatch_size, gpu_layers, gpu_memory_fit, gpu_memory_limit_bytes, kv_offload, multimodal_projector, reasoning_budget, temperature, top_k, top_p, min_p, repeat_last_n, repeat_penalty, frequency_penalty, presence_penalty, seed, kv_bucket_tokens, math_mode (cfg), numeric_policy, dispatch_type (cfg), exact_activations, weight_precision, qwen35moe_pre_gather, qwen35moe_persistent_cuts, gdn_prefill_backend, qwen35moe_residency_budget_bytes, dense_weights_budget_bytes, expert_weights_budget_bytes, activations_budget_bytes, kv_cache_budget_bytes, qwen35moe_expert_prefetch, qwen35moe_monolithic_all_low, qwen35moe_layer_window, qwen35moe_monolithic_high_mmap, gpu_correctness_fallback, prefill_one_evaluation, prefill_chunk_positions, cached_attention_fusion, gated_delta_net_fusion, moe_topk_fusion, plan_time_constants, plan_refit, command_buffer_chunks, max_command_buffers_per_token, overlap_transfer_compute, admission_schedule, phase_schedule, expert_residency_schedule, speculative, prompt_cache, attention, prefill`.
    Assert the table length first: `58` without the metal backend and `60` under `cfg(all(feature = "metal", target_os = "macos"))`. Also assert `settings.kv != KvSettings::default()` and `settings.schedule != ScheduleSettings::default()` (the two settings-only sections).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_15 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_default_parity|serving_settings_round_trip/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo check -p proxima-model-interop --features std,metal --all-targets`; `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_(kv|attention_read|prefill_assemble|schedule_idle|placement)_variants|serving_settings_default_parity|serving_settings_round_trip/)'` prints `7 passed`
- stage: `proxima-model-interop/src/serving_settings.rs`
- commit: `test(interop): round trip every serving config field through settings`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a field to `ServingSettings`; weaken the exhaustive literal back to a base.
- gpu: none

### 2.16 Add the block-size conflict row

- id: FT2.16
- needs: FT2.15, FT2.34
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving_settings/refusal.rs::ServingRefusal` and `interop/serving_settings/refusals.rs::ServingSettings::refusals` (FT2.34): the enum and the check list this card extends;
  - `interop/serving_settings.rs::ServingSettings::as_serving_config` (FT2.9) and decision D7: the lowering takes an explicit prompt cache block size and otherwise the kv one, so a mismatch shadows the kv value;
  - `interop/prompt_cache_settings.rs::PromptCacheSettings` field `block_tokens` (FT2.15a: `Option<u32>`).
- change:
  1. `proxima-model-interop/src/serving_settings/refusal.rs`: add the variant `#[error("prompt_cache.block_tokens {prompt_cache} conflicts with kv.block_tokens {kv}")] BlockTokensConflict { kv: u32, prompt_cache: u32 }` and a `"prompt_cache.block_tokens"` arm in `field_path`.
  2. `proxima-model-interop/src/serving_settings/refusals.rs`: add `fn check_block(&self, found: &mut Vec<ServingRefusal>)`, which pushes `BlockTokensConflict { kv: self.kv.block_tokens, prompt_cache: explicit }` when `self.prompt_cache.block_tokens == Some(explicit)` and `explicit != self.kv.block_tokens`, and call it from `refusals()` before `check_read`.
- test: add `serving_settings_refuses_block_tokens_conflict` in `serving_settings/refusals.rs` `tests`:
  - `kv.block_tokens = 256` with `prompt_cache.block_tokens = Some(128)`: `refusals() == vec![ServingRefusal::BlockTokensConflict { kv: 256, prompt_cache: 128 }]` and `validate()` is `Err(conflaguration::Error::Validation { errors })` (matched with `let .. else`) with `errors.len() == 1` and `errors[0].path == "prompt_cache.block_tokens"`;
  - `kv = 256` with `Some(256)`: empty; `kv = 256` with `None`: empty (the lowering gives 256); `kv = 256` with `Some(64)`: `vec![BlockTokensConflict { kv: 256, prompt_cache: 64 }]` (an explicit 64 is refused); `kv = 64` with `Some(32)`: `vec![BlockTokensConflict { kv: 64, prompt_cache: 32 }]`;
  - `ServingSettings::default().validate().is_ok()` and `ServingSettings::default().refusals().is_empty()` (happy path).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_16 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_refuses_block_tokens_conflict/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; the 7-test filter of FT2.15 still prints `7 passed`; `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_refuses_read_hook_not_bound/)'` still prints `1 passed`
- stage: `proxima-model-interop/src/serving_settings/refusal.rs`, `proxima-model-interop/src/serving_settings/refusals.rs`
- commit: `feat(interop): refuse conflicting block sizes in serving settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add variants beyond `BlockTokensConflict`; add a second public type; put a model-dependent row here; add a multiple-of-16 rule (no summary field exists to tile); change the lowering (an explicit value already wins).
- gpu: none

### 2.17 Refuse reads that cannot be served (rows: attention layers, two-range cache)

- id: FT2.17
- needs: FT2.16
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `proxima-tensor/src/spec/descriptor.rs::ModelDescriptor` (~line 53: public `layers: Vec<LayerSchedule>`, `cache_strategy: CacheStrategy`), `::CacheStrategy` (~line 15: `Cacheless`, `TwoRange`, `SingleRange`) and `::mistral_descriptor_from_shape` (~line 247; it takes a trailing `&FamilyProfile` and builds `cache_strategy: SingleRange`);
  - `proxima-tensor/src/spec/single_range_moe_cached.rs::LayerKind` (~line 1452: `Attention`, `ShortConv`) and `interop/profiles/mod.rs::family_profile` (~line 40);
  - `interop/serving_settings/refusals.rs` (FT2.34, FT2.16): `check_read` already pushes `ReadHookNotBound`;
  - the read-hook cards (`tasks-recut/06-read-sets.md`): the read hook is built on the two-range cached attention builder only, and a read on a single-range engine is a named decide-later item there.
- change:
  1. `refusal.rs`: add variants `#[error("attention.read needs attention layers only; layer {layer} is not one")] ReadNeedsAttentionLayers { layer: usize }` and `#[error("attention.read needs the two-range kv cache, which this model does not use")] ReadNeedsTwoRangeCache`; `field_path` returns `"attention.read"` for both.
  2. `refusals.rs`: add `pub fn refusals_for(&self, descriptor: &proxima_tensor::spec::ModelDescriptor) -> Vec<ServingRefusal>`: starts from `self.refusals()` (which already holds `ReadHookNotBound` for a non-dense read); then, when `self.attention.read != ReadSpec::Dense`, pushes `ReadNeedsAttentionLayers { layer }` for the first `layer` whose `descriptor.layers[layer].kind != LayerKind::Attention`, and then pushes `ReadNeedsTwoRangeCache` when `descriptor.cache_strategy != CacheStrategy::TwoRange`.
- test: add two tests in `refusals.rs` `tests`. Fixtures (decision D9): `let profile = crate::profiles::family_profile("gemma4").expect("gemma4 profile is embedded");` then `dense = mistral_descriptor_from_shape(262_144, 1536, 6144, 8, 1, 256, 35, 0, 0, false, false, false, false, &profile)` (the first values of `gemma4_e2b/gguf_kv.txt`: vocab 262144, embedding 1536, feed forward 6144, 8 query heads, 1 kv head, head dim 256 (the `gemma4.attention.key_length_swa` value; `gemma4.attention.key_length` is 512), 35 blocks), `two_range = ModelDescriptor { cache_strategy: CacheStrategy::TwoRange, ..dense.clone() }`, `hybrid = ModelDescriptor { layers: ..., ..two_range.clone() }` with `layers[5].kind = LayerKind::ShortConv` (clone the vector, assign the field).
  - `serving_settings_refuses_read_needs_attention_layers`: settings with `attention.read = ReadSpec::Operand`: `refusals_for(&hybrid) == vec![ReadHookNotBound, ReadNeedsAttentionLayers { layer: 5 }]`; `refusals_for(&two_range) == vec![ReadHookNotBound]` (the descriptor rows add nothing); with `ReadSpec::Dense` and `hybrid`: empty.
  - `serving_settings_refuses_read_needs_two_range_cache`: `Operand` with `dense` (single range as built; assert `dense.cache_strategy == CacheStrategy::SingleRange` first): `vec![ReadHookNotBound, ReadNeedsTwoRangeCache]`; `Operand` with `two_range`: `vec![ReadHookNotBound]`; `Dense` with `dense`: empty; `Operand` with a hybrid single-range descriptor (`ModelDescriptor { layers: <hybrid layers>, ..dense.clone() }`): `vec![ReadHookNotBound, ReadNeedsAttentionLayers { layer: 5 }, ReadNeedsTwoRangeCache]` in that order.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_17 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_refuses_read_needs_/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/refusal.rs`, `proxima-model-interop/src/serving_settings/refusals.rs`
- commit: `feat(interop): refuse reads that cannot be served`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: move the two descriptor rows into `validate()`; add or remove the `ReadHookNotBound` row (FT2.34); add a rectify or replay row; add a layer field to `ReadNeedsTwoRangeCache`; add any other not-bound row.
- gpu: none

### 2.18 Refuse unmet assemble, shift and eviction settings

- id: FT2.18
- needs: FT2.17, FT5.20
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/generate/prompt_cache.rs::PromptCache::prompt_cache_lookup` (the `shifting` binding, `config.cache_reuse_min > 0 && config.ring_rewind_slack > 0`, ~line 1055) and `interop/serving.rs::PromptCacheConfig::is_enabled` (~line 596: `byte_budget > 0 && max_entries > 0`);
  - FT5.20 (`LoadedModel::set_eviction_rules`, with `PromptCache::set_eviction_rules`, refuses a list whose last rule is not `Oldest` with `InteropError::UnsupportedServingConfig`, so a full cache always has a victim);
  - `interop/serving_settings/refusals.rs` (FT2.17).
- change:
  1. `refusal.rs`: variants, each with a lowercase `#[error]`: `AssembleNeedsPromptCache` ("prefill.assemble needs the prompt cache on: byte_budget and max_entries must be above 0"), `ShiftNeedsReuse` ("prefill.assemble shift needs prompt_cache.cache_reuse_min and ring_rewind_slack above 0"), `EvictionNeedsOldestLast` ("kv.eviction must end with oldest so a full cache always has a victim"). `field_path`: `AssembleNeedsPromptCache | ShiftNeedsReuse => "prefill.assemble"`, `EvictionNeedsOldestLast => "kv.eviction"`.
  2. `refusals.rs`: `check_prefill(&self, found)`: push `AssembleNeedsPromptCache` when `!self.prefill.assemble.is_empty()` and `!self.prompt_cache.as_prompt_cache_config().is_enabled()`; push `ShiftNeedsReuse` when `self.prefill.assemble` contains `AssembleStep::Shift` and not (`self.prompt_cache.cache_reuse_min > 0 && self.prompt_cache.ring_rewind_slack > 0`). `check_kv(&self, found)`: push `EvictionNeedsOldestLast` unless `self.kv.eviction.last() == Some(&EvictionRule::Oldest)`. Call both from `refusals()` after `check_read`.
- test: three tests in `refusals.rs` `tests` (each also asserts `ServingSettings::default().refusals().is_empty()`):
  - `serving_settings_refuses_assemble_needs_prompt_cache`: `assemble = [Prefix]` with `prompt_cache.byte_budget = 0`: `vec![AssembleNeedsPromptCache]`; with the default budget (2147483648): empty; `assemble = []` with `byte_budget = 0`: empty.
  - `serving_settings_refuses_shift_needs_reuse`: `assemble = [Prefix, Shift]` with the defaults (`cache_reuse_min = 0`): `vec![ShiftNeedsReuse]`; with `cache_reuse_min = 64` (default `ring_rewind_slack = 256`): empty; with `cache_reuse_min = 64` and `ring_rewind_slack = 0`: `vec![ShiftNeedsReuse]`; `assemble = [Prefix]` with the defaults: empty.
  - `serving_settings_refuses_eviction_without_oldest_last`: `[Branch, Oldest]` and `[Oldest]` are empty; `[]`, `[Branch]` and `[Oldest, Branch]` are each `vec![EvictionNeedsOldestLast]`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_18 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_refuses_(assemble_needs_prompt_cache|shift_needs_reuse|eviction_without_oldest_last)/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/refusal.rs`, `proxima-model-interop/src/serving_settings/refusals.rs`
- commit: `feat(interop): refuse unmet assemble and eviction settings`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a row for a tier list (none exists in settings); add a row for a stage the grammar lacks (the grammar holds only `Prefix` and `Shift`, and an unknown stage fails to load); touch the idle or placement rows.
- gpu: none

### 2.19 Refuse idle steps without a worker or room for their drafts

- id: FT2.19
- needs: FT2.18
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - decision D8 and `interop/serving.rs::PromptCacheConfig` fields `byte_budget`, `max_entries`, `prewarm_chunk_tokens`, `follow_up_branches` (~lines 472 to 545; `follow_up_branches` is documented to count against `byte_budget` and `max_entries`, and no check exists: sketch 12 gap on validation);
  - `interop/generate/prewarm_follow_up.rs::IdleStep` and `::DraftStep` (FT10.4): `DraftStep { branches, .. }`; an empty idle list runs one draft step built from `prompt_cache.follow_up_branches`;
  - `interop/serving_settings/refusals.rs` (FT2.18).
- change:
  1. `refusal.rs`: variants `#[error("schedule.idle needs the prewarm worker: prompt_cache byte_budget, max_entries and prewarm_chunk_tokens must be above 0")] IdleNeedsWorker` and `#[error("schedule.idle drafts {branches} branches but the prompt cache holds {max_entries} entries; a draft evicts an earlier one")] DraftBranchesExceedEntries { branches: u32, max_entries: u32 }`; `field_path` returns `"schedule.idle"` for both.
  2. `refusals.rs`: `check_schedule(&self, found)`: push `IdleNeedsWorker` when `!self.schedule.idle.is_empty()` and not (`prompt_cache.byte_budget > 0 && prompt_cache.max_entries > 0 && prompt_cache.prewarm_chunk_tokens > 0`). Then the draft counts: for each `IdleStep::Draft(step)` in `self.schedule.idle`, or, when the list is empty, the single count `self.prompt_cache.follow_up_branches`; for each count `branches > 0` with `branches + 1 > prompt_cache.max_entries`, push `DraftBranchesExceedEntries { branches, max_entries: prompt_cache.max_entries }`. Call it from `refusals()` after `check_kv`.
- test: two tests in `refusals.rs` `tests` (each also asserts `ServingSettings::default().refusals().is_empty()`):
  - `serving_settings_refuses_idle_steps_without_the_prewarm_worker`: `idle = [IdleStep::Prewarm]` with `prompt_cache.prewarm_chunk_tokens = 0`: `vec![IdleNeedsWorker]`; with `prompt_cache.byte_budget = 0`: `vec![IdleNeedsWorker]`; with the defaults (`byte_budget = 2147483648`, `max_entries = 4`, `prewarm_chunk_tokens = 256`): empty; `idle = []` with `prewarm_chunk_tokens = 0`: empty.
  - `serving_settings_refuses_draft_branches_beyond_the_entry_limit`: with `draft(n) = IdleStep::Draft(DraftStep { branches: n, max_tokens: 256, temperature_milli: 800, lead: Vec::new(), keep: false })` and the default `max_entries = 4`: `[Prewarm, draft(3)]` is empty (3 + 1 = 4); `[Prewarm, draft(4)]` is `vec![DraftBranchesExceedEntries { branches: 4, max_entries: 4 }]`; `[Prewarm, draft(5)]` is `vec![DraftBranchesExceedEntries { branches: 5, max_entries: 4 }]`; `[draft(5), draft(6)]` is two refusals in list order; `[draft(0)]` is empty; `idle = []` with `prompt_cache.follow_up_branches = 5` is `vec![DraftBranchesExceedEntries { branches: 5, max_entries: 4 }]`; `idle = []` with `follow_up_branches = 3` is empty.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_19 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_refuses_(idle_steps_without_the_prewarm_worker|draft_branches_beyond_the_entry_limit)/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/refusal.rs`, `proxima-model-interop/src/serving_settings/refusals.rs`
- commit: `feat(interop): refuse idle steps without a worker or entry room`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a byte-budget row (the entry size is unknown before the first request: unmeasured, so it cannot be a settings row); add a compact or judge row.
- gpu: none

### 2.24 Refuse placement settings that would be silently ignored

- id: FT2.24
- needs: FT2.19
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/generate/decode.rs::qwen35moe_monolithic_all_low_enabled` (~line 7264: `pre_gather && uses_gpu && requested`) and the `monolithic_high_mmap_requested` binding (~line 3126: `qwen35_pre_gather_requested && runtime.uses_gpu() && serving_config.qwen35moe_monolithic_high_mmap`): both modes are inert without `pre_gather`;
  - `interop/serving.rs::apply_serving_config` (~lines 1397 to 1411): `layer_window` must be 1 or 2, and 2 requires `pre_gather` and no `persistent_cuts` (a run-time refusal today);
  - `proxima-model-interop/Cargo.toml` feature `qwen35moe-expert-prefetch` (~line 17, `["std"]`) and `interop/generate/decode.rs` (~line 3403): the prefetch field does nothing when the feature is off, and nothing validates it;
  - `interop/serving_settings/refusals.rs` (FT2.19).
- change:
  1. `refusal.rs`: variants (lowercase `#[error]`): `MonolithicAllLowNeedsPreGather` ("placement.experts.monolithic_all_low is ignored unless pre_gather is on"), `MonolithicHighMmapNeedsPreGather` ("placement.experts.monolithic_high_mmap is ignored unless pre_gather is on"), `LayerWindowNeedsPreGatherWithoutCuts` ("placement.experts.layer_window 2 needs pre_gather on and persistent_cuts off"), `PrefetchNeedsBuildFeature` ("placement.experts.prefetch needs a build with the expert prefetch feature"). `field_path`: `"placement.experts.monolithic_all_low"`, `"placement.experts.monolithic_high_mmap"`, `"placement.experts.layer_window"`, `"placement.experts.prefetch"`.
  2. `refusals.rs`: `check_placement(&self, found)`, with `experts = &self.placement.experts`: push `MonolithicAllLowNeedsPreGather` when `experts.monolithic_all_low && !experts.pre_gather`; push `MonolithicHighMmapNeedsPreGather` when `experts.monolithic_high_mmap && !experts.pre_gather`; push `LayerWindowNeedsPreGatherWithoutCuts` when `experts.layer_window == 2 && (!experts.pre_gather || experts.persistent_cuts)`; push `PrefetchNeedsBuildFeature` when `experts.prefetch && !cfg!(feature = "qwen35moe-expert-prefetch")`. Call it from `refusals()` after `check_schedule`.
- test: three tests in `refusals.rs` `tests` (each also asserts `ServingSettings::default().refusals().is_empty()`):
  - `serving_settings_refuses_monolithic_modes_without_pre_gather`: `monolithic_all_low = true`, `pre_gather = false`: `vec![MonolithicAllLowNeedsPreGather]`; `monolithic_high_mmap = true`, `pre_gather = false`: `vec![MonolithicHighMmapNeedsPreGather]`; both true, `pre_gather = false`: both, in that order; both true with `pre_gather = true`: empty.
  - `serving_settings_refuses_layer_window_two_without_pre_gather_or_with_cuts`: `layer_window = 2`, `pre_gather = false`: `vec![LayerWindowNeedsPreGatherWithoutCuts]`; `layer_window = 2`, `pre_gather = true`, `persistent_cuts = true`: the same; `layer_window = 2`, `pre_gather = true`, `persistent_cuts = false`: empty; `layer_window = 1` with `pre_gather = false` and `persistent_cuts = true`: empty.
  - `serving_settings_refuses_prefetch_without_the_build_feature`: `prefetch = true`: `if cfg!(feature = "qwen35moe-expert-prefetch") { Vec::<ServingRefusal>::new() } else { vec![PrefetchNeedsBuildFeature] }`; `prefetch = false`: empty.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_24 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_refuses_(monolithic_modes_without_pre_gather|layer_window_two_without_pre_gather_or_with_cuts|prefetch_without_the_build_feature)/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo clippy -p proxima-model-interop --features std,conflaguration,qwen35moe-expert-prefetch --all-targets`
- stage: `proxima-model-interop/src/serving_settings/refusal.rs`, `proxima-model-interop/src/serving_settings/refusals.rs`
- commit: `feat(interop): refuse placement modes that would be ignored`
- done when: the expect line printed, clippy clean under both feature sets, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a row that reads a model descriptor (the engine's routed notion is per architecture today, so a descriptor row would pass settings the engine ignores); mirror the `1 | 2` window range check (`apply_serving_config` keeps it); add an `execution` enum.
- gpu: none

### 2.20 Share the phase-schedule refusal between the config check and the decode loop, and refuse it in settings

- id: FT2.20
- needs: FT2.24
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/generate/decode.rs::run_decode_loop_from_ids` (~line 2988): the `if !serving_config.phase_schedule.prefill_before_decode { return Err(InteropError::UnsupportedServingConfig(..)) }` block and its comment block (~lines 3092 to 3106), placed before any decode-loop state is built. An uncacheable request reaches it first: `generate/prompt_cache.rs::run_decode_loop_through_cache` (~line 1196) returns into `run_decode_loop_from_ids` at ~line 1221, before its own `apply_serving_config` call at ~line 1237;
  - `interop/serving.rs::apply_serving_config` (~line 1236): the admission check (`max_concurrent_requests`, ~line 1250) is the shape to copy; `supported_default` (~line 2110) and `interop/serving.rs::tests::scheduling_levels_are_independent` (~line 2172 at a7c08c4c; its local binding `phase_changed` at ~line 2187 builds `prefill_before_decode: false` and the test asserts only that the other two schedule levels are unchanged);
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity` (~line 668) and `::Checkpoint::open` (~line 112): how an integration test opens the gemma4 E2B checkpoint (env `PROXIMA_ARCH_GEMMA4_E2B_GGUF`, never skips), tokenizes with `proxima_tokenizer::gguf::vocab_from_metadata`, and calls `LoadedModel::generate_from_ids` with `PromptCacheConfig::off()`;
  - `interop/serving_settings/refusals.rs` (FT2.24): the `check_*` method style.
- change:
  1. `interop/serving.rs`: add `pub(crate) fn check_phase_schedule(config: &ServingConfig<'_>) -> Result<(), InteropError>`: `Ok(())` when `config.phase_schedule.prefill_before_decode`, otherwise `Err(InteropError::UnsupportedServingConfig("phase_schedule.prefill_before_decode=false: interleaving prefill and decode steps across sequences is not implemented yet".into()))` (the same text as the block it replaces). Call it from `apply_serving_config` right after the admission check: `check_phase_schedule(config)?;`.
  2. `interop/generate/decode.rs`: replace the inline block with `check_phase_schedule(serving_config)?;` in the same place (import it from `crate::serving`); keep the comment block above it, shortened to one lowercase line saying the check runs before any loop state is built; keep `let one_evaluation_prefill_requested = prefill_one_evaluation_requested(serving_config);` as it is.
  3. `refusal.rs`: variant `#[error("phase_schedule.prefill_before_decode=false: interleaving prefill and decode steps across sequences is not implemented yet")] PhaseInterleaveUnsupported`; `field_path` returns `"phase_schedule.prefill_before_decode"`. `refusals.rs`: `check_phase(&self, found)` pushes it when `!self.phase_schedule.prefill_before_decode`; call it from `refusals()` after `check_placement`.
  4. `proxima-model-interop/tests/phase_schedule_refusal.rs` (new; `#![cfg(feature = "std")]` and `#![allow(clippy::unwrap_used, clippy::expect_used)]` as `arch_data_baseline.rs` has): the one model-loading test below.
- test: three tests.
  - `serving_settings_refuses_phase_interleave_unsupported` in `refusals.rs` `tests`: `phase_schedule.prefill_before_decode = false`: `refusals() == vec![PhaseInterleaveUnsupported]`, `validate()` is `Err(Validation { errors })` with `errors[0].path == "phase_schedule.prefill_before_decode"`; `true`: empty.
  - `serving_section_apply_serving_config_refuses_phase_interleave` in `serving.rs` `tests`: `apply_serving_config(&ServingConfig { phase_schedule: PhaseSchedule { prefill_before_decode: false }, ..supported_default() }, 6)` is `Err(InteropError::UnsupportedServingConfig(message))` with `message.contains("prefill_before_decode=false")`; `apply_serving_config(&supported_default(), 6)` is `Ok(())`.
  - `decode_entry_refuses_phase_interleave_on_the_uncached_path` in `tests/phase_schedule_refusal.rs`: the path that returns into the decode loop before any config check. Load the gemma4 E2B checkpoint once (path from env `PROXIMA_ARCH_GEMMA4_E2B_GGUF`, default the path `arch_data_baseline.rs::GEMMA4_E2B` names; a missing file fails the test with its path and the env name). Tokenize `"The capital of France is"` with the checkpoint's vocabulary and BOS; assert the ids are not empty. Build `config = ServingConfig { prompt_cache: PromptCacheConfig::off(), gpu_layers: 0, ..ServingConfig::default() }` and set `config.phase_schedule.prefill_before_decode = false` (`PhaseSchedule` is not exported, so the field is assigned). Assert `model.generate_from_ids(&ids, 1, &config, &mut |_event| ControlFlow::Continue(()))` is `Err(InteropError::UnsupportedServingConfig(message))` with `message == "phase_schedule.prefill_before_decode=false: interleaving prefill and decode steps across sequences is not implemented yet"` (the exact text, so a changed message or a deleted refusal fails). Then the same call with `prompt_cache: PromptCacheConfig::standard()` (the cached path, which refuses in `apply_serving_config`) returns `Err(InteropError::UnsupportedServingConfig(message))` with the same `message`. `gpu_layers: 0` keeps the run on the CPU path; the refusal fires before any forward, so the run costs one model load.
  Placement of the shared call is checked by two greps in `validate`, because no test can observe that it runs before loop state is built.
- validate: run in order, one model-loading process at a time and only when `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` prints nothing:
  1. `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_20 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_refuses_phase_interleave_unsupported|serving_section_apply_serving_config_refuses_phase_interleave/)'`
  2. `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_20 cargo nextest run -p proxima-model-interop --features std,conflaguration -j 1 -E 'binary(phase_schedule_refusal) & test(/decode_entry_refuses_phase_interleave_on_the_uncached_path/)'`
  3. `git grep -c "check_phase_schedule" -- proxima-model-interop/src/generate/decode.rs`
  4. `git grep -n -A1 "check_phase_schedule(serving_config)?;" -- proxima-model-interop/src/generate/decode.rs`
- expect: 1: `2 passed`; 2: `1 passed`; 3: `proxima-model-interop/src/generate/decode.rs:2` (the import and the call); 4: two lines, the second of which contains `one_evaluation_prefill_requested` (the call sits where the inline block stood, before that statement)
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`; `cargo clippy -p proxima-model-interop --features std,metal --all-targets`; `cargo nextest run -p proxima-model-interop --features std -E 'test(/^serving::tests::/)'` prints `30 passed` (derived: the 27 tests on main, plus one each from the attention-section and assemble-list cards that land earlier in this chain, plus this card's one; not run)
- stage: `proxima-model-interop/src/serving_settings/refusal.rs`, `proxima-model-interop/src/serving_settings/refusals.rs`, `proxima-model-interop/src/serving.rs`, `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/tests/phase_schedule_refusal.rs`
- commit: `fix(interop): refuse phase interleaving in settings and config checks`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list (four source files plus the new test file, which the size rule does not count: this card exceeds the three-file rule by one on purpose; `decode.rs` and `serving.rs` are one behaviour-preserving sharing of one guard, and splitting it would leave the check in two copies or the knob silently ignored), and the commit landed with that message. If the executor's plan needs a sixth file, it stops and reports.
- do not: delete the guard from `decode.rs` without the shared call in the same place; move the check later in the decode function; change the message text; add a second model-loading test; use a qwen checkpoint.
- gpu: one run (`-j 1`, one model at a time, the CPU path), waiting for a quiet box (CARDS.md machine safety)

### 2.21 Check every refusal names its field path, and the exit count

- id: FT2.21
- needs: FT2.20
- budget: 20 min
- crate(s): proxima-model-interop (features: std,conflaguration)
- read first:
  - `interop/serving_settings/refusal.rs` and `refusals.rs` (FT2.20);
  - `interop/lib.rs` (~line 157): the `pub use` lines this card extends.
- change: `proxima-model-interop/src/lib.rs`: add `ServingRefusal` to the `pub use serving_settings::{..}` line (it is the public error of the public `refusals()` method, so it must be nameable). Nothing else outside tests, unless the test finds a path that disagrees with the table below; then fix `ServingRefusal::field_path` in `refusal.rs`.
- test: add `serving_settings_refusal_field_paths` in `refusals.rs` `tests`. For each of the 14 rows build a settings (and for the two descriptor rows the descriptors of FT2.17) that triggers that row (rows 2 and 3 excepted, below), call `refusals_for(&descriptor)`, and assert the refusal's `field_path()` equals the path; and, through `validate()` for the twelve model-free rows, that `Err(conflaguration::Error::Validation { errors })` has `errors[0].path` equal to the same path and `errors[0].message` is non-empty:

  | row | variant | path |
  |---|---|---|
  | 1 | BlockTokensConflict { kv: 256, prompt_cache: 128 } | `prompt_cache.block_tokens` |
  | 2 | ReadNeedsAttentionLayers { layer: 5 } | `attention.read` |
  | 3 | ReadNeedsTwoRangeCache | `attention.read` |
  | 4 | AssembleNeedsPromptCache | `prefill.assemble` |
  | 5 | ShiftNeedsReuse | `prefill.assemble` |
  | 6 | EvictionNeedsOldestLast | `kv.eviction` |
  | 7 | IdleNeedsWorker | `schedule.idle` |
  | 8 | DraftBranchesExceedEntries { branches: 5, max_entries: 4 } | `schedule.idle` |
  | 9 | MonolithicAllLowNeedsPreGather | `placement.experts.monolithic_all_low` |
  | 10 | MonolithicHighMmapNeedsPreGather | `placement.experts.monolithic_high_mmap` |
  | 11 | LayerWindowNeedsPreGatherWithoutCuts | `placement.experts.layer_window` |
  | 12 | PrefetchNeedsBuildFeature | `placement.experts.prefetch` |
  | 13 | PhaseInterleaveUnsupported | `phase_schedule.prefill_before_decode` |
  | 14 | ReadHookNotBound | `attention.read` |

  The test iterates this table and asserts `rows.len() == 14` first. Rows 2 and 3 use `Operand` settings, so their `refusals_for` list also holds `ReadHookNotBound` ahead of the row (FT2.34, FT2.17): for those two rows the assertion finds the row's variant in the list and checks its `field_path()`. Row 12 fires only when the build feature is off: assert its `field_path()` on the constructed variant always, and its `validate()` path only under `cfg(not(feature = "qwen35moe-expert-prefetch"))`. The `validate()` assertions run for every row except rows 2 and 3.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_21 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_(kv_variants|attention_read_variants|prefill_assemble_variants|schedule_idle_variants|placement_variants|default_parity|round_trip|refuses_block_tokens_conflict|refuses_read_hook_not_bound|refuses_read_needs_attention_layers|refuses_read_needs_two_range_cache|refuses_assemble_needs_prompt_cache|refuses_shift_needs_reuse|refuses_eviction_without_oldest_last|refuses_idle_steps_without_the_prewarm_worker|refuses_draft_branches_beyond_the_entry_limit|refuses_monolithic_modes_without_pre_gather|refuses_layer_window_two_without_pre_gather_or_with_cuts|refuses_prefetch_without_the_build_feature|refuses_phase_interleave_unsupported|refusal_field_paths)/)'`
- expect: `21 passed` (5 section variants, 2 whole-surface tests, 13 refusal tests, 1 field-path test); zero tests matched is red
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/refusals.rs`, `proxima-model-interop/src/lib.rs`
- commit: `test(interop): check every refusal names its config field path`
- done when: the expect line printed with `21 passed`, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a 15th refusal variant; add a doctest.
- gpu: none

## spec drift

Each item is a place where the spec, a sketch, the previous cut or a sibling card disagrees with main a7c08c4c, or leaves a choice these cards decided.

1. `interop/serving.rs` module doc (lines ~17 to 24) says the crate has no `serde`; `proxima-model-interop/Cargo.toml` declares `serde = { workspace = true, features = ["derive", "alloc"] }` unconditionally (~line 327). These cards add serde derives to `serving_grammar.rs`, `RopeScaling`, `GdnPrefillBackend`, `EvictionRule`, `IdleStep` and `DraftStep`, and keep `ServingConfig` and its `Copy` section structs serde-free. The 7B forward regression the module doc cites (reverted at `23a6688`) is not re-measured here (unmeasured).
2. The previous cut's `omega::MathMode` premise (two variants) is false at a7c08c4c: `Safe`, `Relaxed`, `Fast` (`omega/src/metal/pipeline_buffers_upload.rs` ~line 34). The previous cut's `mistral_descriptor_from_shape` call shape is stale: the function takes a trailing `&FamilyProfile` and no `RopePairing` (`proxima-tensor/src/spec/descriptor.rs` ~line 247).
3. The previous cut said `ServingConfig` has five full literals; at a7c08c4c it has six (the `Default` impl, `fully_supported_config_applies_without_error` and four `via_full_literal`).
4. The previous cut deleted the `decode.rs` phase guard on the premise that `apply_serving_config` runs first. It does not on the uncached path (decision D10); the guard is shared instead.
5. Sketch 08 and the triage verdict name `ReadSpec::Dense | Block { keep_ratio_milli, min_blocks, local_blocks }` with a separate `summary_tokens`. This file builds `Dense | Operand` (the sibling read-hook cards' premise, owner direction that techniques are not library code); see "designs abandoned".
6. Sketch 10 and the previous cut name a seal summary list, `kv.seal.horizon_rows`, a host tier size and an allocation mode. None is built: the horizon is `prompt_cache.seal_horizon_rows` (the seal card), the host tier is `prompt_cache`, and the allocation mode is a technique with no hook gap.
7. `ServingSettings::refusals_for(&ModelDescriptor)` holds the two descriptor read rows (decision D9). `validate()` covers the model-free rows, including `ReadHookNotBound` (FT2.34), which refuses `operand` until the decode arm exists.
8. `ServingSettings::as_serving_config` takes the caller-held weight-precision slice (D2); a settings text that writes `as_serving_config()` with no argument would not compile.
9. `speculative` and `prompt_cache` keep their own env prefixes (D6), so `PROXIMA_PROMPT_CACHE_BYTE_BUDGET=0` still works.
10. Sketch 03 lists nine placement gaps. This file answers only the ones that need no engine change: the inert combinations of the monolithic modes and the layer window (rows), the prefetch feature gate (row), and the section shape with the four budgets (data). Not built, because each sits inside `generate/decode.rs` or `generate/pregather.rs` paths that run only for the one architecture whose `Architecture::ffn_routing` returns `Routed`: the `execution` enum, the decision constants (`ema_rate`, `hysteresis_margin`, `min_dwell_tokens`), the sidecar loader, the retention enums, a read-cache limit separate from `residency_budget_bytes` (`generate/decode.rs` ~line 3422), and making "routed" a descriptor fact. A granite proof of the placement stage needs both the slice-0 granite card and that engine change; neither exists at a7c08c4c.
11. CC-1 (a hash-chain cache key over layer groups) and CC-2 (an exactness contract on a plan) change the key's shape for producers that do not exist yet (non-identity codecs, adapters, steering). The key gains exactly one field here (`read`); the chain arrives with its first producer.
12. CC-4 (a replay deeper than the seal horizon) is the replay-versus-seal row in the rectify slice (FT7.7, amended above). Its second half, "the widest verify shape", is not built here: the width is computed by `interop/generate/kv_ring.rs::speculative_draft_limit`, a `pub(super)` function, and a settings row would copy its body.
13. `kv.eviction` is a settings-only list passed to a setter (decision D1). It is therefore outside `ServingConfig`, and the cache key; no field of it can change a cached row (it picks which entry a full cache gives up, not what a row holds). There is no tier list in settings (see "dropped").
14. `conflaguration::Error` and `proxima_tensor::NumericPolicy` are `#[non_exhaustive]`: tests match `Error::Validation` with `let .. else`, and build a `NumericPolicy` by assigning fields on `bit_exact()`.

## slice exit

Run, in order, from the checkout holding the cards' commits, with `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_2_exit` (remove it after). Zero tests run is red for every filter.

1. `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_(kv_variants|attention_read_variants|prefill_assemble_variants|schedule_idle_variants|placement_variants|default_parity|round_trip|refuses_block_tokens_conflict|refuses_read_hook_not_bound|refuses_read_needs_attention_layers|refuses_read_needs_two_range_cache|refuses_assemble_needs_prompt_cache|refuses_shift_needs_reuse|refuses_eviction_without_oldest_last|refuses_idle_steps_without_the_prewarm_worker|refuses_draft_branches_beyond_the_entry_limit|refuses_monolithic_modes_without_pre_gather|refuses_layer_window_two_without_pre_gather_or_with_cuts|refuses_prefetch_without_the_build_feature|refuses_phase_interleave_unsupported|refusal_field_paths)/)'`: expect `21 passed`.
2. `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_grammar_|serving_scalars_|serving_section_|text_form_|cache_key_separates_read_specs|cache_key_ignores_the_assemble_list/)'`: expect `22 passed` (grammar 2; scalars 9 without the metal feature: the four scalar-card tests, the admission, phase and expert residency level tests, the speculative and prompt cache test, and `serving_scalars_prompt_cache_block_tokens_is_optional`; section 7: attention default, prefill default, cache type, attention section, expert placement, placement budget, apply-config phase refusal; text form 2; cache key 2).
3. `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets` and `cargo clippy -p proxima-model-interop --features std,metal --all-targets`: clean.
4. `cargo check -p proxima-model-interop --no-default-features`: passes (the alloc floor still builds with `serving_grammar.rs`).
5. `cargo check -p proxima-model-interop --features std,metal --all-targets` plus `cargo check --examples` at the repo root: every `ServingConfig { .. }` literal in `examples/` still compiles.
6. Model oracle for the two edited hot files (`serving.rs`, `generate/decode.rs`), one GPU run, only when `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` prints nothing: `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^(arch_data_digest|llama_parity)_gemma4/)'`: expect `4 passed` (digests and recorded-id parity for gemma4 E2B and gemma4 26B). Then, in a separate process, `cargo nextest run -p proxima-model-interop --features std,conflaguration -j 1 -E 'binary(phase_schedule_refusal)'`: expect `1 passed` (the uncached refusal, one gemma4 E2B load on the CPU path). The granite checkpoint joins the first command when its card lands.
7. `git grep -n '```' -- proxima-model-interop/src/serving_settings.rs proxima-model-interop/src/serving_settings | wc -l` prints `0`.
8. `git diff --stat main..HEAD` lists only: `proxima-model-interop/src/{serving_grammar.rs, serving_settings.rs, serving_settings/*.rs, serving.rs, lib.rs, rope_scaling.rs, prompt_cache_settings.rs, generate/prompt_cache_key.rs, generate/resident_plans.rs, generate/prompt_cache.rs, generate/prewarm_follow_up.rs, generate/decode.rs}` and `proxima-model-interop/tests/phase_schedule_refusal.rs`.
9. Remove `/private/tmp/cargo_target_ft_2_*` and any scratch under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/`.

Slice 2 is a series of 32 commits, each green. The slice is marked done only with the printed lines above.

## not built here, and why

- A bridge that calls the prompt-cache setters from a loaded settings value (`set_eviction_rules` and the idle-schedule setter). Call site both ways: `model.set_eviction_rules(&settings.kv.eviction)` against a `LoadedModel::install(settings)` that makes the same three calls; the two do the same work, so no function is built (decision D1).
- A decode section and a cascade section (dropped, above).
- Placement engine changes and a granite placement proof (spec drift item 10).
- The widest-verify-shape half of the seal-horizon row (spec drift item 12).
