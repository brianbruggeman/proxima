# fsm techniques: slices

Each slice is one commit, green at every commit. It lands by `git merge --ff-only` in the
checkout holding main. AC commands are spelled out in SPEC.md. Every slice also runs AC24
(the llama parity count stays at 4 passed of 7).

Paths are relative to the proxima repo root. New examples belong to `proxima-model-interop`.
Each gets an `[[example]]` entry in `proxima-model-interop/Cargo.toml`, with `test = true`
where an AC runs nextest on it.

Order constraints (SPEC.md "dependencies"):
- slice 0 runs on main as it is;
- slice 1 onward follows architecture-as-data 10b;
- slice 2 follows architecture-as-data 4c (formerly 10a);
- slice 6 follows decode-as-data 1, 2, 3 and 7, and this spec's slices 3 and 4;
- slice 7 follows decode-as-data 4 and 7, and slice 6;
- slice 8 follows decode-as-data 2;
- slice 9 follows decode-as-data 7, and slice 8;
- slice 10 follows slices 1, 4 and 5; slice 2 is split around it: its cards FT2.32 to FT2.15a land first, and its cards from FT2.22 onward follow FT5.7, FT5.20 and FT10.4 (FT2.22 needs FT5.7 and FT10.4, FT2.18 needs FT5.20; no slice 5 or slice 10 card cites a slice 2 card);
- slice 11 follows decode-as-data 3;
- slice 12 follows slice 11;
- slice 13 follows slice 12;
- slice 14 follows slices 5 and 8;
- slice 15 follows slice 7;
- slice 16 follows architecture-as-data 10c and every other slice here;
- slice 17 follows every other slice here.

### slice 0: oracles, worked examples and fixtures before code; bounded fixes

ACs: AC25, AC0 control, AC24 baseline.

Files:
1. Re-read the UNVERIFIED rows of `~/.cache/proxima-fsm-audit/techniques.md` against the PDFs: ReSA's p direction, CRE's accept rule, and Milvus seal ordering. Correct the SPEC source register.
2. `proxima-tensor/specs/fsm-techniques/worked-examples.md` (`/algorithm-development`), covering:
   - PAV on 8 points;
   - q̂ at α=0.1, n=20;
   - k-means k=2 on 6 points, plus silhouette;
   - λ argmin over 2 models x 3 clusters;
   - Tchebycheff over 3 thresholds;
   - the seal trace: 10 appends and 3 rewinds;
   - the LRU trace;
   - HKVD per-layer selection on 3 chunks;
   - ReSA block scoring on 4 blocks, and W9 for AC9;
   - the CLT budget at ε = δ = 0.05;
   - τ for rectify, and the logprob tolerance for AC18;
   - the rotate_rows concatenation;
   - the five judge traces;
   - W7, a 12-element score vector with its hand-derived top-fraction set;
   - W8, 4 sealed blocks plus a 5-row tail at keep_ratio 0.5, with its attended row set;
   - W14, a 2-token 3-vocab teacher/student pair with its hand-derived KL.
3. `proxima-tensor/specs/fsm-techniques/env.sh`: source long-context's env.sh, then export `QWEN3_0_6B`.
4. Vendor from llama-server f1ea20621 into `proxima-model-interop/tests/fixtures/llama-parity/<checkpoint>/`, for the 4 parity checkpoints:
   - `followup_ids.json`: a 2-turn prompt where turn 2 reuses turn 1's prefix;
   - `n_probs.json`: `n_probs = 2` on the 3 vendored prompts;
   - `cache_reuse_ids.json`: llama-server started with `--cache-reuse 64`, two requests where request 2 repeats request 1's middle chunk at a shifted position.
5. Add the `fsm_oracle_control_` tests.
6. Run AC24 at the slice-0 HEAD with no other GPU suite running, and record the printed line here. A count other than SPEC AC24's literal is a regression to root-cause before slice 1. It is not a new literal.
7. Bounded fixes: delete `ServingFsmError::NotSupported` (`proxima-model-interop/src/serving_fsm.rs:83-87`), and point `generate/block_bloom.rs:9` at `chunk_shift`.

Validation: AC25, then AC0, then AC24, then `cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_fsm/)'`.

Expected: 5 passed; 0; 4 passed of 7; 4 passed.

Done: [~] 2026-10-04.
- Item 7 is applied, uncommitted: serving_fsm 4/4 passed, clippy clean.
- Item 6 is done: AC24 at edd4163c gave 7 run, 4 passed (qwen2, gemma4_e2b, openchat, qwen3). The failures are qwen35 and qwen35moe (no fixture) and gemma4_26b (D2). Log: `~/.cache/proxima-fsm-audit/logs/p0-retry.log`.
- Items 1-5 are open.

Also landed in the working tree, uncommitted: 51 + 8 rustdoc link errors on main fixed, docs only. Together with item 7, 15 files under `proxima-model-interop/src` changed: 14 doc-only (including `block_bloom.rs`), plus `serving_fsm.rs`. Gates were re-run at edd4163c. `RUSTDOCFLAGS='-D warnings' cargo doc` gives 0 errors on std, std+metal and std+conflaguration, and clippy is clean. Logs are in `~/.cache/proxima-fsm-audit/logs/`.

### slice 1: FSM generic over its entry, at tier 1; action worked example

ACs: AC1, AC23, AC27, AC30.

Files:
- `proxima-core/src/serving_state.rs` is new, holding the body of `serving_fsm.rs`:
  - `Entry` replaces `u32` in `positions`, `last` and `draft` (:52, :55, :60, :68, :71);
  - `draft_prompt_lookup<Entry: PartialEq + Clone>` (:97);
  - the transitions (:116-277);
  - the four moved tests get a `serving_state_` prefix, with the names listed in SPEC AC1.
- The test module adds the three R19 action-speculation tests over a `BTreeMap` environment with a `u64` version.
- `proxima-core/src/lib.rs`: mod plus `pub use`.
- Delete `proxima-model-interop/src/serving_fsm.rs` and `lib.rs:69-70`; re-export from `proxima_core`.
- `generate/serving_backend.rs:121-241`: imports.

Validation: AC1, then AC23, then AC27, then AC30.

Expected: 4 passed; 3 passed; 1, 0 and 1 passed; 6 passed.

Feature wiring: `proxima-model-interop/Cargo.toml:314` declares `proxima-core` as `optional = true, features = ["alloc"]`. The re-export (AC27) needs it reachable under interop's `std` feature: add `dep:proxima-core` to `std`, a features edit, not a dependency edit.

Also in this slice: `proxima-core/src/speculation_settings.rs` (new, behind the existing `config` feature, `proxima-core/Cargo.toml:65`). It holds `SpeculationSettings` with the `accept` grammar of R22a and its lowering to the accept pipe (R22b).

Done: [ ]

### slice 2: `ServingSettings`, the whole `ServingConfig` as conflaguration, plus the reachability matrix

ACs: AC2.

Files:
- `proxima-model-interop/src/serving_settings.rs` is new:
  - `#[derive(Builder, Deserialize, Serialize, Settings, Validate)]` with `#[settings(prefix = "PROXIMA_SERVING")]`;
  - one field per `ServingConfig` field (`serving.rs:725-1066`);
  - nested `speculative: SpeculativeSettings` and `prompt_cache: PromptCacheSettings`;
  - the full sections `kv`, `attention`, `decode`, `prefill`, `schedule` and `cascade`, with every field R4, R8a, R10e, R11c, R15e and R17f name, and defaults that keep today's behaviour;
  - `as_serving_config`, modelled on `speculative_settings.rs:340`;
  - `Validate` implementing the 12 matrix rows.
- `serving.rs`: `Copy` mirrors of each section on `ServingConfig`; extend `Default` (:1165-1210).
- `decode.rs:3101-3106`: delete the run-time refusal.
- `lib.rs:153-155`: export.
- `generate/prompt_cache_key.rs`: `CacheKey::of` destructures the new fields.

Validation: AC2.

Expected: 23 passed.

Done: [ ]

### slice 3: top-fraction row selection, one expression and one lowering

ACs: AC7.

Files:
- `proxima-tensor/src/spec/primitives.rs`: `top_fraction_mask(program, scores, fraction, min_rows, keep_rows) -> NodeId`, next to `causal_mask_merged` (:1354).
  - rank = `Reduce Add` over `Elementwise Greater`, with an `IndexMap` broadcast;
  - selected = rank < max(min_rows, ceil(fraction·M)), OR membership in keep_rows.
- `omega/src/msl/selection_render.rs` is new: a radix-select recognizer for that subgraph.
- `omega/src/msl/emit_and_classify.rs`: classify it at `selection.top_fraction_min_rows`.
- `omega/omega-runtime.toml`: add that key.
- CPU reference tests in `proxima-tensor/src/cpu`.

Validation: AC7.

Expected: 2 passed, then 4 passed (including the two kernel-count tests: 0 below the threshold, 1 at it).

Done: [ ]

### slice 4: block seal and reserved allocation

ACs: AC3, AC6.

Files:
- `generate/residency_caches.rs:112-160`, `LayerCache`:
  - add a sealed-end row and per-block key min/max when key_minmax is configured;
  - `append` (:131) writes into the reservation under `allocation = reserve`;
  - `truncate` (:155) refuses below the sealed end with `InteropError::RewindIntoSealed`.
- `generate/kv_ring.rs:121`: ring layers seal only rows outside the ring.
- `generate/arena.rs` and `generate/prefix_trie.rs`: sealed blocks carry the content key; trie insert moves from request end to seal.
- `generate/device_kv.rs:61`: placed buffers come from the reservation.
- `generate/kv_reserve.rs` is new:
  - one `libc::mmap(MAP_ANON | MAP_NORESERVE)` per layer cache, sized for the resolved context;
  - pages are touched `page_blocks · block_tokens` rows at a time, with a mapped-page counter;
  - on the placed path, Metal `newBufferWithBytesNoCopy` over the range.

Validation: AC3, then AC6.

Expected: 6 passed; 5 passed.

Done: [ ]

### slice 5: tiers (device, host, disk), eviction policy, restore from disk

ACs: AC5, AC4.

Files:
- `generate/prompt_cache.rs:539`, `PromptCache`:
  - holds the `kv.tiers` list;
  - `take_best` (:629), `take_best_shifting` (:644) and `restore_nearest` (:510) search across tiers;
  - `kv.policy` demotes before it drops.
- `generate/block_file.rs` is new: the sealed-block file, with the header from R13a, read through `proxima-storage`'s mapping.
- `residency_caches.rs:707`: `PrefixState` restores from block refs.

Validation: AC5, then AC4.

Expected: 4 passed; 4 passed.

Done: [ ]

### slice 6: read sets, block top-n and sampled

ACs: AC28, AC8, AC10, AC26.

Files:
- `proxima-tensor/src/spec/primitives.rs`: `read_mask`, ANDed into decode-as-data's visibility expression.
  - The block arm: `top_fraction_mask` over block scores, expanded to rows, OR the local blocks, OR the unsealed tail. A block's score is the sum over the head group of max(q·kmax, q·kmin), with `block_kmax`/`block_kmin` as `Input` leaves.
  - The sampled arm: sink OR local OR top-k OR (uniform `Input` < budget/ns), with the 1/p weight as an `Elementwise Multiply` on the softmax numerator.
  - The CLT budget is computed in graph from base-rate samples, at the max-budget cost.
- `proxima-tensor/src/spec/descriptor.rs:299`: `build_forward` takes the read config. Under dense, no leaves are added, so the digest is unchanged.
- `omega/src/msl/cached_attention_render.rs:51` and `cached_attention_row_tiled.rs:31`: iterate selected block indices when a read-set input is bound.
- `generate/residency_caches.rs:1206`: `build_position_inputs_at` (added by decode-as-data slice 2; TODO.md item C7) emits the summary and uniform inputs.
- `generate/resident_plans.rs` (`DecodePlanKey`) and `resolve_cached_plan` (`residency_caches.rs:2755`; the bounded map from decode-as-data slice 7, TODO.md item D6): add the read config to the plan key, so a sparse-read plan and the dense plan are held side by side.
- Vendor `proxima-tensor/tests/fixtures/kv-captures/gemma4-e2b-layer2.f32`: 4096 K/V rows plus 2000 query rows from a real gemma4 E2B prefill of the war_and_peace prefix, captured through decode-as-data's layer-tap readout.

Validation: AC28, then AC8, then AC10, then AC26.

Expected: 4 passed; 4 passed; 1 passed; 4 passed. Then the supplementary `arch_data_digest_` run gives 7 passed.

Done: [ ]

### slice 7: Verify entry as config; ReSA rectify

ACs: AC11.

Files:
- `proxima-core/src/serving_state.rs`: `enter_verify` takes the proposer index the enter pipe chose. No new variant.
- The live loop (wired by architecture-as-data 10b):
  - the enter pipe evaluates `decode.rectify.every` and the drafter set, in config order;
  - the rectify proposer re-proposes the last f committed ids at their own positions;
  - accept takes all;
  - commit is decode-as-data's gather, writing over the same rows.
- `DecodePlanKey` gains the rectify verify shape: f rows at committed positions. It is held in decode-as-data D6's bounded map next to decode-1.

Validation: AC11.

Expected: 7 passed.

Done: [ ]

### slice 8: Prefill from an assembled cache

ACs: AC12.

Files:
- `proxima-core/src/serving_state.rs`: `Prefill { positions, start_position, cache }`; `start` accepts a non-empty cache; rewrite the `Prefill` doc.
- The live prefill builds positions from `start_position` through `build_position_inputs_at`, which decode-as-data slice 2 (TODO.md item C7) adds next to `build_position_inputs` (`residency_caches.rs:1206`).
- `generate/prompt_cache.rs:1219-1225`: express the `cacheable` bypass conditions as the absence of an assemble pipe.
- `prefill.assemble` lowers prefix -> shift -> load in config order.

Validation: AC12.

Expected: 5 passed.

Done: [ ]

### slice 9: CacheBlend selective recompute

ACs: AC13, AC14.

Files:
- `proxima-tensor/src/spec/descriptor.rs:299`, for blend:
  - all loaded K/V enter as `Input` leaves;
  - at the check layer, compute K/V for every row and the `Reduce Add` of |loaded - computed| per row;
  - `top_fraction_mask` picks the rows;
  - later layers narrow by the per-layer list;
  - unselected rows read loaded K/V through `Select`.
- `generate/chunk_shift.rs:173`: `rotate_rows` is reused for re-rotation.
- `DecodePlanKey` gains a blend flag, so a blend prefill plan never evicts the plain prefill plan.
- `prefill.assemble` gains `blend`.

Validation: AC13, then AC14.

Expected: 4 passed; 5 passed.

Done: [ ]

### slice 10: idle jobs, sleep and compact

ACs: AC17.

Files:
- `generate/prewarm.rs:321,397` and `generate/prewarm_queue.rs`: the queue holds `schedule.idle` jobs in config order.
  - sleep generates from the configured prompt over the source context, then seals its output blocks under that context's content key;
  - compact rewrites disk block files into segment files of `segment_blocks`.
- `generate/prewarm_gate.rs:21`: preemption at block boundaries for every job kind.

Validation: AC17.

Expected: 4 passed.

Done: [ ]

### slice 11: readouts (logprob, top-2 margin, N samples)

ACs: AC18.

Files:
- `generate/residency_caches.rs:3169`: `TokenEvent` gains `logprob: f32` and `top2_margin: f32`, computed from the logits row `select_decoded_token` already holds (`decode.rs:1319`).
- `decode.rs`: `decode.samples` completions, seeded `seed + i`.

Validation: AC18.

Expected: 9 passed.

Done: [ ]

### slice 12: cascade as one composed pipe

ACs: AC19.

Files:
- `proxima-model-interop/src/cascade.rs` is new: lowers the `cascade` settings to `router.and_then(tier_0).and_then(settle_0)…` over `Result<Answer, Request>` inside `Ok`, using `proxima-primitives/src/pipe/ext.rs:48`.
  - each tier is a loaded model's generate call;
  - each settle reads R16 readouts or a classifier model's output;
  - no new combinator and no judge trait.
- Tests on fake tiers driven by the worked traces.

Validation: AC19.

Expected: 9 passed.

Done: [ ]

### slice 13: calibration fitters and the cascade bench

ACs: AC20, AC22.

Files:
- `proxima-model-interop/examples/calibration_fit.rs` with `[[example]] test = true`: PAV, q̂, k-means with silhouette, λ argmin and Tchebycheff; writes the tables the `cascade` section reads.
- `proxima-model-interop/examples/cascade_bench.rs`.
- `proxima-model-interop/tests/fixtures/cascade/{calibration,held_out}.jsonl`: 200 rows each, exact-match QA built from war_and_peace needle questions, with labels from the needle text.

Validation: AC20, then AC22.

Expected: 5 passed; the per-arm counts and miscovered <= 49 assert.

Done: [ ]

### slice 14: cartridge load and trainer

ACs: AC15, AC16, AC29.

Files:
- `generate/block_file.rs`: a cartridge is a block file with base position 0; the descriptor-digest refusal.
- `prefill.assemble` `load{cartridge}`.
- `proxima-model-interop/examples/cartridge_train.rs`:
  - self-study conversations from the 5 seed-prompt kinds;
  - KL distillation with K/V rows as `proxima-autograd` leaves;
  - writes the cartridge.
- First, run `cargo nextest run -p proxima-autograd -E 'test(/grad_/)'` and list the `Op` kinds it covers. Any missing gradient of the five lands in this slice.

Validation: AC15, then AC16, then AC29 (the training run, then `trained_cartridge_loads_` bound to the printed path).

Expected: 4 passed; 3 passed; 200 `step=` lines, 1 `kl_start=`, 1 `kl_end=` and 1 `cartridge=`, then 1 passed.

The `cartridge_train` example's `[[example]]` entry sets `test = true`.

Done: [ ]

### slice 15: long-context read arms

ACs: AC9.

Files:
- `proxima-model-interop/examples/long_context_niah.rs`: add `--read` (a list of dense, block, block-rectify) and `--rectify`, parsed beside `--kv` (:217). Arms are interleaved per iteration; print `arm=`, `found=` and `kv_rows_read=`.

Validation: AC9.

Expected: 3 `arm=` lines, and the block arm's `kv_rows_read` = W9.

Done: [ ]

### slice 16: one validated config per technique; final guard

ACs: AC21, AC0.

Files:
- `proxima-model-interop/src/serving_settings.rs`: 12 doctests on `ServingSettings`, for T1-T9, T13, T14a and T14b.
- `proxima-core/src/speculation_settings.rs`: 3 doctests on `SpeculationSettings`, for T10-T12.
- Each doctest loads, validates and lowers its TOML, and asserts that the row's section is not the default.
- `docs/configuration.md`: link each to its doctest.

Validation: AC21, then AC0.

Expected: 12 passed then 3 passed (15 in total); 0.

Done: [ ]

### slice 17: sans-IO conformance suite (the end-state gate)

ACs: AC31.

Files:
- `proxima-core/src/serving_state/scripted.rs` (test-only module, `#[cfg(test)]`): the scripted
  backend, a pure function from (entries, positions, cache) to per-row readouts, read from a
  table.
- `proxima-core/src/serving_state/sansio_tests.rs`: 15 tests (transitions 2, places 7,
  properties 6).
- `proxima-model-interop/src/generate/sansio_tests.rs`: 4 tests (seal, place, assemble, host
  read), over in-memory tiers and the scripted backend.

Validation: AC31.

Expected: 15 passed; 4 passed; 0.

Done: [ ]

Cards: every slice is cut into cards in `tasks/NN-<slice>.md`, under the rules in CARDS.md.
