---
status: admitted (spec-auditor ADMIT on pass 12, 2026-10-04; R 49, AC 31, 0 orphaned, 0 dangling)
---

# fsm techniques: fifteen serving techniques (fourteen named; vAttention is two) as configuration of one FSM

## problem

For a caller who configures proxima serving only through conflaguration, this spec raises the
number of the 15 source-register rows (fourteen named techniques; vAttention is two) whose
configuration loads into `ServingSettings`, validates, and lowers to a `ServingConfig` with
that row's section set, from 0 of 15 (AC21 runs 0 doctests at main edd4163c) to 15 of 15
(AC21: 15 passed).

## context

Owner, 2026-10-04: "I want those to be _supportable_ through the programmable logic that lowers
with conflaguration", and "the full todo setup so we can impl directly". The fourteen
techniques are T1-T14 in the source register below.

Cited paths, all at main edd4163c:
- FSM: `proxima-model-interop/src/serving_fsm.rs`.
- Live loop: `proxima-model-interop/src/generate/decode.rs`.
- KV and events: `proxima-model-interop/src/generate/residency_caches.rs`.
- Prompt cache:
  - `proxima-model-interop/src/generate/prompt_cache.rs`;
  - `prefix_trie.rs`, `arena.rs`, `block_bloom.rs`, `chunk_shift.rs`;
  - `kv_ring.rs`, `device_kv.rs`;
  - `prewarm.rs`, `prewarm_gate.rs`, `prewarm_queue.rs`.
- Config:
  - `proxima-model-interop/src/serving.rs`;
  - `speculative_settings.rs`, `prompt_cache_settings.rs`, `lib.rs`.
- Lowering:
  - `proxima-tensor/src/spec/descriptor.rs`, `proxima-tensor/src/spec/primitives.rs`;
  - `proxima-tensor/src/op.rs`, `proxima-tensor/src/map.rs`.
- Kernels:
  - `omega/src/msl/cached_attention_render.rs`, `cached_attention_row_tiled.rs`;
  - `emit_and_classify.rs`, `omega/omega-runtime.toml`.
- Pipe algebra: `proxima-primitives/src/pipe/ext.rs`.
- Harness and corpus:
  - `proxima-model-interop/examples/long_context_niah.rs`;
  - `proxima-model-interop/examples/data/war_and_peace.txt`;
  - `proxima-tensor/specs/long-context/env.sh`.
- Parity tests: `proxima-model-interop/tests/arch_data_baseline.rs` (7 `llama_parity_*` fns).
- Research extraction: `~/.cache/proxima-fsm-audit/techniques.md`.

## refutation condition

This is the wrong thing to build if any of these is observed:

- A technique needs a sixth `Op` variant (`proxima-tensor/src/op.rs:191`). Today there are
  five: Input, Elementwise, Reduce, Iota and Constant. If a technique needs a sixth, it is
  not configuration of the existing algebra; it becomes a recorded irreducible, like MoE
  routing in `architecture-as-data`.
- A technique needs a `ServingState` variant beyond decode-as-data's set. That set is
  Prefill, Decode, Verify (with refine), Accept, Rollback and Done. If one is needed, the
  FSM is not the composition point and the frame is wrong.
- A lossless configuration changes token ids against its non-technique control. The
  lossless configurations are LMCache tiers, the vAttention-2024 allocator, CacheBlend at
  ratio 1.0, ReSA at keep ratio 1.0, and agent speculation under exact verification.

## source register (primary sources; parameters as published)

The research extraction lives at `~/.cache/proxima-fsm-audit/techniques.md`. It was read
through WebFetch extraction, so its UNVERIFIED rows are re-read from the PDF in slice 0
before an AC encodes them.

| # | technique | source | mechanism the FSM must host | published defaults |
|---|---|---|---|---|
| T1 | ReSA (Rectified Sparse Attention) | arXiv:2506.04108 | Block-sparse decode. Each block is scored against its key min/max, and the top-n blocks plus the local block are read. Every f tokens a dense re-encode overwrites the last f KV rows. | b=16, p=0.9, n_min=16, n_local=1, f=32 |
| T2 | Cartridges | arXiv:2506.06266 | A trained KV prefix Z (p rows per layer), loaded into the prefix slots. Cartridges compose by concatenation. Training is self-study plus context distillation (KL). | p in {128..8192}; batch 64; seq 1024; 1 epoch |
| T3 | sleep-time compute | arXiv:2504.13171 | While idle, S(c) -> c'. At query time, T_b(q, c'). | up to 10 rethink calls; k in {1,5,10} |
| T4 | CacheBlend | arXiv:2405.16444 | Non-prefix chunk KV reuse: re-rotate K, then recompute only the high-KV-deviation (HKVD) rows. The fraction narrows layer by layer, starting from the check layer. | ratio 0.15; check layer 1 |
| T5 | Milvus segments | milvus-proto `SegmentState`; datacoord config | Growing (mutable, brute-force searched) -> Sealed (immutable, index built) -> Flushed (persisted) -> compacted -> Dropped. Seal triggers: size, age, idle, memory pressure, manual. | maxSize 1024 MB, sealProportion 0.12, maxIdleTime 600 s |
| T6 | Cascadia | arXiv:2506.04203 | A model chain with per-stage acceptance thresholds h_k, chosen by an offline Tchebycheff search. | thresholds H; GPU budget N |
| T7 | cluster-route-escalate | arXiv:2606.27457 | Embed, find the nearest k-means centroid, and route to argmin Error + λ·Cost. A quality-estimation classifier then accepts or escalates. | k by silhouette in [2,10]; λ 0.06/0.07 |
| T8 | conformal cascade | arXiv:2607.25018 | N samples. Score s = 1 - votes/N. q̂_k is the ceil((1-α)(n+1))-th order statistic. Accept iff \|C_k\| = 1. | N=16, T=0.7, n>=200 |
| T9 | UCCI | arXiv:2605.18796 | u = 1 - mean_t(p_top1 - p_top2), then isotonic g(u), then keep if g(u) <= θ*. | 30% calibration / 20% validation |
| T10 | AOSpec | arXiv:2608.00881 | Action and observation forks on copy-on-write environment snapshots. A fork is reused iff its action is equal AND its environment version is equal. | 8 action forks, 5 observation branches |
| T11 | Sherlock | arXiv:2511.00330 | Per-node verify. Downstream nodes are speculated inside the verifier's latency window. On failure, rollback is selective by similarity. | verifier budget k, speculation budget B, λ |
| T12 | Speculative Macro Commit | arXiv:2609.03236 | A drafter chain runs on a snapshot. Commit when the anchor action matches, the macro aligns, and at least L_min steps are skipped. | n_min, τ, δ, D, L_min |
| T13 | LMCache | arXiv:2510.09665; docs.lmcache.ai | Hash-keyed chunks across device, host, disk and remote tiers; asynchronous offload; LRU eviction. | chunk_size 256, max_local_cpu 5 GB, policy LRU |
| T14a | vAttention (memory) | arXiv:2405.04437 | A contiguous virtual KV range per request, with physical pages mapped on demand. | page group 2 MB (64-256 KB variants) |
| T14b | vAttention (verified sparse) | arXiv:2510.05688 | Read sink + local + approximate top-k + uniformly sampled rows, importance-weighted. The budget is sized by the CLT to give relative error <= ε with probability 1-δ. | fs, fl, ft, fb, ε, δ |

## audit of the FSM and its config surface (main edd4163c, read 2026-10-04; every line re-read at that HEAD)

| # | site | finding | closed by |
|---|---|---|---|
| A1 | `proxima-model-interop/src/serving_fsm.rs:38` | `#![allow(dead_code)]`. Nothing outside the file's own tests drives the FSM, and `generate/serving_backend.rs:121-241` uses it only under `#[cfg(test)]`. | architecture-as-data slice 10b |
| A2 | `generate/decode.rs:3530-3537` | The live loop runs a second, hand-written state machine: `DrafterSet` plus a `pending` queue drained at the top of a closure. That is the lambda fork decode-as-data forbids. | architecture-as-data 10b |
| A3 | `serving_fsm.rs:83-87` | `ServingFsmError::NotSupported` documents `accept`/`rollback` as unbuilt. Both are built (:186, :253). The variant is dead. | removed in this spec's slice 0 |
| A4 | `serving_fsm.rs:48-74` | Entries are hard-coded `u32` token ids (`positions: Vec<u32>`, `last: u32`, `draft: Vec<u32>`). `Cache` is generic but the log entry is not. An action log (T10-T12) cannot instantiate it. | slice 1 |
| A5 | `proxima-model-interop/src/lib.rs:69-70` | `mod serving_fsm` is private and std-gated, yet the file uses only `alloc` (:40). No consumer outside interop can reach it. | slice 1 |
| A6 | `serving_fsm.rs:158` | `enter_verify` is legal whenever the caller calls it. No config decides when Verify is entered, so a periodic re-encode (T1) has no trigger. | slice 7 |
| A7 | `serving_fsm.rs:49-52` | `Prefill` documents `cache` as always empty. A pre-assembled cache (loaded segments, cartridges, blended chunks: T2, T4, T13) is unrepresentable. | slice 8 |
| A8 | `generate/residency_caches.rs:3169` | `TokenEvent` carries `token_id`, `text_piece`, `phase`, `step` and `elapsed_ms`. It has no logprob and no top-2 margin, so no cascade judge (T7-T9) can read confidence. | slice 11 |
| A9 | `generate/decode.rs:1319` | `select_decoded_token` takes one `rng`. There is no N-samples-per-step mode for conformal sets (T8). | slice 11 |
| A10 | `serving.rs:719` | `ServingConfig` derives `Debug, Clone, Copy, PartialEq` only: no serde, `Settings`, `Validate` or builder. Only `SpeculativeSettings` (`speculative_settings.rs:253`) and `PromptCacheSettings` (`prompt_cache_settings.rs:17`) are conflaguration mirrors (`lib.rs:153-155`). | slice 2 |
| A11 | `proxima-tensor/src/spec/descriptor.rs:14,53` | `CacheStrategy` and `ModelDescriptor` have no config derive. | architecture-as-data 10a |
| A12 | `generate/residency_caches.rs:131` | `LayerCache::append` is `Vec::extend_from_slice`. Growth reallocates and copies all prior rows, and there is no reservation (T14a). | slice 4 |
| A13 | `generate/prompt_cache.rs:539`, `residency_caches.rs:707` | `PromptCache` and `PrefixState` live only on the heap. No KV byte ever reaches a file (searched `File::create\|std::fs\|write_all` in `generate/`; only debug dumps at `decode.rs:1057,1185-1266`). There is no tier below host (T5, T13). | slice 5 |
| A14 | `generate/block_bloom.rs:9` | The doc says "moving the rows (spec R5) is not built". `generate/chunk_shift.rs:106,173` builds it. The doc is stale. | fixed in slice 0 |
| A15 | `generate/chunk_shift.rs:30-35` | Moved rows keep the older context's values, and nothing recomputes the rows whose context changed. CacheBlend's selective recompute is absent (searched `recompute_tokens\|selective_recompute\|cacheblend`). | slice 9 |
| A16 | `proxima-tensor/src/op.rs:191` and `map.rs:186` | There is no top-k op. Row selection must be written as rank-count (`Reduce Add` over `Greater`), which is O(M^2) in block count M. At 128K context with b=16, M=8192, which violates the no-O(n^2) rule unless lowering specializes it. | slice 3 |
| A17 | `generate/prewarm.rs:321,397`, `prewarm_gate.rs:21` | Idle compute exists, but only as anticipatory prefill. It has no job kind for generating derived context (T3) and no compaction job (T5). | slice 10 |
| A18 | `proxima-primitives/src/pipe/ext.rs:48-78` | The pipe extensions are `and_then`, `filter`, `fanout` and `fanin`; there is no `or_else`. Escalation (T6-T9) is `and_then` over `Result<Answer, Request>` values carried inside `Ok`, so no combinator is added. | slice 12 |
| A19 | `serving.rs:630-655`, `decode.rs:3101-3106` | `AdmissionSchedule` is a scalar cap. `PhaseSchedule.prefill_before_decode = false` is accepted by config and rejected at run time with "not implemented yet". Validate does not reject it. | slice 2 |

## the frame: one log, eleven places, each a pipe

decode-as-data's frame: the committed sequence is an append-only log, the KV cache is a fold
over it, and a technique is a choice of pipe at a fixed place. Its places are propose, shape,
accept and commit. The fourteen techniques need seven more places, which makes eleven.
None of them is a type; each is a config-selected pipe at a fixed point of the existing
transitions.

| place | form, in -> out | runs at | techniques |
|---|---|---|---|
| propose | transform: (log, readouts) -> candidate entries | `Decode -> Verify` | decode-as-data set; T1 re-proposes the last f committed entries; T10-T12 action drafts |
| shape | transform: candidates -> (`mask_pos`, `seq_member`, `seq_primary`) | Verify plan inputs | decode-as-data set |
| accept | transform: (candidates, per-row readouts) -> accepted row indices, or refine | `Verify -> Accept/Rollback` | decode-as-data set; T1 accepts all; T10 action and version equality; T11 verifier; T12 anchor and macro |
| commit | sink: accepted rows -> fold | after accept | decode-as-data gather; T1 overwrites rows in place; T10-T12 environment commit or discard |
| enter | transform: (state, step) -> enter Verify with proposer i, or stay | top of `Decode` | n-gram (non-empty draft); T1 (`step % f == 0`) |
| read | transform: (query rows, sealed block summaries, live rows) -> per-query read set and weights, as graph inputs | inside every cached-attention op | dense (today); T1 block top-n; T14b sink+local+top-k+sampled |
| seal | transform: growing block -> sealed block (content key, key min/max, tier) | commit, once a block is full and below the rewind horizon | T5, T13, prompt cache trie and bloom, T1 summaries |
| place | sink: sealed block -> tier (device, host, disk) | after seal; under budget pressure | T13 tiers, T5 flush, T14a physical mapping |
| assemble | transform: request -> (preloaded cache, fresh entries, recompute selection) | before `Prefill` | prefix reuse (today); `chunk_shift` (today); T2 cartridges; T4 blend; T13 retrieve |
| schedule | transform: (queue, device state) -> which FSM instance runs next | between transitions | admission (today); prewarm (today); T3 sleep jobs; T5 compaction |
| settle | transform: `Result<Answer, Request>` -> `Result<Answer, Request>` (judge a tier's answer, or pass the request to the next tier) | after `Done` of one tier's FSM instance | T6-T9 |

Three consequences, each a requirement below:

1. The FSM changes in exactly four ways.
   - It is generic over its log entry (A4) and public at tier 1 (A5).
   - Verify entry is a config rule (A6).
   - `Prefill` may start from a preassembled cache at a start position (A7).
   - Every transition's output carries the readouts the judges need (A8, A9).
   Everything else is a pipe at a place.
   - Accept takes the verifier's own per-row choices as `&[Entry]` beside the draft. The
     accept rule is then a pure function over (draft, choices, config): `take_while`
     equality for token chains, `min_skip` for anchor and macro, and so on.
   - No separate readout type is added: the per-row choice is an `Entry`. This lets R19's
     anchor-and-macro policy reject a correct prefix shorter than `min_skip`.
   - Decided 2026-10-04, from the slice-1 card writer's finding.
   - `proxima-core` sits below `proxima-primitives`, which defines `Pipe`. The accept rule is
     therefore a plain function in core, which interop wraps as a pipe.
2. One in-graph expression, top fraction of rows by score, serves three techniques: read for
   T1 and T14b, and assemble's recompute for T4. It is written once in the algebra as a
   rank-count. It is lowered once as a selection kernel, a lowering specialization
   admissible by decode-as-data's two-kinds-of-split rule. A16 makes the specialization
   mandatory, not optional.
3. One block, `prompt_cache.block_tokens`, is the unit of seal, key, summary, tier and
   allocation page. Its default is 64 (`prompt_cache_settings.rs:67-68`). Milvus segments,
   LMCache chunks, ReSA blocks and vAttention page groups are this block at different sizes.
   No second chunk size exists. ReSA's b=16 and LMCache's 256 are values of the one key, and
   Validate requires `block_tokens` to be a multiple of 16 so summary blocks tile.

### where each technique lands

| technique | places | substrate added | lossless? |
|---|---|---|---|
| T1 ReSA | read (block top-n + local), enter (`every`), propose (last f), accept (all), commit (overwrite) | key min/max summary per sealed block (seal) | at keep ratio 1.0, yes |
| T2 Cartridges | assemble (load and concatenate with re-rotation) | sealed-block file format (place); trainer (slice 14) | n/a (trained) |
| T3 sleep-time | schedule (idle job), then seal and place its output | job kinds in the prewarm queue | the prewarm form is lossless; the derived-context form is not |
| T4 CacheBlend | assemble (re-rotate + top-fraction recompute rows) | deviation readout at the check layer | at ratio 1.0, yes |
| T5 segments | seal (growing -> sealed), place (flush), schedule (compaction) | disk tier; segment files | yes |
| T6 Cascadia | settle (per-tier threshold on a judge score) | multiple descriptors loaded; threshold search tool | n/a |
| T7 cluster-route-escalate | settle (classifier judge), with routing as assemble's first pipe | centroid table file; embedding readout | n/a |
| T8 conformal | settle (set size from N samples vs q̂_k) | N-sample decode; calibration table | coverage guarantee |
| T9 UCCI | settle (isotonic g(margin) vs θ*) | per-token top-2 margin readout; isotonic table | cost-optimal under the stated assumptions |
| T10 AOSpec | propose/accept/commit with Entry = action, Cache = versioned environment; config section `SpeculationSettings.accept = equality{fields, fork_budget}` (proxima-core, feature `config`) | none in proxima beyond the section (environment is the consumer's) | exact, for state the environment versions |
| T11 Sherlock | per-node FSM instances; accept = verifier; commit = selective rollback; config section `SpeculationSettings.accept = verifier{rollback: exact or similarity{threshold}}` | none in proxima beyond the section | exact under `rollback = exact`; similarity-gated otherwise |
| T12 Speculative Macro Commit | propose (macro library), accept (anchor + align + L_min); config section `SpeculationSettings.accept = anchor_macro{library path, min_skip, min_occurrences, tau, delta}` | none in proxima beyond the section | exact on commit |
| T13 LMCache | seal (content key), place (tiers + LRU), assemble (retrieve) | disk tier; hash-keyed lookup across tiers | yes |
| T14a vAttention memory | place (allocation page = block) | reserved virtual range per layer cache | yes |
| T14b vAttention sparse | read (sink + local + top-k + sampled, importance-weighted) | uniform-random input leaf; CLT budget in graph | ε, δ guarantee |

## requirements

Each requirement can fail on its own.

- **R1a.** `ServingState<Entry, Cache>` replaces `ServingState<Cache>`: `positions`, `last`
  and `draft` hold `Entry`.
- **R1b.** The type lives in `proxima-core` and builds under `--no-default-features
  --features alloc`.
- **R1c.** `proxima-model-interop` re-exports it, and its token path is the instantiation
  `Entry = u32`.
- **R2a.** `ServingSettings` loads every `ServingConfig` field (`serving.rs:725-1066`) to equal
  values from TOML, from env vars under the prefix `PROXIMA_SERVING`, and from its fluent
  builder. Implementation pattern: conflaguration `Settings`, `Validate` and `Builder`
  derives, as at `speculative_settings.rs:253`.
- **R2b.** `ServingSettings::default().as_serving_config()` equals `ServingConfig::default()`.
  The lowering is the `as_*_config` pattern at `speculative_settings.rs:340`.
- **R3a.** `ServingSettings::validate` refuses each row of the reachability matrix below with
  that row's refusal variant.
- **R3b.** Each refusal's error names the config field path the row lists (for example
  `decode.rectify.every`).
- **R4.** The `kv` section has these fields, and each variant round-trips through TOML, env
  and the builder with every field set:
  - `block_tokens`: the one block size, a multiple of 16;
  - `seal.horizon_rows`;
  - `seal.summaries`: a subset of {content_key, key_minmax};
  - `tiers`: a list of {device | host | disk{path}}, each with a `bytes` budget;
  - `policy`: lru | lfu | fifo;
  - `allocation`: grow | reserve{page_blocks}.
- **R10e.** The `decode` section's `rectify{every}` and `samples` fields round-trip through
  TOML, env and the builder.
- **R11c.** The `prefill.assemble` list grammar round-trips through TOML, env and the builder:
  a list of prefix | shift | load{block keys or cartridge path} | blend{recompute_ratio,
  check_layers}, with every field set.
- **R15e.** The `schedule.idle` list grammar round-trips through TOML, env and the builder: a
  list of prewarm | sleep{prompt, budget_tokens} | compact{segment_blocks}.
- **R17f.** The `cascade` grammar round-trips through TOML, env and the builder:
  - tiers of {model, judge};
  - router: first | cluster{centroids path, table path, embed readout};
  - judge: threshold{readout, h} | classifier{model, accept_class} | conformal{samples,
    qhat table} | isotonic{table, theta} | always.
- **R5a.** A block that is not full is never sealed.
- **R5d.** A full block is sealed iff all of its rows are older than `seal.horizon_rows`.
- **R5b.** A sealed block's bytes never change.
- **R5c.** Rewinding below a sealed block's end is refused with `InteropError::RewindIntoSealed`.
- **R6a.** Moving a sealed block device->host, host->disk or disk->host is byte-identical.
- **R6b.** A follow-up request whose prefix is restored from the disk tier produces ids equal
  to llama-server's ids for the same follow-up.
- **R7a.** Under `allocation = reserve`, appending a row never copies an earlier row.
- **R7b.** Under `allocation = reserve`, physical pages are backed one page (`page_blocks`
  blocks) at a time. After r rows, `ceil(r / (page_blocks * block_tokens))` pages are mapped.
- **R7c.** Under `allocation = reserve`, ids equal llama-server's ids.
- **R8a.** The `attention.read` section is dense | block{keep_ratio, min_blocks,
  local_blocks} | sampled{sink_fraction, local_fraction, topk_fraction, base_rate, epsilon,
  delta}. Each variant, with every field, round-trips through TOML, env and the builder.
- **R8e.** With `attention.read = dense` set explicitly, ids equal llama-server's.
- **R8b.** A block read attends to the top max(min_blocks, ceil(keep_ratio * M)) sealed
  blocks by summary score, chosen among the sealed blocks that are NOT local blocks (R8f).
  - Excluding the local blocks keeps the attended count data-independent, so W9 is exact
    (decided 2026-10-04, card-writer finding).
- **R8f.** A block read always attends to the `local_blocks` most recent sealed blocks.
- **R8g.** A block read always attends to every row of the unsealed tail.
- **R8h.** A block read attends to no row outside the union of R8b, R8f and R8g.
- **R8d.** With block read at keep_ratio 1.0, ids equal llama-server's.
- **R8c.** A sampled read's output has relative error <= epsilon with probability >= 1 -
  delta.
- **R9a.** At M >= `selection.top_fraction_min_rows`, the rank-count top-fraction expression
  lowers to exactly one selection kernel.
- **R9c.** At M < `selection.top_fraction_min_rows`, it lowers to zero selection kernels,
  leaving the plain expression.
- **R9b.** Both lowerings select the hand-derived index set of worked example W7.
- **R10a.** Under `decode.rectify.every = f`, the enter pipe enters Verify at exactly steps
  f, 2f, ....
- **R10d.** A rectify proposal is the last f committed entries, at their own positions.
- **R10b.** A rectify pass accepts all f rows and commits by overwriting exactly those f KV
  rows. The committed length does not change.
- **R10c.** After a rectify pass, the f rows equal the rows a dense prefill of the same
  entries writes, within tolerance τ. τ is derived in `worked-examples.md`.
- **R11a.** `Prefill` takes `start_position` and a cache that may already hold rows.
- **R11b.** The `prefill.assemble` pipes run in config order.
- **R12a.** Blend recomputes the rows whose |KV_loaded - KV_recomputed| at the check layer
  ranks in the top `recompute_ratio`. The selection narrows per later layer, and the
  per-layer counts must match the worked example.
- **R12b.** At blend ratio 1.0, ids equal llama-server's ids for the full prompt.
- **R12c.** At blend ratio 0.0, a prompt whose earlier chunk is reused at a shifted position
  gives ids equal to llama-server f1ea20621 run with `--cache-reuse` on the same two-request
  sequence. llama-server's `n_cache_reuse` keeps the moved rows' older-context values, the
  same property `chunk_shift.rs:30-35` documents.
- **R13a.** The sealed-block file, which is also the cartridge file, round-trips
  byte-identically. Its header holds the format version, descriptor digest, content key,
  base position, block_tokens and per-layer row bytes.
- **R13b.** Loading a block file whose descriptor digest differs from the model's is refused.
- **R13c.** Two cartridges concatenate, with K re-rotated by `chunk_shift::rotate_rows`.
- **R14a.** In `cartridge_train`, the only `proxima-autograd` leaves that receive gradients are
  the cartridge's K/V rows. Model weights receive none.
- **R14b.** The training loss is KL(teacher || student). The teacher is the model with the
  source chunk in context; the student is the model with the cartridge prefix.
- **R14c.** Training conversations are generated by the same model over the source chunk,
  seeded by the 5 seed-prompt kinds.
- **R14d.** An end-to-end run of `cartridge_train` writes a cartridge file that AC15's reader
  loads.
- **R15a.** A sleep job seals its generated output under the source context's content key.
- **R15b.** An admitted request preempts a running idle job at the next block boundary.
- **R15c.** A compact job merges disk block files into segment files with byte-identical
  reads.
- **R15d.** No idle job starts while a request is admitted.
- **R16a.** `TokenEvent.logprob` for the chosen token equals the log of llama-server's
  `n_probs = 2` top probability, within tolerance.
- **R16c.** `TokenEvent.top2_margin` equals llama-server's `n_probs = 2` top-1 minus top-2
  probability, within tolerance.
- **R16b.** `decode.samples = N` produces N completions seeded `seed + i`, reproducible across
  runs.
- **R17a.** The `cascade` section lowers to one pipe composed only with `and_then`
  (`proxima-primitives/src/pipe/ext.rs:48`) over `Result<Answer, Request>` carried in `Ok`.
- **R17d.** A settled answer passes every later tier untouched: later tiers run 0
  generations for it.
- **R17e.** The last tier always settles.
- **R17b.** Each of the five judge kinds settles or escalates as its worked trace says.
- **R17c.** The cluster router sends a query to the tier its nearest centroid's table names.
- **R18.** The fitters are examples in `proxima-model-interop`: PAV isotonic, the conformal
  q̂, k-means with silhouette, per-cluster λ argmin and Tchebycheff thresholds. Each
  reproduces `worked-examples.md` bit-exact.
- **R19.** At `Entry = action`, over an in-memory versioned environment, each of three accept
  policies gives a speculative run whose final history and environment equal the sequential
  run's. The policies are action and version equality, anchor and macro, and verifier with
  exact rollback.
- **R20.** No `struct`, `enum` or `trait` in `proxima-model-interop/src`,
  `proxima-tensor/src`, `proxima-core/src` or `omega/src` is named after a technique.
- **R22a.** `SpeculationSettings` lives in `proxima-core` behind its existing `config`
  feature (`proxima-core/Cargo.toml:65`, conflaguration). Its `accept` grammar is
  equality{fields, fork_budget} | verifier{rollback: exact or similarity{threshold}} |
  anchor_macro{library path, min_skip, min_occurrences, tau, delta}. Each variant, with every
  field set, round-trips through TOML, env and the builder.
- **R22b.** `SpeculationSettings::accept` lowers to the accept pipe that R19's `Entry =
  action` run uses. Each of the three lowered variants reproduces R19's worked run exactly.
- **R21.** Each of the 15 source-register rows (T1-T13, T14a and T14b; vAttention names two
  distinct techniques) has a TOML configuration that loads, validates and lowers, with the
  section named in "where each technique lands" not at its default.
  - The 12 serving rows (T1-T9, T13, T14a, T14b) use `ServingSettings`, lowering to
    `ServingConfig`.
  - The 3 agent rows (T10-T12) use `SpeculationSettings`, lowering to the accept pipe.

- **R23.** The FSM and every place's pipe are proven correct sans-IO. Owner, 2026-10-04: "at the
  end we'll be able to fully test sansio for correctness."
  - **R23a.** A scripted backend drives `ServingState<Entry, Cache>`. The backend is a pure
    function from (entries, positions, cache) to per-row readouts, read from a script table;
    it does no IO and holds no weights. The suite exercises every legal transition, and every
    illegal (state, event) pair is refused with its error.
  - **R23b.** For each place whose pipe runs on the host, each config variant reproduces its
    worked trace (`worked-examples.md`): the log, accepted rows and cache rows, exactly. Those
    places are propose, shape, accept, commit, enter, schedule and settle in `proxima-core`,
    and seal, place, assemble and the host half of read in `proxima-model-interop` over
    in-memory tiers.
    - Placement rule, binding on every slice:
      - Every place's decision is a pure function in `proxima-core`, tier 1. That covers the
        seal rule (which block seals), the enter rule, the schedule choice and every settle
        judge.
      - `proxima-model-interop` holds only the IO and kernel halves that apply the decision.
      - A slice that puts a decision in interop violates R23b.
  - **R23c.** Properties hold over 256 random cases each:
    1. the committed log only grows, except that a rectify overwrite keeps its length and
       positions;
    2. Rollback leaves the pre-Verify cache rows as a byte-equal prefix, and the cache length
       is pre + 1 + accepted. (On main, `accept` returns `row_caches[accepted]`, and
       `Verify.snapshot` is written but never read; the property is stated against that
       behaviour.)
    3. under every exact accept policy (token chain, action equality, verifier with exact
       rollback, anchor and macro), the speculative final log equals the sequential final
       log. Tree accept joins this property when decode-as-data's row-index accept lands;
       that spec owns it.
    4. Done is reached within `max_tokens` steps;
    5. a sealed block's bytes never change;
    6. exactly one cascade tier settles each request.
  - **R23d.** The `proxima-core` half builds as a non-test library under
    `--no-default-features --features alloc,sansio-script`, so it cannot touch std IO.
    - The test build alone does not prove this: proxima-core dev-depends on the root `proxima`
      crate, which enables std.

### reachability matrix (R3)

| config value | requires | refusal variant |
|---|---|---|
| `kv.block_tokens % 16 != 0` | (never valid) | `BlockNotMultipleOf16` |
| `attention.read != dense` | every KV-bearing layer is attention | `ReadNeedsAttentionLayers { layer }` |
| `attention.read = block` | `kv.seal.summaries` contains key_minmax | `ReadNeedsSummary` |
| `decode.rectify.every > 0` | `attention.read != dense` | `RectifyNeedsSparseRead` |
| `decode.rectify.every > 0` | every layer can rewind | `RectifyNeedsRewind` |
| `prefill.assemble` contains load or blend | `kv.seal.summaries` contains content_key | `AssembleNeedsContentKey` |
| `kv.tiers` contains disk | seal enabled | `TierNeedsSeal` |
| `kv.allocation = reserve` | the context length resolves to a bound | `ReserveNeedsBoundedContext` |
| `schedule.idle` contains sleep or compact | the prewarm worker is configured | `IdleJobNeedsWorker` |
| `schedule.idle` contains compact | `kv.tiers` contains disk | `CompactNeedsDisk` |
| `cascade.tiers.len() >= 2` | each judge's readout is enabled | `JudgeNeedsReadout { tier }` |
| `phase_schedule.prefill_before_decode = false` | (unbuilt) | `PhaseInterleaveUnsupported` (moved from run time, `decode.rs:3101-3106`) |
| `prompt_cache.block_tokens != kv.block_tokens` (either set explicitly) | one block size | `BlockTokensConflict { kv, prompt_cache }` |
| `cascade.tiers.len() > 4` | `and_then` fixes the chain type; `Cascade` has arms for 1-4 tiers | `CascadeTooManyTiers { tiers, max: 4 }` |

## acceptance criteria

Commands run from the checkout holding main, after
`source proxima-tensor/specs/fsm-techniques/env.sh`. Slice 0 writes that file. It sources
`long-context/env.sh` (which sets `CARGO_TARGET_DIR`, `$GEMMA4` and `$QWEN3_8B`, and defines
`niah`), then adds `$QWEN3_0_6B` as the absolute blob path of the Ollama `qwen3:0.6b` model.

The four parity checkpoints are gemma4 E2B, openchat, qwen2 and qwen3. They are the set that
architecture-as-data AC6 records as passing under the first-EOG comparison. This spec's oracle
tests compare through llama's first EOG id, using that spec's policy.

Kinds:
- ORACLE: the expected value comes from llama.cpp f1ea20621 artifacts vendored in slice 0.
- INVARIANT: an equality between two proxima paths, where the reference path is itself held to
  llama by AC24 or AC4. It shows the technique path is equivalent to the reference path; it is
  never the correctness oracle.
- CONSISTENCY: proxima against its own other path, never as the correctness oracle.
- WORKED: the expected value is hand-derived in `worked-examples.md` before code.
- MEASUREMENT: the AC asserts counts; the values themselves are recorded, not gated.

A capability AC's control is "absent", meaning 0 tests run, which counts as failing.

- **AC0** (R20, guard; consistency).
  - Command:
    `git grep -nP '\b(struct|enum|trait)\s+\w*(Resa|ReSA|Cartridge|SleepTime|CacheBlend|Milvus|SealedSegment|GrowingSegment|Cascadia|Conformal|Ucci|UCCI|AoSpec|Sherlock|SpeculativeMacro|LmCache|VAttention)' -- proxima-model-interop/src proxima-tensor/src proxima-core/src omega/src | wc -l`
  - Expected: 0.
  - Control: 0 at edd4163c. It is a guard, not a capability. Its discriminating twin is
    `git grep -nP '\b(struct|enum|trait)\s+\w*(Architecture)' -- proxima-model-interop/src proxima-tensor/src proxima-core/src omega/src | wc -l`,
    which prints 8 at edd4163c and so shows the pattern form matches. AC21 discharges the
    capability.
- **AC1** (R1b; consistency).
  - Command:
    `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/serving_state_/)'`
  - Expected: 4 passed. Slice 1 moves the four tests in `serving_fsm.rs` and renames them:
    - `walkthrough_drives_every_legal_transition` (:318) becomes
      `serving_state_walkthrough_drives_every_legal_transition`;
    - `accept_rejects_mismatched_or_empty_draft` (:461) becomes
      `serving_state_accept_rejects_mismatched_or_empty_draft`;
    - `draft_prompt_lookup_finds_the_earlier_occurrence` (:499) becomes
      `serving_state_draft_prompt_lookup_finds_the_earlier_occurrence`;
    - `speculation_matches_plain_greedy_for_thirty_two_tokens` (:527) becomes
      `serving_state_speculation_matches_plain_greedy_for_thirty_two_tokens`.
  - Control: absent.
- **AC27** (R1c; consistency).
  - Command:
    `git grep -nP 'pub use proxima_core::.*\bServingState\b' -- proxima-model-interop/src/lib.rs | wc -l`,
    then
    `git grep -nP '^\s*(pub\(crate\) )?mod serving_fsm\b' -- proxima-model-interop/src/lib.rs | wc -l`,
    then
    `cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_backend_/)'`.
  - Expected: 1; then 0; then 1 passed
    (`serving_backend_drives_serving_state_through_prefill_and_decode`, now compiled against
    the re-export).
  - Control at edd4163c: 0; then 1; then 1 passed. The first two counts discriminate.
- **AC2** (R2a, R2b, R3a, R3b, R4, R8a, R10e, R11c, R15e, R17f; consistency).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_/)'`
  - Expected: 23 passed.
  - Whole surface (R2a, R2b), 2 tests:
    - `serving_settings_round_trip` (R2a): every `ServingConfig` field at a non-default
      value, TOML vs builder vs `PROXIMA_SERVING_*` env;
    - `serving_settings_default_parity` (R2b): `ServingSettings::default().as_serving_config() ==
      ServingConfig::default()`.
  - Section round trips, 6 tests. Each has every variant and every field at a non-default
    value, and runs TOML -> settings -> builder -> env to equal values:
    - `serving_settings_kv_variants` (R4);
    - `serving_settings_attention_read_variants` (R8a);
    - `serving_settings_decode_variants` (R10e);
    - `serving_settings_prefill_assemble_variants` (R11c);
    - `serving_settings_schedule_idle_variants` (R15e);
    - `serving_settings_cascade_variants` (R17f).
  - Refusals, 15 tests:
    - `serving_settings_refuses_*`: one test per matrix row, each asserting its refusal
      variant (14, R3a);
    - `serving_settings_refusal_field_paths`: all 14 refusal errors name the field path in
      the matrix's first column (R3b).
  - Control: absent.
- **AC3** (R5a, R5b, R5c, R5d; worked).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std -E 'test(/seal_/)'`
  - Expected: 6 passed:
    - the worked seal trace (R5a, R5d);
    - full-but-inside-horizon is not sealed (R5d);
    - partial is never sealed (R5a);
    - sealed bytes are unchanged after later appends;
    - rewind into a sealed block is refused;
    - a proptest of 256 random append/rewind schedules never seals a rewindable row.
  - Control: absent.
- **AC4** (R6b; oracle).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/tier_disk_followup_oracle_/)'`
  - Expected: 4 passed, one per parity checkpoint. A follow-up served from a disk-restored
    prefix equals the vendored `followup_ids.json`.
  - Control: absent.
- **AC5** (R6a; consistency).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std -E 'test(/tier_round_trip_/)'`
  - Expected: 4 passed:
    - device->host, host->disk and disk->host round trips are byte-identical on a real
      gemma4 E2B block;
    - eviction order on the worked LRU trace.
  - Control: absent.
- **AC6** (R7a, R7b, R7c; oracle and consistency).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/kv_reserve_/)'`
  - Expected: 5 passed:
    - 4 parity checkpoints with ids equal to the vendored llama-server ids;
    - one counter test over a 4096-token decode, asserting row copies = 0 and mapped pages
      = ceil(rows / (page_blocks * block_tokens)).
  - Control: absent.
- **AC7** (R9a, R9b, R9c; worked).
  - Command: `cargo nextest run -p proxima-tensor -E 'test(/top_fraction_/)'`, then
    `cargo nextest run -p omega --features metal -E 'test(/top_fraction_/)'`.
  - Expected: 2 passed, then 4 passed:
    - proxima-tensor: the CPU evaluation of W7 at M = min_rows - 1 and at M = min_rows
      selects W7's hand-derived set (R9b);
    - omega: the same two selections on Metal (R9b);
    - `top_fraction_kernel_count_below`: the lowered plan at M = min_rows - 1 contains 0
      selection kernels (R9c);
    - `top_fraction_kernel_count_at`: the lowered plan at M = min_rows contains exactly 1
      (R9a).
  - Control: absent.
- **AC28** (R8b, R8f, R8g, R8h; worked).
  - Command:
    `cargo nextest run -p proxima-tensor -E 'test(/block_read_selection_worked_/)'`
  - Expected: 4 passed, on worked example W8 (4 sealed blocks plus a 5-row unsealed tail,
    keep_ratio 0.5, min_blocks 1, local_blocks 1):
    - `_top`: the top-scored blocks attended equal W8's hand-derived set (R8b);
    - `_local`: the local block is attended even with the lowest score (R8f);
    - `_tail`: all 5 tail rows are attended (R8g);
    - `_nothing_else`: the attended row count equals W8's union count exactly (R8h).
  - Control: absent.
- **AC8** (R8d; oracle).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/block_read_full_keep_oracle_/)'`
  - Expected: 4 passed. At keep_ratio 1.0, ids equal the vendored llama-server ids.
  - Control: absent.
- **AC26** (R8e; oracle).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/read_dense_oracle_/)'`
  - Expected: 4 passed. With `attention.read = dense` set explicitly through
    `ServingSettings`, each parity checkpoint's ids equal the vendored llama-server ids.
  - Control: absent.
  - Supplementary consistency check (not an oracle, discharges nothing):
    `arch_data_digest_` stays 7 passed after slice 6, which shows dense adds no graph nodes.
- **AC9** (R8b; measurement).
  - Command:
    `niah --model "$QWEN3_8B" --ctx 32768 --needles 8 --read dense,block,block-rectify --rectify 32`
  - Expected: exactly 3 `arm=` lines, each carrying `found=`, `kv_rows_read=`,
    `prompt_tokens=` and `decode_tokens=`.
  - The block arm's `kv_rows_read` equals W9's formula. W9 is derived in
    `worked-examples.md` as a function of b, keep_ratio, min_blocks, local_blocks, layers,
    prompt length P and decode length T.
  - The formula is evaluated at the P and T the same run prints, because both depend on the
    seed.
  - Control: absent (the flags do not parse).
- **AC10** (R8c; worked, statistical).
  - Command: `cargo nextest run -p proxima-tensor -E 'test(/sampled_read_error_bound_/)'`
  - Expected: 1 passed. Over 2000 query rows of the vendored
    `proxima-tensor/tests/fixtures/kv-captures/gemma4-e2b-layer2.f32`, at least 1900 have
    relative error <= 0.05 against exact attention computed in f64.
  - Control: absent.
- **AC11** (R10a, R10b, R10c, R10d; worked and invariant).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/rectify_/)'`
  - Expected: 7 passed:
    - `rectify_enter_steps_worked` (R10a): over 100 decode steps at f = 32, Verify is entered
      at exactly steps {32, 64, 96};
    - `rectify_proposal_worked` (R10d): at step 64, the proposal is ids[32..64] at positions
      32..64 of the worked trace;
    - `rectify_commit_overwrites_worked` (R10b): accepted = f, overwritten rows = f, and
      committed length unchanged;
    - `rectify_rows_equal_dense_` (R10c, invariant): 4 parity checkpoints. The reference is
      the dense prefill that AC24 holds to llama.
  - Control: absent.
- **AC12** (R11a, R11b; oracle and consistency).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/prefill_start_position_oracle_|assemble_order_/)'`
  - Expected: 5 passed:
    - 4 runs in which a host-restored prefix plus Prefill at `start_position` equals the
      vendored follow-up ids;
    - 1 run in which assemble pipes run in config order, asserted by event order.
  - Control: absent.
- **AC13** (R12b; oracle).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/blend_ratio_one_oracle_/)'`
  - Expected: 4 passed.
  - Control: absent.
- **AC14** (R12a, R12c; worked and oracle).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/blend_ratio_zero_cache_reuse_oracle_|blend_selection_worked_/)'`
  - Expected: 5 passed:
    - 4 runs at ratio 0.0, one per parity checkpoint, each equal to the vendored
      `cache_reuse_ids.json` (oracle);
    - 1 run in which the per-layer selected row counts and indices equal the 3-chunk worked
      example (worked).
  - Control: absent.
- **AC15** (R13a, R13b, R13c; consistency, worked and oracle).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/block_file_/)'`
  - Expected: 4 passed:
    - byte round trip (consistency);
    - digest refusal (consistency);
    - concatenation re-rotation equal to the `rotate_rows` worked example (worked);
    - `block_file_cartridge_followup_oracle_` on gemma4 E2B: a cartridge written from the
      follow-up prompt's prefix, then loaded, gives ids equal to the vendored
      `followup_ids.json` (oracle).
  - Control: absent.
- **AC16** (R14a, R14b, R14c; worked).
  - Command:
    `cargo nextest run -p proxima-model-interop --example cartridge_train -E 'test(/cartridge_train_/)'`
  - Expected: 3 passed:
    - `cartridge_train_leaves` (R14a): after one step, gradient tensors exist for exactly the
      2·L·p cartridge rows and for 0 weight tensors;
    - `cartridge_train_kl_worked` (R14b): the loss on worked example W14 (a 2-token, 3-vocab
      teacher/student pair) equals the hand-derived KL;
    - `cartridge_train_self_study_source` (R14c): each batch carries the generating model's
      descriptor digest equal to the trained model's, and a seed-prompt kind from the 5.
  - Control: absent.
- **AC29** (R14d; measurement).
  - Command:
    `cargo run -p proxima-model-interop --release --example cartridge_train --features std,metal -- --model "$GEMMA4" --corpus proxima-model-interop/examples/data/war_and_peace.txt --p 512 --steps 200`
  - Then, with `CARTRIDGE` set to the printed `cartridge=` path:
    `PROXIMA_TRAINED_CARTRIDGE="$CARTRIDGE" cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/trained_cartridge_loads_/)'`
  - Expected:
    - exactly 200 `step=` lines, 1 `kl_start=` line, 1 `kl_end=` line and 1 `cartridge=`
      line;
    - then 1 passed: `trained_cartridge_loads_` reads the file at `PROXIMA_TRAINED_CARTRIDGE`
      through the block-file reader. It asserts:
      - the descriptor digest equals gemma4 E2B's;
      - the row count equals 2 · layers · 512;
      - assemble `load{cartridge}` accepts it.
      The test fails, never skips, when the variable is unset or the file is absent.
  - Control: absent.
- **AC17** (R15a, R15b, R15c, R15d; consistency).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std -E 'test(/idle_job_/)'`
  - Expected: 4 passed, one per requirement.
  - Control: absent.
- **AC18** (R16a, R16b, R16c; oracle and consistency).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/token_readout_/)'`
  - Expected: 9 passed:
    - `token_readout_logprob_oracle_`: 4 parity checkpoints whose `logprob` equals the
      vendored `n_probs.json` within the worked tolerance (R16a);
    - `token_readout_margin_oracle_`: 4 parity checkpoints whose `top2_margin` equals the
      same within tolerance (R16c);
    - 1 run of `samples = 4` with fixed seeds, identical across 2 runs.
  - Control: absent.
- **AC19** (R17a, R17b, R17c, R17d, R17e; worked).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std -E 'test(/cascade_/)'`
  - Expected: 9 passed, on deterministic fake tiers driven by worked traces:
    - 5 judge kinds (R17b);
    - the cluster router (R17c);
    - the lowered pipe is a chain of `and_then` nodes only, asserted by walking the composed
      type's tiers, with count = 2·tiers + 1 (R17a);
    - a settled answer causes 0 generations in later tiers (R17d);
    - the last tier settles every request that reaches it (R17e).
  - Control: absent.
- **AC20** (R18; worked).
  - Command:
    `cargo nextest run -p proxima-model-interop --example calibration_fit -E 'test(/fit_worked_/)'`
  - Expected: 5 passed.
  - Control: absent (the example is not declared).
- **AC21** (R21; consistency; the problem's bound).
  - Command: `cargo test -p proxima-model-interop --doc --features std,conflaguration serving_settings`,
    then `cargo test -p proxima-core --doc --features config speculation_settings`.
  - Expected: 12 passed, then 3 passed (15 in total), one doctest per source-register row:
    - 12 doctests on `ServingSettings`, for T1-T9, T13, T14a and T14b;
    - 3 doctests on `SpeculationSettings`, for T10-T12.
    Each loads its TOML, validates it, lowers it, and asserts that its section is not the
    default.
  - Control: 0 run, then 0 run, at edd4163c.
- **AC30** (R22a, R22b; worked).
  - Command:
    `cargo nextest run -p proxima-core --features config -E 'test(/speculation_settings_/)'`
  - Expected: 6 passed:
    - 3 round trips, one per `accept` variant with every field at a non-default value (R22a);
    - 3 lowered-policy runs: each variant, lowered, drives AC23's worked environment to the
      same final history and environment as the matching `action_speculation_` test (R22b).
  - Control: absent.
- **AC22** (R17a, R16a; measurement).
  - Command:
    `cargo run -p proxima-model-interop --release --example cascade_bench --features std,metal -- --tiers "$QWEN3_0_6B","$QWEN3_8B" --judges isotonic,conformal --calibration proxima-model-interop/tests/fixtures/cascade/calibration.jsonl --held-out proxima-model-interop/tests/fixtures/cascade/held_out.jsonl`
  - Expected:
    - for each of 2 judge arms and 2 tiers, `settled + escalated = 200`;
    - the conformal arm's held-out miscovered count is <= 49 of 200. That bound is K·α =
      0.2 at α = 0.1 and K = 2, i.e. 40, plus 1.645·sqrt(200·0.2·0.8) = 9.3.
  - Accuracy and cost per arm are recorded.
  - Control: absent.
- **AC23** (R19, R1a; worked). The `Entry = action` instantiation compiling and passing is
  R1a's evidence.
  - Command:
    `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/action_speculation_/)'`
  - Expected: 3 passed.
  - Control: absent.
- **AC24** (guard on R8e and R7c: the default dense, grow-allocated path keeps llama parity
  through every slice; oracle).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std,metal -E 'binary(arch_data_baseline) & test(/^llama_parity_/)'`
  - Expected: 7 run, 4 passed (gemma4 E2B, openchat, qwen2, qwen3).
  - The 7 `llama_parity_*` fns are at `arch_data_baseline.rs:725-755`.
  - The comparator (`compared_len`, :706, from f214d173) truncates proxima's ids to the length
    of llama's recorded ids, which end at llama's first EOG id. A proxima output shorter
    than llama's is a divergence.
  - Measured at edd4163c on 2026-10-04, in a clean detached worktree: 7 run, 4 passed, 3
    failed in 54.7 s. Log: `~/.cache/proxima-fsm-audit/logs/p0-retry.log`.
    - Passed: qwen2, gemma4_e2b, openchat, qwen3.
    - Failed: qwen35 and qwen35moe, each missing `llama_ids.json`
      (`arch_data_baseline.rs:622`; architecture-as-data finding O1).
    - Failed: gemma4_26b, 2 of 3 prompts diverging at index 0 and 1
      (`arch_data_baseline.rs:715`; finding D2).
  - Control: 4 passed of 7 at edd4163c, measured as above. It agrees with architecture-as-data
    TASKS.md slice 0's recorded "AC6 4/7" (committed at 0ac67c63).
- **AC25** (guard on R6b, R12b, R13c and R16a: positive control for every oracle fixture this
  spec vendors; oracle).
  - Command:
    `cargo nextest run -p proxima-model-interop --features std -E 'test(/fsm_oracle_control_/)'`
  - Expected: 5 passed:
    - `followup_ids.json` compared with itself passes;
    - `n_probs.json` compared with itself passes;
    - `cache_reuse_ids.json` compared with itself passes;
    - one flipped id is rejected;
    - one probability perturbed beyond tolerance is rejected.
  - Control: absent.

- **AC31** (R23a, R23b, R23c, R23d; worked and property; the end-state gate).
  - Commands:
    - `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_/)'`
      (nextest matches the module-qualified name, so an anchored `^sansio_` would list 0)
    - then `cargo nextest run -p proxima-model-interop --features std -E 'test(/sansio_tests::sansio_/)'`
      (std only, no metal)
    - then `cargo check -p proxima-core --no-default-features --features alloc,sansio-script`
      (the non-test proof of R23d)
    - then the no-IO guard:
      - first, `git ls-files proxima-core/src/serving_state.rs 'proxima-core/src/serving_state/*' | wc -l`;
      - then `git grep -nE 'std::(fs|net|io|env|thread|time)|File::|TcpStream|tokio|libc::|println!' -- proxima-core/src/serving_state.rs 'proxima-core/src/serving_state/*' | wc -l`.
      - The file count must be nonzero, so that an empty pathspec cannot pass as 0.
  - Expected: 15 passed, then 4 passed, then exit 0, then a file count of at least 3 and 0 IO
    matches.
    - proxima-core, 15 tests:
      - `sansio_transitions_legal` and `sansio_transitions_illegal`;
      - 7 place tests: `sansio_place_propose`, `_shape`, `_accept`, `_commit`, `_enter`,
        `_schedule`, `_settle`;
      - 6 property tests: `sansio_prop_log_append_only`, `_rollback_restores`,
        `_exact_accept_equals_sequential`, `_terminates`, `_sealed_immutable`,
        `_one_tier_settles`.
    - proxima-model-interop, 4 tests: `sansio_place_seal`, `sansio_place_place`,
      `sansio_place_assemble`, `sansio_place_read_host`, over in-memory tiers and the
      scripted backend.
  - Control: absent (0 run).

## decisions from card writing (2026-10-04, against main 8e98e5be)

These override any earlier wording in this file. Cards in `tasks/` implement them.

- **Accept.**
  - The accept rule is a pure function over (draft, the verifier's per-row choices as
    `&[Entry]`, config). No readout type is added.
  - `SpeculationSettings` has 9 scalar fields. `anchor_macro_tau` is a similarity floor.
    `anchor_macro_delta` is a macro-depth cap, where 0 means no cap.
- **Top fraction (slice 3).**
  - The expression is `top_fraction_mask(program, scores, keep_count, keep_rows)`, plus a host
    function `top_fraction_keep_count(fraction_milli, min_keep, rows)`. The algebra has no
    ceil, and M is symbolic.
  - Ties go to the lower index.
  - `min_keep` (the minimum rows kept) is distinct from `selection.top_fraction_min_rows` (the
    lowering threshold). The threshold defaults to 256, unmeasured.
  - The recognizer is a cfg-gated `BoundOpKind` matcher in `proxima-tensor/src/bind/`, following
    `match_moe_topk`. omega holds only the kernel.
- **Seal (slice 4).**
  - Ring layers never seal: rows outside the ring are overwritten.
  - The content key is a chained key in `block_bloom.rs`. `arena.rs` is untouched.
  - Trie insert stays at request end. Insert-at-seal is dropped: the decode loop holds no
    `PromptCache`, so an insert-at-seal path would have no production caller and would be dead
    code.
    - It returns only if a later change gives the decode loop the cache.
- **Reserve (slice 4).** mmap plus `newBufferWithBytesNoCopy` serves the placed path only. The
  host `LayerCache` uses `Vec::reserve_exact`. R7a is asserted on qwen3, which has no sliding
  layers.
- **Tiers (slice 5).**
  - Mapping uses `memmap2`, which is already in `std`; not `proxima-storage`.
  - `model_digest = xxh3_128(format!("{program:?}"))`.
  - Ring-layer restore from tiers is open: if gemma4 diverges in AC4, that case is reported as
    a finding.
- **Decisions in proxima-core (R23b).**
  - `proxima-core/src/kv_decision.rs` holds the pure functions `seal_target`, `sealed_blocks`,
    `eviction_victim`, `demotion_target`, `lookup_tier` and `initial_tier`.
  - Interop calls them.
- **ServingSettings (slice 2).**
  - `kv.block_tokens` is the one block size.
    - `PromptCacheSettings.block_tokens` becomes `Option<u32>`, default `None`. Standalone
      lowering is `unwrap_or(64)`, so it behaves as before.
    - Under ServingSettings, `Some(x)` with `x != kv.block_tokens` is refused with
      `BlockTokensConflict { kv, prompt_cache }`. It is never silently overwritten, and an
      explicit 64 against kv 256 is refused too.
    - This is a 13th matrix row, so AC2's count is 22.
  - The nested `speculative` and `prompt_cache` sections keep their own env prefixes
    (`PROXIMA_SPECULATIVE_*`, `PROXIMA_PROMPT_CACHE_*`). Every other field uses
    `PROXIMA_SERVING_*`.
  - Matrix rows 2 and 5 need per-layer facts. They are checked by
    `refusals_for(&ModelDescriptor)`; the other rows are checked by `validate()`.
  - The `decode.rs` run-time refusal moves into `apply_serving_config` as well, so a
    hand-built `ServingConfig` is still refused.
- **W9.** The formula is evaluated at the printed P and T. Its instance at P=4096, T=2 gives
  83052 and 267372.

## dependencies

- **architecture-as-data:**
  - slice 4c (formerly 10a; moved 2026-10-04, main 9f0647da) precedes slice 2 here;
  - slice 10b precedes slice 1 onward here (10b edits `serving_fsm.rs`, which slice 1
    moves);
  - slice 10c precedes slice 16 here.
- **decode-as-data** (untracked, `proxima-tensor/specs/decode-as-data/`):
  - its slice 1 (visibility inputs) precedes slice 6 here;
  - its slice 2 (host shape pipes; RoPE from a positions slice, which TODO.md item C7 names
    `build_position_inputs_at`) precedes slices 6 and 8 here;
  - its slice 7 (per-config static plan cache; TODO.md item D6) precedes slices 6, 7 and 9
    here;
  - its slice 3 (readout list; TODO.md item C8) precedes slices 6 and 11 here (slice 6
    captures its K/V fixture through a layer tap);
  - its slice 4 (row-index accept, gather commit) precedes slice 7 here.
- **long-context:** this spec reuses its `env.sh`, the `niah` harness and
  `proxima-model-interop/examples/data/war_and_peace.txt`.

## out of scope (each names its owner)

- The consumer instantiations of R19, the real tool sandbox, and remote-provider cascade
  tiers. These belong to the consuming application, outside this repository, which also owns
  the policy for which tools may be speculated. A proxima library naming a consumer is
  forbidden.
- Cascadia's GPU-allocation MILP. On one device its inner level is degenerate, so R18's
  threshold search is the whole optimizer at N = 1.
- Pipelining propose(t+1) with verify(t). decode-as-data names it out of scope.
