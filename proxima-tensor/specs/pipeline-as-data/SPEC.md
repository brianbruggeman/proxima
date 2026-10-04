---
status: draft (not audited; supersedes the scope of fsm-techniques and joins architecture-as-data)
---

# pipeline as data

## problem

Owner, 2026-10-04:
- "every model should be expressible entirely in conflaguration";
- "it should be for the entire pipeline, right?";
- "I asked the original agent to build the hooks so they could be implemented. the thought
  is... if the hooks are mostly programmable, the surface becomes very hackable and future
  techniques can be quickly vetted."

The goal is hooks, not techniques. Every stage from request to response is:
- a pipe selected by configuration;
- whose decision is a pure function in `proxima-core`;
- whose default reproduces today's behaviour exactly.

A model, a tokenizer, a template, a sampler or a serving technique is then a configuration,
plus at most one small pipe written against an existing hook. It never edits the core.

## what is proven, and what is not (main 4b4be6cf)

| claim | status | evidence |
|---|---|---|
| gemma4's forward graph lowers from a data descriptor | proven | byte-identical to the hand builder, 13314 ops (`arch_data_baseline.rs` digests) |
| 7 checkpoints' op graphs are stable under the data refactors so far | proven | `arch_data_digest_` 7/7 |
| token parity with llama.cpp | proven for 4 (gemma4 E2B, openchat, qwen2, qwen3) | `llama_parity_` |
| recurrent (GDN) and MoE families fit the descriptor | unmeasured | each has its own builder today |
| hooks can express the listed techniques | unmeasured | no technique has been expressed against a hook |
| configuration-driven lowering costs no speed | unmeasured | baseline binaries built, timing not run |
| the serving FSM drives the live loop | false today | `serving_fsm.rs` is `#![allow(dead_code)]` |
| a model descriptor can be written in configuration | false today | `ModelDescriptor` derives only `Debug, Clone, PartialEq` (`spec/descriptor.rs:53`) |
| stages compose through one mechanism | false today | 1 of 17 stages composes from config; 9 unrelated mechanisms (section below) |

## how much composes today (main, read 2026-10-04)

Composition here means: two or more pieces chosen and combined by data, rather than one piece
picked from N, or a fixed sequence with toggles.

| stage | today | mechanism | anchor |
|---|---|---|---|
| H11 step | composes, reachable from TOML/env | enum bitset of drafters, walked in a fixed priority order; first non-empty draft wins | `generate/drafter.rs:123`, `serving.rs:273`, `speculative_settings.rs:253` |
| H4 describe | composes, Rust only | `Vec<LayerSchedule>`, each `LayerKind` x attention config x FFN config | `spec/descriptor.rs:54`; no serde on `ModelDescriptor` (:53); qwen35, qwen35moe and lfm2 bypass it |
| H5 bind | lookup, not composition | ordered first-match `WeightPrecisionRule` list; fusion passes are a fixed sequence toggled by feature, env or bool | `serving.rs:107`, `bind/gdn_moe_fusion_apply.rs:62` |
| the other 14 | none | one-of-N enums, numeric knobs, one callback closure, or fixed code | per-stage rows in the audit report |

Fully fixed in code: template (plain `role: content` concat), stop (one eos id plus budget),
detokenize, the sampler chain order, the prompt-cache lookup order, and readback. Seal and tier
has no code at all.

Mechanisms found: nine, sharing no type or trait.
- Five combine things: the layer list, the drafter bitset, the precision rule list,
  conflaguration source layering, and the `Pipe` algebra.
- Four only select or toggle: the architecture registry, fusion kill switches, the `on_token`
  closure, and the serving FSM, which is dead code.

`Pipe` is the intended single model:
- `LoadedModel` implements it (`generate/residency_caches.rs:3146`), but only tests call it, and
  nothing composes it.
- The one server (`examples/openai_serve_gguf.rs:141`) bypasses it.
- `and_then` appears on the tensor path only in tests and benches.
- The only wired `Pipe` composition is in telemetry (`proxima-telemetry/src/pipes.rs:84`).

Consequence for this spec: the hooks are not N new mechanisms. Each hook is a `Pipe` slot whose
contents are a configured list. The drafter set is the one working instance of that shape, and
it is the template for the other sixteen.

## the hook catalog (one row per stage, request to response)

Code anchors come from the request-to-response map of main 9dd9deef. Re-read each one before
designing against it.

| # | stage | pipe shape (in -> out) | decision in proxima-core | configured by | default = today | anchor |
|---|---|---|---|---|---|---|
| H1 | intake | request -> messages | none (transport is outside proxima's model stack) | server config | OpenAI-shaped JSON | `examples/openai_serve_gguf.rs` |
| H2 | template | messages -> prompt text | template choice | model config `template` (GGUF `tokenizer.chat_template` as a layer) | plain concatenation | absent today |
| H3 | tokenize | text -> ids | splitter choice, special-token handling | model config `tokenizer` (GGUF `tokenizer.ggml.*` as a layer) | shape probe plus regex | `proxima-tokenizer/src/pipe.rs:103` |
| H4 | describe | GGUF header + profile -> model descriptor | descriptor assembly | model config (profile -> GGUF -> env -> TOML) | today's per-family builders | `spec/gguf_descriptor.rs`, `profiles/` |
| H5 | bind | descriptor + tensor directory -> bound weights | name resolution, codec, fused splits | model config | per-family name tables | `gemma4/bind.rs`, `bind.rs` |
| H6 | lower | descriptor -> op graph | none: one generic builder | model config | `build_forward` | `spec/descriptor.rs` |
| H7 | specialize | op graph -> kernel plan | kernel choice by op shape | lowering thresholds (`omega-runtime.toml`) | today's classifiers | `msl/emit_and_classify.rs` |
| H8 | place | plan + device -> placed buffers | residency and offload policy | placement config | all on GPU | `residency_caches.rs`, `expert_slab.rs` |
| H9 | assemble | request -> starting cache | which cached rows to reuse | serving config `prefill.assemble` | prefix reuse | `prompt_cache.rs` |
| H10 | schedule | queue -> next instance | admission, idle jobs | serving config `schedule` | one request | `prewarm.rs` |
| H11 | step | FSM state -> FSM state | enter, propose, shape, accept, commit | serving config `decode`, `speculative` | greedy plus n-gram | `serving_fsm.rs` (dead today), `decode.rs` |
| H12 | read | query rows + cache -> read set | which cache rows a step reads | serving config `attention.read` | dense | cached-attention ops |
| H13 | seal and tier | cache rows -> sealed blocks in tiers | seal, evict, demote | serving config `kv` | host memory only | `kv_ring.rs`, `prompt_cache.rs` |
| H14 | sample | logits row -> entry | sampler chain | model config `sampling` + request | greedy | `select_decoded_token` |
| H15 | stop | entry -> continue or stop | stop set | request policy | eos id only (owner policy, 2026-09-17) | `decode_until_stop_or_budget` |
| H16 | detokenize | ids -> text pieces | streaming rules | tokenizer config | UTF-8 hold | `decode_streamed_piece` |
| H17 | settle | answer -> answer, or escalate | judge | serving config `cascade` | always settle | absent today |

### stages added by ablation (from `frontier.md`, 2026-10-04)

Every row below is backed by recent primary sources (status ACTIVE in `frontier.md`), so it
must be a hook (owner rule).

| # | stage | pipe shape | sits between | what it hosts |
|---|---|---|---|---|
| N1 | shape logits | logits row -> logits row | H11 and H14 | grammar and JSON masks, classifier-free and contrastive guidance, watermarking, the truncation chain |
| N2 | tap and edit | hidden state -> hidden state, plus readouts | between layers, inside H6's program | steering, probes, lens readouts, fast-weight writes, activation sparsity, observability |
| N3 | depth program | (row, layer) -> run, skip, exit or loop | inside H6 and H11 | layer skip, early exit, recurrent depth such as looped models |
| N4 | adapter bind | request -> adapter set | H5 per request | multi-adapter hot swap |
| N5 | compress | prompt -> shorter prompt | H2 and H3 | prompt compression |
| N6 | route | router logits -> expert set | inside MoE layers | expert-routing override and offload-aware routing |
| N7 | seal codec | KV rows -> encoded rows | inside H13 | KV quantization, merging and eviction encodings |
| N8 | import and export | cache or hidden state <-> outside | H9 and H13 | cross-instance and cross-agent cache or latent hand-off |
| N9 | position | ids -> positions and rotation | H6 and H9 | position-id schemes, RoPE scaling, position-independent caching |
| N10 | determinism mode | constraint on H7 | H7 | batch-invariant and reproducible kernels |

Cross-cutting rule, plausible and not proven: steering, position-independent caching,
cartridges, fast-weight writes and latent hand-off all change what a cached row means.
- So an H9 or H13 cache key binds the producing configuration: the intervention, adapter,
  position scheme and codec, not token ids alone.
- The prompt-cache key already binds the build configuration
  (`generate/prompt_cache_key.rs`), so this extends an existing mechanism.

### thin in the literature: a research track, not engineering

Owner, 2026-10-04: "it would need to be research, not engineering."

These have no or very few recent sources; `frontier.md` records the searches. Nobody has an
answer to reproduce, so none of them gets a hook, a card or a slice here.

Each is a pre-registered experiment run by `/discovery-loop`, and states four things before any
code:
- a hypothesis;
- the measurement and the held-out data it runs on;
- a fairly tuned baseline;
- a kill criterion.

A hook follows only from a win that survives ablation and variance. That win becomes a worked
example test through `/algorithm-development`, and only then enters the catalog above. A
negative result is recorded here and the item closes.

Few sources in the literature does not by itself make an item research. An item is research
only when no known method answers its core question. By that test, three of the six are
research and three are engineering. The engineering three have a known correct answer, so
they are specified and tested like any other hook:
- token healing (back the prompt up one token and constrain the first draw);
- streaming detokenize with stop-string hold-back (streamed bytes must equal batch decode);
- the generic engine (op-graph digests and llama token parity are its oracle).

Model merging is dropped. Owner, 2026-10-04: it is a tuning and training knob, not an
inference benefit.

By the same rule, training-side work is out of scope. That covers cartridge training
(`cartridge_train` in the fsm-techniques draft) and the calibration fitters. Inference loads a
trained cartridge (sketch 11) and reads a fitted calibration table. Producing either one is a
separate tool outside the serving pipeline. When the fsm-techniques cards are re-cut, these two
lose their cards.

An in-loop readout that picks an exit point is active research, not open. Training-free
layer-skipping self-speculation already exists (Draft and Verify, 2023; SWIFT, 2024). Both
were opened by the research sweep (`research.md`), along with ConfLayers and SimLens. It
belongs with speculation at H11 and the depth program at N3.

The table below keeps the rows written as experiments, each with its baseline and kill
criterion. The merging row is removed.
The data for each is still to be chosen at pre-registration.

| question | baseline | kill criterion |
|---|---|---|
| Does token healing reduce boundary-token errors on prompts ending mid-word? | today's tokenize | no reduction on held-out mid-word prompts, or a parity loss on whole-word prompts |
| Can streaming detokenize with stop-string hold-back be specified so streamed text always equals batch decode? | batch decode of the same ids | any streamed byte sequence differs from batch decode |
| Can one generic engine run every supported family from data alone, at today's speed? | today's per-family builders | op-graph digests or llama token parity diverge, or decode is slower than the interleaved baseline |
| Can a per-op error bound, derived from each Metal kernel pair's reduction tree, certify that the multi-row path picks the same argmax as decode-1 whenever the top-1/top-2 margin exceeds the bound, with the other steps re-run at decode-1? | verify-all at decode-1 shape; an empirical margin threshold | the bound is violated on any recorded step, more than 25% of steps fall below twice the bound, or argmax flips above twice the bound |

Results of the research sweep (`research.md`, 2026-10-04; sources opened unless labelled
snippet):
- Steering inside one batch has two parts. The exact case is engineering: dependency-closure
  memoization, where the cache key is a hash chain over layer groups that binds the producing
  configuration (as in Bazel or salsa, and like vLLM's `(block_hash, group_id)`). The
  approximate case is active research.
- The last row above is the one open question the sweep found.
  - Speculation, verify, tree, lookahead, chunked prefill and prefix-cache continuation all
    assume byte-identical output across evaluation shapes. omega picks kernels by row count.
  - arXiv 2607.17283 measured quantized Metal logits shifting by up to about 0.1 between a
    batch of 5 and a batch of 1, which is enough to flip argmax at near-ties.
  - No certified-margin method for block-quantized kernels was found.
  - The 25% kill figure is to be re-derived from the verify-width cost curve before the run.
- Certified retrieval attention (`research-retrieval-attention.md`) is active research, not open.
  The certifying math exists: arXiv 2512.07647 gives a total-variation bound from per-cell logit
  upper bounds, and the output error is bounded by that total variation times a value bound. It
  was tested only on bert-base at n=128/256. What remains open is the measured fire rate on real
  long-context decoding. One record reports certifiable steps peaking near 3.6%, with the bound
  overestimating tail mass 80-200x (record summary only). It binds the read stage (H12):
  - a per-head or per-KV-group key bound. A query pooled over heads, as sketch 08 does,
    bounds no single head, so pooled selection cannot be certified;
  - a per-block value bound;
  - the bound taken after the kernel's own logit transform, plus a rounding slack, which ties
    it to the open kernel-exactness question above;
  - a read-everything fallback variant when the bound fails;
  - tiered summaries. Summaries are about 6.3% of KV bytes at block 16 (arithmetic, not
    measured).
- The sweep's implied catalog changes are listed in `research.md` under "catalog changes"
  (cache key as a layer-group hash chain, an exactness contract on the determinism mode,
  validation rows, tier grain per layer capability, and others). They are folded into the
  catalog after the remaining paper sketches land.

The last row is this spec's own thesis, and it is the one the engineering above depends on:
- The 14 paper sketches are its first measurement: they count the core edits each sketch
  needs.
- The three sketches written so far each need 9 to 12 hook changes. So the thesis is
  unmeasured, not refuted.

## the paper test (before any code)

Each sketch writes, against the catalog above:
- the configuration;
- and, only if needed, one pipe implementation, at most about 40 lines, at one hook.

A sketch that needs a core edit is a hook defect, recorded as a finding.

Sketches, chosen to stress different hooks:
1. gemma4 E2B, entirely from configuration (H2-H8, H14-H16).
2. qwen3.5 (GDN layers) from configuration. This tests whether H4 and H6 hold for recurrent
   layers.
3. qwen3.6 MoE with expert offload (H8).
4. A model whose chat template GGUF carries (H2).
5. A tokenizer that needs a different pre-split (H3).
6. A min-p plus repetition-penalty sampler (H14).
7. n-gram speculation (H11), which exists today and is the control.
8. Rectified sparse attention (H11 enter and commit, H12, H13 summaries).
9. CacheBlend (H9, H12).
10. Tiered chunk cache (H13).
11. Cartridges, load path only (H9).
12. Sleep-time prewarm (H10).
13. A conformal cascade (H17).
14. Action speculation over a non-token entry (H11 generic entry).

Pass condition, per sketch: written, reviewed, and needing zero core edits. Any failure names
the hook and the missing input, and becomes a requirement here.

### results so far (paper only; nothing built or run)

| sketch | verdict | hook changes needed | the change that matters most |
|---|---|---|---|
| qwen3.5 GDN (`sketches/02-qwen35-gdn.md`) | fails as written | 12 | `build_forward` cannot run recurrent state, so the layer loop must become schedule-driven. `tokenizer.ggml.pre` is never read. |
| MoE offload (`sketches/03-moe-offload.md`) | fails as written | 9 | the residency decision is already pure and can move to core. Four booleans encode one mode. Retention lives in six env vars. Only one family can offload. |
| rectified sparse attention (`sketches/08-rectified-sparse-attention.md`) | fails as written | 12 | no seal concept and no block structure in `LayerCache`. FSM accept discards `last`. Single-range models have no cache-side mask. There is no top-k op. |

| tiered chunk cache (`sketches/10-tiered-chunk-cache.md`) | fails as written | 6 | eviction drops its victim with no slot to hand it to a next tier. Nothing serializes a cache entry. Only whole entries can tier, not blocks. |
| cartridges, load path (`sketches/11-cartridges-load.md`) | fails as written | 5 | the assemble arms are inline. There is no file format or loader. Prompt ids need placeholders for the loaded rows. |
| sleep-time prewarm (`sketches/12-sleep-time-prewarm.md`) | fails as written | 6 | the job list is fixed. The worker discards drafts. The queue is single-slot, newest wins. |
| conformal cascade (`sketches/13-conformal-cascade.md`) | fits as is | 0 | one pure settle function and one pipe of about 38 lines. Only validation rules are missing. |
| action speculation (`sketches/14-action-speculation.md`) | fails as written | 1 | `ServingState` is crate-private and dead, so a consumer cannot name it. Interning actions to u32 ids needs no generic entry. |

| gemma4 E2B from configuration (`sketches/01-gemma4-from-config.md`) | fails as written | 8, plus those of sketches 4-6 | no serde on the descriptor. GGUF keys are read in three places. Kernel thresholds are build constants. `PROXIMA_HEAD_REPEATS` is read inside the graph builder. |
| chat template (`sketches/04-chat-template.md`) | fails as written | 6 | no template stage. The template-to-tokenizer seam is a bare string, so control markers in user text are converted. gemma4's template is 18,155 bytes of Jinja, and no Jinja engine is in the lockfile. |
| pre-split (`sketches/05-pre-split.md`) | fails as written | 5 | `tokenizer.ggml.pre` is never read. Digit runs are hard-coded to 3, where llama splits qwen digits one at a time. |
| min-p plus repetition penalty (`sketches/06-min-p-repetition.md`) | fails as written | 5 | the chain order is fixed. There is no library sampling loader and no per-request sampling. |
| n-gram speculation, the control (`sketches/07-ngram-speculation.md`) | config loads today, but the slot fails | 5 | the live loop is open-coded. The drafter table is written three times. Five config keys have no consumer. Speculation is silently off for every family without a verify program. |
| CacheBlend (`sketches/09-cacheblend.md`) | fails as written | 6 | lookup lifts from one entry only. Positions must be contiguous. There is no per-layer residual hand-off. Entries carry no provenance. |

All 14 sketches are written. One fits as is (the conformal cascade). The other 13 need 1 to 12
named hook changes each, and no sketch needed a new pipe type. The thesis, "every model and
technique is configuration over existing hooks", is therefore false on main today. The gaps are
inputs and data shapes behind slots, not missing abstractions.

Correctness defects on main found by the sketches (being fixed, not tracked):
- `tokenizer.ggml.pre` is never read: every BPE vocab takes the llama3 rule
  (`proxima-tokenizer/src/pipe.rs:76`).
  - The digit difference is real in the rule but changes no ids. The qwen vocabs contain zero
    multi-digit tokens (0 of 151936; 0 of 248320).
  - The ids that do change, confirmed against llama-tokenize at f1ea20621:
    - qwen3.5 treats combining marks as part of words (Devanagari, Thai and Arabic text);
    - Rust's `char` classes differ from llama.cpp's Unicode tables on roman numerals and
      circled letters or digits, in all five families.
- Two tokenizer parity tests are bare `#[ignore]` (`proxima-tokenizer` `tests.rs:332, 370`).
- An `unreachable!` in non-test source (`gemma4/bind.rs:707`).
- A `std::env::var` read inside the op-graph builder (`attention_forward.rs:2121`).
- Bare `std::sync::Mutex` at `load_model.rs:1005, 1014`. No `proxima-lock` crate exists anywhere
  in slot-0, so the workspace lock rule cannot be met as written.

Structural findings from the second batch, which bind the design:
- **`ServingConfig` is `Copy`** (`serving.rs:719`), so no list-valued stage can be a field of it.
  This affects tiers, assemble, idle jobs and the cascade. Today's precedent is setters into
  `PromptCache`, outside the cache key. The hook-as-configured-list shape needs a config type
  that can hold lists, and the cache key must be derived from it.
- **"Decision in proxima-core" has no precedent.** `proxima-core/src` has no serving-decision
  module. Today's pure decisions are `pub(super)` functions in interop (`entry_is_reusable`,
  `draft_limit_for_step`).
- **The draft's `lru|lfu|fifo` default does not reproduce today's victim rule.** Today, unused
  follow-up branches go first.
- **`proxima_lock::Mutex` does not exist in this workspace.** The prewarm gate and queue use
  `std::sync::Mutex`, and their docs say so.
- **Duplicate surfaces:**
  - two n-gram drafters (`draft_prompt_lookup`, which is dead, and `ngram_simple_draft`);
  - `kv.block_tokens` duplicating `prompt_cache.block_tokens`;
  - `speculative` and `speculation` sections one edit apart.
- **`byte_len` counts `Vec` capacity, not length** (`prompt_cache.rs:332-355`).
- **No digest function exists for weights or descriptors.** The cache key and cartridges both
  need one.

Findings the first three share:
- No sketch needed a new pipe *type*. Each hook is a configured slot, and the missing parts
  are inputs and data shapes.
- The placement callback in residency is already a sink pipe. Wrapping it in a named type
  would only relocate it.
- `ModelDescriptor` and `ServingConfig` both lack a config derive, so nothing on these paths
  can be written in TOML yet.

Defects found while sketching:
- Three public qwen35moe pre-gather functions have no caller.
- A bare `std::sync::Mutex<ExpertSlab>` sits at `load_model.rs:1005`.
- Residency `reconcile` runs only on GPU, while `observe` runs on both backends. Not yet
  explained.

## relation to the other specs

- **architecture-as-data:** becomes H4-H7. Its slices stand.
- **fsm-techniques:** contributes H9-H13 and H17. Its cards were written to implement the
  techniques, which is out of scope here.
  - Its useful substrate carries over: the generic entry, the pure decision functions,
    the settings sections, and the sans-IO conformance suite.
  - The cards get re-cut after the paper test, not before.
