# prefix-cache-reuse

status: draft
owner: brian bruggeman
created: 2026-10-01

## problem

proxima has no prompt cache across requests. `prefill_prefix` / `generate_from_prefix`
(`proxima-model-interop/src/generate/decode.rs:2152-2310`) let a caller hold a
`PrefixState` (`residency_caches.rs:707-711`) and continue from it, but nothing matches a
new prompt against cached state: a caller passes an exact `PrefixState` or `None`, and on
any mismatch the whole prompt is prefilled again. No server, example, or downstream application path keeps
state between turns. llama-server, the incumbent (upstream `f1ea20621`), reuses the
longest common token prefix (`tools/server/server-common.cpp:697-709`,
`server-context.cpp:3198`), shifts reusable chunks after a divergence
(`server-context.cpp:3217-3264`, `n_cache_reuse`), and restores sliding-window state from
checkpoints (`server-context.cpp:3276-3365`). This spec gives proxima the same three, in
that order, token-identical to a fresh prefill.

## refutation condition

Written before evidence: if, on gemma4-E2B with a multi-turn transcript where each turn
extends the previous prompt, time-to-first-token with the cache is not lower than
without it by at least the prefill time of the shared prefix minus 10%, the cache is not
doing its job and the spec is reopened.

## facts the design rests on (read, file:line)

- `LayerCache::truncate` rewinds full-attention layers by shrinking their rows, and is a
  no-op for ring layers: "its rows are addressed by absolute position, rewinding is
  `cached_len` going back, and the ring's slack keeps every row the window needs"
  (`residency_caches.rs:140-162`). So a ring layer can rewind only as far as its slack.
- Ring geometry keeps `min(cached_len, window)` recent rows (`kv_ring.rs:1-18`);
  `reserve_ring_rows` never shrinks (`kv_ring.rs:105-115`).
- gemma4-E2B: 35 layers, sliding window 512 on the ring layers, full attention on the
  rest; some layers share KV from an earlier layer (`LayerCacheState::SharedFromLayer`,
  `residency_caches.rs:644-653`).
- Cached K is stored after RoPE, so moving a cached chunk to a new position requires
  re-rotating K by the position delta (llama: `seq_add` with a K-shift).

## requirements

| id | requirement |
|---|---|
| R1 | A per-model prompt cache inside proxima-model-interop: one or more cached entries, each = token ids + `PrefixState`, bounded by a byte budget (config), least-recently-used eviction. |
| R2 | A generate entry point that takes the full prompt token ids, finds the cached entry with the longest common token prefix (LCP), reuses it up to the LCP, and prefills only the remaining tokens. Matching is on token ids, never text. |
| R3 | Rewind to the LCP: full-attention layers truncate to LCP; ring layers rewind when `stored_len - LCP` is within the ring's slack. When it is not, R4 applies; with no checkpoint at or before LCP, the ring layers are rebuilt by a full prefill (correct, slow, logged at debug with the reason). |
| R4 | Sliding-window checkpoints: a bounded number of ring-layer snapshots per cache entry, taken at caller-marked boundaries (turn ends) and every N tokens (config), so a rewind past the slack restores the nearest checkpoint at or before LCP and re-prefills from there. |
| R5 | Chunk reuse after divergence (llama `n_cache_reuse`): when a run of at least `cache_reuse_min` tokens after the LCP reappears at a new position, move those full-attention K/V rows and re-rotate K by the position delta instead of recomputing them. Ring layers are recomputed for moved chunks unless a checkpoint covers them. Default off until measured, like llama's (`n_cache_reuse = 0`). |
| R6 | Output through the cache is token-identical to output from a fresh prefill of the same token ids, for every reuse path (pure extension, rewind within slack, checkpoint restore, chunk shift), and token-identical to llama-server on the same ids (the oracle). |
| R7 | Config surface: `PromptCacheConfig` (byte budget, checkpoint interval, max checkpoints, cache_reuse_min) via builder and conflaguration, both producing identical configs; `ServingConfig` carries it. |
| R8 | Telemetry per request: `cache_lcp`, `cache_reused_tokens`, `cache_prefilled_tokens`, `cache_path` (extend / rewind / checkpoint / shift / miss), at debug; a miss with a reason. |
| R9 | Speculative caching: the speculative drafter's state is cached with the prompt cache entry and survives across requests. The drafter history and n-gram tables (`DrafterSet`: ngram-simple / map-k / map-k4v / mod / cache) for the reused prefix are restored instead of rebuilt, and rewound to the LCP like the KV (tables built from tokens past the LCP are dropped or rebuilt). The ngram-cache drafter's dynamic cache persists across requests and sessions and can be saved and loaded in llama.cpp's format (`common_ngram_cache_save` / `common_ngram_cache_load`; proxima already ports the loader at `proxima-tokenizer/src/draft/ngram_cache.rs:738`), with an optional static cache path in config (llama `-lcs` / `-lcd`). |
| R10 | Anticipatory prefill: a `prewarm(ids)` API prefills token ids into the prompt cache before any request needs them, at low priority. It runs only while no request is decoding, in chunks, and a real request preempts it at the next chunk boundary; the partially prewarmed entry stays usable up to what was prefilled. The existing LCP lookup consumes prewarmed entries; there is no second path. The end of a generation (EOS or stop) is the built-in prewarm point, because the answer is complete and the GPU is idle while the user reads. At that moment proxima prewarms automatically, in order: (1) the answer's own trailing tokens plus the turn-boundary suffix (end-of-turn and next-user-turn opener token ids, supplied once through config or registration by the caller, so proxima applies it with no per-request policy); (2) optional deeper anticipation: the model drafts K likely follow-up user turns from the formed answer (config: count, max tokens, default off), each prefilled as a branch entry sharing the prefix; the next request's LCP lookup picks whichever branch matches, and unused branches are evicted first. The caller may also hand proxima a prefix it expects (a system prompt at load, retrieved documents while a tool call runs, a user's partial input as they type); what to anticipate beyond the end-of-answer trigger is the caller's policy. Proxima provides prewarm, preemption and accounting (telemetry: `prewarm_tokens`, `prewarm_preempted`, `prewarm_hit_tokens` on the request that consumed it). |
| R11 | Similarity floor (llama `slot_prompt_similarity`, `server-context.cpp:1542-1571`, default 0.1): an entry is reused only when it is extended whole (LCP equals its stored length) or when the LCP covers strictly more than `min_similarity_milli` thousandths of the incoming prompt. Otherwise the request misses (`MissReason::BelowSimilarity`), builds an entry of its own and leaves every older entry to LRU eviction, so an unrelated prompt sharing a chat header cannot rewind a long cached conversation to a few rows. `0` reuses any entry sharing the first token. |
| R12 | The end-of-answer prewarm only queues its prefix (the answer, then the turn-boundary suffix once: the model's repeated end-of-turn tokens are the suffix's first token, not extra ones) and the request returns; a worker the caller scopes around its serving code (`with_prewarm_worker`), or a `run_pending_prewarm` call, prefills it behind the same gate, keeping one backend runtime across jobs. |
## acceptance criteria

| id | check | expected |
|---|---|---|
| AC1 | unit: LCP of token-id sequences, property test over random pairs | LCP equals the naive scan on 10,000 generated pairs |
| AC2 | gemma4-E2B, 3-turn transcript, each turn extends the last: ids via cache vs fresh prefill | 3 of 3 turns identical, `cache_prefilled_tokens` per turn equals the new-turn token count |
| AC3 | same transcript, turn 2 rewrites the last 50 tokens of turn 1 (within slack) | ids identical to fresh prefill; `cache_path = rewind`; `cache_prefilled_tokens` = 50 + new tokens |
| AC4 | rewrite 2,000 tokens back (beyond slack), checkpoint exists before it | ids identical; `cache_path = checkpoint`; prefilled tokens = LCP-to-checkpoint gap + suffix |
| AC5 | squashed history: system prompt kept byte-identical, middle turns replaced by a summary, last turn kept | ids identical; with R5 on, `cache_path = shift` and reused tokens = system prompt + kept last turn |
| AC6 | oracle: AC2-AC5 prompts through llama-server f1ea20621 with `cache_prompt` on | proxima ids equal llama's ids on every prompt (count reported) |
| AC7 | TTFT, quiet box, 10 interleaved pairs, cache on vs off, AC2 transcript turn 3 | median TTFT ratio reported with p10/p90; refutation condition applied |
| AC8 | builder vs conflaguration parity fixture for `PromptCacheConfig` | identical config, 1 fixture |
| AC9 | multi-turn transcript, speculation on: drafted and accepted counts and output ids with drafter state restored from the cache vs a run that rebuilds the drafter from the full prompt each turn | identical on every turn; per-turn drafter rebuild work drops to the new tokens only (count reported) |
| AC10 | save the dynamic ngram cache, reload in a fresh process; llama's `common_ngram_cache_load` reads proxima's file and proxima reads llama's | round trip, byte and entry counts equal |
| AC11 | prewarm the next-turn prefix after turn N; turn N+1's request | prefills only the user's new tokens (count); ids identical to a fresh prefill; TTFT ratio with vs without prewarm reported (10 interleaved pairs) |
| AC12 | a real request arrives mid-prewarm | the request's first prefill dispatch starts within one prewarm chunk's time (measured delay); the partial prewarm is reused (`prewarm_hit_tokens` > 0) |
| AC13 | after an answer ends, with no further call from the caller, the next turn's request | prefills only the user's new tokens (count), proving the end-of-answer trigger fired; ids identical to a fresh prefill |
| AC14 | deeper anticipation, default off: K follow-up branches prewarmed | hit rate (requests whose LCP extended into a branch) and tokens saved per hit reported on a multi-turn transcript set; preemption still within one chunk |
| AC15 | conversation A (over 1,000 tokens), an unrelated conversation B sharing only the chat header, then A's next turn | B reports `BelowSimilarity` and reuses 0 tokens; A's next turn reuses at least the tokens A held (count); ids identical to a fresh prefill |
| AC16 | the answer call with the end-of-answer suffix registered vs not (10 interleaved pairs) | answer-call time within noise of the unregistered call; turn N+1 still prefills only the user's tokens |
## slices

| slice | scope | validation |
|---|---|---|
| S1 | R1 + R2 for pure extension and full-attention truncate; R3 rewind within slack; R8 telemetry | AC1, AC2, AC3, AC8 |
| S2 | R4 checkpoints | AC4 |
| S3 | R5 chunk shift with K re-rotation | AC5 |
| S4 | oracle + timing | AC6, AC7 |
| S5 | R9 drafter state cached with the entry, ngram-cache persistence in llama's format | AC9, AC10 |
| S6 | R10 prewarm API, end-of-answer trigger, chunked preemption, optional follow-up branches | AC11, AC12, AC13, AC14 |

## out of scope here, owned elsewhere

How a client squashes history is prompt policy and belongs to the consuming application (owner decision,
2026-09-17). The cheapest win is policy, not cache: keep the system prompt and tool
definitions byte-identical at the front and put any summary after them, so R2 alone
reuses them.
