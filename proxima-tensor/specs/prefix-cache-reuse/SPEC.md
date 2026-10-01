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

## slices

| slice | scope | validation |
|---|---|---|
| S1 | R1 + R2 for pure extension and full-attention truncate; R3 rewind within slack; R8 telemetry | AC1, AC2, AC3, AC8 |
| S2 | R4 checkpoints | AC4 |
| S3 | R5 chunk shift with K re-rotation | AC5 |
| S4 | oracle + timing | AC6, AC7 |

## out of scope here, owned elsewhere

How a client squashes history is prompt policy and belongs to the consuming application (owner decision,
2026-09-17). The cheapest win is policy, not cache: keep the system prompt and tool
definitions byte-identical at the front and put any summary after them, so R2 alone
reuses them.
