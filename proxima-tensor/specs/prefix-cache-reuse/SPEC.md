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
| R5 | Chunk reuse after divergence (llama `n_cache_reuse`): when a run of at least `cache_reuse_min` tokens after the LCP reappears at a new position, move those K/V rows and re-rotate K by the position delta instead of recomputing them, with the table each layer's own program rotates with (`rope_cos`/`rope_cos_swa`, partial-rotary factors, scaling); the tokens between runs are prefilled. Runs are found at any offset by an exact rolling hash over the entry's ids, in prompt order, each past the end of the previous one in the entry (llama's pointer walk only steps over cache tokens, so a summary where the middle turns were hides every later run from it; this steps over prompt tokens too). A run's sliding-window rows are moved only while the entry's ring still holds them (llama moves them only with `--swa-full` and otherwise ignores `n_cache_reuse`, `server-context.cpp:3205-3215`, `llama-kv-cache-iswa.cpp:253-257`); a run whose ring rows are gone is prefilled, and the entry is marked restored at the run's end so no later rewind reads the ring rows the move did not write. Default off, like llama's (`n_cache_reuse = 0`). Status: built (`generate/chunk_shift.rs`); measured in S3, see the findings below. |
| R6 | Output through the cache is token-identical to output from a fresh prefill of the same token ids, for every reuse path (pure extension, rewind within slack, checkpoint restore, chunk shift), and token-identical to llama-server on the same ids (the oracle). Chunk shift is the exception the S3 measurements record: the moved rows were computed under the entry's older context, so the ids it generates are the ids llama-server's own chunk reuse generates, not a fresh prefill's. |
| R7 | Config surface: `PromptCacheConfig` (byte budget, checkpoint interval, max checkpoints, cache_reuse_min) via builder and conflaguration, both producing identical configs; `ServingConfig` carries it. |
| R8 | Telemetry per request: `cache_lcp`, `cache_reused_tokens`, `cache_prefilled_tokens`, `cache_path` (extend / rewind / checkpoint / shift / miss), at debug; a miss with a reason. |
| R9 | Speculative caching: the speculative drafter's state is cached with the prompt cache entry and survives across requests. The drafter history and n-gram tables (`DrafterSet`: ngram-simple / map-k / map-k4v / mod / cache) for the reused prefix are restored instead of rebuilt, and rewound to the LCP like the KV (tables built from tokens past the LCP are dropped or rebuilt). The ngram-cache drafter's dynamic cache persists across requests and sessions and can be saved and loaded in llama.cpp's format (`common_ngram_cache_save` / `common_ngram_cache_load`; proxima already ports the loader at `proxima-tokenizer/src/draft/ngram_cache.rs:738`), with an optional static cache path in config (llama `-lcs` / `-lcd`). |
| R10 | Anticipatory prefill: a `prewarm(ids)` API prefills token ids into the prompt cache before any request needs them, at low priority. It runs only while no request is decoding, in chunks, and a real request preempts it at the next chunk boundary; the partially prewarmed entry stays usable up to what was prefilled. The existing LCP lookup consumes prewarmed entries; there is no second path. The end of a generation (EOS or stop) is the built-in prewarm point, because the answer is complete and the GPU is idle while the user reads. At that moment proxima prewarms automatically, in order: (1) the answer's own trailing tokens plus the turn-boundary suffix (end-of-turn and next-user-turn opener token ids, supplied once through config or registration by the caller, so proxima applies it with no per-request policy); (2) optional deeper anticipation: the model drafts K likely follow-up user turns from the formed answer (config: count, max tokens, default off), each prefilled as a branch entry sharing the prefix; the next request's LCP lookup picks whichever branch matches, and unused branches are evicted first. The caller may also hand proxima a prefix it expects (a system prompt at load, retrieved documents while a tool call runs, a user's partial input as they type); what to anticipate beyond the end-of-answer trigger is the caller's policy. Proxima provides prewarm, preemption and accounting (telemetry: `prewarm_tokens`, `prewarm_preempted`, `prewarm_hit_tokens` on the request that consumed it). |
| R11 | Similarity floor (llama `slot_prompt_similarity`, `server-context.cpp:1542-1571`, default 0.1): an entry is reused only when it is extended whole (LCP equals its stored length) or when the LCP covers strictly more than `min_similarity_milli` thousandths of the incoming prompt. Otherwise the request misses (`MissReason::BelowSimilarity`), builds an entry of its own and leaves every older entry to LRU eviction, so an unrelated prompt sharing a chat header cannot rewind a long cached conversation to a few rows. `0` reuses any entry sharing the first token. |
| R12 | The end-of-answer prewarm only queues its prefix (the answer, then the turn-boundary suffix once: the model's repeated end-of-turn tokens are the suffix's first token, not extra ones) and the request returns; a worker the caller scopes around its serving code (`with_prewarm_worker`), or a `run_pending_prewarm` call, prefills it behind the same gate, keeping one backend runtime across jobs. |
| R13 | Prefix index and bloom: the cache finds the entry sharing the longest prefix without comparing the prompt against every entry. Each entry's ids are cut into whole blocks of `block_tokens` (default 64) and held in a radix trie over blocks in a capacity-bounded arena (`u32` handles, a free list, no allocation per node once the arena has grown; at most two nodes per entry, so the capacity comes from `max_entries` and an insert past it evicts or refuses). A node holds a run of whole blocks every entry through it shares; it names an entry through it and the block range of that entry's ids instead of copying tokens, splits when entries diverge, and merges into its only child when it stops branching. Children are found through one table keyed by (parent, hash of the first block of the run), and every hit is verified against the owner's ids (never reuse on a hash alone). A lookup walks the prompt's blocks from the root, offers the entries of the deepest level first, and refines each token by token; the similarity floor (R11) still gates reuse; the chosen entry and prefix length equal those of the scan it replaced. Each entry also carries a bloom filter (`bloom_bits_per_entry`, `bloom_hashes`) over its blocks hashed on content alone, so a prompt whose prefix diverged early (squashed history) can ask which entries probably hold which of its later blocks; reported in telemetry as `bloom_candidate_entries` and `bloom_candidate_blocks` until R5 consumes it. Status: built (`generate/arena.rs`, `generate/prefix_trie.rs`, `generate/block_bloom.rs`). A lookup that reaches a node offers every entry below it, so a prompt sharing only a header with N entries costs O(N) offers; an extension of one entry costs one. |
| R14 | Future, spec only: KV storage owned by the index. Entries that share a prefix would share its KV blocks: a second arena of fixed-size KV blocks with reference counts, trie nodes holding block indices, an entry holding the blocks of its path plus a private tail. Removes the copy each follow-up branch makes of its parent's rows (the AC14 run held 16.5 to 51.0 MB after drafting three branches per answer). Entries keep owning contiguous `LayerCache` buffers until then. |## acceptance criteria

| id | check | expected |
|---|---|---|
| AC1 | unit: LCP of token-id sequences, property test over random pairs | LCP equals the naive scan on 10,000 generated pairs |
| AC2 | gemma4-E2B, 3-turn transcript, each turn extends the last: ids via cache vs fresh prefill | 3 of 3 turns identical, `cache_prefilled_tokens` per turn equals the new-turn token count |
| AC3 | same transcript, turn 2 rewrites the last 50 tokens of turn 1 (within slack) | ids identical to fresh prefill; `cache_path = rewind`; `cache_prefilled_tokens` = 50 + new tokens |
| AC4 | rewrite 2,000 tokens back (beyond slack), checkpoint exists before it | ids identical; `cache_path = checkpoint`; prefilled tokens = LCP-to-checkpoint gap + suffix |
| AC5 | squashed history: system prompt kept byte-identical, middle turns replaced by a summary, last turn kept | ids identical; with R5 on, `cache_path = shift` and reused tokens = system prompt + kept last turn (S3 measured the path and the counts; the ids are in the findings below) |
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
| AC17 | the trie against the scan it replaced; lookup cost against the entry count | the same entry and prefix length on 10,000 generated caches and prompts and over a 10,000-operation random run of store, evict and take (counts asserted); every node freed once every entry is removed; lookup ns at 4, 64, 256 and 1,024 entries reported, hit and miss, with and without a shared 512-token header; bytes of index and bloom per entry reported |
| AC18 | the 8 follow-up chats squashed (opening turn and last user turn kept, the middle replaced by a summary) | blocks the chain and the bloom filters find, against an exact comparison of every aligned block and against exact windows at any offset; the bloom false-positive rate against blocks no entry holds |## slices

| slice | scope | validation |
|---|---|---|
| S1 | R1 + R2 for pure extension and full-attention truncate; R3 rewind within slack; R8 telemetry | AC1, AC2, AC3, AC8 |
| S2 | R4 checkpoints | AC4 |
| S3 | R5 chunk shift with K re-rotation | AC5 |
| S4 | oracle + timing | AC6, AC7 |
| S5 | R9 drafter state cached with the entry, ngram-cache persistence in llama's format | AC9, AC10 |
| S6 | R10 prewarm API, end-of-answer trigger, chunked preemption, optional follow-up branches | AC11, AC12, AC13, AC14 |

## S3 findings: chunk shift (R5, AC5)

Evidence: `proxima-speculative-decode-evidence/prefix_cache/s3/` (`ac5.stdout`, `ac5.jsonl`, `real1.log`, `real5.log`, `gate_real1.log`, `ttft.jsonl`), host gemma4-E2B, llama-server `f1ea20621`.

What moves, read from the code: on gemma4-E2B 15 layers own a KV cache; layers 4, 9, 14 are full attention (head 512, base 1e6, only the first 64 pairs rotate), the other 12 are sliding (window 512, head 256, base 1e4), and layers 15 to 34 read layer 13's and 14's rows. A key is stored rotated for its position, so the key at the new position is the old one rotated by the delta once (`rotate_rows`); V is not rotated.

Rotation against a fresh prefill (`a_chunk_moved_by_the_models_own_rotation_matches_the_rows_a_fresh_prefill_stores`): the same 4,071-token chunk after two openings of different lengths, rows rotated by the delta against the second prefill's rows. Layers 0 to 3 (last 512 rows) differ by at most 1.3e-4 against a key magnitude near 1.0, layer 4 rows 2,200 or more tokens into the chunk by 2.4e-5. These are the layers that read only the chunk (the sliding layers before the first full layer see the last 511 positions through each layer). Layer 4 rows within 256 tokens of the chunk start differ by 0.17, layers 5 to 14 by 0.02 to 1.28, V by up to 18.8: those rows depend on the text before the chunk.

The pipeline writes what it lifted (`the_squash_stores_the_lifted_rows_in_every_layer_not_the_fresh_ones`, squash of a 4,368-token transcript to 2,490): the entry the shifted request leaves holds the lifted rows to within 4.8e-5 in all 15 layers, and a fresh prefill's rows to within 7.4e-4 in layers 0 to 3 and 0.06 to 1.03 in layers 4 to 14.

AC5, 12 squashed transcripts (kept 1, 2 or 3 turns; summary of 90 or 700 characters, or the middle dropped; 3 of the 12 rows repeat the dropped case), 16 tokens greedy:

- proxima shifted ids equal a fresh prefill's in 0 of 12 (first differing token at index 0 to 7); with `cache_reuse_min = 0` the ids equal a fresh prefill's in 12 of 12. `cache_path` is `shift` in 12 of 12; reused tokens are the system prompt (722, restored from its checkpoint) plus 1,038 to 2,122 shifted.
- llama-server `--cache-reuse 256 --swa-full`, middle dropped (6 rows): `cache_n` 2,291, 1,763, 2,842 against proxima's reused 2,288, 1,760, 2,839 (llama counts the 3 shared tokens after the system prompt in place, proxima restores the checkpoint 3 tokens earlier and prefills them); the shifted chunk is the same length (1,566, 1,038, 2,117 tokens for llama's `cache_n` minus its 725 shared tokens). llama's ids equal proxima's shifted ids in 6 of 6 and differ from llama's own fresh prefill in 6 of 6.
- llama-server `--cache-reuse 256 --swa-full`, summary in place of the middle (6 rows): `cache_n` 725 (the shared prefix only) in 6 of 6; its ids equal a fresh prefill's in 6 of 6 and proxima's shifted ids in 0 of 6.
- llama-server `--cache-reuse 256` without `--swa-full`: `cache_n` 0 in 12 of 12 (the shared prefix is not reused either: the SWA window cannot be restored without a checkpoint, `server-context.cpp:3337-3365`; `n_cache_reuse` is ignored, `:3205-3215`).

TTFT, summary variant (kept 2 turns, 2,517-token prompt), 10 interleaved pairs after a warm-up pair: shifted TTFT median 1,264.0 ms (p10 1,236.6, p90 1,283.3, CoV 1.8%) against a full prefill of the same prompt at 11,260.3 ms (p10 11,043.4, p90 11,281.3, CoV 1.0%), median ratio 0.112; the 10 shifted ids differ from the full prefill's in 10 of 10 pairs. The shifted arm reused 2,293 tokens (722 in place, 1,571 moved) and prefilled 224. Release build with the production features; the box was not idle: another process held 100 to 147% CPU and the GPU device utilization read 95 to 98% in the host snapshots between pairs (`ttft.jsonl`), the same for both arms of a pair, which alternate order.

## out of scope here, owned elsewhere

How a client squashes history is prompt policy and belongs to the consuming application (owner decision,
2026-09-17). The cheapest win is policy, not cache: keep the system prompt and tool
definitions byte-identical at the front and put any summary after them, so R2 alone
reuses them.
