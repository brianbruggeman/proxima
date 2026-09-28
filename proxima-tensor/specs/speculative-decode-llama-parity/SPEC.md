# speculative-decode-llama-parity

status: audited
owner: brian bruggeman
created: 2026-09-28

## problem

proxima's speculative decode covers 1 of llama.cpp's 10 speculation types (upstream
`f1ea20621`, `common/common.h:173-186`), runs only on gemma4
(`Architecture::speculative_verify_program` returns `Ok(None)` for every other arch,
`proxima-model-interop/src/architecture.rs:292-309`), is reachable only through an env
var (`decode.rs` `PROXIMA_SPECULATIVE_DECODE`), and drafted 0 tokens on a 64-token prose
prompt -- so for a caller of `LoadedModel::generate_with_serving_config`, speculative
decode is measured as absent; this spec makes every llama.cpp speculation type available
through ServingConfig, token-identical to non-speculative decode, on every shipped
architecture.

## refutation condition

Written before evidence: if, on the same model/prompt/seed, the four llama.cpp
self-speculative drafters (ngram-simple, ngram-map-k, ngram-mod, ngram-cache) each accept
a mean of < 0.1 draft tokens per verify step on the corpus named in AC17, the n-gram half
of this spec (R4-R7, R13) is struck.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | verify is sample-and-match: each verify row selects its token through the exact per-step selection the non-speculative path uses (override, penalties window, `sample_next_token`, shared rng); speculative output is byte-identical to non-speculative output for any ServingConfig and seed | yes |
| R2 | every `Architecture` impl in proxima-model-interop returns `Some` from `speculative_verify_program` | yes |
| R3a | rejected drafts are rolled back in attention KV via `LayerCache::truncate` | yes |
| R3b | rejected drafts are rolled back in recurrent/SSM state via snapshot-restore of the pre-verify state | yes |
| R4 | `ngram-simple` drafter, token-identical to `common_ngram_simple_draft` (`common/ngram-map.cpp:49`) | yes |
| R5 | `ngram-map-k` drafter, token-identical to `common_ngram_map_begin/draft/accept` with `key_only = true` | yes |
| R13 | `ngram-map-k4v` drafter, token-identical to the same functions with `key_only = false` | yes |
| R6 | `ngram-mod` drafter, token-identical to `common_speculative_impl_ngram_mod` (`common/speculative.cpp:1849-2022`) including the 0.25 occupancy reset and the 5-round low-acceptance reset | yes |
| R7 | `ngram-cache` drafter, token-identical to `common_ngram_cache_draft` (`common/ngram-cache.cpp`), and loads llama.cpp static/dynamic cache files | yes |
| R8 | `draft_ngram_lookup` is removed; `ngram-simple` (R4) replaces it -- one prompt-lookup primitive, the incumbent's | yes |
| R9 | `ServingConfig` carries a speculative section mirroring `common_params_speculative`; builder and conflaguration loader produce identical configs; the env var is removed | yes |
| R10 | `draft-simple`: a second GGUF draft model, compatibility check equivalent to `common_speculative_are_compatible`, drafting equivalent to `common_speculative_impl_draft_simple` (`common/speculative.cpp:179`) | yes |
| R11a | an audited sub-spec exists for `draft-mtp` (`common/speculative.cpp:1331`); draft correctness is that sub-spec's requirement | yes |
| R11b | an audited sub-spec exists for `draft-eagle3` (`:455`) | yes |
| R11c | an audited sub-spec exists for `draft-dflash` (`:923`) | yes |
| R11d | an audited sub-spec exists for `draft-dspark` | yes |
| R12 | per-request `draft_n` / `draft_n_accepted` emitted as proxima telemetry events, matching llama-server's timings fields | yes |
| R14 | the refutation measurement: mean accepted draft tokens per verify step is recorded per n-gram type on the AC17 corpus | yes |
| R15 | speculation ON is faster than OFF: per n-gram type, on the AC17 corpus, on the metal path (`gpu_layers = all`) and the CPU path, median per-pair speedup (OFF ms/token ÷ ON ms/token) > 1.0 with pair-ratio CoV ≤ 5% over ≥ 5 interleaved OFF/ON pairs per prompt | yes |
| R16 | idle overhead is bounded: on a corpus where the drafter proposes nothing, ON ms/token ≤ 1.02 × OFF ms/token, same interleaving and CoV rules | yes |
| R17 | proxima's speedup ratio ≥ llama.cpp `f1ea20621`'s speedup ratio (`llama-server --spec-type <type>` vs no spec) per n-gram type, same GGUF, prompts, sampling config, hardware | yes |
| R18 | drafters allocate nothing per call: `draft()` writes into a caller-owned buffer; a counting allocator records 0 allocations over 100 000 `draft()` calls per n-gram type on the fixture streams | yes |
| R19 | the verify-width cost curve is recorded: ms per verify forward at width k+1 for k ∈ {0, 1, 4, 8, 16, 48} on metal and CPU, plus the break-even accepted-tokens-per-step computed from it | yes |
| R20 | speculation runs on the metal path: with `gpu_layers = all` the verify program binds and runs and R1's parity holds | yes |

## architecture

Two pure halves plus one loop seam, all sans-IO:

- **drafters** live in `proxima-tokenizer/src/draft/` (one file per llama type). Each is a
  pure state machine over token ids: `begin(&[u32])`, `draft(history: &[u32], last: u32,
  out: &mut Vec<u32>)`, `accept(accepted: u16)`. The decode loop holds a closed
  `enum Drafter { NgramSimple(..), NgramMapK(..), NgramMod(..), NgramCache(..), DraftModel(..) }`
  and matches (box-free).
- **verify** is one shared per-row selection function in `decode.rs`, called by both the
  non-speculative step and each verify row (R1).
- **all-positions program** per architecture (R2) is the existing capability method; each
  impl pins `logits_root` to every position.
- **rollback** (R3) is per `LayerCacheState`: `Attention` truncates, recurrent restores a
  pre-verify snapshot.
- **parity example CLI** (`examples/speculative_decode_parity.rs`) grows flags:
  `--drafter <type>`, `--draft-model <path>`, `--seed-mismatch-control`, `--telemetry-file <path>`, `--gpu-layers <n|all>`.
- **performance harness** (`examples/speculative_bench.rs`): per prompt, OFF and ON run as
  interleaved pairs (order swapped every pair, warmup pairs discarded); per arm it records
  ms/token over the decode window, TTFT, p50/p99 per-token latency, verify steps, accepted
  tokens, drafts proposed, RSS, CPU%, and on metal GPU utilization (ioreg IOAccelerator
  "Device Utilization %", ~10 Hz, idle baseline); it prints per-pair ratios with median, p90,
  CoV, win fraction, and the count of contaminated pairs excluded.
- **measurement protocol**: an exclusive phase -- no concurrent cargo, Metal tests, or agents;
  `ollama ps` recorded before/after, Ollama quit for the run and reopened after; load checked
  before each arm; a pair whose idle window samples > 5% GPU is contaminated and excluded.
- **incumbent arm**: `llama-server` built from `~/repos/others/llama.cpp` at `f1ea20621`
  (Metal), same prompts over its HTTP API with and without `--spec-type <type>`, reading its
  own `timings.predicted_per_second`, `draft_n`, `draft_n_accepted`.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| verify rule | sample-and-match (llama `common_sampler_sample_and_accept_n`) | p/q rejection sampling cannot be byte-identical to non-speculative decode, so parity would only be statistical |
| oracle | llama.cpp `f1ea20621` ngram functions compiled into a C++ fixture generator; fixtures vendored with the commit recorded | re-deriving expected values from reading C++ is principle 14's "from memory" failure |
| drafter dispatch | closed enum + match | `Box<dyn Drafter>` violates box-free; the set is llama's enum, closed |
| prompt-lookup | llama `ngram-simple` replaces `draft_ngram_lookup` | two prompt-lookup primitives differing only in tie-break and sizes is RISC debt |
| draft-model families | sub-spec per family | each needs a GGUF head format + forward graph read from upstream before its ACs can be named |
| draft-simple fixture | target = draft = gemma4-E2B blob | same-vocab by construction, available locally; exercises the full second-model path |
| timing arms | interleaved per pair, ratio per pair | back-to-back arm blocks measured different GPU clock states: a 2.7x "win" that was 1.1x (2026-09-21) |
| win metric | speedup ratio vs OFF and vs llama.cpp's own ratio | absolute tokens/s moves with hardware and quant; the ratio isolates what speculation buys |
| idle overhead gate | ≤ 2% | drafting runs every step even when nothing matches; unbounded idle cost makes default-on wrong |

## acceptance criteria

`$EX` = `cargo run --release -p proxima-model-interop --features std --example speculative_decode_parity --`
`$E2B` = `/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `$EX` | exit 0; 2 config blocks (greedy; t=0.8 top_k 40 top_p 0.9 min_p 0.05 repeat 1.1), each `identical = true` and `speculative_verify_steps` ≥ 1 |
| AC2 | R1 | `$EX --seed-mismatch-control` | exit 0; sampled block prints `identical = false` (ON run seeded differently -- a control that must diverge) |
| AC3 | R2 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/verify_program_bound_for_every_architecture/)'` | 1 passed; test prints `archs_asserted = N registry = N` with equal N |
| AC4a | R3a | `cargo nextest run -p proxima-model-interop --features std -E 'test(/layer_cache_truncate/)'` | ≥ 3 passed, 0 skipped |
| AC4b | R3b | `cargo nextest run -p proxima-model-interop --features std -E 'test(/rollback_recurrent/)'` | ≥ 2 passed, 0 skipped |
| AC5 | R4 | `cargo nextest run -p proxima-tokenizer -E 'test(/ngram_simple_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N ≥ 200 |
| AC6 | R5 | `cargo nextest run -p proxima-tokenizer -E 'test(/ngram_map_k_matches_llama_fixture/)'` | 1 passed; `cases` ≥ 200 |
| AC7 | R13 | `cargo nextest run -p proxima-tokenizer -E 'test(/ngram_map_k4v_matches_llama_fixture/)'` | 1 passed; `cases` ≥ 200 |
| AC8 | R6 | `cargo nextest run -p proxima-tokenizer -E 'test(/ngram_mod_matches_llama_fixture/)'` | 1 passed; `cases` ≥ 200, `occupancy_resets` ≥ 1, `low_accept_resets` ≥ 1 |
| AC9 | R7 | `cargo nextest run -p proxima-tokenizer -E 'test(/ngram_cache_matches_llama_fixture/) or test(/ngram_cache_loads_llama_file/)'` | 2 passed; `cases` ≥ 200 |
| AC10 | R4,R5,R13,R6,R7 | `for t in ngram-simple ngram-map-k ngram-map-k4v ngram-mod ngram-cache; do $EX --drafter $t "$E2B" "Repeat exactly five times: the quick brown fox jumps over the lazy dog." 40; done` | 5 runs, each exit 0, both config blocks `identical = true` |
| AC11 | R8 | `git grep -c draft_ngram_lookup -- ':!proxima-tensor/specs'` | 0 matches |
| AC12 | R9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_builder_matches_loader/)'` | 1 passed |
| AC13 | R9 | `git grep -c PROXIMA_SPECULATIVE_DECODE -- ':!proxima-tensor/specs'` | 0 matches |
| AC14 | R10 | `$EX --drafter draft-simple --draft-model "$E2B" "$E2B"` and `cargo nextest run -p proxima-model-interop --features std -E 'test(/draft_model_vocab_mismatch_is_typed_error/)'` | exit 0 with both blocks `identical = true`, steps ≥ 1; 1 passed |
| AC15 | R11a,R11b,R11c,R11d | `for f in mtp eagle3 dflash dspark; do grep -c '^status: audited' proxima-tensor/specs/speculative-draft-$f/SPEC.md; done` | 4 lines, each `1` |
| AC16 | R12 | `$EX --telemetry-file /tmp/spec_tel.log && grep -c 'draft_n_accepted' /tmp/spec_tel.log` | count equals the example's printed `speculative_verify_steps` summed over both blocks |
| AC17 | R14 | `cargo run --release -p proxima-model-interop --features std --example speculative_acceptance_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl "$E2B"` | corpus has ≥ 50 prompts (printed `prompts = N`); 5 rows (one per n-gram type) of `mean_accepted_per_step`; any row < 0.1 invokes the refutation clause |

`$BENCH` = `cargo run --release -p proxima-model-interop --features std,metal --example speculative_bench --`
`$CORPUS` = `proxima-model-interop/examples/data/speculative_corpus.jsonl`
`$TYPES` = `ngram-simple ngram-map-k ngram-map-k4v ngram-mod ngram-cache`

| id | discharges | command | expected |
|---|---|---|---|
| AC18 | R15 | `for t in $TYPES; do $BENCH --drafter $t --corpus $CORPUS --pairs 5 --gpu-layers all "$E2B"; done` | 5 rows; each `median_speedup` > 1.00, `pair_cov` ≤ 0.05, `contaminated_pairs = 0` |
| AC19 | R15 | `for t in $TYPES; do $BENCH --drafter $t --corpus $CORPUS --pairs 5 --gpu-layers 0 "$E2B"; done` | 5 rows; each `median_speedup` > 1.00, `pair_cov` ≤ 0.05 |
| AC20 | R16 | `$BENCH --drafter ngram-simple --corpus proxima-model-interop/examples/data/speculative_no_repeat.jsonl --pairs 5 --gpu-layers all "$E2B"` | `verify_steps_total = 0`, `median_overhead` ≤ 1.02, `pair_cov` ≤ 0.05 |
| AC21 | R17 | `for t in $TYPES; do $BENCH --drafter $t --incumbent llama-server --corpus $CORPUS --pairs 5 --gpu-layers all "$E2B"; done` | 5 rows; each prints `proxima_speedup` and `llama_speedup` with `proxima_speedup ≥ llama_speedup` |
| AC22 | R18 | `cargo nextest run -p proxima-tokenizer -E 'test(/drafter_zero_alloc/)'` | 5 passed; each prints `allocs = 0 over 100000 calls` |
| AC23 | R19 | `$BENCH --verify-width-sweep 0,1,4,8,16,48 --gpu-layers all "$E2B"` then the same with `--gpu-layers 0` | 6 rows per run with `ms_per_verify` and `cov` ≤ 0.05; 1 `break_even_accepted_per_step` line per run |
| AC24 | R20 | `$EX --gpu-layers all` | exit 0; both config blocks `identical = true`, `speculative_verify_steps` ≥ 1 |

## out of scope

- tree / multi-sequence drafting (`examples/speculative` branch splitting, `n_seq_dft > 1`) -- not in llama-server's path
- synthetic acceptance rates (`server_sample_and_accept_synth`) -- bench-only upstream
- multimodal speculation -- disabled upstream (server mmproj abort)
- draft correctness for mtp/eagle3/dflash/dspark -- owned by their sub-specs (R11a-d)

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| recurrent-state rollback needs a snapshot per verify step, costing more than speculation saves | medium | R3 lands but speculation is slower on SSM archs | measure per arch; read upstream `tests/test-recurrent-state-rollback` first |
| all-positions program for MoE archs changes routing cost at width k+1 | medium | verify slower than k single steps | bench per arch; default-on only with a measured win |
| fixture generator drifts from upstream | low | tests pass against a stale oracle | fixture header records upstream commit; generator source checked in |
| draft-model families need GGUF variants proxima-gguf cannot parse | high | R11 sub-specs grow | each sub-spec names its parse gap first |

## context

- proxima: `proxima-tokenizer/src/draft/` (`mod.rs`, `ngram_simple.rs`), `proxima-tokenizer/src/sample.rs`, `proxima-model-interop/src/generate/decode.rs` (draft ~3033, verify ~4962), `proxima-model-interop/src/architecture.rs:292-309`, `proxima-model-interop/src/gemma4/bind.rs:1105-1110`, `proxima-model-interop/src/generate/residency_caches.rs` (`LayerCache::truncate`), `proxima-model-interop/examples/speculative_decode_parity.rs`
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`: `common/speculative.cpp` (impls at 179, 455, 923, 1331, 1750, 1795, 1849, 2024), `common/ngram-map.cpp`, `common/ngram-mod.cpp`, `common/ngram-cache.cpp`, `common/sampling.cpp` (`common_sampler_sample_and_accept_n`), `tools/server/server-context.cpp:3885-3900`, `common/common.h:173-186`, `:355-364`
