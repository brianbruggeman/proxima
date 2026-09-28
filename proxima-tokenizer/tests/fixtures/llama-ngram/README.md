# llama-ngram fixtures

Oracle fixtures for proxima's ports of llama.cpp's four self-speculative
n-gram drafters (`speculative-decode-llama-parity` SPEC.md, slice 2). Every
case in every `.json` file here was produced by calling llama.cpp's own
functions directly -- `common_ngram_simple_draft`, `common_ngram_map_begin` /
`_draft` / `_accept`, `common_ngram_cache_update` / `_draft`, and the
`common_speculative_*` public API driving type `ngram-mod` -- never by
re-deriving their behaviour by reading the C++.

## regenerate

```
LLAMA_CPP_SRC=~/repos/others/llama.cpp \
LLAMA_CPP_BUILD=/tmp/llama-ngram-build \
generator/build.sh

generator/fixturegen \
  ~/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd \
  fixtures/
```

`LLAMA_CPP_SRC` must be at commit `f1ea20621` (the commit recorded in every
fixture header). The first argument to `fixturegen` is a GGUF path loaded
vocab-only (gemma4-E2B); the second is the output directory.

## schema

### `fixtures/streams.json`

```json
{
  "vocab_gguf": "<path to the GGUF used for tokenization>",
  "streams": [
    { "id": 0, "source": "<file path or a description of a constructed stream>", "tokens": [<token ids>] }
  ]
}
```

Twelve streams: `llama.cpp README.md`, `proxima README.md`,
`llama.cpp common/ngram-map.cpp`, `proxima proxima-tokenizer/src/vocab.rs`
(real prose and code, tokenized with `common_tokenize` and `add_special =
true`; `streams[0..3]`), plus constructed-but-real-language streams: the
repeated-sentence stream (`streams[4]`), the ngram-mod low-acceptance trap
(`streams[5]`), and six ngram-map key_only-vs-k4v divergence batches
(`streams[6..11]`) -- both described below.

### `fixtures/ngram_simple.json`, `ngram_map_k.json`, `ngram_map_k4v.json`, `ngram_mod.json`, `ngram_cache.json`

```json
{
  "llama_cpp_commit": "f1ea20621",
  "generator": "llama-ngram-fixturegen v1",
  "drafter": "ngram-simple | ngram-map-k | ngram-map-k4v | ngram-mod | ngram-cache",
  "params": { "...": "the llama.cpp DEFAULT params struct for this drafter type" },
  "sources": [],
  "cases": [
    { "stream_id": 0, "position": 70, "sampled": 1234, "draft": [1235, 1236], "accepted": 2 }
  ]
}
```

Each case replays one decode step: `history = stream[..position]`,
`sampled = stream[position]` is fed to the drafter exactly as
`common_speculative_draft_params.id_last` / `common_ngram_*_draft`'s
`sampled` argument. `draft` is the drafter's output for that step.
`accepted` is the longest prefix of `draft` that matches
`stream[position+1 ..]` -- the number a Rust port's parity test should also
compute and feed back into its own `accept(n)` before advancing by
`accepted + 1`, exactly as this generator does for the stateful drafters
(`ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`).

### `fixtures/ngram_cache_static.bin`

A real static n-gram cache, built via `common_ngram_cache_update` over
`streams[2]` (`llama.cpp common/ngram-map.cpp`, `ngram_min=1`,
`ngram_max=4`) and written with llama.cpp's own `common_ngram_cache_save`.
Used for the load round-trip test (`ngram_cache_loads_llama_file`).

## per-drafter defaults (recorded verbatim in each file's `params`)

| drafter | defaults | source |
|---|---|---|
| ngram-simple | `size_ngram=12, size_mgram=48` | `common_params_speculative_ngram_map` defaults, `common/common.h:361-365`, consumed via `common/speculative.cpp:2660-2673` |
| ngram-map-k | `size_key=12, size_value=48, key_only=true, min_hits=1` | same struct, `key_only` forced true for this type by `get_common_ngram_map`, `common/speculative.cpp:2178-2187` |
| ngram-map-k4v | same numeric defaults, `key_only=false` | same |
| ngram-mod | `n_match=24, n_max=64, n_min=48`, table `size=4*1024*1024` (hardcoded, not config), `occupancy_threshold=0.25`, `low_accept_threshold=0.25` over a 5-round streak | `common_params_speculative_ngram_mod`, `common/common.h:354-359`; table size and thresholds at `common/speculative.cpp:1876,1914,2006-2008` |
| ngram-cache | `n_draft=8` (hardcoded in `create_state_ngram_cache`), `ngram_min=1, ngram_max=4` (`LLAMA_NGRAM_MIN`/`MAX`), `nc_dynamic`/`nc_static` empty (no `--lookup-cache-*` path) | `common/speculative.cpp:2189-2203`, `common/ngram-cache.h:9-11` |

## ngram-mod reset mechanism (invariant 3)

ngram-mod's shared hash table has no collision resolution -- `entries[i] =
tokens[n]` always overwrites. Two resets are required by SPEC.md and both
fire exactly once per generator run (see `fixtures/generator.log`):

- **occupancy reset** (`trigger_ngram_mod_occupancy_reset` in
  `generator/main.cpp`): a *throwaway* `common_speculative` instance (never
  contributes to `ngram_mod.json`) is warmed with ~2.2M real tokens
  tokenized from `llama.cpp/ggml/src/*.{c,cpp,h}` in one `begin()` call.
  Real C code has enough repeated boilerplate (headers, includes) that used
  slots trail raw token count roughly 0.55-0.66:1, so the warmup target is
  oversized well past the naive `0.25 * 4*1024*1024 = 1,048,576` threshold.
  Measured: 2,200,000 tokens -> 1,186,686 used slots (0.28) -> reset logged.

- **low-acceptance reset**: relying on chance hash collisions between
  unrelated real text produced **zero** non-empty ngram-mod drafts in
  testing (`n_min=48` requires 48 *consecutive* correct hash hits before any
  draft is returned at all, which chance collisions essentially never
  clear). Instead `streams[5]` is constructed deterministically: the same
  >=24-token real English sentence (`COMMON`) precedes seven distinct real
  English sentences (`TAIL_1..TAIL_7`, on unrelated topics so each starts
  with a different token). Replaying this stream once, in order: the first
  `COMMON` occurrence trains the table (untested, empty draft); the second
  occurrence's query chains through `TAIL_1` (the only thing in the table at
  that hash address) and is compared against the real `TAIL_2` -- wrong from
  token 0, a "low" round. Ground truth then naturally overwrites the
  `COMMON` slot to point at `TAIL_2` as replay advances past it, so the
  third occurrence chains through `TAIL_2` against real `TAIL_3` -- another
  low round, and so on. Six transitions (`TAIL_1`->`TAIL_2` through
  `TAIL_6`->`TAIL_7`) each score `accepted=0`, and no other non-empty draft
  occurs in between (the interior of each real-language tail is novel
  content, so mod.get() returns empty and does not disturb the streak
  counter) -- the fifth consecutive low round fires the reset. This is
  deterministic, not probabilistic: it does not depend on the hash
  function's collision behaviour at all, only on the documented
  last-write-wins overwrite semantics.

Both mechanisms are honest applications of SPEC.md invariant 3's
"choose/extend inputs until they occur" -- no llama.cpp parameter was
changed from its default to force either reset.

## ngram-map key_only vs k4v divergence (streams[6..11])

Without a dedicated stream, `ngram_map_k.json` and `ngram_map_k4v.json` were
byte-identical: with `size_key=12, size_value=48`, real prose/code almost
never repeats the exact same 12-token key with two different 48-token
continuations, so `common_ngram_map` only ever sees one distinct "value"
per key and both modes draft the same match_pos content at the same length.

`key_only` (`common/ngram-map.cpp:382-398`) always drafts `match_pos`'s
content, length-capped by `values[0].n_accepted`, with no ambiguity check.
`k4v` (`:401-516`) instead tallies up to `COMMON_NGRAM_MAX_VALUES=4` distinct
continuations per key and, once a key has recurred with two or more, applies
`sum_occur > 0 && max_occur < 2*sum_occur` (`:495-499`): if no single
continuation clearly dominates, k4v drafts **nothing** while key_only still
drafts from `match_pos` regardless. The two modes can therefore only diverge
in the narrow window right when a second distinct continuation for a key
*first* appears (`slot0.count=1`, freshly-created `slot1.count=1` -- an exact
tie) -- confirmed in isolation with a standalone `size_key=3` diagnostic
(not part of the fixtures) driving a single fresh `common_ngram_map` by
hand: the tie fires at the key's second occurrence, then resolves (one slot
pulls ahead) and both modes reconverge for all later occurrences of that
same key.

Two failed constructions, kept in `generator/main.cpp`'s comments as a
record: (1) concatenating many `key_i + tail_A once + tail_B` scenarios into
one long stream, so all scenarios share one accumulating `common_ngram_map`
(since `replay_ngram_map` builds one map per *stream*, not per scenario) --
this produced exactly 8 divergences regardless of whether 80 or 250
scenarios were concatenated, meaning something in the shared map's long-run
state (many accumulated keys / `key_map` hash entries) suppresses the tie
effect after the first handful of scenarios; and (2) one stream per
scenario, giving each its own fresh map -- this produced **zero**
divergences, because `common_ngram_map_draft`'s match search requires
`size_last_begin > n+m+1=61` (`:278`) and searches strictly beyond that gap
(`:298-312`), so a lone ~113-token scenario never grows large enough for any
match to become findable at all.

`streams[6..11]` combine both lessons: six independent **batches**, each
concatenating 10 scenarios (long enough, with a real `size_last_begin` from
`begin()`, for matches to become findable at all) but short enough to stay
within the handful of early scenarios that reliably tie before the
shared-map falloff observed above. Each batch's key phrase is "for lookup
and matching purposes the recurring context marker number is `N` right here
now." followed by one of two of four rotating real sentences (freshwater
eels, suspension bridges, beekeeping, pipe organs) -- the distinguishing
marker `N` sits near the *end* of the phrase (within the last 12 tokens)
rather than the start, since a marker placed early in a long phrase never
actually changes the trailing-12-token key at all (also measured and
rejected: 0 divergences). Measured result: 24 cases where `ngram_map_k.json`
and `ngram_map_k4v.json` draft differently (`jq` command and count below),
3-5 per batch, matching the same early-scenario pattern from failed
construction (1) above.

## sources

- `/Users/brianbruggeman/repos/others/llama.cpp/README.md`
- `/Users/brianbruggeman/repos/slot-0/proxima/README.md`
- `/Users/brianbruggeman/repos/others/llama.cpp/common/ngram-map.cpp`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tokenizer/src/vocab.rs`
- constructed, real-language: 700x repetition of "Repeat exactly five
  times: the quick brown fox jumps over the lazy dog." (`streams[4]`,
  guarantees genuine periodic matches for the non-stateful/short-window
  drafters)
- constructed, real-language: the ngram-mod low-acceptance trap described
  above (`streams[5]`)
- constructed, real-language: six ngram-map k4v-divergence batches
  described above (`streams[6..11]`)
- occupancy-reset warmup (not a recorded stream): real `.c`/`.cpp`/`.h`
  files under `llama.cpp/ggml/src`, walked in sorted directory order until
  the token target is reached (69 files consumed at the 2.2M-token target;
  exact file list varies only with the target and is not itself
  fixture-relevant)

## verification

```
diff <(jq -c '.cases' fixtures/ngram_map_k.json) <(jq -c '.cases' fixtures/ngram_map_k4v.json)
# exit 1 (files differ)

jq -n --slurpfile k fixtures/ngram_map_k.json --slurpfile v fixtures/ngram_map_k4v.json '
  ($k[0].cases | map({key: "\(.stream_id):\(.position)", value: .draft}) | from_entries) as $a
  | ($v[0].cases | map({key: "\(.stream_id):\(.position)", value: .draft}) | from_entries) as $b
  | [$a | keys[] | select($b[.] != null and $a[.] != $b[.])] | length'
# 24
```
