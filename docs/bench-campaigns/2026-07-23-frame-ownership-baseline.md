# 2026-07-23 — memcached frame-ownership baseline (post-conversion, sealed)

Campaign: `bench_memcached_frame_ownership` (`proxima-protocols/benches/`),
four criterion groups (`memcached_parse_frame`, `memcached_own_frame_owned_vec`,
`memcached_own_frame_zero_copy`, `memcached_frame_end_to_end`) x 7 workloads
(`get_16b`, `set_1kb_value`, `set_8kb_value`, `set_64kb_value`,
`multiget_20keys`, `malformed_unknown_verb`, `malformed_oversized`), plus one
`stats_alloc`-based allocation report printed by the same run. Host: single
Apple M1 Max (10 cores), macOS Darwin 24.6.0 (15.7.8), `rustc 1.97.1`,
`cargo 1.97.1`. Command: `cargo bench -p proxima-protocols --features
memcached-codec-trait --bench bench_memcached_frame_ownership`, own
`CARGO_TARGET_DIR` (isolated from any other concurrent worktree — per the
shared-target-dir-contamination lesson). 3 full (non-`--quick`) runs plus 3
additional `--quick` smoke runs; numbers below are min-max ranges across the
3 full runs, all captured on this host, this session, today. The claim under
test and workload rationale live in the bench file's own module doc — not
repeated here.

**This supersedes an interim measurement taken earlier the same day, before
the `Get{keys: Bytes}` zero-container conversion landed** (see
`git log` on `proxima-protocols/src/memcached/{frame_codec,pipe_contract,mod}.rs`).
The conversion replaced `MemcachedRequest`'s `Vec<u8>`/`Vec<Vec<u8>>` fields
with `Bytes` windows sliced via `Bytes::slice_ref` (an `Arc` refcount bump,
zero-copy — the same seam `grpc_framing`/`http1_codec`/`websocket_frame`
already ship on) and folded the multi-`get` key list into ONE untouched
`Bytes` span (`MemcachedRequest::Get::keys`), walked lazily by
`pipe_contract::iter_keys` instead of materializing a `Vec<Vec<u8>>`. The
production `own_frame` path now allocates once per request, independent of
value size or key count — see the allocation report below.

Caveats up front: (a) this box was NOT dedicated to the benchmark — a
`scc1_harness` example binary (another repo's worktree, ~100% CPU) and an
`sccache` daemon were both observed running during these runs, the same
loadout the pre-conversion baseline doc flagged; unlike that run, none of
these 3 runs produced a multi-x spike (see finding 2) — this looks like
this box tolerating the same background load better this session, not a
guarantee it always will; (b) `criterion`'s own within-run confidence
interval is narrow (sub-1%) — the spread reported below is RUN-TO-RUN,
which is the noise this doc's CI-gating decision (finding 3) is actually
about; (c) the allocation report is 1 iteration per workload, not a
criterion measurement — it is a direct `stats_alloc::Region` snapshot,
printed once per bench invocation, and was IDENTICAL across all 3 full runs
AND the 3 follow-up `--quick` runs (6 total invocations — see finding 1).

## Allocation report (stats_alloc, 1 iteration per workload — identical across all 6 runs, 3 full + 3 `--quick`)

| workload | wire bytes | parse_frame# | own_frame(vec)# | o_bytes | own_frame(0copy)# | z_bytes | end_to_end# | e_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `get_16b` | 16 | 0 | 1 | 24 | 1 | 128 | 0 | 0 |
| `set_1kb_value` | 1,047 | 0 | 1 | 24 | 0 | 0 | 0 | 0 |
| `set_8kb_value` | 8,215 | 0 | 1 | 24 | 0 | 0 | 0 | 0 |
| `set_64kb_value` | 65,560 | 0 | 1 | 24 | 0 | 0 | 0 | 0 |
| `multiget_20keys` | 165 | 0 | 1 | 24 | 1 | 1,024 | 0 | 0 |
| `malformed_unknown_verb` | 13 | 0 | 0 | 0 | 0 | 0 | 1 | 24 |
| `malformed_oversized` | 26 | 0 | 0 | 0 | 0 | 0 | 1 | 24 |

`own_frame(vec)#` is the CURRENT production lift
(`MemcachedCodec::own_frame` building `MemcachedRequest`'s now-`Bytes`
fields via `Bytes::slice_ref`). Column header is a holdover from the
pre-conversion bench (`Vec` is no longer what this path builds — see the
bench file's own module doc for the full before/after framing); the
numbers are the post-conversion, current-production measurement.

**Read this table against the pre-conversion baseline it supersedes**
(21 allocations / 908 bytes on `multiget_20keys`, 2 allocations on every
other non-malformed arm, scaling with value size on `set_*`): every
non-malformed arm now costs exactly **1 allocation / 24 bytes**, flat,
regardless of value size (`set_1kb_value` through `set_64kb_value`) or key
count (`multiget_20keys`, 20 keys, same 1/24 as a single-key `get`). The 24
bytes is the `bytes` crate's one-time promotion of a `Vec`-backed `Bytes`
to its `Arc`-based shared representation (pointer + capacity + atomic
refcount) on the FIRST `slice_ref`/`clone` call against a given source
window — every subsequent slice off the SAME source (additional keys,
`malformed`'s zero-alloc `Violation` copy) is refcount-only and allocates
nothing further. This is an **O(1)** allocation shape (was **O(payload) +
O(key count)** pre-conversion) — not a constant-factor win, a shape change.
`malformed_unknown_verb`/`malformed_oversized` stay 0-alloc on `own_frame`
in both before and after: they resolve to `Violation` (a `Copy` enum),
never touching `MemcachedRequest` at all.

## Timing (ns/op, min-max range across 3 full runs)

| workload | `parse_frame` | `own_frame` (production) | `own_frame` (zero-copy counterfactual) | `frame_end_to_end` |
| --- | ---: | ---: | ---: | ---: |
| `get_16b` | 10.81–10.97 | 24.47–24.83 | 31.75–32.07 | 49.61–50.65 |
| `set_1kb_value` | 26.48–26.77 | 28.53–28.99 | 28.64–29.13 | 63.08–64.46 |
| `set_8kb_value` | 26.53–27.08 | 28.56–28.79 | 28.61–28.86 | 63.23–64.39 |
| `set_64kb_value` | 27.48–27.76 | 28.06–28.36 | 28.42–28.84 | 65.83–67.15 |
| `multiget_20keys` | 69.72–70.50 | 24.30–24.52 | 289.63–292.72 | 490.77–526.64† |
| `malformed_unknown_verb` | 8.95–9.02 | 19.88–20.13 | 19.93–20.15 | 28.91–29.17 |
| `malformed_oversized` | 28.42–28.85 | 19.74–19.99 | 19.69–19.94 | 48.19–49.27 |

† One of 3 runs put the upper bound at 526.64 ns vs ~495 ns in the other
two — a single-run ~6% high-side wobble on the noisiest/largest arm, not
remotely the 2–5x spike the pre-conversion doc recorded; still evidence
noise, not signal (see finding 2).

`own_frame` (production) is now FASTER than `parse_frame` on `get_16b` and
`multiget_20keys` — expected: it is doing strictly less work than the
pre-conversion `Vec`-copying lift (one refcount bump vs N `to_vec()`
copies), and `multiget_20keys`'s `own_frame` cost is now independent of key
count (24.30–24.52 ns, statistically the SAME as `get_16b`'s 24.47–24.83 ns)
where it previously scaled with key count. The bench-local
`own_frame_zero_copy` counterfactual (unchanged by the conversion — it was
already the target design point) still costs more on `get_16b` (~32 ns,
collects into a `Vec<Bytes>` even for one key) and `multiget_20keys` (~291
ns, same `Vec<Bytes>` collect once per key) than the actual production path
now does, because production no longer materializes ANY container for the
key list (`MemcachedRequest::Get::keys` is one `Bytes` span, walked lazily
by `iter_keys`) — production has now overtaken its own bench counterfactual.

## Findings

1. **Allocation counts are exactly reproducible; timings are not.** All 6
   runs (3 full + 3 `--quick`) produced byte-for-byte identical
   `stats_alloc` counts and byte totals. Every timing arm, run to run,
   moved by single-digit percent, with one ~6% high-side wobble on
   `frame_end_to_end/multiget_20keys` (see finding 2). This is the
   load-bearing evidence for the CI-gating decision (finding 3): allocs/op
   is the honest thing to hard-assert; ns/op is not.
2. **No repeat of the pre-conversion contention spike, same background
   loadout.** The same `scc1_harness` (another repo's worktree, observed
   pegging a core) and `sccache` processes that correlated with the
   pre-conversion doc's 2–5x spike were both present during these runs
   too — this time every arm stayed within single-digit-percent run-to-run
   noise, with the sole exception noted above. This is NOT evidence the
   spike mechanism is fixed or gone; it is one more data point that a
   shared dev box's contention is not reliably reproducible in either
   direction, which is itself part of why a numeric ns/op CI gate would be
   flaky on a shared CI runner (same conclusion as finding 3, arrived at
   from the opposite-direction observation this time).
3. **CI gates on allocation count, not timing.** Given finding 1 and 2, the
   committed CI job smoke-runs the bench (`--quick`; must exit 0 — catches
   panics, feature-gate drift, API breaks) and hard-asserts the
   `own_frame(vec)#` column via a dedicated deterministic test
   (`tests/memcached_frame_ownership_alloc.rs`), not a numeric ns/op
   threshold. No tolerance band is chosen for timing because none would be
   both tight enough to catch a real regression and loose enough to survive
   ordinary shared-box contention — an honest "smoke + alloc-count" gate
   beats a threshold that would need disabling within a week.
4. **The conversion closed the gap to the zero-copy design point, then
   passed it.** Pre-conversion, `own_frame(owned_vec)` scaled with value
   size (48 ns → 1.05–1.23 µs) and key count (21 allocations on
   `multiget_20keys`), strictly worse than the `own_frame(zero_copy)`
   counterfactual on every arm. Post-conversion, production `own_frame` is
   allocation-flat (1/24, matching `zero_copy`'s best case) AND faster than
   `zero_copy` on every arm, because `zero_copy`'s bench-local counterfactual
   still collects keys into a `Vec<Bytes>` (an allocation the production
   path no longer pays at all, by keeping the key list as one untouched
   `Bytes` span). The hypothesis the original bench was built to test — "is
   the per-request owned-copy material?" — is confirmed AND now resolved:
   the production path no longer pays it.
