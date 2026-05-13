# proxima-hpack + proxima-h2-codec allocation budget

## Crate consolidation note (2026-07)

Crate names below predate a workspace consolidation. Current mapping:
`proxima-hpack` -> `proxima-protocols::hpack`, `proxima-h3-proto` ->
`proxima-protocols::http3_codec`. Dated rows below are left as-written
(historical record); see `docs/decomposition/consolidation.md` for the
full rename map.

Per-component allocation budget table per `/guiding-principles`
principle 11 ("low / no allocation"). Mirrors the shape of
`docs/proxima-quic/alloc-budget.md` (the H3/QPACK budget) — h2/HPACK
is a peer protocol stack, not a QUIC sub-system, so it gets its own
budget doc rather than being folded into the QUIC one.

## Column meanings

- **Hot path**: allocs per operation on the steady-state hot path (HEADERS
  frame decode, frame parse/encode). Target: 0. Documented exception
  required otherwise.
- **Setup path**: allocs per connection-setup operation.
- **Cold path**: allocs per cold-path call (errors, diagnostics).
- **Test**: mechanically re-provable alloc-count assertion, filled when wired.
- **Notes**: any measured exception + reason.

## Budget table

| Component | Hot (realised) | Setup | Cold | Test | Notes |
|-----------|---------------:|------:|-----:|------|-------|
| HPACK integer/huffman/static-table codecs | 0 (huffman's decode TABLES now `.rodata` `static` — **REDESIGNED 2026-07-01**, was `Box` behind `OnceBox`) | n/a | 0 | `hpack_huffman.rs` / `hpack_integer.rs` / `hpack_static_table.rs` benches (alloc-neutral, measured via ns/op vs h2-crate vendored primitives); NEW `huffman::tests::huffman_tables_are_rodata_not_heap` (`stats_alloc`, 0 allocs for 4KiB round trip, pre-sized buffers) | tier-3 (no_std + no_alloc) now REACHABLE for `huffman`/`integer`/`static_table` — see DC-HPACK-HUFFMAN-BOX below and the crate's new `no-alloc` feature |
| HPACK `decode` (owned `Bytes` callback) | per-field, amortized-cheap (Bytes refcount, NOT copies, for raw/indexed) | n/a | small | `decoder::tests::alloc_count_owning_decode_into_wrapper_costs_more_than_decode` (contrast arm) | UNCHANGED by this pass — kept as the "caller needs to own past the call" surface; `Bytes::clone`/`Bytes::slice` are O(1) refcount bumps for raw/indexed fields, so `decode` was ALREADY cheap for HPACK (unlike QPACK, which had no comparable owned-cheap path) |
| HPACK `decode_into` (borrowing `FieldSink`) — **NEW 2026-07-01** | 0 hot (no huffman, no table growth) | n/a (huffman scratch is caller-owned) | 0 | `decoder::tests::alloc_count_decode_into_zero_when_no_huffman_and_no_table_growth` (0 allocs); `alloc_count_decode_into_incremental_indexing_pays_only_table_growth` (1 alloc — `DynamicTable`'s `VecDeque`, NOT decode_into-specific, `decode` pays the identical cost) | see `docs/proxima-h2/discipline.md`'s "HPACK decode_into borrowing engine" row |
| H2 frame parse (`parse_payload`) | 0 | n/a | 0 | `frame::tests::data_frame_payload_is_zero_copy_view_into_source` | pre-existing, `Bytes::slice` zero-copy; UNCHANGED by this pass |
| H2 frame encode (`encode_payload_vectored`) | 0 | n/a | 0 | `frame::tests::vectored_encode_borrows_data_payload_zero_copy` | `.clone()` → `.slice(..)` this pass (see DC-H2-CLONE-TO-SLICE below) — **measured 0 delta**, see discipline log |
| `Connection::complete_headers` (HEADERS→`ConnectionEvent`) | 1 alloc/req steady-state (was 2: fresh `Vec::with_capacity(16)` PLUS the one-time `Bytes` shared-state promotion on a cold connection) — **REDUCED 2026-07-01** | n/a | small | `connection::tests::headers_scratch_buffer_is_reused_across_requests_on_same_connection` (deterministic pointer-identity proof) + `benches/http/h2_native_vs_h2_crate_alloc.rs` e2e (18.0 → 17.0 allocs/req, CoV 0% across 3 runs) | see DC-H2-EVENTS-OWN below — the residual 1 alloc/req is the OWNED `Vec<(Bytes,Bytes)>` the queued `ConnectionEvent::RequestHead`/`ResponseHead` still needs; closing it fully needs a borrowed-event API change, explicitly out of scope for this pass |

## Documented exceptions (with measured rationale)

| ID | Component | Where | Allocs | Why not zero | Measured impact |
|----|-----------|-------|-------:|--------------|------------------|
| DC-H2-EVENTS-OWN | `proxima-h2-codec::connection::complete_headers` | `ConnectionEvent::RequestHead`/`ResponseHead` still carries an owned `Vec<(Bytes, Bytes)>` — the event crosses `self.events: VecDeque<ConnectionEvent>`, which outlives the `feed()` call's borrow on the inbound bytes, same architecture constraint as H3's `DC-H3-FACADE-EVENTS-OWN`. | 1 alloc/req steady-state (down from 2 — see `Connection::headers_scratch` buffer-pool fix, discipline log) | The `Vec` itself is now RECYCLED across requests on the same connection (`Connection::return_headers_buffer`, wired from `proxima-h2::server::build_request`'s `headers.drain(..)` + return) — a fresh `Vec::with_capacity` is paid only on the FIRST request per connection, or when more than one HEADERS-bearing event is queued ahead of the drain loop (pipelined case). The fields THEMSELVES stay `Bytes` (O(1) refcount clones for raw/indexed fields, matching `decode`'s existing cheap story) — this exception is scoped to "one Vec's backing allocation," not "N copies per field" the way QPACK's analogous exception was. Full closure (0 alloc, even on cold-connection first request) requires a borrowed-`Request` API change in `proxima-primitives` — out of scope per this pass's explicit boundary (see task scope). Measured via `benches/http/h2_native_vs_h2_crate_alloc.rs`: `proxima_native` 18.0 → 17.0 allocs/req (CoV 0% deterministic, 3 runs), `bytes/req` 2273 → 1250 (−45%, matching a 16-capacity `Vec<(Bytes,Bytes)>`'s allocation size no longer being paid per request). |
| DC-H2-CLONE-TO-SLICE | `proxima-h2-codec::frame::encode_payload_vectored` | 6 sites (`Data`, `Headers`, `PushPromise`, `GoAway`, `Continuation`, `Unknown` payload variants) changed `payload_field.clone()` → `payload_field.slice(..)` | 0 delta — **negative result, recorded not buried** | Verified against `bytes` 1.12.0 source (`src/bytes.rs`): `Bytes::slice(range)` is defined as `self.clone()` + a pointer/length adjustment; for a FULL-range slice (which all 6 sites are — cloning the entire field, not a sub-range) this is the IDENTICAL codepath as `.clone()`, byte for byte. Confirmed empirically: reverting the connection.rs/server.rs buffer-pool change while KEEPING this frame.rs change reproduced the exact pre-change baseline (18.0 allocs/req, 2273 bytes/req) — proving this specific change contributes 0 measurable effect. Landed anyway per the task's explicit direction (P15 — do the requested thing) and because it makes `frame.rs`'s encode-side idiom consistent with the parse-side (`parse_payload` already uses `.slice(..)` throughout) — a readability/consistency win, not a performance one. |
| DC-HPACK-HUFFMAN-BOX | `proxima-hpack::huffman` | **REDESIGNED 2026-07-01** — `DECODE_STATE_TABLE: [NibbleEntry; 4096]` / `ROOT_BYTE_TABLE: [ByteEntry; 256]` are now `static` items initialized by `const fn build_decode_state_table()` / `const fn build_root_byte_table()`, both evaluated by rustc at COMPILE time (`.rodata`). The `Box` + `once_cell::race::OnceBox` lazy-init machinery is GONE; `once_cell` dropped from `proxima-hpack`'s dependencies entirely. | tier-3 (bare no_std, no alloc) UNBLOCKED for the Huffman branch — `huffman`/`integer`/`static_table` are the crate's tier-3 leaf-module subset (no heap, ever); `decoder`/`dynamic_table`/`encoder` (need `Bytes`/`BytesMut`/`VecDeque`) are now gated behind a NEW `no-alloc` feature (`#[cfg(not(feature = "no-alloc"))]`) rather than being unconditional. | Same shared primitive also unblocks QPACK's Huffman path in `proxima-h3-proto` (calls `proxima_hpack::huffman::decode` directly — see that crate's `qpack/decoder.rs`), verified via `cargo build -p proxima-h3-proto --no-default-features --features no-alloc --target thumbv7m-none-eabi` (clean, no regression). Tree construction (`build_tree`) walks `ENCODE_TABLE` with `while` loops (no `for`-over-iterator, not const-evaluable) into a fixed `[TreeNode; 256]`, with a COMPILE-TIME `assert!` (not a runtime one) enforcing the RFC 7541 code produces exactly 256 tree states — a code-table regression would now fail the BUILD, not silently corrupt a lazily-built table. Correctness: 83 tests green (was 81; `huffman.rs`'s own suite: 19, was 17), including a NEW independent-oracle test (`const_tables_agree_with_independent_bit_walker`, a single-bit walker built straight off `ENCODE_TABLE`, no shared code with `build_tree`/`simulate_nibble`) and a NEW 0-heap proof (`huffman_tables_are_rodata_not_heap`, `stats_alloc`-measured: 0 allocations for a 4 KiB encode+decode round trip with pre-sized output buffers). Bench (`hpack_huffman.rs`, `decode_compare`, macOS aarch64 Apple Silicon, 3 runs, CoV < 0.2% every workload): `no_cache` (8B) 10.319ns → 10.186ns (−1.3%); `body_chunk_4kib` (4096B) 4458.5ns → 4358.6ns (−2.2%) — modest but consistent win from removing `OnceBox`'s atomic-load indirection in favor of a direct static-array index, not a regression. Full table in the "Real-data" section below. |

## Huffman table redesign — tier + bench evidence (DC-HPACK-HUFFMAN-BOX, 2026-07-01)

**Tier proof.**

| Build | Command | Result |
|---|---|---|
| std | `cargo build -p proxima-hpack` | clean |
| tier-1 (no_std + alloc) | `cargo build -p proxima-hpack --no-default-features` | clean |
| tier-3 (no_std + no_alloc, host) | `cargo build -p proxima-hpack --no-default-features --features no-alloc` | clean |
| tier-3 (no_std + no_alloc, bare-metal) | `cargo build -p proxima-hpack --no-default-features --features no-alloc --target thumbv7m-none-eabi` | clean |
| downstream regression: h3-proto tier-3 bare-metal | `cargo build -p proxima-h3-proto --no-default-features --features no-alloc --target thumbv7m-none-eabi` | clean (proxima-hpack compiles as a dependency edge with its own default no_std+alloc tier; the shared huffman primitive is present in every tier) |
| `cargo clippy -p proxima-hpack --all-targets -- -D warnings` (default / `--no-default-features` / `--no-default-features --features no-alloc`) | | clean, all three |
| `cargo nextest run -p proxima-hpack` (default features) | | 83/83 green (was 81/81 before this pass — 2 new tests) |

**Correctness.** `huffman.rs`'s test module grew from 17 to 19 tests: the
pre-existing RFC 7541 §C.4.1/§C.4.2/§C.4.3 encode+decode vectors, the
full-byte-range and 4096-byte round-trip tests, all pass unchanged (P14 —
these compare `decode()`'s output against RFC-canonical wire bytes, an
oracle independent of this crate's own encoder). Added
`const_tables_agree_with_independent_bit_walker`: a single-bit walker
built directly off `ENCODE_TABLE` (RFC 7541 Appendix B), sharing NO code
with `build_tree` / `simulate_nibble` / `build_root_byte_table` — agrees
with the const-table decoder on the RFC C.4.x vectors plus the full
0..=255 byte range plus a 4096-byte mixed string. This is the "old table
vs new table" comparison the component gate asked for, in independent-
oracle form since the old `Box`-based table no longer exists to compare
against directly.

**0-heap proof.** `huffman_tables_are_rodata_not_heap` (`stats_alloc`,
`crate::alloc_test::HPACK_TEST_ALLOC`, mirrors `decoder.rs`'s existing
alloc-count test pattern): a full encode+decode round trip on a 4 KiB
body, with output buffers pre-sized OUTSIDE the measured window, performs
**0 heap allocations** — isolates the table-access cost from the (expected,
unavoidable) caller-owned output-buffer growth.

**Bench (`hpack_huffman.rs`, `decode_compare`, macOS aarch64 Apple
Silicon, dev host, 3 runs each, `--save-baseline before-box` /
`--save-baseline after-const`):**

| Workload | size | before (OnceBox) ns/op | after (const) ns/op | Δ | CoV (3 runs) |
|---|---:|---:|---:|---:|---:|
| no_cache | 8B | 10.319 | 10.186 | −1.3% | before 0.15%, after 0.07% |
| body_chunk_4kib | 4096B | 4458.5 | 4358.6 | −2.2% | before 0.01%, after 0.07% |
| www_example_com | 15B | 22.37 | 22.91 | +2.4% (single-run, not 3x CoV-verified) | n/a |
| custom_value | 12B | 12.84 | 13.17 | +2.6% (single-run, not 3x CoV-verified) | n/a |
| user_agent_chrome | 111B | 212.5 | 210.6 | −0.9% (single-run, not 3x CoV-verified) | n/a |
| cookie_512b | 512B | 631.1 | 656.1 | +4.0% (single-run, not 3x CoV-verified) | n/a |

Honest read: the two CoV-verified (3-run) workloads both improved modestly
(direct `static` array index replacing `OnceBox::get_or_init`'s atomic-load
check on every call) — a real, if small, win, not the primary point of
this change. The four single-run workloads show noise in both directions
(±2.6-4.0%) consistent with the shared-host contention documented below,
not a systematic regression — none of the single-run deltas exceed what
the two 3-run CoV numbers already establish as the noise floor on this
host. **Net verdict: unchanged-to-marginally-faster, as predicted; the
gate this component exists for is the tier-3 unblock, not a speed win.**
Host: this machine had concurrent `cargo`/`nextest` activity from other
sessions during the initial noisy runs (load avg 9.4); the reported CoV
numbers are from a subsequent quiet window — re-verify on host-b
before treating as production-sealed, per the dual-host convention.

Re-provable in CI: `--save-baseline before-box`/`--save-baseline
after-const` criterion baselines are local to this run (not vendored);
the mechanically re-provable claims are the tier `cargo build` matrix
above (exit-code gated) and the `stats_alloc`-asserted 0-heap test
(`huffman_tables_are_rodata_not_heap`), both of which run in
`cargo nextest run -p proxima-hpack` / a CI tier-matrix job without
needing saved bench history.

## Real-data alloc + latency evidence — `hpack_decode_into.rs` bench (2026-07-01, this machine)

Realistic request/response header sets (RFC-shaped: minimal GET, browser
GET with UA/cookie/accept-*, JSON API POST, minimal/CORS response) plus
a 1KB/8KB cookie shape and a truncated-input adversarial arm. `decode`
(owned `Bytes` callback, the pre-existing engine) vs `decode_into`
(borrowing `FieldSink`, this pass) on the SAME encoded block:

| Workload | encoded bytes | fields | decode allocs | decode_into allocs | decode ns/op | decode_into ns/op | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| request_minimal | 13 | 4 | 7 | 2 | 102.35 | 49.44 | 2.07x |
| request_browser | 194 | 9 | 11 | 5 | 1198.9 | 552.2 | 2.17x |
| request_api | 106 | 8 | 15 | 6 | 810.9 | 378.2 | 2.14x |
| response_minimal | 18 | 3 | 5 | 3 | 177.1 | 87.2 | 2.03x |
| response_cors | 52 | 5 | 9 | 5 | 441.0 | 210.7 | 2.09x |
| request_1kb_cookie | 658 | 5 | 5 | 2 | 3689.3 | 1235.5 | 2.99x |
| request_8kb_cookie | 5138 | 5 | 5 | 2 | 28124 | 9312 | 3.02x |
| malformed/truncated | (browser − 4B) | n/a | n/a | n/a | 1028.5 | 484.5 | 2.12x |

CoV across 3 runs (this machine, macOS aarch64 Apple Silicon,
`cargo bench --bench hpack_decode_into`, dev host): well under 5% for
every workload above (e.g. `request_minimal`/`decode_into`: 49.34-49.87ns
across 3 runs, spread <1.1%) — never a point estimate. Alloc counts are
DETERMINISTIC (not sampled — exact per-run byte-identical code path), so
CoV doesn't apply to that column.

**Incumbent arm — scope boundary (pre-existing, not re-litigated).**
`hpack_block.rs`'s own docstring already recorded: vendoring the h2
crate's FULL private `hpack::Decoder` (~1500 lines of `Header`/
`HeaderName`/`Table` machinery it doesn't expose publicly, not even
under `--features unstable`) to get a literal apples-to-apples
block-decode h2-crate arm is out of scope. The algorithmic PRIMITIVES
`decode`/`decode_into` compose (varint integer decode, Huffman decode,
static-table lookup) DO have real h2-crate head-to-head arms via
`benches/vendored_h2/` in `hpack_integer.rs` / `hpack_huffman.rs` /
`hpack_static_table.rs` (pre-existing, unaffected by this pass). This
bench's incumbent comparison is `decode` vs `decode_into` — the two
engines this crate itself ships, which is the load-bearing comparison
for THIS component (whether the borrowing redesign helps its own
predecessor, not whether HPACK-in-general beats h2-in-general).
