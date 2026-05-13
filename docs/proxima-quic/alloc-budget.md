# proxima-quic + proxima-h3 allocation budget

## Crate consolidation note (2026-07)

The standalone sans-IO proto crates named below were folded into
consolidated crates:

- `proxima-quic-proto` → `proxima-protocols::quic`
- `proxima-h3-proto` → `proxima-protocols::http3_codec`
- `proxima-hpack` → `proxima-protocols::hpack`

Table rows below use the pre-consolidation names as written at the
time. Full rename map:
[`docs/decomposition/consolidation.md`](../decomposition/consolidation.md).

Per-component allocation budget table per `/guiding-principles`
principle 11 ("low / no allocation"). The binding invariant lives in
[`ai_docs/invariants.jsonl`](../../ai_docs/invariants.jsonl) under
`proxima.decision.sans_io_contract_binding`; clause 2 binds zero
hot-path allocations for tier-3-promotable modules AND for the inner
loop of every tier-1 module. Tier-1 modules whose returned values or
queued events outlive the inbound borrow file an explicit exception
row below. Each exception MUST cite the per-operation alloc count
AND the planned redesign that would re-eliminate it.

**Verification status (current).** The `tracking_allocator`-based
per-exception assertions described in the exception rows below are
**not yet wired**. The exception rows document the per-op counts
that a future test MUST pin, but no `tracking_allocator` test
currently asserts them — a regression that doubled the count would
land unnoticed. This is the explicit subject of the meta-exception
row `DC-H3-ALLOC-TEST-WIRE` below; until that row is marked done,
the budget is *documented* but not *enforced* for the H3 exceptions.
The other rows (the proxima-quic-proto crate's hot-path leaves)
are enforced by their type-system shape — tier-3 builds compile
without `extern crate alloc;` so the alloc surface is structurally
absent, no runtime assertion needed.

This honest split is required by
`proxima.decision.sans_io_contract_binding`: an exception without a
measurement IS a clause-2 failure, but the failure is at the
*wiring* layer (the meta-exception) rather than the per-component
layer. Until that wiring lands, treat the H3 exception rows as
aspirational claims to be verified, not gates that are passing.

## Column meanings

- **Hot path**: allocs per operation on the steady-state hot path (e.g.
  `Connection::handle_datagram` + `Connection::poll_transmit` cycle, frame
  parse, AEAD protect/unprotect). Target: 0. Documented exception required
  otherwise.
- **Setup path**: allocs per setup operation (connection creation, key
  install, transport-param exchange). Bounded, recorded.
- **Cold path**: allocs per cold-path call (errors, diagnostics, drop-on-
  error). Allowed where the bench shows call rate is genuinely cold.
- **Test**: 100k-iteration alloc-counter test result (filled when component
  lands).
- **Notes**: any measured exception + reason.

## proxima-quic-proto budget

| Component | Hot path | Setup path | Cold path | Test | Notes |
|-----------|---------:|----------:|----------:|------|-------|
| C1 varint | 0 | n/a | 0 | DONE by type system (no alloc available in tier-3) | tier-3 PROMOTED; thumbv7m cliff green |
| C2 packet header | 0 | n/a | 0 | DONE by type system (no alloc available in tier-3) | tier-3 PROMOTED; all 6 Header variants borrow into input |
| C3 frame codec | 0 | n/a | 0 | DONE by type system (no alloc available in tier-3) | tier-3 PROMOTED; all 28+ Frame variants borrow into input; AckRanges is borrowed iterator |
| C4 transport parameters | 0 | bounded | 0 | DONE by type system (no alloc available in tier-3) | tier-3 PROMOTED; struct of `Option<T>` + borrowed slices for CIDs/tokens |
| C5 HKDF-Expand-Label | 0 | n/a | 0 | DONE by type system (no alloc available in tier-3) | tier-3 PROMOTED; RustCrypto sha2/hmac/hkdf are all stack-only |
| C6 AEAD packet protection | 0 | n/a | 0 | DONE by type system (no alloc available in tier-3) | tier-3 PROMOTED; RustCrypto aes-gcm + chacha20poly1305 in-place |
| C7 header protection | 0 | n/a | 0 | DONE by type system (no alloc available in tier-3) | tier-3 PROMOTED; AES + ChaCha20 stack ops |
| C8 connection ID stores | 0 | bounded | 0 | DONE by type system | tier-3 PROMOTED; `CidQueue<CAP>` const-generic, `[u8; MAX_CID_LEN]` per entry, no heap |
| C9 packet number spaces | 0 | n/a | 0 | DONE by type system | tier-3 PROMOTED; `SendSpace` POD + `RecvSpace<WINDOW>` `[u8; 64]` bitmap |
| C10 packet protection compose | 0 | n/a | 0 | DONE by type system | tier-3 PROMOTED; composes C5+C6+C7+C9; scope re-cut from "TLS 1.3 sans-IO" — full TLS deferred to C11 |
| C11 connection state machine | 0 | bounded | small | pending | tier-1 — owns sub-state |
| C12 streams + flow control | 0 hot read/write | per-stream-create | small | pending | tier-1 — alloc per new stream |
| C13 ACK generation | 0 | n/a | 0 | pending | `ArrayRangeSet<MAX_ACK_RANGES>` |
| C14 loss detection | 0 | n/a | small | pending | `heapless::Deque<MAX>` for sent-packets |
| C15 NewReno | 0 | n/a | 0 | pending | fixed-size cwnd state |
| C16 CUBIC | 0 | n/a | 0 | pending | fixed-size cwnd state + epoch-start time |
| C17 BBRv2 | 0 | n/a | 0 | pending | bandwidth filter state — fixed-cap window |
| C18 ECN | 0 | n/a | 0 | pending | pure counter logic |
| C19 address validation | 0 | per-token | 0 | pending | retry-token = arrayvec of bounded size |
| C20 anti-amplification | 0 | n/a | 0 | pending | counter only |
| C21 path migration | 0 | per-path | 0 | pending | `arrayvec<Path, MAX_PATHS>` |
| C22 version negotiation | 0 | n/a | 0 | pending | stateless |
| C23 key update | 0 | per-update | 0 | pending | typestate; per-phase key install bounded |
| C24 0-RTT | 0 hot | per-resumption | 0 | pending | session ticket = bounded blob |
| C25 RFC 9221 DATAGRAM | 0 hot send/recv | bounded | 0 | pending | queue caps from sized.rs |
| C26 multipath | 0 | per-path | 0 | pending | fixed `MAX_PATHS_PER_CONNECTION` cap |
| C27 endpoint demux | 0 hot | per-connection | 0 | pending | `heapless::IndexMap<DCID, ConnId, MAX>` |

## proxima-h3-proto budget

The H3 proto layer landed with hot-path allocations the original
budget claimed as 0. The honest current state is the **realised**
column below; the **target** column is the original aspiration plus
the design changes required to reach it. A row whose realised and
target diverge has an entry in the exceptions table.

| Component | Hot (realised) | Hot (target) | Setup | Cold | Test | Notes |
|-----------|---------------:|-------------:|------:|-----:|------|-------|
| C32 H3 frame codec | 0 | 0 | n/a | 0 | DONE by type system (no alloc available in tier-3) | tier-3 PROMOTED — `H3Frame<'_>` borrows into the caller's input slice. |
| C33 QPACK encoder | per-block | 0 hot | per-request | small | pending | dynamic table alloc on insert only; encode writes into caller-owned `Vec<u8>`. |
| C34 QPACK decoder | 0 hot (`decode_into`) | 0 hot | per-request (`decode_bounded`'s `1 + 2*field_count`, alloc-tier convenience wrapper only) | small | DC-H3-QPACK-DECODE-OWNS-VECS — REDESIGNED 2026-07-01 | `decode_into` (tier-3, `FieldSink`-driven borrowing engine) is 0-alloc, measured via `stats_alloc`. `decode_bounded`/`decode` are now thin wrappers over it — same owned-`Vec<DecodedField>` shape as before, moved from "the only path" to "one convenience surface over the 0-alloc engine". |
| C35 H3 server state machine | per-event (default `Owned`) — **0 alloc/request-HEADERS opt-in** (`part-source`, `RequestHeaderMode::Source`) | 0 hot | per-request | small | DC-H3-FACADE-EVENTS-OWN — server half REDESIGNED 2026-07-01 (opt-in, C4 in `docs/proxima-pipe/discipline.md`) | `H3ServerEvent::Request{Headers,Data,Trailers}` (default) still carry owned `Vec`s via `decode_bounded` — byte-for-byte unchanged. A connection in `RequestHeaderMode::Source` (via `ServerConnection::enable_header_source_mode` or the listener spec key `part_source`) queues request-HEADERS sections raw (pool-recycled, 0 steady-state allocations) and the facade builds the dispatch `Request` straight from stepping `poll_request_header_source` — e2e +6.5% mean over owned on the local A/B, 6/6 pairs. |
| C36 H3 client state machine | 1 alloc/response (default `Owned` mode; was per-event `1 + 2*field_count`) — **0 alloc/response opt-in** (`part-source` feature, `ResponseHeaderMode::Source`) | 0 hot | per-request | small | DC-H3-FACADE-EVENTS-OWN — client half REDESIGNED 2026-07-01 (C36-R); the residual 1-alloc/response `header_block` copy REDESIGNED again 2026-07-01 (part-source step 3, opt-in) | `H3ClientEvent::ResponseHeaders` (default `Owned` mode, unchanged since C36-R) still carries `status: Option<u16>` (Copy, 0 alloc) + `header_block: Vec<u8>` (1 alloc). A connection that calls `ClientConnection::enable_header_source_mode` (`part-source` feature, default OFF) instead queues each response's still-encoded HEADERS section in a pool-recycled buffer (0 steady-state allocations, 24-byte queue elements) and lazily steps a borrowed `qpack::part_source::FieldSectionSource` from `poll_response_header_source` (0 allocations to poll + step — the C3 fix for C2's multi-KB arena-move regression, see `docs/proxima-pipe/discipline.md`); `ResponseTrailers` is unaffected (still `{ stream_id }` only, 0 alloc). |
| C37 H3-Datagrams | 0 | 0 | n/a | 0 | pending | composes RFC 9221 (datagram queues are bounded heapless caps). |
| C38 extended CONNECT | 0 hot | 0 hot | per-CONNECT | small | pending | request-shape state. |

## Documented exceptions (with measured rationale)

Format: component / where / how-much / why-not-zero / measured-impact.
Each exception is also a discipline-log row (column links).

| ID | Component | Where | Allocs | Why not zero | Measured impact / planned redesign |
|----|-----------|-------|-------:|--------------|------------------------------------|
| DC-H3-QPACK-DECODE-OWNS-VECS | C34 QPACK decoder | **REDESIGNED 2026-07-01** — `qpack::decoder::decode_into(input, cap, &mut scratch, &mut sink: impl FieldSink)` is the engine now; `decode_bounded`/`decode` are thin `VecFieldSink` wrappers over it | `decode_into`: 0. `decode_bounded`: unchanged shape, `1 + 2*field_count` (now an explicit CHOICE at the wrapper layer, not the only path) | Huffman-encoded literals still need somewhere to write variable-length output — that's now the caller/engine-owned `scratch: &mut [u8]` (stack array in `decode_bounded`'s case), not a `Vec`. `decode_into` yields BORROWED views (`'static` static-table / `&input` raw / `&scratch` Huffman) to `FieldSink`, so the engine itself never owns a byte. | Measured via `stats_alloc`: `decode_into` = 0 allocations on a 5-field nginx-shaped fixture (`qpack::decoder::tests::alloc_count_decode_into_zero_decode_bounded_one_plus_two_per_field` + `bench_c34_decode.rs`'s alloc report). `decode_bounded` unchanged at `1 + 2*field_count` (11 for the same fixture) — now a documented CHOICE for callers that want an owned, request-lifetime-independent copy, not a forced cost. See `docs/proxima-quic/discipline.md`'s "C34 — QPACK decoder borrowing engine" entry. |
| DC-H3-FACADE-EVENTS-OWN | C35 / C36 connection FSMs | **Client half (C36) REDESIGNED 2026-07-01 (C36-R)**, then **REDESIGNED again 2026-07-01 (part-source step 3, opt-in, `part-source` feature default OFF)** — default mode unchanged: `H3ClientEvent::ResponseHeaders { stream_id, status: Option<u16>, header_block: Vec<u8> }`; `ResponseTrailers { stream_id }` (signal only). A connection may instead call `ClientConnection::enable_header_source_mode()` to route response HEADERS to `poll_response_header_source() -> Option<(StreamId, qpack::part_source::FieldSectionSource<'_>)>` — 0 heap allocations to feed, poll, AND step (C3 lazy-source shape, 2026-07-01). Mutually exclusive per connection; the default path is byte-for-byte unchanged for callers who never opt in. **Server half (C35) REDESIGNED 2026-07-01 (opt-in, same shape as the client's C3)** — default `Owned` events unchanged; `RequestHeaderMode::Source` + the listener's `request_head_from_source` build the dispatch `Request` from the stepped source with 0 proto-layer allocations (C4 row, `docs/proxima-pipe/discipline.md`). Request DATA/trailers events unchanged. | Client HEADERS (`Owned`, default): 1 alloc/response (`header_block.to_vec()`), down from `1 + 2*field_count`. Client HEADERS (`Source`, opt-in): **0 allocations** — measured through `ClientConnection::feed_response` itself, not just the underlying adapter in isolation. Client TRAILERS: 0 (down from `1 + 2*field_count`, no consumer read the fields). Client DATA: unchanged, 1 Vec/event. Server: unchanged, `1 Vec per emitted event + 2 per decoded field (HEADERS / Trailers)`. | Client-side (`Owned` mode): events are still queued in a `VecDeque` that outlives the borrow on the inbound bytes, so SOMETHING must be owned by the time it lands in the queue — `header_block` (the still-QPACK-encoded bytes) is now that one owned thing, instead of `field_count` owned name+value pairs. `status` rides along as a `Copy` field extracted during the SAME `decode_into` pass that validates the section (0 marginal cost). Client-side (`Source` mode, C3 shape): the "something must be owned to cross the queue boundary" constraint is satisfied by the still-encoded block itself, copied once into a pool-recycled `Vec<u8>` (0 steady-state allocations) — decode is deferred to `poll_response_header_source`, which steps a borrowed `qpack::part_source::FieldSectionSource` over the block. The prior shape (eager decode into `HeaderBlockPartSource`'s fixed inline arena, queued by value) moved ~6 KB per queue hop and measured 0.73× the `Owned` throughput — replaced by C3 (`docs/proxima-pipe/discipline.md`), which restored parity. A caller needing full header enumeration in `Owned` mode (`H3NativeUpstream`) decodes `header_block` itself via `decode_into`, straight into its own target type — skipping the `DecodedField` intermediate the old `decode_bounded` path forced on every response. | Measured via `stats_alloc` (`client::tests::alloc_count_apply_response_frame_headers_is_one_not_one_plus_two_per_field` + `alloc_count_decode_status_is_zero` for `Owned`; `client::tests::alloc_count_feed_response_source_mode_is_zero_owned_mode_is_greater_than_zero` for `Source`): `apply_response_frame` (`Owned`) on a 5-field nginx-shaped response = 1 allocation (was 11 via `decode_bounded`); `decode_status` alone = 0; `feed_response` in `Source` mode on the SAME 5-field fixture = 0 allocations, `Owned` mode through the same `feed_response` entry point = >0 (consistent with the 1-alloc claim). `H3NativeUpstream`'s full-forward path (status + all 4 non-pseudo headers) = 9 allocations (was 19: `1 + 2*5` `DecodedField` Vecs + `2*4` `HeaderList` `Bytes`) — unaffected by this row, `H3NativeUpstream` itself hasn't been ported to the source path. **Planned redesign, still open**: the still-fully-owned server half (C35); a per-protocol dispatch-hot-path default flip (design doc step 3's own gate: needs a composed end-to-end bench proving the source path wins before any default changes — not attempted this row, see `docs/proxima-pipe/discipline.md`); `H3NativeUpstream`'s reverse-proxy forward path onto the source. Recorded as edges entry "H3 event-queue alloc redesign". |
| DC-H3-DRIVER-PASS-ALLOCS | proxima-h3 driver pass | `route_inbound_streams_*` / `drain_request_streams_*` / `drain_complete_frames` allocate per pass | 1 Vec per pass for `connection.stream_ids().collect()`, 1 Vec per stream for `request_recv_buf.entry().or_default().extend_from_slice`, 1 Vec per complete-frame drain | The driver is in the `proxima-h3` facade crate (tier-2, std + tokio), NOT in the sans-IO proto crate. Tier-2 facades are NOT bound by the 0-hot-alloc budget — the budget binds the proto crates. | Acknowledged. The facade-vs-proto split is the relevant invariant; the proto crate's hot-path budget remains 0 hot for the **proto layer's** entry points (the bytes-in / bytes-out / event-out API). |
| DC-H3-ALLOC-TEST-WIRE | **meta-row** — wires the per-exception assertions for DC-H3-QPACK-DECODE-OWNS-VECS + DC-H3-FACADE-EVENTS-OWN | **PARTIALLY WIRED 2026-07-01** — QPACK decoder (C34) and H3 client (C36-R) exceptions now have live `stats_alloc`-backed assertions; server (C35) does NOT yet. | n/a (test wiring) | `qpack::decoder`'s and `client`'s alloc-count claims are now mechanically re-provable in CI (`stats_alloc::StatsAlloc<System>` wired as `#[cfg(all(test, feature = "std"))]` global allocator in `crate::alloc_test::QPACK_TEST_ALLOC`, `proxima-h3-proto/src/lib.rs`) — a regression that doubled either's alloc count fails `cargo nextest run -p proxima-h3-proto`. Server (C35)'s `feed_request` alloc count is still unenforced. | **Done**: `qpack::decoder::tests::alloc_count_decode_into_zero_decode_bounded_one_plus_two_per_field` (asserts `decode_into`=0, `decode_bounded`=`1+2*field_count`); `client::tests::alloc_count_apply_response_frame_headers_is_one_not_one_plus_two_per_field` + `alloc_count_decode_status_is_zero` (asserts the C36-R claims); `client::tests::alloc_count_feed_response_source_mode_is_zero_owned_mode_is_greater_than_zero` (asserts the part-source step 3 `Source`-mode-is-0/`Owned`-mode-is->0 claim, `part-source` feature). **Done 2026-07-01 (Source path)**: `server::tests::alloc_count_feed_request_source_mode_is_zero_owned_mode_is_greater_than_zero` asserts `feed_request` (Source) = 0 AND poll+step = 0 AND `Owned` > 0. **Still open**: an exact per-event alloc-count assertion for the `Owned` server path (the `> 0` bound is asserted, the exact `1 + 2*field_count`-shaped formula is not). |
| DC-HPACK-HUFFMAN-BOX | QPACK's Huffman path (`proxima_hpack::huffman::decode`, which QPACK's `decode_into` composes) | **`proxima-hpack`-side REDESIGNED 2026-07-01** (see `docs/proxima-h2/alloc-budget.md`) — `DECODE_STATE_TABLE` / `ROOT_BYTE_TABLE` are now `const fn`-built `static` `.rodata`, no `Box`, no `once_cell`. `proxima-hpack::huffman` is tier-3 (no_std + no_alloc) capable via its new `no-alloc` feature, verified on `thumbv7m-none-eabi`. | The SHARED blocker (proxima-hpack needing a heap for its tables) is GONE. **STILL OPEN**: `proxima-h3-proto`'s OWN `qpack::decoder::{resolve_value, resolve_name, resolve_both_huffman}` remain `#[cfg(feature = "alloc")]`-gated independently of `proxima-hpack`'s tier — a bare `no-alloc` h3-proto build still declines Huffman literals with `DecodeError::HuffmanUnsupported` (still RFC-permitted, not a correctness gap), but the reason cited in that module's doc comment is now stale. | Verified no regression: `cargo build -p proxima-h3-proto --no-default-features --features no-alloc --target thumbv7m-none-eabi` stays clean (proxima-hpack compiles fine as a dependency at any tier). **Follow-up, separate component**: drop the `alloc` cfg gate on QPACK's three Huffman-resolving functions and wire them to `proxima-hpack::huffman`'s now-tier-3-available `decode`, closing this row fully — needs its own bench + tier proof, not done in this pass (which was scoped to `proxima-hpack`'s own table storage, per the task's own component boundary). |
