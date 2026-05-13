# proxima-quic + proxima-h3 hot-path performance targets

## Crate consolidation note (2026-07)

`proxima-h3` (std stack) was folded into `proxima-http::http3`.
Component rows below use the pre-consolidation name as written at
the time. Full rename map:
[`docs/decomposition/consolidation.md`](../decomposition/consolidation.md).

Per-component target numbers for the hot-path invariants from
the workspace AGENTS.md §"hot-path requirements". Filled
in as components land. The numbers here are GATES, not aspirations —
a component that misses its target does NOT land.

## Workspace-wide hot-path gates (bind for every component)

| Gate | Target |
|------|-------:|
| RAM at 10k connections | ≤ 500 MB |
| Sustained throughput per connection on single core | ≥ 55 MB/s (over 60s) |
| p99 latency for `handle_datagram` + `poll_transmit` cycle | < 1 ms |
| Heap allocations in hot path | 0 per [`alloc-budget.md`](./alloc-budget.md); tier-3-promotable modules verified structurally by the tier-3 build path. Tier-1 modules whose event payload outlives the inbound borrow MAY file a documented exception row with a per-op count; per-op assertion via `tracking_allocator` test is the OPEN meta-row `DC-H3-ALLOC-TEST-WIRE` (until that lands, the H3 exception rows are aspirational, not enforced). |
| O(N²) in query / scoring / index path | forbidden |
| Lock-free concurrent reads at facade boundary | required |

## Per-component target table

(Filled in as components land. Format: component / scope / target /
incumbent reference / measured.)

### Codec layer (C1–C4)

| Component | Scope | Target | Incumbent ref (quinn-proto) | Measured |
|-----------|-------|-------:|----------------------------:|----------|
| C1 varint encode | 1 / 2 / 4 / 8 byte | beat quinn-proto on incumbent home turf | quinn-proto VarInt | **0.50 / 0.69 / 0.82 / 0.97 ns** (3-7× faster than quinn-proto) |
| C1 varint decode | 1 / 2 / 4 / 8 byte | beat quinn-proto on single-value; stream within 25% | quinn-proto VarInt | single: **0.66 / 0.85 / 0.94 / 1.01 ns** (beats quinn on all classes); stream: ~25% slower at 1KB+ (quinn's Buf design wins streaming) |
| C2 packet header parse | 1200-byte Initial / Short | beat quinn-proto::PartialDecode | quinn-proto PartialDecode | parse Initial **8.86 ns** (6.2× faster); parse Short **0.90 ns** (51× faster, includes quinn's BytesMut alloc) |
| C2 packet header encode | long header (Initial, 1220 B) | sub-30 ns | n/a (quinn encoder gated on cipher context) | encode Initial **24.79 ns** |
| C3 frame parse (ACK) | typical ACK frame | sub-100 ns at 50 ranges | quinn-proto Frame is pub(crate) — no direct compare | 5 ranges **18 ns**; 50 ranges **82 ns**; 500 ranges **745 ns** (~1.5 ns/range) |
| C3 frame parse (STREAM) | 1200-byte STREAM | sub-10 ns (zero-copy on data) | n/a | **4.90 ns** (flat across 16 B / 1 KB / 1200 B / 8 KB) — data borrowed |
| C3 frame parse (mixed) | PADDING+PING+ACK+STREAM+HSDONE | sub-100 ns | n/a | **67 ns total** (~13 ns/frame) |
| C3 frame encode (STREAM) | 1200-byte payload | sub-30 ns | n/a | **28 ns** (~24 ns in copy_from_slice) |
| C4 transport params parse | full param set (17 params, ~88 B) | sub-100 ns | quinn-proto::TransportParameters (deferred to C10) | **57 ns** (~3.4 ns/param) |
| C4 transport params encode | full param set | sub-50 ns | n/a | **32.6 ns** |
| C4 transport params parse | preferred-address subset | sub-20 ns | n/a | **13.5 ns** |

### Crypto layer (C5–C9)

| Component | Scope | Target | Incumbent ref (quinn-proto + aws-lc-rs) | Measured |
|-----------|-------|-------:|----------------------------------------:|----------|
| C5 HKDF-Expand-Label | full initial key pair derivation (RFC 9001 §A.1 vectors bit-exact) | sub-10 µs per derivation | quinn-proto crypto::ring (crate-private, deferred to C10) | **5.07 µs** (HKDF-Extract + 8× Expand-Label); single Expand-Label **781 ns** |
| C6 AEAD protect (ChaCha20) | 1200 B | clear AGENTS.md 55 MB/s gate | quinn::crypto::Cipher (deferred to C10) | **2.83 µs / ~424 MB/s** |
| C6 AEAD protect (AES-128-GCM) | 1200 B | clear AGENTS.md 55 MB/s gate | quinn::crypto::Cipher (deferred to C10) | **8.12 µs / ~147 MB/s** (aarch64; will improve on x86_64 with AES-NI) |
| C6 AEAD protect (AES-128-GCM) | 1452 B | clear AGENTS.md 55 MB/s gate | quinn::crypto::Cipher (deferred to C10) | **9.56 µs / ~152 MB/s** |
| C6 build_nonce | n/a | sub-5 ns | n/a (pure XOR) | **1.86 ns** |
| C7 AES-128 mask | per packet | sub-µs | quinn::crypto::HeaderKey (deferred to C10) | **497 ns** (aarch64; AES-NI on x86_64 should improve) |
| C7 ChaCha20 mask | per packet | sub-µs | n/a | **115 ns** (aarch64 SIMD) |
| C7 apply_mask | per packet | sub-10 ns | n/a (pure XOR) | **4.08 ns** |
| C8 CidQueue insert × 4 + find_by_sequence | typical multi-CID lookup | sub-100 ns | quinn::cid_queue (pub(crate); deferred to C10) | **8.69 ns** |
| C9 SendSpace::assign | per packet sent | sub-10 ns | quinn::packet_builder (pub(crate); deferred to C10) | **322 ps** |
| C9 RecvSpace::record_received | per packet received | sub-50 ns | n/a | **9.84 ns** (in-order; bitmap shift dominates) |
| C9 encode_packet_number | per packet sent | sub-5 ns | RFC 9000 §A.2 algorithm | **1.14 ns** |
| C9 decode_packet_number | per packet received | sub-5 ns | RFC 9000 §A.3 algorithm | **1.19 ns** (bit-exact match on §A.3 fixture) |

### State machine (C10–C18)

| Component | Scope | Target | Incumbent ref | Measured |
|-----------|-------|-------:|--------------:|----------|
| C10 protect_initial (1200B) | per packet | clear 55 MB/s gate | quinn::Connection::handle_packet (deferred to C11) | **8.78 µs / ~136 MB/s** (AES-128-GCM + AES-128 HP) |
| C10 protect_initial (1452B) | per packet | clear 55 MB/s gate | n/a | **10.2 µs / ~143 MB/s** |
| C10 unprotect_initial (1200B) | per packet | clear 55 MB/s gate | n/a | **8.73 µs / ~138 MB/s** |
| C10 full TLS 1.3 sans-IO (Session trait + handshake driver) | full handshake | deferred to C11 | quinn::Connection | scope re-cut; see discipline.md C10 row |
| C11 connection state machine | handshake → established | TBD ns/op + max ms | TBD | pending |
| C12 streams | 1000 concurrent bidi | TBD ops/s | TBD | pending |
| C13 ACK generation | high-loss reorder | TBD ns/op | TBD | pending |
| C14 loss detection | 1% loss + reorder | TBD events/s | TBD | pending |
| C15 NewReno | slow-start → CA → recovery | full ack-process round | TBD | pending |
| C16 CUBIC | sustained 100ms RTT | TBD | TBD | pending |
| C17 BBRv2 | high-BDP bursty loss | TBD | TBD (vs s2n-quic-core) | pending |
| C18 ECN | ECN-CE marked path | TBD | TBD | pending |

### Validation, migration, multipath (C19–C27)

| Component | Scope | Target | Incumbent ref | Measured |
|-----------|-------|-------:|--------------:|----------|
| C19 address validation | spoof-flood Retry path | TBD | TBD | pending |
| C20 anti-amplification | unvalidated peer | TBD | TBD | pending |
| C21 path migration | NAT rebind under load | TBD ms validation | TBD | pending |
| C22 version negotiation | unknown-version probe | TBD ns/op | TBD | pending |
| C23 key update | KEY_UPDATE in-flight | TBD ms switchover | TBD | pending |
| C24 0-RTT | resumption → early data | TBD ms RTT | TBD | pending |
| C25 RFC 9221 DATAGRAM | reliable + unreliable mix | TBD ops/s | TBD | pending |
| C26 multipath | dual-path active/active | TBD aggregate MB/s | TBD (vs s2n-quic-core / picoquic) | pending |
| C27 endpoint demux | 10k concurrent connections | TBD ops/s | TBD | pending |

### I/O integration (C28–C31)

| Component | Scope | Target | Incumbent ref | Measured |
|-----------|-------|-------:|--------------:|----------|
| C28 UDP datagram source | 1 Gbps loopback per connection | TBD GSO batch / pkt rate | quinn-udp + tokio::net::UdpSocket | pending |
| C29 Future facade | bulk stream + multi-stream interleave | TBD MB/s | quinn::Connection | pending |
| C30 config + parity | env / builder / TOML equivalence | identical state | n/a | pending |
| C31 tokio-compat | tokio-side throughput parity | TBD MB/s | quinn (tokio-driven) | pending |

### H3 proto (C32–C38)

| Component | Scope | Target | Incumbent ref | Measured |
|-----------|-------|-------:|--------------:|----------|
| C32 H3 frame codec | full frame mix | TBD ns/op | `h3::proto::frame` | pending |
| C33 QPACK encoder | typical HTTP header set | TBD ns/op | `h3::qpack::Encoder` | pending |
| C34 QPACK decoder | typical decode under blocked limit | TBD ns/op | `h3::qpack::Decoder` | pending |
| C35 H3 server | concurrent request fan-in | TBD req/s | `h3::server::Connection` | pending |
| C36 H3 client | concurrent request fan-out | TBD req/s | `h3::client::Connection` | pending |
| C37 H3-Datagrams | mixed CONNECT-UDP + DATA + DATAGRAM | TBD ops/s | h3 limited | pending |
| C38 extended CONNECT | CONNECT round-trip | TBD ns/op | `h3::ext` (partial) | pending |

### H3 I/O (C39–C41)

| Component | Scope | Target | Incumbent ref | Measured |
|-----------|-------|-------:|--------------:|----------|
| C39 Server / Client facade | full GET / POST loopback | TBD req/s | `h3::server::Connection` driven by h3-quinn | pending |
| C40 config + parity | env / builder / TOML | identical | n/a | pending |
| C41 cutover | `tests/listener_h3.rs` + `benches/bench_h3_upstream.rs` | green on native | h3-quinn baseline kept until cutover | pending |
