# C25 — RFC 9221 unreliable DATAGRAM extension (paper proof)

Per [RFC 9221]. QUIC's unreliable-datagram extension: app-layer data
that bypasses streams' ordering + retransmit, riding directly on QUIC's
packet protection + congestion control.

[RFC 9221]: https://www.rfc-editor.org/rfc/rfc9221

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## Scope

**C25 v1**:
- Per-direction bounded send + recv queues. Bounded by
  `proxima-quic-proto.toml [datagram].{send_queue_cap, recv_queue_cap}`
  per principle 12 (RFC 9221 doesn't mandate caps; quinn / msquic use
  similar bounded queues).
- Transport-parameter negotiation: `max_datagram_frame_size` (RFC 9221
  §3) — local advertises; peer-TP parse stash + boolean enabled flag.
- `Connection::send_datagram(payload)` queues outbound; returns
  `DatagramSendError::QueueFull` / `TooLarge` / `NotEnabled`.
- `Connection::recv_datagram() -> Option<Vec<u8>>` drains inbound.
- The C3 `Frame::Datagram` parse path (which is already done!) plumbs
  inbound payloads into the recv queue.
- `poll_transmit_established` opportunistically appends DATAGRAM frames
  from the send queue when budget allows.

**C25 deferred to follow-on**:
- 0-RTT datagrams (defer to C24).
- Multiple datagram frames per packet — needs frame-fitting algorithm.
- Datagram-level pacing (separate from stream pacing). RFC 9221 says
  congestion control treats DATAGRAMs the same as STREAM bytes.

## Per principle 14 (incumbent wins on correctness)

Wire format per RFC 9221 §4 — already done in C3. Bound caps + the
enable-bit semantics per RFC 9221 §3 — that's the new code.

Cross-reference for behavior parity: quinn-proto's `Datagrams` type
+ `send_datagram` API. We don't have an exact-output incumbent test
for C25 (DATAGRAMs are opaque app-layer payloads), but the BEHAVIOUR
must match:

- `max_datagram_frame_size = 0` (not advertised) → peer cannot send;
  our outbound send returns `NotEnabled`.
- `max_datagram_frame_size > 0` → both sides can send; payloads up to
  the advertised limit are accepted.
- Queue full → caller-visible error; we drop nothing silently.

## Worked example

State at t=Established: `flow_control` initial; `send_queue.len = 0`;
`recv_queue.len = 0`; peer advertised `max_datagram_frame_size = 1200`.

| Event | Action | State after |
|---|---|---|
| `send_datagram(b"hello")` | enqueue payload | `send_queue = [b"hello"]` |
| `send_datagram(b"world")` | enqueue | `send_queue = [b"hello", b"world"]` |
| `poll_transmit_established` | drain queue, emit DATAGRAM frames | `send_queue = []` |
| inbound 1-RTT packet w/ `Frame::Datagram(b"resp")` | enqueue into recv | `recv_queue = [b"resp"]` |
| `recv_datagram()` | pop front | returns `Some(b"resp")`; `recv_queue = []` |
| `recv_datagram()` | nothing pending | returns `None` |
| `send_datagram(b"x" * 9999)` w/ max=1200 | reject — too large | `Err(TooLarge { max: 1200 })` |
| `send_datagram(b"y")` w/ queue full | reject — queue full | `Err(QueueFull { cap: N })` |

## Code site

- `proxima-quic-proto/src/datagram.rs` — `DatagramSendError` +
  `DatagramConfig` + per-direction queues.
- `proxima-quic-proto/src/connection/state.rs` — add
  `datagram_send_queue` / `datagram_recv_queue` / `datagram_enabled`
  fields to EstablishedState.
- `proxima-quic-proto/src/connection/mod.rs` — `send_datagram` /
  `recv_datagram` methods on Connection.
- `proxima-quic-proto/src/connection/mod.rs::parse_and_apply_established`
  — when this exists (currently NotImplemented; arrives with 1-RTT
  ingress), handle `Frame::Datagram` → push to recv queue.
- For C25 v1 we add the queue + the API but the actual 1-RTT wire
  round-trip defers to the same future component that handles
  general 1-RTT ingress.

## Sizing (principle 12)

```toml
[datagram]
# Per-direction queue caps (RFC 9221 doesn't mandate; matches quinn).
send_queue_cap = 256
recv_queue_cap = 256
# Local-advertised max_datagram_frame_size in transport parameters.
# 0 disables DATAGRAM advertisement entirely.
local_max_datagram_frame_size = 1200
```

## Tier

The `datagram` queue types are tier-1 (alloc; bounded `heapless::Deque`
with `Vec<u8>` payloads — alloc per-payload because sizes vary).
The wire-format parse/encode (Frame::Datagram in C3) is already tier-3.

## Self-critique

- **Pass 1 — paper before code**: yes.
- **Pass 2 — algorithm walk produces exact expected output**: yes;
  the 8-row worked example covers send + recv + size-reject + queue-
  full-reject.
- **Pass 3 — code maps step-by-step to algorithm**: deferred.
- **Pass 4 — test uses exact inputs from worked example**: planned.
- **Pass 5 — would the test fail on bugs**: yes; forgetting the
  `NotEnabled` check (would silently send DATAGRAMs the peer can't
  decode), off-by-one in queue-full detection, or wrong size-limit
  comparison would all break specific assertions.
- **Pass 6 — paper linked to test**: yes.
