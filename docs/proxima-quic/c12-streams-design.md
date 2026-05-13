# C12 — Stream multiplexing + flow control (paper proof)

Per [RFC 9000 §2] + §3 (stream lifecycle) + §4 (flow control) +
§19.8 (STREAM frame) + §19.9–§19.14 (flow-control frames).

Composes:

- C3 — `Frame::Stream` (parse + encode already done).
- C9 — packet-number space for in-flight STREAM frame retransmit
  bookkeeping (not in v1; defer to C12.1).
- C13 — AckScheduler (no direct hook; STREAM-frame inclusion already
  marks the packet ack-eliciting).
- C14 — LossDetection (STREAM-frame data in a lost packet must be
  retransmitted; defer to C12.1).
- C15 — CongestionController (send_budget gates STREAM-frame
  emission).

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## Scope choices

C12 is a multi-week algorithm if everything is done at once. Scope
cut for **C12 v1**:

| Feature | Status |
|---|---|
| StreamId newtype + direction/initiator helpers | YES |
| Per-stream Send state (Ready/Send/DataSent/DataRecvd) | YES |
| Per-stream Recv state (Recv/SizeKnown/DataRecvd/DataRead) | YES |
| StreamTable\<MAX_BIDI, MAX_UNI\> with `heapless::IndexMap` | YES |
| Connection-level + stream-level flow-control credits | YES |
| STREAM frame parse → enqueue into recv buffer | YES |
| STREAM frame encode in `poll_transmit_established` | YES |
| `open_stream(bidi)` + `send_application(id, bytes)` | YES |
| MAX_DATA / MAX_STREAM_DATA frame handling on inbound | YES |
| STREAM_DATA_BLOCKED + DATA_BLOCKED on outbound (when stuck) | YES |
| `RESET_STREAM` / `STOP_SENDING` frame handling | **C12.1** |
| `MAX_STREAMS` / `STREAMS_BLOCKED` negotiation | **C12.1** |
| Stream-data retransmit on loss | **C12.1** |
| 0-RTT stream resumption | **C24** |
| Server-side open_stream | **C12.1** (client-side first) |

The v1 scope is enough to ship a client that can send AND receive
data on a single bidi stream over a 1-RTT-protected connection. That
exercises the full FSM lifecycle through the Established state. The
deferred features land in C12.1 + C12.2 incrementally.

## Types

```rust
/// RFC 9000 §2.1 — 62-bit stream ID. Low 2 bits encode direction +
/// initiator.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u64);

impl StreamId {
    pub const fn direction(self) -> StreamDirection {
        if self.0 & 0x2 == 0 { StreamDirection::Bidi } else { StreamDirection::Uni }
    }
    pub const fn initiator(self) -> Side {
        if self.0 & 0x1 == 0 { Side::Client } else { Side::Server }
    }
    pub const fn next_local(prev: Option<Self>, side: Side, dir: StreamDirection) -> Self {
        let base = match (side, dir) {
            (Side::Client, StreamDirection::Bidi) => 0,
            (Side::Server, StreamDirection::Bidi) => 1,
            (Side::Client, StreamDirection::Uni)  => 2,
            (Side::Server, StreamDirection::Uni)  => 3,
        };
        match prev {
            Some(StreamId(p)) => StreamId(p + 4),
            None => StreamId(base),
        }
    }
}
```

## Per-stream Send state

```
pub enum SendState {
    /// Ready — peer signalled it can receive; no data sent yet.
    Ready,
    /// Send — bytes are queued or in flight.
    Send {
        send_buffer: ArrayVec<u8, STREAM_SEND_INLINE>,
        offset_next: u64,       // next byte we'd put into the queue
        offset_acked: u64,      // bytes peer has ack'd
    },
    /// DataSent — caller called close_send; the FIN bit is queued.
    DataSent { offset_final: u64, offset_acked: u64 },
    /// DataRecvd — peer ack'd all bytes including FIN. Terminal.
    DataRecvd { offset_final: u64 },
    /// ResetSent — caller called reset_send. Terminal. (C12.1)
    ResetSent,
}
```

## Per-stream Recv state

```
pub enum RecvState {
    /// Recv — receiving STREAM bytes, no FIN seen.
    Recv {
        recv_buffer: ArrayVec<u8, STREAM_RECV_INLINE>,
        offset_next: u64,       // next contiguous offset expected
    },
    /// SizeKnown — FIN seen; collecting any trailing gaps.
    SizeKnown { recv_buffer: ArrayVec<u8, STREAM_RECV_INLINE>, offset_final: u64 },
    /// DataRecvd — all data through offset_final present.
    DataRecvd { offset_final: u64 },
    /// DataRead — caller drained the recv buffer. Terminal.
    DataRead { offset_final: u64 },
}
```

## Flow control state

```
pub struct ConnectionFlowControl {
    /// Bytes peer has authorised us to send across all streams (MAX_DATA).
    pub credit_send: u64,
    /// Bytes we've authorised peer (sent in MAX_DATA frames).
    pub credit_recv: u64,
    /// Bytes we've actually sent (counts against credit_send).
    pub sent_offset: u64,
    /// Bytes we've consumed via recv (counts toward credit_recv expansion).
    pub recv_offset: u64,
}

pub struct StreamFlowControl {
    pub credit_send: u64,    // MAX_STREAM_DATA peer sent us
    pub credit_recv: u64,    // we'll grant via MAX_STREAM_DATA
    pub sent_offset: u64,
    pub recv_offset: u64,
}
```

## StreamTable

```
pub struct StreamTable<const MAX_BIDI: usize, const MAX_UNI: usize> {
    bidi: heapless::IndexMap<StreamId, Stream, MAX_BIDI>,
    uni:  heapless::IndexMap<StreamId, Stream, MAX_UNI>,
    next_local_bidi: Option<StreamId>,
    next_local_uni:  Option<StreamId>,
}

pub struct Stream {
    pub id: StreamId,
    pub send: SendState,
    pub recv: RecvState,
    pub flow: StreamFlowControl,
}
```

`heapless::IndexMap` is tier-3-friendly. Caps come from `prime-runtime.toml`
`[quic].max_concurrent_bidi_streams = 1024` and `max_concurrent_uni_streams = 1024`.

## API on Connection

```
fn open_stream(&mut self, bidi: bool) -> Result<StreamId, ConnectionError>;
fn send_application(&mut self, id: StreamId, data: &[u8]) -> Result<usize, ConnectionError>;
fn read_stream(&mut self, id: StreamId, out: &mut [u8]) -> Result<usize, ConnectionError>;
fn close_send(&mut self, id: StreamId) -> Result<(), ConnectionError>;
fn poll_stream_event(&mut self) -> Option<StreamEvent>;
```

`StreamEvent`:
- `StreamReadable { id }` — recv buffer has data.
- `StreamWritable { id }` — send buffer has capacity.
- `StreamFinished { id }` — peer sent FIN AND we've drained the buffer.
- `StreamReset { id, error_code }` — peer reset; data dropped. (C12.1)

## Worked example: client opens bidi stream, sends "hello", recvs "world", closes

### State at t=Established

- side: Client
- `streams = StreamTable::new()`
- `connection_flow_control.credit_send = peer_TP.initial_max_data` (assume 1 MB)
- `connection_flow_control.credit_recv = local_TP.initial_max_data` (1 MB)

### Step 1: `open_stream(bidi=true)`

- `next_local_bidi == None` → first client-initiated bidi → `StreamId(0)`.
- Insert into `bidi` table:
  - `send = Ready`
  - `recv = Recv { recv_buffer: empty, offset_next: 0 }`
  - `flow = { credit_send: peer_TP.initial_max_stream_data_bidi_remote, credit_recv: local_TP.initial_max_stream_data_bidi_local, ... }`
- Return `StreamId(0)`.

### Step 2: `send_application(StreamId(0), b"hello")`

- Look up stream `0`.
- Transition send `Ready → Send { send_buffer: b"hello", offset_next: 5, offset_acked: 0 }`.
- Connection FC: `sent_offset` not yet incremented (only on actual transmit).
- Return `5` (bytes accepted into buffer).

### Step 3: `poll_transmit_established(now, buf)`

- send_budget from controller is e.g. 12000.
- For each stream with pending data: emit a STREAM frame.
- Stream 0: `STREAM(stream_id=0, offset=0, data=b"hello", fin=false)`.
- Update `send_buffer.drain(0..5)`, `offset_acked` stays (acks come on inbound).
- Increment `connection_flow_control.sent_offset += 5`.
- Return a 1-RTT-protected datagram.

### Step 4: `handle_datagram` — server replies with `STREAM(0, 0, b"world")`

- Parse STREAM frame.
- Look up stream `0`; append to recv buffer: `[w, o, r, l, d]`, `offset_next = 5`.
- Emit `StreamEvent::StreamReadable { id: StreamId(0) }`.

### Step 5: `read_stream(StreamId(0), &mut buf)`

- Drain recv buffer; return `5`.

### Step 6: `close_send(StreamId(0))`

- Transition send `Send → DataSent { offset_final: 5, offset_acked: 0 }`.

### Step 7: next `poll_transmit_established` emits STREAM with FIN bit

- `STREAM(stream_id=0, offset=5, data=b"", fin=true)`.

### Step 8: peer ACKs PNs covering the STREAM frames

- Update `offset_acked = 5`.
- `DataSent → DataRecvd { offset_final: 5 }` (terminal).
- Stream slot can be retired.

## Code site

- `proxima-quic-proto/src/streams/mod.rs` — re-exports.
- `proxima-quic-proto/src/streams/id.rs` — `StreamId` + `StreamDirection` + helpers.
- `proxima-quic-proto/src/streams/send_state.rs` — `SendState` enum.
- `proxima-quic-proto/src/streams/recv_state.rs` — `RecvState` enum.
- `proxima-quic-proto/src/streams/flow.rs` — `ConnectionFlowControl` + `StreamFlowControl`.
- `proxima-quic-proto/src/streams/table.rs` — `StreamTable<MAX_BIDI, MAX_UNI>` + `Stream`.
- `proxima-quic-proto/src/streams/event.rs` — `StreamEvent` enum.
- `proxima-quic-proto/src/connection/state.rs` — replace `stub::StreamTable` with the real `streams::StreamTable<…>`.
- `proxima-quic-proto/src/connection/mod.rs` — replace `NotImplemented` stubs in `open_stream` / `send_application` / `initiate_key_update`-adjacent paths.

## Tier

`streams::*` modules target tier-3. `heapless::IndexMap` is no_std + no_alloc.
The inline `STREAM_SEND_INLINE` + `STREAM_RECV_INLINE` capacities are tunable per profile;
default 16 KiB per buffer per stream. (Could be much larger via build.rs constant.)

## Self-critique

- **Pass 1 — paper before code**: yes; this doc precedes any C12 code.
- **Pass 2 — algorithm walk produces exact expected output**: yes;
  the 8-step worked example traces stream creation through close + final
  ACK; each step has a state assertion.
- **Pass 3 — code maps step-by-step to algorithm**: deferred.
- **Pass 4 — test uses exact inputs from worked example**: planned —
  `client_opens_bidi_writes_reads_closes`.
- **Pass 5 — would the test fail on bugs**: yes; swapping stream-id
  base (0 vs 1 for client-bidi), forgetting to increment connection FC
  sent_offset, transitioning Send → DataRecvd without going through
  DataSent would all break specific assertions.
- **Pass 6 — paper linked to test**: yes.

## C12 v1 deferred follow-ons

- **C12.1** — RESET_STREAM + STOP_SENDING + MAX_STREAMS + retransmit-on-loss + server-side open_stream.
- **C12.2** — Per-stream backpressure on caller's `send_application` when buffer fills.
- **C24** — 0-RTT stream resumption.

## Implementation effort estimate

Multi-slice landing similar to C11:
- C12.0 — types (StreamId, SendState, RecvState, flow control) — ~200 LOC + 20 tests.
- C12.1 — StreamTable<MAX_BIDI, MAX_UNI> — ~300 LOC + 15 tests.
- C12.2 — Connection FSM wire-up (open_stream + send_application + read_stream + parse STREAM + emit STREAM) — ~500 LOC.
- C12.3 — Lifecycle test walking the 8-step worked example end to end.
- C12.4 — Bench + discipline log + journal + /doc note.

Each slice is a separate commit. Total ≈ 1000-1500 LOC + 60-80 tests.
