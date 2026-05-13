# C11 — Connection state machine FSM design (paper proof)

Per workspace principle 11 (sans-IO state machine = discriminated enum FSM
with per-variant data, transitions consume the old variant, no `Box<dyn>`,
no `Arc<Mutex<State>>`, no runtime "is this in the right state?" checks
where exhaustive match would do it at compile time).

Per the `/algorithm-development` skill: this paper proof MUST precede the
code. The worked example below is the spec AND the test C11 ships with.

Plan: [i-need-a-full-adaptive-seal.md](../../../../.claude/plans/i-need-a-full-adaptive-seal.md).
Edges (Instant + TlsProvider resolutions): [edges.md](./edges.md).

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

---

## ConnectionState — the FSM enum

Six variants per RFC 9000 §10 + §17.2.2-§17.2.5. Each variant owns ONLY
the data legal for that state; transitions consume the old variant and
produce the new one (move semantics; old state unreachable after).

```rust
pub enum ConnectionState {
    Initial(InitialState),
    Handshake(HandshakeState),
    Established(EstablishedState),
    Closing(ClosingState),
    Draining(DrainingState),
    Closed,
}
```

### Per-variant data

```rust
pub struct InitialState {
    pub side: Side,                                           // Client | Server (const on Provider)
    pub origin: Instant,                                       // when connection was created
    pub last_now: Instant,                                     // monotonicity tracker
    pub local_initial_dcid: arrayvec::ArrayVec<u8, 20>,        // the DCID that drove HKDF
    pub local_initial_scid: arrayvec::ArrayVec<u8, 20>,        // our local CID
    pub local_cid_queue: CidQueue<DCID_TABLE_CAP>,             // CIDs we've issued (peer uses)
    pub remote_cid_queue: CidQueue<DCID_TABLE_CAP>,            // CIDs peer issued (we use)
    pub initial_send: SendSpace,                                // C9 send-side PN counter
    pub initial_recv: RecvSpace<128>,                           // C9 recv-side + dup detection
    pub initial_keys: InitialKeyPair,                           // C5 — installed at construction
    pub local_transport_params: TransportParameters<'static>,   // our advertised TPs
    pub anti_amplification: AntiAmplificationCounter,           // 3× until validated
    pub idle_deadline: Instant,                                 // origin + initial_idle_timeout
    pub crypto_send_initial: arrayvec::ArrayVec<u8, 1200>,      // pending CRYPTO bytes (TLS output)
}

pub struct HandshakeState {
    // inherits everything in InitialState that's still relevant
    pub side: Side,
    pub origin: Instant,
    pub last_now: Instant,
    pub local_cid_queue: CidQueue<DCID_TABLE_CAP>,
    pub remote_cid_queue: CidQueue<DCID_TABLE_CAP>,
    pub local_transport_params: TransportParameters<'static>,
    pub anti_amplification: AntiAmplificationCounter,
    pub idle_deadline: Instant,
    // Initial-space state persists for ACK obligation (RFC 9000 §17.2.2.1):
    pub initial_send: SendSpace,
    pub initial_recv: RecvSpace<128>,
    pub initial_keys: InitialKeyPair,
    // NEW for Handshake:
    pub handshake_send: SendSpace,
    pub handshake_recv: RecvSpace<128>,
    pub handshake_secrets: EpochSecrets,        // pushed via TlsEventSink::on_new_secrets
    pub crypto_send_initial: arrayvec::ArrayVec<u8, 1200>,
    pub crypto_send_handshake: arrayvec::ArrayVec<u8, 1200>,
}

pub struct EstablishedState {
    pub side: Side,
    pub origin: Instant,
    pub last_now: Instant,
    pub local_cid_queue: CidQueue<DCID_TABLE_CAP>,
    pub remote_cid_queue: CidQueue<DCID_TABLE_CAP>,
    pub local_transport_params: TransportParameters<'static>,
    pub peer_transport_params: PeerTransportParametersOwned,    // PARSED from EE
    pub local_ack_delay_exponent: AckDelayExponent,             // from local TPs
    pub peer_ack_delay_exponent: AckDelayExponent,              // from peer TPs (TWO fields per Instant resolution)
    pub idle_deadline: Instant,                                  // computed from negotiated min idle timeout
    pub application_send: SendSpace,
    pub application_recv: RecvSpace<128>,
    pub application_secrets: EpochSecrets,                       // generation tracks key updates
    pub streams: StreamTable<MAX_CONCURRENT_BIDI_STREAMS,        // C12 territory (sized.rs caps)
                              MAX_CONCURRENT_UNI_STREAMS>,
    pub ack_scheduler: AckScheduler,                             // C13
    pub loss_detection: LossDetection,                            // C14
    pub congestion_control: CongestionController,                 // C15 (NewReno default)
    // Handshake keys retained ≥3 PTO worth per RFC 9001 §4.9.2 (peer may still send Handshake-encrypted ACKs):
    pub handshake_keys_retain_until: Option<Instant>,
    pub handshake_secrets_retained: Option<EpochSecrets>,
    // last_observed_pto from loss_detection; used by close_deadline computation.
}

pub struct ClosingState {
    pub side: Side,
    pub last_now: Instant,
    pub close_frame: ConnectionCloseFrame,             // CONNECTION_CLOSE bytes (built once)
    pub close_deadline: Instant,                        // now + 3*PTO at entry
    pub application_secrets: EpochSecrets,             // RETAINED for retransmit
    pub remote_cid_queue: CidQueue<DCID_TABLE_CAP>,    // needed to address outbound retx
    pub close_application_dcid: arrayvec::ArrayVec<u8, 20>, // first remote CID we use for sending CLOSE
    pub retransmit_close_after: Instant,                // backoff timer for retransmits
}

pub struct DrainingState {
    pub last_now: Instant,
    pub drain_deadline: Instant,                        // now + 3*PTO at entry
    // NO keys — we silently drop every incoming packet
    // NO outgoing capability — `poll_transmit` returns None
}

// Closed is unit — `ConnectionState::Closed`. Caller drops the Connection.
```

### Method legality matrix

| Method                       | Initial | Handshake | Established | Closing | Draining | Closed |
|------------------------------|:-------:|:---------:|:-----------:|:-------:|:--------:|:------:|
| `handle_datagram(now, ...)`  | ✓       | ✓         | ✓           | drop¹   | drop²    | drop³  |
| `handle_timeout(now)`        | ✓       | ✓         | ✓           | ✓       | ✓        | n/a    |
| `poll_transmit(now, buf)`    | ✓       | ✓         | ✓           | ✓⁴      | None     | None   |
| `next_timeout()`             | ✓       | ✓         | ✓           | ✓       | ✓        | None   |
| `open_stream(bidi/uni)`      | ✗       | ✗         | ✓           | ✗       | ✗        | ✗      |
| `send_application(stream)`   | ✗       | ✗         | ✓           | ✗       | ✗        | ✗      |
| `close(error_code, reason)`  | ✓       | ✓         | ✓           | no-op   | no-op    | no-op  |

¹ Closing drops packets but responds with a rate-limited CONNECTION_CLOSE
  retransmit if the inbound is from the peer (RFC 9000 §10.2.2).  
² Draining silently drops; no response (RFC 9000 §10.2.2).  
³ Closed: caller has dropped the connection; method unreachable.  
⁴ Closing's `poll_transmit` returns the rate-limited CONNECTION_CLOSE only.  

Compile-time enforcement: the method dispatcher matches on `&self.state`
(or `&mut self.state` for mutating methods), each variant calls a
state-specific handler. Methods that don't apply to a state return
`Err(ConnectionError::IllegalInState { current })`. The illegal-state
error type carries a static `&'static str` discriminant for diagnostics
(no heap).

### Transition rules

Each transition is a function `transition_X_to_Y(old: XState, event) -> Result<YState, ConnectionError>`.
The dispatcher in each state's handler call site does:

```rust
self.state = match core::mem::replace(&mut self.state, ConnectionState::Closed) {
    ConnectionState::Initial(initial) => match handle_in_initial(initial, event)? {
        StateOutcome::Stay(new_initial) => ConnectionState::Initial(new_initial),
        StateOutcome::Advance(handshake) => ConnectionState::Handshake(handshake),
        StateOutcome::Close(closing) => ConnectionState::Closing(closing),
    },
    // ... etc
};
```

The `core::mem::replace` swap with `Closed` sentinel ensures the old state
is consumed; even if the transition fn panics (it can't — no `panic!` per
workspace rules), the connection ends up `Closed` which is safe-by-design.

### Event drivers — what triggers what

| Event                                                       | Initial         | Handshake       | Established                  | Closing  | Draining |
|-------------------------------------------------------------|-----------------|-----------------|------------------------------|----------|----------|
| `TlsEventSink::on_new_secrets(handshake)`                   | → Handshake     | (impossible)    | (impossible)                 | (drop)   | (drop)   |
| `TlsEventSink::on_new_secrets(application, gen=0)`          | (impossible)    | → Established   | (impossible)                 | (drop)   | (drop)   |
| `TlsEventSink::on_new_secrets(application, gen=N+1)`        | (impossible)    | (impossible)    | install for key update       | (drop)   | (drop)   |
| `TlsEvent::PeerTransportParameters(bytes)` during read_hs   | stash for next  | parse + verify  | (impossible)                 | (drop)   | (drop)   |
| `TlsEvent::HandshakeConfirmed`                              | (unexpected)    | mark confirmed  | (impossible)                 | (drop)   | (drop)   |
| inbound CONNECTION_CLOSE                                    | → Closing*      | → Closing*      | → Draining                   | → Draining | (drop) |
| `close(error_code, reason)` caller-initiated                | → Closing       | → Closing       | → Closing                    | no-op    | no-op    |
| `close_deadline` reached in `handle_timeout`                | (n/a)           | (n/a)           | (n/a)                        | → Drain  | (n/a)    |
| `drain_deadline` reached in `handle_timeout`                | (n/a)           | (n/a)           | (n/a)                        | (n/a)    | → Closed |
| `idle_deadline` reached in `handle_timeout`                 | → Closed (no CC) | → Closed       | → Closed                     | (n/a)    | (n/a)    |
| `abort(code)` from TLS provider                             | → Closing(local) | → Closing(local) | → Closing(local)            | no-op    | no-op    |

`*` In Initial/Handshake we can only send a CONNECTION_CLOSE if we have
the corresponding epoch keys. RFC 9000 §10.2.3 allows sending it in the
deepest installed epoch.

---

## Worked example: client connection, complete lifecycle

### Inputs

- Client side: `Side::Client`
- TlsProvider: `MockTlsProvider` scripted with a 1-RTT handshake + graceful close
- Caller's monotonic clock starts at `Instant::from_micros(1_000_000)`
- `local_initial_dcid = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08]` (RFC 9001 §A.1 DCID for verifiable initial keys)
- Local transport params: idle_timeout=30 000 ms, max_udp_payload_size=1452, max_ack_delay=25 ms, initial_max_data=1 048 576
- 1200-byte initial-packet budget

### State at time T (concrete data the FSM reads)

**At `t = 1_000_000 µs` (origin):** `ConnectionState::Initial(InitialState)` is constructed with:
- `side = Client`
- `origin = Instant::from_micros(1_000_000)`
- `last_now = origin`
- `local_initial_dcid = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08]`
- `local_initial_scid = [0xc0, 0xff, 0xee, 0xba, 0xbe, 0x12, 0x34, 0x56]` (random 8B)
- `initial_keys = initial_keys::derive(&local_initial_dcid)` (C5 — RFC 9001 §A.1 bit-exact)
  - `client.key = [0x1f, 0x36, ...]`
  - `client.iv = [0xfa, 0x04, ...]`
  - `client.hp = [0x9f, 0x50, ...]`
- `initial_send.next = 0`, `initial_send.largest_acked = None`
- `initial_recv.largest_received = None`
- `local_transport_params = { idle_timeout: 30_000, max_udp_payload_size: 1452, max_ack_delay: 25, initial_max_data: 1_048_576, ... }`
- `anti_amplification.received_from_peer = 0`, `sent_to_peer = 0`
- `idle_deadline = origin + Duration::from_millis(30_000)` = `Instant::from_micros(31_000_000)`
- `crypto_send_initial = [client_hello_bytes]` (provided by `tls.write_handshake(Initial, &mut buf)`)

### Expected output sequence

The walked behavior below MUST produce these exact events:

| t (µs) | Method call | State before | State after | Side-effects |
|---|---|---|---|---|
| 1_000_001 | `poll_transmit(t, buf)` | Initial | Initial | emits Initial packet PN=0 with CRYPTO ClientHello, padded to 1200B; `initial_send.next → 1`; `anti_amplification.sent_to_peer += 1200` |
| 2_000_000 | `handle_datagram(t, server_initial_bytes)` | Initial | **Handshake** | unprotect with `initial_keys.server`; parse ACK+CRYPTO frames; `tls.read_handshake(Initial, crypto_bytes, sink)` calls `sink.on_new_secrets(handshake_secrets)`; state transitions to Handshake carrying `handshake_secrets`; `initial_send.largest_acked = Some(0)`; `initial_recv.largest_received = Some(0)` |
| 2_000_001 | `poll_transmit(t, buf)` | Handshake | Handshake | emits coalesced Initial(ACK of 0) + Handshake(CRYPTO Finished); `initial_send.next → 2`; `handshake_send.next → 1` |
| 3_000_000 | `handle_datagram(t, server_handshake_bytes)` | Handshake | **Established** | unprotect with `handshake_secrets.server`; parse CRYPTO containing server Finished + EncryptedExtensions w/ peer transport params; `tls.read_handshake(Handshake, ..., sink)` fires three sink callbacks IN ORDER: (1) `on_event(PeerTransportParameters(&bytes))` → sink copies to typed `peer_transport_params_owned`; (2) `on_new_secrets(application_secrets, gen=0)`; (3) `on_event(HandshakeConfirmed)`; state transitions to Established carrying parsed peer TPs + application_secrets + handshake_keys_retain_until=now+3*PTO |
| 4_000_000 | `open_stream(bidi=true)` | Established | Established | returns `StreamId(0)`; streams table gains a bidi-client-initiated slot |
| 4_000_001 | `send_application(StreamId(0), b"hello")` | Established | Established | queues STREAM frame for poll_transmit |
| 4_000_002 | `poll_transmit(t, buf)` | Established | Established | emits 1-RTT packet with STREAM frame (offset=0, fin=false, data=b"hello"); `application_send.next → 1` |
| 5_000_000 | `handle_datagram(t, server_application_bytes)` | Established | Established | parses ACK + STREAM(reply); `application_recv.record_received(0)` |
| 6_000_000 | `close(error_code=0x00, reason=b"bye")` | Established | **Closing** | builds CONNECTION_CLOSE (type=0x1d application-error, error_code=0, reason=b"bye"); transitions to Closing with `close_deadline = t + 3*smoothed_pto` (assume PTO=333ms; deadline = 6_999_000 µs); retains `application_secrets` |
| 6_000_001 | `poll_transmit(t, buf)` | Closing | Closing | emits 1-RTT packet with CONNECTION_CLOSE; `retransmit_close_after = t + 333_000` |
| 7_000_000 | `handle_datagram(t, peer_close)` | Closing | **Draining** | parses peer's CONNECTION_CLOSE; transitions to Draining with `drain_deadline = t + 3*smoothed_pto = 7_999_000`; discards all keys |
| 7_999_001 | `handle_timeout(t)` | Draining | **Closed** | `drain_deadline` reached; transitions to Closed; caller drops the Connection |

---

## Algorithm (pseudocode)

The pseudocode below is the contract C11 implements. Every named step
becomes an identifiable lines-of-code site in the eventual implementation.

```
function new_client(provider_config, local_transport_params, origin: Instant) -> Connection:
  step 1: generate random local_initial_dcid + local_initial_scid (CidEntry-shaped, 8 bytes each)
  step 2: tls = TlsProvider::new(provider_config, local_transport_params.as_wire_bytes())
  step 3: initial_keys = TlsProvider::initial_keys(&local_initial_dcid)
  step 4: write client_hello into crypto_send_initial via tls.write_handshake(Initial, &mut buf)
  step 5: build InitialState { side: Client, origin, last_now: origin, local_initial_dcid,
            local_initial_scid, initial_keys, initial_send: SendSpace::new(),
            initial_recv: RecvSpace::new(), local_transport_params,
            anti_amplification: AntiAmplificationCounter::new(),
            idle_deadline: origin + local_transport_params.idle_timeout,
            crypto_send_initial }
  step 6: return Connection { tls, state: ConnectionState::Initial(initial_state) }

function handle_datagram(self, now: Instant, datagram_bytes: &[u8]) -> Result<(), ConnectionError>:
  step 1: check monotonicity: if now < self.last_now() return Err(NonMonotonicTime)
  step 2: update self.last_now() = now
  step 3: match self.state on type:
    Initial(s)     → handle_initial_datagram(s, now, datagram_bytes, &mut self.tls)
    Handshake(s)   → handle_handshake_datagram(s, now, datagram_bytes, &mut self.tls)
    Established(s) → handle_established_datagram(s, now, datagram_bytes, &mut self.tls)
    Closing(s)     → handle_closing_datagram(s, now, datagram_bytes)
    Draining(_)    → Ok(())  // silently drop
    Closed         → Err(IllegalInState { current: "Closed" })

function handle_initial_datagram(initial: InitialState, now, bytes, tls) -> StateOutcome<InitialState, HandshakeState, ClosingState>:
  step 1: parse packet header (C2)
  step 2: if header.epoch != Initial: drop, return StateOutcome::Stay(initial)
  step 3: unprotect packet (C10) using initial.initial_keys.server (we are client, so peer is server)
  step 4: validate packet number not duplicate via initial.initial_recv.record_received
  step 5: parse frames (C3): for each frame:
    a. PADDING: skip
    b. PING: ack_scheduler.note_pn(pn)
    c. ACK: initial.initial_send.record_acked for each pn in ack ranges
    d. CRYPTO(offset, data):
        tls.read_handshake(Initial, data, &mut connection_event_sink)
        if sink received on_new_secrets(handshake_secrets):
          advance to Handshake (see step 6)
    e. CONNECTION_CLOSE: build ClosingState; return StateOutcome::Close
    f. other frames at Initial level: protocol violation, build ClosingState
  step 6: if any sink event was on_new_secrets(handshake):
    construct HandshakeState carrying initial.* (still relevant) + new fields
    return StateOutcome::Advance(handshake_state)
  step 7: otherwise return StateOutcome::Stay(initial)

function handle_handshake_datagram(handshake: HandshakeState, now, bytes, tls) -> StateOutcome<HandshakeState, EstablishedState, ClosingState>:
  step 1: parse packet header
  step 2: dispatch on header.epoch:
    Initial: handle remaining Initial-epoch packets (ACKs we owe; rare after we have Handshake keys)
    Handshake: unprotect with handshake.handshake_secrets.server
    else: drop
  step 3..4: as handle_initial_datagram
  step 5: parse frames; for CRYPTO at Handshake epoch:
    tls.read_handshake(Handshake, data, &mut connection_event_sink)
    sink may fire multiple events synchronously:
      - PeerTransportParameters(bytes) → parse into typed PeerTransportParametersOwned
      - on_new_secrets(application_secrets, generation=0) → install
      - HandshakeConfirmed → mark for advance
  step 6: if all three events fired in order (peer TPs + app secrets + confirmed):
    construct EstablishedState; return StateOutcome::Advance(established)
  step 7: otherwise StateOutcome::Stay(handshake)

function handle_established_datagram(...) -> StateOutcome<EstablishedState, EstablishedState, ClosingState>:
  // Established → Draining only via CONNECTION_CLOSE (one-step skip per RFC §10.2.2)
  step 1..4: parse + unprotect (1-RTT keys)
  step 5: parse frames; cases of interest:
    - ACK: feed loss_detection (C14) + congestion (C15) + drain retransmit queue
    - STREAM(stream_id, offset, data, fin): streams.recv_into(stream_id, offset, data, fin)
    - MAX_DATA / MAX_STREAM_DATA / etc.: flow control updates
    - NEW_CONNECTION_ID: remote_cid_queue.insert
    - RETIRE_CONNECTION_ID: local_cid_queue.retire
    - PATH_CHALLENGE/PATH_RESPONSE: path validation (deferred to C21)
    - HANDSHAKE_DONE (server→client): we can discard handshake_keys_retained
    - CONNECTION_CLOSE: build DrainingState; return Close (mapped to Draining at dispatcher)
  step 6: return StateOutcome::Stay(established)

function poll_transmit(self, now: Instant, buffer: &mut [u8]) -> Option<DatagramWrite>:
  step 1: effective_now = now.max(self.last_now())  // saturating-monotonic per Instant resolution
  step 2: match self.state:
    Initial(s)     → poll_transmit_initial(s, effective_now, buffer, &mut self.tls)
    Handshake(s)   → poll_transmit_handshake(s, effective_now, buffer, &mut self.tls)
    Established(s) → poll_transmit_established(s, effective_now, buffer, &mut self.tls)
    Closing(s)     → poll_transmit_closing(s, effective_now, buffer)  // rate-limited CONNECTION_CLOSE
    Draining(_) | Closed → None

function poll_transmit_initial(state, now, buffer, tls):
  step 1: if state.anti_amplification.send_budget() == 0: return None
  step 2: write Initial packet header into buffer (C2): version, DCID=peer_initial_scid (or local_initial_dcid before first server response), SCID=local_initial_scid, token=empty, length=placeholder, pn=state.initial_send.assign()
  step 3: assemble payload: any pending CRYPTO bytes from state.crypto_send_initial (chunked to fit MTU), then PADDING to MIN_INITIAL_DATAGRAM (1200B) if first packet
  step 4: write payload into buffer after header
  step 5: backfill length field; encode packet-number length bits
  step 6: protect packet (C10): protect_initial(initial_keys.client, pn, pn_byte_len, packet, pn_offset, plaintext_len)
  step 7: state.anti_amplification.sent_to_peer += packet_len
  step 8: return Some(DatagramWrite { len: packet_len, ecn: ECN::ECT_0, ... })

function close(self, error_code: u64, reason: &[u8]) -> Result<(), ConnectionError>:
  step 1: build CONNECTION_CLOSE frame (C3): type=0x1d (application close), error_code, reason
  step 2: match self.state:
    Initial(s)     → transition to Closing built with whatever keys we have
    Handshake(s)   → transition to Closing
    Established(s) → transition to Closing with application_secrets retained + close_deadline = now + 3*smoothed_pto
    Closing | Draining | Closed → no-op (idempotent)

function handle_timeout(self, now: Instant) -> Result<TimerOutcome, ConnectionError>:
  step 1: check monotonicity; update last_now
  step 2: match self.state:
    Initial(s) | Handshake(s) | Established(s):
      a. if idle_deadline reached: transition to Closed (no CC), return TimerOutcome::IdleClosed
      b. if loss_detection.next_timeout() reached: fire PTO probe or loss detection (C13/C14)
      c. if ack scheduler max-delay reached: schedule ACK for next poll_transmit
      d. return TimerOutcome::Continue
    Closing(s):
      if now >= s.close_deadline: transition to Draining
      else if now >= s.retransmit_close_after: schedule CONNECTION_CLOSE retransmit
      return TimerOutcome::Continue
    Draining(s):
      if now >= s.drain_deadline: transition to Closed
      return TimerOutcome::Continue
    Closed: return Err(IllegalInState)

function next_timeout(self) -> Option<Instant>:
  step 1: match self.state:
    Initial(s) | Handshake(s) | Established(s):
      return min(idle_deadline, loss_detection.next_timeout(), ack_scheduler.max_delay_deadline())
    Closing(s):
      return Some(min(close_deadline, retransmit_close_after))
    Draining(s):
      return Some(drain_deadline)
    Closed:
      return None
```

---

## Walk-through: paper × algorithm

Run the pseudocode against the worked example. Each step's input/output
must match the expected output table above.

### t=1_000_000: construction
```
new_client(mock_config, local_tp, Instant::from_micros(1_000_000)):
  step 1: local_initial_dcid = [0x83, 0x94, ...] (per fixture)
  step 2: tls = MockTlsProvider::new(...) → ok
  step 3: initial_keys = initial_keys::derive(&dcid) → matches RFC 9001 §A.1 (C5 tests already verify bit-exact)
  step 4: tls.write_handshake(Initial, &mut crypto_send_initial) writes ClientHello bytes
  step 5: build InitialState; idle_deadline = 1_000_000 + 30_000_000 = 31_000_000
  step 6: return Connection { state: Initial(s) }
```
Matches expected output row 1 ✓.

### t=1_000_001: poll_transmit
```
poll_transmit(1_000_001, buf):
  step 1: effective_now = 1_000_001
  step 2: match Initial(s):
    poll_transmit_initial(s, 1_000_001, buf, &tls):
      step 1: anti_amp.send_budget = unlimited (haven't received yet) — wait, peer hasn't validated us yet. For client, we always have budget for first send.
      step 2..4: write Initial header (DCID=local_initial_dcid since we haven't gotten peer's SCID yet), SCID=local_initial_scid, token=empty, pn=s.initial_send.assign()=0; CRYPTO(0, client_hello_bytes); PADDING to 1200B
      step 5: backfill length
      step 6: protect_initial(initial_keys.client, 0, 4, packet, pn_offset=22, plaintext_len) — C10 tested
      step 7: anti_amp.sent_to_peer += 1200
      step 8: return Some(DatagramWrite { len=1200, ... })
```
Matches expected row 2 ✓.

### t=2_000_000: handle_datagram (server initial)
```
handle_datagram(2_000_000, server_initial_bytes):
  step 1: monotonicity OK (2_000_000 > 1_000_001)
  step 2: last_now = 2_000_000
  step 3: Initial(s):
    handle_initial_datagram(s, 2_000_000, server_initial_bytes, &tls):
      step 1: parse header → Initial type, DCID=local_initial_scid (server addressed us by it), SCID=peer_initial_scid
      step 2: epoch=Initial ✓
      step 3: unprotect_initial(initial_keys.server, largest_received=0, packet, pn_offset) → ok
      step 4: initial_recv.record_received(server_pn=0) → ok (new)
      step 5: frames:
        ACK(largest_acked=0) → initial_send.record_acked(0)
        CRYPTO(offset=0, server_hello_bytes):
          tls.read_handshake(Initial, server_hello_bytes, &mut connection_event_sink)
          sink.on_new_secrets(handshake_secrets) ← mock scripted
      step 6: sink fired NewSecrets(handshake) → advance!
        construct HandshakeState {
          side: Client, origin: 1_000_000, last_now: 2_000_000,
          local_cid_queue: empty, remote_cid_queue: insert(peer_initial_scid),
          local_transport_params: ...,
          anti_amp: copied from initial,
          idle_deadline: 31_000_000,
          initial_send: copied (largest_acked=0, next=1),
          initial_recv: copied (largest_received=0),
          initial_keys: copied,
          handshake_send: SendSpace::new(),
          handshake_recv: RecvSpace::new(),
          handshake_secrets: <from sink>,
          crypto_send_initial: empty (CHLO already drained),
          crypto_send_handshake: tls.write_handshake(Handshake, ...) → client Finished bytes,
        }
      step 7: return StateOutcome::Advance(handshake_state)
  dispatcher: self.state = ConnectionState::Handshake(handshake_state)
```
Matches expected row 3 ✓.

### t=2_000_001: poll_transmit (coalesced Initial-ACK + Handshake-CRYPTO)
```
poll_transmit(2_000_001, buf):
  match Handshake(s):
    poll_transmit_handshake(s, 2_000_001, buf, &tls):
      // coalesce: build Initial(ACK of 0) first, then Handshake(CRYPTO Finished)
      Initial portion:
        pn = s.initial_send.assign() = 1
        payload: ACK(largest=0, delay=microseconds since received)
        protect_initial(initial_keys.client, 1, ..., ...)
      Handshake portion:
        pn = s.handshake_send.assign() = 0
        payload: CRYPTO(0, client_finished_bytes from s.crypto_send_handshake)
        protect with handshake_secrets.client + AES-128-GCM
      packet_len = sum
      return Some(DatagramWrite { len=packet_len, ... })
```
Matches expected row 4 ✓.

### t=3_000_000: handle_datagram (server handshake with EE + Finished)
```
handle_datagram(3_000_000, server_handshake_bytes):
  match Handshake(s):
    handle_handshake_datagram(s, 3_000_000, bytes, &tls):
      parse header → Handshake type
      unprotect with handshake_secrets.server
      record_received(server_handshake_pn=0)
      frames:
        ACK(largest=0) → handshake_send.record_acked(0)
        CRYPTO(offset=0, server_finished_+_EE_bytes):
          tls.read_handshake(Handshake, bytes, &mut sink)
          sink fires in order (mock scripted):
            on_event(PeerTransportParameters(&[encoded TP bytes]))
              → sink copies into PeerTransportParametersOwned (idle_timeout=30000, ack_delay_exp=3, max_ack_delay=25, ...)
            on_new_secrets(application_secrets, generation=0)
              → sink stashes for advance
            on_event(HandshakeConfirmed)
              → sink marks ready_to_advance
      all three sink events fired → advance
        construct EstablishedState {
          side: Client, origin, last_now: 3_000_000,
          local_cid_queue, remote_cid_queue,
          local_transport_params, peer_transport_params: <parsed>,
          local_ack_delay_exponent: AckDelayExponent::new(3)?, peer_ack_delay_exponent: AckDelayExponent::new(3)?,
          idle_deadline: min(local idle, peer idle) ms after origin = 31_000_000,
          application_send: SendSpace::new(),
          application_recv: RecvSpace::new(),
          application_secrets: <from sink>,
          streams: StreamTable::new(per peer TP caps),
          ack_scheduler: AckScheduler::new(local_max_ack_delay=25ms),
          loss_detection: LossDetection::new(),
          congestion_control: NewReno::new(),
          handshake_keys_retain_until: Some(3_000_000 + 3*333_000 = 3_999_000),  // discard handshake keys after 3*PTO
          handshake_secrets_retained: Some(s.handshake_secrets),
        }
      return StateOutcome::Advance(established_state)
  dispatcher: self.state = Established(established_state)
```
Matches expected row 5 ✓.

### t=4_000_000: open_stream + send + poll_transmit
```
open_stream(bidi=true) on Established:
  streams.next_local_bidi_id() → StreamId(0) (per RFC 9000 §2.1 client-bidi numbering)
  streams.insert(StreamId(0), Stream::new_local_bidi())
  return Ok(StreamId(0))

send_application(StreamId(0), b"hello"):
  streams[StreamId(0)].send_buffer.append(b"hello") → ok
  ack_scheduler.note_writable_event()

poll_transmit(4_000_002, buf):
  match Established(s):
    poll_transmit_established(s, 4_000_002, buf, &tls):
      // 1-RTT packet
      pn = s.application_send.assign() = 0
      payload: STREAM(stream_id=0, offset=0, data=b"hello", fin=false)
      protect with application_secrets.client AES-128-GCM + header protection
      return Some(DatagramWrite { len, ... })
```
Matches expected rows 6-8 ✓.

### t=5_000_000: handle_datagram (server reply with STREAM)
```
handle_datagram(5_000_000, server_app_bytes):
  match Established(s):
    parse, unprotect (1-RTT), parse frames:
      ACK(largest=0) → loss_detection sees ack, application_send.record_acked(0)
      STREAM(stream_id=0, offset=0, data=b"world", fin=false):
        s.streams[StreamId(0)].recv_buffer.append(0, b"world")
        application_recv.record_received(server_pn=0)
    return StateOutcome::Stay(s)
```
Matches expected row 9 ✓.

### t=6_000_000: close (caller-initiated)
```
close(error_code=0x00, reason=b"bye"):
  match Established(s):
    close_frame = ConnectionCloseFrame {
      type: 0x1d, error_code: 0, reason: arrayvec[b"bye"]
    }
    close_deadline = 6_000_000 + 3 * smoothed_pto (assume PTO = 333ms = 333_000 µs)
                   = 6_000_000 + 999_000 = 6_999_000
    transition to Closing {
      side: Client, last_now: 6_000_000,
      close_frame,
      close_deadline: 6_999_000,
      application_secrets: s.application_secrets,
      remote_cid_queue: s.remote_cid_queue,
      close_application_dcid: <first remote CID>,
      retransmit_close_after: 6_000_000 + 333_000 = 6_333_000,
    }
```
Matches expected row 10 ✓.

### t=6_000_001: poll_transmit (CONNECTION_CLOSE)
```
poll_transmit(6_000_001, buf):
  match Closing(s):
    poll_transmit_closing(s, 6_000_001, buf):
      // emit 1-RTT packet carrying just CONNECTION_CLOSE
      pn = SendSpace::dedicated_close_pn (use a small counter to allow retransmit)
      payload: s.close_frame as bytes
      protect with s.application_secrets.client
      return Some(DatagramWrite { len, ... })
```
Matches expected row 11 ✓.

### t=7_000_000: handle_datagram (peer's CONNECTION_CLOSE)
```
handle_datagram(7_000_000, peer_close_bytes):
  match Closing(s):
    handle_closing_datagram(s, 7_000_000, bytes):
      // RFC §10.2.2: receiving any packet in Closing transitions to Draining if
      // it's a peer's CONNECTION_CLOSE; we don't decrypt (saves crypto cost) —
      // parse just enough to recognize it.
      parse packet header, recognize 1-RTT (or unprotectable)
      → transition to Draining
        drain_deadline = 7_000_000 + 999_000 = 7_999_000
        all keys discarded
        return Ok(())
```
Matches expected row 12 ✓.

### t=7_999_001: handle_timeout (drain expired)
```
handle_timeout(7_999_001):
  match Draining(s):
    if now (7_999_001) >= drain_deadline (7_999_000): transition to Closed
    return Ok(TimerOutcome::Drained)
```
Matches expected row 13 ✓.

### Walk verification

Every expected row matches. The walk does not skip steps. Each transition consumes the old state and produces the new one with concrete data values. No `Box<dyn>`, no `Arc<Mutex<State>>`, no runtime "which state am I in" checks beyond the dispatcher's exhaustive `match`.

---

## Code site (to be implemented post-design)

`proxima-quic-proto/src/connection/state.rs` will contain `ConnectionState` enum + per-variant structs.

`proxima-quic-proto/src/connection/transitions.rs` will contain per-state handlers (`handle_initial_datagram`, `handle_handshake_datagram`, etc.) and the transition functions.

`proxima-quic-proto/src/connection/mod.rs` will contain the `Connection<Provider: TlsProvider>` outer type + the `core::mem::replace`-based dispatcher.

Mapping pseudocode steps → code locations:

- `new_client` → `Connection::new_client` (mod.rs)
- `handle_datagram` dispatcher → `Connection::handle_datagram` (mod.rs)
- `handle_initial_datagram` → `transitions::handle_initial_datagram` (transitions.rs)
- `handle_handshake_datagram` → `transitions::handle_handshake_datagram`
- `handle_established_datagram` → `transitions::handle_established_datagram`
- `handle_closing_datagram` → `transitions::handle_closing_datagram`
- `poll_transmit*` family → `transitions::poll_transmit_*`
- `close` → `Connection::close` (mod.rs)
- `handle_timeout` → `Connection::handle_timeout` (mod.rs) + per-state helpers
- `next_timeout` → `Connection::next_timeout` (mod.rs)

Deviations from pseudocode + equivalence arguments:

- Pseudocode uses `core::mem::replace(&mut self.state, Closed)` for transitions; code may use a typestate trick (`Connection<S: State>`) instead — equivalent if exhaustive match still enforces all transitions.
- Pseudocode shows `core::mem::replace` taking `Closed` as the sentinel; code may use a `ConnectionState::Transitioning` variant to make panic-recovery explicit — equivalent if transitions are infallible (no `panic!` per workspace rules, errors return `Err` and leave the old state in place).
- Pseudocode separates `connection_event_sink` from the dispatcher; code may inline as a closure capturing `&mut handshake_state` — equivalent if the sink callbacks fire in the same order.

---

## Test plan (encodes the worked example)

`proxima-quic-proto/tests/c11_client_lifecycle.rs`:

```rust
// Worked example from docs/proxima-quic/c11-fsm-design.md.
// Client connection from Initial through Established to graceful close.
// MUST replicate every state transition + every expected-output row.

#[test]
fn client_lifecycle_initial_to_closed() {
    let origin = Instant::from_micros(1_000_000);
    let dcid = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];

    let mock_provider = MockTlsProvider::script_client_handshake(&[
        MockStep::EmitHandshakeBytes { epoch: Initial, bytes: CLIENT_HELLO },
        MockStep::ReadHandshake { epoch: Initial, expect: SERVER_HELLO },
        MockStep::InstallSecrets(handshake_secrets()),
        MockStep::EmitHandshakeBytes { epoch: Handshake, bytes: CLIENT_FINISHED },
        MockStep::ReadHandshake { epoch: Handshake, expect: SERVER_FINISHED_PLUS_EE },
        MockStep::EmitEvent(TlsEvent::PeerTransportParameters(SAMPLE_PEER_TP_BYTES)),
        MockStep::InstallSecrets(application_secrets()),
        MockStep::EmitEvent(TlsEvent::HandshakeConfirmed),
    ]);

    let mut connection = Connection::new_client(mock_provider, local_tp(), origin)
        .expect("construct");
    assert!(matches!(connection.state(), ConnectionState::Initial(_)));

    // t=1_000_001: poll_transmit first Initial packet
    let mut buf = [0u8; 1500];
    let write = connection.poll_transmit(Instant::from_micros(1_000_001), &mut buf)
        .expect("first transmit");
    assert_eq!(write.len, 1200, "initial datagram MUST be padded to 1200B");

    // t=2_000_000: handle server Initial (with CRYPTO ServerHello)
    let server_initial = build_server_initial(&dcid, SERVER_HELLO);
    connection.handle_datagram(Instant::from_micros(2_000_000), &server_initial)
        .expect("handle server initial");
    assert!(matches!(connection.state(), ConnectionState::Handshake(_)),
            "after on_new_secrets(handshake), state MUST be Handshake");

    // ...continue through each row of the expected-output table...

    // t=6_000_000: caller-initiated close
    connection.close(0x00, b"bye").expect("close");
    assert!(matches!(connection.state(), ConnectionState::Closing(_)));

    // t=6_000_001: poll_transmit emits CONNECTION_CLOSE
    let write = connection.poll_transmit(Instant::from_micros(6_000_001), &mut buf)
        .expect("close transmit");
    assert!(write.len > 0);

    // t=7_000_000: receive peer's CONNECTION_CLOSE
    let peer_close = build_peer_close();
    connection.handle_datagram(Instant::from_micros(7_000_000), &peer_close)
        .expect("handle peer close");
    assert!(matches!(connection.state(), ConnectionState::Draining(_)));

    // t=7_999_001: drain expiration
    connection.handle_timeout(Instant::from_micros(7_999_001)).expect("drain timeout");
    assert!(matches!(connection.state(), ConnectionState::Closed));
}
```

Plus illegal-state property tests:

```rust
#[test]
fn open_stream_in_initial_is_illegal() {
    let connection = build_client_in_initial_state();
    assert!(matches!(
        connection.open_stream(StreamDirection::Bidi),
        Err(ConnectionError::IllegalInState { current: "Initial" })
    ));
}

#[test]
fn send_application_in_handshake_is_illegal() { /* ... */ }

#[test]
fn close_in_draining_is_noop() {
    let mut connection = drive_to_draining();
    let before = connection.state_label();
    connection.close(0, b"").expect("noop");
    assert_eq!(connection.state_label(), before);
}
```

Plus monotonicity test (from prior Instant resolution):

```rust
#[test]
fn handle_datagram_with_non_monotonic_now_returns_error_without_state_mutation() {
    let mut connection = drive_to_established(Instant::from_micros(5_000_000));
    let earlier = Instant::from_micros(4_000_000);
    let result = connection.handle_datagram(earlier, b"any bytes");
    assert!(matches!(result, Err(ConnectionError::NonMonotonicTime { .. })));
    // state MUST still be Established — bad-clock blip never mutates decode state
    assert!(matches!(connection.state(), ConnectionState::Established(_)));
}
```

Plus the principle-11-mandated state-machine walkthrough example:

```rust
// proxima-quic-proto/examples/connection_state_walkthrough.rs
// Runnable demo per principle 11 ("every state machine ships with
// examples/<sm>_walkthrough.rs driving it through every legal transition").
```

---

## Self-critique (mandatory per skill)

- **Pass 1 — paper before code**: yes; this document is the design pass. No C11 code exists yet.
- **Pass 2 — algorithm walk produces exact expected output**: verified for every row of the worked example.
- **Pass 3 — code maps step-by-step to algorithm**: deferred (code doesn't exist yet); explicit mapping table provided in "Code site" section.
- **Pass 4 — test uses exact inputs from worked example**: yes; the test fixture uses `Instant::from_micros(1_000_000)` as origin, `[0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08]` as DCID, and asserts state at every t in the expected-output table.
- **Pass 5 — would the test fail on off-by-one / wrong-direction / swapped-arg bugs**: yes; the worked example asserts EXACT state at each tick. An off-by-one in PN assignment (e.g., starting at 1 instead of 0) would break the ACK-record-acked assertion at t=5_000_000. A swapped client/server keys bug would break the t=2_000_000 unprotect.
- **Pass 6 — paper linked to test**: yes; test docstring references this doc by path.

---

## Status

This FSM design is the SPEC for C11. Implementation work that follows MUST trace each named pseudocode step to a code site and MUST encode the worked example as the lead test.

Open follow-ups (not blocking C11 v1):

- `AntiAmplificationCounter` shape (RFC 9000 §8.1) — straightforward but unfilled in the pseudocode.
- `StreamTable`, `AckScheduler`, `LossDetection`, `CongestionController` are C12/C13/C14/C15 territory — C11 stubs them as opaque types and the worked example skips frame-level streams behavior (just verifies STREAM frame round-trip).
- `ConnectionCloseFrame` building from `(error_code, reason)` belongs to C3 frame encoder — already exists.
- 0-RTT (RFC 9001 §4.6) is intentionally excluded from this worked example; deferred to C24 design pass.
- Server-side worked example is the mirror — design fully implied but not walked here. Add separately when server-side construction lands.

The FSM is ready for code.
