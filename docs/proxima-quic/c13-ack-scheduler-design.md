# C13 — ACK generation + scheduling (paper proof)

Per RFC 9000 §13.2 + §19.3. Resolves the deferred edges from
[`edges.md`](./edges.md):

- ACK range-set storage shape — `ArrayRangeSet<MAX>` overflow policy.
- ACK scheduler emit conditions (delayed vs immediate).

Composes:

- C3 `Frame::Ack` for encoding the wire format.
- C9 `RecvSpace<WINDOW>` for duplicate-detection.
- C9 packet-number truncation for the ACK-frame `largest` field.

Per workspace principle 11 — every multi-step protocol behavior is a
discriminated enum FSM with per-variant data. The scheduler here is
state-machine-light (one struct per epoch); the range-set is a pure
data primitive.

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

---

## ArrayRangeSet<MAX> — sorted descending interval set

### Representation

```rust
pub struct ArrayRangeSet<const MAX: usize> {
    /// Inclusive packet-number ranges in DESCENDING `end` order.
    /// `ranges[0]` has the largest `end`; `ranges[len-1]` has the smallest.
    /// Wire encoding starts from the largest and walks down via gap+length
    /// pairs (RFC 9000 §19.3.1).
    ranges: arrayvec::ArrayVec<RangeInclusive, MAX>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeInclusive {
    pub start: u64,  // inclusive
    pub end: u64,    // inclusive (>= start)
}
```

`MAX` comes from `prime-runtime.toml [quic].max_ack_ranges` (default 32),
threaded through `proxima-build` constants per the existing pattern.

### `insert(pn: u64)` algorithm

Find the right slot:

1. Binary-search descending for the first range with `end < pn - 1`. The
   slot above that range is where `pn` would land.
2. Three cases:
   - **Extends-down**: the range just above has `start - 1 == pn` →
     decrement `start` to `pn`.
   - **Extends-up**: the range just below has `end + 1 == pn` →
     increment `end` to `pn`.
   - **Both**: range above's `start - 1 == pn` AND range below's
     `end + 1 == pn` — merge the two by extending the higher's `start`
     down to the lower's `start`, then removing the lower range.
   - **Neither**: insert a new singleton `{ start: pn, end: pn }` at the
     correct sorted position.
3. **Duplicate** (`pn` is already within an existing range): no-op.

### Overflow policy

When `ranges.is_full()` and we need to insert a new range that doesn't
extend any existing range, drop the smallest-end range (the oldest
acknowledged region). Rationale:

- RFC 9000 §13.2.4 only REQUIRES acknowledging the largest packet
  number to reconstruct loss; older ranges are an optimization for
  retransmit avoidance.
- Quinn-proto uses the same drop-oldest policy under `MAX_ACK_BLOCKS`.
- Drop-oldest preserves recent recovery info (where loss-detection cares).

When we'd need to MERGE but the merge would still fit, that's a free
operation — merging reduces the range count.

### Wire encoding via `Frame::Ack`

```
largest = ranges[0].end
first_range = ranges[0].end - ranges[0].start  // RFC 9000 §19.3.1 "ACK Range Length"

for i in 1..ranges.len() {
    gap_i    = ranges[i-1].start - ranges[i].end - 2  // RFC §19.3.1: gap encodes (Ranges[i-1].smallest - 2 - Ranges[i].largest)
    length_i = ranges[i].end - ranges[i].start         // ack-range length
    write_varint(gap_i)
    write_varint(length_i)
}
```

The `ranges_raw` field of `Frame::Ack` is the back-to-back encoded `(gap,
length)` pair stream; `range_count` is `ranges.len() - 1`.

---

## AckScheduler — when to emit

### Per-epoch state

```rust
pub struct AckScheduler {
    /// Sorted descending range set of received packet numbers.
    pub ranges: ArrayRangeSet<MAX_ACK_RANGES>,
    /// Number of ack-eliciting packets received since last ACK emitted.
    pub ack_eliciting_since_last_ack: u32,
    /// When the first ack-eliciting packet of the current pending batch
    /// arrived (used to enforce max_ack_delay).
    pub pending_ack_deadline: Option<Instant>,
    /// "Send ACK immediately on next poll_transmit" — set by reorder /
    /// PING / HANDSHAKE_DONE / ECN-CE triggers per RFC 9000 §13.2.1.
    pub immediate_ack_requested: bool,
    /// Largest packet number we've ack'd in a previously-sent ACK frame.
    /// Used to know whether the next ACK is informationally new.
    pub largest_acked_sent: Option<u64>,
}
```

### `record_received(pn, is_ack_eliciting, now, max_ack_delay)`

```
1. ranges.insert(pn)
2. if !is_ack_eliciting: return  // PURE-ACK packets do not trigger ACK emission
3. ack_eliciting_since_last_ack += 1
4. if pending_ack_deadline is None:
       // Never delay: the deadline is `now`. It is retained only so the I/O
       // layer can wake to flush a lone ACK when the connection is otherwise
       // idle; the ACK is emitted on the next transmit opportunity.
       pending_ack_deadline = Some(now)
5. // Reorder detection: if pn < ranges[0].end (i.e. not the new largest), set immediate.
//    This includes the case where pn fills a gap — gap fills trigger immediate ACK
//    per RFC 9000 §13.2.1.
   if pn != ranges[0].end:
       immediate_ack_requested = true
```

### `should_emit(now)` → `bool`

```
return immediate_ack_requested
    OR (pending_ack_deadline is Some(deadline) AND now >= deadline)
```

We never delay ACK emission: `pending_ack_deadline` is set to `now` on the
first unacked ack-eliciting packet, so this fires on the next transmit
opportunity. Acking every packet trivially satisfies the RFC 9000 §13.2.2
"at least every other ack-eliciting packet" SHOULD; holding a lone ACK for
`max_ack_delay` would only add request/response latency. The eliciting
counter (`ack_eliciting_since_last_ack`) still tracks unacked packets but no
longer gates emission.

### `on_emitted(largest)` — clear pending state after a frame is sent

```
1. ack_eliciting_since_last_ack = 0
2. pending_ack_deadline = None
3. immediate_ack_requested = false
4. largest_acked_sent = Some(largest)
```

### `next_deadline()` → `Option<Instant>`

```
return pending_ack_deadline
```

Plumbed into `Connection::next_timeout` so the I/O facade knows when to
wake up specifically for an ACK flush.

### `has_pending()` → `bool`

```
return ranges.len() > 0 && largest_acked_sent != Some(ranges[0].end)
```

A pending ACK exists iff we've received at least one packet AND the
largest received packet is informationally newer than the last largest
we ack'd. Used by `poll_transmit` to coalesce ACK with CRYPTO opportunistically.

---

## Per-epoch instances

Each epoch gets its own `AckScheduler`:

- `InitialState.initial_ack_scheduler`
- `HandshakeState.initial_ack_scheduler` (we may still owe Initial ACKs after promoting to Handshake per RFC 9000 §17.2.2.1) + `HandshakeState.handshake_ack_scheduler`
- `EstablishedState.application_ack_scheduler`

The Initial-epoch scheduler in `HandshakeState` is consumed and emitted
when `poll_transmit_handshake` coalesces; once acked, the scheduler can
be dropped (the connection has moved past Initial). For C13 v1 we
emit each epoch's ACK in its own datagram; coalescing per RFC 9000 §12.2
lands in C11.6.

---

## Worked example (never-delay emission)

Mirrors `scheduler::tests::worked_example_from_design_doc`. Every ack-eliciting
packet sets the deadline to `now`, so `should_emit` is true on the next
transmit opportunity. A pure-ACK packet bumps nothing. A gap-fill (reorder)
sets the immediate flag (RFC 9000 §13.2.1) — also immediate.

State at `t = 5_000_000 µs`: `EstablishedState` with empty scheduler.

| t (µs) | call | ranges after | ack_eliciting_since | pending_deadline | immediate | should_emit |
|---|---|---|---|---|---|---|
| 5_000_000 | record(100, eliciting) | `[{100,100}]` | 1 | Some(5_000_000) | false | **true** (deadline=now) |
| 5_010_000 | record(101, eliciting) | `[{100,101}]` | 2 | Some(5_000_000) | false | **true** |
| 5_010_001 | on_emitted(101) | `[{100,101}]` | 0 | None | false | false |
| 5_015_000 | record(102, eliciting) | `[{100,102}]` | 1 | Some(5_015_000) | false | **true** (deadline=now) |
| 5_015_001 | on_emitted(102) | `[{100,102}]` | 0 | None | false | false |
| 6_000_000 | record(150, eliciting=false) | `[{150,150},{100,102}]` | 0 (PURE-ACK) | None | false | false |
| 6_001_000 | record(151, eliciting) | `[{150,151},{100,102}]` | 1 | Some(6_001_000) | false | **true** (deadline=now) |

A gap-fill reorder (e.g. receiving 104 then 103) sets `immediate_ack_requested`
and emits at once — see `scheduler::tests::reorder_triggers_immediate_emit`.

Walk verifies: every-2 rule + reorder triggers immediate + pure-ACK packets don't restart the timer + deadline-driven flush via handle_timeout.

---

## Sized.rs constants

- `MAX_ACK_RANGES = 32` (already in `prime-runtime.toml [quic]`).
- `DEFAULT_MAX_ACK_DELAY_MICROS = 25_000` (RFC 9000 §18.2 default 25 ms).

---

## Code site mapping

- `proxima-quic-proto/src/range_set.rs` — `ArrayRangeSet<MAX>` + `RangeInclusive`.
- `proxima-quic-proto/src/ack/mod.rs` + `scheduler.rs` — `AckScheduler`.
- `proxima-quic-proto/src/connection/state.rs` — add `initial_ack_scheduler` /
  `handshake_ack_scheduler` / `application_ack_scheduler` fields.
- `proxima-quic-proto/src/connection/mod.rs` — wire `record_received` calls
  into `parse_and_apply_initial` / `parse_and_apply_handshake`; wire ACK
  emission into `build_initial_datagram` / `build_handshake_datagram` when
  the scheduler signals pending.
- `proxima-quic-proto/benches/bench_c13_range_set.rs` — bench arms:
  in-order insert / reverse-order insert / random insert / wire encode /
  worst-case 32-range encode.

---

## Self-critique

- **Pass 1 — paper before code**: yes; this doc precedes any C13 code.
- **Pass 2 — algorithm walk produces exact expected output**: verified for the
  every-2 rule + reorder trigger + deadline flush; pure-ACK packets correctly
  excluded from the eliciting counter.
- **Pass 3 — code maps step-by-step to algorithm**: deferred until code lands;
  the implementation MUST keep `record_received` / `should_emit` / `on_emitted`
  / `next_deadline` / `has_pending` as named methods that match the paper.
- **Pass 4 — test uses exact worked example**: yes; unit test
  `ack_scheduler_walked_example` will encode every row of the table above.
- **Pass 5 — would the test fail on bugs**: yes — off-by-one in the
  `gap = smallest - 2 - largest` arithmetic, swap of `start`/`end`,
  reorder trigger missing the `pn != ranges[0].end` check, or pure-ACK
  packets advancing the eliciting counter would all break specific table rows.
- **Pass 6 — paper linked to test**: test docstring references this doc.
