# C17 — BBR congestion control (foundational primitives)

Per [draft-ietf-ccwg-bbr-05] (latest as of 2026-05-25; pinned in
[`rfc-reference.md`](./rfc-reference.md)). The IETF draft generalizes
Google's BBR / BBRv2 algorithm into a transport-agnostic specification.

Tagged per principle 13: `/algorithm-development` (the per-state
control logic has 4 primary states with sub-states, ~20 per-state
parameters, and intricate state transitions that the draft devotes
~3000 lines to). The full algorithm is a multi-week landing; this
row ships the foundational primitives that everything else composes.

[draft-ietf-ccwg-bbr-05]: https://www.ietf.org/archive/id/draft-ietf-ccwg-bbr-05.txt

## Pinned revision

`draft-ietf-ccwg-bbr-05` (dated 2 March 2026). State-machine
nomenclature, filter window constants, and parameter names taken
verbatim from this revision. **If the draft revises** (it has
revised ~5 times to date) the pin must be re-cut with a discipline-
log row documenting changed parameter values.

## Scope split

The full BBR algorithm is too large for a single component. C17 is
decomposed:

| Slice | Scope | Lands here |
|---|---|---|
| **C17.0** (this row) | Foundational primitives — MinRttFilter (10-sec window), MaxBwFilter (2-cycle window), BbrState enum (Startup / Drain / ProbeBW(SubState) / ProbeRTT) discriminated FSM. **Per-state control logic + delivery-rate sampling defer.** | tier-3 v1 |
| C17.1 | Delivery-rate sampling per draft §4.1 — per-packet `delivered`/`delivered_time`/`is_app_limited` state + `RateSample` computation | defers; needs per-packet bookkeeping that ties into the existing C14 SentPacket state |
| C17.2 | Per-state control logic — pacing_rate + cwnd_gain for each {Startup, Drain, ProbeBW.{Up, Down, Cruise, Refill}, ProbeRTT} | defers; needs C17.1 rate samples + the `CongestionController` trait wire-up |
| C17.3 | ProbeBW cycle scheduling + ProbeRTT entry/exit logic | defers; needs C17.2 |

Principle 14 binds hard here: BBR is an active research area, and
parity-or-prove-bug against Google's reference C implementation is
the gate for landing the per-state control. C17.0 ships only the
data-structure primitives that have unambiguous semantics from the
draft — the algorithmic content waits for paired bench against the
reference impl.

## C17.0 — Foundational primitives

### MinRttFilter

Per draft §2.13.1: `BBR.min_rtt` = the windowed-min of RTT samples
over `BBR.MinRTTFilterLen` (= 10 seconds).

```rust
pub struct MinRttFilter {
    min_rtt: Option<Duration>,
    min_rtt_stamp: Instant,  // wall-clock when the current min was sampled
    window: Duration,        // 10 seconds per draft
}
```

Methods:
- `new(window: Duration) -> Self`
- `note_sample(rtt: Duration, now: Instant)` — update; if `rtt < min_rtt` OR `(now - min_rtt_stamp) > window`, replace with new sample
- `get() -> Option<Duration>` — current min (None if no samples ever recorded)
- `is_expired(now: Instant) -> bool` — whether the current min has aged past the window

### MaxBwFilter

Per draft §2.10: `BBR.max_bw_filter` = a windowed-max filter for
delivery-rate samples with window `BBR.MaxBwFilterLen` = 2 (up to
2 ProbeBW cycles).

```rust
pub struct MaxBwFilter {
    /// Tracks the max delivery rate over the last N cycles.
    /// Index 0 = current cycle, index 1 = previous cycle. Cycle
    /// advance shifts index 1 ← 0, clears index 0.
    cycles: [u64; 2],
    cycle_count: u8,  // single bit needed per draft §2.10
}
```

Methods:
- `new() -> Self`
- `note_sample(delivery_rate: u64)` — `cycles[0] = max(cycles[0], delivery_rate)`
- `advance_cycle()` — `cycles[1] = cycles[0]; cycles[0] = 0; cycle_count ^= 1`
- `get() -> u64` — `max(cycles[0], cycles[1])`

### BbrState discriminated enum

Per draft §3.3 (state-machine overview) + §3.5 (per-state parameters):

```rust
pub enum BbrState {
    Startup,
    Drain,
    ProbeBw(ProbeBwSubState),
    ProbeRtt,
}

pub enum ProbeBwSubState {
    Up,      // probe for higher bandwidth (pacing_gain > 1)
    Down,    // drain queue after probe (pacing_gain < 1)
    Cruise,  // hold steady (pacing_gain = 1)
    Refill,  // refill pipe between probe cycles
}
```

State-transition primitives ship as methods on a wrapper that owns
the enum + the filters; per-state control logic defers.

## Worked example (MinRttFilter — 10-second window)

3 RTT samples observed over a 12-second wall-clock window.

| t (s) | RTT sample | Action | Filter state after |
|-------|------------|--------|--------------------|
| 0     | 50 ms      | note_sample(50ms, t=0)    | min_rtt=50ms, stamp=0 |
| 4     | 30 ms      | note_sample(30ms, t=4)    | min_rtt=30ms (better), stamp=4 |
| 11    | 80 ms      | note_sample(80ms, t=11)   | min_rtt=30ms (still in window: 11-4=7s < 10s), stamp=4 |
| 15    | 80 ms      | note_sample(80ms, t=15)   | min_rtt=80ms (window expired: 15-4=11s > 10s), stamp=15 |

## Worked example (MaxBwFilter — 2-cycle window)

| event | Action | cycles[0] | cycles[1] | get() |
|-------|--------|-----------|-----------|-------|
| init  | new()  | 0         | 0         | 0     |
| sample 100 | note_sample(100) | 100  | 0       | 100   |
| sample 80  | note_sample(80)  | 100  | 0       | 100   |
| advance    | advance_cycle()  | 0    | 100     | 100   |
| sample 60  | note_sample(60)  | 60   | 100     | 100   |
| advance    | advance_cycle()  | 0    | 60      | 60    |
| sample 70  | note_sample(70)  | 70   | 60      | 70    |

## Security review (per principle 13)

| Concern | Mitigation |
|---|---|
| RTT-sample spoofing via forged ACKs | RTT samples already feed through the integrity-protected ACK path (C13); BBR layer trusts the upstream measurement |
| Bandwidth-overestimate DoS (peer ACKs at higher rate than actually delivered) | RFC 9002 + draft §4.1: rate samples include `is_app_limited` bit so the bound is conservative |
| ProbeBW oscillation under loss | Draft §3.5: pacing_gain values are bounded constants (Up=1.25, Down=0.75 in v2 drafts) — no caller input |

No crypto material in this layer; no `/security-review` triage table
beyond the above.

## Tier

C17.0 is tier-3 (POD filter state + state enum; no alloc).

## Per principle 14 (incumbent wins)

- `MinRTTFilterLen = 10 seconds` taken verbatim from draft §2.13.1.
- `MaxBwFilterLen = 2 cycles` taken verbatim from draft §2.10.
- State names (Startup / Drain / ProbeBW / ProbeRTT) and sub-states
  (Up / Down / Cruise / Refill) taken verbatim from draft §3.3 + §3.5.
- The actual per-state control parameters (pacing gains, cwnd gains,
  probe interval) defer until C17.2 — at which point paired bench
  vs Google's reference C impl is the parity gate.

## Sized constants (principle 12)

```toml
[bbr]
# BBR.MinRTTFilterLen per draft §2.13.1 — 10 seconds.
min_rtt_filter_window_micros = 10_000_000
```

## Self-critique

- **Pass 1 — paper before code**: yes.
- **Pass 2 — algorithm walk produces exact expected output**: yes (2 worked examples — min_rtt filter + max_bw filter).
- **Pass 3 — code maps step-by-step**: planned.
- **Pass 4 — test uses exact inputs from worked example**: planned.
- **Pass 5 — would the test fail on bugs**: yes; off-by-one in the window expiry, dropping the previous-cycle max on advance, or wrong sub-state nesting in BbrState would all break specific assertions.
- **Pass 6 — paper linked to test**: yes.
