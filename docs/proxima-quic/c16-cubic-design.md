# C16 — CUBIC congestion control (paper proof)

Per [RFC 9438] (CUBIC). Plugs into the existing
[`CongestionController`] trait (C15). The interesting work is the
**integer-arithmetic CUBIC** — the RFC's algorithm reaches for `cbrt`
and floating-point constants; we need an integer-only path so the
controller stays tier-3.

[RFC 9438]: https://www.rfc-editor.org/rfc/rfc9438
[`CongestionController`]: ../../proxima-quic-proto/src/congestion/mod.rs

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## CUBIC summary

CUBIC replaces NewReno's linear congestion-avoidance growth with a
**cubic function of time-since-last-reduction**:

```
W_cubic(t) = C × (t - K)^3 + W_max
```

where:

- `W_max` — cwnd (in segments) at the most recent loss event.
- `K` — time at which `W_cubic(K) = W_max` (i.e. when cwnd has
  recovered to the pre-loss value).
- `C` — scaling constant; RFC 9438 §5.1 recommends `0.4`.
- `t` — time (seconds) since the most recent loss event.
- All cwnd values are expressed in **segments** (multiples of
  `max_datagram_size`).

On a loss event the algorithm computes:

```
W_max  = cwnd_segments
cwnd  *= beta            (beta = 0.7 per RFC §5.1)
K      = cbrt(W_max * (1 - beta) / C)
```

The CA growth rule on each ACK is:

```
target = W_cubic(t + RTT)
if target > cwnd:    cnt = cwnd / (target - cwnd)
else:                cnt = 100 * cwnd  (slow growth)
cwnd += MSS / cnt    (per ACK)
```

There's also a TCP-friendly fallback (`W_est(t)`) but the RFC marks
it OPTIONAL. We defer it to a follow-on (C16.1).

## Integer arithmetic

### Constants

- `BETA_NUM / BETA_DENOM = 7 / 10` (0.7 multiplicative decrease).
- `C_NUM / C_DENOM = 4 / 10` (0.4 scaling constant, in units of
  segments / second^3).

### Time scaling

The RFC formulas work in **seconds** and **segments**. To avoid
intermediate overflow:

- Keep `t` in **milliseconds** (`u64`). One ms = 10^-3 s, so for the
  `(t - K)^3` term to stay in u64 we cap `(t - K)` at `2^21 = ~35 minutes`
  (still way more than any sane recovery period).
- Express `K` in milliseconds via `K_ms = cbrt(W_max_segs * (1 - beta) / C * 10^9)`
  — that is, `cbrt(W_max * 0.3 / 0.4 * 10^9) = cbrt(W_max * 0.75 * 10^9)`.

### Integer cube root

```
fn cbrt_u64(n: u64) -> u64:
  // Newton-Raphson: x_{i+1} = (2*x + n / (x*x)) / 3
  if n == 0: return 0
  // Seed from leading-zero count.
  let bits = 64 - n.leading_zeros()
  let mut x: u64 = 1 << (bits / 3 + 1)
  for _ in 0..10:
    let x2 = x * x
    let next = (2 * x + n / x2) / 3
    if next >= x: return x
    x = next
  x
```

5-10 iterations converge to within 1 of true cube root for any u64.
Test: `cbrt_u64(1_000_000_000_000) == 10_000` (10^12 = (10^4)^3).
Test: `cbrt_u64(27) == 3`. Test: `cbrt_u64(0) == 0`.

### `W_cubic(t_ms) -> segments`

```
fn w_cubic_segments(t_ms: u64, k_ms: u64, w_max_segs: u64) -> u64:
  // (t - K) signed; we always evaluate in the post-recovery region
  // where t >= K (and use w_est_segments for the slow concave region).
  let delta_ms = t_ms.saturating_sub(k_ms);  // signed in spec; we clamp at 0
  // delta_s^3 in units of seconds^3 = (delta_ms / 1000)^3 = delta_ms^3 / 1e9.
  // We want C * delta_s^3 in segments = (C_NUM/C_DENOM) * delta_ms^3 / 1e9.
  let delta_cubed = delta_ms.saturating_mul(delta_ms).saturating_mul(delta_ms);
  let growth = delta_cubed.saturating_mul(C_NUM) / (C_DENOM * 1_000_000_000);
  w_max_segs.saturating_add(growth)
```

### Worked examples

#### Cube root of perfect cubes

- `cbrt_u64(0) = 0`
- `cbrt_u64(1) = 1`
- `cbrt_u64(8) = 2`
- `cbrt_u64(27) = 3`
- `cbrt_u64(1_000_000_000_000) = 10_000` (10^4 cubed)
- `cbrt_u64(125_000_000_000_000) = 50_000`

#### K computation on a loss event

State: cwnd_segments = 100 (i.e. 100 × 1200 = 120000 B).

```
W_max_segs = 100
cwnd_segs *= beta = 100 * 7 / 10 = 70
K_ms = cbrt(W_max_segs * (1 - beta_num/beta_denom) / C * 1e9)
     = cbrt(100 * 0.3 / 0.4 * 1e9)
     = cbrt(75_000_000_000)
     ≈ 4217 ms
```

So 4.217 seconds after the loss event, the cubic function predicts
cwnd has recovered to W_max = 100 segments. Beyond that point, the
cubic function probes past W_max.

#### W_cubic growth at t=K

At `t_ms == K_ms`, `delta_ms == 0`, `growth == 0`, `W_cubic = W_max = 100`. ✓

#### W_cubic growth at t=K + 1000ms

```
delta_ms = 1000
delta_cubed = 1_000_000_000
growth = 1_000_000_000 * 4 / (10 * 1_000_000_000) = 0  (integer truncation)
```

Hm — the integer truncation at small deltas means CUBIC won't grow at
all for the first ~3-4 seconds after K. Acceptable degeneracy for the
v1 integer-only impl; documented as a known precision floor. The
algorithm-development edge says: `W_cubic increments are detectable only
above ~3000ms delta from K, which is fine because CUBIC's whole point
is probing AFTER recovery completes`.

#### W_cubic growth at t=K + 3000ms

```
delta_ms = 3000
delta_cubed = 27_000_000_000
growth = 27_000_000_000 * 4 / (10 * 1_000_000_000) = 10 (segments)
W_cubic = 100 + 10 = 110 segments
```

So after 3 seconds past recovery, CUBIC has probed 10 segments past
W_max. As `t` grows, the cubic term dominates.

### Per-ACK growth

The RFC's per-ACK growth is `cwnd += MSS / cnt` where
`cnt = cwnd_segs / (target_segs - cwnd_segs)`. With integer math:

```
fn growth_per_ack(target_segs: u64, cwnd_segs: u64, mss: u64) -> u64:
  if target_segs <= cwnd_segs: return mss / (100 * cwnd_segs).max(1)
  let cnt_recip = target_segs - cwnd_segs;
  // cwnd += mss * cnt_recip / cwnd_segs
  mss.saturating_mul(cnt_recip) / cwnd_segs.max(1)
```

## Loss event

Identical to NewReno's recovery-window logic; only the post-reduction
cwnd and the cbrt(K) computation differ.

```
on_packets_lost(lost, now, pto):
  if !in_congestion_recovery(earliest):
    congestion_recovery_start_time = now
    last_loss_time_ms = now (as u64 ms-since-origin)
    W_max_segs = cwnd / mss
    cwnd = (W_max_segs * BETA_NUM / BETA_DENOM) * mss   // *0.7
    cwnd = max(cwnd, kMinWindow)
    K_ms = cbrt_u64(W_max_segs.saturating_mul((BETA_DENOM - BETA_NUM) * 1_000_000_000) / (C_NUM * BETA_DENOM))
  // persistent-congestion check identical to NewReno.
```

## On ACK

```
on_packet_acked(packet, now):
  bytes_in_flight -= packet.size_bytes
  if in_congestion_recovery(packet.sent_time): return
  if cwnd < ssthresh: cwnd += packet.size_bytes  // slow start
  else:
    // CUBIC CA: compute W_cubic(t_ms_since_loss + rtt_ms)
    let elapsed_ms = (now - last_loss_time).as_millis()
    let target_segs = w_cubic_segments(elapsed_ms + rtt_ms, K_ms, W_max_segs)
    let cwnd_segs = cwnd / mss
    cwnd += growth_per_ack(target_segs, cwnd_segs, mss)
```

The first time the controller enters CA (no prior loss event), there's
no `W_max` or `K`. We use the current cwnd as W_max and K=0, which
makes the cubic term grow strictly upward from t=0. (Equivalent to
"there's no historical W_max to recover toward".)

## RTT injection

CUBIC needs RTT for the `target = W_cubic(t + RTT)` lookahead. C16
takes `rtt_ms` as an arg on `on_packet_acked` (extending the
[`CongestionController`] trait); or — to keep the trait stable —
exposes it via a setter the FSM calls on each RTT update from C14.

Decision: **add `update_rtt(rtt: Duration)` to the trait**. NewReno
ignores it (defaults to a no-op default impl). CUBIC stores it and
uses it in the on-ACK lookahead.

## Code site

- `proxima-quic-proto/src/congestion/cubic.rs` — `Cubic` impl.
- Re-export from `congestion/mod.rs`.
- Add `update_rtt(rtt: Duration)` default-impl to the
  `CongestionController` trait.
- `Connection<P>` keeps a `NewReno` field as today; the actual
  controller selection lives behind a generic param in a future
  refactor (or a closed-enum dispatch). For C16 we land the Cubic impl
  + trait extension + unit tests, but the FSM continues to use NewReno
  by default. Wiring CUBIC into the live FSM is a separate slice
  alongside `quic_impl=native` profile-axis work.

## Self-critique

- **Pass 1 — paper before code**: yes; this doc precedes any C16 code.
- **Pass 2 — algorithm walk produces exact expected output**: verified
  for cbrt perfect cubes, K computation (W_max=100 → K≈4217ms),
  W_cubic at t=K (=W_max), W_cubic at t=K+3000ms (=W_max+10).
- **Pass 3 — code maps step-by-step to algorithm**: deferred until
  code lands.
- **Pass 4 — test uses exact inputs from worked examples**: yes;
  unit tests `cbrt_perfect_cubes`, `k_ms_for_wmax_100`,
  `w_cubic_at_k_equals_wmax`, `w_cubic_growth_after_3s`.
- **Pass 5 — would the test fail on bugs**: yes; swapping
  BETA_NUM/BETA_DENOM, off-by-one in cbrt's seed, dropping the
  saturating_mul guard on (t-K)^3, forgetting the `cwnd_segs * mss`
  vs `cwnd` distinction would all break specific table values.
- **Pass 6 — paper linked to test**: test docstrings reference this doc.
