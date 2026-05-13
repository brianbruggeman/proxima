# prime TCP connect — state machine (algorithm-development proof)

Paper proof for `prime::os::net::TcpStream::connect` (the `Connect` future). Gates the
disciplined-component row in `discipline-net-connect.md`. Written to verify the killed-agent draft and
to fix the latent spurious-poll bug it carries.

## Worked example (paper)

**Inputs:** `addr = 127.0.0.1:P` where `P` is a bound, listening localhost `TcpListener` (accept loop
ready). Polled by a prime worker (CURRENT_REACTOR set).

**State at time T:**
- reactor is **edge-triggered**: epoll `EPOLLET` (reactor.rs:614), kqueue `EV_ADD|EV_CLEAR`
  (reactor.rs:290). `register(fd, interest)` on an fd that is *already* ready delivers an event
  (epoll/kqueue report current readiness at ADD). This is the property the accept path relies on.
- nonblocking `connect()` on a TCP socket returns `EINPROGRESS`/`WouldBlock` while the handshake runs,
  then the socket becomes **writable** when it resolves; `getsockopt(SO_ERROR)` (`take_error`) then
  yields `Ok(None)` on success or `Ok(Some(e))` on failure.
- **async contract:** `Future::poll` MAY be called spuriously (without the registered waker firing).

**Expected output (derived by hand):**
- loopback connect usually completes synchronously → first poll returns `Ready(Ok(TcpStream))`.
- otherwise: first poll registers write-interest + waker, returns `Pending`; when the writable edge
  fires, a poll re-probes completion and returns `Ready(Ok(TcpStream))` (or `Ready(Err)` on refused).
- **a spurious poll while still connecting MUST return `Pending`, not a half-open `TcpStream`.**

## Algorithm (pseudocode) — spurious-poll-safe

```
poll(state, waker):
  match state:
    Init:
      sock = nonblocking TCP socket; set_nodelay
      r = connect(sock, addr)
      if r == Ok or err==EISCONN:           return Ready(Ok(TcpStream::from(sock)))   # immediate
      if err not in {EINPROGRESS,EALREADY,WouldBlock}: return Ready(Err(err))
      key = reactor.register(fd, Write); reactor.register_write_waker(key, waker)
      state = Pending{sock,key}
      return Pending
    Pending{sock,key}:
      # DO NOT trust take_error alone — a spurious poll before writable would see Ok(None).
      # Re-probe completion the way accept re-attempts its syscall:
      r = connect(sock, addr)                # second connect on a connecting socket:
      match classify(r):
        Connected  (Ok | EISCONN):           deregister(key); return Ready(Ok(TcpStream::from(sock)))
        InProgress (EINPROGRESS|EALREADY|WouldBlock):
                                             reactor.register_write_waker(key, waker); return Pending
        Failed(e) (else, e.g. ECONNREFUSED): also confirmed by take_error(); return Ready(Err(e))
    Done: return Ready(Err("polled after completion"))
```

`classify` maps the second-`connect()` result: `EISCONN` ⇒ Connected (the kernel says the socket is
already connected — unambiguous, unlike `take_error`); `EALREADY`/`EINPROGRESS` ⇒ still connecting
(spurious poll or not-yet-resolved) ⇒ stay Pending; any other error ⇒ Failed (cross-check with
`take_error`/`SO_ERROR` for the precise errno, e.g. `ECONNREFUSED`).

## Walk-through (paper × algorithm)

- **Immediate (loopback):** Init → `connect()` Ok → `Ready(Ok)`. ✓ matches expected.
- **Async success:** Init → `connect()` EINPROGRESS → register+waker → Pending. Writable edge fires →
  poll → `connect()` ⇒ `EISCONN` ⇒ Connected → `Ready(Ok)`. ✓
- **Spurious poll mid-connect (the bug case):** Init → Pending. *Spurious* poll before writable →
  `connect()` ⇒ `EALREADY` ⇒ InProgress → re-register waker → `Pending` (NOT a half-open stream). ✓
  — the draft's `take_error()` path returns `Ok(None)` here and wrongly yields `Ready(Ok)`. ✗
- **Refused:** Init → EINPROGRESS → Pending. Writable edge (error is also a wakeup) → `connect()` ⇒
  `ECONNREFUSED` ⇒ Failed → `Ready(Err)`. ✓

## Draft divergence (the bug) + fix

The killed-agent draft (net.rs:221-315): (a) wraps the match in a `loop` that never iterates
(`clippy::never_loop`); (b) the `Pending` arm calls only `socket.take_error()` and treats `Ok(None)`
as connected — **ambiguous under a spurious poll** (errno is 0 while still connecting). Fix: drop the
dead `loop`; in `Pending`, re-probe with a second `connect()` and classify via `EISCONN`/`EALREADY`
(unambiguous), falling back to `take_error` only to extract the precise failure errno.

## Code site
`prime/src/os/net.rs` `impl Future for Connect::poll` — Init arm (socket+connect+register) and the
hardened Pending arm (re-probe). Lost-wakeup: register-then-Pending is safe because edge-triggered
ADD reports current readiness (same as the proven `poll_accept`); spurious-poll safety comes from the
re-probe, not from assuming poll⟹writable.

## Test (encodes the worked example)
- `prime_tcp_upstream_connects_and_round_trips_bytes` — async-success path (exists, passes).
- `prime_tcp_upstream_connect_refused_returns_error` — Failed path (exists, passes).
- NEW `connect_pending_survives_spurious_poll` — poll the `Connect` future once *before* the listener
  accepts / before writable, assert it returns `Pending` (not a half-open stream). Locks the bug fix.
