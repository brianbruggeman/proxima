# Transport-unification — runtime-selected wire for `proxima::Client`

> STATUS 2026-06-19: IMPLEMENTED (slices 1A/1B/2, green across prime / tokio-only /
> both-wires builds; sanity-benched). Commits: `002fe80` from_handle, `87601d9`
> tokio dispatch hop, `1c84842` per-upstream `"wire":"tokio"` selector + shared
> tokio sidecar, `e76b66e` latency measurement. Default stays prime;
> `{"http": url, "wire": "tokio"}` dials a tokio-only upstream from a prime
> process. The §5 compat-cell `Direct` (experiment E) is orthogonal and unchanged.
> The original design text below is preserved; the implemented selector is simpler
> than the §4 state machine because the wire is chosen by the per-upstream spec
> field, not inferred — so no PipeHandle reactor-kind tag was needed (the Client
> reads its own spec). Formal criterion gate (§8) remains a follow-up.

> Design proposal. The dispatch-selection state machine at its
> core was vetted by an algorithm-rigor tournament (3-author bundle competition +
> 3-judge Borda); the winning bundle and its locked worked-example test are in
> §4–§6. Status: proposal awaiting the disciplined-component gate in §8.

## 1. The gap

The runtime "umbrella" is already **executor-unified**: `proxima::runtime::Runtime`
(`proxima-runtime/src/lib.rs`) is implemented by `PrimeRuntime`,
`TokioPerCoreRuntime`, and `MockRuntime`, and `Client::builder().runtime(rt)` takes
any of them. `prime-tokio-compat` (P2; `PrimeRuntime::builder().tokio_compat()`)
runs a sister tokio runtime per prime worker so pipes that call `tokio::{spawn,
sync, time}` / hyper keep working **while hosted on prime** — the *prime-hosts-tokio*
direction.

It is **not** transport-unified. Two compile-time facts force an all-or-nothing
build choice:

1. **`src/load.rs:175-191`** registers the `http` upstream factory by mutually
   exclusive `cfg`: `PrimeHttpPipeFactory` (prime H1 over `PrimeTcpUpstream`) **xor**
   the hyper `HttpPipeFactory` (tokio). Exactly one wire per build.
2. **`Client::dispatch` (`src/client/handle.rs:185-195`)** gates the off-worker hop
   (`call_on_worker`) behind `#[cfg(feature = "runtime-prime")]`. On a tokio build
   that branch is compiled out and dispatch is direct-on-ambient — so an **injected
   `Runtime` is ignored on the tokio path**.

Consequence: a process cannot dial prime *and* tokio wires from one `Client`, and a
**tokio-hosted application** (an event-loop GUI, an embedder that already owns a
tokio runtime) cannot ask `proxima::Client` to ride *its* runtime — it must either
rebuild proxima with the tokio transport (the wire then rides the host tokio by
ambient dispatch, one runtime) or accept a second, client-owned `shared_prime`
runtime beside the host's. Neither is runtime-selectable.

## 2. Where this slots

`docs/runtime-prime/discipline-prime-tokio-compat.md` tracks the tokio-elimination
plan. P2 = `prime-tokio-compat` = **prime-hosts-tokio** (landed, parked on its bench
gate). This proposal is the symmetric and superset capability:

- **tokio-hosts-proxima** — a `TokioPerCoreRuntime::from_handle(Handle)` that wraps
  the host's already-running tokio runtime as a `Runtime` (instead of spawning its
  own threads, which `::new` does today), so an injected host runtime is honored.
- **runtime-selected transport** — both http factories register; the resolved
  `PipeHandle` carries a `reactor_kind` tag; dispatch routes each request so the
  stream is constructed+polled where its reactor is *driven*.

It depends on the same residency invariant P2 realizes via `EnterGuard`, and reuses
P2's sister-runtime handle (`tokio_compat_handles()`, `prime/src/os/runtime.rs`) as a
hop target. Call it **P-TU** in the plan; it lands *after* P2's gate clears (it reads
P2's compat machinery) and subsumes the executor-only umbrella.

## 3. The two laws (the oracle)

The decision is governed by two laws; every cell's expected output derives from them
(principle 14), never from preference.

- **L1 — reactor residency (correctness).** The future that *constructs and polls* a
  transport stream MUST run on a thread where the matching reactor is **driven** —
  not merely *enterable*. `PrimeStream` ⟹ a prime worker (`current_core().is_some()`);
  `TokioStream` ⟹ a thread that *drives* a tokio reactor. Violation is a panic
  (`there is no reactor running, must be called from the context of a Tokio 1.x
  runtime`) or the prime mirror.
  - **L1-compat (derived from `prime/src/os/tokio_compat.rs`).** A prime worker built
    `tokio_compat()` holds a `'static EnterGuard`, so `Handle::current()` *resolves*
    on it — but the tokio mio reactor is **driven on a separate sister OS thread**.
    Therefore a compat worker is **prime-resident** (its CoreShard reactor is driven
    inline) but **NOT tokio-driven inline**. "Enterable" ≠ "driven here." This is the
    load-bearing subtlety: a residency probe of `Handle::try_current().is_ok()` is
    *unsound* here — it is true on a compat worker yet an inline `TokioStream` poll's
    readiness is delivered by the sister thread.

- **L2 — parity (principle 14).** For the cells the incumbent already handles,
  reproduce its path exactly. Incumbent (`handle.rs:185-195`, `call_on_worker`
  `:203-226`, `shared_prime_runtime` `:482`), which only ever sees `PrimeStream`:
  off-worker ⟹ `HopTo(injected ?? shared_prime, CoreId(0))`; on-worker ⟹ `Direct`.

## 4. The algorithm (vetted winner)

State space: `injected: Option<{Prime, TokioHostHandle, TokioPerCore}>` ×
`ambient: {OnPrimeWorker, OnPrimeWorkerCompat, OnTokioReactor, BareThread}` ×
`transport: {PrimeStream, TokioStream}`. Output:
`DispatchPath ∈ {Direct, HopTo(Runtime, CoreId), Error(reason)}`. `CoreId` is always
`0` on a hop (incumbent invariant).

```
fn dispatch_path(injected, ambient, transport) -> DispatchPath {
    let needed = reactor_kind(transport)           // PrimeStream->Prime, TokioStream->Tokio

    // honor an explicit injection that cannot host the wire — fail loud, never reroute.
    if injected == Some(Prime) && needed == Tokio { return Error("prime injected, tokio stream") }

    // L1: poll in place iff the matching reactor is DRIVEN on this thread.
    if is_resident(ambient, needed) { return Direct }

    // off-thread: hop onto a runtime whose reactor kind matches `needed`.
    pick_target(injected, ambient, needed)
}

fn is_resident(ambient, needed) -> bool {
    match (ambient, needed) {
        (OnPrimeWorker,       Prime) => true,
        (OnPrimeWorkerCompat, Prime) => true,   // prime reactor driven here
        (OnTokioReactor,      Tokio) => true,
        (OnPrimeWorkerCompat, Tokio) => true,   // E (2026-06-19): inline connect completes — sister reactor wakes the inline future. was conservatively FALSE pre-E.
        _ => false,
    }
}

fn pick_target(injected, ambient, needed) -> DispatchPath {
    match needed {
        Prime => match injected {
            Some(Prime(rt)) => HopTo(rt, 0),
            _               => HopTo(shared_prime(), 0),   // L2: never Error a PrimeStream
        },
        Tokio => match injected {
            Some(TokioHostHandle(h)) => HopTo(TokioPerCoreRuntime::from_handle(h), 0),
            Some(TokioPerCore(rt))   => HopTo(rt, 0),
            None if ambient == OnPrimeWorkerCompat
                                     => HopTo(sister_tokio(), 0),   // conservative; see §5
            None                     => HopTo(shared_tokio(), 0) or Error("no tokio runtime"),
            Some(Prime(_))           => unreachable, // handled by the top guard
        },
    }
}
```

### Decision table (the worked example — the spec)

`C0 = CoreId(0)`. Expected paths are derived by hand from L1/L1-compat/L2, *independent
of the function* (so the test in §6 is not tautological).

| injected | ambient | transport | → path | law |
| --- | --- | --- | --- | --- |
| Prime | OnPrimeWorker | Prime | Direct | L2 on-worker |
| Prime | OnPrimeWorkerCompat | Prime | Direct | L1 prime driven here |
| Prime | OnTokioReactor | Prime | HopTo(InjectedPrime, C0) | L2 off-worker |
| Prime | BareThread | Prime | HopTo(InjectedPrime, C0) | L2 off-worker |
| Tokio* | OnPrimeWorker\|Compat | Prime | Direct | L2 on-worker (injection ignored when resident) |
| Tokio* | OnTokioReactor\|BareThread | Prime | HopTo(shared_prime, C0) | L2: PrimeStream never Errors → shared_prime fallback |
| None | OnPrimeWorker\|Compat | Prime | Direct | L1 |
| None | OnTokioReactor\|BareThread | Prime | HopTo(shared_prime, C0) | L2 (canonical incumbent off-worker) |
| Prime | * | Tokio | Error("prime injected, tokio stream") | top guard — honor injection, fail loud |
| TokioHostHandle | OnTokioReactor | Tokio | Direct | L1 tokio driven here |
| TokioHostHandle | OnPrimeWorker\|Compat\|Bare | Tokio | HopTo(from_handle, C0) | L1: hop to host tokio |
| TokioPerCore | OnTokioReactor | Tokio | Direct | L1 |
| TokioPerCore | OnPrimeWorker\|Bare | Tokio | HopTo(TokioPerCore, C0) | L1 |
| None | OnTokioReactor | Tokio | Direct | L1 |
| None | **OnPrimeWorkerCompat** | Tokio | **Direct** (E: completes) | **E refuted the hang — was conservative HopTo pre-E; see §5** |
| None | OnPrimeWorker | Tokio | HopTo(shared_tokio, C0) | L1 |
| None | BareThread | Tokio | HopTo(shared_tokio, C0) or Error | L1; Error if no shared tokio in build |

**Parity by restriction.** Set `transport ≡ PrimeStream` and drop the compat/tokio
ambients (a prime-only build never produces them). Then `needed ≡ Prime`, the top
guard never fires, `is_resident` reduces to `current_core().is_some()`, and
`pick_target(Prime)` reduces to `injected_prime ?? shared_prime` at `C0`. That is
`dispatch` + `call_on_worker` line-for-line. ∎

## 5. The one contested cell — `(OnPrimeWorkerCompat, TokioStream)`

> **RESOLVED BY EXPERIMENT E — RUN 2026-06-19, result `completed=true ok=true`.**
> The contested premise (an inline tokio poll on a compat worker silently hangs
> because the mio reactor is on the sister thread) is **REFUTED for the connect
> case**: `Direct` **completes**. Cross-thread waker propagation from the sister
> reactor to a future polled inline on the prime worker works. So
> `(OnPrimeWorkerCompat, TokioStream)` is **Direct**, not the conservative Hop — the
> tournament's "silent-hang" finding (the decisive axis on which the unanimous Borda
> verdict turned) did not survive measurement. This is the principle-16 correction:
> the verdict was scored on a plausible claim no agent had run.

The original (now-refuted) reasoning, kept for the record: the sister mio reactor is
*driven* on a separate OS thread, so it was unclear whether a stream registered there
but polled inline on the prime worker would ever receive its readiness. It does.

**Caveat (what E proves and does not).** E exercises a single inline `connect` on a
**non-inverted** compat worker — the critical boundary (an I/O readiness event from
the sister reactor waking an inline-polled future). It does **not** yet cover a
full read/write round-trip under load, nor the `prime-tokio-compat-inverted` worker.
A read/write-under-load extension (needs tokio `io-util`) fully closes it; the
direction is unambiguous (Direct works, does not hang).

> **Experiment E** (now a passing test:
> `prime::os::runtime::tests::experiment_e_inline_tokio_connect_on_compat_worker`,
> features `prime-tokio-compat,prime-tokio-compat-inverted,runtime-prime-executor,
> runtime-prime-reactor,runtime-prime-inbox-alloc,runtime-prime-bgpool`): builds
> `PrimeRuntime::new_with_tokio_compat(1)`; on a compat worker (`CoreId(0)`) inline
> `tokio::net::TcpStream::connect` to a loopback acceptor under a 2s bound; records
> completion. GREEN.

The console / GUI case (a tokio-hosted app) never reaches this cell anyway — that
ambient is `OnTokioReactor` or `BareThread`, not a compat prime worker.

## 6. Locked test (non-tautological)

Two parts. The decision-table test asserts the §4 hand-derived spec (oracle = the
LAWS, not the function). The live test enforces L1 against a real reactor.

- **Table test** — `#[rstest] case::…` over the §4 cells, asserting the exact
  `DispatchPath` and `CoreId(0)` on every hop. Fails under wrong selection-ordering
  (residency vs guard vs target), a swapped-reactor `is_resident`, or a skipped
  parity hop. The compat-None×Tokio row asserts `HopTo(sister_tokio)` (the conservative
  default), flipping to `Direct` only after E licenses it.
- **Live residency gate** — build a real 1-core `PrimeRuntime`, serve a loopback echo,
  dial it with a runtime-less `Client` from a bare thread; assert the request
  *completes* (not merely that a path was chosen). This is what A's tautological
  table-test (oracle == its own output) could never catch.
- **Experiment E** (§5) — the runnable Direct-vs-Hop decider.

## 7. The seam (code-mapping — honest about scope)

This is larger than "remove the `#[cfg]` guard":

1. **Runtime-tagged dual-factory registry** replacing `load.rs:175-191`: both
   `PrimeHttpPipeFactory` and the hyper `HttpPipeFactory` register; the resolved
   `PipeHandle` carries `reactor_kind: ReactorKind`. This is what `reactor_kind(transport)`
   reads. The transport choice moves from compile-time `cfg` to a per-resolution tag.
2. **Probe order** — `dispatch` resolves `self.handle().await` *first* (the factory that
   won decides the stream type), then probes `reactor_kind` + `ambient`.
3. **Ambient probe** — `current_core().is_some()` + a new `on_compat_worker()`
   thread-local (set by the compat `WorkerSetup`) to split `OnPrimeWorker` from
   `OnPrimeWorkerCompat`, and a `tokio_reactor_driven_here()` that detects a *driven*
   tokio reactor (a worker / active `block_on`), explicitly NOT an `EnterGuard`
   (`Handle::try_current().is_ok()` alone is the unsound probe — it is true on a compat
   worker). Getting this inverse right is itself a correctness obligation.
4. **`hop(runtime, handle, request)`** — lift `call_on_worker` (`handle.rs:215-226`:
   oneshot + `spawn_on_core(CoreId(0), …)` + receiver) into one helper shared by prime
   and tokio targets (both impl `Runtime`).
5. **New `Runtime` constructors** — `TokioPerCoreRuntime::from_handle(Handle)` (wraps an
   existing host runtime, no new threads — the missing primitive for tokio-hosts-proxima)
   and `shared_tokio_runtime()` (mirror of `shared_prime_runtime`).

`dispatch`'s `#[cfg(feature = "runtime-prime")]` guard is removed; the build is unified,
so the decision is data-driven, not cfg-driven.

## 8. Disciplined-component gate (what must clear before P-TU lands)

| cell | claim it proves |
| --- | --- |
| build / clippy / parity test | the §6 table test passes; PrimeStream restriction is bit-identical to the incumbent (L2). |
| live residency gate | a real off-worker PrimeStream dial completes on the prime reactor (L1), not just a chosen path. |
| experiment E | the compat-None×Tokio cell's Direct-vs-Hop fact, recorded. |
| micro-bench: hop overhead | the generalized `hop()` is within ±5% of the incumbent `call_on_worker` (no regression on the parity path). |
| micro-bench: probe cost | the per-request `ambient` + `reactor_kind` probe is sub-µs (it runs on every dispatch). |
| compare-bench: one-runtime vs two | tokio-hosts-proxima (host runtime injected) vs the shared-prime-beside-host baseline — the win this capability claims. |

## 8b. Verified post-implementation (2026-06-19, branch feat/ptu-compat-tests)

- **Prime *serve* survives the both-wires build (no unification wall).** Running
  `cargo test --test prime_serve --features "http-hyper,tokio-runtime,runtime-tokio"`
  passes — a process can serve on prime while the tokio wire is compiled in. The
  early-session "no reactor on prime core" panic does NOT recur on the proxima prime
  serve path. This was the load-bearing risk for a serving daemon; cleared.
- **`wire:tokio` over `https://` routes through hyper+TLS on the sidecar.**
  `tests/ptu_https_wire.rs`: a self-signed TLS loopback is reached over TCP
  (`accepts=1`) and the dial fails at the TLS layer (webpki roots correctly reject
  the self-signed cert). Real, publicly-CA-signed upstreams (provider APIs) would
  succeed. **GOTCHA:** `http-hyper` alone is PLAINTEXT — https needs the `tls`
  feature too (`proxima-h1/tls` → hyper-rustls `HttpsConnector` with webpki roots).
  A self-signed/internal upstream would need a custom-CA/insecure hook on the hyper
  client, which does NOT exist yet (a follow-up if internal-TLS compat is needed).

## 9. Risk register (weakest tournament axes)

- **Minimality (the winner's weakest axis).** The dual-factory registry + runtime-tagged
  handle is a real, non-trivial seam. It is necessary (it *is* the transport-unification),
  but it is the heaviest part. Keep `from_handle` / `shared_tokio` thin.
- **`None × BareThread × TokioStream` (table last row): `shared_tokio` fallback vs hard
  Error.** A deliberate choice — lazily leasing a process-shared tokio runtime vs failing
  loud when no tokio runtime is reachable. Defaulting to a fallback softens the
  no-silent-failure stance; decide explicitly when implementing (recommend: Error unless a
  `shared-tokio` feature is on, mirroring how `shared_prime` is always available under
  `runtime-prime`).
- **`tokio_reactor_driven_here()` probe.** It must distinguish a *driven* tokio reactor
  from a bare `EnterGuard` for the OTHER ambients — but note E (§5) shows that for the
  compat worker specifically, inline poll works regardless, so this probe is a
  classification nicety there, not a correctness gate.
- **E covers connect only.** Experiment E is run and green for an inline `connect` on a
  non-inverted compat worker; a read/write-under-load + inverted-worker extension fully
  closes it. Direction is unambiguous.

## 10. Provenance — and the correction measurement forced

Dispatch-selection algorithm vetted by `/algorithm-rigor`: incumbent A (3-state ambient,
`try_current` residency probe), blind author B (4-state ambient with `OnPrimeWorkerCompat`,
hand-derived cells), synthesis (B-base + A's parity column + re-walk corrections +
conservative compat cell). 3-judge Borda: **synthesis 6, B 3, A 0 — unanimous**, on the
"decisive axis" that the synthesis treats the compat × TokioStream cell as a silent-hang
risk and hops it conservatively.

**Then experiment E was run and refuted that decisive axis** (`completed=true`): an inline
`Direct` poll on a compat worker COMPLETES — there is no silent hang for connect. So the
unanimous verdict was correct about *form* (B's 4-state model, the parity reconciliation —
a `PrimeStream` never Errors off-worker, found by the tournament not the first pass) but
WRONG about its headline *correctness* claim, because no agent ran the experiment. The
durable lesson is principle 16: a 3-0 judge sweep on an unmeasured claim is a hypothesis,
not a result. Author B's blind instinct (compat worker is tokio-usable inline) beat the
synthesis on the one cell that mattered — and only measurement showed it.
