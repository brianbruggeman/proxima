# integration — front, fake, and replay a third-party API

## Builds on

- [proxy](../proxy/README.md) — the edge is the same shape: a pipe that forwards
  via `Client`. Here that pipe is `RecordUpstream<Client>` instead of a bare
  `Client` forward.
- [record](../record/README.md) — the cassette tee: `RecordUpstream` wraps the
  forward and writes every interaction to disk as it flows.
- [replay](../replay/README.md) — the fake: `ReplayUpstream` reads that cassette
  back and becomes a `SendPipe` in its own right, no upstream call required.

## What it demonstrates

Integrating a third-party API is these three composed, not new machinery.
Phase 1 mounts `RecordUpstream<Client>` as the edge's own pipe — a real
`App` on a real bind address, fronting a real vendor over real HTTP, tee'd
onto a cassette as it goes. Phase 2 drains the vendor's `App` for real (it is
gone, not mocked-out), rebuilds the edge from `ReplayUpstream` alone at the
*same* bind address, and proves the same client code gets back the same
answer with no vendor call made.

| phase | mechanism | what serves the request |
|---|---|---|
| 1 — LIVE | `RecordUpstream<Client>` fronting the vendor over real HTTP | the vendor, tee'd to a cassette on the way through |
| 2 — REPLAY/FAKE | `ReplayUpstream` loaded from that cassette | the cassette; no vendor process exists anymore |

The rigorous proof is byte-for-byte and framing-agnostic: the cassette is
read back directly (`JsonlSource`, concatenating `ResponseChunk` events —
the same technique `record`'s own proof uses) and compared with
`assert_eq!` against what `ReplayUpstream` serves for the identical request,
called in-process. That comparison can't be fooled by transport details
(chunking, header order) because neither side goes through a socket. A
second, softer check — hitting the edge's real bind address with a plain
blocking `GET`, once live and once faked — proves the fake is also a genuine
drop-in over the wire, not just an in-process trick.

## Run

```
cargo run --example integration
```

Same `runtime-prime-*` quartet as `hello`/`proxy`/`record`/`replay` — no
extra features.

## What you'll see

```
phase 1: LIVE — front the vendor, record every response

vendor (third-party) listening on 127.0.0.1:8095
edge (live front) listening on 127.0.0.1:8096, forwards to 127.0.0.1:8095

client -> edge -> vendor:
HTTP/1.1 200 OK
x-vendor: acme-quotes-api
...
{"symbol":"ACME","price":42.17}

awaiting terminal.drained() before the cassette is read back...
edge (live) drained: cores_acked=1 hooks_drained=0
vendor drained: cores_acked=1 hooks_drained=0 -- the vendor is now GONE

phase 2: REPLAY — serve the capture, no vendor required

cassette loaded, known match keys: ["GET /?"]
in-process proof: 32 bytes recorded == 32 bytes replayed, no vendor call made

same client, same address 127.0.0.1:8096, vendor is dead:
HTTP/1.1 200 OK
x-vendor: acme-quotes-api
...
{"symbol":"ACME","price":42.17}

edge (fake) drained: cores_acked=1 hooks_drained=0

PASS: acme-quotes-api was fronted live, recorded, and replayed byte-identical with the vendor removed.
```

The vendor's `App` is drained before phase 2 ever builds a `ReplayUpstream` —
by the time the fake answers, the process that could have answered for real
no longer exists. The 32-byte body and `x-vendor` header are identical in
both phases; the edge's own bind address never changes.
