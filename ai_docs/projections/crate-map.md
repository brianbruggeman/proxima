# the crate map

**Audience:** anyone — agent or person — who needs to know which crate owns a
thing before touching code, and in which direction the dependencies point.

The prose introduction to the same map lives in
[`README.md`](../../README.md#the-crate-map). This page is the version with the
edges written down, because the question an agent actually asks is "may X
depend on Y", and that is answered by the layer, not by the name.

35 workspace members that ship code, in seven groups. **A crate may depend on
crates in its own group or any group above it, never below.** Every edge below
was read out of `cargo metadata`, not inferred from the names — and the names
mislead in one specific place, called out under primitives.

## primitives

The algebra and what it stands on.

| crate | owns | depends on (internal) |
|---|---|---|
| `proxima-core` | error, ring, arena, time, markers | *nothing* — the root |
| `proxima-runtime` | the `Runtime` trait: per-core spawn, cross-core dispatch, background pool | core |
| `proxima-macros` | `#[proxima::piped]`, `#[proxima::instrument]`, `#[proxima::main]`, `#[proxima::test]` | *nothing* |
| `proxima-config` | format registry (JSON/JSON5/TOML/YAML/RON/XML), typed schema IR, desired-state store | core, macros |
| `proxima-primitives` | **`Pipe`**, `SendPipe`, `UnpinPipe`, `UnpinSendPipe`, `Handler`, `SourcePipe`, `Body`, `HeaderList`, sans-IO byte-stream traits, concurrency primitives | core, config, runtime |
| `proxima-clock` | time as a pipe — tick sources, wall-clock anchoring, coarse shared-time cell | primitives |
| `proxima-telemetry` | traces, metrics, logs | core, clock, macros, primitives |
| `proxima-codec` | body marshalling registry | core |
| `proxima-build` | `build.rs` profile resolution, const + cfg emission | *nothing* |
| `proxima-test` | the `#[proxima::test]` harness | prime, runtime |

**The naming trap.** `proxima-core` is *not* where the pipe algebra lives, and
`proxima-primitives` is not beneath it. `proxima-core` is the leaf (error, ring,
arena) and `proxima-primitives` depends on it. `Pipe` is in
`proxima-primitives`. Anything reasoning from the names alone gets this
backwards.

## patterns

Pipes wired into a shape. No wire of their own.

| crate | owns | depends on (internal) |
|---|---|---|
| `proxima-auth` | token lifecycle, SigV4, RFC 7616 Digest, SPNEGO — sans-IO FSMs | *nothing* |
| `proxima-recording` | event model, binary + JSONL formats, capture, causal replay | core, primitives, protocols, runtime, telemetry |
| `proxima-patterns` | one feature per pattern: `alert`, `balancer`, `middleware`, `control_plane`, `kv` | auth, core, primitives, runtime, telemetry |

`proxima-auth` depends on no proxima crate at all. Its docs describe "an FSM in
the middle with `Pipe` at the edges" — the caller supplies the pipes; the crate
does not import them.

## protocols

Sans-IO. Bytes in, frames out; never touches a socket.

| crate | owns | depends on (internal) |
|---|---|---|
| `proxima-protocols` | 24 feature-gated codecs: HTTP/1, HTTP/2 + HPACK, HTTP/3, QUIC, TCP/IP, WebSocket, redis, pgwire, DNS, MQTT, AMQP, Kafka, memcached, gRPC framing, protobuf wire, JSON-RPC, PROXY protocol, NVMe, inet, process, time | codec, core, macros, primitives |
| `proxima-centauri` | IKE-style key agreement, rekey, stateless cookies, ESP child SA with AEAD + anti-replay. No-alloc; entropy and time are pipe inputs | clock, primitives |

## stacks

A codec plus a listener plus a client, so `proxima::Client` and
`proxima::Listener` speak the wire. Each protocol stack splits into a `client`
and a `listen` feature; `--no-default-features` leaves the bare codec.

| crate | depends on (internal) |
|---|---|
| `proxima-tls` | core, primitives |
| `proxima-listen` | codec, core, net, primitives, protocols, runtime, telemetry, tls |
| `proxima-quic` | prime, core, listen, primitives, protocols, telemetry |
| `proxima-http` | prime, core, listen, net, primitives, protocols, quic, runtime, telemetry, tls |
| `proxima-redis` `proxima-dns` `proxima-kafka` `proxima-mqtt` `proxima-amqp` `proxima-memcached` | core, listen, primitives, protocols, runtime, telemetry (+ codec for dns/kafka) |
| `proxima-pgwire` | auth, core, listen, net, primitives, protocols, runtime, telemetry, tls |

Every protocol stack depends on `proxima-listen` and `proxima-protocols` and on
nothing from a lower group. That edge set is the load-bearing one: it is what
makes "the codec is reusable without the listener" true rather than aspirational.

## backends, as features

A backend is a feature flag, never a second API.

| crate | owns | depends on (internal) |
|---|---|---|
| `prime` | the per-core runtime: one thread per core, no work-stealing, bounded SPSC inbox, reactor + executor + timer + background pool | clock, core, primitives, runtime |
| `proxima-net` | UDP `PacketListener`, addressing, and the platform backends `prime` / `tokio` / `wasm` / `dpdk` / `xdp` as feature-gated modules | prime, core, primitives, protocols |

## substrates

Where bytes land, where processes run.

| crate | owns | depends on (internal) |
|---|---|---|
| `proxima-storage` | NVMe queue-pair engine (`nvme`), pmem crash-consistency leaf (unconditional, `no_std` + no-alloc), DAX mmap facade (`dax`) | primitives, protocols |
| `proxima-process` | capability-typed commands, PTY and FD pipes, libc shim dispatch, host grounds | primitives, protocols |

`proxima_storage::nvme::QueuePair` is a `Pipe`. A block device is not a special
kind of object at this layer.

## apps

Things you run.

| crate | is | depends on (internal) |
|---|---|---|
| `proxima` | the umbrella library, and `proximad` | 24 of the above |
| `proxima-cli` | the `proxima` binary: call, serve, describe, daemon control, hot-swap, replay | proxima |
| `proxima-intercept` | TLS-terminating CONNECT proxy, per-host cert generation | core, http, primitives, protocols, recording, tls |
| `rekt` | the `rek` binary — a load tester built on proxima | net, primitives, recording, runtime, telemetry |
| `proxima-vm` | a VM-backed `Pipe` proof surface over KVM / Hypervisor.framework | core, primitives, protocols |

## folded crate names that no longer exist

These were real crates and are gone. All twelve are absent from `Cargo.lock`.
If you meet one in a doc comment or an older note, it is history, not a
dependency you can add:

| gone | folded into |
|---|---|
| `proxima-pipe`, `proxima-stream`, `proxima-sync`, `proxima-transport` | `proxima-primitives` |
| `proxima-notify`, `proxima-balancer`, `proxima-middleware`, `proxima-control-plane`, `proxima-kv` | `proxima-patterns` |
| `proxima-nvme`, `proxima-pmem`, `proxima-pmem-dax` | `proxima-storage` |

`proxima-h2`, `proxima-h3`, `proxima-quic-proto` and `proxima-h3-proto` are
likewise gone: the codecs are `proxima-protocols` modules and the I/O facades
are `proxima-http` and `proxima-quic`. `proxima-lock` folded away too — the
tier-resolved blocking `Mutex` is reachable through `proxima-primitives`.
