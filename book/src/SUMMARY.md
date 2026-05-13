# proxima by example — table of contents

[proxima by example](index.md)

# Start here

- [hello](start/hello.md)

# The pipe algebra

- [overview — form · chain · primitive · pattern](algebra/index.md)
  - [transform — the pipe and its four forms](algebra/transform.md)
  - [send — the same pipe across a core (the tier)](algebra/send.md)
  - [filter — a primitive: pass or drop](algebra/filter.md)
  - [fan-out — a primitive: one to many](algebra/fan-out.md)
  - [fan-in — a primitive: many to one](algebra/fan-in.md)
  - [gate — a pattern: readiness by composition](algebra/gate.md)
  - [signal — completion on the substrate](algebra/signal.md)
  - [the pattern gallery — retry · auth · iam · wal · cron · etl](algebra/patterns.md)

# Configure it

- [config](configure/config.md)

# Make it resilient

- [clock](resilience/clock.md)
- [retry](resilience/retry.md)
- [backoff](resilience/backoff.md)
- [rate-limit](resilience/rate-limit.md)
- [circuit-breaker](resilience/circuit-breaker.md)
- [deadline](resilience/deadline.md)
- [fallback](resilience/fallback.md)

# Flow & delivery

- [backpressure](flow/backpressure.md)
- [cancellation](flow/cancellation.md)
- [delivery](flow/delivery.md)
- [best-effort](flow/best-effort.md)

# Prove it holds

- [chaos](prove/chaos.md)
- [fuzz](prove/fuzz.md)
- [differential](prove/differential.md)

# Fake & replay

- [record](fake-replay/record.md)
- [replay](fake-replay/replay.md)
- [cache](fake-replay/cache.md)

# Observe it

- [logs](observe/logs.md)
- [metrics](observe/metrics.md)
- [traces](observe/traces.md)
- [instrument](observe/instrument.md)
- [export](observe/export.md)
- [distributed-trace](observe/distributed-trace.md)

# Runtimes

- [runtime-select](runtimes/runtime-select.md)
- [multi-runtime](runtimes/multi-runtime.md)

# The frontier

- [no-std](frontier/no-std.md)
- [new-platform](frontier/new-platform.md)
- [wasm](frontier/wasm.md)
- [dpdk](frontier/dpdk.md)
- [spdk](frontier/spdk.md)
- [pmem](frontier/pmem.md)

# Extend it

- [plugin](extend/plugin.md)
- [codec](extend/codec.md)

# Applied — build a real thing

- [proxy](applied/proxy.md)
- [gateway](applied/gateway.md)
- [load-balance](applied/load-balance.md)
- [crud](applied/crud.md)
- [integration](applied/integration.md)
- [load](applied/load.md)

# Reference — baselines & tools

- [floor](reference/floor.md)
- [boot-time](reference/boot-time.md)
- [benches](reference/benches.md)
- [proxima-main](reference/proxima-main.md)
