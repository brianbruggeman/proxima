# bench-tokio-compat results — m1

Measures whether `prime+compat` costs 15-25% on multi-conn h2 vs
`TokioPerCoreRuntime` (the apples-to-apples baseline). Three libraries
(hyper, pingora, proxima) × three workloads (single_stream, h2_fanin,
multicore_fanin). `per_core` column = TokioPerCoreRuntime; `compat/per_core`
ratio > 1.0x means compat is slower by that factor.

Run `scripts/bench-tokio-compat.sh` to populate this file.

---

## single_stream (1 req/iter, 1 TCP connection)

| library | current_thread | multi_thread | per_core | prime_compat | compat/per_core ratio | p99 (compat) | p99 (per_core) |
|---------|---------------|-------------|----------|-------------|----------------------|-------------|----------------|
| hyper   | pending | pending | pending | pending | pending | pending | pending |
| pingora | pending | pending | pending | pending | pending | pending | pending |
| proxima | pending | pending | pending | pending | pending | pending | pending |

---

## h2_fanin (32 concurrent streams, 1 TCP connection)

| library | current_thread | multi_thread | per_core | prime_compat | compat/per_core ratio | p99 (compat) | p99 (per_core) |
|---------|---------------|-------------|----------|-------------|----------------------|-------------|----------------|
| hyper   | pending | pending | pending | pending | pending | pending | pending |
| pingora | pending | pending | pending | pending | pending | pending | pending |
| proxima | pending | pending | pending | pending | pending | pending | pending |

---

## multicore_fanin (4 ports × 8 streams = 32 total — the key cost-model workload)

| library | current_thread | multi_thread | per_core | prime_compat | compat/per_core ratio | p99 (compat) | p99 (per_core) |
|---------|---------------|-------------|----------|-------------|----------------------|-------------|----------------|
| hyper   | n/a | pending | pending | pending | pending | pending | pending |
| pingora | n/a | pending | pending | pending | pending | pending | pending |
| proxima | n/a | pending | pending | pending | pending | pending | pending |

---

_Run `scripts/bench-tokio-compat.sh` to replace `pending` values with measured results._
