# proxima-h3 docs

`proxima-h3` folded into the workspace crate consolidation: the std/
runtime stack is now `proxima-http::http3`, and the sans-IO wire codec
is `proxima-protocols::http3_codec`. The docs below (and the
proxima-quic cross-references) still describe the design; only the
crate names moved.

H3 lives inside the unified QUIC + H3 rewrite initiative. The
discipline log, RFC reference, allocation-budget table, performance
targets, edges, and bench logs all live in
[`proxima/docs/proxima-quic/`](../proxima-quic/) — H3 components
(C32–C41) are tracked there alongside the QUIC components.

H3-specific axioms (RFC 9114 / 9204 / 9220 / 9297 invariants; QPACK
dynamic-table cap policy; H3 stream-state-machine shape) live in the
`/guiding-principles` skill under the "proxima-h3 axioms" section,
alongside the workspace principles and the proxima-quic overlay.

- Unified initiative log: [`../proxima-quic/discipline.md`](../proxima-quic/discipline.md)
- Unified RFC reference: [`../proxima-quic/rfc-reference.md`](../proxima-quic/rfc-reference.md)
- Unified alloc budget: [`../proxima-quic/alloc-budget.md`](../proxima-quic/alloc-budget.md)
- Unified perf targets: [`../proxima-quic/perf-targets.md`](../proxima-quic/perf-targets.md)
- Unified edges: [`../proxima-quic/edges.md`](../proxima-quic/edges.md)
- Bench logs: [`../proxima-quic/bench-logs/`](../proxima-quic/bench-logs/)
