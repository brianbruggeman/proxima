# benches — throughput/latency harnesses

Reference, not a lesson: `benches/` is 52 Criterion harnesses (throughput,
latency, allocation-counted, CoV-tracked), one concern each — not a single
rung a `main.rs` + `README.md` pair can teach. This chapter does not
enumerate all 52; it includes one representative harness, unmodified, to
show the shape. Every other file in `benches/` follows the same
`criterion_group!`/`criterion_main!` skeleton around a different primitive.

Run the full suite with `wf`, or a single harness directly:

```
cargo bench --bench causal_record
```

## `benches/causal_record.rs` — `CausalIndex::record` / `explain` microbenches

```rust
{{#include ../../../benches/causal_record.rs}}
```
