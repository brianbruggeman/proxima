# load — proxima load-testing proxima

*(builds on: transform)*

A load generator built on proxima itself: the **`rekt`** crate (its `rek`
binary and its `rekt_load` example). There is no `examples/load/` rung —
`rekt` lives under `tools/`, so this chapter includes
straight from `tools/rekt/` instead of `examples/`.

## `tools/rekt/README.md` — what rekt is and why it exists

{{#include ../../../tools/rekt/README.md}}

## `tools/rekt/examples/rekt_load.rs` — the closed-loop throughput driver

```rust
{{#include ../../../tools/rekt/examples/rekt_load.rs}}
```
