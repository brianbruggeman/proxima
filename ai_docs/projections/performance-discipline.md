# Performance Discipline

This projection summarizes `proxima/ai_docs/invariants.jsonl`.

## Rules

- Prefer stack allocation over heap allocation in hot paths.
- Prefer bytes internally; convert to strings only at semantic edges.
- Borrow or view existing data before instantiating owned data.
- Prefer zero-copy first, `Copy` for small semantic value types, and
  `Clone` only when duplication is intentional and measured.
- Use SIMD-backed search primitives such as `memchr`, `memchr2`,
  `memchr3`, `memchr::memmem`, or `aho-corasick` before scalar loops.
- State an allocation budget before landing a performance-sensitive
  component.
- Keep formatting, diagnostics, allocation, and recovery on cold paths.
- Prefer compact data layout over pointer-heavy object graphs.
- Box-free by default (whole workspace): avoid `Box<dyn Trait>`,
  `Box<dyn Future>`, `Box::pin`, and `#[async_trait]`; prefer a
  discriminated enum + match, typestate, generic params, or a
  state-machine future. Legitimate `Box` (open/unbounded dyn set,
  recursion, measured large enum variant) carries a one-line why.
- RPITIT for async/impl-returning trait methods
  (`fn f(&self) -> impl Future` / `async fn` in trait), never
  `#[async_trait]` or `Pin<Box<dyn Future>>`; poll-based `poll_*` is the
  other box-free async surface, preferred for reactor-driven code.

## Exception Rule

Exceptions are allowed only with concrete evidence: a measured bench,
allocation trace, code-level proof from fixed storage and borrowed
outputs, or a documented semantic boundary that requires ownership.

## Review Prompt

Before a component row can pass, answer these questions with data:

- What allocates on the hot path?
- What copies bytes, and why is zero-copy not enough?
- Which values are borrowed, which are owned, and where is the ownership
  boundary?
- Which `Clone` calls remain, and are they cold-path or measured?
- Which byte scans use SIMD-backed primitives?
- Which data structures make steady-state traversal strict O(1)?
