# proxima TODO

## Payload substrate — no-cell design (2026-07-07)

Status: **design settled, not code.** This note is a public design backlog,
not part of the shipped API contract.

### The decision
The pipe substrate moves **`P` and nothing else** — a pipe is `P -> Q`.
No `Envelope` / `Payload` container cell. No `Body` type.

- **Metadata is a capability, not a slot.** Concerns are composable wrapper types
  `W<P>` — `Header<P>`, `Trailer<P>`, `Framed<P>`, `Sealed<P>` — and each is
  *itself a `P`*, so it pushes into the next layer through the same `P -> Q`
  pipes with zero special-casing.
- **Presence** = `P` (empty is `P = ()`, absence, not a variant).
- **Delivery** = the *type* of `P` (streamed = `P` is a stream type), never an enum arm.
- **Nesting = protocol layering.** HTTP's control-headers vs `Content-*`
  representation-headers split = outer-wrap vs inner-wrap. gzip = `map_inner`
  on the inner wrapper — structurally cannot touch control headers.

### Load-bearing constraint (the thing that can break it)
Wrapper **forwarding**: a pipe that doesn't care about the header must see
through `Header<Bytes>` to `Bytes`, or every wrapper becomes unwrap-boilerplate.
- functor `map_inner` (transform inner, wrapper preserved), and/or
- `Deref<Target = P>` transparency.

### Naming (owner-settled)
- reject `Envelope` (wrapper/contents metaphor breaks under self-similar nesting;
  fights already-landed `body -> payload` vocabulary)
- reject `Metadata` (weasel-word)
- content is `P`; wrapper noun TBD (`Header` favored)
- names (`Body`, `Request`, `Response`) = type aliases; newtype ONLY to enforce an invariant

### Open sub-questions
- [ ] `context` — ambient (context var) vs field
- [ ] dynamic-depth boundary — where the type bottoms out to opaque `Bytes` + `Box<dyn>` at the edge

### Next action — prove it before any code
- [ ] Model **multipart**, a **CONNECT tunnel**, and **h2 frames** as nested wrapper
      `P`s against how proxima does them today. Confirm: no loss, no per-chunk
      metadata, and no pipe hand-unwraps (forwarding holds).
- [ ] Only then scope the migration (blast radius: replaces main's fat
      `Request<P> { method, path, metadata, payload, stream, context }` — every
      listener/upstream/middleware).

---

## Doc consolidation — comb + collapse every .md

**297 tracked `.md` files** (147 under `docs/`). Comb through all of them and
systematically collapse them **one by one** into their precise needed form —
merge overlapping discipline logs, drop superseded/landed records, keep only
what's load-bearing. Do it per-file, not in a bulk sweep.

- [ ] `docs/` (126) — the bulk; per-subsystem discipline logs, likely heavy overlap
- [ ] top-level (`SHAPE.md`, `FEATURES.md`, `parking-lot.md`, `README.md`)
- [ ] per-crate `*.md` (telemetry 5, intercept 5, benches 8, examples 5, ai_docs 3)
- [ ] `spec/`, `scenarios/`

---

## RFC 9112 — authoring form, and the half-done citation migration (2026-07-16)

Two items that look like one. Neither is built.

### 1. The message syntax exists on the wire, not as an authoring surface

proxima already encodes and parses HTTP/1.1 message syntax —
`proxima-protocols/src/http1_codec/h1_client.rs` (`encode_request_head`,
`parse_response_head`) and `h1_body.rs` (chunked framing). What does not exist
is that same syntax as a thing a human *writes* to drive the client:

```
GET /path HTTP/1.1
Host: example.com
Accept: application/json

<body>
```

The codec that reads this shape is already here, so the seam is wiring rather
than new machinery. `.http` / `.rest` files (JetBrains HTTP Client, VS Code
REST Client) are the established convention for the same text, so the form has
prior art and tooling.

Unresolved: where it attaches. An `H1ClientConfig` variant? A `Pipe` whose `In`
is the raw text? A `Spec` form? Config and code are isomorphic here, so the
choice decides whether a request-as-text is config, input, or spec — pick
deliberately.

### 2. The RFC citations mostly point at superseded documents

RFC 9110 (Semantics) and RFC 9112 (HTTP/1.1) replaced RFC 7230/7231 in June
2022. Counts in the tree today:

| cited | count | status |
| --- | --- | --- |
| RFC 7230 | 24 | superseded by 9112 (syntax) + 9110 (semantics) |
| RFC 7231 | 6 | superseded by 9110 |
| RFC 9110 | 6 | current |
| RFC 9112 | 1 | current |

The implementation follows the current syntax; the citations largely do not.
Someone began the migration and stopped. Section numbers were reorganised
between the old and new documents, so this is a read-and-remap job per
citation — a find/replace of the RFC number alone would produce confidently
wrong section references, which is worse than the stale ones.

---

## DynPipe / SendDynPipe — the last two blanket impls (2026-07-17)

Status: **open design question, deliberately unresolved.** `algebra-lint`
flags both and should keep flagging them until this is answered.

`proxima-primitives/src/pipe/alloc_tier.rs` erases a pipe for `dyn` dispatch
through two traits — `DynPipe`, `SendDynPipe` — each a restatement of `Pipe`
with `In`/`Out` demoted to generics, `Err` pinned to `ProximaError`, and the
future boxed. They are implemented as blanket impls over an open set (`impl<P:
Pipe> DynPipe for P`), which the no-blanket rule forbids.

Both remedies are refused, and that is the finding:
- deleting the blanket impl needs a type to host the impl instead — that
  newtype IS the blanket impl renamed (`Erased<P>`, tried and reverted in
  `39e7c035`/`bd69cb5f`: `into_handle(pipe)` was the identical call site
  before and after, so nothing was gained).
- so the defect is upstream: the two traits themselves. The question is
  whether erasure can key off `Pipe` directly (or whether `Pipe` can be made
  object-safe enough) so the restatement disappears, rather than being
  relocated.

Note `alloc_tier.rs` already asserts the equivalence it is compensating for,
ten lines below the impls: `impl<In, Out> SendPipe for dyn SendDynPipe<In,
Out>` — the erased handle IS a pipe of the same form. That round-trip is the
evidence the two traits are redundant, not the justification for them.

Do not "fix" this by exempting it in the lint or by adding an adapter. It is a
real design decision about the erasure boundary.

## Also pending
- Review pre-public cleanup branches and cherry-pick only still-relevant changes.
- Keep scratch worktrees and private assistant state out of public commits.
