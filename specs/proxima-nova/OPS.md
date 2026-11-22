# shared ops (c14) — inventory of omega and nova, and the O2 slice

status: inventory complete 2026-11-21 (card PROXIMANOVA-53). Every row was read from the tree
on branch `feat/proxima-nova`; the `file:line` is the definition site. The overlap is the exact
set, not a guess: `ops=28 shared=6 omega_only=11 nova_only=11 cited=28 of=28` (counting
comparison conditions as separate nova leaves and width-changes as nova leaves; `cited` counts
table rows carrying a `file:line`).
goal: `pm/GOALS.md` `shared-ops:c14`; requirement shape in `SPEC.md` "shared ops (c14)".

## 1. the two tables

omega's scalar vocabulary is one closed enum, `ScalarOp` (`proxima-tensor/src/op.rs:77-95`),
a body of `Op::Elementwise` and `Reduce` (the tensor graph at `op.rs:9-17` is backward-only
and stays that way; card PROXIMANOVA-43 chose "share the table, never the graph").

nova's scalar vocabulary is `Inst` (`proxima-nova/src/ir/inst.rs:22`): integer arithmetic,
comparison conditions, width changes and memory/call forms. It has no float form today
(`F64` is refused, `tests/refused/float.rs`).

The CPU reference for omega is `proxima-tensor/src/cpu/typed_eval.rs:100-115` (the integer arm
block) and `:280-300` (f64); the width-specialised kernels repeat the same arms at
`proxima-tensor/src/cpu/width_kernels.rs:360-380`. Each emitter repeats it again with its own
string template: MSL `omega/src/msl/signature_tokens_prelude.rs:626`, WGSL
`omega/src/wgsl.rs:769`, CUDA `omega/src/cuda.rs:1854`, and the name table
`omega/src/identity.rs:503`. nova repeats it in `proxima-nova/src/aarch64/lower.rs:410`
(add/sub), `:477` (div) and `proxima-nova/src/aarch64/encode.rs:214` (`MachInst::Add`). The
six are therefore spread across these arms — omega: `typed_eval.rs:100-115` (CPU),
`width_kernels.rs:360-380`, `signature_tokens_prelude.rs:626`, `wgsl.rs:769`, `cuda.rs:1854`,
`identity.rs:503`; nova: `lower.rs:410` (Iadd/Isub), `:465` (Imul), `:477` (Sdiv/Udiv), `:494`
(Icmp), `encode.rs:214` (Add), `:286` (Mul), `:303` (Cmp), `:319` (Cset). No count is claimed:
what matters is that no single site defines all six.

## 2. the three sets

| op | omega (`op.rs`) | nova (`ir/inst.rs`) | class |
|---|---|---|---|
| `Add` | `:79` | `Iadd` `:27` | **shared** |
| `Subtract` | `:80` | `Isub` `:28` | **shared** |
| `Multiply` | `:81` | `Imul` `:29` | **shared** |
| `Divide` | `:82` | `Sdiv` `:31` / `Udiv` `:33` | **shared** (signedness is the width/type, not a second op) |
| `Greater` | `:92` | `Icmp { Sgt/Ugt }` `:35` | **shared** |
| `Equal` | `:93` | `Icmp { Eq }` `:35` | **shared** |
| `Identity` | `:78` | - | omega only (no instruction) |
| `Maximum` | `:83` | - | omega only, expressible as `Icmp` + select |
| `Minimum` | `:84` | - | omega only, expressible as `Icmp` + select |
| `Negate` | `:85` | - | omega only, expressible as `Isub(0, x)` |
| `Select` | `:94` | - | omega only (nova has branches, not select) |
| `Reciprocal` | `:86` | - | omega only (float; nova has no float) |
| `Exponential` | `:87` | - | omega only (float) |
| `Logarithm` | `:88` | - | omega only (float) |
| `SquareRoot` | `:89` | - | omega only (float) |
| `Tanh` | `:90` | - | omega only (float) |
| `Erf` | `:91` | - | omega only (float) |
| `Icmp { Ne }` | - | `:35` | nova only |
| `Icmp { Slt }` | - | `:35` | nova only |
| `Icmp { Sle }` | - | `:35` | nova only |
| `Icmp { Sge }` | - | `:35` | nova only |
| `Icmp { Ult }` | - | `:35` | nova only |
| `Icmp { Ule }` | - | `:35` | nova only |
| `Icmp { Ugt }` | - | `:35` | nova only |
| `Icmp { Uge }` | - | `:35` | nova only |
| `Sextend` | - | `:41` | nova only (width change) |
| `Uextend` | - | `:46` | nova only (width change) |
| `Ireduce` | - | `:51` | nova only (width change) |

Counts: `ops=28 shared=6 omega_only=11 nova_only=11`. The shared six are exactly the integer
subset omega's CPU reference already defines at `typed_eval.rs:100-115` and nova already lowers;
nothing needs a new instruction to start.

## 3. float finding (the card's case c4)

`proxima-numeric` already has one float description, `FloatLayout`
(`proxima-numeric/src/float_layout.rs`, used as a settings value at
`proxima-numeric/src/flow_match.rs:87`). omega's `ScalarOp` is **not** keyed by it: each
emitter writes float literals and intrinsic names per op (`wgsl.rs:397` `0.0`,
`msl/emit_and_classify.rs:2840` `0.0f`). So the float half is one type but five renderings,
which is the same duplication as the integer half. O4 attaches the table to `FloatLayout`
rather than adding a second float description.

## 4. O2 decision (the first executable slice)

**Op: `Add`, widths `i32` and `i64`, wrapping semantics.**

| consumer | site | today |
|---|---|---|
| omega CPU reference | `proxima-tensor/src/cpu/typed_eval.rs:101` `wrapping_add` | one arm per width family, repeated in `width_kernels.rs` |
| omega MSL | `omega/src/msl/signature_tokens_prelude.rs:626` `(a + b)` | string template; reduce `simd_sum` at `emit_and_classify.rs:2803` |
| omega WGSL | `omega/src/wgsl.rs:769` `(a + b)` | string template; reduce `0.0` at `:397` |
| omega CUDA | `omega/src/cuda.rs:1854` `(a + b)` | string template; reduce `acc + sh` at `:1667` |
| nova lower | `proxima-nova/src/aarch64/lower.rs:410` | `add` instruction selection |
| nova encode | `proxima-nova/src/aarch64/encode.rs:214` (`MachInst::Add`; `:228` `AddImm` for a slot offset) | encoder table |

The slice adds `pub fn shared_add(width: Width, a: i64, b: i64) -> i64` to `proxima-numeric`
(`no_std + alloc`) and wires **only** `Add` through it on both sides: omega's `typed_eval` Add
arm and nova's `Iadd` arm each call it in their own body. The existing `ScalarOp` keeps its
shape; only the Add arm's body changes. `proxima-numeric` carries `op_table::wrapping_i32` and
`op_table::wrapping_i64`; each consumer carries `op_table::shared_add_i32` and
`op_table::shared_add_i64`. Every test pins `3 + 4 = 7` and wrapping at the type maximum, so a
saturating table fails a test rather than passing `3 + 4`.

Acceptance (`proxima-nova/scripts/nova-omega-ops.sh`, written by the slice):

    ops_shared=1 of=1
    results=4 of=4
    definitions=1 of=1

`definitions` asserts exactly one `pub fn shared_add` across the four crates. `results` is the
sum of `cargo val -p proxima-tensor op_table::shared_add` (2 passed) and the same filter on
`proxima-nova` (2 passed). A mutant that makes `shared_add` saturate fails `results` at
`i32::MAX + 1`; a duplicate definition fails `definitions`; removing either arm's call fails
`ops_shared`.

## 5. ladder after O2

| slice | content | check |
|---|---|---|
| O3 | `Subtract`, `Multiply`, `Divide` (signed and unsigned), `Greater`, `Equal` at `i32`/`i64`; comparisons gain their conditions from the table | `ops_shared=6 of=6 results=24 of=24` |
| O4 | float ops (`Reciprocal`, `Exponential`, `Logarithm`, `SquareRoot`, `Tanh`, `Erf`) keyed by `FloatLayout`; nova's float lowering is its own card (today `F64` is refused) | identical bit patterns omega-CPU vs an IEEE reference; `ops_shared=12 of=12` |
| O5 | omega's MSL, WGSL, CUDA and MIL emitters resolve per-op code through the table | every emitter's `ScalarOp::` arm count equals the table's coverage; a new op added once appears in all |

## 6. kill criterion

If an op's emitter code is shape- or storage-dependent in a way a typed table cannot key (for
example a matmul body that must be rendered as a different kernel per layout), stop at the
reference-evaluator table (omega CPU plus nova) and record the measured emitter gap in this
file. Do not widen the table to force an emitter match.

## 7. provenance (read 2026-11-21, branch `feat/proxima-nova`)

    sed -n '1,190p' proxima-tensor/src/op.rs
    sed -n '1,120p' proxima-nova/src/ir/inst.rs
    grep -rn 'ScalarOp::' omega/src proxima-tensor/src/cpu
    grep -n 'Inst::Iadd\|Inst::Sdiv' proxima-nova/src/aarch64/lower.rs
    grep -n 'MachInst::Sdiv\|MachInst::Udiv' proxima-nova/src/aarch64/encode.rs
