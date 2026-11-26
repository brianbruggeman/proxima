load when: designing, briefing, changing or reviewing how any op is computed on any backend (Metal, CUDA/HIP, WGSL, CPU) in proxima or omega; before any lowering, kernel, fusion, format or backend work

## Source of truth

The proxima repo's `docs/omega/08-the-ir.md` builds the IR from the lowest algebra up and shows how a backend lowers it. Read it first, then the files it cites. Every claim cites a `file:line` opened this session; a claim from memory is a guess. `cargo val -p proxima-numeric --test doc_citations` keeps the doc's citations true.

## The IR is proxima-tensor's, not omega's

- A program is `Vec<Op>`: `Input`, `Elementwise`, `Reduce`, `Iota`, `Constant` (`proxima-tensor/src/op.rs`). Its scalar vocabulary is the closed `ScalarOp`.
- The bound form is `BoundOp` with `BoundOpKind` (`proxima-tensor/src/bind/types_layout_boundop.rs`).
- There is no other IR. A second one is always the mistake: a new kernel description, a trait standing in for the bound op, a per-backend op enum. Owner, 2026-11-26: "haven't we already reduced?"

## Layers, bottom up

1. `ScalarOp`, and the formats `DType`/`FloatLayout`/`NumericFormat` (proxima-numeric). `DType` names one-number formats only. `NumericFormat` also holds `Block` and `Levels`.
2. Addressing: `IndexMap` in a program, then `Layout` (base plus strides) after binding.
3. `ComposedBody`: an ordered list of scalar steps. Fusion is this one shape.
4. The two general ops:
   - `Elementwise`, a map;
   - `Reduce`, a fold over the axes the output drops. A matmul is a `Reduce` over a product body, and there is no matmul op.
5. The fused forms in `BoundOpKind`. Each is defined by the unfused chain it replaces, which is its reference.

## Lowering is a simple transform into an instruction set

- `omega::render::Dialect` spells tokens per machine (Msl, Cuda/HIP, Wgsl). `render::load`/`store` lower a format to integer steps for every dialect.
- Machines differ in facts, not code. Matrix-unit shapes and the formats they accept, wave width, shared memory and async copy are device data the shared lowering reads. The instruction words are the dialect's spelling.
- A backend that lags its incumbent is missing a fact or a schedule transform in the shared lowering, never a per-backend kernel.

## The known gap

A bound op's operand carries no format; weights arrive beside it (`PackedOperands`, `omega/src/lowering.rs`). That is why code gets tempted to pick a format. The fix is to carry the format on the bound operand, never to choose one.

## The debt map

`docs/omega/08-the-ir.md` section 6 lists every place a backend still has its own text:
- the per-backend `ScalarOp` spelling;
- the elementwise and reduce bodies;
- the fused-form bodies;
- the CPU evaluator's per-format kernels that omega's `CpuPlan` wraps.
Work deletes rows and never adds one.

## Binding rules (proxima `AGENTS.md`)

1. No second IR.
2. No bespoke paths: no backend-specific kernel text, no policy rule inside one backend's emitter, no kernel in the CPU evaluator. Not as an interim, not for a measured speedup.
3. No fixed weights until the format is within the lowered kernel. No code chooses a format by name, constant or property search (widest, half width, most mantissa bits). No default rounds or recodes to a weight's stored format outside that kernel.
4. The acceptance is the incumbent (llama.cpp, whisper.cpp, stable-diffusion.cpp) on every backend, never sameness with our old text.
5. A change to the IR or a lowering updates `docs/omega/08-the-ir.md` in the same change.

Hooks enforce rules 2 and 3: `numeric-format-guard` (`format-chooser`) and `bespoke-path-guard`.
