# Task brief — GPU parity plan for proxima-tensor through omega, ONE RISC

## The ask (owner, 2026-09-03, verbatim intent)

"see if you can determine and build a plan for why gpu is not performant on proxima-tensor
through omega. we need one risc. [guiding-principles, disciplined-component, plan-rigor] we need
this to be built so luna could follow the tasks. I'm expecting 4k or more lines. don't go cheap.
we need to expressly identify and then execute on what's going wrong. llama, ggml, ort and torch
can beat us on gpu. we have cpu afaik if we turn on acceleration."

## What the plan must be

An implementation plan, stepwise, that a cheap "hands" model (Luna: tool use, bounded edits,
runs commands, no design judgment) can execute task by task. Every task therefore carries:
worktree + branch + target dir, the exact commands, the file:line to open, the expected N
(tests, dispatch counts, op counts) with "N==0 is RED", the pre-registered prediction one rung
ahead on the bench ladder (nano -> micro -> milli -> bench), the kill criterion, the rollback,
the discipline-log row template, and the re-prove command. Time estimates are FORBIDDEN (owner
rule). Verdicts are forbidden; tasks produce evidence rows.

## Binding constraints (from the loaded skills and workspace rules — do not restate, obey)

- guiding-principles §1 reuse-first (write the expression before minting a type), §3 tiers
  (omega msl/wgsl/cuda emitters are alloc-tier; drivers are std), §4 config+builder parity,
  §6 read the code, §11 sans-IO/no-alloc hot path, §12 no magic numbers (every geometry constant
  traces to `omega-runtime.toml` via build.rs), §14 incumbent wins on correctness, §15 no punt,
  §16 re-provable now, §18/§19 measurement provenance and the evidence ladder, §20 box-free,
  §21 lock-free.
- disciplined-component 16-point gate per component; default-off compile-time feature per
  component; home-turf incumbent arm (llama.cpp-Metal at its own shape set) on every compare
  row; frequency-weighted scorecard; N asserted; one measurer on the box; interleaved arms.
- The bench ladder: nano -> micro -> milli -> bench, prediction ONE rung ahead only, a miss kills
  the climb and must be decomposed into inconsistency vs understanding-gap with a work item.
- Every response/row carries the scoreboard vs llama.cpp-Metal (and torch-MPS / ORT-CoreML once
  their cells exist) with CoV and a roofline cell; roofline = silicon physics only.
- The pipe question and the relocation question before ANY new type in omega or proxima-tensor.
- No commits without owner authorization; every commit a green bisect point; conventional commits.
- Worktrees are slot-0 siblings (`/Users/brianbruggeman/repos/slot-0/proxima-wt-<name>`), each
  with its own CARGO_TARGET_DIR; ~70 already exist, so names must not collide (listed in ledger).
- Row numbers are assigned at land time from main's last row (233); branches carry placeholders.
- ai_docs: add JSONL records (index/task-routes/invariants) for the GPU lane; do not bypass.

## The evidence ledger

`/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6e203711-bd50-48cc-9ade-409668bdafdd/scratchpad/ledger.md`
is the register of everything read or measured this session, tagged MEASURED / READ / MEMORY /
DERIVED. Cite it by section (R0..R11). Anything tagged MEMORY must be re-verified by a task
before another task depends on it. You may open the repo (read-only) to check any claim: main
checkout `/Users/brianbruggeman/repos/slot-0/proxima` at 4be2f3a; weld
`cd /Users/brianbruggeman/repos/slot-0/proxima && ` into every command; never enter a
`proxima-wt-*` sibling.

## The one-line diagnosis the plan must either confirm or overturn with a task

The GPU lane loses ~3.5-4x to llama.cpp-Metal on the same silicon and the same Metal API for
three classes of reason, in this order of mass: (1) the graph the model emits is not the RISC's
minimal graph — attention is duplicated because K/V cannot be written in place into a persistent
device buffer, so ~26 attention ops/layer exist where the incumbent has 3-5, the KV cache
round-trips through the host every token, and the plan/op-setup is rebuilt per token though the
shape never changes; (2) the Q4_K matvec kernel body does ~6x the ALU work of the incumbent's
per 8 weights and the cooperative reduce launches 32 threads where the incumbent launches up to
1024; (3) the lowering is three unequal hand-written emitters routed by hand-ordered gates whose
decision is recovered by grepping generated source, so no census can say which route each op
took or why, and geometry constants are bare source consts outside the sizing config.
Fixes for (2) exist as MEASURED wins (-17.2% and -4.9% gpu_exec) but are UNCOMMITTED, unrebased
diffs in worktrees based on a commit main has moved 9 commits past; a second agent's 42-commit
branch on today's main works the same lane with its own numbering.

## What "one RISC" means for this plan (bind it or argue it in the plan)

One bound plan (`&[BoundOp]`, 4 kinds) from one rewrite engine, identical for every backend;
one first-class route enum decided before emission and censused `(NodeId, reason)`; one emitter
core over the 4 kinds with backend-specific text only, every backend covering every kind; one
sizing config owning every geometry constant; write placement expressed with the existing
`Reduce.out_map`/`out_layout.base` and a driver-level persistent-buffer alias, NOT a new Op;
the Llama-arch graph emitted at 23 real ops/layer or fewer (the incumbent's count at its checkout).

## Output shape required from a Plan author

A numbered, phased plan. Phase 0 = seal + land what exists (re-seal on quiet box, rebase and
land the measured-but-uncommitted wins one green commit each, land the log rows, reconcile the
parallel branch). Then phases ordered by mass-removed-per-risk with explicit dependencies,
rollback per step, blast radius per step, observability per step (which counter proves it),
and the exact gates. Name what you abandoned and why (the constraints must have changed the
plan). Total length is not the goal here — precision is; the main thread expands the winner
into Luna task cards.
