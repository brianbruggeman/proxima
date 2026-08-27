# proxima-vm discipline log

Per `/disciplined-component`. Scope: `tools/proxima-vm/src/elf.rs`, reshaped
under `tools/proxima-vm/ROADMAP.md`'s "Hand-rolling a spec parser" three
binding conditions (2026-08-26, elf-reshape worktree).

Build profile for every number below: `cargo bench` default (release,
`opt-level = 3` via the `bench` cargo profile, debuginfo on). Host: macOS
aarch64 (Apple Silicon), quiet box — no other benches running concurrently,
some background IDE/agent processes present (not pinned/isolated via
`taskset`-equivalent; macOS has no direct core-pinning CLI used here). This
is a **wall-clock note, not a pinned-core guarantee** — see Notes below.

## C1 — elf.rs (`parse_elf`, `Cursor`/`Step` FSM, `Segment`, `LoaderError`)

| Build+Tiers | Tests (N) | Lint | Micro | Compare | E2E | Opt | SIMD/SM/no-dyn | O(1) | Cfg/API | Home-turf | Audit | Consts | Re-prove | Δ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DONE (see below) | DONE (24) | DONE | DONE | DONE | N/A | PARTIAL | DONE | DONE | N/A | DONE | DONE | N/A | DONE | initial landing | see below |

**Incumbent design point(s):**
- `object` (v0.40.0): a general multi-format (ELF/Mach-O/PE/COFF/Wasm)
  binary-analysis library — its design point is a **full dynamic
  executable/object file** (`ET_DYN`, `PT_DYNAMIC`, symbol tables,
  relocations), the shape a real dynamic loader or disassembler hands it.
- `goblin` (v0.10.7): likewise general-purpose, ELF64 module built around
  the same dynamic-linking shape (`PT_DYNAMIC`, dynamic symbol table).

Neither incumbent's design point is "validate a static `ET_EXEC` bare-metal
guest's `PT_LOAD` segments for W^X/overlap/congruence before mapping it into
a hypervisor" — that is `parse_elf`'s own, narrower design point, named in
its own module doc comment. The `large_dynamic_elf` fixture below is the
`design-favors: incumbent` arm; the `small_guest_elf` fixtures are
`parse_elf`'s own turf, `design-favors: neutral` for `object`/`goblin`
(their relocation/symbol machinery is not engaged by either arm).

**Tier evidence:**
- `cargo check -p proxima-vm --lib --no-default-features --features alloc --target aarch64-unknown-none` → exit 0.
- `cargo check -p proxima-vm --lib --no-default-features --features alloc --target x86_64-unknown-none` → exit 0.
- Mechanically verified `elf.rs` is actually compiled under the tier-3
  build (not a `cfg`'d-out N==0 pass): a `compile_error!("tier3-proof-marker")`
  injected at the top of `parse_elf` made the tier-3 build fail at that
  exact line (`error: tier3-proof-marker` at `elf.rs:581`), then was
  removed and the build re-verified clean. Only `elf.rs`'s non-test code
  (`Cursor`, `Step`, `Segment`, `LoaderError`, `parse_elf`) is in scope for
  this build — `#[cfg(all(test, feature = "std"))]` excludes
  `test_support`/`tests` from a plain `check` regardless of tier.

**Test N:** 24, via `cargo nextest run -p proxima-vm --lib --no-fail-fast`
(23 in `elf::tests`, 1 pre-existing `tests::hello_guest_declares_expected_output`,
2 in `dispatch::tests`). Baseline (before this reshape, after fixing the
pre-existing wiring/workspace gaps below): 24 passed, 0 failed. After the
FSM reshape: 24 passed, 0 failed, 0 skipped — behavior-identical.

**Opt-sweep findings (sans-IO axes, condition 1):**

| axis | status | evidence |
|---|---|---|
| state machine (enum) | DONE | `Cursor<'a, MAX_SEGMENTS>` — 3 variants, see below |
| bytes-first | DONE | `&[u8]` in, no text anywhere on the parse path |
| borrowed views | DONE | `Segment<'a>.data: &'a [u8]` borrows the caller's image |
| zero-copy | DONE | no byte is copied; `Segment` fields are POD reads |
| copy-over-clone | DONE | `Segment` is `Copy`; `Cursor`/`Step` move by value, no `.clone()` anywhere |
| SIMD | N/A — rationale | fixed 56-byte struct reads (`read_u16/32/64`), not a byte-scan/search; nothing here is memchr/SIMD-shaped |
| stack-over-heap | DONE | `ArrayVec<Segment, MAX_SEGMENTS>` — fixed-cap, no heap |
| branchless inner loop | PARTIAL | per-entry validation is a straight-line sequence of early-return checks (`?`/`if` on already-validated data), not a data-dependent hot loop; the accept/reject branches ARE the contract (gABI rejection conditions), not something to eliminate |
| no dynamic dispatch | DONE | no `dyn`, no `Box`, no trait objects anywhere in `elf.rs` |
| O(1) per token | DONE — see O(1) row | per-entry cost is O(1) except the O(k) overlap scan (k = segments accepted so far, bounded by `MAX_SEGMENTS`) |

**Opt-sweep (disciplined-component axes, condition 2) — PARTIAL, honest gap:**
only one design tried (the FSM reshape itself, vs. the pre-reshape linear
function) — no *second* tweak was attempted and re-benched this session
(time-boxed; condition 2 was explicitly deprioritized behind conditions 1
and 3 for this pass). The FSM-vs-linear-function comparison is a shape
change with identical generated logic per branch, not a perf tweak — no
throughput delta is claimed or expected between them, and none was
measured (out of scope for the compare-bench, which targets the named
incumbents, not our own prior revision). **This row is left open**: a
follow-up pass should re-run `bench_elf` against a `git stash`ed
pre-reshape copy of `elf.rs` to confirm zero regression from the FSM
shape change quantitatively, not just via the identical 24/24 test count.

**SIMD/SM/no-dyn pass:** state machine — DONE (see `Cursor` below). SIMD —
N/A, no byte-scan/search present (see opt-sweep table). No dynamic dispatch —
DONE, verified by grep: `grep -n "dyn \|Box<"  tools/proxima-vm/src/elf.rs`
→ no matches.

**O(1):** every `Cursor::advance` transition does fixed-size reads
(`ELF64_HEADER_LEN`=64 or `PROGRAM_HEADER_LEN`=56 bytes) plus a linear scan
over `accepted` (already-pushed segments) for the overlap check — bounded
by `MAX_SEGMENTS`, a caller-chosen **compile-time** constant, so this is
O(`MAX_SEGMENTS`) per entry, not unbounded. `ArrayVec` push/read are O(1).

**Internal-primitive audit:** `proxima_protocols::nvme::raw::{read_u16,
read_u32, read_u64}` (host-DRAM little-endian field accessors, identical in
shape and byte order to the trio `elf.rs` was hand-rolling) is reused
instead of a fourth copy of the same three functions — the only change
required was flipping `mod raw;` to `pub mod raw;` in
`proxima-protocols/src/nvme/mod.rs` (that module was already `pub` at the
function level but private at the module level, so nothing outside `nvme`
could reach it before this). No existing primitive expresses the
gABI-specific validation sequence itself (bounds/size/overflow/congruence/
overlap/W^X checks driven by an explicit stage enum) — `Cursor`/`Step` stay.

**Tunable axes:** `MAX_SEGMENTS` is the one numeric axis, and it is
**already** a caller-supplied `const usize` generic parameter, not a
hidden magic number — satisfies principle 12 as originally designed; no
change needed in this pass.

**Re-prove command:**
```
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-vm --lib --no-fail-fast
CARGO_TARGET_DIR=<scratch> cargo clippy -p proxima-vm --lib --profile test
CARGO_TARGET_DIR=<scratch> cargo check -p proxima-vm --lib --no-default-features --features alloc --target aarch64-unknown-none
CARGO_TARGET_DIR=<scratch> cargo check -p proxima-vm --lib --no-default-features --features alloc --target x86_64-unknown-none
# bench (bins temporarily moved out of tools/proxima-vm/src/bin/ — see Notes):
CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-vm --bench bench_elf --features elf-bench,std
```
Every step above is a plain command against the artifact in this worktree;
no step depends on a prior run's memory. **Not yet CI-wired** — no CI job
runs these on a schedule (out of scope for this pass; see Notes).

### Bench results (2 runs, `--sample-size 30 --measurement-time 1 --warm-up-time 1`, criterion default profile)

| fixture | arm | design-favors | run 1 (median) | run 2 (median) | CoV |
|---|---|---|---|---|---|
| aarch64 guest ELF (~600 B, 2×PT_LOAD, ET_EXEC) | `proxima_vm_parse_elf` | neutral | 29.249 ns | 30.079 ns | ~2.8% |
| aarch64 guest ELF | `object::File::parse` | neutral | 60.574 ns | 62.408 ns | ~3.0% |
| aarch64 guest ELF | `goblin::elf::Elf::parse` | neutral | 1.0009 µs | 1.0154 µs | ~1.4% |
| x86_64 guest ELF (~200 B, 2×PT_LOAD, ET_EXEC) | `proxima_vm_parse_elf` | neutral | 29.149 ns | (not re-run) | n/a — 1 run |
| x86_64 guest ELF | `object::File::parse` | neutral | 58.944 ns | (not re-run) | n/a — 1 run |
| x86_64 guest ELF | `goblin::elf::Elf::parse` | neutral | 880.60 ns | (not re-run) | n/a — 1 run |
| **large dynamic ELF (4.58 MB, 10 phdrs, ET_DYN, real `aarch64-unknown-linux-gnu` std binary)** | `object::File::parse` | **incumbent** | 142.81 ns | 148.39 ns | ~3.9% |
| **large dynamic ELF** | `goblin::elf::Elf::parse` | **incumbent** | 53.740 µs | 54.265 µs | ~1.0% |
| **large dynamic ELF** | `parse_elf` | **incumbent (feature gap)** | rejects with `LoaderError::UnsupportedType` (asserted, not timed) | — | — |

**Honest read:**
- On `parse_elf`'s own turf (the two real M1 guest ELFs), `parse_elf` is
  ~2x faster than `object` and ~30-34x faster than `goblin` — expected and
  not the headline: neither incumbent's relocation/dynamic-symbol
  machinery is engaged by a static two-segment `ET_EXEC` file, so this is
  a `design-favors: neutral` comparison, not a win over their design
  point (gate 13). The honest driver is that `parse_elf` does strictly
  less work (no format-dispatch, no section/symbol table indexing) for an
  input that doesn't need it.
- On the incumbents' home turf (the large real dynamic executable),
  `object` parses a 4.58 MB `ET_DYN` binary in ~143-148 ns (metadata-only
  parse — it does not decode every section eagerly) and `goblin` in
  ~54 µs (goblin does more eager work — dynamic symbol table walk, string
  table indexing — on the same input). **`parse_elf` cannot run this arm
  at all: it is out of its documented design point** (`ET_EXEC` only, no
  relocation processing — the module's own doc comment names this
  boundary). This is gate 13's "cannot run the arm" case: a **feature
  gap**, not a performance loss, and correctness-verified: `parse_elf`
  names it `LoaderError::UnsupportedType`, not a silent mis-map or a
  panic — asserted in the bench itself (`expect_err` on every bench
  invocation, not amortized away).
- **No `parse_elf` throughput number is claimed against either incumbent's
  home-turf workload.** The frequency-weighted read: `parse_elf`'s actual
  production call frequency is once per guest boot (M1's exit criterion),
  never per-byte-of-a-dynamic-binary — its own design point (the guest
  ELFs) is the 100% case for this component, and it wins there on strictly
  less scope, honestly labeled `neutral`, not `incumbent`.

**Correctness parity (principle 14):**
- `elf::tests::matches_readelf_on_the_real_aarch64_guest` /
  `..._x86_64_guest` — `parse_elf`'s entry point + both `PT_LOAD` segments'
  virtual address, size, and permission bits cross-checked field-for-field
  against `llvm-readobj --program-headers` (LLVM 20, this toolchain's
  bundled binary — no `readelf` on this host, so `llvm-readobj` is the
  incumbent per principle 14). **Both pins in this test were stale
  relative to a clean rebuild of the checked-in guest source** (aarch64:
  pinned 532/40, rebuilt+verified 332/32; x86_64: pinned 291/12 @ vaddr
  0x124, rebuilt+verified 179/12 @ vaddr 0xB4) — chased down against a
  fresh `llvm-readobj` run before touching `parse_elf` (principle 14: the
  disagreement is ours to disprove first), confirmed `parse_elf`'s output
  already matched the fresh `llvm-readobj` run byte-for-byte, and only the
  test's hardcoded pin was corrected. This is the disagreement-with-the-
  incumbent case principle 14 exists for, worked exactly as specified.
- The large dynamic-ELF fixture's `LoaderError::UnsupportedType` rejection
  is itself a parity check: `object`/`goblin` both successfully parse
  `e_type == ET_DYN`; `parse_elf` naming it as an explicit, typed rejection
  (not a panic, not a silent wrong-answer) is the correct behavior for a
  loader that has documented itself as `ET_EXEC`-only.

**Implication:** condition 1 and condition 3's required arms are landed
and passing. Condition 2's opt-sweep is PARTIAL — one design (the FSM
itself) shipped and tested, but no second measured tweak was attempted
this pass; that is an open item, not a closed gate (see the open row
above). Bench CoV across 2 runs stayed under ~4% on every arm re-run;
3-5 runs is the skill's stated minimum and this pass only had budget for 2
— reported as a range, not a point estimate, per the skill's own rule for
this exact situation.

## Notes — scope, host loadout, and what this pass did NOT do

- **Host loadout:** macOS, no explicit core pinning; ambient IDE/tooling
  processes may have been running. CoV stayed low (<4%) across the 2 runs
  captured, which is some evidence the box was reasonably quiet, but this
  is not a controlled/isolated bench environment claim.
- **`cargo bench -p proxima-vm --bench <name>` builds every target in the
  package, including its `[[bin]]`s**, unlike `cargo check`/`cargo test`
  with the same target-selection flags. Two of this crate's three bins
  (`dispatch_probe`, `proxima-vm`) do not currently compile — pre-existing
  gaps from the M1 checkpoint (`RecordingDispatcher` and
  `run_hypercall_guest` do not exist in `dispatch.rs`; `loader.rs` is not
  wired into `lib.rs`), unrelated to `elf.rs`. To run the bench at all,
  those two bin source files were **temporarily moved out of
  `src/bin/` and moved back immediately after each bench run** (verified
  restored via `ls`/`git status` after both runs). This is a
  bench-running workaround, not a fix, and not a permanent change — the
  worktree's `src/bin/` directory is unchanged from before this pass.
- **`elf.rs`-unrelated pre-existing breakage found and left as-is** (out
  of scope for "reshape `elf.rs`" — these are ROADMAP resume-checklist
  steps 1/2/4, not step 3): `tools/proxima-vm/tests/boot.rs` and
  `tests/lambda_run.rs` reference `tempfile`/`postcard`, neither declared
  as a dependency; `tests/dispatch_hypercall.rs` and
  `src/bin/dispatch_probe.rs` reference `dispatch::RecordingDispatcher`
  and `dispatch::run_hypercall_guest`, neither of which exists in
  `dispatch.rs`; `src/bin/proxima-vm.rs` additionally references
  `proxima_vm::loader`, not wired into `lib.rs` (attempting to wire it
  produces an **undefined-symbol link error**, not a warning:
  `proxima_vm_map_guest_memory`/`proxima_vm_unmap_guest_memory` are
  declared via `unsafe extern "C"` in `loader.rs` but never implemented in
  `build.rs`/the `.c` sources it compiles). `cargo nextest run -p
  proxima-vm` (full target set, no `--lib` restriction) still exits 101
  for these reasons; `cargo nextest run -p proxima-vm --lib` is the
  correct, green, elf.rs-inclusive scope this pass used throughout.
- **Wiring fixes made in this pass, in scope because `elf.rs`'s own tests
  could not otherwise run:**
  - `pub mod elf;` added to `tools/proxima-vm/src/lib.rs` — the module
    existed on disk but was never declared as part of the crate
    (`unresolved import proxima_vm::elf` from every external caller,
    including the pinned parity tests' own crate-external imports).
  - `arrayvec` and `proptest` added as real dependencies of
    `tools/proxima-vm` via `cargo add` (`arrayvec` regular,
    `proptest --dev`) — `elf.rs` used both without either being declared.
  - `"tools/proxima-vm/guests/lambda"` added to the workspace `members`
    array — the guest crate existed on disk, un-registered, so
    `cargo build -p proxima-vm-guest-lambda --target ...` (what the
    parity tests' own `guest_elf_bytes` helper runs) failed with
    `package ID specification did not match any packages`. Risk noted: a
    hypothetical blanket `cargo check --workspace`/`--all` at the host
    default target would now also attempt to build this `#![no_std]`,
    custom-linker-script bare-metal binary crate and fail — no such
    blanket command is part of this task's gates, and none was run.
  - `proxima-protocols/src/nvme/mod.rs`: `mod raw;` → `pub mod raw;`
    (one line) so `elf.rs` can reuse `read_u16`/`read_u32`/`read_u64`
    instead of a fourth hand-rolled copy (see the internal-primitive
    audit above).

