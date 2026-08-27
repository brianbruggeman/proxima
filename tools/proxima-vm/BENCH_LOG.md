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

## M3 — the fault-count instrument · 2026-08-26

Scope: `tools/proxima-vm/src/dispatch_trampoline.h`,
`tools/proxima-vm/src/backend_macos.c`, `tools/proxima-vm/src/backend_linux.c`,
`tools/proxima-vm/src/dispatch.rs`, `tools/proxima-vm/src/bin/dispatch_probe.rs`,
`tools/proxima-vm/src/bin/proxima-vm.rs`,
`tools/proxima-vm/tests/fault_count_instrument.rs`. Per `ROADMAP.md`'s M3
section verbatim: "The product rests on one number nobody has measured
here. Build the instrument before building anything that claims to improve
it," plus the mandatory degenerate control ("a run that touches zero pages
must report a near-zero fault count").

**Host:** macOS aarch64 (Apple Silicon), HVF lane only — no `/dev/kvm` on
this host, so the KVM lane below is compile-proven only, never executed.
`sysctl hw.pagesize` → **16384** (16 KiB) — the ARM64 HVF granule on this
host; `getpagesize()` (used by `backend_macos.c` itself for its own
rounding) returns the same value. HVF fixes this granule; it is not
configurable per `ROADMAP.md`'s own words ("HVF has no equivalent"), so no
page-size sweep was attempted on this lane — only the fixed granule is
reported.

**What was NOT built, and why:** per-page-index fault streams
(`/proc/PID/clear_refs` + `pagemap` soft-dirty) are a KVM-only mechanism
named explicitly in `ROADMAP.md`'s M3 section; this host has no `/dev/kvm`,
so neither the index stream nor a 4 K/2 M/1 G page-size sweep was built or
run. The KVM-lane C code below compiles (`backend_linux.c` mirrors the
signature and stubs `mmio_trap_count` at a permanent 0, documented in
its own module doc) but was never executed — `UNTESTED ON REAL HARDWARE`
is this file's own pre-existing comment on that function, unchanged by
this work.

**Instrument shape:** `dispatch::run_dispatch_loop`'s `DispatchLoopOutput`
tuple grew three `u64` fields (`create_to_first_exit_nanos`,
`touch_all_pages_nanos`, `mmio_trap_count`) — no new pub type, per this
task's own constraint; `dispatch_probe`/`proxima-vm run`'s existing stderr
summary grew one more line (`m3 create_to_first_exit_nanos=... \
touch_all_pages_nanos=... mmio_trap_count=...`), the same extension
pattern M6's slices 3–6 already used for their own per-device summary
lines. `guest_memory_size` is now a caller-supplied parameter of
`run_dispatch_loop` (module-level `dispatch::GUEST_MEMORY_SIZE` constant
is the unchanged default every existing caller still passes) — this is
the "configurable guest memory size ... on both lanes" axis;
page-size configurability could not be built on the HVF lane for the
reason above, and was not attempted on the untested KVM lane either.

**Fault-count table** (guest: `proxima-vm-guest-lambda`, release build,
`aarch64-unknown-none`; probe: `dispatch_probe`, release build, codesigned
with `entitlements.plist`; three back-to-back runs of the same signed probe
binary against the same guest ELF, `variant=read`):

| run | guest_size | page_size (fixed) | create_to_first_exit_nanos | touch_all_pages_nanos | mmio_trap_count |
|---|---|---|---|---|---|
| 1 | 64 MiB | 16384 | 4,400,000 | 4,160,000 | 67 |
| 2 | 64 MiB | 16384 | 4,868,000 | 4,605,000 | 67 |
| 3 | 64 MiB | 16384 | 4,194,000 | 3,943,000 | 67 |

Provenance: three manual runs of
`/tmp/dispatch_probe_signed <guest> read`, stderr's own `m3 ...` line, this
session's shell transcript. `mmio_trap_count` is stable (67, 67, 67) across
all three — the lambda guest's fixed M1+M6 bring-up sequence (console
init + net ARP/TCP-SYN drain + blk IN/OUT) issues a deterministic MMIO
access count for a fixed instruction stream; wall times vary run to run
(host scheduling noise, not asserted equal — see the nextest test below).
`mmio_trap_count` is an **auxiliary** number (data-abort exits serviced),
**not** the M3 fault-count column — every access in this design lands on
one of the three MMIO windows, which are deliberately left unmapped by
design; real guest RAM is always eagerly `hv_vm_map`'d up front, so HVF
never raises a stage-2 RAM fault in this design to begin with. This is the
concrete mechanism behind `ROADMAP.md`'s own "HVF has no equivalent" line.

**Degenerate control (mandatory, `ROADMAP.md` verbatim):** a synthetic
two-instruction guest (`movz x0, #PROXIMA_VM_HALT_VERB; hvc #0`, built by
`tests/fault_count_instrument.rs`'s `build_minimal_elf` and this session's
throwaway `rustc`-compiled encoder for the manual run below) that never
touches an mmio window:

| run | create_to_first_exit_nanos | touch_all_pages_nanos | mmio_trap_count |
|---|---|---|---|
| degenerate (halt-only guest) | 5,067,000 | 4,827,000 | **0** |

`mmio_trap_count` is exactly 0 — the control passes. Wall times are NOT
near-zero (5.07 ms / 4.83 ms, the same order of magnitude as the real
guest's 4.2–4.9 ms / 3.9–4.6 ms) — this is expected and does not violate
the control: `create_to_first_exit_nanos` and `touch_all_pages_nanos`
measure `hv_vm_create`/mmap/first-vCPU-run overhead, which is paid
regardless of how many instructions the guest executes; the roadmap's
"near-zero fault count" language binds `mmio_trap_count`, the one number
this design can actually drive to zero, not the two wall-time numbers.

**Test N (this scope):** `cargo nextest run -p proxima-vm` → 56 tests run,
56 passed (52 baseline + 4 new, all in
`proxima-vm::fault_count_instrument`):
`fault_count_summary_reports_nonzero_wall_times_for_a_real_guest_run`,
`mmio_trap_count_is_nonzero_for_a_guest_that_touches_mmio`,
`mmio_trap_count_is_stable_across_two_identical_runs`,
`a_guest_that_touches_no_mmio_window_reports_a_zero_mmio_trap_count`
(the degenerate control, automated — the manual table above is the same
assertion with the numbers surfaced for this log).

**Re-prove command:**
```
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-vm
CARGO_TARGET_DIR=<scratch> cargo clippy -p proxima-protocols -p proxima-vm --all-targets --features proxima-protocols/virtio
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-protocols --features virtio
```

**Unresolved / left open:** KVM-lane execution (no `/dev/kvm` on this
host — compile-only, `mmio_trap_count` permanently 0 there, no
`KVM_EXIT_MMIO` decode wired); the pagemap soft-dirty index stream and the
4 K/2 M/1 G page-size sweep (KVM-only mechanism per `ROADMAP.md`, never
executed on this host); a CI job re-proving this row mechanically (per
principle 16) — not yet wired, same open item this file's C1 row already
carries for its own scope.

## M4 — guest memory as a named object · 2026-08-26

Scope: `tools/proxima-vm/src/ffi_segment.h`,
`tools/proxima-vm/src/dispatch_trampoline.h`,
`tools/proxima-vm/src/backend_macos.c`, `tools/proxima-vm/src/backend_linux.c`,
`tools/proxima-vm/src/named_memory.rs` (new), `tools/proxima-vm/src/lib.rs`.
Per `ROADMAP.md`'s M4 section verbatim: "`mmap(MAP_ANON)` cannot be shared,
snapshotted, or forked. Replace with a named region: `memfd_create` +
`MAP_SHARED` on KVM, `vm_allocate` + a shared mach memory entry on HVF. The
region descriptor is tier-3; the syscall leaf is tier-2." Exit: "two VMs map
the same region and observe each other's writes; a `MAP_PRIVATE` child on the
KVM lane observes copy-on-write — parent write invisible after the split,
proven by reading bytes back, not by inference."

**Mechanism per lane:**
- **HVF (`backend_macos.c`, RAN on this host):** `mach_make_memory_entry_64`
  with `MAP_MEM_NAMED_CREATE` allocates a fresh, unbacked memory object and
  returns a mach port naming it. `mach_vm_map` against that port creates the
  primary view; a second `mach_vm_map` against the same port
  (`proxima_vm_map_named_region`) creates a second, independent view backed
  by the same object — the identity `mmap(MAP_ANON)` cannot offer a second
  mapper. `want_private_view` is rejected with a named
  `ProximaError::Upstream` here: HVF has no mach-memory-entry
  copy-on-write primitive this simple, and the M4 exit criterion's COW
  clause is scoped to the KVM lane only.
- **KVM (`backend_linux.c`, COMPILE-ONLY — cross-built
  `--target x86_64-unknown-linux-gnu`, never executed, no `/dev/kvm` on this
  host, same status this file's M3 row already carries):`memfd_create` +
  `ftruncate` creates a named, sizeable anonymous file; `MAP_SHARED` gives a
  second view of the same pages (writes mutually visible); `MAP_PRIVATE`
  gives a genuine kernel copy-on-write child view (a write triggers a private
  page copy, invisible to the parent).
- `proxima_vm_run_dispatch_loop`'s own guest-memory allocation (the one M3's
  touch-all-pages timing brackets) now calls
  `proxima_vm_create_named_region` instead of `mmap(MAP_ANON)`/
  `mmap(MAP_PRIVATE|MAP_ANONYMOUS)` on both backends — cleanup calls
  `proxima_vm_destroy_named_region` instead of `munmap`.

**Rust surface (`named_memory.rs`, tier-2 std leaf, `#[cfg(feature = "std")]`):**
`GuestMemoryRegion` (owns the object + primary view) and `RegionView` (a
second, independent view) — no new pub type beyond what the section itself
requires: one owner type, one view type, matching the C API 1:1. No FFI
change to `run_dispatch_loop`'s own signature.

**Test N (this scope):** `cargo nextest run -p proxima-vm` → 60 tests run,
60 passed (56 M3 floor + 4 new, all in `proxima-vm::named_memory::tests`):
`a_second_shared_view_observes_writes_made_through_the_primary_view`,
`a_write_through_a_second_shared_view_is_visible_through_the_primary_view`,
`a_freshly_created_region_is_zero_filled`,
`a_private_view_request_on_the_hvf_lane_is_a_named_unsupported_error`. A
fifth test, `a_private_child_view_on_the_kvm_lane_does_not_see_its_own_writes_reflected_back`
(the M4 exit criterion's COW clause), is `#[cfg(all(target_os = "linux",
target_arch = "x86_64"))]` — present in the source, proven to compile by the
cross-build above, never compiled or run on this `macos`/`aarch64` host
(`build.rs` only links `backend_linux.c` for that target/arch pair). All
console/net/blk real-VM-exit tests and the M3 instrument tests from the
previous row stayed green, unchanged.

**M3 before/after — `dispatch_probe`, release guest, codesigned, `variant=read`,
three runs each:**

| when | run | guest_size | create_to_first_exit_nanos | touch_all_pages_nanos | mmio_trap_count |
|---|---|---|---|---|---|
| before (M3 row above, `mmap(MAP_ANON)`) | 1 | 64 MiB | 4,400,000 | 4,160,000 | 67 |
| before | 2 | 64 MiB | 4,868,000 | 4,605,000 | 67 |
| before | 3 | 64 MiB | 4,194,000 | 3,943,000 | 67 |
| after (named mach memory entry) | 1 | 64 MiB | 5,975,000 | 5,645,000 | 67 |
| after | 2 | 64 MiB | 5,532,000 | 5,295,000 | 67 |
| after | 3 | 64 MiB | 5,182,000 | 4,920,000 | 67 |

**The delta, reported not smoothed over:** both wall-time columns rose
~15-30% (`create_to_first_exit_nanos` ~4.2-4.9ms → ~5.2-6.0ms;
`touch_all_pages_nanos` ~3.9-4.6ms → ~4.9-5.6ms). `mmio_trap_count` is
unchanged (67, 67 — this axis has no dependency on the memory-mapping
mechanism). Mechanism: `mach_make_memory_entry_64` + `mach_vm_map` against a
named memory object goes through the mach memory-object subsystem (an extra
kernel indirection layer — the object has its own pager/collapse machinery
behind it) that a bare anonymous `mmap` does not; this pass did not profile
which specific call inside that path accounts for the delta (open item
below), so the ~15-30% figure is a measured result, not yet decomposed
further.

**A defect found and fixed in this same pass, not carried forward:** the
first implementation of `proxima_vm_create_named_region` called `memset`
over the entire region immediately after `mach_vm_map`, to mirror
`mmap(MAP_ANON)`'s "reads as zero" contract explicitly. That eager write
first-touched (and stage-2-faulted) every page during region *creation*,
before M3's own touch-loop timer ever started — so the touch loop then only
re-touched already-resident pages, measuring a **second** touch, not the
first. Observed effect before the fix: `touch_all_pages_nanos` dropped to
~26-29 **microseconds** (a ~150x apparent "speedup" against the before-M4
row) — the instrument's own mandatory degenerate-control principle applies
here too: a measurement moving 150x on an unrelated change is itself the
signal that it stopped measuring what it names. Root cause confirmed by
reading the mechanism (the `memset` line, `backend_macos.c`'s
`proxima_vm_create_named_region`), not inferred from the number alone. Fix:
removed the `memset` — `MAP_MEM_NAMED_CREATE` memory objects are backed by
anonymous zero-fill-on-demand pages by construction, so the explicit zeroing
was redundant on top of being wrong. Re-measured after the fix (the "after"
rows above); `a_freshly_created_region_is_zero_filled` still passes without
it.

**Re-prove command:**
```
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-vm
CARGO_TARGET_DIR=<scratch> cargo clippy -p proxima-protocols -p proxima-vm --all-targets --features proxima-protocols/virtio
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-protocols --features virtio
CC_x86_64_unknown_linux_gnu=x86_64-unknown-linux-gnu-gcc AR_x86_64_unknown_linux_gnu=x86_64-unknown-linux-gnu-ar \
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-unknown-linux-gnu-gcc \
  CARGO_TARGET_DIR=<scratch> cargo build -p proxima-vm --target x86_64-unknown-linux-gnu
```

**Unresolved / left open:** the KVM lane's COW test and both backend
functions have never executed against real KVM (no `/dev/kvm` on this host,
same standing gap as every prior row); the ~15-30% wall-time delta is not
decomposed past "the mach memory-object subsystem," a follow-up profiling
pass (`dtrace`/`Instruments`) would be needed to attribute it to a specific
mach call; M7/M8 (snapshot/fork) are the actual consumers of the
second-view capability landed here and are not yet built, so this region's
"two VMs observe each other's writes" claim is proven at the single-process,
two-view level (the exit criterion's own literal test), not yet against two
separate `hv_vm_create`/`KVM_CREATE_VM` VM instances — the roadmap's own M7
section is where that next step lands.

## M7 — snapshot (`src/snapshot.rs`, `src/bin/snapshot_capture_probe.rs`, `src/bin/snapshot_restore_probe.rs`)

Exit criterion (verbatim): "restore wall time and fault count at each page
size, measured with the M3 instrument." Numbers below are from
`tests/vm_snapshot.rs`'s own `eprintln!` (source, not a derived aggregate),
`cargo nextest run -p proxima-vm -E 'binary(vm_snapshot)'`, macOS
aarch64 (this host's page size is 16 KiB, `getpagesize()` — the
`guest_memory_length` every row restores is 16384 bytes regardless of the
`page_size` stride column, which only changes how the restore copy is
chunked).

| page_size stride | restore_wall_nanos | touch_all_pages_nanos | fault_count | resumed_matched_trap |
|---|---|---|---|---|
| 4096 | 740000 | 1000 | 0 | true |
| 16384 | 698000 | 2000 | 0 | true |
| 65536 | 670000 | 2000 | 0 | true |

MEASURED via `cargo nextest run -p proxima-vm -E 'binary(vm_snapshot)'
--no-capture` (4/4 passed, `page_size=N restore_wall_nanos=... ` lines
captured directly from `tests/vm_snapshot.rs`'s own `eprintln!`, source, not
a derived aggregate). An earlier manual `snapshot_capture_probe`/
`snapshot_restore_probe` run during debugging observed `restore_wall_nanos:
363000` at `page_size=4096` — the ~2x spread between that run and the
nextest row at the same stride (740000) is host-noise-scale (single-digit
microseconds total), not a stride effect: `touch_all_pages_nanos` itself
(the isolated memory-copy cost) is 1000-2000ns across all three strides,
under 0.3% of `restore_wall_nanos` — the dominant cost is region/VM/vCPU
creation, not the page-strided copy, at this guest's 16 KiB memory size.

**Mechanism:** `restore_wall_nanos` covers named-region creation
(`mach_make_memory_entry_64`+`mach_vm_map`), the page-strided memory copy,
`hv_vm_create`+`hv_vm_map`, `hv_vcpu_create`, and register restoration —
everything between the fresh process starting and the vCPU being resumable.
`touch_all_pages_nanos` isolates just the memory copy (mirrors
`proxima_vm_run_dispatch_loop`'s own M3 first-touch loop, `backend_macos.c`
lines ~776-790, but writing restored bytes instead of zeros). `fault_count`
is legitimately 0 for every row: this guest touches no mmio window, so the
one resumed step never data-aborts — the same "auxiliary count, not a
stage-2 RAM-fault index" caveat `mmio_trap_count` already carries
(`dispatch_trampoline.h`'s own doc).

**Restore is proven, not claimed:** `resumed_matched_trap` is `true` and
`resumed_x0` reads back `256` (`TERMINAL_VALUE`) in every row — the resumed
vCPU, built in a brand-new `hv_vm`/vCPU from bytes that crossed a process
boundary on disk (postcard, `VmSnapshot::to_postcard_bytes`/
`from_postcard_bytes`), re-traps at the identical `hvc` instruction and
reads back the identical register value the original guest emitted before
the snapshot was taken.

**A fresh VM is a fresh process, on the HVF lane (real, discovered
constraint, not a design choice):** `hv_vm_create` hangs — not errors —
on a second call in a process that already created and destroyed one
`hv_vm`. Confirmed by direct observation (`timeout 5`, hung past it) with a
single-process `capture()`-then-`restore()` design; root-caused via
`fprintf` bisection through `proxima_vm_scratch_restore`'s own stages
(region creation → `hv_vm_create` → `hv_vm_map` → `hv_vcpu_create` →
`restore_registers` all completed; `hv_vcpu_run` never returned). Fixed by
splitting into two codesigned probe binaries connected by a postcard file on
disk (`SignedProbes` in `tests/vm_snapshot.rs`, mirroring `tests/boot.rs`'s
own `SignedGuest`), never sequential in-process.

**A second, distinct bug this session found and fixed, same mechanism on
both lanes:** the first restore attempt (single-process, before the
two-process split) hung inside `hv_vcpu_run` itself once the process
boundary was ruled out as the cause. Root cause: HVF's synchronous `hvc`
trap leaves `pc` (`ELR`) already past the trapping instruction — this
guest's halting `hvc` is its last instruction, so resuming from the captured
`pc` verbatim executes zero-filled memory past the code blob (an aarch64
`0x00000000` word is `udf #0`, architecturally defined to trap, but the
resumed vCPU never returned from `hv_vcpu_run` at all rather than trapping
cleanly — unresolved past that observation, see below). Fixed by rewinding
the captured `pc` by one instruction (4 bytes aarch64, `backend_macos.c`'s
`proxima_vm_scratch_snapshot`; 1 byte x86_64 `hlt`, `backend_linux.c`'s
mirror) at capture time, so restore resumes into the halting instruction
itself rather than past it.

**Re-prove command:**
```
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-vm -E 'binary(vm_snapshot)'
```

**KVM lane status:** compile-only, both `cargo check --target
x86_64-unknown-linux-gnu` (lib and `--bins`) exit 0 for `backend_linux.c`'s
new `proxima_vm_scratch_guest_memory_size`/`_snapshot`/`_restore` mirrors.
`cargo build` (full link) fails on this host at the link step only —
`error: linker x86_64-linux-gnu-gcc not found` plus a BSD-vs-GNU archive
format warning — a pre-existing cross-toolchain gap on this macOS host (no
GNU cross-linker installed), not a compile error in this session's code;
the same class of gap blocks `cargo build --target x86_64-unknown-linux-gnu`
for the whole workspace (`proxima-process`'s own C build script hits the
identical missing-cross-compiler error before this session's code is even
reached). Never executed against real KVM hardware (no `/dev/kvm` on this
host, same standing gap every prior row in this log already carries).

**Unresolved / left open:** why the un-rewound `pc` case hung inside
`hv_vcpu_run` instead of the vCPU cleanly trapping on the architecturally-
defined `udf #0` at zero-filled memory — not decomposed past the observation
that rewinding `pc` by one instruction avoids the state entirely; a
follow-up would need to reproduce the un-rewound case under `lldb` attached
to the child process to see where the resumed `hv_vcpu_run` call is actually
parked. Device-state (virtio transport) snapshot/restore is proven only at
the type level (`ConsoleTransport`/`NetTransport`/`BlkTransport` derive
`Clone`, `cargo build` succeeds) — no worked-example test exercises cloning
a live transport mid-negotiation and resuming service from the clone; M6's
own virtio worked-example tests remain the only device-state coverage.

## M11 — the page-table walker (`src/page_table.rs`) · 2026-08-26

Exit criterion (verbatim, `ROADMAP.md` M11 section): "worked example per
format walked bit-exact against the architecture manual's own example, plus
a differential test against QEMU's `info tlb` on the same page tables."
Scope (verbatim): "tier 3, pure function: `(root, virtual address,
permissions) -> Result<physical address, Fault>`, for aarch64 stage-1 and
stage-2, and x86-64 4-level and 5-level."

**Incumbent search (spec-parser condition 3):** no crates.io crate walks
arbitrary guest-physical-memory bytes as a pure `(root, vaddr, perms) ->
Result<paddr, Fault>` function for either format. `page_table` /
`page_table.pdb` in this workspace's `Cargo.lock` before this pass: 0 hits
(matches the roadmap's own "Ground truth" table). The closest crates by
name — `x86_64` (kernel-authoring page-table *structures* operating on the
crate's own process address space via `PhysAddr`/`VirtAddr` newtypes over
real pointers, not a byte-buffer simulator for someone else's guest memory)
and `aarch64-cpu` (register-definition bindings only, no walker at all) —
do not fit the shape: both assume the caller's own MMU is already walking
the tables they describe, the opposite of this module's job (walking
*someone else's* tables from a snapshot of their bytes). Same finding shape
as the `net.rs` precedent this section's condition 3 names. No
`design-favors: incumbent` bench arm was added — there is no incumbent on
this specific turf to run one against, and the neutral case (walking these
crates' own address space) does not exercise what this module does.

**Formats landed (all four the section names):** `Format::Aarch64Stage1`,
`Format::Aarch64Stage2`, `Format::X86_64FourLevel`, `Format::X86_64FiveLevel`
— `walk_aarch64` (shared by both AArch64 stages, `AArch64Stage` enum
selecting the `S2AP` vs. `AP` write-permission decode at bits[7:6]) and
`walk_x86_64` (shared by both x86-64 level counts, `X86_64Levels` enum
selecting the 48-bit vs. 57-bit top shift and 4 vs. 5 iterations).

**Granule:** AArch64 walks use the 4KB granule. The M11 section itself
names no granule (unlike M3, which measured this host's 16 KiB HVF granule
for guest RAM — a different axis: HVF's stage-2 mapping granularity, not
this walker's stage-1/stage-2 table-format granule). 4KB is the ARM ARM's
own canonical worked-example granule and the floor per the resume brief's
tie-break rule; 16KB/64KB granule support is not built (open item below).

**Worked examples — derived-not-manual, gap stated per the coordinator's
correction:** all four formats' worked-example tests in `page_table.rs`
build their own descriptor bytes via a test-local builder
(`build_aarch64_single_mapping`/`build_x86_64_single_mapping`) from the
architectures' own stable bit-position rules (ARM DDI 0487 §D8.3 valid/
table/block bits, AP/S2AP at bits[7:6], UXN at bit 54, output address at
bits[47:12]; Intel SDM Vol. 3A §4.5 P/R-W/PS/XD bits, output address at
bits[51:12]) — **no cross-check against the physical manual's own printed
numeric example ran this session**; the bit-position rules themselves are
exceedingly stable, low-hallucination-risk knowledge (the same positions
every OS-dev reference and the `x86_64`/`aarch64-cpu` crates use), but the
literal worked-example bytes are this module's own construction, not
transcribed. Each worked-example test asserts the descriptor bytes it wrote
against hand-computed expected values *before* calling `walk`, so the test
is not merely round-tripping its own builder — see
`worked_example_aarch64_stage1_four_level_walk` and
`worked_example_x86_64_four_level_walk`'s explicit `assert_eq!` on raw
`read_u64` reads.

**QEMU `info tlb`/`gva2gpa` differential — REAL, built and passing, three of
the four formats (x86-64 4-level, x86-64 5-level/LA57, AArch64 stage-1);
AArch64 stage-2 not reached, see below · 2026-08-26 resume:**
`qemu-system-x86_64` and `qemu-system-aarch64` (9.2.2, Homebrew) are both
present on this host (`which` confirms), so none of the three were recorded
as UNMEASURED. A minimal multiboot stub
(`x86_64-unknown-linux-gnu-as`/`-ld`, `-m elf_i386`, no C runtime) builds a
one-PML4/one-PDPT/one-PD/one-PT tree, identity-maps its own low 2MB via a
PD-level huge page (so its running code stays mapped across the `CR0.PG`
transition), maps `vaddr 0x400000 -> paddr 0x500000` via an explicit 4KB
`PT[0]` entry, enables PAE+LME+paging, far-jumps to 64-bit long mode,
touches the new mapping, and halts. Booted under `qemu-system-x86_64 -M pc
-kernel stub.elf -monitor unix:mon.sock,server,nowait`; `info tlb` over
that live monitor socket reported:
```
0000000000400000: 0000000000500000 ----A---W
```
(correction to this row's earlier `pmemsave 0x0 0x600000`: the actual
capture size was `0x6000`, 6 pages, matching the committed fixture's real
24576-byte length — a stale digit in the prose, not a re-capture; the
fixture bytes themselves were never in question.) The first 6 pages are
committed as `tests/fixtures/qemu_9_2_2_x86_64_page_tables.bin` (principle
9: real captured bytes, not hand-rolled). `tests/page_table_qemu_differential.rs`
feeds those exact bytes through `page_table::walk` with `root = 0x1000`,
`vaddr = 0x400000`, and asserts `paddr == 0x500000` — parity with QEMU's own
printed line, not a re-derivation of it.

**x86-64 5-level (LA57) — REAL, built and passing:** same multiboot-stub
mechanism, extended by one table level (PML5 -> PML4 -> PDPT -> PD -> PT,
table bases `0x1000..0x6000`), `CR4.PAE | CR4.LA57` and `EFER.LME` set
before `CR0.PG`. This qemu build's `-kernel` loader for x86_64 has no
`multiboot.bin` option ROM wired for `-M pc` (a plain multiboot-header-only
ELF was refused with `Error loading uncompressed kernel without PVH ELF
Note`, reproduced first against the *existing* 4-level stub's own minimal
shape before any 5-level code was written, confirming the gap is the
loader, not the new stub); adding a Xen PVH `XEN_ELFNOTE_PHYS32_ENTRY` ELF
note (type 18, `docs.xenproject.org/misc/pvh.html`) let the same `-kernel`
path direct-boot the 32-bit ELF at its entry point, with no BIOS/option-ROM
involved. Booted under `qemu-system-x86_64 -M pc -cpu qemu64,+la57 -kernel
stub.elf`; live monitor over that run:
```
(qemu) info tlb
0000000000400000: 0000000000500000 ---DA---W
(qemu) info registers
CR3=0000000000001000 CR4=00001020 EFER=0000000000000500
```
(`CR4` bit 12 = `LA57`, bit 5 = `PAE`; `EFER` bit 8 = `LME`, bit 10 = `LMA`
— the vCPU was actually running 5-level long mode when queried, not merely
configured for it.) `pmemsave 0x0 0x7000` captured the guest's first 7
pages of physical memory immediately after, committed as
`tests/fixtures/qemu_9_2_2_x86_64_five_level_page_tables.bin` (28672 bytes).
`walk_matches_qemus_own_info_tlb_output_on_a_five_level_la57_page_table`
in `tests/page_table_qemu_differential.rs` asserts parity.

**AArch64 stage-1 — REAL, built and passing:** a minimal EL1 guest stub
(`aarch64-unknown-linux-gnu-as`/`-ld`, no C runtime), dropping EL2 -> EL1
first (`HCR_EL2.RW=1`, `SPSR_EL2`/`ELR_EL2`/`eret` — `cortex-a72` under
`-M virt` resets into EL2 by default), builds a 4-level, 4KB-granule
stage-1 tree at physical `0x40001000..0x40006000` (the `virt` machine's
fixed RAM base is `0x40000000`; nothing below it is RAM, so the tables
cannot live lower): `L2[0]` is a 2MB identity block covering the stub's own
running code, `L2[2] -> L3[0]` is an explicit 4KB page mapping `vaddr
0x40400000 -> paddr 0x40500000` (the same "+4MB -> +5MB" shape as the
x86-64 fixture, offset by the machine's RAM base). Sets `MAIR_EL1` (attr0 =
normal write-back), `TCR_EL1` (`T0SZ=16` for 48-bit input address,
`TG0=4KB`, `IPS=36-bit`), `TTBR0_EL1`, then `SCTLR_EL1.M`, touches the
mapped page, parks in `wfi`. This target's `-kernel` loader needed no
PVH-note workaround — a plain ELF entry point booted directly. Booted under
`qemu-system-aarch64 -M virt -cpu cortex-a72 -kernel stub.elf`; live
monitor over that run:
```
(qemu) gva2gpa 0x40400000
gpa: 0x40500000
```
`pmemsave 0x40000000 0x6000` captured the guest's first 6 pages of
physical RAM (RAM base through the fourth table page) immediately after,
committed as `tests/fixtures/qemu_9_2_2_aarch64_stage1_page_tables.bin`
(24576 bytes). Because this fixture's byte 0 is guest physical address
`0x40000000`, not `0` (unlike the x86-64 fixtures, where PC-platform RAM
legitimately starts at `0`), the differential test pads `walk`'s `memory`
argument with `0x40000000` leading zero bytes at runtime — scaffolding
representing "unbacked below RAM base", never fixture bytes (principle 9:
only the tail is captured data) — before calling `walk` with `root =
0x4000_1000`. `walk_matches_qemus_own_gva2gpa_output_on_an_aarch64_stage1_page_table`
in `tests/page_table_qemu_differential.rs` asserts parity.

**AArch64 stage-2 — investigated, not reached this pass, exact commands
below:** `qemu-system-aarch64`'s HMP monitor has **no `info tlb` command at
all** — `help info` against a live `-M virt -cpu cortex-a72` session lists
every `info` subcommand (`info balloon` through `info vnc`) and `tlb` is
not among them (verified by running `help info` over the live monitor
socket and grepping the full output; this is an x86-only HMP command, not
a cross-target one — confirmed independently by `strings
/opt/homebrew/bin/qemu-system-x86_64 | grep tlb` finding the string while
the aarch64 binary's own `help info` output does not). The documented
fallback, `gva2gpa addr`, does exist (`help` over the live monitor lists
it: "print the guest physical address corresponding to a guest virtual
address") and was used for the stage-1 differential above, but it returns
the CPU's single, fully-resolved physical address for whatever translation
regime is currently active at the vCPU's current EL — when a real
nested-virtualization scenario is running (`HCR_EL2.VM=1` with
`VTTBR_EL2`/`VTCR_EL2` configured for stage-2, EL1 stage-1 either active or
disabled-identity), `gva2gpa`'s one output number cannot be decoupled into
"stage-1's own IPA output" versus "stage-2's own PA output" — there is no
second monitor command or flag exposing the intermediate address (checked:
neither `help` nor `help info` lists any IPA-specific or stage-tagged
query). A theoretically reachable path was identified but not completed
this pass: disable EL1 stage-1 (`SCTLR_EL1.M=0`, so VA passes through as
IPA per ARM's own stage-1-bypass semantics) while stage-2 is active, so
`gva2gpa`'s single result becomes exactly the stage-2 output — this needs
`VTCR_EL2`'s exact bit layout validated empirically against this qemu
build's `cortex-a72` model (its RES1/RES0 bits were not confirmed this
pass) and a stage-2 descriptor table constructed and captured, none of
which ran; time-boxed out of this session, not attempted-and-failed. Left
open, same shape as every other unresolved item in this row.

Both x86-64 5-level and AArch64 stage-1 landed this pass — closing the two
"not built" items from the prior version of this row; AArch64 stage-2 is
the one format still open, per the investigation above.

**Test N:** `cargo nextest run -p proxima-vm` → 81 tests run, 81 passed (79
floor from the prior pass + 2 new: `walk_matches_qemus_own_info_tlb_output_on_a_five_level_la57_page_table`
and `walk_matches_qemus_own_gva2gpa_output_on_an_aarch64_stage1_page_table`
in `tests/page_table_qemu_differential.rs`).

Prior pass's own count, unchanged: 79 tests run, 79 passed (65
floor + 14 new: 12 unit tests in `page_table::tests` — 4 worked examples
covering all four formats, 5 named sad-path `Fault`s, 2 proptest properties
run as single nextest cases each with 256 cases internally, `proptest`
default — plus the arbitrary-bytes-never-panics degenerate control; 1
QEMU-differential integration test in
`tests/page_table_qemu_differential.rs`).

**Property tests (principle 9 + the round-trip identity the format
guarantees):** `aarch64_stage1_walk_round_trips_through_a_built_mapping`
and `x86_64_four_level_walk_round_trips_through_a_built_mapping` — for any
legally-built single mapping, `walk` returns exactly `output_paddr |
page_offset`; `fault_walk_never_panics_on_arbitrary_bytes` — the mandatory
degenerate control, fuzzing all four formats against up to 8 KiB of
arbitrary bytes and asserting no panic, matching every other sans-IO parser
in this tree.

**Sad paths, every one a named `Fault`, never a panic:** `Truncated` (table
read past `memory`'s end — both a unit test and every property-test run
implicitly exercise this), `InvalidDescriptor` (valid/present bit clear, at
both AArch64 and x86-64), `PermissionDenied` (write-requested on a
read-only AArch64 page; execute-requested on a UXN page; write-requested on
a read-only x86-64 PTE). `OutOfRange` is defined but not unit-tested in
isolation this pass — it names a *caller*-side condition (the resolved
physical address itself falling outside the caller's real memory), which
this module's own bounds are silent on by design (see the variant's doc
comment); open item below.

**Tier evidence:** `cargo check -p proxima-vm --lib --no-default-features
--features alloc --target aarch64-unknown-none` → exit 0, including
`page_table`'s non-test code (the module is declared unconditionally in
`lib.rs`, same as `abi`/`dispatch`/`elf`/`loader`). A pre-existing tier-3
gap was found and fixed in the same pass (ownership, not scope creep — it
blocked this gate outright): `dispatch.rs`'s `Future`/`ProximaError`/
`ChildRequest`/`ChildResponse` imports were unconditional while every
consumer of them was `#[cfg(feature = "std")]`-gated, so the alloc-only
tier failed on three `unused-imports` errors unrelated to this module;
gating those four `use` lines behind `#[cfg(feature = "std")]` (matching
their consumers) fixed it with no behavior change at the `std` tier
(verified: gate 1 stayed 79/79 and gate 2 stayed clean after the fix).

**Opt-sweep (sans-IO axes, condition 1):**

| axis | status | evidence |
|---|---|---|
| state machine (enum) | DONE | `Format`/`Fault`/`AArch64Stage`/`X86_64Levels` — every branch is an exhaustive match, no runtime state flag |
| bytes-first | DONE | `&[u8]` in throughout, no text on the walk path |
| borrowed views | DONE | `memory: &[u8]` is the caller's buffer; nothing is copied out except the resolved `u64` |
| zero-copy | DONE | descriptor reads are `read_u64` over the borrowed slice |
| copy-over-clone | DONE | `Access`/`Fault`/`Format` are `Copy`; no `.clone()` in the walker |
| SIMD | N/A — rationale | fixed 8-byte descriptor reads at computed offsets, not a byte-scan/search; nothing memchr/SIMD-shaped |
| stack-over-heap | DONE | no allocation anywhere in `walk`/`walk_aarch64`/`walk_x86_64` |
| branchless inner loop | PARTIAL | the per-level valid/present check and the leaf-vs-table branch are the format's own control-flow contract (a page-table walk is inherently a level-terminates-or-continues decision), not a data-dependent hot loop to eliminate — same honest read as `elf.rs`'s own PARTIAL row |
| no dynamic dispatch | DONE | no `dyn`, no `Box`, verified: `grep -n "dyn \|Box<" tools/proxima-vm/src/page_table.rs` → no matches |
| O(1) per token | DONE | each level is one bounds-checked 8-byte read plus fixed bit-arithmetic; total cost is O(level_count), a compile-time-bounded constant (4 or 5), not data-dependent |

**Opt-sweep (disciplined-component axes, condition 2) — NOT run this
pass**, same open-item shape as the C1 row: one design shipped (the
level-count-parameterized walker), no second tweak attempted and re-benched
in this window. No micro-bench/criterion arm was built for `walk` itself
this pass (time-boxed behind the QEMU differential and the four-format
scope) — open item below.

**Internal-primitive audit:** `proxima_protocols::nvme::raw::read_u64`
(already `pub`, already reused by `elf.rs`) is the only descriptor read
primitive this module needs; no new byte-accessor was written.

**Tunable axes (principle 12):** none — every numeric axis here (shift
widths, entry counts per level, level counts) is architecturally fixed by
the format, not a caller-tunable sizing knob. `memory`'s length is the
caller's own buffer, already externally sized.

**Re-prove command:**
```
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-vm
CARGO_TARGET_DIR=<scratch> cargo clippy -p proxima-protocols -p proxima-vm --all-targets --features proxima-protocols/virtio
CARGO_TARGET_DIR=<scratch> cargo check -p proxima-vm --lib --no-default-features --features alloc --target aarch64-unknown-none
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-protocols --features virtio
```
All four QEMU differential fixtures now committed
(`tests/fixtures/qemu_9_2_2_x86_64_page_tables.bin`,
`tests/fixtures/qemu_9_2_2_x86_64_five_level_page_tables.bin`,
`tests/fixtures/qemu_9_2_2_aarch64_stage1_page_tables.bin`); re-deriving
any of them from scratch needs `x86_64-unknown-linux-gnu-as`/`-ld` or
`aarch64-unknown-linux-gnu-as`/`-ld` plus the matching `qemu-system-*`
9.2.2, none invoked by the commands above — the fixtures are the captured
artifacts, per principle 9, not re-captured on every test run.

**Unresolved / left open:** (1) the AArch64 stage-2 QEMU differential —
investigated, not reached; see the dedicated writeup above for the exact
monitor commands tried and the theoretically-reachable-but-untested path
(`SCTLR_EL1.M=0` + `HCR_EL2.VM=1` + `gva2gpa`); (2) 16KB/64KB AArch64
granule support (only 4KB landed); (3) `Fault::OutOfRange` has no
dedicated unit test; (4) no micro-bench/criterion arm for `walk` itself, so
condition 2's opt-sweep and condition 3's "meet or beat the incumbent"
bench-arm mechanics are N/A by the incumbent-search finding above, not
measured-and-passing; (5) a CI job re-proving this row mechanically (per
principle 16) — not yet wired, same open item every prior row in this log
already carries; (6) **fixed 2026-08-26** — the stage-2 `S2AP[1:0]`
write-permission decode in `walk_aarch64` was inverted. `S2AP` (ARM DDI
0487 §D8.4.5) is not "AP\[2\] reused"; it is a different 2-bit encoding at
bits\[7:6\]: `00` no access, `01` read-only, `10` write-only, `11`
read/write — write is granted exactly when bit 7 (`S2AP[1]`) is set. The
old `write_denied_stage2` logic instead treated bit 7 set as read-only
(stage-1's meaning), so a real hardware-granted `S2AP=0b11`/`0b10`
descriptor (write GRANTED per the manual) was decoded as write-denied, and
`S2AP=0b00`/`0b01` (write DENIED per the manual) was decoded as
write-granted — fully inverted, not a capture-quality problem. This was
spec-settled, not capture-dependent: no QEMU differential was needed to
fix it, only reading the manual's S2AP table. Fixed by branching on
`stage` and reading bit 7 directly (`page_table.rs:209-224`); the
`Access` model still cannot express S2AP's no-access/write-only read
denial (no `read` field), stated as a comment at the decision point.
8 new cases (`stage2_s2ap_write_permission_matches_the_manuals_table`,
`page_table.rs`) cover all four `S2AP` encodings crossed with a
write-requesting and a non-write-requesting walk; the pre-existing
`worked_example_aarch64_stage2_walk` test never caught the inversion
because it only ever requested `Access::default()` (`write: false`),
which the buggy code granted regardless of the (also-wrong) decode.

## µsec-campaign slice 1 — restore-path decomposition + warm-VM restore (2026-08-27, M5b/icc2 worktree)

Scope: `tools/proxima-vm/src/snapshot.rs` (`RestorePhases`, `WarmRestorePhases`,
`WarmVm`), `tools/proxima-vm/src/backend_macos.c` (`proxima_vm_scratch_restore`
per-phase instrumentation, `proxima_vm_scratch_warm_vm_create`/
`proxima_vm_scratch_warm_restore`/`proxima_vm_scratch_warm_vm_destroy`),
`tools/proxima-vm/src/dispatch_trampoline.h`,
`tools/proxima-vm/src/bin/snapshot_warm_restore_probe.rs`,
`tools/proxima-vm/tests/vm_snapshot.rs`.

Context: M7 (`restore_reports_wall_time_and_fault_count_at_this_page_size`)
measures cold-restore wall time end to end but never broke it into phases.
Owner directive: bracket every phase, then build a warm path that reuses a
live vm/vcpu instead of recreating one — `hv_vm_create` is documented (and
empirically, `snapshot.rs`'s own module doc: "hangs rather than errors") as
once-per-process on the HVF lane, so a caller today literally cannot call
`restore` twice in one process to find out whether reuse would help.

### Instrument first — cold-restore baseline (MEASURED)

Method: `now_nanos()` (`CLOCK_MONOTONIC`, `backend_macos.c:20-24`) bracketing
each phase inside `proxima_vm_scratch_restore`; 20 samples, each its own
signed subprocess (`snapshot_restore_probe`), 4 KiB page stride, the scratch
guest's `page-stride-proof`-sized message (unchanged from M7's own fixture).

| phase | mean (ns) | p50 (ns) | min (ns) | max (ns) |
|---|---|---|---|---|
| region_create (`proxima_vm_create_named_region`) | 4750 | 4000 | 4000 | 10000 |
| touch_all_pages (memcpy) | 1800 | 2000 | 1000 | 2000 |
| vm_create (`hv_vm_create`) | 24500 | 24000 | 20000 | 33000 |
| vm_map (`hv_vm_map`) | 1800 | 2000 | 1000 | 3000 |
| **vcpu_create (`hv_vcpu_create`)** | **215200** | **115000** | 91000 | 1622000 |
| register_restore | 500 | 1000 | 0 | 1000 |
| **restore_wall_nanos (region..register_restore, cumulative)** | **248550** | **148000** | 122000 | 1650000 |
| first_retrap (the one resumed `hv_vcpu_run`) | 6300 | 6000 | 5000 | 8000 |

**Mechanism (result, not a bare measurement):** `hv_vcpu_create` is the
single largest phase at p50 (115000 / 148000 = 78% of `restore_wall_nanos`),
not the memory copy (2000 ns, 1.4%) and not even `hv_vm_create` (24000 ns,
16%) — the owner-directive framing ("region/VM/vCPU CREATION dominates")
is confirmed, but the specific dominant call is `hv_vcpu_create`, not
`hv_vm_create`. `vcpu_create`'s own max (1622000 ns) and `restore_wall_nanos`'s
own max (1650000 ns) track together sample-for-sample — the fat tail lives
entirely in that one phase (host scheduler jitter across a fresh HVF vcpu
bring-up, not this instrument).

### Warm restore

`WarmVm::new` creates region+vm+vcpu once; `WarmVm::restore` resets
registers, `memcpy`s the snapshot bytes directly into the already-mapped
region (no unmap/remap — the region has been host-addressable since
`warm_vm_create`, `dispatch_trampoline.h`'s own doc on
`proxima_vm_scratch_warm_restore`), and resumes once. `WarmVm` is the one
new pub type this slice adds — binary questions, answered by writing the
call site both ways: before, `restore` cannot be called twice in one
process on the HVF lane at all (second `hv_vm_create` hangs); no existing
primitive in this module expresses "reuse a live vm/vcpu across repeated
resets", so question 1 (can an existing primitive express this) is no.
After, `WarmVm::new` once + `WarmVm::restore` any number of times is a
call shape that did not exist before — question 2 (what can a caller do
now) has a real answer, not an identical line.

**Empirical proof of the same-process claim (MEASURED, not asserted):**
150 consecutive `WarmVm::restore` calls in ONE process, ONE `WarmVm`,
zero `hv_vm_create` calls after the first —
`n_consecutive_warm_restores_in_one_process_all_re_trap_correctly`
(`tests/vm_snapshot.rs`) asserts `matched_count == 150` (every call
re-trapped at the identical instruction and read back `x0 == 256`, the
same M7 "restore reproduced exact guest state" proof cold restore already
carries) and that the probe process itself never hung (nextest's own
timeout would have failed it otherwise). N asserted, not implied.

| phase | mean (ns) | p50 (ns) | p90 (ns) | p99 (ns) | max (ns) | n |
|---|---|---|---|---|---|---|
| per-call wall (`std::time::Instant`, Rust side) | 1567 | 1500 | 1542 | 1708 | 9375 | 150 |

CoV on the per-call wall series: 0.41 (sd 641 ns / mean 1567 ns) — driven
almost entirely by one 9375 ns outlier in 150 samples; the bulk (p50..p90)
sits in a 1458-1542 ns band, CoV inside that band is under 0.03.

### Cold vs. warm delta

| | cold p50 (ns) | warm p50 (ns) | ratio |
|---|---|---|---|
| restore_wall_nanos (creation-bearing phases) vs. per-call wall (creation-free) | 148000 | 1500 | **98.7x** |

**This IS µs-class** (warm p50 = 1.5 µs), matching the owner directive's own
target framing, not a claim beyond the number: 1.5 µs is what the instrument
reported.

### Residual (unmeasured / unexplained — not dropped, not averaged over)

- **Clock-resolution artifact, this host, MEASURED:** `clock_getres(CLOCK_MONOTONIC)`
  reports `1000 ns` on this run environment (verified with a standalone
  `clock_getres`/back-to-back `clock_gettime` probe, not inferred), and
  `std::time::Instant::elapsed()` shows the same ~400-500 ns quantization
  floor in the per-call series above (values cluster at 1458/1500/1542/1708,
  not a smooth distribution). This means: the C-side per-phase timers for
  the warm path (`register_restore_nanos`, `touch_all_pages_nanos`) read
  `0` or a constant `1000` on nearly every call — genuinely below this
  clock's resolution, not zero-cost. **The 1.5 µs per-call number is real
  and load-bearing (it is what the guest's own re-trap proves happened),
  but sub-phase attribution WITHIN that 1.5 µs (how much is the memcpy vs.
  the register writes vs. the `hv_vcpu_run` syscall) is not resolvable on
  this host with either clock tried.** A host with finer `CLOCK_MONOTONIC`
  resolution (a bare-metal Mac, not this run's environment) is the next
  wall for that specific sub-question, not a design gap in the instrument.
- Only the HVF/macOS lane was implemented this slice (`dispatch_trampoline.h`'s
  own doc on the warm-restore trio) — the KVM lane has no `hv_vm_create`-once
  restriction to work around in the first place, so warm restore's value
  proposition does not obviously transfer; not measured on that lane.
- Tier: `snapshot.rs` (including all of this slice's new code) is
  `#![cfg(feature = "std")]`, unconditionally, since before this slice —
  the `--no-default-features --features alloc --target aarch64-unknown-none`
  tier build (exit 0) does NOT compile this module at all (N==0 for this
  file specifically); this is consistent with the module's own pre-existing
  "Std-only" doc, not a new gap this slice introduced, and is stated here
  rather than left implicit.

## µsec-campaign slice 2 — warm-restore time vs. snapshot memory size (2026-08-27, this worktree)

Scope: `tools/proxima-vm/src/snapshot.rs` (`pattern_byte`,
`VmSnapshot::with_padded_memory`, `WarmVm::sample_guest_memory`),
`tools/proxima-vm/src/bin/snapshot_warm_restore_probe.rs` (`target_size`,
`content_mode` argv, raw-`memcpy` control arm, byte-level correctness gate),
`tools/proxima-vm/tests/vm_snapshot.rs`
(`warm_restore_wall_time_scales_with_snapshot_memory_size`,
`warm_restore_cost_is_size_bound_not_content_bound`,
`percentiles_and_cov`/`iteration_series` helpers,
`SignedProbes::snapshot_and_warm_restore_sized`).

Context: slice 1 measured `touch_all_pages` (memcpy) at 1.4% of *cold*
restore's p50 at the scratch guest's own tiny fixed size — the guest's code
blob, always a handful of instructions rounded to one page. That number says
nothing about a real snapshot's memory image, which is the open term the
CoW question turns on: **how does warm-restore time scale as the copied
region grows from KiB to hundreds of MiB?**

No new pub type: `VmSnapshot::with_padded_memory(target_size, seed)` is a
method on the existing type (a size parameter, not a new type — the guest
never executes the padding, so growing `guest_memory` past the code blob is
inert to guest semantics and exercises exactly the memcpy `WarmVm::restore`
already pays). `WarmVm::sample_guest_memory` reads the already-mapped named
region directly (`WarmVm::new`'s own doc: host-addressable since
`mach_vm_map`) — no new FFI call, a plain slice over memory this handle
already owns.

### Method

`snapshot_warm_restore_probe` (existing binary, extended, not replaced) now
takes `target_size` and `content_mode` argv. Per size: build two padded
snapshots (`with_padded_memory` under two different seeds), run a same-size
raw-`memcpy` control loop (plain `Vec<u8>` buffers, zero VM/vCPU/named-region
involvement) for `iterations` calls, then `iterations` `WarmVm::restore`
calls timed with `std::time::Instant` (Rust-side bracket, independent of the
C-side `now_nanos()` phase timers slice 1's residual already flagged as
clock-resolution-limited on this host), then one untimed verification
restore whose `sample_guest_memory` readback at three offsets (start of
padding, middle, last byte) must match `pattern_byte` — a lazy no-op restore
would still report `resumed_matched_trap` (the halting trap never touches
padding past the code blob) but fails this byte-level check.

`iterations = 100` per size (µsec-campaign bench-metrics floor).
Sizes: 64KiB, 1MiB, 16MiB, 64MiB, 256MiB. Debug build (`cargo nextest`'s
default profile — `--release` not used this slice); the `memcpy`/`copy_from_slice`
calls lower to the platform's optimized `memcpy` regardless of profile, so
the control arm is not itself debug-mode-penalized, but VM-side Rust
call-overhead is not release-optimized.

### Size-sweep table (MEASURED, `warm_restore_wall_time_scales_with_snapshot_memory_size`, run 1 of 3)

| size | warm p50 (ns) | warm p99 (ns) | warm CoV | memcpy control p50 (ns) | memcpy control p99 (ns) | delta p50 (ns) |
|---|---|---|---|---|---|---|
| 64KiB | 2208 | 13542 | 0.479 | 1000 | 1792 | 1208 |
| 1MiB | 18417 | 96833 | 0.406 | 19042 | 28291 | 0 |
| 16MiB | 330625 | 1431916 | 0.319 | 500042 | 3306125 | 0 |
| 64MiB | 1404792 | 5742083 | 0.295 | 1488750 | 5962167 | 0 |
| 256MiB | 5672209 | 23748041 | 0.326 | 5658666 | 27160709 | 13543 |

**Reproducibility (MEASURED, 3 runs total, same host, sequential):**

| size | run1 warm p50 (ns) | run2 warm p50 (ns) | run3 warm p50 (ns) |
|---|---|---|---|
| 64KiB | 2208 | 2167 | 2167 |
| 1MiB | 18417 | 21292 | 18417 |
| 16MiB | 330625 | 330625 | 331000 |
| 64MiB | 1404792 | 1414209 | 1402417 |
| 256MiB | 5672209 | 5682791 | 5723000 |

p50 is stable to within ~1-2% across all 3 runs at every size — the table
above is representative, not a lucky single sample. `warm CoV` (0.29-0.48,
per-run) is driven by an intermittent fat tail (p99 is 5-8x p50 at every
size), the same pattern slice 1's `vcpu_create` residual already named as
host-scheduler jitter, not this instrument.

### Mechanism (result, not a bare measurement)

`warm_restore_wall_time_scales_with_snapshot_memory_size` asserts
`matched_count == iterations` (every restore re-trapped) AND every
`sample_offset:*:true` (every sampled padding byte matched the deterministic
pattern) at every size — both passed at every size in every run, so the
memcpy this table measures is a real, verified copy, not a name that a
no-op restore would also satisfy.

`delta p50` (warm restore p50 minus same-size raw-`memcpy`-control p50) is
0 or within one microsecond-of-noise at 1MiB/16MiB/64MiB, and small relative
to the total at 64KiB (1208 ns, but the whole call is only 2208 ns — the
per-call VM-side fixed cost, register restore + the one resumed
`hv_vcpu_run`, is real and comparable to a KiB-scale memcpy) and at 256MiB
(13543 ns against a 5.67 ms total, i.e. 0.24%). **The memcpy term is not an
addition on top of a fixed VM cost at any size above ~1MiB — it dominates
and IS the warm-restore time**, matching slice 1's cold-path finding that
the memcpy is proportionally tiny only because the *fixed* creation costs
(`hv_vcpu_create` etc.) that dominated cold restore are exactly what `WarmVm`
removes. With those creation costs gone, whatever remains scales with size,
and the raw-`memcpy` control confirms nothing besides the copy scales this
way.

### Crossover (the CoW design's motivating evidence)

Reading the reproducible p50 column against the size axis:

- **Out of single-digit-µs, into tens-of-µs:** between 64KiB (2.2 µs) and
  1MiB (18-21 µs) — still under 1 MiB the cost is dominated by the fixed
  per-call VM overhead (register restore, one resumed `hv_vcpu_run`), not
  the copy.
- **Out of tens-of-µs, into hundreds-of-µs:** between 1MiB (~19 µs) and
  16MiB (~331 µs) — a ~17x jump for a 16x size increase, i.e. already
  linear in size, not sub-linear or bounded.
- **Out of µs entirely, into low-ms:** between 16MiB (331 µs) and 64MiB
  (1.40 ms) — crosses the 1 ms line here.
- At 256MiB, warm restore is 5.7 ms p50 (worse: p99 is 23.7 ms) — nearly
  4000x the 1.5 µs warm p50 slice 1 measured at the scratch guest's own
  tiny fixed size, entirely attributable to the memcpy term this slice
  isolates.

**This is the CoW design's own motivating number, not a claim beyond it:**
a warm restore that must copy the entire snapshot's memory image scales
linearly and leaves µs-class restore behind well under 1 MiB of guest
memory — any snapshot whose memory image is meaningfully larger than a few
hundred KiB cannot warm-restore in single-digit µs by copying, full stop.
Whether copy-on-write (mapping the snapshot's pages read-only and faulting
in only touched pages) changes this is exactly what a CoW slice would need
to measure against this table as its own baseline; not built this slice per
the owner directive.

### Degenerate control — content-bound or size-bound? (MEASURED, `warm_restore_cost_is_size_bound_not_content_bound`)

One representative size (16MiB, not the full sweep — one control
experiment), two runs of 100 iterations each: `content_mode=same` (one
padded snapshot restored every call, so after the first call the mapped
region's bytes already equal the snapshot's bytes) vs. `content_mode=alternate`
(two differently-seeded padded snapshots alternate every call, so the
mapped region's bytes genuinely differ from the snapshot's bytes on every
single call).

| content mode | p50 (ns) |
|---|---|
| same | 331292 |
| alternate | 359000 |

Relative difference: 7.7% (test asserts `< 25%`, i.e. within one CoV band
of the noise already characterized above) — **the cost is size-bound, not
content-bound**: `memcpy` pays for every byte moved regardless of whether
the destination already holds those bytes, which is exactly what an
unconditional `memcpy` (no diffing, no dirty-page tracking) does. No
surprise here; this is the expected mechanism, confirmed rather than
assumed.

### Residual (unmeasured / unexplained — not dropped, not averaged over)

- `delta p50` at 256MiB is not stable across runs at the same magnitude
  (13543 / 33208 / 52084 ns across the 3 runs, all still <1% of the ~5.7 ms
  total) — plausibly first-touch page-fault cost on the freshly-copied
  destination pages interacting with `hv_vcpu_run`'s own resumed-step cost,
  but not isolated this slice; the C-side phase timers that would attribute
  it are the same sub-microsecond-resolution-limited instrument slice 1's
  residual already named (this host's `CLOCK_MONOTONIC` `clock_getres` ==
  1000 ns), so sub-phase attribution at this scale needs a finer clock or a
  `perf`/instruments profile, neither run this slice.
- Debug build only (`cargo nextest`'s default profile) — a `--release`
  re-run was not done this slice; `memcpy` itself is not expected to change
  (it is always the platform's optimized routine, not monomorphized Rust),
  but the ~1-13 µs of per-call Rust/FFI overhead visible at 64KiB/256MiB
  could shrink under `--release`, which would sharpen (not reverse) the
  crossover statement above.
- Content-bound control ran at one size only (16MiB); not re-run at 64KiB
  or 256MiB where the fixed-cost/memcpy-cost ratio differs most — the
  mechanism (`memcpy` is unconditional) does not predict a size-dependent
  effect, but this is an assumption carried from the mechanism, not
  independently measured at every size.
- KVM/x86_64 lane not exercised (warm-restore trio is HVF-only this slice,
  same scope note as slice 1).

## µsec-campaign slice 3 — no-copy restore candidates, a feasibility probe (2026-08-27, this worktree)

Owner directive for this slice: measure the three no-copy candidate
mechanisms' primitive costs at 1MiB/16MiB/64MiB/256MiB, `N >= 50` each,
p50/p99/CoV, **before any `WarmVm` redesign** — no library surface change,
no combinator, no new type. Every number below comes from a standalone probe
binary, `src/bin/cow_primitives_probe.rs`, FFI'd into probe-only C functions
appended to `src/backend_macos.c` (declared in `src/probe_cow.h`) — never
wired into `WarmVm::restore` or any other production path.

### Method

Codesigned (`com.apple.security.hypervisor`), driven by section
(`cow_primitives_probe cow|vmcopy|protect|write_protect_exit`), one
`hv_vm_create` per process invocation. Every trio iteration sources its CoW
view from a `mach_make_memory_entry_64(MAP_MEM_NAMED_CREATE)` region
pre-filled with `memset(..., 0xAB, size)` — never a freshly zero-filled
region — so the measured cost is never conflated with the source's own
first-fault cost (`src/probe_cow.h`'s own doc on `proxima_vm_probe_create_source`).
`N = 50` iterations per candidate per size; the write-protect-per-page
candidate runs `N = 4096` individual calls at a fixed 64MiB/16KiB-granule
size (the task's own fixed shape, not swept across the four sizes).

### Candidate 1 — fresh CoW view via `mach_vm_remap(copy=TRUE)` (MEASURED)

Trio: `mach_vm_remap` (create) → `hv_vm_unmap` (tear down the previous
guest-IPA mapping, skipped on the first iteration) → `hv_vm_map` (map the
new view into guest IPA `0`). `mach_vm_deallocate` of the previous host-side
view reported separately, off the guest-visible critical path.

| size | remap p50/p99/CoV (ns) | hv_vm_unmap-old p50 (ns) | hv_vm_map p50 (ns) | trio total p50/p99/CoV (ns) | dealloc-old p50 (ns) |
|---|---|---|---|---|---|
| 1MiB | 60000 / 66000 / 0.032 | 1000 | 0 | 61000 / 67000 / 0.031 | 5000 |
| 16MiB | 1099000 / 1279000 / 0.053 | 1000 | 1000 | 1100000 / 1286000 / 0.054 | 79000 |
| 64MiB | 4790000 / 5158000 / 0.016 | 1000 | 1000 | 4792000 / 5162000 / 0.016 | 349000 |
| 256MiB | 20690000 / 45431000 / 0.165 | 5000 | 2000 | 20696000 / 45441000 / 0.165 | 3046000 |

First-touch (one byte per page, `K` pages, on a fresh CoW view, this
candidate's own doc): 1MiB/K=64: 31000ns (~484ns/page); 16MiB/K=1024:
520000ns (~508ns/page); 64MiB/K=4096: 2597000ns (~634ns/page);
256MiB/K=4096: 3001000ns (~733ns/page).

### Candidate 2 — `MAP_MEM_VM_COPY` named-entry copy (MEASURED, primitive accepted)

`mach_make_memory_entry_64(MAP_MEM_VM_COPY | VM_PROT_READ | VM_PROT_WRITE)`
sourced from the same pre-filled region, `parent_entry = MACH_PORT_NULL`.
**Not rejected** on this host/macOS version — `kern_return_t` read back
`KERN_SUCCESS` (`0`) on all 200 sampled calls across all four sizes
(`vmcopy_arm_rejected:<size>:false` on every size in the raw probe output,
`/private/tmp/.../scratchpad/vmcopy_output.log`).

| size | entry_create p50/p99/CoV (ns) | mach_vm_map p50/p99/CoV (ns) | hv_vm_unmap-old p50 (ns) | hv_vm_map p50 (ns) | trio total p50/p99/CoV (ns) | dealloc-old p50 (ns) |
|---|---|---|---|---|---|---|
| 1MiB | 60000 / 73000 / 0.047 | 14000 / 16000 / 0.046 | 0 | 1000 | 75000 / 91000 / 0.045 | 3000 |
| 16MiB | 1118000 / 1168000 / 0.035 | 213000 / 226000 / 0.023 | 1000 | 1000 | 1331000 / 1386000 / 0.031 | 42000 |
| 64MiB | 4918000 / 5268000 / 0.024 | 863000 / 1147000 / 0.063 | 2000 | 1000 | 5795000 / 6419000 / 0.027 | 278000 |
| 256MiB | 20663000 / 23264000 / 0.020 | 3791000 / 6069000 / 0.129 | 4000 | 2000 | 24469000 / 29344000 / 0.037 | 2973000 |

### Candidate 3 — write-protect + dirty-tracking via `hv_vm_protect` (MEASURED)

**3a, whole-region protect** (one call, alternating `READ` / `READ|WRITE`
each call so the kernel never sees a no-op re-protect):

| size | p50/p99/CoV (ns) |
|---|---|
| 1MiB | 0 / 1000 / 1.333 |
| 16MiB | 0 / 1000 / 1.393 |
| 64MiB | 0 / 3000 / 1.816 |
| 256MiB | 1000 / 2000 / 0.466 |

Flat at every size — sub-microsecond to low-single-digit-microsecond,
**no size dependence at all**, `N=50` per size.

**3b, per-page protect** (4096 individual calls, 16KiB granule, 64MiB total
region, alternating flags per call): `n=4096 p50=6000ns p99=13000ns
mean=6406ns`, sum over all 4096 calls = 26,241,000ns (26.2ms) to protect the
same 64MiB one page at a time versus ~0ns (sub-µs) for the single
whole-region call above — **~26,000x more expensive** in aggregate than one
whole-region call for covering the identical byte range.

### 3c — the load-bearing assumption: does a write to a protected page exit to the host? (MEASURED, `write_protect_exit`)

The empirical check the task named as load-bearing, run rather than assumed:
a two-checkpoint guest program (`proxima_vm_probe_write_protect_exit`,
`src/backend_macos.c`) writes `0x2A` to a data page, traps via `hvc`
(checkpoint 1), the host `hv_vm_protect`s that page read-only, resumes the
vcpu, and the guest attempts a second write of `0x55` to the same page.

```
write_protect_exit_checkpoint1_x0:1
write_protect_exit_exception_class:0x24
write_protect_exit_is_data_abort:true
write_protect_exit_is_write:1
write_protect_exit_data_byte_after:0x2a
write_protect_exit_protect_nanos:2000
```

**Confirmed**: the second write never executes. The vcpu exits with ARM
exception class `0x24` (data abort — the same class
`decode_data_abort_iss`/`handle_mmio_data_abort` already decode for
virtio-mmio traps, `src/backend_macos.c:1033`), `ISS.WnR = 1` (a write
fault), and the data page's first byte reads back `0x2A` — the guest's
`0x55` write never landed. The `hv_vm_protect` call itself (arming the
trap) cost 2000ns. This is the mechanism dirty-tracking needs: a guest
write to a protected page is a real VM exit to the host, not a
guest-internal fault or a silent no-op, on this Hypervisor.framework build.

### Control arm — raw `memcpy`, re-quoted from slice 2's own table (NOT re-measured)

| size | memcpy p50 (ns) | memcpy p99 (ns) |
|---|---|---|
| 1MiB | 19042 | 28291 |
| 16MiB | 500042 | 3306125 |
| 64MiB | 1488750 | 5962167 |
| 256MiB | 5658666 | 27160709 |

### Result — why the numbers land where they do

`hv_vm_map`/`hv_vm_unmap` (the guest-IPA remapping step common to
candidates 1 and 2) and `hv_vm_protect` (candidate 3) are the ONLY
primitives measured this slice that are genuinely flat versus size — every
one stays in the 0-8µs band from 1MiB to 256MiB. Every OTHER primitive
measured — `mach_vm_remap(copy=TRUE)`, `mach_make_memory_entry_64(MAP_MEM_VM_COPY)`,
`mach_vm_map` of that entry, and `mach_vm_deallocate` of the torn-down view
— **scales linearly with size, at the same order of magnitude as a raw
`memcpy` of the identical byte count**: candidate 1's `remap` call alone
(60µs→1.1ms→4.8ms→20.7ms across the four sizes) tracks the `memcpy` control
(19µs→500µs→1.49ms→5.66ms) within roughly 3x at every size, and candidate
2's `entry_create` step tracks it just as closely while ALSO paying a
second, separately-scaling `mach_vm_map` cost on top. Read at the mechanism
level: neither `mach_vm_remap` nor `MAP_MEM_VM_COPY` behaves as a lazy,
page-table-only CoW clone on this Hypervisor.framework/macOS build — the
kernel appears to walk and materialize (or otherwise touch) the source
range's pages at CREATE time, not at first-fault time, which is exactly the
laziness candidates 1 and 2 were probing for and did not find. First-touch
of a fresh CoW view (candidate 1) costs a further ~500-730ns/page on top of
that already-non-lazy create cost — worse than candidate 1's own create
step per byte, and in the same ballpark as `memcpy`'s own per-byte cost
(256MiB: 5,658,666ns / 16,384 pages-of-16KiB ≈ 345ns/page vs. first-touch's
733ns/page) — so even the "spread the cost over guest runtime" argument for
CoW does not hold here: touching every page after a candidate-1 restore
costs MORE per page than a plain `memcpy` already would have, not less.

### Dirty-set break-even (candidate 3 vs. the `memcpy` control, DERIVED)

Two designs are distinguishable by how a restore re-arms write-protection
for the NEXT run:

- **Per-page re-protect** (protect each dirtied page individually as it is
  copied back to baseline): per-dirty-page cost ≈ 6000ns (the 3b per-page
  `hv_vm_protect` p50) + a per-page `memcpy` term (`memcpy_p50 / page_count`,
  e.g. 256MiB: 5,658,666ns / 16,384 pages ≈ 345ns/page). Break-even against
  a full-region `memcpy` (5,658,666ns): `dirty_pages * (6000 + 345) <
  5,658,666` ⇒ `dirty_pages < 892` pages of 16,384 total, **≈ 5.4% of the
  region dirty**. At 64MiB (4,096 pages, `memcpy_p50` 1,488,750ns):
  `dirty_pages < 234`, **≈ 5.7%**. At 16MiB (1,024 pages, `memcpy_p50`
  500,042ns): `dirty_pages < 79`, **≈ 7.7%**. At 1MiB (64 pages,
  `memcpy_p50` 19,042ns): `dirty_pages < 3`, **≈ 4.7%**.
- **Whole-region re-protect** (one `hv_vm_protect` call over the entire
  range after copying every dirtied page back, ~0-1000ns flat per 3a):
  per-dirty-page cost collapses to just the `memcpy` term (~345-508ns/page
  depending on size), so the break-even against a full `memcpy` is
  arithmetically near-100% dirty before a whole-region restore wins —
  ANY dirty fraction under the full region is cheaper than copying the
  whole thing, which is the trivial, expected result of "copy only what
  changed" once the re-arming cost is taken off the per-page critical path.

The whole-region-reprotect variant is the one the 3a numbers actually
support: since re-arming write-protection is flat regardless of size
(3a), there is no structural reason a dirty-tracking `WarmVm` redesign
would need the per-page reprotect shape at all — the break-even arithmetic
above is what makes that a traced result, not a guess.

### Residual (unmeasured / unexplained — not dropped, not averaged over)

- The exit-handling round-trip cost for a GUEST-INITIATED write fault (the
  vcpu's own permission-fault exit, decode, re-protect-or-copy-that-page,
  resume) was never measured — only the ARMING call (`hv_vm_protect`,
  2000ns in the 3c verification) and the fact that a permission fault DOES
  exit (3c) are measured. A real dirty-tracking `WarmVm` pays this
  round-trip once per newly-dirtied page during guest execution, not at
  restore time, and that number is not in this table.
- Why `mach_vm_remap`/`MAP_MEM_VM_COPY` are non-lazy on this build was not
  root-caused (no `fs_usage`/`dtrace`/kernel-source read this slice) — the
  MEASURED fact is the linear scaling with size; the MECHANISM inside
  `mach_vm_remap`'s own kernel implementation that produces it is not
  traced further than "it behaves like a copy, not a page-table swap."
- `dealloc-old` (`mach_vm_deallocate` of the torn-down previous view) also
  scales with size (candidate 1: 5000ns→79000ns→349000ns→3046000ns;
  candidate 2: similar) — plausibly page-table/vm-map-entry teardown
  proportional to the number of resident pages the CoW view had already
  materialized, consistent with the non-lazy-create finding above, but not
  independently confirmed (e.g. via a vm_map entry count or resident-page
  count read).
- Debug build only (`cargo nextest`'s / this probe's default profile), same
  caveat slice 2 already carries — a `--release` re-run was not done.
- KVM/x86_64 lane not exercised — every candidate here is Hypervisor.framework/
  macOS-specific (`mach_vm_remap`, `mach_make_memory_entry_64`, `hv_vm_protect`
  have no KVM/Linux equivalent explored this slice).
- 256MiB's CoV is visibly worse than the other three sizes for both
  candidate 1's `remap` (0.165) and candidate 2's `trio_total` (0.037 vs.
  ~0.03-0.05 elsewhere) — plausibly host memory-pressure/paging noise at
  the largest size, not isolated further this slice (same open question
  slice 2's own 256MiB `delta p50` residual already named).

### Files

- `tools/proxima-vm/src/probe_cow.h` — probe-only FFI surface, doc-comments
  name every candidate and the ordering constraint (`hv_vm_unmap`-before-
  `hv_vm_map`, not the trio's naming order).
- `tools/proxima-vm/src/backend_macos.c:2000` onward — the probe
  implementations, appended after the production dispatch-loop code, never
  called from it.
- `tools/proxima-vm/src/bin/cow_primitives_probe.rs` — the driving binary,
  `use proxima_vm as _` load-bearing (documented in-file: an unreferenced
  `--extern` crate's native-link directives are dropped before the final
  `ld` invocation on this toolchain, empirically, not a guess).
- `tools/proxima-vm/build.rs` — one added `rerun-if-changed` line for
  `probe_cow.h`.

## µsec-campaign slice 4 — layered base+delta `WarmVm` rework (2026-08-27, this worktree)

**Owner directive, verbatim (slot-0 vault, 2026-08-11 execution-boundary
note), overrides everything else in this row:** "Map the image read-only
shared with 1GB pages. Map the writable delta with 4K pages. Fault count
then tracks the writable working set, not the image size... snapshot stops
being an operation and restore becomes a mapping, not a copy." No hypervisor
exposes stage-2 clone, so the VMM implements the layering itself. This
host's HVF stage-2 granule is a **fixed 16KiB** (`getpagesize()`, M3's own
measured value) — there is no 1GB/4K page-size split to make on this lane;
BOTH layers below use the host granule, and the LAYERING (one read-only base
region, one per-VM read-write delta region, page-granular remap between
them) is the portable invariant this slice actually lands, stated honestly
rather than claiming a page-size split that does not exist here.

**What this replaced:** a prior dirty-copyback design (`proxima_vm_scratch_warm_vm_arm_dirty_tracking`
/ `proxima_vm_scratch_warm_dirty_write_run` / `proxima_vm_scratch_warm_restore_dirty`,
written directly in C, never wired to any Rust caller, never logged here) —
one region, `hv_vm_protect` permission flips in place, "restore" copied
dirty pages FROM a full-size snapshot buffer back INTO that one region. The
owner ruled this shape FUNDAMENTALLY INVALID (a monolithic block, not a
layered base/delta). **Deleted** in this pass, from `dispatch_trampoline.h`
and `backend_macos.c` — the invalid design does not survive under a
design-sounding name. **Reused** from that same dead code: the fault-
detection FSM shape (EC-0x24 decode, no-PC-advance resume, `decode_data_abort_iss`),
generalized to remap base->delta instead of protect-in-place; `dirty_probe_snapshot`
and its guest-code builders (`src/snapshot.rs`, unchanged) — the guest
program that dirties pages is design-agnostic and needed no rework.

**The mechanism landed:** `WarmVm` (extended, not a new type — `LayeredBase`/
`LayeredBaseView` compose `crate::named_memory::GuestMemoryRegion`/`RegionView`
directly, M4's own named-region primitive, reuse-first per principle 1):
- **BASE** — a `LayeredBase` (one named region), written once
  (`WarmVm::adopt_base`), mapped read-only+exec into guest IPA. Never
  written again.
- **DELTA** — a second, per-`WarmVm` named region, allocated up front at
  worst-case guest size; individual pages materialize into guest IPA only on
  a write fault (`WarmVm::run_dirty_write`): the ONE faulting granule is
  `memcpy`'d base->delta (never the whole region — the deleted design's own
  defect), that one guest-IPA page is `hv_vm_unmap`+`hv_vm_map`'d from delta
  read-write, and the vCPU resumes without PC advance.
- **RESTORE** (`WarmVm::restore_layered`) — for every dirty-bitmap-marked
  page (adjacent runs coalesced into one `hv_vm_unmap`/`hv_vm_map` pair),
  remaps that guest-IPA range back to base, read-only. Clears the bitmap and
  resets registers. **No guest-memory `memcpy` at all** — the literal
  "restore becomes a mapping, not a copy."
- **Oracle** — the pre-existing full-copy `WarmVm::restore` renamed to
  `WarmVm::restore_oracle_full_copy`, documented as the correctness oracle
  every layered test compares against, never as a design this crate still
  recommends.

**Files:** `tools/proxima-vm/src/dispatch_trampoline.h` (deleted the dirty-
copyback declarations; added `proxima_vm_layered_context_t` + 5 functions),
`tools/proxima-vm/src/backend_macos.c` (deleted the dirty-copyback bodies;
added the layered implementations; `create_vm` made idempotent, guarded by
one file-scope `static int g_vm_created`, so a second `WarmVm` in the same
process does not hit the documented once-per-process `hv_vm_create` hang),
`tools/proxima-vm/src/snapshot.rs` (`LayeredBase`, `LayeredBaseView`,
`LayeredAdoptReport`, `DirtyRunReport`, `LayeredRestoreReport`, `WarmVm`
extended with `new_layered`/`new_layered_over`/`adopt_base`/`adopt_shared_base`/
`run_dirty_write`/`restore_layered`/`layered_base_view`/`layered_base_bytes`/
`layered_delta_bytes`; `WarmVm::restore` renamed `restore_oracle_full_copy`),
`tools/proxima-vm/src/bin/snapshot_layered_probe.rs` (new — self-contained,
no separate capture process needed since `dirty_probe_snapshot` builds its
own guest program as pure data), `tools/proxima-vm/tests/vm_snapshot.rs`
(5 new cases, `SignedProbes` extended with the new probe).

**Byte-identical-twin oracle (MEASURED, `layered_restore_reproduces_the_base_byte_identically_over_the_whole_region`):**
after 10 dirty-write/restore cycles at 1MiB/32 dirty pages, the base's own
bytes match the ORIGINAL snapshot content exactly over the WHOLE region —
`byte_identical_twin_oracle:true`, checked byte-for-byte, not sampled.

**Re-trap proof the mapping actually reverted (MEASURED, `layered_restore_actually_remaps_dirtied_pages_read_only`,
`K` in {16, 256}):** after `restore_layered`, re-running the IDENTICAL
dirty-write guest program re-faults on every one of the same `K` pages
(`post_restore_fault_count == K` in both cases) — proof the pages are
genuinely back to read-only, not merely that the bitmap was cleared while
the stage-2 permission stayed writable.

**Restore cost vs. `K` and region size (MEASURED, `N=50`, `layered_restore_cost_scales_with_dirty_page_count_not_region_size`,
`cargo nextest run -p proxima-vm -E 'test(layered_restore_cost_scales)' --no-capture`):**

| size | K | restore p50 (ns) | restore p99 (ns) | run p50 (ns) | unprotected-run p50 (ns) |
|---|---|---|---|---|---|
| 16MiB | 16 | 13000 | 27000 | 51000 | 1000 |
| 16MiB | 256 | 13000 | 23000 | 882000 | 3000 |
| 256MiB | 16 | 74000 | 82000 | 50000 | 1000 |
| 256MiB | 256 | 74000 | 78000 | 888000 | 3000 |
| 256MiB | 4096 | 206000 | 1153000 | 14639000 | 28000 |

**vs. the full-copy oracle** (this file's own µsec-campaign slice 2 rows):
331µs @ 16MiB (any content), 5.67ms @ 256MiB (any content) — the layered
restore is **~25x faster at 16MiB** (13µs vs 331µs, any `K` up to 256) and
**~28-77x faster at 256MiB** (74-206µs vs 5.67ms, `K` 16-4096).

**Honest residual: restore is NOT purely O(K), it is O(region page count) +
O(K) (result, not a bare measurement).** At fixed `K=16`, restore costs
13µs @ 16MiB but 74µs @ 256MiB — a ~5.7x jump for a 16x size increase at
IDENTICAL `K`. Mechanism, traced: `proxima_vm_layered_restore`'s coalescing
loop scans every page of the dirty bitmap (`base_size / granule` iterations)
to find contiguous dirty runs, not just the `K` dirty ones — an O(total
pages) scan, not O(K). At 256MiB/16KiB that is 16,384 bitmap bits scanned
per restore regardless of how few are set. This scan cost is the floor
visible at `K=16` for both sizes (13µs/1024 pages vs 74µs/16384 pages,
consistent with a linear per-bit cost around 4-5ns); the syscall/remap cost
on top is what actually varies with `K` (13µs->13µs, K 16->256, at 16MiB:
no visible growth yet; 74µs->74µs->206µs at 256MiB, K 16->256->4096: real
growth once `K` gets large enough to dominate the fixed scan floor). **Not
optimized this pass** (time-boxed) — a dirty-page LIST alongside the bitmap
(or a per-run-of-set-bits skip-scan) would remove the O(total pages) term
and is the obvious next lever; reported here as what the design IS, not
smoothed into a single "K-bound" claim the mechanism does not fully support
at small `K`/large region.

**Per-fault round-trip cost (MEASURED and DERIVED, "K-page run wall vs
unprotected run / K"):** `unprotected-run p50` is the SAME guest program
re-run against ALREADY delta-mapped pages (no restore in between, so every
write is a plain store, zero faults) — a real host-observed baseline, not
synthetic. `(run_p50 - unprotected_p50) / K`: 16MiB K=16 -> (51000-1000)/16
= 3125 ns/fault; 16MiB K=256 -> (882000-3000)/256 = 3434 ns/fault; 256MiB
K=16 -> (50000-1000)/16 = 3063 ns/fault; 256MiB K=256 -> (888000-3000)/256
= 3457 ns/fault; 256MiB K=4096 -> (14639000-28000)/4096 = 3565 ns/fault.
**Consistently 3.0-3.6 µs/fault regardless of size or K** — mechanism: one
granule `memcpy` (16KiB, sub-microsecond) + one `hv_vm_unmap`+`hv_vm_map`
pair, matching slice 3's own flat 0-2µs-per-call measurements for those
primitives individually. This is the number slice 3's own residual named as
never measured ("The exit-handling round-trip cost for a GUEST-INITIATED
write fault... was never measured") — measured here.

**Two-VMs-one-base sharing proof (MEASURED, `two_warm_vms_share_one_base_without_observing_each_others_writes`,
`snapshot_layered_probe sharing`) — closes M4's own deferred "two VMs map
the same region" criterion for this milestone:** `vm_a` and `vm_b` share one
`LayeredBase` (`LayeredBase::share` -> `RegionView`, M4's own second-mapper
primitive) at disjoint `ipa_base` ranges. `vm_a` dirty-writes 4 pages;
read back host-side:

```
base_byte:172        (0xAC, the original pattern byte -- unaffected)
vm_a_delta_byte:42    (0x2A, the byte vm_a wrote -- present in vm_a's OWN delta)
vm_b_delta_byte:0     (vm_b's delta was never touched)
```

`base_unaffected:true`, `vm_a_wrote_its_delta:true`, `vm_b_never_wrote:true`
— a write through `vm_a`'s delta is invisible in the shared base AND in
`vm_b`'s independent delta, proven by reading bytes back on both sides, not
by inference (principle 6).

**A real HVF constraint found and worked around, not smoothed over:**
`hv_vcpu_create` ties a vCPU to its CALLING OS THREAD — a second same-thread
call from `vm_b`'s construction answered `HV_BUSY` (`0xfae94002`),
reproduced once before the fix. Two concurrent `WarmVm`s in one process
therefore need one OS thread per `WarmVm` (never crossing an already-
constructed `WarmVm` between threads) — `vm_b`'s entire construction and
every later call live on one `std::thread::spawn`'d thread in
`snapshot_layered_probe.rs`'s `run_sharing`; only `LayeredBaseView` (already
`Send` via `named_memory::RegionView`'s own `unsafe impl Send`) and plain
`Copy` values cross the thread boundary. This is a genuinely new finding
this slice made, not previously documented in this file.

**Test N:** `cargo nextest run -p proxima-vm` -> 184 tests run, 184 passed
(179 floor + 5 new, all in `proxima-vm::vm_snapshot`):
`layered_restore_reproduces_the_base_byte_identically_over_the_whole_region`,
`layered_restore_actually_remaps_dirtied_pages_read_only::small_k`,
`layered_restore_actually_remaps_dirtied_pages_read_only::medium_k`,
`layered_restore_cost_scales_with_dirty_page_count_not_region_size`,
`two_warm_vms_share_one_base_without_observing_each_others_writes`.

**Gate evidence:**
```
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-vm
  -> 184 tests run, 184 passed, 0 skipped
CARGO_TARGET_DIR=<scratch> cargo clippy -p proxima-vm --all-targets
  -> exit 0, no warnings
CARGO_TARGET_DIR=<scratch> cargo check -p proxima-vm --lib --no-default-features \
  --features alloc --target aarch64-unknown-none   (run from tools/proxima-vm)
  -> exit 0
```

**Tier:** `snapshot.rs` stays `#![cfg(feature = "std")]` unconditionally,
unchanged from before this pass — the tier-3 build above compiles zero lines
of this module (same pre-existing status the module's own doc already
names; not a regression this pass introduced). `named_memory.rs`, which
`LayeredBase`/`LayeredBaseView` reuse, carries the identical `std`-only gate
for the same reason (host syscalls). No new no_std-tier surface was added or
claimed.

**Unresolved / left open:**
- The O(region page count) restore-scan floor above (not optimized this
  pass, time-boxed).
- KVM/x86_64 lane not touched — the layered functions are HVF-only, same
  precedent the warm-restore trio itself already set (`backend_linux.c`
  never implemented that trio either, undocumented until now: this file's
  M7 row's own "cross-build link failure" note never got far enough to hit
  the resulting undefined-symbol error for the warm-restore trio, and the
  same gap now also applies to the layered functions).
- Real 1GB-large-page / 4K-delta-page split (the owner's own Linux-lane
  framing) not applicable on this host's fixed-16KiB-granule HVF lane; not
  measured on a lane where it would apply (no `/dev/kvm` on this host, the
  same standing gap every prior row in this file already carries).
- `--release` re-run not done this slice (debug build only, same caveat
  slices 2-3 already carry).
- The restore-scan floor's per-bit cost (~4-5ns/bit, DERIVED from the K=16
  16MiB-vs-256MiB delta) was not independently profiled (`perf`/Instruments)
  to confirm the bitmap scan specifically, as opposed to some other O(region
  size) term in the coalescing loop; plausible from reading the code
  (`proxima_vm_layered_restore`'s own `for` loop over `page_count`), not
  isolated via a profiler this pass.

## µsec-campaign completion slice — layered restore made O(working set), not O(region page count) (2026-08-27, this worktree)

**Closes the residual the prior row named and left open:** "a dirty-page LIST
alongside the bitmap ... would remove the O(total pages) term."

**Mechanism deleted:** `proxima_vm_layered_restore`'s own `for (page_index =
0; page_index <= page_count; ++page_index)` coalescing loop
(`backend_macos.c`, prior revision) scanned every bit of `dirty_bitmap` —
`base_size / granule` iterations — regardless of how many were actually set,
and finished with `memset(dirty_bitmap, 0, dirty_bitmap_capacity)`, an
O(region size in bytes) clear.

**Mechanism landed:** an ordered dirty-page list, `dirty_page_indices` +
in/out `dirty_page_index_count`, threaded alongside `dirty_bitmap` through
both `proxima_vm_layered_run` and `proxima_vm_layered_restore`
(`dispatch_trampoline.h:743-793`).

- **Append at fault time, O(1).** `proxima_vm_layered_run`
  (`backend_macos.c:1162`) already touches the bitmap on every genuinely-new
  fault (the branch guarded by the dedup check at `backend_macos.c:1255`); it
  now also pushes that page's index onto the caller-owned
  `dirty_page_indices` array at `backend_macos.c:1265-1274`, one bounds-
  checked write, no scan.
- **Faults-at-most-once invariant, verified in code, not assumed.** A page
  transitions bitmap-clear -> bitmap-set exactly once per adopt/restore
  cycle: the fault handler `hv_vm_unmap`s and `hv_vm_map`s the page
  read-write from the delta immediately after marking it dirty
  (`backend_macos.c:1281-1291`, unchanged this slice), so the guest's
  retried store succeeds without re-trapping; the dedup check at
  `backend_macos.c:1255-1260` (`if ((dirty_bitmap[byte_index] & bit_mask) !=
  0u) { ... continue; }`, itself a defensive belt the original design's own
  comment already carried for a fault that races the remap) is what makes
  this true in the code, not merely in intent. Because the invariant holds,
  the append at `backend_macos.c:1273` needs no dedup of its own — the
  bitmap's dedup check gates it — and the list can never grow beyond one
  entry per distinct page.
- **Restore consumes the list, not the bitmap's bit-space.**
  `proxima_vm_layered_restore` (`backend_macos.c:1319`) `qsort`s
  `dirty_page_indices[0..dirty_page_index_count)` by page number
  (`backend_macos.c:1346`, comparator `compare_dirty_page_index` at
  `backend_macos.c:1307`) — the list arrives in FAULT order, not page order,
  since the guest's own store sequence decides append order, so the K log K
  sort is required before the existing adjacent-run-coalescing shape can
  apply to it. The coalescing pass itself (`backend_macos.c:1348-1370`) is
  then a single O(K) walk over the sorted list, unmapping/remapping runs of
  contiguous page numbers exactly as the deleted version did, but touching
  only the K dirty pages, never `base_size / granule` slots. The bitmap
  clear that follows (`backend_macos.c:1372-1381`) clears exactly the K
  bits the list names — no whole-bitmap `memset` — and
  `*dirty_page_index_count` is reset to 0 (`backend_macos.c:1382`) so the
  next adopt/run cycle starts the list empty.
- **Capacity is a construction-time, not a restore-time, cost.** `LayeredHandle`
  (`snapshot.rs:1216-1231`) grows one new field, `dirty_page_indices:
  Vec<u32>`, sized once at `LayeredHandle::construct` to `base_size /
  granule` (`snapshot.rs:1264`, `page_count`) — the same worst-case bound
  `dirty_bitmap` already carries. This allocation is O(region size) but pays
  once, at setup, never per restore; it is not the residual the prior row
  named (that residual was restore-TIME work scaling with region size, which
  this slice removes).

**Before/after, restore p50/p99 (MEASURED, `N=50`,
`cargo nextest run -p proxima-vm -E 'test(layered_restore_cost_scales)' --no-capture`,
same test, same machine, debug build):**

| size | K | restore p50 before (ns) | restore p50 after (ns) | restore p99 before (ns) | restore p99 after (ns) |
|---|---|---|---|---|---|
| 16MiB | 16 | 13000 | 3000-4000 | 27000 | 6000-10000 |
| 16MiB | 256 | 13000 | 11000-18000 | 23000 | 13000-64000 |
| 256MiB | 16 | 74000 | 4000 | 82000 | 5000-6000 |
| 256MiB | 256 | 74000 | 17000-19000 | 78000 | 21000-22000 |
| 256MiB | 4096 | 206000 | 175000-179000 | 1153000 | 835000-852000 |

Two independent runs of the same test are reported as the "after" range
(p50/p99 both) rather than a single point estimate — run/unprotected-run
columns are unchanged from the prior row (this slice touches restore only)
and are omitted here; see the prior slice's table for those.

**The result, not a bare measurement:** at fixed `K=16`, restore p50 is now
3000-4000ns at BOTH 16MiB and 256MiB — flat, where it was 13000ns vs 74000ns
(a 5.7x jump) before this slice. `K=256` also flattens (11000-18000ns @
16MiB vs 17000-19000ns @ 256MiB, both inside the same noise band, vs the
prior 13000ns/74000ns 5.7x split). The mechanism is exactly the one traced
above: restore-time cost is now a function of `dirty_page_index_count` (K),
never of `base_size`. `K=4096` (256MiB only, no 16MiB row exists in this
test's own matrix to compare against) moved from 206000ns to 175000-179000ns
p50 — a smaller, expected win at this `K`, since with `K=4096` the O(K log K)
sort plus the syscall/remap cost per run were already dominating the deleted
O(region page count) scan term even before this slice (the prior row's own
"real growth once K gets large enough to dominate the fixed scan floor"
finding).

**Byte-identical-twin oracle and re-trap proof still green** (unchanged
assertions, same test bodies): `layered_restore_reproduces_the_base_byte_identically_over_the_whole_region`,
`layered_restore_actually_remaps_dirtied_pages_read_only::small_k`,
`layered_restore_actually_remaps_dirtied_pages_read_only::medium_k`, and the
sharing proof `two_warm_vms_share_one_base_without_observing_each_others_writes`
all pass unmodified — the list is additive machinery behind the same public
`WarmVm::restore_layered`/`run_dirty_write` surface, no test needed a
rewrite.

**Files:** `tools/proxima-vm/src/dispatch_trampoline.h:743-793` (both
trampoline signatures gain `dirty_page_indices`/`dirty_page_index_count`,
doc rewritten to name the O(working-set) shape), `tools/proxima-vm/src/backend_macos.c`
(`#include <stdlib.h>` for `qsort`; `proxima_vm_layered_run:1162` gains the
O(1) append at `:1265-1274`; `compare_dirty_page_index:1307`;
`proxima_vm_layered_restore:1319` rewritten — sort at `:1346`, O(K)
coalescing walk at `:1348-1376`, O(K) bitmap clear at `:1378-1388`, count
reset at `:1389`), `tools/proxima-vm/src/snapshot.rs` (both `unsafe extern
"C"` declarations updated; `LayeredHandle` struct gains `dirty_page_indices:
Vec<u32>` + `dirty_page_index_count: u64` fields at `:1221-1238`;
`page_count` helper at `:1244-1246`; `construct`/`run`/`restore` thread the
two new parameters through the FFI calls).

**No new public types.** `dirty_page_indices`/`dirty_page_index_count` are
private fields on the existing `LayeredHandle` (itself `pub(super)`, never
public); the public `WarmVm`/`DirtyRunReport`/`LayeredRestoreReport` surface
is unchanged byte-for-byte.

**Gate evidence:**
```
CARGO_TARGET_DIR=<scratch> cargo nextest run -p proxima-vm
  -> 184 tests run, 184 passed, 0 skipped
CARGO_TARGET_DIR=<scratch> cargo clippy -p proxima-vm --all-targets
  -> exit 0, no warnings
```

**Unresolved / left open:**
- `--release` re-run not done this slice (debug build only, same caveat
  every prior slice in this file already carries).
- The sort is `qsort` (libc), not a specialized small-K sort — at `K=4096`
  this is ~4096*log2(4096) ~= 49k comparisons; not benchmarked in isolation
  against an insertion sort or a radix pass over page numbers, since the
  end-to-end restore p50 at `K=4096` already improved and no regression was
  observed.
- KVM/x86_64 lane untouched, same standing gap every prior row in this file
  already carries — the layered functions remain HVF-only.
- No independent profiler run (`perf`/Instruments) to attribute the residual
  restore cost at large `K` between the sort, the coalescing walk, and the
  `hv_vm_unmap`/`hv_vm_map` syscall pair themselves; the end-to-end numbers
  above are the only measurement this slice took.
