# proxima-vm roadmap — scratch guest to firecracker-style lambda, and to a simulated machine

Two tracks off one substrate.

- **Track A — the boundary.** A proxima `Pipe` whose `call` forks a pre-booted
  snapshot, runs a tenant payload, and drops it. The µsec guarantee is on the
  **boundary**, not the tenant payload: however much code the tenant adds lives
  inside the snapshot and does not move the boundary number.
- **Track B — the machine.** Device models and a CPU as **sans-IO state
  machines**, so the same device codec is driven by a hypervisor exit
  (KVM/HVF), by an interpreter with no hardware virtualization at all (the
  QEMU/TCG shape), or by real hardware (VFIO/uio — already proven in the nvme
  leaf). One codec, three drivers.

Rent the machine where renting is honest; own the three things that actually
differentiate: guest memory + snapshot/fork, the device plane as pipes, and the
capability protocol. Track B is what makes "own the device plane" mean
something stronger than "we wrote a virtio shim."

## The tier ladder — binding, and the point of the whole exercise

Principle 3 (`slot-0/AGENTS.md:415`) and principle 11's tier-3 clause
(`:858`) bind here harder than anywhere else in the workspace, because a
device model that allocates cannot run in the place a device model most wants
to run.

| tier | means | what lands here |
| --- | --- | --- |
| 3 — bare `no_std`, **no alloc** | no `Box`/`Vec`/`Arc`/`String`; fixed-cap via `heapless`/`arrayvec`/const generics; caller owns every buffer | **every device codec, every descriptor parser, the CPU decode/execute step, the page-table walker** |
| 1 — `no_std` + `alloc` | `alloc::*` + `bytes::Bytes` | only where state genuinely scales unbounded with guest behavior, with a discipline-log row saying which state and why |
| 2 — `std` | fds, mmap, ioctl, the hypervisor leaves | the **drivers only**: KVM ioctls, HVF calls, uio/VFIO, the host reactor |

The rule that makes this mechanical: **the sans-IO half of any milestone below
lands at tier 3 or it does not land.** The std half is the leaf that feeds it
bytes. `prime`, `proxima-net`, `proxima-storage`, and `proxima-protocols`
already build for `aarch64-unknown-none` and `thumbv7em-none-eabihf` — the
device plane joins that gate, it does not get an exemption.

Ruled out on the alloc-free path (principle 3 verbatim list applies): `std::io::Error`,
`RawFd`, `std::process::Command`, `OsStr`, `std::cell::RefCell`, `std::sync::Arc`,
`std::sync::Mutex`/`parking_lot::Mutex` (principle 21: lock-free first, else
`proxima_lock::Mutex`, never a spin), `std::time::Instant`, `Box::pin(async move {})`.

## Ground truth, measured 2026-08-12

| fact | where |
| --- | --- |
| guest boots and emits bytes on Apple silicon | `cargo nextest run -p proxima-vm` → 9 tests, boot emits 22 bytes |
| HVF lane: synthesized aarch64, `hvc #0` per byte | `src/backend_macos.c:69-74` |
| KVM lane: 16-bit x86, `KVM_EXIT_IO` per byte | `src/backend_linux.c:141-155` |
| the virtqueue seam already exists | `proxima-storage/src/nvme/backend.rs:12` — `QueueBackend`, 4 methods |
| sans-IO submission/completion rings, `Pipe + SendPipe` | `proxima-storage/src/nvme/engine.rs:6` |
| DMA/BAR/pagemap pinning proven vs real NVMe + QEMU 1.4 | `proxima-storage/src/nvme/uio.rs:1-17` |
| shared-memory producer/consumer rings | `proxima-net/src/xdp/ring.rs`, `umem.rs` |
| sans-IO TCP for a virtio-net backend | `proxima-net/src/tcp_stack.rs` (686 lines) |
| bare-metal tier is real and gated | `prime`, `proxima-net`, `proxima-storage`, `proxima-protocols`, `examples/no-std` |
| `virtio` / `vhost-user` / `memfd` / `userfaultfd` / `page_table` | **0 hits in the tree** |
| `target_os = "windows"` | 0 hits; reactor is kqueue/epoll/io_uring, no IOCP |
| KVM host (versailles, 192.168.1.78) | offline as of 2026-08-12 |

## The bottleneck, stated before any work is done

Stage-2 (EPT/NPT) page-table population — not mmap, not vCPU setup, not
devices. A fresh VM's stage-2 tables start empty; every guest page touched
faults to the host at ~1-2µs.

| guest size | page size | faults | wall |
| --- | --- | --- | --- |
| 4 GB | 4 KB | 1,048,576 | 1-2 s |
| 4 GB | 2 MB | 2,048 | 2-4 ms |
| 4 GB | 1 GB | 4 | ~µs |

The lever fights itself: 1 GB pages make CoW useless (one write copies 1 GB).
Resolution to **test, not assume**: image read-only shared @1 GB, writable
delta @4 K, so fault count tracks the writable working set, not image size.
1 GB pages are boot-reserved and non-reclaimable, so µsec startup trades
directly against density. No hypervisor exposes "clone stage-2 tables" — that
absence is why every snapshot-restore product on the market is ms-class.

Everything below M3 is unmeasured. M3 exists to stop that.

## How a milestone lands

Each milestone is a `/disciplined-component` component, not a task. That means,
before the row seals:

- the 13-point gate, every cell filled — blank is not done
- tier declared and gated (tier 3 for sans-IO halves; `--no-default-features`
  build proves it, not a comment)
- principle 13 skill tagged and run **first**: `/algorithm-development` for the
  virtio ring rules, the page-table walker, and the instruction decoder (each
  has a known answer to reproduce — a spec, and QEMU as the incumbent);
  `/research-rigor` for contested shapes; `/security-review` for anything on
  the isolation boundary
- principle 14: **QEMU is the incumbent.** A divergence from QEMU's behavior is
  our bug until proven otherwise by the six steps. Byte-for-byte parity on
  device register semantics and descriptor handling is the bar.
- principle 9: real data. Descriptor bytes come from a real capture against
  QEMU or real hardware, never hand-rolled literals.
- discipline log in the vault at `20 - Proxima/Discipline/proxima-vm/discipline.md`

- **principle 4, both roles, on every milestone that adds a knob.** Config-as-mirror:
  one built thing has a 1:1 serializable config that round-trips with its builder.
  Config-as-composition: the config *language* composes compiled primitives into
  instances the binary never enumerated, so a new device, a new guest, a new
  capability route is **config, not a recompile**. The compiled set grows only for
  a genuinely new primitive. Every knob reaches the user through both the
  conflaguration surface and the fluent builder, interchangeably, with a
  round-trip parity fixture proving the two forms identical.
- **registries are the extension mechanism, not `match` arms.** `PipeFactory` +
  `RegistryEntry` for upstreams and listeners; `SchemaRegistry`
  (`proxima-config/src/schema/mod.rs:573`) for validation;
  `ConfigFormatRegistry` (`proxima-config/src/lib.rs:124`) for formats. A device
  model, a guest, or a capability handler is a registered factory keyed by its
  `type` discriminator. Adding one must never require editing a central `match`.

- **the gate is the test suite, not a shell script.** A milestone's exit criterion
  is asserted by `#[proxima::test]` cases with `#[case::...]` parameterization, run
  by `cargo nextest run`. Setup a shell script would do — codesigning, copying an
  image, launching a helper — happens inside the harness, in Rust, in a tempdir.
  Shell captures lie: `$(...)` strips trailing newlines, an exit code cannot
  separate zero work from all the work, and neither failure is visible in a green
  log. Compile-time properties (bare-metal target builds, clippy, rustdoc) belong
  in CI matrix entries, not in a script that pretends to be a test.

- **policy is the caller's, and the shape keeps it that way.** Proxima owns the
  machine — memory, devices, the exit loop — and never bakes in an authority
  model. An external policy layer must be able to sit on the boundary without
  proxima knowing it exists. Three shape rules, binding, and free to hold:
  1. **admission is a pipe, not logic.** Every capability decision point (which
     verb, which handle, which region) is a `SendPipe` seam the caller supplies.
     Proxima ships a default that says yes; a policy engine replaces it. Never a
     `Box<dyn Fn>` hook, never a hardcoded policy branch — a pipe slot.
  2. **requests carry the fields a total decision procedure needs.** A policy
     that must always terminate can only read fields that exist on the request:
     verb, handle, addressing, payload-presence, sizes — typed, not stringly.
     If deciding requires consulting state outside the request, the request is
     missing a field; fix the request. (This is also why P0's fd-keying matters
     beyond hygiene: `path().starts_with(prefix)` is open-ended string matching;
     a typed handle is a field a bounded rule can read.)
  3. **protection is data, not code.** Permission levels, region bounds, and
     privilege live as plain values on requests and page-table entries — the
     M11 walker is already the right shape: a pure
     `(root, vaddr, perms) -> Result<paddr, Fault>` total function. A level
     comparison a policy can express is a level stored where it can read it.

Exit criteria below are **numbers or byte sequences**, never checkboxes. A gate
that cannot report the N it processed is not a gate.

---

# Track A — the boundary

### M0 — the loop · DONE 2026-08-12

Build, sign, boot, assert the guest's output bytes. `cargo run` alone cannot do
this: Hypervisor.framework returns `HV_DENIED` (`0xfae94007`) to an unsigned
binary, and codesigning is a post-link step cargo has no hook for.

**The gate is the test suite. There is no gate script.** `tests/boot.rs` copies
the built guest into a per-case tempdir, applies the entitlement itself, runs it
and asserts the emitted bytes. Shell gates are not used for this component: the
first thing the test suite caught was that the shell version's `$(...)` capture
had been stripping the guest's trailing newline, so it compared 21 bytes against
a guest that emits 22 — a passing gate asserting the wrong number.

- exit: `cargo nextest run -p proxima-vm` → 9 tests, 9 passed
- cases: default greeting, caller-supplied message, single byte, **empty guest
  emits zero bytes** (the degenerate control), and unsigned-guest denial naming
  `hv_vm_create` in its error
- also fixed: two rustdoc links, one of which named a parity test that did not
  exist — `wire_format_round_trips_for_parity` now pins the postcard bytes

### M1 — the guest ABI and ELF loader: run an arbitrary proxima binary · ~3-5 days

**This is the first shippable thing.** The tree can already build proxima for
`aarch64-unknown-none` with no std, no OS, and no allocator — which is precisely
what an HVF guest executes. What is missing is not a kernel; it is a loader and
a calling convention.

- a guest crate at `tools/proxima-vm/guests/lambda/`: `#![no_std] #![no_main]`,
  a `_start` at a fixed load address, a linker script, built for
  `aarch64-unknown-none` (KVM lane: `x86_64-unknown-none`)
- an ELF loader in `proxima-vm`: parse PT_LOAD segments, map them at their
  `p_vaddr` into guest memory with per-segment permissions (W^X honored, not
  RWX-everything as the scratch guest does today at `backend_macos.c:87`),
  set PC to `e_entry`, set SP below the image
- **the guest ABI is the capability protocol, not a new invention**: `hvc #0`
  (aarch64) / `out dx, al` (x86) with `x0` = verb, `x1` = shared-page offset,
  `x2` = length. The payload in the shared page is a postcard-encoded
  `ChildRequest`; the reply is a `ChildResponse`. Those bytes are already
  pinned by `wire_format_round_trips_for_parity`, and the libc-shim already
  speaks them — one contract, two mechanisms, per
  `proxima.decision.libc_shim_vm_parity`.
- the guest-side hypercall wrapper is tier 3: no alloc, `heapless` framing into
  the shared page, caller owns the buffer
- exit: `proxima-vm run <path-to-elf>` loads a guest the loader has never seen,
  the guest issues ≥2 distinct `ChildRequest` verbs, and the host's responses
  change the bytes the guest emits — proving the channel is bidirectional and
  not a replay. Assert the request count and the response bytes, not the exit
  status. Sad path asserted too: a malformed ELF and an out-of-range `p_vaddr`
  are rejected with a named error, not a fault.

### M2 — the lambda is a registry entry, not a binary · ~3-5 days

**Conflaguration-based is required, and it is the headline capability, not a
surface.** A new function is a TOML block. Not a recompile, not a new crate, not
a plugin build — a block. That is principle 4's config-as-composition
(`slot-0/AGENTS.md:510`), and for this component it is the load-bearing claim.

The seam already exists and is exactly the right shape:
`src/upstreams/process.rs:676` — `impl PipeFactory for ProcessPipeFactory` with
`name() -> "process"` and `build(spec: &Value) -> PipeHandle`, resolved through
`RegistryEntry` (`src/settings/mod.rs:53`: discriminator `type` plus a flattened
untyped spec, late-deserialized by the registered factory) off
`ProximaSettings.upstreams` at `src/app.rs:1054`. `ProcessConfig`
(`src/upstreams/process.rs:551`) is the canonical shape to mirror: one struct
deriving `Builder + Deserialize + Serialize + Settings`, `#[settings(prefix =
"PROXIMA_PROCESS")]`, per-field `#[setting]`/`#[serde]`/`#[builder]` defaults
that agree.

So M2 is not "write a binary." It is:

- `VmConfig` / `LambdaConfig` deriving `Builder + Deserialize + Serialize +
  Settings + Validate`, `#[settings(prefix = "PROXIMA_VM")]` — the guest image
  path, memory size, page size, the capability routes the guest may reach, the
  device set, and the lane. Principle 12: every numeric cap traces to the sizing
  TOML, none hard-coded in source.
- `VmPipeFactory` with `name() -> "vm"`, registered exactly the way
  `ProcessPipeFactory` is. From that moment `[upstream.my-fn] type = "vm"`
  in a proxima config spins up a guest, and it composes with routing,
  middleware, `.any()`, recording, and telemetry because it is a `PipeHandle`
  like everything else.
- the fluent half, first-class and not the lesser variant: `Vm::builder()`
  mutable → `.build()` immutable, `builder_from(&VmConfig)` to start from a
  loaded config and keep chaining, getters for every field that participated in
  construction.
- `proxima-lambda run <guest.elf>` exists as a one-shot convenience for the
  dev loop, and it is a thin caller of the same config — never a second path.

Exit, and gate point 12's **config-only-variant proof** is the load-bearing part:

1. a fixture that registers a novel function purely via TOML — **zero new Rust** —
   and drives it end to end, returning the guest's bytes
2. the both-ways parity fixture: the same `Vm` built via the conflaguration env
   loader, via `from_path`, and via `.builder().build()`, asserted to identical
   internal state
3. p50/p99 of the boot→run→teardown cycle, recorded as the **baseline every
   later milestone is measured against**. Fresh VM per request, no snapshot, no
   fork. This number is expected to be bad. Record it anyway — it is the
   denominator for the µsec claim, and without it M8's win has nothing to be a
   win over.

A per-guest Rust `impl` would be a recompile and does **not** satisfy this.

### M2b — tensor verbs: the LLM lambda · ~1-2 weeks, after P0

The guest never gets the GPU; it gets a **capability**. The tenant's ELF streams
tensor work — proxima-tensor's `&[Expr]` form — out over the M1 hypercall
channel, and the host executes it: prime interpreting, or omega emitting MSL to
Metal. That is the `Nest` seam doing exactly what it was built for, and on Apple
hardware it is the only path, not a workaround: Metal inside an `hv_` guest is
not a thing. The guest stays a thin tier-3 binary; weights, KV cache, and the
GPU driver stay host-side.

The fork model and the LLM serving model are the same shape, which is why this
belongs on this roadmap and not somewhere else:

| LLM serving piece | VM piece it maps onto |
| --- | --- |
| weights (immutable, GB-scale) | the read-only image shared @1 GB pages — dedups across tenants, ~4 faults |
| KV cache (per-session, growing) | the writable delta @4 K — per-fork dirty pages |
| new chat session | fork from snapshot (M8) |
| tokenizer / sampling / agent logic | the tenant's guest ELF |

KV cache is the one unsolved design question in the tensor product frame; this
frame gives it a concrete home — the writable delta — rather than leaving it
abstract.

- **hard prerequisite: P0.** Tensor verbs added onto path-keyed `ChildRequest`
  addressing would inherit the defect fifteen-fold. fd-keying lands first.
- the verb set follows the tensor design of record: submit an expr stream,
  outputs-as-request; batch derives from the stream. No new node kinds — the
  gap derivations showed quant/radix/topk all express in the existing algebra.
- host side is a registered capability handler (registry, not a `match` arm),
  config-selected per guest: which device, which memory budget, which model
  image. A new model is a TOML block naming an image path — config, not Rust.
- guest-side wrapper is tier 3 like the rest of the hypercall surface.
- exit: a guest ELF drives one full decode loop (prompt in, tokens out) through
  the boundary; report tokens/sec through the hypercall channel vs omega driven
  directly on the same host — the boundary's cost as a number against ~10-100 ms
  of GPU work per step. `design-favors: incumbent` arm: llama.cpp on the same
  model and quant, per the product frame's rule that we are not duplicating it —
  the claim is the isolation boundary's price, not a kernel-speed win.
- the CPython dirty-on-read caveat does not apply here: a proxima guest has no
  refcount writes on read.

### M3 — the fault-count instrument · ~1-2 days

The product rests on one number nobody has measured here. Build the instrument
before building anything that claims to improve it.

- configurable guest memory size and page size on both lanes
- record per boot: wall `create → first guest instruction`, wall to touch every
  mapped page, and the **page indices** faulted (not just counts) —
  `/proc/PID/clear_refs` + `pagemap` soft-dirty on KVM; HVF has no equivalent,
  so the mac lane reports wall time only and says so in the row
- exit: a table of (guest size, page size, fault count, µs) across 4 K / 2 M /
  1 G, plus the raw index stream on disk
- **degenerate control, mandatory**: a run that touches zero pages must report a
  near-zero fault count. If it does not, the instrument measures something other
  than what it names and every number after it is void.

### M4 — guest memory as a named object · ~2-3 days

`mmap(MAP_ANON)` cannot be shared, snapshotted, or forked. Replace with a named
region: `memfd_create` + `MAP_SHARED` on KVM, `vm_allocate` + a shared mach
memory entry on HVF. The region descriptor is tier-3; the syscall leaf is tier-2.

- exit: two VMs map the same region and observe each other's writes; a
  `MAP_PRIVATE` child on the KVM lane observes copy-on-write — parent write
  invisible after the split, proven by reading bytes back, not by inference

### M5 — a real kernel boots · ~3-5 days

The scratch guest is a byte emitter, not an OS. PVH or bare `vmlinux` direct
boot on the KVM lane, serial console only, no BIOS, no PCI.

- exit: kernel console output arrives through our byte channel; the assertion is
  on kernel-emitted bytes and the count is asserted nonzero
- M5b: the same on HVF, which needs GIC + PSCI + a DTB — that is Track B work
  arriving early, and it is where the two tracks first touch

### M6 — virtio as a sans-IO codec, next to nvme · ~1-2 weeks

The device plane is the second thing we own. The hard part is already done:
`QueueBackend` is the virtqueue seam, and the DMA/doorbell/ring code is proven
against real hardware. Missing is a sans-IO virtio codec beside `nvme` plus a
transport.

- tier 3, no exceptions: descriptor chain walking, available/used ring
  arithmetic, and feature negotiation are fixed-shape and allocate nothing
- order: virtio-console (replaces the M5 byte channel), then virtio-net over
  `proxima-net/src/tcp_stack.rs`, then virtio-blk over `proxima-storage`
- worked-example tests carry **real captured descriptor bytes** from a QEMU
  session, walked bit-exact (principle 9 + `/algorithm-development`)
- **P0 below lands first** — do not grow the contract on a defect
- exit: a guest kernel drives virtio-console; the ring codec's worked-example
  tests pass with no VM in the loop, and the crate builds
  `--no-default-features` for `aarch64-unknown-none`

### M7 — snapshot · ~1 week

Serialize vCPU registers, device state, and memory; restore into a fresh VM.
Device state serializes because the device is already a state machine — that is
the payoff of M6's shape, and it is why "restore, not renegotiate" is available
to us and not to a shim.

- exit: restore wall time and fault count at each page size, measured with the
  M3 instrument. This is where "ms-class vs µs-class" gets evidence instead of a
  claim.

### M8 — fork · ~1-2 weeks

Image read-only shared @1 GB, writable delta @4 K, `userfaultfd` for demand
population. This is the µsec claim.

- exit: N forks from one snapshot; per-fork wall time distribution and the
  dirty-page **index** stream per fork. Report p50 and p99, and report the
  density cost (reserved 1 GB pages per idle tenant) in the same table. A
  latency number without the density number it bought is half a result.
- KVM lane only — HVF has no `userfaultfd` equivalent

### M9 — the lambda edge · ~3-5 days

A `Vm` pipe: `In = Request`, `Out = Response`. Request arrival forks a snapshot,
runs, drops.

- exit: end-to-end p50/p99 request→response with the cold fork **included** in
  every sample, against a warm-process baseline on the same host

---

# Track B — the simulated machine

QEMU is the incumbent for this entire track (principle 14). Every device's
register semantics, reset values, and error behavior are theirs until we prove a
divergence by the six steps.

### M10 — the device is a sans-IO state machine · ~1-2 weeks

Generalize what M6 does for virtio into the shape every other device uses: a
device is `(MMIO/PIO access, DMA view) -> (state transition, interrupt
assertion)`, with no I/O in the signature and no allocation.

- the driver is a separate leaf per source: KVM exit, HVF exit, interpreter
  step, real hardware via uio/VFIO. Same codec object under all four.
- first devices after virtio-console, in dependency order: **serial 16550A**
  (every kernel's first output), **RTC/CMOS**, **PIC/IOAPIC + local APIC**
  (interrupt delivery is what makes a kernel able to do anything), **PCI
  configuration space**, **fw_cfg** (how a kernel learns the machine's shape)
- exit: a Linux kernel reaches userspace init driving only our device models,
  with the register-access trace diffed against QEMU's for the same kernel and
  the divergence count reported as a number

### M11 — the page-table walker · ~1 week

**No single address space. Settled 2026-08-11** — slot-0 vault
`10 - Journals/12 - Notes/2026/Week 33/2026-08-11/proxima-execution-boundary-direction.md:257-262`,
correction #4. Verbatim: *"Single-address-space design lets you skip most of the
MMU work." Wrong, and it broke an earlier claim: CoW fork is a page-table
mechanism, so there is no µsec fork without page tables. SASOS isolation
(Singularity, Theseus) only holds for code the system compiled and verified — it
is structurally inapplicable to BYO containers, any `unsafe` in the TCB voids it
globally, and it puts rustc in the TCB. Page tables are critical path, not for
isolation but for CoW fork, W^X, guard pages, `mmap`/`brk`, and demand paging.*

So `page_table|TTBR|VBAR_EL1` = 0 hits is a gap to close, not a design choice to
defend. Any later proposal that reaches for a shared address space to avoid MMU
work is re-litigating a closed decision.

- tier 3, pure function: `(root, virtual address, permissions) -> Result<physical
  address, Fault>`, for aarch64 stage-1 and stage-2, and x86-64 4-level and
  5-level
- exit: worked example per format walked bit-exact against the architecture
  manual's own example, plus a differential test against QEMU's
  `info tlb` on the same page tables

### M12 — the interpreted CPU · ~4-8 weeks, the long pole

The QEMU/TCG axis: execute guest instructions with no hardware virtualization.
This is what makes the tree able to sim hardware on any host, cross-arch, and
without KVM or HVF at all.

- decode is tier 3 and pure: `&[u8] -> Option<(Instruction, usize)>`, a
  discriminated enum, borrowed views, no allocation — the same shape as every
  proxima wire codec
- execute is a state machine over an architectural-state struct plus the M9
  walker plus the M10 device bus
- start with the subset the M5 kernel actually executes, measured by tracing the
  KVM lane — not the full ISA up front
- exit: the M5 kernel boots to the same console bytes under the interpreter as
  under KVM, byte-for-byte; plus instructions/second reported against QEMU-TCG on
  the same kernel, on their turf (`design-favors: incumbent`)
- honest expectation: an interpreter loses to a JIT by 10-100x. The result to
  report is the number, not a verdict, and the interesting claim is the
  tier-3/no-alloc property that TCG does not have, not raw speed.

### M13 — hardware-in-the-loop parity · ~1 week

The nvme leaf already talks to real hardware through uio with zero C linked.
Close the loop: run one device codec against (a) the interpreter, (b) KVM, and
(c) the real device, and diff the register-access traces.

- exit: three traces, one diff, divergence count per pair reported as a number

---

## P0 — fix before M6 grows the contract

`proxima-protocols/src/process/protocol.rs:113`: `ChildRequest` has 5 variants,
all filesystem, all **path-keyed**, while `ChildResponse::Open` returns a
`handle: i32` the request side cannot name.
`proxima-process/src/operators.rs:52-56` is the entire routing layer —
`request.path().starts_with(prefix)`. WASI and Linux are both fd-keyed after
open. Adding the ~15 verbs a VM needs (sock, clock, random, poll, seek, readdir)
on top of path-keying makes all 15 inherit the defect. Routing is localized to
one function; the fix is cheap now and compounds later. Principle 15: this is
greenfield, so the legitimate-deferral surface is zero.

Second item, same file set: `tools/proxima-vm/src/dispatch.rs:52` defines
`VmDispatchHandler` — one method, `ChildRequest -> Future<ChildResponse>`. That
is `SendPipe` with `In = ChildRequest, Out = ChildResponse`, which its own doc
comment says out loud. Answer the binary question by writing the call site both
ways; if the two lines are identical, the trait is a relocation, not a type.

## Hard external caps

- macOS **guests** run only under Virtualization.framework on Apple hardware,
  2-VM limit; `hv_` cannot boot macOS. A macOS guest means Apple's device model:
  no owned devices, no µsec fork. Separate product, weaker promises. Not an
  engineering choice. Track B's interpreter is the only path that changes this,
  and it changes it slowly.
- Windows host support needs an IOCP reactor leaf, which does not exist yet.
- CPython dirties pages on **read** (`Py_INCREF` writes the object header). No
  LSM, DAX, or CoW scheme reaches that; the fix is upstream — immortal objects
  (PEP 683, 3.12+), side-table refcounts, `gc.freeze()`. Any Python-tenant fork
  number is meaningless until this is controlled for.

## Hand-rolling a spec parser — the three binding conditions

Hand-rolling is legitimate when the format is a specification (ELF gABI, virtio,
page-table formats) and nothing in the tree expresses it. It is legitimate ONLY
under all three of these, no exceptions:

1. **sans-IO enum FSM** — the house parser shape per principle 11: explicit
   discriminated-enum state machine, `parse(&[u8])` -> borrowed view + consumed
   length, zero alloc on the hot path, tier 3, no I/O in the signature. Mirror
   the protocol crates' codec shape and reuse their byte-scan/bounds helpers.
2. **disciplined component** — the full 13-point gate: feature flag, tests,
   clippy, micro-bench, compare-bench vs a named incumbent, opt sweep, and the
   discipline-log row with cells filled by numbers, not checkmarks.
3. **meet or beat the incumbent on their home turf** — for the ELF loader the
   named incumbents are the `object`/`goblin` crates (parse throughput on real
   binaries) with correctness parity vs `readelf -l` on the same file. A
   `design-favors: incumbent` bench arm must exist; a loss is documented
   honestly and claims no win. Principle 14 governs any divergence: our parser
   disagreeing with readelf/objdump is our bug until the six steps prove
   otherwise.

Everything else composes what exists first: `proxima-process/src/ipc.rs`
(`read_frame`/`write_frame`/`run_dispatch_loop`) already frames and dispatches
this protocol; `proxima-primitives/src/pipe/` holds ~58 primitives and the VM
boundary is their forms — the exit loop is a source of `ChildRequest`s, dispatch
is routing over `SendPipe`, response write-back is a sink, admission is a
transform in front. A hand-rolled loop that a source + routing + sink
composition expresses does not land.

## State of work — closed down 2026-08-23, primed for resume

**Where things stand:**

- **M0 DONE and verified** (2026-08-16): the proxima-vm nextest suite was 9/9
  green, clippy and rustdoc clean, boot cases asserting exact byte counts
  including the zero-byte degenerate control and the unsigned-guest denial.
- **M1 code is checkpointed but NOT sealed.** An orchestrated build produced
  the M1 surface — `src/{abi,elf,loader}.rs`, `guests/lambda/` (no_std guest
  crate + link.ld + hypercall wrappers), the C trampoline headers, the
  `proxima-vm run` CLI, `tests/{boot,dispatch_hypercall}.rs` — landed in
  `chore: checkpoint working tree`. The run was stopped mid-flight twice for
  reuse violations (a bespoke postcard decode/dispatch loop where
  `run_dispatch_loop` exists; a named test dispatcher where a closure serves)
  and the checkpoint has NOT been re-verified since. Treat every claim in it
  as unproven until the gate runs.
- The `VmDispatchHandler` trait was deleted as a relocation — a VM dispatcher
  is `SendPipe<In = ChildRequest, Out = ChildResponse>` directly.

**Resume checklist, in order:**

1. Run the proxima-vm nextest suite (`--no-fail-fast`) — assert the tests-run
   count is nonzero and note exactly which cases exist. This is the ground
   truth for everything below.
2. **Reuse audit of the checkpoint** against the inventory above: for each of
   `abi.rs` / `loader.rs` / `dispatch.rs` additions, either name the existing
   primitive that replaces it (and delete it) or keep what has no existing
   expression, with the grep that proves it.
3. **`elf.rs` (963 lines) is kept but reshaped** under the three spec-parser
   conditions above: enum FSM, 13-point gate, object/goblin + readelf arms.
4. Close M1's exit criterion: `proxima-vm run <elf>` with a guest issuing >= 2
   distinct verbs, host responses provably changing guest output, counts
   asserted; sad paths (malformed ELF, out-of-range p_vaddr) named errors.
5. Then M2 (registry entry / conflaguration surface) per its section above.

**Known-good baselines to protect:** the pinned postcard wire bytes in
`dispatch.rs` (`wire_format_round_trips_for_parity`), the 22-byte hello boot,
and the per-case self-signing harness in `tests/boot.rs` (each case copies the
binary to a tempdir before codesigning — parallel nextest processes corrupt a
shared path).

## State update — 2026-08-26, resume checklist executed (uncommitted)

Every step above ran, in order, with asserted counts. Nothing here is a seal —
the owner reviews the diff and rules. What ran:

1. **Gate first**: the checkpoint did not compile — exit 101, 0 tests (missing
   dev-deps, unexported `elf`/`loader`, undefined `proxima_vm_map_guest_memory`
   in both C backends, three mid-refactor symbols). Repaired; suite reached
   42/42, then 43/43 after the trap loop landed.
2. **Reuse audit ran adversarially**; the claim was refuted three times and all
   three were fixed: postcard framing now composes
   `proxima_process::framing::{FrameDecoder,FrameEncoder}` (the earlier
   rationale had rebutted only `run_dispatch_loop` — a strawman); the flat RWX
   guest image was replaced by per-segment W^X mapping through
   `RawSegment` + a page-window merge (driven by an empirical
   `hv_vm_map` `HV_BAD_ARGUMENT` on byte-exact mapping; Linux KVM expresses
   only READONLY — no exec bit without guest paging); the C
   create/run/teardown skeleton is factored and shared by the scratch and
   dispatch loops in both backends. `FfiRecordingDispatcher` survived both
   binary questions: extern "C" cannot be generic, so a concrete pointee is a
   language constraint, not a relocation.
3. **`elf.rs` reshaped** to the sans-IO enum FSM
   (`Cursor{Header, ProgramHeaderTable, EntryPointCheck}`), reusing
   `proxima_protocols::nvme::raw` byte-read helpers; `elf-bench`-gated
   compare-bench + `BENCH_LOG.md` landed. Numbers: ours ~29ns vs `object`
   ~61ns vs `goblin` ~1.0µs on the M1 guest (neutral arm); the
   design-favors-incumbent arm (real 4.58MB dynamic ELF) is a documented
   feature gap — ET_EXEC-only, named-error rejection, **no win claimed on the
   incumbents' turf**. Condition 2's second-design tweak-loop and bench CI
   wiring remain open rows in `BENCH_LOG.md`.
4. **Exit criterion exercised through real VM exits** on the HVF lane: the
   lambda guest issues `Read` then `Close` as genuine `hvc #0` traps, host
   responses provably change guest-emitted bytes (`00 00` vs `03 03` from the
   identical binary), request counts asserted, both sad paths named errors.
   The KVM lane compiles under cross-toolchain but has never executed —
   `/dev/kvm` absent on this host; RIP auto-advance after `out` is an
   inherited, unverified assumption.
5. M2 is next per its section above.

Full gate at close: nextest 43/43, clippy `--all-targets` clean,
`aarch64-unknown-none` alloc-tier check clean, x86_64-linux cross build clean
(host needs `CC/AR/LINKER=x86_64-unknown-linux-gnu-gcc` scoped per command),
all `bench_elf` arms running. Worktree `../proxima-elf-reshape` holds the
reshape's original (now ported) work — removal is the owner's call.
