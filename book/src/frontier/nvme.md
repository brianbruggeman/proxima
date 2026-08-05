# nvme — kernel-bypass storage

*(builds on: dpdk)*

`dpdk` took the kernel off the network path. This does the same for storage,
and by the same method: talk to the device's queues directly from userspace,
and never make a syscall on the hot path.

**A note on the name.** You will see this rung called "spdk" in older notes.
SPDK is Intel's C storage-bypass framework, and proxima does not use it — there
is no SPDK dependency and no C linked at all. What proxima ships is its own
pure-Rust NVMe queue-pair engine, so `nvme` is what this chapter is called.

## the shape

An NVMe controller is driven through **queue pairs**: a submission queue you
write 64-byte commands into, a completion queue the device writes 16-byte
results into, and a pair of doorbell registers you ring to say "I have added
entries" / "I have consumed entries". Which entries are new is not tracked by a
count but by a **phase bit** that flips each time the device wraps the ring.

proxima splits that in half at the sans-IO line, exactly like every protocol in
the workspace:

- `proxima_protocols::nvme` — the codec and the ring FSM. Encode an SQE, decode
  a CQE, track head/tail cursors and the phase bit. No device, no mapping, no
  syscall. `no_std`, no allocator.
- `proxima_storage::nvme` — the engine that drives that codec over a pluggable
  `QueueBackend`, and exposes "submit a command, await its completion" as a
  **`Pipe`** (`proxima-storage/src/nvme/mod.rs:1-16`).

That last point is the one worth stopping on. A block device, at this layer, is
not a special kind of object with a `read`/`write` API of its own. It is a
`Pipe` — `QueuePair` implements the per-core `!Send` root form, because NVMe
queue pairs are core-affine and that is precisely the promise the root `Pipe`
makes, plus a `SendPipe` for when the handle has to cross a core. The same
`and_then` you chained two HTTP transforms with chains a storage submission.

The engine itself is tier-3: `no_std`, no allocation, touching only the codec,
the ring cursors (held in atomics so the pair is `Sync`), and the backend
trait. Acquiring a real device — mapping the controller BAR through VFIO or
uio, allocating hugepage queues, ringing real MMIO doorbells — is a separate
backend behind the `std` boundary, the same way `dpdk` sits over the `no_std`
inet codec.

## the runnable proof

`proxima-storage/examples/uio_rw.rs` brings up a real controller through the
UIO backend, creates an I/O queue pair, then submits an NVM Write and an NVM
Read through the `QueuePair` pipe — codec, engine, and cooperative poll, end to
end on hardware:

```rust
{{#include ../../../proxima-storage/examples/uio_rw.rs}}
```

```bash
cargo build -p proxima-storage --example uio_rw --features nvme-uio
# inside a Linux guest, as root, device bound to uio_pci_generic:
sudo ./uio_rw 0000:00:02.0
```

Unlike the rest of the curriculum this one cannot run on your laptop. It reads
`/sys/bus/pci` and `/proc/self/pagemap` and expects `uio_pci_generic`, so it is
Linux by construction and gated on the target as well as on the feature
(`proxima-storage/src/nvme/mod.rs:21-24`) — compiling it elsewhere would prove
nothing it could ever run. The codec and engine underneath it are ordinary unit
tests that run anywhere.

If you want a storage rung you *can* execute right now, [pmem](pmem.md) is the
one: unconditional, no feature flag, and its walkthrough runs on any host.
