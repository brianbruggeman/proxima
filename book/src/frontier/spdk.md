# spdk — kernel-bypass storage

*(builds on: dpdk)*

Userspace NVMe: a sans-IO SQE/CQE codec + ring FSM, byte-addressable storage
without the block layer's syscall path. Proxima does not link SPDK directly;
the `proxima-storage` `spdk` feature exposes a C-ABI callback adapter for an
embedding process that owns an SPDK qpair and reactor. The same `QueuePair`
engine and phase-bit codec are then reused without moving SPDK's thread
affinity or completion policy into Rust.

```bash
cargo test -p proxima-storage --features spdk
```

The adapter is intentionally an integration seam, not a claim that SPDK is
installed on the host: a local C shim must populate `SpdkCallbacks` from the
SPDK qpair. The pure-Rust `nvme-uio` backend remains the directly runnable
kernel-bypass option when no SPDK installation is present.
