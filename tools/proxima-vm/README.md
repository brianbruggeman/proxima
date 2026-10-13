# proxima-vm

Minimal VM-backed proxima `Pipe` proof surface for scratch guests. `ScratchVm` does not boot an operating system: it synthesizes a tiny architecture-native guest program whose only effect is emitting its configured bytes and halting. The platform leaf is KVM on Linux x86_64 and Hypervisor.framework on Apple silicon.

Per the workspace decision on libc-shim parity, proxima-vm and the `proxima-process` libc-shim share a dispatch contract: `proxima_protocols::process::ChildRequest` / `ChildResponse`.

## What it composes

- `proxima-primitives` (`Pipe`), `proxima-core`, `proxima-net`
- `proxima-process` and `proxima-protocols` (the shared dispatch contract)
- optional: `proxima-onnx`, `proxima-tensor`

## Features

`std` (default), `alloc`, and opt-in `kernel-boot`, `elf-bench`, `restore-probe-demo`, `proxima-onnx`, `proxima-tensor`; see `[features]` in `Cargo.toml`. Real-VM boot tests are gated by `kernel-boot`.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
