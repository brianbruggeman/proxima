# proxima-storage

Kernel-bypass storage as pipes. One crate with feature-gated modules, each keeping its own tier discipline:

- `nvme` (feature `nvme`): tier-3 `no_std` + no-alloc queue-pair engine driving the sans-IO `proxima_protocols::nvme` codec
- `pmem`: tier-3 persistence and crash-consistency primitives (copy-on-write atomic root swap, cache-line persist barriers), always available, no PMDK linked
- `dax` (feature `dax`): std DAX/file-backed mmap facade over `pmem`; Linux-only for the pmem-native fast tier, with a portable `FileCell` floor

## What it composes

`proxima-protocols` (NVMe codec), `proxima-primitives`.

## Features

`nvme`, `nvme-uio`, `spdk`, `dax`, `std` (default); see `[features]` in `Cargo.toml`.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
