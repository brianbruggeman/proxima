# proxima-net

Network primitives for proxima: UDP packet listeners, address helpers, and every
platform network backend as a feature-gated module.

Stream traits and `BindAddr`/`PeerInfo` live in `proxima-primitives::stream`; this
crate carries the UDP `PacketListener`, addressing helpers, and the sans-IO
`stack`/`tcp_stack` logic. `packet` is std-gated (it returns `std::io::Result`);
the sans-IO halves compile under `no_std` + `alloc`.

## Backends

Each backend is a former standalone crate, now a feature-gated module — turn one on
to get its listeners and upstreams:

- `prime` — prime-reactor `StreamUpstream`/`AcceptorFactory`, zero tokio
- `tokio` — tokio-backed listeners and upstreams
- `wasm` — browser/wasm datapath
- `dpdk` — DPDK poll-mode userspace networking
- `xdp` — AF_XDP zero-copy datapath

Part of the [proxima](..) workspace.
