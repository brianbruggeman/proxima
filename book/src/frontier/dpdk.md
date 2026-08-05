# dpdk — kernel-bypass networking

*(builds on: runtime-select)*

Userspace NIC rx/tx rings, poll-mode — kernel-bypass networking as a `Pipe`.
The backend ships today as `proxima_net::dpdk`, one of the five platform
backends `proxima-net` carries as feature-gated modules alongside `prime`,
`tokio`, `wasm` and `xdp` (`proxima-net/src/lib.rs:35-48`). That is the shape
worth taking away: DPDK is not a different API you port your pipe to, it is a
feature you turn on underneath the pipe you already wrote.

Three flat, reference-tier examples drive it, each gated on `required-features
= ["dpdk"]`:

- `examples/dpdk_tcp_connect.rs`
- `examples/dpdk_tcp_echo.rs`
- `examples/dpdk_udp_echo.rs`

They are not included inline here because they cannot be built — let alone run
— without DPDK's own libraries and a bound NIC present on the host, so the
build is host-dependent in a way no other chapter in this book is. Read them in
the tree.

They are also flat files rather than the dir-per-example shape the rest of the
curriculum uses (`examples/<name>/main.rs` beside an `examples/<name>/README.md`
teaching the one lesson). When that rung lands, this chapter becomes a normal
two-include chapter like its siblings.
