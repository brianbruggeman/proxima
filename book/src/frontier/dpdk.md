# dpdk — kernel-bypass networking

*(builds on: runtime-select)*

Userspace NIC rx/tx rings, poll-mode — kernel-bypass networking as a
`Pipe`. Not yet landed as a dir-per-example rung (no `examples/dpdk/README.md`
teaching this one lesson end to end), so there is nothing here to include
yet without re-typing unverified prose.

The underlying mechanics already exist as reference-tier, flat (no-README)
examples, compiled and runnable today:

- `examples/dpdk_tcp_connect.rs`
- `examples/dpdk_tcp_echo.rs`
- `examples/dpdk_udp_echo.rs`

Per `examples/README.md`'s own status line, `dpdk` is listed among the rungs
**not yet built**. When the dir-per-example rung lands (`examples/dpdk/main.rs`
+ `examples/dpdk/README.md`), this chapter becomes a normal two-include
chapter like its siblings.
