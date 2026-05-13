//! Probe which AF_XDP bind mode a NIC supports: `copy`, `zerocopy`, or the
//! kernel's default choice. On veth (copy-only) the zerocopy bind is expected
//! to fail — this records that honestly rather than faking a zerocopy number.
//!
//! Usage: `xdp_mode_probe <ifname> <copy|zerocopy|default>`  (root; veth-only).

#[cfg(all(feature = "xdp", target_os = "linux"))]
fn main() {
    use proxima_net::xdp::uapi;
    use proxima_net::xdp::{RingSizes, UmemConfig, XskSocket};

    let mut args = std::env::args().skip(1);
    let ifname = args.next().unwrap_or_else(|| "veth0".to_string());
    let mode = args.next().unwrap_or_else(|| "copy".to_string());
    let flags = match mode.as_str() {
        "copy" => uapi::XDP_COPY,
        "zerocopy" => uapi::XDP_ZEROCOPY,
        _ => 0,
    };

    let umem_cfg = UmemConfig {
        frame_count: proxima_net::xdp::sized::UMEM_FRAME_COUNT,
        frame_size: proxima_net::xdp::sized::UMEM_FRAME_SIZE,
    };
    let ring_sizes = RingSizes {
        fill: proxima_net::xdp::sized::RINGS_FILL,
        completion: proxima_net::xdp::sized::RINGS_COMPLETION,
        rx: proxima_net::xdp::sized::RINGS_RX,
        tx: proxima_net::xdp::sized::RINGS_TX,
    };
    match XskSocket::bind(&ifname, 0, umem_cfg, ring_sizes, flags) {
        Ok(socket) => println!("{ifname} {mode}: bind OK (fd={})", socket.fd()),
        Err(error) => println!("{ifname} {mode}: bind FAILED: {error}"),
    }
}

#[cfg(not(all(feature = "xdp", target_os = "linux")))]
fn main() {
    eprintln!("this example requires --features xdp on linux");
}
