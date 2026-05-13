//! AF_XDP zerocopy probe, done the RIGHT way: attach the redirect program in
//! NATIVE/DRV mode first, then bind the xsk with `XDP_ZEROCOPY`. Reports the
//! empirical result of each step (OK, or the exact errno) so the discipline log
//! records the truth for this box rather than assuming.
//!
//! Usage: `xdp_zerocopy_probe <ifname>`  (root; veth-only).

#[cfg(all(feature = "xdp", target_os = "linux"))]
fn main() {
    use proxima_net::xdp::bpf::{XDP_FLAGS_DRV_MODE, XdpProgram};
    use proxima_net::xdp::uapi;
    use proxima_net::xdp::{RingSizes, UmemConfig, XskSocket, sys};

    let ifname = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "veth0".to_string());
    let ifindex = match sys::if_nametoindex(&ifname) {
        Ok(index) => index,
        Err(error) => {
            eprintln!("if_nametoindex({ifname}): {error}");
            return;
        }
    };

    // 1. load the redirect program and attach it in native/DRV mode.
    let mut program = match XdpProgram::load(1) {
        Ok(program) => program,
        Err(error) => {
            println!("{ifname}: program load FAILED: {error}");
            return;
        }
    };
    match program.attach_with_flags(ifindex, XDP_FLAGS_DRV_MODE) {
        Ok(()) => println!("{ifname}: native/DRV XDP attach OK"),
        Err(error) => println!("{ifname}: native/DRV XDP attach FAILED: {error}"),
    }

    // 2. bind the AF_XDP socket in zerocopy mode (no XDP_COPY).
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
    match XskSocket::bind(&ifname, 0, umem_cfg, ring_sizes, uapi::XDP_ZEROCOPY) {
        Ok(socket) => println!("{ifname}: XDP_ZEROCOPY bind OK (fd={})", socket.fd()),
        Err(error) => println!("{ifname}: XDP_ZEROCOPY bind FAILED: {error}"),
    }
    // program detaches on drop.
}

#[cfg(not(all(feature = "xdp", target_os = "linux")))]
fn main() {
    eprintln!("this example requires --features xdp on linux");
}
