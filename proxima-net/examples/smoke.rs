//! Manual bring-up smoke check for the AF_XDP datapath: exercises the Umem
//! frame allocator in isolation, then attempts a full `XskSocket::bind` on a
//! real interface so failures show the exact kernel errno rather than a type
//! error. Run explicitly on a linux host with the `xdp` feature; not part of
//! the automated test suite because it requires a live interface and usually
//! `CAP_NET_RAW`/root.
//!
//! Usage: `cargo run --example smoke --features xdp -- <ifname> <queue_id>`

#[cfg(all(feature = "xdp", target_os = "linux"))]
fn main() {
    use proxima_net::xdp::Umem;
    use proxima_net::xdp::xsk::{RingSizes, UmemConfig, XskSocket};

    let mut args = std::env::args().skip(1);
    let ifname = args.next().unwrap_or_else(|| "lo".to_string());
    let queue_id: u32 = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let mode = args.next().unwrap_or_else(|| "both".to_string());

    let mut umem = match Umem::new(
        proxima_net::xdp::sized::UMEM_FRAME_COUNT,
        proxima_net::xdp::sized::UMEM_FRAME_SIZE,
    ) {
        Ok(umem) => umem,
        Err(error) => {
            eprintln!("umem allocation failed: {error}");
            return;
        }
    };
    let Some(frame) = umem.alloc_frame() else {
        eprintln!("freshly created umem unexpectedly has no free frames");
        return;
    };
    println!(
        "umem: allocated frame at offset {frame}, base_addr=0x{:x}",
        umem.base_addr()
    );
    umem.free_frame(frame);

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

    if mode == "both" || mode == "xsk" {
        println!("attempting XskSocket::bind(ifname={ifname:?}, queue_id={queue_id})");
        match XskSocket::bind(&ifname, queue_id, umem_cfg, ring_sizes, 0) {
            Ok(socket) => println!("bind succeeded: fd={}", socket.fd()),
            Err(error) => println!("bind failed: {error}"),
        }
    }

    if mode == "both" || mode == "listener" {
        use proxima_net::xdp::XdpPacketListener;
        use std::net::{Ipv4Addr, SocketAddrV4};

        let our_mac = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
        let bind_addr = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 9000);
        println!("attempting XdpPacketListener::bind(ifname={ifname:?}, queue_id={queue_id})");
        match XdpPacketListener::bind(&ifname, queue_id, our_mac, bind_addr) {
            Ok(_listener) => println!("XdpPacketListener::bind succeeded (fill ring seeded)"),
            Err(error) => println!("XdpPacketListener::bind failed: {error}"),
        }
    }
}

#[cfg(not(all(feature = "xdp", target_os = "linux")))]
fn main() {
    eprintln!("this example requires --features xdp on linux");
}
