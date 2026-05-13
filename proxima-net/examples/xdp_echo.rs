//! Live AF_XDP UDP echo: brings up an `XdpPacketListener` on a real interface
//! (loads + attaches the redirect XDP program, seeds the fill ring), then
//! busy-polls the RX ring echoing every UDP datagram back to its sender. ARP
//! and ICMP are answered inline by the listener so a kernel peer can resolve
//! and ping us first.
//!
//! Usage: `xdp_echo <ifname> <our_ip> <port> <seconds> <max_echoes>`
//! The interface MAC is read from `/sys/class/net/<ifname>/address`.
//! Requires `CAP_NET_ADMIN`/root (BPF load + XDP attach). Run on a veth so
//! attaching the redirect program does not disturb a production NIC.

#[cfg(all(feature = "xdp", target_os = "linux"))]
fn main() {
    use proxima_net::packet::{Packet, PacketListener};
    use proxima_net::xdp::XdpPacketListener;
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    use std::time::{Duration, Instant};

    fn parse_mac(text: &str) -> Option<[u8; 6]> {
        let mut mac = [0u8; 6];
        let mut parts = text.trim().split(':');
        for slot in &mut mac {
            let byte = parts.next()?;
            *slot = u8::from_str_radix(byte, 16).ok()?;
        }
        Some(mac)
    }

    const VTABLE: RawWakerVTable =
        RawWakerVTable::new(|data| RawWaker::new(data, &VTABLE), |_| {}, |_| {}, |_| {});

    let mut args = std::env::args().skip(1);
    let ifname = args.next().unwrap_or_else(|| "veth0".to_string());
    let our_ip: Ipv4Addr = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(Ipv4Addr::new(10, 0, 0, 1));
    let port: u16 = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9000);
    let seconds: u64 = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);
    let max_echoes: u64 = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);

    let mac_path = format!("/sys/class/net/{ifname}/address");
    let mac_text = match std::fs::read_to_string(&mac_path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("cannot read {mac_path}: {error}");
            return;
        }
    };
    let Some(our_mac) = parse_mac(&mac_text) else {
        eprintln!("cannot parse mac {mac_text:?}");
        return;
    };
    println!("echo: ifname={ifname} our_ip={our_ip} port={port} mac={our_mac:02x?}");

    let bind_addr = SocketAddrV4::new(our_ip, port);
    let listener = match XdpPacketListener::bind(&ifname, 0, our_mac, bind_addr) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("bind failed: {error}");
            return;
        }
    };
    println!("listener up: xdp program attached, fill ring seeded; echoing for {seconds}s");

    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut context = Context::from_waker(&waker);
    let mut scratch = [0u8; 2048];
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut echoes = 0u64;

    while Instant::now() < deadline && echoes < max_echoes {
        match listener.poll_recv(&mut context, &mut scratch) {
            Poll::Ready(Ok(packet)) => {
                println!(
                    "recv from {}: {} bytes: {:02x?}",
                    packet.src,
                    packet.data.len(),
                    &packet.data[..]
                );
                let reply = Packet {
                    src: packet.src,
                    dst: packet.dst,
                    data: packet.data.clone(),
                };
                loop {
                    match listener.poll_send(&mut context, &reply) {
                        Poll::Ready(Ok(())) => {
                            let SocketAddr::V4(peer) = reply.src else {
                                break;
                            };
                            println!("echoed {} bytes back to {peer}", reply.data.len());
                            echoes += 1;
                            break;
                        }
                        Poll::Ready(Err(error)) => {
                            eprintln!("send failed: {error}");
                            break;
                        }
                        Poll::Pending => {}
                    }
                }
            }
            Poll::Ready(Err(error)) => {
                eprintln!("recv failed: {error}");
                break;
            }
            Poll::Pending => {}
        }
    }
    println!("echo loop done: {echoes} datagram(s) echoed");
}

#[cfg(not(all(feature = "xdp", target_os = "linux")))]
fn main() {
    eprintln!("this example requires --features xdp on linux");
}
