// Userspace TCP echo server over dpdk: the proxima-tcp stack driven off the
// net_tap RX ring. ARP/ICMP are answered so the kernel can reach us; TCP to
// our_ip:our_port runs through the sans-IO EchoListener and the replies are
// serialized back onto the TX ring. Build/run on a dpdk host (sudo):
//   cargo run --features dpdk --bin tcp-echo
// then from the kernel side: ip link/addr ptap0, and `nc 10.0.0.2 7`.

#[cfg(feature = "dpdk")]
fn main() -> std::process::ExitCode {
    imp::main()
}

#[cfg(not(feature = "dpdk"))]
fn main() {
    eprintln!("proxima-net-dpdk: build with --features dpdk on a dpdk host");
}

#[cfg(feature = "dpdk")]
mod imp {
    use proxima_protocols::inet::ethernet::{self, EtherType, EthernetFrame};
    use proxima_protocols::inet::ipv4::{self, Ipv4Header, Ipv4Protocol};
    use proxima_protocols::inet::tcp::{self, TcpHeader};
    use proxima_net::stack::{self, Action};
    use proxima_net::tcp_listener::{EchoListener, Endpoint, Inbound, OutSegment};
    use proxima_net::dpdk::{Eal, Mempool, Port, RawMbuf, port};
    use proxima_protocols::tcp::time::Instant as TcpInstant;
    use std::env;
    use std::process::ExitCode;
    use std::time::{Duration, Instant};

    const BURST: usize = 32;
    const ETH: usize = 14;
    const IP: usize = 20;
    const TCP: usize = 20;

    fn parse_ipv4(text: &str) -> Option<[u8; 4]> {
        let mut octets = [0u8; 4];
        let mut parts = text.split('.');
        for octet in &mut octets {
            *octet = parts.next()?.parse().ok()?;
        }
        if parts.next().is_some() {
            None
        } else {
            Some(octets)
        }
    }

    // A self-contained inbound segment (payload copied) so the source mbuf can be
    // freed before we build replies.
    struct InboundOwned {
        source_mac: [u8; 6],
        source_ip: [u8; 4],
        source_port: u16,
        flags: tcp::TcpFlags,
        seq: u32,
        ack: u32,
        window: u16,
        payload: Vec<u8>,
    }

    // Classify a frame as TCP-for-us, returning the owned segment; None means
    // "not our TCP" (let the ARP/ICMP responder handle it).
    fn classify(frame: &[u8], our_ip: [u8; 4], our_port: u16) -> Option<InboundOwned> {
        let eth = EthernetFrame::parse(frame).ok()?;
        if eth.ether_type() != EtherType::Ipv4 {
            return None;
        }
        let ip = Ipv4Header::parse(eth.payload()).ok()?;
        if ip.protocol() != Ipv4Protocol::Tcp || ip.destination() != our_ip {
            return None;
        }
        let segment = TcpHeader::parse(ip.payload()).ok()?;
        if segment.destination_port() != our_port {
            return None;
        }
        Some(InboundOwned {
            source_mac: eth.source(),
            source_ip: ip.source(),
            source_port: segment.source_port(),
            flags: segment.flags(),
            seq: segment.sequence(),
            ack: segment.acknowledgement(),
            window: segment.window(),
            payload: segment.payload().to_vec(),
        })
    }

    fn transmit(
        pool: &Mempool,
        dev: &Port,
        our_mac: [u8; 6],
        our_ip: [u8; 4],
        our_port: u16,
        peer: Endpoint,
        seg: &OutSegment,
    ) {
        let payload_len = u16::try_from(seg.payload.len()).unwrap_or(0);
        let total = u16::try_from(ETH + IP + TCP).unwrap_or(0) + payload_len;
        let mbuf: RawMbuf = pool.alloc();
        if mbuf.is_null() {
            eprintln!("warn: mbuf pool exhausted, dropping reply");
            return;
        }
        let Some(buf) = (unsafe { port::frame_append(mbuf, total) }) else {
            unsafe { port::free(mbuf) };
            return;
        };
        let _ = ethernet::write_header(&mut buf[..ETH], peer.mac, our_mac, EtherType::Ipv4);
        let l4_len = u16::try_from(TCP).unwrap_or(0) + payload_len;
        let _ = ipv4::write_header(
            &mut buf[ETH..ETH + IP],
            our_ip,
            peer.ip,
            Ipv4Protocol::Tcp,
            64,
            l4_len,
            0,
        );
        buf[ETH + IP + TCP..].copy_from_slice(&seg.payload);
        let _ = tcp::write_header(
            &mut buf[ETH + IP..],
            our_ip,
            peer.ip,
            our_port,
            peer.port,
            seg.seq,
            seg.ack,
            seg.flags,
            seg.window,
            &seg.payload,
        );
        let mut one = [mbuf];
        if dev.tx_burst(&mut one) == 0 {
            unsafe { port::free(mbuf) };
        }
    }

    fn run() -> Result<(), Box<dyn std::error::Error>> {
        let iface = env::var("PROXIMA_TAP_IFACE").unwrap_or_else(|_| "ptap0".to_string());
        let pmd_dir =
            env::var("PROXIMA_PMD_DIR").unwrap_or_else(|_| "/usr/lib/dpdk/pmds-26.0".to_string());
        let run_secs: u64 = env::var("PROXIMA_RUN_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        let our_ip =
            parse_ipv4(&env::var("PROXIMA_OUR_IP").unwrap_or_else(|_| "10.0.0.2".to_string()))
                .ok_or("PROXIMA_OUR_IP is not a dotted-quad ipv4 address")?;
        let our_port: u16 = env::var("PROXIMA_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(7);

        let vdev = format!("--vdev=net_tap0,iface={iface}");
        let eal_args = ["tcp-echo", "-l", "0", "--no-pci", "-d", &pmd_dir, &vdev];
        let _eal = Eal::init(&eal_args)?;

        let pool = Mempool::create("pndtcp_pool", 8192, -1)?;
        let dev = Port::init(0, &pool)?;
        let our_mac = dev.mac()?;
        let mut listener = EchoListener::new(our_ip, our_port, 0x1000);
        println!(
            "tcp-echo up: {}.{}.{}.{}:{our_port}, mac {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, {run_secs}s",
            our_ip[0],
            our_ip[1],
            our_ip[2],
            our_ip[3],
            our_mac[0],
            our_mac[1],
            our_mac[2],
            our_mac[3],
            our_mac[4],
            our_mac[5],
        );

        let start = Instant::now();
        let deadline = start + Duration::from_secs(run_secs);
        let mut rx_bufs: [RawMbuf; BURST] = [core::ptr::null_mut(); BURST];
        let mut connections: u64 = 0;

        while Instant::now() < deadline {
            let received = usize::from(dev.rx_burst(&mut rx_bufs));
            let now = TcpInstant::from_micros(
                u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX),
            );
            for &mbuf in &rx_bufs[..received] {
                let frame = unsafe { port::frame_bytes_mut(mbuf) };
                if let Some(inbound) = classify(frame, our_ip, our_port) {
                    let parsed = Inbound {
                        source_mac: inbound.source_mac,
                        source_ip: inbound.source_ip,
                        source_port: inbound.source_port,
                        flags: inbound.flags,
                        seq: inbound.seq,
                        ack: inbound.ack,
                        window: inbound.window,
                        payload: &inbound.payload,
                    };
                    if inbound.flags.syn && !inbound.flags.ack {
                        connections += 1;
                    }
                    if let Some(response) = listener.on_inbound(&parsed, now) {
                        for seg in &response.segments {
                            transmit(&pool, &dev, our_mac, our_ip, our_port, response.peer, seg);
                        }
                    }
                    unsafe { port::free(mbuf) };
                } else {
                    match stack::handle_frame(frame, our_mac, our_ip) {
                        Action::Transmit => {
                            let mut one = [mbuf];
                            if dev.tx_burst(&mut one) == 0 {
                                unsafe { port::free(mbuf) };
                            }
                        }
                        Action::Drop => unsafe { port::free(mbuf) },
                    }
                }
            }
        }

        println!("done: {connections} connections opened");
        Ok(())
    }

    pub fn main() -> ExitCode {
        match run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("tcp-echo failed: {err}");
                ExitCode::FAILURE
            }
        }
    }
}
