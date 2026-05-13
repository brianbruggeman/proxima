// Userspace ARP + ICMP-echo responder over dpdk: brings up a net_tap vdev and
// answers ping for our address entirely in our own code (EAL -> RX -> sans-IO
// stack -> TX). Build/run on a dpdk host: cargo run --features dpdk --bin
// ping-responder (sudo); then ip link/addr the kernel side and ping 10.0.0.2.

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
    use proxima_net::stack::{self, Action};
    use proxima_net::dpdk::{Eal, Mempool, Port, RawMbuf, port};
    use std::env;
    use std::process::ExitCode;
    use std::time::{Duration, Instant};

    const BURST: usize = 32;

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

    fn run() -> Result<(), Box<dyn std::error::Error>> {
        let iface = env::var("PROXIMA_TAP_IFACE").unwrap_or_else(|_| "ptap0".to_string());
        let pmd_dir =
            env::var("PROXIMA_PMD_DIR").unwrap_or_else(|_| "/usr/lib/dpdk/pmds-26.0".to_string());
        let run_secs: u64 = env::var("PROXIMA_RUN_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(15);
        let our_ip =
            parse_ipv4(&env::var("PROXIMA_OUR_IP").unwrap_or_else(|_| "10.0.0.2".to_string()))
                .ok_or("PROXIMA_OUR_IP is not a dotted-quad ipv4 address")?;

        let vdev = format!("--vdev=net_tap0,iface={iface}");
        let eal_args = [
            "ping-responder",
            "-l",
            "0",
            "--no-pci",
            "-d",
            &pmd_dir,
            &vdev,
        ];
        let _eal = Eal::init(&eal_args)?;

        let pool = Mempool::create("pndresp_pool", 8192, -1)?;
        let dev = Port::init(0, &pool)?;
        let our_mac = dev.mac()?;
        println!(
            "responder up: ip {}.{}.{}.{}, mac {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, answering arp+icmp for {run_secs}s",
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

        let deadline = Instant::now() + Duration::from_secs(run_secs);
        let mut rx_bufs: [RawMbuf; BURST] = [core::ptr::null_mut(); BURST];
        let mut tx_bufs: [RawMbuf; BURST] = [core::ptr::null_mut(); BURST];
        let mut answered: usize = 0;

        while Instant::now() < deadline {
            let received = usize::from(dev.rx_burst(&mut rx_bufs));
            let mut to_send = 0usize;
            for &mbuf in &rx_bufs[..received] {
                let frame = unsafe { port::frame_bytes_mut(mbuf) };
                match stack::handle_frame(frame, our_mac, our_ip) {
                    Action::Transmit => {
                        tx_bufs[to_send] = mbuf;
                        to_send += 1;
                    }
                    Action::Drop => unsafe { port::free(mbuf) },
                }
            }
            if to_send == 0 {
                continue;
            }
            let sent = usize::from(dev.tx_burst(&mut tx_bufs[..to_send]));
            answered += sent;
            for &mbuf in &tx_bufs[sent..to_send] {
                unsafe { port::free(mbuf) };
            }
        }

        println!("done: answered {answered} frames");
        Ok(())
    }

    pub fn main() -> ExitCode {
        match run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("ping-responder failed: {err}");
                ExitCode::FAILURE
            }
        }
    }
}
