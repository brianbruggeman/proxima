// Prime-driven UDP echo over a dpdk net_tap PMD: proves DpdkPacketListener works
// inside the real prime runtime (no tokio). Run on a dpdk host (sudo):
//   cargo run --features dpdk --example dpdk_udp_echo
// then from the kernel side:
//   sudo ip link set ptap0 up && sudo ip addr add 10.0.0.1/24 dev ptap0
//   echo hi | nc -u -w1 10.0.0.2 9999
//
// Env: PROXIMA_TAP_IFACE (ptap0), PROXIMA_OUR_IP (10.0.0.2), PROXIMA_PORT (9999),
// PROXIMA_PMD_DIR (/usr/lib/dpdk/pmds-26.0).

use prime::core::local_executor::LocalExecutor;
use proxima::PacketListenerExt;
use proxima::listeners::DpdkPacketListener;
use std::env;
use std::error::Error;
use std::net::{Ipv4Addr, SocketAddrV4};

fn parse_ipv4(text: &str) -> Option<Ipv4Addr> {
    let mut octets = [0u8; 4];
    let mut parts = text.split('.');
    for octet in &mut octets {
        *octet = parts.next()?.parse().ok()?;
    }
    if parts.next().is_some() {
        None
    } else {
        Some(Ipv4Addr::from(octets))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let iface = env::var("PROXIMA_TAP_IFACE").unwrap_or_else(|_| "ptap0".to_string());
    let pmd_dir =
        env::var("PROXIMA_PMD_DIR").unwrap_or_else(|_| "/usr/lib/dpdk/pmds-26.0".to_string());
    let port: u16 = env::var("PROXIMA_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9999);
    let our_ip = parse_ipv4(&env::var("PROXIMA_OUR_IP").unwrap_or_else(|_| "10.0.0.2".to_string()))
        .ok_or("PROXIMA_OUR_IP must be a dotted-quad ipv4 address")?;

    let listener = DpdkPacketListener::bind(SocketAddrV4::new(our_ip, port), &iface, &pmd_dir)?;
    println!("dpdk-udp-echo up on {our_ip}:{port} (prime LocalExecutor); kill to stop");

    // prime per-core executor drives the busy-poll listener (PrimeRuntime is N of
    // these pinned per core; one core suffices for a single PMD queue).
    let executor = LocalExecutor::new();
    executor.block_on(async move {
        let mut buf = vec![0_u8; 2048];
        loop {
            match listener.recv(&mut buf).await {
                // the Packet's `src` is the peer; sending it straight back echoes.
                Ok(packet) => {
                    let _ = listener.send(&packet).await;
                }
                Err(err) => eprintln!("recv error: {err}"),
            }
        }
    });
    Ok(())
}
