// Active-open TCP client over a dpdk net_tap PMD: proves DpdkStreamUpstream
// (ARP resolve + handshake + AsyncRead/AsyncWrite) by connecting to a *kernel*
// TCP echo server on the tap, under the prime runtime (no tokio).
//
// The tap is created by this process, so it waits PROXIMA_CONNECT_DELAY seconds
// after bring-up to let you configure the kernel side + start the echo server:
//   cargo run --features dpdk --example dpdk_tcp_connect &   (sudo)
//   sudo ip link set ptap0 up && sudo ip addr add 10.0.0.1/24 dev ptap0
//   ncat -l 10.0.0.1 8080 --keep-open --exec /bin/cat
//
// Env: PROXIMA_TAP_IFACE (ptap0), PROXIMA_LOCAL_IP (10.0.0.2),
// PROXIMA_PEER_IP (10.0.0.1), PROXIMA_PEER_PORT (8080),
// PROXIMA_CONNECT_DELAY (6), PROXIMA_PMD_DIR (/usr/lib/dpdk/pmds-26.0).

use futures::io::{AsyncReadExt, AsyncWriteExt};
use prime::core::local_executor::LocalExecutor;
use proxima::listeners::DpdkStreamUpstream;
use proxima::stream::StreamUpstreamExt;
use std::env;
use std::error::Error;
use std::io;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::time::Duration;

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
    let local_ip =
        parse_ipv4(&env::var("PROXIMA_LOCAL_IP").unwrap_or_else(|_| "10.0.0.2".to_string()))
            .ok_or("PROXIMA_LOCAL_IP must be a dotted-quad")?;
    let peer_ip =
        parse_ipv4(&env::var("PROXIMA_PEER_IP").unwrap_or_else(|_| "10.0.0.1".to_string()))
            .ok_or("PROXIMA_PEER_IP must be a dotted-quad")?;
    let peer_port: u16 = env::var("PROXIMA_PEER_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let delay: u64 = env::var("PROXIMA_CONNECT_DELAY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let local_port: u16 = env::var("PROXIMA_LOCAL_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40000);

    let local = SocketAddrV4::new(local_ip, local_port);
    let peer = SocketAddrV4::new(peer_ip, peer_port);
    let upstream = DpdkStreamUpstream::bind(local, peer, &iface, &pmd_dir)?;
    println!(
        "ptap0 created; waiting {delay}s to set up the kernel echo server, then connecting to {peer}"
    );
    std::thread::sleep(Duration::from_secs(delay));

    let executor = LocalExecutor::new();
    let result: io::Result<Vec<u8>> = executor.block_on(async move {
        let mut conn = upstream.connect().await?;
        conn.write_all(b"hello from the dpdk client\n").await?;
        let mut buf = vec![0_u8; 256];
        let read = conn.read(&mut buf).await?;
        buf.truncate(read);
        Ok(buf)
    });

    match result {
        Ok(bytes) => println!("CLIENT_RECV: {}", String::from_utf8_lossy(&bytes)),
        Err(err) => {
            eprintln!("connect/echo failed: {err}");
            return Err(err.into());
        }
    }
    Ok(())
}
