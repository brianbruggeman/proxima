// Prime-driven TCP echo server over a dpdk net_tap PMD: proves DpdkStreamListener
// (accept + AsyncRead/AsyncWrite) inside the real prime runtime (no tokio). Run
// on a dpdk host (sudo):
//   cargo run --features dpdk --example dpdk_tcp_echo
// then from the kernel side:
//   sudo ip link set ptap0 up && sudo ip addr add 10.0.0.1/24 dev ptap0
//   printf 'hello stream\n' | nc -q1 10.0.0.2 7
//
// Env: PROXIMA_TAP_IFACE (ptap0), PROXIMA_OUR_IP (10.0.0.2), PROXIMA_PORT (7),
// PROXIMA_PMD_DIR (/usr/lib/dpdk/pmds-26.0).

use futures::io::{AsyncReadExt, AsyncWriteExt};
use prime::core::local_executor::LocalExecutor;
use proxima::listeners::DpdkStreamListener;
use proxima::stream::StreamListenerExt;
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
        .unwrap_or(7);
    let our_ip = parse_ipv4(&env::var("PROXIMA_OUR_IP").unwrap_or_else(|_| "10.0.0.2".to_string()))
        .ok_or("PROXIMA_OUR_IP must be a dotted-quad ipv4 address")?;

    let listener = DpdkStreamListener::bind(SocketAddrV4::new(our_ip, port), &iface, &pmd_dir)?;
    println!("dpdk-tcp-echo up on {our_ip}:{port} (prime LocalExecutor); kill to stop");

    let executor = LocalExecutor::new();
    executor.block_on(async move {
        // one connection at a time is enough to prove accept + read + write + EOF;
        // the pump runs on every poll, so other peers still handshake meanwhile.
        loop {
            let mut conn = match listener.accept().await {
                Ok(conn) => conn,
                Err(err) => {
                    eprintln!("accept error: {err}");
                    continue;
                }
            };
            let mut buf = vec![0_u8; 2048];
            loop {
                match conn.read(&mut buf).await {
                    Ok(0) => break, // peer closed
                    Ok(read) => {
                        if conn.write_all(&buf[..read]).await.is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        eprintln!("read error: {err}");
                        break;
                    }
                }
            }
        }
    });
    Ok(())
}
