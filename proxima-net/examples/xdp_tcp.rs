//! Live AF_XDP TCP over the `proxima_primitives::stream` traits, driven by
//! `futures::executor::block_on` (busy-poll; no prime worker needed).
//!
//! `listen` mode: `XdpStreamListener` accepts `N` sequential connections and
//! byte-echoes each until the peer half-closes — proving return-to-LISTEN.
//!
//! `connect` mode: `XdpStreamUpstream` ARP-resolves a peer, drives the active
//! open, writes a line, and reads the echo back from a kernel server.
//!
//! Usage: `xdp_tcp <ifname> listen  <our_ip> <port> <connections>`
//!        `xdp_tcp <ifname> connect <our_ip> <peer_ip> <port>`
//! Reads the interface MAC from `/sys/class/net/<ifname>/address`. Root + veth.

#[cfg(all(feature = "xdp", target_os = "linux"))]
fn main() {
    use futures::executor::block_on;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_net::xdp::{XdpStreamListener, XdpStreamUpstream};
    use proxima_primitives::stream::{StreamListenerExt, StreamUpstreamExt};
    use std::net::{Ipv4Addr, SocketAddrV4};

    fn parse_mac(text: &str) -> Option<[u8; 6]> {
        let mut mac = [0u8; 6];
        let mut parts = text.trim().split(':');
        for slot in &mut mac {
            *slot = u8::from_str_radix(parts.next()?, 16).ok()?;
        }
        Some(mac)
    }

    let mut args = std::env::args().skip(1);
    let ifname = args.next().unwrap_or_else(|| "veth0".to_string());
    let mode = args.next().unwrap_or_else(|| "listen".to_string());

    let mac_path = format!("/sys/class/net/{ifname}/address");
    let Ok(mac_text) = std::fs::read_to_string(&mac_path) else {
        eprintln!("cannot read {mac_path}");
        return;
    };
    let Some(our_mac) = parse_mac(&mac_text) else {
        eprintln!("cannot parse mac {mac_text:?}");
        return;
    };

    if mode == "listen" {
        let our_ip: Ipv4Addr = args
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(Ipv4Addr::new(10, 0, 0, 1));
        let port: u16 = args
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9100);
        let connections: u32 = args
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(2);
        let bind_addr = SocketAddrV4::new(our_ip, port);

        let listener = match XdpStreamListener::bind(bind_addr, &ifname, 0, our_mac) {
            Ok(listener) => listener,
            Err(error) => {
                eprintln!("listener bind failed: {error}");
                return;
            }
        };
        println!("tcp listen: {our_ip}:{port} for {connections} connection(s)");
        block_on(async move {
            for index in 0..connections {
                let mut conn = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(error) => {
                        eprintln!("accept failed: {error}");
                        return;
                    }
                };
                let mut buf = [0u8; 2048];
                let mut echoed = 0usize;
                loop {
                    let read = match conn.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(read) => read,
                        Err(error) => {
                            eprintln!("read failed: {error}");
                            break;
                        }
                    };
                    if conn.write_all(&buf[..read]).await.is_err() {
                        break;
                    }
                    echoed += read;
                }
                // close our write side so a half-closing peer (SHUT_WR then read)
                // gets EOF after the echo — proper passive close.
                let _ = conn.close().await;
                println!("connection {index}: echoed {echoed} bytes byte-for-byte");
            }
            println!("listener done: all connections served");
        });
    } else if mode == "connect" {
        let our_ip: Ipv4Addr = args
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(Ipv4Addr::new(10, 0, 0, 1));
        let peer_ip: Ipv4Addr = args
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(Ipv4Addr::new(10, 0, 0, 2));
        let port: u16 = args
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9200);
        let local = SocketAddrV4::new(our_ip, 40000);
        let peer = SocketAddrV4::new(peer_ip, port);

        let upstream = match XdpStreamUpstream::bind(local, peer, &ifname, 0, our_mac) {
            Ok(upstream) => upstream,
            Err(error) => {
                eprintln!("upstream bind failed: {error}");
                return;
            }
        };
        println!("tcp connect: {our_ip} -> {peer_ip}:{port}");
        block_on(async move {
            let mut conn = match upstream.connect().await {
                Ok(conn) => conn,
                Err(error) => {
                    eprintln!("connect failed: {error}");
                    return;
                }
            };
            let line = b"hello-from-xdp-active-open\n";
            if let Err(error) = conn.write_all(line).await {
                eprintln!("write failed: {error}");
                return;
            }
            let mut buf = [0u8; 64];
            match conn.read(&mut buf).await {
                Ok(read) => {
                    let echoed = &buf[..read];
                    let byte_exact = echoed == line;
                    println!("upstream sent : {line:02x?}");
                    println!("upstream recvd: {echoed:02x?}");
                    println!(
                        "RESULT: {}",
                        if byte_exact { "BYTE_EXACT" } else { "DIVERGED" }
                    );
                }
                Err(error) => eprintln!("read failed: {error}"),
            }
        });
    } else {
        eprintln!("unknown mode {mode:?}; expected listen|connect");
    }
}

#[cfg(not(all(feature = "xdp", target_os = "linux")))]
fn main() {
    eprintln!("this example requires --features xdp on linux");
}
