//! Reactor-parked AF_XDP TCP, run on a real prime worker so the shared xsk fd
//! parks on the per-core reactor (no busy-poll) across accept/read/write.
//!
//! `listen`  mode: accept `N` sequential connections and byte-echo each until
//!           the peer half-closes; then send our FIN (proper passive close).
//! `connect` mode: active-open to a kernel echo server and round-trip a line.
//!
//! Usage: `xdp_tcp_reactor <ifname> listen  <our_ip> <port> <connections> <secs>`
//!        `xdp_tcp_reactor <ifname> connect <our_ip> <peer_ip> <port>`
//! Reads the interface MAC from `/sys/class/net/<ifname>/address`. Root + veth.

#[cfg(all(feature = "xdp", target_os = "linux"))]
fn main() {
    use core::future::poll_fn;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use prime::os::core_shard;
    use proxima_net::xdp::{XdpStreamListener, XdpStreamUpstream};
    use proxima_runtime::CoreId;
    use proxima_primitives::stream::{StreamListener, StreamUpstream};
    use std::net::{Ipv4Addr, SocketAddrV4};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

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
    println!("PID={}", std::process::id());

    let handle = match core_shard::launch(CoreId(0), None) {
        Ok(handle) => handle,
        Err(error) => {
            eprintln!("prime launch failed: {error}");
            return;
        }
    };
    let ready = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let echoed_total = Arc::new(AtomicU64::new(0));
    let result: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

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
        let ready_worker = ready.clone();
        let done_worker = done.clone();
        let echoed_worker = echoed_total.clone();
        let result_worker = result.clone();

        let dispatched = handle.dispatch_send_inline(async move {
            let listener = match XdpStreamListener::bind(bind_addr, &ifname, 0, our_mac) {
                Ok(listener) => listener,
                Err(error) => {
                    if let Ok(mut slot) = result_worker.lock() {
                        *slot = Some(format!("bind failed: {error}"));
                    }
                    ready_worker.store(true, Ordering::Release);
                    done_worker.store(true, Ordering::Release);
                    return;
                }
            };
            ready_worker.store(true, Ordering::Release);
            for _ in 0..connections {
                let mut conn = match poll_fn(|cx| listener.poll_accept(cx)).await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };
                let mut buf = [0u8; 2048];
                loop {
                    let read = match conn.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(read) => read,
                        Err(_) => break,
                    };
                    if conn.write_all(&buf[..read]).await.is_err() {
                        break;
                    }
                    echoed_worker.fetch_add(read as u64, Ordering::Relaxed);
                }
                let _ = conn.close().await;
            }
            done_worker.store(true, Ordering::Release);
        });
        if let Err(error) = dispatched {
            eprintln!("dispatch failed: {error}");
            let _ = handle.shutdown_and_join();
            return;
        }
        wait_ready(&ready);
        if let Ok(slot) = result.lock()
            && let Some(message) = slot.as_ref()
        {
            eprintln!("{message}");
        }
        println!("listener ready (reactor-parked); accepting {connections} connection(s)");
        wait_done(&done, Duration::from_secs(30));
        println!(
            "listener done: echoed {} bytes total",
            echoed_total.load(Ordering::Relaxed)
        );
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
        let ready_worker = ready.clone();
        let done_worker = done.clone();
        let result_worker = result.clone();

        let dispatched = handle.dispatch_send_inline(async move {
            ready_worker.store(true, Ordering::Release);
            let upstream = match XdpStreamUpstream::bind(local, peer, &ifname, 0, our_mac) {
                Ok(upstream) => upstream,
                Err(error) => {
                    if let Ok(mut slot) = result_worker.lock() {
                        *slot = Some(format!("bind failed: {error}"));
                    }
                    done_worker.store(true, Ordering::Release);
                    return;
                }
            };
            let outcome = async {
                let mut conn = poll_fn(|cx| upstream.poll_connect(cx)).await?;
                let line = b"hello-from-xdp-reactor-connect\n";
                conn.write_all(line).await?;
                let mut buf = [0u8; 64];
                let read = conn.read(&mut buf).await?;
                Ok::<bool, std::io::Error>(&buf[..read] == line)
            }
            .await;
            let message = match outcome {
                Ok(true) => "RESULT: BYTE_EXACT".to_string(),
                Ok(false) => "RESULT: DIVERGED".to_string(),
                Err(error) => format!("connect/round-trip failed: {error}"),
            };
            if let Ok(mut slot) = result_worker.lock() {
                *slot = Some(message);
            }
            done_worker.store(true, Ordering::Release);
        });
        if let Err(error) = dispatched {
            eprintln!("dispatch failed: {error}");
            let _ = handle.shutdown_and_join();
            return;
        }
        wait_done(&done, Duration::from_secs(20));
        if let Ok(slot) = result.lock()
            && let Some(message) = slot.as_ref()
        {
            println!("{message}");
        }
    } else {
        eprintln!("unknown mode {mode:?}; expected listen|connect");
    }
    let _ = handle.shutdown_and_join();

    fn wait_ready(ready: &std::sync::Arc<std::sync::atomic::AtomicBool>) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ready.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    fn wait_done(done: &std::sync::Arc<std::sync::atomic::AtomicBool>, budget: Duration) {
        let deadline = Instant::now() + budget;
        while !done.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

#[cfg(not(all(feature = "xdp", target_os = "linux")))]
fn main() {
    eprintln!("this example requires --features xdp on linux");
}
