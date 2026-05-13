//! Reactor-driven AF_XDP UDP echo, run on a real prime worker so the xsk fd
//! parks on the per-core reactor (no busy-poll). Echoes every UDP datagram
//! back to its sender; the netns burst client floods far more datagrams than
//! the ring depth (several wraps) and verifies byte-exactness per sequence.
//! The server side only echoes + counts; the client does the corruption check.
//!
//! Usage: `xdp_reactor_burst <ifname> <our_ip> <port> <run_seconds>`
//! Reads the interface MAC from `/sys/class/net/<ifname>/address`. Requires
//! root (BPF load + XDP attach). veth only.

#[cfg(all(feature = "xdp", target_os = "linux"))]
fn main() {
    use prime::os::core_shard;
    use proxima_net::packet::{Packet, PacketListenerExt};
    use proxima_net::xdp::XdpPacketListener;
    use proxima_runtime::CoreId;
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
    let our_ip: Ipv4Addr = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(Ipv4Addr::new(10, 0, 0, 1));
    let port: u16 = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9000);
    let run_seconds: u64 = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(15);

    let mac_path = format!("/sys/class/net/{ifname}/address");
    let Ok(mac_text) = std::fs::read_to_string(&mac_path) else {
        eprintln!("cannot read {mac_path}");
        return;
    };
    let Some(our_mac) = parse_mac(&mac_text) else {
        eprintln!("cannot parse mac {mac_text:?}");
        return;
    };
    println!("reactor-burst: ifname={ifname} our_ip={our_ip} port={port} mac={our_mac:02x?}");

    let handle = match core_shard::launch(CoreId(0), None) {
        Ok(handle) => handle,
        Err(error) => {
            eprintln!("prime launch failed: {error}");
            return;
        }
    };

    let count = Arc::new(AtomicU64::new(0));
    let ready = Arc::new(AtomicBool::new(false));
    let failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let count_worker = count.clone();
    let ready_worker = ready.clone();
    let failure_worker = failure.clone();
    let bind_addr = SocketAddrV4::new(our_ip, port);

    let dispatched = handle.dispatch_send_inline(async move {
        let listener = match XdpPacketListener::bind(&ifname, 0, our_mac, bind_addr) {
            Ok(listener) => listener,
            Err(error) => {
                if let Ok(mut slot) = failure_worker.lock() {
                    *slot = Some(error.to_string());
                }
                ready_worker.store(true, Ordering::Release);
                return;
            }
        };
        ready_worker.store(true, Ordering::Release);
        let mut scratch = [0u8; 2048];
        while let Ok(packet) = listener.recv(&mut scratch).await {
            let reply = Packet {
                src: packet.src,
                dst: packet.dst,
                data: packet.data,
            };
            if listener.send(&reply).await.is_ok() {
                count_worker.fetch_add(1, Ordering::Relaxed);
            }
        }
    });
    if let Err(error) = dispatched {
        eprintln!("dispatch failed: {error}");
        let _ = handle.shutdown_and_join();
        return;
    }

    let ready_deadline = Instant::now() + Duration::from_secs(5);
    while !ready.load(Ordering::Acquire) {
        if Instant::now() >= ready_deadline {
            eprintln!("listener never became ready");
            let _ = handle.shutdown_and_join();
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if let Ok(slot) = failure.lock()
        && let Some(message) = slot.as_ref()
    {
        eprintln!("listener bind failed: {message}");
        let _ = handle.shutdown_and_join();
        return;
    }

    println!("listener ready (reactor-parked); echoing for {run_seconds}s");
    std::thread::sleep(Duration::from_secs(run_seconds));
    println!(
        "server echoed {} datagram(s)",
        count.load(Ordering::Relaxed)
    );
    let _ = handle.shutdown_and_join();
}

#[cfg(not(all(feature = "xdp", target_os = "linux")))]
fn main() {
    eprintln!("this example requires --features xdp on linux");
}
