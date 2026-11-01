// a bench's setup has no caller to return an error to; a failed setup must abort the run.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::hint::black_box;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_net::tcp_listener::{Endpoint, Inbound};
use proxima_net::tcp_stack::{ConnId, TcpStack};
use proxima_protocols::inet::tcp::TcpFlags;
use proxima_protocols::tcp::time::Instant;

const SIZES: [usize; 3] = [54, 402, 1514];
const OUR_IP: [u8; 4] = [10, 0, 2, 15];
const PEER: Endpoint = Endpoint {
    mac: [2, 0, 0, 0, 0, 2],
    ip: [10, 0, 2, 2],
    port: 80,
};
const PEER_ISN: u32 = 900;
const NEUTRAL: &str = "neutral";
const FAVORS_INCUMBENT: &str = "design-favors-incumbent";

struct Established {
    stack: TcpStack,
    id: ConnId,
    our_sequence: u32,
    peer_sequence: u32,
}

fn flags_ack() -> TcpFlags {
    TcpFlags {
        ack: true,
        ..TcpFlags::default()
    }
}

fn from_peer(sequence: u32, acknowledgement: u32, flags: TcpFlags, payload: &[u8]) -> Inbound<'_> {
    Inbound {
        source_mac: PEER.mac,
        source_ip: PEER.ip,
        source_port: PEER.port,
        flags,
        seq: sequence,
        ack: acknowledgement,
        window: u16::MAX,
        payload,
    }
}

fn establish() -> Established {
    let mut stack = TcpStack::new(OUR_IP, 1234, 100);
    let (id, syn) = stack.connect(PEER).expect("ephemeral port available");
    let syn_ack = TcpFlags {
        syn: true,
        ack: true,
        ..TcpFlags::default()
    };
    let our_sequence = syn[0].1.seq.wrapping_add(1);
    let reply = stack.on_inbound_to(
        id.0,
        &from_peer(PEER_ISN, our_sequence, syn_ack, &[]),
        Instant::ZERO,
    );
    assert_eq!(reply.len(), 1, "syn-ack must produce exactly the handshake ack");
    assert_eq!(stack.poll_connected(), Some(id), "connection must surface");
    Established {
        stack,
        id,
        our_sequence,
        peer_sequence: PEER_ISN.wrapping_add(1),
    }
}

fn inbound_data_to_ack(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tcp_stack/inbound_segment_to_ack");
    group.throughput(Throughput::Elements(1));
    for size in SIZES {
        let payload = vec![0xA5_u8; size];
        let mut conn = establish();
        let mut drain = vec![0_u8; size];
        let first = conn.stack.on_inbound_to(
            conn.id.0,
            &from_peer(conn.peer_sequence, conn.our_sequence, flags_ack(), &payload),
            Instant::ZERO,
        );
        assert_eq!(first.len(), 1, "data segment must produce its ack");
        assert_eq!(conn.stack.read(conn.id, &mut drain), size);
        conn.peer_sequence = conn.peer_sequence.wrapping_add(size as u32);
        let id = BenchmarkId::new(format!("{NEUTRAL}/on_inbound_to+drain_read"), size);
        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                let reply = conn.stack.on_inbound_to(
                    conn.id.0,
                    &from_peer(conn.peer_sequence, conn.our_sequence, flags_ack(), &payload),
                    Instant::ZERO,
                );
                black_box(&reply);
                conn.peer_sequence = conn.peer_sequence.wrapping_add(size as u32);
                black_box(conn.stack.read(conn.id, &mut drain));
            });
        });
    }
    group.finish();
}

fn request_response_round_trip(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tcp_stack/request_response");
    group.throughput(Throughput::Elements(1));
    for size in SIZES {
        let request = vec![0x47_u8; size];
        let response = vec![0x52_u8; size];
        let mut conn = establish();
        let mut drain = vec![0_u8; size];
        let id = BenchmarkId::new(format!("{NEUTRAL}/write+inbound_response+read"), size);
        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                let sent = conn.stack.write(conn.id, &request, Instant::ZERO);
                black_box(&sent);
                conn.our_sequence = conn.our_sequence.wrapping_add(size as u32);
                let reply = conn.stack.on_inbound_to(
                    conn.id.0,
                    &from_peer(conn.peer_sequence, conn.our_sequence, flags_ack(), &response),
                    Instant::ZERO,
                );
                black_box(&reply);
                conn.peer_sequence = conn.peer_sequence.wrapping_add(size as u32);
                black_box(conn.stack.read(conn.id, &mut drain));
            });
        });
    }
    group.finish();
}

fn spawn_echo_pair(size: usize) -> (TcpStream, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let address = listener.local_addr().expect("listener address");
    let echo = thread::spawn(move || {
        let (mut server, _) = listener.accept().expect("accept client");
        server.set_nodelay(true).expect("server nodelay");
        let mut buffer = vec![0_u8; size];
        while server.read_exact(&mut buffer).is_ok() {
            if server.write_all(&buffer).is_err() {
                break;
            }
        }
    });
    let client = TcpStream::connect(address).expect("connect loopback");
    client.set_nodelay(true).expect("client nodelay");
    (client, echo)
}

fn host_kernel_keepalive(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tcp_stack/request_response");
    group.throughput(Throughput::Elements(1));
    for size in SIZES {
        let request = vec![0x47_u8; size];
        let mut response = vec![0_u8; size];
        let (mut client, echo) = spawn_echo_pair(size);
        let id = BenchmarkId::new(format!("{FAVORS_INCUMBENT}/std_net_loopback_echo"), size);
        group.bench_function(id, |bencher| {
            bencher.iter(|| {
                client.write_all(&request).expect("write request");
                client.read_exact(&mut response).expect("read response");
                black_box(&response);
            });
        });
        drop(client);
        echo.join().expect("echo thread exits on client close");
    }
    group.finish();
}

criterion_group!(
    benches,
    inbound_data_to_ack,
    request_response_round_trip,
    host_kernel_keepalive
);
criterion_main!(benches);
