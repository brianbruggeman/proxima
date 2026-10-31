//! OS-1 exit proof: a guest that is only a network host (virtio-net on the
//! sans-IO driver codec, served by proxima-net's ARP/ICMP responder) is pinged
//! by smoltcp, an independent implementation, over the device's frame pipes.
//! The peer sees only `Pipe` ends, so the same peer runs unchanged against a
//! relay that records frames in between.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use futures::FutureExt;
use proxima_primitives::pipe::{AndThen, Pipe};
use proxima_vm::elf::parse_elf;
use proxima_vm::frame_pipe::{FrameError, FrameSink, FrameSource, frame_channel};
use proxima_vm::net_run::run_net_guest;
use proxima_vm::virtio_net::NetTransport;
use smoltcp::iface::{Config, Interface, SocketSet};
use smoltcp::phy::{Device, DeviceCapabilities, Medium, RxToken, TxToken};
use smoltcp::socket::icmp::{Endpoint, PacketBuffer, PacketMetadata, Socket};
use smoltcp::time::Instant as SmoltcpInstant;
use smoltcp::wire::{
    EthernetAddress, HardwareAddress, Icmpv4Packet, Icmpv4Repr, IpAddress, IpCidr, Ipv4Address,
};

const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");
const GUEST_BIN: &str = "proxima-vm-guest-lambda-netstack";
const GUEST_MEMORY_SIZE: u64 = 64 * 1024 * 1024;
const WATCHDOG_MILLIS: u64 = 60_000;
const GUEST_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x02];
const PEER_MAC: [u8; 6] = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
const GUEST_IP: [u8; 4] = [10, 0, 0, 2];
const PEER_IP: [u8; 4] = [10, 0, 0, 1];
const STACK_DISABLED_BIT: u64 = 1 << 32;
const PING_COUNT: u16 = 100;
const PAYLOAD_LEN: usize = 56;
const REPLY_DEADLINE: Duration = Duration::from_secs(5);
const SILENCE_DEADLINE: Duration = Duration::from_millis(500);
const IDENT: u16 = 0x4f53;

fn guest_elf() -> Vec<u8> {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    let path = PATH.get_or_init(|| {
        let target_dir = std::env::var_os("CARGO_TARGET_DIR")
            .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"), PathBuf::from)
            .join("proxima-vm-netstack-guest");
        let status = Command::new("cargo")
            .args(["build", "--release", "--target", "x86_64-unknown-none", "--bin", GUEST_BIN])
            .args(["--manifest-path", &format!("{GUEST_MANIFEST_DIR}/Cargo.toml")])
            .env("CARGO_TARGET_DIR", &target_dir)
            .status()
            .expect("run cargo build for the guest ELF");
        assert!(status.success(), "guest ELF build failed");
        target_dir.join("x86_64-unknown-none/release").join(GUEST_BIN)
    });
    std::fs::read(path).expect("read the built guest ELF")
}

fn boot_argument(stack_enabled: bool) -> u64 {
    let ip = u64::from(u32::from_be_bytes(GUEST_IP));
    if stack_enabled { ip } else { ip | STACK_DISABLED_BIT }
}

struct GuestRun {
    rx_delivered: u64,
    tx_forwarded: u64,
    virtio_exits: u64,
}

fn start_guest(
    ingress: FrameSource,
    egress: FrameSink,
    stack_enabled: bool,
) -> thread::JoinHandle<GuestRun> {
    let image = guest_elf();
    thread::spawn(move || {
        let (entry, segments) = parse_elf::<4>(&image).expect("parse the guest ELF");
        let mut transport = NetTransport::with_pipes(GUEST_MAC, ingress, egress);
        let virtio_exits = run_net_guest(
            entry,
            &segments,
            GUEST_MEMORY_SIZE,
            boot_argument(stack_enabled),
            &mut transport,
            WATCHDOG_MILLIS,
        )
        .expect("the guest halts when the peer closes the link");
        GuestRun {
            rx_delivered: transport.rx_delivered(),
            tx_forwarded: transport.tx_forwarded(),
            virtio_exits,
        }
    })
}

struct PipeDevice<Inbound, Outbound> {
    inbound: Inbound,
    outbound: Outbound,
}

struct InboundFrame(Vec<u8>);

struct OutboundSlot<'pipe, Outbound>(&'pipe Outbound);

impl RxToken for InboundFrame {
    fn consume<R, F: FnOnce(&[u8]) -> R>(self, f: F) -> R {
        f(&self.0)
    }
}

impl<Outbound> TxToken for OutboundSlot<'_, Outbound>
where
    Outbound: Pipe<In = Vec<u8>, Out = (), Err = FrameError>,
{
    fn consume<R, F: FnOnce(&mut [u8]) -> R>(self, len: usize, f: F) -> R {
        let mut frame = vec![0u8; len];
        let result = f(&mut frame);
        self.0.call(frame).now_or_never().expect("sink is ready").ok();
        result
    }
}

impl<Inbound, Outbound> Device for PipeDevice<Inbound, Outbound>
where
    Inbound: Pipe<In = (), Out = Vec<u8>, Err = FrameError>,
    Outbound: Pipe<In = Vec<u8>, Out = (), Err = FrameError>,
{
    type RxToken<'a>
        = InboundFrame
    where
        Self: 'a;
    type TxToken<'a>
        = OutboundSlot<'a, Outbound>
    where
        Self: 'a;

    fn receive(&mut self, _timestamp: SmoltcpInstant) -> Option<(InboundFrame, OutboundSlot<'_, Outbound>)> {
        let frame = self.inbound.call(()).now_or_never()?.ok()?;
        Some((InboundFrame(frame), OutboundSlot(&self.outbound)))
    }

    fn transmit(&mut self, _timestamp: SmoltcpInstant) -> Option<OutboundSlot<'_, Outbound>> {
        Some(OutboundSlot(&self.outbound))
    }

    fn capabilities(&self) -> DeviceCapabilities {
        let mut capabilities = DeviceCapabilities::default();
        capabilities.medium = Medium::Ethernet;
        capabilities.max_transmission_unit = 1500;
        capabilities
    }
}

#[derive(Debug, PartialEq)]
enum PingFailure {
    Timeout { sequence: u16 },
    WrongReply { sequence: u16 },
}

fn payload_for(sequence: u16) -> Vec<u8> {
    (0..PAYLOAD_LEN)
        .map(|index| (index as u8).wrapping_mul(7).wrapping_add(sequence as u8))
        .collect()
}

fn timestamp(start: Instant) -> SmoltcpInstant {
    SmoltcpInstant::from_micros(start.elapsed().as_micros() as i64)
}

fn ping_guest<Inbound, Outbound>(
    device: &mut PipeDevice<Inbound, Outbound>,
    count: u16,
    reply_deadline: Duration,
) -> Result<Vec<Duration>, PingFailure>
where
    Inbound: Pipe<In = (), Out = Vec<u8>, Err = FrameError>,
    Outbound: Pipe<In = Vec<u8>, Out = (), Err = FrameError>,
{
    let start = Instant::now();
    let config = Config::new(HardwareAddress::Ethernet(EthernetAddress(PEER_MAC)));
    let mut interface = Interface::new(config, device, timestamp(start));
    interface.update_ip_addrs(|addresses| {
        addresses
            .push(IpCidr::new(IpAddress::Ipv4(Ipv4Address::from_octets(PEER_IP)), 24))
            .expect("one address fits");
    });
    let buffer = || PacketBuffer::new(vec![PacketMetadata::EMPTY; 4], vec![0; 2048]);
    let mut sockets = SocketSet::new(vec![]);
    let handle = sockets.add(Socket::new(buffer(), buffer()));
    sockets.get_mut::<Socket>(handle).bind(Endpoint::Ident(IDENT)).expect("bind icmp ident");
    let checksums = device.capabilities().checksum;
    let target = IpAddress::Ipv4(Ipv4Address::from_octets(GUEST_IP));

    let mut round_trips = Vec::new();
    for sequence in 0..count {
        let payload = payload_for(sequence);
        let request = Icmpv4Repr::EchoRequest { ident: IDENT, seq_no: sequence, data: &payload };
        let mut bytes = vec![0u8; request.buffer_len()];
        request.emit(&mut Icmpv4Packet::new_unchecked(&mut bytes), &checksums);
        sockets.get_mut::<Socket>(handle).send_slice(&bytes, target).expect("send echo request");
        let sent = Instant::now();
        loop {
            interface.poll(timestamp(start), device, &mut sockets);
            let socket = sockets.get_mut::<Socket>(handle);
            if socket.can_recv() {
                let (reply, source) = socket.recv().expect("reply is queued");
                let packet = Icmpv4Packet::new_checked(reply).expect("well-formed icmp");
                let parsed = Icmpv4Repr::parse(&packet, &checksums).expect("valid icmp checksum");
                let expected = Icmpv4Repr::EchoReply { ident: IDENT, seq_no: sequence, data: &payload };
                let from_guest = source == target;
                if !from_guest || parsed != expected {
                    return Err(PingFailure::WrongReply { sequence });
                }
                round_trips.push(sent.elapsed());
                break;
            }
            if sent.elapsed() > reply_deadline {
                return Err(PingFailure::Timeout { sequence });
            }
            std::hint::spin_loop();
        }
    }
    Ok(round_trips)
}

fn percentile(sorted: &[Duration], fraction: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

fn report(label: &str, round_trips: &[Duration]) {
    let mut sorted = round_trips.to_vec();
    sorted.sort();
    eprintln!(
        "{label}: {} round trips, p50 {:?}, p99 {:?}, max {:?}",
        sorted.len(),
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.99),
        sorted.last().unwrap()
    );
}

#[proxima::test]
async fn smoltcp_pings_the_guest_one_hundred_times_over_the_frame_pipes() {
    let (to_guest, ingress) = frame_channel(64);
    let (egress, from_guest) = frame_channel(64);
    let guest = start_guest(ingress, egress, true);
    let mut device = PipeDevice { inbound: from_guest, outbound: to_guest };

    let outcome = ping_guest(&mut device, PING_COUNT, REPLY_DEADLINE);
    device.outbound.close();
    let run = guest.join().expect("guest thread");

    let round_trips = outcome.expect("every echo request is answered with its own payload");
    report("direct", &round_trips);
    assert_eq!(round_trips.len(), usize::from(PING_COUNT));
    assert_eq!(run.rx_delivered, u64::from(PING_COUNT) + 1, "arp request + echo requests");
    assert_eq!(run.tx_forwarded, u64::from(PING_COUNT) + 1, "arp reply + echo replies");
    eprintln!("guest virtio exits: {}", run.virtio_exits);
}

#[proxima::test]
async fn guest_with_its_stack_disabled_never_answers_and_the_ping_times_out() {
    let (to_guest, ingress) = frame_channel(64);
    let (egress, from_guest) = frame_channel(64);
    let guest = start_guest(ingress, egress, false);
    let mut device = PipeDevice { inbound: from_guest, outbound: to_guest };

    let outcome = ping_guest(&mut device, 1, SILENCE_DEADLINE);
    device.outbound.close();
    let run = guest.join().expect("guest thread");

    assert_eq!(outcome, Err(PingFailure::Timeout { sequence: 0 }));
    assert!(run.rx_delivered >= 1, "the guest received the arp request and dropped it");
    assert_eq!(run.tx_forwarded, 0, "a guest that drops everything transmits nothing");
}

#[derive(Clone)]
struct Recorder {
    frames: Arc<AtomicUsize>,
    ipv4: Arc<AtomicUsize>,
    arp: Arc<AtomicUsize>,
}

impl Pipe for Recorder {
    type In = Vec<u8>;
    type Out = Vec<u8>;
    type Err = FrameError;

    fn call(&self, frame: Vec<u8>) -> impl Future<Output = Result<Vec<u8>, FrameError>> {
        async move {
            self.frames.fetch_add(1, Ordering::Relaxed);
            match frame.get(12..14) {
                Some([0x08, 0x00]) => self.ipv4.fetch_add(1, Ordering::Relaxed),
                Some([0x08, 0x06]) => self.arp.fetch_add(1, Ordering::Relaxed),
                _ => 0,
            };
            Ok(frame)
        }
    }
}

impl Recorder {
    fn new() -> Self {
        Self {
            frames: Arc::new(AtomicUsize::new(0)),
            ipv4: Arc::new(AtomicUsize::new(0)),
            arp: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[proxima::test]
async fn an_observe_pipe_between_device_and_peer_sees_every_frame_of_the_exchange() {
    let (to_guest, ingress) = frame_channel(64);
    let (egress, from_guest) = frame_channel(64);
    let (peer_tx, relay_rx) = frame_channel(64);
    let (relay_tx, peer_rx) = frame_channel(64);
    let (guest_recorder, peer_recorder) = (Recorder::new(), Recorder::new());
    let guest = start_guest(ingress, egress, true);

    let stop = Arc::new(AtomicBool::new(false));
    let relay = {
        let (stop, guest_recorder, peer_recorder) = (stop.clone(), guest_recorder.clone(), peer_recorder.clone());
        let to_guest_end = to_guest.clone();
        thread::spawn(move || {
            let toward_peer = AndThen::new(AndThen::new(from_guest, guest_recorder), relay_tx);
            let toward_guest = AndThen::new(AndThen::new(relay_rx, peer_recorder), to_guest);
            while !stop.load(Ordering::Acquire) {
                toward_peer.call(()).now_or_never().expect("pipes are ready").ok();
                if let Some(Err(FrameError::Closed)) = toward_guest.call(()).now_or_never() {
                    to_guest_end.close();
                }
            }
        })
    };
    let mut device = PipeDevice { inbound: peer_rx, outbound: peer_tx };

    let outcome = ping_guest(&mut device, PING_COUNT, REPLY_DEADLINE);
    device.outbound.close();
    let run = guest.join().expect("guest thread");
    stop.store(true, Ordering::Release);
    relay.join().expect("relay thread");

    let round_trips = outcome.expect("pings succeed through the relay unchanged");
    report("relayed", &round_trips);
    let echoes = usize::from(PING_COUNT);
    assert_eq!(guest_recorder.ipv4.load(Ordering::Relaxed), echoes, "echo replies seen leaving the guest");
    assert_eq!(peer_recorder.ipv4.load(Ordering::Relaxed), echoes, "echo requests seen entering the guest");
    assert_eq!(guest_recorder.arp.load(Ordering::Relaxed), 1);
    assert_eq!(peer_recorder.arp.load(Ordering::Relaxed), 1);
    assert_eq!(guest_recorder.frames.load(Ordering::Relaxed) as u64, run.tx_forwarded);
    assert_eq!(peer_recorder.frames.load(Ordering::Relaxed) as u64, run.rx_delivered);
}
