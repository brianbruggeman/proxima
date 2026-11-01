//! A resident OCI whoami guest answers an HTTP/1.1 GET dialed from the host
//! over the virtio-net window, through the frame-stack driver.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};
use std::fs;
use std::io;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::path::Path;
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::executor::block_on;
use futures::future::{Either, select};
use futures::io::{AsyncReadExt, AsyncWriteExt};
use http::{Request, Response};
use proxima_net::frame_stack::{Driver, FrameUpstream};
use proxima_primitives::pipe::Pipe;
use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::recorder::Recorder;
use proxima_vm::frame_pipe::FrameError;
use proxima_protocols::http1_codec::h1_client::encode_request_head;
use proxima_vm::kernel_modules::load_modules;
use proxima_vm::frame_pipe::FrameSource;
use proxima_vm::kvm::host_wire::{HOST_IP, HOST_MAC, RingingTx, ZeroMint, dial_guest_when_listening};
use proxima_vm::kvm::resident::{HostLink, LinkConfig, ResidentLinux, net_device_cmdline};
use proxima_vm::linux::LinuxBootInputs;
use proxima_vm::linux::LinuxGuestLimits;
use proxima_vm::oci::digest::Digest;
use proxima_vm::oci::error::OciError;
use proxima_vm::oci::{cache, initramfs_with_modules};
use proxima_vm::oci::reference::ImageRef;
use proxima_vm::oci::registry::{PulledImage, pull};
use proxima_vm::x86_boot::SetupHeader;

const GUEST_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
const GUEST_HTTP: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::new(10, 0, 2, 15), 80);
const MODULES_ROOT: &str = "/usr/lib/modules";
const DEADLINE: Duration = Duration::from_secs(60);
const POLL_CEILING: Duration = Duration::from_millis(5);
const CMDLINE: &str = "console=ttyS0 reboot=t panic=-1 i8042.noaux loglevel=4 proxima.resident";
const STATUS_LINE: &[u8] = b"HTTP/1.1 200";
const READ_CHUNK: usize = 1024;
const TRACE_FILE: &str = "whoami-trace.jsonl";
const TRACE_MESSAGE: &str = "resident whoami answered";

const HOST_PORT: u16 = 49152;
const HOST_INITIAL_SEQUENCE: u32 = 0x4000;
// a per-request watchdog this short would end the guest before eth0 is up, so
// a 200 proves the resident lane does not arm it.
const PER_REQUEST_WATCHDOG_MILLIS: u64 = 1;
const ETHERNET_HEADER: usize = 14;
const IPV4_PROTOCOL_OFFSET: usize = 23;
const TCP_PROTOCOL: u8 = 6;
const TCP_FLAGS_OFFSET: usize = 13;

type Upstream = FrameUpstream<Tap, RingingTx, ZeroMint>;
type Seen = Sender<String>;

struct Tap {
    inner: FrameSource,
    seen: Seen,
}

impl Pipe for Tap {
    type In = ();
    type Out = Vec<u8>;
    type Err = FrameError;

    fn call(&self, (): ()) -> impl Future<Output = Result<Vec<u8>, FrameError>> {
        async move {
            let frame = self.inner.call(()).await?;
            self.seen.send(describe_frame(&frame)).map_err(|_| FrameError::Closed)?;
            Ok(frame)
        }
    }
}

fn describe_frame(frame: &[u8]) -> String {
    let ether_type = frame.get(12..14).map_or(0, |bytes| u16::from_be_bytes([bytes[0], bytes[1]]));
    let tcp_flags = (frame.get(IPV4_PROTOCOL_OFFSET) == Some(&TCP_PROTOCOL))
        .then(|| frame.get(ETHERNET_HEADER).map(|version_ihl| ETHERNET_HEADER + usize::from(version_ihl & 0xf) * 4 + TCP_FLAGS_OFFSET))
        .flatten()
        .and_then(|offset| frame.get(offset));
    format!("ethertype={ether_type:#06x} len={} tcp_flags={tcp_flags:?}", frame.len())
}

fn dial_tapped(link: HostLink, seen: Seen) -> Upstream {
    let pipes = (
        Tap { inner: link.from_guest, seen },
        RingingTx { sink: link.to_guest, doorbell: link.doorbell },
        ZeroMint,
    );
    let local = SocketAddrV4::new(HOST_IP, HOST_PORT);
    FrameUpstream::new(Driver::new(pipes, HOST_MAC, local, HOST_INITIAL_SEQUENCE), GUEST_HTTP)
}

fn whoami_image(cache_root: &Path) -> PulledImage {
    for digest in common::WHOAMI_BLOBS {
        cache::write(cache_root, &Digest::parse(digest).expect("digest"), &common::vendored(digest)).expect("seed cache from the vendored fixture");
    }
    let no_network = async |_: Request<Bytes>| -> Result<Response<Bytes>, OciError> { panic!("the whoami image must already be cached") };
    let image = ImageRef::parse(&format!("traefik/whoami@{}", common::INDEX_DIGEST)).expect("reference");
    block_on(pull(&no_network, cache_root, &image, "linux", "amd64")).expect("the cached whoami image verifies and loads")
}

fn kernel_release(image: &[u8]) -> String {
    let header = SetupHeader::parse(image).expect("the guest kernel has a setup header");
    let version = header.version_string(image).expect("the setup header names the kernel version");
    let release = version.split(|&byte| byte == b' ').next().expect("the version string starts with the release");
    String::from_utf8(release.to_vec()).expect("the kernel release is utf-8")
}

fn boot_inputs(image: &PulledImage) -> LinuxBootInputs {
    let release_dir = Path::new(MODULES_ROOT).join(kernel_release(common::kernel()));
    let modules = load_modules(&release_dir, &["virtio_mmio", "virtio_net"]).expect("kernel modules are readable");
    let initramfs = initramfs_with_modules(image, &common::guest("oci-init"), &modules).expect("initramfs builds");
    LinuxBootInputs {
        kernel: Bytes::from_static(common::kernel()),
        initramfs: Bytes::from(initramfs),
        cmdline: format!("{CMDLINE} {}", net_device_cmdline()),
        limits: LinuxGuestLimits { watchdog_millis: PER_REQUEST_WATCHDOG_MILLIS, ..LinuxGuestLimits::default() },
    }
}

fn resident_set_kib() -> u64 {
    let status = fs::read_to_string("/proc/self/status").expect("proc status is readable");
    let line = status.lines().find(|line| line.starts_with("VmRSS:")).expect("status has a VmRSS line");
    line.split_whitespace().nth(1).and_then(|kib| kib.parse().ok()).expect("VmRSS carries a kib count")
}

fn request_bytes() -> Vec<u8> {
    let mut request = Vec::new();
    let headers = [("Host", "10.0.2.15"), ("User-Agent", "proxima"), ("Accept", "*/*"), ("Connection", "close")];
    encode_request_head("GET", "/", &headers, &mut request);
    request
}

fn drive<Value>(future: impl Future<Output = Value>, deadline: Instant) -> Option<Value> {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return Some(value);
        }
        let remaining = deadline.checked_duration_since(Instant::now())?;
        thread::park_timeout(remaining.min(POLL_CEILING));
    }
}

struct Answer {
    bytes: Vec<u8>,
    first_200: Option<Instant>,
}

async fn read_response(connection: &mut (impl AsyncReadExt + Unpin), answer: &mut Answer) -> io::Result<()> {
    let mut chunk = [0_u8; READ_CHUNK];
    loop {
        let read = connection.read(&mut chunk).await?;
        if read == 0 {
            return Ok(());
        }
        answer.bytes.extend_from_slice(&chunk[..read]);
        if answer.first_200.is_none() && answer.bytes.windows(STATUS_LINE.len()).any(|window| window == STATUS_LINE) {
            answer.first_200 = Some(Instant::now());
        }
    }
}

async fn fetch(upstream: &Upstream, request: &[u8], answer: &mut Answer) -> io::Result<()> {
    let mut connection = dial_guest_when_listening(upstream).await?;
    connection.write_all(request).await?;
    read_response(&mut connection, answer).await?;
    connection.close().await
}

async fn fetch_pumped(upstream: &Upstream, request: &[u8], answer: &mut Answer) -> io::Result<()> {
    match select(pin!(fetch(upstream, request, answer)), pin!(upstream.pump())).await {
        Either::Left((outcome, _)) => outcome,
        Either::Right((failure, _)) => Err(io::Error::other(format!("the frame pump ended before the fetch finished: {failure:?}"))),
    }
}

#[test]
fn a_resident_whoami_guest_answers_an_http_get_from_the_host() {
    global::install(EnvFilter::parse("info"));
    let scratch = tempfile::tempdir().expect("tempdir");
    let trace_path = scratch.path().join(TRACE_FILE);
    let recorder = Recorder::builder().export(Exporter::file(&trace_path).format(Formatter::Json)).expect("file sink").install().expect("recorder installs");
    let image = whoami_image(&scratch.path().join("oci-cache"));
    let inputs = boot_inputs(&image);
    let link = LinkConfig { mac: GUEST_MAC, frame_capacity: 256, host_consumer: Some(thread::current()) };
    let rss_before = resident_set_kib();
    let started = Instant::now();
    let (resident, host) = ResidentLinux::start(inputs, link).expect("the resident guest starts");
    let doorbell = host.doorbell.clone();
    let (seen, frames_seen) = mpsc::channel();
    let upstream = dial_tapped(host, seen);

    let mut answer = Answer { bytes: Vec::new(), first_200: None };
    let fetched = drive(fetch_pumped(&upstream, &request_bytes(), &mut answer), started + DEADLINE);
    let rss_after = resident_set_kib();
    let load = fs::read_to_string("/proc/loadavg").expect("loadavg is readable");
    let was_running = resident.is_running();
    let outcome = resident.stop();
    doorbell.retire();
    let console = String::from_utf8_lossy(&outcome.report.console);
    let text = String::from_utf8_lossy(&answer.bytes);

    println!("rss_before_kib={rss_before} rss_after_kib={rss_after} rss_delta_kib={}", i128::from(rss_after) - i128::from(rss_before));
    println!("loadavg={}", load.trim());
    let frames: Vec<String> = frames_seen.try_iter().collect();
    println!("frames from guest ({}): {frames:?}", frames.len());
    let first_200 = answer.first_200.unwrap_or_else(|| {
        panic!("no 200; fetch {fetched:?}; guest running {was_running}; failure {:?}; response {text:?}; console:\n{console}", outcome.failure)
    });
    println!("start_to_first_200_ms={}", first_200.duration_since(started).as_millis());
    proxima_telemetry::info!(start_to_first_200_ms = first_200.duration_since(started).as_millis() as u64, rss_before_kib = rss_before, rss_after_kib = rss_after, loadavg = %load.trim(), frames_from_guest = frames.len() as u64, "resident whoami answered");
    let drained = recorder.drain();
    let trace = fs::read_to_string(&trace_path).expect("the file sink wrote the trace");
    println!("trace record:\n{trace}");
    println!("response:\n{text}");
    println!("console:\n{console}");
    assert!(trace.contains(TRACE_MESSAGE), "timing record missing from trace after drain returned {drained} (managed drainer {}): {trace:?}", recorder.is_managed_drainer());
    assert!(trace.contains("start_to_first_200_ms"), "timing field missing from trace: {trace}");
    assert!(text.starts_with("HTTP/1.1 200"), "status line: {text}");
    assert!(text.contains("GET / HTTP/1.1"), "whoami echoes the request line: {text}");
    assert!(was_running, "the guest was still running after it answered: {:?}", outcome.failure);
}
