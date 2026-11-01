//! The resident lifetime of the `vm` upstream: the guest boots once, when the
//! factory builds the pipe, and every request is an HTTP/1.1 exchange with the
//! server inside it. The transport is `proxima_vm::kvm::host_wire::dial_guest`
//! (a `StreamUpstream` over the guest's virtio-net window) and the client is
//! [`H1ClientUpstream`], so connection reuse is the client's lock-free idle
//! queue and nothing here frames a byte.

use std::io;
use std::net::SocketAddrV4;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use futures::io::AsyncWriteExt;
use proxima_http::http1::H1ClientUpstream;
use proxima_net::frame_stack::FrameUpstream;
use proxima_primitives::pipe::SendPipe;
use proxima_vm::kernel_modules::load_modules;
use proxima_vm::frame_pipe::FrameSource;
use proxima_vm::kvm::host_wire::{RingingTx, ZeroMint, dial_guest, dial_guest_when_listening};
use proxima_vm::kvm::resident::{LinkConfig, ResidentLinux, net_device_cmdline};
use proxima_vm::linux::{self, LinuxBootInputs, LinuxGuestLimits};
use proxima_vm::oci;
use proxima_vm::x86_boot::SetupHeader;

use super::vm::{GuestSpec, LinuxRoot, VmSpec, read_guest_file};
use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::request::{Request, Response};

const GUEST_IP: std::net::Ipv4Addr = std::net::Ipv4Addr::new(10, 0, 2, 15);
const GUEST_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
const FRAME_CAPACITY: usize = 256;
const NET_MODULES: [&str; 2] = ["virtio_mmio", "virtio_net"];
const RESIDENT_FLAG: &str = "proxima.resident";
const MODULES_DISABLED: &str = "nomodules";

static GUESTS_BOOTED: AtomicU64 = AtomicU64::new(0);

/// Resident guests this process has started; one per built resident upstream,
/// however many requests they served.
#[must_use]
pub fn guests_booted() -> u64 {
    GUESTS_BOOTED.load(Ordering::Relaxed)
}

type GuestDial =FrameUpstream<FrameSource, RingingTx, ZeroMint>;

/// A guest kept up for the life of the upstream, and the HTTP client that
/// talks to it. The guest is stopped by `ResidentLinux`'s own drop, so it
/// outlives no request and no handle: `client` is declared first and drops
/// first, releasing its end of the link before the vCPU thread is joined.
pub struct ResidentVmUpstream {
    client: H1ClientUpstream<GuestDial>,
    guest: ResidentLinux,
}

impl ResidentVmUpstream {
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.guest.is_running()
    }
}

impl SendPipe for ResidentVmUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        SendPipe::call(&self.client, request)
    }
}

fn upstream_error(label: &str, what: &str, error: impl std::fmt::Display) -> ProximaError {
    ProximaError::Upstream(format!("vm `{label}`: {what}: {error}"))
}

fn kernel_release(kernel: &[u8]) -> Option<String> {
    let header = SetupHeader::parse(kernel).ok()?;
    let version = header.version_string(kernel)?;
    let release = version.split(|&byte| byte == b' ').next()?;
    String::from_utf8(release.to_vec()).ok()
}

fn resident_cmdline(cmdline: &str) -> String {
    let kept: Vec<&str> = cmdline
        .split_whitespace()
        .filter(|token| *token != MODULES_DISABLED)
        .collect();
    format!("{} {RESIDENT_FLAG} {}", kept.join(" "), net_device_cmdline())
}

fn initramfs(
    label: &str,
    root: &LinuxRoot,
    modules: &[proxima_vm::kernel_modules::KernelModule],
) -> Result<Vec<u8>, ProximaError> {
    match root {
        LinuxRoot::Init { init_path } => {
            let init = read_guest_file(label, "init", init_path)?;
            linux::initramfs_with_modules(&init, modules)
                .map_err(|error| upstream_error(label, "archive init", error))
        }
        LinuxRoot::Image { reference, init_path, cache_dir } => {
            let init = read_guest_file(label, "oci init", init_path)?;
            let image = oci::ImageRef::parse(reference)
                .map_err(|error| upstream_error(label, reference, error))?;
            let pulled = oci::pull_blocking(cache_dir, &image)
                .map_err(|error| upstream_error(label, reference, error))?;
            oci::initramfs_with_modules(&pulled, &init, modules)
                .map_err(|error| upstream_error(label, reference, error))
        }
    }
}

fn boot_inputs(label: &str, spec: &VmSpec, modules_root: &Path) -> Result<LinuxBootInputs, ProximaError> {
    let GuestSpec::Linux { kernel_path, root, cmdline, ram_mib, watchdog_millis } = &spec.guest else {
        return Err(ProximaError::Config(format!("vm `{label}`: a resident guest is a linux guest")));
    };
    let kernel = read_guest_file(label, "kernel", kernel_path)?;
    let release = kernel_release(&kernel).ok_or_else(|| {
        upstream_error(label, "parse kernel", format_args!("`{}` names no release", kernel_path.display()))
    })?;
    let modules = load_modules(&modules_root.join(&release), &NET_MODULES)
        .map_err(|error| upstream_error(label, "load net modules", error))?;
    Ok(LinuxBootInputs {
        kernel: Bytes::from(kernel),
        initramfs: Bytes::from(initramfs(label, root, &modules)?),
        cmdline: resident_cmdline(cmdline),
        limits: LinuxGuestLimits {
            ram_mib: *ram_mib,
            watchdog_millis: *watchdog_millis,
            ..LinuxGuestLimits::default()
        },
    })
}

async fn connect_when_listening(dial: &GuestDial) -> io::Result<()> {
    dial_guest_when_listening(dial).await?.close().await
}

async fn await_listening(label: &str, dial: &GuestDial, limit: Duration) -> Result<(), ProximaError> {
    match proxima_core::time::timeout(limit, connect_when_listening(dial)).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(upstream_error(label, "dial the resident guest", error)),
        Err(_) => Err(upstream_error(label, "resident guest never listened", format_args!("{limit:?}"))),
    }
}

/// Boots the guest once and returns the pipe that proxies into it. The pipe
/// is handed back only after the guest accepts a connection, so the first
/// request never races the boot.
pub(super) async fn build_resident(label: &str, spec: &VmSpec) -> Result<PipeHandle, ProximaError> {
    let inputs = boot_inputs(label, spec, &spec.modules_root)?;
    let link = LinkConfig { mac: GUEST_MAC, frame_capacity: FRAME_CAPACITY, host_consumer: None };
    let (guest, host) =
        ResidentLinux::start(inputs, link).map_err(|error| upstream_error(label, "start the resident guest", error))?;
    GUESTS_BOOTED.fetch_add(1, Ordering::Relaxed);
    let dial = dial_guest(host, SocketAddrV4::new(GUEST_IP, spec.guest_port));
    await_listening(label, &dial, Duration::from_millis(spec.connect_timeout_millis)).await?;
    let client = H1ClientUpstream::new(dial, GUEST_IP.to_string(), label);
    Ok(into_handle(ResidentVmUpstream { client, guest }))
}
