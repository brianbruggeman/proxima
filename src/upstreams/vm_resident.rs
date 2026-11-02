//! The resident lifetime of the `vm` upstream: the guest boots once, when the
//! factory builds the pipe, and every request is an HTTP/1.1 exchange with the
//! server inside it. The transport is `proxima_vm::kvm::host_wire::dial_guest`
//! (a `StreamUpstream` over the guest's virtio-net window) and the client is
//! [`H1ClientUpstream`], so connection reuse is the client's lock-free idle
//! queue and nothing here frames a byte.

use core::pin::pin;
use std::cell::Cell;
use std::collections::BTreeSet;
use std::io;
use std::net::SocketAddrV4;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use bytes::Bytes;
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::future::{Either, select};
use futures::io::AsyncWriteExt;
use proxima_http::http1::H1ClientUpstream;
use proxima_core::time::{self, Instant};
use proxima_net::frame_stack::FrameUpstream;
use proxima_core::signal::{Signal, SignalGuard};
use proxima_primitives::pipe::clock::TimeClock;
use proxima_primitives::pipe::resilience::Retry;
use proxima_primitives::pipe::{Backoff, Jitter, Pipe, RetryController, RetryRules, Retryable, SendPipe};
use proxima_primitives::sync::watch;
use proxima_vm::kernel_modules::load_modules;
use proxima_vm::frame_pipe::FrameSource;
use proxima_vm::kvm::host_wire::{RingingTx, ZeroMint, dial_guest, dial_guest_when_listening};
use proxima_vm::kvm::resident::{LinkConfig, ResidentLinux, net_device_cmdline};
use proxima_vm::linux::{self, LinuxBootInputs, LinuxGuestLimits};
use proxima_vm::oci;
use proxima_vm::x86_boot::SetupHeader;

use super::process::RestartPolicy;
use super::vm::{GuestSpec, LinuxRoot, VmSpec, read_guest_file};
use crate::SwappablePipe;
use crate::control_plane::{PipeState, PipeStatus};
use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::request::{Request, Response};

const GUEST_IP: std::net::Ipv4Addr = std::net::Ipv4Addr::new(10, 0, 2, 15);
const GUEST_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
const FRAME_CAPACITY: usize = 256;
const NET_MODULES: [&str; 2] = ["virtio_mmio", "virtio_net"];
const RESIDENT_FLAG: &str = "proxima.resident";
const MODULES_DISABLED: &str = "nomodules";
const PUMP_THREAD: &str = "proxima-vm-resident-pump";

const SUPERVISOR_THREAD: &str = "proxima-vm-resident-supervisor";
// the kubernetes crash-loop ceiling, so a restart storm backs off like a pod's
const RESTART_BACKOFF_MAX: Duration = Duration::from_secs(300);
const RESTART_BACKOFF_FACTOR: u32 = 2;

static GUESTS_BOOTED: AtomicU64 = AtomicU64::new(0);
static GUEST_RESTARTS: AtomicU64 = AtomicU64::new(0);

/// Resident guests this process has started; one per built resident upstream,
/// however many requests they served.
#[must_use]
pub fn guests_booted() -> u64 {
    GUESTS_BOOTED.load(Ordering::Relaxed)
}

type GuestDial = FrameUpstream<FrameSource, RingingTx, ZeroMint>;

/// Resident guests booted again after one ended; read beside [`guests_booted`].
#[must_use]
pub fn guest_restarts() -> u64 {
    GUEST_RESTARTS.load(Ordering::Relaxed)
}

/// The swappable HTTP client in front of whichever guest is current, the status
/// the supervisor publishes, and the guard that ends the supervisor (and with it
/// the guest) when the upstream drops.
pub struct ResidentVmUpstream {
    pipe: Arc<SwappablePipe>,
    status: watch::Receiver<PipeStatus>,
    _shutdown: SignalGuard,
}

impl ResidentVmUpstream {
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.status.borrow().state == PipeState::Running
    }

    #[must_use]
    pub fn status(&self) -> watch::Receiver<PipeStatus> {
        self.status.clone()
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
        SendPipe::call(&*self.pipe, request)
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

struct Timeline {
    began: Instant,
    last: Instant,
}

impl Timeline {
    fn begin() -> Self {
        let began = time::now();
        Self { began, last: began }
    }

    fn lap(&mut self) -> (u64, u64) {
        let now = time::now();
        let elapsed_ms = millis(now.saturating_duration_since(self.began));
        let phase_ms = millis(now.saturating_duration_since(self.last));
        self.last = now;
        (elapsed_ms, phase_ms)
    }
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn initramfs(
    label: &str,
    root: &LinuxRoot,
    modules: &[proxima_vm::kernel_modules::KernelModule],
    timeline: &mut Timeline,
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
            let (elapsed_ms, phase_ms) = timeline.lap();
            proxima_telemetry::info!(
                label = %label,
                phase = "image_pulled",
                reference = %reference,
                elapsed_ms,
                phase_ms,
                "resident guest image pulled and verified before the initramfs is built"
            );
            oci::initramfs_with_modules(&pulled, &init, modules)
                .map_err(|error| upstream_error(label, reference, error))
        }
    }
}

fn boot_inputs(
    label: &str,
    spec: &VmSpec,
    modules_root: &Path,
    timeline: &mut Timeline,
) -> Result<LinuxBootInputs, ProximaError> {
    let GuestSpec::Linux { kernel_path, root, cmdline, ram_mib, watchdog_millis } = &spec.guest else {
        return Err(ProximaError::Config(format!("vm `{label}`: a resident guest is a linux guest")));
    };
    let kernel = read_guest_file(label, "kernel", kernel_path)?;
    let (elapsed_ms, phase_ms) = timeline.lap();
    proxima_telemetry::info!(
        label = %label,
        phase = "kernel_read",
        kernel_bytes = kernel.len(),
        elapsed_ms,
        phase_ms,
        "resident guest kernel read from disk before modules are matched to its release"
    );
    let release = kernel_release(&kernel).ok_or_else(|| {
        upstream_error(label, "parse kernel", format_args!("`{}` names no release", kernel_path.display()))
    })?;
    let modules = load_modules(&modules_root.join(&release), &NET_MODULES)
        .map_err(|error| upstream_error(label, "load net modules", error))?;
    let (elapsed_ms, phase_ms) = timeline.lap();
    proxima_telemetry::info!(
        label = %label,
        phase = "modules_loaded",
        release = %release,
        modules = modules.len(),
        elapsed_ms,
        phase_ms,
        "resident guest net modules loaded for the kernel release"
    );
    let initramfs = initramfs(label, root, &modules, timeline)?;
    let (elapsed_ms, phase_ms) = timeline.lap();
    proxima_telemetry::info!(
        label = %label,
        phase = "initramfs_built",
        initramfs_bytes = initramfs.len(),
        elapsed_ms,
        phase_ms,
        "resident guest initramfs built, nothing left to read before the vm starts"
    );
    Ok(LinuxBootInputs {
        kernel: Bytes::from(kernel),
        initramfs: Bytes::from(initramfs),
        cmdline: resident_cmdline(cmdline),
        limits: LinuxGuestLimits {
            ram_mib: *ram_mib,
            watchdog_millis: *watchdog_millis,
            ..LinuxGuestLimits::default()
        },
    })
}

async fn connect_when_listening(dial: &GuestDial) -> io::Result<u32> {
    let (mut connection, dial_attempts) = dial_guest_when_listening(dial).await?;
    connection.close().await?;
    Ok(dial_attempts)
}

async fn pump_until_stopped(label: String, dial: GuestDial, mut stop: oneshot::Receiver<()>, ended: Signal) {
    let failure = {
        let pump = pin!(dial.pump());
        match select(pump, &mut stop).await {
            Either::Left((failure, _)) => failure,
            Either::Right(_) => return,
        }
    };
    if matches!(stop.try_recv(), Ok(None)) {
        proxima_telemetry::error!(
            label = %label,
            failure = ?failure,
            "resident guest frame pump ended while the upstream was live; its connections are dead"
        );
    }
    ended.fire();
}

fn spawn_pump(label: &str, dial: GuestDial, ended: Signal) -> Result<oneshot::Sender<()>, ProximaError> {
    let (stop_sender, stop_receiver) = oneshot::channel();
    let pump_label = label.to_owned();
    thread::Builder::new()
        .name(PUMP_THREAD.to_owned())
        .spawn(move || block_on(pump_until_stopped(pump_label, dial, stop_receiver, ended)))
        .map_err(|error| upstream_error(label, "spawn the frame pump", error))?;
    Ok(stop_sender)
}

async fn await_listening(label: &str, dial: &GuestDial, limit: Duration) -> Result<u32, ProximaError> {
    match time::timeout(limit, connect_when_listening(dial)).await {
        Ok(Ok(dial_attempts)) => Ok(dial_attempts),
        Ok(Err(error)) => Err(upstream_error(label, "dial the resident guest", error)),
        Err(_) => Err(upstream_error(label, "resident guest never listened", format_args!("{limit:?}"))),
    }
}


struct Booted {
    guest: ResidentLinux,
    dial: GuestDial,
    pump_stop: oneshot::Sender<()>,
    pump_ended: Signal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GuestEnd {
    Ended,
    Stopped,
}

impl Retryable for GuestEnd {
    fn retry_status(&self) -> Option<u16> {
        None
    }

    fn is_success(&self) -> bool {
        true
    }
}

fn guest_client(label: &str, dial: &GuestDial) -> PipeHandle {
    into_handle(H1ClientUpstream::new(dial.clone(), GUEST_IP.to_string(), label))
}

async fn boot_guest(label: &str, spec: &VmSpec) -> Result<Booted, ProximaError> {
    let mut timeline = Timeline::begin();
    let inputs = boot_inputs(label, spec, &spec.modules_root, &mut timeline)?;
    let link = LinkConfig { mac: GUEST_MAC, frame_capacity: FRAME_CAPACITY, host_consumer: None };
    let (guest, host) =
        ResidentLinux::start(inputs, link).map_err(|error| upstream_error(label, "start the resident guest", error))?;
    GUESTS_BOOTED.fetch_add(1, Ordering::Relaxed);
    let (elapsed_ms, phase_ms) = timeline.lap();
    proxima_telemetry::info!(
        label = %label,
        phase = "guest_started",
        elapsed_ms,
        phase_ms,
        "resident guest vcpu thread started, the host now waits for its server to listen"
    );
    let dial = dial_guest(host, SocketAddrV4::new(GUEST_IP, spec.guest_port));
    let pump_ended = Signal::new();
    let pump_stop = spawn_pump(label, dial.clone(), pump_ended.clone())?;
    let (elapsed_ms, phase_ms) = timeline.lap();
    proxima_telemetry::info!(
        label = %label,
        phase = "first_dial",
        elapsed_ms,
        phase_ms,
        "frame pump spawned, the first dial is attempted next"
    );
    let dial_attempts = await_listening(label, &dial, Duration::from_millis(spec.connect_timeout_millis)).await?;
    let (elapsed_ms, phase_ms) = timeline.lap();
    proxima_telemetry::info!(
        label = %label,
        phase = "listening",
        dial_attempts,
        elapsed_ms,
        phase_ms,
        "resident guest accepted a connection, the upstream is ready for requests"
    );
    Ok(Booted { guest, dial, pump_stop, pump_ended })
}

struct ResidentRun {
    label: String,
    spec: VmSpec,
    pipe: Arc<SwappablePipe>,
    shutdown: Signal,
    status: watch::Sender<PipeStatus>,
    first: Cell<Option<Booted>>,
    restarts: Cell<u64>,
}

impl ResidentRun {
    fn publish(&self, state: PipeState, message: Option<String>) {
        let status = PipeStatus {
            name: self.label.clone(),
            state,
            uptime_ms: None,
            restart_count: self.restarts.get(),
            last_message: message,
        };
        // no receiver left means the upstream was dropped, and nobody reads the status
        let _ = self.status.send(status);
    }

    async fn next_guest(&self) -> Result<Booted, ProximaError> {
        if let Some(first) = self.first.take() {
            return Ok(first);
        }
        let booted = boot_guest(&self.label, &self.spec).await.inspect_err(|error| {
            self.publish(PipeState::Failed, Some(error.to_string()));
        })?;
        self.pipe.swap(guest_client(&self.label, &booted.dial));
        self.restarts.set(self.restarts.get() + 1);
        GUEST_RESTARTS.fetch_add(1, Ordering::Relaxed);
        Ok(booted)
    }

    fn end_of(&self, booted: Booted) -> Result<GuestEnd, ProximaError> {
        let Booted { guest, pump_stop, .. } = booted;
        let outcome = guest.stop();
        drop(pump_stop);
        match outcome.failure {
            Some(failure) => {
                self.publish(PipeState::Failed, Some(failure.clone()));
                Err(upstream_error(&self.label, "resident guest ended", failure))
            }
            None => {
                self.publish(PipeState::Stopped, Some("resident guest ended".to_owned()));
                match self.spec.restart {
                    RestartPolicy::Always => {
                        Err(upstream_error(&self.label, "resident guest ended", "a clean end under restart always"))
                    }
                    RestartPolicy::OnFailure | RestartPolicy::Never => Ok(GuestEnd::Ended),
                }
            }
        }
    }

    async fn run(&self) -> Result<GuestEnd, ProximaError> {
        if self.shutdown.is_fired() {
            return Ok(GuestEnd::Stopped);
        }
        let booted = self.next_guest().await?;
        self.publish(PipeState::Running, None);
        let ended = booted.pump_ended.fired();
        match select(pin!(ended), pin!(self.shutdown.fired())).await {
            Either::Left(_) => self.end_of(booted),
            Either::Right(_) => Ok(GuestEnd::Stopped),
        }
    }
}

impl Pipe for ResidentRun {
    type In = ();
    type Out = GuestEnd;
    type Err = ProximaError;

    fn call(&self, _input: ()) -> impl Future<Output = Result<GuestEnd, ProximaError>> {
        self.run()
    }
}

fn restart_controller(spec: &VmSpec) -> RetryController {
    let retry_on_error = !matches!(spec.restart, RestartPolicy::Never);
    RetryController {
        rules: RetryRules { retry_on_status: BTreeSet::new(), retry_on_error, idempotent_only: false },
        backoff: Backoff::Exponential {
            initial: Duration::from_millis(spec.restart_delay_ms),
            factor: RESTART_BACKOFF_FACTOR,
            max: RESTART_BACKOFF_MAX,
        },
        jitter: Jitter::None,
        max_attempts: match spec.max_restart_attempts {
            0 => u32::MAX,
            restarts => restarts.saturating_add(1),
        },
        deadline: None,
    }
}

fn spawn_supervisor(run: ResidentRun, controller: RetryController) -> Result<(), ProximaError> {
    let label = run.label.clone();
    let thread_label = label.clone();
    thread::Builder::new()
        .name(SUPERVISOR_THREAD.to_owned())
        .spawn(move || {
            let supervised = Retry::new(run, controller, TimeClock, 0);
            match block_on(Pipe::call(&supervised, ())) {
                Ok(end) => proxima_telemetry::info!(
                    label = %thread_label,
                    end = ?end,
                    "resident guest supervisor finished, the restart policy declines another boot"
                ),
                Err(error) => proxima_telemetry::error!(
                    label = %thread_label,
                    error = %error,
                    "resident guest supervisor gave up; the upstream serves no more requests"
                ),
            }
        })
        .map(drop)
        .map_err(|error| upstream_error(&label, "spawn the guest supervisor", error))
}

/// Boots the guest and returns the pipe that proxies into it, served by a
/// supervisor that boots a new guest when this one ends and the `RestartPolicy`
/// allows. The pipe is handed back only after the first guest accepts a
/// connection, so the first request never races the boot. Each restart rebuilds
/// its boot inputs from the spec through the same path as the first boot, so
/// no 25 MB image is held between boots.
pub(super) async fn build_resident(label: &str, spec: &VmSpec) -> Result<PipeHandle, ProximaError> {
    build_upstream(label, spec).await.map(into_handle)
}

/// [`build_resident`] before the pipe is erased, for a caller that reads the
/// published [`ResidentVmUpstream::status`].
pub async fn build_upstream(label: &str, spec: &VmSpec) -> Result<ResidentVmUpstream, ProximaError> {
    let first = boot_guest(label, spec).await?;
    let pipe = Arc::new(SwappablePipe::new(guest_client(label, &first.dial)));
    let (status_sender, status) = watch::channel(PipeStatus {
        name: label.to_owned(),
        state: PipeState::Running,
        uptime_ms: None,
        restart_count: 0,
        last_message: None,
    });
    let shutdown = Signal::new();
    let run = ResidentRun {
        label: label.to_owned(),
        spec: spec.clone(),
        pipe: pipe.clone(),
        shutdown: shutdown.clone(),
        status: status_sender,
        first: Cell::new(Some(first)),
        restarts: Cell::new(0),
    };
    spawn_supervisor(run, restart_controller(spec))?;
    Ok(ResidentVmUpstream { pipe, status, _shutdown: shutdown.guard() })
}
