//! A guest that boots once and serves many connections: the vCPU runs on its
//! own thread for as long as the guest lives, a virtio-net window carries
//! Ethernet frames to and from the host through two frame pipes, and a
//! [`Doorbell`] wakes the vCPU when the host has put a frame in the pipe.
//!
//! Nothing here depends on what runs in the guest: a Linux guest and a
//! purpose-built guest with its own TCP stack plug into the same [`HostLink`].

use core::cell::RefCell;
use core::future::Future;
use std::panic::{self, AssertUnwindSafe};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle, Thread};

use proxima_primitives::pipe::Pipe;

use super::devices::{Transport, VirtioWindow, declined};
use super::doorbell::Doorbell;
use super::linux_lane::{LinuxOutcome, NetWiring};
use super::trap::{End, Reply, RunError, Trap, VcpuExit};
use crate::frame_pipe::{FrameSink, FrameSource, frame_channel, frame_channel_waking};
use crate::linux::{LinuxBootInputs, boot_resident, console_tail};
use crate::virtio_net::NetTransport;

pub const NET_WINDOW_BASE: u64 = 0xd000_0000;
pub const NET_WINDOW_SIZE: u64 = 0x1000;
pub const NET_IRQ: u32 = 5;

/// The kernel command-line argument that makes a Linux guest probe the net
/// window: `virtio_mmio.device=<size>@<base>:<irq>`.
#[must_use]
pub fn net_device_cmdline() -> String {
    format!("virtio_mmio.device=4K@{NET_WINDOW_BASE:#x}:{NET_IRQ}")
}

/// Claims the exit a doorbell causes. The vCPU lowers `immediate_exit`, then
/// either ends the run (stop requested) or delivers whatever the host queued.
pub struct Wake<'run> {
    window: &'run VirtioWindow<'run>,
    doorbell: &'run Doorbell,
    stop: &'run AtomicBool,
}

impl<'run> Wake<'run> {
    #[must_use]
    pub const fn new(window: &'run VirtioWindow<'run>, doorbell: &'run Doorbell, stop: &'run AtomicBool) -> Self {
        Self { window, doorbell, stop }
    }

    fn wake(&self) -> Result<Reply, RunError> {
        self.doorbell.clear();
        if self.stop.load(Ordering::Acquire) {
            return Ok(Reply::End(End::Stopped));
        }
        let irq = self.window.service_rx()?;
        Ok(Reply::Io { value: None, irq })
    }
}

impl Pipe for Wake<'_> {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match trap {
            Trap::Exit(VcpuExit::Interrupted) => core::future::ready(self.wake()),
            other => declined(other),
        }
    }
}

/// How the host's end of the guest's network is made.
#[derive(Clone, Debug)]
pub struct LinkConfig {
    /// The guest NIC's MAC, e.g. `02:00:00:00:00:01`.
    pub mac: [u8; 6],
    /// Frames each direction's pipe holds before refusing more.
    pub frame_capacity: usize,
    /// A thread to unpark whenever the guest sends a frame or the guest ends,
    /// so the host's consumer can park between frames.
    pub host_consumer: Option<Thread>,
}

/// The host's end of the guest's network, and the means to wake the guest.
pub struct HostLink {
    /// Host to guest frames; ring `doorbell` after a push so the vCPU delivers it.
    pub to_guest: FrameSink,
    /// Guest to host frames; reads `FrameError::Closed` once the guest has ended.
    pub from_guest: FrameSource,
    pub doorbell: Arc<Doorbell>,
}

#[derive(Debug, thiserror::Error)]
pub enum ResidentError {
    #[error("could not spawn the vcpu thread: {0}")]
    Spawn(#[source] io::Error),
    #[error("the resident guest failed before it was running: {0}")]
    Boot(String),
}

/// A guest running on a dedicated vCPU thread until it ends or is stopped.
pub struct ResidentLinux {
    stop: Arc<AtomicBool>,
    doorbell: Arc<Doorbell>,
    thread: Option<JoinHandle<LinuxOutcome>>,
}

fn run_on_vcpu_thread(
    boot: LinuxBootInputs,
    transport: Transport,
    guest_to_host: FrameSink,
    stop: Arc<AtomicBool>,
    doorbell_out: mpsc::SyncSender<Arc<Doorbell>>,
) -> LinuxOutcome {
    let cell = RefCell::new(Some(transport));
    let net = NetWiring { transport: &cell, stop: &stop, doorbell_out: &doorbell_out };
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| boot_resident(boot, net)))
        .unwrap_or_else(|_| LinuxOutcome::failed("the vcpu thread panicked".to_owned()));
    if !stop.load(Ordering::Acquire) {
        proxima_telemetry::error!(
            failure = ?outcome.failure,
            end = ?outcome.report.end,
            exits = outcome.report.exit_count,
            console_dropped = outcome.report.console_dropped,
            console_tail = ?console_tail(&outcome.report.console),
            "resident guest ended on its own; its link is closed for good"
        );
    }
    guest_to_host.close();
    outcome
}

impl ResidentLinux {
    /// Boots the guest on its own thread and returns once the vCPU exists and
    /// its doorbell is bound, or with the reason it could not get that far.
    pub fn start(boot: LinuxBootInputs, link: LinkConfig) -> Result<(Self, HostLink), ResidentError> {
        let (to_guest, ingress) = frame_channel(link.frame_capacity);
        let (guest_to_host, from_guest) = match link.host_consumer {
            Some(consumer) => frame_channel_waking(link.frame_capacity, consumer),
            None => frame_channel(link.frame_capacity),
        };
        let transport = Transport::Net(NetTransport::with_pipes(link.mac, ingress, guest_to_host.clone()));
        let stop = Arc::new(AtomicBool::new(false));
        let (doorbell_out, doorbell_in) = mpsc::sync_channel(1);
        let thread_stop = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("proxima-vm-resident-vcpu".to_owned())
            .spawn(move || run_on_vcpu_thread(boot, transport, guest_to_host, thread_stop, doorbell_out))
            .map_err(ResidentError::Spawn)?;
        match doorbell_in.recv() {
            Ok(doorbell) => {
                let resident = Self { stop, doorbell: Arc::clone(&doorbell), thread: Some(thread) };
                Ok((resident, HostLink { to_guest, from_guest, doorbell }))
            }
            Err(RecvError) => Err(ResidentError::Boot(Self::failure_of(thread))),
        }
    }

    fn failure_of(thread: JoinHandle<LinuxOutcome>) -> String {
        match thread.join() {
            Ok(outcome) => outcome.failure.unwrap_or_else(|| "the vcpu thread ended before binding its doorbell".to_owned()),
            Err(_) => "the vcpu thread panicked before binding its doorbell".to_owned(),
        }
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.thread.as_ref().is_some_and(|thread| !thread.is_finished())
    }

    fn shut_down(&mut self) -> Option<LinuxOutcome> {
        let thread = self.thread.take()?;
        self.stop.store(true, Ordering::Release);
        self.doorbell.ring();
        Some(thread.join().unwrap_or_else(|_| LinuxOutcome::failed("the vcpu thread panicked".to_owned())))
    }

    /// Ends the guest and returns what it produced. The guest-to-host pipe
    /// reads `FrameError::Closed` afterwards.
    #[must_use]
    pub fn stop(mut self) -> LinuxOutcome {
        self.shut_down().unwrap_or_else(|| LinuxOutcome::failed("the resident guest was already stopped".to_owned()))
    }
}

impl Drop for ResidentLinux {
    fn drop(&mut self) {
        let _ = self.shut_down();
    }
}

