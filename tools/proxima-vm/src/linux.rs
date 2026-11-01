//! A Linux guest: a Linux bzImage booted in a KVM microVM runs one function
//! per invocation.
//!
//! The function is the `/init` of a newc initramfs built in memory. The host
//! talks to it over COM2 with an 8-byte little-endian length prefix followed by
//! that many bytes, in both directions; the kernel console is COM1. The boot
//! protocol is [`crate::x86_boot`]; the KVM run is [`crate::kvm::linux_lane`],
//! two 16550s ([`crate::uart16550`]) and the CMOS RTC ([`crate::cmos_rtc`])
//! as devices on a pipe bus.

use std::io::Write;
use std::time::Duration;

use bytes::Bytes;
use proxima_core::ProximaError;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use crate::kvm::linux_lane::{LinuxOutcome, LinuxReport, LinuxRun, Load, NetWiring, run_linux};
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use crate::kvm::trap::End;
use crate::kernel_modules::KernelModule;
use crate::x86_boot::{BOOT_PARAMS_SIZE, BootPlan, GDT_ADDRESS, PAGE_TABLES_ADDRESS, SetupHeader, fill_boot_params};

const FRAME_PREFIX: usize = 8;
const CONSOLE_TAIL: usize = 1024;
const DEVICE_MODE: u32 = 0o020_600;
const INIT_MODE: u32 = 0o100_755;
const DIRECTORY_MODE: u32 = 0o040_755;
const FILE_MODE: u32 = 0o100_644;
pub const MODULES_DIR: &str = ".proxima/modules";
const TTY_MAJOR: u32 = 4;
const COM2_MINOR: u32 = 65;
const CONSOLE_MAJOR: u32 = 5;
const CONSOLE_MINOR: u32 = 1;

/// Plain run bounds. `Default` is a 128 MiB guest with a 1 MiB console, a
/// 1 MiB output budget, and a 10 s wall-clock watchdog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinuxGuestLimits {
    pub ram_mib: u32,
    pub console_capacity: usize,
    pub output_capacity: usize,
    pub watchdog_millis: u64,
}

impl Default for LinuxGuestLimits {
    fn default() -> Self {
        Self {
            ram_mib: 128,
            console_capacity: 1 << 20,
            output_capacity: 1 << 20,
            watchdog_millis: 10_000,
        }
    }
}

/// Everything a resident boot needs, owned because it crosses to the vCPU
/// thread. `limits.watchdog_millis` is ignored: a resident run has no deadline.
#[derive(Clone, Debug)]
pub struct LinuxBootInputs {
    pub kernel: Bytes,
    pub initramfs: Bytes,
    pub cmdline: String,
    pub limits: LinuxGuestLimits,
}

/// How the guest stopped: it wrote the i8042 reset byte (`reboot=k`), or KVM
/// reported a shutdown (triple fault).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinuxEnd {
    ResetPort,
    Shutdown,
}

/// Instrument timings, all measured from the start of the run call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinuxGuestTimings {
    /// Host work before the first `KVM_RUN`: vm, irqchip, ram, loads, page tables, vcpu.
    pub setup: Duration,
    pub first_exit: Duration,
    pub init_exec: Option<Duration>,
    /// When `oci-init` was about to spawn the image entrypoint; `None` for guests that never write marker 3.
    pub entrypoint: Option<Duration>,
    pub total: Duration,
}

/// Port accesses nothing answered: reads returned 0xff, writes were dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnhandledPort {
    pub port: u16,
    pub reads: u64,
    pub writes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinuxGuestRun {
    pub output: Vec<u8>,
    pub console: Vec<u8>,
    /// Console bytes the guest wrote past `console_capacity`; zero when nothing was lost.
    pub console_dropped: usize,
    pub timings: LinuxGuestTimings,
    pub end: LinuxEnd,
    pub exit_count: u64,
    pub mmio_exit_count: u64,
    pub unhandled_ports: Vec<UnhandledPort>,
    pub unhandled_port_overflow: u64,
}

fn write_entry(
    archive: Vec<u8>,
    builder: cpio::newc::Builder,
    contents: &[u8],
) -> Result<Vec<u8>, ProximaError> {
    let size = u32::try_from(contents.len())
        .map_err(|_| ProximaError::Config(format!("initramfs entry of {} bytes exceeds newc's 4 GiB", contents.len())))?;
    let mut writer = builder.write(archive, size);
    writer.write_all(contents)?;
    Ok(writer.finish()?)
}

/// A newc initramfs holding `/init` plus the two device nodes it needs:
/// `/dev/console` (the kernel opens it for init's stdio) and `/dev/ttyS1`
/// (COM2, the invocation channel).
pub fn initramfs_from_init(init: &[u8]) -> Result<Vec<u8>, ProximaError> {
    initramfs_with_modules(init, &[])
}

/// [`initramfs_from_init`] plus `modules` under [`MODULES_DIR`], each file
/// prefixed with its load position (`00-failover.ko.zst`) so the guest loads
/// them in the order given.
pub fn initramfs_with_modules(init: &[u8], modules: &[KernelModule]) -> Result<Vec<u8>, ProximaError> {
    let mut archive = Vec::with_capacity(init.len() + 2048);
    archive = write_entry(archive, cpio::newc::Builder::new("dev").ino(1).mode(DIRECTORY_MODE).nlink(2), &[])?;
    archive = write_entry(
        archive,
        cpio::newc::Builder::new("dev/console").ino(2).mode(DEVICE_MODE).rdev_major(CONSOLE_MAJOR).rdev_minor(CONSOLE_MINOR),
        &[],
    )?;
    archive = write_entry(
        archive,
        cpio::newc::Builder::new("dev/ttyS1").ino(3).mode(DEVICE_MODE).rdev_major(TTY_MAJOR).rdev_minor(COM2_MINOR),
        &[],
    )?;
    archive = write_entry(archive, cpio::newc::Builder::new("init").ino(4).mode(INIT_MODE), init)?;
    archive = write_modules(archive, modules)?;
    Ok(cpio::newc::trailer(archive)?)
}

fn write_modules(mut archive: Vec<u8>, modules: &[KernelModule]) -> Result<Vec<u8>, ProximaError> {
    if modules.is_empty() {
        return Ok(archive);
    }
    for (inode, directory) in (5_u32..).zip([".proxima", MODULES_DIR]) {
        archive = write_entry(archive, cpio::newc::Builder::new(directory).ino(inode).mode(DIRECTORY_MODE).nlink(2), &[])?;
    }
    for (index, module) in modules.iter().enumerate() {
        let path = format!("{MODULES_DIR}/{index:02}-{}", module.file_name);
        let inode = u32::try_from(index + 7).map_err(|_| ProximaError::Config("too many kernel modules".into()))?;
        archive = write_entry(archive, cpio::newc::Builder::new(&path).ino(inode).mode(FILE_MODE), &module.bytes)?;
    }
    Ok(archive)
}

/// The invocation frame the guest reads from COM2: `u64` LE length, then bytes.
///
/// ```
/// use proxima_vm::linux::{frame, unframe};
///
/// let framed = frame(b"hello, linux");
/// assert_eq!(&framed[..8], &[12, 0, 0, 0, 0, 0, 0, 0]);
/// assert_eq!(unframe(&framed).expect("a frame decodes as itself"), b"hello, linux");
/// ```
#[must_use]
pub fn frame(payload: &[u8]) -> Vec<u8> {
    let mut framed = Vec::with_capacity(FRAME_PREFIX + payload.len());
    framed.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    framed.extend_from_slice(payload);
    framed
}

/// Decodes the guest's COM2 transmit bytes as exactly one frame.
pub fn unframe(transmitted: &[u8]) -> Result<&[u8], ProximaError> {
    let prefix: [u8; FRAME_PREFIX] = transmitted
        .get(..FRAME_PREFIX)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| {
            ProximaError::Upstream(format!(
                "guest output has no length prefix: {} bytes transmitted, need {FRAME_PREFIX}",
                transmitted.len()
            ))
        })?;
    let declared = u64::from_le_bytes(prefix);
    let body = &transmitted[FRAME_PREFIX..];
    match (body.len() as u64).cmp(&declared) {
        std::cmp::Ordering::Equal => Ok(body),
        std::cmp::Ordering::Less => Err(ProximaError::Upstream(format!(
            "guest output frame truncated: declared {declared} bytes, received {}",
            body.len()
        ))),
        std::cmp::Ordering::Greater => Err(ProximaError::Upstream(format!(
            "guest output frame has trailing bytes: declared {declared}, received {}",
            body.len()
        ))),
    }
}

fn console_tail(console: &[u8]) -> String {
    let start = console.len().saturating_sub(CONSOLE_TAIL);
    String::from_utf8_lossy(&console[start..]).into_owned()
}

/// Everything the lane needs, laid out and owned by the caller.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
struct Prepared<'bytes> {
    plan: BootPlan,
    kernel: &'bytes [u8],
    initramfs: &'bytes [u8],
    boot_params: [u8; BOOT_PARAMS_SIZE],
    cmdline: Vec<u8>,
    framed_input: Vec<u8>,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn prepare<'bytes>(
    kernel: &'bytes [u8],
    initramfs: &'bytes [u8],
    cmdline: &str,
    input: &[u8],
    limits: &LinuxGuestLimits,
) -> Result<Prepared<'bytes>, ProximaError> {
    let header = SetupHeader::parse(kernel).map_err(|error| ProximaError::Config(format!("kernel rejected: {error}")))?;
    let ram_bytes = u64::from(limits.ram_mib) << 20;
    let plan = BootPlan::new(&header, kernel.len(), ram_bytes, cmdline.len(), initramfs.len())
        .map_err(|error| ProximaError::Config(format!("boot plan rejected: {error}")))?;
    let mut boot_params = [0_u8; BOOT_PARAMS_SIZE];
    fill_boot_params(kernel, &plan, &mut boot_params)
        .map_err(|error| ProximaError::Config(format!("boot params rejected: {error}")))?;
    let mut cmdline = cmdline.as_bytes().to_vec();
    cmdline.push(0);
    Ok(Prepared { plan, kernel, initramfs, boot_params, cmdline, framed_input: frame(input) })
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn launch(prepared: &Prepared<'_>, limits: &LinuxGuestLimits, net: Option<NetWiring<'_>>) -> LinuxOutcome {
    let plan = &prepared.plan;
    let loads = [
        Load { guest_address: plan.kernel_address, data: &prepared.kernel[plan.kernel_offset..] },
        Load { guest_address: plan.initrd_address, data: prepared.initramfs },
        Load { guest_address: plan.boot_params_address, data: &prepared.boot_params[..] },
        Load { guest_address: plan.cmdline_address, data: &prepared.cmdline[..] },
    ];
    run_loaded(plan, &prepared.framed_input, limits, net, loads)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn run_loaded<Data: AsRef<[u8]>>(
    plan: &BootPlan,
    framed_input: &[u8],
    limits: &LinuxGuestLimits,
    net: Option<NetWiring<'_>>,
    loads: [Load<Data>; 4],
) -> LinuxOutcome {
    let wall_clock_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_secs()).unwrap_or(0));
    let run = LinuxRun {
        ram_bytes: plan.ram_bytes,
        entry: plan.entry,
        boot_params_address: plan.boot_params_address,
        gdt_address: GDT_ADDRESS,
        page_tables_address: PAGE_TABLES_ADDRESS,
        input: framed_input,
        console_capacity: limits.console_capacity,
        output_capacity: limits.output_capacity,
        watchdog_millis: limits.watchdog_millis,
        wall_clock_seconds,
        net,
    };
    run_linux(&run, loads)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn execute(prepared: &Prepared<'_>, limits: &LinuxGuestLimits) -> Result<LinuxGuestRun, ProximaError> {
    let outcome = launch(prepared, limits, None);
    let report = outcome.report;
    if let Some(message) = outcome.failure {
        return Err(ProximaError::Upstream(format!(
            "linux vm failed: {message}; console tail: {}",
            console_tail(&report.console)
        )));
    }
    collect(report, limits)
}

/// Boots the guest described by `inputs` with the net device wired in and runs it
/// until `net.stop` ends it or the guest ends itself. The wall-clock watchdog in
/// `inputs.limits` is not armed: it bounds one request's guest, and a resident
/// guest's lifetime is its owner's. `inputs` is consumed: the kernel and
/// initramfs live only until they are copied into guest ram.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn boot_resident(inputs: LinuxBootInputs, net: NetWiring<'_>) -> LinuxOutcome {
    let LinuxBootInputs { kernel, initramfs, cmdline, limits } = inputs;
    let limits = LinuxGuestLimits { watchdog_millis: 0, ..limits };
    let prepared = match prepare(&kernel, &initramfs, &cmdline, &[], &limits) {
        Ok(prepared) => prepared,
        Err(error) => return LinuxOutcome::failed(error.to_string()),
    };
    let Prepared { plan, boot_params, cmdline, framed_input, .. } = prepared;
    let loads = [
        Load { guest_address: plan.kernel_address, data: kernel.slice(plan.kernel_offset..) },
        Load { guest_address: plan.initrd_address, data: initramfs },
        Load { guest_address: plan.boot_params_address, data: Bytes::copy_from_slice(&boot_params) },
        Load { guest_address: plan.cmdline_address, data: Bytes::from(cmdline) },
    ];
    drop(kernel);
    run_loaded(&plan, &framed_input, &limits, Some(net), loads)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn collect(
    report: LinuxReport,
    limits: &LinuxGuestLimits,
) -> Result<LinuxGuestRun, ProximaError> {
    let console = report.console;
    let end = match report.end {
        Some(End::ResetPort) => LinuxEnd::ResetPort,
        Some(End::Shutdown) => LinuxEnd::Shutdown,
        Some(End::OutputOverflow) => {
            return Err(ProximaError::Upstream(format!(
                "guest output exceeded the {} byte budget; console tail: {}",
                limits.output_capacity,
                console_tail(&console)
            )));
        }
        Some(End::Halted | End::Stopped) | None => {
            return Err(ProximaError::Upstream("linux vm reported unknown end reason".into()));
        }
    };
    let payload = unframe(&report.output).map_err(|error| match error {
        ProximaError::Upstream(message) => ProximaError::Upstream(format!("{message}; console tail: {}", console_tail(&console))),
        other => other,
    })?;
    let timings = LinuxGuestTimings {
        setup: Duration::from_nanos(report.setup_nanos),
        first_exit: Duration::from_nanos(report.first_exit_nanos),
        init_exec: (report.init_marker_nanos != 0).then(|| Duration::from_nanos(report.init_marker_nanos)),
        entrypoint: (report.entrypoint_marker_nanos != 0).then(|| Duration::from_nanos(report.entrypoint_marker_nanos)),
        total: Duration::from_nanos(report.total_nanos),
    };
    let unhandled_ports = report
        .unhandled
        .iter()
        .map(|&(port, reads, writes)| UnhandledPort { port, reads, writes })
        .collect();
    Ok(LinuxGuestRun {
        output: payload.to_vec(),
        console,
        console_dropped: report.console_dropped,
        timings,
        end,
        exit_count: report.exit_count,
        mmio_exit_count: report.mmio_exit_count,
        unhandled_ports,
        unhandled_port_overflow: report.unhandled_overflow,
    })
}

/// Boots `kernel` (a bzImage) with `init` as the initramfs `/init`, feeds the
/// guest `input` over COM2, and returns what it framed back. The wall-clock
/// watchdog in `limits` bounds a guest that never powers off.
pub fn run_linux_guest(
    kernel: &[u8],
    init: &[u8],
    cmdline: &str,
    input: &[u8],
    limits: &LinuxGuestLimits,
) -> Result<LinuxGuestRun, ProximaError> {
    let initramfs = initramfs_from_init(init)?;
    run_linux_initramfs(kernel, &initramfs, cmdline, input, limits)
}

/// [`run_linux_guest`] with the initramfs already built by
/// [`initramfs_from_init`], for a caller that boots the same function many
/// times and builds the archive once.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub fn run_linux_initramfs(
    kernel: &[u8],
    initramfs: &[u8],
    cmdline: &str,
    input: &[u8],
    limits: &LinuxGuestLimits,
) -> Result<LinuxGuestRun, ProximaError> {
    let prepared = prepare(kernel, initramfs, cmdline, input, limits)?;
    execute(&prepared, limits)
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub fn run_linux_initramfs(
    _kernel: &[u8],
    _initramfs: &[u8],
    _cmdline: &str,
    _input: &[u8],
    _limits: &LinuxGuestLimits,
) -> Result<LinuxGuestRun, ProximaError> {
    Err(ProximaError::Config("the Linux guest needs linux/x86_64 KVM".into()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::io::Read;

    use super::*;

    #[test]
    fn frame_prefixes_the_length_as_eight_little_endian_bytes() {
        let framed = frame(b"hello, linux");

        assert_eq!(&framed[..8], &[12, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&framed[8..], b"hello, linux");
    }

    #[test]
    fn unframe_round_trips_a_frame_and_an_empty_frame() {
        assert_eq!(unframe(&frame(b"HELLO, LINUX")).unwrap(), b"HELLO, LINUX");
        assert_eq!(unframe(&frame(b"")).unwrap(), b"");
    }

    #[test]
    fn unframe_names_a_missing_prefix() {
        let error = unframe(&[1, 2, 3]).expect_err("3 bytes cannot hold a prefix");

        assert_eq!(error.to_string(), "upstream error: guest output has no length prefix: 3 bytes transmitted, need 8");
    }

    #[test]
    fn unframe_names_a_truncated_body() {
        let mut transmitted = frame(b"abcdef");
        transmitted.truncate(8 + 4);

        let error = unframe(&transmitted).expect_err("declared 6, received 4");

        assert_eq!(error.to_string(), "upstream error: guest output frame truncated: declared 6 bytes, received 4");
    }

    #[test]
    fn unframe_names_trailing_bytes() {
        let mut transmitted = frame(b"abc");
        transmitted.extend_from_slice(b"zz");

        let error = unframe(&transmitted).expect_err("declared 3, received 5");

        assert_eq!(error.to_string(), "upstream error: guest output frame has trailing bytes: declared 3, received 5");
    }

    #[test]
    fn initramfs_is_newc_with_init_console_and_invocation_node() {
        let init = b"\x7fELF-pretend-static-binary";

        let archive = initramfs_from_init(init).unwrap();

        assert_eq!(&archive[..6], b"070701");
        assert_eq!(archive.len() % 4, 0);
        let mut names = Vec::new();
        let mut cursor = std::io::Cursor::new(&archive);
        loop {
            let mut reader = cpio::newc::Reader::new(&mut cursor).unwrap();
            if reader.entry().is_trailer() {
                break;
            }
            names.push(reader.entry().name().to_string());
            if reader.entry().name() == "init" {
                let mut contents = Vec::new();
                reader.read_to_end(&mut contents).unwrap();
                assert_eq!(contents, init);
                assert_eq!(reader.entry().mode(), INIT_MODE);
            }
            reader.finish().unwrap();
        }
        assert_eq!(names, ["dev", "dev/console", "dev/ttyS1", "init"]);
    }
}
