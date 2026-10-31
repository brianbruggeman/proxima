//! The KVM lane for a Linux bzImage: in-kernel irqchip and PIT, two 16550s
//! (COM1 the console, COM2 the invocation channel), the CMOS RTC, and the reset
//! and marker ports, each a device on one bus.

use core::cell::{Cell, RefCell};

use proxima_primitives::pipe::AndThen;
use proxima_primitives::pipe::fan_in::{FanIn, Select};

use super::describe::describe_failure;
use super::devices::Lifecycle;
use super::drive::drive;
use super::error::{KvmError, KvmResult};
use super::exit_source::TrapSource;
use super::memory::{GuestRam, HostMapping};
use super::pipes::{ExitLedger, FirstClaim, MAX_TOTAL_EXITS, Meter, Resume};
use super::serial::{
    COM1_BASE, COM1_IRQ, COM2_BASE, COM2_IRQ, InitMarker, PortTally, ResetPort, RtcPorts, SerialPort, Transmit,
    Unhandled,
};
use super::session::{Session, release_off_the_return_path};
use super::slots::{SlotOp, SlotRegion};
use super::trap::End;
use super::watchdog::Watchdog;
use crate::page_table::{IdentitySpan, X86_TABLE_PAGE_SIZE, write_x86_64_identity_2m};
use crate::x86_boot::{GDT_SIZE, write_boot_gdt};

const GIGABYTE: u64 = 1024 * 1024 * 1024;
const MAX_RAM: u64 = 3 * GIGABYTE;
const IDENTITY_MAP_ADDRESS: u64 = 0xfffb_c000;
const TSS_ADDRESS: usize = 0xfffb_d000;
const HUGE_ENTRY_FLAGS: u64 = 0x83;

#[derive(Clone, Copy, Debug)]
pub struct Load<'a> {
    pub guest_address: u64,
    pub data: &'a [u8],
}

pub struct LinuxRun<'a> {
    pub ram_bytes: u64,
    pub entry: u64,
    pub boot_params_address: u64,
    pub gdt_address: u64,
    pub page_tables_address: u64,
    pub loads: &'a [Load<'a>],
    pub input: &'a [u8],
    pub console_capacity: usize,
    pub output_capacity: usize,
    pub watchdog_millis: u64,
    pub wall_clock_seconds: i64,
}

#[derive(Debug)]
pub struct LinuxReport {
    pub console: Vec<u8>,
    pub console_dropped: usize,
    pub output: Vec<u8>,
    pub setup_nanos: u64,
    pub first_exit_nanos: u64,
    pub init_marker_nanos: u64,
    pub total_nanos: u64,
    pub exit_count: u64,
    pub mmio_exit_count: u64,
    pub end: Option<End>,
    pub unhandled: Vec<(u16, u64, u64)>,
    pub unhandled_overflow: u64,
}

#[derive(Debug)]
pub struct LinuxOutcome {
    pub report: LinuxReport,
    pub failure: Option<String>,
}

struct Prepared {
    session: Session,
    ram: GuestRam,
}

fn fits(ram_bytes: u64, address: u64, length: u64) -> bool {
    address <= ram_bytes && length <= ram_bytes - address
}

fn lay_out_boot_memory(run: &LinuxRun<'_>, ram: &mut HostMapping) -> KvmResult<()> {
    for load in run.loads {
        let length = load.data.len() as u64;
        if !fits(run.ram_bytes, load.guest_address, length) {
            return Err(KvmError::guest("linux load exceeds guest ram"));
        }
        let start = load.guest_address as usize;
        ram.bytes_mut()[start..start + load.data.len()].copy_from_slice(load.data);
    }
    let directories = run.ram_bytes.div_ceil(GIGABYTE);
    let table_bytes = (2 + directories) * X86_TABLE_PAGE_SIZE;
    if !fits(run.ram_bytes, run.page_tables_address, table_bytes) {
        return Err(KvmError::guest("linux page tables exceed guest ram"));
    }
    let tables = run.page_tables_address as usize;
    let span = IdentitySpan { start: 0, end: directories * GIGABYTE, flags: HUGE_ENTRY_FLAGS };
    write_x86_64_identity_2m(&mut ram.bytes_mut()[tables..tables + table_bytes as usize], run.page_tables_address, &[span])
        .map_err(|error| KvmError::guest(error.to_string()))?;
    if !fits(run.ram_bytes, run.gdt_address, GDT_SIZE as u64) {
        return Err(KvmError::guest("linux gdt exceeds guest ram"));
    }
    let gdt = run.gdt_address as usize;
    write_boot_gdt(&mut ram.bytes_mut()[gdt..gdt + GDT_SIZE]).map_err(|error| KvmError::guest(error.to_string()))
}

fn prepare(run: &LinuxRun<'_>) -> KvmResult<Prepared> {
    if run.ram_bytes == 0 || run.ram_bytes > MAX_RAM || !run.ram_bytes.is_multiple_of(X86_TABLE_PAGE_SIZE) {
        return Err(KvmError::guest("linux ram must be a nonzero page multiple of at most 3 GiB"));
    }
    let mut session = Session::open()?;
    session.machine().create_interrupt_devices(IDENTITY_MAP_ADDRESS, TSS_ADDRESS)?;
    let mut ram = HostMapping::anonymous(run.ram_bytes as usize, false)?;
    let slot = SlotOp::Submit(SlotRegion {
        slot: 0,
        flags: 0,
        guest_address: 0,
        size: run.ram_bytes,
        host: ram.as_ptr() as u64,
    });
    session.machine().apply(&[slot], "KVM_SET_USER_MEMORY_REGION linux ram")?;
    session.create_vcpu()?;
    lay_out_boot_memory(run, &mut ram)?;
    let guest_ram = GuestRam::of(&ram);
    session.attach_mapping(ram);
    session.vcpu()?.start_linux_boot(run.entry, run.boot_params_address, run.gdt_address, run.page_tables_address)?;
    Ok(Prepared { session, ram: guest_ram })
}

/// Boots the guest and runs it until it resets, shuts down, overflows its
/// output budget, or is stopped. The report carries what the guest produced
/// either way.
pub fn run_linux(run: &LinuxRun<'_>) -> LinuxOutcome {
    let ledger = ExitLedger::begin();
    let console = RefCell::new(Transmit::with_capacity(run.console_capacity));
    let output = RefCell::new(Transmit::with_capacity(run.output_capacity));
    let tally = RefCell::new(PortTally::default());
    let marker_stamp = Cell::new(None);
    let mut setup_nanos = 0;
    let mut total_nanos = 0;
    let (failure, end) = match prepare(run) {
        Ok(prepared) => {
            setup_nanos = ledger.elapsed_nanos();
            let outcome = drive_prepared(&prepared, run, &ledger, [&console, &output], &tally, &marker_stamp);
            total_nanos = ledger.elapsed_nanos();
            release_off_the_return_path(prepared.session);
            outcome
        }
        Err(error) => (Some(error.to_string()), None),
    };
    if total_nanos == 0 {
        total_nanos = ledger.elapsed_nanos();
    }
    let console = console.into_inner();
    let tally = tally.into_inner();
    LinuxOutcome {
        report: LinuxReport {
            console_dropped: console.dropped(),
            console: console.into_bytes(),
            output: output.into_inner().into_bytes(),
            setup_nanos,
            first_exit_nanos: ledger.first_exit_nanos(),
            init_marker_nanos: marker_stamp.get().unwrap_or(0),
            total_nanos,
            exit_count: ledger.exits(),
            mmio_exit_count: ledger.mmio_exits(),
            end,
            unhandled: tally.entries,
            unhandled_overflow: tally.overflow,
        },
        failure,
    }
}

fn drive_prepared(
    prepared: &Prepared,
    run: &LinuxRun<'_>,
    ledger: &ExitLedger,
    [console, output]: [&RefCell<Transmit>; 2],
    tally: &RefCell<PortTally>,
    marker_stamp: &Cell<Option<u64>>,
) -> (Option<String>, Option<End>) {
    let Ok(vcpu) = prepared.session.vcpu() else {
        return (Some("the vcpu is created before the run starts".to_owned()), None);
    };
    let watchdog = match Watchdog::arm(vcpu.immediate_exit_flag(), run.watchdog_millis) {
        Ok(armed) => armed,
        Err(error) => return (Some(error.to_string()), None),
    };
    let com1 = SerialPort::new(COM1_BASE, COM1_IRQ, &[], console, false);
    let com2 = SerialPort::new(COM2_BASE, COM2_IRQ, run.input, output, true);
    let bus = FirstClaim::new(
        Lifecycle::new(true),
        FirstClaim::new(
            com1,
            FirstClaim::new(
                &com2,
                FirstClaim::new(
                    RtcPorts::at(run.wall_clock_seconds),
                    FirstClaim::new(
                        ResetPort,
                        FirstClaim::new(InitMarker::new(&com2, ledger, marker_stamp), Unhandled::new(tally)),
                    ),
                ),
            ),
        ),
    );
    let merge = FanIn::new(
        [TrapSource::Watchdog(watchdog.as_ref()), TrapSource::Exit { vcpu, hypercall_memory: None }],
        Select::Fifo,
    );
    let meter = Meter::new(ledger, MAX_TOTAL_EXITS, "exceeded total exit budget without the guest ending");
    let step = AndThen::new(
        AndThen::new(AndThen::new(merge, meter), bus),
        Resume::new(prepared.session.machine(), vcpu, prepared.ram),
    );
    match drive(&step) {
        Ok(end) => (None, Some(end)),
        Err(error) => (Some(describe_failure(&error, Some(vcpu))), None),
    }
}
