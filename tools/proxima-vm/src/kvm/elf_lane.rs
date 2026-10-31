//! The KVM lane for an ELF guest: segments mapped at their real permissions
//! under a long-mode identity map, the hypercall endpoint and the virtio-mmio
//! windows on one bus, and the run driven until a hypercall halts it.

use core::cell::{Cell, RefCell};

use proxima_core::ProximaError;
use proxima_primitives::pipe::fan_in::{FanIn, Select};
use proxima_primitives::pipe::{AndThen, Pipe};
use proxima_protocols::process::{ChildRequest, ChildResponse};

use super::describe::describe_failure;
use super::devices::{Channel, Hypercall, Lifecycle, RejectUnmapped, Transport, VirtioWindow};
use super::drive::drive;
use super::error::{KvmError, KvmResult};
use super::exit_source::TrapSource;
use super::memory::{GuestRam, HostMapping, NamedRegion, page_size, round_up_to_page};
use super::pipes::{ExitLedger, FirstClaim, MAX_TOTAL_EXITS, Meter, Resume};
use super::session::{Session, release_off_the_return_path};
use super::slots::{SlotOp, SlotRegion};
use super::trap::{RunError, Trap};
use super::watchdog::Watchdog;
use super::windows::{SegmentSpan, merge_into_windows};
use crate::page_table::{IdentitySpan, X86_HUGE_PAGE_SIZE, X86_TABLE_PAGE_SIZE, write_x86_64_identity_2m};

pub const CONSOLE_WINDOW_BASE: u64 = 0x10_0000_0000;
pub const NET_WINDOW_BASE: u64 = 0x10_0000_1000;
pub const BLK_WINDOW_BASE: u64 = 0x10_0000_2000;
pub const WINDOW_SIZE: u64 = 0x1000;

const GIGABYTE: u64 = 1024 * 1024 * 1024;
const RAM_ENTRY_FLAGS: u64 = 0xe3;
const UNCACHED_ENTRY_FLAGS: u64 = 0xfb;

/// One loadable segment: `data` is copied to `guest_address`, and the segment
/// owns `memory_size` bytes there (the tail beyond `data` stays zero).
#[derive(Clone, Copy, Debug)]
pub struct GuestSegment<'a> {
    pub guest_address: u64,
    pub data: &'a [u8],
    pub memory_size: u64,
    pub writable: bool,
}

pub struct Transports {
    pub console: Option<Transport>,
    pub net: Option<Transport>,
    pub blk: Option<Transport>,
}

#[derive(Clone, Copy, Debug)]
pub struct Capacities {
    pub emitted: usize,
    pub console: usize,
    pub net: usize,
    pub blk: usize,
}

pub struct ElfRun<'run, Dispatcher> {
    pub segments: &'run [GuestSegment<'run>],
    pub memory_size: u64,
    pub entry: u64,
    pub boot_argument: u64,
    pub dispatcher: &'run Dispatcher,
    pub transports: Transports,
    pub max_hypercalls: usize,
    pub watchdog_millis: u64,
    pub capacities: Capacities,
}

#[derive(Debug, Default)]
pub struct ElfReport {
    pub emitted: Vec<u8>,
    pub console: Vec<u8>,
    pub net: Vec<u8>,
    pub blk: Vec<u8>,
    pub create_to_first_exit_nanos: u64,
    pub touch_all_pages_nanos: u64,
    pub mmio_trap_count: u64,
    pub virtio_trap_count: u64,
    pub dispatch_sentinel: i64,
    pub exit_count: u64,
}

#[derive(Debug)]
pub struct ElfOutcome {
    pub report: ElfReport,
    pub failure: Option<String>,
}

struct Prepared {
    session: Session,
    ram: GuestRam,
    touch_all_pages_nanos: u64,
}

fn check_segments_fit(segments: &[GuestSegment<'_>], memory_size: u64) -> KvmResult<()> {
    let fits = segments.iter().all(|segment| {
        segment.guest_address <= memory_size
            && segment.memory_size <= memory_size - segment.guest_address
            && segment.data.len() as u64 <= segment.memory_size
    });
    match fits {
        true => Ok(()),
        false => Err(KvmError::guest("guest segment exceeds guest memory reservation")),
    }
}

fn build_tables(mapped_size: u64) -> KvmResult<(HostMapping, u64)> {
    let mapped_ram_end = mapped_size.div_ceil(X86_HUGE_PAGE_SIZE) * X86_HUGE_PAGE_SIZE;
    let directory_pages = mapped_ram_end.div_ceil(GIGABYTE) + 1;
    let byte_size = (2 + directory_pages) * X86_TABLE_PAGE_SIZE;
    if mapped_ram_end + byte_size > CONSOLE_WINDOW_BASE {
        return Err(KvmError::guest("guest memory reaches the virtio-mmio window"));
    }
    let mut tables = HostMapping::anonymous(usize::try_from(byte_size).unwrap_or(usize::MAX), true)?;
    let uncached_start = CONSOLE_WINDOW_BASE & !(X86_HUGE_PAGE_SIZE - 1);
    let spans = [
        IdentitySpan { start: 0, end: mapped_ram_end, flags: RAM_ENTRY_FLAGS },
        IdentitySpan { start: uncached_start, end: uncached_start + X86_HUGE_PAGE_SIZE, flags: UNCACHED_ENTRY_FLAGS },
    ];
    write_x86_64_identity_2m(tables.bytes_mut(), mapped_ram_end, &spans)
        .map_err(|_| KvmError::guest("guest page tables exceed their reserved pages"))?;
    Ok((tables, mapped_ram_end))
}

fn slots_for(segments: &[GuestSegment<'_>], ram: &HostMapping, tables: &HostMapping, tables_address: u64) -> Vec<SlotOp> {
    let spans: Vec<SegmentSpan> = segments
        .iter()
        .map(|segment| SegmentSpan {
            guest_address: segment.guest_address,
            memory_size: segment.memory_size,
            writable: segment.writable,
        })
        .collect();
    let mut ops: Vec<SlotOp> = merge_into_windows(&spans, page_size() as u64)
        .iter()
        .enumerate()
        .map(|(index, window)| {
            SlotOp::Submit(SlotRegion {
                slot: index as u32,
                flags: if window.writable { 0 } else { kvm_bindings::KVM_MEM_READONLY },
                guest_address: window.start,
                size: window.end - window.start,
                host: ram.as_ptr() as u64 + window.start,
            })
        })
        .collect();
    let slot = ops.len() as u32;
    ops.push(SlotOp::Submit(SlotRegion {
        slot,
        flags: 0,
        guest_address: tables_address,
        size: tables.len() as u64,
        host: tables.as_ptr() as u64,
    }));
    ops
}

fn prepare<Dispatcher>(spec: &ElfRun<'_, Dispatcher>) -> KvmResult<Prepared> {
    check_segments_fit(spec.segments, spec.memory_size)?;
    let mut session = Session::open()?;
    let mapped_size = round_up_to_page(usize::try_from(spec.memory_size).unwrap_or(usize::MAX).max(1));
    let mut region = NamedRegion::create(mapped_size)?;
    let touch_all_pages_nanos = region.primary_mut().touch_every_page();
    for segment in spec.segments {
        let start = usize::try_from(segment.guest_address).unwrap_or(usize::MAX);
        region.primary_mut().bytes_mut()[start..start + segment.data.len()].copy_from_slice(segment.data);
    }
    let (tables, tables_address) = build_tables(mapped_size as u64)?;
    let ops = slots_for(spec.segments, region.primary(), &tables, tables_address);
    session.machine().apply(&ops[..ops.len() - 1], "KVM_SET_USER_MEMORY_REGION")?;
    session.machine().apply(&ops[ops.len() - 1..], "KVM_SET_USER_MEMORY_REGION page tables")?;
    let ram = session.attach_named_region(region);
    session.attach_mapping(tables);
    session.create_vcpu()?;
    session.vcpu()?.start_flat(spec.entry, mapped_size as u64, spec.boot_argument, Some(tables_address))?;
    Ok(Prepared { session, ram, touch_all_pages_nanos })
}

fn channel(capacity: usize) -> RefCell<Channel> {
    RefCell::new(Channel::with_capacity(capacity))
}

/// Runs the guest to its halt hypercall, or to the first thing that ends it
/// otherwise. The outcome always carries what the guest produced before it
/// ended, success or not.
pub fn run_elf<Dispatcher>(spec: ElfRun<'_, Dispatcher>) -> ElfOutcome
where
    Dispatcher: Pipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>,
{
    let unobserved = ExitLedger::begin();
    run_elf_observed(spec, Meter::new(&unobserved, u64::MAX, "exit budget"))
}

/// [`run_elf`] with an observe pipe interposed between the exit sources and the
/// bus. It sees every trap the bus will see, in order, and hands each back
/// untouched; neither the sources nor the devices know it is there.
pub fn run_elf_observed<Dispatcher, Observer>(spec: ElfRun<'_, Dispatcher>, observer: Observer) -> ElfOutcome
where
    Dispatcher: Pipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>,
    Observer: Pipe<In = Trap, Out = Trap, Err = RunError>,
{
    let ledger = ExitLedger::begin();
    let channels = [
        channel(spec.capacities.emitted),
        channel(spec.capacities.console),
        channel(spec.capacities.net),
        channel(spec.capacities.blk),
    ];
    let sentinel = Cell::new(0_i64);
    let prepared = prepare(&spec);
    let (failure, touch_all_pages_nanos) = match prepared {
        Ok(prepared) => {
            let touch = prepared.touch_all_pages_nanos;
            let failure = drive_prepared(&prepared, spec, observer, &ledger, &channels, &sentinel);
            release_off_the_return_path(prepared.session);
            (failure, touch)
        }
        Err(error) => (Some(error.to_string()), 0),
    };
    let [emitted, console, net, blk] = channels.map(|channel| channel.into_inner().into_bytes());
    ElfOutcome {
        report: ElfReport {
            emitted,
            console,
            net,
            blk,
            create_to_first_exit_nanos: ledger.first_exit_nanos(),
            touch_all_pages_nanos,
            mmio_trap_count: ledger.mmio_exits(),
            virtio_trap_count: ledger.virtio_exits(),
            dispatch_sentinel: sentinel.get(),
            exit_count: ledger.exits(),
        },
        failure,
    }
}

fn drive_prepared<Dispatcher, Observer>(
    prepared: &Prepared,
    spec: ElfRun<'_, Dispatcher>,
    observer: Observer,
    ledger: &ExitLedger,
    channels: &[RefCell<Channel>; 4],
    sentinel: &Cell<i64>,
) -> Option<String>
where
    Dispatcher: Pipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>,
    Observer: Pipe<In = Trap, Out = Trap, Err = RunError>,
{
    let (machine, vcpu) = (prepared.session.machine(), prepared.session.vcpu().ok()?);
    let watchdog = match Watchdog::arm(vcpu.immediate_exit_flag(), spec.watchdog_millis) {
        Ok(armed) => armed,
        Err(error) => return Some(error.to_string()),
    };
    let ram = prepared.ram;
    let Transports { console, net, blk } = spec.transports;
    let window = |name, base, transport, index: usize| {
        VirtioWindow::new(name, base, WINDOW_SIZE, transport, &channels[index], ram, ledger)
    };
    let hypercall = Hypercall::new(spec.dispatcher, &channels[0], sentinel, spec.max_hypercalls);
    let bus = FirstClaim::new(
        Lifecycle::new(false),
        FirstClaim::new(
            window("console", CONSOLE_WINDOW_BASE, console, 1),
            FirstClaim::new(
                window("net", NET_WINDOW_BASE, net, 2),
                FirstClaim::new(window("blk", BLK_WINDOW_BASE, blk, 3), FirstClaim::new(hypercall, RejectUnmapped)),
            ),
        ),
    );
    let merge = FanIn::new(
        [
            TrapSource::Watchdog(watchdog.as_ref()),
            TrapSource::Exit { vcpu, hypercall_memory: Some(ram) },
        ],
        Select::Fifo,
    );
    let meter = Meter::new(ledger, MAX_TOTAL_EXITS, "exceeded total exit budget without halting");
    let observed = AndThen::new(AndThen::new(merge, meter), observer);
    let step = AndThen::new(AndThen::new(observed, bus), Resume::new(machine, vcpu, ram));
    drive(&step).err().map(|error| describe_failure(&error, Some(vcpu)))
}
