//! Layered base-plus-delta warm restore on KVM. The base maps read-only, so
//! every guest store to it exits as MMIO; a device copies that one page base to
//! delta, applies the store the guest attempted (the x86 emulator has already
//! decoded it and will retire it on the next `KVM_RUN`, so it is not retried),
//! and the reply re-points the page at the delta read-write.

use core::cell::{Cell, RefCell};
use core::future::Future;
use std::time::Instant;

use kvm_bindings::{KVM_MEM_READONLY, kvm_regs};
use proxima_primitives::pipe::fan_in::{FanIn, Select};
use proxima_primitives::pipe::{AndThen, Pipe};

use super::describe::describe_failure;
use super::devices::{Lifecycle, answered, declined, failed};
use super::dirty::{DirtyLedger, Marked};
use super::drive::drive;
use super::error::{KvmError, KvmResult};
use super::exit_source::TrapSource;
use super::memory::{GuestRam, page_size};
use super::pipes::{ExitLedger, FirstClaim, Meter, Resume};
use super::session::Session;
use super::slots::SlotTable;
use super::trap::{End, Reply, RunError, Trap, VcpuExit};

const RFLAGS_RESERVED: u64 = 2;
const EXIT_SLACK: u64 = 64;

pub struct LayeredMachine {
    session: Session,
    slots: RefCell<SlotTable>,
    ledger: RefCell<DirtyLedger>,
    base: GuestRam,
    delta: GuestRam,
    ipa_base: u64,
    granule: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AdoptTimings {
    pub map_nanos: u64,
    pub register_reset_nanos: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RunReport {
    pub run_wall_nanos: u64,
    pub fault_count: u64,
    pub newly_dirty_page_count: u64,
    pub halted_ok: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RestoreReport {
    pub restore_wall_nanos: u64,
    pub remap_nanos: u64,
    pub register_reset_nanos: u64,
    pub remapped_page_count: u64,
}

fn nanos_since(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

struct CopyOnWrite<'run> {
    machine: &'run LayeredMachine,
    faults: &'run Cell<u64>,
    newly_dirty: &'run Cell<u64>,
}

impl CopyOnWrite<'_> {
    fn fault(&self, address: u64, data: &[u8]) -> Result<Reply, RunError> {
        let machine = self.machine;
        if address < machine.ipa_base || address >= machine.ipa_base + machine.base.len() as u64 {
            return Err(KvmError::guest("layered run fault address outside the base range").into());
        }
        let page = (address - machine.ipa_base) / machine.granule;
        self.faults.set(self.faults.get() + 1);
        let marked = machine.ledger.borrow_mut().mark(page).map_err(|error| KvmError::guest(error.to_string()))?;
        let offset_in_page = (address - machine.ipa_base) - page * machine.granule;
        if data.len() > 8 || offset_in_page + data.len() as u64 > machine.granule {
            return Err(KvmError::guest("layered run saw a guest write spanning two pages").into());
        }
        let page_offset = page * machine.granule;
        if marked == Marked::Already {
            return machine.store(page_offset + offset_in_page, data).map(|()| Reply::ACK);
        }
        self.newly_dirty.set(self.newly_dirty.get() + 1);
        if !machine.base.copy_to(&machine.delta, page_offset, machine.granule) {
            return Err(KvmError::guest("layered run page index exceeds delta capacity").into());
        }
        machine.store(page_offset + offset_in_page, data)?;
        let host = machine.delta.as_ptr() as u64 + page_offset;
        let ops = machine
            .slots
            .borrow_mut()
            .retarget(page, page + 1, host, 0)
            .map_err(|error| KvmError::guest(error.to_string()))?;
        Ok(Reply::Remap(ops))
    }
}

impl Pipe for CopyOnWrite<'_> {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match &trap {
            Trap::Exit(VcpuExit::Hlt) => answered(Reply::End(End::Halted)),
            Trap::Exit(VcpuExit::MmioWrite { address, data }) => core::future::ready(self.fault(*address, data)),
            Trap::Exit(VcpuExit::MmioRead { .. }) => failed("layered run saw a guest read of the base mmio window"),
            _ => declined(trap),
        }
    }
}

impl LayeredMachine {
    /// # Safety
    ///
    /// `base` and `delta` must stay valid for the machine's whole life.
    pub unsafe fn create(base: GuestRam, delta: GuestRam, ipa_base: u64) -> KvmResult<Self> {
        let mut session = Session::open()?;
        session.create_vcpu()?;
        session.vcpu()?.start_flat(ipa_base, 0, 0, None)?;
        let granule = page_size() as u64;
        let pages = (base.len() as u64).div_ceil(granule).max(1);
        Ok(Self {
            session,
            slots: RefCell::new(SlotTable::new(ipa_base, granule)),
            ledger: RefCell::new(DirtyLedger::new(pages as usize)),
            base,
            delta,
            ipa_base,
            granule,
        })
    }

    fn store(&self, offset: u64, data: &[u8]) -> Result<(), RunError> {
        match self.delta.write(offset, data) {
            true => Ok(()),
            false => Err(KvmError::guest("layered run write falls outside the delta region").into()),
        }
    }

    fn reset_vcpu(&self) -> KvmResult<()> {
        self.session.vcpu()?.set_registers(&kvm_regs { rip: self.ipa_base, rflags: RFLAGS_RESERVED, ..kvm_regs::default() })
    }

    fn map_base(&self) -> KvmResult<()> {
        let mut slots = self.slots.borrow_mut();
        let mut ops = slots.clear();
        let pages = (self.base.len() as u64).div_ceil(self.granule);
        ops.push(slots.install(0, pages, self.base.as_ptr() as u64, KVM_MEM_READONLY));
        self.session.machine().apply(&ops, "KVM_SET_USER_MEMORY_REGION layered base")
    }

    pub fn adopt_base(&self) -> KvmResult<AdoptTimings> {
        let started = Instant::now();
        self.map_base()?;
        let map_nanos = nanos_since(started);
        self.ledger.borrow_mut().clear();
        let started = Instant::now();
        self.reset_vcpu()?;
        Ok(AdoptTimings { map_nanos, register_reset_nanos: nanos_since(started) })
    }

    pub fn run(&self, expected_page_count: u64) -> KvmResult<RunReport> {
        self.reset_vcpu()?;
        let started = Instant::now();
        let (faults, newly_dirty) = (Cell::new(0), Cell::new(0));
        let vcpu = self.session.vcpu()?;
        let ledger = ExitLedger::begin();
        let bus = FirstClaim::new(Lifecycle::new(false), CopyOnWrite { machine: self, faults: &faults, newly_dirty: &newly_dirty });
        let merge = FanIn::new(
            [TrapSource::Watchdog(None), TrapSource::Exit { vcpu, hypercall_memory: None }],
            Select::Fifo,
        );
        let meter = Meter::new(&ledger, expected_page_count + EXIT_SLACK, "layered run exceeded exit budget without halting");
        let step = AndThen::new(
            AndThen::new(AndThen::new(merge, meter), bus),
            Resume::new(self.session.machine(), vcpu, self.base),
        );
        let ended = drive(&step);
        let report = RunReport {
            run_wall_nanos: nanos_since(started),
            fault_count: faults.get(),
            newly_dirty_page_count: newly_dirty.get(),
            halted_ok: matches!(ended, Ok(End::Halted)),
        };
        ended.map_err(|error| KvmError::guest(describe_failure(&error, Some(vcpu))))?;
        Ok(report)
    }

    /// Re-points every page the run dirtied back at the base. Retiring every
    /// run and installing one base slot is O(runs), and runs are bounded by
    /// 2K+1 for K dirty pages: O(working set), never a scan of the whole base.
    pub fn restore(&self) -> KvmResult<RestoreReport> {
        let started = Instant::now();
        let dirty = self.ledger.borrow_mut().drain_sorted();
        self.map_base()?;
        let remap_nanos = nanos_since(started);
        let reset_started = Instant::now();
        self.reset_vcpu()?;
        Ok(RestoreReport {
            restore_wall_nanos: nanos_since(started),
            remap_nanos,
            register_reset_nanos: nanos_since(reset_started),
            remapped_page_count: dirty.len() as u64,
        })
    }
}
