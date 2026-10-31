//! The copy-on-write primitive probe on KVM (`tools/proxima-vm/ROADMAP.md`,
//! µsec campaign slice 3). The Hypervisor.framework candidates map onto Linux
//! like so: a mach memory entry is a `memfd`; `mach_vm_remap(copy)` is a
//! `MAP_PRIVATE` view of that memfd; `MAP_MEM_VM_COPY` is an eager copy into a
//! second memfd; `hv_vm_map`/`hv_vm_unmap`/`hv_vm_protect` are memslot submit,
//! retire, and flag change. Probe-only: nothing else calls it.

use std::time::Instant;

use futures::FutureExt;
use kvm_bindings::KVM_MEM_READONLY;
use proxima_primitives::pipe::UnpinPipe;

use super::error::{KvmError, KvmResult};
use super::exit_source::TrapSource;
use super::memory::{HostMapping, NamedRegion, page_size, round_up_to_page};
use super::scratch::OUTPUT_PORT;
use super::session::Session;
use super::slots::SlotTable;
use super::trap::{Trap, VcpuExit};

const SOURCE_FILL: u8 = 0xab;
const WRITE_FLAG: u64 = 2;
const KVM_EXIT_MMIO: u64 = 6;

#[derive(Clone, Copy, Debug, Default)]
pub struct CowTimings {
    pub remap_nanos: u64,
    pub unmap_old_nanos: u64,
    pub map_nanos: u64,
    pub dealloc_old_nanos: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CopyTimings {
    pub entry_create_nanos: u64,
    pub map_nanos: u64,
    pub unmap_old_nanos: u64,
    pub guest_map_nanos: u64,
    pub dealloc_old_nanos: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WriteProtectExit {
    pub checkpoint1_x0: u64,
    pub exception_class: u64,
    pub is_data_abort: bool,
    pub is_write: bool,
    pub data_byte_after: u8,
    pub protect_nanos: u64,
}

pub struct ProbeVm {
    session: Session,
    table: SlotTable,
    previous_copy: Option<NamedRegion>,
}

fn nanos_since(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn pages_of(size: usize) -> u64 {
    (round_up_to_page(size) / page_size()) as u64
}

impl ProbeVm {
    pub fn create() -> KvmResult<Self> {
        Ok(Self {
            session: Session::open()?,
            table: SlotTable::new(0, page_size() as u64),
            previous_copy: None,
        })
    }

    pub fn create_source(size: usize) -> KvmResult<NamedRegion> {
        let mut source = NamedRegion::create(size)?;
        source.primary_mut().bytes_mut().fill(SOURCE_FILL);
        Ok(source)
    }

    fn replace_guest_view(&mut self, view: &HostMapping, size: usize) -> KvmResult<(u64, u64)> {
        let machine = self.session.machine();
        let started = Instant::now();
        machine.apply(&self.table.clear(), "KVM_SET_USER_MEMORY_REGION probe view")?;
        let unmap_nanos = nanos_since(started);
        let started = Instant::now();
        let install = self.table.install(0, pages_of(size), view.as_ptr() as u64, 0);
        machine.apply(&[install], "KVM_SET_USER_MEMORY_REGION probe view")?;
        Ok((unmap_nanos, nanos_since(started)))
    }

    pub fn cow_view(
        &mut self,
        source: &NamedRegion,
        size: usize,
        previous: Option<HostMapping>,
    ) -> KvmResult<(HostMapping, CowTimings)> {
        let started = Instant::now();
        let view = source.map_view(true)?;
        let remap_nanos = nanos_since(started);
        let (unmap_old_nanos, map_nanos) = self.replace_guest_view(&view, size)?;
        let started = Instant::now();
        drop(previous);
        Ok((view, CowTimings { remap_nanos, unmap_old_nanos, map_nanos, dealloc_old_nanos: nanos_since(started) }))
    }

    pub fn copy_view(
        &mut self,
        source: &NamedRegion,
        size: usize,
        previous: Option<HostMapping>,
    ) -> KvmResult<(HostMapping, CopyTimings)> {
        let started = Instant::now();
        let mut copy = NamedRegion::create(size)?;
        copy.primary_mut().bytes_mut().copy_from_slice(source.primary().bytes());
        let entry_create_nanos = nanos_since(started);
        let started = Instant::now();
        let view = copy.map_view(false)?;
        let map_nanos = nanos_since(started);
        self.previous_copy = Some(copy);
        let (unmap_old_nanos, guest_map_nanos) = self.replace_guest_view(&view, size)?;
        let started = Instant::now();
        drop(previous);
        Ok((
            view,
            CopyTimings {
                entry_create_nanos,
                map_nanos,
                unmap_old_nanos,
                guest_map_nanos,
                dealloc_old_nanos: nanos_since(started),
            },
        ))
    }

    pub fn first_touch(view: &mut HostMapping, page_count: usize) -> u64 {
        let started = Instant::now();
        let stride = page_size();
        for index in 0..page_count {
            view.bytes_mut()[index * stride] = 0x5a;
        }
        nanos_since(started)
    }

    pub fn map_guest(&mut self, view: &HostMapping, flags: u64) -> KvmResult<()> {
        let kvm_flags = if flags & WRITE_FLAG != 0 { 0 } else { KVM_MEM_READONLY };
        let install = self.table.install(0, pages_of(view.len()), view.as_ptr() as u64, kvm_flags);
        self.session.machine().apply(&[install], "KVM_SET_USER_MEMORY_REGION probe map")
    }

    pub fn unmap_guest(&mut self) -> KvmResult<()> {
        self.session.machine().apply(&self.table.clear(), "KVM_SET_USER_MEMORY_REGION probe unmap")
    }

    pub fn protect_whole(&mut self, read_only: bool) -> KvmResult<u64> {
        let started = Instant::now();
        let ops = self.table.set_all_flags(if read_only { KVM_MEM_READONLY } else { 0 });
        self.session.machine().apply(&ops, "KVM_SET_USER_MEMORY_REGION protect")?;
        Ok(nanos_since(started))
    }

    pub fn protect_per_page(&mut self, granule: usize, page_count: usize) -> KvmResult<Vec<u64>> {
        let table_granule = self.table.granule();
        let pages_per_granule = granule as u64 / table_granule;
        let mut timings = Vec::with_capacity(page_count);
        for index in 0..page_count as u64 {
            let first = index * pages_per_granule;
            let host = self
                .table
                .host_of(first)
                .ok_or_else(|| KvmError::guest("protect_per_page range is not mapped"))?;
            let started = Instant::now();
            let ops = self
                .table
                .retarget(first, first + pages_per_granule, host, KVM_MEM_READONLY)
                .map_err(|error| KvmError::guest(error.to_string()))?;
            self.session.machine().apply(&ops, "KVM_SET_USER_MEMORY_REGION protect page")?;
            timings.push(nanos_since(started));
        }
        Ok(timings)
    }

    /// A real-mode guest at page 1 stores to page 2, signals on port 0xe9,
    /// stores again, and halts. The host makes page 2 read-only at the signal,
    /// so the second store exits as MMIO: KVM's spelling of a write to a
    /// read-only slot.
    pub fn write_protect_exit(&mut self) -> KvmResult<WriteProtectExit> {
        const GUEST_CODE: [u8; 15] = [
            0xc6, 0x06, 0x00, 0x20, 0x2a, 0xb0, 0x01, 0xe6, OUTPUT_PORT as u8, 0xc6, 0x06, 0x00, 0x20, 0x55, 0xf4,
        ];
        let granule = page_size();
        let mut memory = HostMapping::anonymous(2 * granule, true)?;
        memory.bytes_mut()[..GUEST_CODE.len()].copy_from_slice(&GUEST_CODE);
        let (code_host, data_host) = (memory.as_ptr() as u64, memory.as_ptr() as u64 + granule as u64);
        let machine = self.session.machine();
        machine.apply(&self.table.clear(), "KVM_SET_USER_MEMORY_REGION probe pages")?;
        let installs = [self.table.install(1, 2, code_host, 0), self.table.install(2, 3, data_host, 0)];
        machine.apply(&installs, "KVM_SET_USER_MEMORY_REGION probe pages")?;
        self.session.create_vcpu()?;
        let vcpu = self.session.vcpu()?;
        vcpu.start_real_mode(granule as u64)?;
        let source = TrapSource::Exit { vcpu, hypercall_memory: None };
        let first = source.call(()).now_or_never();
        if !matches!(first, Some(Ok(Trap::Exit(VcpuExit::PortOut { .. })))) {
            return Err(KvmError::guest("write-protect probe did not reach checkpoint 1"));
        }
        let checkpoint1_x0 = vcpu.registers()?.rax & 0xff;
        let started = Instant::now();
        let ops = self
            .table
            .retarget(2, 3, data_host, KVM_MEM_READONLY)
            .map_err(|error| KvmError::guest(error.to_string()))?;
        self.session.machine().apply(&ops, "KVM_SET_USER_MEMORY_REGION protect data page")?;
        let protect_nanos = nanos_since(started);
        let second = source.call(()).now_or_never();
        let is_mmio_write = matches!(second, Some(Ok(Trap::Exit(VcpuExit::MmioWrite { .. }))));
        let is_mmio = is_mmio_write || matches!(second, Some(Ok(Trap::Exit(VcpuExit::MmioRead { .. }))));
        let data_byte_after = memory.bytes()[granule];
        self.session.machine().apply(&self.table.clear(), "KVM_SET_USER_MEMORY_REGION probe pages")?;
        Ok(WriteProtectExit {
            checkpoint1_x0,
            exception_class: if is_mmio { KVM_EXIT_MMIO } else { 0 },
            is_data_abort: is_mmio,
            is_write: is_mmio_write,
            data_byte_after,
            protect_nanos,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn a_store_to_a_slot_made_read_only_exits_as_an_mmio_write_and_never_lands() {
        let mut probe = ProbeVm::create().expect("kvm vm");

        let outcome = probe.write_protect_exit().expect("probe runs");

        assert_eq!(outcome.checkpoint1_x0, 1);
        assert!(outcome.is_data_abort && outcome.is_write);
        assert_eq!(outcome.exception_class, 6);
        assert_eq!(outcome.data_byte_after, 0x2a, "the first store landed, the protected second did not");
    }

    #[test]
    fn a_private_cow_view_reads_the_source_and_diverges_on_its_first_write() {
        let mut probe = ProbeVm::create().expect("kvm vm");
        let source = ProbeVm::create_source(1 << 20).expect("source");

        let (mut view, _) = probe.cow_view(&source, 1 << 20, None).expect("cow view");
        let before = view.bytes()[0];
        ProbeVm::first_touch(&mut view, 4);

        assert_eq!(before, SOURCE_FILL);
        assert_eq!(source.primary().bytes()[0], SOURCE_FILL, "the source is untouched");
        assert_eq!(view.bytes()[0], 0x5a);
    }
}
