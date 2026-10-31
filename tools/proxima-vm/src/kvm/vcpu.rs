use std::cell::RefCell;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use bytes::Bytes;
use kvm_bindings::{
    KVM_EXIT_FAIL_ENTRY, KVM_EXIT_INTERNAL_ERROR, KVM_EXIT_IO, KVM_EXIT_MMIO, KVM_MAX_CPUID_ENTRIES,
    kvm_cpuid_entry2, kvm_pit_config, kvm_regs, kvm_run, kvm_segment, kvm_sregs,
    kvm_userspace_memory_region,
};
use kvm_ioctls::{Kvm, VcpuExit as RawExit, VcpuFd, VmFd};

use super::error::{KvmError, KvmResult};
use super::slots::SlotOp;
use super::trap::VcpuExit;

const EINTR: i32 = 4;
const CPUID_LEAF_FEATURES: u32 = 1;
const CPUID_LEAF_EXTENDED_TOPOLOGY: u32 = 0xb;
const CPUID_LEAF_V2_EXTENDED_TOPOLOGY: u32 = 0x1f;
const CPUID_EBX_APIC_ID_MASK: u32 = 0x00ff_ffff;
const RFLAGS_RESERVED: u64 = 2;
const CODE_SEGMENT_TYPE: u8 = 0x0b;
const DATA_SEGMENT_TYPE: u8 = 0x03;
const FLAT_CODE_SELECTOR: u16 = 0x08;
const FLAT_DATA_SELECTOR: u16 = 0x10;
const CR0_PROTECTED_FLAT: u64 = 0x33;
const CR0_PAGING_WRITE_PROTECT: u64 = 0x8001_0000;
const CR4_PAE_PGE_OSFXSR: u64 = 0x620;
const EFER_LME_LMA_NXE: u64 = 0x500;
const LINUX_BOOT_STACK: u64 = 0x6ff0;
const LINUX_CR0: u64 = 0x8001_0033;
const LINUX_CR4_PAE: u64 = 0x20;
const LINUX_GDT_LIMIT: u16 = 4 * 8 - 1;

pub struct Machine {
    kvm: Kvm,
    vm: VmFd,
}

pub struct Vcpu {
    fd: RefCell<VcpuFd>,
}

fn assign_apic_id_zero(entries: &mut [kvm_cpuid_entry2]) {
    for entry in entries {
        match entry.function {
            CPUID_LEAF_FEATURES => entry.ebx &= CPUID_EBX_APIC_ID_MASK,
            CPUID_LEAF_EXTENDED_TOPOLOGY | CPUID_LEAF_V2_EXTENDED_TOPOLOGY => entry.edx = 0,
            _ => {}
        }
    }
}

#[must_use]
pub fn flat_segment(selector: u16, type_: u8, is_code: bool, long_mode: bool) -> kvm_segment {
    let long_code = is_code && long_mode;
    kvm_segment {
        base: 0,
        limit: 0xffff_ffff,
        selector,
        type_,
        present: 1,
        dpl: 0,
        db: u8::from(!long_code),
        s: 1,
        l: u8::from(long_code),
        g: 1,
        ..kvm_segment::default()
    }
}

impl Machine {
    pub fn create() -> KvmResult<Self> {
        let kvm = Kvm::new().map_err(|error| KvmError::from_ioctl("open /dev/kvm", error))?;
        let vm = kvm.create_vm().map_err(|error| KvmError::from_ioctl("KVM_CREATE_VM", error))?;
        Ok(Self { kvm, vm })
    }

    pub fn create_vcpu(&self) -> KvmResult<Vcpu> {
        let fd = self.vm.create_vcpu(0).map_err(|error| KvmError::from_ioctl("KVM_CREATE_VCPU", error))?;
        let mut cpuid = self
            .kvm
            .get_supported_cpuid(KVM_MAX_CPUID_ENTRIES)
            .map_err(|error| KvmError::from_ioctl("KVM_GET_SUPPORTED_CPUID", error))?;
        assign_apic_id_zero(cpuid.as_mut_slice());
        fd.set_cpuid2(&cpuid).map_err(|error| KvmError::from_ioctl("KVM_SET_CPUID2", error))?;
        Ok(Vcpu { fd: RefCell::new(fd) })
    }

    pub fn apply(&self, ops: &[SlotOp], operation: &'static str) -> KvmResult<()> {
        ops.iter().try_for_each(|op| self.submit(*op, operation))
    }

    fn submit(&self, op: SlotOp, operation: &'static str) -> KvmResult<()> {
        let region = match op {
            SlotOp::Submit(region) => kvm_userspace_memory_region {
                slot: region.slot,
                flags: region.flags,
                guest_phys_addr: region.guest_address,
                memory_size: region.size,
                userspace_addr: region.host,
            },
            SlotOp::Retire { slot, guest_address, host } => kvm_userspace_memory_region {
                slot,
                flags: 0,
                guest_phys_addr: guest_address,
                memory_size: 0,
                userspace_addr: host,
            },
        };
        // SAFETY: every region names host memory its owner keeps mapped for the VM's life.
        unsafe { self.vm.set_user_memory_region(region) }
            .map_err(|error| KvmError::from_ioctl(operation, error))
    }

    pub fn create_interrupt_devices(&self, identity_map: u64, tss: usize) -> KvmResult<()> {
        let step = |operation, outcome: Result<(), kvm_ioctls::Error>| {
            outcome.map_err(|error| KvmError::from_ioctl(operation, error))
        };
        step("KVM_SET_IDENTITY_MAP_ADDR", self.vm.set_identity_map_address(identity_map))?;
        step("KVM_SET_TSS_ADDR", self.vm.set_tss_address(tss))?;
        step("KVM_CREATE_IRQCHIP", self.vm.create_irq_chip())?;
        step("KVM_CREATE_PIT2", self.vm.create_pit2(kvm_pit_config::default()))
    }

    pub fn set_irq_line(&self, line: u32, level: bool) -> KvmResult<()> {
        self.vm.set_irq_line(line, level).map_err(|error| KvmError::from_ioctl("KVM_IRQ_LINE", error))
    }

    pub fn duplicate_vm_descriptor(&self) -> KvmResult<OwnedFd> {
        // SAFETY: the descriptor is live for the duration of the call.
        let duplicate = unsafe { libc::fcntl(self.vm.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
        if duplicate < 0 {
            return Err(KvmError::last_os("dup vm descriptor"));
        }
        // SAFETY: a fresh descriptor nothing else owns.
        Ok(unsafe { OwnedFd::from_raw_fd(duplicate) })
    }
}

impl Vcpu {
    fn sregs(&self) -> KvmResult<kvm_sregs> {
        self.fd.borrow().get_sregs().map_err(|error| KvmError::from_ioctl("KVM_GET_SREGS", error))
    }

    fn set_sregs(&self, sregs: &kvm_sregs) -> KvmResult<()> {
        self.fd.borrow().set_sregs(sregs).map_err(|error| KvmError::from_ioctl("KVM_SET_SREGS", error))
    }

    pub fn registers(&self) -> KvmResult<kvm_regs> {
        self.fd.borrow().get_regs().map_err(|error| KvmError::from_ioctl("KVM_GET_REGS", error))
    }

    pub fn set_registers(&self, registers: &kvm_regs) -> KvmResult<()> {
        self.fd.borrow().set_regs(registers).map_err(|error| KvmError::from_ioctl("KVM_SET_REGS", error))
    }

    pub fn instruction_pointer(&self) -> u64 {
        self.registers().map_or(0, |registers| registers.rip)
    }

    pub fn start_real_mode(&self, entry: u64) -> KvmResult<()> {
        let mut sregs = self.sregs()?;
        sregs.cs.base = 0;
        sregs.cs.selector = 0;
        self.set_sregs(&sregs)?;
        self.set_registers(&kvm_regs { rip: entry, rflags: RFLAGS_RESERVED, ..kvm_regs::default() })
    }

    pub fn start_flat(
        &self,
        entry: u64,
        stack_pointer: u64,
        argument: u64,
        long_mode_root: Option<u64>,
    ) -> KvmResult<()> {
        let long_mode = long_mode_root.is_some();
        let mut sregs = self.sregs()?;
        sregs.cs = flat_segment(FLAT_CODE_SELECTOR, CODE_SEGMENT_TYPE, true, long_mode);
        sregs.ds = flat_segment(FLAT_DATA_SELECTOR, DATA_SEGMENT_TYPE, false, long_mode);
        sregs.es = sregs.ds;
        sregs.fs = sregs.ds;
        sregs.gs = sregs.ds;
        sregs.ss = sregs.ds;
        sregs.cr0 = CR0_PROTECTED_FLAT;
        sregs.cr3 = 0;
        sregs.cr4 = 0;
        sregs.efer = 0;
        if let Some(root) = long_mode_root {
            sregs.cr0 |= CR0_PAGING_WRITE_PROTECT;
            sregs.cr3 = root;
            sregs.cr4 = CR4_PAE_PGE_OSFXSR;
            sregs.efer = EFER_LME_LMA_NXE;
        }
        self.set_sregs(&sregs)?;
        self.set_registers(&kvm_regs {
            rip: entry,
            rflags: RFLAGS_RESERVED,
            rsp: stack_pointer,
            rbx: argument,
            ..kvm_regs::default()
        })
    }

    pub fn start_linux_boot(
        &self,
        entry: u64,
        boot_params: u64,
        gdt_address: u64,
        page_tables: u64,
    ) -> KvmResult<()> {
        let mut sregs = self.sregs()?;
        sregs.cs = flat_segment(crate::x86_boot::BOOT_CODE_SELECTOR, CODE_SEGMENT_TYPE, true, true);
        sregs.ds = flat_segment(crate::x86_boot::BOOT_DATA_SELECTOR, DATA_SEGMENT_TYPE, false, true);
        sregs.es = sregs.ds;
        sregs.fs = sregs.ds;
        sregs.gs = sregs.ds;
        sregs.ss = sregs.ds;
        sregs.gdt.base = gdt_address;
        sregs.gdt.limit = LINUX_GDT_LIMIT;
        sregs.cr0 = LINUX_CR0;
        sregs.cr3 = page_tables;
        sregs.cr4 = LINUX_CR4_PAE;
        sregs.efer = EFER_LME_LMA_NXE;
        self.set_sregs(&sregs)?;
        self.set_registers(&kvm_regs {
            rip: entry,
            rflags: RFLAGS_RESERVED,
            rsp: LINUX_BOOT_STACK,
            rsi: boot_params,
            ..kvm_regs::default()
        })
    }

    pub fn set_rax(&self, value: u64) -> KvmResult<()> {
        let mut registers = self.registers()?;
        registers.rax = value;
        self.set_registers(&registers)
    }

    pub fn immediate_exit_flag(&self) -> *mut u8 {
        &raw mut self.fd.borrow_mut().get_kvm_run().immediate_exit
    }

    pub fn run(&self) -> VcpuExit {
        let mut fd = self.fd.borrow_mut();
        let sketch = match fd.run() {
            Ok(RawExit::IoOut(port, data)) => Sketch::PortOut { port, data: Bytes::copy_from_slice(data) },
            Ok(RawExit::IoIn(port, _)) => Sketch::PortIn { port },
            Ok(RawExit::MmioRead(address, data)) => Sketch::MmioRead { address, length: data.len() },
            Ok(RawExit::MmioWrite(address, data)) => {
                Sketch::MmioWrite { address, data: Bytes::copy_from_slice(data) }
            }
            Ok(RawExit::Hlt) => Sketch::Hlt,
            Ok(RawExit::Shutdown) => Sketch::Shutdown,
            Ok(_) => Sketch::Other,
            Err(error) if error.errno() == EINTR => Sketch::Interrupted,
            Err(error) => return VcpuExit::RunFailed { errno: error.errno() },
        };
        sketch.finish(fd.get_kvm_run())
    }

    pub fn complete_read(&self, bytes: &[u8]) {
        let mut fd = self.fd.borrow_mut();
        let run = fd.get_kvm_run();
        match run.exit_reason {
            KVM_EXIT_IO => {
                // SAFETY: the kernel reported an io exit, so the `io` arm is the live one.
                let io = unsafe { run.__bindgen_anon_1.io };
                let length = (io.count as usize * usize::from(io.size)).min(bytes.len());
                let base = core::ptr::from_mut(run).cast::<u8>();
                // SAFETY: `data_offset` addresses the io buffer inside the mapped run page.
                let destination = unsafe { base.add(io.data_offset as usize) };
                // SAFETY: `length` fits both buffers and they do not overlap.
                unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, length) };
            }
            KVM_EXIT_MMIO => {
                // SAFETY: the kernel reported an mmio exit, so the `mmio` arm is the live one.
                let mmio = unsafe { &mut run.__bindgen_anon_1.mmio };
                let length = bytes.len().min(mmio.data.len());
                mmio.data[..length].copy_from_slice(&bytes[..length]);
            }
            _ => {}
        }
    }
}

enum Sketch {
    PortOut { port: u16, data: Bytes },
    PortIn { port: u16 },
    MmioRead { address: u64, length: usize },
    MmioWrite { address: u64, data: Bytes },
    Hlt,
    Shutdown,
    Interrupted,
    Other,
}

impl Sketch {
    fn finish(self, run: &mut kvm_run) -> VcpuExit {
        match self {
            Self::PortOut { port, data } => {
                let (size, count) = io_geometry(run);
                VcpuExit::PortOut { port, size, count, data, arguments: None, payload: None }
            }
            Self::PortIn { port } => {
                let (size, count) = io_geometry(run);
                VcpuExit::PortIn { port, size, count }
            }
            Self::MmioRead { address, length } => VcpuExit::MmioRead { address, length: length as u8 },
            Self::MmioWrite { address, data } => VcpuExit::MmioWrite { address, data },
            Self::Hlt => VcpuExit::Hlt,
            Self::Shutdown => VcpuExit::Shutdown,
            Self::Interrupted => VcpuExit::Interrupted,
            Self::Other => describe_other(run),
        }
    }
}

fn io_geometry(run: &kvm_run) -> (u8, u32) {
    // SAFETY: only called for exits the kernel reported as io.
    let io = unsafe { run.__bindgen_anon_1.io };
    (io.size, io.count)
}

fn describe_other(run: &kvm_run) -> VcpuExit {
    match run.exit_reason {
        KVM_EXIT_FAIL_ENTRY => VcpuExit::FailEntry {
            // SAFETY: the kernel reported a failed entry, so `fail_entry` is the live arm.
            hardware_reason: unsafe { run.__bindgen_anon_1.fail_entry.hardware_entry_failure_reason },
        },
        KVM_EXIT_INTERNAL_ERROR => VcpuExit::InternalError {
            // SAFETY: the kernel reported an internal error, so `internal` is the live arm.
            suberror: unsafe { run.__bindgen_anon_1.internal.suberror },
        },
        reason => VcpuExit::Other { reason },
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn the_64_bit_flat_code_segment_sets_l_and_clears_d() {
        let code = flat_segment(0x10, 0x0b, true, true);

        assert_eq!((code.l, code.db, code.g, code.present), (1, 0, 1, 1));
        assert_eq!((code.base, code.limit, code.selector, code.type_), (0, 0xffff_ffff, 0x10, 0x0b));
    }

    #[test]
    fn protected_mode_flat_code_is_32_bit_and_data_is_never_long() {
        let code = flat_segment(0x08, 0x0b, true, false);
        let data = flat_segment(0x10, 0x03, false, true);

        assert_eq!((code.l, code.db), (0, 1));
        assert_eq!((data.l, data.db), (0, 1), "a data segment carries D/B even in long mode");
    }

    #[test]
    fn the_cpuid_fixup_zeroes_the_apic_id_the_guest_would_otherwise_see() {
        let mut leaf_one = kvm_cpuid_entry2 { function: 1, ebx: 0x1234_5678, ..kvm_cpuid_entry2::default() };
        let mut topology = kvm_cpuid_entry2 { function: 0xb, edx: 9, ..kvm_cpuid_entry2::default() };
        let mut unrelated = kvm_cpuid_entry2 { function: 7, ebx: 0xdead, edx: 3, ..kvm_cpuid_entry2::default() };
        let mut entries = [leaf_one, topology, unrelated];

        assign_apic_id_zero(&mut entries);
        [leaf_one, topology, unrelated] = entries;

        assert_eq!(leaf_one.ebx, 0x0034_5678, "bits 31:24 of EBX hold the initial APIC id");
        assert_eq!(topology.edx, 0);
        assert_eq!((unrelated.ebx, unrelated.edx), (0xdead, 3));
    }
}
