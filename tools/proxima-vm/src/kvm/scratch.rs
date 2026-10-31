//! The scratch guest on KVM: a real-mode program that writes its message to
//! port 0xe9 a byte at a time and halts. It is the smallest machine that
//! exercises a run, a snapshot, and a restore end to end.

use core::cell::RefCell;
use core::future::Future;
use std::time::Instant;

use futures::FutureExt;
use kvm_bindings::kvm_regs;
use proxima_primitives::pipe::fan_in::{FanIn, Select};
use proxima_primitives::pipe::{AndThen, Pipe, UnpinPipe};

use super::describe::describe_failure;
use super::devices::{Lifecycle, answered, declined, failed};
use super::drive::drive;
use super::error::{KvmError, KvmResult};
use super::exit_source::TrapSource;
use super::memory::{GuestRam, NamedRegion, page_size, round_up_to_page};
use super::pipes::{ExitLedger, FirstClaim, MAX_TOTAL_EXITS, Meter, Resume};
use super::session::{Session, release_off_the_return_path};
use super::slots::{SlotOp, SlotRegion};
use super::trap::{End, Reply, RunError, Trap, VcpuExit};
use super::vcpu::Vcpu;

pub const OUTPUT_PORT: u16 = 0xe9;
pub const CODE_ADDRESS: u64 = 0x1000;
const RUN_MEMORY_SIZE: usize = 2 * 1024 * 1024;
const TERMINAL_VALUE: u16 = 256;
const MOV_AL_IMM8: u8 = 0xb0;
const OUT_IMM8_AL: u8 = 0xe6;
const MOV_AX_IMM16: u8 = 0xb8;
const HLT: u8 = 0xf4;

/// The scratch guest: for each message byte `mov al, imm8; out 0xe9, al`, then
/// `mov ax, 256; hlt`. 256 does not fit `al`, so no `out` can be mistaken for
/// the sentinel a snapshot captures at the halting trap.
#[must_use]
pub fn scratch_code(message: &[u8]) -> Vec<u8> {
    let mut code = Vec::with_capacity(message.len() * 4 + 4);
    for byte in message {
        code.extend_from_slice(&[MOV_AL_IMM8, *byte, OUT_IMM8_AL, OUT_PORT_BYTE]);
    }
    code.push(MOV_AX_IMM16);
    code.extend_from_slice(&TERMINAL_VALUE.to_le_bytes());
    code.push(HLT);
    code
}

const OUT_PORT_BYTE: u8 = 0xe9;

#[must_use]
pub fn memory_size_for(message_length: usize) -> usize {
    round_up_to_page(CODE_ADDRESS as usize + message_length * 4 + 4)
}

/// Collects the bytes the guest writes to the output port and ends the run at
/// its `hlt`, once the declared message has been emitted.
struct ScratchOutput<'run> {
    emitted: &'run RefCell<Vec<u8>>,
    declared: usize,
}

impl Pipe for ScratchOutput<'_> {
    type In = Trap;
    type Out = Reply;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Reply, RunError>> {
        match &trap {
            Trap::Exit(VcpuExit::PortOut { port: OUTPUT_PORT, size: 1, data, .. }) => {
                let mut emitted = self.emitted.borrow_mut();
                emitted.extend_from_slice(data);
                match emitted.len() > self.declared {
                    true => failed("scratch guest emitted more bytes than declared"),
                    false => answered(Reply::ACK),
                }
            }
            Trap::Exit(VcpuExit::Hlt) => match self.emitted.borrow().len() == self.declared {
                true => answered(Reply::End(End::Halted)),
                false => failed("scratch guest halted before emitting declared output"),
            },
            _ => declined(trap),
        }
    }
}

pub struct ScratchMachine {
    session: Session,
    ram: GuestRam,
}

fn real_mode_machine(message: &[u8], memory_size: usize) -> KvmResult<ScratchMachine> {
    let mut session = Session::open()?;
    let mut region = NamedRegion::create(memory_size)?;
    let code = scratch_code(message);
    let start = CODE_ADDRESS as usize;
    region.primary_mut().bytes_mut()[start..start + code.len()].copy_from_slice(&code);
    map_whole_region(&session, &region)?;
    let ram = session.attach_named_region(region);
    session.create_vcpu()?;
    session.vcpu()?.start_real_mode(CODE_ADDRESS)?;
    Ok(ScratchMachine { session, ram })
}

fn map_whole_region(session: &Session, region: &NamedRegion) -> KvmResult<()> {
    let slot = SlotOp::Submit(SlotRegion {
        slot: 0,
        flags: 0,
        guest_address: 0,
        size: region.primary().len() as u64,
        host: region.primary().as_ptr() as u64,
    });
    session.machine().apply(&[slot], "KVM_SET_USER_MEMORY_REGION")
}

fn run_to_halt(machine: &ScratchMachine, message_length: usize, emitted: &RefCell<Vec<u8>>) -> KvmResult<()> {
    let session = &machine.session;
    let vcpu = session.vcpu()?;
    let ledger = ExitLedger::begin();
    let bus = FirstClaim::new(Lifecycle::new(false), ScratchOutput { emitted, declared: message_length });
    let merge = FanIn::new(
        [TrapSource::Watchdog(None), TrapSource::Exit { vcpu, hypercall_memory: None }],
        Select::Fifo,
    );
    let meter = Meter::new(&ledger, MAX_TOTAL_EXITS, "exceeded total exit budget without halting");
    let step = AndThen::new(AndThen::new(AndThen::new(merge, meter), bus), Resume::new(session.machine(), vcpu, machine.ram));
    drive(&step)
        .map(|_| ())
        .map_err(|error| KvmError::guest(describe_failure(&error, Some(vcpu))))
}

/// Runs the scratch guest and returns the bytes it wrote to the output port.
pub fn run_scratch(message: &[u8]) -> KvmResult<Vec<u8>> {
    if CODE_ADDRESS as usize + message.len() * 4 + 4 > RUN_MEMORY_SIZE {
        return Err(KvmError::guest("scratch guest message does not fit guest memory"));
    }
    let machine = real_mode_machine(message, RUN_MEMORY_SIZE)?;
    let emitted = RefCell::new(Vec::new());
    let outcome = run_to_halt(&machine, message.len(), &emitted);
    release_off_the_return_path(machine.session);
    outcome.map(|()| emitted.into_inner())
}

/// One vCPU's general registers as a snapshot keeps them: `rax, rbx, rcx, rdx,
/// rsi, rdi, rsp, rbp, r8..r15` in `struct kvm_regs` order, then `rip` and
/// `rflags`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuestRegisters {
    pub gpr: [u64; 16],
    pub pc: u64,
    pub flags: u64,
}

impl GuestRegisters {
    fn from_kvm(registers: &kvm_regs) -> Self {
        Self {
            gpr: [
                registers.rax, registers.rbx, registers.rcx, registers.rdx, registers.rsi, registers.rdi,
                registers.rsp, registers.rbp, registers.r8, registers.r9, registers.r10, registers.r11,
                registers.r12, registers.r13, registers.r14, registers.r15,
            ],
            pc: registers.rip,
            flags: registers.rflags,
        }
    }

    fn to_kvm(self) -> kvm_regs {
        let [rax, rbx, rcx, rdx, rsi, rdi, rsp, rbp, r8, r9, r10, r11, r12, r13, r14, r15] = self.gpr;
        kvm_regs {
            rax, rbx, rcx, rdx, rsi, rdi, rsp, rbp, r8, r9, r10, r11, r12, r13, r14, r15,
            rip: self.pc,
            rflags: self.flags,
        }
    }
}

pub struct ScratchCapture {
    pub output: Vec<u8>,
    pub registers: GuestRegisters,
    pub memory: Vec<u8>,
}

/// Runs the scratch guest to its halting trap and captures the registers and
/// memory there. `hlt` retires before KVM reports it, so `rip` is rewound one
/// byte to "about to execute the halting trap", the state a restore re-traps at.
pub fn capture_scratch(message: &[u8]) -> KvmResult<ScratchCapture> {
    let memory_size = memory_size_for(message.len());
    let machine = real_mode_machine(message, memory_size)?;
    let emitted = RefCell::new(Vec::new());
    let outcome = run_to_halt(&machine, message.len(), &emitted).and_then(|()| {
        let mut registers = GuestRegisters::from_kvm(&machine.session.vcpu()?.registers()?);
        registers.pc -= 1;
        let memory = machine.ram.read(0, memory_size as u64).map_or_else(Vec::new, |bytes| bytes.to_vec());
        Ok((registers, memory))
    });
    release_off_the_return_path(machine.session);
    let (registers, memory) = outcome?;
    Ok(ScratchCapture { output: emitted.into_inner(), registers, memory })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResumeObservation {
    pub fault_count: u64,
    pub resumed_x0: u64,
    pub resumed_ok: bool,
    pub first_retrap_nanos: u64,
}

fn restore_registers(vcpu: &Vcpu, registers: &GuestRegisters) -> KvmResult<()> {
    vcpu.start_real_mode(registers.pc)?;
    vcpu.set_registers(&registers.to_kvm())
}

/// The one resumed step every restore ends with: the snapshot was taken at the
/// guest's halting `hlt`, so one run re-traps at the identical instruction.
fn resume_once(vcpu: &Vcpu) -> KvmResult<ResumeObservation> {
    let started = Instant::now();
    let source = TrapSource::Exit { vcpu, hypercall_memory: None };
    let trap = source.call(()).now_or_never();
    let first_retrap_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let mut observation = ResumeObservation { first_retrap_nanos, ..ResumeObservation::default() };
    match trap {
        Some(Ok(Trap::Exit(VcpuExit::RunFailed { errno }))) => {
            return Err(KvmError::syscall("KVM_RUN resumed step", std::io::Error::from_raw_os_error(errno)));
        }
        Some(Ok(Trap::Exit(VcpuExit::MmioRead { .. } | VcpuExit::MmioWrite { .. }))) => observation.fault_count = 1,
        Some(Ok(Trap::Exit(VcpuExit::Hlt))) => {
            observation.resumed_x0 = vcpu.registers()?.rax;
            observation.resumed_ok = true;
        }
        _ => {}
    }
    Ok(observation)
}

/// Copies `source` into `destination` one `stride` chunk at a time and returns
/// the bytes moved, counted per chunk as it copied.
fn copy_page_strided(destination: &mut [u8], source: &[u8], stride: usize) -> u64 {
    destination
        .chunks_mut(stride)
        .zip(source.chunks(stride))
        .map(|(to, from)| {
            to[..from.len()].copy_from_slice(from);
            from.len() as u64
        })
        .sum()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RestoreTimings {
    pub restore_wall_nanos: u64,
    pub touch_all_pages_nanos: u64,
    pub bytes_copied: u64,
    pub region_create_nanos: u64,
    pub vm_create_nanos: u64,
    pub vm_map_nanos: u64,
    pub vcpu_create_nanos: u64,
    pub register_restore_nanos: u64,
}

fn elapsed_nanos(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Restores into a brand-new VM, region, and vCPU, copying the memory in
/// `page_size`-strided chunks, then resumes once.
pub fn restore_scratch(
    registers: &GuestRegisters,
    memory: &[u8],
    page_size_hint: usize,
) -> KvmResult<(RestoreTimings, ResumeObservation)> {
    let begun = Instant::now();
    let stride = if page_size_hint > 0 { page_size_hint } else { page_size() };
    let mut timings = RestoreTimings::default();
    let phase = Instant::now();
    let mut region = NamedRegion::create(memory.len())?;
    timings.region_create_nanos = elapsed_nanos(phase);
    let phase = Instant::now();
    timings.bytes_copied = copy_page_strided(region.primary_mut().bytes_mut(), memory, stride);
    timings.touch_all_pages_nanos = elapsed_nanos(phase);
    let phase = Instant::now();
    let mut session = Session::open()?;
    timings.vm_create_nanos = elapsed_nanos(phase);
    let phase = Instant::now();
    map_whole_region(&session, &region)?;
    session.attach_named_region(region);
    timings.vm_map_nanos = elapsed_nanos(phase);
    let phase = Instant::now();
    session.create_vcpu()?;
    timings.vcpu_create_nanos = elapsed_nanos(phase);
    let phase = Instant::now();
    restore_registers(session.vcpu()?, registers)?;
    timings.register_restore_nanos = elapsed_nanos(phase);
    timings.restore_wall_nanos = elapsed_nanos(begun);
    let observation = resume_once(session.vcpu()?);
    release_off_the_return_path(session);
    Ok((timings, observation?))
}

/// A live VM, vCPU, and region held open across repeated restores.
pub struct WarmMachine {
    session: Session,
    ram: GuestRam,
    capacity: usize,
}

impl WarmMachine {
    pub fn create(capacity: usize) -> KvmResult<Self> {
        let mut session = Session::open()?;
        let region = NamedRegion::create(capacity)?;
        map_whole_region(&session, &region)?;
        let ram = session.attach_named_region(region);
        session.create_vcpu()?;
        Ok(Self { session, ram, capacity })
    }

    pub fn restore(
        &self,
        registers: &GuestRegisters,
        memory: &[u8],
        page_size_hint: usize,
    ) -> KvmResult<(RestoreTimings, ResumeObservation)> {
        let begun = Instant::now();
        if memory.len() > self.capacity {
            return Err(KvmError::guest("warm restore guest memory exceeds context capacity"));
        }
        let stride = if page_size_hint > 0 { page_size_hint } else { page_size() };
        let mut timings = RestoreTimings::default();
        let phase = Instant::now();
        timings.bytes_copied = self.ram.with_mut(|ram| copy_page_strided(ram, memory, stride));
        timings.touch_all_pages_nanos = elapsed_nanos(phase);
        let phase = Instant::now();
        let vcpu = self.session.vcpu()?;
        restore_registers(vcpu, registers)?;
        timings.register_restore_nanos = elapsed_nanos(phase);
        timings.restore_wall_nanos = elapsed_nanos(begun);
        Ok((timings, resume_once(vcpu)?))
    }

    #[must_use]
    pub fn sample(&self, offset: usize, length: usize) -> Vec<u8> {
        self.ram.read(offset as u64, length as u64).map_or_else(Vec::new, |bytes| bytes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn the_scratch_program_is_mov_out_per_byte_then_mov_ax_256_and_hlt() {
        let code = scratch_code(b"hi");

        assert_eq!(
            code,
            [0xb0, b'h', 0xe6, 0xe9, 0xb0, b'i', 0xe6, 0xe9, 0xb8, 0x00, 0x01, 0xf4]
        );
    }

    #[test]
    fn the_scratch_guest_emits_its_message_through_a_real_vm_exit() {
        assert_eq!(run_scratch(b"hello from proxima-vm\n").expect("kvm runs the scratch guest"), b"hello from proxima-vm\n");
    }

    #[test]
    fn a_snapshot_taken_at_the_halt_restores_into_a_fresh_vm_and_re_traps() {
        let capture = capture_scratch(b"snap").expect("capture");

        let (_, observation) = restore_scratch(&capture.registers, &capture.memory, 4096).expect("restore");

        assert_eq!(capture.output, b"snap");
        assert!(observation.resumed_ok, "the resumed step re-traps at the hlt");
        assert_eq!(observation.resumed_x0, 256);
    }

    #[test]
    fn copy_page_strided_copies_every_chunk_including_the_ragged_tail() {
        let source: Vec<u8> = (0..10).collect();
        let mut destination = vec![0xaa; 10];

        let copied = copy_page_strided(&mut destination, &source, 4);

        assert_eq!(destination, source);
        assert_eq!(copied, 10, "the counter is the bytes moved, ragged tail included");
    }
}
