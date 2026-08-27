//! Host-side PSCI (ARM DEN0022, revision D) call handler — the M5b
//! counterpart to [`crate::abi::decode_hypercall`]. `src/dtb.rs`'s
//! `write_psci` advertises `compatible = "arm,psci-0.2"`, `method = "hvc"`
//! (`src/dtb.rs:174-179`), so this guest issues PSCI calls as `hvc #0` with
//! the function ID in `x0` (SMCCC calling convention) — the same trap class
//! (`0x16`, `backend_macos.c:1156-1189`) the existing `ChildRequest`
//! hypercall path already catches. This module is the pure decision
//! (`function_id`, `args`) -> [`PsciResult`]; it reads no vCPU register and
//! performs no I/O, mirroring `abi.rs`'s own free-function shape (a decode
//! decision over already-recovered register values, never a `Pipe`).
//!
//! # Routing: PSCI is disjoint from the existing hypercall ABI, by value
//!
//! `src/dispatch_trampoline.h` reserves `0xfffe`/`0xffff` for
//! `PROXIMA_VM_EMIT_VERB`/`PROXIMA_VM_HALT_VERB`, and the `ChildRequest`
//! postcard discriminants that flow through
//! `proxima_vm_dispatch_hypercall` occupy `0..=4`
//! (`proxima-protocols/src/process/protocol.rs:72-77`). Every PSCI function
//! ID SMCCC defines lives at `0x8400_0000` (32-bit fast call) or
//! `0xC400_0000` (64-bit fast call) plus a small offset — six orders of
//! magnitude above the existing verb space, so the two ranges never
//! collide. `is_psci_function_id` is the range test the C-side trap loop
//! uses to route a raw `hvc` exit to this module instead of
//! `proxima_vm_dispatch_hypercall`, before either sentinel check runs.
//!
//! # Single-vCPU scope
//!
//! This VM boots exactly one vCPU (`src/dtb.rs`'s `write_cpus` writes one
//! `cpu@0`, `reg = <0>`). Per the devicetree PSCI/CPU binding, `reg` is the
//! CPU's MPIDR affinity fields, so [`RESIDENT_MPIDR_AFFINITY`] (`0`) is the
//! only valid `CPU_ON` target this guest could ever legally address itself
//! (or another core) by.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`): no allocation, no register access, no
//! syscall — a pure match over integers.

/// PSCI 0.2 `PSCI_VERSION`: no arguments, returns `(major << 16) | minor`.
pub const PSCI_VERSION: u32 = 0x8400_0000;

/// PSCI 0.2 `CPU_OFF`: no arguments. Per the spec, this call does not
/// return on success — the calling core powers down. Every return value
/// this handler produces for it is therefore a failure return.
pub const PSCI_CPU_OFF: u32 = 0x8400_0002;

/// PSCI 0.2 `CPU_ON` (32-bit SMC calling convention): `target_cpu`,
/// `entry_point_address`, `context_id`.
pub const PSCI_CPU_ON_32: u32 = 0x8400_0003;

/// PSCI 0.2 `CPU_ON` (64-bit SMC calling convention) — same arguments and
/// semantics as [`PSCI_CPU_ON_32`], the form the roadmap names explicitly.
pub const PSCI_CPU_ON_64: u32 = 0xC400_0003;

/// PSCI 0.2 `SYSTEM_OFF`: no arguments, no return on success.
pub const PSCI_SYSTEM_OFF: u32 = 0x8400_0008;

/// PSCI 0.2 `SYSTEM_RESET`: no arguments, no return on success.
pub const PSCI_SYSTEM_RESET: u32 = 0x8400_0009;

/// PSCI 0.2 `PSCI_FEATURES`: one argument, the function ID being queried.
pub const PSCI_FEATURES: u32 = 0x8400_000a;

/// `SUCCESS` — spec §5.1, table 5.
const RETURN_SUCCESS: i32 = 0;
/// `NOT_SUPPORTED` — the function ID is not implemented by this PSCI
/// provider.
const RETURN_NOT_SUPPORTED: i32 = -1;
/// `INVALID_PARAMETERS` — `CPU_ON`'s `target_cpu` does not name a CPU that
/// exists in this system (spec §5.6.2: any value other than a real MPIDR).
const RETURN_INVALID_PARAMS: i32 = -2;
/// `DENIED` — spec §5.1: the legal return when a call this implementation
/// does not support cannot be carried out (`CPU_OFF` on the only core: there
/// is nothing to hand execution to).
const RETURN_DENIED: i32 = -3;
/// `ALREADY_ON` — `CPU_ON`'s `target_cpu` names a core that is already
/// running (spec §5.6.2), which for this single-vCPU guest is only ever the
/// caller's own core.
const RETURN_ALREADY_ON: i32 = -4;

/// PSCI 0.2 version this handler answers `PSCI_VERSION` with: major 0,
/// minor 2, matching `src/dtb.rs`'s advertised `arm,psci-0.2` compatible
/// string (`(major << 16) | minor`).
const PSCI_VERSION_VALUE: i32 = 0x0000_0002;

/// The only `target_cpu` value this single-vCPU guest could legally name —
/// the MPIDR affinity fields for `cpu@0`'s devicetree `reg = <0>`
/// (`src/dtb.rs`'s `write_cpus`).
pub const RESIDENT_MPIDR_AFFINITY: u64 = 0;

/// What one PSCI call resolves to: a register value the caller must write
/// back into `x0` before resuming the guest, or one of the two calls that
/// end this VM's run instead of returning to it — composing with the exit
/// path `backend_macos.c`'s `PROXIMA_VM_HALT_VERB` check already provides,
/// not a second exit channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsciResult {
    /// Write this signed 32-bit value into the guest's `x0` and resume.
    ReturnValue(i32),
    /// `SYSTEM_OFF` was called: end the dispatch loop the same way
    /// `PROXIMA_VM_HALT_VERB` does, never returning to the guest.
    SystemOff,
    /// `SYSTEM_RESET` was called: this handler has no reset capability to
    /// offer, so it ends the dispatch loop identically to `SystemOff`
    /// rather than silently returning a success the guest would then run
    /// past as if nothing happened.
    SystemReset,
}

/// True when `function_id` falls in either SMCCC fast-call range PSCI
/// occupies (`0x8400_0000..=0x8400_001f` 32-bit, `0xC400_0000..=0xC400_001f`
/// 64-bit) — the range test the C-side trap loop uses to route a raw `hvc`
/// exit here instead of `proxima_vm_dispatch_hypercall`.
#[must_use]
pub fn is_psci_function_id(function_id: u32) -> bool {
    const RANGE_WIDTH: u32 = 0x20;
    let in_range = |base: u32| function_id >= base && function_id < base.wrapping_add(RANGE_WIDTH);
    in_range(0x8400_0000) || in_range(0xc400_0000)
}

/// Resolves one PSCI call. `args` are the guest's `x1`/`x2`/`x3` at the
/// trap, in order — `target_cpu`/`entry_point_address`/`context_id` for
/// `CPU_ON`, the queried function ID in `args[0]` for `PSCI_FEATURES`,
/// unused otherwise.
#[must_use]
pub fn handle_psci_call(function_id: u32, args: [u64; 3]) -> PsciResult {
    match function_id {
        PSCI_VERSION => PsciResult::ReturnValue(PSCI_VERSION_VALUE),
        PSCI_CPU_ON_32 | PSCI_CPU_ON_64 => {
            let target_cpu = args[0];
            if target_cpu == RESIDENT_MPIDR_AFFINITY {
                PsciResult::ReturnValue(RETURN_ALREADY_ON)
            } else {
                PsciResult::ReturnValue(RETURN_INVALID_PARAMS)
            }
        }
        PSCI_CPU_OFF => PsciResult::ReturnValue(RETURN_DENIED),
        PSCI_SYSTEM_OFF => PsciResult::SystemOff,
        PSCI_SYSTEM_RESET => PsciResult::SystemReset,
        PSCI_FEATURES => {
            let queried = args[0] as u32;
            if is_implemented(queried) {
                PsciResult::ReturnValue(RETURN_SUCCESS)
            } else {
                PsciResult::ReturnValue(RETURN_NOT_SUPPORTED)
            }
        }
        _ => PsciResult::ReturnValue(RETURN_NOT_SUPPORTED),
    }
}

fn is_implemented(function_id: u32) -> bool {
    matches!(
        function_id,
        PSCI_VERSION
            | PSCI_CPU_ON_32
            | PSCI_CPU_ON_64
            | PSCI_CPU_OFF
            | PSCI_SYSTEM_OFF
            | PSCI_SYSTEM_RESET
            | PSCI_FEATURES
    )
}

/// The `extern "C"` trampoline entry `backend_macos.c`'s hvc trap loop calls
/// once it has range-tested `x0` as a PSCI function ID
/// (`is_psci_function_id`) — the PSCI mirror of
/// [`crate::dispatch::proxima_vm_dispatch_hypercall`], stateless so it takes
/// no transport pointer. Writes the signed return value into
/// `*return_value_out` (defined only when the return is `0`, meaning
/// "resume the guest"); `*action_out` carries `0` (resume), `1`
/// (`SystemOff`), or `2` (`SystemReset`) so the C loop can end its own
/// dispatch loop without this module knowing anything about hypervisor
/// exit control.
///
/// # Safety
///
/// `return_value_out` and `action_out` must be valid, non-aliasing, writable
/// pointers for the duration of the call — same contract as every other
/// trampoline in this crate (e.g. `mmio_trampoline::proxima_vm_dispatch_mmio`).
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_dispatch_psci(
    function_id: u32,
    arg0: u64,
    arg1: u64,
    arg2: u64,
    return_value_out: *mut i64,
    action_out: *mut u8,
) -> i32 {
    match handle_psci_call(function_id, [arg0, arg1, arg2]) {
        PsciResult::ReturnValue(value) => {
            // SAFETY: caller contract — `return_value_out`/`action_out` are
            // valid, non-aliasing, writable pointers for the duration of
            // this call (same contract as every other trampoline in this
            // crate, e.g. `mmio_trampoline::proxima_vm_dispatch_mmio`).
            unsafe {
                *return_value_out = i64::from(value);
                *action_out = 0;
            }
        }
        PsciResult::SystemOff => {
            // SAFETY: see above.
            unsafe {
                *return_value_out = 0;
                *action_out = 1;
            }
        }
        PsciResult::SystemReset => {
            // SAFETY: see above.
            unsafe {
                *return_value_out = 0;
                *action_out = 2;
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{
        PSCI_CPU_OFF, PSCI_CPU_ON_32, PSCI_CPU_ON_64, PSCI_FEATURES, PSCI_SYSTEM_OFF,
        PSCI_SYSTEM_RESET, PSCI_VERSION, PsciResult, RESIDENT_MPIDR_AFFINITY, handle_psci_call,
        is_psci_function_id,
    };

    #[test]
    fn psci_version_reports_major_zero_minor_two_matching_the_advertised_dtb_compatible() {
        let result = handle_psci_call(PSCI_VERSION, [0, 0, 0]);
        assert_eq!(result, PsciResult::ReturnValue(0x0000_0002));
    }

    #[test]
    fn cpu_on_targeting_the_resident_core_reports_already_on() {
        let result = handle_psci_call(PSCI_CPU_ON_64, [RESIDENT_MPIDR_AFFINITY, 0x4000_0000, 0]);
        assert_eq!(result, PsciResult::ReturnValue(-4), "ALREADY_ON");
    }

    #[test]
    fn cpu_on_32_bit_form_targeting_the_resident_core_also_reports_already_on() {
        let result = handle_psci_call(PSCI_CPU_ON_32, [RESIDENT_MPIDR_AFFINITY, 0x4000_0000, 0]);
        assert_eq!(result, PsciResult::ReturnValue(-4), "ALREADY_ON");
    }

    #[test]
    fn cpu_on_targeting_any_nonresident_core_reports_invalid_params() {
        let nonexistent_second_core = 1_u64;
        let result = handle_psci_call(PSCI_CPU_ON_64, [nonexistent_second_core, 0x4000_0000, 0]);
        assert_eq!(result, PsciResult::ReturnValue(-2), "INVALID_PARAMETERS");
    }

    #[test]
    fn cpu_off_on_the_only_core_is_denied_rather_than_powering_down_the_whole_guest() {
        let result = handle_psci_call(PSCI_CPU_OFF, [0, 0, 0]);
        assert_eq!(result, PsciResult::ReturnValue(-3), "DENIED");
    }

    #[test]
    fn system_off_ends_the_dispatch_loop_instead_of_returning_a_register_value() {
        let result = handle_psci_call(PSCI_SYSTEM_OFF, [0, 0, 0]);
        assert_eq!(result, PsciResult::SystemOff);
    }

    #[test]
    fn system_reset_ends_the_dispatch_loop_since_no_real_reset_exists() {
        let result = handle_psci_call(PSCI_SYSTEM_RESET, [0, 0, 0]);
        assert_eq!(result, PsciResult::SystemReset);
    }

    #[test]
    fn features_reports_success_for_every_function_id_this_handler_implements() {
        for implemented in [
            PSCI_VERSION,
            PSCI_CPU_ON_32,
            PSCI_CPU_ON_64,
            PSCI_CPU_OFF,
            PSCI_SYSTEM_OFF,
            PSCI_SYSTEM_RESET,
            PSCI_FEATURES,
        ] {
            let result = handle_psci_call(PSCI_FEATURES, [u64::from(implemented), 0, 0]);
            assert_eq!(
                result,
                PsciResult::ReturnValue(0),
                "SUCCESS expected for implemented id {implemented:#x}"
            );
        }
    }

    #[test]
    fn features_reports_not_supported_for_an_unimplemented_function_id() {
        let migrate_info_type = 0x8400_0006_u32;
        let result = handle_psci_call(PSCI_FEATURES, [u64::from(migrate_info_type), 0, 0]);
        assert_eq!(result, PsciResult::ReturnValue(-1), "NOT_SUPPORTED");
    }

    #[test]
    fn calling_an_unimplemented_function_id_directly_reports_not_supported() {
        let migrate_32 = 0x8400_0005_u32;
        let result = handle_psci_call(migrate_32, [0, 0, 0]);
        assert_eq!(result, PsciResult::ReturnValue(-1), "NOT_SUPPORTED");
    }

    #[test]
    fn every_implemented_function_id_is_recognized_as_psci_range() {
        for id in [
            PSCI_VERSION,
            PSCI_CPU_ON_32,
            PSCI_CPU_ON_64,
            PSCI_CPU_OFF,
            PSCI_SYSTEM_OFF,
            PSCI_SYSTEM_RESET,
            PSCI_FEATURES,
        ] {
            assert!(is_psci_function_id(id), "{id:#x} must be in PSCI range");
        }
    }

    #[test]
    fn existing_hypercall_verb_space_is_never_mistaken_for_a_psci_function_id() {
        const EMIT_VERB: u32 = 0xfffe;
        const HALT_VERB: u32 = 0xffff;
        for child_request_verb in 0_u32..=4 {
            assert!(!is_psci_function_id(child_request_verb));
        }
        assert!(!is_psci_function_id(EMIT_VERB));
        assert!(!is_psci_function_id(HALT_VERB));
    }
}
