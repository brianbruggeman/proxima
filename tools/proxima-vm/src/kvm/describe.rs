use super::trap::{RunError, Trap, VcpuExit};
use super::vcpu::Vcpu;

const KVM_EXIT_IO: u32 = 2;
const KVM_EXIT_MMIO: u32 = 6;
const KVM_EXIT_HLT: u32 = 5;
const KVM_EXIT_SHUTDOWN: u32 = 8;
const KVM_EXIT_INTR: u32 = 10;

fn exit_reason(exit: &VcpuExit) -> u32 {
    match exit {
        VcpuExit::PortOut { .. } | VcpuExit::PortIn { .. } => KVM_EXIT_IO,
        VcpuExit::MmioRead { .. } | VcpuExit::MmioWrite { .. } => KVM_EXIT_MMIO,
        VcpuExit::Hlt => KVM_EXIT_HLT,
        VcpuExit::Shutdown => KVM_EXIT_SHUTDOWN,
        VcpuExit::Interrupted | VcpuExit::RunFailed { .. } => KVM_EXIT_INTR,
        VcpuExit::FailEntry { .. } => kvm_bindings::KVM_EXIT_FAIL_ENTRY,
        VcpuExit::InternalError { .. } => kvm_bindings::KVM_EXIT_INTERNAL_ERROR,
        VcpuExit::Other { reason } => *reason,
    }
}

/// Positional evidence for an exit nothing claimed: what the exit was and
/// where the guest was executing.
#[must_use]
pub fn unexpected_exit(exit: &VcpuExit, rip: u64) -> String {
    match exit {
        VcpuExit::FailEntry { hardware_reason } => {
            format!("KVM entry failed: hardware reason 0x{hardware_reason:x} at rip=0x{rip:x}")
        }
        VcpuExit::InternalError { suberror } => format!("KVM internal error: suberror {suberror} at rip=0x{rip:x}"),
        other => format!("unexpected KVM exit reason {} at rip=0x{rip:x}", exit_reason(other)),
    }
}

/// The text of a run that did not end cleanly, with `rip` read from `vcpu`
/// when the failure is one a position explains.
#[must_use]
pub fn describe_failure(error: &RunError, vcpu: Option<&Vcpu>) -> String {
    let rip = || vcpu.map_or(0, Vcpu::instruction_pointer);
    match error {
        RunError::Watchdog => format!(
            "watchdog forced exit: guest never trapped within the watchdog window, still executing at rip=0x{:x}",
            rip()
        ),
        RunError::Declined(Trap::Exit(exit)) => unexpected_exit(exit, rip()),
        RunError::Declined(Trap::WatchdogFired) => describe_failure(&RunError::Watchdog, vcpu),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unclaimed_hlt_names_kvm_exit_reason_5_and_the_instruction_pointer() {
        assert_eq!(unexpected_exit(&VcpuExit::Hlt, 1), "unexpected KVM exit reason 5 at rip=0x1");
    }

    #[test]
    fn a_triple_fault_names_kvm_exit_reason_8() {
        assert_eq!(unexpected_exit(&VcpuExit::Shutdown, 0x20), "unexpected KVM exit reason 8 at rip=0x20");
    }

    #[test]
    fn a_failed_entry_carries_the_hardware_reason() {
        assert_eq!(
            unexpected_exit(&VcpuExit::FailEntry { hardware_reason: 0x80000021 }, 0x1000),
            "KVM entry failed: hardware reason 0x80000021 at rip=0x1000"
        );
    }

    #[test]
    fn an_internal_error_carries_the_suberror() {
        assert_eq!(
            unexpected_exit(&VcpuExit::InternalError { suberror: 1 }, 0x2),
            "KVM internal error: suberror 1 at rip=0x2"
        );
    }

    #[test]
    fn the_watchdog_message_names_itself_and_the_stuck_rip() {
        let message = describe_failure(&RunError::Watchdog, None);

        assert!(message.starts_with("watchdog forced exit") && message.ends_with("rip=0x0"), "{message}");
    }
}
