//! Guest that is a network host and nothing else: a virtio-net NIC served by
//! proxima's own sans-IO L2/L3 responder (`proxima_net::stack::handle_frame`,
//! ARP and ICMP echo) over the driver codec in `netdev`. The loop is
//! pipe-shaped: the receive source feeds the stack, and a reply goes to the
//! transmit sink. Idle is a bounded poll — every iteration reads the link
//! status, and the run ends when the host takes the link down.
//!
//! The boot register carries this guest's IPv4 address in its low 32 bits
//! (network order as a number, `10.0.0.2` = `0x0a000002`) and, in bit 32, a
//! switch that disables the stack so received frames are dropped unanswered.
//! The MAC comes from virtio config space, where the device already offers it.

#![no_std]
#![no_main]

#[cfg(target_arch = "x86_64")]
use core::arch::naked_asm;
use core::future::Future;
use core::panic::PanicInfo;
use core::pin::pin;
use core::task::{Context, Poll, Waker};

#[path = "../../hypercall.rs"]
mod hypercall;
mod netdev;

use proxima_net::stack::{Action, handle_frame};
use proxima_primitives::pipe::Pipe;

/// Matches `main.rs`'s `HALT_VERB` and `PROXIMA_VM_HALT_VERB`.
const HALT_VERB: u16 = 0xffff;

const STACK_DISABLED_BIT: u64 = 1 << 32;

#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[unsafe(naked)]
extern "C" fn _start() -> ! {
    naked_asm!(
        "lea rsp, [rip + __stack_top]",
        "mov rdi, rbx",
        "call {entry}",
        entry = sym entry,
    );
}

#[cfg(not(target_arch = "x86_64"))]
#[unsafe(no_mangle)]
extern "C" fn _start() -> ! {
    entry(0)
}

fn ready<Output>(future: impl Future<Output = Output>) -> Option<Output> {
    let mut context = Context::from_waker(Waker::noop());
    match pin!(future).poll(&mut context) {
        Poll::Ready(output) => Some(output),
        Poll::Pending => None,
    }
}

fn frame_bytes(address: u64, length: u32) -> &'static mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(address as *mut u8, length as usize) }
}

fn halt() -> ! {
    let mut empty = [0_u8; 0];
    unsafe { hypercall::hypercall(HALT_VERB, &mut empty) };
    loop {
        core::hint::spin_loop();
    }
}

extern "C" fn entry(boot: u64) -> ! {
    let ip = (boot as u32).to_be_bytes();
    let stack_enabled = boot & STACK_DISABLED_BIT == 0;
    let mac = netdev::read_mac();
    let Ok((rx, tx)) = (unsafe { netdev::bring_up() }) else {
        halt()
    };
    while netdev::link_up() {
        let Some(Ok(received)) = ready(rx.call(())) else {
            break;
        };
        let Some((address, length)) = received else {
            continue;
        };
        if stack_enabled
            && handle_frame(frame_bytes(address, length), mac, ip) == Action::Transmit
            && !matches!(ready(tx.call((address, length))), Some(Ok(())))
        {
            break;
        }
    }
    halt()
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
