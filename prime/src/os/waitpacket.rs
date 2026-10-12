//! wait completion packets: the thread-free way to turn a waitable handle into a port packet.
//!
//! `NtAssociateWaitCompletionPacket` makes the kernel queue one completion on the reactor's
//! port when the target handle signals, so an exited process or a set event wakes a task
//! without a thread-pool wait. The entry points are not in
//! windows-sys, so they are linked with `raw-dylib` like the AFD calls in [`super::afd`].
//! A packet is one-shot: after it fires, a signaled handle needs no second association.

use std::ffi::c_void;
use std::io;
use std::os::windows::io::{AsRawHandle, BorrowedHandle, FromRawHandle, OwnedHandle, RawHandle};
use std::ptr;

use windows_sys::Win32::Foundation::{GENERIC_ALL, HANDLE, NTSTATUS};

use super::afd::check;

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    #[link_name = "NtCreateWaitCompletionPacket"]
    fn nt_create_wait_completion_packet(
        packet: *mut HANDLE,
        desired_access: u32,
        object_attributes: *const c_void,
    ) -> NTSTATUS;

    #[link_name = "NtAssociateWaitCompletionPacket"]
    fn nt_associate_wait_completion_packet(
        packet: HANDLE,
        port: HANDLE,
        target: HANDLE,
        key_context: *mut c_void,
        apc_context: *mut c_void,
        io_status: NTSTATUS,
        io_status_information: usize,
        already_signaled: *mut u8,
    ) -> NTSTATUS;

    #[link_name = "NtCancelWaitCompletionPacket"]
    fn nt_cancel_wait_completion_packet(packet: HANDLE, remove_signaled_packet: u8) -> NTSTATUS;
}

pub(crate) fn create() -> io::Result<OwnedHandle> {
    let mut packet: HANDLE = ptr::null_mut();
    // SAFETY: `packet` is a valid out-pointer and a null OBJECT_ATTRIBUTES requests an unnamed object.
    let status = unsafe { nt_create_wait_completion_packet(&mut packet, GENERIC_ALL, ptr::null()) };
    check(status)?;
    // SAFETY: the kernel just returned this handle and nothing else owns it.
    Ok(unsafe { OwnedHandle::from_raw_handle(packet as RawHandle) })
}

/// queue a packet on `port` carrying `key` when `target` signals. `true` means `target` was
/// already signaled, so the caller treats the gate as open instead of waiting for a packet.
pub(crate) fn associate(
    packet: BorrowedHandle<'_>,
    port: BorrowedHandle<'_>,
    target: BorrowedHandle<'_>,
    key: usize,
) -> io::Result<bool> {
    let mut already_signaled = 0u8;
    // SAFETY: all three handles are valid for the call and `already_signaled` is a valid out-pointer.
    let status = unsafe {
        nt_associate_wait_completion_packet(
            packet.as_raw_handle() as HANDLE,
            port.as_raw_handle() as HANDLE,
            target.as_raw_handle() as HANDLE,
            key as *mut c_void,
            ptr::null_mut(),
            0,
            0,
            &mut already_signaled,
        )
    };
    check(status)?;
    Ok(already_signaled != 0)
}

/// disassociate the packet and drop any packet it already queued, so a stale completion
/// cannot reach a recycled slot.
pub(crate) fn cancel(packet: BorrowedHandle<'_>) -> io::Result<()> {
    // SAFETY: the packet handle is valid for the call.
    let status = unsafe { nt_cancel_wait_completion_packet(packet.as_raw_handle() as HANDLE, 1) };
    match status {
        status if status >= 0 => Ok(()),
        failed => check(failed),
    }
}
