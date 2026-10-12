//! windows socket readiness through `\Device\Afd`, completing on prime's own [`CompletionPort`].
//!
//! Copyright (c) Stjepan Glavina <stjepang@gmail.com>
//! Copyright (c) John Nunley <dev@notgull.net>
//!
//! Ported from `polling` 3.11.0, `src/iocp/afd.rs` (https://github.com/smol-rs/polling),
//! licensed `Apache-2.0 OR MIT`. polling's LICENSE-MIT and LICENSE-APACHE carry no
//! copyright line; the holders above are the `authors` of the polling 3.11.0 crate manifest.
//! Redistributed here under the MIT license.
//!
//! Differences from the original: the packet/`Completion` trait machinery is replaced by one
//! caller-owned [`AfdSlot`] per socket; ntdll entry points are linked with `raw-dylib`
//! instead of resolved through `GetProcAddress`; the poll is armed without a lock flag
//! because the slot's pending state is the lock.
//!
//! [`CompletionPort`]: super::iocp::CompletionPort

use std::ffi::c_void;
use std::io;
use std::marker::PhantomPinned;
use std::mem::size_of;
use std::os::windows::io::{
    AsRawHandle, BorrowedHandle, FromRawHandle, OwnedHandle, RawHandle, RawSocket,
};
use std::ptr;

use windows_sys::Win32::Foundation::{
    HANDLE, NTSTATUS, STATUS_CANCELLED, STATUS_NOT_FOUND, STATUS_PENDING, STATUS_SUCCESS,
    UNICODE_STRING,
};
use windows_sys::Win32::Networking::WinSock::{
    SIO_BASE_HANDLE, SIO_BSP_HANDLE_POLL, SOCKET_ERROR, WSAIoctl,
};
use windows_sys::Win32::Storage::FileSystem::{FILE_SHARE_READ, FILE_SHARE_WRITE, SYNCHRONIZE};
use windows_sys::Win32::System::IO::{IO_STATUS_BLOCK, OVERLAPPED};

use super::reactor::Interest;

const IOCTL_AFD_POLL: u32 = 0x0001_2024;
const FILE_OPEN: u32 = 1;

const RECEIVE: u32 = 0x001;
const RECEIVE_EXPEDITED: u32 = 0x002;
const SEND: u32 = 0x004;
const DISCONNECT: u32 = 0x008;
const ABORT: u32 = 0x010;
const LOCAL_CLOSE: u32 = 0x020;
const ACCEPT: u32 = 0x080;
const CONNECT_FAIL: u32 = 0x100;

const READABLE_EVENTS: u32 = RECEIVE | ACCEPT | DISCONNECT | RECEIVE_EXPEDITED;
const ERROR_EVENTS: u32 = ABORT | CONNECT_FAIL;

#[repr(C)]
struct ObjectAttributes {
    length: u32,
    root_directory: HANDLE,
    object_name: *const UNICODE_STRING,
    attributes: u32,
    security_descriptor: *const c_void,
    security_quality_of_service: *const c_void,
}

#[repr(C)]
struct AfdPollHandleInfo {
    handle: HANDLE,
    events: u32,
    status: NTSTATUS,
}

#[repr(C)]
struct AfdPollInfo {
    timeout: i64,
    handle_count: u32,
    exclusive: u32,
    handles: [AfdPollHandleInfo; 1],
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    #[link_name = "NtCreateFile"]
    fn nt_create_file(
        file_handle: *mut HANDLE,
        desired_access: u32,
        object_attributes: *const ObjectAttributes,
        io_status_block: *mut IO_STATUS_BLOCK,
        allocation_size: *mut i64,
        file_attributes: u32,
        share_access: u32,
        create_disposition: u32,
        create_options: u32,
        ea_buffer: *mut c_void,
        ea_length: u32,
    ) -> NTSTATUS;

    #[link_name = "NtDeviceIoControlFile"]
    fn nt_device_io_control_file(
        file_handle: HANDLE,
        event: HANDLE,
        apc_routine: *mut c_void,
        apc_context: *mut c_void,
        io_status_block: *mut IO_STATUS_BLOCK,
        io_control_code: u32,
        input_buffer: *mut c_void,
        input_buffer_length: u32,
        output_buffer: *mut c_void,
        output_buffer_length: u32,
    ) -> NTSTATUS;

    #[link_name = "NtCancelIoFileEx"]
    fn nt_cancel_io_file_ex(
        file_handle: HANDLE,
        io_request_to_cancel: *mut IO_STATUS_BLOCK,
        io_status_block: *mut IO_STATUS_BLOCK,
    ) -> NTSTATUS;

    #[link_name = "RtlNtStatusToDosError"]
    fn rtl_nt_status_to_dos_error(status: NTSTATUS) -> u32;
}

/// what a completed AFD poll reported for its socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AfdOutcome {
    Ready { readable: bool, writable: bool },
    LocalClose,
    Cancelled,
    Failed(NTSTATUS),
}

/// one in-flight AFD poll: the kernel writes the status block and poll info in place, so a
/// slot must stay at a fixed address from `poll` until its completion is dequeued.
#[repr(C)]
pub(crate) struct AfdSlot {
    status_block: IO_STATUS_BLOCK,
    info: AfdPollInfo,
    key: usize,
    _pinned: PhantomPinned,
}

impl AfdSlot {
    pub(crate) fn new() -> Self {
        Self {
            status_block: IO_STATUS_BLOCK::default(),
            info: AfdPollInfo {
                timeout: 0,
                handle_count: 0,
                exclusive: 0,
                handles: [AfdPollHandleInfo {
                    handle: ptr::null_mut(),
                    events: 0,
                    status: STATUS_SUCCESS,
                }],
            },
            key: 0,
            _pinned: PhantomPinned,
        }
    }

    pub(crate) fn key(&self) -> usize {
        self.key
    }

    /// the address a completion entry carries in `lpOverlapped` for this slot.
    pub(crate) fn overlapped(&self) -> *mut OVERLAPPED {
        ptr::from_ref(&self.status_block).cast_mut().cast()
    }

    pub(crate) fn is_pending(&self) -> bool {
        self.status() == STATUS_PENDING
    }

    pub(crate) fn outcome(&self) -> AfdOutcome {
        match self.status() {
            STATUS_SUCCESS => self.ready_outcome(),
            STATUS_CANCELLED => AfdOutcome::Cancelled,
            status => AfdOutcome::Failed(status),
        }
    }

    fn ready_outcome(&self) -> AfdOutcome {
        let events = match self.info.handle_count {
            0 => 0,
            _ => self.info.handles[0].events,
        };
        match events & LOCAL_CLOSE != 0 {
            true => AfdOutcome::LocalClose,
            false => AfdOutcome::Ready {
                readable: events & (READABLE_EVENTS | ERROR_EVENTS) != 0,
                writable: events & (SEND | ERROR_EVENTS) != 0,
            },
        }
    }

    fn status(&self) -> NTSTATUS {
        // SAFETY: the kernel may write the status concurrently; a volatile read of the union's
        // status arm, which the block always holds as an initialized integer, is the contract.
        unsafe { ptr::read_volatile(&raw const self.status_block.Anonymous.Status) }
    }
}

impl Default for AfdSlot {
    fn default() -> Self {
        Self::new()
    }
}

/// the `\Device\Afd` handle every slot polls through.
pub(crate) struct Afd {
    handle: OwnedHandle,
}

impl Afd {
    pub(crate) fn new() -> io::Result<Self> {
        let device_name: Vec<u16> = r"\Device\Afd\Smol".encode_utf16().collect();
        let byte_len = u16::try_from(device_name.len() * size_of::<u16>())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let name = UNICODE_STRING {
            Length: byte_len,
            MaximumLength: byte_len,
            Buffer: device_name.as_ptr().cast_mut(),
        };
        let attributes = ObjectAttributes {
            length: size_of::<ObjectAttributes>() as u32,
            root_directory: ptr::null_mut(),
            object_name: &name,
            attributes: 0,
            security_descriptor: ptr::null(),
            security_quality_of_service: ptr::null(),
        };
        let mut handle: HANDLE = ptr::null_mut();
        let mut status_block = IO_STATUS_BLOCK::default();
        // SAFETY: `name`, `device_name` and `attributes` outlive the call and every pointer
        // argument is valid for the call; the handle is only used when the status is success.
        let status = unsafe {
            nt_create_file(
                &mut handle,
                SYNCHRONIZE,
                &attributes,
                &mut status_block,
                ptr::null_mut(),
                0,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                FILE_OPEN,
                0,
                ptr::null_mut(),
                0,
            )
        };
        check(status)?;
        // SAFETY: NtCreateFile succeeded, so `handle` is a fresh handle nothing else owns.
        let handle = unsafe { OwnedHandle::from_raw_handle(handle as RawHandle) };
        Ok(Self { handle })
    }

    /// borrow the device handle so the reactor can associate it with a completion port.
    pub(crate) fn as_handle(&self) -> BorrowedHandle<'_> {
        // SAFETY: `handle` is open for as long as `self` is, which bounds the borrow.
        unsafe { BorrowedHandle::borrow_raw(self.handle.as_raw_handle()) }
    }

    /// arm one poll of `socket` for `interest`; success and pending both queue a completion
    /// packet on the port the device is associated with.
    ///
    /// # Safety
    ///
    /// `slot` must not move or drop until its completion is dequeued or `cancel` has
    /// completed, and no poll may already be pending on it.
    pub(crate) unsafe fn poll(
        &self,
        slot: &mut AfdSlot,
        socket: RawSocket,
        interest: Interest,
        key: usize,
    ) -> io::Result<()> {
        let base = base_socket(socket)?;
        slot.key = key;
        slot.info.exclusive = 0;
        slot.info.handle_count = 1;
        slot.info.timeout = i64::MAX;
        slot.info.handles[0].handle = base as HANDLE;
        slot.info.handles[0].status = STATUS_SUCCESS;
        slot.info.handles[0].events = afd_mask(interest);
        slot.status_block.Anonymous.Status = STATUS_PENDING;
        let block: *mut IO_STATUS_BLOCK = &raw mut slot.status_block;
        let info: *mut c_void = (&raw mut slot.info).cast();
        // SAFETY: the caller guarantees `slot` stays put; block and info are the slot's own
        // fields, and the status block doubles as the apc context the completion returns.
        let status = unsafe {
            nt_device_io_control_file(
                self.handle.as_raw_handle() as HANDLE,
                ptr::null_mut(),
                ptr::null_mut(),
                block.cast(),
                block,
                IOCTL_AFD_POLL,
                info,
                size_of::<AfdPollInfo>() as u32,
                info,
                size_of::<AfdPollInfo>() as u32,
            )
        };
        match status {
            STATUS_SUCCESS | STATUS_PENDING => Ok(()),
            failed => check(failed),
        }
    }

    /// cancel a pending poll; the slot still completes, with `AfdOutcome::Cancelled`.
    ///
    /// # Safety
    ///
    /// `slot` must be the one armed through this device and still alive.
    pub(crate) unsafe fn cancel(&self, slot: &AfdSlot) -> io::Result<()> {
        if !slot.is_pending() {
            return Ok(());
        }
        let mut cancel_block = IO_STATUS_BLOCK::default();
        // SAFETY: the request block is the slot's live status block; `cancel_block` is a
        // valid out-parameter for the call.
        let status = unsafe {
            nt_cancel_io_file_ex(
                self.handle.as_raw_handle() as HANDLE,
                ptr::from_ref(&slot.status_block).cast_mut(),
                &mut cancel_block,
            )
        };
        match status {
            STATUS_SUCCESS | STATUS_NOT_FOUND => Ok(()),
            failed => check(failed),
        }
    }
}

fn afd_mask(interest: Interest) -> u32 {
    let (readable, writable) = match interest {
        Interest::Read => (true, false),
        Interest::Write => (false, true),
        Interest::ReadWrite => (true, true),
    };
    let mut mask = ABORT | CONNECT_FAIL;
    if readable {
        mask |= READABLE_EVENTS;
    }
    if writable {
        mask |= SEND;
    }
    mask
}

fn check(status: NTSTATUS) -> io::Result<()> {
    match status {
        STATUS_SUCCESS => Ok(()),
        failed => {
            // SAFETY: a pure status-to-error translation with no pointer arguments.
            let code = unsafe { rtl_nt_status_to_dos_error(failed) };
            Err(io::Error::from_raw_os_error(code as i32))
        }
    }
}

/// resolve the base service-provider socket AFD needs; layered providers wrap it.
pub(crate) fn base_socket(socket: RawSocket) -> io::Result<RawSocket> {
    match socket_ioctl(socket, SIO_BASE_HANDLE) {
        Ok(base) => Ok(base),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => Err(error),
        Err(_) => {
            let polled = socket_ioctl(socket, SIO_BSP_HANDLE_POLL)?;
            if polled == socket {
                return Err(io::Error::from(io::ErrorKind::InvalidInput));
            }
            socket_ioctl(polled, SIO_BASE_HANDLE)
        }
    }
}

fn socket_ioctl(socket: RawSocket, control_code: u32) -> io::Result<RawSocket> {
    let mut output: RawSocket = 0;
    let mut returned = 0u32;
    // SAFETY: `output` and `returned` are valid out-pointers sized as passed; no overlapped
    // or completion routine is supplied, so the call is synchronous.
    let status = unsafe {
        WSAIoctl(
            socket as usize,
            control_code,
            ptr::null(),
            0,
            ptr::from_mut(&mut output).cast(),
            size_of::<RawSocket>() as u32,
            &mut returned,
            ptr::null_mut(),
            None,
        )
    };
    match status == SOCKET_ERROR {
        true => Err(io::Error::last_os_error()),
        false => Ok(output),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::os::windows::io::AsRawSocket;
    use std::time::Duration;

    use super::*;
    use crate::core::sized;
    use crate::os::iocp::CompletionPort;

    const AFD_DEVICE_KEY: usize = 1;
    const SLOT_KEY: usize = 42;

    fn connected_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
        let client = TcpStream::connect(listener.local_addr().expect("listener address"))
            .expect("connect to loopback listener");
        let (server, _) = listener.accept().expect("accept loopback connection");
        (server, client)
    }

    fn port_with_afd() -> (CompletionPort, Afd) {
        let port = CompletionPort::new(sized::REACTOR_EVENT_BATCH).expect("port creation");
        let afd = Afd::new().expect("open the afd device");
        port.associate(afd.as_handle(), AFD_DEVICE_KEY)
            .expect("associate afd with the port");
        (port, afd)
    }

    #[test]
    fn afd_reports_readable_after_peer_write() {
        let (mut port, afd) = port_with_afd();
        let (server, mut client) = connected_pair();
        let mut slot = AfdSlot::new();
        // SAFETY: `slot` stays on this frame, unmoved, until its completion is dequeued below.
        unsafe { afd.poll(&mut slot, server.as_raw_socket(), Interest::Read, SLOT_KEY) }
            .expect("arm the afd poll");
        assert!(
            slot.is_pending(),
            "no data yet, so the poll must be pending"
        );

        client
            .write_all(b"GET /health HTTP/1.1\r\n\r\n")
            .expect("peer write");
        let entries = port.dequeue(Some(Duration::from_secs(5))).expect("dequeue");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].lpCompletionKey, AFD_DEVICE_KEY);
        assert_eq!(entries[0].lpOverlapped, slot.overlapped());
        assert_eq!(slot.key(), SLOT_KEY);
        assert_eq!(
            slot.outcome(),
            AfdOutcome::Ready {
                readable: true,
                writable: false
            }
        );
    }

    #[test]
    fn afd_cancel_completes_with_cancelled_status() {
        let (mut port, afd) = port_with_afd();
        let (server, _client) = connected_pair();
        let mut slot = AfdSlot::new();
        // SAFETY: `slot` stays on this frame, unmoved, until its completion is dequeued below.
        unsafe { afd.poll(&mut slot, server.as_raw_socket(), Interest::Read, SLOT_KEY) }
            .expect("arm the afd poll");

        // SAFETY: `slot` is the live slot armed through this device just above.
        unsafe { afd.cancel(&slot) }.expect("cancel the pending poll");
        let entries = port.dequeue(Some(Duration::from_secs(5))).expect("dequeue");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].lpOverlapped, slot.overlapped());
        assert_eq!(slot.outcome(), AfdOutcome::Cancelled);
        assert!(!slot.is_pending());
    }
}
