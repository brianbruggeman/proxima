//! windows completion port owned by prime.
//!
//! the reactor owns its port directly so foreign overlapped I/O (named pipes, files) can
//! complete on it beside AFD socket polls and wake packets; each dequeued entry is identified
//! by its completion key and `OVERLAPPED` address.

use std::io;
use std::os::windows::io::{
    AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, OwnedHandle, RawHandle,
};
use std::sync::Arc;
use std::time::Duration;

use windows_sys::Win32::Foundation::{
    ERROR_INVALID_PARAMETER, HANDLE, INVALID_HANDLE_VALUE, WAIT_TIMEOUT,
};
use windows_sys::Win32::System::IO::{
    CreateIoCompletionPort, GetQueuedCompletionStatusEx, OVERLAPPED_ENTRY,
    PostQueuedCompletionStatus,
};
use windows_sys::Win32::System::Threading::INFINITE;

/// completion key of a wake packet posted by [`post_to`]; no I/O source holds it.
pub(crate) const WAKE_KEY: usize = usize::MAX;

pub(crate) struct CompletionPort {
    handle: Arc<OwnedHandle>,
    // one setup allocation sized by the reactor event batch; the kernel writes into it
    entries: Box<[OVERLAPPED_ENTRY]>,
}

impl CompletionPort {
    pub(crate) fn new(batch: usize) -> io::Result<Self> {
        let capacity = u32::try_from(batch)
            .ok()
            .filter(|capacity| *capacity > 0)
            .ok_or_else(|| io::Error::from_raw_os_error(ERROR_INVALID_PARAMETER as i32))?;
        // SAFETY: creating a fresh port takes no existing handle; the result is checked below.
        let raw =
            unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, std::ptr::null_mut(), 0, 1) };
        if raw.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `raw` is a valid handle this call just created and nothing else owns.
        let handle = Arc::new(unsafe { OwnedHandle::from_raw_handle(raw as RawHandle) });
        let entries = vec![OVERLAPPED_ENTRY::default(); capacity as usize].into_boxed_slice();
        Ok(Self { handle, entries })
    }

    pub(crate) fn associate(&self, source: BorrowedHandle<'_>, key: usize) -> io::Result<()> {
        // SAFETY: both handles are valid for the call; the kernel drops the association
        // when either handle closes.
        let associated =
            unsafe { CreateIoCompletionPort(source.as_raw_handle() as HANDLE, self.raw(), key, 0) };
        match associated.is_null() {
            true => Err(io::Error::last_os_error()),
            false => Ok(()),
        }
    }

    pub(crate) fn as_handle(&self) -> BorrowedHandle<'_> {
        self.handle.as_handle()
    }

    /// the port handle shared with wake handles that outlive the reactor.
    pub(crate) fn shared_handle(&self) -> Arc<OwnedHandle> {
        Arc::clone(&self.handle)
    }

    pub(crate) fn dequeue(
        &mut self,
        timeout: Option<Duration>,
    ) -> io::Result<&[OVERLAPPED_ENTRY]> {
        let mut removed = 0u32;
        let capacity = self.entries.len() as u32;
        // SAFETY: `entries` holds `capacity` writable slots and `removed` is a valid out-pointer.
        let status = unsafe {
            GetQueuedCompletionStatusEx(
                self.raw(),
                self.entries.as_mut_ptr(),
                capacity,
                &mut removed,
                timeout_millis(timeout),
                0,
            )
        };
        match status {
            0 => self.dequeue_failure(),
            _ => Ok(&self.entries[..removed as usize]),
        }
    }

    fn dequeue_failure(&self) -> io::Result<&[OVERLAPPED_ENTRY]> {
        let error = io::Error::last_os_error();
        match error.raw_os_error() == Some(WAIT_TIMEOUT as i32) {
            true => Ok(&self.entries[..0]),
            false => Err(error),
        }
    }

    fn raw(&self) -> HANDLE {
        self.handle.as_raw_handle() as HANDLE
    }
}

/// queue a wake packet with a null `OVERLAPPED` on `port` from any thread.
pub(crate) fn post_to(port: &OwnedHandle, key: usize) -> io::Result<()> {
    // SAFETY: the port handle is valid for the call; a null OVERLAPPED is the documented wake packet.
    let posted = unsafe {
        PostQueuedCompletionStatus(port.as_raw_handle() as HANDLE, 0, key, std::ptr::null())
    };
    match posted {
        0 => Err(io::Error::last_os_error()),
        _ => Ok(()),
    }
}

fn timeout_millis(timeout: Option<Duration>) -> u32 {
    match timeout {
        None => INFINITE,
        Some(duration) => {
            let millis = duration.as_nanos().div_ceil(1_000_000);
            u32::try_from(millis).map_or(INFINITE - 1, |millis| millis.min(INFINITE - 1))
        }
    }
}

#[cfg(test)]
mod tests {
    use windows_sys::Win32::Foundation::{ERROR_IO_PENDING, ERROR_PIPE_CONNECTED, GENERIC_WRITE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_FLAG_OVERLAPPED, OPEN_EXISTING, PIPE_ACCESS_INBOUND, ReadFile, WriteFile,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;
    use windows_sys::Win32::System::Pipes::{
        ConnectNamedPipe, CreateNamedPipeW, PIPE_TYPE_BYTE, PIPE_WAIT,
    };

    use super::*;
    use crate::core::sized;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn overlapped_pipe_pair(name: &str) -> (OwnedHandle, OwnedHandle) {
        let name = wide(name);
        // SAFETY: `name` is a nul-terminated wide string that outlives the call.
        let server = unsafe {
            CreateNamedPipeW(
                name.as_ptr(),
                PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
                PIPE_TYPE_BYTE | PIPE_WAIT,
                1,
                4096,
                4096,
                0,
                std::ptr::null(),
            )
        };
        assert_ne!(
            server,
            INVALID_HANDLE_VALUE,
            "server pipe: {}",
            io::Error::last_os_error()
        );
        // SAFETY: `server` is a fresh valid handle owned by nothing else.
        let server = unsafe { OwnedHandle::from_raw_handle(server as RawHandle) };
        // SAFETY: same wide-string contract; no security attributes or template.
        let client = unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_WRITE,
                0,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_ne!(
            client,
            INVALID_HANDLE_VALUE,
            "client pipe: {}",
            io::Error::last_os_error()
        );
        // SAFETY: `client` is a fresh valid handle owned by nothing else.
        let client = unsafe { OwnedHandle::from_raw_handle(client as RawHandle) };
        (server, client)
    }

    fn accept_connected_client(server: &OwnedHandle) {
        // SAFETY: the client already opened the pipe, so this returns synchronously.
        let connected =
            unsafe { ConnectNamedPipe(server.as_raw_handle() as HANDLE, std::ptr::null_mut()) };
        let error = io::Error::last_os_error();
        assert!(
            connected != 0 || error.raw_os_error() == Some(ERROR_PIPE_CONNECTED as i32),
            "connect: {error}"
        );
    }

    #[test]
    fn posted_keys_dequeue_in_one_batch() {
        let mut port = CompletionPort::new(sized::REACTOR_EVENT_BATCH).expect("port creation");
        for key in 0..5usize {
            post_to(&port.handle, key).expect("post a wake packet");
        }

        let entries = port.dequeue(Some(Duration::from_secs(5))).expect("dequeue");

        let keys: Vec<usize> = entries.iter().map(|entry| entry.lpCompletionKey).collect();
        assert_eq!(keys, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn timeout_returns_empty_batch() {
        let mut port = CompletionPort::new(sized::REACTOR_EVENT_BATCH).expect("port creation");

        let entries = port
            .dequeue(Some(Duration::from_millis(10)))
            .expect("dequeue");

        assert!(entries.is_empty());
    }

    #[test]
    fn overlapped_pipe_read_completes_on_port() {
        let mut port = CompletionPort::new(sized::REACTOR_EVENT_BATCH).expect("port creation");
        let name = format!(r"\\.\pipe\prime-iocp-{}", std::process::id());
        let (server, client) = overlapped_pipe_pair(&name);
        accept_connected_client(&server);
        port.associate(server.as_handle(), 7)
            .expect("associate read end");
        let mut overlapped = Box::new(OVERLAPPED::default());
        let mut buffer = [0u8; 32];
        // SAFETY: `buffer` and `overlapped` stay alive and unmoved until the completion is dequeued.
        let started = unsafe {
            ReadFile(
                server.as_raw_handle() as HANDLE,
                buffer.as_mut_ptr(),
                32,
                std::ptr::null_mut(),
                &mut *overlapped,
            )
        };
        let pending = io::Error::last_os_error();
        assert!(
            started == 0 && pending.raw_os_error() == Some(ERROR_IO_PENDING as i32),
            "read: {pending}"
        );

        let payload = b"cargo-val";
        let mut written = 0u32;
        // SAFETY: `payload` is valid for its length and the client handle is synchronous.
        let wrote = unsafe {
            WriteFile(
                client.as_raw_handle() as HANDLE,
                payload.as_ptr(),
                payload.len() as u32,
                &mut written,
                std::ptr::null_mut(),
            )
        };
        assert_ne!(wrote, 0, "write: {}", io::Error::last_os_error());
        let entries = port.dequeue(Some(Duration::from_secs(5))).expect("dequeue");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].lpCompletionKey, 7);
        assert_eq!(entries[0].dwNumberOfBytesTransferred, 9);
        assert_eq!(entries[0].lpOverlapped, &mut *overlapped as *mut OVERLAPPED);
        assert_eq!(&buffer[..9], payload);
    }
}
