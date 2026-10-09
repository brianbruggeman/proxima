#![cfg(all(windows, feature = "std"))]

use std::fs;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};

use proxima_process::{Command, Stdio};
use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
use windows_sys::Win32::System::Threading::{
    OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE,
};

#[test]
fn spawn_wait_inherited_stdio() {
    let temporary = tempfile::tempdir().expect("create child working directory");
    let mut command = Command::new("cmd.exe");
    command
        .args([
            "/D",
            "/V:ON",
            "/C",
            "(echo cargo-val child record v1)>payload.txt&exit /b 23",
        ])
        .current_dir(temporary.path())
        .stdin(Stdio::Inherit)
        .stdout(Stdio::Inherit)
        .stderr(Stdio::Inherit);

    let mut child = command.spawn().expect("spawn inherited-stdio child");
    let child_pid = child.pid();
    assert_ne!(child_pid, 0, "spawned child has a process identifier");
    let process_handle = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, child_pid) };
    assert!(
        !process_handle.is_null(),
        "open spawned child for lifecycle oracle"
    );
    let process_handle = unsafe { OwnedHandle::from_raw_handle(process_handle) };

    assert_eq!(child.wait().expect("wait and reap child"), 23);
    assert_eq!(
        child.try_wait().expect("read cached reaped status"),
        Some(23)
    );
    assert_eq!(
        unsafe { WaitForSingleObject(process_handle.as_raw_handle(), 0) },
        WAIT_OBJECT_0,
        "the launched child process has exited after wait returned",
    );
    assert_eq!(
        fs::read(temporary.path().join("payload.txt")).expect("read child byte record"),
        b"cargo-val child record v1\r\n"
    );
}
