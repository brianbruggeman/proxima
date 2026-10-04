#![cfg(all(windows, feature = "std"))]

use std::env;
use std::ffi::{CString, OsString};
use std::fs;
use std::io::{Read, Write};
use std::mem::size_of;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::pin::Pin;
use std::process::Command as NativeCommand;
use std::sync::mpsc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::executor::block_on;
use futures::{Stream, StreamExt};
use proxima_primitives::pipe::{ProximaError, Request, RequestStream, SendPipe};
use proxima_process::grounds::Deny;
use proxima_process::spawn::spawn;
use proxima_process::{Command, CommandDescriptor, SpawnOptions, Stdio};
use windows_sys::Win32::Foundation::{INVALID_HANDLE_VALUE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS,
};
use windows_sys::Win32::System::Threading::{
    OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
};

fn shell() -> OsString {
    env::var_os("COMSPEC").unwrap_or_else(|| OsString::from("cmd.exe"))
}

fn command(script: &str) -> Command {
    let mut command = Command::new(shell());
    command.args(["/D", "/V:ON", "/C", script]);
    command
}

#[test]
fn windows_process_output_and_exit() {
    let script = "(echo proxima child stdout)&(echo proxima child stderr)1>&2&exit /b 23";
    let oracle = NativeCommand::new(shell())
        .env_clear()
        .args(["/D", "/V:ON", "/C", script])
        .output()
        .expect("native output oracle");
    let mut child = command(script);
    let output = child.output().expect("collect actual child output");
    assert_eq!(output.stdout, b"proxima child stdout\r\n");
    assert_eq!(output.stderr, b"proxima child stderr\r\n");
    assert_eq!(output.status, 23);
    assert_eq!(output.stdout, oracle.stdout);
    assert_eq!(output.stderr, oracle.stderr);
    assert_eq!(Some(output.status), oracle.status.code());
    assert_eq!(child.get_stdout(), Stdio::Inherit);
    assert_eq!(child.get_stderr(), Stdio::Inherit);
}

#[test]
fn windows_process_stdin_round_trip() {
    let mut command = command("set /p payload=&echo(!payload!");
    command
        .stdin(Stdio::Piped)
        .stdout(Stdio::Piped)
        .stderr(Stdio::Null);
    let mut child = command.spawn().expect("spawn stdin echo child");
    assert!(child.pid() > 0);
    let mut input = child.stdin.take().expect("piped child stdin");
    input
        .write_all(b"proxima stdin payload\n")
        .expect("write child stdin");
    drop(input);
    let mut output = Vec::new();
    child
        .stdout
        .take()
        .expect("piped child stdout")
        .read_to_end(&mut output)
        .expect("read child output");
    assert_eq!(output, b"proxima stdin payload\r\n");
    assert_eq!(child.wait().expect("wait child"), 0);
    assert_eq!(child.try_wait().expect("cached child exit"), Some(0));
}

#[test]
fn windows_process_environment_directory_and_kill() {
    let directory = tempfile::tempdir().expect("create child directory fixture");
    let working = directory.path().join("café-日本");
    fs::create_dir(&working).expect("create Unicode working directory");
    let mut writer = command("(echo %PROXIMA_PORT_VALUE%)>payload.txt");
    writer
        .current_dir(&working)
        .env("PROXIMA_PORT_VALUE", "proxima env payload");
    assert_eq!(
        writer.status().expect("child writes environment payload"),
        0
    );
    assert_eq!(
        fs::read(working.join("payload.txt")).expect("independent file oracle"),
        b"proxima env payload\r\n"
    );
    for remove in [false, true] {
        let script = "if defined PROXIMA_SECRET (echo !PROXIMA_SECRET!) else (echo absent)";
        let mut native = NativeCommand::new(shell());
        native.env_clear().args(["/D", "/V:ON", "/C", script]);
        native
            .env("PROXIMA_SECRET", "first")
            .env("proxima_secret", "second")
            .env("PROXIMA_SECRET", "final");
        let mut mixed = command(script);
        mixed
            .env("PROXIMA_SECRET", "first")
            .env("proxima_secret", "second")
            .env("PROXIMA_SECRET", "final");
        if remove {
            native.env_remove("Proxima_Secret");
            mixed.env_remove("Proxima_Secret");
        }
        let expected = if remove {
            b"absent\r\n".as_slice()
        } else {
            b"final\r\n".as_slice()
        };
        let oracle = native
            .output()
            .expect("native mixed-case environment oracle");
        assert_eq!(oracle.stdout, expected);
        assert_eq!(
            mixed
                .output()
                .expect("process mixed-case environment")
                .stdout,
            oracle.stdout
        );
    }

    temp_env::with_vars([("PROXIMA_DESCRIPTOR_SECRET", Some("hidden"))], || {
        let mut descriptor_command = command("exit /b 0");
        descriptor_command
            .inherit_current_env()
            .env_remove("proxima_descriptor_secret");
        let descriptor = descriptor_command
            .to_descriptor()
            .expect("lower case-insensitive environment removal");
        assert!(!descriptor
            .env
            .iter()
            .any(|entry| entry.key.to_bytes().eq_ignore_ascii_case(b"PROXIMA_DESCRIPTOR_SECRET")));
    });

    let mut waiting = command("set /p payload=");
    waiting
        .stdin(Stdio::Piped)
        .stdout(Stdio::Null)
        .stderr(Stdio::Null);
    let mut child = waiting.spawn().expect("spawn waiting child");
    assert_eq!(child.try_wait().expect("child waits for piped input"), None);
    child.kill().expect("terminate waiting child");
    assert_ne!(child.wait().expect("reap terminated child"), 0);
    child.kill().expect("kill after reap is harmless");
}

#[test]
fn windows_process_errors_and_unsupported() {
    let directory = tempfile::tempdir().expect("create missing executable fixture");
    let missing = directory.path().join("missing.exe");
    let oracle = NativeCommand::new(&missing)
        .spawn()
        .expect_err("native missing program");
    match Command::new(&missing)
        .spawn()
        .expect_err("process missing program")
    {
        ProximaError::Io(error) => {
            assert_eq!(error.kind(), oracle.kind());
            assert_eq!(error.raw_os_error(), oracle.raw_os_error());
        }
        other => panic!("expected retained process OS error, got {other:?}"),
    }
    assert!(
        Command::new(shell())
            .arg("invalid\0argument")
            .spawn()
            .is_err()
    );
    assert!(command("exit /b 0").umask(0o022).spawn().is_err());
    assert!(command("exit /b 0").controlling_tty(true).spawn().is_err());
    assert!(command("exit /b 0").libc_shim().spawn().is_err());
    assert!(
        command("exit /b 0")
            .dispatch(Deny::new(libc::EACCES))
            .spawn()
            .is_err()
    );
    assert!(command("exit /b 0").stdin(Stdio::Fd(1)).spawn().is_err());
    let descriptor = CommandDescriptor::new(CString::new("cmd.exe").expect("descriptor program"));
    for options in [
        SpawnOptions {
            dispatch_fd: Some(7),
            ..SpawnOptions::default()
        },
        SpawnOptions {
            controlling_tty: true,
            ..SpawnOptions::default()
        },
        SpawnOptions {
            umask: Some(0o022),
            ..SpawnOptions::default()
        },
    ] {
        assert!(spawn(&descriptor, options).is_err());
    }
}

#[test]
fn windows_process_pipe_round_trip_and_error() {
    let mut command = command(
        "(for /L %i in (1,1,8192) do @echo stderr payload 1>&2)&set /p payload=&echo(!payload!",
    );
    command.stderr(Stdio::Piped);
    let request = Request::builder()
        .method("POST")
        .path("/")
        .payload(Bytes::from_static(b"proxima pipe payload\n"))
        .build()
        .expect("pipe request");
    let response = block_on(command.call(request)).expect("spawn process pipe");
    let mut stream = response.into_chunk_stream();
    let output = block_on(async {
        let mut output = Vec::new();
        while let Some(chunk) = stream.next().await {
            output.extend_from_slice(&chunk.expect("actual child output chunk"));
        }
        output
    });
    assert_eq!(output, b"proxima pipe payload\r\n");
    let directory = tempfile::tempdir().expect("create missing pipe executable fixture");
    let missing = Command::new(directory.path().join("missing.exe"));
    let request = Request::builder()
        .method("POST")
        .path("/")
        .build()
        .expect("empty request");
    assert!(block_on(missing.call(request)).is_err());
    dropped_pipe_terminates_waiting_child();
}

struct PendingInput(mpsc::Sender<()>);

impl Stream for PendingInput {
    type Item = Result<Bytes, ProximaError>;

    fn poll_next(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

impl Drop for PendingInput {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

fn direct_child_handle() -> OwnedHandle {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
        assert_ne!(snapshot, INVALID_HANDLE_VALUE);
        let snapshot = unsafe { OwnedHandle::from_raw_handle(snapshot) };
        let mut entry = PROCESSENTRY32W {
            dwSize: size_of::<PROCESSENTRY32W>() as u32,
            ..PROCESSENTRY32W::default()
        };
        let mut available = unsafe { Process32FirstW(snapshot.as_raw_handle(), &mut entry) };
        while available != 0 {
            if entry.th32ParentProcessID == std::process::id() {
                let process = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, entry.th32ProcessID) };
                assert!(!process.is_null(), "open independent child process handle");
                let process = unsafe { OwnedHandle::from_raw_handle(process) };
                if unsafe { WaitForSingleObject(process.as_raw_handle(), 0) } == WAIT_TIMEOUT {
                    return process;
                }
            }
            available = unsafe { Process32NextW(snapshot.as_raw_handle(), &mut entry) };
        }
        assert!(
            Instant::now() < deadline,
            "waiting child absent from OS process snapshot"
        );
        std::thread::yield_now();
    }
}

fn dropped_pipe_terminates_waiting_child() {
    let (dropped, observed_drop) = mpsc::channel();
    let request = Request::builder()
        .method("POST")
        .path("/")
        .stream(RequestStream::new(PendingInput(dropped)))
        .build()
        .expect("pending input request");
    let response =
        block_on(command("set /p payload=").call(request)).expect("start silent waiting child");
    let child = direct_child_handle();
    assert_eq!(
        unsafe { WaitForSingleObject(child.as_raw_handle(), 0) },
        WAIT_TIMEOUT
    );
    drop(response);
    assert_eq!(
        unsafe { WaitForSingleObject(child.as_raw_handle(), 10_000) },
        WAIT_OBJECT_0,
        "dropping an unread response must terminate the actual waiting child"
    );
    observed_drop
        .recv_timeout(Duration::from_secs(10))
        .expect("pending input future released after cancellation");
}
