#![cfg(all(windows, feature = "std"))]

use std::io::{Read, Write};
use std::process::Command as NativeCommand;

use proxima_process::{Command, Stdio};

#[test]
fn stdout_bytes() {
    let mut command = Command::new("cmd.exe");
    command
        .args(["/D", "/C", "echo cargo-val stdout record v1"])
        .stdout(Stdio::Piped);

    let mut child = command.spawn().expect("spawn stdout fixture child");
    let mut stdout_bytes = Vec::new();
    child
        .stdout
        .take()
        .expect("piped stdout handle")
        .read_to_end(&mut stdout_bytes)
        .expect("read fixture stdout bytes");

    assert_eq!(child.wait().expect("wait for stdout fixture child"), 0);
    assert_eq!(stdout_bytes, b"cargo-val stdout record v1\r\n");
}

#[test]
fn stdin_stderr_bytes() {
    let script = "set /p payload=&(echo(!payload!)1>&2";
    let mut oracle = NativeCommand::new("cmd.exe");
    oracle
        .args(["/D", "/V:ON", "/C", script])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let mut oracle_child = oracle.spawn().expect("spawn native stdin/stderr oracle");
    oracle_child
        .stdin
        .take()
        .expect("native piped stdin")
        .write_all(b"cargo-val stdin record v1\r\n")
        .expect("write native input bytes");
    let oracle_output = oracle_child
        .wait_with_output()
        .expect("wait and reap native oracle");

    let mut command = Command::new("cmd.exe");
    command
        .args(["/D", "/V:ON", "/C", script])
        .stdin(Stdio::Piped)
        .stdout(Stdio::Null)
        .stderr(Stdio::Piped);
    let mut child = command.spawn().expect("spawn stdin/stderr fixture child");
    child
        .stdin
        .take()
        .expect("piped child stdin")
        .write_all(b"cargo-val stdin record v1\r\n")
        .expect("write child input bytes");
    let mut stderr_bytes = Vec::new();
    child
        .stderr
        .take()
        .expect("piped child stderr")
        .read_to_end(&mut stderr_bytes)
        .expect("read child stderr bytes");
    assert_eq!(child.wait().expect("wait and reap fixture child"), 0);
    assert_eq!(child.try_wait().expect("read reaped child status"), Some(0));

    assert_eq!(oracle_output.status.code(), Some(0));
    assert_eq!(oracle_output.stdout, b"");
    assert_eq!(oracle_output.stderr, b"cargo-val stdin record v1\r\n");
    assert_eq!(stderr_bytes, b"cargo-val stdin record v1\r\n");
    assert_eq!(stderr_bytes, oracle_output.stderr);
}
