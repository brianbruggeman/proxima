#![cfg(all(windows, feature = "std"))]

use std::io::Read;

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
