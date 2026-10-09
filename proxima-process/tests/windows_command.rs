#![cfg(all(windows, feature = "std"))]

use std::fs;
use std::process::Command as NativeCommand;

use proxima_process::Command;

#[test]
fn env_and_cwd_environment_override_matches_native_command() {
    let script = "echo %CARGO_VAL_PROXIMA_OVERRIDE%";
    let mut native = NativeCommand::new("cmd.exe");
    native
        .env_clear()
        .args(["/D", "/C", script])
        .env("CARGO_VAL_PROXIMA_OVERRIDE", "literal environment value");
    let native_output = native.output().expect("run native environment oracle");

    let mut command = Command::new("cmd.exe");
    command
        .env_clear()
        .args(["/D", "/C", script])
        .env("CARGO_VAL_PROXIMA_OVERRIDE", "literal environment value");
    let output = command.output().expect("run Proxima environment command");

    assert_eq!(native_output.status.code(), Some(0));
    assert_eq!(output.status, 0);
    assert_eq!(native_output.stdout, b"literal environment value\r\n");
    assert_eq!(output.stdout, b"literal environment value\r\n");
    assert_eq!(output.stdout, native_output.stdout);
}

#[test]
fn env_and_cwd_removal_matches_native_command() {
    temp_env::with_vars(
        [("CARGO_VAL_PROXIMA_REMOVED", Some("ambient secret"))],
        || {
            let script = "if defined CARGO_VAL_PROXIMA_REMOVED (echo present) else (echo absent)";
            let mut native = NativeCommand::new("cmd.exe");
            native
                .args(["/D", "/C", script])
                .env_remove("CARGO_VAL_PROXIMA_REMOVED");
            let native_output = native.output().expect("run native removal oracle");

            let mut command = Command::new("cmd.exe");
            command
                .args(["/D", "/C", script])
                .env_remove("CARGO_VAL_PROXIMA_REMOVED");
            let output = command.output().expect("run Proxima removal command");

            assert_eq!(native_output.status.code(), Some(0));
            assert_eq!(output.status, 0);
            assert_eq!(native_output.stdout, b"absent\r\n");
            assert_eq!(output.stdout, b"absent\r\n");
            assert_eq!(output.stdout, native_output.stdout);
        },
    );
}

#[test]
fn env_and_cwd_working_directory_matches_native_command() {
    let temporary = tempfile::tempdir().expect("create cwd fixture root");
    let native_directory = temporary.path().join("native-working-directory");
    let command_directory = temporary.path().join("proxima-working-directory");
    fs::create_dir(&native_directory).expect("create native cwd fixture");
    fs::create_dir(&command_directory).expect("create Proxima cwd fixture");
    let script = "echo cargo-val cwd record>payload.txt";

    let mut native = NativeCommand::new("cmd.exe");
    native
        .args(["/D", "/C", script])
        .current_dir(&native_directory);
    let native_status = native.status().expect("run native cwd oracle");

    let mut command = Command::new("cmd.exe");
    command
        .args(["/D", "/C", script])
        .current_dir(&command_directory);
    let status = command.status().expect("run Proxima cwd command");

    assert_eq!(native_status.code(), Some(0));
    assert_eq!(status, 0);
    let native_payload =
        fs::read(native_directory.join("payload.txt")).expect("read native cwd payload");
    let payload =
        fs::read(command_directory.join("payload.txt")).expect("read Proxima cwd payload");
    assert_eq!(native_payload, b"cargo-val cwd record\r\n");
    assert_eq!(payload, b"cargo-val cwd record\r\n");
    assert_eq!(payload, native_payload);
}
