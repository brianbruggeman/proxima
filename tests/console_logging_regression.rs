//! Regression test for the silent-failure trap formerly named
//! `proxima::init_tracing_default` / `proxima::init_tracing`: it built a
//! `TracingLayer` over a `Recorder` backed by `NullPipe`, never registered
//! the process-default recorder, and returned success while discarding every
//! record. Both functions were deleted (no callers existed anywhere in the
//! slot-0 workspace); `proxima::telemetry::export::install_console_logging_with`
//! is the surviving init-shaped entry point.
//!
//! This test proves that entry point actually emits. The sink is real
//! process stdout/stderr, not an injectable writer, so the only faithful way
//! to observe it is a child process whose captured output we can inspect.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::process::Command;
use std::time::Duration;

const CHILD_ENV_VAR: &str = "PROXIMA_CONSOLE_LOGGING_REGRESSION_CHILD";
const MARKER: &str = "console-logging-regression-marker";

#[test]
fn install_console_logging_with_emits_a_record() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        emit_from_child();
        return;
    }

    let exe = env::current_exe().expect("test binary has a path");
    let output = Command::new(exe)
        .arg("install_console_logging_with_emits_a_record")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV_VAR, "1")
        .output()
        .expect("spawn child test process");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stderr.contains(MARKER),
        "install_console_logging_with produced no output; \
         stdout={stdout:?} stderr={stderr:?}"
    );
}

fn emit_from_child() {
    use proxima::telemetry::export::{Formatter, install_console_logging_with};

    install_console_logging_with(Formatter::Text).expect("install console logging");
    proxima_telemetry::error!(MARKER);
    std::thread::sleep(Duration::from_millis(200));
}
