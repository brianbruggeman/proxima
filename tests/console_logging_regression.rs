//! Regression tests for the silent-failure trap in `proxima::init_tracing` /
//! `proxima::init_tracing_default`: `init_tracing_default` built a
//! `TracingLayer` over a `Recorder` backed by `NullPipe` (every record
//! discarded at the sink) and neither function ever called
//! `set_default_recorder`, so the crate's most init-shaped name returned
//! `Ok` and emitted nothing via either `tracing::*` or `proxima_telemetry::*`.
//!
//! Both functions were fixed to delegate to
//! `proxima_telemetry::export::install_console_logging`/
//! `install_console_logging_with` (the implementation that already worked),
//! and `proxima::init_telemetry`/`init_telemetry_with` were added as the
//! well-named crate-root entry points. Every sink here is real process
//! stdout/stderr, not an injectable writer (except the BYO-recorder case,
//! which composes an injectable `Exporter::writer`), so a child process is
//! the only faithful way to observe it — a global `tracing` subscriber can
//! only be set once per process, and nextest/`cargo test` share a process
//! across the other unit tests in this binary.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::process::Command;
use std::time::Duration;

const CHILD_ENV_VAR: &str = "PROXIMA_CONSOLE_LOGGING_REGRESSION_CHILD";
const MARKER: &str = "console-logging-regression-marker";

/// Runs `test_name` as a re-exec'd child of this same test binary (so a
/// global `tracing` subscriber set inside it doesn't leak into any other
/// test), and returns its captured (stdout, stderr).
fn run_in_child(test_name: &str) -> (String, String) {
    let exe = env::current_exe().expect("test binary has a path");
    let output = Command::new(exe)
        .arg(test_name)
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV_VAR, "1")
        .output()
        .expect("spawn child test process");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// `install_console_logging_with` is the implementation every init entry
/// point above it delegates to — this proves the foundation actually works.
#[test]
fn install_console_logging_with_emits_a_record() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        use proxima::telemetry::export::{Formatter, install_console_logging_with};

        install_console_logging_with(Formatter::Text).expect("install console logging");
        proxima_telemetry::error!(MARKER);
        std::thread::sleep(Duration::from_millis(200));
        return;
    }

    let (stdout, stderr) = run_in_child("install_console_logging_with_emits_a_record");
    assert!(
        stderr.contains(MARKER),
        "install_console_logging_with produced no output; stdout={stdout:?} stderr={stderr:?}"
    );
}

/// The "simple init" contract: zero arguments, no format, no other setup.
/// Uses `error!`/`error!` rather than `info!` because that is genuinely
/// zero-setup — both proxima's own emit floor and the `tracing` bridge
/// default to a `warn`/`error`-only floor with no `RUST_LOG` set (by
/// design, see the `proxima-log` skill); asserting on `info!` here would
/// require setting `RUST_LOG`, which is exactly the "extra setup" this test
/// exists to rule out.
#[test]
fn init_telemetry_emits_for_both_paths_with_zero_setup() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        proxima::init_telemetry().expect("init_telemetry");
        proxima_telemetry::error!(MARKER);
        tracing::error!(MARKER);
        std::thread::sleep(Duration::from_millis(200));
        return;
    }

    let (stdout, stderr) = run_in_child("init_telemetry_emits_for_both_paths_with_zero_setup");
    let occurrences = stderr.matches(MARKER).count();
    assert!(
        occurrences >= 2,
        "init_telemetry() should emit for both proxima_telemetry:: and tracing:: with zero \
         setup; saw {occurrences} marker occurrences. stdout={stdout:?} stderr={stderr:?}"
    );
}

/// Deprecated-alias coverage: `init_tracing_default` must behave exactly
/// like `init_telemetry_with` (this is the historical name people reach for
/// first, per the original bug report — it must not regress back to a
/// no-op).
#[test]
#[allow(deprecated)]
fn init_tracing_default_emits_for_both_paths() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        proxima::init_tracing_default(proxima::LogFormat::Human).expect("init_tracing_default");
        proxima_telemetry::error!(MARKER);
        tracing::error!(MARKER);
        std::thread::sleep(Duration::from_millis(200));
        return;
    }

    let (stdout, stderr) = run_in_child("init_tracing_default_emits_for_both_paths");
    let occurrences = stderr.matches(MARKER).count();
    assert!(
        occurrences >= 2,
        "init_tracing_default should emit for both proxima_telemetry:: and tracing:: (it must \
         not regress to the NullPipe/no-set_default_recorder trap); saw {occurrences} marker \
         occurrences. stdout={stdout:?} stderr={stderr:?}"
    );
}

/// `init_tracing(recorder, format)` bridges a CALLER-owned recorder: prove
/// it (a) registers that recorder as the ambient default so
/// `proxima_telemetry::*` resolves to it too, and (b) bridges `tracing::*`
/// events into the same recorder, both reaching the recorder's own sink (an
/// injectable in-memory writer here, since this function doesn't hardcode
/// stdout the way `install_console_logging_with` does).
#[test]
fn init_tracing_bridges_caller_recorder_for_both_paths() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        use proxima_telemetry::export::{Exporter, Formatter};
        use proxima_telemetry::recorder::Recorder;

        let path = env::temp_dir().join(format!("init-tracing-bridge-{}.log", std::process::id()));
        let recorder = Recorder::builder()
            .export(Exporter::file(&path).format(Formatter::Text))
            .expect("compose file exporter")
            .core_count(1)
            .start()
            .expect("start recorder");
        let recorder = std::sync::Arc::new(recorder);

        proxima::init_tracing(std::sync::Arc::clone(&recorder), proxima::LogFormat::Human)
            .expect("init_tracing");

        proxima_telemetry::error!(MARKER);
        tracing::error!(MARKER);
        std::thread::sleep(Duration::from_millis(200));
        recorder.drain();

        let contents = std::fs::read_to_string(&path).unwrap_or_default();
        let _ = std::fs::remove_file(&path);
        println!("{contents}");
        return;
    }

    let (stdout, stderr) = run_in_child("init_tracing_bridges_caller_recorder_for_both_paths");
    let occurrences = stdout.matches(MARKER).count();
    assert!(
        occurrences >= 2,
        "init_tracing(recorder, format) should register the recorder as ambient default AND \
         bridge tracing:: events into it, both reaching the recorder's own sink; saw \
         {occurrences} marker occurrences. stdout={stdout:?} stderr={stderr:?}"
    );
}
