//! Regression tests for the silent-failure trap in `proxima::init_tracing` /
//! `proxima::init_tracing_default`: `init_tracing_default` built a
//! `TracingLayer` over a `Recorder` backed by `NullPipe` (every record
//! discarded at the sink) and neither function ever called
//! `set_default_recorder`, so the crate's most init-shaped name returned
//! `Ok` and emitted nothing via either `tracing::*` or `proxima_telemetry::*`.
//!
//! `proxima::init_telemetry`/`init_telemetry_with` are the well-named
//! crate-root entry points added to replace them, and — unlike the
//! `tracing-init`-only functions they delegate to — work with **zero
//! required features**: proxima-native telemetry (the `proxima_telemetry::*`
//! macros, `#[proxima::instrument]`) needs a recorder + sink + drain, not
//! `tracing-subscriber`. Only the `tracing::`-crate bridge needs
//! `tracing-init`. This file has NO `required-features` in `Cargo.toml`, so
//! it compiles and runs under default features; the tests that specifically
//! exercise the bridge are individually gated with `#[cfg(feature =
//! "tracing-init")]`.
//!
//! Every sink here is real process stdout/stderr, not an injectable writer
//! (except the BYO-recorder case, which composes an injectable
//! `Exporter::file`), so a child process is the only faithful way to
//! observe it — a global `tracing` subscriber can only be set once per
//! process, and nextest/`cargo test` share a process across the other unit
//! tests in this binary.

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

/// THE zero-setup contract, proven under DEFAULT features (no
/// `tracing-init`): `init_telemetry()` alone must make
/// `proxima_telemetry::error!` reach the console. Uses `error!` rather than
/// `info!` because that is genuinely zero-setup — proxima's own emit floor
/// defaults to error-only with no `RUST_LOG` set (by design, see the
/// `proxima-log` skill); asserting on `info!` here would require setting
/// `RUST_LOG`, which is exactly the "extra setup" this test exists to rule
/// out.
#[test]
fn init_telemetry_emits_proxima_native_with_zero_setup_and_zero_features() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        proxima::init_telemetry().expect("init_telemetry");
        proxima_telemetry::error!(MARKER);
        std::thread::sleep(Duration::from_millis(200));
        return;
    }

    let (stdout, stderr) =
        run_in_child("init_telemetry_emits_proxima_native_with_zero_setup_and_zero_features");
    assert!(
        stderr.contains(MARKER),
        "init_telemetry() should emit proxima_telemetry:: records with zero features and zero \
         setup; stdout={stdout:?} stderr={stderr:?}"
    );
}

/// The same contract, with `tracing-init` also on: `init_telemetry()` must
/// now ADDITIONALLY bridge `tracing::` events into the same recorder.
#[cfg(feature = "tracing-init")]
#[test]
fn init_telemetry_also_bridges_tracing_crate_events_with_tracing_init() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        proxima::init_telemetry().expect("init_telemetry");
        proxima_telemetry::error!(MARKER);
        tracing::error!(MARKER);
        std::thread::sleep(Duration::from_millis(200));
        return;
    }

    let (stdout, stderr) =
        run_in_child("init_telemetry_also_bridges_tracing_crate_events_with_tracing_init");
    let occurrences = stderr.matches(MARKER).count();
    assert!(
        occurrences >= 2,
        "with tracing-init on, init_telemetry() should emit for both proxima_telemetry:: and \
         tracing::; saw {occurrences} marker occurrences. stdout={stdout:?} stderr={stderr:?}"
    );
}

/// `install_console_logging_with` is the tracing-init implementation
/// `init_telemetry`/`init_telemetry_with` delegate to when the feature is
/// on — this proves the foundation actually works.
#[cfg(feature = "tracing-init")]
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

/// `install_console_recorder_with` is the always-available (no
/// `tracing-init` needed) implementation `init_telemetry`/`init_telemetry_with`
/// delegate to when the feature is off.
#[test]
fn install_console_recorder_with_emits_a_record() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        use proxima::telemetry::export::{Formatter, install_console_recorder_with};

        install_console_recorder_with(Formatter::Text).expect("install console recorder");
        proxima_telemetry::error!(MARKER);
        std::thread::sleep(Duration::from_millis(200));
        return;
    }

    let (stdout, stderr) = run_in_child("install_console_recorder_with_emits_a_record");
    assert!(
        stderr.contains(MARKER),
        "install_console_recorder_with produced no output; stdout={stdout:?} stderr={stderr:?}"
    );
}

/// Deprecated-alias coverage under default features: `init_tracing_default`
/// must still emit proxima-native records with zero features (this is the
/// historical name people reach for first, per the original bug report — it
/// must not regress back to a no-op in any configuration).
#[test]
#[allow(deprecated)]
fn init_tracing_default_emits_native_with_zero_features() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        proxima::init_tracing_default(proxima::LogFormat::Human).expect("init_tracing_default");
        proxima_telemetry::error!(MARKER);
        std::thread::sleep(Duration::from_millis(200));
        return;
    }

    let (stdout, stderr) = run_in_child("init_tracing_default_emits_native_with_zero_features");
    assert!(
        stderr.contains(MARKER),
        "init_tracing_default should emit proxima_telemetry:: records with zero features; \
         stdout={stdout:?} stderr={stderr:?}"
    );
}

/// Deprecated-alias coverage with `tracing-init` on: `init_tracing_default`
/// must behave exactly like `init_telemetry_with` for both emit paths.
#[cfg(feature = "tracing-init")]
#[test]
#[allow(deprecated)]
fn init_tracing_default_emits_for_both_paths_with_tracing_init() {
    if env::var(CHILD_ENV_VAR).is_ok() {
        proxima::init_tracing_default(proxima::LogFormat::Human).expect("init_tracing_default");
        proxima_telemetry::error!(MARKER);
        tracing::error!(MARKER);
        std::thread::sleep(Duration::from_millis(200));
        return;
    }

    let (stdout, stderr) =
        run_in_child("init_tracing_default_emits_for_both_paths_with_tracing_init");
    let occurrences = stderr.matches(MARKER).count();
    assert!(
        occurrences >= 2,
        "init_tracing_default should emit for both proxima_telemetry:: and tracing:: (it must \
         not regress to the NullPipe/no-set_default_recorder trap); saw {occurrences} marker \
         occurrences. stdout={stdout:?} stderr={stderr:?}"
    );
}

/// `init_tracing(recorder, format)`'s entire purpose is the `tracing::`
/// bridge, so — unlike `init_telemetry` — it genuinely requires
/// `tracing-init`. Without it, it must fail loudly, never silently succeed.
#[cfg(not(feature = "tracing-init"))]
#[test]
fn init_tracing_fails_loudly_without_tracing_init() {
    use proxima_telemetry::export::{Exporter, Formatter};
    use proxima_telemetry::recorder::Recorder;
    use std::sync::Arc;

    let recorder = Recorder::builder()
        .export(Exporter::std().format(Formatter::Text))
        .expect("compose exporter")
        .core_count(1)
        .start()
        .expect("start recorder");
    let result = proxima::init_tracing(Arc::new(recorder), proxima::LogFormat::Human);
    assert!(
        result.is_err(),
        "init_tracing must fail loudly without tracing-init, not silently succeed"
    );
}

/// `init_tracing(recorder, format)` bridges a CALLER-owned recorder: prove
/// it (a) registers that recorder as the ambient default so
/// `proxima_telemetry::*` resolves to it too, and (b) bridges `tracing::*`
/// events into the same recorder, both reaching the recorder's own sink (an
/// injectable file sink here, since this function doesn't hardcode stdout
/// the way `install_console_logging_with` does).
#[cfg(feature = "tracing-init")]
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
