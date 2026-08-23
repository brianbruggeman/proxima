//! The out-of-band fault channel for telemetry-internals failures.
//!
//! The drainer, the instrument registry, and the drain-thread spawn path each
//! fail *inside* the machinery that drains the recorder's own ring. Routing
//! their failure logs through `error!`/`warn!` — as a workspace-wide
//! tracing-elimination pass briefly did — feeds the very ring those paths
//! just failed to drain: under sustained failure, every reported fault would
//! enqueue another record onto the backlog it is reporting on, and nothing
//! would ever drain it. [`report_fault`] writes straight to `stderr` with no
//! [`crate::recorder::Recorder`], no [`crate::export::Exporter`], and no ring
//! anywhere in the call path, so it cannot self-reference no matter how
//! sustained the failure is.
//!
//! This is why it does not follow the workspace's usual "emit a structured
//! event and point a file-sink `Exporter` at it" rule for forensics: that
//! rule assumes the exporter/recorder is working. This function exists for
//! the case where it is not — it IS the failure path of the exporter and the
//! drainer, so it must not depend on either.
//!
//! `std`-only: it needs a real `stderr` handle. Every call site lives under
//! this crate's `std` feature gate already (`legacy`, `recorder::drainer`,
//! `recorder::registry`, and `recorder::mod` are all `#[cfg(feature =
//! "std")]` in `lib.rs`), so nothing is forced down a tier to reach this.

use std::io::Write;

/// Report a telemetry-internals failure directly to `stderr`, bypassing the
/// recorder entirely. See the module doc for why this must not route through
/// `error!`/`warn!` or any [`crate::export::Exporter`].
pub fn report_fault(message: &str) {
    let _ = writeln!(std::io::stderr(), "[proxima-telemetry] {message}");
}

// builds a real Recorder, which uses proxima-core's Ring/StaticRing internally
// -- cfg-swapped to loom under `--features loom`, only usable inside an actual
// loom::model(...) closure, which this plain #[test] function doesn't provide
// (mirrors the same gate on `export.rs`'s test module).
#[cfg(all(test, not(feature = "loom")))]
mod tests {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use crate::export::default_recorder;
    use crate::recorder::Recorder;

    #[derive(Clone)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    // proves the regression this module fixes cannot recur: a fault reported
    // while a recorder is installed must NOT land any record in that
    // recorder's own ring. Without the fix (routing through `error!`/`warn!`
    // instead of straight to stderr) this test fails, because the fault would
    // enqueue onto the very ring it is reporting a failure about.
    #[test]
    fn report_fault_does_not_reenter_the_installed_recorder() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let recorder = Recorder::builder()
            .export(crate::export::Exporter::writer(SharedBuf(buf.clone())))
            .unwrap()
            .core_count(1)
            .install()
            .unwrap();

        super::report_fault("pipe dispatch error during drain: synthetic failure");
        recorder.drain();

        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(
            out.is_empty(),
            "the fault channel must not write into the recorder's own ring: {out}"
        );
        assert!(
            default_recorder().is_some(),
            "sanity: a recorder actually was installed for this test"
        );
    }
}
