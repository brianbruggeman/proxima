//! proxima-native log macros — `error!` / `warn!` / `info!` / `debug!` / `trace!`.
//!
//! Drop-in for `tracing`'s level macros: they accept the same call syntax
//! (`?x` Debug, `%x` Display, `field = expr`, bare-ident shorthand, leading
//! message literal) but record into proxima's own recorder, gated by proxima's
//! own runtime filter ([`crate::emit::global`], default `error`,
//! `RUST_LOG`-overridable) — no `tracing` dependency.
//!
//! Each callsite owns a `static` [`CallsiteGate`](crate::emit::CallsiteGate):
//! the first hit at a given filter generation runs the filter once and caches
//! the decision; later hits are two atomic loads. A disabled callsite never
//! touches the recorder, never formats a field, never allocates. The callsite
//! stays compiled in (the compile floor is `trace`), so raising `RUST_LOG`
//! lights it up with no rebuild.
//!
//! Use via the explicit path: `use proxima_telemetry::{debug, warn};`.
//!
//! Every level macro also accepts a leading `recorder = <expr>` form —
//! `error!(recorder = rec, "msg", field = val)` — that emits into `rec`
//! (a `&Recorder`) instead of [`crate::export::default_recorder()`], with no
//! ambient fallback. This mirrors `counter!`/`gauge!`/`histogram!`/`updown!`'s
//! `recorder = rec` seam (`crate::metric`) and `#[instrument(recorder = rec)]`:
//! EXPLICIT by design, so a `capture()`-scoped recorder never depends on
//! process-global state and two `capture()` calls never see each other's
//! logs. The callsite filter gate still applies — `recorder = rec` changes
//! WHERE an enabled log goes, not WHETHER it is enabled. Plain call sites
//! (no `recorder =`) are byte-for-byte unchanged.

/// Gate a callsite against the global filter, then collect fields and emit.
#[macro_export]
#[doc(hidden)]
macro_rules! __emit {
    ($level:expr, recorder = $recorder:expr $(, $($args:tt)*)?) => {{
        static __GATE: $crate::emit::CallsiteGate = $crate::emit::CallsiteGate::new();
        if __GATE.is_enabled(
            $crate::emit::global::current_generation(),
            || $crate::emit::global::decide(
                ::core::module_path!(),
                $crate::emit::Coord::from($level),
            ),
        ) {
            $crate::__emit_collect!(@scan $level, "", [], [$recorder], $($($args)*)?);
        } else {
            $crate::__emit_admit!($level, [$recorder], $($($args)*)?);
        }
    }};
    ($level:expr, $($args:tt)*) => {{
        static __GATE: $crate::emit::CallsiteGate = $crate::emit::CallsiteGate::new();
        if __GATE.is_enabled(
            $crate::emit::global::current_generation(),
            || $crate::emit::global::decide(
                ::core::module_path!(),
                $crate::emit::Coord::from($level),
            ),
        ) {
            $crate::__emit_collect!(@scan $level, "", [], [], $($args)*);
        } else {
            $crate::__emit_admit!($level, [], $($args)*);
        }
    }};
}

/// The below-floor admit branch: when a callsite is gated OFF, still construct
/// and emit the record for a verbose-sampled trace at/above the elevated depth,
/// so `ElevationSink` can replay it. Feature-off, this expands to nothing — the
/// `else` arm is empty and the build is unchanged.
#[cfg(feature = "elevation")]
#[macro_export]
#[doc(hidden)]
macro_rules! __emit_admit {
    ($level:expr, [$($sink:tt)*], $($args:tt)*) => {
        if $crate::current::should_admit_below_floor($level) {
            $crate::__emit_collect!(@scan $level, "", [], [$($sink)*], $($args)*);
        }
    };
}

#[cfg(not(feature = "elevation"))]
#[macro_export]
#[doc(hidden)]
macro_rules! __emit_admit {
    ($($args:tt)*) => {};
}

/// TT-muncher: walk the call args, route the message literal and each field
/// form into a builder chain, then build + emit on the target recorder.
///
/// The 4th slot `[$($sink:tt)*]` threads the emit target through the scan
/// unchanged: `[]` means "the ambient default recorder" (unchanged
/// behavior), `[$recorder]` means "this explicit recorder, no ambient
/// fallback" (see the `recorder = <expr>` form on each level macro).
#[macro_export]
#[doc(hidden)]
macro_rules! __emit_collect {
    // terminal, ambient default recorder — build + emit (no-op when no
    // recorder is installed). module_path!() here expands at the macro
    // invocation site, so the log carries the emitting module (the span
    // pillar already captured it; the log pillar did not).
    (@scan $level:expr, $msg:expr, [$($tags:tt)*], [] $(,)?) => {
        if let ::core::option::Option::Some(__rec) = $crate::export::default_recorder() {
            __rec.log().level($level).message($msg).module_path(::core::module_path!()) $($tags)* .emit();
        }
    };
    // terminal, explicit recorder — emit into `$recorder` only.
    (@scan $level:expr, $msg:expr, [$($tags:tt)*], [$recorder:expr] $(,)?) => {
        $recorder.log().level($level).message($msg).module_path(::core::module_path!()) $($tags)* .emit();
    };
    // message literal (overrides the default empty message)
    (@scan $level:expr, $msg:expr, [$($tags:tt)*], [$($sink:tt)*], $next:literal $(, $($rest:tt)*)?) => {
        $crate::__emit_collect!(@scan $level, $next, [$($tags)*], [$($sink)*] $(, $($rest)*)?)
    };
    // field = ?expr  (Debug-formatted)
    (@scan $level:expr, $msg:expr, [$($tags:tt)*], [$($sink:tt)*], $key:ident = ?$val:expr $(, $($rest:tt)*)?) => {
        $crate::__emit_collect!(@scan $level, $msg,
            [$($tags)* .tag(::core::stringify!($key), $crate::emit::global::fmt_bytes(::core::format_args!("{:?}", &$val)))],
            [$($sink)*]
            $(, $($rest)*)?)
    };
    // field = %expr  (Display-formatted)
    (@scan $level:expr, $msg:expr, [$($tags:tt)*], [$($sink:tt)*], $key:ident = %$val:expr $(, $($rest:tt)*)?) => {
        $crate::__emit_collect!(@scan $level, $msg,
            [$($tags)* .tag(::core::stringify!($key), $crate::emit::global::fmt_bytes(::core::format_args!("{}", &$val)))],
            [$($sink)*]
            $(, $($rest)*)?)
    };
    // field = expr  (typed scalar)
    (@scan $level:expr, $msg:expr, [$($tags:tt)*], [$($sink:tt)*], $key:ident = $val:expr $(, $($rest:tt)*)?) => {
        $crate::__emit_collect!(@scan $level, $msg,
            [$($tags)* .tag(::core::stringify!($key), $val)],
            [$($sink)*]
            $(, $($rest)*)?)
    };
    // ?expr  (Debug, field name = expression text)
    (@scan $level:expr, $msg:expr, [$($tags:tt)*], [$($sink:tt)*], ?$val:expr $(, $($rest:tt)*)?) => {
        $crate::__emit_collect!(@scan $level, $msg,
            [$($tags)* .tag(::core::stringify!($val), $crate::emit::global::fmt_bytes(::core::format_args!("{:?}", &$val)))],
            [$($sink)*]
            $(, $($rest)*)?)
    };
    // %expr  (Display, field name = expression text)
    (@scan $level:expr, $msg:expr, [$($tags:tt)*], [$($sink:tt)*], %$val:expr $(, $($rest:tt)*)?) => {
        $crate::__emit_collect!(@scan $level, $msg,
            [$($tags)* .tag(::core::stringify!($val), $crate::emit::global::fmt_bytes(::core::format_args!("{}", &$val)))],
            [$($sink)*]
            $(, $($rest)*)?)
    };
    // bare ident shorthand (field name = ident, value = ident)
    (@scan $level:expr, $msg:expr, [$($tags:tt)*], [$($sink:tt)*], $key:ident $(, $($rest:tt)*)?) => {
        $crate::__emit_collect!(@scan $level, $msg,
            [$($tags)* .tag(::core::stringify!($key), $key)],
            [$($sink)*]
            $(, $($rest)*)?)
    };
}

#[macro_export]
macro_rules! error {
    ($($args:tt)*) => { $crate::__emit!($crate::level::Level::ERROR, $($args)*) };
}

#[macro_export]
macro_rules! warn {
    ($($args:tt)*) => { $crate::__emit!($crate::level::Level::WARN, $($args)*) };
}

#[macro_export]
macro_rules! info {
    ($($args:tt)*) => { $crate::__emit!($crate::level::Level::INFO, $($args)*) };
}

#[macro_export]
macro_rules! debug {
    ($($args:tt)*) => { $crate::__emit!($crate::level::Level::DEBUG, $($args)*) };
}

#[macro_export]
macro_rules! trace {
    ($($args:tt)*) => { $crate::__emit!($crate::level::Level::TRACE, $($args)*) };
}

// these tests build real Recorders, which use proxima-core's Ring/
// StaticRing internally -- cfg-swapped to loom under `--features loom`
// (forwarded via proxima-core/loom), only usable inside an actual
// loom::model(...) closure, which these plain #[test] functions don't
// provide.
#[cfg(all(test, not(feature = "loom")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    extern crate std;

    use std::future::Future;
    use std::sync::{Arc, Mutex};
    use std::vec::Vec;

    use bytes::Bytes;
    use proxima_primitives::pipe::ProximaError;
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::request::Response;

    use crate::emit::{EnvFilter, global};
    use crate::export::set_default_recorder;
    use crate::level::Level;

    // global::install and set_default_recorder both mutate process-wide
    // statics; the test harness runs #[test] fns in parallel by default,
    // so every test in this module (including recorder_routing below)
    // takes this lock first to serialize around that shared state.
    // poison-recovery: an earlier test's genuine assertion failure must
    // not cascade-fail every later test via a poisoned lock.
    static GLOBAL_STATE_LOCK: Mutex<()> = Mutex::new(());

    fn lock_global_state() -> std::sync::MutexGuard<'static, ()> {
        GLOBAL_STATE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
    use crate::log::{LogBody, LogRecord};
    use crate::pipes::{TelemetryRecord, TelemetryRequest};
    use crate::recorder::Recorder;
    use crate::tag::{ScalarValue, Tag};

    // a terminal telemetry pipe that keeps every LogRecord for assertions —
    // the collecting sibling of CountingPipe.
    #[derive(Clone)]
    struct CollectingPipe {
        logs: Arc<Mutex<Vec<LogRecord>>>,
    }

    impl SendPipe for CollectingPipe {
        type In = TelemetryRequest;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: TelemetryRequest,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let sink = Arc::clone(&self.logs);
            async move {
                match request.payload {
                    TelemetryRecord::Log(record) => sink.lock().unwrap().push(record),
                    TelemetryRecord::LogBatch(records) => sink.lock().unwrap().extend(records),
                    _ => {}
                }
                Ok(Response::ok(Bytes::new()))
            }
        }
    }

    // install a collecting recorder as process default; return it + the captured log vec.
    fn collecting_recorder() -> (Arc<Recorder>, Arc<Mutex<Vec<LogRecord>>>) {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let pipe = CollectingPipe {
            logs: Arc::clone(&logs),
        };
        let recorder = Arc::new(
            Recorder::builder()
                .pipe(pipe)
                .core_count(1)
                .start()
                .expect("recorder build"),
        );
        set_default_recorder(Arc::clone(&recorder));
        (recorder, logs)
    }

    fn drain(recorder: &Recorder) {
        while recorder.drain() > 0 {}
    }

    #[test]
    fn default_floor_is_error_so_debug_drops_and_error_records() {
        let _guard = lock_global_state();
        global::install(EnvFilter::parse(""));
        let (recorder, logs) = collecting_recorder();

        debug!("a per-datagram debug line nobody asked for");
        error!("a contract violation worth surfacing");
        drain(&recorder);

        let captured = logs.lock().unwrap();
        assert_eq!(
            captured.len(),
            1,
            "only the error must record at the default error floor"
        );
        assert_eq!(captured[0].level, Level::ERROR);
    }

    #[test]
    fn rust_log_style_filter_raises_the_floor_to_debug() {
        let _guard = lock_global_state();
        global::install(EnvFilter::parse("debug"));
        let (recorder, logs) = collecting_recorder();

        debug!("now i am wanted");
        drain(&recorder);

        assert_eq!(
            logs.lock().unwrap().len(),
            1,
            "debug records once the floor is debug"
        );
    }

    #[test]
    fn every_tracing_field_form_records_a_tagged_message() {
        let _guard = lock_global_state();
        global::install(EnvFilter::parse("trace"));
        let (recorder, logs) = collecting_recorder();

        let peer = "127.0.0.1:443";
        let len: usize = 1200;
        let err = "connection reset";
        let handle: u64 = 7;

        debug!("bare message only");
        debug!(?err, "debug-formatted field");
        debug!(label = %peer, "display-formatted field");
        debug!(len, %peer, "bare-ident shorthand plus display");
        debug!(handle = handle, "typed scalar field");
        warn!(label = %peer, len, reason = ?err, "mixed multi-field");
        warn!(?err, handle = handle, "multi-line multi-field");
        drain(&recorder);

        let captured = logs.lock().unwrap();
        assert_eq!(captured.len(), 7, "all seven field forms must record");
    }

    #[test]
    fn field_forms_map_to_the_expected_tag_kinds() {
        let _guard = lock_global_state();
        global::install(EnvFilter::parse("trace"));
        let (recorder, logs) = collecting_recorder();

        let err = "reset by peer";
        let len: usize = 1200;
        debug!(?err, len, "recv");
        drain(&recorder);

        let captured = logs.lock().unwrap();
        let record = &captured[0];
        assert!(matches!(record.body, LogBody::Text("recv")));

        let debug_tag = record
            .attrs
            .iter()
            .find(|tag| matches!(tag, Tag::Scalar { key: "err", .. }));
        match debug_tag {
            Some(Tag::Scalar {
                value: ScalarValue::Bytes(bytes),
                ..
            }) => {
                assert_eq!(&bytes[..], b"\"reset by peer\"", "?expr formats via Debug");
            }
            other => panic!("err tag must be Debug-formatted bytes, got {other:?}"),
        }

        let typed_tag = record
            .attrs
            .iter()
            .find(|tag| matches!(tag, Tag::Scalar { key: "len", .. }));
        assert!(
            matches!(
                typed_tag,
                Some(Tag::Scalar {
                    value: ScalarValue::U64(1200),
                    ..
                })
            ),
            "bare usize ident maps to a typed U64 tag, got {typed_tag:?}"
        );
        let _ = len;
    }

    // C9 (log pillar): a bare `error!(...)` only ever reaches the ambient
    // default recorder, so a `capture()`-scoped test recorder can't observe
    // it. `recorder = rec` is the OTHER view -- an explicit, non-ambient seam
    // into a recorder's own log sink, mirroring `counter!`/`gauge!`/
    // `histogram!`/`updown!`'s `recorder = rec` (see `crate::metric::tests::recorder_routing`).
    mod recorder_routing {
        use super::lock_global_state;
        use crate::capture::capture;
        use crate::emit::{EnvFilter, global};
        use crate::level::Level;
        use crate::log::LogBody;
        use crate::tag::{ScalarValue, Tag};

        // error! with `recorder = rec`: level, message, and typed field all land
        // in the capture()-scoped recorder's drained log.
        #[test]
        fn error_macro_routes_to_explicit_recorder() {
            let _guard = lock_global_state();
            global::install(EnvFilter::parse("trace"));
            let captured = capture(|rec| {
                error!(recorder = rec, code = 500u64, "explicit error");
            });
            let logs = captured.logs();
            assert_eq!(logs.len(), 1, "dump: {}", captured.dump());
            assert_eq!(logs[0].level, Level::ERROR);
            assert!(matches!(logs[0].body, LogBody::Text("explicit error")));
            assert!(
                logs[0].attrs.iter().any(|tag| matches!(
                    tag,
                    Tag::Scalar {
                        key: "code",
                        value: ScalarValue::U64(500)
                    }
                )),
                "dump: {}",
                captured.dump()
            );
        }

        // warn! with `recorder = rec`.
        #[test]
        fn warn_macro_routes_to_explicit_recorder() {
            let _guard = lock_global_state();
            global::install(EnvFilter::parse("trace"));
            let captured = capture(|rec| {
                warn!(recorder = rec, "explicit warn");
            });
            let logs = captured.logs();
            assert_eq!(logs.len(), 1, "dump: {}", captured.dump());
            assert_eq!(logs[0].level, Level::WARN);
            assert!(matches!(logs[0].body, LogBody::Text("explicit warn")));
        }

        // info! with `recorder = rec`.
        #[test]
        fn info_macro_routes_to_explicit_recorder() {
            let _guard = lock_global_state();
            global::install(EnvFilter::parse("trace"));
            let captured = capture(|rec| {
                info!(recorder = rec, "explicit info");
            });
            let logs = captured.logs();
            assert_eq!(logs.len(), 1, "dump: {}", captured.dump());
            assert_eq!(logs[0].level, Level::INFO);
            assert!(matches!(logs[0].body, LogBody::Text("explicit info")));
        }

        // debug! with `recorder = rec`.
        #[test]
        fn debug_macro_routes_to_explicit_recorder() {
            let _guard = lock_global_state();
            global::install(EnvFilter::parse("trace"));
            let captured = capture(|rec| {
                debug!(recorder = rec, "explicit debug");
            });
            let logs = captured.logs();
            assert_eq!(logs.len(), 1, "dump: {}", captured.dump());
            assert_eq!(logs[0].level, Level::DEBUG);
            assert!(matches!(logs[0].body, LogBody::Text("explicit debug")));
        }

        // trace! with `recorder = rec`.
        #[test]
        fn trace_macro_routes_to_explicit_recorder() {
            let _guard = lock_global_state();
            global::install(EnvFilter::parse("trace"));
            let captured = capture(|rec| {
                trace!(recorder = rec, "explicit trace");
            });
            let logs = captured.logs();
            assert_eq!(logs.len(), 1, "dump: {}", captured.dump());
            assert_eq!(logs[0].level, Level::TRACE);
            assert!(matches!(logs[0].body, LogBody::Text("explicit trace")));
        }

        // a plain call (no `recorder =`) inside a capture() body only ever targets
        // the ambient default recorder, which is unset here -- proves `recorder =`
        // is the only seam into a capture()-scoped recorder, not accidental cross-talk.
        #[test]
        fn plain_call_inside_capture_does_not_reach_the_capture_recorder() {
            let _guard = lock_global_state();
            global::install(EnvFilter::parse("trace"));
            let captured = capture(|_rec| {
                error!("no recorder given, targets the ambient default only");
            });
            assert_eq!(
                captured.logs().len(),
                0,
                "plain call must not leak into an unrelated capture() recorder; dump: {}",
                captured.dump()
            );
        }

        // two independent `capture()` calls, each its own private recorder, no
        // process-global touched -- never see each other's routed logs. This is
        // the isolation guarantee: the RECORDER side of `recorder = rec` is scoped
        // to whichever recorder the caller explicitly handed it.
        #[test]
        fn explicit_recorder_routing_is_isolated_across_captures() {
            let _guard = lock_global_state();
            global::install(EnvFilter::parse("trace"));

            let first = capture(|rec| {
                error!(recorder = rec, tenant = 1u64, "first capture's log");
            });
            let second = capture(|rec| {
                error!(recorder = rec, tenant = 2u64, "second capture's log");
            });

            let first_logs = first.logs();
            let second_logs = second.logs();

            assert_eq!(first_logs.len(), 1, "dump: {}", first.dump());
            assert_eq!(second_logs.len(), 1, "dump: {}", second.dump());
            assert!(matches!(
                first_logs[0].body,
                LogBody::Text("first capture's log")
            ));
            assert!(matches!(
                second_logs[0].body,
                LogBody::Text("second capture's log")
            ));
            assert!(
                first_logs[0].attrs.iter().any(|tag| matches!(
                    tag,
                    Tag::Scalar {
                        key: "tenant",
                        value: ScalarValue::U64(1)
                    }
                )),
                "first capture sees only its own tenant tag; dump: {}",
                first.dump()
            );
            assert!(
                second_logs[0].attrs.iter().any(|tag| matches!(
                    tag,
                    Tag::Scalar {
                        key: "tenant",
                        value: ScalarValue::U64(2)
                    }
                )),
                "second capture sees only its own tenant tag, never the first's; dump: {}",
                second.dump()
            );
        }
    }
}
