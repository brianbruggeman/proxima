//! P14 parity: proxima's `EnvFilter::parse` + `CompiledEmit::decide` must agree
//! with the REAL `tracing-subscriber::EnvFilter` on a matrix of (target, level)
//! callsites for the same directive string. Truth is the incumbent — this test
//! drives actual `tracing::event!` callsites through a real `EnvFilter` and
//! compares which survive to proxima's decision.
//!
//! Both sides use the ERROR global default: tracing via `EnvFilter::new` (which
//! injects `with_default_directive(LevelFilter::ERROR)`), proxima via its parser
//! default. Gated on `emit`.
#![cfg(feature = "emit")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Arc, Mutex};

use proxima_telemetry::emit::{Coord, Decision, EnvFilter as ProximaEnvFilter};
use proxima_telemetry::level::Level as ProximaLevel;
use tracing::{Event, Id, Level, Metadata, event, span};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

/// Records the (target, level) of every event the EnvFilter lets through.
struct Capturing(Arc<Mutex<Vec<(String, Level)>>>);

impl tracing::Subscriber for Capturing {
    fn new_span(&self, _: &span::Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn event(&self, event: &Event<'_>) {
        let metadata = event.metadata();
        self.0
            .lock()
            .unwrap()
            .push((metadata.target().to_string(), *metadata.level()));
    }
    fn record(&self, _: &Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

fn proxima_level(level: Level) -> ProximaLevel {
    match level {
        Level::TRACE => ProximaLevel::TRACE,
        Level::DEBUG => ProximaLevel::DEBUG,
        Level::INFO => ProximaLevel::INFO,
        Level::WARN => ProximaLevel::WARN,
        Level::ERROR => ProximaLevel::ERROR,
    }
}

#[test]
fn proxima_env_filter_matches_tracing_on_a_matrix() {
    const FILTER: &str = "proxima::h2=debug,proxima::h2::hpack=trace,proxima=info,downstream::store=warn";

    // drive real callsites through the real EnvFilter; capture survivors.
    let captured = Arc::new(Mutex::new(Vec::new()));
    let subscriber = Capturing(Arc::clone(&captured)).with(EnvFilter::new(FILTER));
    tracing::subscriber::with_default(subscriber, || {
        event!(target: "proxima::h2::frame", Level::INFO, x = 1);
        event!(target: "proxima::h2::frame", Level::DEBUG, x = 1);
        event!(target: "proxima::h2::frame", Level::TRACE, x = 1);
        event!(target: "proxima::h2::hpack::evict", Level::TRACE, x = 1);
        event!(target: "proxima::quic", Level::INFO, x = 1);
        event!(target: "proxima::quic", Level::DEBUG, x = 1);
        event!(target: "downstream::store", Level::WARN, x = 1);
        event!(target: "downstream::store", Level::INFO, x = 1);
        event!(target: "other::crate", Level::ERROR, x = 1);
        event!(target: "other::crate", Level::WARN, x = 1);
    });

    let captured = captured.lock().unwrap();
    let tracing_kept =
        |target: &str, level: Level| captured.iter().any(|(t, l)| t == target && *l == level);

    let ours = ProximaEnvFilter::parse(FILTER);
    let our_kept = |target: &str, level: Level| {
        ours.decide(target, Coord::from(proxima_level(level))) == Decision::Keep
    };

    // every cell must agree with the incumbent.
    for (target, level) in [
        ("proxima::h2::frame", Level::INFO),
        ("proxima::h2::frame", Level::DEBUG),
        ("proxima::h2::frame", Level::TRACE),
        ("proxima::h2::hpack::evict", Level::TRACE),
        ("proxima::quic", Level::INFO),
        ("proxima::quic", Level::DEBUG),
        ("downstream::store", Level::WARN),
        ("downstream::store", Level::INFO),
        ("other::crate", Level::ERROR),
        ("other::crate", Level::WARN),
    ] {
        assert_eq!(
            our_kept(target, level),
            tracing_kept(target, level),
            "({target}, {level:?}): proxima kept={}, tracing kept={}",
            our_kept(target, level),
            tracing_kept(target, level),
        );
    }
}
