#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `boundary` — one config-selected control point for integrating into, then
//! replacing, an existing service.
//!
//! You are standing an existing service (`theirs`) next to your replacement
//! (`ours`) and want ONE place you set and forget, then flip via config as the
//! migration walks forward. That place is a `PipeHandle` slot: whatever pipe
//! sits in it IS the boundary's behaviour. A config enum picks which built-in
//! pipe fills the slot; nothing here is a new abstraction.
//!
//! The four built-in strategies are each an existing pipe, `Request -> Response`
//! unchanged:
//!
//! - `Off` -> the inner handle, untouched. Zero-cost pass-through: you have
//!   cut over to `ours` (or are still fronting `theirs`) and want no boundary
//!   machinery at all.
//! - `Record` -> `RecordUpstream` tees every (request, response) to a
//!   cassette as it flows. Capture `theirs` in production — the golden
//!   oracle.
//! - `Replay` -> `ReplayUpstream` serves that cassette back byte-identical,
//!   no upstream call. Feed the recorded requests to `ours` in CI and a diff
//!   is meaningful; an uncaptured request is a typed miss.
//! - `Shadow` -> `Diff` fans out to both and reports where they disagree
//!   (200 identical / 409 diverged). Run `ours` against `theirs` live
//!   without trusting it yet.
//!
//! There is no `Observe` strategy, on purpose: observation is not a boundary
//! pipe, it is `#[proxima::instrument]` — orthogonal, sprinkled on the code and
//! left on in EVERY mode. A boundary "observe pipe" would be `#[instrument]`
//! wearing a costume.
//!
//! And there is no `trait BoundaryStrategy`: a strategy is a
//! `SendPipe<Request, Response>`, full stop. Extending the boundary is not
//! subclassing an abstraction, it is writing (or composing) a pipe. Section 5
//! proves it — a fifth strategy, tagging every response `theirs` serves with
//! a migration marker header, is just `Transform` wrapped around `theirs`
//! and dropped into the same slot: no new type, no new enum arm required.
//!
//! # A gotcha the boundary runs straight into: `Pipe` vs `SendPipe`
//!
//! The obvious fifth strategy to reach for is
//! `proxima_primitives::pipe::Fallback` — "front `theirs`, and when it's
//! down, serve the last-known-good answer off the cassette instead." It
//! reads like the natural composition. It does not compile here, and the
//! reason is worth understanding before you compose your own pipes: `Fallback`
//! implements only the root `Pipe` trait (`!Send`, borrow-shaped), never
//! `SendPipe` (the cross-core, `'static`, erasable form) — and there is NO
//! blanket bridge from one to the other (RTN, the language feature that
//! would let one exist, is unstable — rust#109417). `into_handle`, the only
//! door into a `PipeHandle`, demands `SendPipe` exactly. So `Fallback`
//! cannot fill this slot, full stop, and hand-rolling a one-off `SendPipe`
//! impl just to force it through would BE the "new type" this example is
//! busy proving you don't need. `Transform` already implements `SendPipe`
//! over a `PipeHandle` inner, so it goes through `into_handle` unchanged —
//! that's why section 5 reaches for it instead.
//!
//! Run: `cargo run --example boundary`

use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use proxima::pipe::PipeHandle;
use proxima::runtime::PrimeRuntime;
use proxima::{
    AccumulatingSink, Diff, DynRecordingSink, FormatKind, HttpEvent, JsonlSource, LazyFanOut,
    ProtocolEvent, ProximaError, RecordUpstream, RecordingEvent, RecordingSource, ReplayUpstream,
    Request, Response, ResponseOp, Runtime, SendPipe, SinkSpec, SynthUpstream, TerminalSignal,
    Transform, deferred_runtime, into_handle,
};

// ── the config menu: a named, conflaguration-flippable built-in strategy ─────

/// Which built-in pipe fills the boundary slot. A closed menu on purpose — an
/// enum has to enumerate its values — but adding a named strategy is one arm
/// here plus one arm in [`wire_boundary`]. For a strategy that ISN'T on the
/// menu, don't touch the enum: compose a pipe and hand it to the slot directly
/// (section 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum BoundaryMode {
    Off,
    Record,
    Replay,
    Shadow,
}

impl FromStr for BoundaryMode {
    type Err = ParseError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "off" => Ok(Self::Off),
            "record" => Ok(Self::Record),
            "replay" => Ok(Self::Replay),
            "shadow" => Ok(Self::Shadow),
            other => Err(ParseError(format!(
                "unknown boundary mode `{other}` (expected off|record|replay|shadow)"
            ))),
        }
    }
}

/// Parse error for the mode field; conflaguration's `resolve_with` demands a
/// `std::error::Error` on the parser's error type.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ParseError(String);

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

fn parse_boundary_mode(raw: &str) -> Result<BoundaryMode, ParseError> {
    raw.parse()
}

fn default_mode() -> BoundaryMode {
    BoundaryMode::Off
}

/// The boundary's config surface — the house pattern (`cassette_config.rs`):
/// `#[derive(Builder, Deserialize, Serialize, Settings)]`, resolved from
/// `PROXIMA_BOUNDARY_MODE` (or a config file) so the strategy flips without a
/// recompile. Set the boundary once; conflaguration turns it on/off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_BOUNDARY")]
#[builder(derive(Clone, Debug))]
struct BoundaryConfig {
    #[setting(default_str = "off", resolve_with = "parse_boundary_mode")]
    #[serde(default = "default_mode")]
    #[builder(default = default_mode())]
    mode: BoundaryMode,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            mode: default_mode(),
        }
    }
}

impl Validate for BoundaryConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        Ok(())
    }
}

// ── the boundary: one slot, one match, every arm an existing pipe ────────────

/// `TerminalSignal<Predicate>` is generic over its terminal-event predicate
/// (`proxima-recording/src/pipe/terminal_signal.rs:30`); a non-capturing
/// closure would still be an anonymous per-callsite type, so `is_http_ended`
/// below is a plain `fn` item instead — its value coerces to the concrete,
/// nameable `fn(&RecordingEvent) -> bool` pointer type, which is what lets
/// this alias (and `wire_boundary`'s return type) name it at all.
type BoundaryTerminalSignal = TerminalSignal<fn(&RecordingEvent) -> bool>;

fn is_http_ended(event: &RecordingEvent) -> bool {
    matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. }))
}

/// Everything a boundary arm might need. `ours`/`theirs` are cheap-to-clone
/// handles; `cassette` and `runtime` feed the record/replay arms.
struct BoundaryDeps {
    ours: PipeHandle,
    theirs: PipeHandle,
    cassette: std::path::PathBuf,
    runtime: Arc<dyn Runtime>,
}

/// Resolve a mode to the pipe that fills the boundary slot. The return is a
/// single `PipeHandle` for every arm (they erase through `into_handle`), plus
/// an optional drain signal that only `Record` produces — the caller awaits it
/// to know the cassette is durable.
async fn wire_boundary(
    mode: BoundaryMode,
    deps: BoundaryDeps,
) -> Result<(PipeHandle, Option<Arc<BoundaryTerminalSignal>>), ProximaError> {
    match mode {
        // cut over (or still fronting theirs): the inner handle, nothing added.
        BoundaryMode::Off => Ok((deps.ours, None)),

        // capture theirs — the golden oracle — as traffic flows.
        BoundaryMode::Record => {
            let (sink, terminal) = build_cassette_sink(&deps.cassette, Arc::clone(&deps.runtime));
            let recorder = RecordUpstream::new("boundary", deps.theirs, sink, "http");
            Ok((into_handle(recorder), Some(terminal)))
        }

        // serve the recording back byte-identical; no inner call happens.
        BoundaryMode::Replay => {
            let replay =
                ReplayUpstream::from_jsonl(&deps.cassette, "boundary", deps.runtime).await?;
            Ok((into_handle(replay), None))
        }

        // run ours against theirs live; report where they disagree.
        BoundaryMode::Shadow => Ok((into_handle(Diff::new(deps.theirs, deps.ours)), None)),
    }
}

/// The record/replay sink stack from `examples/record`: a `LazyFanOut` to a
/// jsonl cassette, batched by an `AccumulatingSink`, wrapped in a
/// `TerminalSignal` that fires once the interaction's terminal event is durable
/// — so the caller awaits a signal instead of repolling the file.
fn build_cassette_sink(
    cassette: &Path,
    runtime: Arc<dyn Runtime>,
) -> (DynRecordingSink, Arc<BoundaryTerminalSignal>) {
    let spigot = deferred_runtime();
    spigot.set(runtime).ok();
    let durable = Arc::new(LazyFanOut::new(
        vec![SinkSpec::new(cassette.to_string_lossy(), FormatKind::Json)],
        spigot,
    ));
    let accumulating: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
    let terminal = Arc::new(TerminalSignal::new(
        accumulating,
        is_http_ended as fn(&RecordingEvent) -> bool,
    ));
    (terminal.clone(), terminal)
}

// ── the two services standing side by side ───────────────────────────────────

fn ours() -> PipeHandle {
    into_handle(SynthUpstream::new("ours", 200, "served by ours"))
}

fn theirs() -> PipeHandle {
    into_handle(SynthUpstream::new("theirs", 200, "served by theirs"))
}

fn chat_request() -> Request<Bytes> {
    Request::builder()
        .method("POST")
        .path("/v1/chat")
        .body("what is a cassette?")
        .build()
        .expect("request builds")
}

async fn body_of(response: Response<Bytes>) -> String {
    let bytes = response.collect_body().await.expect("collect body");
    String::from_utf8_lossy(&bytes).into_owned()
}

#[proxima::main(cores = 1)]
async fn main() -> Result<(), ProximaError> {
    println!("boundary: one config-selected slot to integrate-then-replace a service\n");

    let cassette_dir = tempfile::tempdir()?;
    let cassette = cassette_dir.path().join("boundary.jsonl");
    let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1)?);

    // the flip point: read the configured mode once. `PROXIMA_BOUNDARY_MODE`
    // (or a config file) selects it — set and forget, then conflaguration it.
    let configured = BoundaryConfig::from_env()
        .map_err(|error| ProximaError::Config(format!("boundary config: {error}")))?;
    println!(
        "configured boundary mode (PROXIMA_BOUNDARY_MODE): {:?}\n",
        configured.mode
    );

    section_off(&cassette, &runtime).await?;
    section_record(&cassette, &runtime).await?;
    section_replay(&cassette, &runtime).await?;
    section_shadow(&cassette, &runtime).await?;
    section_extend_with_a_composed_strategy().await?;

    println!("\nPASS: one PipeHandle slot, config-selected; every strategy an existing pipe.");
    Ok(())
}

fn deps(cassette: &Path, runtime: &Arc<dyn Runtime>) -> BoundaryDeps {
    BoundaryDeps {
        ours: ours(),
        theirs: theirs(),
        cassette: cassette.to_path_buf(),
        runtime: Arc::clone(runtime),
    }
}

// ── 1. Off: the slot is the inner handle, nothing added ──────────────────────

async fn section_off(cassette: &Path, runtime: &Arc<dyn Runtime>) -> Result<(), ProximaError> {
    println!("--- 1. Off: pass-through to ours, zero boundary machinery ---");
    let (boundary, _) = wire_boundary(BoundaryMode::Off, deps(cassette, runtime)).await?;
    let served = body_of(SendPipe::call(&boundary, chat_request()).await?).await;
    println!("  served: {served:?}");
    assert_eq!(
        served, "served by ours",
        "Off returns the inner handle unchanged"
    );
    Ok(())
}

// ── 2. Record: tee theirs to a cassette as it flows ──────────────────────────

async fn section_record(cassette: &Path, runtime: &Arc<dyn Runtime>) -> Result<(), ProximaError> {
    println!("\n--- 2. Record: serve theirs, tee the interaction to a cassette ---");
    let (boundary, terminal) = wire_boundary(BoundaryMode::Record, deps(cassette, runtime)).await?;
    let served = body_of(SendPipe::call(&boundary, chat_request()).await?).await;
    println!("  served (by theirs, the oracle): {served:?}");
    assert_eq!(served, "served by theirs", "Record's inner is theirs");

    // await the flush signal, then confirm the cassette actually captured it.
    terminal
        .expect("Record yields a drain signal")
        .drained()
        .await;
    let events = read_cassette(cassette, Arc::clone(runtime)).await;
    let captured = events
        .iter()
        .any(|event| matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. })));
    println!(
        "  cassette events: {} (terminal captured: {captured})",
        events.len()
    );
    assert!(captured, "the interaction reached the cassette");
    Ok(())
}

// ── 3. Replay: serve the recording back, no upstream call ────────────────────

async fn section_replay(cassette: &Path, runtime: &Arc<dyn Runtime>) -> Result<(), ProximaError> {
    println!("\n--- 3. Replay: serve the cassette back byte-identical, theirs never called ---");
    let (boundary, _) = wire_boundary(BoundaryMode::Replay, deps(cassette, runtime)).await?;
    let served = body_of(SendPipe::call(&boundary, chat_request()).await?).await;
    println!("  served (off disk): {served:?}");
    assert_eq!(
        served, "served by theirs",
        "replay reproduces theirs' recorded bytes"
    );

    // the flip side of byte-identical: an uncaptured request is a typed miss.
    let unrecorded = Request::builder()
        .method("GET")
        .path("/v1/never-recorded")
        .build()
        .expect("request builds");
    match SendPipe::call(&boundary, unrecorded).await {
        Err(ProximaError::ReplayMiss { fingerprint }) => {
            println!("  unrecorded request correctly missed: {fingerprint}");
        }
        other => panic!("expected a replay miss, got {other:?}"),
    }
    Ok(())
}

// ── 4. Shadow: fan out to both, report divergence ────────────────────────────

async fn section_shadow(_cassette: &Path, _runtime: &Arc<dyn Runtime>) -> Result<(), ProximaError> {
    println!("\n--- 4. Shadow: run ours against theirs, report where they disagree ---");

    // agree: same bytes both sides -> Diff serves 200 identical.
    let agree = into_handle(Diff::new(
        into_handle(SynthUpstream::new("theirs", 200, "same answer")),
        into_handle(SynthUpstream::new("ours", 200, "same answer")),
    ));
    let response = SendPipe::call(&agree, chat_request()).await?;
    println!("  agree   -> status {}", response.status);
    assert_eq!(response.status, 200, "identical responses -> 200");

    // diverge: different bytes -> Diff serves 409 with a report.
    let diverge = into_handle(Diff::new(
        into_handle(SynthUpstream::new("theirs", 200, "old answer")),
        into_handle(SynthUpstream::new("ours", 200, "new answer")),
    ));
    let response = SendPipe::call(&diverge, chat_request()).await?;
    println!("  diverge -> status {}", response.status);
    assert_eq!(response.status, 409, "divergent responses -> 409 conflict");
    Ok(())
}

// ── 5. Extend WITHOUT a new type: compose a pipe, drop it in the same slot ───
//
// The module doc's "gotcha" section explains why this is NOT
// `Fallback { primary: theirs, secondary: replay }`: `Fallback` only
// implements the root `Pipe` trait, never `SendPipe`, and `into_handle`
// demands `SendPipe` exactly. `Transform` DOES implement `SendPipe` over a
// `PipeHandle` inner (`proxima-primitives/src/pipe/transform.rs:74-99`), so
// it goes through the same door `wire_boundary`'s four arms use.

async fn section_extend_with_a_composed_strategy() -> Result<(), ProximaError> {
    println!("\n--- 5. Extend: tag theirs' responses, no new type, no new enum arm ---");

    // a strategy that isn't on the menu: stamp every response `theirs`
    // serves with a migration marker header, so ops dashboards can see the
    // strangler-fig progress live without a new BoundaryMode arm. `Transform`
    // is an existing SendPipe combinator (proxima::Transform); wrapping
    // `theirs()` in it and erasing with `into_handle` is composition, not a
    // new type.
    let tagged = into_handle(
        Transform::new(theirs()).with_response_op(ResponseOp::SetHeader {
            name: "x-served-by".into(),
            value: "theirs (boundary canary)".into(),
        }),
    );

    let response = SendPipe::call(&tagged, chat_request()).await?;
    let marker = response.metadata.get_str("x-served-by").map(str::to_owned);
    let served = body_of(response).await;
    println!("  theirs, tagged -> {served:?} (x-served-by: {marker:?})");
    assert_eq!(
        served, "served by theirs",
        "Transform's inner still answers"
    );
    assert_eq!(
        marker.as_deref(),
        Some("theirs (boundary canary)"),
        "Transform added the migration marker header on the way out"
    );
    Ok(())
}

async fn read_cassette(path: &Path, runtime: Arc<dyn Runtime>) -> Vec<RecordingEvent> {
    let source = JsonlSource::new(path, runtime);
    let mut events = source.events();
    let mut collected = Vec::new();
    while let Some(event) = events.next().await {
        collected.push(event.expect("read recording event"));
    }
    collected
}
