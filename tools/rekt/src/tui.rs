//! the live observability dashboard.
//!
//! shaped as a telemetry sink, not a side channel. the run records via proxima
//! telemetry [`Counter`](proxima_telemetry::metric::Counter)s; a dedicated
//! [`Recorder`] drains them on its own thread and dispatches each window's
//! [`MetricSample`]s through the pipe chain into [`DashboardPipe`] — the same
//! `SendPipe<In = TelemetryRequest>` shape the file/OTLP/stdout exporters wear.
//! the dashboard is "just another exporter": it folds the sampled deltas into
//! shared atomics, and the ratatui loop reads those atomics to draw. nothing
//! here pumps the drain or touches the load hot path.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use proxima::SendPipe;
use proxima::pipe::ProximaError;
use proxima::request::Response;
use proxima_telemetry::metric::MetricSample;
use proxima_telemetry::pipes::{TelemetryRecord, TelemetryRequest};
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::tag::{ScalarValue, Tag};

use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph};

use crate::engine::{LiveCounters, Throughput, drive_adaptive_live, drive_throughput_live};
use crate::error::Error;
use crate::grpcload::drive_grpc_live;
use crate::h2load::drive_h2_live;
use crate::h3load::drive_h3_live;
use crate::live::{Series, human_count};

const HITS: &str = "rekt.hits";
const ERRORS: &str = "rekt.errors";
const DRAIN_INTERVAL: Duration = Duration::from_millis(100);
const FRAME_POLL: Duration = Duration::from_millis(60);
// series points span ~2-3 drains: wider than the drain interval so a point's
// rate averages several windows instead of aliasing against a single one, which
// keeps the line smooth instead of saw-toothed.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);
const SERIES_CAPACITY: usize = 1024;

/// which protocol the live drive speaks, and its shape. the H3 target is a
/// `host:port` (parsed to a `SocketAddr`); the H1/H2 target is an `http://` url.
#[derive(Debug, Clone)]
pub enum Mode {
    /// h1 flat-out: a fixed `connections_per_core` keep-alive clients per core.
    H1FlatOut { connections_per_core: usize },
    /// h1 adaptive: a hillclimb controller raises/lowers the per-core target from
    /// `seed`, chasing the throughput crest.
    H1Adaptive { seed: usize },
    /// multiplexed h2: `connections_per_core` connections, each keeping
    /// `streams_per_conn` streams in flight.
    H2 { connections_per_core: usize, streams_per_conn: usize },
    /// multiplexed native-QUIC h3: `connections_per_core` connections, each
    /// keeping `streams_per_conn` streams in flight, verified against `server_name`.
    H3 { connections_per_core: usize, streams_per_conn: usize, server_name: String },
    /// grpc over multiplexed h2: unary calls to `path` kept in flight.
    Grpc { connections_per_core: usize, streams_per_conn: usize, path: String },
}

impl Mode {
    fn label(&self) -> String {
        match self {
            Mode::H1FlatOut { connections_per_core } => format!("h1 flat-out {connections_per_core}c/core"),
            Mode::H1Adaptive { seed } => format!("h1 adaptive seed {seed}"),
            Mode::H2 { connections_per_core, streams_per_conn } => format!("h2 {connections_per_core}c × {streams_per_conn}s"),
            Mode::H3 { connections_per_core, streams_per_conn, .. } => format!("h3 {connections_per_core}c × {streams_per_conn}s"),
            Mode::Grpc { connections_per_core, streams_per_conn, .. } => format!("grpc {connections_per_core}c × {streams_per_conn}s"),
        }
    }
}

/// the dashboard run spec.
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    pub target: String,
    pub cores: usize,
    pub duration: Duration,
    pub mode: Mode,
}

/// cumulative counts published by the drain and read by the render loop. the
/// drainer folds per-window deltas in via [`DashboardPipe`]; the render thread
/// only reads. relaxed is enough — these feed a display, nothing orders on them.
#[derive(Debug, Default)]
struct DashboardShared {
    hits: AtomicU64,
    errors: AtomicU64,
}

impl DashboardShared {
    fn counts(&self) -> (u64, u64) {
        (self.hits.load(Ordering::Relaxed), self.errors.load(Ordering::Relaxed))
    }
}

/// the telemetry exporter sink. receives one metric sample per registered
/// instrument per drain window and folds its delta into the shared counts.
struct DashboardPipe {
    shared: Arc<DashboardShared>,
}

impl DashboardPipe {
    fn new(shared: Arc<DashboardShared>) -> Self {
        Self { shared }
    }
}

impl SendPipe for DashboardPipe {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(&self, request: TelemetryRequest) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let shared = Arc::clone(&self.shared);
        async move {
            if let TelemetryRecord::Metric(MetricSample::Counter(point)) = request.payload
                && let ScalarValue::U64(delta) = point.value
            {
                match metric_name(&point.attrs) {
                    Some(HITS) => shared.hits.fetch_add(delta, Ordering::Relaxed),
                    Some(ERRORS) => shared.errors.fetch_add(delta, Ordering::Relaxed),
                    _ => 0,
                };
            }
            Ok(Response::ok(Bytes::new()))
        }
    }
}

// the instrument name the drainer stamps into a sample's attrs (`name_attrs`).
fn metric_name(attrs: &[Tag]) -> Option<&str> {
    attrs.iter().find_map(|tag| match tag {
        Tag::Scalar { key: "metric.name", value: ScalarValue::Str(name) } => Some(*name),
        _ => None,
    })
}

/// run the dashboard: spin up the recorder + drain, drive the load on a worker
/// thread, and render until the run finishes and the user dismisses it (or quits
/// early). returns the run's aggregate [`Throughput`].
pub fn run(config: DashboardConfig) -> Result<Throughput, Error> {
    let shared = Arc::new(DashboardShared::default());
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(DashboardPipe::new(Arc::clone(&shared)))
            .flush_interval(DRAIN_INTERVAL)
            .start()
            .map_err(|err| Error::Engine(err.to_string()))?,
    );
    let counters = LiveCounters {
        hits: recorder.counter(HITS),
        errors: recorder.counter(ERRORS),
    };

    // the drain strategy: its own thread pulls the registered instruments into
    // the pipe chain every window. the render loop never pumps it.
    let stop = Arc::new(AtomicBool::new(false));
    let drain = spawn_drain(Arc::clone(&recorder), Arc::clone(&stop));

    // the load: blocks for the whole duration on its own thread, folding deltas
    // into the counters as it goes.
    let (done_tx, done_rx) = mpsc::channel();
    let load = spawn_load(config.clone(), counters, done_tx);

    let outcome = render_loop(&config, &shared, &done_rx);

    stop.store(true, Ordering::Relaxed);
    let _ = drain.join();
    let carried = load.join().ok().flatten();

    // the render loop returns the result if the run finished while watching. on
    // early quit it returns None: the drive still runs to its deadline, so wait
    // on the channel for the aggregate it eventually sends.
    outcome
        .or(carried)
        .or_else(|| done_rx.recv().ok())
        .unwrap_or_else(|| Err(Error::Engine("load ended without a result".into())))
}

fn spawn_drain(recorder: Arc<Recorder>, stop: Arc<AtomicBool>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            recorder.drain();
            thread::sleep(DRAIN_INTERVAL);
        }
        recorder.drain();
    })
}

fn spawn_load(
    config: DashboardConfig,
    counters: LiveCounters,
    done_tx: mpsc::Sender<Result<Throughput, Error>>,
) -> thread::JoinHandle<Option<Result<Throughput, Error>>> {
    thread::spawn(move || {
        let DashboardConfig { target, cores, duration, mode } = config;
        let result = match mode {
            Mode::H1FlatOut { connections_per_core } => drive_throughput_live(&target, connections_per_core, cores, duration, counters),
            Mode::H1Adaptive { seed } => drive_adaptive_live(&target, seed, cores, duration, counters),
            Mode::H2 { connections_per_core, streams_per_conn } => {
                drive_h2_live(&target, connections_per_core, streams_per_conn, cores, duration, counters)
            }
            Mode::H3 { connections_per_core, streams_per_conn, server_name } => match target.parse::<SocketAddr>() {
                Ok(addr) => drive_h3_live(addr, &server_name, connections_per_core, cores, duration, streams_per_conn, counters),
                Err(err) => Err(Error::Engine(format!("h3 target must be host:port: {err}"))),
            },
            Mode::Grpc { connections_per_core, streams_per_conn, path } => {
                drive_grpc_live(&target, &path, connections_per_core, streams_per_conn, cores, duration, counters)
            }
        };
        // the render loop takes the result if it is still watching; if it already
        // left, this send fails and the handle carries the result home instead.
        match done_tx.send(result) {
            Ok(()) => None,
            Err(mpsc::SendError(result)) => Some(result),
        }
    })
}

// draw at ~12fps, sampling the shared counts into the series each frame. returns
// Some(result) once the run finishes (the frozen final frame stays up until the
// user quits); None if the user quits before the run finishes.
fn render_loop(
    config: &DashboardConfig,
    shared: &DashboardShared,
    done_rx: &mpsc::Receiver<Result<Throughput, Error>>,
) -> Option<Result<Throughput, Error>> {
    let mut terminal = ratatui::init();
    let mut series = Series::new(SERIES_CAPACITY);
    let started = Instant::now();
    let mut last_sample = started;
    let mut finished: Option<Result<Throughput, Error>> = None;

    loop {
        let (hits, errors) = shared.counts();
        if last_sample.elapsed() >= SAMPLE_INTERVAL {
            series.push(started.elapsed().as_secs_f64(), hits, errors);
            last_sample = Instant::now();
        }
        let view = View { config, series: &series, hits, errors, running: finished.is_none() };
        if terminal.draw(|frame| draw(frame, &view)).is_err() {
            break;
        }
        if quit_pressed() {
            break;
        }
        if finished.is_none()
            && let Ok(result) = done_rx.try_recv()
        {
            finished = Some(result);
        }
    }

    ratatui::restore();
    finished
}

fn quit_pressed() -> bool {
    if !event::poll(FRAME_POLL).unwrap_or(false) {
        return false;
    }
    matches!(event::read(), Ok(Event::Key(key)) if is_quit(key))
}

fn is_quit(key: KeyEvent) -> bool {
    if key.kind == KeyEventKind::Release {
        return false;
    }
    matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
        || (key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c'))
}

struct View<'a> {
    config: &'a DashboardConfig,
    series: &'a Series,
    hits: u64,
    errors: u64,
    running: bool,
}

fn draw(frame: &mut Frame, view: &View) {
    let [header, chart, footer] =
        Layout::vertical([Constraint::Length(3), Constraint::Min(8), Constraint::Length(9)]).areas(frame.area());
    draw_header(frame, header, view);
    draw_chart(frame, chart, view);
    let [rps, totals] = Layout::horizontal([Constraint::Length(26), Constraint::Min(20)]).areas(footer);
    draw_rps(frame, rps, view);
    draw_totals(frame, totals, view);
}

fn draw_header(frame: &mut Frame, area: Rect, view: &View) {
    let elapsed = view.series.latest().map(|sample| sample.elapsed).unwrap_or(0.0);
    let duration = view.config.duration.as_secs_f64();
    let status = if view.running { "running" } else { "complete" };
    let line = Line::from(vec![
        Span::styled(" rekt ", Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(format!("  {}  ", view.config.target)),
        Span::styled(view.config.mode.label(), Style::default().fg(Color::Yellow)),
        Span::raw(format!("  {}c  ", view.config.cores)),
        Span::styled(format!("{elapsed:.1}s / {duration:.0}s  "), Style::default().fg(Color::DarkGray)),
        Span::styled(status, Style::default().fg(if view.running { Color::Green } else { Color::Magenta })),
    ]);
    frame.render_widget(Paragraph::new(line).block(Block::default().borders(Borders::ALL)), area);
}

fn draw_chart(frame: &mut Frame, area: Rect, view: &View) {
    let hits = view.series.hits_points();
    let errors = view.series.errors_points();
    let [x_lo, x_hi] = view.series.x_bounds();
    let y_hi = (view.series.peak_rps() * 1.15).max(10.0);

    let datasets = vec![
        Dataset::default()
            .name("hits/s")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&hits),
        Dataset::default()
            .name("errors/s")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Red))
            .data(&errors),
    ];

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(" throughput  req/s "))
        .x_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([x_lo, x_hi])
                .labels([format!("{x_lo:.0}s"), format!("{:.0}s", (x_lo + x_hi) / 2.0), format!("{x_hi:.0}s")]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, y_hi])
                .labels(["0".to_string(), human_count(y_hi / 2.0), human_count(y_hi)]),
        );
    frame.render_widget(chart, area);
}

fn draw_rps(frame: &mut Frame, area: Rect, view: &View) {
    let rps = view.series.latest().map(|sample| sample.hits_per_sec).unwrap_or(0.0);
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(human_count(rps), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled("req/s", Style::default().fg(Color::DarkGray))),
        Line::from(""),
        Line::from(Span::styled(format!("peak {}", human_count(view.series.peak_rps())), Style::default().fg(Color::DarkGray))),
    ];
    let block = Block::default().borders(Borders::ALL).title(" rps ");
    frame.render_widget(Paragraph::new(lines).block(block).alignment(Alignment::Center), area);
}

fn draw_totals(frame: &mut Frame, area: Rect, view: &View) {
    let total = view.hits + view.errors;
    let error_rate = if total == 0 { 0.0 } else { view.errors as f64 / total as f64 * 100.0 };
    let errors_per_sec = view.series.latest().map(|sample| sample.errors_per_sec).unwrap_or(0.0);
    let hint = if view.running { "q to quit" } else { "run complete — q to exit" };
    let lines = vec![
        row("hits", human_count(view.hits as f64), Color::Green),
        row("errors", human_count(view.errors as f64), if view.errors == 0 { Color::Green } else { Color::Red }),
        row("err/s", human_count(errors_per_sec), if errors_per_sec == 0.0 { Color::DarkGray } else { Color::Red }),
        row("err rate", format!("{error_rate:.2}%"), if error_rate == 0.0 { Color::Green } else { Color::Red }),
        Line::from(""),
        Line::from(Span::styled(hint, Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))),
    ];
    frame.render_widget(Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" totals ")), area);
}

fn row(label: &str, value: String, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!(" {label:<9}"), Style::default().fg(Color::Gray)),
        Span::styled(value, Style::default().fg(color).add_modifier(Modifier::BOLD)),
    ])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;

    // the wiring under test: a drain snapshots the registered counters into
    // metric samples, dispatches them through DashboardPipe, and the deltas land
    // in the shared counts — cumulatively, since a counter resets each drain.
    #[test]
    fn drain_folds_metric_samples_into_shared_counts() {
        let shared = Arc::new(DashboardShared::default());
        let recorder = Recorder::builder()
            .pipe(DashboardPipe::new(Arc::clone(&shared)))
            .flush_interval(DRAIN_INTERVAL)
            .start()
            .expect("recorder");
        let hits = recorder.counter(HITS);
        let errors = recorder.counter(ERRORS);

        hits.add(300, &[]);
        errors.add(2, &[]);
        recorder.drain();
        assert_eq!(shared.counts(), (300, 2));

        hits.add(50, &[]);
        recorder.drain();
        assert_eq!(shared.counts(), (350, 2), "second window accumulates on the first");
    }

    fn config() -> DashboardConfig {
        DashboardConfig {
            target: "http://127.0.0.1:8080/".to_string(),
            cores: 2,
            duration: Duration::from_secs(10),
            mode: Mode::H1FlatOut { connections_per_core: 8 },
        }
    }

    #[test]
    fn draw_survives_empty_and_populated_series() {
        let mut terminal = Terminal::new(TestBackend::new(90, 26)).expect("terminal");
        let config = config();

        let empty = Series::new(16);
        let view = View { config: &config, series: &empty, hits: 0, errors: 0, running: true };
        terminal.draw(|frame| draw(frame, &view)).expect("draw empty");

        let mut series = Series::new(16);
        series.push(0.0, 0, 0);
        series.push(0.25, 1000, 1);
        series.push(0.5, 2500, 1);
        let view = View { config: &config, series: &series, hits: 2500, errors: 1, running: false };
        terminal.draw(|frame| draw(frame, &view)).expect("draw populated");
    }
}
