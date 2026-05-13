use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_recording::event::{
    FrameMetadata, HttpEvent, InteractionId, ProtocolEvent, RecordMeta, RecordingEvent,
    RequestHeader,
};
use proxima_recording::pipe::{DeferredRuntime, FormatKind, LazyFanOut, SinkSpec};
use time::OffsetDateTime;

const REDACTED_PREFIX: &str = "redacted:";

/// The per-direction chunk buffer. A plain owned `Vec` — capture is a single
/// consumer that drains once at `finish()`, so the broadcast-ring machinery
/// (pre-allocated slots, clone-on-read, lag-drop) buys nothing here. `push`
/// MOVES each chunk in (lazy alloc, grows only as needed); `finish` MOVES them
/// out into events (zero clone). `Mutex` only because the clonable pusher hands
/// `&self` access across the `Arc`; it is single-producer and uncontended.
type ChunkBuffer = Arc<Mutex<Vec<Bytes>>>;

fn lock_buffer(buffer: &Mutex<Vec<Bytes>>) -> std::sync::MutexGuard<'_, Vec<Bytes>> {
    // poison only happens if a holder panicked mid-push; the Vec is still a
    // valid (append-only) buffer, so recover rather than propagate a panic.
    buffer
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Whether a direction's chunk bytes are recorded — a runtime knob (principle
/// 4: config as first-class), not a recompile. (There is no "coalesce" mode:
/// the sink already compresses each interaction's events as one block, so
/// `PerChunk` keeps full fidelity AND compresses on disk — coalescing would
/// only throw away boundaries for no gain.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "intercept-config",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "intercept-config", serde(rename_all = "snake_case"))]
pub enum ChunkGranularity {
    /// One event per pushed chunk — preserves SSE/streaming boundaries so a
    /// replay reproduces the exact wire framing (§14). The block sink compresses
    /// the whole interaction together, so this is faithful AND compact. Default.
    #[default]
    PerChunk,
    /// Don't record chunk bytes at all — only the structural events (`Started`,
    /// headers, `Ended`) land. For logging/metrics without replay: `push`
    /// becomes a no-op, so nothing is buffered, stored, or restored.
    Discard,
}

impl ChunkGranularity {
    /// Whether chunk bytes are retained. `false` for [`Self::Discard`], which
    /// turns `push` into a no-op (no buffering, no store/restore).
    #[must_use]
    pub fn records_chunks(self) -> bool {
        !matches!(self, Self::Discard)
    }
}

#[derive(Clone)]
pub struct Capture {
    durable: Arc<LazyFanOut>,
    chunk_granularity: ChunkGranularity,
}

impl Capture {
    /// Build the capture terminal *disarmed*: it records its destination (a
    /// single `bin` log at `data_path`) + the shared `spigot`, but opens no
    /// file and pumps nothing until the App turns the spigot on at serve.
    pub fn open(data_path: &Path, spigot: DeferredRuntime) -> Result<Self, ProximaError> {
        validate_parent(data_path)?;
        Ok(Self::with_spec(
            SinkSpec::new(data_path.to_string_lossy().into_owned(), FormatKind::Bin),
            spigot,
        ))
    }

    /// Like [`Self::open`] but with an explicit zstd level for the block
    /// compressor (the CPU/ratio lever). Higher = smaller files, more CPU.
    pub fn open_with_level(
        data_path: &Path,
        zstd_level: i32,
        spigot: DeferredRuntime,
    ) -> Result<Self, ProximaError> {
        validate_parent(data_path)?;
        Ok(Self::with_spec(
            SinkSpec::new(data_path.to_string_lossy().into_owned(), FormatKind::Bin)
                .with_zstd_level(zstd_level),
            spigot,
        ))
    }

    fn with_spec(spec: SinkSpec, spigot: DeferredRuntime) -> Self {
        Self {
            durable: Arc::new(LazyFanOut::new(vec![spec], spigot)),
            chunk_granularity: ChunkGranularity::default(),
        }
    }

    /// Select how chunks become events (fidelity vs throughput). Builder-style
    /// so config-driven (`CaptureSettings`) and API-driven setups agree.
    #[must_use]
    pub fn with_chunk_granularity(mut self, granularity: ChunkGranularity) -> Self {
        self.chunk_granularity = granularity;
        self
    }

    /// Whether the terminal has somewhere to pump (spigot on + a destination).
    /// Producers gate work on this — no point building events with no sink.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.durable.is_armed()
    }

    /// Flush the durable terminal — no-op while disarmed.
    pub async fn flush(&self) -> Result<(), ProximaError> {
        self.durable.flush().await
    }

    /// Append a single `ProtocolEvent::Custom` event correlated to an existing
    /// interaction via `parent`. Use this to record enrichment or derived events
    /// that belong to the same exchange but are produced asynchronously after
    /// `finish()` has been called on the primary recorder.
    pub async fn append_custom(
        &self,
        parent: InteractionId,
        kind: &str,
        payload: serde_json::Value,
    ) -> Result<(), ProximaError> {
        self.durable
            .call(vec![RecordingEvent {
                id: InteractionId::new(),
                ts_ms: now_ms(),
                parent: Some(parent),
                event: ProtocolEvent::Custom {
                    kind: kind.to_string(),
                    payload,
                },
            }])
            .await
    }

    pub async fn begin(
        &self,
        target_host: &str,
        request_wire: &[u8],
        started: Instant,
    ) -> Result<InteractionRecorder, ProximaError> {
        let interaction_id = InteractionId::new();
        let request_header = parse_request_header(request_wire, target_host);

        self.durable
            .call(vec![RecordingEvent {
                id: interaction_id,
                ts_ms: now_ms(),
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::Started {
                    ts: OffsetDateTime::now_utc(),
                    pipe: "intercept".into(),
                    request: request_header,
                    meta: None,
                }),
            }])
            .await?;

        Ok(InteractionRecorder {
            durable: Arc::clone(&self.durable),
            interaction_id,
            // Vec::new does not allocate until the first push.
            request_buf: Arc::new(Mutex::new(Vec::new())),
            response_buf: Arc::new(Mutex::new(Vec::new())),
            chunk_granularity: self.chunk_granularity,
            started,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum Direction {
    Request,
    Response,
}

pub struct InteractionRecorder {
    durable: Arc<LazyFanOut>,
    interaction_id: InteractionId,
    request_buf: ChunkBuffer,
    response_buf: ChunkBuffer,
    chunk_granularity: ChunkGranularity,
    started: Instant,
}

#[derive(Clone)]
pub struct RequestPusher {
    buf: ChunkBuffer,
    granularity: ChunkGranularity,
}

#[derive(Clone)]
pub struct ResponsePusher {
    buf: ChunkBuffer,
    granularity: ChunkGranularity,
}

impl RequestPusher {
    pub fn push(&self, chunk: Bytes) {
        if self.granularity.records_chunks() && !chunk.is_empty() {
            lock_buffer(&self.buf).push(chunk);
        }
    }
}

impl ResponsePusher {
    pub fn push(&self, chunk: Bytes) {
        if self.granularity.records_chunks() && !chunk.is_empty() {
            lock_buffer(&self.buf).push(chunk);
        }
    }
}

impl InteractionRecorder {
    #[must_use]
    pub fn interaction_id(&self) -> InteractionId {
        self.interaction_id
    }

    pub fn request_pusher(&self) -> RequestPusher {
        RequestPusher {
            buf: Arc::clone(&self.request_buf),
            granularity: self.chunk_granularity,
        }
    }

    pub fn response_pusher(&self) -> ResponsePusher {
        ResponsePusher {
            buf: Arc::clone(&self.response_buf),
            granularity: self.chunk_granularity,
        }
    }

    pub fn push_request(&self, chunk: Bytes) {
        if self.chunk_granularity.records_chunks() && !chunk.is_empty() {
            lock_buffer(&self.request_buf).push(chunk);
        }
    }

    pub fn push_response(&self, chunk: Bytes) {
        if self.chunk_granularity.records_chunks() && !chunk.is_empty() {
            lock_buffer(&self.response_buf).push(chunk);
        }
    }

    pub async fn finish(self, response_wire: &[u8]) -> Result<(), ProximaError> {
        // collect the whole interaction's events, then hand the durable ONE
        // batch: the terminal writes it as a single zstd block (one data + one
        // index write). one clock read for the batch — per-event timestamps
        // cost a syscall each at no ordering gain (Started, written at begin(),
        // already orders before this batch).
        let ts_ms = now_ms();
        let mut events: Vec<RecordingEvent> = Vec::new();

        // take the owned chunks (move, not clone) and turn each into an event.
        let request_chunks = std::mem::take(&mut *lock_buffer(&self.request_buf));
        collect_chunks(
            &mut events,
            request_chunks,
            self.interaction_id,
            ts_ms,
            Direction::Request,
            self.chunk_granularity,
        );

        events.push(RecordingEvent {
            id: self.interaction_id,
            ts_ms,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::RequestEnded),
        });

        let (status, response_headers, _body_bytes) = parse_response_head(response_wire);
        events.push(RecordingEvent {
            id: self.interaction_id,
            ts_ms,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseStarted {
                status,
                headers: response_headers,
            }),
        });

        let response_chunks = std::mem::take(&mut *lock_buffer(&self.response_buf));
        collect_chunks(
            &mut events,
            response_chunks,
            self.interaction_id,
            ts_ms,
            Direction::Response,
            self.chunk_granularity,
        );

        events.push(RecordingEvent {
            id: self.interaction_id,
            ts_ms,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::Ended {
                latency_ms: elapsed_ms(self.started),
                meta: RecordMeta::default(),
            }),
        });

        self.durable.call(events).await
    }
}

fn collect_chunks(
    events: &mut Vec<RecordingEvent>,
    chunks: Vec<Bytes>,
    interaction_id: InteractionId,
    ts_ms: u64,
    direction: Direction,
    granularity: ChunkGranularity,
) {
    let push_chunk = |events: &mut Vec<RecordingEvent>, data: Bytes| {
        let http_event = match direction {
            Direction::Request => HttpEvent::RequestChunk {
                data,
                metadata: FrameMetadata::new(),
            },
            Direction::Response => HttpEvent::ResponseChunk {
                data,
                metadata: FrameMetadata::new(),
            },
        };
        events.push(RecordingEvent {
            id: interaction_id,
            ts_ms,
            parent: None,
            event: ProtocolEvent::Http(http_event),
        });
    };

    match granularity {
        // push was a no-op for Discard, so `chunks` is already empty; the match
        // arm documents intent and guards against a future caller that fills it.
        ChunkGranularity::Discard => {}
        ChunkGranularity::PerChunk => {
            for chunk in chunks {
                push_chunk(events, chunk);
            }
        }
    }
}

// cheap config fail-fast: the durable opens lazily at serve, so a bad path
// would otherwise only surface on the first interaction. a missing parent dir
// is a config error — catch it at build without opening the file.
fn validate_parent(data_path: &Path) -> Result<(), ProximaError> {
    if let Some(parent) = data_path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.is_dir()
    {
        return Err(ProximaError::Config(format!(
            "capture data_path parent dir does not exist: {}",
            parent.display()
        )));
    }
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

pub fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn parse_request_header(wire: &[u8], host_fallback: &str) -> RequestHeader {
    let text = match std::str::from_utf8(wire) {
        Ok(text_str) => text_str,
        Err(_) => {
            return RequestHeader {
                method: String::new(),
                path: String::new(),
                headers: BTreeMap::new(),
                query: BTreeMap::new(),
            };
        }
    };

    let header_end = text.find("\r\n\r\n").unwrap_or(text.len());
    let header_block = &text[..header_end];
    let mut lines = header_block.split("\r\n");

    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path_and_query = parts.next().unwrap_or("").to_string();

    let (path_only, query_map) = split_path_query(&path_and_query);

    let mut headers: BTreeMap<String, String> = BTreeMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            let lowered = name.trim().to_ascii_lowercase();
            let value_trimmed = value.trim();
            let stored = if is_sensitive_header(&lowered) {
                redact(value_trimmed.as_bytes())
            } else {
                value_trimmed.to_string()
            };
            headers.insert(lowered, stored);
        }
    }
    if !headers.contains_key("host") {
        headers.insert("host".into(), host_fallback.into());
    }

    RequestHeader {
        method,
        path: path_only,
        headers,
        query: query_map,
    }
}

fn split_path_query(path_and_query: &str) -> (String, BTreeMap<String, String>) {
    let mut query: BTreeMap<String, String> = BTreeMap::new();
    let (path_only, query_str) = path_and_query
        .split_once('?')
        .map_or((path_and_query, ""), |(left, right)| (left, right));
    for pair in query_str.split('&').filter(|segment| !segment.is_empty()) {
        if let Some((key, value)) = pair.split_once('=') {
            query.insert(key.to_string(), value.to_string());
        } else {
            query.insert(pair.to_string(), String::new());
        }
    }
    (path_only.to_string(), query)
}

fn parse_response_head(wire: &[u8]) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let text = String::from_utf8_lossy(wire);
    let header_end = text.find("\r\n\r\n");
    let header_end_idx = match header_end {
        Some(idx) => idx,
        None => return (0, Vec::new(), wire.to_vec()),
    };
    let header_block = &text[..header_end_idx];
    let body_bytes = wire[header_end_idx + 4..].to_vec();

    let mut lines = header_block.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status = parse_status_code(status_line);

    let mut headers: Vec<(String, String)> = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            let lowered = name.trim().to_ascii_lowercase();
            let value_trimmed = value.trim();
            let stored = if is_sensitive_header(&lowered) {
                redact(value_trimmed.as_bytes())
            } else {
                value_trimmed.to_string()
            };
            headers.push((lowered, stored));
        }
    }

    (status, headers, body_bytes)
}

fn parse_status_code(status_line: &str) -> u16 {
    let mut parts = status_line.split_whitespace();
    let _ = parts.next();
    parts
        .next()
        .and_then(|code| code.parse::<u16>().ok())
        .unwrap_or(0)
}

fn is_sensitive_header(lowered_name: &str) -> bool {
    matches!(
        lowered_name,
        "authorization" | "cookie" | "set-cookie" | "proxy-authorization"
    ) || lowered_name.starts_with("x-")
        || lowered_name.contains("token")
        || lowered_name.contains("bearer")
}

fn redact(secret_bytes: &[u8]) -> String {
    let hash = blake3::hash(secret_bytes);
    let hex = hash.to_hex();
    format!("{REDACTED_PREFIX}{}", &hex.as_str()[..12])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use proxima::runtime::{PrimeRuntime, Runtime};
    use proxima_recording::pipe::deferred_runtime;

    fn prime() -> Arc<dyn Runtime> {
        Arc::new(PrimeRuntime::new(1).expect("prime"))
    }

    // an armed capture over a fresh bin log: spigot turned on with a 1-core
    // prime runtime (the off-core blocking-I/O backend the durable rides).
    fn armed_capture(path: &Path) -> Capture {
        let spigot = deferred_runtime();
        spigot
            .set(Arc::new(PrimeRuntime::new(1).expect("prime")) as Arc<dyn Runtime>)
            .ok();
        Capture::open(path, spigot).expect("open capture")
    }

    // disarmed: no spigot -> no file, no pump, even after finish + flush.
    #[proxima::test]
    async fn disarmed_capture_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let capture = Capture::open(&path, deferred_runtime()).unwrap();
        assert!(!capture.is_armed());
        let recorder = capture
            .begin(
                "api.example.com",
                b"POST /r HTTP/1.1\r\nHost: api.example.com\r\n\r\n",
                Instant::now(),
            )
            .await
            .unwrap();
        recorder.push_response(Bytes::from_static(b"data"));
        recorder.finish(b"HTTP/1.1 200 OK\r\n\r\n").await.unwrap();
        capture.flush().await.unwrap();
        assert!(!path.exists(), "disarmed capture opens no file");
    }

    // the runtime knob: PerChunk records one event per chunk, Discard records
    // none (push is a no-op — no store/restore).
    #[proxima::test]
    async fn chunk_granularity_controls_recorded_chunk_events() {
        use proxima_recording::BinSource;
        use proxima_recording::source::RecordingSource;

        for (granularity, expected_chunks) in [
            (ChunkGranularity::PerChunk, 3usize),
            (ChunkGranularity::Discard, 0),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("rec.bin");
            let capture = armed_capture(&path).with_chunk_granularity(granularity);

            let request_wire = b"POST /r HTTP/1.1\r\nHost: api.example.com\r\n\r\n";
            let response_wire = b"HTTP/1.1 200 OK\r\n\r\n";
            let recorder = capture
                .begin("api.example.com", request_wire, Instant::now())
                .await
                .unwrap();
            for byte in 0..3u8 {
                recorder.push_response(Bytes::copy_from_slice(&[byte; 8]));
            }
            recorder.finish(response_wire).await.unwrap();
            capture.flush().await.unwrap();

            let source = BinSource::new(&path, prime());
            let mut stream = source.events();
            let mut chunk_events = 0usize;
            while let Some(item) = stream.next().await {
                if let ProtocolEvent::Http(HttpEvent::ResponseChunk { .. }) = item.unwrap().event {
                    chunk_events += 1;
                }
            }
            assert_eq!(chunk_events, expected_chunks, "granularity {granularity:?}");
        }
    }

    #[test]
    fn redact_is_stable_and_short() {
        let token_value = b"Bearer ghp_abcdef";
        let redacted_first = redact(token_value);
        let redacted_second = redact(token_value);
        assert_eq!(redacted_first, redacted_second);
        assert!(redacted_first.starts_with(REDACTED_PREFIX));
        assert_eq!(redacted_first.len(), REDACTED_PREFIX.len() + 12);
        assert!(!redacted_first.contains("ghp_"));
    }

    #[test]
    fn sensitive_classifier_catches_known_headers() {
        for known_header in [
            "authorization",
            "cookie",
            "x-client-session-id",
            "x-session-token",
            "x-anything",
            "some-token-name",
            "the-bearer-value",
        ] {
            assert!(is_sensitive_header(known_header), "{known_header}");
        }
        assert!(!is_sensitive_header("content-type"));
        assert!(!is_sensitive_header("user-agent"));
    }

    #[test]
    fn parse_request_header_redacts_authorization_only() {
        let wire = b"POST /responses HTTP/1.1\r\n\
                     Host: api.example.com\r\n\
                     Content-Type: application/json\r\n\
                     Authorization: Bearer secret123\r\n\
                     X-Session-Token: tok_value\r\n\
                     \r\n";
        let parsed = parse_request_header(wire, "api.example.com");
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.path, "/responses");
        assert_eq!(
            parsed.headers.get("content-type").unwrap(),
            "application/json"
        );
        assert!(
            parsed
                .headers
                .get("authorization")
                .unwrap()
                .starts_with(REDACTED_PREFIX)
        );
        assert!(
            parsed
                .headers
                .get("x-session-token")
                .unwrap()
                .starts_with(REDACTED_PREFIX)
        );
        assert!(
            !parsed
                .headers
                .get("authorization")
                .unwrap()
                .contains("secret123")
        );
    }

    #[test]
    fn parse_request_query_string_is_split() {
        let wire = b"GET /responses?stream=true&model=model-nano HTTP/1.1\r\n\
                     Host: api.example.com\r\n\
                     \r\n";
        let parsed = parse_request_header(wire, "api.example.com");
        assert_eq!(parsed.path, "/responses");
        assert_eq!(parsed.query.get("stream").map(String::as_str), Some("true"));
        assert_eq!(
            parsed.query.get("model").map(String::as_str),
            Some("model-nano")
        );
    }

    #[test]
    fn parse_response_extracts_status_headers_body() {
        let wire = b"HTTP/1.1 200 OK\r\n\
                     Content-Type: application/json\r\n\
                     Set-Cookie: session=value\r\n\
                     \r\n\
                     {\"ok\":true}";
        let (status, headers, body) = parse_response_head(wire);
        assert_eq!(status, 200);
        let cookie = headers
            .iter()
            .find(|(name, _)| name == "set-cookie")
            .map(|(_, value)| value.clone());
        assert!(cookie.unwrap().starts_with(REDACTED_PREFIX));
        assert_eq!(body, b"{\"ok\":true}");
    }

    #[test]
    fn parse_response_with_no_body_yields_empty_body() {
        let wire = b"HTTP/1.1 204 No Content\r\n\
                     Content-Length: 0\r\n\
                     \r\n";
        let (status, _, body) = parse_response_head(wire);
        assert_eq!(status, 204);
        assert!(body.is_empty());
    }

    #[proxima::test]
    async fn event_order_is_started_request_ended_response_ended() {
        use proxima_recording::BinSource;
        use proxima_recording::source::RecordingSource;
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("recording.bin");
        let capture = armed_capture(&data_path);

        let request_wire = b"POST /turn HTTP/1.1\r\nHost: api.example.com\r\n\r\n";
        let response_wire = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n";

        let recorder = capture
            .begin("api.example.com", request_wire, Instant::now())
            .await
            .expect("begin");
        recorder.push_request(Bytes::from_static(b"req-bytes"));
        recorder.push_response(Bytes::from_static(b"resp-bytes"));
        recorder.finish(response_wire).await.expect("finish");
        capture.flush().await.expect("flush");

        let source = BinSource::new(&data_path, prime());
        let mut stream = source.events();
        let mut kinds: Vec<&str> = Vec::new();
        while let Some(item) = stream.next().await {
            let event = item.expect("event ok");
            let kind = match &event.event {
                ProtocolEvent::Http(HttpEvent::Started { .. }) => "started",
                ProtocolEvent::Http(HttpEvent::RequestChunk { .. }) => "req_chunk",
                ProtocolEvent::Http(HttpEvent::RequestEnded) => "req_end",
                ProtocolEvent::Http(HttpEvent::ResponseStarted { .. }) => "resp_start",
                ProtocolEvent::Http(HttpEvent::ResponseChunk { .. }) => "resp_chunk",
                ProtocolEvent::Http(HttpEvent::Ended { .. }) => "ended",
                _ => "other",
            };
            kinds.push(kind);
        }
        assert_eq!(
            kinds,
            vec![
                "started",
                "req_chunk",
                "req_end",
                "resp_start",
                "resp_chunk",
                "ended"
            ],
            "events must appear in canonical http sequence"
        );
    }

    #[proxima::test]
    async fn capture_round_trip_via_shared_ring_tee() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("recording.bin");
        let capture = armed_capture(&data_path);

        let request_wire = b"POST /responses HTTP/1.1\r\nHost: api.example.com\r\nAuthorization: Bearer xyz\r\n\r\n";
        let response_wire = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n";

        let recorder = capture
            .begin("api.example.com", request_wire, Instant::now())
            .await
            .expect("begin");

        recorder.push_request(Bytes::from_static(b"{\"ok\":true}"));
        recorder.push_response(Bytes::from_static(b"{\"id\":42}"));
        recorder.finish(response_wire).await.expect("finish");

        capture.flush().await.expect("flush");
        let metadata = std::fs::metadata(&data_path).expect("recording file");
        assert!(metadata.len() > 0, "recording file should have bytes");
    }

    #[proxima::test]
    async fn multi_chunk_pushes_are_preserved_through_tee() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("recording.bin");
        let capture = armed_capture(&data_path);

        let request_wire = b"POST /chunked HTTP/1.1\r\nHost: api.example.com\r\n\r\n";
        let response_wire = b"HTTP/1.1 200 OK\r\n\r\n";

        let recorder = capture
            .begin("api.example.com", request_wire, Instant::now())
            .await
            .expect("begin");

        for chunk in 0..32u8 {
            recorder.push_request(Bytes::copy_from_slice(&[chunk; 64]));
            recorder.push_response(Bytes::copy_from_slice(&[chunk.wrapping_add(1); 32]));
        }
        recorder.finish(response_wire).await.expect("finish");

        capture.flush().await.expect("flush");
        let metadata = std::fs::metadata(&data_path).expect("recording file");
        assert!(metadata.len() > 0, "recording file should have bytes");
    }
}
