//! C4 — `StdoutAlertPipe`: terminal sink pipe that formats an
//! [`AlertEvent`] and writes it to stdout.
//!
//! Composes:
//! - `SendPipe<In = AlertRequest, Out = Response<Bytes>>` — typed request; the
//!   event is `request.payload`, no serialization.
//! - The method-byte discriminant convention — accepts `b"ALERT"` and
//!   `b"SCHEDULED_TICK"`; rejects everything else with 405.
//! - `crate::alert::event::json_shape::alert_event_to_json` (under the
//!   proto crate's `json-shape` feature, on by default for std builds)
//!   for the `json` output format.
//!
//! Markers:
//! - NOT `WithoutFilesystem` (stdout is fs-shaped).
//! - NOT `IdempotentSideEffectFree` (printing twice ≠ printing once).
//! - `WithoutNetwork`, `WithoutSpawn`, `WithoutRandom`, `WithoutTime`.

use std::io::Write;
use std::sync::Mutex;

use bytes::Bytes;
use proxima_core::markers::{WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime};
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Response;

use crate::alert::event::AlertEvent;
use crate::alert::methods;
use crate::alert::pipes::AlertRequest;

/// How an [`StdoutAlertPipe`] formats events before writing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputFormat {
    /// One line per event: `[severity] kind { labels } payload=<n bytes>`.
    Human,
    /// One line per event: the documented `AlertEvent` JSON shape per
    /// `docs/proxima-notify/ALERT_EVENT_SCHEMA.md`.
    Json,
}

/// Terminal sink pipe that prints `AlertEvent` payloads to stdout.
pub struct StdoutAlertPipe {
    format: OutputFormat,
    writer: Mutex<std::io::Stdout>,
}

impl Default for StdoutAlertPipe {
    fn default() -> Self {
        Self::new(OutputFormat::Human)
    }
}

impl StdoutAlertPipe {
    /// Construct with a chosen format.
    #[must_use]
    pub fn new(format: OutputFormat) -> Self {
        Self {
            format,
            writer: Mutex::new(std::io::stdout()),
        }
    }

    /// Fluent builder entry point (principle 4).
    #[must_use]
    pub fn builder() -> StdoutAlertPipeBuilder {
        StdoutAlertPipeBuilder::default()
    }

    fn format_event(&self, event: &AlertEvent) -> String {
        match self.format {
            OutputFormat::Human => format!(
                "[{severity}] {kind} labels={labels} payload_bytes={payload_len} fired_at_micros={fired_at}",
                severity = event.severity.as_str(),
                kind = event.kind.as_str(),
                labels = format_labels(event),
                payload_len = event.payload.len(),
                fired_at = event.fired_at_micros,
            ),
            OutputFormat::Json => {
                let value = crate::alert::event::json_shape::alert_event_to_json(event);
                serde_json::to_string(&value)
                    .unwrap_or_else(|err| format!("{{\"json_render_error\":\"{err}\"}}"))
            }
        }
    }

    fn write_line(&self, line: &str) -> Result<(), ProximaError> {
        let mut guard = self
            .writer
            .lock()
            .map_err(|err| ProximaError::Record(format!("stdout mutex poisoned: {err}")))?;
        writeln!(*guard, "{line}").map_err(ProximaError::Io)?;
        guard.flush().map_err(ProximaError::Io)
    }
}

fn format_labels(event: &AlertEvent) -> String {
    let mut sorted: Vec<(&str, &str)> = event
        .labels
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    sorted.sort_by(|left, right| left.0.cmp(right.0));
    let pairs: Vec<String> = sorted
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect();
    format!("{{{}}}", pairs.join(","))
}

impl SendPipe for StdoutAlertPipe {
    type In = AlertRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: AlertRequest,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let method_known = matches!(
            request.method.as_bytes(),
            method if method == methods::ALERT || method == methods::SCHEDULED_TICK
        );
        let outcome: Result<Response<Bytes>, ProximaError> = if !method_known {
            Ok(Response::new(405))
        } else {
            let line = self.format_event(&request.payload);
            self.write_line(&line).map(|()| Response::ok(Bytes::new()))
        };
        async move { outcome }
    }
}

impl WithoutNetwork for StdoutAlertPipe {}
impl WithoutSpawn for StdoutAlertPipe {}
impl WithoutTime for StdoutAlertPipe {}
impl WithoutRandom for StdoutAlertPipe {}

/// Builder for [`StdoutAlertPipe`] (principle 4).
#[derive(Default)]
pub struct StdoutAlertPipeBuilder {
    format: Option<OutputFormat>,
}

impl StdoutAlertPipeBuilder {
    /// Choose `OutputFormat::Human` (default) or `OutputFormat::Json`.
    #[must_use]
    pub fn format(mut self, format: OutputFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Build the immutable [`StdoutAlertPipe`].
    #[must_use]
    pub fn build(self) -> StdoutAlertPipe {
        StdoutAlertPipe::new(self.format.unwrap_or(OutputFormat::Human))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use bytes::Bytes;
    use proxima_primitives::pipe::header_list::HeaderList;
    use proxima_primitives::pipe::request::{Request, RequestContext};

    use crate::alert::event::{AlertId, KindString, LabelKey, LabelMap, LabelValue, Payload, Severity};

    use super::*;

    fn sample_event() -> AlertEvent {
        let mut labels = LabelMap::new();
        labels
            .insert(
                LabelKey::try_from("host").unwrap(),
                LabelValue::try_from("test-host").unwrap(),
            )
            .unwrap();
        labels
            .insert(
                LabelKey::try_from("source").unwrap(),
                LabelValue::try_from("test").unwrap(),
            )
            .unwrap();
        AlertEvent {
            id: AlertId(ulid::Ulid::nil()),
            severity: Severity::Warn,
            kind: KindString::try_from("heartbeat").unwrap(),
            labels,
            payload: Payload::from_slice(&[1, 2, 3]).unwrap(),
            fired_at_micros: 1_700_000_000_000_000,
        }
    }

    fn alert_request(event: AlertEvent) -> AlertRequest {
        Request {
            method: methods::alert_method(),
            path: Bytes::from_static(b"/notify/alert"),
            query: HeaderList::new(),
            metadata: HeaderList::new(),
            payload: event,
            stream: None,
            context: RequestContext::default(),
        }
    }

    #[proxima::test]
    async fn unknown_method_returns_405_without_writing_to_stdout() {
        let pipe = StdoutAlertPipe::builder().build();
        let request = Request {
            method: proxima_primitives::pipe::method::Method::from_bytes(b"UNKNOWN"),
            path: Bytes::from_static(b"/"),
            query: HeaderList::new(),
            metadata: HeaderList::new(),
            payload: sample_event(),
            stream: None,
            context: RequestContext::default(),
        };
        let response = pipe.call(request).await.expect("call should not error");
        assert_eq!(response.status, 405);
    }

    #[test]
    fn human_format_renders_severity_kind_labels_payload_len() {
        let pipe = StdoutAlertPipe::builder()
            .format(OutputFormat::Human)
            .build();
        let event = sample_event();
        let line = pipe.format_event(&event);
        assert!(line.contains("[warn]"));
        assert!(line.contains("heartbeat"));
        assert!(line.contains("host=test-host"));
        assert!(line.contains("source=test"));
        assert!(line.contains("payload_bytes=3"));
        assert!(line.contains("fired_at_micros=1700000000000000"));
    }

    #[test]
    fn json_format_renders_documented_schema_keys() {
        let pipe = StdoutAlertPipe::builder()
            .format(OutputFormat::Json)
            .build();
        let event = sample_event();
        let line = pipe.format_event(&event);
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("json parse");
        assert_eq!(parsed["severity"], "warn");
        assert_eq!(parsed["kind"], "heartbeat");
        assert!(parsed["labels"].is_object());
        assert!(parsed["fired_at_micros"].is_number());
        assert!(parsed["payload_bytes_base64"].is_string());
    }

    #[proxima::test]
    async fn call_with_alert_method_and_typed_body_returns_ok() {
        let pipe = StdoutAlertPipe::builder().build();
        let request = alert_request(sample_event());
        let response = pipe.call(request).await.expect("call should succeed");
        assert_eq!(response.status, 200);
    }
}
