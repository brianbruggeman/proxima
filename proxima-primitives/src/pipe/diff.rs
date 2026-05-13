use std::future::Future;

use bytes::Bytes;

use crate::transport::{DEFAULT_REPLAY_CAP_BYTES, Replay};

use crate::pipe::Method;
use crate::pipe::ProximaError;
use crate::pipe::body::{ChunkStream, RequestStream};
use crate::pipe::primitives::Pipe;
use crate::pipe::SendPipe;
use crate::pipe::handler::{Handler, PipeHandle, ThreadLocalPipeHandle, into_handle};
use crate::pipe::request::{Request, Response};
use crate::pipe::telemetry_surface::Labels;

const COUNTER_IDENTICAL: &str = "proxima.diff.identical_total";
const COUNTER_DIVERGENT: &str = "proxima.diff.divergent_total";

/// fan-out → fan-in diff. send the same Request through `left` and
/// `right`, collect both responses, emit a single JSON diff describing
/// where they disagree. used to A/B two implementations against the
/// same input — record once, run both, see which bytes diverged.
///
/// the request body is teed (via the existing backpressure-aware tee) so
/// both branches see identical input bytes. each branch consumes the
/// body to completion before its response body is collected.
/// fan-out diff between two inners. Generic over the inner handle:
/// `Diff<PipeHandle>` impls `Handler`; `Diff<ThreadLocalPipeHandle>`
/// impls `ThreadLocalHandler`. Both branches must share handle type
/// because the resulting future is one Send-ness.
pub struct Diff<Inner = PipeHandle> {
    pub left: Inner,
    pub right: Inner,
    pub replay_cap_bytes: usize,
}

impl<Inner> Diff<Inner> {
    #[must_use]
    pub fn new(left: Inner, right: Inner) -> Self {
        Self {
            left,
            right,
            replay_cap_bytes: DEFAULT_REPLAY_CAP_BYTES,
        }
    }

    #[must_use]
    pub fn with_replay_cap_bytes(mut self, cap: usize) -> Self {
        self.replay_cap_bytes = cap;
        self
    }
}

impl<Inner> SendPipe for Diff<Inner>
where
    Inner: Handler + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let left = self.left.clone();
        let right = self.right.clone();
        let cap = self.replay_cap_bytes;
        let Request {
            method,
            path,
            query,
            metadata,
            payload,
            stream,
            context,
        } = request;
        async move {
            let source = source_stream(payload, stream);
            let (tee, primary_body) = Replay::wrap(source, cap);
            let left_request = rebuild(&method, &path, &query, &metadata, &context, primary_body);
            let right_body = tee.replay()?;
            let right_request = rebuild(&method, &path, &query, &metadata, &context, right_body);

            let (left_outcome, right_outcome) = futures::join!(
                SendPipe::call(&left, left_request),
                SendPipe::call(&right, right_request),
            );
            let left_snapshot = harvest("left", left_outcome).await;
            let right_snapshot = harvest("right", right_outcome).await;
            let report = build_report(&left_snapshot, &right_snapshot);
            let metric = if report.identical {
                COUNTER_IDENTICAL
            } else {
                COUNTER_DIVERGENT
            };
            context.telemetry.counter_inc(metric, &Labels::empty(), 1);
            let body_bytes = serde_json::to_vec_pretty(&report)
                .map_err(|err| ProximaError::Body(format!("diff serialize: {err}")))?;
            Ok(Response::new(if report.identical { 200 } else { 409 })
                .with_header("content-type", "application/json")
                .with_body(Bytes::from(body_bytes)))
        }
    }
}

impl Pipe for Diff<ThreadLocalPipeHandle> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let left = self.left.clone();
        let right = self.right.clone();
        let cap = self.replay_cap_bytes;
        let Request {
            method,
            path,
            query,
            metadata,
            payload,
            stream,
            context,
        } = request;
        async move {
            let source = source_stream(payload, stream);
            let (tee, primary_body) = Replay::wrap(source, cap);
            let left_request = rebuild(&method, &path, &query, &metadata, &context, primary_body);
            let right_body = tee.replay()?;
            let right_request = rebuild(&method, &path, &query, &metadata, &context, right_body);

            let (left_outcome, right_outcome) = futures::join!(
                Pipe::call(&left, left_request),
                Pipe::call(&right, right_request),
            );
            let left_snapshot = harvest("left", left_outcome).await;
            let right_snapshot = harvest("right", right_outcome).await;
            let report = build_report(&left_snapshot, &right_snapshot);
            let metric = if report.identical {
                COUNTER_IDENTICAL
            } else {
                COUNTER_DIVERGENT
            };
            context.telemetry.counter_inc(metric, &Labels::empty(), 1);
            let body_bytes = serde_json::to_vec_pretty(&report)
                .map_err(|err| ProximaError::Body(format!("diff serialize: {err}")))?;
            Ok(Response::new(if report.identical { 200 } else { 409 })
                .with_header("content-type", "application/json")
                .with_body(Bytes::from(body_bytes)))
        }
    }
}

#[derive(Debug)]
enum BranchSnapshot {
    Ok { status: u16, body: Bytes },
    Err { message: String },
}

async fn harvest(
    _label: &'static str,
    outcome: Result<Response<Bytes>, ProximaError>,
) -> BranchSnapshot {
    match outcome {
        Ok(response) => {
            let status = response.status;
            match response.collect_body().await {
                Ok(body) => BranchSnapshot::Ok { status, body },
                Err(err) => BranchSnapshot::Err {
                    message: format!("body collect: {err}"),
                },
            }
        }
        Err(err) => BranchSnapshot::Err {
            message: err.to_string(),
        },
    }
}

#[derive(Debug, serde::Serialize)]
struct DiffReport {
    identical: bool,
    summary: String,
    left: BranchReport,
    right: BranchReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_diff_offset: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
struct BranchReport {
    outcome: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    body_len: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn build_report(left: &BranchSnapshot, right: &BranchSnapshot) -> DiffReport {
    match (left, right) {
        (
            BranchSnapshot::Ok {
                status: left_status,
                body: left_body,
            },
            BranchSnapshot::Ok {
                status: right_status,
                body: right_body,
            },
        ) => {
            if left_status == right_status && left_body == right_body {
                return DiffReport {
                    identical: true,
                    summary: "byte-identical".into(),
                    left: branch_ok(*left_status, left_body.len()),
                    right: branch_ok(*right_status, right_body.len()),
                    first_diff_offset: None,
                };
            }
            let mut summary_parts: Vec<String> = Vec::new();
            if left_status != right_status {
                summary_parts.push(format!("status diverged ({left_status} vs {right_status})"));
            }
            let first_diff = if left_body != right_body {
                summary_parts.push(format!(
                    "body diverged ({} vs {} bytes)",
                    left_body.len(),
                    right_body.len()
                ));
                Some(first_byte_difference(left_body, right_body))
            } else {
                None
            };
            DiffReport {
                identical: false,
                summary: summary_parts.join(", "),
                left: branch_ok(*left_status, left_body.len()),
                right: branch_ok(*right_status, right_body.len()),
                first_diff_offset: first_diff,
            }
        }
        (BranchSnapshot::Err { message }, BranchSnapshot::Ok { .. }) => DiffReport {
            identical: false,
            summary: "left errored, right ok".into(),
            left: branch_err(message),
            right: branch_ok_from(right),
            first_diff_offset: None,
        },
        (BranchSnapshot::Ok { .. }, BranchSnapshot::Err { message }) => DiffReport {
            identical: false,
            summary: "right errored, left ok".into(),
            left: branch_ok_from(left),
            right: branch_err(message),
            first_diff_offset: None,
        },
        (
            BranchSnapshot::Err {
                message: left_message,
            },
            BranchSnapshot::Err {
                message: right_message,
            },
        ) => DiffReport {
            identical: left_message == right_message,
            summary: if left_message == right_message {
                "both errored identically".into()
            } else {
                format!("errors diverged: \"{left_message}\" vs \"{right_message}\"")
            },
            left: branch_err(left_message),
            right: branch_err(right_message),
            first_diff_offset: None,
        },
    }
}

fn first_byte_difference(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left_byte, right_byte)| left_byte == right_byte)
        .count()
}

fn branch_ok(status: u16, body_len: usize) -> BranchReport {
    BranchReport {
        outcome: "ok",
        status: Some(status),
        body_len: Some(body_len),
        error: None,
    }
}

fn branch_ok_from(snapshot: &BranchSnapshot) -> BranchReport {
    match snapshot {
        BranchSnapshot::Ok { status, body } => branch_ok(*status, body.len()),
        BranchSnapshot::Err { message } => branch_err(message),
    }
}

fn branch_err(message: &str) -> BranchReport {
    BranchReport {
        outcome: "error",
        status: None,
        body_len: None,
        error: Some(message.to_string()),
    }
}

fn source_stream(body: Bytes, stream: Option<RequestStream>) -> ChunkStream {
    match stream {
        Some(stream) => stream.into_chunk_stream(),
        None => Box::pin(futures::stream::once(async move { Ok(body) })),
    }
}

fn rebuild(
    method: &Method,
    path: &Bytes,
    query: &crate::pipe::header_list::HeaderList,
    metadata: &crate::pipe::header_list::HeaderList,
    context: &crate::pipe::request::RequestContext,
    body: ChunkStream,
) -> Request<Bytes> {
    Request {
        method: method.clone(),
        path: Bytes::clone(path),
        query: query.clone(),
        metadata: metadata.clone(),
        payload: Bytes::new(),
        stream: Some(RequestStream::from_chunk_stream(body)),
        context: context.clone(),
    }
}

/// helper that wraps two PipeHandles into a Diff Pipe handle.
/// preferred entry point until config-driven dual-inner is supported.
#[must_use]
pub fn diff_handle(left: PipeHandle, right: PipeHandle) -> PipeHandle {
    into_handle(Diff::new(left, right))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::telemetry_surface::Telemetry;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct CounterRecorder {
        counters: Mutex<HashMap<String, u64>>,
    }

    impl Telemetry for CounterRecorder {
        fn counter_inc(&self, metric: &str, _labels: &Labels, by: u64) {
            *self
                .counters
                .lock()
                .unwrap()
                .entry(metric.to_string())
                .or_insert(0) += by;
        }
        fn gauge_set(&self, _: &str, _: &Labels, _: i64) {}
        fn histogram_record(&self, _: &str, _: &Labels, _: f64) {}
    }

    impl CounterRecorder {
        fn counter(&self, metric: &str) -> Option<u64> {
            self.counters.lock().unwrap().get(metric).copied()
        }
    }

    struct ConstantPipe {
        status: u16,
        body: &'static [u8],
    }

    impl SendPipe for ConstantPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let status = self.status;
            let body = self.body;
            async move { Ok(Response::new(status).with_body(Bytes::from_static(body))) }
        }
    }

    fn fresh_request() -> Request<Bytes> {
        Request::builder()
            .method("POST")
            .path("/x")
            .body("input")
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn identical_branches_report_byte_identical() {
        let diff = Diff::new(
            into_handle(ConstantPipe {
                status: 200,
                body: b"hello",
            }),
            into_handle(ConstantPipe {
                status: 200,
                body: b"hello",
            }),
        );
        let response = SendPipe::call(&diff, fresh_request()).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("body");
        let report: serde_json::Value = serde_json::from_slice(&body).expect("parse diff report");
        assert_eq!(report["identical"], serde_json::Value::Bool(true));
    }

    #[proxima::test]
    async fn body_divergence_reported_with_first_diff_offset() {
        let diff = Diff::new(
            into_handle(ConstantPipe {
                status: 200,
                body: b"hello world",
            }),
            into_handle(ConstantPipe {
                status: 200,
                body: b"hello there",
            }),
        );
        let response = SendPipe::call(&diff, fresh_request()).await.expect("call");
        assert_eq!(response.status, 409, "diverged → 409 conflict");
        let body = response.collect_body().await.expect("body");
        let report: serde_json::Value = serde_json::from_slice(&body).expect("parse diff report");
        assert_eq!(report["identical"], serde_json::Value::Bool(false));
        assert_eq!(report["first_diff_offset"], serde_json::json!(6));
    }

    #[proxima::test]
    async fn status_divergence_surfaces_in_summary() {
        let diff = Diff::new(
            into_handle(ConstantPipe {
                status: 200,
                body: b"x",
            }),
            into_handle(ConstantPipe {
                status: 500,
                body: b"x",
            }),
        );
        let response = SendPipe::call(&diff, fresh_request()).await.expect("call");
        assert_eq!(response.status, 409);
        let body = response.collect_body().await.expect("body");
        let report: serde_json::Value = serde_json::from_slice(&body).expect("parse diff report");
        let summary = report["summary"].as_str().expect("summary");
        assert!(
            summary.contains("200"),
            "summary should mention left status: {summary}"
        );
        assert!(
            summary.contains("500"),
            "summary should mention right status: {summary}"
        );
    }

    #[proxima::test]
    async fn identical_call_emits_identical_counter() {
        let metrics = std::sync::Arc::new(CounterRecorder::default());
        let diff = Diff::new(
            into_handle(ConstantPipe {
                status: 200,
                body: b"hello",
            }),
            into_handle(ConstantPipe {
                status: 200,
                body: b"hello",
            }),
        );
        let request = Request::builder()
            .method("POST")
            .path("/x")
            .body("input")
            .telemetry(metrics.clone())
            .build()
            .expect("builder");
        let _response = SendPipe::call(&diff, request).await.expect("call");
        assert_eq!(metrics.counter(COUNTER_IDENTICAL), Some(1));
        assert!(metrics.counter(COUNTER_DIVERGENT).is_none());
    }

    #[proxima::test]
    async fn divergent_call_emits_divergent_counter() {
        let metrics = std::sync::Arc::new(CounterRecorder::default());
        let diff = Diff::new(
            into_handle(ConstantPipe {
                status: 200,
                body: b"left",
            }),
            into_handle(ConstantPipe {
                status: 200,
                body: b"right",
            }),
        );
        let request = Request::builder()
            .method("POST")
            .path("/x")
            .body("input")
            .telemetry(metrics.clone())
            .build()
            .expect("builder");
        let _response = SendPipe::call(&diff, request).await.expect("call");
        assert_eq!(metrics.counter(COUNTER_DIVERGENT), Some(1));
        assert!(metrics.counter(COUNTER_IDENTICAL).is_none());
    }
}
