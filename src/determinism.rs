use std::sync::Arc;

use bytes::Bytes;

use crate::capture_surface::CaptureContext;
use crate::error::ProximaError;
use proxima_primitives::pipe::SendPipe;

use crate::pipe::PipeHandle;
use crate::recording::LiveCaptureContext;
use crate::recording::event::FrameMetadata;
use crate::request::{Request, Response};

/// Run the Handler twice on freshly-built copies of the same request and
/// assert that both calls produced byte-for-byte identical output AND
/// identical captured frame metadata. Returns `Ok` on match, `Err` with a
/// human-readable description of the first divergence found.
///
/// Handler authors should call this from their own test suites to lock the
/// determinism contract in CI — a violation means the Handler is consulting
/// hidden state (clock, RNG, external API) without stashing the entropy
/// into `request.context.capture`, which would silently break replay.
pub async fn check_determinism<F, R>(build: F, request: R) -> Result<(), String>
where
    F: Fn() -> PipeHandle,
    R: Fn() -> Request<Bytes>,
{
    let first = capture_call(build(), request()).await;
    let second = capture_call(build(), request()).await;
    diff_snapshots(&first, &second)
}

#[derive(Debug)]
enum CallSnapshot {
    Ok {
        status: u16,
        headers: Vec<(Bytes, Bytes)>,
        body: Bytes,
        metadata: FrameMetadata,
    },
    Err {
        error: String,
        metadata: FrameMetadata,
    },
}

async fn capture_call(pipe: PipeHandle, mut request: Request<Bytes>) -> CallSnapshot {
    let capture = Arc::new(LiveCaptureContext::new());
    request.context.capture = Some(capture.clone() as Arc<dyn CaptureContext>);
    let outcome = SendPipe::call(&pipe, request).await;
    match outcome {
        Ok(response) => match harvest(response).await {
            Ok((status, headers, body)) => CallSnapshot::Ok {
                status,
                headers,
                body,
                metadata: capture.drain(),
            },
            Err(err) => CallSnapshot::Err {
                error: format!("body collect failed: {err}"),
                metadata: capture.drain(),
            },
        },
        Err(error) => CallSnapshot::Err {
            error: error.to_string(),
            metadata: capture.drain(),
        },
    }
}

async fn harvest(
    response: Response<Bytes>,
) -> Result<(u16, Vec<(Bytes, Bytes)>, Bytes), ProximaError> {
    let status = response.status;
    let header_snapshot: Vec<(Bytes, Bytes)> = response
        .metadata
        .iter()
        .map(|(name, value)| (Bytes::copy_from_slice(name), Bytes::copy_from_slice(value)))
        .collect();
    let body_bytes = response.collect_body().await?;
    Ok((status, header_snapshot, body_bytes))
}

fn diff_snapshots(left: &CallSnapshot, right: &CallSnapshot) -> Result<(), String> {
    match (left, right) {
        (
            CallSnapshot::Ok {
                status: status_left,
                headers: headers_left,
                body: body_left,
                metadata: metadata_left,
            },
            CallSnapshot::Ok {
                status: status_right,
                headers: headers_right,
                body: body_right,
                metadata: metadata_right,
            },
        ) => {
            if status_left != status_right {
                return Err(format!("status diverged: {status_left} vs {status_right}"));
            }
            if headers_left != headers_right {
                return Err("response headers diverged between runs".to_string());
            }
            if body_left != body_right {
                return Err(format!(
                    "response body diverged: {} bytes vs {} bytes",
                    body_left.len(),
                    body_right.len()
                ));
            }
            diff_metadata(metadata_left, metadata_right)
        }
        (
            CallSnapshot::Err {
                error: error_left,
                metadata: metadata_left,
            },
            CallSnapshot::Err {
                error: error_right,
                metadata: metadata_right,
            },
        ) => {
            if error_left != error_right {
                return Err(format!(
                    "errors diverged: \"{error_left}\" vs \"{error_right}\""
                ));
            }
            diff_metadata(metadata_left, metadata_right)
        }
        (left_snapshot, right_snapshot) => Err(format!(
            "shape diverged: {} vs {}",
            shape(left_snapshot),
            shape(right_snapshot)
        )),
    }
}

fn diff_metadata(left: &FrameMetadata, right: &FrameMetadata) -> Result<(), String> {
    if left == right {
        return Ok(());
    }
    for (key, value_left) in left {
        match right.get(key) {
            Some(value_right) if value_right == value_left => {}
            Some(value_right) => {
                return Err(format!(
                    "captured metadata key \"{key}\" diverged: {} bytes vs {} bytes",
                    value_left.len(),
                    value_right.len()
                ));
            }
            None => {
                return Err(format!(
                    "captured metadata key \"{key}\" present in first run, missing in second"
                ));
            }
        }
    }
    for key in right.keys() {
        if !left.contains_key(key) {
            return Err(format!(
                "captured metadata key \"{key}\" present in second run, missing in first"
            ));
        }
    }
    Ok(())
}

fn shape(snapshot: &CallSnapshot) -> &'static str {
    match snapshot {
        CallSnapshot::Ok { .. } => "ok",
        CallSnapshot::Err { .. } => "error",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::into_handle;

    fn fresh_request() -> Request<Bytes> {
        Request::builder()
            .method("GET")
            .path("/x")
            .body("ping")
            .build()
            .expect("builder")
    }

    struct ConstantEcho;
    impl SendPipe for ConstantEcho {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> {
            async { Ok(Response::ok(Bytes::from_static(b"echo"))) }
        }
    }


    struct CapturingClockPipe {
        ticks: std::sync::atomic::AtomicU64,
    }
    impl SendPipe for CapturingClockPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> {
            let now = self.ticks.fetch_add(0, std::sync::atomic::Ordering::SeqCst);
            async move {
                if let Some(capture) = request.context.capture.as_ref() {
                    capture.attach("clock", Bytes::copy_from_slice(&now.to_be_bytes()));
                }
                Ok(Response::ok(Bytes::from_static(b"done")))
            }
        }
    }


    struct DriftingClockPipe {
        ticks: std::sync::atomic::AtomicU64,
    }
    impl SendPipe for DriftingClockPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> {
            // monotonically advancing per-call clock — captures different values
            // each call, which the harness must catch.
            let now = self.ticks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if let Some(capture) = request.context.capture.as_ref() {
                    capture.attach("clock", Bytes::copy_from_slice(&now.to_be_bytes()));
                }
                Ok(Response::ok(Bytes::from_static(b"done")))
            }
        }
    }


    #[proxima::test]
    async fn pure_pipe_passes_determinism_check() {
        let outcome = check_determinism(|| into_handle(ConstantEcho), fresh_request).await;
        outcome.expect("ConstantEcho must be deterministic");
    }

    #[proxima::test]
    async fn pipe_with_stable_captured_clock_passes() {
        let outcome = check_determinism(
            || {
                into_handle(CapturingClockPipe {
                    ticks: std::sync::atomic::AtomicU64::new(42),
                })
            },
            fresh_request,
        )
        .await;
        outcome.expect("stable captured clock must round-trip identically");
    }

    #[proxima::test]
    async fn drifting_clock_fails_determinism_check_with_metadata_diagnostic() {
        // factory shares a single counter so both Pipes see different values
        // — guarantees the captured metadata diverges between runs.
        let ticks = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let ticks_for_build = ticks.clone();
        let outcome = check_determinism(
            move || {
                let pinned = ticks_for_build.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                into_handle(DriftingClockPipe {
                    ticks: std::sync::atomic::AtomicU64::new(pinned),
                })
            },
            fresh_request,
        )
        .await;
        let err = outcome.expect_err("drifting clock must be flagged");
        assert!(
            err.contains("clock"),
            "diagnostic should name the divergent key: {err}"
        );
    }
}
