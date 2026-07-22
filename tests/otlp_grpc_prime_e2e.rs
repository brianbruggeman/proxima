//! OTLP/gRPC export over the prime wire — the "no codec pipe, just a Client
//! configured for h2 + the codec data" shape. The codec data is the grpc-framed
//! `ExportTraceServiceRequest` (the existing `OtlpGrpcPipe` / `OtlpGrpcExporter`
//! produces it); the transport is `proxima::Client` resolved to the native h2
//! client via the `grpc` spec-key factory. The client POSTs the framed bytes to
//! `/opentelemetry.../Export` (content-type `application/grpc`) and a prime h2
//! collector prost-decodes them off the wire.

#![cfg(all(
    feature = "otlp-grpc",
    feature = "http-prime",
    any(target_os = "linux", target_os = "macos")
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use prost::Message as _;

use proxima::Client;
use proxima::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::prime::os::core_shard::spawn_on_current_core;
use proxima::runtime::prime::os::net::TcpListener;
use proxima::telemetry::capture::capture;
use proxima::telemetry::out::otlp_http::proto::ExportTraceServiceRequest;
use proxima::telemetry::pipes::{OtlpGrpcPipe, span_batch_request};
use proxima_primitives::pipe::SendPipe;

const GRPC_FRAME_HEADER_LEN: usize = 5;
const TRACE_EXPORT_PATH: &str = "/opentelemetry.proto.collector.trace.v1.TraceService/Export";

// Prime h2 gRPC collector: drains the request body (the grpc frame), strips the
// 5-byte header, prost-decodes the OTLP ExportTraceServiceRequest, counts spans.
struct GrpcSpanCollector {
    spans: Arc<AtomicUsize>,
}

impl SendPipe for GrpcSpanCollector {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let spans = Arc::clone(&self.spans);
        async move {
            let (_request, body) = request.body_bytes().await?;
            if body.len() < GRPC_FRAME_HEADER_LEN {
                return Err(ProximaError::Upstream(
                    "grpc frame shorter than header".into(),
                ));
            }
            let proto = &body[GRPC_FRAME_HEADER_LEN..];
            let decoded = ExportTraceServiceRequest::decode(proto)
                .map_err(|err| ProximaError::Upstream(format!("decode OTLP/grpc: {err}")))?;
            let count: usize = decoded
                .resource_spans
                .iter()
                .flat_map(|resource| resource.scope_spans.iter())
                .map(|scope| scope.spans.len())
                .sum();
            spans.fetch_add(count, Ordering::SeqCst);
            Ok(Response::new(200).with_body(Bytes::new()))
        }
    }
}


fn seven_spans() -> Vec<proxima::telemetry::trace::SpanRecord> {
    let spans = capture(|recorder| {
        for _ in 0..7 {
            recorder.span("export").tag("k", "v").start();
        }
    })
    .spans();
    assert_eq!(spans.len(), 7);
    spans
}

#[proxima::test]
async fn otlp_grpc_export_over_prime_wire_via_client() {
    // The codec data: grpc-framed ExportTraceServiceRequest from 7 sample spans,
    // produced by the existing OtlpGrpcPipe (buffer via call, frame via flush).
    let grpc_pipe = OtlpGrpcPipe::new("");
    let _ = grpc_pipe.call(span_batch_request(seven_spans())).await;
    let framed = grpc_pipe.flush_spans();
    assert!(
        framed.len() > GRPC_FRAME_HEADER_LEN,
        "grpc frame carries the encoded spans"
    );

    let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
    let bound = listener.local_addr().expect("local_addr");

    let spans = Arc::new(AtomicUsize::new(0));
    let dispatch: PipeHandle = into_handle(GrpcSpanCollector {
        spans: Arc::clone(&spans),
    });
    spawn_on_current_core(Box::pin(async move {
        if let Ok((socket, _peer)) = listener.accept().await {
            let admission = proxima_listen::admission::ConnAdmission::unbounded();
            let _ = serve_h2_connection(socket, dispatch, admission, None).await;
        }
    }));

    // proxima::Client configured for h2 (the `grpc` spec key -> H2ClientUpstream),
    // POSTing the codec data — no bespoke gRPC codec pipe.
    let client = Client::from_value(serde_json::json!({ "grpc": format!("http://{bound}") }))
        .expect("client");
    let response = client
        .post(TRACE_EXPORT_PATH)
        .header("content-type", "application/grpc")
        .body(framed)
        .send()
        .await
        .expect("grpc export over h2");

    assert_eq!(response.status(), 200, "h2 server accepted the gRPC POST");
    assert_eq!(
        spans.load(Ordering::SeqCst),
        7,
        "prime h2 collector prost-decoded the 7 spans from the grpc frame off the wire"
    );
}
