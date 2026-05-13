#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Phase 8 demo (proxima-notify initiative): heartbeat-to-stdout end-to-end.
//!
//! Exercises the substrate + facade composition:
//!   S2 (producer-lifecycle driver in proxima::App)
//!     + S3 (producer-graph in ProximaSettings — via App::pipe / Spec::Handle)
//!     + C3 (ScheduledTriggerPipe — interval producer)
//!     + C4 (StdoutAlertPipe — terminal sink)

#![cfg(all(feature = "scheduled-trigger", feature = "stdout-alert"))]

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::request::{Request, RequestContext, Response};

use proxima_patterns::alert::event::{AlertEvent, Severity};
use proxima_patterns::alert::pipes::{AlertPipeHandle, AlertRequest, into_alert_handle};
use proxima_patterns::alert::scheduled_trigger::{Schedule, ScheduledTriggerPipe};
use proxima_patterns::alert::stdout_alert::StdoutAlertPipe;

/// A test sink that captures AlertEvents AND delegates to a real sink.
struct CaptureThenDelegate {
    captured: Mutex<Vec<AlertEvent>>,
    inner: AlertPipeHandle,
}

impl CaptureThenDelegate {
    fn new(inner: AlertPipeHandle) -> Arc<Self> {
        Arc::new(Self {
            captured: Mutex::new(Vec::new()),
            inner,
        })
    }
}

impl SendPipe for CaptureThenDelegate {
    type In = AlertRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: AlertRequest,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let _ = self
            .captured
            .lock()
            .map(|mut guard| guard.push(request.payload.clone()));
        let inner = self.inner.clone();
        async move { SendPipe::call(inner.as_ref(), request).await }
    }
}

#[proxima::test]
async fn heartbeat_producer_fires_alert_events_to_stdout_sink_end_to_end() {
    let stdout_sink = into_alert_handle(StdoutAlertPipe::default());
    let capture = CaptureThenDelegate::new(stdout_sink);
    let inner = into_alert_handle(Arc::clone(&capture));

    let producer = ScheduledTriggerPipe::builder()
        .schedule(Schedule::Interval(Duration::from_millis(5)))
        .inner(inner)
        .event_kind("heartbeat")
        .source_label("phase8_demo")
        .severity(Severity::Info)
        .build()
        .expect("producer builder");

    // drive the producer as a SourcePipe (Signal -> ()) inline until an
    // event lands, then fire cancel so the loop returns; the loop never
    // resolves on its own before that, so the watcher arm always wins.
    use futures::FutureExt as _;
    let cancel = proxima_core::signal::Signal::new();
    let loop_future = SendPipe::call(&producer, cancel.clone()).fuse();
    let capture_for_watch = Arc::clone(&capture);
    let watcher = async move {
        while capture_for_watch.captured.lock().expect("mutex").is_empty() {
            proxima_core::time::sleep(Duration::from_millis(1)).await;
        }
    }
    .fuse();
    futures::pin_mut!(loop_future, watcher);
    let raced = async {
        futures::select_biased! {
            () = watcher => (),
            _ = loop_future => panic!("producer loop exited without being fired"),
        }
    };
    proxima_core::time::timeout(Duration::from_secs(10), raced)
        .await
        .expect("heartbeat should fire before the timeout");
    cancel.fire();

    let captured = capture.captured.lock().expect("mutex");
    assert!(
        !captured.is_empty(),
        "phase 8 demo: ScheduledTriggerPipe → CaptureThenDelegate → \
         StdoutAlertPipe should have fired at least once on the house clock"
    );

    for event in captured.iter() {
        assert_eq!(event.severity, Severity::Info);
        assert_eq!(event.kind.as_str(), "heartbeat");
        let source = event
            .labels
            .get(&proxima_patterns::alert::event::LabelKey::try_from("source").unwrap())
            .map(|value| value.as_str().to_string());
        assert_eq!(source.as_deref(), Some("phase8_demo"));
    }

    println!(
        "phase 8 demo: captured {} heartbeat alert(s); pipeline runs end-to-end.",
        captured.len()
    );
}

#[proxima::test(start_paused = true)]
async fn unknown_method_routed_to_stdout_sink_returns_405_without_capture() {
    let stdout_sink = into_alert_handle(StdoutAlertPipe::default());
    let bogus_event = proxima_patterns::alert::event::AlertEvent {
        id: proxima_patterns::alert::event::AlertId(ulid::Ulid::nil()),
        severity: Severity::Info,
        kind: proxima_patterns::alert::event::KindString::try_from("test").unwrap(),
        labels: proxima_patterns::alert::event::LabelMap::new(),
        payload: proxima_patterns::alert::event::Payload::new(),
        fired_at_micros: 0,
    };
    let bogus_request = Request {
        method: proxima_primitives::pipe::method::Method::Get,
        path: Bytes::from_static(b"/"),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload: bogus_event,
        stream: None,
        context: RequestContext::default(),
    };
    let response = SendPipe::call(stdout_sink.as_ref(), bogus_request)
        .await
        .expect("call");
    assert_eq!(response.status, 405);
}
