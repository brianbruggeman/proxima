//! C3 — `ScheduledTriggerPipe`: interval-driven producer
//! [`SourcePipe`](proxima_primitives::pipe::SourcePipe).
//!
//! Wraps [`proxima_core::time::interval`] in a `SendPipe<Signal, ()>` impl whose
//! `call` runs the schedule loop until the passed-in [`Signal`] fires.
//! Each tick constructs an `AlertEvent` and dispatches it through the
//! inner Pipe via the method-byte discriminant `b"SCHEDULED_TICK"`.
//!
//! `ScheduledTriggerPipe` is a source, not a handler: mounting one on a
//! listener is a compile error (its `SendPipe` shape is `Signal -> ()`,
//! not `Request<Bytes> -> Response<Bytes>`).
//!
//! # Markers
//!
//! - NOT `WithoutTime` — reads the clock to schedule.
//! - NOT `IdempotentSideEffectFree` — `fired_at_micros` varies per call;
//!   observationally not identical across repeated dispatches even if the
//!   inner Pipe is idempotent.
//! - Inherits `WithoutFilesystem`, `WithoutNetwork`, `WithoutSpawn`,
//!   `WithoutRandom` from the inner Pipe via blanket AND-composition.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures::FutureExt;
use proxima_core::signal::Signal;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_core::factory::Named;
use proxima_primitives::pipe::source::{SourceFactory, SourceHandle, into_source_handle};
use proxima_primitives::pipe::request::{Request, RequestContext, Response};

use crate::alert::event::{
    AlertEvent, AlertId, KindString, LabelKey, LabelMap, LabelValue, Payload, Severity,
};
use crate::alert::methods;
use crate::alert::pipes::{AlertPipeHandle, AlertRequest, into_alert_handle};

/// Schedule shape for a [`ScheduledTriggerPipe`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Schedule {
    /// Fire every `period` after the first tick.
    Interval(Duration),
}

/// Producer Pipe that fires an [`AlertEvent`] into an inner Pipe on a
/// scheduled cadence.
pub struct ScheduledTriggerPipe {
    schedule: Schedule,
    inner: AlertPipeHandle,
    event_kind: String,
    source_label: String,
    severity: Severity,
    fire_count: Arc<AtomicU64>,
}

impl ScheduledTriggerPipe {
    /// Construct directly. Most callers should use [`Self::builder`].
    #[must_use]
    pub fn new(
        schedule: Schedule,
        inner: AlertPipeHandle,
        event_kind: impl Into<String>,
        source_label: impl Into<String>,
        severity: Severity,
    ) -> Self {
        Self {
            schedule,
            inner,
            event_kind: event_kind.into(),
            source_label: source_label.into(),
            severity,
            fire_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Fluent builder entry point.
    #[must_use]
    pub fn builder() -> ScheduledTriggerPipeBuilder {
        ScheduledTriggerPipeBuilder::default()
    }

    /// Number of times the trigger has fired so far.
    #[must_use]
    pub fn fire_count(&self) -> u64 {
        self.fire_count.load(Ordering::Relaxed)
    }
}

impl SendPipe for ScheduledTriggerPipe {
    type In = Signal;
    type Out = ();
    type Err = ProximaError;

    fn call(&self, cancel: Signal) -> impl std::future::Future<Output = Result<(), ProximaError>> + Send {
        let inner = self.inner.clone();
        let schedule = self.schedule.clone();
        let event_kind = self.event_kind.clone();
        let source_label = self.source_label.clone();
        let severity = self.severity;
        let fire_count = self.fire_count.clone();
        async move {
            run_schedule_loop(
                inner,
                schedule,
                event_kind,
                source_label,
                severity,
                fire_count,
                cancel,
            )
            .await;
            Ok(())
        }
    }
}

/// Config-driven [`SourceFactory`] for [`ScheduledTriggerPipe`]. `spec`
/// carries the interval period in milliseconds; the inner alert handle,
/// event kind, source label, and severity are supplied at construction.
pub struct ScheduledTriggerSourceFactory {
    name: String,
    inner: AlertPipeHandle,
    event_kind: String,
    source_label: String,
    severity: Severity,
}

impl ScheduledTriggerSourceFactory {
    /// Register a thin factory that builds a [`ScheduledTriggerPipe`] under
    /// `name`, dispatching every tick to `inner`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        inner: AlertPipeHandle,
        event_kind: impl Into<String>,
        source_label: impl Into<String>,
        severity: Severity,
    ) -> Self {
        Self {
            name: name.into(),
            inner,
            event_kind: event_kind.into(),
            source_label: source_label.into(),
            severity,
        }
    }
}

impl Named for ScheduledTriggerSourceFactory {
    fn name(&self) -> &str {
        &self.name
    }
}

impl SourceFactory for ScheduledTriggerSourceFactory {
    fn build(&self, spec: &serde_json::Value) -> Result<SourceHandle, ProximaError> {
        let period_ms = spec
            .get("period_ms")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                ProximaError::Config("scheduled_trigger source requires period_ms".into())
            })?;
        let pipe = ScheduledTriggerPipe::new(
            Schedule::Interval(Duration::from_millis(period_ms)),
            self.inner.clone(),
            self.event_kind.clone(),
            self.source_label.clone(),
            self.severity,
        );
        Ok(into_source_handle(pipe))
    }
}

async fn run_schedule_loop(
    inner: AlertPipeHandle,
    schedule: Schedule,
    event_kind: String,
    source_label: String,
    severity: Severity,
    fire_count: Arc<AtomicU64>,
    cancel: Signal,
) {
    match schedule {
        Schedule::Interval(period) => {
            let mut interval = proxima_core::time::interval(period);
            interval.set_missed_tick_behavior(proxima_core::time::MissedTickBehavior::Skip);
            interval.tick().await;
            loop {
                let cancel_fut = Box::pin(cancel.fired()).fuse();
                let tick_fut = Box::pin(interval.tick()).fuse();
                futures::pin_mut!(cancel_fut);
                futures::pin_mut!(tick_fut);
                let should_fire = futures::select_biased! {
                    _ = cancel_fut => {
                        tracing::debug!(event_kind = %event_kind, "scheduled_trigger cancelled");
                        return;
                    }
                    _ = tick_fut => true,
                };
                if should_fire {
                    let fired_at_micros = micros_since_epoch();
                    let event = build_alert_event(
                        &event_kind,
                        &source_label,
                        severity,
                        fired_at_micros,
                        fire_count.fetch_add(1, Ordering::Relaxed),
                    );
                    let request = build_scheduled_tick_request(event);
                    if let Err(err) = SendPipe::call(inner.as_ref(), request).await {
                        tracing::error!(
                            ?err,
                            event_kind = %event_kind,
                            "scheduled_trigger inner call failed"
                        );
                    }
                }
            }
        }
    }
}

fn build_alert_event(
    event_kind: &str,
    source_label: &str,
    severity: Severity,
    fired_at_micros: u64,
    fire_seq: u64,
) -> AlertEvent {
    let kind = truncate_to_kind_string(event_kind);
    let mut labels = LabelMap::new();
    let _ = labels.insert(
        LabelKey::try_from("source").unwrap_or_default(),
        truncate_to_label_value(source_label),
    );
    let mut fire_seq_buf = LabelValue::new();
    use core::fmt::Write;
    let _ = write!(&mut fire_seq_buf, "{fire_seq}");
    let _ = labels.insert(
        LabelKey::try_from("fire_seq").unwrap_or_default(),
        fire_seq_buf,
    );

    let id = AlertId(ulid::Ulid::new());
    AlertEvent {
        id,
        severity,
        kind,
        labels,
        payload: Payload::new(),
        fired_at_micros,
    }
}

fn build_scheduled_tick_request(event: AlertEvent) -> AlertRequest {
    Request {
        method: methods::scheduled_tick_method(),
        path: Bytes::from_static(b"/notify/scheduled_tick"),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload: event,
        stream: None,
        context: RequestContext::default(),
    }
}

fn micros_since_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_micros()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn truncate_to_kind_string(value: &str) -> KindString {
    let max = crate::alert::event::sized::ALERT_KIND_MAX;
    let truncated = if value.len() > max {
        &value[..max]
    } else {
        value
    };
    KindString::try_from(truncated).unwrap_or_default()
}

fn truncate_to_label_value(value: &str) -> LabelValue {
    let max = crate::alert::event::sized::ALERT_LABEL_VAL_MAX;
    let truncated = if value.len() > max {
        &value[..max]
    } else {
        value
    };
    LabelValue::try_from(truncated).unwrap_or_default()
}

/// Builder for [`ScheduledTriggerPipe`] (principle 4).
#[derive(Default)]
pub struct ScheduledTriggerPipeBuilder {
    schedule: Option<Schedule>,
    inner: Option<AlertPipeHandle>,
    event_kind: Option<String>,
    source_label: Option<String>,
    severity: Option<Severity>,
}

impl ScheduledTriggerPipeBuilder {
    /// Set the schedule (e.g. `Schedule::Interval(Duration::from_secs(5))`).
    #[must_use]
    pub fn schedule(mut self, schedule: Schedule) -> Self {
        self.schedule = Some(schedule);
        self
    }

    /// Set the inner alert pipe that receives each fired event.
    #[must_use]
    pub fn inner(mut self, inner: AlertPipeHandle) -> Self {
        self.inner = Some(inner);
        self
    }

    /// Convenience: wrap any compatible pipe in an `AlertPipeHandle`.
    #[must_use]
    pub fn inner_pipe<P>(mut self, pipe: P) -> Self
    where
        P: SendPipe<In = AlertRequest, Out = Response<Bytes>, Err = ProximaError>
            + Send
            + Sync
            + 'static,
    {
        self.inner = Some(into_alert_handle(pipe));
        self
    }

    /// Set the AlertEvent `kind` label (e.g. `"heartbeat"`).
    #[must_use]
    pub fn event_kind(mut self, value: impl Into<String>) -> Self {
        self.event_kind = Some(value.into());
        self
    }

    /// Set the AlertEvent `labels.source` label (default `"scheduled_trigger"`).
    #[must_use]
    pub fn source_label(mut self, value: impl Into<String>) -> Self {
        self.source_label = Some(value.into());
        self
    }

    /// Set the AlertEvent severity (default `Severity::Info`).
    #[must_use]
    pub fn severity(mut self, severity: Severity) -> Self {
        self.severity = Some(severity);
        self
    }

    /// Build the immutable [`ScheduledTriggerPipe`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if either `schedule` or `inner` is unset.
    pub fn build(self) -> Result<ScheduledTriggerPipe, BuildError> {
        let schedule = self.schedule.ok_or(BuildError::MissingSchedule)?;
        let inner = self.inner.ok_or(BuildError::MissingInner)?;
        let event_kind = self.event_kind.unwrap_or_else(|| "tick".to_string());
        let source_label = self
            .source_label
            .unwrap_or_else(|| "scheduled_trigger".to_string());
        let severity = self.severity.unwrap_or(Severity::Info);
        Ok(ScheduledTriggerPipe::new(
            schedule,
            inner,
            event_kind,
            source_label,
            severity,
        ))
    }
}

/// Errors from [`ScheduledTriggerPipeBuilder::build`].
#[derive(Debug)]
#[non_exhaustive]
pub enum BuildError {
    /// `.schedule(...)` was not called.
    MissingSchedule,
    /// `.inner(...)` was not called.
    MissingInner,
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSchedule => {
                write!(formatter, "ScheduledTriggerPipe builder missing schedule")
            }
            Self::MissingInner => write!(formatter, "ScheduledTriggerPipe builder missing inner"),
        }
    }
}

impl std::error::Error for BuildError {}

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
    use std::sync::Mutex;

    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::request::Response;

    use crate::alert::pipes::{AlertRequest, into_alert_handle};

    use super::*;

    struct CapturingPipe {
        captured: Mutex<Vec<AlertEvent>>,
    }

    impl CapturingPipe {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                captured: Mutex::new(Vec::new()),
            })
        }
    }

    impl SendPipe for CapturingPipe {
        type In = AlertRequest;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: AlertRequest,
        ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send
        {
            let _ = self
                .captured
                .lock()
                .map(|mut guard| guard.push(request.payload.clone()));
            async { Ok(Response::ok(Bytes::new())) }
        }
    }

    #[proxima::test]
    async fn scheduled_trigger_fires_three_ticks_on_the_house_clock() {
        let capture = CapturingPipe::new();
        let inner = into_alert_handle(Arc::clone(&capture));
        let pipe = ScheduledTriggerPipe::builder()
            .schedule(Schedule::Interval(Duration::from_millis(5)))
            .inner(inner)
            .event_kind("heartbeat")
            .severity(Severity::Info)
            .build()
            .expect("builder");

        // drive the source inline until three events land; the loop never
        // resolves on its own before cancel fires, so the watcher arm
        // always wins.
        let cancel = Signal::new();
        let loop_future = SendPipe::call(&pipe, cancel.clone()).fuse();
        let capture_for_watch = Arc::clone(&capture);
        let watcher = async move {
            while capture_for_watch.captured.lock().unwrap().len() < 3 {
                proxima_core::time::sleep(Duration::from_millis(1)).await;
            }
        }
        .fuse();
        futures::pin_mut!(loop_future, watcher);
        let raced = async {
            futures::select_biased! {
                () = watcher => (),
                _ = loop_future => panic!("schedule loop exited without being fired"),
            }
        };
        proxima_core::time::timeout(Duration::from_secs(10), raced)
            .await
            .expect("scheduled_trigger should fire three ticks before the timeout");
        cancel.fire();

        let captured = capture.captured.lock().unwrap();
        assert!(
            captured.len() >= 3,
            "scheduled_trigger should have fired three ticks on the house clock"
        );
        for event in captured.iter() {
            assert_eq!(event.severity, Severity::Info);
            assert_eq!(event.kind.as_str(), "heartbeat");
            assert!(
                event
                    .labels
                    .get(&LabelKey::try_from("source").unwrap())
                    .is_some(),
                "every event should have a 'source' label"
            );
        }
    }

    #[proxima::test]
    async fn builder_missing_schedule_returns_build_error() {
        let capture = CapturingPipe::new();
        let inner = into_alert_handle(Arc::clone(&capture));
        let result = ScheduledTriggerPipe::builder().inner(inner).build();
        assert!(matches!(result, Err(BuildError::MissingSchedule)));
    }

    #[proxima::test]
    async fn builder_missing_inner_returns_build_error() {
        let result = ScheduledTriggerPipe::builder()
            .schedule(Schedule::Interval(Duration::from_secs(1)))
            .build();
        assert!(matches!(result, Err(BuildError::MissingInner)));
    }
}
