use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::balancer::upstream_ref::{ThreadLocalUpstreamRef, UpstreamRef};
use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_primitives::transport::{DEFAULT_REPLAY_CAP_BYTES, Replay};
use proxima_primitives::pipe::body::{ChunkStream, RequestStream};
use proxima_primitives::pipe::{Pipe, SendPipe};
#[cfg(test)]
use proxima_primitives::pipe::handler::Handler;
use proxima_primitives::pipe::request::{Request, Response};

fn source_stream(body: Bytes, stream: Option<RequestStream>) -> ChunkStream {
    match stream {
        Some(stream) => stream.into_chunk_stream(),
        None => Box::pin(futures::stream::once(async move { Ok(body) })),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissReason {
    NoData,
    Status5xx,
    Status404,
    Timeout,
    ConnectionRefused,
    ConnectionReset,
    Other,
}

#[derive(Debug, Clone)]
pub struct MissPolicy {
    pub on_no_data: bool,
    pub on_status: Vec<u16>,
    pub on_error: bool,
}

impl Default for MissPolicy {
    fn default() -> Self {
        Self {
            on_no_data: true,
            on_status: Vec::new(),
            on_error: false,
        }
    }
}

impl MissPolicy {
    #[must_use]
    pub fn fallthrough_default() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn classifies_as_miss(&self, response: &Response<Bytes>) -> Option<MissReason> {
        if self.on_no_data && response.is_no_data() {
            return Some(MissReason::NoData);
        }
        if self.on_status.contains(&response.status) {
            let reason = match response.status {
                404 => MissReason::Status404,
                500..=599 => MissReason::Status5xx,
                _ => MissReason::Other,
            };
            return Some(reason);
        }
        None
    }

    #[must_use]
    pub fn classifies_error_as_miss(&self, error: &ProximaError) -> Option<MissReason> {
        if !self.on_error && !matches!(error, ProximaError::NoData) {
            return None;
        }
        Some(match error {
            ProximaError::NoData => MissReason::NoData,
            ProximaError::Timeout(_) => MissReason::Timeout,
            ProximaError::Upstream(message) => {
                if message.contains("connection refused") {
                    MissReason::ConnectionRefused
                } else if message.contains("connection reset") {
                    MissReason::ConnectionReset
                } else {
                    MissReason::Other
                }
            }
            _ => MissReason::Other,
        })
    }
}

pub trait Selection: Send + Sync + 'static {
    fn dispatch(
        &self,
        upstreams: &[UpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> + Send;

    fn name(&self) -> &str;
}

#[derive(Debug)]
pub struct DispatchOutcome {
    pub response: Response<Bytes>,
    pub upstream_index: usize,
    pub fallthroughs: Vec<MissReason>,
}

pub struct Fallthrough {
    pub policy: MissPolicy,
    pub replay_cap_bytes: usize,
}

impl Fallthrough {
    #[must_use]
    pub fn new(policy: MissPolicy) -> Self {
        Self {
            policy,
            replay_cap_bytes: DEFAULT_REPLAY_CAP_BYTES,
        }
    }

    #[must_use]
    pub fn miss_on_no_data() -> Self {
        Self::new(MissPolicy::fallthrough_default())
    }

    #[must_use]
    pub fn with_replay_cap_bytes(mut self, cap: usize) -> Self {
        self.replay_cap_bytes = cap;
        self
    }
}

impl Selection for Fallthrough {
    fn dispatch(
        &self,
        upstreams: &[UpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> + Send {
        let policy = self.policy.clone();
        let cap = self.replay_cap_bytes;
        let upstreams = upstreams.to_vec();
        async move { run_fallthrough(&policy, &upstreams, request, cap).await }
    }

    fn name(&self) -> &str {
        "fallthrough"
    }
}

async fn run_fallthrough(
    policy: &MissPolicy,
    upstreams: &[UpstreamRef],
    request: Request<Bytes>,
    replay_cap_bytes: usize,
) -> Result<DispatchOutcome, ProximaError> {
    if upstreams.is_empty() {
        return Err(ProximaError::Config("no upstreams configured".into()));
    }
    let single = upstreams.len() == 1;
    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    } = request;
    let (tee, primary_body) = if single {
        (None, source_stream(payload, stream))
    } else {
        let (tee, primary) = Replay::wrap(source_stream(payload, stream), replay_cap_bytes);
        (Some(tee), primary)
    };
    let mut next_body: Option<ChunkStream> = Some(primary_body);
    let mut fallthroughs: Vec<MissReason> = Vec::new();
    let mut last_error: Option<ProximaError> = None;
    for (index, upstream) in upstreams.iter().enumerate() {
        let body_for_call = match next_body.take() {
            Some(body) => body,
            None => match tee.as_ref() {
                Some(tee) => tee.replay()?,
                None => Box::pin(futures::stream::empty()),
            },
        };
        let forward = Request {
            method: method.clone(),
            path: path.clone(),
            query: query.clone(),
            metadata: metadata.clone(),
            payload: Bytes::new(),
            stream: Some(RequestStream::from_chunk_stream(body_for_call)),
            context: context.clone(),
        };
        let tracker = upstream.track_call();
        let outcome = SendPipe::call(&upstream.pipe, forward).await;
        match outcome {
            Ok(response) => {
                if let Some(reason) = policy.classifies_as_miss(&response) {
                    tracker.settle_failure();
                    fallthroughs.push(reason);
                    continue;
                }
                tracker.settle_success();
                return Ok(DispatchOutcome {
                    response,
                    upstream_index: index,
                    fallthroughs,
                });
            }
            Err(error) => {
                if let Some(reason) = policy.classifies_error_as_miss(&error) {
                    tracker.settle_failure();
                    fallthroughs.push(reason);
                    last_error = Some(error);
                    continue;
                }
                tracker.settle_failure();
                return Err(error);
            }
        }
    }
    Err(last_error.unwrap_or(ProximaError::NoData))
}

pub struct RoundRobin {
    cursor: AtomicUsize,
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self {
            cursor: AtomicUsize::new(0),
        }
    }
}

impl Selection for RoundRobin {
    fn dispatch(
        &self,
        upstreams: &[UpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> + Send {
        let upstreams = upstreams.to_vec();
        let cursor = self.cursor.fetch_add(1, Ordering::Relaxed);
        async move {
            if upstreams.is_empty() {
                return Err(ProximaError::Config("no upstreams configured".into()));
            }
            let index = cursor % upstreams.len();
            let upstream = &upstreams[index];
            let tracker = upstream.track_call();
            match SendPipe::call(&upstream.pipe, request).await {
                Ok(response) => {
                    tracker.settle_success();
                    Ok(DispatchOutcome {
                        response,
                        upstream_index: index,
                        fallthroughs: Vec::new(),
                    })
                }
                Err(error) => {
                    tracker.settle_failure();
                    Err(error)
                }
            }
        }
    }

    fn name(&self) -> &str {
        "round_robin"
    }
}

pub struct WeightedRoundRobin {
    cursor: AtomicUsize,
}

impl Default for WeightedRoundRobin {
    fn default() -> Self {
        Self {
            cursor: AtomicUsize::new(0),
        }
    }
}

impl Selection for WeightedRoundRobin {
    fn dispatch(
        &self,
        upstreams: &[UpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> + Send {
        let upstreams = upstreams.to_vec();
        let cursor = self.cursor.fetch_add(1, Ordering::Relaxed);
        async move {
            if upstreams.is_empty() {
                return Err(ProximaError::Config("no upstreams configured".into()));
            }
            let total_weight: u64 = upstreams
                .iter()
                .filter(|upstream| upstream.metrics.is_healthy())
                .map(|upstream| u64::from(upstream.weight))
                .sum();
            if total_weight == 0 {
                return Err(ProximaError::NoData);
            }
            let target = (cursor as u64) % total_weight;
            let mut accumulated: u64 = 0;
            for (index, upstream) in upstreams.iter().enumerate() {
                if !upstream.metrics.is_healthy() {
                    continue;
                }
                accumulated += u64::from(upstream.weight);
                if target < accumulated {
                    let tracker = upstream.track_call();
                    return match SendPipe::call(&upstream.pipe, request).await {
                        Ok(response) => {
                            tracker.settle_success();
                            Ok(DispatchOutcome {
                                response,
                                upstream_index: index,
                                fallthroughs: Vec::new(),
                            })
                        }
                        Err(error) => {
                            tracker.settle_failure();
                            Err(error)
                        }
                    };
                }
            }
            Err(ProximaError::NoData)
        }
    }

    fn name(&self) -> &str {
        "weighted_round_robin"
    }
}

pub struct WeightedLeastConn;

impl Selection for WeightedLeastConn {
    fn dispatch(
        &self,
        upstreams: &[UpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> + Send {
        let upstreams = upstreams.to_vec();
        async move {
            let pick = upstreams
                .iter()
                .enumerate()
                .filter(|(_, upstream)| upstream.metrics.is_healthy())
                .min_by(|left, right| {
                    let left_load =
                        (left.1.metrics.in_flight() as f64) / f64::from(left.1.weight.max(1));
                    let right_load =
                        (right.1.metrics.in_flight() as f64) / f64::from(right.1.weight.max(1));
                    left_load
                        .partial_cmp(&right_load)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .or_else(|| upstreams.iter().enumerate().next());
            let Some((index, upstream)) = pick else {
                return Err(ProximaError::Config("no upstreams configured".into()));
            };
            let tracker = upstream.track_call();
            match SendPipe::call(&upstream.pipe, request).await {
                Ok(response) => {
                    tracker.settle_success();
                    Ok(DispatchOutcome {
                        response,
                        upstream_index: index,
                        fallthroughs: Vec::new(),
                    })
                }
                Err(error) => {
                    tracker.settle_failure();
                    Err(error)
                }
            }
        }
    }

    fn name(&self) -> &str {
        "weighted_least_connections"
    }
}

pub struct LeastConn;

impl Selection for LeastConn {
    fn dispatch(
        &self,
        upstreams: &[UpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> + Send {
        let upstreams = upstreams.to_vec();
        async move {
            let Some((index, upstream)) = pick_least_loaded(&upstreams) else {
                return Err(ProximaError::Config("no upstreams configured".into()));
            };
            let tracker = upstream.track_call();
            match SendPipe::call(&upstream.pipe, request).await {
                Ok(response) => {
                    tracker.settle_success();
                    Ok(DispatchOutcome {
                        response,
                        upstream_index: index,
                        fallthroughs: Vec::new(),
                    })
                }
                Err(error) => {
                    tracker.settle_failure();
                    Err(error)
                }
            }
        }
    }

    fn name(&self) -> &str {
        "least_conn"
    }
}

fn pick_least_loaded(upstreams: &[UpstreamRef]) -> Option<(usize, &UpstreamRef)> {
    upstreams
        .iter()
        .enumerate()
        .filter(|(_, upstream)| upstream.metrics.is_healthy())
        .min_by_key(|(_, upstream)| upstream.metrics.in_flight())
        .or_else(|| upstreams.iter().enumerate().next())
}

pub type SelectionHandle = Arc<dyn DynSelection>;

pub trait DynSelection: Send + Sync + 'static {
    fn dispatch_dyn<'this, 'upstreams>(
        &'this self,
        upstreams: &'upstreams [UpstreamRef],
        request: Request<Bytes>,
    ) -> std::pin::Pin<
        Box<dyn Future<Output = Result<DispatchOutcome, ProximaError>> + Send + 'upstreams>,
    >
    where
        'this: 'upstreams;

    fn name_dyn(&self) -> &str;
}

impl<S: Selection> DynSelection for S {
    fn dispatch_dyn<'this, 'upstreams>(
        &'this self,
        upstreams: &'upstreams [UpstreamRef],
        request: Request<Bytes>,
    ) -> std::pin::Pin<
        Box<dyn Future<Output = Result<DispatchOutcome, ProximaError>> + Send + 'upstreams>,
    >
    where
        'this: 'upstreams,
    {
        Box::pin(self.dispatch(upstreams, request))
    }

    fn name_dyn(&self) -> &str {
        self.name()
    }
}

/// Per-thread sibling of [`Selection`]. Operates on
/// [`ThreadLocalUpstreamRef`] handles and returns a `?Send` future.
pub trait ThreadLocalSelection: 'static {
    fn dispatch(
        &self,
        upstreams: &[ThreadLocalUpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>>;

    fn name(&self) -> &str;
}

pub type ThreadLocalSelectionHandle = std::rc::Rc<dyn ThreadLocalDynSelection>;

pub trait ThreadLocalDynSelection: 'static {
    fn dispatch_dyn<'this, 'upstreams>(
        &'this self,
        upstreams: &'upstreams [ThreadLocalUpstreamRef],
        request: Request<Bytes>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<DispatchOutcome, ProximaError>> + 'upstreams>>
    where
        'this: 'upstreams;

    fn name_dyn(&self) -> &str;
}

impl<S: ThreadLocalSelection> ThreadLocalDynSelection for S {
    fn dispatch_dyn<'this, 'upstreams>(
        &'this self,
        upstreams: &'upstreams [ThreadLocalUpstreamRef],
        request: Request<Bytes>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<DispatchOutcome, ProximaError>> + 'upstreams>>
    where
        'this: 'upstreams,
    {
        Box::pin(ThreadLocalSelection::dispatch(self, upstreams, request))
    }

    fn name_dyn(&self) -> &str {
        ThreadLocalSelection::name(self)
    }
}

impl ThreadLocalSelection for Fallthrough {
    fn dispatch(
        &self,
        upstreams: &[ThreadLocalUpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> {
        let policy = self.policy.clone();
        let cap = self.replay_cap_bytes;
        let upstreams = upstreams.to_vec();
        async move { run_fallthrough_local(&policy, &upstreams, request, cap).await }
    }

    fn name(&self) -> &str {
        "fallthrough"
    }
}

impl ThreadLocalSelection for RoundRobin {
    fn dispatch(
        &self,
        upstreams: &[ThreadLocalUpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> {
        let upstreams = upstreams.to_vec();
        let cursor = self.cursor.fetch_add(1, Ordering::Relaxed);
        async move {
            if upstreams.is_empty() {
                return Err(ProximaError::Config("no upstreams configured".into()));
            }
            let index = cursor % upstreams.len();
            let upstream = &upstreams[index];
            let tracker = upstream.track_call();
            match Pipe::call(&upstream.pipe, request).await {
                Ok(response) => {
                    tracker.settle_success();
                    Ok(DispatchOutcome {
                        response,
                        upstream_index: index,
                        fallthroughs: Vec::new(),
                    })
                }
                Err(error) => {
                    tracker.settle_failure();
                    Err(error)
                }
            }
        }
    }

    fn name(&self) -> &str {
        "round_robin"
    }
}

impl ThreadLocalSelection for WeightedRoundRobin {
    fn dispatch(
        &self,
        upstreams: &[ThreadLocalUpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> {
        let upstreams = upstreams.to_vec();
        let cursor = self.cursor.fetch_add(1, Ordering::Relaxed);
        async move {
            if upstreams.is_empty() {
                return Err(ProximaError::Config("no upstreams configured".into()));
            }
            let total_weight: u64 = upstreams
                .iter()
                .filter(|upstream| upstream.metrics.is_healthy())
                .map(|upstream| u64::from(upstream.weight))
                .sum();
            if total_weight == 0 {
                return Err(ProximaError::NoData);
            }
            let target = (cursor as u64) % total_weight;
            let mut accumulated: u64 = 0;
            for (index, upstream) in upstreams.iter().enumerate() {
                if !upstream.metrics.is_healthy() {
                    continue;
                }
                accumulated += u64::from(upstream.weight);
                if target < accumulated {
                    let tracker = upstream.track_call();
                    return match Pipe::call(&upstream.pipe, request).await {
                        Ok(response) => {
                            tracker.settle_success();
                            Ok(DispatchOutcome {
                                response,
                                upstream_index: index,
                                fallthroughs: Vec::new(),
                            })
                        }
                        Err(error) => {
                            tracker.settle_failure();
                            Err(error)
                        }
                    };
                }
            }
            Err(ProximaError::NoData)
        }
    }

    fn name(&self) -> &str {
        "weighted_round_robin"
    }
}

impl ThreadLocalSelection for WeightedLeastConn {
    fn dispatch(
        &self,
        upstreams: &[ThreadLocalUpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> {
        let upstreams = upstreams.to_vec();
        async move {
            let pick = upstreams
                .iter()
                .enumerate()
                .filter(|(_, upstream)| upstream.metrics.is_healthy())
                .min_by(|left, right| {
                    let left_load =
                        (left.1.metrics.in_flight() as f64) / f64::from(left.1.weight.max(1));
                    let right_load =
                        (right.1.metrics.in_flight() as f64) / f64::from(right.1.weight.max(1));
                    left_load
                        .partial_cmp(&right_load)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .or_else(|| upstreams.iter().enumerate().next());
            let Some((index, upstream)) = pick else {
                return Err(ProximaError::Config("no upstreams configured".into()));
            };
            let tracker = upstream.track_call();
            match Pipe::call(&upstream.pipe, request).await {
                Ok(response) => {
                    tracker.settle_success();
                    Ok(DispatchOutcome {
                        response,
                        upstream_index: index,
                        fallthroughs: Vec::new(),
                    })
                }
                Err(error) => {
                    tracker.settle_failure();
                    Err(error)
                }
            }
        }
    }

    fn name(&self) -> &str {
        "weighted_least_connections"
    }
}

impl ThreadLocalSelection for LeastConn {
    fn dispatch(
        &self,
        upstreams: &[ThreadLocalUpstreamRef],
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<DispatchOutcome, ProximaError>> {
        let upstreams = upstreams.to_vec();
        async move {
            let Some((index, upstream)) = pick_least_loaded_local(&upstreams) else {
                return Err(ProximaError::Config("no upstreams configured".into()));
            };
            let tracker = upstream.track_call();
            match Pipe::call(&upstream.pipe, request).await {
                Ok(response) => {
                    tracker.settle_success();
                    Ok(DispatchOutcome {
                        response,
                        upstream_index: index,
                        fallthroughs: Vec::new(),
                    })
                }
                Err(error) => {
                    tracker.settle_failure();
                    Err(error)
                }
            }
        }
    }

    fn name(&self) -> &str {
        "least_conn"
    }
}

fn pick_least_loaded_local(
    upstreams: &[ThreadLocalUpstreamRef],
) -> Option<(usize, &ThreadLocalUpstreamRef)> {
    upstreams
        .iter()
        .enumerate()
        .filter(|(_, upstream)| upstream.metrics.is_healthy())
        .min_by_key(|(_, upstream)| upstream.metrics.in_flight())
        .or_else(|| upstreams.iter().enumerate().next())
}

async fn run_fallthrough_local(
    policy: &MissPolicy,
    upstreams: &[ThreadLocalUpstreamRef],
    request: Request<Bytes>,
    replay_cap_bytes: usize,
) -> Result<DispatchOutcome, ProximaError> {
    if upstreams.is_empty() {
        return Err(ProximaError::Config("no upstreams configured".into()));
    }
    let single = upstreams.len() == 1;
    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    } = request;
    let (tee, primary_body) = if single {
        (None, source_stream(payload, stream))
    } else {
        let (tee, primary) = Replay::wrap(source_stream(payload, stream), replay_cap_bytes);
        (Some(tee), primary)
    };
    let mut next_body: Option<ChunkStream> = Some(primary_body);
    let mut fallthroughs: Vec<MissReason> = Vec::new();
    let mut last_error: Option<ProximaError> = None;
    for (index, upstream) in upstreams.iter().enumerate() {
        let body_for_call = match next_body.take() {
            Some(body) => body,
            None => match tee.as_ref() {
                Some(tee) => tee.replay()?,
                None => Box::pin(futures::stream::empty()),
            },
        };
        let forward = Request {
            method: method.clone(),
            path: path.clone(),
            query: query.clone(),
            metadata: metadata.clone(),
            payload: Bytes::new(),
            stream: Some(RequestStream::from_chunk_stream(body_for_call)),
            context: context.clone(),
        };
        let tracker = upstream.track_call();
        let outcome = Pipe::call(&upstream.pipe, forward).await;
        match outcome {
            Ok(response) => {
                if let Some(reason) = policy.classifies_as_miss(&response) {
                    tracker.settle_failure();
                    fallthroughs.push(reason);
                    continue;
                }
                tracker.settle_success();
                return Ok(DispatchOutcome {
                    response,
                    upstream_index: index,
                    fallthroughs,
                });
            }
            Err(error) => {
                if let Some(reason) = policy.classifies_error_as_miss(&error) {
                    tracker.settle_failure();
                    fallthroughs.push(reason);
                    last_error = Some(error);
                    continue;
                }
                tracker.settle_failure();
                return Err(error);
            }
        }
    }
    Err(last_error.unwrap_or(ProximaError::NoData))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::handler::into_handle;
    use rstest::rstest;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct StaticPipe {
        label: String,
        response: Mutex<Option<Result<Response<Bytes>, ProximaError>>>,
        calls: AtomicUsize,
    }

    impl StaticPipe {
        fn ok_with_body(label: &str, body: &'static [u8]) -> Arc<Self> {
            Arc::new(Self {
                label: label.into(),
                response: Mutex::new(Some(Ok(Response::ok(Bytes::from_static(body))))),
                calls: AtomicUsize::new(0),
            })
        }

        fn no_data(label: &str) -> Arc<Self> {
            Arc::new(Self {
                label: label.into(),
                response: Mutex::new(Some(Ok(Response::no_data()))),
                calls: AtomicUsize::new(0),
            })
        }

        fn err(label: &str, error: ProximaError) -> Arc<Self> {
            Arc::new(Self {
                label: label.into(),
                response: Mutex::new(Some(Err(error))),
                calls: AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl SendPipe for StaticPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut guard = self.response.lock().expect("lock");
            let next = guard.take();
            *guard = match &next {
                Some(Ok(response)) => Some(Ok(Response {
                    status: response.status,
                    metadata: response.metadata.clone(),
                    payload: Bytes::new(),
                    stream: None,
                    upgrade: None,
                })),
                Some(Err(error)) => Some(Err(ProximaError::Upstream(error.to_string()))),
                None => None,
            };
            let label = self.label.clone();
            async move {
                next.unwrap_or_else(|| {
                    Err(ProximaError::Upstream(format!(
                        "test stub '{label}' exhausted"
                    )))
                })
            }
        }
    }


    fn build_request() -> Request<Bytes> {
        Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder")
    }

    fn upstream(label: &str, pipe: Arc<StaticPipe>) -> UpstreamRef {
        struct ArcWrapper(Arc<StaticPipe>);
        impl SendPipe for ArcWrapper {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                SendPipe::call(&*self.0, request)
            }
        }
        UpstreamRef::new(into_handle(ArcWrapper(pipe)), label.to_string(), 1)
    }

    #[proxima::test]
    async fn fallthrough_returns_first_data_upstream() {
        let cache = StaticPipe::ok_with_body("cache", b"hit");
        let origin = StaticPipe::ok_with_body("origin", b"origin");
        let upstreams = vec![
            upstream("cache", cache.clone()),
            upstream("origin", origin.clone()),
        ];
        let select = Fallthrough::miss_on_no_data();
        let outcome = Selection::dispatch(&select, &upstreams, build_request())
            .await
            .expect("dispatch");
        assert_eq!(outcome.upstream_index, 0);
        assert!(outcome.fallthroughs.is_empty());
        assert_eq!(cache.calls(), 1);
        assert_eq!(origin.calls(), 0);
    }

    #[proxima::test]
    async fn fallthrough_skips_no_data_to_next() {
        let cache = StaticPipe::no_data("cache");
        let origin = StaticPipe::ok_with_body("origin", b"origin");
        let upstreams = vec![
            upstream("cache", cache.clone()),
            upstream("origin", origin.clone()),
        ];
        let select = Fallthrough::miss_on_no_data();
        let outcome = Selection::dispatch(&select, &upstreams, build_request())
            .await
            .expect("dispatch");
        assert_eq!(outcome.upstream_index, 1);
        assert_eq!(outcome.fallthroughs, vec![MissReason::NoData]);
    }

    #[proxima::test]
    async fn fallthrough_propagates_non_miss_error() {
        let cache = StaticPipe::err("cache", ProximaError::Decode("bad".into()));
        let origin = StaticPipe::ok_with_body("origin", b"origin");
        let upstreams = vec![
            upstream("cache", cache.clone()),
            upstream("origin", origin.clone()),
        ];
        let select = Fallthrough::new(MissPolicy {
            on_no_data: true,
            on_status: vec![],
            on_error: false,
        });
        let outcome = Selection::dispatch(&select, &upstreams, build_request()).await;
        assert!(matches!(outcome, Err(ProximaError::Decode(_))));
        assert_eq!(origin.calls(), 0);
    }

    #[proxima::test]
    async fn fallthrough_no_data_error_treated_as_miss() {
        let cache = StaticPipe::err("cache", ProximaError::NoData);
        let origin = StaticPipe::ok_with_body("origin", b"origin");
        let upstreams = vec![
            upstream("cache", cache.clone()),
            upstream("origin", origin.clone()),
        ];
        let select = Fallthrough::miss_on_no_data();
        let outcome = Selection::dispatch(&select, &upstreams, build_request())
            .await
            .expect("dispatch should succeed via fallthrough");
        assert_eq!(outcome.upstream_index, 1);
    }

    #[rstest]
    #[case::single(1, &[0])]
    #[case::two(2, &[0, 1, 0])]
    #[case::three(3, &[0, 1, 2, 0])]
    #[proxima::test]
    async fn round_robin_cycles_through_upstreams(
        #[case] count: usize,
        #[case] expected_indices: &[usize],
    ) {
        let pipes: Vec<Arc<StaticPipe>> = (0..count)
            .map(|index| StaticPipe::ok_with_body(&format!("svc-{index}"), b"x"))
            .collect();
        let upstreams: Vec<UpstreamRef> = pipes
            .iter()
            .enumerate()
            .map(|(idx, pipe)| upstream(&format!("svc-{idx}"), pipe.clone()))
            .collect();
        let select = RoundRobin::default();
        for &expected in expected_indices {
            let outcome = Selection::dispatch(&select, &upstreams, build_request())
                .await
                .expect("dispatch");
            assert_eq!(outcome.upstream_index, expected);
        }
    }

    #[proxima::test]
    async fn least_conn_picks_lowest_in_flight() {
        let lighter = StaticPipe::ok_with_body("lighter", b"a");
        let heavier = StaticPipe::ok_with_body("heavier", b"b");
        let upstreams = vec![
            upstream("lighter", lighter.clone()),
            upstream("heavier", heavier.clone()),
        ];
        upstreams[1]
            .metrics
            .in_flight
            .fetch_add(5, Ordering::Relaxed);
        let select = LeastConn;
        let outcome = Selection::dispatch(&select, &upstreams, build_request())
            .await
            .expect("dispatch");
        assert_eq!(outcome.upstream_index, 0);
        assert_eq!(lighter.calls(), 1);
        assert_eq!(heavier.calls(), 0);
    }

    #[proxima::test]
    async fn weighted_round_robin_distributes_proportionally() {
        let one = StaticPipe::ok_with_body("one", b"x");
        let two = StaticPipe::ok_with_body("two", b"x");
        let upstreams = vec![
            UpstreamRef::new(into_handle(ArcWrapper(one.clone())), "one", 1),
            UpstreamRef::new(into_handle(ArcWrapper(two.clone())), "two", 3),
        ];
        let select = WeightedRoundRobin::default();
        for _ in 0..16 {
            let _ = Selection::dispatch(&select, &upstreams, build_request()).await;
        }
        let one_calls = one.calls();
        let two_calls = two.calls();
        assert!(
            two_calls > one_calls,
            "weight 3 must serve more than weight 1"
        );
        assert!(
            two_calls >= one_calls * 2,
            "weight 3 should serve roughly 3x weight 1; one={one_calls} two={two_calls}",
        );
    }

    #[proxima::test]
    async fn weighted_least_conn_picks_lower_load_per_unit_weight() {
        let small = StaticPipe::ok_with_body("small", b"x");
        let big = StaticPipe::ok_with_body("big", b"x");
        let upstreams = vec![
            UpstreamRef::new(into_handle(ArcWrapper(small.clone())), "small", 1),
            UpstreamRef::new(into_handle(ArcWrapper(big.clone())), "big", 4),
        ];
        upstreams[0]
            .metrics
            .in_flight
            .fetch_add(2, Ordering::Relaxed);
        upstreams[1]
            .metrics
            .in_flight
            .fetch_add(3, Ordering::Relaxed);
        let select = WeightedLeastConn;
        let outcome = Selection::dispatch(&select, &upstreams, build_request())
            .await
            .expect("dispatch");
        assert_eq!(
            outcome.upstream_index, 1,
            "big (3/4=0.75 load) beats small (2/1=2.0 load) once weight is factored in",
        );
    }

    struct ArcWrapper<S>(Arc<S>);

    impl<S: Handler + Send + Sync + 'static> SendPipe for ArcWrapper<S> {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            SendPipe::call(&*self.0, request)
        }
    }


    #[proxima::test]
    async fn empty_upstreams_returns_config_error() {
        let select = Fallthrough::miss_on_no_data();
        let outcome = Selection::dispatch(&select, &[], build_request()).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    use proxima_primitives::pipe::handler::into_thread_local_handle;
    use std::cell::Cell;

    struct CountingLocalPipe {
        label: String,
        calls: std::rc::Rc<Cell<usize>>,
    }

    impl Pipe for CountingLocalPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            let calls = self.calls.clone();
            let label = self.label.clone();
            async move {
                calls.set(calls.get() + 1);
                Ok(Response::ok(label))
            }
        }
    }


    fn local_upstream(
        label: &str,
        weight: u32,
    ) -> (ThreadLocalUpstreamRef, std::rc::Rc<Cell<usize>>) {
        let calls = std::rc::Rc::new(Cell::new(0));
        let pipe = CountingLocalPipe {
            label: label.into(),
            calls: calls.clone(),
        };
        let handle = into_thread_local_handle(pipe);
        (ThreadLocalUpstreamRef::new(handle, label, weight), calls)
    }

    #[proxima::test(runtime = "tokio")]
    async fn thread_local_round_robin_cycles_local_upstreams() {
        let (upstream_one, calls_one) = local_upstream("svc-0", 1);
        let (upstream_two, calls_two) = local_upstream("svc-1", 1);
        let upstreams = vec![upstream_one, upstream_two];
        let select = RoundRobin::default();
        for _ in 0..4 {
            let outcome = ThreadLocalSelection::dispatch(&select, &upstreams, build_request())
                .await
                .expect("dispatch");
            assert!(outcome.upstream_index < upstreams.len());
        }
        assert_eq!(calls_one.get() + calls_two.get(), 4);
    }

    #[proxima::test(runtime = "tokio")]
    async fn thread_local_fallthrough_skips_no_data_to_next() {
        struct AlwaysNoData;
        impl Pipe for AlwaysNoData {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                _request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
                async { Ok(Response::no_data()) }
            }
        }
        let no_data_handle = into_thread_local_handle(AlwaysNoData);
        let cache = ThreadLocalUpstreamRef::new(no_data_handle, "cache", 1);
        let (origin, origin_calls) = local_upstream("origin", 1);
        let upstreams = vec![cache, origin];
        let select = Fallthrough::miss_on_no_data();
        let outcome = ThreadLocalSelection::dispatch(&select, &upstreams, build_request())
            .await
            .expect("dispatch via fallthrough");
        assert_eq!(outcome.upstream_index, 1);
        assert_eq!(outcome.fallthroughs, vec![MissReason::NoData]);
        assert_eq!(origin_calls.get(), 1);
    }
}
