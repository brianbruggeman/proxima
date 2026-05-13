//! `DatagramListenProtocol` — a connectionless request/reply `ListenProtocol`
//! over UDP-style datagrams. Each inbound datagram becomes a `Request` whose
//! body is the datagram bytes; the `Pipe`'s `Response` body is sent back to the
//! datagram's peer as one reply datagram, and the loop continues. There is no
//! connection state — the peer address rides with every packet (the
//! [`Addressed`](proxima_codec::Addressed) shape), so a reply always knows
//! where to go. An empty response body is fire-and-forget: nothing is sent.
//!
//! Bytes-centric, mirroring [`FramedListenProtocol`](super::FramedListenProtocol):
//! this listener moves only bytes + peer. Typed per-datagram protocols (DNS,
//! RADIUS) decode/encode with [`proxima_codec::Datagram`] inside the consumer
//! `Pipe` at the edge — the same "typing is the Pipe's concern" split framed
//! uses. A future typed driver generic over `C: Datagram` can reuse this
//! recv/dispatch/send loop wholesale: only the per-datagram middle (bytes →
//! `Request<Bytes>` → `Response<Bytes>` → bytes) changes to (bytes →
//! `codec.decode` → typed pipe → `codec.encode` → bytes); the batch I/O,
//! peer plumbing, and shutdown race stay identical. That is the phase-2 seam.
//!
//! Batched I/O: recv drains a burst through
//! [`DatagramSocketBatchExt::poll_fill_recv_batch`] +
//! [`drain_recv_to_empty`](DatagramSocketBatchExt::drain_recv_to_empty) (one
//! `recvmmsg`-shaped drain per wake, so the kernel buffer never overflows on a
//! burst), and replies ship through
//! [`poll_drive_send_batch`](DatagramSocketBatchExt::poll_drive_send_batch)
//! (one `sendmmsg`-shaped flush per tick). Runtime-agnostic like the native h3
//! listener: one serve loop, no per-connection spawn — a datagram has no
//! connection to pin work to.

use std::future::{Future, poll_fn};
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::Poll;

use bytes::Bytes;
use futures::channel::oneshot;
use futures::future::{Either, select};
use serde_json::Value;
use tracing::{debug, warn};

use crate::{ListenProtocol, ServeContext};
use proxima_core::ProximaError;
use proxima_core::datagram_batch::DefaultDatagramBatch;
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::request::{Request, RequestContext};
use proxima_primitives::stream::DatagramSocketBatchExt;

const DEFAULT_METHOD: &str = "DGRAM";
const DEFAULT_PATH: &str = "/";

/// Connectionless request/reply listener. Construct with [`Self::new`], tweak
/// the synthetic request envelope with [`Self::with_method`] / [`Self::with_path`],
/// and register on an `App` via `with_listen_protocol`. The `method` and `path`
/// stamped on every synthetic `Request` are read from the listener `spec` so the
/// control plane can tune a deployment without a recompile.
pub struct DatagramListenProtocol {
    label: String,
    method: String,
    path: String,
}

impl DatagramListenProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            method: DEFAULT_METHOD.into(),
            path: DEFAULT_PATH.into(),
        }
    }

    #[must_use]
    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = method.into();
        self
    }

    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }
}

impl ListenProtocol for DatagramListenProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        mut shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let label = self.label.clone();
        let method = Bytes::from(
            spec.get("method")
                .and_then(Value::as_str)
                .unwrap_or(&self.method)
                .to_string(),
        );
        let path = Bytes::from(
            spec.get("path")
                .and_then(Value::as_str)
                .unwrap_or(&self.path)
                .to_string(),
        );
        let datagram_factory = context.datagram_factory.clone();
        let ready_signal = context.ready_signal.clone();

        Box::pin(async move {
            let datagram_factory = datagram_factory.ok_or_else(|| {
                ProximaError::Config("datagram listener requires a datagram factory".into())
            })?;
            let mut socket = datagram_factory
                .bind(bind)
                .map_err(|err| ProximaError::Io(io::Error::other(format!("{label} bind {bind}: {err}"))))?;
            if let Some(sender) = ready_signal {
                let _ = sender.send(());
            }
            debug!(label = %label, %bind, "datagram listener bound");

            let mut batch = DefaultDatagramBatch::new();
            // Datagrams copied out of the recv slab (owning their bytes) before
            // dispatch, so the slab borrow is not held across the `Pipe::call`
            // await. Hoisted once, cleared per iteration — off the per-tick
            // alloc path. One copy per datagram, like framed's per-frame copy;
            // a zero-copy path is a later optimisation, not a phase-1 concern.
            let mut inbound: Vec<(Bytes, SocketAddr)> = Vec::new();

            loop {
                // Race the burst recv against shutdown. `poll_fill_recv_batch`
                // only resolves once at least one datagram lands (or errors), so
                // an idle socket parks here on the recv waker.
                let filled = {
                    let recv = poll_fn(|cx| socket.poll_fill_recv_batch(cx, &mut batch.recv));
                    match select(recv, &mut shutdown).await {
                        Either::Left((result, _)) => result.map_err(|err| {
                            ProximaError::Io(io::Error::other(format!("{label} recv: {err}")))
                        })?,
                        Either::Right(_) => {
                            debug!(label = %label, "datagram listener shutting down");
                            return Ok(());
                        }
                    }
                };
                if filled == 0 {
                    continue;
                }
                // Drain whatever else the kernel already has, so one wake serves
                // a whole burst rather than one datagram per park.
                socket.drain_recv_to_empty(&mut batch.recv);

                inbound.clear();
                for view in batch.recv.filled_datagrams() {
                    inbound.push((Bytes::copy_from_slice(view.bytes), view.peer));
                }
                batch.recv.clear();

                for (payload, peer) in inbound.drain(..) {
                    let request = Request {
                        method: Method::from_wire(method.clone()),
                        path: path.clone(),
                        query: HeaderList::new(),
                        metadata: HeaderList::new(),
                        payload,
                        stream: None,
                        context: RequestContext::default(),
                    };
                    // A datagram is connectionless: one malformed request must
                    // NOT tear down the listener (as framed's per-connection
                    // bail does) — it drops that reply and moves on.
                    let reply = match SendPipe::call(&dispatch, request).await {
                        Ok(response) => match response.collect_body().await {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                warn!(?error, label = %label, %peer, "datagram response body failed; dropping reply");
                                continue;
                            }
                        },
                        Err(error) => {
                            warn!(?error, label = %label, %peer, "datagram dispatch failed; dropping reply");
                            continue;
                        }
                    };
                    if reply.is_empty() {
                        // fire-and-forget: an empty body sends no datagram.
                        continue;
                    }
                    stage_reply(&mut socket, &mut batch, &reply, peer, &label).await?;
                }

                flush_send(&mut socket, &mut batch, &label).await?;
            }
        })
    }
}

/// Stage one reply into the send batch, flushing first if the send arena is
/// full so a burst larger than one arena still ships. A single reply too large
/// for an empty arena is dropped with a warning — it can never fit.
async fn stage_reply(
    socket: &mut Box<dyn proxima_primitives::stream::DatagramSocket>,
    batch: &mut DefaultDatagramBatch,
    reply: &[u8],
    peer: SocketAddr,
    label: &str,
) -> Result<(), ProximaError> {
    if batch.send.try_append(reply, peer).is_ok() {
        return Ok(());
    }
    flush_send(socket, batch, label).await?;
    if let Err(error) = batch.send.try_append(reply, peer) {
        warn!(?error, label = %label, %peer, len = reply.len(), "datagram reply exceeds send arena; dropping");
    }
    Ok(())
}

/// Fully ship the staged send burst, parking on backpressure via the waker the
/// inner send already registered (mirrors the native h3 listener's flush).
async fn flush_send(
    socket: &mut Box<dyn proxima_primitives::stream::DatagramSocket>,
    batch: &mut DefaultDatagramBatch,
    label: &str,
) -> Result<(), ProximaError> {
    if batch.send.is_empty() {
        return Ok(());
    }
    let staged = batch.send.len();
    let mut span_offset = 0;
    let flush = poll_fn(|cx| match socket.poll_drive_send_batch(cx, &batch.send, &mut span_offset) {
        Poll::Ready(Ok(())) if span_offset >= staged => Poll::Ready(Ok(())),
        Poll::Ready(Ok(())) => Poll::Pending,
        Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
        Poll::Pending => Poll::Pending,
    })
    .await;
    batch.send.reset();
    flush.map_err(|err| ProximaError::Io(io::Error::other(format!("{label} send: {err}"))))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::VecDeque;
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Waker};

    use futures::task::noop_waker;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::request::Response;
    use proxima_primitives::stream::{DatagramFactory, DatagramSocket};
    use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;

    use super::*;

    // in-memory datagram socket: the test queues inbound datagrams and drains
    // whatever the serve loop sent. DatagramSocket needs only recv_from/send_to;
    // the batch ext methods default onto them.
    #[derive(Default)]
    struct SocketState {
        inbound: VecDeque<(Vec<u8>, SocketAddr)>,
        sent: Vec<(Vec<u8>, SocketAddr)>,
        waker: Option<Waker>,
    }

    #[derive(Clone)]
    struct SharedSocket {
        state: Arc<Mutex<SocketState>>,
        local: SocketAddr,
    }

    impl SharedSocket {
        fn new(local: SocketAddr) -> Self {
            Self { state: Arc::new(Mutex::new(SocketState::default())), local }
        }

        fn inject(&self, bytes: Vec<u8>, from: SocketAddr) {
            let mut state = self.state.lock().unwrap();
            state.inbound.push_back((bytes, from));
            if let Some(waker) = state.waker.take() {
                waker.wake();
            }
        }

        fn sent(&self) -> Vec<(Vec<u8>, SocketAddr)> {
            self.state.lock().unwrap().sent.clone()
        }
    }

    impl DatagramSocket for SharedSocket {
        fn poll_recv_from(&mut self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<(usize, SocketAddr)>> {
            let mut state = self.state.lock().unwrap();
            match state.inbound.pop_front() {
                Some((bytes, from)) => {
                    let len = bytes.len().min(buf.len());
                    buf[..len].copy_from_slice(&bytes[..len]);
                    Poll::Ready(Ok((len, from)))
                }
                None => {
                    state.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }

        fn poll_send_to(&mut self, _cx: &mut Context<'_>, buf: &[u8], peer: SocketAddr) -> Poll<io::Result<usize>> {
            self.state.lock().unwrap().sent.push((buf.to_vec(), peer));
            Poll::Ready(Ok(buf.len()))
        }

        fn local_addr(&self) -> io::Result<SocketAddr> {
            Ok(self.local)
        }
    }

    struct SharedFactory {
        socket: SharedSocket,
    }

    impl DatagramFactory for SharedFactory {
        fn bind(&self, _addr: SocketAddr) -> io::Result<Box<dyn DatagramSocket>> {
            Ok(Box::new(self.socket.clone()))
        }
    }

    // uppercases the request body — proves a datagram flows in, through the
    // pipe, and back out addressed to its sender.
    struct UppercasePipe;

    impl SendPipe for UppercasePipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(&self, request: Request<Bytes>) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move {
                let (_, bytes) = request.body_bytes().await?;
                let upper: Vec<u8> = bytes.iter().map(u8::to_ascii_uppercase).collect();
                Ok(Response {
                    status: 200,
                    metadata: HeaderList::new(),
                    payload: Bytes::from(upper),
                    stream: None,
                    upgrade: None,
                })
            }
        }
    }

    // empty-body pipe: proves an empty response is fire-and-forget (no datagram
    // sent back).
    struct DropPipe;

    impl SendPipe for DropPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(&self, _request: Request<Bytes>) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move {
                Ok(Response {
                    status: 200,
                    metadata: HeaderList::new(),
                    payload: Bytes::new(),
                    stream: None,
                    upgrade: None,
                })
            }
        }
    }

    fn drive_until<P: SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError> + Send + Sync + 'static>(
        pipe: P,
        inject: &[(&[u8], SocketAddr)],
        stop: impl Fn(&SharedSocket) -> bool,
    ) -> SharedSocket {
        let bind = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5300);
        let socket = SharedSocket::new(bind);
        let factory = Arc::new(SharedFactory { socket: socket.clone() });
        let protocol = DatagramListenProtocol::new("dgram-test");
        let dispatch = into_handle(pipe);
        let spec = serde_json::json!({});
        let context = ServeContext::new(Arc::new(NoopTelemetry)).with_datagram_factory(factory);
        let (_shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let mut serve = protocol.serve(bind, dispatch, &spec, context, shutdown_rx);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        // prime: bind + park on empty recv.
        let _ = serve.as_mut().poll(&mut cx);
        for (bytes, from) in inject {
            socket.inject(bytes.to_vec(), *from);
        }
        for _ in 0..200 {
            let _ = serve.as_mut().poll(&mut cx);
            if stop(&socket) {
                break;
            }
        }
        socket
    }

    #[proxima::test]
    async fn datagram_round_trips_reply_to_its_peer() {
        let peer = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 7)), 40000);
        let socket = drive_until(UppercasePipe, &[(b"hello datagram", peer)], |s| !s.sent().is_empty());
        let sent = socket.sent();
        assert_eq!(sent.len(), 1, "exactly one reply datagram");
        assert_eq!(sent[0].0, b"HELLO DATAGRAM", "reply body is the pipe output");
        assert_eq!(sent[0].1, peer, "reply is addressed back to the sender");
    }

    #[proxima::test]
    async fn two_datagrams_from_two_peers_each_get_their_own_reply() {
        let a = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)), 41000);
        let b = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)), 42000);
        let socket = drive_until(UppercasePipe, &[(b"one", a), (b"two", b)], |s| s.sent().len() >= 2);
        let sent = socket.sent();
        assert_eq!(sent.len(), 2);
        assert!(sent.contains(&(b"ONE".to_vec(), a)), "peer a gets ONE");
        assert!(sent.contains(&(b"TWO".to_vec(), b)), "peer b gets TWO");
    }

    #[proxima::test]
    async fn empty_response_is_fire_and_forget_no_reply_sent() {
        let peer = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 9)), 43000);
        // drive a fixed number of passes; assert nothing was ever sent.
        let socket = drive_until(DropPipe, &[(b"ingest me", peer)], |_| false);
        assert!(socket.sent().is_empty(), "empty body must send no datagram");
    }
}
