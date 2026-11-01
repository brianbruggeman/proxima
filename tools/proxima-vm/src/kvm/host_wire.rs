//! The host end of a resident guest's eth0 as the three pipes the
//! `proxima_net::frame_stack::Driver` reads, writes and mints through. A
//! [`HostLink`] already is a frame source, a frame sink and a doorbell;
//! [`dial_guest`] only reshapes them: the source's `Err(Empty)` becomes
//! `Ok(None)`, and every transmitted frame is followed by a doorbell ring so
//! the vCPU thread services RX.
//!
//! No combinator in `proxima_primitives::pipe` fits: `AndThen` needs
//! `Infallible: From<FrameError>` to chain the doorbell after the sink, and
//! nothing maps an error to an `Option` output or mints a blank frame.

use core::future::Future;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::sync::Arc;

use proxima_net::frame_stack::{Driver, FrameUpstream};
use proxima_primitives::pipe::Pipe;

use super::doorbell::Doorbell;
use super::resident::HostLink;
use crate::frame_pipe::{FrameError, FrameSink, FrameSource};

pub const HOST_IP: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 2);
pub const HOST_MAC: [u8; 6] = [0x02, 0x50, 0x58, 0x00, 0x02, 0x02];
const HOST_PORT: u16 = 49152;
const HOST_INITIAL_SEQUENCE: u32 = 0x4000;

/// `() -> Option<Vec<u8>>`: the guest-to-host frame source with "nothing
/// ready" reported as `None`. `Closed` stays an error so a dead guest fails
/// the dial instead of reading as an idle wire.
pub struct ReadyRx(pub FrameSource);

impl Pipe for ReadyRx {
    type In = ();
    type Out = Option<Vec<u8>>;
    type Err = FrameError;

    fn call(&self, (): ()) -> impl Future<Output = Result<Option<Vec<u8>>, FrameError>> {
        async move {
            match self.0.call(()).await {
                Ok(frame) => Ok(Some(frame)),
                Err(FrameError::Empty) => Ok(None),
                Err(other) => Err(other),
            }
        }
    }
}

/// `Vec<u8> -> ()`: the host-to-guest frame sink that rings the vCPU's
/// doorbell after each accepted frame.
pub struct RingingTx {
    pub sink: FrameSink,
    pub doorbell: Arc<Doorbell>,
}

impl Pipe for RingingTx {
    type In = Vec<u8>;
    type Out = ();
    type Err = FrameError;

    fn call(&self, frame: Vec<u8>) -> impl Future<Output = Result<(), FrameError>> {
        async move {
            self.sink.call(frame).await?;
            self.doorbell.ring();
            Ok(())
        }
    }
}

/// `u16 -> Vec<u8>`: a zeroed frame of the requested length.
pub struct ZeroMint;

impl Pipe for ZeroMint {
    type In = u16;
    type Out = Vec<u8>;
    type Err = core::convert::Infallible;

    fn call(&self, length: u16) -> impl Future<Output = Result<Vec<u8>, Self::Err>> {
        core::future::ready(Ok(vec![0; usize::from(length)]))
    }
}

pub type HostPipes = (ReadyRx, RingingTx, ZeroMint);

/// An upstream dialing `guest` over `link`, as host 10.0.2.2.
#[must_use]
pub fn dial_guest(link: HostLink, guest: SocketAddrV4) -> FrameUpstream<ReadyRx, RingingTx, ZeroMint> {
    let pipes = (
        ReadyRx(link.from_guest),
        RingingTx { sink: link.to_guest, doorbell: link.doorbell },
        ZeroMint,
    );
    let local = SocketAddrV4::new(HOST_IP, HOST_PORT);
    FrameUpstream::new(Driver::new(pipes, HOST_MAC, local, HOST_INITIAL_SEQUENCE), guest)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use core::pin::pin;
    use core::task::{Context, Poll, Waker};
    use std::io;
    use std::thread;

    use futures::executor::block_on;
    use futures::future::join;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_net::frame_stack::FrameListener;
    use proxima_primitives::stream::{StreamListenerExt, StreamUpstreamExt};
    use proxima_protocols::http1_codec::h1::HttpVersion;
    use proxima_protocols::http1_codec::h1_body::BodyFraming;
    use proxima_protocols::http1_codec::h1_client::encode_request_head;
    use proxima_protocols::http1_codec::h1_response::write_response_head;

    use super::*;
    use crate::frame_pipe::{frame_channel, frame_channel_waking};

    const GUEST_IP: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 15);
    const GUEST_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
    const GUEST_HTTP: SocketAddrV4 = SocketAddrV4::new(GUEST_IP, 80);
    const BODY: &[u8] = b"ok\n";

    type GuestPipes = (ReadyRx, FrameSink, ZeroMint);

    fn request_bytes() -> Vec<u8> {
        let mut request = Vec::new();
        let headers = [("Host", "10.0.2.15"), ("User-Agent", "proxima"), ("Accept", "*/*")];
        encode_request_head("GET", "/healthz", &headers, &mut request);
        request
    }

    fn response_bytes() -> Vec<u8> {
        let mut response = Vec::new();
        let headers = [
            ("Content-Type".to_owned(), "text/plain".to_owned()),
            ("Connection".to_owned(), "close".to_owned()),
        ];
        let framing = BodyFraming::ContentLength(BODY.len() as u64);
        write_response_head(&mut response, HttpVersion::Http11, 200, "OK", &headers, framing);
        response.extend_from_slice(BODY);
        response
    }

    fn crossed_link(flag: &mut u8) -> (HostLink, GuestPipes) {
        let (to_guest, guest_ingress) = frame_channel_waking(64, thread::current());
        let (guest_egress, from_guest) = frame_channel(64);
        let doorbell = Doorbell::for_current_thread(flag).expect("sigusr2 handler installs");
        let link = HostLink { to_guest, from_guest, doorbell: Arc::new(doorbell) };
        (link, (ReadyRx(guest_ingress), guest_egress, ZeroMint))
    }

    fn guest_listener(pipes: GuestPipes) -> FrameListener<ReadyRx, FrameSink, ZeroMint> {
        FrameListener::new(Driver::new(pipes, GUEST_MAC, GUEST_HTTP, 0x9000), GUEST_HTTP)
    }

    async fn serve(listener: &FrameListener<ReadyRx, FrameSink, ZeroMint>, response: &[u8]) -> Vec<u8> {
        let mut connection = listener.accept().await.expect("guest accepts the host's dial");
        let mut request = Vec::new();
        let mut chunk = [0_u8; 256];
        while !request.ends_with(b"\r\n\r\n") {
            let read = connection.read(&mut chunk).await.expect("guest reads the request");
            assert!(read > 0, "host closed before finishing the request head");
            request.extend_from_slice(&chunk[..read]);
        }
        connection.write_all(response).await.expect("guest writes the response");
        connection.close().await.expect("guest closes");
        request
    }

    async fn call(upstream: &FrameUpstream<ReadyRx, RingingTx, ZeroMint>, request: &[u8]) -> Vec<u8> {
        let mut connection = upstream.connect().await.expect("host completes the handshake");
        connection.write_all(request).await.expect("host writes the request");
        let mut response = Vec::new();
        connection.read_to_end(&mut response).await.expect("host reads to the guest's FIN");
        connection.close().await.expect("host closes");
        response
    }

    #[test]
    fn host_http1_request_reaches_the_guest_listener_and_the_response_returns_byte_equal() {
        let mut flag = 0_u8;
        let (link, guest_pipes) = crossed_link(&mut flag);
        let doorbell = Arc::clone(&link.doorbell);
        let listener = guest_listener(guest_pipes);
        let upstream = dial_guest(link, GUEST_HTTP);
        let (request, response) = (request_bytes(), response_bytes());

        let (request_seen, response_seen) =
            block_on(join(serve(&listener, &response), call(&upstream, &request)));
        doorbell.retire();

        assert_eq!(request_seen, request);
        assert_eq!(response_seen, response);
    }

    #[test]
    fn dialing_a_guest_port_nothing_listens_on_is_refused_within_a_bounded_number_of_polls() {
        const POLL_BUDGET: usize = 64;
        let mut flag = 0_u8;
        let (link, guest_pipes) = crossed_link(&mut flag);
        let doorbell = Arc::clone(&link.doorbell);
        let listener = guest_listener(guest_pipes);
        let upstream = dial_guest(link, SocketAddrV4::new(GUEST_IP, 9999));
        let mut context = Context::from_waker(Waker::noop());
        let mut accept = pin!(listener.accept());
        let mut connect = pin!(upstream.connect());

        let outcome = (0..POLL_BUDGET).find_map(|_| {
            assert!(accept.as_mut().poll(&mut context).is_pending(), "nothing listens on the dialed port");
            match connect.as_mut().poll(&mut context) {
                Poll::Ready(result) => Some(result),
                Poll::Pending => None,
            }
        });
        doorbell.retire();

        let error = outcome
            .expect("dial resolves inside the poll budget")
            .err()
            .expect("dial to a closed port must fail");
        assert_eq!(error.kind(), io::ErrorKind::ConnectionRefused);
    }
}
