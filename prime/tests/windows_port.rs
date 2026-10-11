#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-virtual-clock",
))]

use std::fmt::Display;
use std::future::{Future, pending, poll_fn};
use std::io::{self, Read, Write};
use std::net::{
    Ipv4Addr, Shutdown, SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream,
    UdpSocket as StdUdpSocket,
};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::task::{Context as TaskContext, Poll, Wake, Waker};
use std::thread;
use std::time::{Duration, Instant};

use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use prime::core::local_executor::LocalExecutor;
use prime::os::core_shard::{self, CoreShardHandle};
use prime::os::net::{DATAGRAM_BATCH, TcpListener, TcpStream, UdpSocket};
use prime::os::reactor::{Interest, Reactor};
use prime::os::readiness::Readiness;
use proxima_clock::{coarse::TickCell, ticks::Ticks};
use proxima_runtime::CoreId;
use socket2::{Domain, Protocol, SockRef, Socket, Type};

const GUARD: Duration = Duration::from_secs(10);
const REQUEST: &[u8] = b"GET /windows-port HTTP/1.1\r\nHost: localhost\r\n\r\n";
const RESPONSE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
const DATAGRAM: &[u8] = b"proxima windows udp probe";

type Failure = String;

trait Context<Value> {
    fn context(self, what: &str) -> Result<Value, Failure>;
}

impl<Value, Cause: Display> Context<Value> for Result<Value, Cause> {
    fn context(self, what: &str) -> Result<Value, Failure> {
        self.map_err(|cause| format!("{what}: {cause}"))
    }
}

impl<Value> Context<Value> for Option<Value> {
    fn context(self, what: &str) -> Result<Value, Failure> {
        self.ok_or_else(|| what.to_string())
    }
}

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

fn worker() -> Result<CoreShardHandle, Failure> {
    core_shard::launch_with_lanes(CoreId(0), None, 4, 16).context("launch prime worker")
}

fn configure_peer(peer: &StdTcpStream) -> Result<(), Failure> {
    peer.set_read_timeout(Some(GUARD))
        .context("set peer read guard")?;
    peer.set_write_timeout(Some(GUARD))
        .context("set peer write guard")
}

#[test]
fn windows_port_tcp_loopback_payload() -> Result<(), Failure> {
    let handle = worker()?;
    let oracle = StdTcpListener::bind(loopback()).expect("bind independent listener");
    let address = oracle.local_addr().expect("read oracle address");
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let mut stream = TcpStream::connect(address)
                .await
                .expect("prime connects to std peer");
            stream
                .write_all(REQUEST)
                .await
                .expect("prime sends request");
            let mut payload = vec![0; RESPONSE.len()];
            stream
                .read_exact(&mut payload)
                .await
                .expect("prime receives response");
            completed.send(payload).expect("report prime response");
        })
        .expect("dispatch prime connect");
    let (mut peer, _) = oracle.accept().expect("std accepts prime connection");
    configure_peer(&peer)?;
    let mut payload = vec![0; REQUEST.len()];
    peer.read_exact(&mut payload).expect("std reads request");
    assert_eq!(payload, REQUEST);
    peer.write_all(RESPONSE).expect("std writes response");
    assert_eq!(
        result.recv_timeout(GUARD).expect("receive prime result"),
        RESPONSE
    );

    let mut listener = TcpListener::bind(loopback()).expect("bind prime listener");
    let address = listener.local_addr().expect("read prime address");
    #[cfg(windows)]
    assert!(
        TcpListener::bind(address).is_err(),
        "Windows listener bind is exclusive"
    );
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let (mut stream, peer_address) =
                listener.accept().await.expect("prime accepts std peer");
            let mut payload = vec![0; REQUEST.len()];
            stream
                .read_exact(&mut payload)
                .await
                .expect("prime receives request");
            stream
                .write_all(RESPONSE)
                .await
                .expect("prime sends response");
            completed
                .send((payload, peer_address))
                .expect("report prime request");
        })
        .expect("dispatch prime accept");
    let mut peer = StdTcpStream::connect(address).expect("std connects to prime listener");
    configure_peer(&peer)?;
    let peer_address = peer.local_addr().expect("read std client address");
    peer.write_all(REQUEST).expect("std sends request");
    let mut payload = vec![0; RESPONSE.len()];
    peer.read_exact(&mut payload)
        .expect("std receives response");
    assert_eq!(payload, RESPONSE);
    let (request, observed_peer) = result.recv_timeout(GUARD).expect("receive prime request");
    assert_eq!(request, REQUEST);
    assert_eq!(observed_peer, peer_address);
    repeated_pending_accept(&handle)?;
    handle.shutdown_and_join().expect("join prime worker");
    Ok(())
}

fn repeated_pending_accept(handle: &CoreShardHandle) -> Result<(), Failure> {
    let mut listener = TcpListener::bind(loopback()).context("bind pending accept listener")?;
    let address = listener
        .local_addr()
        .context("read pending accept address")?;
    let (registered, registration) = mpsc::channel();
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            for _round in 0..2 {
                let outcome = async {
                    let mut announced = false;
                    let (mut stream, source) = poll_fn(|context| {
                        let outcome = Pin::new(&mut listener).poll_accept(context);
                        if outcome.is_pending() && !announced {
                            announced = true;
                            // a gone receiver means the test already failed on its own side
                            let _ = registered.send(());
                        }
                        outcome
                    })
                    .await
                    .context("accept after client connects")?;
                    assert!(announced, "client must wait for accept Pending");
                    let mut received = vec![0; REQUEST.len()];
                    stream
                        .read_exact(&mut received)
                        .await
                        .context("read accepted request")?;
                    assert_eq!(received, REQUEST);
                    stream
                        .write_all(RESPONSE)
                        .await
                        .context("reply to accepted peer")?;
                    Ok::<_, Failure>(source)
                }
                .await;
                let failed = outcome.is_err();
                drop(completed.send(outcome));
                if failed {
                    break;
                }
            }
        })
        .context("dispatch repeated pending accepts")?;
    for _round in 0..2 {
        registration
            .recv_timeout(GUARD)
            .context("accept returned Pending")?;
        let mut peer = StdTcpStream::connect(address).context("connect after accept Pending")?;
        configure_peer(&peer)?;
        peer.write_all(REQUEST).context("send accepted request")?;
        let mut received = vec![0; RESPONSE.len()];
        peer.read_exact(&mut received)
            .context("read accepted reply")?;
        assert_eq!(received, RESPONSE);
        assert_eq!(
            result
                .recv_timeout(GUARD)
                .context("receive accepted source")??,
            peer.local_addr().context("read client address")?
        );
    }
    Ok(())
}

#[test]
fn windows_port_udp_loopback_payload() -> Result<(), Failure> {
    let truncated_oracle = StdUdpSocket::bind(loopback()).expect("bind std truncation oracle");
    truncated_oracle
        .set_read_timeout(Some(GUARD))
        .expect("set truncation oracle guard");
    let truncation_sender = StdUdpSocket::bind(loopback()).expect("bind truncation sender");
    truncation_sender
        .send_to(
            DATAGRAM,
            truncated_oracle.local_addr().expect("oracle address"),
        )
        .expect("send independent truncation probe");
    let mut oracle_prefix = [0; 7];
    let oracle_result = truncated_oracle.recv_from(&mut oracle_prefix);
    #[cfg(unix)]
    assert_eq!(
        oracle_result.expect("std receives prefix").0,
        oracle_prefix.len()
    );
    #[cfg(windows)]
    assert_eq!(
        oracle_result
            .expect_err("Winsock reports oversized message")
            .raw_os_error(),
        Some(windows_sys::Win32::Networking::WinSock::WSAEMSGSIZE)
    );
    assert_eq!(&oracle_prefix, &DATAGRAM[..7]);
    let handle = worker()?;
    let mut socket = UdpSocket::bind(loopback()).expect("bind prime UDP");
    let address = socket.local_addr().expect("read prime UDP address");
    #[cfg(windows)]
    assert!(
        UdpSocket::bind(address).is_err(),
        "Windows UDP bind is exclusive"
    );
    let oracle = StdUdpSocket::bind(loopback()).expect("bind std UDP");
    oracle.set_read_timeout(Some(GUARD)).expect("set UDP guard");
    let peer_address = oracle.local_addr().expect("read std UDP address");
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let mut buffer = [0; 128];
            let (count, source) =
                poll_fn(|context| Pin::new(&mut socket).poll_recv_from(context, &mut buffer))
                    .await
                    .expect("receive UDP probe");
            assert_eq!(&buffer[..count], DATAGRAM);
            assert_eq!(source, peer_address);
            let sent =
                poll_fn(|context| Pin::new(&mut socket).poll_send_to(context, DATAGRAM, source))
                    .await
                    .expect("return UDP probe");
            assert_eq!(sent, DATAGRAM.len());
            let mut prefix = [0; 7];
            let (truncated, source) =
                poll_fn(|context| Pin::new(&mut socket).poll_recv_from(context, &mut prefix))
                    .await
                    .expect("receive truncated datagram");
            assert_eq!(&prefix[..truncated], &DATAGRAM[..7]);
            assert_eq!(source, peer_address);
            completed
                .send((count, truncated))
                .expect("report UDP counts");
        })
        .expect("dispatch UDP task");
    assert_eq!(
        oracle.send_to(DATAGRAM, address).expect("send UDP probe"),
        DATAGRAM.len()
    );
    let mut buffer = [0; 128];
    let (count, source) = oracle
        .recv_from(&mut buffer)
        .expect("receive prime UDP response");
    assert_eq!(&buffer[..count], DATAGRAM);
    assert_eq!(source, address);
    oracle
        .send_to(DATAGRAM, address)
        .expect("send truncation probe");
    assert_eq!(
        result.recv_timeout(GUARD).expect("receive UDP result"),
        (DATAGRAM.len(), 7)
    );
    udp_receive_batch_limit(&handle)?;
    handle.shutdown_and_join().expect("join prime worker");
    Ok(())
}

fn udp_receive_batch_limit(handle: &CoreShardHandle) -> Result<(), Failure> {
    let mut socket = UdpSocket::bind(loopback()).context("bind batch receiver")?;
    let address = socket.local_addr().context("batch receiver address")?;
    let sender = StdUdpSocket::bind(loopback()).context("bind batch sender")?;
    let source = sender.local_addr().context("batch sender address")?;
    for index in 0_u8..40 {
        let mut payload = vec![index];
        payload.extend_from_slice(DATAGRAM);
        assert_eq!(
            sender
                .send_to(&payload, address)
                .context("prequeue datagram")?,
            payload.len()
        );
    }
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let outcome = async {
                let mut seen = [false; 40];
                let mut received = 0;
                while received < seen.len() {
                    let mut storage = [[0xa5_u8; 64]; 64];
                    let mut buffers: Vec<&mut [u8]> = storage
                        .iter_mut()
                        .map(|buffer| buffer.as_mut_slice())
                        .collect();
                    let sentinel = (usize::MAX, loopback());
                    let mut metadata = [sentinel; 64];
                    let count = poll_fn(|context| {
                        Pin::new(&mut socket).poll_recv_batch(context, &mut buffers, &mut metadata)
                    })
                    .await
                    .context("receive queued datagram batch")?;
                    assert!(count > 0 && count <= DATAGRAM_BATCH);
                    if received == 0 {
                        assert_eq!(count, 32, "prequeued batch must stop at the portable limit");
                    }
                    for index in 0..count {
                        let (length, peer) = metadata[index];
                        assert_eq!(peer, source);
                        assert_eq!(length, DATAGRAM.len() + 1);
                        assert_eq!(&storage[index][1..length], DATAGRAM);
                        let sequence = usize::from(storage[index][0]);
                        assert!(sequence < seen.len());
                        assert!(!seen[sequence], "duplicate batch payload");
                        seen[sequence] = true;
                    }
                    for index in count..metadata.len() {
                        assert_eq!(metadata[index], sentinel);
                        assert_eq!(storage[index], [0xa5; 64]);
                    }
                    received += count;
                }
                assert_eq!(seen, [true; 40]);
                Ok::<_, Failure>(received)
            }
            .await;
            drop(completed.send(outcome));
        })
        .context("dispatch batch limit check")?;
    assert_eq!(
        result
            .recv_timeout(GUARD)
            .context("all prequeued datagrams received")??,
        40
    );
    Ok(())
}

#[test]
fn windows_port_tcp_pending_read_wakes() -> Result<(), Failure> {
    let handle = worker()?;
    let oracle = StdTcpListener::bind(loopback()).expect("bind std listener");
    let address = oracle.local_addr().expect("read oracle address");
    let (registered, registration) = mpsc::channel();
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let mut stream = TcpStream::connect(address)
                .await
                .expect("connect prime stream");
            for expected in [b"first".as_slice(), b"second".as_slice()] {
                let mut buffer = [0; 32];
                let mut announced = false;
                let count = poll_fn(|context| {
                    let outcome = Pin::new(&mut stream).poll_read(context, &mut buffer);
                    if outcome.is_pending() && !announced {
                        announced = true;
                        registered.send(()).expect("announce pending TCP read");
                    }
                    outcome
                })
                .await
                .expect("read after readiness");
                assert!(announced, "peer must wait for the Pending registration");
                assert!(count <= expected.len());
                stream
                    .read_exact(&mut buffer[count..expected.len()])
                    .await
                    .expect("complete partial TCP read");
                assert_eq!(&buffer[..expected.len()], expected);
            }
            completed.send(()).expect("report repeated reads");
        })
        .expect("dispatch pending TCP reader");
    let (mut peer, _) = oracle.accept().expect("accept prime connection");
    configure_peer(&peer)?;
    for payload in [b"first".as_slice(), b"second".as_slice()] {
        registration
            .recv_timeout(GUARD)
            .expect("wait until TCP poll returned Pending");
        peer.write_all(payload)
            .expect("write only after read registration");
    }
    result
        .recv_timeout(GUARD)
        .expect("both TCP reads completed");
    tcp_pending_write_wakes(&handle)?;
    handle.shutdown_and_join().expect("join prime worker");
    Ok(())
}

fn tcp_local_half_close(handle: &CoreShardHandle) -> Result<(), Failure> {
    let oracle = StdTcpListener::bind(loopback()).context("bind half-close peer")?;
    let address = oracle.local_addr().context("half-close peer address")?;
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let outcome = async {
                let mut stream = TcpStream::connect(address)
                    .await
                    .context("connect half-close stream")?;
                stream
                    .write_all(REQUEST)
                    .await
                    .context("write before local half-close")?;
                stream
                    .close()
                    .await
                    .context("close only Prime write direction")?;
                let mut received = vec![0; RESPONSE.len()];
                stream
                    .read_exact(&mut received)
                    .await
                    .context("read reply after local half-close")?;
                Ok::<_, Failure>(received)
            }
            .await;
            drop(completed.send(outcome));
        })
        .context("dispatch half-close check")?;
    let (mut peer, _) = oracle.accept().context("accept half-close stream")?;
    configure_peer(&peer)?;
    let mut received = Vec::new();
    peer.read_to_end(&mut received)
        .context("observe Prime write EOF")?;
    assert_eq!(received, REQUEST);
    peer.write_all(RESPONSE)
        .context("reply after observing peer EOF")?;
    assert_eq!(
        result
            .recv_timeout(GUARD)
            .context("Prime reads after close")??,
        RESPONSE
    );
    tcp_shutdown_error_oracle(handle)
}

type ObservedError = (io::ErrorKind, Option<i32>);

fn tcp_shutdown_error_oracle(handle: &CoreShardHandle) -> Result<(), Failure> {
    let listener = StdTcpListener::bind(loopback()).context("bind reset peer")?;
    let address = listener.local_addr().context("reset peer address")?;
    let (connected, connection) = mpsc::channel();
    let oracle = thread::spawn(move || {
        let mut stream = StdTcpStream::connect(address).context("connect std reset oracle")?;
        configure_peer(&stream)?;
        connected.send(()).context("report connected std oracle")?;
        let error = match stream.read(&mut [0; 1]) {
            Ok(count) => {
                return Err(format!(
                    "std observes abortive peer close: read returned {count} bytes"
                ));
            }
            Err(error) => error,
        };
        let closed = stream
            .shutdown(Shutdown::Write)
            .map_err(|error| (error.kind(), error.raw_os_error()));
        Ok::<_, Failure>(((error.kind(), error.raw_os_error()), closed))
    });
    let (peer, _) = listener.accept().context("accept std reset oracle")?;
    connection
        .recv_timeout(GUARD)
        .context("std connect completed before reset")?;
    SockRef::from(&peer)
        .set_linger(Some(Duration::ZERO))
        .context("set abortive std close")?;
    drop(peer);
    let expected = oracle
        .join()
        .map_err(|_| Failure::from("reset oracle thread panicked"))??;
    assert_eq!(expected.0.0, io::ErrorKind::ConnectionReset);

    let (completed, result) = mpsc::channel();
    let (connected, connection) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let outcome = async {
                let mut stream = TcpStream::connect(address)
                    .await
                    .context("connect Prime reset stream")?;
                connected
                    .send(())
                    .context("report connected Prime stream")?;
                let error = match stream.read(&mut [0; 1]).await {
                    Ok(count) => {
                        return Err(format!(
                            "Prime observes abortive peer close: read returned {count} bytes"
                        ));
                    }
                    Err(error) => error,
                };
                let closed = stream
                    .close()
                    .await
                    .map_err(|error| (error.kind(), error.raw_os_error()));
                Ok::<_, Failure>(((error.kind(), error.raw_os_error()), closed))
            }
            .await;
            drop(completed.send(outcome));
        })
        .context("dispatch reset close oracle")?;
    let (peer, _) = listener.accept().context("accept Prime reset stream")?;
    connection
        .recv_timeout(GUARD)
        .context("Prime connect completed before reset")?;
    SockRef::from(&peer)
        .set_linger(Some(Duration::ZERO))
        .context("set abortive Prime peer close")?;
    drop(peer);
    let observed: (ObservedError, Result<(), ObservedError>) = result
        .recv_timeout(GUARD)
        .context("receive Prime shutdown result")??;
    assert_eq!(observed, expected);
    Ok(())
}

fn tcp_pending_write_wakes(handle: &CoreShardHandle) -> Result<(), Failure> {
    const PATTERN: &[u8] = b"prime write rearm payload\n";
    const REPETITIONS: usize = 512 * 1024;
    let oracle = StdTcpListener::bind(loopback()).context("bind write backpressure peer")?;
    SockRef::from(&oracle)
        .set_recv_buffer_size(64 * 1024)
        .context("bound independent receive buffer")?;
    let address = oracle.local_addr().context("backpressure peer address")?;
    let (registered, registration) = mpsc::channel();
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let outcome = async {
                let mut stream = TcpStream::connect(address)
                    .await
                    .context("connect backpressure stream")?;
                let payload = PATTERN.repeat(REPETITIONS);
                stream
                    .set_send_buffer_size(64 * 1024)
                    .context("request bounded kernel send buffer")?;
                let send_capacity = stream
                    .send_buffer_size()
                    .context("read effective send buffer")?;
                assert!(
                    send_capacity > 0 && send_capacity < payload.len(),
                    "effective send buffer {send_capacity} must be below payload {}",
                    payload.len()
                );
                let mut offset = 0;
                let mut announced = false;
                poll_fn(|context| {
                    while offset < payload.len() {
                        let end = (offset + 16 * 1024).min(payload.len());
                        match Pin::new(&mut stream).poll_write(context, &payload[offset..end]) {
                            Poll::Ready(Ok(count)) => {
                                assert!(count > 0, "nonempty write cannot make zero progress");
                                offset += count;
                            }
                            Poll::Ready(Err(error)) => {
                                panic!("write after backpressure failed: {error}")
                            }
                            Poll::Pending => {
                                if !announced {
                                    announced = true;
                                    let _ = registered.send(offset);
                                }
                                return Poll::Pending;
                            }
                        }
                    }
                    assert!(
                        announced,
                        "bounded payload must fill socket before peer drainage"
                    );
                    Poll::Ready(())
                })
                .await;
                stream
                    .close()
                    .await
                    .context("half-close completed writer")?;
                Ok::<_, Failure>(offset)
            }
            .await;
            drop(completed.send(outcome));
        })
        .context("dispatch backpressure writer")?;
    let (mut peer, _) = oracle.accept().context("accept backpressure stream")?;
    configure_peer(&peer)?;
    let pending_offset = registration
        .recv_timeout(GUARD)
        .context("writer returned Pending before drainage")?;
    assert!(pending_offset > 0 && pending_offset < PATTERN.len() * REPETITIONS);
    let mut payload = Vec::new();
    peer.read_to_end(&mut payload)
        .context("drain writer through EOF")?;
    assert_eq!(payload.len(), PATTERN.len() * REPETITIONS);
    let (chunks, remainder) = payload.as_chunks::<{ PATTERN.len() }>();
    assert!(remainder.is_empty());
    for chunk in chunks {
        assert_eq!(chunk.as_slice(), PATTERN);
    }
    assert_eq!(
        result
            .recv_timeout(GUARD)
            .context("write resumed to completion")??,
        payload.len()
    );
    Ok(())
}

#[test]
fn windows_port_udp_pending_receive_wakes() -> Result<(), Failure> {
    let handle = worker()?;
    let mut socket = UdpSocket::bind(loopback()).expect("bind prime UDP");
    let address = socket.local_addr().expect("read prime address");
    let oracle = StdUdpSocket::bind(loopback()).expect("bind std UDP");
    let peer_address = oracle.local_addr().expect("read std address");
    let (registered, registration) = mpsc::channel();
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            for _round in 0..2 {
                let mut buffer = [0; 128];
                let mut announced = false;
                let (count, source) = poll_fn(|context| {
                    let outcome = Pin::new(&mut socket).poll_recv_from(context, &mut buffer);
                    if outcome.is_pending() && !announced {
                        announced = true;
                        registered.send(()).expect("announce pending UDP receive");
                    }
                    outcome
                })
                .await
                .expect("receive after readiness");
                assert!(announced);
                assert_eq!(&buffer[..count], DATAGRAM);
                assert_eq!(source, peer_address);
            }
            completed.send(()).expect("report repeated UDP receives");
        })
        .expect("dispatch UDP reader");
    for _round in 0..2 {
        registration
            .recv_timeout(GUARD)
            .expect("wait until UDP poll returned Pending");
        oracle
            .send_to(DATAGRAM, address)
            .expect("send after pending registration");
    }
    result
        .recv_timeout(GUARD)
        .expect("both UDP receives completed");
    handle.shutdown_and_join().expect("join prime worker");
    Ok(())
}

#[test]
fn windows_port_tcp_peer_close_reports_eof() -> Result<(), Failure> {
    let handle = worker()?;
    let oracle = StdTcpListener::bind(loopback()).expect("bind std listener");
    let address = oracle.local_addr().expect("read oracle address");
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let mut stream = TcpStream::connect(address)
                .await
                .expect("connect prime stream");
            let mut buffer = [0; 1];
            completed
                .send(stream.read(&mut buffer).await.expect("read peer EOF"))
                .expect("report EOF");
        })
        .expect("dispatch EOF reader");
    let (peer, _) = oracle.accept().expect("accept connection");
    peer.shutdown(Shutdown::Write)
        .expect("close peer write direction");
    assert_eq!(result.recv_timeout(GUARD).expect("receive EOF count"), 0);
    tcp_local_half_close(&handle)?;
    let (mut transferred, mut peer) = registered_stream(&handle)?;
    let mut buffer = [0; 1];
    let mut context = TaskContext::from_waker(Waker::noop());
    assert!(
        matches!(
            Pin::new(&mut transferred).poll_read(&mut context, &mut buffer),
            Poll::Ready(Err(_))
        ),
        "foreign polling must reject a registered Prime socket"
    );
    thread::spawn(move || drop(transferred))
        .join()
        .expect("drop registered socket on another thread");
    assert_eq!(
        peer.read(&mut buffer)
            .expect("owner cancellation closes the socket"),
        0
    );
    let (late, mut peer) = registered_stream(&handle)?;
    handle
        .shutdown_and_join()
        .expect("join prime worker with transferred source");
    drop(late);
    assert_eq!(
        peer.read(&mut buffer)
            .expect("drop after worker shutdown closes the socket"),
        0
    );
    Ok(())
}

fn registered_stream(handle: &CoreShardHandle) -> Result<(TcpStream, StdTcpStream), Failure> {
    let oracle = StdTcpListener::bind(loopback()).context("bind owner-transfer peer")?;
    let address = oracle.local_addr().context("owner-transfer address")?;
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let outcome = async {
                let mut stream = TcpStream::connect(address)
                    .await
                    .context("connect owner-transfer stream")?;
                let mut buffer = [0; 1];
                poll_fn(|context| {
                    assert!(
                        Pin::new(&mut stream)
                            .poll_read(context, &mut buffer)
                            .is_pending(),
                        "source must be registered before transfer"
                    );
                    Poll::Ready(())
                })
                .await;
                Ok::<_, Failure>(stream)
            }
            .await;
            drop(completed.send(outcome));
        })
        .context("dispatch owner-transfer source")?;
    let (peer, _) = oracle
        .accept()
        .context("accept owner-transfer connection")?;
    configure_peer(&peer)?;
    let transferred = result
        .recv_timeout(GUARD)
        .context("receive registered Prime stream")??;
    Ok((transferred, peer))
}

#[test]
fn windows_port_tcp_connect_refused_reports_error() -> Result<(), Failure> {
    let handle = worker()?;
    let bound = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .expect("create unlistened endpoint");
    bound
        .bind(&loopback().into())
        .expect("hold unlistened endpoint bound");
    let address = bound
        .local_addr()
        .expect("read bound address")
        .as_socket()
        .expect("IP address");
    let oracle = thread::spawn(move || {
        StdTcpStream::connect(address)
            .expect_err("std peer must reject the same unlistened endpoint")
            .kind()
    });
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let error = TcpStream::connect(address)
                .await
                .err()
                .expect("unlistened endpoint must refuse connect");
            let sources = core_shard::with_current_reactor(|reactor| reactor.live_sources())
                .expect("worker reactor installed");
            completed
                .send((error.kind(), sources))
                .expect("report refused connect");
        })
        .expect("dispatch refused connect");
    let (kind, sources) = result.recv_timeout(GUARD).expect("receive refusal");
    let oracle_kind = oracle.join().expect("join independent connect oracle");
    assert_eq!(
        kind, oracle_kind,
        "prime must preserve the OS error observed by std on the same endpoint"
    );
    assert_eq!(
        sources, 0,
        "failed connect must remove its borrowed registration"
    );
    handle.shutdown_and_join().expect("join prime worker");
    Ok(())
}

#[test]
fn windows_port_reactor_external_wake() {
    let mut reactor = Reactor::new().expect("create reactor");
    let wakeup = reactor.wakeup();
    reactor.arm_wakeup();
    let producer = thread::spawn(move || wakeup.fire());
    reactor.turn(None).expect("external wake releases reactor");
    producer.join().expect("join wake producer");
    {
        let retained = reactor.wakeup();
        reactor.arm_wakeup();
        drop(reactor);
        retained.fire();
    }

    let clock = Arc::new(TickCell::new(Ticks::ZERO));
    let handle = core_shard::launch_with_virtual_clock(CoreId(0), None, 4, 16, clock.clone())
        .expect("launch virtual-clock worker");
    let (release, mut released) = oneshot::channel();
    let (registered, registration) = mpsc::channel();
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let mut announced = false;
            poll_fn(|context| {
                let outcome = Pin::new(&mut released).poll(context);
                if outcome.is_pending() && !announced {
                    announced = true;
                    registered
                        .send(())
                        .expect("announce external wake registration");
                }
                outcome
            })
            .await
            .expect("external producer releases task");
            core_shard::timer_at(25_000).await;
            completed
                .send(core_shard::current_tick())
                .expect("report timer expiry");
        })
        .expect("dispatch external wake task");
    registration
        .recv_timeout(GUARD)
        .expect("task registered cross-thread waker");
    release
        .send(())
        .expect("release worker from another thread");
    assert_eq!(
        result.recv_timeout(GUARD).expect("virtual timer expired"),
        25_000
    );
    assert_eq!(clock.get(), Ticks::from_raw(25_000));
    handle
        .shutdown_and_join()
        .expect("join virtual-clock worker");
}

struct WakeCount(AtomicUsize);

impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn register(
    reactor: &mut Reactor,
    socket: &StdUdpSocket,
    interest: Interest,
) -> std::io::Result<prime::os::reactor::SourceKey> {
    #[cfg(unix)]
    {
        reactor.register(socket.as_raw_fd(), interest)
    }
    #[cfg(windows)]
    {
        reactor.register(socket, interest)
    }
}

#[test]
fn windows_port_reactor_deregister_rejects_stale_generation() -> Result<(), Failure> {
    let mut reactor = Reactor::new().expect("create reactor");
    let first = StdUdpSocket::bind(loopback()).expect("bind first source");
    let second = StdUdpSocket::bind(loopback()).expect("bind replacement source");
    let oracle = StdUdpSocket::bind(loopback()).expect("bind independent sender");
    first
        .set_nonblocking(true)
        .expect("nonblocking first source");
    second
        .set_nonblocking(true)
        .expect("nonblocking replacement source");
    let counter = Arc::new(WakeCount(AtomicUsize::new(0)));
    let waker = Waker::from(counter.clone());
    let old = register(&mut reactor, &first, Interest::Read).expect("register old source");
    oracle
        .send_to(DATAGRAM, first.local_addr().expect("first address"))
        .expect("queue stale readiness");
    reactor.deregister(old).expect("deregister old source");
    let current = register(&mut reactor, &second, Interest::ReadWrite).expect("reuse source slot");
    assert_ne!(old, current);
    assert!(!reactor.set_read_waker(old, waker.clone()));
    assert!(reactor.reregister(old, Interest::Read).is_err());
    assert!(reactor.set_read_waker(current, waker));
    let writes = Arc::new(WakeCount(AtomicUsize::new(0)));
    assert!(reactor.set_write_waker(current, Waker::from(writes.clone())));
    reactor
        .reregister(current, Interest::Write)
        .expect("select write interest");
    reactor
        .reregister(current, Interest::Read)
        .expect("replace write with read interest");
    reactor
        .deregister(old)
        .expect("stale deregister is harmless");
    assert_eq!(reactor.live_sources(), 1);
    reactor
        .turn(Some(Duration::ZERO))
        .expect("consume writable-only event");
    assert_eq!(counter.0.load(Ordering::SeqCst), 0);
    assert_eq!(
        writes.0.load(Ordering::SeqCst),
        0,
        "narrowed write interest must not fire"
    );
    for expected in 1..=2 {
        #[cfg(windows)]
        reactor
            .rearm(current, Interest::Read)
            .expect("rearm pending reader");
        oracle
            .send_to(DATAGRAM, second.local_addr().expect("replacement address"))
            .expect("send replacement payload");
        let deadline = Instant::now() + GUARD;
        while counter.0.load(Ordering::SeqCst) < expected {
            assert!(
                Instant::now() < deadline,
                "replacement reader lost readiness"
            );
            reactor
                .turn(Some(deadline.saturating_duration_since(Instant::now())))
                .expect("dispatch source readiness");
        }
        let mut buffer = [0; 128];
        let (count, _) = second
            .recv_from(&mut buffer)
            .expect("drain replacement socket");
        assert_eq!(&buffer[..count], DATAGRAM);
        assert_eq!(counter.0.load(Ordering::SeqCst), expected);
        if expected == 1 {
            reactor
                .reregister(current, Interest::ReadWrite)
                .expect("arm simultaneous reader and writer");
            let deadline = Instant::now() + GUARD;
            while writes.0.load(Ordering::SeqCst) == 0 {
                assert!(Instant::now() < deadline, "writable source lost readiness");
                reactor
                    .turn(Some(deadline.saturating_duration_since(Instant::now())))
                    .expect("dispatch write while reader pending");
            }
            assert_eq!(
                counter.0.load(Ordering::SeqCst),
                1,
                "write readiness must retain the pending reader"
            );
        }
    }
    #[cfg(windows)]
    {
        thread::spawn(move || drop(second))
            .join()
            .expect("drop original socket on another thread");
        reactor
            .reregister(current, Interest::Read)
            .expect("reactor still owns its socket duplicate");
    }
    reactor
        .deregister(current)
        .expect("remove replacement source");
    assert_eq!(reactor.live_sources(), 0);
    readiness_owner_collision()?;
    reentrant_reactor_callback()?;
    shutdown_reactor_callback()?;
    retained_executor_waker()?;
    Ok(())
}

fn capture_task_waker(executor: &LocalExecutor) -> Result<Waker, Failure> {
    let (captured, received) = mpsc::channel();
    let mut captured = Some(captured);
    executor.spawn_local(poll_fn(move |context| {
        if let Some(captured) = captured.take() {
            drop(captured.send(context.waker().clone()));
        }
        Poll::<()>::Pending
    }));
    assert_eq!(executor.tick(), 1);
    received.recv_timeout(GUARD).context("task waker captured")
}

fn retained_executor_waker() -> Result<(), Failure> {
    let remote_wakes = Arc::new(AtomicUsize::new(0));
    let notified = Arc::clone(&remote_wakes);
    let executor = LocalExecutor::with_remote_wake(Some(Arc::new(move || {
        notified.fetch_add(1, Ordering::SeqCst);
    })));
    executor.arm();
    let retained = capture_task_waker(&executor)?;
    drop(executor);
    retained.wake_by_ref();
    assert_eq!(remote_wakes.load(Ordering::SeqCst), 1);

    let older = LocalExecutor::new();
    older.arm();
    let newer_wakes = Arc::new(AtomicUsize::new(0));
    let newer_notified = Arc::clone(&newer_wakes);
    let newer = LocalExecutor::with_remote_wake(Some(Arc::new(move || {
        newer_notified.fetch_add(1, Ordering::SeqCst);
    })));
    newer.arm();
    let newer_retained = capture_task_waker(&newer)?;
    drop(older);
    newer_retained.wake_by_ref();
    assert_eq!(newer_wakes.load(Ordering::SeqCst), 0);
    assert_eq!(newer.tick(), 1);
    drop(newer);
    newer_retained.wake_by_ref();
    assert_eq!(newer_wakes.load(Ordering::SeqCst), 1);
    Ok(())
}

struct ShutdownWake(mpsc::Sender<bool>);

struct ShutdownTask(mpsc::Sender<bool>);

impl Drop for ShutdownTask {
    fn drop(&mut self) {
        // drop cannot return a failure; the receiving side times out when this report is lost
        let _ = self.0.send(
            core_shard::current_core().is_none()
                && !core_shard::on_worker()
                && core_shard::with_current_reactor(|_| ()).is_none(),
        );
    }
}

impl Wake for ShutdownWake {
    fn wake(self: Arc<Self>) {
        drop(self);
    }
}

impl Drop for ShutdownWake {
    fn drop(&mut self) {
        let _ = self
            .0
            .send(core_shard::with_current_reactor(|_| ()).is_none());
    }
}

fn shutdown_reactor_callback() -> Result<(), Failure> {
    let handle = worker()?;
    let (completed, result) = mpsc::channel();
    let (dropped, drop_result) = mpsc::channel();
    let (task_dropped, task_drop_result) = mpsc::channel();
    let (installed, installation) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let _task = ShutdownTask(task_dropped);
            let _ = installed.send(());
            pending::<()>().await;
        })
        .context("dispatch pending shutdown task")?;
    installation
        .recv_timeout(GUARD)
        .context("pending shutdown task installed")?;
    handle
        .dispatch_send_inline(async move {
            let outcome = async {
                let source =
                    StdUdpSocket::bind(loopback()).context("bind shutdown callback source")?;
                source
                    .set_nonblocking(true)
                    .context("nonblocking shutdown source")?;
                core_shard::with_current_reactor(|reactor| {
                    let key = register(reactor, &source, Interest::Read)
                        .context("register shutdown callback")?;
                    assert!(
                        reactor.set_read_waker(key, Waker::from(Arc::new(ShutdownWake(dropped))))
                    );
                    Ok::<(), Failure>(())
                })
                .context("borrow reactor for shutdown callback")??;
                Ok::<_, Failure>(source)
            }
            .await;
            drop(completed.send(outcome));
        })
        .context("dispatch shutdown callback check")?;
    let source = result
        .recv_timeout(GUARD)
        .context("receive registered source")??;
    handle
        .shutdown_and_join()
        .context("join shutdown callback worker")?;
    assert!(
        drop_result
            .recv_timeout(GUARD)
            .context("shutdown callback ran")?
    );
    assert!(
        task_drop_result
            .recv_timeout(GUARD)
            .context("pending task destructor ran")?
    );
    drop(source);
    Ok(())
}

struct ReentrantWake {
    sockets: crossbeam_queue::SegQueue<UdpSocket>,
    called: AtomicUsize,
}

impl Wake for ReentrantWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        assert!(core_shard::with_current_reactor(|_| ()).is_none());
        drop(self.sockets.pop());
        self.called.fetch_add(1, Ordering::SeqCst);
    }
}

fn reentrant_reactor_callback() -> Result<(), Failure> {
    let handle = worker()?;
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let outcome = async {
                let mut socket =
                    UdpSocket::bind(loopback()).context("bind deferred release source")?;
                let mut buffer = [0; 128];
                poll_fn(|context| {
                    assert!(
                        Pin::new(&mut socket)
                            .poll_recv_from(context, &mut buffer)
                            .is_pending()
                    );
                    Poll::Ready(())
                })
                .await;
                let callback = Arc::new(ReentrantWake {
                    sockets: crossbeam_queue::SegQueue::new(),
                    called: AtomicUsize::new(0),
                });
                callback.sockets.push(socket);
                let source = StdUdpSocket::bind(loopback()).context("bind callback trigger")?;
                source
                    .set_nonblocking(true)
                    .context("nonblocking callback trigger")?;
                let sender = StdUdpSocket::bind(loopback()).context("bind callback sender")?;
                sender
                    .send_to(DATAGRAM, source.local_addr().context("trigger address")?)
                    .context("queue callback payload")?;
                core_shard::with_current_reactor(|reactor| {
                    assert!(core_shard::with_current_reactor(|_| ()).is_none());
                    let key = register(reactor, &source, Interest::Read)
                        .context("register callback source")?;
                    assert!(reactor.set_read_waker(key, Waker::from(callback.clone())));
                    let deadline = Instant::now() + GUARD;
                    while callback.called.load(Ordering::SeqCst) == 0 {
                        assert!(Instant::now() < deadline, "callback did not run");
                        reactor
                            .turn(Some(deadline.saturating_duration_since(Instant::now())))
                            .context("invoke reentrant callback")?;
                    }
                    reactor.deregister(key).context("remove callback trigger")?;
                    reactor
                        .turn(Some(Duration::ZERO))
                        .context("drain callback's deferred socket")?;
                    assert_eq!(reactor.live_sources(), 0);
                    Ok::<(), Failure>(())
                })
                .context("borrow worker reactor")??;
                assert!(
                    core_shard::with_current_reactor(|_| ()).is_some(),
                    "restore reactor after callback"
                );
                Ok::<_, Failure>(())
            }
            .await;
            drop(completed.send(outcome));
        })
        .context("dispatch callback check")?;
    result
        .recv_timeout(GUARD)
        .context("callback check completed")??;
    handle.shutdown_and_join().context("join callback worker")?;
    Ok(())
}

fn make_readiness(socket: &StdUdpSocket) -> Result<Readiness, Failure> {
    #[cfg(unix)]
    {
        Ok(Readiness::new(socket.as_raw_fd()))
    }
    #[cfg(windows)]
    {
        Readiness::new(socket).context("own readiness socket")
    }
}

fn readiness_owner_collision() -> Result<(), Failure> {
    let first = worker()?;
    let second = worker()?;
    let (completed, result) = mpsc::channel();
    first
        .dispatch_send_inline(async move {
            let outcome = async {
                let socket =
                    StdUdpSocket::bind(loopback()).context("bind first readiness source")?;
                socket
                    .set_nonblocking(true)
                    .context("nonblocking first readiness source")?;
                let mut readiness = make_readiness(&socket)?;
                poll_fn(|context| {
                    Poll::Ready(
                        readiness
                            .poll(context)
                            .context("register first readiness source")
                            .map(|_| ()),
                    )
                })
                .await?;
                Ok::<_, Failure>(readiness)
            }
            .await;
            drop(completed.send(outcome));
        })
        .context("dispatch first readiness owner")?;
    let mut foreign = result
        .recv_timeout(GUARD)
        .context("receive foreign readiness")??;
    let (completed, result) = mpsc::channel();
    second
        .dispatch_send_inline(async move {
            let outcome = async {
                let socket =
                    StdUdpSocket::bind(loopback()).context("bind second readiness source")?;
                socket
                    .set_nonblocking(true)
                    .context("nonblocking second readiness source")?;
                let mut local = make_readiness(&socket)?;
                poll_fn(|context| {
                    let registered = local
                        .poll(context)
                        .context("register second readiness source")
                        .map(|_| ());
                    assert!(
                        foreign.poll(context).is_err(),
                        "equal slab keys from another owner must be rejected"
                    );
                    Poll::Ready(registered)
                })
                .await?;
                drop(foreign);
                let retained = core_shard::with_current_reactor(|reactor| reactor.live_sources())
                    .context("second owner reactor")?;
                assert_eq!(
                    retained, 1,
                    "foreign drop must preserve the second owner's source"
                );
                drop(local);
                Ok::<_, Failure>(())
            }
            .await;
            drop(completed.send(outcome));
        })
        .context("dispatch second readiness owner")?;
    result
        .recv_timeout(GUARD)
        .context("readiness owner collision checked")??;
    first
        .shutdown_and_join()
        .context("join first readiness owner")?;
    second
        .shutdown_and_join()
        .context("join second readiness owner")?;
    Ok(())
}
