use std::future::poll_fn;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::thread;
use std::time::Duration;

use futures::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(feature = "http1-native")]
use proxima::Spec;
#[cfg(feature = "http1-native")]
use proxima::runtime::RuntimeSelection;
use proxima::runtime::{RuntimeBackend, installed_runtime, run_with_cores};
use proxima::stream::StreamUpstream;
use proxima::{App, PrimeTcpUpstream};
#[cfg(feature = "http1-native")]
use serde_json::json;

pub const TCP_PAYLOAD: &[u8] = b"GET /windows-port HTTP/1.1\r\nHost: localhost\r\n\r\n";
pub const UDP_PAYLOAD: &[u8] = b"proxima windows udp probe";
const IO_TIMEOUT: Duration = Duration::from_secs(10);

pub fn check_payload(actual: &[u8], expected: &[u8]) -> io::Result<()> {
    if actual != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload mismatch: received {actual:?}, expected {expected:?}"),
        ));
    }
    Ok(())
}

pub fn tcp_exchange() -> io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    let peer = thread::spawn(move || -> io::Result<()> {
        let (mut stream, _) = listener.accept()?;
        stream.set_read_timeout(Some(IO_TIMEOUT))?;
        stream.set_write_timeout(Some(IO_TIMEOUT))?;
        let mut request = vec![0; TCP_PAYLOAD.len()];
        stream.read_exact(&mut request)?;
        check_payload(&request, TCP_PAYLOAD)?;
        stream.write_all(TCP_PAYLOAD)
    });
    run_with_cores(Some(1), None, async move {
        let selection =
            installed_runtime().ok_or_else(|| io::Error::other("no ambient runtime"))?;
        if selection.backend != RuntimeBackend::Prime {
            return Err(io::Error::other("default runtime did not select prime"));
        }
        let app = App::new().map_err(io::Error::other)?;
        if app.runtime().is_none() || app.acceptor_factory().is_none() {
            return Err(io::Error::other(
                "default app has no runtime or TCP factory",
            ));
        }
        let upstream = PrimeTcpUpstream::new(address);
        let mut stream = upstream.connect_future().await?;
        stream.write_all(TCP_PAYLOAD).await?;
        let mut response = vec![0; TCP_PAYLOAD.len()];
        stream.read_exact(&mut response).await?;
        check_payload(&response, TCP_PAYLOAD)
    })
    .map_err(io::Error::other)??;
    peer.join()
        .map_err(|_| io::Error::other("TCP peer panicked"))??;
    Ok(())
}

pub fn udp_exchange() -> io::Result<()> {
    let peer = UdpSocket::bind("127.0.0.1:0")?;
    peer.set_read_timeout(Some(IO_TIMEOUT))?;
    let address = peer.local_addr()?;
    let worker = thread::spawn(move || -> io::Result<()> {
        let mut buffer = [0; 256];
        let (length, sender) = peer.recv_from(&mut buffer)?;
        check_payload(&buffer[..length], UDP_PAYLOAD)?;
        let sent = peer.send_to(UDP_PAYLOAD, sender)?;
        if sent != UDP_PAYLOAD.len() {
            return Err(io::Error::other("incomplete UDP send"));
        }
        Ok(())
    });
    run_with_cores(Some(1), None, async move {
        let app = App::new().map_err(io::Error::other)?;
        let factory = app
            .datagram_factory()
            .ok_or_else(|| io::Error::other("default app has no UDP factory"))?;
        let mut socket = factory.bind("127.0.0.1:0".parse().map_err(io::Error::other)?)?;
        let sent = poll_fn(|context| socket.poll_send_to(context, UDP_PAYLOAD, address)).await?;
        if sent != UDP_PAYLOAD.len() {
            return Err(io::Error::other("incomplete prime UDP send"));
        }
        let mut response = [0; 256];
        let (length, sender) =
            poll_fn(|context| socket.poll_recv_from(context, &mut response)).await?;
        if sender != address {
            return Err(io::Error::other("UDP reply source changed"));
        }
        check_payload(&response[..length], UDP_PAYLOAD)
    })
    .map_err(io::Error::other)??;
    worker
        .join()
        .map_err(|_| io::Error::other("UDP peer panicked"))??;
    Ok(())
}

#[cfg(feature = "http1-native")]
pub async fn multiworker_listener_exchange() -> io::Result<()> {
    let mut app = App::builder()
        .runtime(RuntimeSelection::prime(2).map_err(io::Error::other)?)
        .with_defaults()
        .map_err(io::Error::other)?
        .build()
        .map_err(io::Error::other)?;
    if app.runtime().map(|runtime| runtime.num_cores()) != Some(2) {
        return Err(io::Error::other(
            "configured runtime must expose two workers",
        ));
    }
    let listeners = app.load_full(Spec::Inline(json!({
        "pipe": [{"name": "windows-probe", "synth": {"status": 200, "body": "proxima windows two workers"}}],
        "listen": [{"type": "http", "bind": "127.0.0.1:0", "pipe": "windows-probe"}]
    })))
    .await
    .map_err(io::Error::other)?;
    let address = listeners
        .first()
        .and_then(|listener| listener.bind_addr())
        .ok_or_else(|| io::Error::other("configured listener has no bound address"))?;
    let mut client = TcpStream::connect(address)?;
    client.set_read_timeout(Some(IO_TIMEOUT))?;
    client.set_write_timeout(Some(IO_TIMEOUT))?;
    client
        .write_all(b"GET /windows-port HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    let mut response = Vec::new();
    client.read_to_end(&mut response)?;
    let header_end = response
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .ok_or_else(|| io::Error::other(format!("missing HTTP headers: {response:?}")))?;
    if !response.starts_with(b"HTTP/1.1 200 ") {
        return Err(io::Error::other(format!(
            "unexpected HTTP response: {response:?}"
        )));
    }
    check_payload(&response[header_end + 4..], b"proxima windows two workers")?;
    drop(client);
    for listener in listeners {
        listener.shutdown();
    }
    Ok(())
}

pub async fn incumbent_exchange(corrupt: bool) -> io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let mut sender = TcpStream::connect(listener.local_addr()?)?;
    let (mut receiver, _) = listener.accept()?;
    receiver.set_read_timeout(Some(IO_TIMEOUT))?;
    sender.write_all(TCP_PAYLOAD)?;
    let mut received = vec![0; TCP_PAYLOAD.len()];
    receiver.read_exact(&mut received)?;
    if corrupt {
        received.pop();
    }
    check_payload(&received, TCP_PAYLOAD)?;
    let sender = UdpSocket::bind("127.0.0.1:0")?;
    let receiver = UdpSocket::bind("127.0.0.1:0")?;
    receiver.set_read_timeout(Some(IO_TIMEOUT))?;
    sender.send_to(UDP_PAYLOAD, receiver.local_addr()?)?;
    let mut received = [0; 256];
    let (length, _) = receiver.recv_from(&mut received)?;
    check_payload(&received[..length], UDP_PAYLOAD)?;
    Ok(())
}
