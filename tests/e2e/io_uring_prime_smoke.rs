//! Prime + io_uring TCP smoke test (P0.2.c of the tokio-elimination plan).
//!
//! Spawns a `PrimeRuntime`, binds the prime-native io_uring `TcpListener` on
//! a worker core, accepts a single connection, echoes bytes back, and shuts
//! down. Correctness only — no perf assertions. Cfg-gated to linux+io_uring
//! per the plan (other platforms fall back to the readiness path).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(
    target_os = "linux",
    feature = "io-uring",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ),
    feature = "runtime-prime-reactor",
))]

use std::sync::Arc;
use std::time::Duration;

use futures::AsyncReadExt;
use futures::AsyncWriteExt;
use futures::channel::oneshot;
use proxima::prime::PrimeRuntime;
use proxima::runtime::prime::os::io_uring::TcpListener;
use proxima::runtime::{CoreId, Runtime};

fn wait_for(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if predicate() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    predicate()
}

#[test]
fn prime_io_uring_listener_accepts_echoes_shuts_down() {
    let runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(1)
            .background_inline()
            .build()
            .expect("build prime runtime"),
    );

    let (addr_tx, addr_rx) = oneshot::channel::<std::net::SocketAddr>();
    let (done_tx, mut done_rx) = oneshot::channel::<()>();

    // bind + accept + echo all on the prime worker so the io_uring
    // CURRENT_URING thread-local is the one polling the listener.
    runtime
        .spawn_on_core(
            CoreId(0),
            Box::pin(async move {
                let mut listener =
                    TcpListener::bind("127.0.0.1:0".parse().expect("addr")).expect("bind listener");
                let local = listener.local_addr().expect("local_addr");
                addr_tx.send(local).expect("send addr");

                let (mut stream, _peer) = listener.accept().await.expect("accept");

                let mut buf = [0_u8; 32];
                let read = stream.read(&mut buf).await.expect("read");
                assert!(read > 0, "client closed before sending");
                stream.write_all(&buf[..read]).await.expect("write echo");
                stream.close().await.expect("close");
                done_tx.send(()).expect("send done");
            }),
        )
        .expect("spawn server on core 0");

    // wait for the listener address, then connect from outside the runtime.
    let listener_addr = futures::executor::block_on(addr_rx).expect("recv addr");

    // tokio client (we're outside any prime worker here; tokio::net is fine
    // as the client side — the server side is the prime-uring path).
    let tokio_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("client tokio runtime");

    tokio_runtime.block_on(async move {
        use tokio::io::{AsyncReadExt as TokioRead, AsyncWriteExt as TokioWrite};
        let mut client = tokio::net::TcpStream::connect(listener_addr)
            .await
            .expect("client connect");
        client
            .write_all(b"prime+uring smoke")
            .await
            .expect("client write");
        client.flush().await.expect("client flush");
        let mut echoed = vec![0_u8; b"prime+uring smoke".len()];
        client
            .read_exact(&mut echoed)
            .await
            .expect("client read echo");
        assert_eq!(&echoed[..], b"prime+uring smoke");
    });

    assert!(
        wait_for(Duration::from_secs(2), || done_rx
            .try_recv()
            .ok()
            .flatten()
            .is_some()),
        "server task did not finish in time",
    );
}
