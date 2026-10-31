//! The one-of-N question for the device bus, answered by writing it each way
//! the algebra offers and keeping the result as evidence.
//!
//! (a) `RoutingPipe`: a port becomes an HTTP-shaped request path, every port a
//!     device owns needs its own mount (`{id}` captures a segment, not a range),
//!     and each mount holds an `Arc<dyn>` handle.
//! (b) `FanOut` under a `FanPolicy`: the policy is two associated consts that
//!     never see the item, and `FanOut::Out` is `()`: every sink is invoked and
//!     no reply can come back, so a device read has nowhere to return its value.
//! (c) the bus as shipped is `FirstClaim` in `proxima_vm::kvm`, covered by its
//!     own unit tests next to its definition.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_primitives::pipe::fanout::{FanOut, IgnoreErrors};
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::routing::RoutingPipe;
use proxima_primitives::pipe::SendPipe;

const COM1_BASE: u16 = 0x3f8;
const COM2_BASE: u16 = 0x2f8;
const UART_PORTS: u16 = 8;

struct Named(&'static str);

impl SendPipe for Named {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(&self, _request: Request<Bytes>) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let name = self.0;
        async move { Ok(Response::ok(Bytes::from_static(name.as_bytes()))) }
    }
}

fn port_request(port: u16) -> Request<Bytes> {
    Request::builder().method("GET").path(format!("/io/{port:x}")).build().expect("a port request builds")
}

#[proxima::test]
async fn routing_pipe_needs_one_mount_per_port_and_answers_through_a_response_body() {
    let mut routes = RoutingPipe::new("ports");
    for offset in 0..UART_PORTS {
        routes = routes.route(&format!("/io/{:x}", COM1_BASE + offset), into_handle(Named("com1")));
        routes = routes.route(&format!("/io/{:x}", COM2_BASE + offset), into_handle(Named("com2")));
    }

    let com1 = SendPipe::call(&routes, port_request(COM1_BASE + 2)).await.expect("routes");
    let com2 = SendPipe::call(&routes, port_request(COM2_BASE)).await.expect("routes");
    let rtc = SendPipe::call(&routes, port_request(0x71)).await.expect("routes");

    assert_eq!(&com1.collect_body().await.expect("body")[..], b"com1");
    assert_eq!(&com2.collect_body().await.expect("body")[..], b"com2");
    assert_eq!(rtc.status, 404, "a port no mount names is a 404, not a decline a next device could take");
}

struct Device {
    base: u16,
    invocations: Arc<AtomicUsize>,
}

impl SendPipe for Device {
    type In = u16;
    type Out = u8;
    type Err = Infallible;

    fn call(&self, port: u16) -> impl Future<Output = Result<u8, Infallible>> + Send {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        let owned = port >= self.base && port - self.base < UART_PORTS;
        async move { Ok(if owned { 0x60 } else { 0xff }) }
    }
}

#[proxima::test]
async fn fan_out_invokes_every_device_and_returns_no_reply() {
    let invocations = Arc::new(AtomicUsize::new(0));
    let bus = FanOut::<Device, IgnoreErrors>::ignore_errors(vec![
        Device { base: COM1_BASE, invocations: Arc::clone(&invocations) },
        Device { base: COM2_BASE, invocations: Arc::clone(&invocations) },
    ]);

    let reply: () = SendPipe::call(&bus, COM1_BASE + 5).await.expect("fan out never fails here");

    assert_eq!(reply, (), "FanOut::Out is (): the LSR value COM1 computed is discarded");
    assert_eq!(invocations.load(Ordering::SeqCst), 2, "the non-owning device was invoked too");
}
