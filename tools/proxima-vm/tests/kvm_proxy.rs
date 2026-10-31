//! The exit seam is an ordinary pipe endpoint: an observe pipe interposed
//! between the exit sources and the device bus, without modifying either, sees
//! the whole exit stream of a real run; that recorded stream, fed to a fresh bus
//! with no VM behind it, reproduces the guest's output byte for byte.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::cell::{Cell, RefCell};
use std::future::Future;

use futures::executor::block_on;
use proxima_core::ProximaError;
use proxima_primitives::pipe::Pipe;
use proxima_protocols::process::{ChildRequest, ChildResponse, WriteResponse};
use proxima_vm::kvm::devices::{Channel, Hypercall, Lifecycle, RejectUnmapped};
use proxima_vm::kvm::trap::{End, Reply, RunError, Trap};
use proxima_vm::kvm::{
    Capacities, ElfRun, FirstClaim, GuestSegment, Transports, run_elf_observed,
};

const GUEST_MEMORY_SIZE: u64 = 2 * 1024 * 1024;
const REQUEST_ADDRESS: usize = 0x100;
const WRITE_VERB: u8 = 1;
const OUTPUT_HANDLE: i32 = 1;
const MESSAGE: &[u8] = b"proxy-shape: recorded once, replayed with no vm";

struct OutputDispatcher {
    written: RefCell<Vec<u8>>,
}

impl OutputDispatcher {
    fn new() -> Self {
        Self { written: RefCell::new(Vec::new()) }
    }
}

impl Pipe for OutputDispatcher {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;

    fn call(&self, request: ChildRequest) -> impl Future<Output = Result<ChildResponse, ProximaError>> {
        let response = match request {
            ChildRequest::Write { handle: OUTPUT_HANDLE, bytes } => {
                self.written.borrow_mut().extend_from_slice(&bytes);
                ChildResponse::Write(WriteResponse { bytes_written: bytes.len() as u32 })
            }
            _ => ChildResponse::Close,
        };
        async move { Ok(response) }
    }
}

struct Recorder<'run> {
    seen: &'run RefCell<Vec<Trap>>,
}

impl Pipe for Recorder<'_> {
    type In = Trap;
    type Out = Trap;
    type Err = RunError;

    fn call(&self, trap: Trap) -> impl Future<Output = Result<Trap, RunError>> {
        self.seen.borrow_mut().push(trap.clone());
        async move { Ok(trap) }
    }
}

/// `write(1, MESSAGE)` as one hypercall (`mov dx, verb; mov rdi, request;
/// mov rsi, length; out dx, al`), then the halt hypercall.
fn guest_image() -> Vec<u8> {
    let request = postcard::to_allocvec(&ChildRequest::Write { handle: OUTPUT_HANDLE, bytes: MESSAGE.to_vec() })
        .expect("postcard encodes the write request");
    let mut image = vec![0_u8; REQUEST_ADDRESS];
    let mut code = vec![0x66, 0xba, WRITE_VERB, 0x00];
    code.extend_from_slice(&[0x48, 0xc7, 0xc7]);
    code.extend_from_slice(&(REQUEST_ADDRESS as u32).to_le_bytes());
    code.extend_from_slice(&[0x48, 0xc7, 0xc6]);
    code.extend_from_slice(&(request.len() as u32).to_le_bytes());
    code.push(0xee);
    code.extend_from_slice(&[0x66, 0xba, 0xff, 0xff, 0xee]);
    image[..code.len()].copy_from_slice(&code);
    image.extend_from_slice(&request);
    image
}

#[proxima::test]
async fn an_interposed_observer_records_the_exit_stream_and_replaying_it_reproduces_the_output() {
    let image = guest_image();
    let dispatcher = OutputDispatcher::new();
    let seen = RefCell::new(Vec::new());
    let segment =
        GuestSegment { guest_address: 0, data: &image, memory_size: image.len() as u64, writable: true };

    let outcome = run_elf_observed(
        ElfRun {
            segments: &[segment],
            memory_size: GUEST_MEMORY_SIZE,
            entry: 0,
            boot_argument: 0,
            dispatcher: &dispatcher,
            transports: Transports { console: None, net: None, blk: None },
            max_hypercalls: 4,
            watchdog_millis: 0,
            capacities: Capacities { emitted: 0, console: 0, net: 0, blk: 0 },
        },
        Recorder { seen: &seen },
    );

    assert_eq!(outcome.failure, None, "the live run must succeed");
    assert_eq!(*dispatcher.written.borrow(), MESSAGE, "the live run wrote the message");
    let recorded = seen.into_inner();
    assert_eq!(
        recorded.len() as u64,
        outcome.report.exit_count,
        "the observer saw exactly the exits the run counted"
    );

    let replayed = OutputDispatcher::new();
    let (emitted, sentinel) = (RefCell::new(Channel::with_capacity(0)), Cell::new(0));
    let bus = FirstClaim::new(
        Lifecycle::new(false),
        FirstClaim::new(Hypercall::new(&replayed, &emitted, &sentinel, 4), RejectUnmapped),
    );
    let mut ended = None;
    for trap in recorded {
        if let Reply::End(end) = block_on(bus.call(trap)).expect("a recorded exit replays") {
            ended = Some(end);
        }
    }

    assert_eq!(ended, Some(End::Halted), "the replayed stream ends the same way");
    assert_eq!(*replayed.written.borrow(), MESSAGE, "identical guest output, with no vm");
}
