#![cfg(feature = "replay")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bytes::Bytes;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::method::Method;
use proxima_primitives::pipe::request::Request;
use proxima_recording::replay::keying::{HeaderAllowlist, request_key};

const CALLS: usize = 100_000;

// per-thread so a parked harness thread cannot tick the counter mid-window
thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

// SAFETY: every operation delegates to `System`; the only addition is a bump of a
// const-initialized thread-local `Cell<usize>`, which needs no allocation or drop.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // growth counts here: a spilled scratch vec must fail this test
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

fn llm_post_request() -> Request<Bytes> {
    let mut request = Request::builder()
        .method(Method::Post)
        .path("/v1/responses")
        .payload(Bytes::from_static(
            b"{\"model\":\"gpt-5\",\"input\":\"what is the weather in oslo\"}",
        ))
        .build()
        .unwrap_or_else(|error| panic!("method and path build a request: {error:?}"));
    let pairs = [
        ("city", "oslo"),
        ("units", "metric"),
        ("lang", "nb"),
        ("tag", "a"),
        ("tag", "b"),
    ];
    request.query = pairs
        .iter()
        .map(|(name, value)| {
            (
                Bytes::from_static(name.as_bytes()),
                Bytes::from_static(value.as_bytes()),
            )
        })
        .collect::<Vec<_>>()
        .into();
    request.metadata = HeaderList::from_pairs([
        ("Accept", "application/json"),
        ("User-Agent", "curl/8.7.1"),
        ("X-API-Version", "2026-10-01"),
    ]);
    request
}

#[test]
fn request_key_allocates_nothing_over_one_hundred_thousand_calls() {
    let request = llm_post_request();
    let allowlist = HeaderAllowlist::new(["accept", "x-api-version"])
        .unwrap_or_else(|error| panic!("valid header names: {error:?}"));
    let expected = request_key(&request, &allowlist);

    let before = allocations();
    let mut last = expected;
    for _ in 0..CALLS {
        last = std::hint::black_box(request_key(
            std::hint::black_box(&request),
            std::hint::black_box(&allowlist),
        ));
    }
    let observed = allocations() - before;

    println!("request_key allocations over {CALLS} calls: {observed}");
    assert_eq!(last, expected, "the key is stable across calls");
    assert_eq!(
        observed, 0,
        "request_key must not allocate for <=16 query pairs"
    );
}

#[test]
fn counting_allocator_observes_a_known_allocation() {
    let before = allocations();
    let buffer = std::hint::black_box(vec![0u8; 64]);
    let observed = allocations() - before;

    assert!(
        observed >= 1,
        "control: the counter must see a Vec allocation"
    );
    drop(buffer);
}
