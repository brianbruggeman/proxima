#![cfg(feature = "replay")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bytes::Bytes;
use proxima_primitives::pipe::method::Method;
use proxima_primitives::pipe::request::Request;
use proxima_recording::replay::fsm::{
    AwaitingPresence, BodyEvent, CassetteHeader, CassetteMode, Disposition, DuplicatePolicy,
    FormatVersion, HeaderEvent, HeaderOutcome, LoadOutcome, LoadedEntry, Loading, Millis, OnStale,
    Opened, Policy, Presence, Ready, RecordedAt, ReplayServing, SegmentId, SegmentStamp, Staleness,
    UnixMillis, UpstreamDigest,
};
use proxima_recording::replay::index::{CassetteIndex, IndexBuilder};
use proxima_recording::replay::keying::{
    HeaderAllowlist, KEY_SCHEMA_VERSION, KeyDigest, RequestDigest, request_digest, request_key,
    response_digest,
};

const CALLS: usize = 100_000;
const ENTRIES: usize = 1_000;
const NOW: u64 = 1_791_676_800_000;
const RECORDED_AT: u64 = 1_790_812_800_000;
const THIRTY_DAYS: u64 = 2_592_000_000;

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
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

fn forecast(city: &str) -> Request<Bytes> {
    let mut request = Request::builder()
        .method(Method::Get)
        .path("/v1/forecast")
        .build()
        .unwrap_or_else(|error| panic!("method and path build a request: {error:?}"));
    request.query = vec![(
        Bytes::from_static(b"city"),
        Bytes::copy_from_slice(city.as_bytes()),
    )]
    .into();
    request
}

struct Fixture {
    index: CassetteIndex<u32>,
    serving: ReplayServing,
    probes: Vec<(KeyDigest, RequestDigest)>,
}

fn city_name(position: usize) -> String {
    format!("city-{position:04}")
}

fn loading(policy: Policy, expected: CassetteHeader) -> Loading {
    let reading = match AwaitingPresence::new(policy).on_presence(Presence::Present) {
        Opened::ReadCassette(state) => state,
        other => panic!("a present strict cassette is read, got {other:?}"),
    };
    match reading.on_header(HeaderEvent::Header(expected)) {
        HeaderOutcome::Accepted(state) => state,
        other => panic!("a matching header is accepted, got {other:?}"),
    }
}

fn advance(outcome: LoadOutcome) -> Loading {
    match outcome {
        LoadOutcome::Continue(state) => state,
        other => panic!("loading must continue, got {other:?}"),
    }
}

// the index is built through the public loader path, not a test-only seam
fn fixture(staleness: Staleness) -> Fixture {
    let allowlist = HeaderAllowlist::new(["accept"])
        .unwrap_or_else(|error| panic!("accept is a valid header name: {error:?}"));
    let expected = CassetteHeader {
        format: FormatVersion::from_raw(3),
        key_schema: KEY_SCHEMA_VERSION,
        key_spec: allowlist.digest(),
        upstream: UpstreamDigest::from_raw(0),
    };
    let policy = Policy {
        mode: CassetteMode::Strict,
        duplicates: DuplicatePolicy::RejectDivergent,
        staleness,
        replace: Disposition::Backup,
        expect: expected,
        now: UnixMillis::from_raw(NOW),
    };
    let mut builder = IndexBuilder::<u32>::new();
    let segment = BodyEvent::Segment {
        line: 2,
        stamp: SegmentStamp {
            segment: SegmentId::from_raw(0),
            recorded_at: RecordedAt::At(UnixMillis::from_raw(RECORDED_AT)),
        },
    };
    let mut state = advance(loading(policy, expected).on_event(segment, &mut builder));
    let mut probes = Vec::with_capacity(ENTRIES);
    for position in 0..ENTRIES {
        let request = forecast(&city_name(position));
        let key = request_key(&request, &allowlist);
        let digest = request_digest(&request, &allowlist);
        probes.push((key, digest));
        let event = BodyEvent::Entry {
            line: 3,
            entry: LoadedEntry {
                key,
                request: digest,
                segment: SegmentId::from_raw(0),
                response: response_digest(200, &[b"{\"temp\":12}"]),
            },
            payload: u32::try_from(position).unwrap_or(u32::MAX),
        };
        state = advance(state.on_event(event, &mut builder));
    }
    let serving = match state.on_event(BodyEvent::End, &mut builder) {
        LoadOutcome::Ready(Ready::Replay(serving)) => serving,
        other => panic!("strict end of load must be ready to replay, got {other:?}"),
    };
    Fixture {
        index: builder.seal(),
        serving,
        probes,
    }
}

#[test]
fn lookup_allocates_nothing_over_one_hundred_thousand_calls_on_a_thousand_entry_index() {
    let Fixture {
        index,
        serving,
        probes,
    } = fixture(Staleness::MaxAge {
        max_age: Millis::from_raw(THIRTY_DAYS),
        on_stale: OnStale::Fail,
    });
    assert_eq!(index.len(), ENTRIES, "every distinct city is one entry");

    let before = allocations();
    let mut hits = 0usize;
    for call in 0..CALLS {
        let (key, request) = std::hint::black_box(probes[call % ENTRIES]);
        if std::hint::black_box(&serving)
            .lookup(std::hint::black_box(&index), key, request)
            .is_ok()
        {
            hits += 1;
        }
    }
    let observed = allocations() - before;

    println!("lookup allocations over {CALLS} calls on {ENTRIES} entries: {observed}");
    assert_eq!(hits, CALLS, "every probe was recorded and fresh");
    assert_eq!(observed, 0, "lookup must not allocate");
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
