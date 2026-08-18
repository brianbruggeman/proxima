#![allow(clippy::expect_used)]
//! Per-call-site itemisation of every heap allocation `evaluate()` performs
//! for the 1024^3 contiguous GEMM (`profile_hot.rs`'s `matmul_program`,
//! reused verbatim). `AllocSite` buckets (`Prepare`/`OutputBuffer`/
//! `ChunkSlices`/`Other`) are too coarse to answer "which of the 65
//! allocations is this" — this wraps the global allocator to capture a real
//! `std::backtrace::Backtrace` per allocation instead, then groups by a
//! short causal chain of frames (skipping known allocator/iterator
//! plumbing), which is what actually names the call site regardless of how
//! deep the crate's own instrumentation reaches. A single frame is not
//! enough: rustc's v0 mangling makes a foreign trait method monomorphised
//! over one of this crate's types (e.g. `<AxisIndex as
//! core::slice::to_vec_in::ConvertVec>::to_vec`) textually indistinguishable
//! from a function this crate actually defines, so one truncated label can
//! misattribute a `Vec<T>::clone` deep in std to the type `T` itself.
//! Debug build only (`cargo run`, not `--release`): release's `lto = "fat"`
//! inlines call sites into each other, collapsing exactly the distinctions
//! this itemisation exists to keep separate. Not part of the crate's public
//! surface; throwaway for this measurement task.

use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use proxima_tensor::{
    DType, Extent, IndexMap, NodeId, Op, ReduceInit, ScalarOp, append, evaluate, map,
};

struct ItemizingAllocator;

static RECORDING: AtomicBool = AtomicBool::new(false);
static RECORDS: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());

std::thread_local! {
    // `Backtrace::force_capture` itself allocates (building its own
    // `Vec<BacktraceFrame>`), which would otherwise re-enter this very
    // `alloc` and capture a backtrace of the backtrace capture, diverging
    // until the stack overflows. This flag makes the itemizer skip any
    // allocation that happens while it is already itemizing one.
    static CAPTURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

unsafe impl GlobalAlloc for ItemizingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if RECORDING.load(Ordering::Relaxed) && !CAPTURING.with(std::cell::Cell::get) {
            // held across both the capture AND the `records.push` below:
            // `Vec::push`'s own reallocation calls back into this `alloc`,
            // and taking `RECORDS`'s lock a second time on the same thread
            // (a plain, non-reentrant `std::sync::Mutex`) would deadlock
            // rather than panic — silent under a `timeout` wrapper.
            CAPTURING.with(|flag| flag.set(true));
            let backtrace = Backtrace::force_capture();
            let site = call_chain(&backtrace);
            let mut records = RECORDS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            records.push((site, layout.size() as u64));
            drop(records);
            CAPTURING.with(|flag| flag.set(false));
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: ItemizingAllocator = ItemizingAllocator;

/// Strips rustc v0 mangling's `[hexhash]` crate/impl disambiguators from a
/// demangled symbol. Can't just split on the first `[`/`]`/`<` — a symbol
/// like `<proxima_tensor[hash]::map::AxisIndex as <[_]>::to_vec_in::
/// ConvertVec>::to_vec` carries a *second*, unrelated bracket pair from the
/// slice-type syntax `[_]`, and a positional split treats it as another
/// hash and truncates mid-path. The disambiguator is always a bare hex
/// string, which real Rust bracket syntax (`[_]`, `[T; N]`) never is —
/// that is the one property that tells them apart.
fn strip_hash_brackets(symbol: &str) -> String {
    let mut output = String::with_capacity(symbol.len());
    let mut index = 0;
    while index < symbol.len() {
        if symbol.as_bytes()[index] == b'['
            && let Some(relative_end) = symbol[index + 1..].find(']')
        {
            let inner = &symbol[index + 1..index + 1 + relative_end];
            if !inner.is_empty() && inner.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                index += relative_end + 2;
                continue;
            }
        }
        let next_char = symbol[index..].chars().next().unwrap_or('\0');
        output.push(next_char);
        index += next_char.len_utf8();
    }
    output
}

struct Frame {
    symbol: String,
    location: Option<String>,
}

/// Parses every `N: 0xADDR - symbol` / `at file:line` stanza out of a
/// rendered backtrace, hash-stripped, in innermost-first order.
fn parse_frames(rendered: &str) -> Vec<Frame> {
    let mut frames = Vec::new();
    let mut lines = rendered.lines().peekable();
    while let Some(line) = lines.next() {
        let Some(dash_position) = line.find(" - ") else {
            continue;
        };
        let raw_symbol = line[dash_position + 3..].trim();
        let symbol = strip_hash_brackets(raw_symbol);
        let location = lines
            .peek()
            .filter(|next_line| next_line.trim_start().starts_with("at "))
            .map(|next_line| next_line.trim().trim_start_matches("at ").to_string());
        frames.push(Frame { symbol, location });
    }
    frames
}

// generic std/alloc bookkeeping that sits between every allocation and its
// real caller (capacity growth, iterator-adapter collect machinery,
// backtrace capture itself) — never the answer to "which of the 65".
const PLUMBING_MARKERS: &[&str] = &[
    "std::backtrace",
    "GlobalAlloc>::alloc",
    "__rust_alloc",
    "alloc::raw_vec::",
    "RawVecInner",
    "RawVec<",
    "with_capacity_in",
    "with_capacity",
    "grow_amortized",
    "grow_exact",
    "grow_one",
    "finish_grow",
    "spec_from_iter",
    "spec_extend",
    "GenericShunt",
    "iter::traits::collect::FromIterator",
    "iter::adapters::try_process",
    "iter::traits::iterator::Iterator>::collect",
    "in_place_collect",
    "try_new_uninit_in",
    "box_new_uninit",
];

// runtime bootstrap above `main` — never informative, and walking into it
// would make an otherwise-short chain misleadingly long.
const BOUNDARY_MARKERS: &[&str] = &[
    "lang_start",
    "__rust_begin_short_backtrace",
    "FnOnce<()>>::call_once",
    "catch_unwind",
];

fn is_plumbing(symbol: &str) -> bool {
    // `alloc::alloc::` can't be a plain substring marker: `alloc::alloc::
    // Global` is also the default `Allocator` type parameter, so it shows
    // up as a *trailing generic argument* (`::<alloc::alloc::Global>`) on
    // almost every collection method, including the real call sites this
    // itemisation exists to find. Only the raw allocator primitives
    // themselves — `alloc::alloc::alloc`, `<alloc::alloc::Global>::...` —
    // have it as the function's *own* path, at the very start of the
    // symbol (after stripping one leading impl-block `<`).
    let is_raw_allocator_frame = symbol
        .strip_prefix('<')
        .unwrap_or(symbol)
        .starts_with("alloc::alloc::");
    is_raw_allocator_frame || PLUMBING_MARKERS.iter().any(|marker| symbol.contains(marker))
}

fn is_boundary(symbol: &str) -> bool {
    BOUNDARY_MARKERS.iter().any(|marker| symbol.contains(marker))
}

const CHAIN_DEPTH: usize = 6;

/// Renders a short causal chain of frames for one allocation, instead of
/// guessing a single label — a chain like `Vec<AxisIndex>::clone <-
/// IndexPattern::clone <- BoundOpBuilder::push` tells a reader this is a
/// clone, where a single truncated frame ("AxisIndex as") would not.
fn call_chain(backtrace: &Backtrace) -> String {
    let rendered = format!("{backtrace:#}");
    if std::env::var("ALLOC_ITEMIZE_DUMP_ONE").is_ok() {
        eprintln!("{rendered}");
        std::process::exit(0);
    }
    let frames = parse_frames(&rendered);
    let start = frames
        .iter()
        .position(|frame| !is_plumbing(&frame.symbol))
        .unwrap_or_else(|| {
            // every frame looked like plumbing: fall back to right after
            // the itemizer's own allocator hook so something is still
            // reported instead of nothing.
            frames
                .iter()
                .position(|frame| frame.symbol.contains("GlobalAlloc>::alloc"))
                .map_or(0, |index| index + 1)
        });
    let chain: Vec<String> = frames[start..]
        .iter()
        .take_while(|frame| !is_boundary(&frame.symbol))
        .take(CHAIN_DEPTH)
        .map(|frame| match &frame.location {
            Some(where_) => format!("{}  ({where_})", frame.symbol),
            None => frame.symbol.clone(),
        })
        .collect();
    if chain.is_empty() {
        return "<no attributable frame found>".to_string();
    }
    chain.join(" <- ")
}

fn matmul_program(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: proxima_tensor::Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    (program, sum)
}

fn main() {
    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32);
    let lhs: Vec<f32> = (0..m * k).map(|value| (value % 13) as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| (value % 7) as f32).collect();

    // one untimed warm-up outside recording so the allocator's own
    // steady-state page tables/free-lists are warm before the itemised run.
    let _ = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("warmup gemm evaluates");

    RECORDS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    RECORDING.store(true, Ordering::Relaxed);
    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("gemm evaluates");
    RECORDING.store(false, Ordering::Relaxed);

    println!(
        "root[0]={} root_len={}",
        evaluated.root()[0],
        evaluated.root().len()
    );

    let records = RECORDS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let total_count = records.len();
    let total_bytes: u64 = records.iter().map(|(_, bytes)| *bytes).sum();
    println!("==== {total_count} allocations, {total_bytes} bytes total ====");

    let mut by_site: HashMap<String, (u64, u64)> = HashMap::new();
    for (site, bytes) in records.iter() {
        let entry = by_site.entry(site.clone()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += bytes;
    }
    let mut rows: Vec<(String, u64, u64)> = by_site
        .into_iter()
        .map(|(site, (count, bytes))| (site, count, bytes))
        .collect();
    rows.sort_by(|left, right| right.2.cmp(&left.2).then(right.1.cmp(&left.1)));
    for (site, count, bytes) in &rows {
        println!("count={count:>3} bytes={bytes:>10} site={site}");
    }
    let attributed_count: u64 = rows.iter().map(|(_, count, _)| *count).sum();
    let attributed_bytes: u64 = rows.iter().map(|(_, _, bytes)| *bytes).sum();
    println!("attributed: count={attributed_count} bytes={attributed_bytes}");
    drop(records);

    // `evaluate_with_scratch` verification: does carrying the output buffer
    // across calls actually remove the 4 MB allocation on the second call,
    // the way `evaluate_parallel_with_scratch` already does? Plain
    // count/bytes here (no backtrace) — the itemisation above already
    // proved the per-site breakdown; this only checks the totals move.
    let mut scratch: Vec<Vec<f32>> = Vec::new();
    for call_number in 1..=3 {
        RECORDS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        RECORDING.store(true, Ordering::Relaxed);
        let evaluated =
            proxima_tensor::evaluate_with_scratch(&program, &[], &[&lhs, &rhs], &[], &mut scratch)
                .expect("scratch gemm evaluates");
        RECORDING.store(false, Ordering::Relaxed);
        let records = RECORDS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let call_count = records.len();
        let call_bytes: u64 = records.iter().map(|(_, bytes)| *bytes).sum();
        drop(records);
        println!(
            "evaluate_with_scratch call {call_number}: allocations={call_count} bytes={call_bytes}"
        );
        evaluated.into_scratch(&mut scratch);
    }
}
