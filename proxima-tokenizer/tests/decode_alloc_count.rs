//! Allocations and frees per `proxima_tokenizer::decode` call on the 84-token
//! stage prompt, on flint's real GGUF vocab (`~/models/llm/flint-gguf/`; no
//! vocab fixture is vendored in this crate, so the test prints `SKIP` loudly
//! when the blob or the prompt file is absent, like `gemma4_llama_oracle.rs`).
//!
//! The counting allocator lives here, in its own integration-test binary, so
//! the process-wide `#[global_allocator]` never touches the crate's other
//! tests. It counts frees as well, which `proxima_test::alloc_count` does not.

#![cfg(feature = "gguf")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use proxima_tokenizer::{Vocab, decode, decode_into, encode_with_bos_eos, gguf::vocab_from_metadata};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
static REALLOCS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREES.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME is set"))
}

fn load_vocab(path: &PathBuf) -> Option<Vocab> {
    let mut file = File::open(path).ok()?;
    let mut header = Vec::new();
    for capacity in [4usize << 20, 16 << 20, 64 << 20] {
        header.resize(capacity, 0);
        file.seek(SeekFrom::Start(0)).expect("seek to gguf start");
        let read = file.read(&mut header).expect("read gguf header region");
        header.truncate(read);
        if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header) {
            return Some(vocab_from_metadata(&parsed).expect("vocab from flint metadata"));
        }
    }
    panic!("gguf metadata region did not fit in 64 MiB of {path:?}");
}

#[test]
fn decode_of_the_84_token_stage_prompt_reports_allocs_and_frees_per_call() {
    let model = home().join("models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf");
    let prompt_path = home().join(".cache/kvgate/proxnum97/wp_84.txt");
    let (Some(vocab), Ok(text)) = (load_vocab(&model), fs::read_to_string(&prompt_path)) else {
        eprintln!("SKIP decode alloc count: {model:?} or {prompt_path:?} absent");
        return;
    };
    let ids = encode_with_bos_eos(text.trim_end_matches('\n'), &vocab, false, false).expect("prompt encodes");
    assert!(!ids.is_empty(), "prompt produced ids");

    let warm = decode(&ids, &vocab).expect("warm decode");
    let (allocs_before, frees_before, reallocs_before) =
        (ALLOCS.load(Ordering::Relaxed), FREES.load(Ordering::Relaxed), REALLOCS.load(Ordering::Relaxed));
    let decoded = decode(&ids, &vocab).expect("measured decode");
    let allocs = ALLOCS.load(Ordering::Relaxed) - allocs_before;
    let reallocs = REALLOCS.load(Ordering::Relaxed) - reallocs_before;
    let frees_with_result_live = FREES.load(Ordering::Relaxed) - frees_before;
    drop(decoded);
    let frees = FREES.load(Ordering::Relaxed) - frees_before;

    let raw = proxima_tokenizer::bpe::decode_ids(&ids, &vocab).expect("raw bytes");
    let marker_replaced = raw != warm.as_bytes();
    eprintln!(
        "decode alloc count: ids={} char_level_bpe={} marker_replaced_stage={} bytes_out={} allocs={} reallocs={} frees_before_result_drop={} frees_after_result_drop={}",
        ids.len(),
        vocab.is_char_level_bpe(),
        marker_replaced,
        warm.len(),
        allocs,
        reallocs,
        frees_with_result_live,
        frees,
    );
    assert!(allocs <= 1, "decode reserves its output once, saw {allocs} allocs");
    assert_eq!(allocs, frees, "every allocation is freed once the result is dropped");

    let mut out = String::new();
    decode_into(&ids, &vocab, &mut out).expect("warming decode_into");
    let (allocs_before, frees_before, reallocs_before) =
        (ALLOCS.load(Ordering::Relaxed), FREES.load(Ordering::Relaxed), REALLOCS.load(Ordering::Relaxed));
    decode_into(&ids, &vocab, &mut out).expect("measured decode_into");
    let warm_allocs = ALLOCS.load(Ordering::Relaxed) - allocs_before;
    let warm_reallocs = REALLOCS.load(Ordering::Relaxed) - reallocs_before;
    let warm_frees = FREES.load(Ordering::Relaxed) - frees_before;
    eprintln!("decode_into warm: allocs={warm_allocs} reallocs={warm_reallocs} frees={warm_frees} decode_allocs={allocs} decode_reallocs={reallocs}");
    assert_eq!(out, warm, "decode_into output equals decode output");
    assert_eq!((warm_allocs, warm_reallocs, warm_frees), (0, 0, 0), "warmed decode_into touches the allocator zero times");
}
