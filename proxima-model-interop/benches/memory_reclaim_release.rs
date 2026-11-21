//! Release and refault over a read-only file mapping.
//!
//! The file is mapped with `Mmap::map`, a read-only file-backed view, the kind
//! `llama_mmap` creates; the bench warms it with one read rather than llama's
//! prefetch flag, against a bench file rather than the server's checkpoint. `ReclaimSettings` supplies the policy values
//! (conflaguration); the timing needs no numeric type beyond the `u128`
//! nanosecond counts and the `i128` signed fault delta printed below.
//!
//! This is a measurement artifact, not a gate. Each round times a warm read, the
//! release, and a read after it, and `fault_ns_median` is the median of the
//! per-round `refault - warm` differences, so the checksum cancels. `advice=`
//! records the code path: `madvise-dontneed` on unix (macOS drops the mapping's
//! pages and refaults them from the unified buffer cache), `no-op` elsewhere.
//!
//! usage: `cargo bench -p proxima-model-interop --bench memory_reclaim_release --features std`

// bench setup is fatal: a failed map or file write must stop the run, not be swallowed.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use memmap2::Mmap;
use proxima_model_interop::memory_reclaim::{ReclaimSettings, release_mapping};

const MAPPING_BYTES: usize = 64 * 1024 * 1024;
const ROUNDS: usize = 5;

static CHUNK: [u8; 1024 * 1024] = [0x5a; 1024 * 1024];

fn main() {
    let settings = ReclaimSettings::builder().build();
    let mut file = tempfile::tempfile().expect("create the bench mapping file");
    file.set_len(MAPPING_BYTES as u64)
        .expect("size the bench mapping file");
    for _ in 0..MAPPING_BYTES / CHUNK.len() {
        file.write_all(&CHUNK).expect("fill the bench mapping file");
    }
    file.flush().expect("flush the bench mapping file");

    // SAFETY: the file is private to this process, never mutated after mapping,
    // and outlives the mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("map the bench file read-only");
    assert_eq!(mapping.len(), MAPPING_BYTES, "mapping covers the whole file");

    let expected = checksum(&mapping);
    let mut warm_nanos = Vec::with_capacity(ROUNDS);
    let mut release_nanos = Vec::with_capacity(ROUNDS);
    let mut refault_nanos = Vec::with_capacity(ROUNDS);
    let mut fault_nanos = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let started = Instant::now();
        let warm = checksum(&mapping);
        warm_nanos.push(started.elapsed().as_nanos());
        assert_eq!(warm, expected, "warm read sees the file");

        let started = Instant::now();
        release_mapping(&mapping).expect("release the mapped pages");
        release_nanos.push(started.elapsed().as_nanos());

        let started = Instant::now();
        let refaulted = checksum(&mapping);
        refault_nanos.push(started.elapsed().as_nanos());
        assert_eq!(refaulted, expected, "refault reads the same bytes");
    }
    fault_nanos.extend(
        refault_nanos
            .iter()
            .zip(&warm_nanos)
            .map(|(refault, warm)| refault.cast_signed() - warm.cast_signed()),
    );

    let rounds = release_nanos.len();
    if rounds != ROUNDS || warm_nanos.len() != rounds || fault_nanos.len() != rounds {
        eprintln!("red: {rounds} timed rounds, expected {ROUNDS}");
        std::process::exit(1);
    }

    println!("rounds={rounds}");
    println!("mapping_bytes={MAPPING_BYTES}");
    println!("pressure_milli={}", settings.pressure_milli);
    println!("idle_secs={}", settings.idle_secs);
    println!(
        "advice={}",
        if cfg!(unix) {
            "madvise-dontneed"
        } else {
            "no-op"
        }
    );
    println!("warm_ns_median={}", median(&mut warm_nanos));
    println!("release_ns_median={}", median(&mut release_nanos));
    println!("refault_ns_median={}", median(&mut refault_nanos));
    println!("fault_ns_median={}", median(&mut fault_nanos));
}

fn checksum(mapping: &[u8]) -> u64 {
    let (words, tail) = mapping.as_chunks::<8>();
    let words_sum = words
        .iter()
        .fold(0_u64, |total, word| total.wrapping_add(u64::from_le_bytes(*word)));
    tail.iter()
        .fold(words_sum, |total, byte| total.wrapping_add(u64::from(*byte)))
        .wrapping_add(mapping.len() as u64)
}

fn median<T: Ord + Copy>(samples: &mut [T]) -> T {
    samples.sort_unstable();
    let middle = samples.len() / 2;
    black_box(samples[middle])
}
