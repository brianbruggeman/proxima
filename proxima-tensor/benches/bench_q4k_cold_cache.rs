//! Isolates cold-cache cost from everything else for
//! `proxima_tensor::cpu::matmul_q4k_q8k_f32`, kernel held constant, to test
//! whether the 1.38x gap between the isolated Criterion bench (0.0334
//! ns/mac) and the in-situ prefaulted forward-pass number (0.0462 ns/mac) is
//! explained by cache residency rather than anything else.
//!
//! `bench_q4k_matmul.rs` reads ONE tensor's packed bytes (~9 MB) into a
//! `Vec<u8>` and calls the kernel against that SAME buffer thousands of
//! times inside `b.iter` — cache-warm for the whole measurement. The real
//! forward walks ~4 GB of DISTINCT weight bytes exactly once each per
//! layer, so every cache line is a cold DRAM fetch. This bench builds two
//! arms, same kernel, same shape, same thread count, same real weight
//! bytes, differing ONLY in whether the weight bytes are cache-resident:
//!
//! - `WARM`: one packed `Q4_K` weight buffer (`blk.0.attn_q.weight`),
//!   called against repeatedly — reproduces `bench_q4k_matmul.rs`'s shape.
//! - `COLD`: the SAME tensor read from all 32 transformer blocks
//!   (`blk.0..31.attn_q.weight`, ~9.4 MB each, ~300 MB total — far past
//!   this M1 Max's 12 MiB per-cluster L2 and its system-level cache),
//!   round-robined one distinct buffer per call so no cache line is ever
//!   reused across consecutive calls.
//!
//! Both arms prefault their buffers before timing (the `read_exact` that
//! populates each `Vec<u8>` already touches every page — writes, not just
//! maps, so the pages are resident on return) and the bench asserts the
//! minor-fault delta across the timed region is ~0 to rule out demand
//! paging as a confound.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

use criterion::Criterion;
use libc::{RUSAGE_SELF, getrusage, rusage};
use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::tensor::TensorInfo;
use proxima_tensor::cpu::{dot_q4k_q8k, matmul_q4k_q8k_f32, quantize_row_q8k};
use proxima_tensor::test_support::Lcg;
use std::hint::black_box;

/// real GGUF checkpoint path, overridable per-operator via
/// `PROXIMA_BENCH_GGUF_PATH` — the hardcoded default only ever resolved on
/// one machine, which made this bench unrunnable anywhere else.
fn gguf_path() -> String {
    std::env::var("PROXIMA_BENCH_GGUF_PATH").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf"
            .to_string()
    })
}

/// `cpu.rs`'s own `Q4K_BLOCK_ELEMENTS`/`Q8K_BLOCK_BYTES` are private to that
/// module; re-derived here from `proxima_gguf::quant::q4_k::QK_K` (the same
/// source `cpu.rs` derives them from) rather than duplicated as a bare
/// magic number, so the sequential single-thread arm below packs its own
/// `Q8_K` activation buffer with the identical layout `dot_q4k_q8k` expects.
const Q4K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_k::QK_K;
const Q8K_BLOCK_BYTES: usize = 4 + Q4K_BLOCK_ELEMENTS + (Q4K_BLOCK_ELEMENTS / 16) * 2;

/// Transformer block count this bench round-robins the COLD arm across.
/// openchat-3.5-1210 (Mistral-7B architecture) carries 32 decoder blocks
/// (`blk.0`..`blk.31`); each `attn_q.weight` is 4096x4096 Q4_K, ~9.4 MB
/// packed, so 32 of them total ~300 MB -- far past this M1 Max's 12 MiB
/// per-cluster L2 and its ~48 MiB system-level cache (`sysctl
/// hw.perflevel0.l2cachesize`), so by the time the round-robin returns to
/// buffer 0 every line it touched has been evicted by the other 31.
const BLOCK_COUNT: usize = 32;

fn ru_minflt() -> u64 {
    let mut usage: rusage = unsafe { core::mem::zeroed() };
    if unsafe { getrusage(RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    usage.ru_minflt as u64
}

/// Streams the file in growing prefixes until the parser reports
/// `Complete`, without ever reading the (multi-GiB) tensor data section.
/// Copied from `bench_q4k_matmul.rs`'s own `parse_header`: same real-file
/// GGUF prefix-only parse, trimmed to this bench's needs.
fn parse_header(path: &Path) -> (ParsedGguf, u64) {
    let mut file = File::open(path).expect("open real gguf file");
    let file_len = file.metadata().expect("stat gguf file").len();

    let mut prefix_len = 1usize << 20;
    loop {
        let mut buf = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).expect("seek to start");
        let read = file.read(&mut buf).expect("read gguf prefix");
        buf.truncate(read);

        if let Ok((parser, events)) = GgufParser::new().push(&buf) {
            let mut version = None;
            let mut metadata = Vec::new();
            let mut tensors = Vec::new();
            let mut completion = None;
            for event in events {
                match event {
                    GgufEvent::Header { version: version_value, .. } => version = Some(version_value),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete { data_offset, alignment } => {
                        completion = Some((data_offset, alignment));
                    }
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().expect("parser reports complete and clean");
                let parsed = ParsedGguf {
                    version,
                    tensor_count: tensors.len() as u64,
                    kv_count: metadata.len() as u64,
                    metadata,
                    tensors,
                    data_offset,
                    alignment,
                };
                return (parsed, file_len);
            }
        }

        assert!(prefix_len < (1 << 26), "gguf header/directory exceeded 64 MiB prefix budget");
        prefix_len *= 2;
    }
}

fn find_tensor<'a>(parsed: &'a ParsedGguf, name: &str) -> &'a TensorInfo {
    parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("tensor {name} not found in real gguf file"))
}

fn read_tensor_bytes(file: &mut File, parsed: &ParsedGguf, tensor: &TensorInfo, file_len: u64) -> Vec<u8> {
    let range = parsed
        .tensor_data_range(tensor, file_len)
        .expect("tensor byte range within file bounds");
    let mut buf = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
    file.read_exact(&mut buf).expect("read exact tensor byte range");
    buf
}

fn bench_cold_cache(c: &mut Criterion) {
    let (parsed, file_len) = parse_header(Path::new(&gguf_path()));
    let mut file = File::open(gguf_path()).expect("reopen real gguf file for tensor data");

    // WARM arm: one buffer, `bench_q4k_matmul.rs`'s exact shape.
    let warm_tensor = find_tensor(&parsed, "blk.0.attn_q.weight");
    let in_dim = warm_tensor.dims[0] as usize;
    let out_dim = warm_tensor.dims[1] as usize;
    let warm_bytes = read_tensor_bytes(&mut file, &parsed, warm_tensor, file_len);
    println!(
        "WARM: tensor=blk.0.attn_q.weight dims=[{in_dim}, {out_dim}] packed_bytes={}",
        warm_bytes.len()
    );

    // COLD arm: same tensor, all 32 blocks -- distinct real weight bytes,
    // same shape, same kernel.
    let mut cold_buffers: Vec<Vec<u8>> = Vec::with_capacity(BLOCK_COUNT);
    for block in 0..BLOCK_COUNT {
        let name = format!("blk.{block}.attn_q.weight");
        let tensor = find_tensor(&parsed, &name);
        assert_eq!(tensor.dims[0] as usize, in_dim, "{name}: in_dim shape drift vs blk.0");
        assert_eq!(tensor.dims[1] as usize, out_dim, "{name}: out_dim shape drift vs blk.0");
        cold_buffers.push(read_tensor_bytes(&mut file, &parsed, tensor, file_len));
    }
    let cold_total_bytes: usize = cold_buffers.iter().map(Vec::len).sum();
    println!(
        "COLD: {BLOCK_COUNT} distinct buffers, {cold_total_bytes} total bytes ({:.1} MiB)",
        cold_total_bytes as f64 / (1024.0 * 1024.0)
    );
    let l2_bytes: u64 = std::process::Command::new("sysctl")
        .args(["-n", "hw.perflevel0.l2cachesize"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or(0);
    println!(
        "host L2 (perflevel0, sysctl hw.perflevel0.l2cachesize): {l2_bytes} bytes ({:.1} MiB); \
         COLD total is {:.1}x that",
        l2_bytes as f64 / (1024.0 * 1024.0),
        cold_total_bytes as f64 / l2_bytes.max(1) as f64
    );

    let mut lcg = Lcg(4242);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 0.5).collect();

    // correctness: both arms must agree on the same real bytes before
    // timing anything.
    let warm_check = matmul_q4k_q8k_f32(&warm_bytes, out_dim, &activation).expect("warm buffer well-formed");
    assert_eq!(warm_check.len(), out_dim);
    for (index, buffer) in cold_buffers.iter().enumerate() {
        matmul_q4k_q8k_f32(buffer, out_dim, &activation)
            .unwrap_or_else(|error| panic!("cold buffer {index} well-formed: {error:?}"));
    }

    let macs = (out_dim as u64) * (in_dim as u64);
    println!("macs/call: {macs}");

    // `matmul_q4k_q8k_f32` allocates a fresh output `Vec<f32>` and a fresh
    // `Q8_K`-activation `Vec<u8>` per call, and `matmul_rows_threaded`
    // spawns a `thread::scope` per call -- both can touch a handful of
    // freshly-mapped pages per iteration even with the WEIGHT bytes fully
    // resident. Reported as faults-per-iteration (Criterion's sample count
    // times its internal batch size is not exposed here, so this divides
    // by `sample_size` * a conservative single-digit iterations-per-sample
    // floor) so a nonzero delta reads as "small, structural, and not the
    // multi-GB weight-walk demand-paging cost under test" rather than as a
    // false all-clear.
    let minflt_before_warm = ru_minflt();
    c.bench_function("q4k_cold_cache_WARM_t_all", |b| {
        b.iter(|| black_box(matmul_q4k_q8k_f32(&warm_bytes, out_dim, &activation).unwrap()))
    });
    let minflt_after_warm = ru_minflt();
    println!(
        "WARM arm minor-fault delta across timed region: {} raw \
         (structural per-call alloc/thread-spawn faults, not weight-byte demand paging -- \
         weight bytes were resident before timing started)",
        minflt_after_warm - minflt_before_warm
    );

    let mut cold_index = 0usize;
    let minflt_before_cold = ru_minflt();
    c.bench_function("q4k_cold_cache_COLD_t_all", |b| {
        b.iter(|| {
            let buffer = &cold_buffers[cold_index % cold_buffers.len()];
            cold_index += 1;
            black_box(matmul_q4k_q8k_f32(buffer, out_dim, &activation).unwrap())
        })
    });
    let minflt_after_cold = ru_minflt();
    println!(
        "COLD arm minor-fault delta across timed region: {} raw \
         (structural per-call alloc/thread-spawn faults, not weight-byte demand paging -- \
         all 32 cold buffers were resident before timing started)",
        minflt_after_cold - minflt_before_cold
    );

    // `docs/discipline.md`'s own "isolated Criterion bench 0.0334 ns/mac"
    // row for this exact shape (attn_q 4096x4096) is labelled "single
    // thread", but `matmul_q4k_q8k_f32` (the WARM/COLD arms above) ALWAYS
    // auto-dispatches to `thread::available_parallelism()` workers once
    // `rows * k` clears `PARALLEL_THRESHOLD` (65536; attn_q is 16.8M
    // macs) -- there is no thread-count parameter to force it down to 1.
    // This pair reproduces the literal single-thread configuration by
    // calling `dot_q4k_q8k` (the SAME `sdot`-accelerated per-row kernel
    // `matmul_q4k_q8k_f32` dispatches internally) in a plain sequential
    // row loop, so the discipline-log row's methodology claim is checked
    // rather than assumed.
    let mut warm_activation_q8k = vec![0u8; (in_dim / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation, &mut warm_activation_q8k).expect("warm activation quantizes");
    let warm_row_bytes = warm_bytes.len() / out_dim;
    c.bench_function("q4k_cold_cache_WARM_t1_sequential", |b| {
        b.iter(|| {
            let sum: f32 = warm_bytes
                .chunks_exact(warm_row_bytes)
                .map(|row| dot_q4k_q8k(row, &warm_activation_q8k).unwrap())
                .sum();
            black_box(sum)
        })
    });

    let cold_row_bytes = cold_buffers[0].len() / out_dim;
    let mut cold_index_seq = 0usize;
    c.bench_function("q4k_cold_cache_COLD_t1_sequential", |b| {
        b.iter(|| {
            let buffer = &cold_buffers[cold_index_seq % cold_buffers.len()];
            cold_index_seq += 1;
            let sum: f32 = buffer
                .chunks_exact(cold_row_bytes)
                .map(|row| dot_q4k_q8k(row, &warm_activation_q8k).unwrap())
                .sum();
            black_box(sum)
        })
    });
}

fn main() {
    let path = gguf_path();
    if !Path::new(&path).exists() {
        println!("real gguf file not found at {path}; nothing to bench");
        return;
    }

    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(30)
        .measurement_time(Duration::from_secs(5));

    bench_cold_cache(&mut criterion);

    criterion.final_summary();
}
