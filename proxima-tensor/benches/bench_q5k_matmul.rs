//! `proxima_tensor::cpu::{matmul_q5k_f32, matmul_q5k_q8k_f32,
//! matmul_q5k_q8k_portable_f32}` vs ggml's `ggml_mul_mat` on a
//! `GGML_TYPE_Q5_K` source, batch-1 decode, at the two weight-matrix shapes
//! the real openchat-3.5-1210 (Mistral architecture) `Q4_K_S` file actually
//! bumps to `Q5_K` per llama.cpp's own mixed-precision quantization recipe:
//! `attn_v` (4096x1024) and `ffn_down` (14336x4096).
//!
//! Both arms read the SAME packed `Q5_K` bytes, sliced directly out of the
//! real GGUF file on disk — no synthetic quantization on either side, no
//! re-quantizing ggml's own output. Mirrors `bench_q4k_matmul.rs`'s
//! structure and correctness-before-timing discipline exactly.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "ggml_ffi.rs"]
mod ggml_ffi;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::raw::c_int;
use std::path::Path;
use std::time::Duration;

use criterion::Criterion;
use ggml_ffi::*;
use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::tensor::TensorInfo;
use proxima_tensor::cpu::matmul_q5k_f32;
#[cfg(feature = "q5k-int8-dot")]
use proxima_tensor::cpu::{matmul_q5k_q8k_f32, matmul_q5k_q8k_portable_f32};
use proxima_tensor::test_support::Lcg;
use std::hint::black_box;

const GGUF_PATH: &str =
    "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

/// Streams the file in growing prefixes until the parser reports
/// `Complete`, without ever reading the (multi-GiB) tensor data section --
/// identical to `bench_q4k_matmul.rs`'s own copy; not shared across bench
/// binaries because criterion benches are each their own crate root.
fn parse_header(path: &Path) -> (ParsedGguf, u64) {
    let mut file = File::open(path).expect("open real gguf file");
    let file_len = file.metadata().expect("stat gguf file").len();

    let mut prefix_len = 1usize << 20; // 1 MiB
    loop {
        let mut buf = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).expect("seek to start");
        let read = file.read(&mut buf).expect("read gguf prefix");
        buf.truncate(read);

        match GgufParser::new().push(&buf) {
            Ok((parser, events)) => {
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
            Err(_) => { /* the truncated prefix parsed partway; grow and retry below */ }
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shape mismatch: {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

struct Plan {
    cplan: ggml_cplan,
    _work: Vec<u8>,
}

unsafe fn make_plan(graph: *mut ggml_cgraph, n_threads: c_int) -> Plan {
    unsafe {
        let mut cplan = ggml_graph_plan(graph, n_threads, std::ptr::null_mut());
        let mut work = vec![0u8; cplan.work_size.max(1)];
        cplan.work_data = work.as_mut_ptr();
        Plan { cplan, _work: work }
    }
}

unsafe fn compute_plan(graph: *mut ggml_cgraph, plan: &mut Plan) {
    unsafe {
        let status = ggml_graph_compute(graph, &mut plan.cplan);
        assert_eq!(status, 0, "ggml_graph_compute failed: {status}");
    }
}

/// One shape's full head-to-head: real packed `Q5_K` bytes feed
/// `matmul_q5k_f32` (dequant-then-fold), `matmul_q5k_q8k_f32`/
/// `matmul_q5k_q8k_portable_f32` (packed int8, feature `q5k-int8-dot`), and
/// a `ggml_mul_mat` graph against a `GGML_TYPE_Q5_K` tensor built from the
/// SAME bytes.
fn bench_shape(c: &mut Criterion, label: &str, tensor_name: &str, seed: u64) {
    let (parsed, file_len) = parse_header(Path::new(GGUF_PATH));
    let tensor = find_tensor(&parsed, tensor_name);
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;
    println!(
        "\n=== {label}: tensor={tensor_name} dims=[{in_dim}, {out_dim}] type={:?} ===",
        tensor.ggml_type
    );
    if tensor.ggml_type != proxima_gguf::types::GgmlType::Q5_K {
        println!(
            "{label} BLOCKED: real file stores this tensor as {:?}, not Q5_K \
             (Q4_K_S's mixed-precision recipe may not bump this tensor on every quantizer version) — skipped, not faked",
            tensor.ggml_type
        );
        return;
    }

    let mut file = File::open(GGUF_PATH).expect("reopen real gguf file for tensor data");
    let weight_bytes = read_tensor_bytes(&mut file, &parsed, tensor, file_len);
    println!("packed weight bytes: {} ({} rows x {} in_dim)", weight_bytes.len(), out_dim, in_dim);

    let mut lcg = Lcg(seed);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 0.5).collect();

    // --- correctness first ---
    let ggml_out = unsafe {
        let ctx = ggml_ctx((weight_bytes.len() + activation.len() * 4) / (1024 * 1024) + 64);
        let weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, in_dim as i64, out_dim as i64);
        std::ptr::copy_nonoverlapping(weight_bytes.as_ptr(), ggml_get_data(weight).cast::<u8>(), weight_bytes.len());
        let vec_tensor = new_f32_1d(ctx, in_dim as i64, &activation);
        let result = ggml_mul_mat(ctx, weight, vec_tensor);
        let graph = build_graph(ctx, result);
        let mut setup_plan = make_plan(graph, 1);
        compute_plan(graph, &mut setup_plan);
        read_f32(result)
    };

    let ours_dequant = matmul_q5k_f32(&weight_bytes, out_dim, &activation).expect("well-formed real q5_k matmul");
    let diff_dequant = max_abs_diff(&ours_dequant, &ggml_out);
    println!("{label} dequant-then-fold max abs diff (ours vs ggml): {diff_dequant:e}");
    // Same band `bench_q4k_matmul.rs` asserts for its own `Q4_K` dequant
    // arm -- a lossy k-quant codec against a real, non-zero-mean weight
    // matrix, dot products thousands of terms wide.
    assert!(diff_dequant < 0.5, "{label} dequant numerical mismatch: {diff_dequant} (fail fast, no timing below this line)");

    #[cfg(feature = "q5k-int8-dot")]
    {
        let ours_packed = matmul_q5k_q8k_f32(&weight_bytes, out_dim, &activation).expect("well-formed packed int8 matmul");
        let diff_packed = max_abs_diff(&ours_packed, &ggml_out);
        println!("{label} packed-int8 max abs diff (ours vs ggml): {diff_packed:e}");
        assert!(
            diff_packed < 0.5,
            "{label} packed-int8 numerical mismatch: {diff_packed} (fail fast, no timing below this line)"
        );

        let ours_portable = matmul_q5k_q8k_portable_f32(&weight_bytes, out_dim, &activation).expect("well-formed portable matmul");
        assert_eq!(ours_packed, ours_portable, "{label}: dispatched and portable packed-int8 arms diverged");

        c.bench_function(&format!("{label}_proxima_matmul_q5k_q8k_dispatched_t1"), |b| {
            b.iter(|| black_box(matmul_q5k_q8k_f32(&weight_bytes, out_dim, &activation).unwrap()))
        });
        c.bench_function(&format!("{label}_proxima_matmul_q5k_q8k_portable_t1"), |b| {
            b.iter(|| black_box(matmul_q5k_q8k_portable_f32(&weight_bytes, out_dim, &activation).unwrap()))
        });
    }

    unsafe {
        let ctx = ggml_ctx((weight_bytes.len() + activation.len() * 4) / (1024 * 1024) + 64);
        let weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, in_dim as i64, out_dim as i64);
        std::ptr::copy_nonoverlapping(weight_bytes.as_ptr(), ggml_get_data(weight).cast::<u8>(), weight_bytes.len());
        let vec_tensor = new_f32_1d(ctx, in_dim as i64, &activation);
        let result = ggml_mul_mat(ctx, weight, vec_tensor);
        let graph = build_graph(ctx, result);

        c.bench_function(&format!("{label}_ggml_q5k_mulmat_t1"), |b| {
            let mut plan = make_plan(graph, 1);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
    }

    c.bench_function(&format!("{label}_proxima_matmul_q5k_f32_dequant_t1"), |b| {
        b.iter(|| black_box(matmul_q5k_f32(&weight_bytes, out_dim, &activation).unwrap()))
    });

    let macs = (out_dim as u64) * (in_dim as u64);
    println!("{label} macs/call: {macs}");
}

unsafe fn ggml_ctx(mem_mb: usize) -> *mut ggml_context {
    let params = ggml_init_params {
        mem_size: (mem_mb + 256) * 1024 * 1024,
        mem_buffer: std::ptr::null_mut(),
        no_alloc: false,
    };
    unsafe { ggml_init(params) }
}

unsafe fn new_f32_1d(ctx: *mut ggml_context, n: i64, data: &[f32]) -> *mut ggml_tensor {
    unsafe {
        let tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        let dst = ggml_get_data_f32(tensor);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        tensor
    }
}

unsafe fn read_f32(tensor: *mut ggml_tensor) -> Vec<f32> {
    unsafe {
        let n = ggml_nelements(tensor) as usize;
        let src = ggml_get_data_f32(tensor);
        std::slice::from_raw_parts(src, n).to_vec()
    }
}

unsafe fn build_graph(ctx: *mut ggml_context, root: *mut ggml_tensor) -> *mut ggml_cgraph {
    unsafe {
        let graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, root);
        graph
    }
}

fn main() {
    if !Path::new(GGUF_PATH).exists() {
        println!("real gguf file not found at {GGUF_PATH}; nothing to bench");
        return;
    }

    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(30)
        .measurement_time(Duration::from_secs(5));

    bench_shape(&mut criterion, "attn_v_4096x1024", "blk.0.attn_v.weight", 203);
    bench_shape(&mut criterion, "ffn_down_14336x4096", "blk.0.ffn_down.weight", 206);

    criterion.final_summary();
}
