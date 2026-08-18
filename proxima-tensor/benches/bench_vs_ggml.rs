//! Structural fusion (proxima-tensor) vs a materializing engine (ggml),
//! both arms in the same process under the same criterion harness.
//!
//! ggml is linked as a static library built from a pinned checkout — see
//! `ggml_ffi.rs` for the extern surface and `build.rs` for the link lines.
//! Both engines read the *same* f32 byte buffers for every shared row: the
//! layout mapping (ggml's `ne0`-is-fastest convention vs this crate's
//! row-major-last-axis-fastest convention) is worked out once per row below
//! so no transposition copy is ever needed to make the two comparable.
//!
//! Row order matches the disciplined-component gate: the incumbent's home
//! turf first, then the closest arm we can actually run on both sides, then
//! the mechanism rows (gather-fused-reduce, a deep chain, a locally
//! connected window) where the architectures structurally diverge, then the
//! original control/hypothesis/elementwise trio. No row is skipped silently
//! — a row we cannot run is reported as BLOCKED with the reason.

#[path = "ggml_ffi.rs"]
mod ggml_ffi;

use std::ffi::c_void;
use std::num::NonZeroUsize;
use std::os::raw::c_int;
use std::time::Duration;

use criterion::Criterion;
use std::hint::black_box;
use ggml_ffi::*;
use proxima_tensor::{
    append, evaluate, evaluate_parallel, map, AxisIndex, AxisTerm, DType, Extent, IndexMap, Keep,
    NodeId, Op, Reduce, ReduceInit, ScalarOp,
};

// ---------------------------------------------------------------------
// deterministic data generation (no `rand` dependency: a tiny LCG is all
// this needs, and it keeps the crate's "minimize dependencies" rule intact
// for a dev-only bench).
// ---------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn random_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..n).map(|_| lcg.next_unit() * scale).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shape mismatch: {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------
// ggml plumbing: one context per row (never freed — the process is
// short-lived and every row's arena is a few hundred MB at most).
// ---------------------------------------------------------------------

// +256 MiB flat headroom on every context: `ggml_graph_compute_with_ctx`'s
// multithreaded path (n_threads > 1) allocates a per-call work buffer from
// the SAME bump-allocated arena as the tensors, on top of the graph node
// table `build_graph` already carved out — an exact tensor-bytes sizing
// underestimates this by a small but nonzero amount that only shows up once
// the t8 bench closures actually run.
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

unsafe fn new_f32_2d(ctx: *mut ggml_context, ne0: i64, ne1: i64, data: &[f32]) -> *mut ggml_tensor {
    unsafe {
        let tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        let dst = ggml_get_data_f32(tensor);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        tensor
    }
}

unsafe fn new_i32_1d(ctx: *mut ggml_context, data: &[i32]) -> *mut ggml_tensor {
    unsafe {
        let tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, data.len() as i64);
        let dst = ggml_get_data(tensor).cast::<i32>();
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

unsafe fn quantize(ty: c_int, src: &[f32], nrows: i64, n_per_row: i64) -> Vec<u8> {
    unsafe {
        let cap = src.len() * 8; // generous upper bound, quantized types are always <= f32 size
        let mut dst = vec![0u8; cap];
        let written = ggml_quantize_chunk(ty, src.as_ptr(), dst.as_mut_ptr().cast::<c_void>(), 0, nrows, n_per_row, std::ptr::null());
        dst.truncate(written);
        dst
    }
}

// ggml's context arena is bump-allocated: `ggml_new_graph` carves out a new
// node table on every call and nothing is freed until the whole context is
// freed, so the graph is built exactly ONCE per row (`build_graph`).
//
// `ggml_graph_compute_with_ctx` allocates its multithreaded work buffer from
// that SAME arena on every single call ("the work data is allocated as a
// part of the context", ggml-cpu.h's own doc) — fine once, fatal inside a
// criterion `b.iter()` loop that calls it thousands of times, which is
// exactly what a first pass of this bench hit
// (`ggml_new_object: not enough space in the context's memory pool`). The
// fix is the plan-based API: `ggml_graph_plan` once, a persistent `Vec<u8>`
// work buffer owned by Rust (never touching ctx), then `ggml_graph_compute`
// repeatedly against that one plan.
unsafe fn build_graph(ctx: *mut ggml_context, root: *mut ggml_tensor) -> *mut ggml_cgraph {
    unsafe {
        let graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, root);
        graph
    }
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

// ---------------------------------------------------------------------
// proxima-tensor program builders. Every matmul-shaped row shares one
// convention: `lhs` is row-major [m, k] (k fastest), `rhs_t` is row-major
// [n, k] (k fastest, i.e. the transpose of the conventional [k, n] rhs).
// That is *exactly* ggml's own `ne0`-fastest weight-storage convention, so
// the identical byte buffer feeds both engines with no copy.
// ---------------------------------------------------------------------

fn matmul_program(
    program: &mut Vec<Op>,
    m: u32,
    k: u32,
    n: u32,
) -> (NodeId, NodeId, NodeId) {
    let lhs = append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs_t = append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(n), Extent::Static(k)],
            name: None,
        },
    );
    let product = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs_t, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    (lhs, rhs_t, sum)
}

fn f32_input(program: &mut Vec<Op>, shape: &[Extent]) -> NodeId {
    append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: shape.to_vec(),
            name: None,
        },
    )
}

/// A [`shape=[1]`] operand broadcast over every axis of `iter_rank` — the
/// `terms: []` `AxisIndex` never varies with the iteration coordinate, so
/// this is a scalar constant, not a per-position read.
fn constant_broadcast(_node: NodeId, iter_rank: u16) -> IndexMap {
    IndexMap::Affine(map::IndexPattern {
        iter_rank,
        axes: vec![AxisIndex::default()],
    })
}

fn unary(program: &mut Vec<Op>, body: ScalarOp, operand: NodeId, rank: u16) -> NodeId {
    let axes: Vec<u16> = (0..rank).collect();
    append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands: vec![(operand, IndexMap::Affine(map::projection(rank, &axes)))],
            name: None,
        },
    )
}

fn binary(program: &mut Vec<Op>, body: ScalarOp, left: NodeId, right: NodeId, rank: u16) -> NodeId {
    let axes: Vec<u16> = (0..rank).collect();
    append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands: vec![
                (left, IndexMap::Affine(map::projection(rank, &axes))),
                (right, IndexMap::Affine(map::projection(rank, &axes))),
            ],
            name: None,
        },
    )
}

fn binary_broadcast_const(program: &mut Vec<Op>, body: ScalarOp, left: NodeId, constant: NodeId, rank: u16) -> NodeId {
    let axes: Vec<u16> = (0..rank).collect();
    append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands: vec![
                (left, IndexMap::Affine(map::projection(rank, &axes))),
                (constant, constant_broadcast(constant, rank)),
            ],
            name: None,
        },
    )
}

// ---------------------------------------------------------------------
// row A: HOME TURF (ggml only). Quantized GEMV: llama.cpp decode is a
// sequence of exactly this operation, one per weight matrix per token.
// proxima-tensor has no quantized dtype yet, so our side is BLOCKED —
// reported as such, not faked, not skipped silently.
// ---------------------------------------------------------------------

fn row_a_home_turf_quantized_gemv(c: &mut Criterion) {
    println!("\n=== ROW A (HOME TURF): quantized GEMV, weight [4096x4096], batch=1 ===");
    println!("ggml SHA 2d191b5dee1a591c41ee8a653ce42bfcd9c8716d");
    println!("proxima-tensor: BLOCKED. no quantized dtype exists in this crate (DType is \
        Float32/Int32 only, cpu.rs is explicitly f32-only in v1). This is not attempted, \
        faked, or approximated below. ggml's number alone is reported so the target this \
        architecture has not yet reached is on the record.");

    let (out_dim, in_dim) = (4096usize, 4096usize);
    let weight_f32 = random_vec(1, out_dim * in_dim, 0.05);
    let activation = random_vec(2, in_dim, 0.5);

    for (label, ty) in [("q4_k", GGML_TYPE_Q4_K), ("q8_0", GGML_TYPE_Q8_0)] {
        unsafe {
            let ctx = ggml_ctx(192);
            let quantized_bytes = quantize(ty, &weight_f32, out_dim as i64, in_dim as i64);
            let weight = ggml_new_tensor_2d(ctx, ty, in_dim as i64, out_dim as i64);
            std::ptr::copy_nonoverlapping(
                quantized_bytes.as_ptr(),
                ggml_get_data(weight).cast::<u8>(),
                quantized_bytes.len(),
            );
            let vec_tensor = new_f32_1d(ctx, in_dim as i64, &activation);
            let result = ggml_mul_mat(ctx, weight, vec_tensor);

            let graph = build_graph(ctx, result);
            let mut setup_plan = make_plan(graph, 1);
            compute_plan(graph, &mut setup_plan);
            println!(
                "ggml {label} quantized weight bytes: {} (f32-equivalent would be {})",
                quantized_bytes.len(),
                out_dim * in_dim * 4
            );

            c.bench_function(&format!("row_a_ggml_{label}_gemv_4096x4096_t1"), |b| {
                let mut plan = make_plan(graph, 1);
                b.iter(|| {
                    compute_plan(graph, &mut plan);
                    black_box(());
                })
            });
            c.bench_function(&format!("row_a_ggml_{label}_gemv_4096x4096_t8"), |b| {
                let mut plan = make_plan(graph, 8);
                b.iter(|| {
                    compute_plan(graph, &mut plan);
                    black_box(());
                })
            });
        }
    }
}

// ---------------------------------------------------------------------
// row B: f32 GEMV at the same decode shape — the closest arm we can
// actually run on both sides. Labeled explicitly as an approximation of
// row A: same memory-bound matrix-vector access pattern, not the same
// quant format ggml's kernels are tuned for.
// ---------------------------------------------------------------------

fn row_b_f32_gemv(c: &mut Criterion) {
    println!("\n=== ROW B: f32 GEMV, weight [4096x4096], batch=1 (approximation of row A, not row A) ===");
    let (m, k, n) = (4096u32, 4096u32, 1u32);
    let lhs_data = random_vec(3, (m * k) as usize, 0.05); // weight [out=4096, in=4096]
    let rhs_t_data = random_vec(4, (n * k) as usize, 0.5); // activation, [1,4096]-transposed == [4096]

    let mut program = Vec::new();
    let (lhs, rhs_t, sum) = matmul_program(&mut program, m, k, n);
    let _ = (lhs, rhs_t, sum);

    let proxima_out = evaluate(&program, &[], &[&lhs_data, &rhs_t_data], &[]).expect("row b evaluates");
    let shapes_b = proxima_tensor::infer(&program, &[]).expect("row b infers");
    let resolved_b = proxima_tensor::bind(&program, &shapes_b, &[]).expect("row b binds");
    println!(
        "materialized tensors: proxima {} (the product never materializes, fused into the \
         reduce), ggml 1 (mul_mat is one kernel call — this row is where both engines already \
         fuse the multiply-then-reduce, by different means: ours structurally, ggml's via a \
         hand-written kernel)",
        resolved_b.len()
    );

    unsafe {
        let ctx = ggml_ctx(256);
        let weight = new_f32_2d(ctx, k as i64, m as i64, &lhs_data);
        let activation = new_f32_1d(ctx, k as i64, &rhs_t_data);
        // ggml_mul_mat(A, B) returns ne0=A.ne1, ne1=B.ne1 (see ggml.h's own
        // comment on the function) — B (the RHS-transpose role) must be the
        // FIRST argument for the result's flat layout to land in our
        // row-major [m, n] order (n fast). Passing (weight, activation)
        // would produce a transposed flat buffer whenever m != n; here m=n=1
        // is degenerate so it happens not to matter, but the order is fixed
        // to match row F/G's real (non-degenerate) shapes.
        let result = ggml_mul_mat(ctx, activation, weight);
        let graph = build_graph(ctx, result);
        let mut setup_plan = make_plan(graph, 1);
        compute_plan(graph, &mut setup_plan);
        let ggml_out = read_f32(result);

        let diff = max_abs_diff(proxima_out.root(), &ggml_out);
        println!("row B max abs diff (ggml vs proxima evaluate): {diff:e}");
        assert!(diff < 5e-3, "row B numerical mismatch: {diff}");

        let workers = NonZeroUsize::new(8).unwrap();
        c.bench_function("row_b_ggml_f32_gemv_4096x4096_t1", |b| {
            let mut plan = make_plan(graph, 1);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        c.bench_function("row_b_ggml_f32_gemv_4096x4096_t8", |b| {
            let mut plan = make_plan(graph, 8);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        c.bench_function("row_b_proxima_f32_gemv_4096x4096_evaluate", |b| {
            b.iter(|| black_box(evaluate(&program, &[], &[&lhs_data, &rhs_t_data], &[]).unwrap()))
        });
        c.bench_function("row_b_proxima_f32_gemv_4096x4096_evaluate_parallel_w8", |b| {
            b.iter(|| {
                black_box(
                    evaluate_parallel(&program, &[], &[&lhs_data, &rhs_t_data], &[], workers).unwrap(),
                )
            })
        });
    }
}

// ---------------------------------------------------------------------
// row C: gather feeding a reduction, fused. Embedding lookup into a
// weighted reduce: ggml materializes the gathered [dim, seq] tensor (and
// its post-multiply tensor) before summing; proxima-tensor's bind() folds
// gather->multiply->reduce into ONE BoundOp (proven by
// `a_gather_fused_into_a_fold_matches_a_hand_written_embedding_matmul_reference`
// in cpu.rs's own test module — this row is that same shape, benched).
// ---------------------------------------------------------------------

fn row_c_gather_fused_reduce(c: &mut Criterion) {
    println!("\n=== ROW C: gather -> scale -> reduce, fused (table [50000x512], 4096 indices) ===");
    let (vocab, dim, seq) = (50_000u32, 512u32, 4096u32);
    let table_data = random_vec(5, (vocab * dim) as usize, 0.1);
    let weight_data = random_vec(6, dim as usize, 0.2); // the "scale" — one f32 per embedding dim
    let ids_u32: Vec<u32> = {
        let mut lcg = Lcg(7);
        (0..seq)
            .map(|_| ((lcg.next_unit() * 0.5 + 0.5) * (vocab - 1) as f32) as u32)
            .collect()
    };
    let ids_f32: Vec<f32> = ids_u32.iter().map(|&value| value as f32).collect();
    let ids_i32: Vec<i32> = ids_u32.iter().map(|&value| value as i32).collect();

    let mut program = Vec::new();
    let table = f32_input(&mut program, &[Extent::Static(vocab), Extent::Static(dim)]);
    let ids = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(seq)],
            name: None,
        },
    );
    let weight = f32_input(&mut program, &[Extent::Static(dim)]);

    // iteration space is (i = index/seq position, d = embedding dim, contracted).
    // rank 2, not 3: there is no separate output-feature axis in this row (the
    // "weight" is a per-dim scale, not a [dim, out] matrix) so a third axis
    // would appear in no operand's index map and shape inference would reject
    // it as unconstrained.
    let gather_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: map::IndexPattern {
            iter_rank: 2,
            axes: vec![
                AxisIndex::default(),
                AxisIndex { terms: vec![AxisTerm::projection(1)], offset: 0 },
            ],
        },
        gathered_dim: 0,
    };
    let weight_map = IndexMap::Affine(map::projection(2, &[1]));
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(table, gather_map), (weight, weight_map)],
            name: None,
        },
    );
    let reduced = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("gather_weighted_reduce".into()),
        }),
    );
    let _ = reduced;

    let shapes = proxima_tensor::infer(&program, &[]).expect("row c infers");
    let resolved = proxima_tensor::bind(&program, &shapes, &[]).expect("row c binds");
    println!(
        "proxima BoundOp count for gather->scale->reduce: {} (1 means gather+scale never materialize)",
        resolved.len()
    );

    let proxima_out = evaluate(&program, &[], &[&table_data, &ids_f32, &weight_data], &[])
        .expect("row c evaluates");

    unsafe {
        let ctx = ggml_ctx(512);
        let table_tensor = new_f32_2d(ctx, dim as i64, vocab as i64, &table_data);
        let ids_tensor = new_i32_1d(ctx, &ids_i32);
        let weight_tensor = new_f32_1d(ctx, dim as i64, &weight_data);
        let gathered = ggml_get_rows(ctx, table_tensor, ids_tensor);
        let scaled = ggml_mul(ctx, gathered, weight_tensor);
        let summed = ggml_sum_rows(ctx, scaled);
        let graph = build_graph(ctx, summed);
        let mut setup_plan = make_plan(graph, 1);
        compute_plan(graph, &mut setup_plan);
        let ggml_out = read_f32(summed);

        let diff = max_abs_diff(proxima_out.root(), &ggml_out);
        println!("row C max abs diff (ggml vs proxima evaluate): {diff:e}");
        println!(
            "ggml materialized intermediates: get_rows [{}x{}]={} bytes, mul [same]={} bytes, sum_rows output={} bytes = 3 buffers",
            dim, seq, dim as usize * seq as usize * 4, dim as usize * seq as usize * 4, seq as usize * 4
        );
        println!("proxima materialized buffers for this chain: 1 (the final reduce output, {} bytes)", seq as usize * 4);
        assert!(diff < 1e-2, "row C numerical mismatch: {diff}");

        let workers = NonZeroUsize::new(8).unwrap();
        c.bench_function("row_c_ggml_gather_scale_sumrows_t1", |b| {
            let mut plan = make_plan(graph, 1);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        c.bench_function("row_c_ggml_gather_scale_sumrows_t8", |b| {
            let mut plan = make_plan(graph, 8);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        c.bench_function("row_c_proxima_gather_fused_reduce_evaluate", |b| {
            b.iter(|| black_box(evaluate(&program, &[], &[&table_data, &ids_f32, &weight_data], &[]).unwrap()))
        });
        c.bench_function("row_c_proxima_gather_fused_reduce_evaluate_parallel_w8", |b| {
            b.iter(|| {
                black_box(
                    evaluate_parallel(&program, &[], &[&table_data, &ids_f32, &weight_data], &[], workers)
                        .unwrap(),
                )
            })
        });
    }
}

// ---------------------------------------------------------------------
// row D: a deep, unhandwritten fusion chain. 9 elementwise ops (unary,
// binary, and one broadcast) feeding a final reduce over [4096, 64].
// IMPORTANT FINDING (see bind.rs's own doc, quoted in the report): this
// crate's fusion is Reduce-absorbs-its-immediate-Elementwise-producer ONLY
// — an Elementwise consuming another Elementwise always materializes the
// producer (`BoundOpBuilder::push`'s `materialize_if_held` call for every
// Elementwise operand). So of these 9 ops, only the LAST is folded into
// the reduce; the other 8 materialize on both engines. That is reported
// here, not hidden — it is the honest reading of "fusion at arbitrary
// depth" for a pure elementwise prefix.
// ---------------------------------------------------------------------

fn row_d_deep_chain(c: &mut Criterion) {
    println!("\n=== ROW D: 9-op elementwise chain + final fused reduce, [4096x64] ===");
    let (rows, width) = (4096u32, 64u32);
    let x_data = random_vec(8, (rows * width) as usize, 0.2);
    let bias_data = random_vec(9, width as usize, 0.1);

    let mut program = Vec::new();
    let x = f32_input(&mut program, &[Extent::Static(rows), Extent::Static(width)]);
    let bias = f32_input(&mut program, &[Extent::Static(width)]);

    let b = binary(&mut program, ScalarOp::Add, x, x, 2);
    let cc = binary(&mut program, ScalarOp::Multiply, b, b, 2);
    let d = unary(&mut program, ScalarOp::SquareRoot, cc, 2);
    let e = unary(&mut program, ScalarOp::Negate, d, 2);
    let f = unary(&mut program, ScalarOp::Exponential, e, 2);
    let g = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (f, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (bias, IndexMap::Affine(map::projection(2, &[1]))),
            ],
            name: None,
        },
    );
    let h = binary(&mut program, ScalarOp::Divide, x, g, 2);
    let i = binary(&mut program, ScalarOp::Multiply, h, x, 2);
    let j = unary(&mut program, ScalarOp::Tanh, i, 2);

    let reduced = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: j,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("deep_chain_reduce".into()),
        }),
    );
    let _ = reduced;

    let shapes = proxima_tensor::infer(&program, &[]).expect("row d infers");
    let resolved = proxima_tensor::bind(&program, &shapes, &[]).expect("row d binds");
    println!(
        "proxima BoundOp count for the 9-op chain + reduce: {} (expect 9: 8 elementwise ops \
         each materialize their own buffer — only op 9, the reduce's direct producer, is \
         absorbed and never materializes). ggml materializes all 9 of its op nodes (add, mul, \
         sqrt, neg, exp, add, div, mul, tanh) plus the sum_rows output = 10 tensors, since ggml \
         has no cross-op fusion at all outside a hand-written single-kernel op.",
        resolved.len()
    );

    let proxima_out = evaluate(&program, &[], &[&x_data, &bias_data], &[]).expect("row d evaluates");

    unsafe {
        let ctx = ggml_ctx(128);
        let xt = new_f32_2d(ctx, width as i64, rows as i64, &x_data);
        let biast = new_f32_1d(ctx, width as i64, &bias_data);
        let bt = ggml_add(ctx, xt, xt);
        let ct = ggml_mul(ctx, bt, bt);
        let dt = ggml_sqrt(ctx, ct);
        let et = ggml_neg(ctx, dt);
        let ft = ggml_exp(ctx, et);
        let gt = ggml_add(ctx, ft, biast);
        let ht = ggml_div(ctx, xt, gt);
        let it = ggml_mul(ctx, ht, xt);
        let jt = ggml_tanh(ctx, it);
        let summed = ggml_sum_rows(ctx, jt);
        let graph = build_graph(ctx, summed);
        let mut setup_plan = make_plan(graph, 1);
        compute_plan(graph, &mut setup_plan);
        let ggml_out = read_f32(summed);

        let diff = max_abs_diff(proxima_out.root(), &ggml_out);
        println!("row D max abs diff (ggml vs proxima evaluate): {diff:e}");
        assert!(diff < 1e-1, "row D numerical mismatch: {diff}");

        let workers = NonZeroUsize::new(8).unwrap();
        c.bench_function("row_d_ggml_deep_chain_t1", |b| {
            let mut plan = make_plan(graph, 1);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        c.bench_function("row_d_ggml_deep_chain_t8", |b| {
            let mut plan = make_plan(graph, 8);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        c.bench_function("row_d_proxima_deep_chain_evaluate", |b| {
            b.iter(|| black_box(evaluate(&program, &[], &[&x_data, &bias_data], &[]).unwrap()))
        });
        c.bench_function("row_d_proxima_deep_chain_evaluate_parallel_w8", |b| {
            b.iter(|| {
                black_box(evaluate_parallel(&program, &[], &[&x_data, &bias_data], &[], workers).unwrap())
            })
        });
    }
}

// ---------------------------------------------------------------------
// row E: locally connected 1D window (stride 2, dilation 2), per-output-
// position kernel. proxima-tensor only, per bind's own reduce-fusion rule.
// ggml has no unshared-weight conv primitive: `ggml_conv_1d` and friends
// share ONE kernel across every output position. Expressing this in ggml
// would need either a per-position loop of 2048 separate small matmuls
// (not a fair single-kernel comparison) or new C. BLOCKED for ggml, per
// gate 13's own allowance — reported as such, not substituted.
// ---------------------------------------------------------------------

fn row_e_locally_connected_window(c: &mut Criterion) {
    println!("\n=== ROW E: locally connected windowed reduce, stride=2 dilation=2 (proxima only) ===");
    println!("ggml: BLOCKED. no unshared-weight conv primitive exists in ggml's op list; \
        every ggml conv op shares one kernel across all output positions. A per-position loop \
        of matmuls would not be a fair single-kernel comparison and is not attempted.");

    let (h, r, stride, dilation) = (2048u32, 3u32, 2u32, 2u32);
    let signal_len = (h - 1) * stride + (r - 1) * dilation + 1;
    let kernel_data = random_vec(10, (h * r) as usize, 0.3);
    let signal_data = random_vec(11, signal_len as usize, 0.4);

    let mut program = Vec::new();
    let kernel = f32_input(&mut program, &[Extent::Static(h), Extent::Static(r)]);
    let signal = f32_input(&mut program, &[Extent::Static(signal_len)]);
    let window = IndexMap::Affine(map::affine(
        2,
        &[(&[AxisTerm::scaled(0, stride as i32), AxisTerm::scaled(1, dilation as i32)], 0)],
    ));
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (kernel, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (signal, window),
            ],
            name: None,
        },
    );
    let reduced = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("locally_connected".into()),
        }),
    );
    let _ = reduced;

    let shapes = proxima_tensor::infer(&program, &[]).expect("row e infers");
    let resolved = proxima_tensor::bind(&program, &shapes, &[]).expect("row e binds");
    println!(
        "proxima BoundOp count: {} (the two-term window*multiply folds directly into the reduce)",
        resolved.len()
    );

    let workers = NonZeroUsize::new(8).unwrap();
    c.bench_function("row_e_proxima_locally_connected_evaluate", |b| {
        b.iter(|| black_box(evaluate(&program, &[], &[&kernel_data, &signal_data], &[]).unwrap()))
    });
    c.bench_function("row_e_proxima_locally_connected_evaluate_parallel_w8", |b| {
        b.iter(|| {
            black_box(
                evaluate_parallel(&program, &[], &[&kernel_data, &signal_data], &[], workers).unwrap(),
            )
        })
    });
}

// ---------------------------------------------------------------------
// row F: CONTROL. bare GEMM, everyone near the ALU ceiling. Parity or a
// loss here is the expected, honest result.
// ---------------------------------------------------------------------

fn row_f_control_bare_gemm(c: &mut Criterion) {
    println!("\n=== ROW F (CONTROL): bare GEMM, square, f32, 512/1024/2048 ===");
    // 2048 included per definitive-measurement run. Single-threaded proxima
    // `evaluate` at 2048 is skipped below under its own budget guard
    // (`if size <= 1024`) — ggml (t1/t8) and proxima evaluate_parallel run at
    // every size including 2048.
    for size in [512u32, 1024, 2048] {
        let (m, k, n) = (size, size, size);
        let lhs_data = random_vec(20 + u64::from(size), (m * k) as usize, 0.05);
        let rhs_t_data = random_vec(30 + u64::from(size), (n * k) as usize, 0.05);

        let mut program = Vec::new();
        let (_lhs, _rhs_t, _sum) = matmul_program(&mut program, m, k, n);
        let proxima_out = evaluate(&program, &[], &[&lhs_data, &rhs_t_data], &[]).expect("gemm evaluates");

        let mem_mb = (m as usize * k as usize * 4 + n as usize * k as usize * 4 + m as usize * n as usize * 4)
            / (1024 * 1024)
            + 64;
        unsafe {
            let ctx = ggml_ctx(mem_mb);
            let a = new_f32_2d(ctx, k as i64, m as i64, &lhs_data);
            let bt = new_f32_2d(ctx, k as i64, n as i64, &rhs_t_data);
            // see row B's comment: B-role (rhs transpose, `bt`) must be the
            // first argument so the result's flat layout matches our
            // row-major [m, n] (n fast) convention.
            let result = ggml_mul_mat(ctx, bt, a);
            let graph = build_graph(ctx, result);
            let mut setup_plan = make_plan(graph, 1);
            compute_plan(graph, &mut setup_plan);
            let ggml_out = read_f32(result);

            let diff = max_abs_diff(proxima_out.root(), &ggml_out);
            println!("row F size={size} max abs diff: {diff:e}");
            println!(
                "row F size={size} materialized tensors: proxima 1 (fused reduce, product \
                 never materializes), ggml 1 (mul_mat is one kernel) — both sides already fuse \
                 the multiply-then-reduce here, so this row's expected honest result is parity \
                 or a ggml win on raw ALU throughput, not a fusion win"
            );
            assert!(diff < 5e-2, "row F ({size}) numerical mismatch: {diff}");

            let workers = NonZeroUsize::new(8).unwrap();
            c.bench_function(&format!("row_f_ggml_gemm_{size}_t1"), |b| {
                let mut plan = make_plan(graph, 1);
                b.iter(|| {
                    compute_plan(graph, &mut plan);
                    black_box(());
                })
            });
            c.bench_function(&format!("row_f_ggml_gemm_{size}_t8"), |b| {
                let mut plan = make_plan(graph, 8);
                b.iter(|| {
                    compute_plan(graph, &mut plan);
                    black_box(());
                })
            });
            #[cfg(target_arch = "aarch64")]
            let tile_counters_before = proxima_tensor::cpu::neon_tile_counters();
            #[cfg(target_arch = "aarch64")]
            let row_remainder_before = proxima_tensor::cpu::neon_tile_row_remainder_invocations();
            if size <= 1024 {
                c.bench_function(&format!("row_f_proxima_gemm_{size}_evaluate"), |b| {
                    b.iter(|| black_box(evaluate(&program, &[], &[&lhs_data, &rhs_t_data], &[]).unwrap()))
                });
            } else {
                println!(
                    "row F size={size}: proxima single-threaded `evaluate` SKIPPED under the \
                     90-minute measurement budget. The 1024^3 case (see row_f_proxima_gemm_1024_evaluate \
                     above) already measured ~5s/call single-threaded; 2048^3 is 8x the FLOPs of \
                     1024^3, so 10 samples would cost several minutes on top of everything else \
                     still queued. evaluate_parallel and ggml both ran to completion at every \
                     size including this one — only the single-threaded proxima arm at the \
                     largest size is missing, and it is missing because of budget, not because \
                     it could not run."
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                let (gate_after, invocations_after, fallback_after) = proxima_tensor::cpu::neon_tile_counters();
                let (gate_before, invocations_before, fallback_before) = tile_counters_before;
                let row_remainder_after = proxima_tensor::cpu::neon_tile_row_remainder_invocations();
                println!(
                    "row F size={size} neon_tile delta (single-threaded evaluate only): \
                     gate_passes={} invocations={} row_remainder_invocations={} fallback_elements={}",
                    gate_after - gate_before,
                    invocations_after - invocations_before,
                    row_remainder_after - row_remainder_before,
                    fallback_after - fallback_before
                );
            }
            #[cfg(target_arch = "aarch64")]
            let tile_counters_before_parallel = proxima_tensor::cpu::neon_tile_counters();
            #[cfg(target_arch = "aarch64")]
            let row_remainder_before_parallel = proxima_tensor::cpu::neon_tile_row_remainder_invocations();
            c.bench_function(&format!("row_f_proxima_gemm_{size}_evaluate_parallel_w8"), |b| {
                b.iter(|| {
                    black_box(
                        evaluate_parallel(&program, &[], &[&lhs_data, &rhs_t_data], &[], workers).unwrap(),
                    )
                })
            });
            #[cfg(target_arch = "aarch64")]
            {
                let (gate_after, invocations_after, fallback_after) = proxima_tensor::cpu::neon_tile_counters();
                let (gate_before, invocations_before, fallback_before) = tile_counters_before_parallel;
                let row_remainder_after = proxima_tensor::cpu::neon_tile_row_remainder_invocations();
                println!(
                    "row F size={size} neon_tile delta (evaluate_parallel_w8 only): \
                     gate_passes={} invocations={} row_remainder_invocations={} fallback_elements={}",
                    gate_after - gate_before,
                    invocations_after - invocations_before,
                    row_remainder_after - row_remainder_before_parallel,
                    fallback_after - fallback_before
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// row G: HYPOTHESIS. rmsnorm -> matmul -> silu -> residual add, an MLP
// block shape (seq=512, d_model=2048, d_ff=8192). Only the matmul step
// fuses (its product never materializes); rmsnorm's and silu's elementwise
// chains materialize on both engines per row D's finding — reported below,
// not hidden.
// ---------------------------------------------------------------------

fn row_g_mlp_chain(c: &mut Criterion) {
    println!("\n=== ROW G (HYPOTHESIS): rmsnorm -> matmul -> silu -> residual add, seq=512 d_model=2048 d_ff=8192 ===");
    let (seq, d_model, d_ff) = (512u32, 2048u32, 8192u32);
    let eps = 1e-5f32;

    let x_data = random_vec(40, (seq * d_model) as usize, 0.3);
    let weight_t_data = random_vec(41, (d_ff * d_model) as usize, 0.02); // stored [d_ff, d_model], d_model fastest
    let residual_data = random_vec(42, (seq * d_ff) as usize, 0.1);
    let dmodel_const = [d_model as f32];
    let eps_const = [eps];
    let one_const = [1.0f32];

    let mut program = Vec::new();
    let x = f32_input(&mut program, &[Extent::Static(seq), Extent::Static(d_model)]);
    let dmodel_node = f32_input(&mut program, &[Extent::Static(1)]);
    let eps_node = f32_input(&mut program, &[Extent::Static(1)]);

    let sq = binary(&mut program, ScalarOp::Multiply, x, x, 2);
    let sumsq = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: sq,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("rmsnorm_sumsq".into()),
        }),
    );
    let meansq = binary_broadcast_const(&mut program, ScalarOp::Divide, sumsq, dmodel_node, 1);
    let with_eps = binary_broadcast_const(&mut program, ScalarOp::Add, meansq, eps_node, 1);
    let rms = unary(&mut program, ScalarOp::SquareRoot, with_eps, 1);
    let normed = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Divide,
            operands: vec![
                (x, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (rms, IndexMap::Affine(map::projection(2, &[0]))),
            ],
            name: None,
        },
    );

    let weight_t = f32_input(&mut program, &[Extent::Static(d_ff), Extent::Static(d_model)]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (normed, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (weight_t, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let hidden = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("mlp_up_proj".into()),
        }),
    );

    let one_node = f32_input(&mut program, &[Extent::Static(1)]);
    let neg = unary(&mut program, ScalarOp::Negate, hidden, 2);
    let expv = unary(&mut program, ScalarOp::Exponential, neg, 2);
    let plus1 = binary_broadcast_const(&mut program, ScalarOp::Add, expv, one_node, 2);
    let sigmoid = unary(&mut program, ScalarOp::Reciprocal, plus1, 2);
    let act = binary(&mut program, ScalarOp::Multiply, hidden, sigmoid, 2);

    let residual = f32_input(&mut program, &[Extent::Static(seq), Extent::Static(d_ff)]);
    let out = binary(&mut program, ScalarOp::Add, act, residual, 2);
    let _ = out;

    let shapes = proxima_tensor::infer(&program, &[]).expect("row g infers");
    let resolved = proxima_tensor::bind(&program, &shapes, &[]).expect("row g binds");
    println!(
        "proxima BoundOp count for the full MLP chain: {} (only the up-projection's product \
         is absorbed into its reduce; rmsnorm's and silu's elementwise steps each materialize)",
        resolved.len()
    );

    let blocks: Vec<&[f32]> = vec![
        &x_data,
        &dmodel_const,
        &eps_const,
        &weight_t_data,
        &one_const,
        &residual_data,
    ];
    let proxima_out = evaluate(&program, &[], &blocks, &[]).expect("row g evaluates");

    unsafe {
        let ctx = ggml_ctx(1024);
        let xt = new_f32_2d(ctx, d_model as i64, seq as i64, &x_data);
        let normed_t = ggml_rms_norm(ctx, xt, eps);
        let weight_tensor = new_f32_2d(ctx, d_model as i64, d_ff as i64, &weight_t_data);
        let hidden_t = ggml_mul_mat(ctx, weight_tensor, normed_t);
        let act_t = ggml_silu(ctx, hidden_t);
        let residual_t = new_f32_2d(ctx, d_ff as i64, seq as i64, &residual_data);
        let out_t = ggml_add(ctx, act_t, residual_t);
        let graph = build_graph(ctx, out_t);
        let mut setup_plan = make_plan(graph, 1);
        compute_plan(graph, &mut setup_plan);
        let ggml_out = read_f32(out_t);

        let diff = max_abs_diff(proxima_out.root(), &ggml_out);
        println!("row G max abs diff (ggml vs proxima evaluate): {diff:e}");
        assert!(diff < 5e-2, "row G numerical mismatch: {diff}");

        let workers = NonZeroUsize::new(8).unwrap();
        println!(
            "ggml materialized tensors for the MLP chain: 4 (rms_norm, mul_mat, silu, add — \
             each is one hand-written kernel; no cross-op fusion between them either)"
        );

        c.bench_function("row_g_ggml_mlp_chain_t1", |b| {
            let mut plan = make_plan(graph, 1);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        c.bench_function("row_g_ggml_mlp_chain_t8", |b| {
            let mut plan = make_plan(graph, 8);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        println!(
            "row G: proxima single-threaded `evaluate` SKIPPED under the 90-minute measurement \
             budget. Its matmul step alone is seq*d_ff*d_model = 512*8192*2048 = 8.6B FLOPs, the \
             same order as row F's skipped 2048^3 case (~40s/call extrapolated); \
             evaluate_parallel and ggml both ran to completion for this row."
        );
        c.bench_function("row_g_proxima_mlp_chain_evaluate_parallel_w8", |b| {
            b.iter(|| black_box(evaluate_parallel(&program, &[], &blocks, &[], workers).unwrap()))
        });
    }
}

// ---------------------------------------------------------------------
// row H: pure elementwise chain, maximally memory-bound, 64M elements.
// Per row D's finding, this crate does NOT fuse elementwise-into-
// elementwise — only reduce-absorbs-elementwise. So this row is the
// falsifying control the hypothesis needs: both engines materialize every
// intermediate, and the honest expectation is parity or a ggml win (its
// per-op kernels are hand-tuned SIMD; ours is a strided interpreter loop).
// ---------------------------------------------------------------------

fn row_h_elementwise_chain(c: &mut Criterion) {
    println!("\n=== ROW H: pure elementwise chain (no reduce), 64M elements, 7 ops ===");
    println!("MECHANISM NOTE: this crate's fusion is Reduce-absorbs-its-immediate-Elementwise- \
        producer ONLY (see bind.rs's own module doc). A chain of elementwise ops with no \
        reduce at the end does not fuse at all — every op materializes on both engines. \
        Expect parity or a loss here; that is the honest result this row exists to surface.");

    let n = 64 * 1024 * 1024usize;
    let x_data = random_vec(50, n, 0.2);

    let mut program = Vec::new();
    let x = f32_input(&mut program, &[Extent::Static(n as u32)]);
    let b = unary(&mut program, ScalarOp::Negate, x, 1);
    let cc = unary(&mut program, ScalarOp::Exponential, b, 1);
    let d = binary(&mut program, ScalarOp::Add, cc, x, 1);
    let e = unary(&mut program, ScalarOp::SquareRoot, d, 1);
    let f = binary(&mut program, ScalarOp::Multiply, e, x, 1);
    let g = unary(&mut program, ScalarOp::Tanh, f, 1);
    let h = unary(&mut program, ScalarOp::Negate, g, 1);
    let _ = h;

    let shapes = proxima_tensor::infer(&program, &[]).expect("row h infers");
    let resolved = proxima_tensor::bind(&program, &shapes, &[]).expect("row h binds");
    println!(
        "proxima BoundOp count for the 7-op pure chain: {} (== 7: no fusion, every op \
         materializes its own {} MB buffer)",
        resolved.len(),
        n * 4 / (1024 * 1024)
    );
    println!("ggml materializes all 7 of its op nodes too (neg, exp, add, sqrt, mul, tanh, neg) — no fusion on either side for this row");

    let proxima_out = evaluate(&program, &[], &[&x_data], &[]).expect("row h evaluates");

    unsafe {
        let ctx = ggml_ctx(3072);
        let xt = new_f32_1d(ctx, n as i64, &x_data);
        let bt = ggml_neg(ctx, xt);
        let ct = ggml_exp(ctx, bt);
        let dt = ggml_add(ctx, ct, xt);
        let et = ggml_sqrt(ctx, dt);
        let ft = ggml_mul(ctx, et, xt);
        let gt = ggml_tanh(ctx, ft);
        let ht = ggml_neg(ctx, gt);
        let graph = build_graph(ctx, ht);
        let mut setup_plan = make_plan(graph, 1);
        compute_plan(graph, &mut setup_plan);
        let ggml_out = read_f32(ht);

        let diff = max_abs_diff(proxima_out.root(), &ggml_out);
        println!("row H max abs diff (ggml vs proxima evaluate): {diff:e}");
        assert!(diff < 1e-2, "row H numerical mismatch: {diff}");

        let workers = NonZeroUsize::new(8).unwrap();
        c.bench_function("row_h_ggml_elementwise_chain_64m_t1", |b| {
            let mut plan = make_plan(graph, 1);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        c.bench_function("row_h_ggml_elementwise_chain_64m_t8", |b| {
            let mut plan = make_plan(graph, 8);
            b.iter(|| {
                compute_plan(graph, &mut plan);
                black_box(());
            })
        });
        c.bench_function("row_h_proxima_elementwise_chain_64m_evaluate", |b| {
            b.iter(|| black_box(evaluate(&program, &[], &[&x_data], &[]).unwrap()))
        });
        c.bench_function("row_h_proxima_elementwise_chain_64m_evaluate_parallel_w8", |b| {
            b.iter(|| black_box(evaluate_parallel(&program, &[], &[&x_data], &[], workers).unwrap()))
        });
    }
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        // sample_size(10)/measurement_time(500ms) previously starved the estimator —
        // a 250ms/iter arm got ~2 iterations for 10 samples, producing >10% CI width
        // that a ratio should never be computed from. Raised for the row_f head-to-head.
        .sample_size(50)
        .measurement_time(Duration::from_secs(10));

    // definitive-measurement run: row_f only, per 75-minute wall-clock ceiling.
    // other rows left defined (unused) rather than deleted.
    let _ = row_a_home_turf_quantized_gemv;
    let _ = row_b_f32_gemv;
    let _ = row_c_gather_fused_reduce;
    let _ = row_d_deep_chain;
    let _ = row_e_locally_connected_window;
    let _ = row_g_mlp_chain;
    let _ = row_h_elementwise_chain;

    row_f_control_bare_gemm(&mut criterion);

    criterion.final_summary();
}
