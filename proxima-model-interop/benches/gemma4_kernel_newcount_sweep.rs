//! SLICE 3 (`bench/kernel-newcount-sweep`): SLICE 1's Component 1 sweep
//! (widths 2/3/5/9) is a CONFOUNDED measurement of the width-1-vs-width>1
//! question — real BOS tokenization floors the real forward at width-2
//! (`prompt_for_width`'s own doc in `gemma4_forward_decomposition.rs`), and
//! width-1 decode uses a DIFFERENT renderer than the multi-position verify
//! path, so no real forward measurement ever isolates "what does new_count
//! alone cost" from "what does the BOS-inflated width-2 floor cost".
//!
//! This slice UNCONFOUNDS it at the KERNEL level, where `new_count=1` IS
//! measurable: an isolated kernel program has no BOS token and no
//! full-program epilogue gate riding along with it. Four kernels, each at
//! real gemma4-E2B dims, each swept at `new_count` ∈ {1,2,4,8,16}:
//!
//! 1. **`reduce-cooperative` at the attention-score shape** — QK score
//!    reduce over `head_dim`, output `[new_count, key_count]`. Swept on
//!    BOTH axes independently (fixed `key_count`, sweep `new_count`; fixed
//!    `new_count=1`, sweep `key_count`) so a `new_count`-only slope is not
//!    confused with a `new_count × key_count` one.
//! 2. **`reduce-cooperative` at the RMSNorm broadcast-epilogue shape** — the
//!    exact op class SLICE 1/2 named (`emit_and_classify.rs:988-1001`'s
//!    `value_norm`-style `x * inv_rms` epilogue folded into the reduce,
//!    `omega/tests/rmsnorm_fused_epilogue_cost.rs`'s own builder,
//!    `INSTANCES=1` here since this slice measures per-dispatch kernel
//!    cost, not the batching that file's `INSTANCES=32` explores). This is
//!    a PER-POSITION cost by construction (one row per `new_count`), so the
//!    fixed-vs-slope fit here is the direct test of whether SLICE 2's fixed-
//!    floor finding survives outside the width-2-to-9 confound.
//! 3. **Packed-row Q4_0 matvec (a projection)** — the REAL
//!    `blk.0.attn_k.weight` bytes from the gemma4-E2B checkpoint
//!    (`[1536, 256]`, `Q4_0`, the same tensor
//!    `omega/tests/q4_0_real_checkpoint_parity.rs` already proves
//!    numerically correct against a from-scratch dequantize+dot oracle),
//!    generalized from that test's `[k,1]` activation to `[k, new_count]`
//!    so the projection sweeps width instead of staying pinned at
//!    batch-1 decode.
//! 4. **LM-head unembed** — `[new_count, EMBEDDING] x [VOCAB, EMBEDDING]^T`,
//!    same shape as `gemma4_forward_decomposition.rs`'s own Component 2,
//!    swept to `new_count=16` here (that file stops at 8).
//!
//! # Compile vs execution axis (kept, per the existing harness)
//!
//! Every sample logs `compile_ns`/`exec_ns`/`pipeline_misses`/
//! `pipeline_hits` via [`metal_stage_totals`] — same
//! snapshot-and-reset thread-local the sibling bench reads, same caveat:
//! read once, immediately after the timed call, never accumulated across
//! iterations.
//!
//! # Run
//!
//! ```sh
//! CARGO_TARGET_DIR=<scratch>/target-uc cargo bench -p proxima-model-interop \
//!     --bench gemma4_kernel_newcount_sweep --features "metal instrument"
//! ```
//!
//! `ollama stop gemma4:e2b-it-qat` first (same GPU-contention reason the
//! sibling bench's module doc gives).

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::hint::black_box;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use memmap2::{Mmap, MmapOptions};
use omega::execute;
use omega::metal::metal_stage_totals;
use proxima_gguf::parse_complete;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_primitives::Codec;
use proxima_tensor::instrument::ticks_to_nanos;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, map,
};

/// `gemma4:e2b-it-qat`'s real blob — same checkpoint the sibling
/// decomposition bench and `omega/tests/q4_0_real_checkpoint_parity.rs`
/// both read.
const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

/// The task brief's own sweep set.
const NEW_COUNTS: [usize; 5] = [1, 2, 4, 8, 16];

/// The attention-score kernel's secondary axis — `cached_len` at
/// `new_count=1` fixed, so a `key_count`-only slope is separable from a
/// `new_count`-only one. Spans up to `SLIDING_WINDOW` (512), the real cap
/// a sliding layer's own key set never exceeds.
const KEY_COUNTS: [usize; 4] = [64, 128, 256, 512];

/// Real gemma4-E2B hparams — same values `gemma4_forward_decomposition.rs`
/// dumped from this checkpoint (2026-09-20; `gemma4.embedding_length`,
/// `gemma4.attention.head_count_kv`, `gemma4.attention.key_length_swa`).
const EMBEDDING: u32 = 1536;
const VOCAB_REAL: u32 = 262_144;
const HEAD_DIM_SWA: u32 = 256;
/// The attention-score kernel's default fixed `key_count` for the
/// `new_count` sweep — mid-range within [`KEY_COUNTS`].
const ATTENTION_SCORE_DEFAULT_KEYS: u32 = 256;

fn model_bytes() -> &'static [u8] {
    static BYTES: OnceLock<Mmap> = OnceLock::new();
    BYTES.get_or_init(|| {
        let file = File::open(MODEL_PATH).expect("open gemma4-E2B blob");
        // SAFETY: same justification as `gemma4_forward_decomposition.rs`'s
        // own `model_bytes` -- the mapping outlives the fd close and lives
        // in a process-lifetime `OnceLock`.
        unsafe { MmapOptions::new().map(&file) }.expect("mmap gemma4-E2B blob")
    })
}

fn parsed_gguf() -> &'static ParsedGguf {
    static PARSED: OnceLock<ParsedGguf> = OnceLock::new();
    PARSED.get_or_init(|| parse_complete(model_bytes()).expect("parse gemma4-E2B header"))
}

/// Real `Q4_0` packed bytes for `name`, sliced directly out of the mmap'd
/// checkpoint (zero-copy borrow -- `parse_complete`'s own doc: only the
/// directory is read, tensor payload bytes are never touched until this
/// slice). Panics (this is a bench fixture, not production code) if the
/// tensor is missing or not `Q4_0` on this host's checkpoint -- matching
/// `q4_0_real_checkpoint_parity.rs`'s "skip, don't fake" posture would
/// require every bench cell to tolerate a missing arm, which the 90-minute
/// ceiling on this slice does not afford; the checkpoint path is the same
/// one the sibling bench already hard-requires.
fn real_q4_0_tensor_bytes(name: &str) -> (&'static [u8], u32, u32) {
    let parsed = parsed_gguf();
    let bytes = model_bytes();
    let tensor = parsed
        .tensors
        .iter()
        .find(|candidate| candidate.name == name)
        .unwrap_or_else(|| panic!("{name} present in gemma4-E2B checkpoint directory"));
    assert_eq!(
        tensor.ggml_type,
        GgmlType::Q4_0,
        "{name} is Q4_0 in this checkpoint (gemma4-E2B's own quantization choice)"
    );
    let range = parsed
        .tensor_data_range(tensor, bytes.len() as u64)
        .expect("tensor byte range within file bounds");
    let in_dim = tensor.dims[0] as u32;
    let out_dim = tensor.dims[1] as u32;
    (
        &bytes[range.start as usize..range.end as usize],
        in_dim,
        out_dim,
    )
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// One raw per-iteration record -- printed, never only aggregated
/// (principle 19: results traced to records). Same field set as the
/// sibling bench's own `log_stage_sample`.
fn log_kernel_sample(kernel: &str, axis: &str, new_count: usize, other: usize, wall_ns: u64) {
    let stage = metal_stage_totals();
    println!(
        "gemma4_kernel_newcount_sweep kernel={kernel} axis={axis} new_count={new_count} \
         other={other} wall_ns={wall_ns} compile_ns={} exec_ns={} pipeline_misses={} \
         pipeline_hits={}",
        ticks_to_nanos(stage.pipeline_compile_ticks),
        ticks_to_nanos(stage.gpu_exec_ticks),
        stage.pipeline_misses,
        stage.pipeline_hits,
    );
}

/// `lhs [m,k]` times `rhs^T [n,k]` -- `omega/benches/metal_vs_cpu.rs`'s own
/// `matmul_rhs_transposed_program`, duplicated here at THIS kernel's own
/// real shape (same crate-boundary reason the sibling bench's own copy
/// states: a bench target cannot depend on another crate's bench target).
/// Used for both the attention-score reduce (`m=new_count`, `k=head_dim`,
/// `n=key_count`) and the LM-head unembed (`m=new_count`, `k=EMBEDDING`,
/// `n=VOCAB_REAL`) -- same op shape, different real dims.
fn matmul_rhs_transposed_program(m: u32, k: u32, n: u32, name: &str) -> (Vec<Op>, NodeId) {
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
            shape: vec![Extent::Static(n), Extent::Static(k)],
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
                (rhs, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some(name.into()),
        }),
    );
    (program, sum)
}

/// Packed-row Q4_0 matvec (a projection): `weight [rows, k]` (real packed
/// `Q4_0` bytes, `UInt8`) times `activation [k, new_count]` -> `[rows,
/// new_count]` -- `q4_0_real_checkpoint_parity.rs`'s own `matmul_program`
/// shape, generalized from that test's fixed `[k,1]` activation to sweep
/// `new_count`.
fn packed_row_q4_0_program(rows: u32, k: u32, new_count: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(new_count)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("q4_0_packed_row_matvec".into()),
        }),
    );
    (program, sum)
}

/// One RMSNorm chain (`x -> x*x -> sum_squares -> mean_square -> +eps ->
/// sqrt -> reciprocal -> x*inv_rms`), `INSTANCES=1` (this slice measures
/// per-dispatch kernel cost, not `rmsnorm_fused_epilogue_cost.rs`'s own
/// 32-chain batching). Requesting ONLY `scaled` as output root (never
/// `sum_squares`) is what makes `bind`'s fusion rule fold the whole tail
/// into the reduce's own epilogue (`BoundOpKind::Reduce::epilogue_body`'s
/// own doc, precondition (b): the pre-fusion sum reduce has no other
/// consumer AND is not itself a requested output) -- the exact
/// broadcast-reduce epilogue shape `emit_and_classify.rs:988-1001` names.
fn rmsnorm_broadcast_epilogue_program(seq: u32, dim: u32) -> (Vec<Op>, NodeId) {
    let mut program: Vec<Op> = Vec::new();
    let full = || IndexMap::Affine(map::projection(2, &[0, 1]));
    let keep_seq = || IndexMap::Affine(map::projection(1, &[0]));
    let broadcast_scalar_seq = || IndexMap::Affine(map::projection(1, &[]));
    let broadcast_seq_over_dim = || IndexMap::Affine(map::projection(2, &[0]));

    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(seq), Extent::Static(dim)],
            name: Some("x".into()),
        },
    );
    let inv_dim = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: Vec::new(),
            name: Some("inv_dim".into()),
        },
    );
    let eps = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: Vec::new(),
            name: Some("eps".into()),
        },
    );
    let squared = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(x, full()), (x, full())],
            name: None,
        },
    );
    let sum_squares = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: squared,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let mean_square = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(sum_squares, keep_seq()), (inv_dim, broadcast_scalar_seq())],
            name: None,
        },
    );
    let mean_square_eps = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![(mean_square, keep_seq()), (eps, broadcast_scalar_seq())],
            name: None,
        },
    );
    let rms = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::SquareRoot,
            operands: vec![(mean_square_eps, keep_seq())],
            name: None,
        },
    );
    let inv_rms = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: vec![(rms, keep_seq())],
            name: None,
        },
    );
    let normed = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(x, full()), (inv_rms, broadcast_seq_over_dim())],
            name: Some("rmsnorm_broadcast_epilogue".into()),
        },
    );
    (program, normed)
}

/// Component A: `reduce-cooperative` at the ATTENTION-SCORE shape --
/// `[new_count, head_dim] x [key_count, head_dim]^T -> [new_count,
/// key_count]`, reduced over `head_dim`. Two sub-sweeps, see module doc.
fn bench_attention_score_reduce(c: &mut Criterion) {
    let _ = metal_stage_totals();
    let mut group = c.benchmark_group("kernel_attention_score_reduce");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for new_count in NEW_COUNTS {
        let (program, root) = matmul_rhs_transposed_program(
            new_count as u32,
            HEAD_DIM_SWA,
            ATTENTION_SCORE_DEFAULT_KEYS,
            "attention_score",
        );
        let query = random_vec(21, new_count * HEAD_DIM_SWA as usize);
        let keys = random_vec(
            22,
            ATTENTION_SCORE_DEFAULT_KEYS as usize * HEAD_DIM_SWA as usize,
        );
        let blocks: [QuantizedBlock<'_>; 2] = [
            QuantizedBlock::Float32(&query),
            QuantizedBlock::Float32(&keys),
        ];
        group.bench_function(format!("new_count_sweep/{new_count}"), |bencher| {
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let started = Instant::now();
                    let evaluated = black_box(
                        execute(&program, &[], &blocks, &[root], NumericPolicy::default())
                            .expect("attention-score reduce executes on a real device"),
                    );
                    let elapsed = started.elapsed();
                    black_box(evaluated.root()[0]);
                    total += elapsed;
                    log_kernel_sample(
                        "attention_score_reduce",
                        "new_count",
                        new_count,
                        ATTENTION_SCORE_DEFAULT_KEYS as usize,
                        elapsed.as_nanos() as u64,
                    );
                }
                total
            });
        });
    }

    for key_count in KEY_COUNTS {
        let (program, root) = matmul_rhs_transposed_program(
            1,
            HEAD_DIM_SWA,
            key_count as u32,
            "attention_score_keys",
        );
        let query = random_vec(23, HEAD_DIM_SWA as usize);
        let keys = random_vec(24, key_count * HEAD_DIM_SWA as usize);
        let blocks: [QuantizedBlock<'_>; 2] = [
            QuantizedBlock::Float32(&query),
            QuantizedBlock::Float32(&keys),
        ];
        group.bench_function(format!("key_count_sweep/{key_count}"), |bencher| {
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let started = Instant::now();
                    let evaluated = black_box(
                        execute(&program, &[], &blocks, &[root], NumericPolicy::default())
                            .expect("attention-score reduce (key sweep) executes on a real device"),
                    );
                    let elapsed = started.elapsed();
                    black_box(evaluated.root()[0]);
                    total += elapsed;
                    log_kernel_sample(
                        "attention_score_reduce",
                        "key_count",
                        1,
                        key_count,
                        elapsed.as_nanos() as u64,
                    );
                }
                total
            });
        });
    }

    group.finish();
}

/// Component B: `reduce-cooperative` at the RMSNorm broadcast-epilogue
/// shape -- reduce over `EMBEDDING`, one row per `new_count`. Probes
/// lowering BEFORE timing (same real-isolation-negative-result posture the
/// sibling bench's Component 3+4 uses) -- see module doc for why this is
/// the exact case that unconfounds the sibling harness's `width=1`
/// `EpilogueNotSupported` finding.
fn bench_rmsnorm_broadcast_epilogue(c: &mut Criterion) {
    let _ = metal_stage_totals();
    let mut group = c.benchmark_group("kernel_rmsnorm_broadcast_epilogue");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for new_count in NEW_COUNTS {
        let (program, root) = rmsnorm_broadcast_epilogue_program(new_count as u32, EMBEDDING);
        let x_data = random_vec(31, new_count * EMBEDDING as usize);
        let blocks: [QuantizedBlock<'_>; 3] = [
            QuantizedBlock::Float32(&x_data),
            QuantizedBlock::Float32(std::slice::from_ref(&INV_DIM)),
            QuantizedBlock::Float32(std::slice::from_ref(&EPS)),
        ];

        let probe = execute(&program, &[], &blocks, &[root], NumericPolicy::default());
        if let Err(error) = probe {
            println!(
                "gemma4_kernel_newcount_sweep kernel=rmsnorm_broadcast_epilogue \
                 new_count={new_count} status=UNSUPPORTED reason={error:?}"
            );
            continue;
        }

        group.bench_function(format!("new_count_{new_count}"), |bencher| {
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let started = Instant::now();
                    let evaluated = black_box(
                        execute(&program, &[], &blocks, &[root], NumericPolicy::default()).expect(
                            "rmsnorm broadcast epilogue executes (probe already proved it lowers)",
                        ),
                    );
                    let elapsed = started.elapsed();
                    black_box(evaluated.root()[0]);
                    total += elapsed;
                    log_kernel_sample(
                        "rmsnorm_broadcast_epilogue",
                        "new_count",
                        new_count,
                        0,
                        elapsed.as_nanos() as u64,
                    );
                }
                total
            });
        });
    }

    group.finish();
}

const INV_DIM: f32 = 1.0 / EMBEDDING as f32;
const EPS: f32 = 1e-6;

/// Component C: packed-row Q4_0 matvec (a projection) -- REAL
/// `blk.0.attn_k.weight` packed bytes, `[1536, 256]`, activation `[1536,
/// new_count]`.
fn bench_packed_row_q4_0_matvec(c: &mut Criterion) {
    let _ = metal_stage_totals();
    let (weight_bytes, in_dim, out_dim) = real_q4_0_tensor_bytes("blk.0.attn_k.weight");

    let mut group = c.benchmark_group("kernel_packed_row_q4_0_matvec");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for new_count in NEW_COUNTS {
        let (program, root) = packed_row_q4_0_program(out_dim, in_dim, new_count as u32);
        let activation = random_vec(41, in_dim as usize * new_count);
        let blocks: [QuantizedBlock<'_>; 2] = [
            QuantizedBlock::Packed {
                codec: Codec::Q4_0,
                bytes: weight_bytes,
            },
            QuantizedBlock::Float32(&activation),
        ];
        group.bench_function(format!("new_count_{new_count}"), |bencher| {
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let started = Instant::now();
                    let evaluated = black_box(
                        execute(&program, &[], &blocks, &[root], NumericPolicy::default())
                            .expect("packed-row Q4_0 matvec executes on real checkpoint bytes"),
                    );
                    let elapsed = started.elapsed();
                    black_box(evaluated.root()[0]);
                    total += elapsed;
                    log_kernel_sample(
                        "packed_row_q4_0_matvec",
                        "new_count",
                        new_count,
                        0,
                        elapsed.as_nanos() as u64,
                    );
                }
                total
            });
        });
    }

    group.finish();
}

/// Component D: LM-head unembed -- same shape as
/// `gemma4_forward_decomposition.rs`'s Component 2, extended to
/// `new_count=16`.
fn bench_lm_head_unembed(c: &mut Criterion) {
    let _ = metal_stage_totals();
    let mut group = c.benchmark_group("kernel_lm_head_unembed");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for new_count in NEW_COUNTS {
        let (program, root) = matmul_rhs_transposed_program(
            new_count as u32,
            EMBEDDING,
            VOCAB_REAL,
            "lm_head_matmul",
        );
        let hidden = random_vec(51, new_count * EMBEDDING as usize);
        let weight = random_vec(52, EMBEDDING as usize * VOCAB_REAL as usize);
        let blocks: [QuantizedBlock<'_>; 2] = [
            QuantizedBlock::Float32(&hidden),
            QuantizedBlock::Float32(&weight),
        ];
        group.bench_function(format!("new_count_{new_count}"), |bencher| {
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let started = Instant::now();
                    let evaluated = black_box(
                        execute(&program, &[], &blocks, &[root], NumericPolicy::default())
                            .expect("lm head projection executes on a real device"),
                    );
                    let elapsed = started.elapsed();
                    black_box(evaluated.root()[0]);
                    total += elapsed;
                    log_kernel_sample(
                        "lm_head_unembed",
                        "new_count",
                        new_count,
                        0,
                        elapsed.as_nanos() as u64,
                    );
                }
                total
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_attention_score_reduce,
    bench_rmsnorm_broadcast_epilogue,
    bench_packed_row_q4_0_matvec,
    bench_lm_head_unembed
);
criterion_main!(benches);
