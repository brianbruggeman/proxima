//! ROW 368: is the ROW 367 slowdown (fewer dispatches, more wall time on the
//! broadcast-reduce epilogue) the fused kernel's own body, or noise around
//! it? Times the UNFUSED rmsnorm pair (a cooperative sum-of-squares reduce
//! with an `Identity` epilogue, followed by the six-step elementwise tail
//! `mean_square`/`+eps`/`sqrt`/`reciprocal`/`x*inv_rms`/`*gamma`) against the
//! FUSED broadcast-reduce epilogue (the same tail folded into
//! [`BoundOpKind::Reduce::epilogue_body`]) for the same real hidden width
//! `[seq, 4096]` this crate's own
//! `reduce_epilogue_fusion_parity.rs::the_rmsnorm_broadcast_epilogue_holds_
//! parity_on_a_real_hidden_width` uses, at `seq=1` (decode) and `seq=7`
//! (prefill-like).
//!
//! Fused/unfused selection needs no cargo-feature toggle at build time
//! (`bind`'s own fusion rule, `proxima-tensor/src/bind.rs`'s `BoundOpKind::
//! Reduce::epilogue_body` doc, precondition (b)): "this fold's output has no
//! other consumer anywhere in the program and is NOT ITSELF A REQUESTED
//! OUTPUT". Requesting the sum-of-squares reduce node as an EXTRA output
//! root alongside the real `scaled` root is enough to keep it unfused --
//! same graph, same six-step tail, no duplicated program text between the
//! two arms.
//!
//! 32 independent chains (`INSTANCES`), one dispatch group per chain,
//! encoded into ONE command buffer via [`omega::plan_named`]/
//! [`omega::execute_plan_named`] (that pair's own doc: one
//! `computeCommandEncoder`, `endEncoding`d once) -- the same batching
//! `matvec_roofline_ladder.rs`'s `multi_tensor_matmul_program`+`plan`/
//! `execute_plan` arm uses, through the public API rather than a hand-rolled
//! encoder. All 32 chains share the SAME `x`/`gamma`/`inv_dim`/`eps` input
//! nodes -- this file measures per-dispatch/kernel-body cost, not memory
//! bandwidth, so re-reading the same 16 KB (seq=1) or 112 KB (seq=7) input 32
//! times is the right shape, not a confound.
//!
//! Warm-up + median-of-7 wall time, same discipline as
//! [`warmed_up_samples`](../matvec_roofline_ladder.rs) (one untimed warm-up
//! dispatch of the whole batch, then `REPEATS` timed repeats, encode loop
//! before the timer starts). The write loop's own division-class instruction
//! count (owner directive, ROW 368: read the AIR before timing) is checked
//! separately -- `xcrun metal -S -emit-llvm` over the SAME emitted kernel
//! text `omega::msl::emit` produces, never hand-rewritten MSL fed to the GPU
//! driver itself this slice (the emitter is not touched here).

#![cfg(all(
    feature = "metal",
    feature = "reduce-epilogue-fusion",
    feature = "instrument",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce,
    ReduceInit, ScalarOp, append, bind, infer, projection,
};

mod support;
use support::{as_named_blocks, production_numeric_policy};

const REPEATS: usize = 7;
const INSTANCES: usize = 32;

/// One rmsnorm chain's own node set -- `scaled` is the real `x * inv_rms *
/// gamma` output every arm requests; `sum_squares` is the pre-fusion
/// cooperative reduce whose extra-output-root trick (this file's module
/// doc) is what keeps a chain unfused when its own root is ALSO requested.
struct RmsnormChain {
    scaled: NodeId,
    sum_squares: NodeId,
}

/// Appends one `[seq, dim]` RMSNorm chain (`x -> x*x -> sum_squares ->
/// mean_square -> +eps -> sqrt -> reciprocal -> x*inv_rms -> *gamma`) to
/// `program`, reusing the SAME `x`/`gamma`/`inv_dim`/`eps` input nodes across
/// every call -- see this file's module doc for why sharing inputs across
/// chains is the right shape for a dispatch-cost measurement.
fn append_rmsnorm_chain(
    program: &mut Vec<Op>,
    x: NodeId,
    gamma: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
) -> RmsnormChain {
    let full = || IndexMap::Affine(projection(2, &[0, 1]));
    let keep_seq = || IndexMap::Affine(projection(1, &[0]));
    let broadcast_scalar_seq = || IndexMap::Affine(projection(1, &[]));
    let broadcast_seq_over_dim = || IndexMap::Affine(projection(2, &[0]));
    let broadcast_dim_over_seq = || IndexMap::Affine(projection(2, &[1]));

    let squared = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(x, full()), (x, full())],
            name: None,
        },
    );
    let sum_squares = append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: squared,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let mean_square = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(sum_squares, keep_seq()), (inv_dim, broadcast_scalar_seq())],
            name: None,
        },
    );
    let mean_square_eps = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![(mean_square, keep_seq()), (eps, broadcast_scalar_seq())],
            name: None,
        },
    );
    let rms = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::SquareRoot,
            operands: vec![(mean_square_eps, keep_seq())],
            name: None,
        },
    );
    let inv_rms = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: vec![(rms, keep_seq())],
            name: None,
        },
    );
    let normed = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(x, full()), (inv_rms, broadcast_seq_over_dim())],
            name: None,
        },
    );
    let scaled = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(normed, full()), (gamma, broadcast_dim_over_seq())],
            name: None,
        },
    );
    RmsnormChain { scaled, sum_squares }
}

/// [`build_batch`]'s own return shape: the program, its (empty, this file
/// never uses decode symbols) symbol table, every root a caller should
/// request as output, and deterministic `(name, data)` pairs for every
/// named input.
type RmsnormBatch = (Vec<Op>, Vec<u64>, Vec<NodeId>, Vec<(String, Vec<f32>)>);

/// Builds `INSTANCES` independent rmsnorm chains sharing one `x`/`gamma`/
/// `inv_dim`/`eps` input set, plus the two named-input byte buffers every
/// chain reads. `fused`: `false` requests every chain's own `sum_squares` as
/// an extra output root (keeps every reduce unfused, per this file's module
/// doc); `true` requests only the real `scaled` roots.
fn build_batch(seq: u32, dim: u32, fused: bool) -> RmsnormBatch {
    let mut program: Vec<Op> = Vec::new();
    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(seq), Extent::Static(dim)],
            name: Some("x".into()),
        },
    );
    let gamma = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(dim)],
            name: Some("gamma".into()),
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

    let mut roots = Vec::with_capacity(INSTANCES * 2);
    for _ in 0..INSTANCES {
        let chain = append_rmsnorm_chain(&mut program, x, gamma, inv_dim, eps);
        roots.push(chain.scaled);
        if !fused {
            roots.push(chain.sum_squares);
        }
    }

    let mut lcg = Lcg(seq as u64 * 97 + dim as u64 * 7 + 3);
    let x_data: Vec<f32> = (0..(seq as u64 * dim as u64) as usize)
        .map(|_| lcg.next_unit())
        .collect();
    let gamma_data: Vec<f32> = (0..dim as usize).map(|_| lcg.next_unit()).collect();
    let named = vec![
        ("x".to_string(), x_data),
        ("gamma".to_string(), gamma_data),
        ("inv_dim".to_string(), vec![1.0f32 / dim as f32]),
        ("eps".to_string(), vec![1e-5f32]),
    ];
    (program, Vec::new(), roots, named)
}

fn epilogued_reduce_count(resolved: &[BoundOp]) -> usize {
    resolved
        .iter()
        .filter(|bound| {
            matches!(
                &bound.kind,
                BoundOpKind::Reduce {
                    epilogue_operands, ..
                } if !epilogue_operands.is_empty()
            )
        })
        .count()
}

struct Stats {
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    cov_pct: f64,
}

fn stats(samples_ms: &[f64]) -> Stats {
    let len = samples_ms.len();
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("no NaN sample"));
    let mean = samples_ms.iter().sum::<f64>() / len as f64;
    let median = if len.is_multiple_of(2) {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    };
    let variance = samples_ms
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / len as f64;
    let cov_pct = if mean.abs() > f64::MIN_POSITIVE {
        100.0 * variance.sqrt() / mean
    } else {
        0.0
    };
    Stats {
        mean_ms: mean,
        median_ms: median,
        min_ms: sorted[0],
        max_ms: sorted[len - 1],
        cov_pct,
    }
}

/// Runs one `(seq, fused)` cell: builds the batch, binds, prints the fused
/// arm's own emitted MSL once (this file's module doc: read before timing),
/// then times [`REPEATS`] warmed-up repeats of `omega::execute_plan_named`
/// over the whole `INSTANCES`-chain program (one command buffer per repeat,
/// `plan`/`execute_plan`'s own doc).
fn run_cell(label: &str, seq: u32, dim: u32, fused: bool) {
    let (program, symbols, roots, named) = build_batch(seq, dim, fused);
    let named_blocks = as_named_blocks(&named);

    let shapes = infer(&program, &symbols).expect("rmsnorm batch program infers");
    let resolved =
        bind(&program, &shapes, &roots, production_numeric_policy()).expect("rmsnorm batch program binds");
    let fused_count = epilogued_reduce_count(&resolved);
    println!(
        "=== ROW 368 {label} seq={seq} dim={dim} fused_requested={fused} \
         fused_reduce_count={fused_count}/{INSTANCES} resolved_ops={} ==="
        , resolved.len()
    );
    if fused {
        assert_eq!(
            fused_count, INSTANCES,
            "{label}: expected every one of {INSTANCES} chains to fuse its rmsnorm tail"
        );
        if let Some(bound) = resolved.iter().find(|bound| {
            matches!(&bound.kind, BoundOpKind::Reduce { epilogue_operands, .. } if !epilogue_operands.is_empty())
        }) {
            let packed_operands = Default::default();
            let kernel = omega::msl::emit(bound, &packed_operands, NumericPolicy::bit_exact())
                .expect("fused rmsnorm reduce emits MSL");
            println!("--- ROW 368 fused rmsnorm MSL (entry={}) ---\n{}", kernel.entry, kernel.source);
        }
    } else {
        assert_eq!(
            fused_count, 0,
            "{label}: expected every one of {INSTANCES} chains to stay unfused \
             (sum_squares requested as an extra output root)"
        );
    }

    let plan = omega::plan_named(
        &program,
        &symbols,
        &named_blocks,
        &roots,
        production_numeric_policy(),
    )
    .expect("metal plans the rmsnorm batch");

    omega::execute_plan_named(&plan, &named_blocks).expect("warm-up run");
    let mut samples_ms = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let started = Instant::now();
        omega::execute_plan_named(&plan, &named_blocks).expect("timed run");
        samples_ms.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let stat = stats(&samples_ms);
    println!(
        "ROW 368 {label} seq={seq} wall_ms: mean={:.4} median={:.4} min={:.4} max={:.4} cov={:.2}%",
        stat.mean_ms, stat.median_ms, stat.min_ms, stat.max_ms, stat.cov_pct
    );

    // ROW 372: GPU-only time, per instance. `metal::execute_plan_named_op_timed`
    // (`instrument`-gated) reads each dispatch's own command buffer
    // `GPUEndTime()-GPUStartTime()` -- see its own doc: one command buffer
    // per `BoundOp`, so summing every op's `gpu_ns` over one warmed-up run
    // gives the batch's total device-side execution time, independent of the
    // host-side encode/commit/wait overhead the wall bracket above also
    // carries. Dividing by `INSTANCES` reports GPU time per rmsnorm chain,
    // the unit the bare-cell table below is keyed on.
    let (_evaluated, warm_timings) =
        omega::metal::execute_plan_named_op_timed(&plan, &named_blocks).expect("warm-up gpu-timed run");
    drop(warm_timings);
    let mut gpu_samples_us = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let (_evaluated, timings) =
            omega::metal::execute_plan_named_op_timed(&plan, &named_blocks).expect("gpu-timed run");
        let total_gpu_ns: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
        gpu_samples_us.push(total_gpu_ns as f64 / INSTANCES as f64 / 1e3);
    }
    let gpu_stat = stats(&gpu_samples_us);
    println!(
        "ROW 372 {label} seq={seq} gpu_us_per_instance: mean={:.3} median={:.3} min={:.3} max={:.3} cov={:.2}%",
        gpu_stat.mean_ms, gpu_stat.median_ms, gpu_stat.min_ms, gpu_stat.max_ms, gpu_stat.cov_pct
    );
}

#[test]
fn rmsnorm_unfused_vs_fused_decode_shape() {
    run_cell("a_unfused_decode", 1, 4096, false);
    run_cell("b_fused_decode", 1, 4096, true);
}

#[test]
fn rmsnorm_unfused_vs_fused_prefill_shape() {
    run_cell("c_unfused_prefill", 7, 4096, false);
    run_cell("d_fused_prefill", 7, 4096, true);
}

// ---- ROW 350-method rewrite: the broadcast-epilogue write loop's own
// generic N-D coordinate decomposition, applied to a single-reduction-axis
// shape where it is a provable no-op (owner directive, ROW 368) ----

/// [`push_broadcast_epilogue_write`]'s (`omega/src/msl.rs:6479`) own
/// generic decomposition for the SOLE reduction axis this rmsnorm shape
/// ever has (`reduce_dims.len() == 1`, the `dim` axis at output index `1`):
/// `r` already ranges over `[0, u.reduction_total)`, and
/// `u.reduction_total == u.reduction_extents[0]` for a rank-1 reduction, so
/// `r % u.reduction_extents[0]` is `r` itself and the paired `remaining_r /=
/// ...` division result is never read again -- ROW 350's exact "provable
/// no-op" shape (`docs/discipline.md` ROW 350: `output_extents[0]==1` made
/// the packed-row preamble's own `%`/`/` pair unreachable data; here it is
/// `reduce_dims.len()==1` making the write loop's pair unreachable data).
/// `assert!` on the exact literal (never a regex/best-effort match) so a
/// silent divergence from the real emitted text fails loudly rather than
/// applying to the wrong bytes.
fn row368_direct_write_loop_rewrite(source: &str) -> String {
    let old = "        long reduction_coord[1];\n        long remaining_r = r;\n        reduction_coord[0] = remaining_r % u.reduction_extents[0]; remaining_r /= u.reduction_extents[0];\n        full_coord[1] = reduction_coord[0];\n";
    let new = "        full_coord[1] = r;\n";
    assert!(
        source.contains(old),
        "row368 write-loop pattern not found in emitted source -- emitted text changed shape"
    );
    source.replace(old, new)
}

/// AIR division-class instruction counts for one compiled kernel --
/// `sdiv`/`srem` (Metal's `long`/`int` are SIGNED, so the write loop's own
/// `%`/`/` against `u.reduction_extents`/`remaining_r` -- both declared
/// `long` -- lower to the SIGNED forms, never `udiv`/`urem`; counting the
/// unsigned mnemonics on signed source reads zero regardless of what the
/// rewrite changed, which is not evidence of anything). Both signed and
/// unsigned are counted so a future kernel that mixes `uint` coordinate math
/// is not silently missed either.
struct DivisionCounts {
    sdiv: usize,
    srem: usize,
    udiv: usize,
    urem: usize,
}

impl DivisionCounts {
    fn total(&self) -> usize {
        self.sdiv + self.srem + self.udiv + self.urem
    }
}

/// Compiles `source` to AIR text (`xcrun metal -S -emit-llvm`, the same
/// tool `compiler-output.md` used) and counts every division-class
/// instruction. Returns `(air_text, counts)`. `#[allow(clippy::expect_used)]`
/// via the file-level attribute: a metal toolchain failure here IS the test
/// failing, same convention as every `compile_pipeline` panic in this
/// crate's own Metal test harnesses.
fn compile_to_air(source: &str, label: &str) -> (String, DivisionCounts) {
    let dir = std::env::temp_dir().join(format!("row368-air-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir for AIR compilation");
    let metal_path = dir.join("kernel.metal");
    let air_path = dir.join("kernel.ll");
    std::fs::write(&metal_path, source).expect("writes the .metal source");

    let output = std::process::Command::new("xcrun")
        .args(["metal", "-S", "-emit-llvm", "-o"])
        .arg(&air_path)
        .arg(&metal_path)
        .output()
        .expect("invokes xcrun metal");
    assert!(
        output.status.success(),
        "xcrun metal -S -emit-llvm failed for {label}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let air_text = std::fs::read_to_string(&air_path).expect("reads the emitted AIR text");
    let counts = DivisionCounts {
        sdiv: air_text.matches("sdiv ").count(),
        srem: air_text.matches("srem ").count(),
        udiv: air_text.matches("udiv ").count(),
        urem: air_text.matches("urem ").count(),
    };
    let _ = std::fs::remove_dir_all(&dir);
    (air_text, counts)
}

/// Owner directive (ROW 368): read the AIR of the fused kernel as emitted
/// vs the ROW-350-method write-loop rewrite, and count division-class
/// instructions in the write loop BEFORE timing either. This cell is the
/// "before timing" half -- the mechanism check that decides whether the
/// rewrite's division-class instructions actually disappear at the AIR
/// level, for both the decode (`seq=1`) and prefill-like (`seq=7`) shapes.
fn air_division_count_for_shape(label: &str, seq: u32, dim: u32) {
    let (program, symbols, roots, _named) = build_batch(seq, dim, true);
    let shapes = infer(&program, &symbols).expect("rmsnorm batch program infers");
    let resolved = bind(&program, &shapes, &roots, NumericPolicy::default()).expect("rmsnorm batch program binds");
    let bound = resolved
        .iter()
        .find(|bound| {
            matches!(&bound.kind, BoundOpKind::Reduce { epilogue_operands, .. } if !epilogue_operands.is_empty())
        })
        .expect("at least one fused rmsnorm reduce");
    let packed_operands = Default::default();
    let kernel = omega::msl::emit(bound, &packed_operands, NumericPolicy::bit_exact())
        .expect("fused rmsnorm reduce emits MSL");

    let rewritten = row368_direct_write_loop_rewrite(&kernel.source);
    assert_ne!(
        kernel.source, rewritten,
        "{label}: rewrite must change the source it was applied to"
    );

    let (_baseline_air, baseline) = compile_to_air(&kernel.source, &format!("{label}-baseline"));
    let (_rewrite_air, rewrite) = compile_to_air(&rewritten, &format!("{label}-rewrite"));

    println!(
        "ROW 368 {label} seq={seq} dim={dim} AIR division-class instruction count: \
         baseline sdiv={} srem={} udiv={} urem={} total={} | \
         row350-rewrite sdiv={} srem={} udiv={} urem={} total={}",
        baseline.sdiv, baseline.srem, baseline.udiv, baseline.urem, baseline.total(),
        rewrite.sdiv, rewrite.srem, rewrite.udiv, rewrite.urem, rewrite.total()
    );
    assert!(
        rewrite.total() < baseline.total(),
        "{label}: row350-method rewrite must strictly reduce the division-class \
         instruction count (baseline={}, rewrite={})",
        baseline.total(),
        rewrite.total()
    );
}

#[test]
fn rmsnorm_fused_epilogue_air_division_count_decode_shape() {
    air_division_count_for_shape("e_fused_decode_air", 1, 4096);
}

#[test]
fn rmsnorm_fused_epilogue_air_division_count_prefill_shape() {
    air_division_count_for_shape("f_fused_prefill_air", 7, 4096);
}
