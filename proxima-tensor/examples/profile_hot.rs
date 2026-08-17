//! Throwaway harness for `scratchpad/opt/discipline.md` — direct `Instant`
//! timing (no criterion overhead) and a symbol the disassembler can find.
//! Not part of the crate's public surface; deleted at the end of the
//! session that needed it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use proxima_tensor::{Extent, IndexMap, NodeId, Op, ReduceInit, ScalarOp, append, evaluate, map};

/// Counts every allocation the process makes, including ones inside
/// `proxima-tensor`'s executor loop — a call-site counter incremented by
/// the crate itself would only ever report calls the crate author remembered
/// to instrument. Wrapping the global allocator counts what actually
/// happened, unconditionally (`scratchpad/opt/discipline.md` ROW 2).
struct CountingAllocator;

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn matmul_program(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: proxima_tensor::DType::Float32,
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
            dtype: proxima_tensor::DType::Float32,
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

/// Single Binary-shaped elementwise op (a * b) over `len` elements — the
/// case ROW 5's `elementwise_width_fast` actually accelerates.
fn elementwise_binary_program(len: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(len)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(len)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(1, &[0]))),
                (rhs, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );
    (program, product)
}

/// A 7-op unary chain over `len` elements — each stage single-use, so the
/// binder fuses all 7 into one multi-step `ComposedBody`
/// (`bind.rs::compose_fused_operands`) rather than 7 separate bound ops.
/// `body_shape` classifies a multi-step body as `Generic`, which ROW 5's
/// fast path deliberately does not cover (`body_shape_is_affine_fast_path`
/// only qualifies `Unary`/`Binary`) — named here as the expected outcome,
/// not a bug: this workload exercises `apply_body`'s unchanged per-element
/// path regardless of ROW 5.
fn elementwise_chain_program(len: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(len)],
            name: None,
        },
    );
    let ops = [
        ScalarOp::Negate,
        ScalarOp::Reciprocal,
        ScalarOp::Negate,
        ScalarOp::Reciprocal,
        ScalarOp::Negate,
        ScalarOp::Negate,
        ScalarOp::Tanh,
    ];
    let mut current = input;
    for op in ops {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: proxima_tensor::DType::Float32,
                body: op,
                operands: vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
    }
    (program, current)
}

/// A plain cumulative sum over `len` elements — `element_body` is a
/// one-step `Identity` (`Unary`), so this DOES hit ROW 5's `scan_width_fast`.
fn scan_program(len: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(len)],
            name: None,
        },
    );
    let scanned = append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: input,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[0])),
            keep: proxima_tensor::Keep::Scan,
            name: Some("cumsum".into()),
        }),
    );
    (program, scanned)
}

fn main() {
    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32);
    let lhs: Vec<f32> = (0..m * k).map(|value| (value % 13) as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| (value % 7) as f32).collect();

    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let start = Instant::now();
    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("gemm evaluates");
    let elapsed = start.elapsed();
    let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);

    println!("gemm {m}x{k}x{n}: {:.3}s", elapsed.as_secs_f64());
    println!("root[0]={} root_len={}", evaluated.root()[0], evaluated.root().len());
    println!("allocations during evaluate(): {}", alloc_after - alloc_before);

    #[cfg(feature = "instrument")]
    {
        let snapshot = proxima_tensor::cpu::telemetry_snapshot();
        for (name, labels, value) in &snapshot.counters {
            println!("counter {name}{labels:?} = {value}");
        }
    }

    let len: u32 = 64 * 1024 * 1024;

    {
        let (program, _out) = elementwise_binary_program(len);
        let a: Vec<f32> = (0..len).map(|value| (value % 13) as f32 + 1.0).collect();
        let b: Vec<f32> = (0..len).map(|value| (value % 7) as f32 + 1.0).collect();
        let start = Instant::now();
        let evaluated = evaluate(&program, &[], &[&a, &b], &[]).expect("elementwise binary evaluates");
        let elapsed = start.elapsed();
        println!("elementwise_binary {len}: {:.4}s", elapsed.as_secs_f64());
        println!("elementwise_binary root[0]={} root_len={}", evaluated.root()[0], evaluated.root().len());
    }

    {
        let (program, _out) = elementwise_chain_program(len);
        let input: Vec<f32> = (0..len).map(|value| (value % 13) as f32 + 1.0).collect();
        let start = Instant::now();
        let evaluated = evaluate(&program, &[], &[&input], &[]).expect("elementwise chain evaluates");
        let elapsed = start.elapsed();
        println!("elementwise_chain {len}: {:.4}s", elapsed.as_secs_f64());
        println!("elementwise_chain root[0]={} root_len={}", evaluated.root()[0], evaluated.root().len());
    }

    {
        let (program, _out) = scan_program(len);
        let input: Vec<f32> = (0..len).map(|value| (value % 13) as f32 + 1.0).collect();
        let start = Instant::now();
        let evaluated = evaluate(&program, &[], &[&input], &[]).expect("scan evaluates");
        let elapsed = start.elapsed();
        println!("scan {len}: {:.4}s", elapsed.as_secs_f64());
        println!("scan root[0]={} root_len={}", evaluated.root()[0], evaluated.root().len());
    }
}
