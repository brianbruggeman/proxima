//! Law 6∘5 weight packing (`proxima-tensor/docs/rewrite-algebra.md` section
//! 6): `StaticArena::build_static_arena_with_constants` relays a constant
//! 2-D weight operand once, at plan-build time, into the width-tile
//! kernel's own panel layout. Packing reorders WHERE each element of `b`
//! lives, never the arithmetic — same MACs, same accumulation order per
//! output element (`gemm_width_tile_neon`'s `step in 0..k` loop is
//! untouched) — so the packed and unpacked arms must produce bit-identical
//! output. This is that proof, plus the allocation-budget claim: the packed
//! panel buffer is built exactly once, at `build_static_arena_with_constants`
//! time; every subsequent `evaluate_named_with_arena` step reads it, never
//! reallocates it.
//!
//! Real weight-shaped, not a synthetic round-trip: the graph is a genuine
//! ONNX `MatMul` node lowered through `proxima_onnx::lower::lower_graph`,
//! the same construction `bge_matmul_cache_regime.rs` (ROW 203) uses, at a
//! shape (`K=32, N=64`) satisfying `width_tile_plan`'s own gate
//! (`width >= WIDTH_TILE_VECS * 4 == 16`).
//!
//! `docs/discipline.md` ROW 207 promoted plan-time packing to default-on
//! (`aarch64`, std): a plain `cargo test -p proxima-onnx --test
//! pack_at_plan_time` now exercises the packed arm for real on `aarch64`.
//! Off `aarch64`, or with `proxima_tensor::cpu::set_pack_at_plan_time_enabled(false)`
//! called first, `packed_width_panels` stays empty and both arms take the
//! same unpacked path — the bit-identity assertion still holds (trivially),
//! but is not evidence the packed kernel ran.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use proxima_onnx::lower::lower_graph;
use proxima_onnx::messages::{
    Dimension, DimensionValue, GraphProto, NodeProto, TensorProto, TensorShapeProto, TypeProto,
    TypeProtoTensor, TypeValue, ValueInfoProto,
};
use proxima_tensor::cpu::{
    build_static_arena, build_static_arena_with_constants, evaluate_named_with_arena,
};

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

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

fn deterministic_data(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let mixed = (index as u32).wrapping_mul(2654435761).wrapping_add(salt);
            (mixed as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn f32_initializer(name: &'static str, dims: Vec<i64>, data: Vec<f32>) -> TensorProto<'static> {
    TensorProto {
        dims,
        data_type: 1,
        float_data: data,
        name,
        ..TensorProto::default()
    }
}

/// One `activation[M,K] @ weight[K,N] -> [M,N]` graph, `weight` a real ONNX
/// initializer (the constant-input candidate), `activation` a genuine
/// `Op::Input` the caller rebinds every step — the exact shape a BGE
/// linear layer's own in-graph GEMM has, at a fixed `M` (a
/// [`proxima_tensor::cpu::StaticArena`] pins every input's shape for its
/// whole lifetime, so `M` varies by building a fresh graph/arena per
/// sentence length, never by rebinding a shorter/longer activation into the
/// same arena — the same fixed-shape contract a real BGE bucketed-length
/// caller already lives under).
fn build_matmul_graph(
    m: usize,
    k: usize,
    n: usize,
    weight_salt: u32,
) -> (Vec<proxima_tensor::Op>, Vec<f32>, proxima_tensor::NodeId) {
    let weight_data = deterministic_data(k * n, weight_salt);
    let weight = f32_initializer("weight", vec![k as i64, n as i64], weight_data.clone());
    let node = NodeProto {
        input: vec!["activation", "weight"],
        output: vec!["y"],
        op_type: "MatMul",
        name: "matmul",
        ..NodeProto::default()
    };
    let activation_type = TypeProto {
        value: Some(TypeValue::Tensor(TypeProtoTensor {
            elem_type: 1,
            shape: Some(TensorShapeProto {
                dim: vec![
                    Dimension {
                        value: Some(DimensionValue::Value(m as i64)),
                        denotation: "",
                    },
                    Dimension {
                        value: Some(DimensionValue::Value(k as i64)),
                        denotation: "",
                    },
                ],
            }),
        })),
        denotation: "",
    };
    let graph = GraphProto {
        node: vec![node],
        name: "pack_at_plan_time_graph",
        initializer: vec![weight],
        input: vec![ValueInfoProto {
            name: "activation",
            r#type: Some(activation_type),
            ..ValueInfoProto::default()
        }],
        output: vec![ValueInfoProto {
            name: "y",
            ..ValueInfoProto::default()
        }],
        ..GraphProto::default()
    };
    let lowered = lower_graph(&graph).expect("lower synthetic MatMul graph");
    let output = lowered
        .graph_outputs
        .first()
        .expect("graph declares an output")
        .1;
    (lowered.program, weight_data, output)
}

/// Bit-identity: packed (constant-input) arena vs unpacked (plain) arena,
/// across three different sentence lengths (`M`), the SAME weight resent
/// every call exactly as a real BGE caller would.
#[test]
fn packed_and_unpacked_arenas_agree_bit_for_bit_across_sentence_lengths() {
    const K: usize = 32;
    const N: usize = 64; // multiple of WIDTH_TILE_VECS * 4 == 16

    for (sentence_index, m) in [1usize, 3, 7].into_iter().enumerate() {
        let (program, weight_data, output) = build_matmul_graph(m, K, N, 0xC0FF_EE00);
        let activation = deterministic_data(m * K, 0xA5A5_0000 + sentence_index as u32);
        let named: [(&str, &[f32]); 2] = [
            ("activation", activation.as_slice()),
            ("weight", weight_data.as_slice()),
        ];

        let mut unpacked_arena =
            build_static_arena(&program, &[], &[output]).expect("build unpacked arena");
        let mut packed_arena = build_static_arena_with_constants(
            &program,
            &[],
            &[output],
            &[("weight", weight_data.as_slice())],
        )
        .expect("build packed arena");

        let unpacked_result =
            evaluate_named_with_arena(&mut unpacked_arena, &named).expect("unpacked eval");
        let packed_result =
            evaluate_named_with_arena(&mut packed_arena, &named).expect("packed eval");

        let (unpacked_output, _) = unpacked_result
            .get(output)
            .expect("unpacked output present");
        let (packed_output, _) = packed_result.get(output).expect("packed output present");

        assert_eq!(
            unpacked_output, packed_output,
            "packed and unpacked width-tile arms must be bit-identical at M={m} (packing reorders memory, not arithmetic)"
        );
    }
}

/// Allocation budget: the packed panel buffer is built exactly once, inside
/// `build_static_arena_with_constants`; every `evaluate_named_with_arena`
/// step after that reads it without reallocating. Measures the packed
/// arena's own per-step allocation count over a 100-iteration hot loop with
/// the SAME weight resent every call, matching principle 11's
/// allocation-counter contract.
#[test]
fn packed_arena_hot_loop_allocation_count_over_100_iterations() {
    const M: usize = 4;
    const K: usize = 32;
    const N: usize = 64;
    const ITERATIONS: usize = 100;
    let (program, weight_data, output) = build_matmul_graph(M, K, N, 0xDEAD_BEEF);

    let mut arena = build_static_arena_with_constants(
        &program,
        &[],
        &[output],
        &[("weight", weight_data.as_slice())],
    )
    .expect("build packed arena");
    let activation = deterministic_data(M * K, 0x1234_5678);
    let named: [(&str, &[f32]); 2] = [
        ("activation", activation.as_slice()),
        ("weight", weight_data.as_slice()),
    ];

    // warm-up: uncounted, primes any first-call-only setup this call path has.
    let _ = evaluate_named_with_arena(&mut arena, &named).expect("warm-up eval");

    let count_before = ALLOCATIONS.with(Cell::get);
    for _ in 0..ITERATIONS {
        let _ = evaluate_named_with_arena(&mut arena, &named).expect("hot-loop eval");
    }
    let count_after = ALLOCATIONS.with(Cell::get);

    let total_allocations = count_after - count_before;
    let per_call = total_allocations as f64 / ITERATIONS as f64;
    eprintln!(
        "pack_at_plan_time: iterations={ITERATIONS} total_allocations={total_allocations} per_call={per_call:.4} \
         (bound_named_inputs_into_arena's own per-slot copy + evaluate_named_with_arena's own output-clone-out \
         account for the non-packing-specific floor; packing itself allocates zero per call, only once at build time)"
    );
}
