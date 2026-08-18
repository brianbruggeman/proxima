#![allow(clippy::expect_used)]
//! What the same contraction costs as its algebra changes.
//!
//! A GEMM is one instantiation of `reduce(body(a, b))` over a shared axis:
//! body `Multiply`, reduce `Add`. ggml ships a hand-written kernel for exactly
//! that pair. This crate expresses the *pair* as data, so every other pair is
//! reachable without new code — but only the `Multiply`/`Add` instantiation
//! meets `cpu::gemm_tile_neon`'s applicability gate, so the others fall to the
//! generic path. This harness measures that difference instead of asserting it.
//!
//! `size` and `iters` come from argv. Every arm runs the identical shape and
//! the identical input buffers, so the only variable is the algebra.

use std::env;
use std::num::NonZeroUsize;
use std::time::Instant;

use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append,
    evaluate_parallel, map,
};

struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn random_vec(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit() * scale).collect()
}

/// The contraction, parameterized by its algebra. `body` combines the two
/// operands elementwise over the full (m, n, k) space; `reduce` folds the
/// contraction axis away.
fn contraction_program(
    size: u32,
    body: ScalarOp,
    reduce: ScalarOp,
    init: ReduceInit,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(size), Extent::Static(size)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(size), Extent::Static(size)],
            name: None,
        },
    );
    let combined = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let folded = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: reduce,
            init,
            operand: combined,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, folded)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let size: u32 = args
        .get(1)
        .map_or(1024, |raw| raw.parse().expect("size must be an integer"));
    let iters: usize = args
        .get(2)
        .map_or(5, |raw| raw.parse().expect("iters must be an integer"));
    let threads: usize = args
        .get(3)
        .map_or(8, |raw| raw.parse().expect("threads must be an integer"));

    let elements = (size as usize) * (size as usize);
    let lhs = random_vec(1, elements, 1.0);
    let rhs = random_vec(2, elements, 1.0);
    let workers = NonZeroUsize::new(threads).expect("threads must be nonzero");
    let macs = (size as f64).powi(3);

    // sum-of-products is the gemm gate's pair; the rest are the same
    // contraction under a different algebra, reachable only because the
    // algebra is data rather than a kernel name.
    let arms: [(&str, ScalarOp, ScalarOp, ReduceInit); 4] = [
        ("sum_of_products", ScalarOp::Multiply, ScalarOp::Add, ReduceInit::Zero),
        ("max_of_products", ScalarOp::Multiply, ScalarOp::Maximum, ReduceInit::NegativeInfinity),
        ("max_of_sums", ScalarOp::Add, ScalarOp::Maximum, ReduceInit::NegativeInfinity),
        ("min_of_diffs", ScalarOp::Subtract, ScalarOp::Minimum, ReduceInit::PositiveInfinity),
    ];

    for (name, body, reduce, init) in arms {
        let (program, _root) = contraction_program(size, body, reduce, init);

        let warm = evaluate_parallel(&program, &[], &[&lhs, &rhs], &[], workers);
        let Ok(_) = warm else {
            println!("arm={name} size={size} threads={threads} UNSUPPORTED");
            continue;
        };

        let mut best_nanos = u64::MAX;
        let mut checksum = 0.0f32;
        for _ in 0..iters {
            let start = Instant::now();
            let evaluated = evaluate_parallel(&program, &[], &[&lhs, &rhs], &[], workers)
                .expect("contraction evaluates");
            let nanos = start.elapsed().as_nanos() as u64;
            checksum = evaluated.root()[0];
            best_nanos = best_nanos.min(nanos);
        }

        let ops_per_second = 2.0 * macs / best_nanos as f64;
        println!(
            "arm={name} size={size} threads={threads} best_ns={best_nanos} \
             gops={ops_per_second:.2} checksum={checksum:.5}"
        );
    }
}
