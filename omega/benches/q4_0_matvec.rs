//! ONE Q4_0 matvec shape in isolation: `[rows, k] x [k, 1] -> [rows, 1]`, the
//! program the packed-weight parity tests already build, repeated by criterion
//! instead of measured once inside a whole decode step.
//!
//! this exists because the per-op profile cannot answer "how fast is our matvec":
//! it gives every op its own command buffer (a ~6 us dispatch floor) and the SAME
//! op on the SAME shape reads 1.23-2.77 ms across runs (2.25x), so it is an
//! ordering, not a budget. A criterion median over one shape is the instrument
//! the question actually needs.
//!
//! shapes are chosen around the checkpoint's real packed matvecs rather than at
//! round numbers: 5.31 MB and 10.62 MB bracket the FFN family (ffn_up 5.31 MB,
//! ffn_down 10.67 MB), and 172.9 MB is the LM-head scale of read. `k = 2048` is
//! the checkpoint's embedding width, so every block count is a whole number.
//!
//! run on a quiet machine: `cargo bench -p proxima-omega --bench q4_0_matvec \
//!   --features metal`
//!
//! read the throughput column: bytes moved over time, so it is directly
//! comparable to the ~152 GB/s a llama.cpp decode step averages across all of
//! its own packed reads.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
use std::mem::size_of;

#[cfg(all(feature = "metal", target_os = "macos"))]
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
#[cfg(all(feature = "metal", target_os = "macos"))]
use proxima_gguf::quant::q4_0;
#[cfg(all(feature = "metal", target_os = "macos"))]
use proxima_tensor::{Extent, IndexMap, Keep, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, projection};
#[cfg(all(feature = "metal", target_os = "macos"))]
use proxima_numeric::{DType, NumericPolicy};

/// `[rows, k] x [k, 1] -> [rows, 1]`, the shape [`crate::`]'s packed-weight gate
/// uses: a packed weight operand, an f32 activation, a multiply, and a reduce
/// over the contraction axis.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn q4_0_matvec_program(rows: u32, k: u32) -> (Vec<Op>, proxima_tensor::NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UINT8,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: Some("weight".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(k), Extent::Static(1)],
            name: Some("activation".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("q4_0_matvec".into()),
        }),
    );
    (program, sum)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn bench_q4_0_matvec(criterion: &mut Criterion) {
    // (rows, k, label): bytes = rows * k * Q4_0's 0.5625 bytes per element
    let shapes = [
        (4608_u32, 2048_u32, "5.3MB_ffn_up"),
        (9216, 2048, "10.6MB_ffn_down"),
        (150_144, 2048, "172.9MB_lm_head_scale"),
    ];
    for (rows, k, label) in shapes {
        let (program, root) = q4_0_matvec_program(rows, k);
        let elements = rows as usize * k as usize;
        let mut weight_bytes = vec![0u8; elements / q4_0::QK4_0 * q4_0::BLOCK_BYTES];
        let weight_f32: Vec<f32> = (0..elements)
            .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
            .collect();
        q4_0::quantize(&weight_f32, &mut weight_bytes).expect("a whole number of q4_0 blocks");
        let activation: Vec<f32> = (0..k as usize)
            .map(|index| ((index % 29) as f32 - 14.0) / 29.0)
            .collect();
        let moved = (weight_bytes.len() + activation.len() * size_of::<f32>()) as u64;

        // the weight and activation bind ONCE, so the timed region is the
        // packed matvec dispatch alone -- `omega::execute` per iteration would
        // re-plan and re-upload the weight every call, which measured 10-20 GB/s
        // for every shape regardless of size (a per-call constant amortized, not
        // a matvec rate).
        let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
            (
                "weight",
                QuantizedBlock::Packed {
                    codec: proxima_numeric::codec::Codec::Q4_0,
                    bytes: &weight_bytes,
                },
            ),
            ("activation", QuantizedBlock::Float32(&activation)),
        ];
        let symbols: [u64; 0] = [];
        let roots = [root];
        let plan = omega::plan_named(
            &program,
            &symbols,
            &named,
            &roots,
            NumericPolicy::default(),
        )
        .expect("the packed Q4_0 matvec plans");

        let mut group = criterion.benchmark_group(format!("q4_0_matvec_{label}"));
        group.throughput(Throughput::Bytes(moved));
        group.warm_up_time(std::time::Duration::from_millis(300));
        group.measurement_time(std::time::Duration::from_millis(1500));
        group.bench_function("packed_row_blocked", |bencher| {
            bencher.iter(|| {
                std::hint::black_box(
                    omega::execute_plan_named(&plan, &named)
                        .expect("the planned packed Q4_0 matvec dispatches"),
                )
            });
        });
        group.finish();
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
criterion_group!(benches, bench_q4_0_matvec);
#[cfg(all(feature = "metal", target_os = "macos"))]
criterion_main!(benches);

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("unsupported target for this benchmark or example");
    std::process::exit(1);
}
