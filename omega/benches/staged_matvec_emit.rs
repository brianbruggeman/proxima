//! The staged-matvec emit path and the shared `LaunchGrid` representation at
//! flint's real packed shapes.
//!
//! the instrument times two arms per shape: a live staged-matvec emit and the
//! same shape with an empty output (the idle representation). together they
//! show the representation the card introduced costs nothing beside the emit it
//! sits in, and that the empty path never reaches a launch.
//!
//! configuration comes from `RuntimeConfig` (conflaguration), numerics from
//! `NumericPolicy` (proxima-numeric); the policy is the llama dp4a lane
//! (`llama_relaxed` plus activation quantization), not a benchmark-only arm.
//!
//! `git wt cargo bench -p proxima-omega --features std,cuda --bench staged_matvec_emit`

#![allow(clippy::expect_used)] // a failed emit in a bench is a broken setup, not a sample

#[cfg(all(feature = "std", feature = "cuda"))]
mod bench {
    use std::time::Duration;

    use criterion::{Criterion, Throughput};
    use proxima_numeric::NumericPolicy;
    use proxima_tensor::{
        Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer, projection,
    };
    use omega::{Codec, PackedOperands, config::RuntimeConfig, emit_cuda_with_policy};

    const SHAPES: [(u32, u32, &str); 3] = [
        (6144, 2048, "flint_ffn_up_gate"),
        (2048, 6144, "flint_ffn_down"),
        (150_144, 2048, "flint_lm_head"),
    ];

    fn llama_serving_policy() -> NumericPolicy {
        NumericPolicy::llama_relaxed().with_activation_quantization(true)
    }

    fn q4_0_matvec_program(rows: u32, k: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: proxima_numeric::DType::UINT8,
                shape: vec![Extent::Static(rows), Extent::Static(k)],
                name: Some("weight".into()),
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: proxima_numeric::DType::FLOAT32,
                shape: vec![Extent::Static(k)],
                name: Some("activation".into()),
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: proxima_numeric::DType::FLOAT32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(2, &[0, 1]))),
                    (activation, IndexMap::Affine(projection(2, &[1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: proxima_numeric::DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(2, &[0, 1])),
                out_map: IndexMap::Affine(projection(2, &[0])),
                keep: Keep::Reduce,
                name: Some("q4_0_matvec".into()),
            }),
        );
        (program, sum)
    }

    fn staged_matvec(
        rows: u32,
        k: u32,
        policy: NumericPolicy,
    ) -> (proxima_tensor::BoundOp, PackedOperands) {
        let (program, root) = q4_0_matvec_program(rows, k);
        let shapes = infer(&program, &[]).expect("the matvec shape infers");
        let bound = bind(&program, &shapes, &[root], policy)
            .expect("the matvec binds")
            .into_iter()
            .next()
            .expect("one bound matvec");
        let mut packed = PackedOperands::new();
        packed.insert(bound.operands()[0].0, Codec::Q4_0);
        (bound, packed)
    }

    pub(crate) fn bench_staged_matvec_emit(criterion: &mut Criterion) {
        let config = RuntimeConfig::from_env().expect("the runtime configuration loads");
        let policy = llama_serving_policy();
        println!(
            "config_source=RuntimeConfig::from_env engine={} gpu_driver={} sample_size={}",
            config.engine,
            config.gpu_driver,
            config.selection.resolved_measurement_runs(),
        );
        println!("numeric_policy=llama_relaxed activation_quantization={}", policy.activation_quantization);
        let sample_size = usize::from(config.selection.resolved_measurement_runs());
        println!("criterion_sample_size={sample_size} (criterion's floor is 10)");

        for (rows, k, label) in SHAPES {
            let (live, packed) = staged_matvec(rows, k, policy);
            let kernel = emit_cuda_with_policy(&live, &packed, policy).expect("the staged matvec emits");
            assert!(!kernel.grid.is_idle(), "{label} must launch");
            let (empty, empty_packed) = staged_matvec(0, k, policy);
            let empty_kernel =
                emit_cuda_with_policy(&empty, &empty_packed, policy).expect("the empty shape emits");
            assert!(empty_kernel.grid.is_idle(), "{label} empty shape is idle");
            println!(
                "shape={label} rows={rows} k={k} threads={} blocks={} source_bytes={} idle_threads={}",
                kernel.grid.threads(),
                kernel.grid.blocks(),
                kernel.source.len(),
                empty_kernel.grid.threads(),
            );

            let moved = kernel.source.len() as u64;
            let mut emit_group = criterion.benchmark_group(format!("staged_matvec_emit_{label}"));
            emit_group.throughput(Throughput::Bytes(moved));
            emit_group.sample_size(sample_size.max(10));
            emit_group.warm_up_time(Duration::from_millis(250));
            emit_group.measurement_time(Duration::from_millis(1_000));
            emit_group.bench_function("live", |bencher| {
                bencher.iter(|| {
                    std::hint::black_box(
                        emit_cuda_with_policy(&live, &packed, policy)
                            .expect("the staged matvec emits"),
                    )
                });
            });
            emit_group.bench_function("idle", |bencher| {
                bencher.iter(|| {
                    std::hint::black_box(
                        emit_cuda_with_policy(&empty, &empty_packed, policy)
                            .expect("the empty staged matvec emits"),
                    )
                });
            });
            emit_group.finish();
        }
    }
}

#[cfg(all(feature = "std", feature = "cuda"))]
use criterion::{criterion_group, criterion_main};

#[cfg(all(feature = "std", feature = "cuda"))]
criterion_group!(benches, bench::bench_staged_matvec_emit);
#[cfg(all(feature = "std", feature = "cuda"))]
criterion_main!(benches);
