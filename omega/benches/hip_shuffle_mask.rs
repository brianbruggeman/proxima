//! HIP reduce emission at the configurations llama.cpp's HIP lane runs.
//!
//! The defect this bench guards is a compile one: hiprtc 9.x rejects a `_sync` shuffle whose mask
//! is 4 bytes, so every HIP kernel emitted at every configured wave must carry an 8-byte all-lanes
//! value. The two rows are the two wavefront widths llama.cpp's ROCm build distinguishes — 64-lane
//! CDNA (the default) and 32-lane RDNA — both loaded through `RuntimeConfig` (the conflaguration
//! surface) and both bound and emitted with the same `config.numeric` transform `silicon::hip::shared::plan`
//! applies, never a bench-only policy.
//!
//! This is an emission bench because the host has no ROCm device; the device compile is
//! `silicon::hip::rocm_tests::hiprtc_compiles_the_emitted_wave32_and_wave64_masks`, which names its
//! skip when the runtime is absent.
//!
//! `cargo bench -p proxima-omega --no-default-features --features std,cuda \
//!   --bench hip_shuffle_mask -- --test` prints one row per wave.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(feature = "cuda")]
use std::collections::BTreeMap;
#[cfg(feature = "cuda")]
use std::hint::black_box;

#[cfg(feature = "cuda")]
use criterion::{BenchmarkId, Criterion};
#[cfg(feature = "cuda")]
use omega::config::RuntimeConfig;
#[cfg(feature = "cuda")]
use omega::OperandDtypes;
#[cfg(feature = "cuda")]
use omega::cuda::{EmitDevice, HipWave, emit_hip_at_wave};
#[cfg(feature = "cuda")]
use proxima_numeric::DType;
#[cfg(feature = "cuda")]
use proxima_tensor::{
    Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer, map,
};

#[cfg(feature = "cuda")]
fn reduce_program(rows: u32, columns: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(rows), Extent::Static(columns)],
            name: None,
        },
    );
    let root = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: input,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("hip_shuffle_mask_reduce".into()),
        }),
    );
    (program, root)
}

#[cfg(feature = "cuda")]
fn selected_wave(config: &RuntimeConfig) -> HipWave {
    HipWave::from_lanes(u64::from(config.hip_wave)).expect("RuntimeConfig validates 32 or 64")
}

#[cfg(feature = "cuda")]
fn bench_hip_reduce_emission(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hip_reduce_emission");
    let (program, root) = reduce_program(1, 4096);
    let shapes = infer(&program, &[]).expect("the reduce infers");
    let packed = BTreeMap::new();

    for (label, toml) in [
        (
            "llama_cdna_wave64",
            "gpu_driver = 'hip'\nhip_wave = 64\n[numeric]\ncontraction = true\nreassociation = true\n",
        ),
        (
            "llama_rdna_wave32",
            "gpu_driver = 'hip'\nhip_wave = 32\n[numeric]\ncontraction = true\nreassociation = true\n",
        ),
    ] {
        let config: RuntimeConfig = toml::from_str(toml).expect("the bench config deserializes");
        let wave = selected_wave(&config);
        let policy = config.numeric.with_accumulator_narrowing().with_flash_narrowing(true);
        assert!(policy.reassociation, "{label} must take the cooperative reduce");
        let bound = bind(&program, &shapes, &[root], policy)
            .expect("the reduce binds under the llama policy");
        let kernel = emit_hip_at_wave(
            wave,
            &bound[0],
            &packed,
            policy,
            EmitDevice::default(),
            &OperandDtypes::new(),
        )
        .expect("the reduce emits");
        let mask = kernel
            .source
            .split_once("#define PROXIMA_ALL_LANES ")
            .expect("the preamble binds the mask")
            .1
            .split('\n')
            .next()
            .expect("the define is one line");
        println!(
            "hip_shuffle_mask label={label} wave={} block_width={:?} token=PROXIMA_ALL_LANES mask={mask} uses={}",
            wave.lanes(),
            kernel.grid.block_width(),
            kernel.source.matches("PROXIMA_ALL_LANES").count()
        );
        assert!(
            mask.ends_with("ull"),
            "{label} must emit an 8-byte mask, got {mask}"
        );
        assert_eq!(kernel.grid.block_width(), Some(wave.lanes()));

        group.bench_with_input(
            BenchmarkId::new(label, wave.lanes()),
            &wave,
            |bencher, &wave| {
                bencher.iter(|| {
                    let emitted = emit_hip_at_wave(
                        wave,
                        &bound[0],
                        &packed,
                        policy,
                        EmitDevice::default(),
                        &OperandDtypes::new(),
                    )
                    .expect("the reduce emits");
                    black_box(emitted.source.len());
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    bench_hip_reduce_emission(&mut criterion);
    criterion.final_summary();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this benchmark requires the `cuda` feature");
    std::process::exit(1);
}
