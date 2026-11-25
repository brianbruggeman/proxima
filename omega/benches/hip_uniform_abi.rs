//! The HIP/CUDA runtime uniform ABI on llama's own serving configuration.
//!
//! `long` is a host-width type: HIP and CUDA device compilation inherits the
//! aux (host) C data model, so on a 64-bit Windows host the device `long` is 4
//! bytes while `pack_cuda_uniforms` writes 8. Every field after the first then
//! reads four bytes early, extents become zero, and the failure surfaces as a
//! copy error after a launch that already ran.
//!
//! This bench is the conformance instrument, not a toy shape list: the knobs
//! are a `conflaguration::Settings` struct whose defaults are llama.cpp's own
//! `-b 2048 -ub 512 -c 4096 -ctk f16 -ctv f16`, and the widths are flint's real
//! FFN and LM-head matvecs. It emits both card-C dialects with llama's relaxed
//! numerics, packs the host words, and refuses any struct whose word count
//! times the numeric word's width is not the packed length. Run it with:
//!
//! ```sh
//! cargo bench -p proxima-omega --features cuda-driver --bench hip_uniform_abi -- --test
//! ```
//!
//! The line `HIP_UNIFORM run=done mismatch=0 ...` is the acceptance output.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use conflaguration::Settings;
use criterion::{Criterion, criterion_group, criterion_main};
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::{
    BoundOp, BoundOpKind, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind,
    infer, projection,
};
use serde::{Deserialize, Serialize};

use omega::cuda::{
    EmitDevice, HipKernel, HipWave, check_uniform_abi, emit_cuda_with_policy, emit_hip_at_wave,
    pack_cuda_uniforms,
};
use omega::{OperandDtypes, PackedOperands};

fn emit_hip_wave64(op: &BoundOp, empty: &PackedOperands, policy: NumericPolicy) -> HipKernel {
    emit_hip_at_wave(HipWave::Wave64, op, empty, policy, EmitDevice::default(), &OperandDtypes::new())
        .expect("the real matvec emits")
}

/// llama-server's flags plus the flint checkpoint's own matvec widths. Every
/// default here is a value llama.cpp or the GGUF metadata already names; an
/// override comes through `conflaguration`'s `HIP_UNIFORM_*` environment, never
/// a bespoke `std::env::var` read in this file.
#[derive(Debug, Clone, Deserialize, Serialize, Settings)]
#[settings(prefix = "HIP_UNIFORM")]
struct HipUniformConfig {
    /// llama-server `-b` (`common.h` `n_batch`).
    #[setting(default = 2048)]
    batch: u32,
    /// llama-server `-ub` (`common.h` `n_ubatch`).
    #[setting(default = 512)]
    ubatch: u32,
    /// llama-server `-c`.
    #[setting(default = 4096)]
    context: u32,
    /// llama-server `-ctk`/`-ctv`.
    #[setting(default = "f16")]
    kv_cache: String,
    /// flint's embedding width, the `k` of every packed matvec.
    #[setting(default = 2048)]
    hidden: u32,
    /// flint's `feed_forward_length`; `ffn_up.weight` is `[hidden, ffn_width]`
    /// and `ffn_down.weight` is `[ffn_width, hidden]`.
    #[setting(default = 6144)]
    ffn_width: u32,
    /// flint's LM-head vocabulary width.
    #[setting(default = 150144)]
    vocab: u32,
}


fn matvec(rows: u32, depth: u32, tokens: u32) -> BoundOp {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(rows), Extent::Static(depth)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(tokens), Extent::Static(depth)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(projection(3, &[0, 2]))),
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
            name: None,
        }),
    );
    let shapes = infer(&program, &[]).expect("the real matvec infers");
    bind(
        &program,
        &shapes,
        &[sum],
        NumericPolicy::llama_relaxed(),
    )
    .expect("the real matvec binds")
    .into_iter()
    .find(|bound| bound.node == sum)
    .expect("the root is in the lowered program")
}

fn real_config_ops(config: &HipUniformConfig) -> Vec<(String, BoundOp)> {
    let tokens = config.ubatch.min(config.context).max(1);
    [
        ("ffn_up", config.ffn_width, config.hidden),
        ("ffn_down", config.hidden, config.ffn_width),
        ("lm_head", config.vocab, config.hidden),
    ]
    .into_iter()
    .map(|(name, rows, depth)| {
        let op = matvec(rows, depth, tokens);
        let BoundOpKind::Reduce { output_axes, .. } = &op.kind else {
            panic!("{name}: the real matvec is a reduce");
        };
        let output: Vec<u64> = output_axes
            .iter()
            .map(|axis| op.extents[*axis as usize])
            .collect();
        let expected = match name {
            "ffn_up" => vec![u64::from(tokens), u64::from(config.ffn_width)],
            "ffn_down" => vec![u64::from(tokens), u64::from(config.hidden)],
            "lm_head" => vec![u64::from(tokens), u64::from(config.vocab)],
            other => panic!("{other}: no expected output extent"),
        };
        assert_eq!(
            output, expected,
            "{name}: the op's output extent does not match flint's tensor"
        );
        (name.to_string(), op)
    })
    .collect()
}

fn hip_uniform_abi(criterion: &mut Criterion) {
    let config = HipUniformConfig::from_env().expect("HIP_UNIFORM_* settings parse");
    let policy = NumericPolicy::llama_relaxed();
    let ops = real_config_ops(&config);
    let mut mismatches = 0_usize;
    for (name, op) in &ops {
        let empty = PackedOperands::new();
        for (dialect, kernel) in [
            (
                "cuda",
                emit_cuda_with_policy(op, &empty, policy).expect("the real matvec emits"),
            ),
            (
                "hip",
                emit_hip_wave64(op, &empty, policy),
            ),
        ] {
            let host = pack_cuda_uniforms(op).expect("the real matvec packs");
            if let Err(error) = check_uniform_abi(&kernel, &host) {
                eprintln!("{dialect}/{name} is not the uniform word: {error}");
                mismatches += 1;
            }
        }
    }
    println!(
        "HIP_UNIFORM run=start mismatch={mismatches} ops={} batch={} ubatch={} context={} kv={}",
        ops.len(),
        config.batch,
        config.ubatch,
        config.context,
        config.kv_cache
    );
    assert_eq!(mismatches, 0, "a uniform word disagrees with its host pack");

    criterion.bench_function("hip_uniform_abi/emit_and_pack", |bencher| {
        bencher.iter(|| {
            for (_, op) in &ops {
                let empty = PackedOperands::new();
                let hip = emit_hip_wave64(op, &empty, policy);
                let host = pack_cuda_uniforms(op).expect("the real matvec packs");
                let words = check_uniform_abi(&hip, &host).expect("the real matvec keeps the contract");
                std::hint::black_box((hip.source.len(), words, host.len()));
            }
        });
    });

    println!(
        "HIP_UNIFORM run=done mismatch={mismatches} ops={} batch={} ubatch={} context={} kv={}",
        ops.len(),
        config.batch,
        config.ubatch,
        config.context,
        config.kv_cache
    );
}

criterion_group!(benches, hip_uniform_abi);
criterion_main!(benches);
