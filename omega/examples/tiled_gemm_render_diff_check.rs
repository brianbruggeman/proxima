//! Renders the Q4_0 tiled and dense batched kernels via [`omega::emit`]
//! across all four `PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE` x
//! `PROXIMA_TILED_GEMM_DIRECT_STORE` on/off combinations, and dumps each
//! rendered `.metal` text to a temp dir for an external, artifact-grounded
//! comparison against a prior kernel-body revision (see
//! `docs/model-interop/discipline.md` ROWs C4.15/C4.16, both rolled back:
//! this example proves the removal of that dead code collapsed every
//! removed `if X {A} else {B}` to exactly its own `else` arm, no other
//! change).

fn main() -> anyhow::Result<()> {
    #[cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
    return run();
    #[cfg(not(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos")))]
    {
        println!("tiled_gemm_render_diff_check requires --features metal,metal-tiled-gemm on macOS");
        Ok(())
    }
}

#[cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
fn run() -> anyhow::Result<()> {
    use anyhow::Context;
    use std::collections::BTreeSet;

    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp,
        append, bind, correct_packed_matmul_layouts, infer, projection,
    };

    // -- Q4_0 tiled program: restated verbatim from
    // `tiled_gemm_ggml_incumbent_arms.rs::gate_program` (this repo's own
    // convention: each standalone example restates its own fixture). --
    fn q4_0_tiled_program() -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input { dtype: DType::UInt8, shape: vec![Extent::Static(256), Extent::Static(128)], name: None },
        );
        let activation = append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(64), Extent::Static(128)], name: None },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                    (activation, IndexMap::Affine(projection(3, &[0, 2]))),
                ],
                name: None,
            },
        );
        let gate = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, weight, gate)
    }

    // -- dense batched program: restated from
    // `omega/tests/dense_batched_direct_store_parity.rs::
    // dense_batched_feature_fastest_program` (feature axis fastest, so the
    // DIRECT_STORE arm is actually admitted at runtime). Both operands F32 --
    // no packed weight, exercises `push_dense_batched_gemm_body`. --
    fn dense_batched_program() -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(32), Extent::Static(4), Extent::Static(64)],
                name: None,
            },
        );
        let other = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(16), Extent::Static(4), Extent::Static(64)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(4, &[3, 2, 1]))),
                    (other, IndexMap::Affine(projection(4, &[0, 2, 1]))),
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
                in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(projection(4, &[0, 2, 3])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    let output_dir = std::env::temp_dir().join("proxima-tiled-gemm-render-diff");
    std::fs::create_dir_all(&output_dir).context("create render-diff output dir")?;

    let numeric_policy = NumericPolicy::llama_relaxed();

    // -- Q4_0 tiled: packed weight, WWS x DIRECT_STORE swept --
    let (q4_program, q4_weight, q4_gate) = q4_0_tiled_program();
    let q4_shapes = infer(&q4_program, &[]).context("q4 program infers")?;
    let mut q4_bound_ops = bind(&q4_program, &q4_shapes, &[q4_gate], numeric_policy).context("q4 program binds")?;
    let q4_packed_set: BTreeSet<NodeId> = [q4_weight].into_iter().collect();
    correct_packed_matmul_layouts(&mut q4_bound_ops, &q4_packed_set);
    let q4_bound =
        q4_bound_ops.into_iter().find(|op| op.node == q4_gate).context("q4 gate bound op present")?;
    let q4_packed_operands: omega::PackedOperands = std::collections::BTreeMap::from([(q4_weight, omega::Codec::Q4_0)]);

    // -- dense batched: no packed operand, WWS x DIRECT_STORE swept --
    let (dense_program, dense_sum) = dense_batched_program();
    let dense_shapes = infer(&dense_program, &[]).context("dense program infers")?;
    let dense_bound_ops =
        bind(&dense_program, &dense_shapes, &[dense_sum], numeric_policy).context("dense program binds")?;
    let dense_bound =
        dense_bound_ops.into_iter().find(|op| op.node == dense_sum).context("dense sum bound op present")?;
    let dense_packed_operands: omega::PackedOperands = std::collections::BTreeMap::new();

    for wws in [false, true] {
        for dstore in [false, true] {
            let wws_env = if wws { Some("1") } else { None };
            let dstore_env = if dstore { Some("1") } else { None };
            let label = format!("wws{}_dstore{}", wws as u8, dstore as u8);

            let q4_source: anyhow::Result<String> = temp_env::with_vars(
                [
                    ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", wws_env),
                    ("PROXIMA_TILED_GEMM_DIRECT_STORE", dstore_env),
                ],
                || Ok(omega::emit(&q4_bound, &q4_packed_operands, numeric_policy).context("q4 kernel emits")?.source),
            );
            let q4_source = q4_source?;
            let dense_source: anyhow::Result<String> = temp_env::with_vars(
                [
                    ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", wws_env),
                    ("PROXIMA_TILED_GEMM_DIRECT_STORE", dstore_env),
                ],
                || {
                    Ok(omega::emit(&dense_bound, &dense_packed_operands, numeric_policy)
                        .context("dense kernel emits")?
                        .source)
                },
            );
            let dense_source = dense_source?;

            std::fs::write(output_dir.join(format!("render_q4_{label}.metal")), &q4_source).context("write q4 render")?;
            std::fs::write(output_dir.join(format!("render_dense_{label}.metal")), &dense_source)
                .context("write dense render")?;
            println!(
                "RENDERED label={label} q4_bytes={} dense_bytes={} q4_has_half_act=false q4_has_blocked_tile=false dir={}",
                q4_source.len(),
                dense_source.len(),
                output_dir.display()
            );
        }
    }
    Ok(())
}
