//! [`omega::metal::KindFilter`] (private; exercised only through
//! [`omega::execute_plan_with_placements`]'s public `PROXIMA_METAL_KIND_FILTER`
//! knob) gained a `family:<substring>` term (ROW 310) alongside its
//! original bare-`kind:` substring term, plus two `plan`-time error checks
//! ROW 308 found missing: an unrecognized `kind:` term and a filter that
//! removes zero (or every) dispatch from the plan it is applied to. This
//! file proves both the new selector and both new errors against a real
//! Metal device, never a stub.
//!
//! The program built below mirrors a real forward pass's own shape (a
//! per-layer weight multiplied against a shared activation, then reduced --
//! `packed_row_blocked_s1_byte_identity.rs`'s own `matmul_program`), except
//! every weight carries a real `blk.{layer}.{family}.weight` name so
//! [`omega::metal::weight_family`]'s aggregation has something to strip.
//! Every weight and the shared activation are filled with `1.0`, so a
//! correctly-dispatched reduce always reads back [`IN_DIM`] and a SKIPPED
//! reduce (its dispatch dropped by the filter, per
//! `register_skipped_output`'s own doc) reads back whatever a fresh Metal
//! buffer holds instead -- `0.0`, never [`IN_DIM`] -- which is what proves a
//! dispatch was actually removed, not just that the filter parsed.

#![cfg(all(feature = "metal-output-placement", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, projection,
};

/// `PROXIMA_METAL_KIND_FILTER` is process-global; this file's own three
/// tests all set it, and `cargo test`'s default multi-threaded runner (as
/// opposed to `cargo nextest run`'s one-process-per-test isolation) would
/// otherwise race two tests' env values against each other. Held for the
/// full set-run-clear span in [`ran_flags`].
static ENV_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

const IN_DIM: u32 = 8;
const OUT_DIM: u32 = 2;
const FFN_UP_LAYERS: u32 = 32;
const ATTN_Q_LAYERS: u32 = 4;
const EXPECTED_SUM: f32 = IN_DIM as f32;

/// `activation -> [named weight_i -> product_i -> sum_i for i in families]`,
/// one independent matvec-shaped dispatch per named weight, sharing one
/// activation input. Returns the program, `(node, family)` for every `sum`
/// output in append order (so [`omega::plan`] keeps every dispatch live),
/// and one caller-ordered `QuantizedBlock::Float32` per `Input` node in the
/// exact order `append` created them (activation first, then one weight per
/// family member) -- every value `1.0`, so `EXPECTED_SUM` is the only
/// correct readback.
/// (program, one `(output node, family)` per dispatch in append order, one
/// `Vec<f32>` per `Input` node in append order) -- named only so
/// [`named_family_program`]'s return type does not trip clippy's
/// `type_complexity` lint.
type FamilyProgram = (Vec<Op>, Vec<(NodeId, &'static str)>, Vec<Vec<f32>>);

fn named_family_program(family_counts: &[(&'static str, u32)]) -> FamilyProgram {
    let mut program = Vec::new();
    let mut blocks: Vec<Vec<f32>> = Vec::new();

    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(IN_DIM)],
            name: None,
        },
    );
    blocks.push(vec![1.0; IN_DIM as usize]);

    let mut outputs = Vec::new();
    for (family, layers) in family_counts {
        for layer in 0..*layers {
            let weight = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: vec![Extent::Static(OUT_DIM), Extent::Static(IN_DIM)],
                    name: Some(format!("blk.{layer}.{family}.weight")),
                },
            );
            blocks.push(vec![1.0; (OUT_DIM * IN_DIM) as usize]);

            let product = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
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
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: product,
                    in_map: IndexMap::Affine(projection(2, &[0, 1])),
                    out_map: IndexMap::Affine(projection(2, &[0])),
                    keep: Keep::Reduce,
                    name: None,
                }),
            );
            outputs.push((sum, *family));
        }
    }
    (program, outputs, blocks)
}

/// Runs `family_counts`' program once under `env_value` (restored to unset
/// afterward) and returns, per output node, whether its readback equals
/// [`EXPECTED_SUM`] -- `true` means that dispatch ran, `false` means the
/// filter skipped it.
fn ran_flags(
    family_counts: &[(&'static str, u32)],
    env_value: Option<&str>,
) -> Result<Vec<(&'static str, bool)>, omega::metal::MetalError> {
    let (program, outputs, blocks) = named_family_program(family_counts);
    let quantized: Vec<QuantizedBlock<'_>> = blocks
        .iter()
        .map(|values| QuantizedBlock::Float32(values.as_slice()))
        .collect();
    let output_nodes: Vec<NodeId> = outputs.iter().map(|(node, _)| *node).collect();
    let plan =
        omega::plan(&program, &[], &quantized, &output_nodes).expect("plans the named family program");

    // `ENV_SERIAL` covers the whole set/run/clear span: `PROXIMA_METAL_KIND_FILTER`
    // is process-global, and this file's tests race it under a
    // multi-threaded runner otherwise (`cargo nextest run`'s
    // one-process-per-test isolation would not need this, but this file
    // does not assume its own runner).
    let _guard = ENV_SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    match env_value {
        // SAFETY: no other thread touches this process's environment while
        // `_guard` is held.
        Some(value) => unsafe { std::env::set_var("PROXIMA_METAL_KIND_FILTER", value) },
        None => unsafe { std::env::remove_var("PROXIMA_METAL_KIND_FILTER") },
    }
    let result = omega::execute_plan_with_placements(&plan, &quantized, &[], &[]);
    // SAFETY: same as above, still under `_guard`.
    unsafe { std::env::remove_var("PROXIMA_METAL_KIND_FILTER") };

    let evaluated = result?;
    Ok(outputs
        .into_iter()
        .map(|(node, family)| {
            let ran = evaluated
                .get(node)
                .is_some_and(|(values, _shape)| values.first().copied() == Some(EXPECTED_SUM));
            (family, ran)
        })
        .collect())
}

#[test]
fn family_exclusion_drops_exactly_the_named_family_layer_count() {
    let family_counts = [("ffn_up", FFN_UP_LAYERS), ("attn_q", ATTN_Q_LAYERS)];
    let flags = ran_flags(&family_counts, Some("!family:ffn_up")).expect("family exclusion runs");

    let dropped = flags.iter().filter(|(_, ran)| !ran).count();
    let ffn_up_dropped = flags
        .iter()
        .filter(|(family, ran)| *family == "ffn_up" && !ran)
        .count();
    let attn_q_kept = flags
        .iter()
        .filter(|(family, ran)| *family == "attn_q" && *ran)
        .count();

    assert_eq!(
        dropped, FFN_UP_LAYERS as usize,
        "excluding `family:ffn_up` must drop exactly {FFN_UP_LAYERS} dispatches total"
    );
    assert_eq!(
        ffn_up_dropped, FFN_UP_LAYERS as usize,
        "every one of the {FFN_UP_LAYERS} ffn_up layers must be the dropped ones"
    );
    assert_eq!(
        attn_q_kept, ATTN_Q_LAYERS as usize,
        "all {ATTN_Q_LAYERS} attn_q dispatches must still have run"
    );
}

#[test]
fn unknown_kind_term_is_a_typed_error_not_a_silent_full_drop() {
    let family_counts = [("ffn_up", 2)];
    let error = ran_flags(&family_counts, Some("!not_a_real_kind"))
        .expect_err("a kind term outside classify_kind's vocabulary must error");
    assert!(
        matches!(
            error,
            omega::metal::MetalError::UnknownKindFilterTerm { .. }
        ),
        "expected UnknownKindFilterTerm, got {error:?}"
    );
}

#[test]
fn family_typo_matching_zero_dispatches_is_a_typed_error() {
    let family_counts = [("ffn_up", 2)];
    let error = ran_flags(&family_counts, Some("!family:not_a_real_family"))
        .expect_err("a family term matching nothing in this plan must error");
    assert!(
        matches!(
            error,
            omega::metal::MetalError::KindFilterMatchesNothing { .. }
        ),
        "expected KindFilterMatchesNothing, got {error:?}"
    );
}
