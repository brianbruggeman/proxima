use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::{Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer, map};
use proxima_numeric::{DType};

use super::*;

const PREFILL_COMBINE_EXTENTS: [u32; 3] = [1000, 8, 1024];

/// `inputs[a] * inputs[b]` over `extents`, summed across `reduce_axes`: the
/// serial combine fold's shape (two operands, `Add`, `Zero` init), built for
/// any rank so the same fixture covers the sole-axis and multi-axis decode.
fn product_fold_op(extents: &[u32], reduce_axes: &[u16]) -> BoundOp {
    product_fold_op_seeded(extents, reduce_axes, ReduceInit::Zero)
}

/// [`product_fold_op`] with an explicit init: `FirstElement` is the one seed the
/// static-extent body does not own, so it keeps the runtime-extent body under test.
fn runtime_extent_fold_op(extents: &[u32], reduce_axes: &[u16]) -> BoundOp {
    product_fold_op_seeded(extents, reduce_axes, ReduceInit::FirstElement)
}

fn product_fold_op_seeded(extents: &[u32], reduce_axes: &[u16], init: ReduceInit) -> BoundOp {
    let rank = extents.len() as u16;
    let shape: Vec<Extent> = extents
        .iter()
        .map(|&extent| Extent::Static(extent))
        .collect();
    let all_axes: Vec<u16> = (0..rank).collect();
    let output_axes: Vec<u16> = all_axes
        .iter()
        .copied()
        .filter(|axis| !reduce_axes.contains(axis))
        .collect();
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: shape.clone(),
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape,
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(rank, &all_axes))),
                (rhs, IndexMap::Affine(map::projection(rank, &all_axes))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init,
            operand: product,
            in_map: IndexMap::Affine(map::projection(rank, &all_axes)),
            out_map: IndexMap::Affine(map::projection(rank, &output_axes)),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let root = NodeId((program.len() - 1) as u32);
    let shapes = infer(&program, &[]).expect("product fold infers");
    bind(&program, &shapes, &[root], NumericPolicy::default())
        .expect("product fold lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

fn emitted_source(bound: &BoundOp, coord_index32: bool) -> String {
    let knobs = EmitKnobs::default().with_coord_index32(coord_index32);
    emit_with_knobs(bound, &BTreeMap::new(), NumericPolicy::default(), &knobs)
        .expect("serial fold emits")
        .source
}

fn cache_key(bound: &BoundOp) -> String {
    kernel_cache_key_with_knobs(
        bound,
        &BTreeMap::new(),
        NumericPolicy::default(),
        &EmitKnobs::default(),
    )
    .expect("cache key builds")
}

const NARROW_OUTPUT_DECODE: &str = "output_coord[1] = (long)((uint)remaining % (uint)u.output_extents[1]); \
     remaining = (long)((uint)remaining / (uint)u.output_extents[1]);";

#[test]
fn prefill_combine_fold_decodes_outputs_in_32_bit_and_drops_the_sole_axis_divmod() {
    let bound = runtime_extent_fold_op(&PREFILL_COMBINE_EXTENTS, &[1]);

    let source = emitted_source(&bound, true);

    assert!(
        !source.contains("simd_sum"),
        "this shape must take the serial body\n{source}"
    );
    assert!(source.contains(NARROW_OUTPUT_DECODE), "{source}");
    assert!(
        source.contains("output_coord[0] = (long)((uint)remaining % (uint)u.output_extents[0]);")
    );
    assert!(source.contains("reduction_coord[0] = r;"), "{source}");
    assert!(
        !source.contains("remaining_r"),
        "a sole reduction axis needs no running remainder"
    );
    assert!(
        !source.contains("u.reduction_extents[0];"),
        "the identity divmod must be gone"
    );
    assert!(
        !source.contains("remaining % u.output_extents"),
        "no 64-bit modulo survives"
    );
}

#[test]
fn an_iteration_space_past_u32_keeps_the_64_bit_decode() {
    let bound = runtime_extent_fold_op(&[70_000, 8, 70_000], &[1]);

    let source = emitted_source(&bound, true);

    assert!(source.contains(
        "output_coord[1] = remaining % u.output_extents[1]; remaining /= u.output_extents[1];"
    ));
    assert!(!source.contains("(uint)remaining"));
    assert!(
        source.contains("reduction_coord[0] = r;"),
        "the identity elision holds at any width"
    );
}

#[test]
fn the_coordinate_override_set_to_zero_restores_the_64_bit_decode() {
    let bound = runtime_extent_fold_op(&PREFILL_COMBINE_EXTENTS, &[1]);

    let source = emitted_source(&bound, false);

    assert!(source.contains("output_coord[1] = remaining % u.output_extents[1];"));
    assert!(!source.contains("(uint)remaining"));
}

#[test]
fn a_two_axis_reduction_narrows_its_inner_divmod_and_hands_the_outer_axis_the_remainder() {
    let bound = runtime_extent_fold_op(&PREFILL_COMBINE_EXTENTS, &[1, 2]);

    let source = emitted_source(&bound, true);

    assert!(source.contains("long remaining_r = r;"), "{source}");
    assert!(
        source.contains(
            "reduction_coord[1] = (long)((uint)remaining_r % (uint)u.reduction_extents[1]); \
         remaining_r = (long)((uint)remaining_r / (uint)u.reduction_extents[1]);"
        ),
        "{source}"
    );
    assert!(source.contains("reduction_coord[0] = remaining_r;"));
    assert!(!source.contains("remaining_r % u.reduction_extents[0]"));
}

#[test]
fn the_narrow_decode_is_part_of_the_pipeline_identity() {
    let narrow = product_fold_op(&PREFILL_COMBINE_EXTENTS, &[1]);
    let wide = product_fold_op(&[70_000, 8, 70_000], &[1]);

    assert!(
        cache_key(&narrow).contains("_c32"),
        "{}",
        cache_key(&narrow)
    );
    assert!(!cache_key(&wide).contains("_c32"), "{}", cache_key(&wide));
}

fn unflatten(flat: u64, extents: &[u64], narrow: bool) -> Vec<u64> {
    let mut coords = vec![0u64; extents.len()];
    let mut remaining = flat;
    for (slot, extent) in extents.iter().enumerate().rev() {
        if narrow {
            coords[slot] = u64::from((remaining as u32) % (*extent as u32));
            remaining = u64::from((remaining as u32) / (*extent as u32));
        } else {
            coords[slot] = remaining % extent;
            remaining /= extent;
        }
    }
    coords
}

#[test]
fn the_narrow_unflatten_equals_the_wide_one_across_the_captured_output_space() {
    let bound = product_fold_op(&PREFILL_COMBINE_EXTENTS, &[1]);
    let BoundOpKind::Reduce { output_axes, .. } = &bound.kind else {
        panic!("a fold binds as a reduce");
    };
    assert!(serial_reduce_index32_fits(&bound, output_axes));
    let output_extents: Vec<u64> = output_axes
        .iter()
        .map(|&axis| bound.extents[axis as usize])
        .collect();
    let output_total: u64 = output_extents.iter().product();

    let mismatches = (0..output_total)
        .filter(|&flat| {
            unflatten(flat, &output_extents, true) != unflatten(flat, &output_extents, false)
        })
        .count();

    assert_eq!(output_total, 1_024_000);
    assert_eq!(mismatches, 0);
}

#[test]
fn a_static_fold_bakes_its_extents_and_carries_no_runtime_loop_bound_or_output_divmod() {
    let bound = product_fold_op(&[512, 6, 2048], &[1]);

    let source = emitted_source(&bound, true);

    assert!(!source.contains("simd_sum"), "serial body\n{source}");
    assert!(!source.contains("u.reduction_total;"), "{source}");
    assert!(!source.contains("% u.output_extents"), "{source}");
    assert!(!source.contains("u.output_extents["), "{source}");
    assert!(!source.contains("u.reduction_extents"), "{source}");
    assert!(!source.contains("seeded"), "{source}");
    assert!(source.contains("fold0 < 6L"), "{source}");
    assert!(source.contains("(uint)2048L"), "{source}");
    assert!(source.contains("output_coord[0] = remaining;"), "{source}");
    assert!(cache_key(&bound).contains("_sf2048x6"), "{}", cache_key(&bound));
}

#[test]
fn a_static_fold_advances_each_operand_by_its_reduce_stride_instead_of_recomputing_offsets() {
    let bound = product_fold_op(&[512, 6, 2048], &[1]);

    let source = emitted_source(&bound, true);

    assert!(source.contains("run0_0 += u.operand_strides[0][1];"), "{source}");
    assert!(source.contains("run1_0 += u.operand_strides[1][1];"), "{source}");
    assert!(
        !source.contains("off0 += full_coord"),
        "the per-step offset recompute must be gone\n{source}"
    );
}

#[test]
fn a_static_two_axis_fold_nests_one_loop_per_reduced_axis_in_flat_order() {
    let bound = product_fold_op(&PREFILL_COMBINE_EXTENTS, &[1, 2]);

    let source = emitted_source(&bound, true);

    let outer = source.find("fold0 < 8L").expect("outer reduced axis loop");
    let inner = source.find("fold1 < 1024L").expect("inner reduced axis loop");
    assert!(outer < inner, "{source}");
    assert!(!source.contains("remaining_r"), "{source}");
}

#[test]
fn the_baked_extents_are_part_of_the_pipeline_identity() {
    let six = product_fold_op(&[512, 6, 2048], &[1]);
    let eight = product_fold_op(&[512, 8, 2048], &[1]);
    let other_tokens = product_fold_op(&[256, 6, 2048], &[1]);

    assert_ne!(cache_key(&six), cache_key(&eight));
    assert_eq!(
        cache_key(&six),
        cache_key(&other_tokens),
        "the outermost output extent stays a runtime uniform, so token-count changes share a pipeline"
    );
}
