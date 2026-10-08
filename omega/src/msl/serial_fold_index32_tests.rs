use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::{
    DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer, map,
};

use super::*;

const PREFILL_COMBINE_EXTENTS: [u32; 3] = [1000, 8, 1024];

/// `inputs[a] * inputs[b]` over `extents`, summed across `reduce_axes`: the
/// serial combine fold's shape (two operands, `Add`, `Zero` init), built for
/// any rank so the same fixture covers the sole-axis and multi-axis decode.
fn product_fold_op(extents: &[u32], reduce_axes: &[u16]) -> BoundOp {
    let rank = extents.len() as u16;
    let shape: Vec<Extent> = extents.iter().map(|&extent| Extent::Static(extent)).collect();
    let all_axes: Vec<u16> = (0..rank).collect();
    let output_axes: Vec<u16> = all_axes
        .iter()
        .copied()
        .filter(|axis| !reduce_axes.contains(axis))
        .collect();
    let mut program = Vec::new();
    let lhs = append(&mut program, Op::Input { dtype: DType::Float32, shape: shape.clone(), name: None });
    let rhs = append(&mut program, Op::Input { dtype: DType::Float32, shape, name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
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
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
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

fn emitted_source(bound: &BoundOp, coord_override: Option<&str>) -> String {
    temp_env::with_var("PROXIMA_COORD_INDEX32", coord_override, || {
        emit(bound, &BTreeMap::new(), NumericPolicy::default())
            .expect("serial fold emits")
            .source
    })
}

fn cache_key(bound: &BoundOp) -> String {
    temp_env::with_var("PROXIMA_COORD_INDEX32", None::<&str>, || {
        kernel_cache_key(bound, &BTreeMap::new(), NumericPolicy::default()).expect("cache key builds")
    })
}

const NARROW_OUTPUT_DECODE: &str = "output_coord[1] = (long)((uint)remaining % (uint)u.output_extents[1]); \
     remaining = (long)((uint)remaining / (uint)u.output_extents[1]);";

#[test]
fn prefill_combine_fold_decodes_outputs_in_32_bit_and_drops_the_sole_axis_divmod() {
    let bound = product_fold_op(&PREFILL_COMBINE_EXTENTS, &[1]);

    let source = emitted_source(&bound, None);

    assert!(!source.contains("simd_sum"), "this shape must take the serial body\n{source}");
    assert!(source.contains(NARROW_OUTPUT_DECODE), "{source}");
    assert!(source.contains("output_coord[0] = (long)((uint)remaining % (uint)u.output_extents[0]);"));
    assert!(source.contains("reduction_coord[0] = r;"), "{source}");
    assert!(!source.contains("remaining_r"), "a sole reduction axis needs no running remainder");
    assert!(!source.contains("u.reduction_extents[0];"), "the identity divmod must be gone");
    assert!(!source.contains("remaining % u.output_extents"), "no 64-bit modulo survives");
}

#[test]
fn an_iteration_space_past_u32_keeps_the_64_bit_decode() {
    let bound = product_fold_op(&[70_000, 8, 70_000], &[1]);

    let source = emitted_source(&bound, None);

    assert!(source.contains("output_coord[1] = remaining % u.output_extents[1]; remaining /= u.output_extents[1];"));
    assert!(!source.contains("(uint)remaining"));
    assert!(source.contains("reduction_coord[0] = r;"), "the identity elision holds at any width");
}

#[test]
fn the_coordinate_override_set_to_zero_restores_the_64_bit_decode() {
    let bound = product_fold_op(&PREFILL_COMBINE_EXTENTS, &[1]);

    let source = emitted_source(&bound, Some("0"));

    assert!(source.contains("output_coord[1] = remaining % u.output_extents[1];"));
    assert!(!source.contains("(uint)remaining"));
}

#[test]
fn a_two_axis_reduction_narrows_its_inner_divmod_and_hands_the_outer_axis_the_remainder() {
    let bound = product_fold_op(&PREFILL_COMBINE_EXTENTS, &[1, 2]);

    let source = emitted_source(&bound, None);

    assert!(source.contains("long remaining_r = r;"), "{source}");
    assert!(source.contains(
        "reduction_coord[1] = (long)((uint)remaining_r % (uint)u.reduction_extents[1]); \
         remaining_r = (long)((uint)remaining_r / (uint)u.reduction_extents[1]);"
    ), "{source}");
    assert!(source.contains("reduction_coord[0] = remaining_r;"));
    assert!(!source.contains("remaining_r % u.reduction_extents[0]"));
}

#[test]
fn the_narrow_decode_is_part_of_the_pipeline_identity() {
    let narrow = product_fold_op(&PREFILL_COMBINE_EXTENTS, &[1]);
    let wide = product_fold_op(&[70_000, 8, 70_000], &[1]);

    assert!(cache_key(&narrow).contains("_c32"), "{}", cache_key(&narrow));
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
    let output_extents: Vec<u64> = output_axes.iter().map(|&axis| bound.extents[axis as usize]).collect();
    let output_total: u64 = output_extents.iter().product();

    let mismatches = (0..output_total)
        .filter(|&flat| unflatten(flat, &output_extents, true) != unflatten(flat, &output_extents, false))
        .count();

    assert_eq!(output_total, 1_024_000);
    assert_eq!(mismatches, 0);
}
