//! The Metal source of a twin-output elementwise, over the op `fuse_twin_elementwise` builds from a
//! real `spec::fused_rope_pair` program: what the kernel declares, what it reads, and that its
//! extra output lands at the buffer index `crate::metal::encode_op` binds an op's extra outputs at.

use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::spec::{RopePairing, fused_rope_pair, input_leaf};
use proxima_tensor::{DType, Extent, NumericPolicy, Op, bind, fuse_twin_elementwise, infer};

use super::*;

const SEQUENCE: u32 = 3;
const HEADS: u32 = 2;
const PAIRS: u32 = 4;

fn rope_ops(head_dim: u32, pairing: RopePairing) -> (Vec<BoundOp>, Vec<Op>) {
    let mut program = Vec::new();
    let source = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(SEQUENCE),
            Extent::Static(HEADS),
            Extent::Static(head_dim),
        ],
        "x",
    );
    let trig = || vec![Extent::Static(SEQUENCE), Extent::Static(PAIRS)];
    let cosine = input_leaf(&mut program, DType::Float32, trig(), "cos");
    let sine = input_leaf(&mut program, DType::Float32, trig(), "sin");
    let (first, second) = fused_rope_pair(&mut program, source, 'h', cosine, sine, pairing)
        .expect("rope pair builds");
    let shapes = infer(&program, &[]).expect("rope program infers");
    let bound = bind(&program, &shapes, &[first, second], NumericPolicy::default())
        .expect("rope program binds");
    (bound, program)
}

fn twin_kernel(head_dim: u32, pairing: RopePairing) -> (BoundOp, Kernel) {
    let (bound, program) = rope_ops(head_dim, pairing);
    let mut fused = fuse_twin_elementwise(bound, &program);
    assert_eq!(fused.len(), 1, "the rope pair fuses to one op");
    let twin = fused.remove(0);
    let kernel = emit(&twin, &PackedOperands::new(), NumericPolicy::default())
        .expect("a twin elementwise renders");
    (twin, kernel)
}

fn extra_output_buffer_index(source: &str) -> usize {
    let after = source
        .split("extra_out0 [[buffer(")
        .nth(1)
        .expect("the kernel declares extra_out0");
    after
        .split(")]]")
        .next()
        .expect("the buffer attribute closes")
        .parse()
        .expect("the buffer index is a number")
}

#[proxima::test]
#[case::split_half(8, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(8, RopePairing::Interleaved)]
async fn the_second_output_is_bound_right_past_the_kernels_own_bindings(
    #[case] head_dim: u32,
    #[case] pairing: RopePairing,
) {
    let (twin, kernel) = twin_kernel(head_dim, pairing);

    let declared = extra_output_buffer_index(&kernel.source);

    assert_eq!(declared, kernel.bindings.len(), "encode_op binds extras at bindings.len()");
    assert_eq!(kernel.bindings.len(), twin.operands().len() + 2, "inputs, output, uniforms");
    assert!(kernel.bindings.contains(&Binding::Output(twin.node)));
    assert!(
        !kernel
            .bindings
            .contains(&Binding::Output(twin.twin_node().expect("twin node"))),
        "the twin node is an extra output, not a named binding"
    );
}

#[proxima::test]
#[case::split_half(8, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(8, RopePairing::Interleaved)]
async fn one_kernel_reads_each_operand_once_and_stores_two_outputs(
    #[case] head_dim: u32,
    #[case] pairing: RopePairing,
) {
    let (twin, kernel) = twin_kernel(head_dim, pairing);

    let operand_reads = kernel
        .source
        .lines()
        .filter(|line| line.starts_with("    scratch["))
        .count();
    let primary_stores = kernel.source.matches("out[gid] = ").count();
    let twin_stores = kernel.source.matches("extra_out0[gid] = ").count();

    assert_eq!(operand_reads, twin.operands().len(), "one read per shared operand");
    assert_eq!(primary_stores, 1);
    assert_eq!(twin_stores, 1);
    assert_eq!(
        kernel.grid.threads,
        u64::from(SEQUENCE * HEADS * PAIRS),
        "one thread per rotated element, for both outputs"
    );
    assert!(kernel.entry.starts_with("omega_elementwise_twin_"));
}

#[proxima::test]
async fn a_plain_elementwise_kernel_declares_no_extra_output() {
    let (bound, _) = rope_ops(8, RopePairing::SplitHalf { pairs: PAIRS });

    let kernel = emit(&bound[0], &PackedOperands::new(), NumericPolicy::default())
        .expect("a plain elementwise renders");

    assert!(!kernel.source.contains("extra_out0"));
    assert_eq!(kernel.bindings.len(), bound[0].operands().len() + 2);
}

#[proxima::test]
async fn the_twin_entry_name_differs_from_both_plain_halves_it_replaces() {
    let (twin, kernel) = twin_kernel(8, RopePairing::SplitHalf { pairs: PAIRS });
    let (primary, secondary) = twin.twin_halves().expect("a twin splits");

    let primary_kernel = emit(&primary, &PackedOperands::new(), NumericPolicy::default())
        .expect("the plain primary renders");
    let secondary_kernel = emit(&secondary, &PackedOperands::new(), NumericPolicy::default())
        .expect("the plain secondary renders");

    assert_ne!(kernel.entry, primary_kernel.entry);
    assert_ne!(kernel.entry, secondary_kernel.entry);
    assert_ne!(primary_kernel.entry, secondary_kernel.entry, "different bodies, different kernels");
}
