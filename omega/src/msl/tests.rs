use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::{
    AxisTerm, DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind,
    infer, map,
};

use super::*;

/// The last node `program` builds -- what every fixture in this module
/// treats as "the answer" by construction (`proxima_tensor::op`'s own
/// doc: "the last element is the root"). `bind_plain`'s reachability
/// pass (ROW 541, `proxima-tensor/docs/discipline.md`) binds only what
/// `outputs` names, so a fixture that wants its whole constructed chain
/// bound must pass this instead of `&[]` -- an empty `outputs`
/// correctly binds nothing.
fn terminal(program: &[Op]) -> NodeId {
    NodeId((program.len() - 1) as u32)
}

#[test]
fn dense_layout_accepts_contiguous_and_rejects_broadcast() {
    let dense = Layout {
        base: 0,
        strides: vec![8_i64, 4, 1].into(),
    };
    assert!(dense_layout(&dense, &[2, 2, 4]));

    let broadcast = Layout {
        base: 0,
        strides: vec![0_i64, 4, 1].into(),
    };
    assert!(!dense_layout(&broadcast, &[2, 2, 4]));
}

#[test]
fn elementwise_broadcast_decodes_omitted_axes_before_selected_axes() {
    let mut program = Vec::new();
    let matrix = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(7), Extent::Static(256)],
            name: None,
        },
    );
    let row = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(7)],
            name: None,
        },
    );
    let equal = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Equal,
            operands: vec![
                (matrix, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (row, IndexMap::Affine(map::projection(2, &[0]))),
            ],
            name: None,
        },
    );
    let shapes = infer(&program, &[]).expect("broadcast elementwise infers");
    let bound = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("broadcast elementwise lowers")
        .into_iter()
        .find(|candidate| candidate.node == equal)
        .expect("broadcast elementwise bound op");
    let source = render_elementwise(&bound, "broadcast_test", &[None, None])
        .expect("broadcast elementwise renders");
    let divide_axis_one = "remaining /= (uint)u.extents[1];";
    let assign_axis_zero = "coord[0] = remaining % (uint)u.extents[0];";
    assert!(source.find(divide_axis_one) < source.find(assign_axis_zero));
}

fn elementwise_tanh_op(extent: u32) -> BoundOp {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let shapes = infer(&program, &[]).expect("elementwise infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("elementwise lowers")
        .into_iter()
        .next()
        .expect("one bound emitted")
}

fn matmul_op(m: u32, k: u32, n: u32) -> BoundOp {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("matmul infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("matmul lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

fn gathered_matmul_op(tokens: u32, experts: u32, rows: u32, k: u32) -> BoundOp {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(experts),
                Extent::Static(rows),
                Extent::Static(k),
            ],
            name: None,
        },
    );
    let route = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(tokens)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(tokens), Extent::Static(k)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (
                    weight,
                    IndexMap::Computed {
                        indices: route,
                        index_map: map::projection(3, &[0]),
                        base: map::IndexPattern {
                            iter_rank: 3,
                            axes: vec![
                                map::AxisIndex::default(),
                                map::AxisIndex {
                                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                                    offset: 0,
                                    len: None,
                                },
                                map::AxisIndex {
                                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                                    offset: 0,
                                    len: None,
                                },
                            ],
                        },
                        gathered_dim: 0,
                    },
                ),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let shapes = infer(&program, &[]).expect("gathered matmul infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("gathered matmul lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

#[expect(
    dead_code,
    reason = "fixture retained for the rejected physical-axis permutation case"
)]
fn permuted_reduction_matmul_op() -> BoundOp {
    let mut program = Vec::new();
    let gated = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(64), Extent::Static(2), Extent::Static(2)],
            name: None,
        },
    );
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(3), Extent::Static(256)],
            name: None,
        },
    );
    let product = proxima_tensor::spec::elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gated, "jug->jugd"), (weight, "d,4*j+2*u+g->jugd")],
    )
    .expect("permuted product builds");
    proxima_tensor::spec::reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "jugd->jugd",
        "d->jugd",
    )
    .expect("permuted reduce builds");
    let shapes = infer(&program, &[]).expect("permuted reduction matmul infers");
    let mut resolved = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("permuted reduction matmul lowers");
    let mut packed = BTreeSet::new();
    packed.insert(weight);
    proxima_tensor::correct_packed_matmul_layouts(&mut resolved, &packed);
    resolved
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

fn cached_attention_op() -> BoundOp {
    let operands = (0..8)
        .map(|index| {
            (
                NodeId(index),
                Layout {
                    base: 0,
                    strides: vec![1].into(),
                },
                None,
            )
        })
        .collect();
    BoundOp {
        node: NodeId(8),
        dtype: DType::Float32,
        extents: vec![1, 1, 1, 4],
        kind: BoundOpKind::CachedAttention {
            operands,
            query_rows: 1,
            cached_key_rows: 1,
            new_key_rows: 1,
            kv_heads: 1,
            query_groups: 1,
            head_dim: 4,
            rotary_dim: 4,
            scale: 0.5,
            cached_lower_inclusive: i64::MIN,
            new_upper_inclusive: 0,
        },
    }
}

/// The single-range fused (nine-operand, `single_range_dynamic`) sibling
/// of [`cached_attention_op`] -- the ninth operand carries the live
/// `cached_len` at run time (`BoundOpKind::CachedAttention`'s own doc),
/// and that shape is real, structural `cached_key_rows == 0`
/// (`cached_attention_single_range_candidates`'s own bind-time
/// construction folds the WHOLE context into the "new" slot). The two
/// `u64` parameters here are kept as the COMPILED `kv-capacity-bucket`
/// extent's two halves for every existing call site's own doc/comments
/// ("capacity N") to stay accurate, but both fold into `new_key_rows`
/// alone so the constructed op matches the real single-range invariant
/// -- a test can still vary their SUM freely to prove `entry_name`/the
/// rendered body do not key on it (redesign §5 option 2), since neither
/// field is ever baked as a literal on this path (`row_count_decl`'s own
/// `single_range_dynamic` arm reads both off `u.cached_key_rows`/
/// `u.new_key_rows` at run time instead).
fn cached_attention_op_dynamic(cached_key_rows: u64, new_key_rows: u64) -> BoundOp {
    let operands = (0..9)
        .map(|index| {
            (
                NodeId(index),
                Layout {
                    base: 0,
                    strides: vec![1].into(),
                },
                None,
            )
        })
        .collect();
    BoundOp {
        node: NodeId(9),
        dtype: DType::Float32,
        extents: vec![1, 1, 1, 4],
        kind: BoundOpKind::CachedAttention {
            operands,
            query_rows: 1,
            cached_key_rows: 0,
            new_key_rows: cached_key_rows + new_key_rows,
            kv_heads: 1,
            query_groups: 1,
            head_dim: 4,
            rotary_dim: 4,
            scale: 0.5,
            cached_lower_inclusive: i64::MIN,
            new_upper_inclusive: 0,
        },
    }
}

/// Same shape as [`matmul_op`] but with a caller-chosen reduce op, so a
/// test can hold the fused `weight * activation` body fixed and vary only
/// `reduce_op` — the one axis [`is_plain_product_reduce`] gates on beyond
/// the body shape itself.
fn matmul_op_with_reduce(m: u32, k: u32, n: u32, reduce_op: ScalarOp) -> BoundOp {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: reduce_op,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul_reduce".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("matmul infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("matmul lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

/// [`matmul_op`]'s `Float16` counterpart -- `q4k_pair_dot`'s own
/// `plain_product` arm (`push_packed_row_blocked_body`'s own gate) is
/// `DType::Float32`-only, so a fixture that needs to reach a
/// DIFFERENT row-blocked Q4_K arm (mask-fma's, single-fetch's) must NOT
/// be plain-`Float32`-shaped, or `q4k_pair_dot` wins over it every time.
#[cfg(all(feature = "metal-q4k-single-fetch", not(feature = "metal-q4k-split-k")))]
fn matmul_op_f16(m: u32, k: u32, n: u32) -> BoundOp {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float16,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float16,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float16,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float16,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul_f16".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("f16 matmul infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("f16 matmul lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

#[test]
fn q4k_row_blocked_matmul_uses_paired_nibble_decode() {
    // 256 == Q4K_BLOCK_ELEMENTS exactly: one super-block, so
    // packed_row_block matches and this is the real matmul shape the
    // the paired decode path exists for (`docs/discipline.md` ROW 257).
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    assert!(
        packed_row_block(&bound, &operand_codecs(&bound, &q4k)).is_some(),
        "test fixture must actually take the row-blocked path for this assertion to mean anything"
    );

    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        source.contains("q4k_pair_dot"),
        "Add-reduce over a plain weight*activation body must take the paired decode path:\n{source}"
    );
    assert!(
        !source.contains("hdr.scale * levels[j] - hdr.minimum"),
        "the per-element dequant expression must not remain once the scale-deferred path is taken:\n{source}"
    );
}

/// Gate (1) of `docs/discipline.md`'s horizontal-packed-merge design
/// note: the merged `mv_row_blocked_z` kernel's text must differ from the
/// unmerged kernel's ONLY by the base-table preamble
/// ([`splice_horizontal_merge_base_table`]'s own doc), and its
/// `kernel_identity` must carry `_z8` so it can never share a
/// `PIPELINE_CACHE` entry with the unmerged N=1 kernel.
#[cfg(feature = "metal-horizontal-merge")]
mod horizontal_merge_base_table_splice_tests {
    use alloc::collections::BTreeMap;

    use proxima_tensor::NumericPolicy;

    use super::super::{EmitError, Codec, emit, splice_horizontal_merge_base_table};
    use super::matmul_op;
    use crate::identity::{KernelLanguage, MetalOnlyExtras, kernel_identity};

    /// The exact literal `splice_horizontal_merge_base_table` inserts --
    /// kept here, spelled out, rather than re-deriving it by calling the
    /// function again, so the assertion below is a genuine round-trip
    /// proof and not a tautology.
    const STRUCT_DECL: &str =
        "struct SliceBase { ulong weight_base; ulong activation_base; ulong output_base; };\n";

    #[test]
    fn merged_kernel_text_differs_from_unmerged_only_by_the_base_table_preamble()
    -> Result<(), EmitError> {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, Codec::Q4K);

        let unmerged = emit(&bound, &q4k, NumericPolicy::default()).expect("unmerged emits");
        let mut merged = unmerged.clone();
        splice_horizontal_merge_base_table(
            &mut merged,
            bound.node,
            0,
            "uchar",
            1,
            "float",
            "float",
        )?;

        assert!(
            merged.source.contains(STRUCT_DECL),
            "merged kernel must declare SliceBase:\n{}",
            merged.source
        );
        assert!(
            merged.source.contains("base_table[merge_gid.z]"),
            "merged kernel must index the base table by the z grid coordinate:\n{}",
            merged.source
        );
        // Metal rejects a signature mixing a scalar and a vector
        // thread-position attribute, so the splice widens the existing
        // `gid` parameter to `uint3 merge_gid` instead of adding a
        // second, separately-attributed parameter.
        assert!(
            !merged.source.contains("uint gid [[thread_position_in_grid]]"),
            "the merged kernel must not keep the scalar gid parameter:\n{}",
            merged.source
        );
        assert!(
            merged
                .source
                .contains("uint3 merge_gid [[thread_position_in_grid]]"),
            "the merged kernel must declare the widened vector gid parameter:\n{}",
            merged.source
        );

        let vector_gid = "uint3 merge_gid [[thread_position_in_grid]]";
        let scalar_gid = "uint gid [[thread_position_in_grid]]";
        let extra_params = ",\n    device const SliceBase* base_table [[buffer(4)]]";
        let preamble = "    uint gid = merge_gid.x;\n    SliceBase merge_base = base_table[merge_gid.z];\n    device const uchar* sliced_weight = (device const uchar*)((device const uchar*)in0 + merge_base.weight_base);\n    device const float* sliced_other = (device const float*)((device const uchar*)in1 + merge_base.activation_base);\n    device float* sliced_out = (device float*)((device uchar*)out + merge_base.output_base);\n";

        let mut restored = merged.source.replacen(STRUCT_DECL, "", 1);
        restored = restored.replacen(extra_params, "", 1);
        restored = restored.replacen(preamble, "", 1);
        restored = restored.replacen(vector_gid, scalar_gid, 1);
        restored = restored.replace("sliced_weight", "in0");
        restored = restored.replace("sliced_other", "in1");
        restored = restored.replace("sliced_out", "out");

        assert_eq!(
            restored, unmerged.source,
            "reversing the splice's known insertions and renames must exactly recover the \
             unmerged kernel text -- any other diff is an UNDOCUMENTED change to the body"
        );
        Ok(())
    }

    #[test]
    fn merged_identity_carries_a_z8_suffix_the_unmerged_identity_never_carries() {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, Codec::Q4K);
        let policy = NumericPolicy::default();

        let unmerged_extras = MetalOnlyExtras::default();
        let merged_extras = MetalOnlyExtras {
            merged_z: Some(8),
            ..MetalOnlyExtras::default()
        };

        let unmerged_identity =
            kernel_identity(KernelLanguage::Metal, &bound, &q4k, unmerged_extras, policy);
        let merged_identity =
            kernel_identity(KernelLanguage::Metal, &bound, &q4k, merged_extras, policy);

        assert!(
            merged_identity.contains("_z8"),
            "a size-8 merge group must carry _z8 in its identity: {merged_identity}"
        );
        assert_eq!(
            merged_identity,
            format!("{unmerged_identity}_z8"),
            "the merged identity must be the unmerged identity plus exactly the _z8 suffix, \
             so a size-8 merge never collides with the unmerged N=1 kernel's cache entry"
        );
    }
}

/// A `RoundBatchedReduce` fixture: `gathered_matmul_op`'s own gathered
/// reduce, hand-wrapped into a `round_count`-round fold the way
/// `bind::apply_moe_round_group_fusion` would have collapsed it. A real
/// round-batched fold names `round_count` DISTINCT sibling route nodes in
/// `round_routes` -- this fixture reuses one `NodeId` for every round
/// since [`splice_round_batched_reduce_base_table`] only renders TEXT from
/// the gather SLOT number, never from which concrete `NodeId` each round
/// names.
#[cfg(feature = "metal-moe-mul-mat-id")]
fn round_batched_matmul_op(round_count: u32) -> BoundOp {
    let BoundOp {
        node,
        dtype,
        extents,
        kind,
    } = gathered_matmul_op(4, 8, 16, 256);
    let BoundOpKind::Reduce {
        element_body,
        reduce_op,
        init,
        keep,
        operands,
        output_axes,
        out_layout,
        out_scatter,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
    } = kind
    else {
        unreachable!("gathered_matmul_op always binds to a Keep::Reduce fold")
    };
    let route_node = operands
        .iter()
        .find_map(|(_, _, lookup)| lookup.as_ref().map(|lookup| lookup.indices))
        .expect("gathered_matmul_op's own product operand gathers a route");
    BoundOp {
        node,
        dtype,
        extents,
        kind: BoundOpKind::RoundBatchedReduce {
            element_body,
            reduce_op,
            init,
            keep,
            operands,
            output_axes,
            out_layout,
            out_scatter,
            epilogue_body,
            epilogue_operands,
            epilogue_broadcast_axes,
            round_count,
            round_routes: vec![route_node; round_count as usize],
            round_outputs: vec![node; round_count as usize],
        },
    }
}

/// Mirrors `horizontal_merge_base_table_splice_tests` (gate (1) of its own
/// doc): the round-batched kernel's text must differ from round 0's own
/// unspliced kernel ONLY by the `RoundBase` preamble and the k bound
/// `route_buf_{z}` parameters
/// ([`splice_round_batched_reduce_base_table`]'s own doc) -- the route side
/// is a `round_gid.z`-switched buffer SELECTION now, never a CPU-copied
/// offset into one shared buffer -- and it must slice ONLY the gathered
/// route and the output -- the shared weight/activation operands must
/// remain untouched, never renamed.
#[cfg(feature = "metal-moe-mul-mat-id")]
mod round_batched_reduce_base_table_splice_tests {
    use alloc::collections::BTreeMap;

    use proxima_tensor::NumericPolicy;

    use super::super::{
        Codec, EmitError, emit, round_zero_reduce_bound, splice_round_batched_reduce_base_table,
    };
    use super::round_batched_matmul_op;

    const STRUCT_DECL: &str = "struct RoundBase { ulong output_base; };\n";

    #[test]
    fn round_batched_kernel_text_differs_from_round_zero_only_by_the_round_base_preamble()
    -> Result<(), EmitError> {
        let round_batched = round_batched_matmul_op(4);
        let round_zero = round_zero_reduce_bound(&round_batched);
        let weight_node = round_zero.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, Codec::Q4K);

        let unspliced = emit(&round_zero, &q4k, NumericPolicy::default()).expect("round 0 emits");
        let mut spliced = unspliced.clone();
        splice_round_batched_reduce_base_table(&mut spliced, &round_batched)?;

        assert!(
            spliced.source.contains(STRUCT_DECL),
            "spliced kernel must declare a single-field, output-only RoundBase:\n{}",
            spliced.source
        );
        assert!(
            spliced.source.contains("round_table[round_gid.z]"),
            "spliced kernel must index the round table by the z grid coordinate:\n{}",
            spliced.source
        );
        assert!(
            spliced.source.contains("switch (round_gid.z)"),
            "spliced kernel must switch on the z grid coordinate to pick this round's own \
             bound route buffer, never CPU-copy route contents:\n{}",
            spliced.source
        );
        assert!(
            !spliced.source.contains("uint gid [[thread_position_in_grid]]"),
            "the spliced kernel must not keep the scalar gid parameter:\n{}",
            spliced.source
        );
        assert!(
            spliced
                .source
                .contains("uint3 round_gid [[thread_position_in_grid]]"),
            "the spliced kernel must declare the widened vector gid parameter:\n{}",
            spliced.source
        );
        // every round's own route buffer is a separately bound parameter --
        // 4 rounds means route_buf_0..route_buf_3, each selected only by the
        // switch above, never read through gather_idx0 directly.
        for round in 0..4 {
            assert!(
                spliced.source.contains(&format!("route_buf_{round}")),
                "spliced kernel must bind round {round}'s own route buffer:\n{}",
                spliced.source
            );
        }
        // the shared weight/activation operands are gathered's OWN `in0`
        // (packed Q4K weight) and the plain `in1` activation -- neither may
        // be renamed: only the gathered route (`gather_idx0`) and `out`
        // move per round.
        assert!(
            spliced.source.contains("in0"),
            "the shared packed weight operand must remain untouched:\n{}",
            spliced.source
        );
        assert!(
            spliced.source.contains("in1"),
            "the shared activation operand must remain untouched:\n{}",
            spliced.source
        );
        Ok(())
    }
}

/// `Q5_K` sibling of the test above: the same Add-reduce-over-plain-
/// product shape must select `q5k_pair_dot` (`Codec::supports_pair_dot`,
/// a structural fact of `Q5_K`'s block layout, not a cargo feature)
/// rather than the scalar per-element `q5k_value` loop
/// `push_packed_row_blocked_body`'s `Codec::Q5K` arm falls back to
/// when the reduce is not a plain product.
#[test]
fn q5k_row_blocked_matmul_uses_paired_nibble_decode() {
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut q5k = BTreeMap::new();
    q5k.insert(weight_node, Codec::Q5K);

    assert!(
        packed_row_block(&bound, &operand_codecs(&bound, &q5k)).is_some(),
        "test fixture must actually take the row-blocked path for this assertion to mean anything"
    );

    let source = emit(&bound, &q5k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        source.contains("q5k_pair_dot"),
        "Add-reduce over a plain weight*activation body must take the paired decode path by default:\n{source}"
    );
    assert!(
        !source.contains("q5k_value(blk"),
        "the scalar per-element q5k_value dequant expression must not remain once the paired path is taken:\n{source}"
    );
}

/// `Q6_K` sibling of `q5k_row_blocked_matmul_uses_paired_nibble_decode`:
/// the same Add-reduce-over-plain-product shape must select
/// `q6k_pair_dot` (`Codec::supports_pair_dot`, a structural fact
/// of `Q6_K`'s block layout, not a cargo feature) rather than the
/// scalar per-element `q6k_value` loop `push_packed_row_blocked_body`'s
/// `Codec::Q6K` arm falls back to when the reduce is not a plain
/// product. This subsumes the pair of feature-gated marker tests this
/// landing replaced (`q6k_row_blocked_matmul_uses_paired_nibble_decode`/
/// `_uses_scalar_decode_by_default`) -- there is now exactly one
/// selection to assert, not a feature-on/feature-off pair.
#[test]
fn q6k_row_blocked_matmul_uses_paired_nibble_decode() {
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut q6k = BTreeMap::new();
    q6k.insert(weight_node, Codec::Q6K);

    assert!(
        packed_row_block(&bound, &operand_codecs(&bound, &q6k)).is_some(),
        "test fixture must actually take the row-blocked path for this assertion to mean anything"
    );

    let source = emit(&bound, &q6k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        source.contains("q6k_pair_dot"),
        "Add-reduce over a plain weight*activation body must take the paired decode path by default:\n{source}"
    );
    assert!(
        !source.contains("q6k_value(blk"),
        "the scalar per-element q6k_value dequant expression must not remain once the paired path is taken:\n{source}"
    );
}

#[test]
fn one_token_gathered_q4k_matmul_uses_row_blocked_gather_body() {
    let bound = gathered_matmul_op(1, 3, 4, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let codecs = operand_codecs(&bound, &q4k);

    assert!(
        classify_packed_row_block(&bound, &codecs).is_ok(),
        "one routed activation row selects one expert before the row-blocked reduction"
    );

    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("gathered q4k matmul emits")
        .source;
    assert!(
        source.contains("gather_idx0")
            && source.contains("weight_base[q] += fetched0 * u.gather_element_stride[0]")
            && source.contains("q4k_pair_dot"),
        "the row-blocked body must select the routed expert slab before its paired q4k decode:\n{source}"
    );
}

#[test]
fn expert_source_uses_descriptor_codec_instead_of_checkpoint_codec() {
    let bound = gathered_matmul_op(1, 3, 4, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let kernel = emit_with_expert_sources(&bound, &q4k, NumericPolicy::default(), weight_node)
        .expect("emits expert source kernel");
    assert!(kernel.source.contains("mixed_expert_element_from_offset"));
    assert!(
        kernel
            .source
            .contains("mixed_expert_element_from_local(expert_base0, expert_codec0, walk0)"),
        "the selected descriptor must be hoisted into the hot loop:\n{}",
        kernel.source
    );
    assert!(
        kernel.source.matches("q4k_pair_dot(").count() == 1,
        "a mixed source must not retain the checkpoint's fixed Q4_K decoder:\n{}",
        kernel.source
    );
    assert!(!kernel.source.contains("q4k_element(in0"));
    assert!(!kernel.source.contains("q6k_element(in0"));
    if std::env::var_os("PROXIMA_DUMP_EXPERT_TEST_SOURCE").is_some() {
        eprintln!("{}", kernel.source);
    }
}

#[test]
fn uniform_expert_source_retains_packed_row_decoder() {
    let bound = gathered_matmul_op(1, 3, 4, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let kernel = emit_with_uniform_expert_source(
        &bound,
        &q4k,
        NumericPolicy::default(),
        weight_node,
        Codec::Q4K,
    )
    .expect("emits uniform expert source kernel");
    assert!(kernel.source.contains("q4k_pair_dot("));
    assert_eq!(
        kernel
            .source
            .matches("mixed_expert_element_from_offset(")
            .count(),
        1,
        "uniform lowering keeps only the shared helper declaration"
    );
    assert!(
        kernel
            .source
            .contains("expert_descriptors[expert_route_index[q]]")
    );
}

#[cfg(not(feature = "metal-gathered-packed-row"))]
#[test]
fn flattened_selected_axis_gathered_q4k_matmul_is_not_row_blocked_without_feature() {
    // `[sequence = 1, selected = 2]` flattened to two rows has the same
    // physical `[2, d_in]` activation shape as a two-token gather. Each
    // row may name a different expert, so it cannot reuse the packed
    // multi-row body's one weight row across its activation group.
    let flattened_sequence_selected = 2;
    let bound = gathered_matmul_op(flattened_sequence_selected, 3, 4, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    assert_eq!(
        classify_packed_row_block(&bound, &operand_codecs(&bound, &q4k)).err(),
        Some(PackedRowBlockRejection::GatheredOperand),
        "flattened selected rows may route to different expert slabs, so the packed multi-row body must not share one weight row"
    );
}

#[cfg(feature = "metal-gathered-packed-row")]
#[test]
fn flattened_selected_axis_gathered_q4k_matmul_gathers_expert_base_once_per_token() {
    let bound = gathered_matmul_op(2, 3, 4, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let codecs = operand_codecs(&bound, &q4k);

    assert!(
        classify_packed_row_block(&bound, &codecs).is_ok(),
        "the opt-in gathered packed-row body must admit multiple selected rows"
    );

    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits gathered packed-row source")
        .source;
    // `selected`/`sequence` now take the SAME token/feature split a
    // dense op would -- proof this is the multi-row body, not the old
    // "treat every output element as its own row" collapse.
    assert!(
        source.contains("token_first = token_group"),
        "gathered rows must take the dense token/feature split, not the flattened-feature fallback:\n{source}"
    );
    assert!(
        source.contains("weight_expert_base"),
        "each token slot's expert base must be its own value, not shared:\n{source}"
    );
    assert_eq!(
        source.matches("long fetched0").count(),
        1,
        "the route must be resolved ONCE per token slot, not once per feature row:\n{source}"
    );
    assert!(
        source.contains("q4k_element"),
        "the routed weight decode still runs through the packed Q4_K element reader:\n{source}"
    );
}

/// Bytes proof for ROW 536/537's routed-expert regression: derives the
/// total weight-ROW count the dispatch reads (`feature_total *
/// token_total`, the exact product [`push_packed_row_multi_row_body`]
/// walks -- `weight_base[q]` once per feature row, `weight_expert_base[s]`
/// once per token slot, never re-derived per `(s, q)` pair) straight from
/// [`classify_packed_row_block`]'s own output, the same source the
/// codegen itself reads. 8 selected experts times 512 rows is the exact
/// slab-once shape the invariant names; the DENSE decode matvec of the
/// same `[rows=512, k=2048]` weight reads exactly 512 rows, one token.
#[cfg(feature = "metal-gathered-packed-row")]
#[test]
fn gathered_packed_row_reads_each_selected_experts_slab_exactly_once() {
    let selected = 8u32;
    let experts = 16u32;
    let rows = 512u32;
    let k = 2048u32;

    let gathered = gathered_matmul_op(selected, experts, rows, k);
    let gathered_weight = gathered.operands()[0].0;
    let mut gathered_codecs = BTreeMap::new();
    gathered_codecs.insert(gathered_weight, Codec::Q4K);
    let gathered_block =
        classify_packed_row_block(&gathered, &operand_codecs(&gathered, &gathered_codecs))
            .expect("the gathered decode matvec shape classifies as packed-row");
    let gathered_feature_total: u64 = gathered_block
        .feature_axes
        .iter()
        .map(|&axis| gathered.extents[axis as usize])
        .product();
    let gathered_token_total =
        packed_row_block_token_total(&gathered_block, &gathered.extents);

    assert_eq!(
        gathered_feature_total,
        u64::from(rows),
        "one weight row per d_out -- never one per (token, d_out) pair"
    );
    assert_eq!(
        gathered_token_total,
        u64::from(selected),
        "one token slot per selected expert -- the flattened axis gathered_matmul_op builds"
    );
    let gathered_row_reads = gathered_feature_total * gathered_token_total;
    assert_eq!(
        gathered_row_reads,
        u64::from(selected) * u64::from(rows),
        "total row reads across the whole dispatch: each of the 8 selected experts' \
         512 rows, exactly once -- not selected*sequence*d_out re-fetches per output element"
    );

    let dense = matmul_op(1, k, rows);
    let dense_weight = dense.operands()[0].0;
    let mut dense_codecs = BTreeMap::new();
    dense_codecs.insert(dense_weight, Codec::Q4K);
    let dense_block = classify_packed_row_block(&dense, &operand_codecs(&dense, &dense_codecs))
        .expect("the dense decode matvec of the identical [rows, k] shape classifies as packed-row");
    let dense_feature_total: u64 = dense_block
        .feature_axes
        .iter()
        .map(|&axis| dense.extents[axis as usize])
        .product();
    let dense_token_total = packed_row_block_token_total(&dense_block, &dense.extents);
    let dense_row_reads = dense_feature_total * dense_token_total;

    assert_eq!(
        dense_row_reads,
        u64::from(rows),
        "the dense kernel for the identical weight shape reads 512 rows, one token"
    );
    assert_eq!(
        gathered_row_reads / dense_row_reads,
        u64::from(selected),
        "gathering 8 experts must scale row reads by exactly 8x over the dense baseline, \
         not by selected*sequence*d_out/rows (the 39x-class blowup this fix removes)"
    );
}

/// `metal-q4k-single-fetch` sibling of the test above: the lane remap
/// renames the scale-deferred accumulators (`raw_low`/`raw_high` in
/// place of `q4k_pair_dot`, one pair per sub-block half — see
/// `push_q4k_single_fetch_body`'s own algebra note) but the SAME
/// dichotomy holds — Add-reduce over a plain product still defers the
/// scale, never falls back to per-element dequant. `matmul_op_f16`, not
/// `matmul_op`: `q4k_pair_dot`'s own `plain_product` arm is
/// `DType::Float32`-only and takes priority over this feature
/// (see `metal-q4k-single-fetch`'s own Cargo.toml doc), so a `Float32`
/// fixture would silently exercise `q4k_pair_dot` instead and this
/// test would assert nothing about single-fetch at all.
#[cfg(all(feature = "metal-q4k-single-fetch", not(feature = "metal-q4k-split-k")))]
#[test]
fn q4k_row_blocked_matmul_defers_scale_to_once_per_sub_block_single_fetch() {
    let bound = matmul_op_f16(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    assert!(
        packed_row_block(&bound, &operand_codecs(&bound, &q4k)).is_some(),
        "test fixture must actually take the row-blocked path for this assertion to mean anything"
    );

    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        source.contains("raw_low") && source.contains("raw_high"),
        "Add-reduce over a plain weight*activation body must take the scale-deferred path, split across both sub-block halves:\n{source}"
    );
    assert!(
        !source.contains("hdr_low.scale * low_levels[j] - hdr_low.minimum")
            && !source.contains("hdr_high.scale * high_levels[j] - hdr_high.minimum"),
        "the per-element dequant expression must not remain once the scale-deferred path is taken:\n{source}"
    );
}

/// `classify_packed_row_block` now admits [`Codec::Q4_0`] the same way it
/// admits [`Codec::Q8_0`] (`emit_and_classify.rs`'s whitelist match):
/// `Q4_0`'s own block is 32 elements, and eight contiguous real blocks span
/// exactly the same 256-element byte span one K-quant super-block occupies,
/// so [`Q4_0_SUPER_ELEMENT_MSL`] emulates the super-block-relative read this
/// path needs instead of falling back to the fully generic per-element
/// accessor. `matmul_op`'s reduce is a plain product (`Add` of `Multiply`),
/// so [`push_packed_row_blocked_body`]'s `Codec::Q4_0 if is_plain_product_
/// reduce` arm fires and the body calls the batched `q4_0_pair_dot`, not
/// the per-element `q4_0_super_element` this test asserted before
/// perf/q4_0-pair-dot landed the batched arm -- see
/// [`push_packed_row_blocked_body_emits_a_q4_0_row_blocked_kernel`] below
/// for the emitter-level half of this proof.
#[test]
fn q4_0_codec_takes_the_row_blocked_path_at_a_256_extent() {
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    assert!(
        classify_packed_row_block(&bound, &operand_codecs(&bound, &q4_0)).is_ok(),
        "Q4_0 must be admitted by codec now that Q4_0_SUPER_ELEMENT_MSL emulates its super-block read"
    );
    assert!(
        packed_row_block(&bound, &operand_codecs(&bound, &q4_0)).is_some(),
        "packed_row_block must agree with classify_packed_row_block's own admission"
    );

    let source = emit(&bound, &q4_0, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        source.contains("q4_0_pair_dot(blk"),
        "a plain-product row-blocked Q4_0 weight must call the batched pair-dot accessor:\n{source}"
    );
    assert!(
        !source.contains("q4_0_super_element(blk"),
        "the batched arm must fully replace the per-element accessor for a plain product:\n{source}"
    );
    assert!(
        !source.contains("q4k_run8(blk")
            && !source.contains("q5k_value(blk")
            && !source.contains("q6k_value(blk"),
        "a Q4_0 weight must never emit a K-quant row-blocked unpack call:\n{source}"
    );
}

/// Companion to [`q4_0_codec_takes_the_row_blocked_path_at_a_256_extent`]:
/// proves the per-element `q4_0_super_element` fallback still fires for a
/// reduce shape `is_plain_product_reduce` does not cover (`Maximum` here,
/// not `Add` of a bare `Multiply`) -- the same "batched arm is conditional,
/// not absolute" proof the K-quant codecs' own fallback arms rely on.
#[test]
fn q4_0_codec_falls_back_to_per_element_for_a_non_plain_product_reduce() {
    let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Maximum);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let source = emit(&bound, &q4_0, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        source.contains("q4_0_super_element(blk"),
        "a non-plain-product Q4_0 reduce must keep the per-element accessor:\n{source}"
    );
    assert!(
        !source.contains("q4_0_pair_dot(blk"),
        "the batched arm requires is_plain_product_reduce and must not fire here:\n{source}"
    );
}

/// Same landmine `q4_0_codec_never_takes_the_row_blocked_path_even_at_a_256_extent`
/// closes, proven for BOTH half-precision codecs at once: neither is a
/// K-quant, so both must reject via `NotKQuantCodec` and render through
/// the generic per-element accessor -- `Float16`'s direct `half` index,
/// `BFloat16`'s `bf16_element` widen -- never the row-blocked path's
/// `q4k_run8`/`q5k_value`/`q6k_value` calls.
#[proxima::test]
#[case::float16(Codec::Float16, "in0[")]
#[case::bfloat16(Codec::BFloat16, "bf16_element(")]
async fn half_precision_codec_never_takes_the_row_blocked_path_even_at_a_256_extent(
    #[case] codec: Codec,
    #[case] expected_read: &str,
) {
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut operands = BTreeMap::new();
    operands.insert(weight_node, codec);

    assert_eq!(
        classify_packed_row_block(&bound, &operand_codecs(&bound, &operands)).err(),
        Some(PackedRowBlockRejection::NotKQuantCodec),
        "{codec:?} must be rejected by codec, not admitted just because 256 is a multiple of its own block size"
    );
    assert!(
        packed_row_block(&bound, &operand_codecs(&bound, &operands)).is_none(),
        "packed_row_block must agree with classify_packed_row_block's own rejection"
    );

    let source = emit(&bound, &operands, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        source.contains(expected_read),
        "a {codec:?} weight must render through its own generic per-element accessor:\n{source}"
    );
    assert!(
        !source.contains("q4k_run8(blk")
            && !source.contains("q5k_value(blk")
            && !source.contains("q6k_value(blk"),
        "a {codec:?} weight must never emit a K-quant row-blocked unpack call:\n{source}"
    );
}

#[cfg(not(all(feature = "metal-q4k-single-fetch", not(feature = "metal-q4k-split-k"))))]
#[test]
fn q4k_row_blocked_non_add_reduce_keeps_the_per_element_path() {
    // Same fused `weight * activation` body as the matmul shape above,
    // but `Maximum` in place of `Add` — the identity
    // `sum_j (scale*nibble_j - min)*act_j == scale*sum(...) - min*sum(...)`
    // does not hold under `max`, so this must fall back to dequantizing
    // per element exactly as before this landing.
    let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Maximum);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    assert!(
        packed_row_block(&bound, &operand_codecs(&bound, &q4k)).is_some(),
        "test fixture must actually take the row-blocked path for this assertion to mean anything"
    );

    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        !source.contains("raw_acc"),
        "a Maximum reduce must never take the scale-deferred path, its identity does not hold under max:\n{source}"
    );
    assert!(
        source.contains("hdr.scale * levels[j] - hdr.minimum"),
        "a Maximum reduce must keep dequantizing per element:\n{source}"
    );
}

/// `metal-q4k-single-fetch` sibling of the test above: the lane remap
/// renames the per-element dequant expression (`hdr_low`/`hdr_high` in
/// place of `hdr`, one pair per sub-block half — see
/// `push_q4k_single_fetch_body`'s own doc) but the SAME dichotomy holds
/// — a Maximum reduce still falls back to dequantizing per element,
/// never the scale-deferred accumulators either arm uses for Add.
#[cfg(all(feature = "metal-q4k-single-fetch", not(feature = "metal-q4k-split-k")))]
#[test]
fn q4k_row_blocked_non_add_reduce_keeps_the_per_element_path_single_fetch() {
    let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Maximum);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    assert!(
        packed_row_block(&bound, &operand_codecs(&bound, &q4k)).is_some(),
        "test fixture must actually take the row-blocked path for this assertion to mean anything"
    );

    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        !source.contains("raw_low") && !source.contains("raw_high"),
        "a Maximum reduce must never take the scale-deferred path, its identity does not hold under max:\n{source}"
    );
    assert!(
        source.contains("hdr_low.scale * low_levels[j] - hdr_low.minimum")
            && source.contains("hdr_high.scale * high_levels[j] - hdr_high.minimum"),
        "a Maximum reduce must keep dequantizing per element, both sub-block halves:\n{source}"
    );
}

/// Same shape as [`matmul_op`] (`lhs=[features,k]` weight,
/// `rhs=[k,tokens]` activation), but with the out_map listing the TOKEN
/// axis before the feature axis -- `output_axes = [1, 0]` instead of
/// `matmul_op`'s `[0, 1]`. This is the convention every real matmul in
/// `proxima-tensor/src/spec.rs` follows (`"sg->sdg"`, `"so->sugdo"`,
/// ...: token/sequence letters listed first, the weight's own letters
/// last) and [`classify_tiled_gemm`]'s own doc names as load-bearing for
/// `native_packed_layout`'s packed-stride reconstruction — `matmul_op`'s
/// own `[0, 1]` order fails that check by construction, so the tiled
/// path needs its own fixture rather than reusing `matmul_op` (which
/// several PRE-EXISTING structural tests already pin to its current
/// order).
fn tiled_gemm_op(tokens: u32, k: u32, features: u32) -> BoundOp {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(features), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(tokens)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[1, 0])),
            keep: Keep::Reduce,
            name: Some("tiled_gemm".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("tiled gemm op infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("tiled gemm op lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

/// Same fused body as [`matmul_op`], but a 3-output-axis shape (`h`, `d`
/// weight-owned, `s` activation-owned) mirroring the multi-head Q/K/V
/// projections `proxima-tensor/src/spec.rs`'s `"ihd->shdi"` pattern
/// takes — `classify_tiled_gemm`'s own doc names this the documented
/// scope limit (ROW 107), not a silent gap: [`push_tiled_gemm_body`]
/// only understands a 2-D tile, so this shape must always stay on the
/// row-blocked path regardless of token count.
#[cfg(feature = "metal-tiled-gemm")]
fn multi_head_matmul_op(seq: u32, heads: u32, head_dim: u32, embed: u32) -> BoundOp {
    let mut program = Vec::new();
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(seq), Extent::Static(embed)],
            name: None,
        },
    );
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(embed),
                Extent::Static(heads),
                Extent::Static(head_dim),
            ],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(4, &[3, 1, 2]))),
                (activation, IndexMap::Affine(map::projection(4, &[0, 3]))),
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
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 1, 2])),
            keep: Keep::Reduce,
            name: Some("multi_head_matmul".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("multi-head matmul infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("multi-head matmul lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

/// [`push_tiled_gemm_body`]'s empty-group guard, driven with a hand-built
/// [`TiledGemmBlock`] -- [`classify_tiled_gemm`]'s own `is_empty()` gate
/// never lets a real caller build one of these, so this drives the
/// emitter's internal contract directly.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn push_tiled_gemm_body_rejects_an_empty_token_axis_group() {
    let bound = tiled_gemm_op(16, 256, 4);
    let block = TiledGemmBlock {
        weight: 0,
        other: 1,
        reduce_dim: 1,
        token_axes: Vec::new(),
        feature_axes: vec![0],
    };
    let mut source = String::new();
    let error = push_tiled_gemm_body(&mut source, bound.node, &[0], 2, &block, "float")
        .expect_err("an empty token axis group is never built by classify_tiled_gemm");
    assert!(matches!(
        error,
        EmitError::EmptyAxisGroup { group: "token", .. }
    ));
}

/// [`push_tiled_gemm_body`]'s axis-lookup guard: a hand-built
/// [`TiledGemmBlock`] naming an axis outside `output_axes` --
/// [`classify_tiled_gemm`] only ever builds `token_axes`/`feature_axes`
/// as a subset of `output_axes`, so this too drives the internal
/// contract directly.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn push_tiled_gemm_body_rejects_an_axis_not_in_output_axes() {
    let bound = tiled_gemm_op(16, 256, 4);
    let block = TiledGemmBlock {
        weight: 0,
        other: 1,
        reduce_dim: 1,
        token_axes: vec![5],
        feature_axes: vec![0],
    };
    let mut source = String::new();
    let error = push_tiled_gemm_body(&mut source, bound.node, &[0], 2, &block, "float")
        .expect_err("axis 5 is never in output_axes [0]");
    assert!(matches!(
        error,
        EmitError::AxisNotInOutputAxes { axis: 5, .. }
    ));
}

/// [`push_tiled_gemm_body`]'s `#[cfg(not(feature = "metal-tiled-gemm"))]`
/// stub and [`tiled_gemm_threadgroups`]'s own non-feature arm both
/// name this exact state: the tiled path reached without the feature
/// that alone can build a real `TiledGemmBlock`.
#[cfg(not(feature = "metal-tiled-gemm"))]
#[test]
fn push_tiled_gemm_body_is_disabled_without_the_metal_tiled_gemm_feature() {
    let bound = tiled_gemm_op(16, 256, 4);
    let block = TiledGemmBlock {
        token_axes: Vec::new(),
        feature_axes: Vec::new(),
    };
    let mut source = String::new();
    let error = push_tiled_gemm_body(&mut source, bound.node, &[0], 2, &block, "float")
        .expect_err("the tiled path never exists without metal-tiled-gemm");
    assert!(matches!(error, EmitError::TiledGemmFeatureDisabled { .. }));

    let error = tiled_gemm_threadgroups(bound.node, 4, 16)
        .expect_err("the tiled path never exists without metal-tiled-gemm");
    assert!(matches!(error, EmitError::TiledGemmFeatureDisabled { .. }));
}

#[cfg(not(feature = "metal-tiled-gemm"))]
#[test]
fn tiled_gemm_never_triggers_without_the_metal_tiled_gemm_feature() {
    // 16 tokens clears every plausible threshold; without the feature
    // compiled in, `TILED_GEMM_MIN_TOKENS` does not exist at all and
    // `classify_tiled_gemm` always returns `None` — see that function's
    // own doc. This test is cfg-gated the OPPOSITE way from the
    // `metal-tiled-gemm`-only tests below: it proves the tiled path is
    // invisible in the build that does not opt into it.
    let bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        !source.contains("simdgroup_multiply_accumulate"),
        "the tiled GEMM path must not exist at all without `metal-tiled-gemm`:\n{source}"
    );
    assert!(
        source.contains("sumf["),
        "16 tokens must still take the row-blocked path when the feature is off:\n{source}"
    );
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn decode_shape_stays_on_the_row_blocked_path_with_tiled_gemm_compiled_in() {
    // ONE token (real decode's own shape) is below
    // `TILED_GEMM_MIN_TOKENS` (8) regardless of how large the feature
    // axis is — proves decode keeps taking the vector path even when
    // the tiled kernel is compiled into the binary, the exact
    // correctness requirement ROW 107 states.
    let bound = tiled_gemm_op(1, 256, 4096);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    assert!(
        tiled_gemm_block(
            &bound,
            &operand_codecs(&bound, &q4k),
            ScalarOp::Add,
            ReduceInit::Zero,
            &[1, 0]
        )
        .is_none(),
        "one token must never clear TILED_GEMM_MIN_TOKENS"
    );
    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        !source.contains("simdgroup_multiply_accumulate"),
        "a one-token (decode-shaped) dispatch must not take the tiled GEMM path:\n{source}"
    );
    assert!(
        source.contains("sumf["),
        "a one-token dispatch must still take the row-blocked path:\n{source}"
    );
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn many_token_matmul_takes_the_tiled_gemm_path() {
    // 16 tokens clears TILED_GEMM_MIN_TOKENS (8); 4 weight rows is
    // deliberately NOT a multiple of TILE_DIM (8), exercising the
    // boundary-tile mask on the feature axis in the same test that
    // proves the path is taken at all.
    let bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    assert!(
        tiled_gemm_block(
            &bound,
            &operand_codecs(&bound, &q4k),
            ScalarOp::Add,
            ReduceInit::Zero,
            &[1, 0]
        )
        .is_some(),
        "16 tokens must clear TILED_GEMM_MIN_TOKENS"
    );
    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        source.contains("simdgroup_multiply_accumulate"),
        "a 16-token dispatch must take the tiled GEMM path:\n{source}"
    );
    assert!(
        source.contains("simdgroup_load"),
        "the tiled path must stage both operand tiles:\n{source}"
    );
    assert!(
        source.contains("feature_extent"),
        "the boundary mask must read the feature extent from uniforms, never bake it in:\n{source}"
    );
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn non_q4k_codec_never_takes_the_tiled_gemm_path() {
    // Q5_K/Q6_K are explicitly out of scope (ROW 107) -- unmeasured on
    // this path, and their unpack has no batched form to reuse.
    let bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;
    let mut q6k = BTreeMap::new();
    q6k.insert(weight_node, Codec::Q6K);

    assert!(
        tiled_gemm_block(
            &bound,
            &operand_codecs(&bound, &q6k),
            ScalarOp::Add,
            ReduceInit::Zero,
            &[1, 0]
        )
        .is_none(),
        "a Q6_K weight must never take the tiled GEMM path"
    );
    let source = emit(&bound, &q6k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        !source.contains("simdgroup_multiply_accumulate"),
        "a Q6_K weight must not emit the tiled GEMM kernel:\n{source}"
    );
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn multi_head_shaped_matmul_stays_on_the_row_blocked_path_regardless_of_token_count() {
    // 32 sequence positions clears TILED_GEMM_MIN_TOKENS handily, but
    // this op keeps TWO weight-owned output axes (`heads`, `head_dim`)
    // -- `classify_tiled_gemm`'s documented scope limit, not a silent
    // gap.
    let bound = multi_head_matmul_op(32, 8, 128, 4096);
    let weight_node = bound.operands()[1].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    let codecs = operand_codecs(&bound, &q4k);
    assert!(
        packed_row_block(&bound, &codecs).is_some(),
        "test fixture must actually clear the row-blocked gate for this assertion to mean anything"
    );
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        ..
    } = &bound.kind
    else {
        panic!("multi_head_matmul_op always builds a Keep::Reduce fold")
    };
    assert!(
        tiled_gemm_block(&bound, &codecs, *reduce_op, *init, output_axes).is_none(),
        "a 3-output-axis matmul must never take the 2-D tiled GEMM path"
    );
    let source = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert!(
        !source.contains("simdgroup_multiply_accumulate"),
        "a multi-head-shaped matmul must stay on the row-blocked path:\n{source}"
    );
}

fn cumsum_op(extent: u32) -> BoundOp {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[0])),
            keep: Keep::Scan,
            name: None,
        }),
    );
    let shapes = infer(&program, &[]).expect("cumsum infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("cumsum lowers")
        .into_iter()
        .next()
        .expect("one bound emitted")
}

/// `table[ids[s], d]` over iteration space `(s, d)`: the same worked
/// example `map.rs`'s docs use, as a standalone elementwise gather.
fn embedding_lookup_op(vocab: u32, dim: u32, seq: u32) -> BoundOp {
    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(vocab), Extent::Static(dim)],
            name: None,
        },
    );
    let ids = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(seq)],
            name: None,
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: map::IndexPattern {
            iter_rank: 2,
            axes: vec![
                map::AxisIndex::default(),
                map::AxisIndex {
                    terms: vec![AxisTerm::projection(1)].into(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(table, gathered_map)],
            name: None,
        },
    );
    let shapes = infer(&program, &[]).expect("embedding lookup infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("embedding lookup lowers")
        .into_iter()
        .next()
        .expect("one bound emitted")
}

#[test]
fn a_gather_op_emits_an_indices_binding_and_the_fetch_uniforms() {
    let bound = embedding_lookup_op(50_000, 8, 4);
    let kernel =
        emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("gather emits");

    assert_eq!(
        kernel.entry, "omega_elementwise_r2_n1_identity_g1",
        "the gather bit is part of the structural fingerprint"
    );
    assert_eq!(
        kernel.bindings,
        vec![
            Binding::Input(bound.operands()[0].0),
            Binding::Indices(
                bound.operands()[0]
                    .2
                    .as_ref()
                    .expect("operand 0 gathers")
                    .indices
            ),
            Binding::Output(bound.node),
            Binding::Uniforms,
            Binding::Fault,
        ],
        "inputs, then indices, then output, then uniforms, then the fault buffer"
    );
    assert!(kernel.source.contains("gather_idx0"));
    assert!(kernel.source.contains("gather_index_base"));
    assert!(kernel.source.contains("gather_element_stride"));
    assert!(kernel.source.contains("gather_extent"));
    assert_eq!(kernel.grid.threads, 4 * 8, "seq x feature, vocab absent");
}

#[test]
fn expert_source_emission_appends_payload_and_descriptor_bindings() {
    let bound = embedding_lookup_op(50_000, 8, 4);
    let kernel = emit_with_expert_sources(
        &bound,
        &BTreeMap::new(),
        NumericPolicy::default(),
        bound.operands()[0].0,
    )
    .expect("expert-source ABI emits");

    assert_eq!(
        &kernel.bindings[kernel.bindings.len() - 2..],
        &[
            Binding::ExpertPayloads(bound.operands()[0].0),
            Binding::ExpertDescriptors(bound.operands()[0].0),
        ],
        "expert buffers follow the ordinary input/index/output/uniform ABI"
    );
    assert!(kernel.source.contains("expert_payloads"));
    assert!(kernel.source.contains("expert_descriptors"));
    assert!(kernel.source.contains("struct ExpertPayloadDescriptor"));
}

#[test]
fn preamble_contains_mixed_expert_codec_selector() {
    let source = MIXED_EXPERT_READ_MSL;
    assert!(source.contains("mixed_expert_element"));
    assert!(source.contains("selected.codec == 1u"));
    assert!(source.contains("selected.codec == 2u"));
    assert!(source.contains("selected.codec == 3u"));
    assert!(source.contains("selected.codec == 4u"));
}

/// The routed Q4_K expert reduction from `gathered_expert_product`: a
/// cooperative reduce (not row-blocked), so `push_cooperative_gather_fetch`
/// emits both the clamp line (`fetched0 = max((long)0, min(fetched0`) and
/// the broadcast line (`fetched0 = (long)simd_broadcast_first`) for the
/// same weight index -- the exact pair
/// `expert_codec_declared_once_when_both_substitution_markers_fire` below
/// checks does not double-declare `expert_codec0`.
fn routed_q4k_reduce_kernel() -> Kernel {
    let mut program = Vec::new();
    let stack = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(3), Extent::Static(256), Extent::Static(2)],
            name: Some("expert_stack".into()),
        },
    );
    let route = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(4)],
            name: Some("route".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4), Extent::Static(256)],
            name: Some("activation".into()),
        },
    );
    let gathered =
        proxima_tensor::spec::gathered_expert_product(&mut program, stack, route, activation);
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: gathered,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: Some("routed_q4k".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("routed reduction infers");
    let bound = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("routed reduction lowers")
        .into_iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::Reduce { .. }))
        .expect("reduction bound op exists");
    let mut packed = BTreeMap::new();
    packed.insert(bound.operands()[0].0, Codec::Q4K);
    emit_with_expert_sources(
        &bound,
        &packed,
        NumericPolicy::default(),
        bound.operands()[0].0,
    )
    .expect("routed reduction emits")
}

#[test]
fn routed_q4k_reduce_uses_cooperative_gather_fetch() {
    let kernel = routed_q4k_reduce_kernel();
    assert!(kernel.source.contains("simd_sum(accumulator)"));
    assert!(
        kernel
            .source
            .contains("simd_broadcast_first((uint)fetched0)")
    );
    assert!(kernel.source.contains("q4k_header_for"));
    assert!(
        kernel.source.contains("byte_length == 0u")
            && kernel
                .source
                .contains("0x80000000u | min((uint)fetched0 + 1u"),
        "a missing routed expert must fault before the descriptor byte offset is read"
    );
}

#[test]
fn expert_codec_declared_once_when_both_substitution_markers_fire() {
    let kernel = routed_q4k_reduce_kernel();
    assert!(
        kernel.source.contains("fetched0 = max((long)0, min(fetched0"),
        "fixture must exercise the clamp marker for this regression to mean anything:\n{}",
        kernel.source
    );
    assert!(
        kernel.source.contains("fetched0 = (long)simd_broadcast_first"),
        "fixture must exercise the broadcast marker for this regression to mean anything:\n{}",
        kernel.source
    );
    assert_eq!(
        kernel.source.matches("uint expert_codec0 = ").count(),
        1,
        "both the clamp and broadcast substitution markers fire for weight 0 in this \
         cooperative (non-row-blocked) kernel -- the codec must be declared exactly once, \
         not once per marker, or Metal rejects the source as `redefinition of 'expert_codec0'`:\n{}",
        kernel.source
    );
}

#[test]
fn a_gather_kernel_binds_and_declares_the_fault_buffer() {
    let bound = embedding_lookup_op(50_000, 8, 4);
    let kernel =
        emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("gather emits");

    assert!(
        kernel.bindings.contains(&Binding::Fault),
        "a gather kernel must bind a fault buffer"
    );
    assert!(kernel.source.contains("device atomic_uint* fault"));
    assert!(
        kernel
            .source
            .contains("atomic_fetch_max_explicit(&fault[0]")
    );
    assert!(
        kernel
            .source
            .contains("fetched0 < 0 || fetched0 >= u.gather_extent[0]"),
        "the fault check must run before the clamp, on the unclamped fetched value"
    );
}

#[test]
fn a_gather_free_op_names_and_binds_exactly_as_before_gather_existed() {
    let bound = elementwise_tanh_op(10);
    let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
        .expect("gather-free elementwise emits");
    assert!(
        !kernel.entry.contains("_g"),
        "a gather-free kernel's name must not grow a gather suffix"
    );
    assert!(!kernel.source.contains("gather_idx"));
    assert!(
        !kernel.source.contains("fault") && !kernel.source.contains("atomic_uint"),
        "a gather-free kernel must not gain any fault-reporting machinery"
    );
    assert_eq!(
        kernel.bindings,
        vec![
            Binding::Input(bound.operands()[0].0),
            Binding::Output(bound.node),
            Binding::Uniforms,
        ],
        "gather-free bindings are unchanged: input, output, uniforms — no fault buffer"
    );
}

#[test]
fn elementwise_op_emits_one_input_one_output_and_a_matching_grid() {
    let bound = elementwise_tanh_op(10);
    let kernel =
        emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("elementwise emits");

    assert_eq!(kernel.entry, "omega_elementwise_r1_n1_tanh");
    assert_eq!(
        kernel.bindings,
        vec![
            Binding::Input(bound.operands()[0].0),
            Binding::Output(bound.node),
            Binding::Uniforms
        ]
    );
    assert!(
        kernel
            .source
            .contains("kernel void omega_elementwise_r1_n1_tanh")
    );
    assert!(kernel.source.contains("tanh(clamp(scratch[0], -20.0f, 20.0f))"));
    assert_eq!(kernel.grid.threads, 10);
}

/// A plain, unfused `Reduce` (identity element body, `Add`/`Zero`) over a
/// 3D input, keeping exactly `output_rank_axes` of its 3 iteration axes
/// and folding the rest — the minimal-pair generator ROW 93's
/// `kernel_cache_key` regression test needs: two calls with the SAME
/// `rank` (3) and operand count (1) but a DIFFERENT `output_rank_axes.len()`
/// share every field `entry_name` recorded before this row (rank, operand
/// count, body, reduce op, keep, init) while `render_reduce` still sizes
/// `output_extents`/`reduction_extents` differently for each — proving
/// `output_axes.len()` had to join the key, not just decorate a doc-comment.
fn rank3_identity_sum_op(output_rank_axes: &[u16]) -> BoundOp {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(2), Extent::Static(2), Extent::Static(2)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, output_rank_axes)),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let shapes = infer(&program, &[]).expect("rank3 identity sum infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("rank3 identity sum lowers")
        .into_iter()
        .next()
        .expect("one bound emitted")
}

#[test]
fn distinct_output_rank_at_same_total_rank_yields_distinct_cache_keys_and_source() {
    let keeps_two_axes = rank3_identity_sum_op(&[0, 1]);
    let keeps_one_axis = rank3_identity_sum_op(&[0]);

    assert_eq!(
        keeps_two_axes.extents.len(),
        keeps_one_axis.extents.len(),
        "same total rank"
    );
    assert_eq!(
        keeps_two_axes.operands().len(),
        keeps_one_axis.operands().len(),
        "same operand count"
    );

    let empty = BTreeMap::new();
    let key_two_axes = kernel_cache_key(&keeps_two_axes, &empty, NumericPolicy::default())
        .expect("cache key builds");
    let key_one_axis = kernel_cache_key(&keeps_one_axis, &empty, NumericPolicy::default())
        .expect("cache key builds");
    assert_ne!(
        key_two_axes, key_one_axis,
        "a coarser key would let a 1-output-axis fold hit the 2-output-axis pipeline"
    );

    let source_two_axes = emit(&keeps_two_axes, &empty, NumericPolicy::default())
        .expect("emits")
        .source;
    let source_one_axis = emit(&keeps_one_axis, &empty, NumericPolicy::default())
        .expect("emits")
        .source;
    assert_ne!(
        source_two_axes, source_one_axis,
        "output_extents/reduction_extents array sizes must differ in the rendered source"
    );
}

/// The test the class defect this module fixes needed: sweep every
/// render option one axis at a time from a shared base op, and for each
/// assert BOTH the emitted source changes AND `kernel_cache_key`'s
/// identity changes with it. ROW 290 (a forgotten cooperative-reduce
/// width) and main 7312713 (wgsl/cuda forgetting the reduce epilogue)
/// are exactly the shape this would have caught: a renderer whose body
/// text moved on some axis while its own identity function stayed
/// silent about it.
#[test]
fn every_axis_that_changes_emitted_source_also_changes_kernel_cache_key() {
    let empty = BTreeMap::new();

    // reduce op.
    let add = matmul_op_with_reduce(4, 8, 5, ScalarOp::Add);
    let max = matmul_op_with_reduce(4, 8, 5, ScalarOp::Maximum);
    assert_ne!(
        emit(&add, &empty, NumericPolicy::default())
            .expect("emits")
            .source,
        emit(&max, &empty, NumericPolicy::default())
            .expect("emits")
            .source,
        "reduce op must change the emitted body"
    );
    assert_ne!(
        kernel_cache_key(&add, &empty, NumericPolicy::default()).expect("cache key builds"),
        kernel_cache_key(&max, &empty, NumericPolicy::default()).expect("cache key builds"),
        "reduce op must change the identity"
    );

    // dtype (half vs wide).
    let mut program = Vec::new();
    let f32_source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: vec![(f32_source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let mut f16_program = Vec::new();
    let f16_source = append(
        &mut f16_program,
        Op::Input {
            dtype: DType::Float16,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    append(
        &mut f16_program,
        Op::Elementwise {
            dtype: DType::Float16,
            body: ScalarOp::Tanh,
            operands: vec![(f16_source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let f32_shapes = infer(&program, &[]).expect("f32 infers");
    let f32_bound = bind(
        &program,
        &f32_shapes,
        &[terminal(&program)],
        NumericPolicy::default(),
    )
    .expect("f32 lowers")
    .into_iter()
    .next()
    .expect("one bound emitted");
    let f16_shapes = infer(&f16_program, &[]).expect("f16 infers");
    let f16_bound = bind(
        &f16_program,
        &f16_shapes,
        &[terminal(&f16_program)],
        NumericPolicy::default(),
    )
    .expect("f16 lowers")
    .into_iter()
    .next()
    .expect("one bound emitted");
    assert_ne!(
        emit(&f32_bound, &empty, NumericPolicy::default())
            .expect("emits")
            .source,
        emit(&f16_bound, &empty, NumericPolicy::default())
            .expect("emits")
            .source,
        "dtype must change the emitted body (half vs. float declarations)"
    );
    assert_ne!(
        kernel_cache_key(&f32_bound, &empty, NumericPolicy::default())
            .expect("cache key builds"),
        kernel_cache_key(&f16_bound, &empty, NumericPolicy::default())
            .expect("cache key builds"),
        "dtype must change the identity"
    );

    // packed codec (the census's own finding for wgsl/cuda -- confirmed
    // here it was ALREADY correct for Metal).
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let mut q5k = BTreeMap::new();
    q5k.insert(weight_node, Codec::Q5K);
    assert_ne!(
        emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source,
        emit(&bound, &q5k, NumericPolicy::default())
            .expect("emits")
            .source,
        "packed codec must change the emitted body"
    );
    assert_ne!(
        kernel_cache_key(&bound, &q4k, NumericPolicy::default()).expect("cache key builds"),
        kernel_cache_key(&bound, &q5k, NumericPolicy::default()).expect("cache key builds"),
        "packed codec must change the identity"
    );

    // numeric policy -- folded into `kernel_cache_key` directly now
    // (never embedded in `emit`'s own source text, so no source-side
    // assertion here). A permission not reflected in `MathMode` (e.g.
    // `contraction` alone) must still change the identity, or two
    // kernels compiling the SAME `MathMode` under different
    // `NumericPolicy`s could share one PIPELINE_CACHE entry.
    assert_ne!(
        kernel_cache_key(&bound, &q4k, NumericPolicy::bit_exact()).expect("cache key builds"),
        kernel_cache_key(
            &bound,
            &q4k,
            NumericPolicy::bit_exact().with_contraction(true)
        )
        .expect("cache key builds"),
        "numeric policy must change the identity, or a bit_exact- and a \
         contraction-only-compiled kernel could share one PIPELINE_CACHE \
         entry even though both compile MathMode::Safe or MathMode::Relaxed"
    );
    assert_ne!(
        kernel_cache_key(&bound, &q4k, NumericPolicy::bit_exact()).expect("cache key builds"),
        kernel_cache_key(&bound, &q4k, NumericPolicy::llama_relaxed())
            .expect("cache key builds"),
        "numeric policy must change the identity, or a bit_exact- and a \
         llama_relaxed-compiled kernel could share one PIPELINE_CACHE entry"
    );
    assert_ne!(
        kernel_cache_key(&bound, &q4k, NumericPolicy::llama_relaxed())
            .expect("cache key builds"),
        kernel_cache_key(&bound, &q4k, NumericPolicy::fast()).expect("cache key builds"),
        "numeric policy must change the identity, or a llama_relaxed- and a \
         fast-compiled kernel could share one PIPELINE_CACHE entry"
    );

    // cooperative-reduce width -- ROW 290's own defect, at the same two
    // reduce lengths that test proved COOPERATIVE_REDUCE_MIN_LEN against.
    #[cfg(feature = "metal-wide-cooperative-reduce")]
    {
        let narrow = single_axis_sum_op(34);
        let wide = single_axis_sum_op(4096);
        assert_ne!(
            kernel_cache_key(&narrow, &empty, NumericPolicy::default())
                .expect("cache key builds"),
            kernel_cache_key(&wide, &empty, NumericPolicy::default())
                .expect("cache key builds"),
            "two cooperative reduces at different widths must never share a pipeline \
             (ROW 290: a stale narrower kernel silently drops reduction terms)"
        );
    }
}

/// The regression this row's first cut of `kernel_cache_key` actually
/// shipped with (caught by `omega::metal_parity
/// attention_block_spec_parity_matches_within_epsilon` and
/// `omega::backend_parity the_wrapper_agrees_with_itself_across_cpu_and_metal`
/// going from PASS to FAIL against a real, unrelated forward): two folds
/// can share `rank` AND `output_axes.len()` (so the SAME "how many axes
/// this key" check the prior test guards would still pass both) while
/// keeping a DIFFERENT axis SET or the same set in a DIFFERENT ORDER --
/// `render_reduce`/`push_cooperative_reduce_body` bake the literal `dim`
/// tied to each `u.output_extents[index]` slot straight into the source,
/// so either change alone must also change the key.
#[test]
fn distinct_output_axis_set_at_the_same_output_rank_yields_distinct_cache_keys_and_source() {
    let keeps_first_and_second = rank3_identity_sum_op(&[0, 1]);
    let keeps_first_and_third = rank3_identity_sum_op(&[0, 2]);
    let empty = BTreeMap::new();

    assert_eq!(
        keeps_first_and_second.extents.len(),
        keeps_first_and_third.extents.len(),
        "same total rank"
    );
    let key_first_second =
        kernel_cache_key(&keeps_first_and_second, &empty, NumericPolicy::default())
            .expect("cache key builds");
    let key_first_third =
        kernel_cache_key(&keeps_first_and_third, &empty, NumericPolicy::default())
            .expect("cache key builds");
    assert_ne!(
        key_first_second, key_first_third,
        "output_axes.len() alone cannot tell {{0,1}} from {{0,2}}"
    );

    let source_first_second = emit(&keeps_first_and_second, &empty, NumericPolicy::default())
        .expect("emits")
        .source;
    let source_first_third = emit(&keeps_first_and_third, &empty, NumericPolicy::default())
        .expect("emits")
        .source;
    assert_ne!(
        source_first_second, source_first_third,
        "the reduce dim, and every operand_strides[..][dim] read, must differ"
    );
}

#[test]
fn output_axis_order_at_the_same_axis_set_yields_distinct_cache_keys_and_source() {
    let ascending = rank3_identity_sum_op(&[0, 1]);
    let descending = rank3_identity_sum_op(&[1, 0]);
    let empty = BTreeMap::new();

    let key_ascending = kernel_cache_key(&ascending, &empty, NumericPolicy::default())
        .expect("cache key builds");
    let key_descending = kernel_cache_key(&descending, &empty, NumericPolicy::default())
        .expect("cache key builds");
    assert_ne!(
        key_ascending, key_descending,
        "the SEQUENCE order of output_axes selects which u.output_extents slot each dim reads"
    );

    let source_ascending = emit(&ascending, &empty, NumericPolicy::default())
        .expect("emits")
        .source;
    let source_descending = emit(&descending, &empty, NumericPolicy::default())
        .expect("emits")
        .source;
    assert_ne!(
        source_ascending, source_descending,
        "reversing output_axes must reverse which dim each output_extents index feeds"
    );
}

#[test]
fn distinct_packed_codec_on_the_same_shape_yields_distinct_cache_keys_and_source() {
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;

    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let mut q6k = BTreeMap::new();
    q6k.insert(weight_node, Codec::Q6K);

    let key_q4k =
        kernel_cache_key(&bound, &q4k, NumericPolicy::default()).expect("cache key builds");
    let key_q6k =
        kernel_cache_key(&bound, &q6k, NumericPolicy::default()).expect("cache key builds");
    assert_ne!(
        key_q4k, key_q6k,
        "entry_name alone cannot see which codec an operand reads through"
    );

    let source_q4k = emit(&bound, &q4k, NumericPolicy::default())
        .expect("emits")
        .source;
    let source_q6k = emit(&bound, &q6k, NumericPolicy::default())
        .expect("emits")
        .source;
    assert_ne!(
        source_q4k, source_q6k,
        "Q4_K and Q6_K unpack through different MSL functions"
    );
}

#[test]
fn distinct_dtype_on_the_same_shape_yields_distinct_cache_keys_and_source() {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let shapes = infer(&program, &[]).expect("f32 elementwise infers");
    let f32_bound = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("f32 elementwise lowers")
        .into_iter()
        .next()
        .expect("one bound emitted");

    let mut half_program = Vec::new();
    let half_source = append(
        &mut half_program,
        Op::Input {
            dtype: DType::Float16,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    append(
        &mut half_program,
        Op::Elementwise {
            dtype: DType::Float16,
            body: ScalarOp::Tanh,
            operands: vec![(half_source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let half_shapes = infer(&half_program, &[]).expect("f16 elementwise infers");
    let f16_bound = bind(
        &half_program,
        &half_shapes,
        &[terminal(&half_program)],
        NumericPolicy::default(),
    )
        .expect("f16 elementwise lowers")
        .into_iter()
        .next()
        .expect("one bound emitted");

    let empty = BTreeMap::new();
    let key_f32 = kernel_cache_key(&f32_bound, &empty, NumericPolicy::default())
        .expect("cache key builds");
    let key_f16 = kernel_cache_key(&f16_bound, &empty, NumericPolicy::default())
        .expect("cache key builds");
    assert_ne!(
        key_f32, key_f16,
        "entry_name does not encode dtype on its own"
    );

    let source_f32 = emit(&f32_bound, &empty, NumericPolicy::default())
        .expect("emits")
        .source;
    let source_f16 = emit(&f16_bound, &empty, NumericPolicy::default())
        .expect("emits")
        .source;
    assert_ne!(
        source_f32, source_f16,
        "float vs half declarations must differ in source"
    );
}

#[test]
fn same_structure_different_extents_share_one_cache_key() {
    let small = elementwise_tanh_op(4);
    let large = elementwise_tanh_op(4096);
    let empty = BTreeMap::new();

    assert_eq!(
        kernel_cache_key(&small, &empty, NumericPolicy::default()).expect("cache key builds"),
        kernel_cache_key(&large, &empty, NumericPolicy::default()).expect("cache key builds"),
        "a cache keyed on structure must still hit across concrete extents"
    );
}

#[test]
fn fused_matmul_op_emits_two_inputs_a_reduction_loop_and_a_row_by_col_grid() {
    let bound = matmul_op(4, 3, 5);
    assert!(
        matches!(bound.kind, BoundOpKind::Reduce { .. }),
        "the elementwise op must have fused into the reduce"
    );
    let kernel =
        emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("matmul emits");

    assert_eq!(kernel.entry, "omega_reduce_r3_o2_n2_multiply_add_zero");
    assert_eq!(kernel.bindings.len(), 4, "two inputs, one output, uniforms");
    assert!(matches!(kernel.bindings[2], Binding::Output(_)));
    assert!(matches!(kernel.bindings[3], Binding::Uniforms));
    assert!(
        kernel
            .source
            .contains("kernel void omega_reduce_r3_o2_n2_multiply_add_zero")
    );
    assert!(kernel.source.contains("reduction_total"));
    assert!(kernel.source.contains("(scratch[0] * scratch[1])"));
    assert!(kernel.source.contains("(accumulator + value)"));
    // `NumericPolicy::default()` is `bit_exact()` -- "bit-parity with
    // `cpu::evaluate`" (`NumericPolicy::bit_exact`'s own doc). `simd_sum`
    // is the cross-lane TREE reduction (`reduce_is_cooperative_for_
    // policy`'s own doc): a real reordering of this Add-fold versus the
    // left-to-right serial accumulate `cpu::evaluate` performs, so it
    // needs `NumericRewrite::TreeReduce`'s `reassociation` permission,
    // which `bit_exact` withholds by construction -- this plain
    // (unpacked, ungathered, non-broadcast-epilogue) matmul reduce now
    // takes `push_serial_reduce_body`'s one-thread-per-output loop
    // instead, matching every other bit-exact-policy reduce.
    assert!(
        !kernel.source.contains("simd_sum(accumulator)"),
        "bit-exact policy must not take the reassociating SIMD-group path"
    );
    assert!(
        kernel
            .source
            .contains("if ((long)gid >= u.output_total) { return; }"),
        "bit-exact policy must take the serial one-thread-per-output path"
    );
    assert_eq!(
        kernel.grid.threads,
        4 * 5,
        "one thread per (row, col) on the serial, bit-exact-policy path"
    );
    assert_eq!(
        kernel.grid.threadgroup_width,
        Some(32),
        "the driver dispatch width default is unaffected by the reduce's own cooperative/serial choice"
    );
}

/// A single 1-D input folded fully to a scalar via `Add` — the plain
/// shape [`reduce_is_cooperative`]'s length gate reasons about, without
/// `matmul_op`'s fused elementwise-multiply step muddying which reduce
/// length is under test.
fn single_axis_sum_op(reduce_len: u32) -> BoundOp {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(reduce_len)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let shapes = infer(&program, &[]).expect("single-axis sum infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("single-axis sum lowers")
        .into_iter()
        .next()
        .expect("one bound emitted")
}

/// Proven against the COMPILED `COOPERATIVE_REDUCE_MIN_LEN` constant,
/// not a hardcoded 128 — this same test body is the re-prove artifact for
/// BOTH claims the short-reduce initiative makes: at the
/// `omega-runtime.toml` default (2) it covers the unit-axis boundary
/// (1/2) while the 34 `attended`, 64 `score_even`/`score_odd`, and 4096
/// `sum_squares` shapes stay cooperative; re-run under
/// `OMEGA_COOPERATIVE_REDUCE_MIN_LEN=0`
/// (a distinct build — the constant is compile-time) it proves 0
/// restores every-qualifying-reduce-stays-cooperative, the routing every
/// build before this key existed used, because `>= 0` is vacuously true
/// for every case including the 34-length one.
#[proxima::test]
#[case::unit_axis(1)]
#[case::at_threshold_2(2)]
#[case::attended_34(34)]
#[case::score_even_odd_64(64)]
#[case::sum_squares_4096(4096)]
async fn reduce_routes_on_reduced_axis_length_against_min_len(#[case] reduce_len: u32) {
    let bound = single_axis_sum_op(reduce_len);
    let structurally_cooperative = meets_cooperative_min_len(u64::from(reduce_len));

    assert_eq!(
        reduce_is_cooperative(&bound),
        structurally_cooperative,
        "reduce_len={reduce_len} vs COOPERATIVE_REDUCE_MIN_LEN={}",
        crate::sized::COOPERATIVE_REDUCE_MIN_LEN
    );

    for (policy, expected_cooperative) in [
        (NumericPolicy::default(), false),
        (NumericPolicy::llama_relaxed(), structurally_cooperative),
    ] {
        let kernel = emit(&bound, &BTreeMap::new(), policy).expect("single-axis sum emits");
        assert_eq!(
            reduce_is_cooperative_for_policy(&bound, policy),
            expected_cooperative,
            "policy gate must agree with the structural route"
        );
        assert_eq!(
            kernel.source.contains("simd_sum(accumulator)"),
            expected_cooperative,
            "emitted kernel source must agree with the policy-aware route"
        );
    }
}

#[test]
fn cached_attention_emits_one_online_softmax_dispatch() {
    let bound = cached_attention_op();
    let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
        .expect("cached attention emits");

    let BoundOpKind::CachedAttention {
        cached_key_rows,
        new_key_rows,
        query_groups,
        head_dim,
        ..
    } = &bound.kind
    else {
        panic!("cached_attention_op must build a CachedAttention bound op");
    };
    let context_chunks = context_chunks_for(
        *cached_key_rows + *new_key_rows,
        *query_groups,
        *head_dim,
        NumericPolicy::default(),
    );

    assert!(kernel.source.contains("long relative ="));
    assert!(kernel.source.contains("simd_sum(partial_score)"));
    assert!(kernel.source.contains("vector_index = (long)gid / 32L"));
    assert!(kernel.source.contains("weighted[local_dimension] / sum"));
    assert_eq!(kernel.bindings.len(), 10, "eight inputs, output, uniforms");
    assert_eq!(kernel.grid.threads, 32 * context_chunks);
    assert_eq!(
        kernel.grid.threadgroup_width,
        Some(query_groups * context_chunks * SIMD_WIDTH),
        "query_groups=1 -- one threadgroup per (query_row, kv_head), width \
         widened by context_chunks under `[attention_context_chunks]`"
    );
    assert!(
        !kernel.source.contains("threadgroup float shared_k_even"),
        "K/V rows are read straight from device memory into registers, \
         never staged through threadgroup memory"
    );
    assert_eq!(
        kernel.source.contains("threadgroup float shared_m["),
        context_chunks > 1,
        "the cross-simdgroup merge only exists when context splits across \
         more than one simdgroup"
    );
    assert_eq!(
        kernel
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        context_chunks > 1,
        "a threadgroup_barrier only exists for the cross-simdgroup merge, \
         never inside the per-key loop"
    );
}

/// A one-chunk dispatch must render EXACTLY the pre-context-parallel
/// body: no `context_chunks`/`chunk`/merge tokens anywhere in the
/// source, and -- since the key loop no longer stages K/V through
/// `threadgroup` memory -- no `threadgroup_barrier` token anywhere
/// either, because chunks==1 has nothing left to synchronize on. This is
/// the byte-identity guarantee `render_cached_attention`'s
/// `context_chunks <= 1` branch reuses the original string construction
/// for, verified here rather than by a stored fixture (none pre-existed
/// to snapshot against). Skipped (not failed) under a sizing override
/// aggressive enough that even this tiny fixture's context splits --
/// `cached_attention_emits_one_online_softmax_dispatch` already checks
/// grid shape agrees with `context_chunks_for` at ANY sizing, this test
/// only adds the byte-identity claim for the sizing where chunks==1.
#[test]
fn cached_attention_at_one_chunk_never_emits_context_chunk_machinery() {
    let bound = cached_attention_op();
    let BoundOpKind::CachedAttention {
        cached_key_rows,
        new_key_rows,
        query_groups,
        head_dim,
        ..
    } = &bound.kind
    else {
        panic!("cached_attention_op must build a CachedAttention bound op");
    };
    if context_chunks_for(
        *cached_key_rows + *new_key_rows,
        *query_groups,
        *head_dim,
        NumericPolicy::default(),
    ) != 1
    {
        return;
    }

    let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
        .expect("cached attention emits");
    assert!(!kernel.source.contains("context_chunks"));
    assert!(!kernel.source.contains("long chunk ="));
    assert!(!kernel.source.contains("shared_m["));
    assert!(!kernel.source.contains("threadgroup_barrier"));
}

#[test]
fn cumsum_op_emits_a_scan_kernel_with_one_thread_per_line() {
    let bound = cumsum_op(8);
    let kernel =
        emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("cumsum emits");

    assert_eq!(kernel.entry, "omega_scan_r1_o1_n1_identity_add_zero");
    assert!(kernel.source.contains("inner_len"));
    assert!(kernel.source.contains("out_running"));
    assert_eq!(
        kernel.grid.threads, 1,
        "no leading dims: a single scan line"
    );
}

#[test]
fn emit_is_deterministic_byte_equal() {
    let bound = matmul_op(4, 3, 5);
    let first =
        emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("first emit succeeds");
    let second =
        emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("second emit succeeds");
    assert_eq!(first, second);
}

#[test]
fn same_structure_different_extents_yield_identical_source_but_different_grid() {
    let small = elementwise_tanh_op(4);
    let large = elementwise_tanh_op(4096);

    let small_kernel =
        emit(&small, &BTreeMap::new(), NumericPolicy::default()).expect("small emits");
    let large_kernel =
        emit(&large, &BTreeMap::new(), NumericPolicy::default()).expect("large emits");

    assert_eq!(small_kernel.source, large_kernel.source);
    assert_eq!(small_kernel.entry, large_kernel.entry);
    assert_ne!(small_kernel.grid.threads, large_kernel.grid.threads);
}

#[test]
fn an_arity_mismatched_op_is_rejected() {
    let mut bound = elementwise_tanh_op(4);
    if let BoundOpKind::Elementwise { body, .. } = &mut bound.kind {
        body.steps[0].op = ScalarOp::Add; // arity 2, but the step still carries 1 arg
    }

    let error = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
        .expect_err("mismatched arity is rejected");
    assert!(matches!(error, EmitError::ArityMismatch { .. }), "{error}");
}

#[test]
fn a_select_reduction_body_is_rejected() {
    let mut bound = matmul_op(4, 3, 5);
    if let BoundOpKind::Reduce { reduce_op, .. } = &mut bound.kind {
        *reduce_op = ScalarOp::Select;
    }

    let error = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
        .expect_err("select reduction body is rejected");
    assert!(
        matches!(error, EmitError::ReductionBodyIsSelect { .. }),
        "{error}"
    );
}

#[test]
fn a_keep_scan_over_zero_axes_is_rejected() {
    let mut bound = cumsum_op(8);
    bound.extents.clear();
    if let BoundOpKind::Reduce { output_axes, .. } = &mut bound.kind {
        output_axes.clear();
    }

    let error = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
        .expect_err("an empty scan is rejected");
    assert!(matches!(error, EmitError::EmptyScan { .. }), "{error}");
}

#[test]
fn render_reduce_rejects_an_elementwise_bound_op() {
    let bound = elementwise_tanh_op(8);
    let error = render_reduce(&bound, "entry", &[None], NumericPolicy::default(), false)
        .expect_err("an elementwise chain is not a Reduce fold");
    assert!(matches!(
        error,
        EmitError::RenderKindMismatch {
            expected: "keep::reduce fold",
            found: "elementwise",
            ..
        }
    ));
}

/// `bind::BoundOpKind::Reduce::epilogue_broadcast_axes`'s own doc: a
/// non-empty value is the RMSNorm-shaped "broadcast-reduce" epilogue
/// (`x * inv_rms`, re-broadcasting the fold's scalar back over the
/// reduced axis) — no Metal kernel here widens `push_reduce_epilogue_
/// write`'s own `output_rank`-only uniform declaration to match, so this
/// must reject with a typed error rather than emit a kernel that reads
/// or writes the wrong element count. The Metal HALF of this fusion is
/// the next slice; this proves main stays correct if the bind-time half
/// lands first.
#[test]
fn render_reduce_rejects_a_broadcast_reduce_epilogue() {
    let mut bound = matmul_op(4, 8, 4);
    let BoundOpKind::Reduce {
        epilogue_broadcast_axes,
        ..
    } = &mut bound.kind
    else {
        panic!("matmul_op always builds a Keep::Reduce fold")
    };
    epilogue_broadcast_axes.push(1);

    let error = render_reduce(&bound, "entry", &[None], NumericPolicy::default(), false)
        .expect_err("a broadcast-reduce epilogue has no Metal renderer yet");
    assert!(
        matches!(error, EmitError::EpilogueNotSupported { .. }),
        "{error}"
    );
}

#[test]
fn render_scan_rejects_an_elementwise_bound_op() {
    let bound = elementwise_tanh_op(8);
    let error = render_scan(&bound, "entry", &[None])
        .expect_err("an elementwise chain is not a Reduce fold");
    assert!(matches!(
        error,
        EmitError::RenderKindMismatch {
            expected: "keep::scan fold",
            found: "elementwise",
            ..
        }
    ));
}

#[test]
fn render_cached_attention_rejects_an_elementwise_bound_op() {
    let bound = elementwise_tanh_op(8);
    let error = render_cached_attention(&bound, "entry", NumericPolicy::default())
        .expect_err("an elementwise chain is not a CachedAttention op");
    assert!(matches!(
        error,
        EmitError::RenderKindMismatch {
            expected: "cached_attention",
            found: "elementwise",
            ..
        }
    ));
}

#[test]
fn simd_combine_fn_rejects_a_non_cooperative_reduce_op() {
    let bound = matmul_op_with_reduce(4, 8, 3, ScalarOp::Subtract);
    let error = simd_combine_fn(bound.node, ScalarOp::Subtract)
        .expect_err("subtract is not associative-commutative");
    assert!(matches!(
        error,
        EmitError::NonCooperativeReduceOp { op: "subtract", .. }
    ));
}

#[test]
fn cooperative_identity_token_rejects_a_non_cooperative_reduce_op() {
    let bound = matmul_op_with_reduce(4, 8, 3, ScalarOp::Subtract);
    let error = cooperative_identity_token(bound.node, ScalarOp::Subtract)
        .expect_err("subtract has no cooperative SIMD-group identity");
    assert!(matches!(
        error,
        EmitError::NonCooperativeReduceOp { op: "subtract", .. }
    ));
}

/// [`push_packed_row_blocked_body`]'s per-codec match, reached with a
/// hand-built [`PackedRowBlock`] naming a codec that is neither a K-quant
/// nor `Q8_0`/`Q4_0` -- [`classify_packed_row_block`]'s own `NotKQuantCodec`
/// gate never builds one of these in practice, so this drives the emitter's
/// internal contract directly rather than through [`emit`]. `Q5_1` is the
/// still-rejected exemplar: `Q8_0` and `Q4_0` each gained a row-blocked arm
/// (see `codec_row_block_step_bytes`/`Q8_0_SUPER_ELEMENT_MSL`/
/// `Q4_0_SUPER_ELEMENT_MSL`'s own docs) and are proved to SUCCEED through
/// this same function by
/// [`push_packed_row_blocked_body_emits_a_q8_0_row_blocked_kernel`]/
/// [`push_packed_row_blocked_body_emits_a_q4_0_row_blocked_kernel`] below,
/// not rejected here anymore.
#[test]
fn push_packed_row_blocked_body_rejects_a_non_k_quant_codec() {
    let bound = matmul_op(4, 256, 3);
    let block = PackedRowBlock {
        weight: 0,
        other: 1,
        reduce_dim: 1,
        codec: Codec::Q5_1,
        token_axes: Vec::new(),
        feature_axes: vec![0, 1],
    };
    let mut source = String::new();
    let error = push_packed_row_blocked_body(
        &mut source,
        &bound,
        ScalarOp::Add,
        ReduceInit::Zero,
        &[0, 1],
        2,
        &[Some(Codec::Q5_1), None],
        "float",
        &block,
        &ComposedBody::leaf(ScalarOp::Identity),
        &[],
        false,
    )
    .expect_err("Q5_1 never reaches the row-blocked path");
    assert!(matches!(
        error,
        EmitError::NonKQuantCodec { codec: "q5_1", .. }
    ));
}

/// [`classify_packed_row_block`] now admits [`Codec::Q8_0`]
/// (`emit_and_classify.rs`'s whitelist match) and
/// [`push_packed_row_blocked_body`] renders a real kernel body for it
/// instead of `EmitError::NonKQuantCodec` -- the gate-level half of the
/// proof; `q8_0_real_checkpoint_parity.rs`'s device test is the
/// execution-level half (real `Q8_0` checkpoint bytes, Metal fast path vs
/// dequantized-f32 CPU oracle, relative error ~2e-6).
#[test]
fn push_packed_row_blocked_body_emits_a_q8_0_row_blocked_kernel() {
    let bound = matmul_op(4, 256, 3);
    let block = PackedRowBlock {
        weight: 0,
        other: 1,
        reduce_dim: 1,
        codec: Codec::Q8_0,
        token_axes: Vec::new(),
        feature_axes: vec![0, 1],
    };
    let mut source = String::new();
    push_packed_row_blocked_body(
        &mut source,
        &bound,
        ScalarOp::Add,
        ReduceInit::Zero,
        &[0, 1],
        2,
        &[Some(Codec::Q8_0), None],
        "float",
        &block,
        &ComposedBody::leaf(ScalarOp::Identity),
        &[],
        false,
    )
    .expect("Q8_0 now reaches the row-blocked path and renders a kernel body");
    assert!(
        source.contains("q8_0_pair_dot(blk"),
        "row-blocked Q8_0 body for a plain-product reduce must call the batched pair-dot accessor: {source}"
    );
}

/// [`classify_packed_row_block`] now admits [`Codec::Q4_0`] the same way it
/// admits [`Codec::Q8_0`] (`emit_and_classify.rs`'s whitelist match) and
/// [`push_packed_row_blocked_body`] renders a real kernel body for it
/// instead of `EmitError::NonKQuantCodec` -- the gate-level half of the
/// proof; the greedy-decode token-parity run against the pre-change
/// baseline (`Q4_0` falling to the generic cooperative reduce) is the
/// execution-level half.
#[test]
fn push_packed_row_blocked_body_emits_a_q4_0_row_blocked_kernel() {
    let bound = matmul_op(4, 256, 3);
    let block = PackedRowBlock {
        weight: 0,
        other: 1,
        reduce_dim: 1,
        codec: Codec::Q4_0,
        token_axes: Vec::new(),
        feature_axes: vec![0, 1],
    };
    let mut source = String::new();
    push_packed_row_blocked_body(
        &mut source,
        &bound,
        ScalarOp::Add,
        ReduceInit::Zero,
        &[0, 1],
        2,
        &[Some(Codec::Q4_0), None],
        "float",
        &block,
        &ComposedBody::leaf(ScalarOp::Identity),
        &[],
        false,
    )
    .expect("Q4_0 now reaches the row-blocked path and renders a kernel body");
    assert!(
        source.contains("q4_0_pair_dot(blk"),
        "row-blocked Q4_0 body for a plain-product reduce must call the batched pair-dot accessor: {source}"
    );
}

/// Reachability proof for [`packed_row_split_factor`]'s row-count gate
/// (`omega-runtime.toml`'s `[packed_row_block].split_k_max_rows`,
/// default 4096): a 1024-row op (the `attn_k`/`attn_v` shape) sits
/// under both the row ceiling and `target_simdgroups`, so split-K must
/// engage (`split > 1`); a 14336-row op (the `ffn_up`/`ffn_gate` shape)
/// sits well past the row ceiling, so split-K must stay a no-op
/// (`split == 1`) regardless of what the simdgroup-target arithmetic
/// alone would compute. Calls [`packed_row_dispatch`] directly -- the
/// SAME function [`grid_threads`] and [`tiled_gemm_threadgroup_width`]
/// call -- so a passing test here is a guarantee those call sites see
/// the identical factor, not a duplicate policy that could drift.
#[cfg(feature = "metal-q4k-split-k")]
#[test]
fn split_k_engages_for_a_1024_row_op_and_declines_for_a_14336_row_op() {
    let (base_starved, split_starved) = packed_row_dispatch(1024, 1, Codec::Q4K);
    assert!(
        split_starved > 1,
        "a 1024-row op (attn_k/attn_v shape) must engage split-K under the default \
         split_k_max_rows(4096)/target_simdgroups gate: got split={split_starved} at \
         base_simdgroups={base_starved}"
    );

    let (base_wide, split_wide) = packed_row_dispatch(14336, 1, Codec::Q4K);
    assert_eq!(
        split_wide, 1,
        "a 14336-row op (ffn_up/ffn_gate shape) must stay split-K's no-op factor: got \
         split={split_wide} at base_simdgroups={base_wide}"
    );
}

/// The row-count ceiling itself, isolated from `target_simdgroups`: a
/// shape whose base-simdgroup count would otherwise clear
/// `packed_row_split_factor`'s target (so the simdgroup arithmetic
/// alone would still pick a factor > 1) must nonetheless collapse to
/// `1` once its row count crosses [`crate::sized::PACKED_ROW_SPLIT_K_MAX_ROWS`].
/// Proves the ceiling is a genuine additional gate, not merely
/// redundant with the simdgroup-target fall-off already in place.
#[cfg(feature = "metal-q4k-split-k")]
#[test]
fn the_row_ceiling_overrides_a_simdgroup_target_that_would_otherwise_split() {
    let max_rows = crate::sized::PACKED_ROW_SPLIT_K_MAX_ROWS;
    assert_ne!(
        max_rows, 0,
        "this test requires a non-zero configured ceiling"
    );

    let rows_at_ceiling = max_rows;
    let rows_past_ceiling = max_rows + PACKED_ROWS_PER_GROUP as u64;

    let base_at = rows_at_ceiling.div_ceil(PACKED_ROWS_PER_GROUP as u64);
    let split_at = packed_row_split_factor(base_at, rows_at_ceiling);
    let base_past = rows_past_ceiling.div_ceil(PACKED_ROWS_PER_GROUP as u64);
    let split_past = packed_row_split_factor(base_past, rows_past_ceiling);

    assert_eq!(
        split_past, 1,
        "one row-group past the configured ceiling must decline split-K even though its \
         base_simdgroups({base_past}) barely differs from the still-eligible shape's \
         ({base_at}), which split at factor {split_at}"
    );
}

/// Reachability proof for [`packed_row_nsg_factor`] (found dead: the
/// `#[cfg(feature = "metal-packed-row-nsg2")]` arm in
/// `tiled_gemm_threadgroup_width` sat AFTER an unconditional
/// `packed_row_block` return in the `Keep::Reduce` `if let` above it, so
/// it could never run for any op that reaches this function -- every
/// `packed_row_block` match IS a `Keep::Reduce` op by construction, see
/// `PackedRowBlock`'s own classification). With `metal-packed-row-nsg2`
/// on (and `metal-q4k-split-k` off, so `split == 1`), the packed
/// row-blocked matmul's threadgroup width must be exactly double the
/// one-simdgroup default.
#[cfg(all(feature = "metal-packed-row-nsg2", not(feature = "metal-q4k-split-k")))]
#[test]
fn packed_row_nsg2_doubles_the_threadgroup_width_for_a_packed_row_blocked_matmul() {
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let quantized = operand_codecs(&bound, &q4k);

    assert!(
        packed_row_block(&bound, &quantized).is_some(),
        "test fixture must actually take the row-blocked path for this assertion to mean anything"
    );

    let width = tiled_gemm_threadgroup_width(&bound, &quantized, NumericPolicy::default())
        .expect("a packed row-blocked reduce always has a threadgroup width");
    assert_eq!(
        width,
        SIMD_WIDTH * 2,
        "metal-packed-row-nsg2 must double the one-simdgroup default width to 2 \
         simdgroups (PACKED_ROW_NSG); got {width}"
    );
}

/// [`packed_row_nsg2_doubles_the_threadgroup_width_for_a_packed_row_blocked_matmul`]'s
/// own twin for the OTHER feature that shares [`packed_row_nsg_factor`]:
/// `metal-q4k-ggml-port` dispatches ggml's own body at ggml's own
/// nsg=2, so it must double the threadgroup width the identical way
/// `metal-packed-row-nsg2` does -- same assertion, different feature,
/// proving the two features compose through one factor rather than two
/// competing nsg constants.
#[cfg(all(feature = "metal-q4k-ggml-port", not(feature = "metal-q4k-split-k")))]
#[test]
fn ggml_port_doubles_the_threadgroup_width_for_a_packed_row_blocked_matmul() {
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let quantized = operand_codecs(&bound, &q4k);

    assert!(
        packed_row_block(&bound, &quantized).is_some(),
        "test fixture must actually take the row-blocked path for this assertion to mean anything"
    );

    let width = tiled_gemm_threadgroup_width(&bound, &quantized, NumericPolicy::default())
        .expect("a packed row-blocked reduce always has a threadgroup width");
    assert_eq!(
        width,
        SIMD_WIDTH * 2,
        "metal-q4k-ggml-port must double the one-simdgroup default width to 2 \
         simdgroups (PACKED_ROW_NSG); got {width}"
    );
}

/// [`packed_row_nsg2_doubles_the_threadgroup_width_for_a_packed_row_blocked_matmul`]'s
/// feature-off twin: without EITHER nsg2 feature compiled in, the NSG
/// factor itself must stay at one simdgroup -- proves the nsg2 fix is
/// additive, not a change to the default dispatch geometry. This cfg arm
/// is also the one `metal-q4k-split-k` alone reaches (neither nsg2
/// feature is on), and split-K widens the SAME threadgroup for an
/// unrelated, real reason (starved shapes get more simdgroups
/// cooperating on one reduction, [`packed_row_split_factor`]'s own doc),
/// so the expected width is derived from the SAME
/// [`packed_row_dispatch`]/[`packed_row_nsg_factor`] production reads
/// rather than a literal -- a hardcoded `SIMD_WIDTH` here is feature-blind
/// to split-K's own widening and fails every cell it does not predict.
#[cfg(not(any(feature = "metal-packed-row-nsg2", feature = "metal-q4k-ggml-port")))]
#[test]
fn packed_row_nsg2_off_leaves_the_threadgroup_width_unchanged() {
    let bound = matmul_op(4, 256, 5);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let quantized = operand_codecs(&bound, &q4k);

    let block = packed_row_block(&bound, &quantized)
        .expect("test fixture must actually take the row-blocked path for this assertion to mean anything");

    let feature_total: u64 = block
        .feature_axes
        .iter()
        .map(|&axis| bound.extents[axis as usize])
        .product();
    let token_total = packed_row_block_token_total(&block, &bound.extents);
    let (_base, split) = packed_row_dispatch(feature_total, token_total, block.codec);
    let expected_width = SIMD_WIDTH * split * packed_row_nsg_factor();

    let width = tiled_gemm_threadgroup_width(&bound, &quantized, NumericPolicy::default())
        .expect("a packed row-blocked reduce always has a threadgroup width");
    assert_eq!(
        width, expected_width,
        "without either nsg2 feature the packed row-blocked path's width must match \
         production's own SIMD_WIDTH * split * packed_row_nsg_factor() derivation; got {width}"
    );
}

/// `omega-runtime.toml`'s `[attention_context_chunks]` ships
/// `keys_per_chunk = 16`, so a real 64-key merged context
/// (`cached_key_rows + new_key_rows`) is exactly the shape
/// `context_chunks_for` splits across simdgroups today -- the algebra
/// design's own worked example (`op.rs:107`'s `is_associative` has no
/// caller that gates this reassociation; this is the gate).
#[test]
fn context_chunk_merge_is_gated_by_numeric_policy_for_a_real_64_key_context() {
    let context_length: u64 = 64;
    let query_groups: u64 = 1;
    let head_dim: u64 = 4;
    assert_eq!(
        context_chunks_for(
            context_length,
            query_groups,
            head_dim,
            NumericPolicy::bit_exact()
        ),
        1,
        "bit_exact() withholds reassociation, ContextChunkMerge, so this falls back to the \
         single-pass chunk<=1 kernel `render_cached_attention` already renders"
    );
    let chunks = context_chunks_for(
        context_length,
        query_groups,
        head_dim,
        NumericPolicy::llama_relaxed(),
    );
    assert!(
        chunks > 1,
        "llama_relaxed() grants reassociation, clearing NumericRewrite::ContextChunkMerge, \
         so a 64-key context (4x omega-runtime.toml's 16-key chunk) must split across more \
         than one simdgroup; got {chunks}"
    );
}

/// ROW 388: `effective_context_chunk_cap` must return the compiled
/// `ATTENTION_CONTEXT_CHUNK_CAP` unchanged for openchat's real shape
/// (`kv_heads` 4, `group` 4 -> `query_groups=4`, `head_dim=128`) --
/// `4 * 4 * (128 + 2) = 2080` bytes per chunk, comfortably under the
/// 32768-byte threadgroup ceiling even at the compiled maximum.
#[test]
fn effective_context_chunk_cap_holds_the_compiled_cap_for_the_openchat_shape() {
    assert_eq!(
        effective_context_chunk_cap(4, 128),
        crate::sized::ATTENTION_CONTEXT_CHUNK_CAP,
        "openchat's per-chunk threadgroup footprint is well under budget"
    );
}

/// ROW 388's own defect: qwen35's `query_groups=8`/`head_dim=256` shape
/// declares `8 * 4 * (256 + 2) = 33024` bytes at the compiled cap (4),
/// past Metal's 32768-byte `threadgroup` ceiling
/// (`CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES`) -- this must clamp down
/// to 3, the largest chunk count whose declared footprint (24768 bytes)
/// still fits.
#[test]
fn effective_context_chunk_cap_clamps_below_the_threadgroup_budget_for_qwen35_shape() {
    assert_eq!(
        effective_context_chunk_cap(8, 256),
        3,
        "qwen35's query_groups=8/head_dim=256 shape must clamp the compiled cap down to what \
         the 32768-byte threadgroup budget actually admits"
    );
}

/// `render_cached_attention`'s single-range dynamic path must never
/// declare more `shared_m`/`shared_l`/`shared_o` threadgroup bytes than
/// [`CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES`] admits, for any shape
/// this crate can bind -- the assertion this whole cap exists to
/// guarantee, checked directly against the rendered source rather than
/// inferred from the cap function alone.
#[test]
fn render_cached_attention_never_declares_past_the_threadgroup_budget() {
    let mut bound = cached_attention_op_dynamic(0, 512);
    let BoundOpKind::CachedAttention {
        query_groups,
        head_dim,
        rotary_dim,
        ..
    } = &mut bound.kind
    else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    *query_groups = 8;
    *head_dim = 256;
    *rotary_dim = 256;

    let source = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
        .expect("qwen35-shaped single-range dynamic attention renders");
    let cap = effective_context_chunk_cap(8, 256);
    assert!(cap < crate::sized::ATTENTION_CONTEXT_CHUNK_CAP);
    let declared_bytes = 4 * 8 * cap * (256 + 2);
    assert!(
        declared_bytes <= crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES,
        "declared shared_m/shared_l/shared_o bytes ({declared_bytes}) must stay within the \
         threadgroup budget"
    );
    assert!(
        source.contains(&format!("constexpr long cap = {cap};")),
        "rendered source must declare the SAME effective cap this test computed"
    );
}

/// The same gate, exercised through the real emitter
/// ([`render_cached_attention`]) instead of `context_chunks_for` in
/// isolation -- the rendered kernel source must not contain the
/// cross-simdgroup merge block under `BitExact`, and must contain it
/// once the caller opts up.
#[test]
fn render_cached_attention_omits_the_merge_block_under_bit_exact_policy() {
    let mut bound = cached_attention_op();
    let BoundOpKind::CachedAttention {
        cached_key_rows,
        new_key_rows,
        ..
    } = &mut bound.kind
    else {
        unreachable!("cached_attention_op always returns a CachedAttention kind");
    };
    *cached_key_rows = 48;
    *new_key_rows = 16;

    let rejected = render_cached_attention(&bound, "entry", NumericPolicy::bit_exact())
        .expect("bit_exact() still renders -- it falls back to the single-pass kernel");
    let admitted = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
        .expect("llama_relaxed() renders the cross-simdgroup merge kernel");

    assert!(
        !rejected.contains("merged_max"),
        "bit_exact() must never reassociate the online-softmax fold across simdgroups"
    );
    assert!(
        admitted.contains("merged_max"),
        "llama_relaxed() is expected to emit the cross-simdgroup merge block"
    );
}

/// Redesign §4c's own inert proof: `bit_exact()` renders the split
/// kernel's SOURCE and BINDINGS exactly as `render_cached_attention`
/// already did before `ContextSplitMerge` existed (no `attn_scratch`,
/// no `Binding::Scratch`, still a single `Binding::Output`), and emits
/// no companion merge kernel at all. `llama_relaxed()` is the ONLY
/// policy that changes either: the split kernel's final store moves to
/// `attn_scratch` and its output binding becomes `Binding::Scratch`,
/// and `emit_cached_attention_merge` returns the companion kernel that
/// reads that scratch layout back and writes the real output. 256 keys
/// (`omega-runtime.toml`'s `keys_per_split_at_scale = 128`, ROW 383) is
/// deliberately past the split threshold under EITHER policy that admits
/// the rewrite --
/// `cached_attention_merge_needed` now takes context length, not just
/// policy (its own doc), so a fixture at or below `keys_per_split`
/// would stay single-dispatch even under `llama_relaxed`, proving
/// nothing about the two-dispatch form this test exists to check.
#[test]
fn cached_attention_two_dispatch_form_is_inert_under_bit_exact() {
    let mut bound = cached_attention_op_dynamic(200, 56);
    let BoundOpKind::CachedAttention { head_dim, rotary_dim, .. } = &mut bound.kind else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    // multiple of 8, `render_cached_attention`'s own alignment
    // requirement once `block_width_for` admits `TreeReduce`
    // (`llama_relaxed()` below) -- `AttentionBlockMisaligned`'s own doc.
    *head_dim = 8;
    *rotary_dim = 8;
    let packed_operands = PackedOperands::new();

    let bit_exact_kernel = emit(&bound, &packed_operands, NumericPolicy::bit_exact())
        .expect("bit_exact() still renders the single-dispatch kernel");
    assert!(
        !bit_exact_kernel.source.contains("attn_scratch"),
        "bit_exact() must render the byte-identical, pre-redesign kernel body"
    );
    assert!(
        bit_exact_kernel
            .bindings
            .iter()
            .any(|binding| matches!(binding, Binding::Output(node) if *node == bound.node)),
        "bit_exact() must still bind its own node as a real Binding::Output"
    );
    assert!(
        !bit_exact_kernel
            .bindings
            .iter()
            .any(|binding| matches!(binding, Binding::Scratch)),
        "bit_exact() must never bind a scratch slot"
    );
    assert!(
        emit_cached_attention_merge(&bound, NumericPolicy::bit_exact())
            .expect("emit_cached_attention_merge never errors on a well-formed op")
            .is_none(),
        "bit_exact() must not need a merge dispatch at all"
    );

    let relaxed_kernel = emit(&bound, &packed_operands, NumericPolicy::llama_relaxed())
        .expect("llama_relaxed() renders the split kernel");
    assert!(
        relaxed_kernel.source.contains("attn_scratch"),
        "llama_relaxed() must write its final partial into scratch, not `out`"
    );
    assert!(
        relaxed_kernel
            .bindings
            .iter()
            .any(|binding| matches!(binding, Binding::Scratch)),
        "llama_relaxed()'s split kernel binds a scratch slot instead of Binding::Output"
    );
    assert!(
        !relaxed_kernel
            .bindings
            .iter()
            .any(|binding| matches!(binding, Binding::Output(_))),
        "llama_relaxed()'s split kernel must not ALSO claim to write the real output"
    );

    let merge_kernel = emit_cached_attention_merge(&bound, NumericPolicy::llama_relaxed())
        .expect("emit_cached_attention_merge never errors on a well-formed op")
        .expect("llama_relaxed() needs a companion merge dispatch");
    assert!(merge_kernel.entry.ends_with("_merge"));
    assert_eq!(
        merge_kernel.bindings,
        alloc::vec![
            Binding::Scratch,
            Binding::Output(bound.node),
            Binding::Uniforms
        ],
        "the merge kernel reads the split's scratch and writes the real output"
    );
}

/// Redesign §4c, item 3, corrected by ROW 385: a ROW 376-scoreboard-shaped
/// context (40 keys) renders the single, byte-identical-in-STRUCTURE
/// kernel under EITHER policy now -- [`cached_attention_merge_needed`]
/// additionally requires `context_length >= ATTENTION_SPLIT_KEYS_PER_
/// SPLIT_AT_SCALE` (128), so a 40-key context never engages the scratch
/// hop regardless of what `splits_for`'s own small-divisor branch would
/// report in isolation. ROW 381's split form measured faster at this
/// window than the byte-identical sequential body (36.1us vs 43.6us
/// bare), but ROW 385 supersedes it: the per-query-head grid
/// ([`cached_attention_per_query_head_grid`]) is the mechanism that wins
/// this window now, by giving the GPU more, narrower threadgroups to
/// schedule instead of a second dispatch to pay for.
#[test]
fn forty_key_plan_uses_the_per_query_head_grid_with_no_merge_under_either_policy() {
    let mut bound = cached_attention_op_dynamic(32, 8);
    let BoundOpKind::CachedAttention { head_dim, rotary_dim, .. } = &mut bound.kind else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    *head_dim = 8;
    *rotary_dim = 8;
    let packed_operands = PackedOperands::new();

    for policy in [NumericPolicy::bit_exact(), NumericPolicy::llama_relaxed()] {
        let kernel =
            emit(&bound, &packed_operands, policy).expect("a 40-key context always renders");
        assert!(
            !kernel.source.contains("attn_scratch"),
            "a 40-key context must never engage the scratch hop under {policy:?}"
        );
        assert!(
            kernel
                .source
                .contains("long kv_head_and_group = (long)tgid % (kv_heads * query_groups);"),
            "a 40-key context must decode `group` off `tgid`, not shared threadgroup memory, \
             under {policy:?}"
        );
        assert!(
            emit_cached_attention_merge(&bound, policy)
                .expect("emit_cached_attention_merge never errors on a well-formed op")
                .is_none(),
            "a 40-key context must never need a companion merge dispatch under {policy:?}"
        );
    }
}

/// ROW 385: below the split-at-scale knee the per-query-head grid must
/// widen the THREADGROUP COUNT by `query_groups`, never the total thread
/// count -- [`grid_threads`] is untouched by this feature by
/// construction ([`cached_attention_per_query_head_grid`]'s own doc), so
/// this is the one source of truth that the reshaping is real: the same
/// total threads land in more, narrower threadgroups instead of fewer,
/// wider ones.
#[test]
fn per_query_head_grid_narrows_threadgroup_width_without_changing_total_threads() {
    let mut narrow = cached_attention_op_dynamic(32, 8);
    let BoundOpKind::CachedAttention {
        head_dim,
        rotary_dim,
        query_groups,
        ..
    } = &mut narrow.kind
    else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    *head_dim = 8;
    *rotary_dim = 8;
    *query_groups = 4;

    let mut wide = narrow.clone();
    let BoundOpKind::CachedAttention {
        cached_key_rows,
        new_key_rows,
        ..
    } = &mut wide.kind
    else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    // Structural `cached_key_rows == 0` (the real single-range
    // invariant, `cached_attention_op_dynamic`'s own doc) -- the past-
    // the-knee 256-key total this test's own doc names lands entirely
    // in `new_key_rows`.
    *cached_key_rows = 0;
    *new_key_rows = 256;

    let quantized: Vec<Option<Codec>> = Vec::new();
    let narrow_width =
        tiled_gemm_threadgroup_width(&narrow, &quantized, NumericPolicy::bit_exact())
            .expect("a CachedAttention op always has a threadgroup width");
    let wide_width =
        tiled_gemm_threadgroup_width(&wide, &quantized, NumericPolicy::bit_exact())
            .expect("a CachedAttention op always has a threadgroup width");
    assert_eq!(
        wide_width,
        4 * narrow_width,
        "past the knee (256 keys) the width must still carry the full query_groups factor \
         (4x the below-the-knee width): narrow={narrow_width} wide={wide_width}"
    );

    let narrow_threads = grid_threads(&narrow, &quantized, NumericPolicy::bit_exact(), false)
        .expect("a CachedAttention op always has a thread count");
    let wide_threads = grid_threads(&wide, &quantized, NumericPolicy::bit_exact(), false)
        .expect("a CachedAttention op always has a thread count");
    assert_eq!(
        narrow_threads, wide_threads,
        "total dispatched threads must be identical across the knee -- only the \
         threadgroup width narrows, never grid_threads' own total: narrow={narrow_threads} \
         wide={wide_threads}"
    );
}

/// Redesign §4c, item 1: the split kernel's own slice formula and
/// scratch index expression, read straight out of the emitted MSL text
/// -- an off-by-one in either is a device OOB write (design risk 3), so
/// this pins the exact rendered arithmetic rather than trusting the
/// hand-derivation.
#[test]
fn split_kernel_emits_the_slice_formula_and_scratch_index() {
    let mut bound = cached_attention_op_dynamic(200, 56);
    let BoundOpKind::CachedAttention { head_dim, rotary_dim, .. } = &mut bound.kind else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    *head_dim = 8;
    *rotary_dim = 8;

    let source = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
        .expect("llama_relaxed renders the split kernel for a 256-key context");

    assert!(
        source.contains("uint tgid [[threadgroup_position_in_grid]]"),
        "the split kernel must read its threadgroup index off Metal's own coordinate"
    );
    assert!(
        source.contains(
            "long slice_len = (live + splits - 1L) / splits;\n    long lo = split * slice_len;\n    long hi = min(lo + slice_len, live);"
        ),
        "the slice formula must be exactly ceil_div(live, splits), [lo, hi)"
    );
    assert!(
        source
            .contains("long scratch_index = (query_index * splits + split) * (2L + head_dim);"),
        "the scratch write must index by (query_index * splits + split) using the LIVE \
         u.splits value cached_attention_scratch_len sized the buffer with, not the \
         compiled ATTENTION_SPLIT_MAX -- ROW: production, 2026-09-07, striding by the \
         compiled max wrote past every query_index > 0's allocated slot"
    );
    assert!(
        source.contains("constexpr long grid_splits = 32;"),
        "the decode modulus must be the compiled grid multiplier (ATTENTION_SPLIT_MAX = 32 \
         here), the SAME constant grid_threads widened the dispatch by -- never the live \
         u.splits, or a tgid the widened grid always dispatches decodes a query_row past \
         the real row count"
    );
    assert!(
        source.contains("long split = query_row_and_split % grid_splits;")
            && source.contains("long query_row = query_row_and_split / grid_splits;"),
        "query_row/split must divide out grid_splits, not the live splits field"
    );
    assert!(
        source.contains("if (split >= splits) { return; }"),
        "an idle split (split >= the live u.splits) must return before touching Q/K/V or \
         scratch -- the merge kernel already reads only i < u.splits, so it must not write \
         an identity partial either"
    );
    assert!(
        source.contains("long slice_start = lo + chunk;"),
        "the block-staged walk (llama_relaxed grants TreeReduce too) must start from this \
         threadgroup's own [lo, hi) slice, not the whole live range"
    );

    let sequential_source = render_cached_attention(&bound, "entry", NumericPolicy::default())
        .expect("bit_exact still renders (splits collapse to 1, but the shape is shared)");
    assert!(
        sequential_source.contains("for (long key = lo + chunk; key < hi; key += chunks)"),
        "the strictly-sequential per-key walk must be bounded to this threadgroup's own \
         [lo, hi) slice"
    );
}

/// Redesign §4c: the merge kernel's own combine, read straight out of
/// the emitted MSL text -- `simd_max`/`simd_sum` over the up-to-32
/// per-split partials, exactly llama.cpp's `kernel_flash_attn_ext_vec_
/// reduce` shape (`ggml-metal-ops.cpp:2063-2097` on `origin/master`).
#[test]
fn merge_kernel_emits_the_online_softmax_combine() {
    let bound = cached_attention_op_dynamic(200, 56);
    let source = render_cached_attention_merge(&bound, "entry")
        .expect("the merge kernel always renders");

    assert!(source.contains("long splits = u.splits;"));
    assert!(source.contains("float global_max = simd_max(own_max);"));
    assert!(source.contains("float total_sum = simd_sum(own_weight * own_sum);"));
    assert!(
        source.contains("for (long split = 0; split < splits; split++)"),
        "each lane's own dimensions must accumulate over every live split"
    );
}

/// Redesign §5 option 2, the ROW 369 residual this closes: crossing a
/// `kv-capacity-bucket` boundary (39 rows of compiled capacity vs. 71)
/// on the single-range fused (dynamic) path must neither change
/// `entry_name` (the plan-cache key) nor the rendered MSL text -- both
/// values now depend only on the compiled MAXIMUM chunk count (`cap`),
/// never the live capacity, so one compiled `MTLComputePipelineState`
/// serves both buckets and `pipeline_compile_ms` never fires again at a
/// crossing (ROW 369's own residual, item 3).
#[test]
fn dynamic_cached_attention_kernel_identity_is_stable_across_kv_capacity_buckets() {
    let smaller_bucket = cached_attention_op_dynamic(32, 7); // capacity 39
    let larger_bucket = cached_attention_op_dynamic(64, 7); // capacity 71
    // `cached_attention_op_dynamic` folds both arguments into
    // `new_key_rows` alone (`cached_key_rows` stays the real, structural
    // `0` every single-range op carries) -- the compiled capacity this
    // test crosses is that SUM, so the fixture-sanity check reads it off
    // `new_key_rows`, not `cached_key_rows`.
    assert_ne!(
        match &smaller_bucket.kind {
            BoundOpKind::CachedAttention { new_key_rows, .. } => *new_key_rows,
            _ => unreachable!(),
        },
        match &larger_bucket.kind {
            BoundOpKind::CachedAttention { new_key_rows, .. } => *new_key_rows,
            _ => unreachable!(),
        },
        "the fixture must actually cross a different compiled capacity"
    );

    let smaller_name = entry_name(&smaller_bucket);
    let larger_name = entry_name(&larger_bucket);
    assert_eq!(
        smaller_name, larger_name,
        "kernel identity must be capacity-free on the dynamic path: got {smaller_name:?} \
         vs {larger_name:?}"
    );
    assert!(
        !smaller_name.contains("_c32") && !smaller_name.contains("_n7"),
        "row-count tokens must not appear in the dynamic path's entry name: {smaller_name:?}"
    );

    let smaller_source =
        render_cached_attention(&smaller_bucket, "entry", NumericPolicy::default())
            .expect("smaller bucket renders");
    let larger_source =
        render_cached_attention(&larger_bucket, "entry", NumericPolicy::default())
            .expect("larger bucket renders");
    assert_eq!(
        smaller_source, larger_source,
        "the generated MSL text itself must be capacity-free on the dynamic path"
    );
    assert!(
        smaller_source.contains("shared_m["),
        "the merge block is always present in the unified dynamic body"
    );
    assert!(
        smaller_source.contains("if (chunk < chunks)"),
        "chunk activity is a runtime guard, not a compile-time branch"
    );
    assert!(
        !smaller_source.contains("constexpr long context_chunks"),
        "context_chunks must be a runtime uniform, never baked as constexpr, \
         on the dynamic path"
    );
}

/// §4b's `block_width == 1` arm ([`block_width_for`] under `bit_exact`,
/// where `NumericRewrite::TreeReduce` is rejected) must render the exact
/// strictly-sequential per-key body -- never the `ss[]`-staged,
/// `simd_shuffle_down`-reducing block body -- so a `bit_exact` plan's
/// generated MSL text never regresses onto the reassociated path.
#[test]
fn block_width_one_renders_the_strictly_sequential_body_never_the_block_staged_one() {
    let bound = cached_attention_op_dynamic(32, 7);

    let source = render_cached_attention(&bound, "entry", NumericPolicy::bit_exact())
        .expect("bit_exact renders");

    assert!(
        !source.contains("block_width"),
        "bit_exact (block_width=1) must not emit the block-staging constant"
    );
    assert!(
        !source.contains("simd_shuffle_down"),
        "bit_exact (block_width=1) must not emit the block's tree-reduce shuffles"
    );
    assert!(
        source.contains("simd_broadcast_first(simd_sum(partial_score))"),
        "bit_exact (block_width=1) must still emit today's per-key simd_sum reduce"
    );
}

/// §4b's `block_width > 1` arm ([`block_width_for`] under
/// `llama_relaxed`, which grants `NumericRewrite::TreeReduce`) must
/// render the block-staged Q·K/softmax/V body: `ss[]` staging, the
/// 8-lane `simd_shuffle_down` tree reduce, and a `simd_max`/`simd_sum`
/// combine fired once per 32-lane sub-block rather than once per key.
#[test]
fn block_width_above_one_renders_the_block_staged_body_under_llama_relaxed() {
    let mut bound = cached_attention_op_dynamic(32, 7);
    let BoundOpKind::CachedAttention { head_dim, rotary_dim, .. } = &mut bound.kind else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    *head_dim = 8; // a multiple of 8, so the float4 K/Q loads stay aligned
    *rotary_dim = 8;

    let source = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
        .expect("llama_relaxed renders the block-staged body for an aligned head_dim");

    assert!(
        source.contains("block_width"),
        "the block-staging constant must be emitted"
    );
    assert!(
        source.contains("threadgroup float ss["),
        "scores must stage into ss[]"
    );
    assert!(
        source.contains("simd_shuffle_down(partial_score, 4)"),
        "the Q*K reduce must tree-reduce over the 8-lane key group"
    );
    assert!(
        source.contains("simd_max("),
        "one simd_max per 32-lane sub-block"
    );
    assert!(
        !source.contains("simd_broadcast_first(simd_sum(partial_score)) * scale"),
        "the block-staged body must not keep the old per-key reduce expression"
    );
}

/// §4b's V-accumulate port (this crate's `attention-kernel-design.md`
/// §4b, llama's `kernel_flash_attn_ext_vec` register form,
/// ggml-metal.metal:4125-4197): once `head_dim` is a multiple of 32 the
/// block-staged body must load V as `float4` per `tx` lane and fold the
/// four `ty` groups' partial sums back together with a `simd_shuffle_xor`
/// cross-group reduce, rather than the scalar per-key V loop the
/// smaller/unaligned `head_dim` fixtures above still render.
#[test]
fn block_staged_v_accumulate_uses_float4_and_a_cross_ty_reduce_when_aligned() {
    let mut bound = cached_attention_op_dynamic(32, 7);
    let BoundOpKind::CachedAttention { head_dim, rotary_dim, .. } = &mut bound.kind else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    *head_dim = 32; // a multiple of 32, so the V float4 loads stay aligned
    *rotary_dim = 32;

    let source = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
        .expect("llama_relaxed renders the block-staged body for a 32-aligned head_dim");

    assert!(
        source
            .contains("float4* v4 = (device const float4*)((cached ? in6 : in7) + kbase * 2)"),
        "V must be reinterpreted as a float4 pointer at the same kbase * 2 offset the \
         scalar form used"
    );
    assert!(
        source.contains("v_acc[register_index] += valid ? float4(v4[(long)tx + 8L * register_index]) * key_weight"),
        "each tx lane accumulates its own float4 chunk of V, weighted by its ty group's own key"
    );
    assert!(
        source.contains("v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 8)")
            && source.contains(
                "v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 16)"
            ),
        "the four ty groups' partials must be folded together by a butterfly \
         simd_shuffle_xor reduce (strides 8 then 16), not left un-combined"
    );
    assert!(
        !source.contains(
            "weighted[local_dimension] += key_weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);"
        ),
        "an aligned head_dim must not keep the scalar per-key V loop the unaligned fixtures use"
    );
}

/// The scalar V-accumulate fallback (§4b: `head_dim` not a multiple of
/// 32) must still be exactly today's per-key loop, byte for byte --
/// `block_width_above_one_renders_the_block_staged_body_under_llama_relaxed`
/// covers the Q·K side at this same `head_dim = 8`; this pins the V side.
#[test]
fn block_staged_v_accumulate_falls_back_to_scalar_when_head_dim_not_32_aligned() {
    let mut bound = cached_attention_op_dynamic(32, 7);
    let BoundOpKind::CachedAttention { head_dim, rotary_dim, .. } = &mut bound.kind else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    *head_dim = 8; // multiple of 8 (Q/K stays aligned) but not of 32
    *rotary_dim = 8;

    let source = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
        .expect("llama_relaxed renders the block-staged body for an unaligned head_dim");

    assert!(
        source.contains(
            "weighted[local_dimension] += key_weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);"
        ),
        "an unaligned head_dim must keep the scalar per-key V loop"
    );
    assert!(
        !source.contains("simd_shuffle_xor"),
        "an unaligned head_dim must never emit the float4 cross-ty reduce"
    );
}

/// `bit_exact()` still withholds [`NumericRewrite::ContextSplitMerge`]
/// at the scoreboard window (40 keys) regardless of `keys_per_split` --
/// bit-identical output has no split/merge path at all under that
/// policy. Under `llama_relaxed()`, ROW 381 lowered `omega-runtime.toml`'s
/// `[attention_splits] keys_per_split` from 128 to 16 (measured 36.1us
/// vs 43.6us bare at this window, beyond 2x CoV), so this window now
/// DOES split: `ceil(40/16) = 3`, not the old single-threadgroup form.
#[test]
fn forty_keys_stays_one_split_under_bit_exact_but_splits_under_llama_relaxed() {
    assert_eq!(splits_for(40, NumericPolicy::bit_exact()), 1);
    assert_eq!(splits_for(40, NumericPolicy::llama_relaxed()), 3);
}

/// `bit_exact()` withholds `ContextSplitMerge` regardless of context
/// length -- the split kernel-plus-merge machinery must never engage
/// under a policy that demands bit-identical output, the same
/// `admit`-rejects-so-fall-back-to-`1` shape [`context_chunks_for`]/
/// [`block_width_for`] already use.
#[test]
fn long_contexts_stay_one_split_under_bit_exact() {
    assert_eq!(splits_for(512, NumericPolicy::bit_exact()), 1);
    assert_eq!(splits_for(4096, NumericPolicy::bit_exact()), 1);
}

/// `llama_relaxed()` grants reassociation, clearing
/// `ContextSplitMerge`, so ROW 376's longer decode shapes (512/4096
/// keys) must split across more than one threadgroup -- the whole
/// point of the port: more of the GPU's 32 cores get a share of one
/// attention op on the contexts where today's 8-threadgroup dispatch
/// (`query_rows * kv_heads` for openchat's shape) leaves most of the
/// GPU idle.
#[test]
fn long_contexts_split_across_more_than_one_threadgroup_under_llama_relaxed() {
    let splits_512 = splits_for(512, NumericPolicy::llama_relaxed());
    let splits_4096 = splits_for(4096, NumericPolicy::llama_relaxed());
    assert!(
        splits_512 > 1,
        "512 keys (4x omega-runtime.toml's 128-key keys_per_split_at_scale) must split; got {splits_512}"
    );
    assert!(splits_4096 > 1, "4096 keys must split; got {splits_4096}");
    assert!(
        splits_4096 >= splits_512,
        "a longer context must never split into FEWER threadgroups than a shorter one"
    );
}

/// ROW 383's own knee, pinned at the five lengths its own log row
/// tabulates: below `keys_per_split_at_scale` (128) the small divisor
/// (16) applies (40 keys -> 3 splits, ROW 381's decode-window win,
/// unchanged); at or above it the large divisor (128, the ORIGINAL
/// pre-ROW-381 value) applies, which is what measured fastest at 512
/// keys (splits=4, 89.5us bare vs 170-177us at 8/16/32) and reaches the
/// SAME split count (32) as the small divisor already did once clamped
/// to `max` at 4096 keys, so nothing regresses at scale.
#[test]
fn splits_for_has_a_knee_at_the_scaled_divisor_threshold() {
    assert_eq!(splits_for(40, NumericPolicy::llama_relaxed()), 3);
    assert_eq!(splits_for(256, NumericPolicy::llama_relaxed()), 2);
    assert_eq!(splits_for(512, NumericPolicy::llama_relaxed()), 4);
    assert_eq!(splits_for(1024, NumericPolicy::llama_relaxed()), 8);
    assert_eq!(splits_for(4096, NumericPolicy::llama_relaxed()), 32);
}

/// `ATTENTION_SPLIT_MAX` (`omega-runtime.toml`'s `[attention_splits].max`,
/// default 32) is a hard ceiling regardless of how long the context
/// grows -- the compiled dispatch always issues exactly this many
/// threadgroups per `(query_row, kv_head)` pair on the dynamic path
/// (`grid_threads`'s own doc), so the live count this function returns
/// can never exceed it without under-provisioning the grid.
#[test]
fn split_count_never_exceeds_the_compiled_maximum() {
    let splits = splits_for(1_000_000, NumericPolicy::llama_relaxed());
    assert_eq!(splits, crate::sized::ATTENTION_SPLIT_MAX);
}

/// [`EmitError::AttentionBlockMisaligned`]'s own reachable gate: a
/// `head_dim` not a multiple of 8 leaves `head_dim / 2` not a multiple
/// of 4, so the block-staged body's `float4` K/Q reinterpretation would
/// read past an unaligned offset -- rejected here, named, rather than
/// emitted.
#[test]
fn block_staged_attention_rejects_a_head_dim_not_a_multiple_of_eight() {
    let mut bound = cached_attention_op_dynamic(32, 7);
    let BoundOpKind::CachedAttention { head_dim, rotary_dim, .. } = &mut bound.kind else {
        unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
    };
    *head_dim = 12; // not a multiple of 8
    *rotary_dim = 12;

    let error = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
        .expect_err("a misaligned head_dim must be rejected under the block-staged policy");

    assert_eq!(
        error,
        EmitError::AttentionBlockMisaligned {
            node: bound.node,
            head_dim: 12,
        }
    );
}
