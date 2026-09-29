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

/// `[rows, k] x [tokens, k] -> [tokens, rows]`, reduced over `k`, with the
/// packed weight and the activation EACH storing `k` as their own innermost
/// (contiguous, stride-1) dim -- the shape
/// `packed_row_multi_row_unroll_ab.rs`'s own `matmul_program` builds.
/// Unlike [`matmul_op`] (a literal `[m,k] x [k,n] -> [m,n]` matrix multiply,
/// where the SECOND operand's own `k` dim is never contiguous), this is the
/// shape `push_packed_row_blocked_body`'s multi-row admission
/// (`split_token_feature_axes`, `classify_packed_row_block`'s own doc) needs
/// to actually classify a token axis: output axis 0 (`tokens`) depends only
/// on the activation, axis 1 (`rows`) only on the weight, so `token_axes =
/// [0]` and `feature_axes = [1]` — the `[token, feature]` output order
/// `split_token_feature_axes` requires for `reassembled == output_axes` to
/// hold.
fn packed_row_multi_token_op(tokens: u32, k: u32, rows: u32) -> BoundOp {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
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
                (weight, IndexMap::Affine(map::projection(3, &[1, 2]))),
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
            name: Some("packed_row_multi_token_matmul".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("packed row multi-token op infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("packed row multi-token op lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

/// Every multi-row-experiment test in this module reads a `PROXIMA_MULTI_
/// ROW_*` env var, and `cargo test` (unlike `cargo nextest run`, which gives
/// each test its own process) runs this module's tests as multiple THREADS
/// sharing one process -- env vars are process-global, so an unguarded
/// "baseline" read (no override set) can observe a DIFFERENT test's
/// concurrently-active `temp_env::with_var` override. `temp_env`'s own
/// internal lock serializes `with_var` calls against each other but not
/// against an unguarded read, so every baseline capture must ALSO go
/// through `with_var` (explicitly unset) to take that same lock.
fn with_every_multi_row_env_unset<T>(closure: impl FnOnce() -> T) -> T {
    temp_env::with_var("PROXIMA_MULTI_ROW_UNROLL", None::<&str>, || {
        temp_env::with_var("PROXIMA_MULTI_ROW_INDEX32", None::<&str>, || {
            temp_env::with_var("PROXIMA_COORD_INDEX32", None::<&str>, closure)
        })
    })
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
        Binding, Codec, EmitError, emit, round_zero_reduce_bound,
        splice_round_batched_reduce_base_table,
    };
    use super::round_batched_matmul_op;
    use proxima_tensor::{BoundOpKind, NodeId};

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

    #[test]
    fn each_round_reads_the_route_buffer_bound_at_its_own_slot() -> Result<(), EmitError> {
        let mut round_batched = round_batched_matmul_op(4);
        let BoundOpKind::RoundBatchedReduce { round_routes, .. } = &mut round_batched.kind else {
            unreachable!("round_batched_matmul_op builds a RoundBatchedReduce")
        };
        let sibling_routes: Vec<NodeId> = (0..4).map(|round| NodeId(900 + round)).collect();
        round_routes.clone_from(&sibling_routes);
        let weight_node = round_zero_reduce_bound(&round_batched).operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, Codec::Q4K);

        let kernel = emit(&round_batched, &q4k, NumericPolicy::default())?;

        let route_slots_start = kernel.bindings.len() - sibling_routes.len();
        let expected_tail: Vec<Binding> = sibling_routes
            .iter()
            .map(|route| Binding::Indices(*route))
            .collect();
        assert_eq!(
            kernel.bindings[route_slots_start..],
            expected_tail,
            "the last k bindings must be the k rounds' own routes, in round order, so the hazard \
             walk sees each MoeTopK write as a read of this dispatch"
        );
        for round in 0..sibling_routes.len() {
            let declaration = format!(
                "route_buf_{round} [[buffer({})]]",
                route_slots_start + round
            );
            assert!(
                kernel.source.contains(&declaration),
                "round {round}'s route_buf must sit at the slot its binding occupies \
                 ({declaration}):\n{}",
                kernel.source
            );
        }
        let table_declaration = format!("round_table [[buffer({})]]", kernel.bindings.len());
        assert!(
            kernel.source.contains(&table_declaration),
            "the round table binds right after every kernel binding ({table_declaration}):\n{}",
            kernel.source
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
    // `metal-q4_0-native` (default-off) takes priority over the batched
    // `q4_0_pair_dot` arm for a plain-product reduce -- see
    // `push_q4_0_native_body`'s own doc. Under that feature this same
    // matmul renders ggml's own inline dot (`sumy * -8.0f`), not a call to
    // the named accessor, so the assertion below is feature-conditional
    // rather than a second copy of this whole test.
    if cfg!(feature = "metal-q4_0-native") {
        assert!(
            source.contains("sumy * -8.0f"),
            "metal-q4_0-native must render ggml's inline nibble dot for a plain-product Q4_0 matmul:\n{source}"
        );
        assert!(
            !source.contains("q4_0_pair_dot(blk") && !source.contains("q4_0_super_element(blk"),
            "metal-q4_0-native must fully replace both the pair-dot and per-element accessors:\n{source}"
        );
    } else {
        assert!(
            source.contains("q4_0_pair_dot(blk"),
            "a plain-product row-blocked Q4_0 weight must call the batched pair-dot accessor:\n{source}"
        );
        assert!(
            !source.contains("q4_0_super_element(blk"),
            "the batched arm must fully replace the per-element accessor for a plain product:\n{source}"
        );
    }
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
    // Real weight layout: heads outermost, head_dim next, embed (the
    // reduce dim) innermost -- `classify_packed_row_block` requires the
    // packed operand's reduce-dim stride to be exactly 1
    // (`NonUnitWeightStride`), the same contiguity every real
    // `attn_q`/`attn_k`/`attn_v` GGUF weight has. The original
    // `[embed, heads, head_dim]` shape put the reduce dim OUTERMOST
    // (stride `heads * head_dim`), which is not a layout any real packed
    // weight ever takes -- `classify_packed_row_block` correctly declines
    // it (`NonUnitWeightStride { stride: 1024 }`), so the fixture never
    // reached the row-blocked gate this test means to exercise.
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(heads),
                Extent::Static(head_dim),
                Extent::Static(embed),
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
                (weight, IndexMap::Affine(map::projection(4, &[1, 2, 3]))),
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
            // `[token, head_dim, heads]` -- head_dim BEFORE heads, the
            // reverse of the weight's own outer-to-inner nesting (heads
            // outer, head_dim inner). This is the genuinely
            // non-contiguous multi-feature-axis shape
            // `classify_tiled_gemm`'s axis-group fold still declines:
            // reversing the declared order breaks `axes_fold_contiguously`
            // for the feature group even though each axis individually is
            // weight-owned -- unlike the NATURAL `[token, heads, head_dim]`
            // order, which folds contiguously and (correctly, since
            // ROW 114) now admits the tiled-GEMM path.
            out_map: IndexMap::Affine(map::projection(4, &[0, 2, 1])),
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

/// The REAL node-139-shaped ("score", this repo's own admitted
/// census) dense (unquantized) batched matmul: `[token, feature, batch,
/// reduce] -> [token, feature, batch]`, weight (operand 0) reading
/// `(feature, batch, reduce)`, other (operand 1) reading `(token, batch,
/// reduce)` -- a real stride on `batch` for BOTH operands (no broadcast),
/// the exact shape `classify_dense_batched_gemm` classifies as one row
/// axis, one column axis, one batch axis.
#[cfg(feature = "metal-tiled-gemm")]
fn dense_batched_score_shaped_op(token: u32, feature: u32, batch: u32, reduce_len: u32) -> BoundOp {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(feature),
                Extent::Static(batch),
                Extent::Static(reduce_len),
            ],
            name: None,
        },
    );
    let other = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(token),
                Extent::Static(batch),
                Extent::Static(reduce_len),
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
                (weight, IndexMap::Affine(map::projection(4, &[1, 2, 3]))),
                (other, IndexMap::Affine(map::projection(4, &[0, 2, 3]))),
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
            name: Some("dense_batched_score".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("dense batched score op infers");
    bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("dense batched score op lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

/// Dumps the dense-batched (`score_even`/`score_odd`) copy-out loop's
/// emitted MSL, at the real gemma4-E2B weather-prompt shape
/// (`feature_extent=576 token_extent=563 batch_extent=8
/// reduction_k=128`, batch-innermost -- `DIRECT_STORE` never engages for
/// this layout), to `PROXIMA_STAGE_DUMP_DIR` when that env var is set -- a
/// no-op assertion-only pass otherwise, mirroring
/// `wide_weight_stage_msl_dump_for_the_real_q4_0_shape`'s own dump
/// convention so the rewritten copy-out loop's IR can be inspected directly
/// (`xcrun -O2 -S -emit-llvm`) rather than only measured through the
/// byte-parity tests.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn dense_batched_restage_msl_dump_for_the_real_weather_score_shape() {
    let bound = dense_batched_score_shaped_op(563, 576, 8, 128);
    let source = temp_env::with_var("PROXIMA_TILED_GEMM_DENSE", Some("1"), || {
        emit(&bound, &BTreeMap::new(), NumericPolicy::default())
            .expect("emits")
            .source
    });
    assert!(
        source.contains("out_offset_base"),
        "rewritten copy-out loop must hoist the batch/base offset out of the per-element loop"
    );
    if let Ok(dir) = std::env::var("PROXIMA_STAGE_DUMP_DIR") {
        std::fs::write(format!("{dir}/dense_score_weather.metal"), &source)
            .expect("writes the dense score MSL dump");
    }
}

/// Coordinator-required proof (2026-09-27 mid-task addition, updated when
/// `PROXIMA_TILED_GEMM_DENSE` flipped to default-on): a small
/// assertion-bearing test, run against THIS tree's own `classify_dense_
/// batched_gemm`, that the dense-batched admission count is nonzero with the
/// switch on (either unset, the new default, or explicit `"1"`) and exactly
/// zero with it explicitly disabled (`"0"`) -- the same property
/// a real 510-token prefill run shows
/// (`admitted=true` count 210 on, 0 off), reproduced here as a fast,
/// deterministic, GPU-free unit test rather than only a captured log.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn dense_batched_gemm_admission_is_switch_gated() {
    use alloc::collections::BTreeMap;

    let shapes: &[(u32, u32, u32, u32)] = &[(510, 512, 8, 128), (16, 32, 8, 128), (64, 96, 4, 256)];

    let count_admitted = |var_value: Option<&str>| -> usize {
        temp_env::with_var("PROXIMA_TILED_GEMM_DENSE", var_value, || {
            shapes
                .iter()
                .filter(|&&(token, feature, batch, reduce_len)| {
                    let bound = dense_batched_score_shaped_op(token, feature, batch, reduce_len);
                    let codecs = operand_codecs(&bound, &BTreeMap::new());
                    let BoundOpKind::Reduce { reduce_op, init, ref output_axes, .. } = bound.kind
                    else {
                        panic!("dense_batched_score_shaped_op always builds a Keep::Reduce fold")
                    };
                    classify_dense_batched_gemm(&bound, &codecs, reduce_op, init, output_axes).is_ok()
                })
                .count()
        })
    };

    let admitted_on = count_admitted(Some("1"));
    assert!(
        admitted_on > 0,
        "PROXIMA_TILED_GEMM_DENSE=1 must admit at least one of the real score-shaped ops"
    );
    assert_eq!(
        admitted_on,
        shapes.len(),
        "every one of these no-broadcast-batch score-shaped ops must admit when the switch is on"
    );

    let admitted_default = count_admitted(None);
    assert_eq!(
        admitted_default,
        shapes.len(),
        "PROXIMA_TILED_GEMM_DENSE unset (new default) must admit every one of these ops, same as \
         explicit \"1\""
    );

    let admitted_off = count_admitted(Some("0"));
    assert_eq!(
        admitted_off, 0,
        "PROXIMA_TILED_GEMM_DENSE=0 must admit none of these ops (explicit disable)"
    );
}

/// Regression: `grid_threads`' own `Reduce` arm lost its
/// `dense_batched_gemm_block` check somewhere between the design session
/// that added it and the commit that landed the feature on main -- it
/// stayed in `tiled_gemm_threadgroup_width` (that function's own
/// `dense_batched_gemm_block` arm, just above), so a dense-admitted op fell
/// through to the `reduce_is_cooperative_dispatch` branch below and
/// dispatched `output_total * cooperative_reduce_width` threads (`feature *
/// token * batch * SIMD_WIDTH`) while the body still rendered
/// [`push_dense_batched_gemm_body`]'s tiled shape (`tile_index = gid /
/// block_threads`, masked only on the final write) -- thousands of times
/// more threadgroups than the real tile grid, each redundantly executing
/// the full tile-load-and-accumulate loop. Node-139 (score) and node-162
/// (value) shaped ops, this repo's own admitted-census shapes.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn dense_batched_gemm_grid_spec_matches_tiled_shape() {
    use alloc::collections::BTreeMap;

    let cases: &[(&str, u32, u32, u32, u32)] = &[
        ("node139_score", 510, 512, 8, 128),
        ("node162_value", 510, 256, 8, 512),
    ];

    temp_env::with_var("PROXIMA_TILED_GEMM_DENSE", Some("1"), || {
        for &(name, token, feature, batch, reduce_len) in cases {
            let bound = dense_batched_score_shaped_op(token, feature, batch, reduce_len);
            let codecs = operand_codecs(&bound, &BTreeMap::new());
            let BoundOpKind::Reduce {
                reduce_op,
                init,
                ref output_axes,
                ..
            } = bound.kind
            else {
                panic!("dense_batched_score_shaped_op always builds a Keep::Reduce fold")
            };
            let block = dense_batched_gemm_block(&bound, &codecs, reduce_op, init, output_axes)
                .unwrap_or_else(|| panic!("{name} must admit to the dense-batched-gemm path"));
            let _ = block;

            let row_tiles = u64::from(feature).div_ceil(crate::sized::TILED_GEMM_BLOCK_M);
            let col_tiles = u64::from(token).div_ceil(crate::sized::TILED_GEMM_BLOCK_N);
            let block_threads = (TILED_GEMM_NSG as u64) * SIMD_WIDTH;
            let expected_threads = row_tiles * col_tiles * block_threads;
            let expected_threadgroups = row_tiles * col_tiles;
            let cooperative_fallback_threads =
                u64::from(feature) * u64::from(token) * u64::from(batch) * SIMD_WIDTH;

            let grid_threads_total = grid_threads(&bound, &codecs, NumericPolicy::default(), false)
                .unwrap_or_else(|err| panic!("{name}: grid_threads must not error: {err:?}"));
            assert_eq!(
                grid_threads_total, expected_threads,
                "{name}: grid_threads must dispatch the tiled-gemm x-count ({row_tiles} row \
                 tiles x {col_tiles} col tiles x {block_threads} threads = {expected_threads}), \
                 not the cooperative-reduce fallback (feature*token*batch*SIMD_WIDTH = \
                 {cooperative_fallback_threads})"
            );

            let packed_operands = PackedOperands::new();
            let (_, grid) =
                kernel_dispatch_shape(&bound, &packed_operands, NumericPolicy::default())
                    .unwrap_or_else(|err| {
                        panic!("{name}: kernel_dispatch_shape must not error: {err:?}")
                    });
            assert_eq!(grid.threads, expected_threads, "{name}: GridSpec.threads mismatch");
            assert_eq!(
                grid.threadgroup_width,
                Some(block_threads),
                "{name}: GridSpec.threadgroup_width must be TILED_GEMM_NSG * SIMD_WIDTH"
            );
            assert_eq!(
                grid.depth,
                u64::from(batch),
                "{name}: GridSpec.depth must carry the batch axes' flattened extent"
            );

            let dispatched_threadgroups = grid.threads / grid.threadgroup_width.unwrap_or(1);
            assert_eq!(
                dispatched_threadgroups, expected_threadgroups,
                "{name}: dispatched threadgroups (before the z/depth multiply) must equal the \
                 tile grid ({expected_threadgroups}), not a cooperative-reduce blowup"
            );
        }
    });
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
        codec: Codec::Q4K,
    };
    let mut source = String::new();
    let error = push_tiled_gemm_body(
        &mut source,
        bound.node,
        &[0],
        2,
        &block,
        "float",
        &ComposedBody::leaf(ScalarOp::Identity),
        &[],
        &crate::identity::MetalOnlyExtras::default(),
    )
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
        codec: Codec::Q4K,
    };
    let mut source = String::new();
    let error = push_tiled_gemm_body(
        &mut source,
        bound.node,
        &[0],
        2,
        &block,
        "float",
        &ComposedBody::leaf(ScalarOp::Identity),
        &[],
        &crate::identity::MetalOnlyExtras::default(),
    )
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
    let error = push_tiled_gemm_body(
        &mut source,
        bound.node,
        &[0],
        2,
        &block,
        "float",
        &ComposedBody::leaf(ScalarOp::Identity),
        &[],
        &crate::identity::MetalOnlyExtras::default(),
    )
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

/// a decode-shaped (`token_total == 1`) op must never carry `_u`, and its
/// cache key/source/grid/threadgroup_width must be identical regardless of
/// `PROXIMA_MULTI_ROW_UNROLL`, since unroll changes only body text, never
/// dispatch geometry.
#[test]
fn multi_row_unroll_decode_shape_keeps_current_key_source_grid_and_width() {
    let bound = packed_row_multi_token_op(1, 256, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let (baseline_key, baseline_source, baseline_dispatch) = with_every_multi_row_env_unset(|| {
        let key = kernel_cache_key(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline cache key");
        let source = emit(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline emits")
            .source;
        let dispatch = kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline dispatch shape");
        (key, source, dispatch)
    });

    let (shared_key, shared_source, shared_dispatch) =
        temp_env::with_var("PROXIMA_MULTI_ROW_UNROLL", Some("1"), || {
            let key = kernel_cache_key(&bound, &q4_0, NumericPolicy::default())
                .expect("unroll-env cache key");
            let source = emit(&bound, &q4_0, NumericPolicy::default())
                .expect("unroll-env emits")
                .source;
            let dispatch = kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
                .expect("unroll-env dispatch shape");
            (key, source, dispatch)
        });

    assert!(
        !baseline_key.contains("_u"),
        "a decode-shaped op must never carry the _u suffix: {baseline_key}"
    );
    assert_eq!(
        baseline_key, shared_key,
        "decode-shaped cache key must be identical whether PROXIMA_MULTI_ROW_UNROLL is set"
    );
    assert_eq!(
        baseline_source, shared_source,
        "decode-shaped emitted source must be byte-identical whether the override is set"
    );
    assert_eq!(
        baseline_dispatch.1.threads, shared_dispatch.1.threads,
        "decode-shaped grid_threads must be identical whether the override is set"
    );
    assert_eq!(
        baseline_dispatch.1.threadgroup_width, shared_dispatch.1.threadgroup_width,
        "decode-shaped threadgroup_width must be identical whether the override is set"
    );
}

/// `c4-7-reduction-literal.md` AC1: with `PROXIMA_REDUCTION_LITERAL` unset
/// (today's default posture, whether or not `metal-reduction-literal` is
/// even compiled in), [`MetalOnlyExtras::reduction_literal`] must stay
/// `None` and the cache key must carry no `_rl` token -- the unset-env
/// default folds no new token into the key, matching every other override in
/// this crate.
#[test]
fn reduction_literal_default_is_none_and_key_carries_no_rl_token() {
    let bound = matmul_op(4, 6144, 3);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let key = temp_env::with_var("PROXIMA_REDUCTION_LITERAL", None::<&str>, || {
        kernel_cache_key(&bound, &q4_0, NumericPolicy::default()).expect("matmul_op emits")
    });
    assert!(
        !key.contains("_rl"),
        "unset PROXIMA_REDUCTION_LITERAL must never bake a reduction-length literal into the \
         cache key: {key}"
    );
}

/// `c4-7-reduction-literal.md` AC5's negative boundary: `K=288` is not a
/// whole multiple of `Q4K_BLOCK_ELEMENTS` (256), so
/// `classify_packed_row_block` rejects it (`ExtentNotBlockMultiple`) and
/// [`packed_row_block`] returns `None` -- `reduction_literal_value`'s own
/// admission short-circuits on that `None` regardless of the env override,
/// so the key must carry no `_rl` token even with `PROXIMA_REDUCTION_LITERAL=1`.
#[cfg(feature = "metal-reduction-literal")]
#[test]
fn reduction_literal_is_none_for_a_non_block_multiple_extent() {
    let bound = matmul_op(4, 288, 3);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let key = temp_env::with_var("PROXIMA_REDUCTION_LITERAL", Some("1"), || {
        kernel_cache_key(&bound, &q4_0, NumericPolicy::default()).expect("matmul_op emits")
    });
    assert!(
        !key.contains("_rl"),
        "K=288 is not a multiple of 256 -- classify_packed_row_block rejects it \
         (ExtentNotBlockMultiple), so reduction_literal must stay None even with the env A/B on: \
         {key}"
    );
}

/// `c4-7-reduction-literal.md` AC5's positive boundary set: `K=256`
/// (minimum admitted length) and `K=12288` each yield `Some` and a distinct
/// `_rl{K}` token -- proven directly against [`matmul_op`]'s own synthetic
/// shape rather than only through the real-checkpoint device test, so this
/// runs without a Metal device or the real gguf blob.
#[cfg(feature = "metal-reduction-literal")]
#[test]
fn reduction_literal_some_at_the_minimum_and_a_wide_boundary() {
    for k in [256u32, 12288u32] {
        let bound = matmul_op(4, k, 3);
        let weight_node = bound.operands()[0].0;
        let mut q4_0 = BTreeMap::new();
        q4_0.insert(weight_node, Codec::Q4_0);

        let key = temp_env::with_var("PROXIMA_REDUCTION_LITERAL", Some("1"), || {
            kernel_cache_key(&bound, &q4_0, NumericPolicy::default()).expect("matmul_op emits")
        });
        assert!(
            key.contains(&format!("_rl{k}")),
            "K={k} is a whole multiple of 256 and packed_row_block admits it, so the key must \
             carry _rl{k}: {key}"
        );
    }
}

/// this switch's own toggle: `PROXIMA_REDUCTION_LITERAL=decode`
/// narrows admission to single-token ops. A genuinely multi-token op
/// ([`packed_row_multi_token_op`]'s own `token_axes = [0]` classification,
/// unlike [`matmul_op`]'s shape) must keep `reduction_literal` at `None` --
/// the same invariant [`reduction_literal_is_none_for_a_non_block_multiple_extent`]
/// checks for a different admission gate.
#[cfg(feature = "metal-reduction-literal")]
#[test]
fn reduction_literal_decode_mode_is_none_for_a_multi_token_op() {
    let bound = packed_row_multi_token_op(27, 6144, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let key = temp_env::with_var("PROXIMA_REDUCTION_LITERAL", Some("decode"), || {
        kernel_cache_key(&bound, &q4_0, NumericPolicy::default()).expect("multi-token op emits")
    });
    assert!(
        !key.contains("_rl"),
        "PROXIMA_REDUCTION_LITERAL=decode must never bake a literal for a 27-token op: {key}"
    );
}

/// `PROXIMA_REDUCTION_LITERAL=decode`'s positive case: a single-token
/// ([`packed_row_multi_token_op`] with `tokens = 1`) op still bakes, the
/// same admission `=1` already grants it.
#[cfg(feature = "metal-reduction-literal")]
#[test]
fn reduction_literal_decode_mode_is_some_for_a_single_token_op() {
    let bound = packed_row_multi_token_op(1, 6144, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let key = temp_env::with_var("PROXIMA_REDUCTION_LITERAL", Some("decode"), || {
        kernel_cache_key(&bound, &q4_0, NumericPolicy::default()).expect("single-token op emits")
    });
    assert!(
        key.contains("_rl6144"),
        "PROXIMA_REDUCTION_LITERAL=decode must bake a literal for a single-token op: {key}"
    );
}

/// [`multi_row_unroll_decode_shape_keeps_current_key_source_grid_and_width`]'s
/// multi-row counterpart: a real prefill shape (27 tokens) DOES take the
/// `_u` cache-key suffix, but its dispatch geometry is UNCHANGED (unroll
/// touches body text only, never `grid_threads`/`tiled_gemm_threadgroup_
/// width`).
#[test]
fn multi_row_unroll_27_token_shape_gets_u_suffix_with_unchanged_geometry() {
    let bound = packed_row_multi_token_op(27, 256, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let baseline_dispatch = with_every_multi_row_env_unset(|| {
        kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline dispatch shape")
    });

    let (shared_key, shared_dispatch) =
        temp_env::with_var("PROXIMA_MULTI_ROW_UNROLL", Some("1"), || {
            let key = kernel_cache_key(&bound, &q4_0, NumericPolicy::default())
                .expect("unroll-env cache key");
            let dispatch = kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
                .expect("unroll-env dispatch shape");
            (key, dispatch)
        });

    assert!(
        shared_key.contains("_u"),
        "a 27-token op with PROXIMA_MULTI_ROW_UNROLL=1 must carry the _u suffix: {shared_key}"
    );
    assert_eq!(
        shared_dispatch.1.threads, baseline_dispatch.1.threads,
        "unroll must not change grid_threads"
    );
    assert_eq!(
        shared_dispatch.1.threadgroup_width, baseline_dispatch.1.threadgroup_width,
        "unroll must not change threadgroup_width"
    );
}

/// [`multi_row_unroll_decode_shape_keeps_current_key_source_grid_and_width`]'s
/// index32 counterpart: checked both alone and with
/// `PROXIMA_MULTI_ROW_UNROLL` also on, since index32 composes with unroll.
#[test]
fn multi_row_index32_decode_shape_keeps_current_key_source_grid_and_width() {
    let bound = packed_row_multi_token_op(1, 256, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let (baseline_key, baseline_source, baseline_dispatch) = with_every_multi_row_env_unset(|| {
        let key = kernel_cache_key(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline cache key");
        let source = emit(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline emits")
            .source;
        let dispatch = kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline dispatch shape");
        (key, source, dispatch)
    });

    for (label, unroll_env) in [("index32 alone", None), ("unroll+index32", Some("1"))] {
        let (shared_key, shared_source, shared_dispatch) =
            temp_env::with_var("PROXIMA_MULTI_ROW_UNROLL", unroll_env, || {
                temp_env::with_var("PROXIMA_MULTI_ROW_INDEX32", Some("1"), || {
                    let key = kernel_cache_key(&bound, &q4_0, NumericPolicy::default())
                        .expect("index32-env cache key");
                    let source = emit(&bound, &q4_0, NumericPolicy::default())
                        .expect("index32-env emits")
                        .source;
                    let dispatch = kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
                        .expect("index32-env dispatch shape");
                    (key, source, dispatch)
                })
            });

        assert!(
            !baseline_key.contains("_i32"),
            "[{label}] a decode-shaped op must never carry the _i32 suffix: {baseline_key}"
        );
        assert_eq!(
            baseline_key, shared_key,
            "[{label}] decode-shaped cache key must be identical whether PROXIMA_MULTI_ROW_INDEX32 is set"
        );
        assert_eq!(
            baseline_source, shared_source,
            "[{label}] decode-shaped emitted source must be byte-identical whether the override is set"
        );
        assert_eq!(
            baseline_dispatch.1.threads, shared_dispatch.1.threads,
            "[{label}] decode-shaped grid_threads must be identical whether the override is set"
        );
        assert_eq!(
            baseline_dispatch.1.threadgroup_width, shared_dispatch.1.threadgroup_width,
            "[{label}] decode-shaped threadgroup_width must be identical whether the override is set"
        );
    }
}

/// [`multi_row_index32_decode_shape_keeps_current_key_source_grid_and_width`]'s
/// multi-row counterpart: a real prefill shape (27 tokens) DOES take the
/// `_i32` cache-key suffix alone, or `_u_i32` with unroll also on, with
/// UNCHANGED dispatch geometry either way (index32 touches body text only,
/// same as unroll -- neither changes `grid_threads`/`tiled_gemm_
/// threadgroup_width`).
#[test]
fn multi_row_index32_27_token_shape_gets_i32_suffix_with_unchanged_geometry() {
    let bound = packed_row_multi_token_op(27, 256, 256);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let baseline_dispatch = with_every_multi_row_env_unset(|| {
        kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline dispatch shape")
    });

    for (unroll_env, expected_suffix) in [(None, "_i32"), (Some("1"), "_u_i32")] {
        let (shared_key, shared_dispatch) =
            temp_env::with_var("PROXIMA_MULTI_ROW_UNROLL", unroll_env, || {
                temp_env::with_var("PROXIMA_MULTI_ROW_INDEX32", Some("1"), || {
                    let key = kernel_cache_key(&bound, &q4_0, NumericPolicy::default())
                        .expect("index32-env cache key");
                    let dispatch = kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
                        .expect("index32-env dispatch shape");
                    (key, dispatch)
                })
            });

        assert!(
            shared_key.contains(expected_suffix),
            "a 27-token op with unroll_env={unroll_env:?} PROXIMA_MULTI_ROW_INDEX32=1 must carry \
             the {expected_suffix} suffix: {shared_key}"
        );
        assert_eq!(
            shared_dispatch.1.threads, baseline_dispatch.1.threads,
            "index32 must not change grid_threads (unroll_env={unroll_env:?})"
        );
        assert_eq!(
            shared_dispatch.1.threadgroup_width, baseline_dispatch.1.threadgroup_width,
            "index32 must not change threadgroup_width (unroll_env={unroll_env:?})"
        );
    }
}

/// A shape whose packed weight's own flat element count
/// (`feature_total * reduction_total`) exceeds `u32::MAX` must NEVER be
/// admitted -- `multi_row_index32_active`'s own doc: `weight_base[q] + k`
/// ranges over exactly that space, and this admission rejects by that
/// arithmetic rather than by naming a codec or a hardcoded shape.
/// `65536 * 65792 = 4_313_157_632 > u32::MAX` (`4_294_967_295`), both
/// multiples of 256 so `classify_packed_row_block`'s own block-multiple gate
/// still admits the op structurally -- this is a rejection purely from the
/// index32 fit check, not from any other admission gate.
#[test]
fn multi_row_index32_oversized_shape_not_admitted() {
    let bound = packed_row_multi_token_op(27, 65792, 65536);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let (baseline_key, baseline_source, baseline_dispatch) = with_every_multi_row_env_unset(|| {
        let key = kernel_cache_key(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline cache key");
        let source = emit(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline emits")
            .source;
        let dispatch = kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
            .expect("baseline dispatch shape");
        (key, source, dispatch)
    });

    let (shared_key, shared_source, shared_dispatch) =
        temp_env::with_var("PROXIMA_MULTI_ROW_INDEX32", Some("1"), || {
            let key = kernel_cache_key(&bound, &q4_0, NumericPolicy::default())
                .expect("index32-env cache key");
            let source = emit(&bound, &q4_0, NumericPolicy::default())
                .expect("index32-env emits")
                .source;
            let dispatch = kernel_dispatch_shape(&bound, &q4_0, NumericPolicy::default())
                .expect("index32-env dispatch shape");
            (key, source, dispatch)
        });

    assert!(
        !shared_key.contains("_i32"),
        "an oversized shape must never carry the _i32 suffix even with the override set: {shared_key}"
    );
    assert_eq!(
        baseline_key, shared_key,
        "an oversized shape's cache key must be identical whether PROXIMA_MULTI_ROW_INDEX32 is set"
    );
    assert_eq!(
        baseline_source, shared_source,
        "an oversized shape's emitted source must be byte-identical whether the override is set"
    );
    assert_eq!(
        baseline_dispatch.1.threads, shared_dispatch.1.threads,
        "an oversized shape's grid_threads must be identical whether the override is set"
    );
    assert_eq!(
        baseline_dispatch.1.threadgroup_width, shared_dispatch.1.threadgroup_width,
        "an oversized shape's threadgroup_width must be identical whether the override is set"
    );
}

/// `PROXIMA_COORD_INDEX32=1` at the GENERIC cooperative-reduce seam
/// (`coord_index32_active`'s own doc): `matmul_op` with NEITHER operand
/// quantized takes the plain cooperative-reduce path (not packed-row, not
/// tiled-GEMM), so this exercises `push_cooperative_reduce_body`'s
/// coordinate-decomposition branch directly. Override on vs off must differ
/// in EXACTLY the coordinate-decomposition lines (`remaining`'s declared
/// type and the two `u.output_extents[N]` divisor casts) -- proved by a
/// line-level diff, not just "not equal" -- carry the `_c32` suffix alone,
/// and leave `grid_threads`/`threadgroup_width` unchanged (this experiment
/// narrows body text only).
#[test]
fn coord_index32_generic_reduce_source_differs_only_in_coordinate_lines() {
    let bound = matmul_op(4, 65536, 5);
    let empty_codecs = BTreeMap::new();

    // `PROXIMA_COORD_INDEX32` now defaults ON, so the "baseline" this test
    // means -- the wide, non-narrowed coordinate decomposition -- must be
    // pinned explicitly rather than read off `with_every_multi_row_env_unset`.
    let (baseline_key, baseline_source, baseline_dispatch) =
        temp_env::with_var("PROXIMA_COORD_INDEX32", Some("0"), || {
            let key = kernel_cache_key(&bound, &empty_codecs, NumericPolicy::llama_relaxed())
                .expect("baseline cache key");
            let source = emit(&bound, &empty_codecs, NumericPolicy::llama_relaxed())
                .expect("baseline emits")
                .source;
            let dispatch = kernel_dispatch_shape(&bound, &empty_codecs, NumericPolicy::llama_relaxed())
                .expect("baseline dispatch shape");
            (key, source, dispatch)
        });

    let (shared_key, shared_source, shared_dispatch) =
        temp_env::with_var("PROXIMA_COORD_INDEX32", Some("1"), || {
            let key = kernel_cache_key(&bound, &empty_codecs, NumericPolicy::llama_relaxed())
                .expect("coord-index32-env cache key");
            let source = emit(&bound, &empty_codecs, NumericPolicy::llama_relaxed())
                .expect("coord-index32-env emits")
                .source;
            let dispatch = kernel_dispatch_shape(&bound, &empty_codecs, NumericPolicy::llama_relaxed())
                .expect("coord-index32-env dispatch shape");
            (key, source, dispatch)
        });

    assert!(
        shared_key.contains("_c32") && !baseline_key.contains("_c32"),
        "override must add exactly the _c32 suffix: baseline={baseline_key} shared={shared_key}"
    );
    assert_eq!(
        baseline_key.replace("_c32", ""),
        shared_key.replace("_c32", ""),
        "the only cache-key difference must be the _c32 suffix"
    );
    assert_eq!(
        baseline_dispatch.1.threads, shared_dispatch.1.threads,
        "coord_index32 must not change grid_threads"
    );
    assert_eq!(
        baseline_dispatch.1.threadgroup_width, shared_dispatch.1.threadgroup_width,
        "coord_index32 must not change threadgroup_width"
    );

    let baseline_lines: Vec<&str> = baseline_source.lines().collect();
    let shared_lines: Vec<&str> = shared_source.lines().collect();
    assert_eq!(
        baseline_lines.len(),
        shared_lines.len(),
        "override must not add or remove lines, only change coordinate-decomposition lines"
    );
    let mut differing_lines = Vec::new();
    for (index, (baseline_line, shared_line)) in baseline_lines.iter().zip(shared_lines.iter()).enumerate() {
        if baseline_line != shared_line {
            differing_lines.push((index, *baseline_line, *shared_line));
        }
    }
    assert!(
        !differing_lines.is_empty(),
        "coord_index32 must change at least one line when admitted"
    );
    for (index, baseline_line, shared_line) in &differing_lines {
        let touches_remaining = baseline_line.contains("remaining") && shared_line.contains("remaining");
        assert!(
            touches_remaining,
            "line {index} differs but is not a coordinate-decomposition line: \
             baseline={baseline_line:?} shared={shared_line:?}"
        );
    }
}

/// A decode/single-token-shaped op through the SAME generic cooperative-
/// reduce seam: unlike the `multi_row_*` family (which structurally
/// excludes `token_total == 1`), `coord_index32_active` has no per-token
/// concept at all -- it is admitted here (`n = 1` still gives every output
/// axis an extent in `[1, u32::MAX]`), and the narrowed vs wide decomposition
/// still produce byte-identical OUTPUT VALUES (this unit test proves the
/// narrowing is admitted and changes only the same coordinate lines; the
/// actual output-bit equality is the Metal A/B gate, not reachable from this
/// no-device unit test).
#[test]
fn coord_index32_single_token_shape_is_admitted_with_same_narrowing() {
    let bound = matmul_op(4, 4096, 1);
    let empty_codecs = BTreeMap::new();

    let shared_key = temp_env::with_var("PROXIMA_COORD_INDEX32", Some("1"), || {
        kernel_cache_key(&bound, &empty_codecs, NumericPolicy::llama_relaxed())
            .expect("coord-index32-env cache key")
    });

    assert!(
        shared_key.contains("_c32"),
        "a single-token op reaching the generic cooperative-reduce seam must still be admitted: {shared_key}"
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

/// Real-production-shaped matmul: weight `[features, k]` (k innermost),
/// activation `[tokens, k]` (k innermost), output `[tokens, features]`
/// (FEATURE innermost) -- `lowering_census.rs`'s own `matmul_op` convention,
/// the actual einsum shape the real spec's weight matmuls take. UNLIKE
/// `tiled_gemm_op` above (whose activation and output both happen to lay
/// out the OPPOSITE axis as contiguous -- confirmed empirically),
/// this fixture is what
/// item 3c (`wide_activation_load`, needs `k` contiguous on the
/// activation) actually admits.
#[cfg(feature = "metal-tiled-gemm")]
fn real_shaped_tiled_gemm_op(tokens: u32, k: u32, features: u32) -> (BoundOp, proxima_tensor::NodeId) {
    let mut program = Vec::new();
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(tokens), Extent::Static(k)],
            name: None,
        },
    );
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(features), Extent::Static(k)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (weight, IndexMap::Affine(map::projection(3, &[1, 2]))),
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
            name: Some("real_shaped_tiled_gemm".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("real-shaped tiled gemm op infers");
    let bound = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
        .expect("real-shaped tiled gemm op lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted");
    (bound, weight)
}

/// Dumps the wide-weight-stage kernel's emitted MSL for the real
/// `[K=1536, M=6144] x N=510` `Q4_0` shape (see `docs/model-interop/
/// discipline.md` ROW C4.12's own IR verification) to
/// `PROXIMA_STAGE_DUMP_DIR` when that env var is set --
/// a no-op assertion-only pass otherwise, so this stays a normal fast test
/// in every other run and never writes files in CI.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn wide_weight_stage_msl_dump_for_the_real_q4_0_shape() {
    let (bound, weight_node) = real_shaped_tiled_gemm_op(510, 1536, 6144);
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let off_source = emit(&bound, &q4_0, NumericPolicy::default()).expect("emits").source;
    let on_source = temp_env::with_var("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("1"), || {
        emit(&bound, &q4_0, NumericPolicy::default()).expect("emits").source
    });
    assert_ne!(off_source, on_source, "the switch must change the emitted source");

    if let Ok(dir) = std::env::var("PROXIMA_STAGE_DUMP_DIR") {
        std::fs::write(format!("{dir}/q4_0_wide_off.metal"), &off_source)
            .expect("writes the off-arm MSL dump");
        std::fs::write(format!("{dir}/q4_0_wide_on.metal"), &on_source)
            .expect("writes the on-arm MSL dump");
    }
}

/// The staging-loop switches (`PROXIMA_TILED_GEMM_WIDE_ACT_LOAD`, item 3c,
/// and `PROXIMA_TILED_GEMM_SLIM_TGMEM`, phase 2), now default ON: explicit
/// `"0"` must leave the emitted source and cache key byte-identical to the
/// phase-1 (pre-switch) baseline -- same posture as `switch_off_keeps_the_
/// source_and_cache_key_on_the_serial_path` (`dense_batched_tiled_gemm_
/// parity.rs`) and every other override in this file. Unset (the new
/// default) must render the same marker and cache key as explicit `"1"`, so
/// this test cannot pass by the new code paths silently never firing on the
/// default path either.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn staging_switches_default_on_render_unless_explicitly_disabled() {
    let (bound, weight_node) = real_shaped_tiled_gemm_op(16, 256, 4);
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    let off_source = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", Some("0")),
            ("PROXIMA_TILED_GEMM_SLIM_TGMEM", Some("0")),
        ],
        || emit(&bound, &q4k, NumericPolicy::default()).expect("emits").source,
    );
    let off_key = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", Some("0")),
            ("PROXIMA_TILED_GEMM_SLIM_TGMEM", Some("0")),
        ],
        || kernel_cache_key(&bound, &q4k, NumericPolicy::default()).expect("cache key derives"),
    );
    for marker in ["act_tile_interior", "tg_shared"] {
        assert!(
            !off_source.contains(marker),
            "explicitly disabled staging switches must never emit {marker:?}:\n{off_source}"
        );
    }

    // each case holds the OTHER staging switch explicitly at "0" while
    // varying the named one, so this isolates a single switch's own effect
    // instead of letting the other switch's now-default-on behavior leak in
    // (both switches default ON unset, so an unwrapped ambient env would
    // silently fold the other switch's marker into every arm here).
    let cases: &[(&str, &str, &str)] = &[
        (
            "PROXIMA_TILED_GEMM_WIDE_ACT_LOAD",
            "PROXIMA_TILED_GEMM_SLIM_TGMEM",
            "act_tile_interior",
        ),
        (
            "PROXIMA_TILED_GEMM_SLIM_TGMEM",
            "PROXIMA_TILED_GEMM_WIDE_ACT_LOAD",
            "tg_shared",
        ),
    ];
    for &(var, other_var, marker) in cases {
        let source_with = |value: Option<&str>| {
            temp_env::with_vars([(var, value), (other_var, Some("0"))], || {
                emit(&bound, &q4k, NumericPolicy::default()).expect("emits").source
            })
        };
        let key_with = |value: Option<&str>| {
            temp_env::with_vars([(var, value), (other_var, Some("0"))], || {
                kernel_cache_key(&bound, &q4k, NumericPolicy::default()).expect("cache key derives")
            })
        };

        let on_source = source_with(Some("1"));
        assert!(
            on_source.contains(marker),
            "{var}=1 must actually render its new code path (missing {marker:?}):\n{on_source}"
        );
        let on_key = key_with(Some("1"));
        assert_ne!(
            on_key, off_key,
            "{var}=1 must render under a DIFFERENT cache key than the off baseline, or the two \
             kernels would collide in the pipeline cache"
        );

        let unset_source = source_with(None);
        assert_eq!(
            unset_source, on_source,
            "{var} unset (new default) must be byte-identical to explicit \"1\""
        );
        let unset_key = key_with(None);
        assert_eq!(unset_key, on_key, "{var} unset (new default) must share explicit \"1\"'s cache key");

        let zero_source = source_with(Some("0"));
        assert_eq!(zero_source, off_source, "{var}=0 must be byte-identical to the all-off baseline");
    }
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

/// `PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE` default OFF: unset emits today's
/// one-thread-per-row staging loop (no `wws_`-prefixed locals at all);
/// explicit `"1"` emits the per-thread block-pointer setup and `half4`
/// vector stores this switch adds, and a `Q4_K` weight decodes through
/// `q4k_header_for`/`q4k_run8` unchanged -- never `Q4_0`'s own
/// `q4_0_run8_wide`.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn wide_weight_stage_emits_wide_decode_and_vector_stores_when_switch_on() {
    let bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;
    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);

    let off_source = emit(&bound, &q4k, NumericPolicy::default()).expect("emits").source;
    assert!(
        !off_source.contains("wws_blk0"),
        "the default (unset) path must not emit the wide-weight-stage schedule:\n{off_source}"
    );

    let on_source = temp_env::with_var("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("1"), || {
        emit(&bound, &q4k, NumericPolicy::default()).expect("emits").source
    });
    assert!(
        on_source.contains("wws_blk0") && on_source.contains("half4("),
        "PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE=1 must emit the per-thread block pointer and half4 \
         vector stores:\n{on_source}"
    );
    assert!(
        !on_source.contains("q4_0_run8_wide"),
        "a Q4_K weight must decode through q4k_run8, never CALL Q4_0's own q4_0_run8_wide -- \
         and since that function is only Q4_0-eligible, its OWN definition text must not even \
         be spliced into a Q4_K kernel's preamble either:\n{on_source}"
    );
}

/// `Q4_0`'s own arm of the same switch -- decodes through `q4_0_run8_wide`'s
/// `ushort` loads, not `q4_0_run8`'s per-byte `uchar` loads.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn wide_weight_stage_emits_ushort_wide_q4_0_decode_when_switch_on() {
    let bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let on_source = temp_env::with_var("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("1"), || {
        emit(&bound, &q4_0, NumericPolicy::default()).expect("emits").source
    });
    assert!(
        on_source.contains("q4_0_run8_wide") && on_source.contains("wws_blk0"),
        "a Q4_0 weight with the switch on must decode through q4_0_run8_wide's ushort loads:\n{on_source}"
    );
}

/// [`preamble`]'s own doc: `Q4_0_RUN8_WIDE_MSL` must never appear -- neither
/// its call site nor its own definition text -- while `PROXIMA_TILED_GEMM_
/// WIDE_WEIGHT_STAGE` is unset, the same "switch unset renders identically
/// to the switch explicitly disabled" contract [`staging_switches_default_
/// on_render_unless_explicitly_disabled`] proves for the crate's default-ON
/// switches, mirrored here for this default-OFF one. Checked for BOTH a
/// Q4_0 shape that actually admits the tiled-GEMM path AND a Q4_0 shape
/// below `TILED_GEMM_MIN_TOKENS` that takes the packed-row-blocked path
/// instead -- the second case is exactly what `packed_row_blocked_s1_byte_
/// identity.rs`'s own fixture pins, and is the shape the unconditional-
/// prelude splice bloated before this gate existed.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn wide_weight_stage_unset_is_byte_identical_to_switch_disabled_for_tiled_and_packed_row_q4_0() {
    let tiled_bound = tiled_gemm_op(16, 256, 4);
    let tiled_weight = tiled_bound.operands()[0].0;
    let mut tiled_q4_0 = BTreeMap::new();
    tiled_q4_0.insert(tiled_weight, Codec::Q4_0);
    temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("1"), || {
        let unset_source = temp_env::with_var("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", None::<&str>, || {
            emit(&tiled_bound, &tiled_q4_0, NumericPolicy::default())
                .expect("emits")
                .source
        });
        let disabled_source =
            temp_env::with_var("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("0"), || {
                emit(&tiled_bound, &tiled_q4_0, NumericPolicy::default())
                    .expect("emits")
                    .source
            });
        assert_eq!(
            unset_source, disabled_source,
            "a Q4_0 tiled-GEMM shape must render identically whether the switch is unset or \
             explicitly disabled"
        );
        assert!(
            !unset_source.contains("q4_0_run8_wide"),
            "the wide Q4_0 decoder must not appear while the switch is off:\n{unset_source}"
        );
    });

    let packed_row_bound = tiled_gemm_op(1, 256, 4);
    let packed_row_weight = packed_row_bound.operands()[0].0;
    let mut packed_row_q4_0 = BTreeMap::new();
    packed_row_q4_0.insert(packed_row_weight, Codec::Q4_0);
    let unset_source = temp_env::with_var("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", None::<&str>, || {
        emit(&packed_row_bound, &packed_row_q4_0, NumericPolicy::default())
            .expect("emits")
            .source
    });
    let disabled_source = temp_env::with_var("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("0"), || {
        emit(&packed_row_bound, &packed_row_q4_0, NumericPolicy::default())
            .expect("emits")
            .source
    });
    assert_eq!(
        unset_source, disabled_source,
        "a packed-row (non-tiled) Q4_0 shape must render identically whether the switch is \
         unset or explicitly disabled"
    );
    assert!(
        !unset_source.contains("q4_0_run8_wide"),
        "a packed-row (non-tiled) Q4_0 kernel must never carry the wide decoder's text at all:\n{unset_source}"
    );
}

/// The dense-batched-gemm path (`push_dense_batched_gemm_body`) has no
/// codec-decode arm at all -- both operands are plain `float` -- so this
/// switch, unlike `tiled_gemm_direct_store`, must never activate for it
/// even when both switches admit.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn wide_weight_stage_never_applies_to_the_dense_batched_gemm_path() {
    temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_DENSE", Some("1")),
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("1")),
        ],
        || {
            let bound = dense_batched_score_shaped_op(510, 512, 8, 128);
            let codecs = operand_codecs(&bound, &BTreeMap::new());
            let BoundOpKind::Reduce {
                reduce_op,
                init,
                ref output_axes,
                ..
            } = bound.kind
            else {
                panic!("dense_batched_score_shaped_op always builds a Keep::Reduce fold")
            };
            assert!(
                !wide_weight_stage_active(&bound, &codecs, reduce_op, init, output_axes),
                "the dense-batched-gemm path has no codec-decode arm; the wide-weight-stage \
                 switch must never activate for it"
            );
        },
    );
}

/// The regression this switch's own doc warns about (`grid_threads` over-
/// dispatch, ROW 113's precedent): the kernel's OWN attribute form and the
/// dispatched grid shape must never disagree. `source.contains(..
/// threadgroup_position_in_grid..)` (the body took the 2D-attribute form) iff
/// `grid.grid2d.is_some()` (the driver dispatches `dispatchThreadgroups`) --
/// checked for BOTH the tiled (packed) body and the dense-batched body, and
/// for BOTH switch states, so a future change to either side alone (the
/// renderer's own signature swap, or `grid2d_for`'s own admission) trips this
/// test the moment the two stop agreeing.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn grid2d_kernel_attribute_form_and_dispatched_grid_always_agree() {
    let tiled_bound = tiled_gemm_op(510, 1536, 128);
    let tiled_weight = tiled_bound.operands()[0].0;
    let mut tiled_q4_0 = BTreeMap::new();
    tiled_q4_0.insert(tiled_weight, Codec::Q4_0);

    let dense_bound = dense_batched_score_shaped_op(510, 512, 8, 128);

    for grid2d_env in [None, Some("1")] {
        temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", grid2d_env, || {
            let tiled_kernel =
                emit(&tiled_bound, &tiled_q4_0, NumericPolicy::default()).expect("tiled emits");
            let (_, tiled_grid) =
                kernel_dispatch_shape(&tiled_bound, &tiled_q4_0, NumericPolicy::default())
                    .expect("tiled dispatch shape");
            let tiled_2d_attrs = tiled_kernel.source.contains("[[threadgroup_position_in_grid]]");
            assert_eq!(
                tiled_2d_attrs,
                tiled_grid.grid2d.is_some(),
                "tiled body: source carries 2D attributes ({tiled_2d_attrs}) must match \
                 grid.grid2d.is_some() ({:?}) at PROXIMA_TILED_GEMM_GRID2D={grid2d_env:?}",
                tiled_grid.grid2d,
            );
            if let Some(grid2d) = tiled_grid.grid2d {
                assert_eq!(
                    grid2d.form,
                    Grid2DForm::TileCoordinates,
                    "the tiled-GEMM lever's launch reads tile coordinates, never the flat thread index"
                );
                assert_eq!(
                    grid2d.threadgroups_x * grid2d.threadgroups_y
                        * grid2d.threads_per_threadgroup_x
                        * grid2d.threads_per_threadgroup_y,
                    tiled_grid.threads,
                    "the 2D threadgroup-count grid and the flattened thread count must describe \
                     the identical launch"
                );
            }

            let dense_kernel =
                emit(&dense_bound, &BTreeMap::new(), NumericPolicy::default()).expect("dense emits");
            let (_, dense_grid) =
                kernel_dispatch_shape(&dense_bound, &BTreeMap::new(), NumericPolicy::default())
                    .expect("dense dispatch shape");
            let dense_2d_attrs = dense_kernel.source.contains("[[threadgroup_position_in_grid]]");
            assert_eq!(
                dense_2d_attrs,
                dense_grid.grid2d.is_some(),
                "dense body: source carries 2D attributes ({dense_2d_attrs}) must match \
                 grid.grid2d.is_some() ({:?}) at PROXIMA_TILED_GEMM_GRID2D={grid2d_env:?}",
                dense_grid.grid2d,
            );
        });
    }
}

/// `PROXIMA_TILED_GEMM_Q4_0` default ON: a `Q4_0` weight, otherwise
/// tiled-GEMM-eligible (same shape [`many_token_matmul_takes_the_tiled_gemm_path`]
/// admits for `Q4_K`), falls back to the row-blocked path only when the
/// switch is EXPLICITLY disabled (`"0"`) -- the emitted MSL for that
/// fallback is asserted here, matching `tiled_gemm_q4_0_override`'s own doc.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn q4_0_never_takes_the_tiled_gemm_path_with_the_switch_explicitly_off() {
    let bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("0"), || {
        assert!(
            tiled_gemm_block(
                &bound,
                &operand_codecs(&bound, &q4_0),
                ScalarOp::Add,
                ReduceInit::Zero,
                &[1, 0]
            )
            .is_none(),
            "a Q4_0 weight must never take the tiled GEMM path with the switch explicitly off"
        );

        let off_source = emit(&bound, &q4_0, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            !off_source.contains("simdgroup_multiply_accumulate"),
            "a Q4_0 weight must not emit the tiled GEMM kernel with the switch explicitly off:\n{off_source}"
        );
    });
}

/// `PROXIMA_TILED_GEMM_Q4_0` unset (the new default): the same `Q4_0` weight
/// must take the tiled path exactly as explicit `"1"` does, byte-identically
/// -- unset is no longer a no-op for this switch.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn q4_0_takes_the_tiled_gemm_path_when_unset() {
    let bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let unset_source = temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", None::<&str>, || {
        assert!(
            tiled_gemm_block(
                &bound,
                &operand_codecs(&bound, &q4_0),
                ScalarOp::Add,
                ReduceInit::Zero,
                &[1, 0]
            )
            .is_some(),
            "a Q4_0 weight must take the tiled GEMM path with the switch unset (new default)"
        );
        emit(&bound, &q4_0, NumericPolicy::default())
            .expect("emits")
            .source
    });
    assert!(
        unset_source.contains("simdgroup_multiply_accumulate"),
        "a Q4_0 weight must emit the tiled GEMM kernel with the switch unset:\n{unset_source}"
    );
    let on_source = temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("1"), || {
        emit(&bound, &q4_0, NumericPolicy::default())
            .expect("emits")
            .source
    });
    assert_eq!(
        unset_source, on_source,
        "unset (new default) and explicit \"1\" must emit byte-identical MSL for a Q4_0 weight"
    );
}

/// `PROXIMA_TILED_GEMM_Q4_0=1`: the same `Q4_0` weight now takes the
/// `simdgroup_matrix`-tiled path, decoding through the per-element
/// `q4_0_element` accessor (already unconditional in the prelude via
/// [`crate::identity`]'s own `Q4_0_UNPACK_MSL`) rather than `Q4_K`'s
/// batched `q4k_header_for`/`q4k_run8` pair -- scheduling, packing and
/// tiling stay the SAME emitted lines either codec takes; only this decode
/// differs, matching the owner's standing rule that only the decoder may be
/// codec-specific.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn q4_0_takes_the_tiled_gemm_path_with_the_switch_on() {
    let bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("1"), || {
        assert!(
            tiled_gemm_block(
                &bound,
                &operand_codecs(&bound, &q4_0),
                ScalarOp::Add,
                ReduceInit::Zero,
                &[1, 0]
            )
            .is_some(),
            "a Q4_0 weight must take the tiled GEMM path with the switch on"
        );
        let source = emit(&bound, &q4_0, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            source.contains("simdgroup_multiply_accumulate"),
            "a Q4_0 weight with the switch on must take the tiled GEMM path:\n{source}"
        );
        assert!(
            source.contains("q4_0_block_scale(blk)") && source.contains("q4_0_run8(blk"),
            "the Q4_0 tiled-GEMM arm must decode through the batched q4_0_run8 arm:\n{source}"
        );
        // the prelude declares `q4k_header_for`/`q4k_run8` unconditionally
        // (every decode helper is always emitted, see `signature_tokens_
        // prelude.rs`'s own comment), so this checks the BODY invocation
        // pattern, not mere textual presence of the declaration.
        assert!(
            !source.contains("q4k_header_for(blk") && !source.contains("q4k_run8(blk"),
            "a Q4_0 weight must never call the Q4_K decode helpers:\n{source}"
        );
    });
}

/// [`TiledGemmRejection::BroadcastEpilogueNotSupported`]'s own admission
/// gate: a broadcast-reduce epilogue can never structurally reach a real
/// packed matmul (its own fold body is never a bare `weight * activation`
/// product, `is_plain_product_reduce` and `render_reduce`'s own broadcast
/// gate both exclude it), so this hand-mutates the ONE field the gate reads
/// on an otherwise-real tiled-eligible `Q4_0` op -- driving the admission
/// contract directly, the same "hand-built shape a real classifier never
/// produces" posture [`push_tiled_gemm_body_rejects_an_empty_token_axis_group`]
/// already takes for [`TiledGemmBlock`] itself. The fix under test: admission
/// declines BEFORE any renderer runs (`classify_tiled_gemm`, `kernel_cache_key`
/// both return `Ok`/the expected `Err` with no `_tgq0` token folded in),
/// never an [`EmitError`] surfacing at emit time the way it did before this
/// gate existed.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn q4_0_broadcast_epilogue_declines_tiled_gemm_admission_without_erroring() {
    let mut bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;
    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);

    let BoundOpKind::Reduce {
        output_axes,
        epilogue_broadcast_axes,
        ..
    } = &mut bound.kind
    else {
        panic!("tiled_gemm_op always builds a Keep::Reduce fold")
    };
    *epilogue_broadcast_axes = output_axes.clone();

    let codecs = operand_codecs(&bound, &q4_0);
    temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("1"), || {
        let admission = classify_tiled_gemm(&bound, &codecs, ScalarOp::Add, ReduceInit::Zero, &[1, 0]);
        assert!(
            matches!(
                admission,
                Err(TiledGemmRejection::BroadcastEpilogueNotSupported)
            ),
            "a broadcast-reduce epilogue must decline tiled-GEMM admission naming that gate, \
             not error later at render time"
        );
        let key = kernel_cache_key(&bound, &q4_0, NumericPolicy::default())
            .expect("an admission decline must fall back, never error, at cache-key time");
        assert!(
            !key.contains("_tgq0"),
            "a declined broadcast epilogue must never fold the tiled-Q4_0 token into the \
             cache key: {key}"
        );
    });
}

/// [`TiledGemmRejection::BlockKNotChunkAligned`]'s own doc: the predicate
/// this test exercises directly ([`tiled_gemm_block_k_chunk_aligned`]) is
/// the exact arithmetic `push_tiled_gemm_body`'s weight-staging chunk loop
/// runs (`num_chunks = block_k.div_ceil(chunk_width)`, last chunk's write
/// extent `num_chunks * chunk_width`) -- `block_k=48, chunk_width=32` is
/// the overrun this names concretely: `num_chunks=2`, extent `64` overruns
/// the row's own 48-wide bound by 16 `half` slots into the next row's own
/// staged data (or past `weight_tile`'s end, on the tile's last row). No
/// build-time `tiled_gemm.block_k` reaches this combination today
/// (`omega/build.rs`'s `require_divides_q4k_block` + `require_multiple_
/// of_eight` jointly rule it out -- every admitted value is a power-of-two
/// divisor of 256 that is also a multiple of 8, and every such value is
/// either <= 32 or a whole multiple of 32), so this predicate is the
/// explicit backstop against a future relaxation of either rule, not a
/// path reachable through today's legal build.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn tiled_gemm_block_k_chunk_aligned_rejects_a_ragged_combination() {
    assert!(
        !tiled_gemm_block_k_chunk_aligned(48, 32),
        "block_k=48 is not <= chunk_width=32 and not a multiple of it -- \
         push_tiled_gemm_body's chunk loop would write chunks at offsets \
         0 and 32, the second one extending to 64, 16 half-slots past the \
         row's own 48-wide bound"
    );
    assert!(
        tiled_gemm_block_k_chunk_aligned(32, 32),
        "block_k == chunk_width is exactly one covering chunk"
    );
    assert!(
        tiled_gemm_block_k_chunk_aligned(16, 32),
        "block_k < chunk_width clamps to a single block_k-wide chunk"
    );
    assert!(
        tiled_gemm_block_k_chunk_aligned(64, 32),
        "block_k a whole multiple of chunk_width covers with no remainder"
    );
}

/// The classification-time counterpart to the predicate test above: proves
/// TODAY's actual build-time `crate::sized::TILED_GEMM_BLOCK_K` still
/// clears [`TiledGemmRejection::BlockKNotChunkAligned`] for both codecs
/// this path admits, so the new gate never regresses the real, currently
/// working shapes.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn classify_tiled_gemm_admits_todays_real_sized_block_k_for_both_codecs() {
    let bound = tiled_gemm_op(16, 256, 4);
    let weight_node = bound.operands()[0].0;

    let mut q4k = BTreeMap::new();
    q4k.insert(weight_node, Codec::Q4K);
    let codecs = operand_codecs(&bound, &q4k);
    let admission =
        classify_tiled_gemm(&bound, &codecs, ScalarOp::Add, ReduceInit::Zero, &[1, 0]);
    assert!(
        admission.is_ok(),
        "today's build-time TILED_GEMM_BLOCK_K must stay chunk-aligned for Q4_K"
    );

    let mut q4_0 = BTreeMap::new();
    q4_0.insert(weight_node, Codec::Q4_0);
    let codecs = operand_codecs(&bound, &q4_0);
    temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("1"), || {
        let admission =
            classify_tiled_gemm(&bound, &codecs, ScalarOp::Add, ReduceInit::Zero, &[1, 0]);
        assert!(
            admission.is_ok(),
            "today's build-time TILED_GEMM_BLOCK_K must stay chunk-aligned for Q4_0"
        );
    });
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn multi_head_shaped_matmul_stays_on_the_row_blocked_path_regardless_of_token_count() {
    // 32 sequence positions clears TILED_GEMM_MIN_TOKENS handily, but
    // this op keeps TWO weight-owned output axes (`heads`, `head_dim`)
    // declared in the REVERSE of the weight's own outer-to-inner nesting
    // (`multi_head_matmul_op`'s own doc) -- `axes_fold_contiguously`
    // declines that group, so this is `classify_tiled_gemm`'s documented
    // scope limit, not a silent gap. A NATURAL declared order (heads
    // before head_dim, matching the weight's own nesting) folds
    // contiguously and correctly clears tiled-GEMM admission instead --
    // this fixture exists to prove the limit still exists for the
    // shapes it should.
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
    let error = render_reduce(
        &bound,
        "entry",
        &[None],
        NumericPolicy::default(),
        false,
        &MetalOnlyExtras::default(),
    )
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

    let error = render_reduce(
        &bound,
        "entry",
        &[None],
        NumericPolicy::default(),
        false,
        &MetalOnlyExtras::default(),
    )
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
        &MetalOnlyExtras::default(),
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
        &MetalOnlyExtras::default(),
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
        &MetalOnlyExtras::default(),
    )
    .expect("Q4_0 now reaches the row-blocked path and renders a kernel body");
    // See the sibling assertion in `q4_0_codec_takes_the_row_blocked_path_
    // at_a_256_extent` for why this branches on `metal-q4_0-native`.
    if cfg!(feature = "metal-q4_0-native") {
        assert!(
            source.contains("sumy * -8.0f"),
            "metal-q4_0-native must render ggml's inline nibble dot: {source}"
        );
    } else {
        assert!(
            source.contains("q4_0_pair_dot(blk"),
            "row-blocked Q4_0 body for a plain-product reduce must call the batched pair-dot accessor: {source}"
        );
    }
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

/// Candidate B's fused softmax op (`BoundOpKind::CachedSoftmaxWeights`'s own
/// doc): three operands collapsed to exactly the axes `run_cached_softmax_
/// weights` reads -- `cached_scores` at `[key, row]` (KEY-outer/ROW-inner,
/// `stride(key) == attention_rows`, `stride(row) == 1`, the real R9-dump
/// shape that CPU evaluator's own doc records, bug 1's fix), `new_scores`
/// at `[row]`, `new_value` at `[row, dim]` (ROW-outer/DIM-inner). `node`'s
/// own output extents follow the same dense `[cached_key_rows,
/// attention_rows]` convention that evaluator's doc names for 154.
fn cached_softmax_weights_op(attention_rows: u64, cached_key_rows: u64, head_dim: u64) -> BoundOp {
    let operands = vec![
        (
            NodeId(0),
            Layout {
                base: 0,
                strides: vec![attention_rows as i64, 1].into(),
            },
            None,
        ),
        (
            NodeId(1),
            Layout {
                base: 0,
                strides: vec![1].into(),
            },
            None,
        ),
        (
            NodeId(2),
            Layout {
                base: 0,
                strides: vec![head_dim as i64, 1].into(),
            },
            None,
        ),
    ];
    BoundOp {
        node: NodeId(3),
        dtype: DType::Float32,
        extents: vec![cached_key_rows, attention_rows],
        kind: BoundOpKind::CachedSoftmaxWeights {
            operands,
            cached_weight_sum: NodeId(4),
            new_weight_sum: NodeId(5),
            new_attended: NodeId(6),
            cached_key_rows,
            new_key_rows: 1,
            query_rows: attention_rows,
            attention_rows,
            head_dim,
        },
    }
}

/// `render_cached_softmax_weights` is deterministic over its `BoundOp`
/// (`emit_is_deterministic_byte_equal`'s own doc establishes the same
/// property for the general `emit` entry point) at both the narrow
/// (`cached_key_rows == 32`, one physical simdgroup, `wide_cooperative_
/// reduce_width` stays at `SIMD_WIDTH`) and wide (`cached_key_rows == 512`,
/// `wide_cooperative_reduce_width` widens to 128, the cross-simdgroup
/// `threadgroup float partials{N}[...]` combine path) shapes -- two
/// independently constructed `BoundOp`s of the identical shape must render
/// byte-identical text, and the two shapes must render DIFFERENT text (the
/// wide shape's cooperative combine is structurally distinct from the
/// narrow shape's direct `simd_max`/`simd_sum`, `push_cooperative_fold`'s
/// own doc). Both texts are also saved to a temp dir as a readable artifact
/// of what got gated here.
#[test]
fn cached_softmax_weights_render_is_deterministic_and_width_dependent() {
    let narrow_a = cached_softmax_weights_op(8, 32, 256);
    let narrow_b = cached_softmax_weights_op(8, 32, 256);
    let wide_a = cached_softmax_weights_op(8, 512, 256);
    let wide_b = cached_softmax_weights_op(8, 512, 256);

    let narrow_text_a = render_cached_softmax_weights(&narrow_a, "omega_cached_softmax_weights_c32_a8_d256")
        .expect("narrow shape renders");
    let narrow_text_b = render_cached_softmax_weights(&narrow_b, "omega_cached_softmax_weights_c32_a8_d256")
        .expect("narrow shape renders (second construction)");
    let wide_text_a = render_cached_softmax_weights(&wide_a, "omega_cached_softmax_weights_c512_a8_d256")
        .expect("wide shape renders");
    let wide_text_b = render_cached_softmax_weights(&wide_b, "omega_cached_softmax_weights_c512_a8_d256")
        .expect("wide shape renders (second construction)");

    assert_eq!(narrow_text_a, narrow_text_b, "narrow shape must render byte-identical text across constructions");
    assert_eq!(wide_text_a, wide_text_b, "wide shape must render byte-identical text across constructions");
    assert_ne!(narrow_text_a, wide_text_a, "narrow and wide shapes take structurally different cooperative-fold paths");

    // narrow: one physical simdgroup, direct `simd_max`/`simd_sum`, no
    // threadgroup partials array.
    assert!(narrow_text_a.contains("simd_max(accumulator0)"));
    assert!(!narrow_text_a.contains("threadgroup float partials0"));
    // wide: `wide_cooperative_reduce_width(512) == 128` -> 4 simdgroups.
    assert!(wide_text_a.contains("threadgroup float partials0[4]"));
    assert!(wide_text_a.contains("threadgroup float partials1[4]"));

    let integration_dir = std::env::temp_dir().join("proxima-omega-softmax-render-parity");
    if std::fs::create_dir_all(&integration_dir).is_ok() {
        let _ = std::fs::write(integration_dir.join("softmax_c32.metal"), &narrow_text_a);
        let _ = std::fs::write(integration_dir.join("softmax_c512.metal"), &wide_text_a);
    }
}

/// `PROXIMA_SOFTMAX_RUNTIME_ROWS` unset (default) must reproduce the exact
/// kernel body emitted before this switch existed: the `struct Uniforms`
/// stays the one-word leaf shape, the entry opens with `(void)u;` (never a
/// `cached_key_rows` read), and every one of the three key-loop bounds
/// stays the compiled literal (`32`, not a variable read). This is the
/// literal tail `render_cached_softmax_weights` emits for
/// `cached_softmax_weights_op(8, 32, 256)`, captured against the pre-switch
/// renderer -- any future edit that moves so much as one byte of this text
/// with the switch off fails here.
#[test]
fn softmax_runtime_rows_default_off_is_byte_identical_to_the_literal_bound_emit() {
    let narrow = cached_softmax_weights_op(8, 32, 256);
    let text = temp_env::with_var("PROXIMA_SOFTMAX_RUNTIME_ROWS", None::<&str>, || {
        render_cached_softmax_weights(&narrow, "omega_cached_softmax_weights_c32_a8_d256")
            .expect("narrow shape renders with the switch off")
    });
    let expected_tail = "struct Uniforms { long total_elements; };\n\n\
kernel void omega_cached_softmax_weights_c32_a8_d256(\n\
\tdevice const float* cached_scores [[buffer(0)]],\n\
\tdevice const float* new_scores [[buffer(1)]],\n\
\tdevice const float* new_value [[buffer(2)]],\n\
\tdevice float* out [[buffer(3)]],\n\
\tconstant Uniforms& u [[buffer(4)]],\n\
\tdevice float* cached_weight_sum [[buffer(5)]],\n\
\tdevice float* new_weight_sum [[buffer(6)]],\n\
\tdevice float* new_attended [[buffer(7)]],\n\
\tuint local [[thread_position_in_threadgroup]],\n\
\tuint tg [[threadgroup_position_in_grid]])\n\
{\n\
\t(void)u;\n\
\tlong row = (long)tg;\n\n\
\tfloat accumulator0 = -INFINITY;\n\
\tbool seeded0 = false;\n\
\tfor (long key = (long)local; key < 32; key += 32) {\n\
\t\tlong offset = 0 + key * 8 + row * 1;\n\
\t\tfloat value = cached_scores[offset];\n\
\t\taccumulator0 = seeded0 ? max(accumulator0, value) : value;\n\
\t\tseeded0 = true;\n\
\t}\n\
\tfloat reduced0 = simd_max(accumulator0);\n\
\tthreadgroup float group_max_shared;\n\
\tif (local == 0u) { group_max_shared = reduced0; }\n\
\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
\tfloat group_max = group_max_shared;\n\n\
\tlong new_offset = 0 + row * 1;\n\
\tgroup_max = max(group_max, new_scores[new_offset]);\n\n\
\tfor (long key = (long)local; key < 32; key += 32) {\n\
\t\tlong offset = 0 + key * 8 + row * 1;\n\
\t\tfloat step0 = (cached_scores[offset] - group_max);\n\
\t\tout[key * 8 + row] = exp(step0);\n\
\t}\n\
\tthreadgroup_barrier(mem_flags::mem_device);\n\n\
\tfloat new_shifted = exp(new_scores[new_offset] - group_max);\n\n\
\tfloat accumulator1 = 0.0f;\n\
\tbool seeded1 = false;\n\
\tfor (long key = (long)local; key < 32; key += 32) {\n\
\t\tfloat value = out[key * 8 + row];\n\
\t\taccumulator1 = seeded1 ? (accumulator1 + value) : value;\n\
\t\tseeded1 = true;\n\
\t}\n\
\tfloat reduced1 = simd_sum(accumulator1);\n\
\tif (local == 0u) {\n\
\t\tcached_weight_sum[row] = reduced1;\n\
\t\tnew_weight_sum[row] = new_shifted;\n\
\t}\n\n\
\tfor (long dim = (long)local; dim < 256; dim += 32) {\n\
\t\tlong voffset = 0 + row * 256 + dim * 1;\n\
\t\tnew_attended[row * 256 + dim] = new_shifted * new_value[voffset];\n\
\t}\n\
}\n";
    assert!(
        text.ends_with(expected_tail),
        "switch-off emit must stay byte-identical to the literal-bound kernel:\n{text}"
    );
    let entry = entry_name(&narrow);
    assert_eq!(
        entry, "omega_cached_softmax_weights_c32_a8_d256",
        "switch-off entry name must keep the c{{n}} token"
    );
    let key = kernel_cache_key(&narrow, &BTreeMap::new(), NumericPolicy::default())
        .expect("cache key computes with the switch off");
    assert!(
        key.contains("_c32_"),
        "switch-off cache key must keep the c{{n}} token: {key}"
    );
    assert!(
        !key.contains("rtrows"),
        "switch-off cache key must never carry the runtime-rows marker: {key}"
    );
}

/// `PROXIMA_SOFTMAX_RUNTIME_ROWS=1` must change exactly three things versus
/// the switch-off emit above: the `Uniforms` struct gains a `cached_key_rows`
/// field, the entry body reads it into a local instead of `(void)u;`, and
/// every key-loop bound becomes that local's name instead of the literal
/// `32` -- `width`, `attention_rows`, and `head_dim` stay compiled literals
/// either way. The entry name and cache key both drop the `c{n}` token and
/// carry `_rtrows` instead, so the two pipelines can never collide.
#[test]
fn softmax_runtime_rows_on_changes_only_the_bound_and_the_name() {
    let narrow = cached_softmax_weights_op(8, 32, 256);

    let off_text = temp_env::with_var("PROXIMA_SOFTMAX_RUNTIME_ROWS", None::<&str>, || {
        render_cached_softmax_weights(&narrow, "omega_cached_softmax_weights_c32_a8_d256")
            .expect("switch-off renders")
    });
    let (on_text, on_entry, on_key) = temp_env::with_var(
        "PROXIMA_SOFTMAX_RUNTIME_ROWS",
        Some("1"),
        || {
            let entry = entry_name(&narrow);
            let text = render_cached_softmax_weights(&narrow, &entry)
                .expect("switch-on renders");
            let key = kernel_cache_key(&narrow, &BTreeMap::new(), NumericPolicy::default())
                .expect("switch-on cache key computes");
            (text, entry, key)
        },
    );

    assert_eq!(
        on_entry, "omega_cached_softmax_weights_rtrows_a8_d256",
        "switch-on entry name must drop c{{n}} and carry _rtrows"
    );
    assert!(
        on_key.contains("rtrows"),
        "switch-on cache key must carry the runtime-rows marker: {on_key}"
    );
    assert!(
        !on_key.contains("_c32_"),
        "switch-on cache key must drop the c{{n}} token: {on_key}"
    );

    assert!(
        on_text.contains("struct Uniforms { long total_elements; long cached_key_rows; };\n\n"),
        "switch-on must widen the Uniforms struct by one word:\n{on_text}"
    );
    assert!(
        on_text.contains("\tlong cached_key_rows = u.cached_key_rows;\n"),
        "switch-on must read cached_key_rows off the uniform buffer:\n{on_text}"
    );
    assert!(
        !on_text.contains("(void)u;"),
        "switch-on no longer leaves u unused:\n{on_text}"
    );
    let bound_count = on_text.matches("key < cached_key_rows; key += 32").count();
    assert_eq!(
        bound_count, 3,
        "all three key-loops must read the runtime bound, same stride/width as before:\n{on_text}"
    );
    assert!(
        !on_text.contains("key < 32;"),
        "switch-on must never leave a literal 32 loop bound behind:\n{on_text}"
    );
    // Only the entry name, the bound, and the uniform declaration differ --
    // every other line (the accumulation expressions, the cooperative fold,
    // the AV-fold tail) stays byte-for-byte the same text as the switch-off
    // emit, entry names normalized to a shared placeholder first.
    let off_normalized = off_text
        .replace(
            "omega_cached_softmax_weights_c32_a8_d256",
            "omega_cached_softmax_weights_TEST_PLACEHOLDER",
        )
        .replace("\t(void)u;\n", "")
        .replace(
            "for (long key = (long)local; key < 32; key += 32)",
            "for (long key = (long)local; key < cached_key_rows; key += 32)",
        );
    let on_normalized = on_text
        .replace(
            "omega_cached_softmax_weights_rtrows_a8_d256",
            "omega_cached_softmax_weights_TEST_PLACEHOLDER",
        )
        .replace(
            "struct Uniforms { long total_elements; long cached_key_rows; };\n\n",
            "struct Uniforms { long total_elements; };\n\n",
        )
        .replace("\tlong cached_key_rows = u.cached_key_rows;\n", "");
    assert_eq!(
        off_normalized, on_normalized,
        "switch-on must change only the entry name, the bound expression, and the uniform declaration"
    );
}


/// A grid wider than the 32 bits of `uint gid [[thread_position_in_grid]]`
/// (Metal launches `threads mod 2^32` of it, no error) must be described by
/// ONE decision -- `grid2d_for` -- that the kernel text, the launch shape and
/// the pipeline cache key all follow. Each case below is a real op whose grid
/// is past `u32::MAX`, built without allocating a byte of tensor data.
mod flat_grid_form {
    use alloc::collections::BTreeMap;

    use proxima_tensor::NumericPolicy;

    use super::*;

    const FLAT_MARKER: &str = "ulong wide_group_index";
    const LINEAR_SIGNATURE: &str = "uint gid [[thread_position_in_grid]]";

    struct FlatCase {
        label: &'static str,
        bound: BoundOp,
        narrow: BoundOp,
        packed: BTreeMap<NodeId, Codec>,
        policy: NumericPolicy,
    }

    fn elementwise_tanh_op_2d(rows: u32, columns: u32) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(rows), Extent::Static(columns)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
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

    fn cumsum_rank3_op(outer: u32, middle: u32, inner: u32) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![
                    Extent::Static(outer),
                    Extent::Static(middle),
                    Extent::Static(inner),
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
                operand: source,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
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

    /// gemma4-E2B's `per_layer_model_proj`: an F16 `[features, k]` weight
    /// against `[rows, k]` f32 activations. F16 keeps it off the tiled-GEMM
    /// path (dense x dense and Q4_0/Q4_K take that), so it is a cooperative
    /// reduce at `wide_cooperative_reduce_width(k)` lanes.
    fn per_layer_projection_op(
        rows: u32,
        reduction: u32,
        features: u32,
    ) -> (BoundOp, BTreeMap<NodeId, Codec>) {
        let mut program = Vec::new();
        let weights = append(
            &mut program,
            Op::Input {
                dtype: DType::Float16,
                shape: vec![Extent::Static(features), Extent::Static(reduction)],
                name: None,
            },
        );
        let activations = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(rows), Extent::Static(reduction)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weights, IndexMap::Affine(map::projection(3, &[1, 2]))),
                    (activations, IndexMap::Affine(map::projection(3, &[0, 2]))),
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
                name: Some("per_layer_model_proj".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("projection infers");
        let bound = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
            .expect("projection lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted");
        let weight_node = bound.operands()[0].0;
        (bound, BTreeMap::from([(weight_node, Codec::Float16)]))
    }

    fn unpacked(
        label: &'static str,
        bound: BoundOp,
        narrow: BoundOp,
        policy: NumericPolicy,
    ) -> FlatCase {
        FlatCase {
            label,
            bound,
            narrow,
            packed: BTreeMap::new(),
            policy,
        }
    }

    fn gated_delta_net_op(num_v_heads: u64) -> BoundOp {
        let operands = (0..6)
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
            node: NodeId(6),
            dtype: DType::Float32,
            extents: vec![1, num_v_heads, 128],
            kind: BoundOpKind::GatedDeltaNet {
                operands,
                n_tokens: 1,
                kv_heads: 1,
                num_v_heads,
                head_k_dim: 4,
                head_v_dim: 128,
                query_key_head_stride: 4,
                query_key_dim_stride: 1,
                inv_sqrt_key_dim: 0.5,
                state_out: NodeId(7),
            },
        }
    }

    fn cached_attention_with_query_vectors(query_vectors: u64) -> BoundOp {
        let mut bound = cached_attention_op_dynamic(0, 8);
        bound.extents = vec![query_vectors, 1, 1, 4];
        bound
    }

    fn position_only_op(kind: BoundOpKind, extent: u64) -> BoundOp {
        BoundOp {
            node: NodeId(0),
            dtype: DType::Float32,
            extents: vec![extent, extent],
            kind,
        }
    }

    /// Every kernel form that can take the flat path, each as a `(wide,
    /// narrow)` pair: the same op structure at a shape past `u32::MAX` threads
    /// and at one that fits. The wide shapes only ever reach `emit` and
    /// `kernel_dispatch_shape`, so no tensor data exists for any of them.
    fn flat_cases() -> Vec<FlatCase> {
        let packed_row = packed_row_multi_token_op(1, 256, 4_000_000_000);
        let packed_narrow = packed_row_multi_token_op(1, 256, 64);
        let packed_weight = packed_row.operands()[0].0;
        let (projection, projection_codecs) = per_layer_projection_op(1873, 1536, 8960);
        let (projection_narrow, _) = per_layer_projection_op(2, 1536, 64);
        vec![
            FlatCase {
                label: "cooperative reduce: per_layer_model_proj at 1873 rows, 256 lanes",
                bound: projection,
                narrow: projection_narrow,
                packed: projection_codecs,
                policy: NumericPolicy::llama_relaxed(),
            },
            unpacked(
                "serial reduce: one thread per output past u32::MAX",
                matmul_op_with_reduce(70_000, 8, 70_000, ScalarOp::Maximum),
                matmul_op_with_reduce(64, 8, 64, ScalarOp::Maximum),
                NumericPolicy::default(),
            ),
            unpacked(
                "elementwise: 70000 x 70000 elements",
                elementwise_tanh_op_2d(70_000, 70_000),
                elementwise_tanh_op_2d(8, 8),
                NumericPolicy::default(),
            ),
            unpacked(
                "scan: one thread per outer line past u32::MAX",
                cumsum_rank3_op(70_000, 70_000, 4),
                cumsum_rank3_op(8, 8, 4),
                NumericPolicy::default(),
            ),
            unpacked(
                "iota",
                position_only_op(BoundOpKind::Iota, 70_000),
                position_only_op(BoundOpKind::Iota, 8),
                NumericPolicy::default(),
            ),
            unpacked(
                "constant",
                position_only_op(BoundOpKind::Constant { value: 1.5 }, 70_000),
                position_only_op(BoundOpKind::Constant { value: 1.5 }, 8),
                NumericPolicy::default(),
            ),
            FlatCase {
                label: "packed row-blocked Q4_K matvec with 4e9 output rows",
                bound: packed_row,
                narrow: packed_narrow,
                packed: BTreeMap::from([(packed_weight, Codec::Q4K)]),
                policy: NumericPolicy::default(),
            },
            unpacked(
                "cached attention: split kernel over 3e8 query vectors",
                cached_attention_with_query_vectors(300_000_000),
                cached_attention_with_query_vectors(1),
                NumericPolicy::default(),
            ),
            unpacked(
                "cached softmax weights: 2e8 attention rows",
                cached_softmax_weights_op(200_000_000, 1024, 64),
                cached_softmax_weights_op(8, 1024, 64),
                NumericPolicy::default(),
            ),
            unpacked(
                "gated delta net: 2^25 heads x 128 rows",
                gated_delta_net_op(1 << 25),
                gated_delta_net_op(4),
                NumericPolicy::default(),
            ),
        ]
    }

    #[test]
    fn every_kernel_past_the_thread_index_renders_and_dispatches_the_flat_form_together() {
        let limit = u64::from(u32::MAX);
        let cases = flat_cases();
        assert_eq!(cases.len(), 10, "one case per kernel form that can take the flat path");

        for case in &cases {
            let kernel = emit(&case.bound, &case.packed, case.policy)
                .unwrap_or_else(|error| panic!("{}: emit failed: {error}", case.label));
            let (_, shape) = kernel_dispatch_shape(&case.bound, &case.packed, case.policy)
                .unwrap_or_else(|error| panic!("{}: dispatch shape failed: {error}", case.label));

            assert!(
                kernel.grid.threads > limit,
                "{}: the case must be past u32::MAX threads, got {}",
                case.label,
                kernel.grid.threads
            );
            assert_eq!(kernel.grid, shape, "{}: emit and the cache-hit dispatch shape disagree", case.label);
            let spec = shape
                .grid2d
                .unwrap_or_else(|| panic!("{}: a {}-thread grid dispatched 1D", case.label, shape.threads));
            assert_eq!(spec.form, Grid2DForm::FlatThreadgroupIndex, "{}", case.label);

            assert!(
                kernel.source.contains(FLAT_MARKER),
                "{}: source must take threadgroup coordinates:\n{}",
                case.label,
                kernel.source
            );
            for scalar_attribute in [
                "uint tg [[",
                "uint tgid [[",
                "uint tptg [[",
                "uint local [[",
                "uint3 dense_batch_gid [[",
            ] {
                assert!(
                    !kernel.source.contains(scalar_attribute),
                    "{}: Metal rejects a scalar grid attribute beside the vector ones, but the source \
                     still declares `{scalar_attribute}`:\n{}",
                    case.label,
                    kernel.source
                );
            }
            assert!(
                !kernel.source.contains(LINEAR_SIGNATURE),
                "{}: source still indexes a 32-bit thread position:\n{}",
                case.label,
                kernel.source
            );
            assert!(
                kernel.source.contains("ulong wide_group_index = "),
                "{}: the flat threadgroup index must be rebuilt in 64 bits",
                case.label
            );

            let narrow = emit(&case.narrow, &case.packed, case.policy)
                .unwrap_or_else(|error| panic!("{}: narrow sibling emit failed: {error}", case.label));
            assert_eq!(narrow.grid.grid2d, None, "{}: the narrow sibling must stay 1D", case.label);
            assert!(
                !narrow.source.contains(FLAT_MARKER),
                "{}: the narrow sibling must not carry the flat form",
                case.label
            );

            let width = spec.threads_per_threadgroup_x;
            assert_eq!(spec.threads_per_threadgroup_y, 1, "{}", case.label);
            assert_eq!(
                spec.threadgroups_x * spec.threadgroups_y,
                shape.threads.div_ceil(width),
                "{}: the rectangle must hold exactly the threadgroups the grid needs",
                case.label
            );
            assert!(spec.threadgroups_x <= crate::sized::GRID_MAX_THREADGROUPS_X, "{}", case.label);
            assert!(spec.threadgroups_y <= limit, "{}", case.label);
            match shape.threadgroup_width {
                Some(pinned) if shape.threads % pinned == 0 => {
                    assert_eq!(width, pinned, "{}: a pinned width that divides the grid is launched as-is", case.label);
                }
                Some(_) => assert_eq!(width, SIMD_WIDTH, "{}: a pinned width that does not divide the grid falls back to one simdgroup", case.label),
                None => assert_eq!(width, SIMD_WIDTH, "{}", case.label),
            }
        }
    }

    #[test]
    fn the_flat_form_gets_its_own_pipeline_identity_and_the_linear_sibling_keeps_its_own() {
        for case in flat_cases() {
            let wide_key = kernel_cache_key(&case.bound, &case.packed, case.policy)
                .unwrap_or_else(|error| panic!("{}: cache key failed: {error}", case.label));
            let narrow_key = kernel_cache_key(&case.narrow, &case.packed, case.policy)
                .unwrap_or_else(|error| panic!("{}: sibling cache key failed: {error}", case.label));

            assert!(wide_key.contains("_wg"), "{}: {wide_key}", case.label);
            assert!(!narrow_key.contains("_wg"), "{}: {narrow_key}", case.label);
        }
    }

    #[test]
    fn a_grid_that_fits_a_32_bit_thread_index_keeps_the_linear_form() {
        let (bound, packed) = per_layer_projection_op(1872, 1536, 8960);

        let kernel = emit(&bound, &packed, NumericPolicy::llama_relaxed()).expect("emits");

        assert_eq!(kernel.grid.threads, 1872 * 8960 * 256, "the last row count that fits is 4_294_082_560 threads");
        assert!(kernel.grid.threads <= u64::from(u32::MAX));
        assert_eq!(kernel.grid.grid2d, None);
        assert!(kernel.source.contains(LINEAR_SIGNATURE));
        assert!(!kernel.source.contains(FLAT_MARKER));
    }

    #[test]
    fn a_thread_count_that_wraps_u64_is_rejected_instead_of_truncated() {
        let bound = BoundOp {
            node: NodeId(4),
            dtype: DType::Float32,
            extents: vec![u64::from(u32::MAX), u64::from(u32::MAX), 8],
            kind: BoundOpKind::Iota,
        };

        let overflow = kernel_dispatch_shape(&bound, &BTreeMap::new(), NumericPolicy::default());

        assert!(
            matches!(
                overflow,
                Err(EmitError::GridExceedsThreadIndex { node, threads: u64::MAX, .. }) if node == NodeId(4)
            ),
            "(2^32-1)^2 * 8 threads is 1.5e20, past u64::MAX: {overflow:?}"
        );
    }

    #[test]
    fn a_grid_with_more_threadgroups_than_two_axes_can_hold_is_rejected() {
        let bound = BoundOp {
            node: NodeId(4),
            dtype: DType::Float32,
            extents: vec![u64::from(u32::MAX), u64::from(u32::MAX)],
            kind: BoundOpKind::Iota,
        };

        let rejected = kernel_dispatch_shape(&bound, &BTreeMap::new(), NumericPolicy::default());

        assert!(
            matches!(rejected, Err(EmitError::GridExceedsThreadIndex { .. })),
            "(2^32-1)^2 threads is 5.8e17 threadgroups of 32, and their largest divisor under \
             {} leaves y above u32::MAX: {rejected:?}",
            crate::sized::GRID_MAX_THREADGROUPS_X
        );
    }

    #[test]
    fn a_threadgroup_count_with_no_divisor_to_spill_into_y_is_rejected() {
        let prime_groups = 4_294_967_311u64;

        let rejected = flat_grid2d(NodeId(3), prime_groups * SIMD_WIDTH, None, false);

        assert!(
            matches!(
                rejected,
                Err(EmitError::GridExceedsThreadIndex { node, limit, .. })
                    if node == NodeId(3) && limit == u64::from(u32::MAX) * SIMD_WIDTH
            ),
            "a prime group count above u32::MAX cannot be split: {rejected:?}"
        );
    }

    #[test]
    fn a_composite_threadgroup_count_above_u32_max_splits_across_both_axes() {
        let composite_groups = 3 * 1_431_655_771u64;

        let spec = flat_grid2d(NodeId(3), composite_groups * SIMD_WIDTH, None, false)
            .expect("3 * 1431655771 threadgroups split as 3 x 1431655771");

        assert_eq!(spec.threadgroups_x * spec.threadgroups_y, composite_groups);
        assert!(spec.threadgroups_y <= u64::from(u32::MAX));
    }

    #[test]
    fn the_rectangle_never_launches_a_threadgroup_the_grid_does_not_need() {
        let spec = flat_grid2d(NodeId(17), 1873 * 8960 * 256, Some(256), false).expect("node 17 at 1873 rows");

        assert_eq!(spec.threadgroups_x * spec.threadgroups_y, 1873 * 8960);
        assert!(spec.threadgroups_x <= crate::sized::GRID_MAX_THREADGROUPS_X);
        assert_eq!(spec.threads_per_threadgroup_x, 256);
    }

    #[test]
    fn a_pinned_threadgroup_width_that_does_not_divide_the_grid_launches_one_simdgroup_wide() {
        let threads = SIMD_WIDTH * 4_000_000_001;

        let spec = flat_grid2d(NodeId(5), threads, Some(64), true).expect("an odd number of simdgroups still launches");

        assert_eq!(spec.threads_per_threadgroup_x, SIMD_WIDTH);
        assert_eq!(spec.threadgroups_x * spec.threadgroups_y, threads.div_ceil(SIMD_WIDTH));
    }

    #[test]
    fn expert_source_substitution_refuses_the_flat_form() {
        let (bound, packed) = per_layer_projection_op(1873, 1536, 8960);

        let codecs = operand_codecs(&bound, &packed);
        let threads = grid_threads(&bound, &codecs, NumericPolicy::llama_relaxed(), true).expect("threads");

        let refused = grid2d_for(&bound, &codecs, NumericPolicy::llama_relaxed(), true, threads);

        assert!(
            matches!(refused, Err(EmitError::WideGridUnsupported { .. })),
            "{refused:?}"
        );
    }

    #[cfg(feature = "metal-moe-mul-mat-id")]
    #[test]
    fn a_round_batched_reduce_past_the_thread_index_reads_its_round_from_the_flat_group_z() {
        let mut bound = round_batched_matmul_op(3);
        bound.extents[0] = 20_000_000;
        bound.extents[1] = 1_000;

        let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("emits");

        assert_eq!(
            kernel.grid.grid2d.map(|spec| spec.form),
            Some(Grid2DForm::FlatThreadgroupIndex)
        );
        assert_eq!(kernel.grid.depth, 3, "the round axis stays the z extent");
        assert!(kernel.source.contains("round_table[wide_group.z]"), "{}", kernel.source);
        assert!(!kernel.source.contains("round_gid"), "{}", kernel.source);
        assert!(kernel.source.contains("ulong gid = "));
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn the_tiled_gemm_one_d_form_past_the_thread_index_takes_the_flat_form() {
        let wide = tiled_gemm_op(300_000, 1536, 4_000_000);
        let narrow = tiled_gemm_op(510, 1536, 128);
        let weight = wide.operands()[0].0;
        let packed = BTreeMap::from([(weight, Codec::Q4_0)]);

        temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", None::<&str>, || {
            let kernel = emit(&wide, &packed, NumericPolicy::default()).expect("wide tiled emits");
            let sibling = emit(&narrow, &packed, NumericPolicy::default()).expect("narrow tiled emits");

            assert!(kernel.grid.threads > u64::from(u32::MAX));
            let spec = kernel.grid.grid2d.expect("a tiled grid past u32::MAX threads cannot dispatch 1D");
            assert_eq!(spec.form, Grid2DForm::FlatThreadgroupIndex);
            assert_eq!(spec.threads_per_threadgroup_x, (TILED_GEMM_NSG as u64) * SIMD_WIDTH);
            assert_eq!(spec.threadgroups_x * spec.threadgroups_y * spec.threads_per_threadgroup_x, kernel.grid.threads);
            assert!(kernel.source.contains(FLAT_MARKER), "{}", kernel.source);
            assert!(!kernel.source.contains(LINEAR_SIGNATURE), "{}", kernel.source);
            assert_eq!(sibling.grid.grid2d, None);
            assert!(!sibling.source.contains(FLAT_MARKER));
        });
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn the_tile_coordinate_form_needs_no_widening_however_wide_the_grid() {
        let wide = tiled_gemm_op(300_000, 1536, 4_000_000);
        let weight = wide.operands()[0].0;
        let packed = BTreeMap::from([(weight, Codec::Q4_0)]);

        temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", Some("1"), || {
            let kernel = emit(&wide, &packed, NumericPolicy::default()).expect("wide tiled emits");

            assert!(kernel.grid.threads > u64::from(u32::MAX));
            let spec = kernel.grid.grid2d.expect("the lever dispatches threadgroup coordinates");
            assert_eq!(spec.form, Grid2DForm::TileCoordinates);
            assert!(!kernel.source.contains(FLAT_MARKER), "{}", kernel.source);
        });
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn the_dense_batched_gemm_one_d_form_past_the_thread_index_reads_its_batch_from_the_flat_group_z() {
        let bound = dense_batched_score_shaped_op(2_000_000, 2_000_000, 8, 128);

        temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", None::<&str>, || {
            let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("dense batched emits");

            assert!(kernel.grid.threads > u64::from(u32::MAX));
            assert_eq!(kernel.grid.depth, 8, "the batch axis stays the z extent");
            assert_eq!(
                kernel.grid.grid2d.map(|spec| spec.form),
                Some(Grid2DForm::FlatThreadgroupIndex)
            );
            assert!(kernel.source.contains("(long)wide_group.z"), "{}", kernel.source);
            assert!(!kernel.source.contains("dense_batch_gid"), "{}", kernel.source);
            assert!(kernel.source.contains(FLAT_MARKER));
        });
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn a_batched_launch_whose_slice_fits_but_whose_whole_grid_does_not_takes_the_flat_form() {
        let bound = dense_batched_score_shaped_op(102_400, 200_000, 8, 128);

        temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", None::<&str>, || {
            let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("emits");

            assert!(kernel.grid.threads <= u64::from(u32::MAX), "one z slice fits: {}", kernel.grid.threads);
            assert_eq!(kernel.grid.depth, 8);
            assert!(kernel.grid.threads * kernel.grid.depth > u64::from(u32::MAX));
            assert_eq!(
                kernel.grid.grid2d.map(|spec| spec.form),
                Some(Grid2DForm::FlatThreadgroupIndex),
                "the launch covers threads x depth, and that product is what a 32-bit width cannot hold"
            );
            assert!(kernel.source.contains(FLAT_MARKER));
        });
    }

    #[test]
    fn a_pinned_width_that_does_not_divide_the_grid_is_refused_for_a_kernel_that_reads_it() {
        let refused = flat_grid2d(NodeId(5), SIMD_WIDTH * 4_000_000_001, Some(64), false);

        assert!(
            matches!(refused, Err(EmitError::WideGridUnsupported { node, .. }) if node == NodeId(5)),
            "{refused:?}"
        );
    }

    #[test]
    fn cached_attention_with_grouped_queries_is_refused_when_its_width_does_not_divide_the_grid() {
        let mut bound = cached_attention_op_dynamic(0, 1_000_000);
        bound.extents = vec![300_000_001, 1, 1, 4];
        let BoundOpKind::CachedAttention { query_groups, .. } = &mut bound.kind else {
            unreachable!("cached_attention_op_dynamic builds a CachedAttention")
        };
        *query_groups = 3;

        let refused = emit(&bound, &BTreeMap::new(), NumericPolicy::default());

        assert!(
            matches!(refused, Err(EmitError::WideGridUnsupported { .. })),
            "three query groups share one threadgroup, so a grid that is not a whole number of them \
             cannot fall back to 32 lanes: {:?}",
            refused.as_ref().map(|kernel| kernel.grid)
        );
    }

    #[test]
    fn a_flat_launch_is_re_cut_for_a_pipeline_that_cannot_run_the_pinned_width() {
        let spec = flat_grid2d(NodeId(17), 1873 * 8960 * 256, Some(256), false).expect("node 17 at 1873 rows");

        let fitted = fit_flat_width(spec, 128);

        assert_eq!(fitted.threads_per_threadgroup_x, 128);
        assert_eq!(
            fitted.threadgroups_x * fitted.threadgroups_y * fitted.threads_per_threadgroup_x,
            1873 * 8960 * 256,
            "halving the width doubles the threadgroups and launches the same threads"
        );
        assert!(fitted.threadgroups_x <= crate::sized::GRID_MAX_THREADGROUPS_X);
        assert_eq!(fit_flat_width(spec, 256), spec, "a width the pipeline allows is left alone");
        assert_eq!(fit_flat_width(spec, 1024), spec);
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn the_tile_coordinate_launch_is_never_re_cut_for_a_pipeline_width() {
        let tiles = Grid2DSpec {
            form: Grid2DForm::TileCoordinates,
            threadgroups_x: 47,
            threadgroups_y: 140,
            threads_per_threadgroup_x: SIMD_WIDTH,
            threads_per_threadgroup_y: TILED_GEMM_NSG as u64,
        };

        assert_eq!(fit_flat_width(tiles, 64), tiles);
    }

    /// Offline compile with the real Metal toolchain (`xcrun metal -c`, no
    /// device); a missing toolchain fails the test, never skips it.
    #[cfg(target_os = "macos")]
    fn assert_compiles_with_the_metal_toolchain(label: &str, source: &str) {
        let directory = tempfile::tempdir().expect("tempdir creation must not fail in ci");
        let metal_path = directory.path().join("kernel.metal");
        std::fs::write(&metal_path, source).expect("write metal source to a temp file");
        let output = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&metal_path)
            .arg("-o")
            .arg(directory.path().join("kernel.air"))
            .output()
            .unwrap_or_else(|error| panic!("metal toolchain unavailable ({error}) -- a red gate, not a skip"));
        assert!(
            output.status.success(),
            "{label}: metal compile failed:\n--- source ---\n{source}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_cached_attention_merge_kernel_past_the_thread_index_takes_the_flat_form_and_compiles() {
        let mut bound = cached_attention_op_dynamic(0, 1_000_000);
        bound.extents = vec![300_000_000, 1, 1, 4];

        let merge = emit_cached_attention_merge(&bound, NumericPolicy::llama_relaxed())
            .expect("emits")
            .expect("a long context under the relaxed policy needs the merge dispatch");

        assert!(merge.grid.threads > u64::from(u32::MAX), "{}", merge.grid.threads);
        assert_eq!(merge.grid.grid2d.map(|spec| spec.form), Some(Grid2DForm::FlatThreadgroupIndex));
        assert!(merge.source.contains(FLAT_MARKER), "{}", merge.source);
        assert_compiles_with_the_metal_toolchain("cached attention merge", &merge.source);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn cached_attention_with_grouped_queries_past_the_thread_index_launches_its_pinned_width_and_compiles() {
        let mut bound = cached_attention_op_dynamic(0, 1_000_000);
        bound.extents = vec![300_000_000, 1, 1, 4];
        let BoundOpKind::CachedAttention { query_groups, .. } = &mut bound.kind else {
            unreachable!("cached_attention_op_dynamic builds a CachedAttention")
        };
        *query_groups = 4;

        let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("emits");
        let spec = kernel.grid.grid2d.expect("a grid past u32::MAX cannot dispatch 1D");

        assert_eq!(Some(spec.threads_per_threadgroup_x), kernel.grid.threadgroup_width, "pinned width is launched as pinned");
        assert!(spec.threads_per_threadgroup_x > SIMD_WIDTH, "the four query groups share one threadgroup");
        assert_compiles_with_the_metal_toolchain("cached attention, query_groups=4", &kernel.source);
    }

    #[cfg(all(target_os = "macos", feature = "metal-moe-mul-mat-id"))]
    #[test]
    fn a_round_batched_reduce_past_the_thread_index_compiles() {
        let mut bound = round_batched_matmul_op(3);
        bound.extents[0] = 20_000_000;
        bound.extents[1] = 1_000;

        let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("emits");

        assert_compiles_with_the_metal_toolchain("round-batched reduce", &kernel.source);
    }

    #[cfg(all(target_os = "macos", feature = "metal-moe-mul-mat-id"))]
    #[test]
    fn a_round_batched_reduce_within_the_thread_index_compiles() {
        let bound = round_batched_matmul_op(3);

        let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("emits");

        assert_compiles_with_the_metal_toolchain("round-batched reduce, scalar gid", &kernel.source);
    }

    #[cfg(feature = "metal-q4k-split-k")]
    #[test]
    fn the_split_k_packed_row_body_gets_its_threadgroup_width_back_as_a_local() {
        let bound = packed_row_multi_token_op(1, 256, 4_000_000_000);
        let weight = bound.operands()[0].0;
        let packed = BTreeMap::from([(weight, Codec::Q4K)]);

        let kernel = emit(&bound, &packed, NumericPolicy::default()).expect("emits");

        assert!(kernel.source.contains("uint tptg = wide_width.x;"), "{}", kernel.source);
        assert!(!kernel.source.contains("uint tptg [["), "{}", kernel.source);
    }
}

/// What one encode of a decode-shaped op costs in heap allocations, counted
/// over the cache-key and dispatch-shape pair every `resolve_steps` and cold
/// `encode_op` performs. The counted budget for each op is what the same pair
/// allocated on `73fd1cbd` (main, before the grid decision existed), measured
/// by running this module against that commit with the pair spelled
/// `kernel_cache_key` then `kernel_dispatch_shape`; the tip must not exceed it.
/// `proxima_test::alloc_count` installs the counting allocator for this test
/// binary only under the `alloc-count` feature, and nextest gives each test its
/// own process, so no other test shares the counter.
#[cfg(feature = "alloc-count")]
mod encode_allocation_budget {
    use proxima_test::alloc_count::{CountingAllocator, allocations, reset};

    use super::*;

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    struct EncodeCase {
        label: &'static str,
        bound: BoundOp,
        packed: PackedOperands,
        base_allocations: usize,
    }

    fn packed_weight(bound: &BoundOp, codec: Codec) -> PackedOperands {
        let mut packed = PackedOperands::new();
        packed.insert(bound.operands()[0].0, codec);
        packed
    }

    fn encode_cases() -> Vec<EncodeCase> {
        let q4k_matvec = packed_row_multi_token_op(1, 1536, 8960);
        let q6k_matvec = packed_row_multi_token_op(1, 4096, 2048);
        let tiled = tiled_gemm_op(64, 256, 128);
        vec![
            EncodeCase {
                label: "q4k_matvec_decode",
                packed: packed_weight(&q4k_matvec, Codec::Q4K),
                bound: q4k_matvec,
                base_allocations: BASE_Q4K_MATVEC,
            },
            EncodeCase {
                label: "q6k_matvec_decode",
                packed: packed_weight(&q6k_matvec, Codec::Q6K),
                bound: q6k_matvec,
                base_allocations: BASE_Q6K_MATVEC,
            },
            EncodeCase {
                label: "tiled_gemm_prefill",
                packed: packed_weight(&tiled, Codec::Q4K),
                bound: tiled,
                base_allocations: BASE_TILED_GEMM,
            },
            EncodeCase {
                label: "f32_matmul",
                bound: matmul_op(1, 256, 5),
                packed: PackedOperands::new(),
                base_allocations: BASE_F32_MATMUL,
            },
            EncodeCase {
                label: "elementwise_tanh",
                bound: elementwise_tanh_op(4096),
                packed: PackedOperands::new(),
                base_allocations: BASE_ELEMENTWISE,
            },
            EncodeCase {
                label: "cached_attention",
                bound: cached_attention_op_dynamic(64, 1),
                packed: PackedOperands::new(),
                base_allocations: BASE_CACHED_ATTENTION,
            },
            EncodeCase {
                label: "cached_softmax_weights",
                bound: cached_softmax_weights_op(4, 64, 64),
                packed: PackedOperands::new(),
                base_allocations: BASE_CACHED_SOFTMAX,
            },
        ]
    }

    const BASE_Q4K_MATVEC: usize = 297;
    const BASE_Q6K_MATVEC: usize = 258;
    const BASE_TILED_GEMM: usize = 254;
    const BASE_F32_MATMUL: usize = 179;
    const BASE_ELEMENTWISE: usize = 14;
    const BASE_CACHED_ATTENTION: usize = 13;
    const BASE_CACHED_SOFTMAX: usize = 9;

    fn encode_pair(case: &EncodeCase) -> usize {
        let policy = NumericPolicy::default();
        let before = allocations();
        let (_, grid) = kernel_dispatch_shape(&case.bound, &case.packed, policy)
            .expect("the fixture's grid is describable");
        let key = kernel_cache_key_for_grid(&case.bound, &case.packed, policy, &grid)
            .expect("the fixture has a cache key");
        let count = allocations() - before;
        core::hint::black_box(key);
        count
    }

    #[test]
    fn encoding_a_decode_op_allocates_no_more_than_it_did_before_the_grid_decision() {
        let mut over_budget = Vec::new();
        for case in encode_cases() {
            encode_pair(&case);
            reset();
            let measured = encode_pair(&case);
            eprintln!("encode_allocations label={} measured={measured}", case.label);
            if measured > case.base_allocations {
                over_budget.push((case.label, measured, case.base_allocations));
            }
        }

        assert!(
            over_budget.is_empty(),
            "(label, measured, base) over budget: {over_budget:?}"
        );
    }
}

/// The merge splice widens a kernel's scalar `uint gid [[thread_position_in_grid]]`
/// into a `uint3` and reads the member index off its `z`. A kernel in the
/// tile-coordinate form has no scalar `gid` at all (it declares
/// `[[threadgroup_position_in_grid]]` instead), so the splice cannot apply to
/// it: `build_merged_dispatch` declines a leader with `grid2d` set rather than
/// let this error fail plan resolution. Merging tile-form kernels would need a
/// splice over threadgroup coordinates, which changes kernel text and needs a
/// device to check, so it stays a separate measured change.
#[cfg(all(feature = "metal-horizontal-merge", feature = "metal-tiled-gemm"))]
#[test]
fn the_merge_splice_has_no_scalar_gid_to_widen_in_a_tile_form_kernel() {
    let bound = tiled_gemm_op(510, 1536, 128);
    let weight = bound.operands()[0].0;
    let packed = BTreeMap::from([(weight, Codec::Q4_0)]);

    temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", Some("1"), || {
        let mut kernel = emit(&bound, &packed, NumericPolicy::default()).expect("tiled emits");
        assert_eq!(
            kernel.grid.grid2d.map(|spec| spec.form),
            Some(Grid2DForm::TileCoordinates),
            "the lever must put this fixture in the tile form for the test to mean anything"
        );

        let outcome = splice_horizontal_merge_base_table(
            &mut kernel,
            bound.node,
            0,
            "uchar",
            1,
            "float",
            "float",
        );

        assert!(
            matches!(
                outcome,
                Err(EmitError::RenderKindMismatch {
                    expected: "scalar thread_position_in_grid parameter",
                    ..
                })
            ),
            "{outcome:?}"
        );
    });
}
