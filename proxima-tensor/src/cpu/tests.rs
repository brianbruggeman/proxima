use super::*;
use crate::bind::BodyStep;
use crate::map::{self, AxisTerm, IndexMap};
use crate::op::{Extent, Reduce, append};

use std::time::Instant;

use crate::test_support::Lcg;
use proptest::proptest;

/// `spec.rs`'s real `wo` shape (`ROW 431`,
/// `append_qwen35_dense_attention_only_with_taps`, `spec.rs:4358-4404`
/// and `spec.rs:9017-9031`): `gated_attended = attended * sigmoid_gate`
/// (a genuine two-leaf composed chain), `wo = wo_flat * o_head_ones`
/// (the packed weight's own ones-broadcast reshape, the multi-term
/// packed-row contraction), `wo_product = gated_attended * wo`, reduced
/// over the packed contraction axes. Built at tiny dims with `seq` left
/// free so callers can drive both the `s == 1` (decode) and `s > 1`
/// (prefill) shapes through the identical graph.
fn wo_shaped_program(seq: u32) -> (Vec<Op>, NodeId) {
    use crate::op::{Extent, append};
    let mut program = Vec::new();
    let (kv_heads, group, head_dim, embedding) = (1u32, 1u32, 2u32, 5u32);
    let attended = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(seq),
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_dim),
            ],
            name: None,
        },
    );
    let sigmoid_gate = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(seq),
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_dim),
            ],
            name: None,
        },
    );
    let gated = crate::spec::elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugd"), (sigmoid_gate, "sugd->sugd")],
    )
    .expect("gated elementwise builds");
    let wo_flat = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(kv_heads * group * head_dim),
                Extent::Static(embedding),
            ],
            name: None,
        },
    );
    let ones = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_dim),
            ],
            value: 1.0,
        },
    );
    let wo = crate::spec::elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (
                wo_flat,
                alloc::format!("{}*u+{head_dim}*g+d,e->ugde", head_dim * group).as_str(),
            ),
            (ones, "ugd->ugde"),
        ],
    )
    .expect("wo elementwise builds");
    let wo_product = crate::spec::elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gated, "sugd->sugde"), (wo, "ugde->sugde")],
    )
    .expect("wo_product elementwise builds");
    let attn_out = crate::spec::reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugde->sugde",
        "se->sugde",
    )
    .expect("attn_out reduce builds");
    (program, attn_out)
}

/// ROW 431 memory boundary: for the `s == 1` (decode) shape, the fused
/// reduce reaching `packed_reduce_activation_operand` must carry
/// EXACTLY the gated activation (`[u, g, d]`, never the output axis
/// `o`/`e`) and the packed weight leaf as its two operands, one
/// `Multiply` step -- proving `composed_packed_product_activation`
/// materializes the ACTIVATION side, not the `[u, g, d, o]` product ROW
/// 430 used to. `quarantine_broadcast_operands` already force-
/// materializes `gated_attended` here (its own extent lacks the `o`
/// axis) independent of this rule; this test's own contribution is
/// proving no OTHER buffer in the bound program carries the `o` axis
/// either -- i.e. the product itself was never materialized.
#[test]
fn packed_reduce_planner_materializes_the_gated_activation_not_the_product_at_decode() {
    let (program, attn_out) = wo_shaped_program(1);
    let shapes = shape::infer(&program, &[]).expect("shape inference succeeds");
    let resolved = bind::bind(&program, &shapes, &[attn_out], NumericPolicy::bit_exact())
        .expect("bind succeeds");
    let bound = resolved
        .iter()
        .find(|op| op.node == attn_out)
        .expect("attn_out is present in the bound program");
    let BoundOpKind::Reduce {
        element_body,
        operands,
        ..
    } = &bound.kind
    else {
        panic!("attn_out must bind to a Reduce");
    };
    assert_eq!(
        element_body.steps,
        vec![BodyStep {
            op: ScalarOp::Multiply,
            args: vec![StepArg::Operand(0), StepArg::Operand(1)]
        }],
        "the fused reduce must be a bare two-operand product, never a wider composed body"
    );
    assert_eq!(
        operands.len(),
        2,
        "packed_reduce_activation_operand requires exactly two operands"
    );
    let pre_reduction_extent: u64 = bound.extents.iter().product();
    for op in &resolved {
        if op.node == bound.node {
            continue;
        }
        let own_extent: u64 = op.extents.iter().product();
        assert_ne!(
            own_extent, pre_reduction_extent,
            "node {:?} (extents {:?}) shares the reduce's full [u,g,d,o] pre-reduction extent -- \
             the activation-weight PRODUCT must never be materialized standalone, \
             only the smaller [u,g,d] activation is",
            op.node, op.extents
        );
    }
}

/// ROW 431 admission contract, RED-before/GREEN-after: a fused reduce
/// whose `element_body` composes TWO steps over THREE physical operands
/// -- `sum(W * (x * g))`, the shape `sum(W * (x * sigmoid(g)))`
/// collapses to structurally once the planner fuses the gate multiply
/// straight into the same reduce -- must never reach
/// `matmul_q4k_f32`'s buffer read at all. Before this admission check
/// existed, `resolved.operands().find(|n| *n != weight_node)` picked
/// whichever of `x`/`g` happened to sort first and fed IT verbatim to
/// the kernel, silently computing `matmul(W, x)` (or `matmul(W, g)`)
/// instead of `matmul(W, x * g)` -- a wrong number with no error at all.
/// This test drives that exact composed body straight at
/// `run_reduce_quantized`, bypassing `bind::bind` entirely (the planner
/// would never emit this shape unfused today, per
/// `packed_reduce_planner_materializes_the_gated_activation_not_the_product_at_decode`
/// above -- this proves the EXECUTOR's own contract independent of
/// whether the planner currently honors it).
#[test]
fn run_reduce_quantized_rejects_a_composed_body_wider_than_a_bare_weight_activation_product() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let rows = 1u32;
    let k = QK_K as u32;
    let weight_f32 = random_vec(701, rows as usize * k as usize);
    let mut weight_bytes = vec![0u8; rows as usize * BLOCK_BYTES];
    quantize(&weight_f32, &mut weight_bytes).expect("row length is QK_K by construction");

    let x: Vec<f32> = random_vec(702, k as usize);
    let g: Vec<f32> = random_vec(703, k as usize);
    let weight_node = NodeId(0);
    let x_node = NodeId(1);
    let g_node = NodeId(2);
    let flat_layout = || bind::Layout {
        base: 0,
        strides: smallvec::smallvec![1],
    };

    let resolved = BoundOp {
        node: NodeId(3),
        dtype: DType::Float32,
        extents: alloc::vec![rows as u64, k as u64],
        kind: BoundOpKind::Reduce {
            element_body: ComposedBody {
                steps: alloc::vec![
                    step(
                        ScalarOp::Multiply,
                        &[StepArg::Operand(1), StepArg::Operand(2)]
                    ),
                    step(ScalarOp::Multiply, &[StepArg::Operand(0), StepArg::Step(0)]),
                ],
            },
            reduce_op: ScalarOp::Add,
            init: ReduceInit::Zero,
            keep: Keep::Reduce,
            operands: alloc::vec![
                (weight_node, flat_layout(), None),
                (x_node, flat_layout(), None),
                (g_node, flat_layout(), None),
            ],
            output_axes: smallvec::smallvec![0],
            out_layout: flat_layout(),
            out_scatter: None,
            epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
            epilogue_operands: Vec::new(),
            epilogue_broadcast_axes: smallvec::smallvec![],
        },
    };
    let buffers: Vec<Option<Cow<'_, [f32]>>> = alloc::vec![
        None,
        Some(Cow::Borrowed(x.as_slice())),
        Some(Cow::Borrowed(g.as_slice())),
    ];
    let mut output = vec![0.0f32; rows as usize];

    let error = run_reduce_quantized(
        &resolved,
        &buffers,
        QuantizedBlock::Q4K(&weight_bytes),
        weight_node,
        None,
        None,
        false,
        &mut output,
    )
    .expect_err("a composed body wider than a bare product must be a typed rejection");
    assert_eq!(
        error,
        TensorError::NotLowerable {
            node: resolved.node,
            reason: "packed reduce admits only W\u{b7}a with a materialized a",
        },
        "the admission contract must name itself, not surface a downstream shape error"
    );
}

/// `b = a * scale; c = b + bias; d = c * c` -- the same shape
/// `bind::tests::elementwise_chain_program` builds (private to that
/// module's own test scope), inlined here rather than reused across the
/// module boundary since this is `plan_trace_named`'s only consumer.
fn elementwise_chain_program() -> (Vec<Op>, NodeId, NodeId) {
    use crate::op::{Extent, append};

    let mut program = Vec::new();
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let scale = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let bias = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let identity = || crate::map::IndexMap::Affine(crate::map::projection(1, &[0]));
    let b = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(a, identity()), (scale, identity())],
            name: None,
        },
    );
    let c = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![(b, identity()), (bias, identity())],
            name: None,
        },
    );
    let d = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(c, identity()), (c, identity())],
            name: None,
        },
    );
    (program, b, d)
}

/// This is the exact shape `bind::tests::an_elementwise_intermediate_requested_as_an_output_prevents_fusion`
/// proves at the `resolved` level -- `plan_trace_named` must report the
/// SAME output-set-dependent flip as a `PlanDecision`, since a caller
/// diffing two `plan_trace_named` calls on the SAME program with
/// different output sets needs exactly this kind of node to show up as
/// a difference.
#[test]
fn plan_trace_named_reports_fusion_flip_when_an_intermediate_becomes_an_output() {
    let (program, b, d) = elementwise_chain_program();

    let fused_only = plan_trace_named(&program, &[], &[], &[d]).expect("plans with one output");
    let with_intermediate_output =
        plan_trace_named(&program, &[], &[], &[b, d]).expect("plans with two outputs");

    assert!(
        fused_only
            .iter()
            .any(|decision| decision.node == b && decision.decision == "absorbed"),
        "b must be absorbed into d's composed body when only d is requested: {fused_only:?}"
    );
    assert!(
        !with_intermediate_output
            .iter()
            .any(|decision| decision.node == b && decision.decision == "absorbed"),
        "b must NOT be absorbed once it is itself a requested output: {with_intermediate_output:?}"
    );
}

/// `z = reduce(Multiply(x, w))` (a dense dot-product fold, `w` a NAMED
/// 2-D input so `build_packed_width_panels` sees it as a packable
/// constant, the exact shape a real weight matrix takes) then
/// `g = sigmoid(z)` (`Negate`/`Exponential`/`Add`/`Reciprocal`), `z`'s
/// ONLY consumer -- the identical GDN output-gate shape row NNN's own
/// discipline note reproduces. Requesting `g` alone (small) lets
/// `bind::bind`'s `reduce-epilogue-fusion` fold `z`'s sigmoid chain onto
/// the reduce itself (`z` is not a requested output, has exactly one
/// consumer); requesting `[z, g]` (wide) keeps them separate since a
/// requested output must still materialize on its own
/// (`reduce_epilogue_candidates`'s own admission rule). Before the fix,
/// `width_tile_pack_candidate` admitted the SMALL case's fused node for
/// `law 6∘5` width-tile packing without checking its epilogue was
/// trivial, and `run_resolved_nodes_in_arena`'s packed branch calls
/// `run_reduce` directly -- bypassing `run_node_into`'s own
/// `apply_reduce_epilogue` call entirely -- so `g` silently came back as
/// raw `z`, never sigmoided.
#[cfg(all(feature = "reduce-epilogue-fusion", target_arch = "aarch64"))]
fn reduce_with_sigmoid_epilogue_program() -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(8), Extent::Static(2)],
            name: Some("x".to_string()),
        },
    );
    let w = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(2), Extent::Static(16)],
            name: Some("w".to_string()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (x, IndexMap::Affine(map::projection(3, &[0, 1]))),
                (w, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let z = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let identity_2d = || IndexMap::Affine(map::projection(2, &[0, 1]));
    let negated = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: vec![(z, identity_2d())],
            name: None,
        },
    );
    let exponentiated = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Exponential,
            operands: vec![(negated, identity_2d())],
            name: None,
        },
    );
    let one = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: vec![Extent::Static(8), Extent::Static(16)],
            value: 1.0,
        },
    );
    let one_plus = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![(exponentiated, identity_2d()), (one, identity_2d())],
            name: None,
        },
    );
    let g = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: vec![(one_plus, identity_2d())],
            name: None,
        },
    );
    (program, z, g)
}

#[test]
#[cfg(all(feature = "reduce-epilogue-fusion", target_arch = "aarch64"))]
fn a_reduce_with_a_sigmoid_epilogue_is_not_stranded_by_width_tile_packing() {
    let (program, z, g) = reduce_with_sigmoid_epilogue_program();
    let mut lcg = Lcg(0x5eed);
    let x_data: Vec<f32> = (0..16).map(|_| lcg.next_unit()).collect();
    let w_data: Vec<f32> = (0..32).map(|_| lcg.next_unit()).collect();
    let named: Vec<(&str, &[f32])> = vec![("x", &x_data), ("w", &w_data)];

    let small = evaluate_named(&program, &[], &named, &[g]).expect("small (g alone) evaluates");
    let wide = evaluate_named(&program, &[], &named, &[z, g]).expect("wide (z and g) evaluates");

    let small_g = small.get(g).expect("small g present").0;
    let wide_z = wide.get(z).expect("wide z present").0;
    let wide_g = wide.get(g).expect("wide g present").0;

    let manual_sigmoid: Vec<f32> = wide_z
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect();

    assert_eq!(
        small_g, wide_g,
        "requesting g alone (forcing the packed-panel width-tile fast path) must match \
         requesting [z, g] (the un-packed, epilogue-correct path): small={small_g:?} wide={wide_g:?}"
    );
    for (index, (actual, expected)) in small_g.iter().zip(manual_sigmoid.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "g[{index}] = {actual}, expected sigmoid(z[{index}]) = {expected} (z[{index}]={})",
            wide_z[index]
        );
    }
}

/// A placed output (`caller_owned` in `omega::metal::finish`) reports
/// `is_placed() == true` and `get() == None` -- distinct from a node
/// that was simply never requested, which reports `is_placed() ==
/// false` alongside the same `get() == None`. Before
/// `Evaluated::from_parts_with_placed` existed, both cases collapsed to
/// the same `None`, so a caller reading a placed node through
/// `Evaluated` instead of the placement it actually landed in saw
/// `MissingEvaluatedNode` with no way to tell that from a genuine bug.
#[test]
fn placed_output_reports_placed_not_missing() {
    let requested = NodeId(1);
    let never_requested = NodeId(2);
    let mut placed = BTreeSet::new();
    placed.insert(requested);

    let evaluated = Evaluated::from_parts_with_placed(requested, Vec::new(), None, placed);

    assert!(evaluated.is_placed(requested));
    assert!(evaluated.get(requested).is_none());
    assert!(!evaluated.is_placed(never_requested));
    assert!(evaluated.get(never_requested).is_none());
}

/// `max(reduce + bias, 0)` — [`EpilogueKind::Clip`]'s own shape, walking
/// [`match_epilogue`] the authored (never eliminated) way, per P17: a
/// worked example over a hand-built [`ComposedBody`], not a fixture that
/// happens to hit the pattern. `reduce_slot=0` is the anchor
/// [`matches_clip_head`] discovers `bias`/`zero` against; this fixture's
/// authored order happens to already have them at 1/2, so the
/// discovered slots equal the old hardcoded literals here -- ROW NNN's
/// `diagnostic_clip_epilogue_engages_when_reduce_is_not_operand_zero`
/// below is the case where they do not.
#[test]
fn match_epilogue_recognizes_clip_authored() {
    let body = ComposedBody {
        steps: vec![
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Operand(0), StepArg::Operand(1)],
            },
            BodyStep {
                op: ScalarOp::Maximum,
                args: vec![StepArg::Step(0), StepArg::Operand(2)],
            },
        ],
    };
    assert_eq!(
        match_epilogue(&body, 0, None),
        Some((EpilogueKind::Clip, EpilogueSlots::Clip { bias: 1, zero: 2 }))
    );
}

/// The same `Clip` value with the `+ bias` step already eliminated
/// because `bias == 0` (`push_canonical_step`'s own identity fold) —
/// `Maximum` reads `Operand(0)` (the reduce) directly instead of a
/// `Step`. Unreachable under `NumericPolicy::bit_exact` (the only
/// policy any current caller binds with): `identity_element_signed_zero_nan`
/// only fires once a caller grants `IdentityEliminationSignedZero`,
/// which `bit_exact` never does (`bind.rs`'s own doc). ROW NNN's
/// slot-discovery fix (`matches_clip_head`/`resolve_other_operand`)
/// rejects this shape rather than guess a bias slot that no longer
/// exists in the composed body -- a safe fall-back to the unfused path,
/// not a silent wrong-value risk, so `match_epilogue` now returns
/// `None` here instead of `Some(Clip)`.
#[test]
fn match_epilogue_rejects_clip_after_bias_identity_elimination() {
    let body = ComposedBody {
        steps: vec![BodyStep {
            op: ScalarOp::Maximum,
            args: vec![StepArg::Operand(0), StepArg::Operand(2)],
        }],
    };
    assert_eq!(match_epilogue(&body, 0, None), None);
}

fn clip_norm_body() -> ComposedBody {
    ComposedBody {
        steps: vec![
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Operand(0), StepArg::Operand(1)],
            },
            BodyStep {
                op: ScalarOp::Maximum,
                args: vec![StepArg::Step(0), StepArg::Operand(2)],
            },
            BodyStep {
                op: ScalarOp::Subtract,
                args: vec![StepArg::Step(1), StepArg::Operand(3)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Operand(4), StepArg::Operand(5)],
            },
            BodyStep {
                op: ScalarOp::SquareRoot,
                args: vec![StepArg::Step(3)],
            },
            BodyStep {
                op: ScalarOp::Divide,
                args: vec![StepArg::Step(2), StepArg::Step(4)],
            },
            BodyStep {
                op: ScalarOp::Multiply,
                args: vec![StepArg::Step(5), StepArg::Operand(6)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Step(6), StepArg::Operand(7)],
            },
        ],
    }
}

/// `((max(reduce + bias, 0) - mean) / sqrt(var + eps)) * gamma + beta`
/// authored in full, per P17's worked-example requirement.
#[test]
fn match_epilogue_recognizes_clip_norm_authored() {
    assert_eq!(
        match_epilogue(&clip_norm_body(), 0, None),
        Some((EpilogueKind::ClipNorm, EpilogueSlots::Other))
    );
}

fn norm_body() -> ComposedBody {
    ComposedBody {
        steps: vec![
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Operand(0), StepArg::Operand(1)],
            },
            BodyStep {
                op: ScalarOp::Subtract,
                args: vec![StepArg::Step(0), StepArg::Operand(2)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Operand(3), StepArg::Operand(4)],
            },
            BodyStep {
                op: ScalarOp::SquareRoot,
                args: vec![StepArg::Step(2)],
            },
            BodyStep {
                op: ScalarOp::Divide,
                args: vec![StepArg::Step(1), StepArg::Step(3)],
            },
            BodyStep {
                op: ScalarOp::Multiply,
                args: vec![StepArg::Step(4), StepArg::Operand(5)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Step(5), StepArg::Operand(6)],
            },
        ],
    }
}

/// `((reduce + bias - mean) / sqrt(var + eps)) * gamma + beta`, no relu
/// — authored in full.
#[test]
fn match_epilogue_recognizes_norm_authored() {
    assert_eq!(
        match_epilogue(&norm_body(), 0, None),
        Some((EpilogueKind::Norm, EpilogueSlots::Other))
    );
}

/// `Norm` with the leading `reduce + bias` step already eliminated
/// (`bias == 0`) — `Subtract` reads `Operand(0)` (the raw reduce)
/// directly instead of a `Step`. Unreachable under
/// `NumericPolicy::bit_exact` (see
/// `match_epilogue_rejects_clip_after_bias_identity_elimination`'s own
/// doc for why); `resolve_other_operand` rejects it structurally now
/// rather than reporting a bias slot that does not exist.
#[test]
fn match_epilogue_rejects_norm_after_bias_identity_elimination() {
    let body = ComposedBody {
        steps: vec![
            BodyStep {
                op: ScalarOp::Subtract,
                args: vec![StepArg::Operand(0), StepArg::Operand(2)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Operand(3), StepArg::Operand(4)],
            },
            BodyStep {
                op: ScalarOp::SquareRoot,
                args: vec![StepArg::Step(1)],
            },
            BodyStep {
                op: ScalarOp::Divide,
                args: vec![StepArg::Step(0), StepArg::Step(2)],
            },
            BodyStep {
                op: ScalarOp::Multiply,
                args: vec![StepArg::Step(3), StepArg::Operand(5)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Step(4), StepArg::Operand(6)],
            },
        ],
    };
    assert_eq!(match_epilogue(&body, 0, None), None);
}

fn layer_norm_body() -> ComposedBody {
    ComposedBody {
        steps: vec![
            BodyStep {
                op: ScalarOp::Multiply,
                args: vec![StepArg::Operand(1), StepArg::Operand(2)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Step(0), StepArg::Operand(3)],
            },
            BodyStep {
                op: ScalarOp::SquareRoot,
                args: vec![StepArg::Step(1)],
            },
            BodyStep {
                op: ScalarOp::Divide,
                args: vec![StepArg::Operand(0), StepArg::Step(2)],
            },
            BodyStep {
                op: ScalarOp::Multiply,
                args: vec![StepArg::Step(3), StepArg::Operand(4)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Step(4), StepArg::Operand(5)],
            },
        ],
    }
}

/// `(centered / sqrt(reduce * (1/N) + eps)) * gamma + beta` authored in
/// full — BERT-style `LayerNormalization`'s own unrolled tail.
#[test]
fn match_epilogue_recognizes_layer_norm_authored() {
    assert_eq!(
        match_epilogue(&layer_norm_body(), 1, Some(0)),
        Some((
            EpilogueKind::LayerNorm,
            EpilogueSlots::LayerNorm {
                primary: 0,
                reciprocal_n: Some(2),
                epsilon: 3,
                gamma: 4,
                beta: 5,
            }
        ))
    );
}

/// The `docs/discipline.md`/`cpu.rs:2570-2626` "hidden=1 confluence
/// gap" reproduced directly: `hidden == 1` makes `1/N == 1.0`, so
/// `push_canonical_step` folds `reduce * (1/N)` to the bare reduce
/// operand, one step shorter than `layer_norm_body`'s own 6 —
/// `var_eps` reads `Operand(1)` directly instead of `Step(0)`.
/// `detect_epilogue_kind` (main) hard-coded `steps.len() == 6` and
/// `steps[0] == Multiply(Operand(1), Operand(2))`; this body fails
/// that check on main and passes here, which is the point: a real
/// LayerNorm program at `hidden=1` must not silently fall off the
/// fused path depending on which upstream fold fired first.
#[test]
fn match_epilogue_recognizes_layer_norm_after_hidden_one_confluence() {
    let body = ComposedBody {
        steps: vec![
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Operand(1), StepArg::Operand(3)],
            },
            BodyStep {
                op: ScalarOp::SquareRoot,
                args: vec![StepArg::Step(0)],
            },
            BodyStep {
                op: ScalarOp::Divide,
                args: vec![StepArg::Operand(0), StepArg::Step(1)],
            },
            BodyStep {
                op: ScalarOp::Multiply,
                args: vec![StepArg::Step(2), StepArg::Operand(4)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Step(3), StepArg::Operand(5)],
            },
        ],
    };
    assert_eq!(
        match_epilogue(&body, 1, Some(0)),
        Some((
            EpilogueKind::LayerNorm,
            EpilogueSlots::LayerNorm {
                primary: 0,
                reciprocal_n: None,
                epsilon: 3,
                gamma: 4,
                beta: 5,
            }
        ))
    );
}

/// A shape none of the four kinds is (`Subtract` at the top, not
/// `Add`/`Maximum`) is rejected outright — `match_epilogue` is a
/// recognizer, not a permissive fallback.
#[test]
fn match_epilogue_rejects_unrelated_shape() {
    let body = ComposedBody {
        steps: vec![BodyStep {
            op: ScalarOp::Subtract,
            args: vec![StepArg::Operand(0), StepArg::Operand(1)],
        }],
    };
    assert_eq!(match_epilogue(&body, 0, None), None);
}

/// The real regression this row fixes: `reduce` at operand slot 2 (NOT
/// 0), `bias` at slot 1, `zero` at slot 0 -- the exact permutation
/// `bind::compose_body`'s commutative canonicalization produces for
/// `clip_epilogue_program`-shaped chains (`bind.rs`'s leaf presort
/// sorts `(NodeId, IndexMap)` pairs before slot assignment, and a
/// `Constant`/`Input` upstream of the reduce in program order gets a
/// smaller `NodeId` than the reduce itself). Before ROW NNN,
/// `matches_clip_head`'s hardcoded `Operand(2)`/`Add(0,1)` literals
/// rejected this shape outright (`hits == 0` for every shape
/// `law1_clip_epilogue_fused_matches_unfused_bit_identical` tried,
/// since this permutation is a property of the PROGRAM STRUCTURE, not
/// the `m`/`k`/`n` shape values).
#[test]
fn match_epilogue_recognizes_clip_when_reduce_is_not_operand_zero() {
    let body = ComposedBody {
        steps: vec![
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Operand(1), StepArg::Operand(2)],
            },
            BodyStep {
                op: ScalarOp::Maximum,
                args: vec![StepArg::Step(0), StepArg::Operand(0)],
            },
        ],
    };
    assert_eq!(
        match_epilogue(&body, 2, None),
        Some((EpilogueKind::Clip, EpilogueSlots::Clip { bias: 1, zero: 0 }))
    );
}

/// `push_canonical_step` canonicalizes a commutative op's operand order
/// before minting a step — `a*b+c` and `c+a*b` compose to the identical
/// [`bind::BodyStep`] sequence, never two different `BoundOpKind`s for
/// the same algebraic value. Built through `bind`'s own program/bind
/// path (not a hand-built body) so this is a real authored-order test,
/// per the brief's "bind to the identical `BoundOpKind`" requirement.
#[test]
fn commutative_operand_order_is_canonical_regardless_of_authored_order() {
    let build = |c_plus_a_times_b: bool| -> ComposedBody {
        let mut program = Vec::new();
        let a = f32_block(&mut program, &[Extent::Static(4)]);
        let b = f32_block(&mut program, &[Extent::Static(4)]);
        let c = f32_block(&mut program, &[Extent::Static(4)]);
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let a_times_b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(a, identity()), (b, identity())],
                name: None,
            },
        );
        let operands = if c_plus_a_times_b {
            alloc::vec![(c, identity()), (a_times_b, identity())]
        } else {
            alloc::vec![(a_times_b, identity()), (c, identity())]
        };
        let root = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands,
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("shape inference succeeds");
        let resolved = bind::bind(
            &program,
            &shapes,
            &[terminal(&program)],
            NumericPolicy::bit_exact(),
        )
        .expect("bind succeeds");
        let BoundOpKind::Elementwise { body, .. } = &resolved
            .iter()
            .find(|op| op.node == root)
            .expect("root node bound")
            .kind
        else {
            panic!("root node did not bind to an Elementwise kind");
        };
        body.clone()
    };

    assert_eq!(
        build(true),
        build(false),
        "c+a*b and a*b+c must mint the identical BodyStep sequence"
    );
}

/// Bit-identity, per the brief's requirement that canonicalizing operand
/// order never changes a computed value: a real-shaped elementwise chain
/// (`a*b+c`, four-element vectors) authored two ways evaluates to
/// bit-identical `f32` output, not merely an equal `ComposedBody`.
#[test]
fn commutative_operand_order_preserves_bit_exact_output() {
    let build_and_evaluate = |c_plus_a_times_b: bool| -> Vec<f32> {
        let mut program = Vec::new();
        let a = f32_block(&mut program, &[Extent::Static(4)]);
        let b = f32_block(&mut program, &[Extent::Static(4)]);
        let c = f32_block(&mut program, &[Extent::Static(4)]);
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let a_times_b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(a, identity()), (b, identity())],
                name: None,
            },
        );
        let operands = if c_plus_a_times_b {
            alloc::vec![(c, identity()), (a_times_b, identity())]
        } else {
            alloc::vec![(a_times_b, identity()), (c, identity())]
        };
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands,
                name: None,
            },
        );

        let a_data = [1.0, 2.0, 3.0, 4.0f32];
        let b_data = [2.0, 0.5, -1.0, 3.0f32];
        let c_data = [1.0, 1.0, 1.0, 1.0f32];
        let evaluated = evaluate(&program, &[], &[&a_data, &b_data, &c_data], &[])
            .expect("elementwise chain evaluates");
        evaluated.root().to_vec()
    };

    assert_eq!(
        build_and_evaluate(true).as_slice(),
        build_and_evaluate(false).as_slice(),
        "reordering a*b+c's commutative Add operand must not change one bit of the result"
    );
}

fn packed_block<'a>(codec: &str, bytes: &'a [u8]) -> QuantizedBlock<'a> {
    match codec {
        "q4_k" => QuantizedBlock::Q4K(bytes),
        "q5_k" => QuantizedBlock::Q5K(bytes),
        "q6_k" => QuantizedBlock::Q6K(bytes),
        "q8_0" => QuantizedBlock::Q8_0(bytes),
        "q4_0" => QuantizedBlock::Q4_0(bytes),
        "q5_1" => QuantizedBlock::Q5_1(bytes),
        "iq4_nl" => QuantizedBlock::Iq4Nl(bytes),
        "iq2_xs" => QuantizedBlock::Iq2Xs(bytes),
        "iq3_xxs" => QuantizedBlock::Iq3Xxs(bytes),
        "float16" => QuantizedBlock::Float16(bytes),
        "bfloat16" => QuantizedBlock::BFloat16(bytes),
        other => panic!("packed_block: unknown test codec {other}"),
    }
}

/// One block-count times one codec's own `elements_for_blocks` is the
/// exact contract [`QuantizedBlock::element_count`] promises every
/// non-`Float32` variant -- this is the single owner
/// `omega::metal`/`omega::wgpu_driver` both now call instead of
/// restating the per-codec bytes-to-elements table themselves.
#[proxima::test]
#[case::q4_k_two_super_blocks("q4_k", q4_k::BLOCK_BYTES, q4_k::QK_K, 2)]
#[case::q5_k_three_super_blocks("q5_k", q5_k::BLOCK_BYTES, q5_k::QK_K, 3)]
#[case::q6_k_one_super_block("q6_k", q6_k::BLOCK_BYTES, q6_k::QK_K, 1)]
#[case::q8_0_four_blocks("q8_0", q8_0::BLOCK_BYTES, q8_0::QK8_0, 4)]
#[case::q4_0_five_blocks("q4_0", q4_0::BLOCK_BYTES, q4_0::QK4_0, 5)]
#[case::q5_1_three_blocks("q5_1", q5_1::BLOCK_BYTES, q5_1::QK5_1, 3)]
#[case::iq4_nl_five_blocks("iq4_nl", iq4_nl::BLOCK_BYTES, iq4_nl::QK4_NL, 5)]
#[case::iq2_xs_two_super_blocks("iq2_xs", iq2_xs::BLOCK_BYTES, iq2_xs::QK_K, 2)]
#[case::iq3_xxs_three_super_blocks("iq3_xxs", iq3_xxs::BLOCK_BYTES, iq3_xxs::QK_K, 3)]
#[case::float16_seven_elements("float16", gguf_f16::BLOCK_BYTES, 1, 7)]
#[case::bfloat16_two_elements("bfloat16", gguf_bf16::BLOCK_BYTES, 1, 2)]
async fn quantized_block_element_count_multiplies_block_count_by_elements_per_block(
    #[case] codec: &str,
    #[case] block_bytes: usize,
    #[case] elements_per_block: usize,
    #[case] block_count: usize,
) {
    let bytes = vec![0u8; block_bytes * block_count];
    let block = packed_block(codec, &bytes);

    let elements = block
        .element_count()
        .expect("a whole multiple of block_bytes never errors");

    assert_eq!(elements, elements_per_block * block_count);
}

#[proxima::test]
async fn quantized_block_element_count_reports_float32_slice_length_directly() {
    let data = [0.0f32; 5];
    let block = QuantizedBlock::Float32(&data);

    assert_eq!(block.element_count().expect("float32 never errors"), 5);
}

#[proxima::test]
#[case::q4_k("q4_k", q4_k::BLOCK_BYTES)]
#[case::q5_k("q5_k", q5_k::BLOCK_BYTES)]
#[case::q6_k("q6_k", q6_k::BLOCK_BYTES)]
#[case::q8_0("q8_0", q8_0::BLOCK_BYTES)]
#[case::q4_0("q4_0", q4_0::BLOCK_BYTES)]
#[case::float16("float16", gguf_f16::BLOCK_BYTES)]
#[case::bfloat16("bfloat16", gguf_bf16::BLOCK_BYTES)]
async fn quantized_block_element_count_rejects_a_byte_length_not_a_whole_block_multiple(
    #[case] codec: &'static str,
    #[case] block_bytes: usize,
) {
    let bytes = vec![0u8; block_bytes + 1];
    let block = packed_block(codec, &bytes);

    let error = block
        .element_count()
        .expect_err("one byte past a whole block is never a legal length");

    assert_eq!(
        error,
        TensorError::PackedBlockBytesNotAMultiple {
            codec,
            bytes: block_bytes + 1,
            block_bytes,
        }
    );
}

/// `stage_offsets` for `stage_count` stages of a UNIFORM `chunks_per_stage`
/// width -- the shape every pre-existing test used before `StagedRound`
/// grew variable-width stages; kept as a fixture so those tests read the
/// same as before, with the new field spelled out.
fn uniform_stage_offsets(stage_count: usize, chunks_per_stage: usize) -> Vec<usize> {
    (0..=stage_count)
        .map(|stage| stage * chunks_per_stage)
        .collect()
}

/// The property [`StagedRound`] exists for: a chunk in stage `s` never
/// observes stage `s - 1` incomplete. Driven from real threads against
/// the real `run_chunk`, with chunks handed out off a monotonic cursor
/// exactly the way `prime`'s cohort hands them out — that ordering is
/// the precondition the barrier's deadlock-freedom argument rests on, so
/// the test reproduces it rather than assuming it.
///
/// [`parse_cpu_list_count`] against the exact shapes
/// `/sys/devices/cpu_core/cpus` produces on a real hybrid Linux host --
/// this crate's dev boxes are aarch64-darwin, so the sysfs file itself
/// is unreachable here; this exercises the parser this Mac CAN run,
/// leaving the `std::fs::read_to_string` wiring in
/// [`performance_core_count`] compiled (`cargo check --target
/// x86_64-unknown-linux-gnu`) but unexecuted on this machine.
#[proxima::test]
#[case::single_range("0-7,16-23", Some(16))]
#[case::bare_id("4", Some(1))]
#[case::single_cpu_range("4-4", Some(1))]
#[case::trailing_newline("0-3\n", Some(4))]
#[case::empty_file("", None)]
#[case::whitespace_only("   \n", None)]
#[case::malformed_range("abc", None)]
async fn parse_cpu_list_count_matches_sysfs_shapes(
    #[case] text: &str,
    #[case] expected: Option<usize>,
) {
    assert_eq!(parse_cpu_list_count(text), expected);
}

/// Each stage's chunk asserts every earlier stage is fully published,
/// then publishes its own slot. A missing barrier shows up as a stage
/// reading a slot its predecessor had not written yet.
#[test]
fn staged_round_never_runs_a_stage_before_its_predecessor_completes() {
    const STAGES: usize = 6;
    const CHUNKS: usize = 4;
    const MEMBERS: usize = 3;

    let stage_offsets = uniform_stage_offsets(STAGES, CHUNKS);
    let completed: Vec<AtomicUsize> = (0..STAGES).map(|_| AtomicUsize::new(0)).collect();
    let published: Vec<AtomicUsize> = (0..STAGES * CHUNKS).map(|_| AtomicUsize::new(0)).collect();
    let violations = AtomicUsize::new(0);

    let round = StagedRound {
        stage_offsets: &stage_offsets,
        completed: &completed,
        run_stage_chunk: |stage: usize, within: usize| {
            for earlier in 0..stage {
                for slot in 0..CHUNKS {
                    if published[earlier * CHUNKS + slot].load(Ordering::Acquire) != 1 {
                        violations.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            published[stage * CHUNKS + within].store(1, Ordering::Release);
            Ok(())
        },
    };

    let cursor = AtomicUsize::new(0);
    let total = round.chunks();
    std::thread::scope(|scope| {
        for _ in 0..MEMBERS {
            scope.spawn(|| {
                loop {
                    let claimed = cursor.fetch_add(1, Ordering::Relaxed);
                    if claimed >= total {
                        break;
                    }
                    round
                        .run_chunk(ChunkIndex(claimed))
                        .expect("staged chunk must not fail");
                }
            });
        }
    });

    assert_eq!(
        violations.load(Ordering::Relaxed),
        0,
        "a stage ran before its predecessor completed"
    );
    assert_eq!(
        total,
        STAGES * CHUNKS,
        "flat chunk space must cover every stage"
    );
    for (index, slot) in published.iter().enumerate() {
        assert_eq!(slot.load(Ordering::Relaxed), 1, "chunk {index} never ran");
    }
}

/// Fewer members than stages is the case the deadlock-freedom argument
/// has to cover: a member that claims a late stage waits on chunks whose
/// owners may themselves be waiting. One member is the extreme.
#[test]
fn staged_round_completes_with_a_single_member() {
    const STAGES: usize = 5;
    const CHUNKS: usize = 2;

    let stage_offsets = uniform_stage_offsets(STAGES, CHUNKS);
    let completed: Vec<AtomicUsize> = (0..STAGES).map(|_| AtomicUsize::new(0)).collect();
    let ran = AtomicUsize::new(0);
    let round = StagedRound {
        stage_offsets: &stage_offsets,
        completed: &completed,
        run_stage_chunk: |_stage: usize, _within: usize| {
            ran.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    };
    for chunk in 0..round.chunks() {
        round
            .run_chunk(ChunkIndex(chunk))
            .expect("staged chunk must not fail");
    }
    assert_eq!(ran.load(Ordering::Relaxed), STAGES * CHUNKS);
}

/// A failing chunk must still publish, or every member behind it hangs
/// on the barrier instead of seeing the error through the round's report.
#[test]
fn staged_round_publishes_a_failed_chunk_so_later_stages_do_not_hang() {
    const CHUNKS: usize = 2;
    let stage_offsets = uniform_stage_offsets(2, CHUNKS);
    let completed: Vec<AtomicUsize> = (0..2).map(|_| AtomicUsize::new(0)).collect();
    let round = StagedRound {
        stage_offsets: &stage_offsets,
        completed: &completed,
        run_stage_chunk: |stage: usize, _within: usize| {
            if stage == 0 {
                Err(TensorError::NotLowerable {
                    node: NodeId(0),
                    reason: "staged round error propagation fixture",
                })
            } else {
                Ok(())
            }
        },
    };
    assert!(round.run_chunk(ChunkIndex(0)).is_err());
    assert!(round.run_chunk(ChunkIndex(1)).is_err());
    assert_eq!(
        completed[0].load(Ordering::Relaxed),
        CHUNKS,
        "a failed chunk must still publish"
    );
    assert!(
        round.run_chunk(ChunkIndex(2)).is_ok(),
        "stage 1 must not be blocked by stage 0's error"
    );
}

/// The property the whole matmul-fold design depends on
/// (`docs/discipline.md` ROW 96's "what remains open"): a wide
/// matmul-shaped stage (many chunks, real cross-worker parallelism) and
/// a narrow elementwise-shaped stage (exactly one chunk) coexisting in
/// the SAME round, neither one forced to match the other's width. Stage
/// widths here are deliberately irregular (3, 1, 5, 1, 2) rather than a
/// clean power of two, so a bug that only reproduces at a stage
/// boundary math edge (off-by-one in `partition_point`'s translation
/// back to `within_stage`) has somewhere to show up.
#[test]
fn staged_round_supports_variable_width_stages_in_one_round() {
    const WIDTHS: [usize; 5] = [3, 1, 5, 1, 2];
    let stage_offsets: Vec<usize> = core::iter::once(0)
        .chain(WIDTHS.iter().scan(0usize, |total, width| {
            *total += width;
            Some(*total)
        }))
        .collect();
    let completed: Vec<AtomicUsize> = (0..WIDTHS.len()).map(|_| AtomicUsize::new(0)).collect();
    let observed_widths: Vec<AtomicUsize> =
        (0..WIDTHS.len()).map(|_| AtomicUsize::new(0)).collect();
    let violations = AtomicUsize::new(0);

    let round = StagedRound {
        stage_offsets: &stage_offsets,
        completed: &completed,
        run_stage_chunk: |stage: usize, within: usize| {
            if within >= WIDTHS[stage] {
                violations.fetch_add(1, Ordering::Relaxed);
            }
            observed_widths[stage].fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    };

    assert_eq!(
        round.chunks(),
        WIDTHS.iter().sum::<usize>(),
        "flat chunk space must cover every stage's own width"
    );
    for chunk in 0..round.chunks() {
        round
            .run_chunk(ChunkIndex(chunk))
            .expect("staged chunk must not fail");
    }
    assert_eq!(
        violations.load(Ordering::Relaxed),
        0,
        "a chunk landed in the wrong stage or read the wrong within-stage index"
    );
    for (stage, width) in WIDTHS.iter().enumerate() {
        assert_eq!(
            observed_widths[stage].load(Ordering::Relaxed),
            *width,
            "stage {stage} did not run exactly its own width in chunks"
        );
    }
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// The scalar reference [`run_elementwise`]'s own `Generic` arm takes:
/// per position, read each operand via its own running offset (advancing
/// by `strides` each step, mirroring [`fill_running_offsets`] plus the
/// per-element loop body), then [`apply_body`]. No gather in any of
/// these fixtures, so this omits [`GatherCursor`] entirely.
fn reference_generic_body(
    body: &ComposedBody,
    raw: &[&[f32]],
    strides: &[i64],
    width: usize,
) -> Vec<f32> {
    let mut running = vec![0i64; raw.len()];
    let mut operand_values = vec![0.0f32; raw.len()];
    let mut step_values = vec![0.0f32; body.steps.len()];
    let mut out = Vec::with_capacity(width);
    for _ in 0..width {
        for (index, data) in raw.iter().enumerate() {
            operand_values[index] = data[running[index] as usize];
            running[index] += strides[index];
        }
        out.push(apply_body(body, &operand_values, &mut step_values));
    }
    out
}

fn step(op: ScalarOp, args: &[StepArg]) -> BodyStep {
    BodyStep {
        op,
        args: args.to_vec(),
    }
}

fn bound_f32_identity(lookup: Option<bind::Lookup>) -> BoundOp {
    BoundOp {
        node: NodeId(1),
        dtype: DType::Float32,
        extents: vec![3],
        kind: BoundOpKind::Elementwise {
            body: ComposedBody::leaf(ScalarOp::Identity),
            operands: vec![(
                NodeId(0),
                bind::Layout {
                    base: 0,
                    strides: smallvec::smallvec![1],
                },
                lookup,
            )],
        },
    }
}

#[test]
fn evaluate_bound_f32_into_runs_only_the_supplied_dense_bound_op() {
    let input = [1.25f32, -2.5, 4.0];
    let buffers = [Some(input.as_slice()), None];
    let mut output = [0.0f32; 3];

    evaluate_bound_f32_into(&bound_f32_identity(None), &buffers, &[], &mut output)
        .expect("dense bound identity evaluates from the supplied snapshot");

    assert_eq!(output, input);
}

#[test]
fn evaluate_bound_f32_into_rejects_packed_and_gathered_operands() {
    let input = [1.25f32, -2.5, 4.0];
    let buffers = [Some(input.as_slice()), None];
    let mut output = [0.0f32; 3];
    let packed_error = evaluate_bound_f32_into(
        &bound_f32_identity(None),
        &buffers,
        &[NodeId(0)],
        &mut output,
    )
    .expect_err("a packed operand is not a dense f32 snapshot");
    assert!(matches!(
        packed_error,
        TensorError::BoundF32PackedOperand {
            node: NodeId(1),
            operand: NodeId(0),
        }
    ));

    let lookup = bind::Lookup {
        indices: NodeId(2),
        index_layout: bind::Layout {
            base: 0,
            strides: smallvec::smallvec![1],
        },
        element_stride: 1,
        extent: 3,
    };
    let gathered_error = evaluate_bound_f32_into(
        &bound_f32_identity(Some(lookup)),
        &buffers,
        &[],
        &mut output,
    )
    .expect_err("a gathered operand needs its index payload and is rejected");
    assert!(matches!(
        gathered_error,
        TensorError::BoundF32GatherOperand {
            node: NodeId(1),
            operand: NodeId(0),
        }
    ));
}

/// A synthetic instance of the 6-step SwiGLU chain named in
/// `proxima-tensor/docs/discipline.md` ROW 5
/// (`[Negate, Exponential, Add, Reciprocal, Multiply, Multiply]`):
/// `silu(gate) * up` computed as `gate * sigmoid(gate) * up`, with
/// `sigmoid(gate) = 1 / (1 + exp(-gate))` unrolled into the same five
/// scalar steps a real fused body has. Operand 0 is `gate` (contiguous),
/// operand 1 is a broadcast `1.0` (stride 0, standing in for a
/// constant), operand 2 is `up` (contiguous) — exactly the affine
/// precondition [`generic_body_is_affine_fast_path`] checks.
fn swiglu_body() -> ComposedBody {
    ComposedBody {
        steps: vec![
            step(ScalarOp::Negate, &[StepArg::Operand(0)]),
            step(ScalarOp::Exponential, &[StepArg::Step(0)]),
            step(ScalarOp::Add, &[StepArg::Step(1), StepArg::Operand(1)]),
            step(ScalarOp::Reciprocal, &[StepArg::Step(2)]),
            step(ScalarOp::Multiply, &[StepArg::Step(3), StepArg::Operand(0)]),
            step(ScalarOp::Multiply, &[StepArg::Step(4), StepArg::Operand(2)]),
        ],
    }
}

/// A synthetic instance of the 6-step RMSNorm chain named in
/// `proxima-tensor/docs/discipline.md` ROW 5
/// (`[Multiply, Add, SquareRoot, Reciprocal, Multiply, Multiply]`):
/// `x * (1 / sqrt(x*x + eps)) * weight`. Operand 0 is `x` (contiguous),
/// operand 1 is a broadcast `eps` (stride 0), operand 2 is `weight`
/// (contiguous).
fn rmsnorm_body() -> ComposedBody {
    ComposedBody {
        steps: vec![
            step(
                ScalarOp::Multiply,
                &[StepArg::Operand(0), StepArg::Operand(0)],
            ),
            step(ScalarOp::Add, &[StepArg::Step(0), StepArg::Operand(1)]),
            step(ScalarOp::SquareRoot, &[StepArg::Step(1)]),
            step(ScalarOp::Reciprocal, &[StepArg::Step(2)]),
            step(ScalarOp::Multiply, &[StepArg::Operand(0), StepArg::Step(3)]),
            step(ScalarOp::Multiply, &[StepArg::Step(4), StepArg::Operand(2)]),
        ],
    }
}

/// A synthetic instance of the 3-step RoPE chains named in
/// `proxima-tensor/docs/discipline.md` ROW 5
/// (`[Multiply, Multiply, Add]` / `[Multiply, Multiply, Subtract]`):
/// `x * cos <op> y * sin`. All four operands (`x`, `cos`, `y`, `sin`)
/// are contiguous.
fn rope_body(combine: ScalarOp) -> ComposedBody {
    ComposedBody {
        steps: vec![
            step(
                ScalarOp::Multiply,
                &[StepArg::Operand(0), StepArg::Operand(1)],
            ),
            step(
                ScalarOp::Multiply,
                &[StepArg::Operand(2), StepArg::Operand(3)],
            ),
            step(combine, &[StepArg::Step(0), StepArg::Step(1)]),
        ],
    }
}

#[proxima::test]
#[case::swiglu(swiglu_body(), vec![1, 0, 1])]
#[case::rmsnorm(rmsnorm_body(), vec![1, 0, 1])]
#[case::rope_add(rope_body(ScalarOp::Add), vec![1, 1, 1, 1])]
#[case::rope_subtract(rope_body(ScalarOp::Subtract), vec![1, 1, 1, 1])]
async fn elementwise_width_generic_matches_scalar_apply_body(
    #[case] body: ComposedBody,
    #[case] strides: Vec<i64>,
) {
    let width = 32;
    let operand_len = |stride: i64| if stride == 0 { 1 } else { width };
    let buffers: Vec<Vec<f32>> = strides
        .iter()
        .enumerate()
        .map(|(index, &stride)| {
            let values = random_vec(0x5eed_0000 + index as u64, operand_len(stride));
            // keep divisor-side and sqrt-side operands away from zero so
            // Reciprocal/SquareRoot stay finite and comparable exactly
            values.into_iter().map(|value| value.abs() + 0.25).collect()
        })
        .collect();
    let raw: Vec<&[f32]> = buffers.iter().map(Vec::as_slice).collect();

    let expected = reference_generic_body(&body, &raw, &strides, width);

    let running = vec![0i64; raw.len()];
    let mut step_values = vec![0.0f32; body.steps.len() * width];
    let mut actual = vec![0.0f32; width];
    elementwise_width_generic(
        &body,
        &raw,
        &running,
        &strides,
        &mut actual,
        &mut step_values,
    );

    assert_eq!(
        actual, expected,
        "width-fast generic path must be bit-identical to the scalar apply_body path"
    );
}

/// The exact shape `specs/rope.toml` maps against operand `x`
/// (`"s,2*i->si"` / `"s,2*i+1->si"`): two views of the SAME underlying
/// buffer, one reading even positions (stride 2, base 0), one reading odd
/// positions (stride 2, base 1) — `x * cos + y * sin` with `x`/`y` both
/// drawn from one physical buffer via [`OperandSpan::at`]. The reference
/// pre-slices each view into its own contiguous buffer instead of relying
/// on `reference_generic_body`'s hardcoded zero starting offset, so the
/// two calls read identical values through different addressing.
#[test]
fn elementwise_width_generic_matches_scalar_apply_body_for_a_rope_shaped_stride_two_operand() {
    let width = 16;
    let x: Vec<f32> = random_vec(0x50fe_0000, 2 * width)
        .into_iter()
        .map(|value| value.abs() + 0.25)
        .collect();
    let cos = random_vec(0x50fe_0001, width);
    let sin = random_vec(0x50fe_0002, width);

    let body = rope_body(ScalarOp::Add);
    let strides = vec![2i64, 1, 2, 1];
    let raw: Vec<&[f32]> = vec![x.as_slice(), cos.as_slice(), x.as_slice(), sin.as_slice()];
    let running = vec![0i64, 0, 1, 0];

    let x_even: Vec<f32> = (0..width).map(|position| x[2 * position]).collect();
    let x_odd: Vec<f32> = (0..width).map(|position| x[2 * position + 1]).collect();
    let reference_raw: Vec<&[f32]> = vec![
        x_even.as_slice(),
        cos.as_slice(),
        x_odd.as_slice(),
        sin.as_slice(),
    ];
    let reference_strides = vec![1i64, 1, 1, 1];
    let expected = reference_generic_body(&body, &reference_raw, &reference_strides, width);

    let mut step_values = vec![0.0f32; body.steps.len() * width];
    let mut actual = vec![0.0f32; width];
    elementwise_width_generic(
        &body,
        &raw,
        &running,
        &strides,
        &mut actual,
        &mut step_values,
    );

    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "width-fast generic path must be bit-identical to the scalar apply_body path for a stride-2 operand"
    );
}

/// [`reduce_dot_binary_monomorphic_strided`]'s load-bearing invariant:
/// a strided dot fold must combine in the same strict left-to-right order
/// [`Iterator::sum`] does, never reassociated the way the contiguous
/// `DOT_LANES` path is. `x` is read at stride 2 (`x[2*k]`), `y` at stride
/// 1 — the same shape a RoPE-adjacent contraction over an interleaved
/// buffer would take.
#[test]
fn reduce_dot_binary_stride_two_matches_a_scalar_reference() {
    let k = 6usize;
    let mut program = Vec::new();
    let x = f32_block(&mut program, &[Extent::Static((2 * k) as u32)]);
    let y = f32_block(&mut program, &[Extent::Static(k as u32)]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (
                    x,
                    IndexMap::Affine(map::affine(1, &[(&[AxisTerm::scaled(0, 2)], 0)]))
                ),
                (y, IndexMap::Affine(map::projection(1, &[0]))),
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
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let x_data = random_vec(0x51de_0000, 2 * k);
    let y_data = random_vec(0x51de_0001, k);
    let evaluated =
        evaluate(&program, &[], &[&x_data, &y_data], &[]).expect("stride-2 dot reduce evaluates");

    let expected: f32 = (0..k)
        .map(|position| x_data[2 * position] * y_data[position])
        .sum();
    assert_eq!(
        evaluated.root()[0].to_bits(),
        expected.to_bits(),
        "fast path must be bit-identical to the scalar reference for a stride-2 operand"
    );
}

/// [`scan_width_unary_monomorphic_strided`]'s equivalent invariant: a
/// running sum over a stride-2 read must land on the exact same running
/// total, position by position, as the plain scalar accumulation below.
/// Built as a direct [`BoundOp`] (mirroring [`bound_op_for_gate`], for a
/// `Reduce`/`Scan` shape instead of an `Elementwise` one) and run through
/// [`run_scan`] directly, rather than through [`evaluate`]'s shape
/// inference — a single scaled operand with no plain-projection sibling
/// leaves iteration axis 0's extent unconstrained for inference to solve.
#[test]
fn cached_attention_bound_step_runs_online_softmax() {
    let mut buffers = vec![None; 9];
    let inputs = [
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![2.0, 3.0],
        vec![4.0, 5.0],
    ];
    for (index, input) in inputs.iter().enumerate() {
        buffers[index] = Some(input.as_slice());
    }
    let operands = inputs
        .iter()
        .enumerate()
        .map(|(index, _)| {
            (
                NodeId(index as u32),
                bind::Layout {
                    base: 0,
                    strides: smallvec::smallvec![1],
                },
                None,
            )
        })
        .collect();
    let resolved = BoundOp {
        node: NodeId(8),
        dtype: DType::Float32,
        extents: vec![1, 1, 1, 2],
        kind: BoundOpKind::CachedAttention {
            operands,
            query_rows: 1,
            cached_key_rows: 1,
            new_key_rows: 1,
            kv_heads: 1,
            query_groups: 1,
            head_dim: 2,
            rotary_dim: 2,
            scale: 1.0,
            cached_lower_inclusive: -1,
            new_upper_inclusive: 0,
        },
    };
    let mut output = vec![0.0; 2];
    run_node_into(&resolved, &buffers, None, None, None, false, &mut output)
        .expect("cached attention bound step runs");
    let cached_weight = 1.0f32.exp() / (1.0f32.exp() + 1.0);
    assert!((output[0] - (2.0 * cached_weight + 4.0 * (1.0 - cached_weight))).abs() < 1e-6);
    assert!((output[1] - (3.0 * cached_weight + 5.0 * (1.0 - cached_weight))).abs() < 1e-6);
}

/// [`cached_attention_bound_step_runs_online_softmax`]'s counterpart for
/// `rotary_dim < head_dim` (qwen35's partial-rotary shape,
/// `BoundOpKind::CachedAttention`'s own doc, ROW 556/557
/// `docs/discipline.md`): `head_dim` 4, `rotary_dim` 2, so the trailing
/// three-operand pass plane scores an extra dot product alongside the
/// even/odd rotary planes. Expected weights are hand-computed the same
/// online-softmax way this file's own reference test already does.
#[test]
fn cached_attention_bound_step_scores_the_partial_rotary_pass_plane() {
    let mut buffers = vec![None; 11];
    let inputs = [
        vec![1.0],                  // query_even
        vec![0.0],                  // query_odd
        vec![1.0],                  // cached_key_even
        vec![0.0],                  // cached_key_odd
        vec![0.0],                  // new_key_even
        vec![1.0],                  // new_key_odd
        vec![2.0, 3.0, 10.0, 11.0], // cached_value
        vec![4.0, 5.0, 12.0, 13.0], // new_value
        vec![1.0, 0.0],             // pass_query
        vec![1.0, 0.0],             // pass_cached_key
        vec![0.0, 1.0],             // pass_new_key
    ];
    for (index, input) in inputs.iter().enumerate() {
        buffers[index] = Some(input.as_slice());
    }
    let operands = inputs
        .iter()
        .enumerate()
        .map(|(index, _)| {
            (
                NodeId(index as u32),
                bind::Layout {
                    base: 0,
                    strides: smallvec::smallvec![1],
                },
                None,
            )
        })
        .collect();
    let resolved = BoundOp {
        node: NodeId(10),
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
            rotary_dim: 2,
            scale: 1.0,
            cached_lower_inclusive: -1,
            new_upper_inclusive: 0,
        },
    };
    let mut output = vec![0.0; 4];
    run_node_into(&resolved, &buffers, None, None, None, false, &mut output)
        .expect("partial-rotary cached attention bound step runs");
    // score_cached = (1*1 + 0*0) + (1*1 + 0*0) = 2; score_new = (1*0 + 0*1) + (1*0 + 0*1) = 0.
    let cached_weight = 1.0 / (1.0 + (-2.0f32).exp());
    let new_weight = 1.0 - cached_weight;
    for dimension in 0..4 {
        let expected = cached_weight * inputs[6][dimension] + new_weight * inputs[7][dimension];
        assert!(
            (output[dimension] - expected).abs() < 1e-6,
            "dimension {dimension}: got {}, expected {expected}",
            output[dimension]
        );
    }
}

#[test]
fn scan_width_unary_stride_two_matches_a_running_sum_reference() {
    let k = 6usize;
    let data = random_vec(0x5ca4_0000, 2 * k);

    let resolved = BoundOp {
        node: NodeId(1),
        dtype: DType::Float32,
        extents: alloc::vec![k as u64],
        kind: BoundOpKind::Reduce {
            element_body: ComposedBody {
                steps: alloc::vec![step(ScalarOp::Identity, &[StepArg::Operand(0)])],
            },
            reduce_op: ScalarOp::Add,
            init: ReduceInit::Zero,
            keep: Keep::Scan,
            operands: alloc::vec![(
                NodeId(0),
                bind::Layout {
                    base: 0,
                    strides: smallvec::smallvec![2],
                },
                None,
            )],
            output_axes: smallvec::smallvec![0],
            out_layout: bind::Layout {
                base: 0,
                strides: smallvec::smallvec![1],
            },
            out_scatter: None,
            epilogue_body: ComposedBody {
                steps: alloc::vec![step(ScalarOp::Identity, &[StepArg::Operand(0)])],
            },
            epilogue_operands: Vec::new(),
            epilogue_broadcast_axes: smallvec::smallvec![],
        },
    };

    let mut actual = vec![0.0f32; k];
    run_scan(&resolved, &[Some(data.as_slice())], &mut actual).expect("stride-2 cumsum runs");

    let mut running = 0.0f32;
    let reference: Vec<f32> = (0..k)
        .map(|position| {
            running += data[2 * position];
            running
        })
        .collect();

    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        reference
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "fast path must be bit-identical to the scalar reference for a stride-2 operand"
    );
}

/// A minimal [`BoundOp`] carrying `body` and one operand per stride in
/// `strides` — `Layout`/`node` are placeholders `operand_is_affine`
/// never reads, only `strides` (passed separately, matching
/// `run_elementwise`'s own precomputed table) and `gather` matter.
fn bound_op_for_gate(
    body: ComposedBody,
    strides: &[i64],
    gather_operand: Option<usize>,
) -> BoundOp {
    let operands = strides
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let gather = (Some(index) == gather_operand).then(|| bind::Lookup {
                indices: NodeId(0),
                index_layout: bind::Layout {
                    base: 0,
                    strides: smallvec::smallvec![0],
                },
                element_stride: 1,
                extent: 1,
            });
            (
                NodeId(index as u32),
                bind::Layout {
                    base: 0,
                    strides: smallvec::smallvec![0],
                },
                gather,
            )
        })
        .collect();
    BoundOp {
        node: NodeId(strides.len() as u32),
        dtype: DType::Float32,
        extents: vec![32],
        kind: BoundOpKind::Elementwise { body, operands },
    }
}

#[proxima::test]
#[case::swiglu(swiglu_body(), vec![1, 0, 1])]
#[case::rmsnorm(rmsnorm_body(), vec![1, 0, 1])]
#[case::rope_add(rope_body(ScalarOp::Add), vec![1, 1, 1, 1])]
async fn generic_body_is_affine_fast_path_accepts_gather_free_affine_operands(
    #[case] body: ComposedBody,
    #[case] strides: Vec<i64>,
) {
    let resolved = bound_op_for_gate(body.clone(), &strides, None);
    assert!(generic_body_is_affine_fast_path(&resolved, &body, &strides));
}

#[proxima::test]
#[case::swiglu(swiglu_body(), vec![1, 0, 1])]
#[case::rmsnorm(rmsnorm_body(), vec![1, 0, 1])]
async fn generic_body_is_affine_fast_path_rejects_a_gathered_operand(
    #[case] body: ComposedBody,
    #[case] strides: Vec<i64>,
) {
    let resolved = bound_op_for_gate(body.clone(), &strides, Some(0));
    assert!(!generic_body_is_affine_fast_path(
        &resolved, &body, &strides
    ));
}

#[proxima::test]
#[case::swiglu(swiglu_body(), vec![2, 0, 1])]
#[case::rmsnorm(rmsnorm_body(), vec![1, 0, 2])]
async fn generic_body_is_affine_fast_path_accepts_a_non_negative_constant_stride(
    #[case] body: ComposedBody,
    #[case] strides: Vec<i64>,
) {
    let resolved = bound_op_for_gate(body.clone(), &strides, None);
    assert!(generic_body_is_affine_fast_path(&resolved, &body, &strides));
}

#[proxima::test]
#[case::swiglu(swiglu_body(), vec![-1, 0, 1])]
#[case::rmsnorm(rmsnorm_body(), vec![1, 0, -1])]
async fn generic_body_is_affine_fast_path_rejects_a_negative_stride(
    #[case] body: ComposedBody,
    #[case] strides: Vec<i64>,
) {
    let resolved = bound_op_for_gate(body.clone(), &strides, None);
    assert!(!generic_body_is_affine_fast_path(
        &resolved, &body, &strides
    ));
}

/// Dispatches `rows` through the same `claim_and_run_rows` shared-cursor
/// mechanism [`matmul_rows_threaded`] uses, but with `oversubscribe`
/// passed in instead of hard-coded to [`crate::sized::ROW_OVERSUBSCRIBE`]
/// — lets [`bench_row_oversubscribe_picks_the_multiplier`] sweep the
/// multiplier without a rebuild per value. Test-only duplication of
/// `matmul_rows_threaded`'s body; not shipped (`#[cfg(test)]`).
fn dispatch_rows_with_oversubscribe<Row>(
    rows: usize,
    workers: usize,
    oversubscribe: usize,
    dot_row: Row,
) -> Vec<f32>
where
    Row: Fn(usize) -> Result<f32, TensorError> + Sync,
{
    // adapts the scalar `Row` this test builds (matches every real
    // caller's own dot-product closure shape) into
    // `claim_and_run_rows`'s `width`-slot form with `width == 1`, then
    // hands off to a generic-over-the-adapted-closure-type inner
    // dispatcher so the unsafe pointer cast below can name that type via
    // its own fresh generic parameter (a bare closure has no nameable
    // type to turbofish with).
    dispatch_rows_widened(rows, workers, oversubscribe, 1, move |row, slot| {
        slot[0] = dot_row(row)?;
        Ok(())
    })
}

fn dispatch_rows_widened<Wide>(
    rows: usize,
    workers: usize,
    oversubscribe: usize,
    width: usize,
    dot_row: Wide,
) -> Vec<f32>
where
    Wide: Fn(usize, &mut [f32]) -> Result<(), TensorError> + Sync,
{
    let mut output = vec![0.0f32; rows * width];
    let chunk_count = (workers.saturating_mul(oversubscribe)).clamp(1, rows.max(1));
    let chunk_len = rows.div_ceil(chunk_count);

    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = output.as_mut_slice();
    let mut row_start = 0usize;
    while !remaining.is_empty() {
        let take_rows = chunk_len.min(remaining.len() / width);
        let (slice, rest) = remaining.split_at_mut(take_rows * width);
        remaining = rest;
        chunk_ranges.push((row_start, slice.as_mut_ptr() as usize, slice.len()));
        row_start += take_rows;
    }
    let chunk_ranges_len = chunk_ranges.len();

    let pool = nest_pool().expect("pool builds under test");
    let dot_row_address = &dot_row as *const Wide as usize;
    let next_index = Arc::new(AtomicUsize::new(0));
    let chunk_ranges: Arc<Vec<(usize, usize, usize)>> = Arc::new(chunk_ranges);
    let spawned_count = workers
        .saturating_sub(1)
        .min(chunk_ranges_len.saturating_sub(1));
    let (result_sender, result_receiver) = sync_channel(chunk_ranges_len);

    for _ in 0..spawned_count {
        let sender = result_sender.clone();
        let next_index = Arc::clone(&next_index);
        let chunk_ranges = Arc::clone(&chunk_ranges);
        drop(pool.spawn(move || {
            claim_and_run_rows::<Wide>(&next_index, dot_row_address, width, &chunk_ranges, &sender);
            Ok::<(), _>(())
        }));
    }
    claim_and_run_rows::<Wide>(
        &next_index,
        dot_row_address,
        width,
        &chunk_ranges,
        &result_sender,
    );
    drop(result_sender);

    for _ in 0..chunk_ranges_len {
        let _ = result_receiver.recv();
    }
    output
}

/// Manual microbench picking [`crate::sized::ROW_OVERSUBSCRIBE`] —
/// principle 18/19: a design constant needs a measurement artifact, not
/// reasoning. Synthetic per-row cost is deliberately imbalanced (the
/// last 1/8 of rows costs ~8x a normal row, echoing the 2.04x
/// equal-row-count spread [`OVERSUBSCRIBE`]'s own doc records for a real
/// GEMM) so a static 1:1 split leaves the calling thread idling in
/// `Receiver::recv` for whichever puller drew the straggler range.
/// `#[ignore]`: manual, not part of the CI gate — run with
/// `cargo test -p proxima-tensor --release bench_row_oversubscribe -- --ignored --nocapture`.
#[test]
#[ignore = "manual microbench, not a CI gate; see this test's own doc"]
fn bench_row_oversubscribe_picks_the_multiplier() {
    let workers = thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(1);
    let rows = 4096usize;
    let straggler_start = rows - rows / 8;
    let cost_of = |row: usize| -> u64 { if row >= straggler_start { 4000 } else { 500 } };
    let dot_row = |row: usize| -> Result<f32, TensorError> {
        let mut accumulator = 0.0f32;
        for iteration in 0..cost_of(row) {
            accumulator += (iteration as f32).sin();
        }
        Ok(accumulator)
    };

    let load = std::process::Command::new("uptime")
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime unavailable".to_string());
    eprintln!("workers={workers} rows={rows} ambient_load={load}");

    eprintln!("-- imbalanced (straggler last 1/8 of rows) --");
    for oversubscribe in [1usize, 2, 4, 8, 16, 32] {
        let mut samples_micros = Vec::with_capacity(5);
        for _ in 0..5 {
            let started = Instant::now();
            let output = dispatch_rows_with_oversubscribe(rows, workers, oversubscribe, dot_row);
            let elapsed = started.elapsed().as_micros() as f64;
            assert_eq!(output.len(), rows);
            samples_micros.push(elapsed);
        }
        let mean = samples_micros.iter().sum::<f64>() / samples_micros.len() as f64;
        let variance = samples_micros
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples_micros.len() as f64;
        let coefficient_of_variation = variance.sqrt() / mean;
        eprintln!(
            "oversubscribe={oversubscribe} mean_us={mean:.1} cov={coefficient_of_variation:.4} samples={samples_micros:?}"
        );
    }

    // degenerate control (principle 19/V5): a UNIFORM per-row cost has
    // nothing to steal around, so this arm isolates the atomic-cursor
    // and `SyncSender` overhead oversubscription adds without any
    // imbalance to pay for it. If a high multiplier regresses here, that
    // is the ceiling on how far oversubscription can be pushed once real
    // rows are small enough for per-chunk overhead to matter.
    let uniform_dot_row = |row: usize| -> Result<f32, TensorError> {
        let mut accumulator = 0.0f32;
        for iteration in 0..1200u64 {
            accumulator += ((row as f32) + iteration as f32).sin();
        }
        Ok(accumulator)
    };
    eprintln!("-- uniform cost (degenerate control) --");
    for oversubscribe in [1usize, 2, 4, 8, 16, 32] {
        let mut samples_micros = Vec::with_capacity(5);
        for _ in 0..5 {
            let started = Instant::now();
            let output =
                dispatch_rows_with_oversubscribe(rows, workers, oversubscribe, uniform_dot_row);
            let elapsed = started.elapsed().as_micros() as f64;
            assert_eq!(output.len(), rows);
            samples_micros.push(elapsed);
        }
        let mean = samples_micros.iter().sum::<f64>() / samples_micros.len() as f64;
        let variance = samples_micros
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples_micros.len() as f64;
        let coefficient_of_variation = variance.sqrt() / mean;
        eprintln!(
            "oversubscribe={oversubscribe} mean_us={mean:.1} cov={coefficient_of_variation:.4} samples={samples_micros:?}"
        );
    }
}

/// KILL-EARLY nano (attention-tile task, 2026-09-01, binding on ROW
/// 198/199): the untiled scalar path vs the NEON width tile widened by
/// this task's `outer_extent` field, vs one `cblas_sgemm` call per head
/// (`try_run_accelerate_sgemm`), at BGE's own real deployed attention
/// shapes -- `Q@K^T` (`K=32`, `N=seq_len`) and `softmax@V`
/// (`K=seq_len`, `N=32`), `seq_len` in `{7, 8, 9}`, `heads=12`. `Q@K^T`'s
/// own `N` is always `< WIDTH_TILE_VECS * 4 == 16` for every `seq_len`
/// in this set, so `width_tile_plan` declines it with `NarrowWidth`
/// regardless of this task's own fix (verified separately by the
/// `bge_route_census` example) -- its NEON number here is `N/A`, never
/// routed. `#[ignore]`: manual, not part of the CI gate -- run with
/// `cargo test -p proxima-tensor --release attention_tile_shapes_nano
/// -- --ignored --nocapture`.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore = "manual nano bench, not a CI gate; see this test's own doc"]
fn attention_tile_shapes_nano() {
    const HEADS: usize = 12;
    const REPEATS: usize = 7;

    fn scalar_reference(
        a: &[f32],
        b_kn: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        for head in 0..heads {
            let a_head = &a[head * m * k..(head + 1) * m * k];
            let b_head = &b_kn[head * k * n..(head + 1) * k * n];
            let out_head = &mut out[head * m * n..(head + 1) * m * n];
            for row in 0..m {
                for col in 0..n {
                    let value = width_tile_scalar_cell(
                        KStridedTile {
                            data: a_head,
                            base: (row * k) as i64,
                            k_stride: 1,
                        },
                        KStridedTile {
                            data: b_head,
                            base: col as i64,
                            k_stride: n as i64,
                        },
                        k,
                        0.0,
                    );
                    out_head[row * n + col] = value;
                }
            }
        }
        out
    }

    fn neon_route<const VECS: usize>(
        a: &[f32],
        b_kn: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        let raw: [&[f32]; 2] = [a, b_kn];
        let plan = WidthTilePlan {
            a_operand: 0,
            b_operand: 1,
            row_stride_a: k as i64,
            base_a: 0,
            k_stride_a: 1,
            base_b: 0,
            k_stride_b: n as i64,
            out_base: 0,
            out_row_stride: n as i64,
            out_col_stride: 1,
            leading_total: m,
            reduction_total: k,
            width: n,
            seed: 0.0,
            outer_extent: heads,
            outer_stride_a: (m * k) as i64,
            outer_stride_b: (k * n) as i64,
            outer_stride_out: (m * n) as i64,
            vecs: VECS,
        };
        run_width_tile_neon::<VECS>(&plan, &raw, None, &mut out);
        out
    }

    #[cfg(target_os = "macos")]
    fn accelerate_route(
        a: &[f32],
        b_nk: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        for head in 0..heads {
            let a_head = &a[head * m * k..(head + 1) * m * k];
            let b_head = &b_nk[head * n * k..(head + 1) * n * k];
            let out_head = &mut out[head * m * n..(head + 1) * m * n];
            // SAFETY: `a_head`/`b_head` are exactly `m*k`/`n*k` elements
            // (`m x k` and `n x k` row-major, this function's own doc),
            // `out_head` exactly `m*n` -- every bound `try_run_accelerate_sgemm`'s
            // own `# Safety` requires.
            let accelerated = unsafe {
                try_run_accelerate_sgemm(
                    a_head, 0, k, b_head, 0, k, out_head, 0, n, m, n, k, 1, 0.0, true,
                )
            };
            assert!(
                accelerated,
                "try_run_accelerate_sgemm declined a shape this nano expects to run: m={m} n={n} k={k}"
            );
        }
        out
    }

    fn mean_cov(samples: &[f64]) -> (f64, f64) {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        (mean, variance.sqrt() / mean)
    }

    let load = std::process::Command::new("uptime")
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime unavailable".to_string());
    eprintln!("attention_tile_shapes_nano: heads={HEADS} repeats={REPEATS} ambient_load={load}");

    for seq_len in [7usize, 8, 9] {
        for (name, k, n) in [("Q@K^T", 32usize, seq_len), ("softmax@V", seq_len, 32usize)] {
            let m = seq_len;
            let mut rng = Lcg(0x9E37_79B9 ^ (m as u64) ^ ((k as u64) << 8) ^ ((n as u64) << 16));
            let a: Vec<f32> = (0..HEADS * m * k).map(|_| rng.next_unit()).collect();
            let b_kn: Vec<f32> = (0..HEADS * k * n).map(|_| rng.next_unit()).collect();
            // `n x k` row-major transpose of `b_kn`, per-head -- the
            // layout `try_run_accelerate_sgemm`'s own doc requires.
            let mut b_nk = vec![0.0f32; HEADS * n * k];
            for head in 0..HEADS {
                for row in 0..k {
                    for col in 0..n {
                        b_nk[head * n * k + col * k + row] = b_kn[head * k * n + row * n + col];
                    }
                }
            }

            let reference = scalar_reference(&a, &b_kn, HEADS, m, k, n);

            let mut scalar_samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let started = Instant::now();
                let out = scalar_reference(&a, &b_kn, HEADS, m, k, n);
                scalar_samples.push(started.elapsed().as_nanos() as f64);
                std::hint::black_box(&out);
            }
            let (scalar_mean, scalar_cov) = mean_cov(&scalar_samples);

            let width_eligible = n >= WIDTH_TILE_VECS * 4;
            let neon_report = if width_eligible {
                let mut neon_out = Vec::new();
                let mut neon_samples = Vec::with_capacity(REPEATS);
                for _ in 0..REPEATS {
                    let started = Instant::now();
                    neon_out = neon_route::<WIDTH_TILE_VECS>(&a, &b_kn, HEADS, m, k, n);
                    neon_samples.push(started.elapsed().as_nanos() as f64);
                }
                for (got, want) in neon_out.iter().zip(&reference) {
                    assert!(
                        (got - want).abs() < 1e-4,
                        "neon width tile diverged from scalar reference: got={got} want={want}"
                    );
                }
                let (neon_mean, neon_cov) = mean_cov(&neon_samples);
                format!(
                    "neon_ns={neon_mean:>9.1} (cov={neon_cov:.3}) neon_speedup={:.3}x",
                    scalar_mean / neon_mean
                )
            } else {
                "neon_ns=N/A (NarrowWidth-declined, N < WIDTH_TILE_VECS*4, never routed)"
                    .to_string()
            };

            #[cfg(target_os = "macos")]
            let accelerate_report = {
                let mut accelerate_out = Vec::new();
                let mut accelerate_samples = Vec::with_capacity(REPEATS);
                for _ in 0..REPEATS {
                    let started = Instant::now();
                    accelerate_out = accelerate_route(&a, &b_nk, HEADS, m, k, n);
                    accelerate_samples.push(started.elapsed().as_nanos() as f64);
                }
                for (got, want) in accelerate_out.iter().zip(&reference) {
                    assert!(
                        (got - want).abs() < 1e-3,
                        "accelerate diverged from scalar reference: got={got} want={want}"
                    );
                }
                let (accelerate_mean, accelerate_cov) = mean_cov(&accelerate_samples);
                format!(
                    "accelerate_ns={accelerate_mean:>9.1} (cov={accelerate_cov:.3}) accelerate_speedup={:.3}x",
                    scalar_mean / accelerate_mean
                )
            };
            #[cfg(not(target_os = "macos"))]
            let accelerate_report = "accelerate_ns=N/A (non-macos)".to_string();

            eprintln!(
                "{name:<10} M={m:<2} K={k:<2} N={n:<2} width_eligible={width_eligible:<5} scalar_ns={scalar_mean:>9.1} (cov={scalar_cov:.3}) {neon_report} {accelerate_report}"
            );
        }
    }
}

/// KILL-EARLY nano (narrow-tile task, 2026-09-01, binding on ROW
/// 198/199): the untiled scalar path vs the narrow-`VECS` NEON width
/// tile at BGE's `Q@K^T` inner shape `(M,32)x(32,M)`, `M` in `{7, 8,
/// 9}`, `heads=12` — the exact class [`attention_tile_shapes_nano`]
/// above reports `neon_ns=N/A` for. Two arms per `M`: `forced_vecs2`
/// runs `gemm_width_tile_neon::<_, 2>` exactly as the binding
/// instruction names it (at `N=7` this walks ZERO full 8-wide tiles —
/// `col_tiles = 7/8 = 0` — so every element falls through the same
/// scalar column-tail loop the untiled path already uses, measuring
/// only the tile/outer-loop overhead with no compensating NEON work);
/// `admitted` runs whichever `VECS` [`width_tile_vecs_for`] actually
/// selects for that `N` (`1` at `N=7`, `2` at `N=8/9`, identical to
/// `forced_vecs2` at `N=8/9`). `#[ignore]`: manual, not part of the CI
/// gate — `cargo test -p proxima-tensor --release
/// narrow_width_tile_shapes_nano -- --ignored --nocapture`.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore = "manual nano bench, not a CI gate; see this test's own doc"]
fn narrow_width_tile_shapes_nano() {
    const HEADS: usize = 12;
    const REPEATS: usize = 7;

    fn scalar_reference(
        a: &[f32],
        b_kn: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        for head in 0..heads {
            let a_head = &a[head * m * k..(head + 1) * m * k];
            let b_head = &b_kn[head * k * n..(head + 1) * k * n];
            let out_head = &mut out[head * m * n..(head + 1) * m * n];
            for row in 0..m {
                for col in 0..n {
                    let value = width_tile_scalar_cell(
                        KStridedTile {
                            data: a_head,
                            base: (row * k) as i64,
                            k_stride: 1,
                        },
                        KStridedTile {
                            data: b_head,
                            base: col as i64,
                            k_stride: n as i64,
                        },
                        k,
                        0.0,
                    );
                    out_head[row * n + col] = value;
                }
            }
        }
        out
    }

    fn neon_route<const VECS: usize>(
        a: &[f32],
        b_kn: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        let raw: [&[f32]; 2] = [a, b_kn];
        let plan = WidthTilePlan {
            a_operand: 0,
            b_operand: 1,
            row_stride_a: k as i64,
            base_a: 0,
            k_stride_a: 1,
            base_b: 0,
            k_stride_b: n as i64,
            out_base: 0,
            out_row_stride: n as i64,
            out_col_stride: 1,
            leading_total: m,
            reduction_total: k,
            width: n,
            seed: 0.0,
            outer_extent: heads,
            outer_stride_a: (m * k) as i64,
            outer_stride_b: (k * n) as i64,
            outer_stride_out: (m * n) as i64,
            vecs: VECS,
        };
        run_width_tile_neon::<VECS>(&plan, &raw, None, &mut out);
        out
    }

    fn mean_cov(samples: &[f64]) -> (f64, f64) {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        (mean, variance.sqrt() / mean)
    }

    let load = std::process::Command::new("uptime")
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime unavailable".to_string());
    eprintln!("narrow_width_tile_shapes_nano: heads={HEADS} repeats={REPEATS} ambient_load={load}");

    for m in [7usize, 8, 9] {
        let (k, n) = (32usize, m);
        let mut rng = Lcg(0x9E37_79B9 ^ (m as u64) ^ ((k as u64) << 8) ^ ((n as u64) << 16));
        let a: Vec<f32> = (0..HEADS * m * k).map(|_| rng.next_unit()).collect();
        let b_kn: Vec<f32> = (0..HEADS * k * n).map(|_| rng.next_unit()).collect();

        let reference = scalar_reference(&a, &b_kn, HEADS, m, k, n);

        let mut scalar_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = Instant::now();
            let out = scalar_reference(&a, &b_kn, HEADS, m, k, n);
            scalar_samples.push(started.elapsed().as_nanos() as f64);
            std::hint::black_box(&out);
        }
        let (scalar_mean, scalar_cov) = mean_cov(&scalar_samples);

        let mut forced2_out = neon_route::<2>(&a, &b_kn, HEADS, m, k, n);
        for (got, want) in forced2_out.iter().zip(&reference) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "vecs=2 not bit-identical to scalar reference: got={got} want={want}"
            );
        }
        let mut forced2_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = Instant::now();
            forced2_out = neon_route::<2>(&a, &b_kn, HEADS, m, k, n);
            forced2_samples.push(started.elapsed().as_nanos() as f64);
        }
        std::hint::black_box(&forced2_out);
        let (forced2_mean, forced2_cov) = mean_cov(&forced2_samples);

        let admitted_vecs = width_tile_vecs_for(n).expect("N=7/8/9 all admit at least VECS=1");
        let (admitted_mean, admitted_cov) = if admitted_vecs == 2 {
            (forced2_mean, forced2_cov)
        } else {
            let mut admitted_out = neon_route::<1>(&a, &b_kn, HEADS, m, k, n);
            for (got, want) in admitted_out.iter().zip(&reference) {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "vecs=1 not bit-identical to scalar reference: got={got} want={want}"
                );
            }
            let mut admitted_samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let started = Instant::now();
                admitted_out = neon_route::<1>(&a, &b_kn, HEADS, m, k, n);
                admitted_samples.push(started.elapsed().as_nanos() as f64);
            }
            std::hint::black_box(&admitted_out);
            mean_cov(&admitted_samples)
        };
        let admitted_speedup = scalar_mean / admitted_mean;

        eprintln!(
            "Q@K^T M={m:<2} K={k:<2} N={n:<2} scalar_ns={scalar_mean:>9.1} (cov={scalar_cov:.3}) forced_vecs2_ns={forced2_mean:>9.1} (cov={forced2_cov:.3}) forced_vecs2_speedup={:.3}x admitted_vecs={admitted_vecs} admitted_ns={admitted_mean:>9.1} (cov={admitted_cov:.3}) admitted_speedup={admitted_speedup:.3}x",
            scalar_mean / forced2_mean
        );
    }
}

/// [`matmul_q4k_f32`] against the reference path (`proxima_gguf`'s own
/// tested dequantize into a full `f32` weight matrix, then a naive f32
/// dot product) — [`super`]'s guiding-principle 14: the incumbent
/// (dequantize-then-matmul) is correct by construction, so this is a
/// parity check, not a round-trip-to-self check. Two real (pseudo-random,
/// non-degenerate — `Lcg`, not zeros/constants) rows x 2 super-blocks
/// (512 elements) each, `Q4_K`'s minimum non-trivial multi-block shape.
#[proxima::test]
#[case::seed_1(1)]
#[case::seed_7(7)]
#[case::seed_1000(1000)]
async fn matmul_q4k_f32_matches_dequantize_then_f32_matmul(#[case] seed: u64) {
    use proxima_gguf::quant::q4_k::{QK_K, dequantize, quantize};

    const ROWS: usize = 2;
    const BLOCKS_PER_ROW: usize = 2;
    const K: usize = BLOCKS_PER_ROW * QK_K;
    const ROW_BYTES: usize = BLOCKS_PER_ROW * Q4K_BLOCK_BYTES;

    let weights_f32 = random_vec(seed, ROWS * K);
    let activation = random_vec(seed.wrapping_add(1), K);

    let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q4K_BLOCK_BYTES];
    for (row_f32, row_packed) in weights_f32
        .as_chunks::<K>()
        .0
        .iter()
        .zip(packed.as_chunks_mut::<ROW_BYTES>().0)
    {
        quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
    }

    let mut dequantized_reference = vec![0.0f32; ROWS];
    let mut dequantized_row = vec![0.0f32; K];
    for (row_index, row_packed) in packed.as_chunks::<ROW_BYTES>().0.iter().enumerate() {
        dequantize(row_packed, &mut dequantized_row)
            .expect("2 whole super-blocks dequantize cleanly");
        dequantized_reference[row_index] = dequantized_row
            .iter()
            .zip(&activation)
            .map(|(weight, value)| weight * value)
            .sum();
    }

    let quantized_result =
        matmul_q4k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

    let mut max_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    for (got, want) in quantized_result.iter().zip(&dequantized_reference) {
        let diff = (got - want).abs();
        max_diff = max_diff.max(diff);
        sum_sq_diff += f64::from(diff) * f64::from(diff);
    }
    let rms_diff = (sum_sq_diff / ROWS as f64).sqrt();
    eprintln!(
        "matmul_q4k_f32 vs dequantize-then-matmul: seed={seed} max_diff={max_diff} rms_diff={rms_diff}"
    );

    // not bit-exact: `dot_q4k_f32` folds one super-block at a time in a
    // single running accumulator, while the reference sums a
    // fully-materialized 512-element row in one linear pass — same
    // terms, different intermediate rounding. Loose bound, not tuned to
    // the measured numbers, matching `q4_k.rs`'s own round-trip tests.
    assert!(
        max_diff < 1e-2,
        "max_diff={max_diff} exceeds parity tolerance"
    );
    assert!(
        rms_diff < 1e-2,
        "rms_diff={rms_diff} exceeds parity tolerance"
    );
}

/// [`matmul_worker_count`]'s only machine-independent invariant: the
/// selected count is never zero (nothing would run) and never more than
/// `available_parallelism()` (oversubscription past the OS-reported
/// core count). The exact value — P-core count on Apple, full logical
/// count elsewhere — is machine-dependent and deliberately not asserted
/// here.
#[test]
fn matmul_worker_count_is_between_one_and_available_parallelism() {
    let available = thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(1);
    let workers = matmul_worker_count();
    assert!(
        workers >= 1,
        "worker count must be at least 1, got {workers}"
    );
    assert!(
        workers <= available,
        "worker count {workers} exceeds available_parallelism {available}"
    );
}

/// [`matmul_rows_threaded`]'s pool dispatch (through [`matmul_q4k_f32`])
/// against the same per-row kernel run sequentially in this test, no
/// threading involved — 128 rows x 512 elements clears
/// [`PARALLEL_THRESHOLD`] (65536 macs vs 4096) and is wide enough that
/// `quantized_matmul_workers` returns `Some` on any machine with fewer
/// than 128 hardware threads, so `matmul_q4k_f32` provably takes the
/// pool path here. Each output row is an independent reduction with no
/// cross-row accumulator (`dot_row` reads only its own row's bytes), so
/// dispatch mechanism cannot perturb any one row's rounding — the pool
/// and a bare sequential loop over the identical `dot_q4k_f32` calls
/// must agree bit-for-bit, not just within a numeric tolerance.
#[test]
fn matmul_q4k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
    use proxima_gguf::quant::q4_k::quantize;

    const ROWS: usize = 128;
    const BLOCKS_PER_ROW: usize = 2;
    const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q4_k::QK_K;
    const ROW_BYTES: usize = BLOCKS_PER_ROW * Q4K_BLOCK_BYTES;

    let weights_f32 = random_vec(42, ROWS * K);
    let activation = random_vec(43, K);

    let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q4K_BLOCK_BYTES];
    for (row_f32, row_packed) in weights_f32
        .as_chunks::<K>()
        .0
        .iter()
        .zip(packed.as_chunks_mut::<ROW_BYTES>().0)
    {
        quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
    }

    assert!(
        quantized_matmul_workers(ROWS, activation.len()).is_some(),
        "test fixture must actually clear the parallel threshold to exercise the pool path"
    );

    let pooled_result =
        matmul_q4k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

    let sequential_reference: Vec<f32> = packed
        .as_chunks::<ROW_BYTES>()
        .0
        .iter()
        .map(|weight_row| dot_q4k_f32(weight_row, &activation).expect("well-formed row"))
        .collect();

    assert_eq!(
        pooled_result, sequential_reference,
        "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
         each row is an independent reduction, so dispatch mechanism cannot move rounding"
    );
}

/// [`matmul_q5k_f32`] was unconditionally sequential (never called
/// [`quantized_matmul_workers`]) before it was routed through the same
/// `matmul_quantized_dispatch` helper `matmul_q4k_f32` uses — same test
/// shape as `matmul_q4k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel`,
/// proving the fix actually reaches the pool path and stays bit-exact.
#[test]
fn matmul_q5k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
    use proxima_gguf::quant::q5_k::quantize;

    const ROWS: usize = 128;
    const BLOCKS_PER_ROW: usize = 2;
    const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q5_k::QK_K;
    const ROW_BYTES: usize = BLOCKS_PER_ROW * Q5K_BLOCK_BYTES;

    let weights_f32 = random_vec(44, ROWS * K);
    let activation = random_vec(45, K);

    let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q5K_BLOCK_BYTES];
    for (row_f32, row_packed) in weights_f32
        .as_chunks::<K>()
        .0
        .iter()
        .zip(packed.as_chunks_mut::<ROW_BYTES>().0)
    {
        quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
    }

    assert!(
        quantized_matmul_workers(ROWS, activation.len()).is_some(),
        "test fixture must actually clear the parallel threshold to exercise the pool path"
    );

    let pooled_result =
        matmul_q5k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

    let sequential_reference: Vec<f32> = packed
        .as_chunks::<ROW_BYTES>()
        .0
        .iter()
        .map(|weight_row| dot_q5k_f32(weight_row, &activation).expect("well-formed row"))
        .collect();

    assert_eq!(
        pooled_result, sequential_reference,
        "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
         each row is an independent reduction, so dispatch mechanism cannot move rounding"
    );
}

/// [`matmul_q6k_f32`]'s counterpart to the `matmul_q5k_f32` test above —
/// same was-always-sequential bug, same fix, same bit-exactness proof.
#[test]
fn matmul_q6k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
    use proxima_gguf::quant::q6_k::quantize;

    const ROWS: usize = 128;
    const BLOCKS_PER_ROW: usize = 2;
    const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q6_k::QK_K;
    const ROW_BYTES: usize = BLOCKS_PER_ROW * Q6K_BLOCK_BYTES;

    let weights_f32 = random_vec(46, ROWS * K);
    let activation = random_vec(47, K);

    let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q6K_BLOCK_BYTES];
    for (row_f32, row_packed) in weights_f32
        .as_chunks::<K>()
        .0
        .iter()
        .zip(packed.as_chunks_mut::<ROW_BYTES>().0)
    {
        quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
    }

    assert!(
        quantized_matmul_workers(ROWS, activation.len()).is_some(),
        "test fixture must actually clear the parallel threshold to exercise the pool path"
    );

    let pooled_result =
        matmul_q6k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

    let sequential_reference: Vec<f32> = packed
        .as_chunks::<ROW_BYTES>()
        .0
        .iter()
        .map(|weight_row| dot_q6k_f32(weight_row, &activation).expect("well-formed row"))
        .collect();

    assert_eq!(
        pooled_result, sequential_reference,
        "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
         each row is an independent reduction, so dispatch mechanism cannot move rounding"
    );
}

/// [`row_chunk_count`] must scale down with total work, not stay pinned
/// to `workers * ROW_OVERSUBSCRIBE` for every shape -- the defect this
/// session fixes: a narrow, low-mac call (`attn_k`/`attn_v`'s real
/// `rows=1024 k=4096` shape, 4.19M macs) was paying the same fixed
/// 40-way dispatch as a wide, high-mac call (`ffn_up`/`ffn_gate`'s
/// shape) despite carrying far less work. `rows` held fixed across both
/// cases so only `contraction_width` (the mac count) drives the
/// difference; the wide case's `contraction_width` is synthetic
/// (comfortably above `MIN_MACS_PER_CHUNK * oversubscribed_ceiling`)
/// purely to prove the cap still applies once a shape carries enough
/// work to earn the full oversubscribed split.
#[test]
fn row_chunk_count_scales_down_for_a_small_shape_and_stays_capped_for_a_large_one() {
    let workers = 10;
    let rows = 1024;

    let small_shape_chunks = row_chunk_count(rows, workers, 4096); // attn_k/attn_v's real k
    let large_shape_chunks = row_chunk_count(rows, workers, 100_000); // comfortably wide

    let oversubscribed_ceiling = workers * ROW_OVERSUBSCRIBE;
    assert!(
        small_shape_chunks < large_shape_chunks,
        "a low-mac shape must produce fewer chunks than a high-mac shape at the same row \
         count: small={small_shape_chunks} large={large_shape_chunks}"
    );
    assert_eq!(
        large_shape_chunks, oversubscribed_ceiling,
        "a shape whose total work clears MIN_MACS_PER_CHUNK * oversubscribed_ceiling must \
         still land on the full oversubscribed split"
    );
    assert!(small_shape_chunks >= 1, "chunk count must never be zero");
}

/// [`matmul_q5k_q8k_f32`] was unconditionally sequential (never called
/// [`quantized_matmul_workers`]) despite [`matmul_q4k_q8k_f32`] already
/// routing through the pool — same test shape as
/// `matmul_q4k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel`,
/// proving the fix actually reaches the pool path and stays bit-exact.
#[cfg(feature = "q5k-int8-dot")]
#[test]
fn matmul_q5k_q8k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
    use proxima_gguf::quant::q5_k::quantize;

    const ROWS: usize = 128;
    const BLOCKS_PER_ROW: usize = 2;
    const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q5_k::QK_K;
    const ROW_BYTES: usize = BLOCKS_PER_ROW * Q5K_BLOCK_BYTES;

    let weights_f32 = random_vec(48, ROWS * K);
    let activation = random_vec(49, K);

    let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q5K_BLOCK_BYTES];
    for (row_f32, row_packed) in weights_f32
        .as_chunks::<K>()
        .0
        .iter()
        .zip(packed.as_chunks_mut::<ROW_BYTES>().0)
    {
        quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
    }

    assert!(
        quantized_matmul_workers(ROWS, activation.len()).is_some(),
        "test fixture must actually clear the parallel threshold to exercise the pool path"
    );

    let pooled_result =
        matmul_q5k_q8k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

    let mut activation_q8k = vec![0u8; BLOCKS_PER_ROW * Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation");
    let sequential_reference: Vec<f32> = packed
        .as_chunks::<ROW_BYTES>()
        .0
        .iter()
        .map(|weight_row| dot_q5k_q8k(weight_row, &activation_q8k).expect("well-formed row"))
        .collect();

    assert_eq!(
        pooled_result, sequential_reference,
        "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
         each row is an independent reduction, so dispatch mechanism cannot move rounding"
    );
}

/// [`matmul_q6k_q8k_f32`]'s counterpart to the `matmul_q5k_q8k_f32` test
/// above — same was-always-sequential bug, same fix, same bit-exactness
/// proof.
#[cfg(feature = "q6k-int8-dot")]
#[test]
fn matmul_q6k_q8k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
    use proxima_gguf::quant::q6_k::quantize;

    const ROWS: usize = 128;
    const BLOCKS_PER_ROW: usize = 2;
    const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q6_k::QK_K;
    const ROW_BYTES: usize = BLOCKS_PER_ROW * Q6K_BLOCK_BYTES;

    let weights_f32 = random_vec(50, ROWS * K);
    let activation = random_vec(51, K);

    let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q6K_BLOCK_BYTES];
    for (row_f32, row_packed) in weights_f32
        .as_chunks::<K>()
        .0
        .iter()
        .zip(packed.as_chunks_mut::<ROW_BYTES>().0)
    {
        quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
    }

    assert!(
        quantized_matmul_workers(ROWS, activation.len()).is_some(),
        "test fixture must actually clear the parallel threshold to exercise the pool path"
    );

    let pooled_result =
        matmul_q6k_q8k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

    let mut activation_q8k = vec![0u8; BLOCKS_PER_ROW * Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation");
    let sequential_reference: Vec<f32> = packed
        .as_chunks::<ROW_BYTES>()
        .0
        .iter()
        .map(|weight_row| dot_q6k_q8k(weight_row, &activation_q8k).expect("well-formed row"))
        .collect();

    assert_eq!(
        pooled_result, sequential_reference,
        "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
         each row is an independent reduction, so dispatch mechanism cannot move rounding"
    );
}

/// [`reject_non_float32`]'s quantized-weight exemption: a `UInt8`-tagged
/// node (standing in for packed `Q4_K` bytes) used as one operand of a
/// `Multiply`-then-`Add`-reduce (matmul) now type-checks when named in
/// the exemption set, and still rejects everything else exactly as
/// before — proving "exactly as far as needed and no further."
#[test]
fn reject_non_float32_exempts_a_quantized_weight_in_matmul_position() {
    let mut program = Vec::new();
    let weight = block(&mut program, DType::UInt8, &[Extent::Static(4)]);
    let activation = f32_block(&mut program, &[Extent::Static(4)]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(1, &[0]))),
                (activation, IndexMap::Affine(map::projection(1, &[0]))),
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
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    assert!(
        reject_non_float32(&program, &BTreeSet::new()).is_err(),
        "an unexempted UInt8 node must still be rejected"
    );

    let mut exempt = BTreeSet::new();
    exempt.insert(weight);
    assert!(
        reject_non_float32(&program, &exempt).is_ok(),
        "a UInt8 node used exclusively as a matmul weight operand must be exempted"
    );
}

/// The exemption is shape-scoped, not tag-scoped: a `UInt8` node that is
/// NOT feeding a `Multiply`-then-`Add` reduce is rejected even when
/// named in the exemption set.
#[test]
fn reject_non_float32_still_rejects_a_quantized_node_outside_matmul_shape() {
    let mut program = Vec::new();
    let weight = block(&mut program, DType::UInt8, &[Extent::Static(4)]);
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: weight,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let mut exempt = BTreeSet::new();
    exempt.insert(weight);
    assert!(
        reject_non_float32(&program, &exempt).is_err(),
        "a quantized node reduced directly (no Multiply) is not the matmul shape and must stay rejected"
    );
}

/// ROW 429's own regression: `per_head_channel_slice`'s former call site
/// against a packed weight (`spec.rs`'s `Qwen35DenseAttentionTaps` doc,
/// ROW 428) built exactly this shape — a `UInt8` weight `Multiply`-ed
/// against a one-hot mask built ENTIRELY from `Op::Iota` and `Equal`
/// (no real data anywhere in its ancestry), feeding an `Add`-reduce.
/// Before [`operand_traces_to_a_real_input`] existed,
/// [`is_quantized_matmul_operand`] recognized this as the ordinary
/// matmul shape purely because the mask's OWN dtype was `Float32`, and
/// [`run_reduce_quantized`] then read `rows`/`k` off whichever axis the
/// select happened to reduce. An axis-position check was tried first and
/// discarded: this crate's own shipped matmul shapes disagree on which
/// position is "the" contraction axis
/// (`quantized_matmul_program`'s `[rows, k]` reduces its LAST axis;
/// `a_reduce_where_activation_and_packed_weight_share_a_kept_output_axis_is_rejected`
/// reduces its FIRST), so a fixed-position rule broke one or the other
/// real shape depending on which position it picked (ROW 429's own
/// RED data). The same weight `Multiply`-ed against a REAL `Op::Input`
/// activation stays exempted — see
/// `reject_non_float32_exempts_a_quantized_weight_in_matmul_position`,
/// unchanged and still passing.
#[test]
fn reject_non_float32_rejects_a_quantized_weight_selected_by_a_synthetic_iota_mask() {
    let mut program = Vec::new();
    let weight = block(&mut program, DType::UInt8, &[Extent::Static(4)]);
    let channel_index = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(4),
        },
    );
    let target = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 1.0,
        },
    );
    let mask = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Equal,
            operands: vec![
                (channel_index, IndexMap::Affine(map::projection(1, &[0]))),
                (target, IndexMap::Affine(map::projection(1, &[]))),
            ],
            name: None,
        },
    );
    let selected = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(1, &[0]))),
                (mask, IndexMap::Affine(map::projection(1, &[0]))),
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
            operand: selected,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let mut exempt = BTreeSet::new();
    exempt.insert(weight);
    assert!(
        reject_non_float32(&program, &exempt).is_err(),
        "a packed weight selected by a mask with no real Input in its ancestry must be \
         rejected, not silently admitted as the matmul shape"
    );
}

/// The dead-leaf exemption's whole point: an ONNX-shaped program where an
/// `Int64` `Op::Input` (standing in for a `Reshape`'s shape initializer)
/// is never read by anything else in `program` must still evaluate its
/// all-`Float32` output cone — reproduces the onnx-model failure this
/// exemption exists for, first as `reject_non_float32` directly (the
/// root cause), then through the real `evaluate_named` entry point (the
/// user-visible symptom).
#[test]
fn reject_non_float32_exempts_an_unreferenced_non_float32_input() {
    let mut program = Vec::new();
    let _dead_shape_leaf = append(
        &mut program,
        Op::Input {
            dtype: DType::Int64,
            shape: vec![Extent::Static(2)],
            name: Some(String::from("reshape_shape")),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some(String::from("activation")),
        },
    );

    assert!(
        reject_non_float32(&program, &BTreeSet::new()).is_ok(),
        "an Int64 Input that nothing in the program reads can never reach the f32-only \
         kernels, so it must not fail the gate"
    );

    // `reshape_shape`'s two Int64 elements are never read as f32 data —
    // `evaluate_named` only needs *a* binding for every `Op::Input`, not
    // one whose bit pattern is meaningful, since the dead leaf's buffer
    // is never handed to a kernel.
    let evaluated = evaluate_named(
        &program,
        &[],
        &[
            ("reshape_shape", &[0.0, 0.0]),
            ("activation", &[1.0, 2.0, 3.0, 4.0]),
        ],
        &[activation],
    )
    .expect("an all-f32 output cone must evaluate even with a dead non-f32 leaf present");
    let (data, _shape) = evaluated
        .get(activation)
        .expect("activation must be a resolved output");
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0].as_slice());
}

/// The other half of the same contract: a non-`Float32` `Op::Input` that
/// IS referenced (here, added into an otherwise-`Float32` elementwise
/// chain) still reaches `run_node_into`'s f32-only kernels regardless of
/// output reachability — `BoundOpBuilder::finish`'s own doc is explicit
/// that a held elementwise op materializes "either [as] a requested
/// output or dead code" — so it must still be rejected, proving the
/// dead-leaf exemption did not widen into "any Input is exempt."
#[test]
fn reject_non_float32_still_rejects_a_referenced_non_float32_input() {
    let mut program = Vec::new();
    let stray_int_leaf = block(&mut program, DType::Int64, &[Extent::Static(4)]);
    let activation = f32_block(&mut program, &[Extent::Static(4)]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (stray_int_leaf, IndexMap::Affine(map::projection(1, &[0]))),
                (activation, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );

    assert!(
        reject_non_float32(&program, &BTreeSet::new()).is_err(),
        "an Int64 Input consumed by a live Elementwise operand still reaches the \
         f32-only kernels and must stay rejected"
    );
}

/// A two-named-input, one-`Elementwise`-node program (`c = a + b`), the
/// smallest fixture that exercises both an [`Op::Input`] rebinding slot
/// AND a resolved-node buffer -- no MNIST dependency, per this row's
/// own task instruction (`docs/discipline.md` ROW 165: `build_static_arena`/
/// `evaluate_named_with_arena` unit-tested on a small synthetic
/// program).
fn named_add_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some(String::from("a")),
        },
    );
    let b = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some(String::from("b")),
        },
    );
    let sum = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (a, IndexMap::Affine(map::projection(1, &[0]))),
                (b, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );
    (program, sum)
}

/// [`build_static_arena`] + [`evaluate_named_with_arena`] run TWICE
/// against a CALLER-OWNED arena, on two different `named` bindings, and
/// must each match [`evaluate_named`]'s own result bit for bit --
/// [`evaluate_named`] now reaches the identical machinery through its
/// OWN, separately-cached arena (`ARENA_CACHE`), so this proves the two
/// independent arena instances (caller-owned vs. process-cached) agree,
/// the same correctness bar `train_step_lane.rs`'s own bench-level
/// `assert_arena_bit_identical_to_baseline` holds the arena to, proved
/// here at the library-unit level instead.
#[test]
fn evaluate_named_with_arena_matches_evaluate_named_over_two_calls() {
    let (program, sum) = named_add_program();
    let mut arena =
        build_static_arena(&program, &[], &[sum]).expect("small program builds a static arena");

    let first_a = [1.0f32, 2.0, 3.0, 4.0];
    let first_b = [10.0f32, 20.0, 30.0, 40.0];
    let arena_first = evaluate_named_with_arena(&mut arena, &[("a", &first_a), ("b", &first_b)])
        .expect("first arena call evaluates");
    let baseline_first = evaluate_named(&program, &[], &[("a", &first_a), ("b", &first_b)], &[sum])
        .expect("first baseline call evaluates");
    assert_eq!(
        arena_first.get(sum).map(|(data, _)| data.to_vec()),
        baseline_first.get(sum).map(|(data, _)| data.to_vec()),
        "first call: arena and fresh-alloc paths must agree bit for bit"
    );

    let second_a = [100.0f32, 200.0, 300.0, 400.0];
    let second_b = [1.0f32, 2.0, 3.0, 4.0];
    let arena_second = evaluate_named_with_arena(&mut arena, &[("a", &second_a), ("b", &second_b)])
        .expect("second arena call evaluates");
    let baseline_second =
        evaluate_named(&program, &[], &[("a", &second_a), ("b", &second_b)], &[sum])
            .expect("second baseline call evaluates");
    assert_eq!(
        arena_second.get(sum).map(|(data, _)| data.to_vec()),
        baseline_second.get(sum).map(|(data, _)| data.to_vec()),
        "second call (same arena, reused buffers): arena and fresh-alloc paths must still agree bit for bit"
    );
    assert_eq!(
        arena_second.get(sum).map(|(data, _)| data.to_vec()),
        Some(vec![101.0, 202.0, 303.0, 404.0])
    );
}

/// [`matmul_program`], but both leaves are NAMED -- `checkout_arena`'s
/// own derivation (every genuine `Op::Input` name in `named` is a
/// `constant_inputs` candidate) only has anything to find when the
/// program's leaves carry names at all, same as any real ONNX-lowered
/// program (`proxima-onnx/src/lower.rs`'s own doc: initializers AND
/// runtime inputs both lower to a named `Op::Input`). `n` is
/// `WIDTH_TILE_VECS * 4` so the `w` operand clears
/// [`width_tile_pack_candidate`]'s tile-width admission.
#[cfg(target_arch = "aarch64")]
fn named_matmul_program(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(m), Extent::Static(k)],
            name: Some(String::from("x")),
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(k), Extent::Static(n)],
            name: Some(String::from("w")),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("named_matmul".into()),
        }),
    );
    (program, sum)
}

/// `docs/discipline.md` ROW 208's own defect closed: plain
/// [`evaluate_named`] on a named-leaf matmul program now packs the `w`
/// (weight) operand with ZERO caller opt-in -- `checkout_arena` derives
/// `constant_inputs` from `named` itself on the cache miss, and
/// [`build_packed_width_panels`]'s own structural gate (2-D, `b`
/// operand of a width-tile-eligible `Reduce`) is what actually decides
/// `w` qualifies. Packed and unpacked results must agree bit for bit
/// (packing is a layout reorder of the SAME source elements, never a
/// different computation, `pack_width_tile_panels`'s own doc).
#[cfg(target_arch = "aarch64")]
#[test]
fn evaluate_named_packs_a_named_weight_with_zero_caller_opt_in() {
    let width_tile_n = (WIDTH_TILE_VECS * 4) as u32;
    let (program, sum) = named_matmul_program(4, WIDTH_TILE_ROWS as u32 * 8, width_tile_n);
    let m = 4usize;
    let k = WIDTH_TILE_ROWS * 8;
    let n = width_tile_n as usize;

    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) - 3.0).collect();
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02) - 5.0).collect();
    let named: [(&str, &[f32]); 2] = [("x", &x), ("w", &w)];

    let packed = evaluate_named(&program, &[], &named, &[sum])
        .expect("named matmul evaluates through the default (packing-eligible) path");

    let mut unpacked_arena = build_static_arena(&program, &[], &[sum])
        .expect("same program builds an unpacked reference arena");
    let unpacked = evaluate_named_with_arena(&mut unpacked_arena, &named)
        .expect("unpacked reference arena evaluates");

    assert_eq!(
        packed.get(sum).map(|(data, _)| data.to_vec()),
        unpacked.get(sum).map(|(data, _)| data.to_vec()),
        "the default (now packing-eligible) path must agree bit for bit with the explicit, \
         never-packed reference arena -- packing is a layout reorder, never a different sum"
    );

    let expected = naive_matmul(&x, &w, m, k, n);
    assert_all_close(
        &packed
            .get(sum)
            .map(|(data, _)| data.to_vec())
            .expect("sum output present"),
        &expected,
        1e-4,
    );
}

/// The soundness half of the fix above: `checkout_arena` derives
/// `constant_inputs` STRUCTURALLY (every named `Op::Input`), never from
/// a caller's promise, so a name it guessed was call-invariant can turn
/// out not to be. Rebinding `w` to DIFFERENT bytes under the identical
/// `(program, symbols, outputs)` cache key must still produce the
/// correct answer for the NEW weights, never the panel packed from the
/// old ones -- [`bind_named_inputs_into_arena`]'s own rebind check
/// (`panel.source`) is what has to catch this.
#[cfg(target_arch = "aarch64")]
#[test]
fn evaluate_named_rebinding_a_packed_weight_never_serves_a_stale_panel() {
    let width_tile_n = (WIDTH_TILE_VECS * 4) as u32;
    let (program, sum) = named_matmul_program(4, WIDTH_TILE_ROWS as u32 * 8, width_tile_n);
    let m = 4usize;
    let k = WIDTH_TILE_ROWS * 8;
    let n = width_tile_n as usize;

    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) - 3.0).collect();
    let w_first: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02) - 5.0).collect();
    let first_named: [(&str, &[f32]); 2] = [("x", &x), ("w", &w_first)];
    let first = evaluate_named(&program, &[], &first_named, &[sum])
        .expect("first call (builds and packs the arena)");
    assert_all_close(
        &first
            .get(sum)
            .map(|(data, _)| data.to_vec())
            .expect("sum output present"),
        &naive_matmul(&x, &w_first, m, k, n),
        1e-4,
    );

    // SAME cache key (identical program/symbols/outputs), DIFFERENT `w`
    // bytes -- exactly the case `checkout_arena`'s doc warns a
    // structural guess must survive.
    let w_second: Vec<f32> = (0..k * n).map(|i| (i as f32 * -0.03) + 1.0).collect();
    let second_named: [(&str, &[f32]); 2] = [("x", &x), ("w", &w_second)];
    let second = evaluate_named(&program, &[], &second_named, &[sum])
        .expect("second call (same cache key, rebinds w)");
    assert_all_close(
        &second
            .get(sum)
            .map(|(data, _)| data.to_vec())
            .expect("sum output present"),
        &naive_matmul(&x, &w_second, m, k, n),
        1e-4,
    );
}

/// The bind-weights-once safety case (`docs/discipline.md`, rebind-identity
/// task, 2026-09-01): `b` is bound ONCE at build time via
/// `constant_inputs`, then never repeated in `named` on the first
/// `evaluate_named_with_arena` call -- `require_all = true` must still
/// succeed (`StaticArena::constant_bound` satisfies it), and the
/// answer must reflect the bound bytes, not zero. A SECOND call then
/// DOES re-pass `b`, with DIFFERENT bytes: this is the safety half --
/// `bind_named_inputs_into_arena`'s `Some(data)` arm never consults
/// `constant_bound`, so a caller who re-sends a constant is exactly as
/// correct as one who never used `constant_inputs` at all, and the
/// answer must reflect the NEW bytes.
#[test]
fn evaluate_named_with_arena_answers_a_constant_bound_input_without_a_repass_and_still_honors_a_repass()
 {
    let (program, sum) = named_add_program();
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_first = vec![10.0f32, 20.0, 30.0, 40.0];

    let mut arena = build_static_arena_with_constants(&program, &[], &[sum], &[("b", &b_first)])
        .expect("build arena with b bound at construction time");

    // First call: `named` carries ONLY `a` -- `b` is never re-passed.
    // `require_all = true` must not raise `UnboundInputName` for `b`.
    let first = evaluate_named_with_arena(&mut arena, &[("a", &a)])
        .expect("require_all is satisfied by a constant-bound input without a repass");
    assert_eq!(
        first.get(sum).map(|(data, _)| data.to_vec()),
        Some(vec![11.0f32, 22.0, 33.0, 44.0]),
        "first call must reflect the constant_inputs-bound b, not a zero-filled slot"
    );

    // Second call: `b` IS re-passed, with DIFFERENT bytes than the
    // constant_inputs bind. The rebind byte-compare must still catch
    // this and the answer must reflect the NEW b, never the old one.
    let b_second = vec![100.0f32, 200.0, 300.0, 400.0];
    let second = evaluate_named_with_arena(&mut arena, &[("a", &a), ("b", &b_second)])
        .expect("a caller may still repass a constant-bound input");
    assert_eq!(
        second.get(sum).map(|(data, _)| data.to_vec()),
        Some(vec![101.0f32, 202.0, 303.0, 404.0]),
        "a repassed constant-bound input with different bytes must invalidate the stale value"
    );
}

/// A two-input program with a genuinely dead node -- `dead = a * b`,
/// consumed by nothing, never a requested output -- alongside the
/// requested `live = a + b`. The smallest fixture for `docs/discipline.md`
/// ROW 167's execution-level elision: `dead` still gets `bind::bind`'s
/// own `BoundOp` (bind-time construction is untouched by design), it
/// just never runs.
fn dead_node_program() -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some(String::from("a")),
        },
    );
    let b = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some(String::from("b")),
        },
    );
    let dead = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (a, IndexMap::Affine(map::projection(1, &[0]))),
                (b, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );
    let live = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (a, IndexMap::Affine(map::projection(1, &[0]))),
                (b, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );
    (program, dead, live)
}

/// `dead` (zero consumers, not requested) never reaches `resolved` at
/// all -- [`bind::bind_plain`]'s reachability pass (ROW 541,
/// `docs/discipline.md`) skips it at bind time, before this arena's own
/// [`StaticArena::dead`]/`run_resolved_nodes_in_arena` skip machinery
/// (ROW 166's) ever sees it, so `dead` is stronger than "bound but
/// elided": there is no `BoundOp` and no pre-sized buffer for it at
/// all. `live` still evaluates bit-identically to [`evaluate_named`]'s
/// own result (which elides the identical way through its own
/// separately-cached arena).
#[test]
fn build_static_arena_elides_a_node_with_zero_consumers() {
    let (program, dead, live) = dead_node_program();
    let mut arena = build_static_arena(&program, &[], &[live])
        .expect("dead-node program builds a static arena");
    assert!(
        !arena.dead.contains(&dead),
        "a node unreachable from every requested output is never bound, so it cannot be a POST-bind dead entry either"
    );
    assert!(
        !arena.dead.contains(&live),
        "the requested output must never be marked dead"
    );

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let elided = evaluate_named_with_arena(&mut arena, &[("a", &a), ("b", &b)])
        .expect("elided arena call evaluates");
    let baseline = evaluate_named(&program, &[], &[("a", &a), ("b", &b)], &[live])
        .expect("baseline call evaluates");
    assert_eq!(
        elided.get(live).map(|(data, _)| data.to_vec()),
        baseline.get(live).map(|(data, _)| data.to_vec()),
        "the live output must be bit-identical whether or not the dead sibling actually executed"
    );
    assert_eq!(
        arena_output(&arena, dead),
        None,
        "the unreachable node was never bound, so it never got a buffer slot at all"
    );
}

/// The other half of the same contract: naming `dead` itself as a
/// requested output un-elides it -- `effective_outputs` membership is
/// what `dead_resolved_nodes` checks, so a caller who genuinely wants
/// that value back still gets it computed.
#[test]
fn build_static_arena_does_not_elide_a_dead_node_that_is_also_a_requested_output() {
    let (program, dead, live) = dead_node_program();
    let mut arena = build_static_arena(&program, &[], &[dead, live])
        .expect("dead-node program builds a static arena with dead requested");
    assert!(
        !arena.dead.contains(&dead),
        "requesting the otherwise-dead node as an output must un-elide it"
    );

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let evaluated = evaluate_named_with_arena(&mut arena, &[("a", &a), ("b", &b)])
        .expect("arena call evaluates");
    let baseline = evaluate_named(&program, &[], &[("a", &a), ("b", &b)], &[dead, live])
        .expect("baseline call evaluates");
    assert_eq!(
        evaluated.get(dead).map(|(data, _)| data.to_vec()),
        baseline.get(dead).map(|(data, _)| data.to_vec()),
        "the now-requested node must actually compute, bit-identical to the non-eliding baseline"
    );
    assert_eq!(
        evaluated.get(dead).map(|(data, _)| data.to_vec()),
        Some(vec![10.0, 40.0, 90.0, 160.0])
    );
}

/// A one-input program with a `Constant` feeding a live `Add` --
/// `docs/discipline.md` ROW 174's own found lever: `c`'s value is baked
/// into its `BoundOp` at `bind::bind` time and never depends on `a`.
fn constant_feeds_live_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some(String::from("a")),
        },
    );
    let constant = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            value: 5.0,
        },
    );
    let live = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (a, IndexMap::Affine(map::projection(1, &[0]))),
                (constant, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );
    (program, a, constant, live)
}

/// [`build_static_arena`] marks a live `Constant` (`bind::bind`'s own
/// `BoundOpKind::Constant` -- no operands, no dependence on `named` at
/// all) as one of its `static_nodes` and runs it exactly once, at build
/// time -- never again on any subsequent [`evaluate_named_with_arena`]
/// call, proven here by corrupting the constant's own resident buffer
/// between steps: if `run_resolved_nodes_in_arena` still executed it,
/// `run_constant`'s `output.fill(value)` would overwrite the corruption
/// back to the literal on the very next step. It does not.
#[test]
fn build_static_arena_runs_a_live_constant_once_and_never_again() {
    let (program, _a, constant, live) = constant_feeds_live_program();
    let mut arena = build_static_arena(&program, &[], &[live])
        .expect("constant-feeds-live program builds a static arena");
    assert!(
        arena.static_nodes.contains(&constant),
        "a live Constant-kind node must be marked static"
    );
    assert!(
        !arena.dead.contains(&constant),
        "a consumed Constant must never also be marked dead"
    );

    let step_one = [1.0f32, 2.0, 3.0, 4.0];
    let evaluated =
        evaluate_named_with_arena(&mut arena, &[("a", &step_one)]).expect("step one evaluates");
    assert_eq!(
        evaluated.get(live).map(|(data, _)| data.to_vec()),
        Some(vec![6.0, 7.0, 8.0, 9.0]),
        "a + the constant's own literal 5.0, computed once at build time"
    );
    assert_eq!(
        arena_output(&arena, constant),
        Some([5.0f32; 4].as_slice()),
        "the constant's own buffer holds its literal after step one"
    );

    arena.buffers[constant.0 as usize] = Some(alloc::vec![999.0f32; 4]);

    let step_two = [10.0f32, 20.0, 30.0, 40.0];
    let evaluated =
        evaluate_named_with_arena(&mut arena, &[("a", &step_two)]).expect("step two evaluates");
    assert_eq!(
        evaluated.get(live).map(|(data, _)| data.to_vec()),
        Some(vec![1009.0, 1019.0, 1029.0, 1039.0]),
        "step two must fold the CORRUPTED buffer, not the literal -- proving run_resolved_nodes_in_arena truly never re-executed the constant"
    );
    assert_eq!(
        arena_output(&arena, constant),
        Some([999.0f32; 4].as_slice()),
        "the corruption survives step two untouched"
    );

    let step_three = [0.0f32, 0.0, 0.0, 0.0];
    let evaluated =
        evaluate_named_with_arena(&mut arena, &[("a", &step_three)]).expect("step three evaluates");
    assert_eq!(
        evaluated.get(live).map(|(data, _)| data.to_vec()),
        Some(vec![999.0, 999.0, 999.0, 999.0]),
        "a third arena step, still folding the same corrupted, never-recomputed buffer"
    );
}

/// The other half of ROW 174's same contract, mirroring
/// [`build_static_arena_does_not_elide_a_dead_node_that_is_also_a_requested_output`]:
/// a `Constant` with zero consumers is never bound at all --
/// [`bind::bind_plain`]'s reachability pass (ROW 541, `docs/discipline.md`)
/// skips it before `dead_resolved_nodes`/`static_resolved_nodes` ever run
/// over `resolved` -- so it lands in neither `dead` nor `static_nodes`,
/// stronger than ROW 174's original "bound but skipped" guarantee.
#[test]
fn a_dead_constant_is_marked_dead_not_static() {
    let (mut program, _a, constant, live) = constant_feeds_live_program();
    let dead_constant = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            value: 42.0,
        },
    );

    let arena = build_static_arena(&program, &[], &[live])
        .expect("program with an unused constant builds a static arena");
    assert!(
        !arena.dead.contains(&dead_constant),
        "a zero-consumer Constant unreachable from `live` is never bound, so it cannot be a POST-bind dead entry either"
    );
    assert!(
        !arena.static_nodes.contains(&dead_constant),
        "a never-bound Constant must not also be marked static"
    );
    assert!(
        arena.static_nodes.contains(&constant),
        "the live constant is unaffected by its dead sibling"
    );
}

/// A `named` binding whose length no longer matches the shape
/// [`build_static_arena`] fixed for that input must return a named
/// [`TensorError::InputSizeMismatch`], not a silent truncation, a
/// panic, or a wrong-shaped result.
#[test]
fn evaluate_named_with_arena_reports_a_shape_mismatched_rebind_by_name() {
    let (program, sum) = named_add_program();
    let mut arena =
        build_static_arena(&program, &[], &[sum]).expect("small program builds a static arena");

    let wrong_length_a = [1.0f32, 2.0, 3.0];
    let full_length_b = [10.0f32, 20.0, 30.0, 40.0];
    let error =
        evaluate_named_with_arena(&mut arena, &[("a", &wrong_length_a), ("b", &full_length_b)])
            .expect_err("a 3-element rebind against a 4-element input slot must be rejected");
    match error {
        TensorError::InputSizeMismatch {
            expected, found, ..
        } => {
            assert_eq!(expected, 4, "the arena's own fixed slot size for `a`");
            assert_eq!(found, 3, "the mismatched rebind's own length");
        }
        other => panic!("expected TensorError::InputSizeMismatch, got {other:?}"),
    }
}

/// `is_quantized_matmul_operand` is called once per candidate node, not
/// once per program — proving the exemption holds for many quantized
/// weights at once (a real checkpoint's 217 `Q4_K` tensors, not the
/// single-weight case the earlier tests above cover), and that each
/// node's own shape is judged independently: a second, unrelated
/// `Multiply`-then-`Add` matmul with its own `UInt8` weight is exempted
/// alongside the first when both are named, and rejected on its own
/// when only the first is named.
#[test]
fn reject_non_float32_exempts_many_independent_quantized_weights() {
    let mut program = Vec::new();
    let weight_a = block(&mut program, DType::UInt8, &[Extent::Static(4)]);
    let activation_a = f32_block(&mut program, &[Extent::Static(4)]);
    let product_a = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight_a, IndexMap::Affine(map::projection(1, &[0]))),
                (activation_a, IndexMap::Affine(map::projection(1, &[0]))),
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
            operand: product_a,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let weight_b = block(&mut program, DType::UInt8, &[Extent::Static(4)]);
    let activation_b = f32_block(&mut program, &[Extent::Static(4)]);
    let product_b = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight_b, IndexMap::Affine(map::projection(1, &[0]))),
                (activation_b, IndexMap::Affine(map::projection(1, &[0]))),
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
            operand: product_b,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let mut only_a = BTreeSet::new();
    only_a.insert(weight_a);
    assert!(
        reject_non_float32(&program, &only_a).is_err(),
        "weight_b is UInt8 and unexempted, so the program must still be rejected"
    );

    let mut both = BTreeSet::new();
    both.insert(weight_a);
    both.insert(weight_b);
    assert!(
        reject_non_float32(&program, &both).is_ok(),
        "two independent matmul-shaped quantized weights must both be exempted when both are named"
    );
}

#[test]
fn matmul_q4k_f32_rejects_a_row_length_not_a_block_multiple() {
    let weights = vec![0u8; Q4K_BLOCK_BYTES + 1];
    let activation = vec![0.0f32; Q4K_BLOCK_ELEMENTS];
    let error = matmul_q4k_f32(&weights, 1, &activation).unwrap_err();
    assert_eq!(
        error,
        TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q4_k block size",
        }
    );
}

#[test]
fn matmul_q4k_f32_rejects_an_activation_length_mismatch() {
    let weights = vec![0u8; Q4K_BLOCK_BYTES];
    let activation = vec![0.0f32; Q4K_BLOCK_ELEMENTS - 1];
    let error = matmul_q4k_f32(&weights, 1, &activation).unwrap_err();
    assert_eq!(
        error,
        TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        }
    );
}

fn block(program: &mut Vec<Op>, dtype: DType, shape: &[Extent]) -> NodeId {
    append(
        program,
        Op::Input {
            dtype,
            shape: shape.to_vec(),
            name: None,
        },
    )
}

fn f32_block(program: &mut Vec<Op>, shape: &[Extent]) -> NodeId {
    block(program, DType::Float32, shape)
}

/// The last node `program` builds -- what a fixture that appends its
/// nodes in dependency order and never binds anything past its own
/// "answer" treats as the requested output. `bind::bind_plain`'s
/// reachability pass (ROW 541, `docs/discipline.md`) binds only what
/// `outputs` names, so a direct `bind::bind` call in this module must
/// pass this instead of `&[]` -- an empty `outputs` correctly binds
/// nothing now.
fn terminal(program: &[Op]) -> NodeId {
    NodeId((program.len() - 1) as u32)
}

/// `run_iota`'s whole contract: `output[i] = i`, evaluated through the
/// real `evaluate` entry point (not the internal `run_node_into` alone),
/// proving the leaf materializes with no external `blocks` entry — the
/// same guarantee `causal_attention.toml`'s `query_index`/`key_index`
/// nodes rely on.
#[test]
fn an_iota_evaluates_to_its_own_position() {
    let mut program = Vec::new();
    let iota = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(6),
        },
    );

    let evaluated = evaluate(&program, &[], &[], &[]).expect("a bare iota evaluates");
    assert_eq!(evaluated.root(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let _ = iota;
}

/// The counterpart of [`an_iota_evaluates_to_its_own_position`] for the
/// other computed leaf: a `Constant` materializes with no external
/// `blocks` entry, and every element is the literal it was built with.
/// Run through the real `evaluate` entry point so the whole
/// bind/schedule path is exercised, not `run_constant` alone.
#[test]
fn a_constant_evaluates_to_its_literal_at_every_position() {
    let mut program = Vec::new();
    append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            value: 0.088_388_35,
        },
    );

    let evaluated = evaluate(&program, &[], &[], &[]).expect("a bare constant evaluates");
    assert_eq!(evaluated.root(), &[0.088_388_35; 4]);
}

/// A rank-0 `Constant` is the shape every scalar literal in
/// `spec.rs` uses: one element, and an empty operand side (`"->i"`)
/// broadcasts it across any consumer's iteration space.
#[test]
fn a_rank_zero_constant_broadcasts_into_a_higher_rank_consumer() {
    let mut program = Vec::new();
    let scale = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 3.0,
        },
    );
    let iota = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(4),
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (iota, IndexMap::Affine(crate::map::projection(1, &[0]))),
                (scale, IndexMap::Affine(crate::map::projection(1, &[]))),
            ],
            name: None,
        },
    );

    let evaluated = evaluate(&program, &[], &[], &[]).expect("rank-0 constant broadcasts");
    assert_eq!(evaluated.root(), &[0.0, 3.0, 6.0, 9.0]);
}

/// `-inf` is the literal the causal mask needs and the one an integer
/// `Iota` derivation reached only through `Reciprocal(-0.0)`. It must
/// survive the leaf verbatim.
#[test]
fn a_constant_carries_negative_infinity_verbatim() {
    let mut program = Vec::new();
    append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(2)],
            value: f32::NEG_INFINITY,
        },
    );

    let evaluated = evaluate(&program, &[], &[], &[]).expect("a -inf constant evaluates");
    assert!(
        evaluated
            .root()
            .iter()
            .all(|value| *value == f32::NEG_INFINITY)
    );
}

/// `reduce_dot_binary_monomorphic`'s `(true, true)` arm reassociates the
/// sum (`DOT_LANES` independent partial accumulators, ROW 12,
/// `proxima-tensor/docs/discipline.md`) — bit-exactness against
/// [`naive_matmul`]'s strict left-to-right fold is no longer the bar for
/// the transposed-RHS (reduce_dot) path, same as Accelerate/OpenBLAS/
/// ggml. Returns the measured max relative error so callers can log it.
fn assert_all_close(actual: &[f32], expected: &[f32], relative_tolerance: f32) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut max_relative_error = 0.0f32;
    for (&value, &reference) in actual.iter().zip(expected) {
        let scale = reference.abs().max(1.0);
        let relative_error = (value - reference).abs() / scale;
        max_relative_error = max_relative_error.max(relative_error);
        assert!(
            relative_error <= relative_tolerance,
            "relative error {relative_error} exceeds tolerance {relative_tolerance} \
             (actual={value}, expected={reference})"
        );
    }
    max_relative_error
}

fn naive_matmul(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += lhs[row * k + inner] * rhs[inner * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    out
}

fn matmul_program(m: u32, k: u32, n: u32, symbolic: bool) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs_shape = if symbolic {
        alloc::vec![Extent::Symbolic(0), Extent::Static(k)]
    } else {
        alloc::vec![Extent::Static(m), Extent::Static(k)]
    };
    let lhs = f32_block(&mut program, &lhs_shape);
    let rhs = f32_block(&mut program, &[Extent::Static(k), Extent::Static(n)]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    (program, sum)
}

/// Same contraction as [`matmul_program`], RHS stored `[n, k]` instead
/// of `[k, n]` (ggml's own `mul_mat` convention) — exercises
/// [`run_reduce`]'s reduction-dim fast path
/// (`proxima-tensor/docs/discipline.md` ROW 10/11): the width dim `n` is not
/// contiguous on the RHS operand here, but the contraction dim `k` is.
fn matmul_program_rhs_transposed(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = f32_block(&mut program, &[Extent::Static(m), Extent::Static(k)]);
    let rhs = f32_block(&mut program, &[Extent::Static(n), Extent::Static(k)]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[1, 2]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul_rhs_transposed".into()),
        }),
    );
    (program, sum)
}

/// `sum_e lhs[e] * rhs[h, e, d]` over iteration space `(h, e, d)`: `lhs`
/// broadcasts over the leading axis `h` AND the width axis `d` (varies
/// only along the reduction axis `e`, ROW 35's `hidden` operand
/// shape), `rhs` varies over all three (ROW 35's per-head `attn_q`
/// weight slice, `[heads, embed, head_dim]`). `sum` is the ONLY
/// consumer of the `Multiply`, so `bind` fuses it into the `Reduce`'s
/// own `BoundOp` when `sum` is the sole requested output — the exact
/// shape `width_tile_plan`'s single-non-degenerate-leading-axis branch
/// (`cpu.rs`) reaches, and the one it silently mishandled before that
/// branch verified `layout_b.stride(leading_dim) == 0`.
fn batched_matmul_program_with_per_row_weight(
    rows: u32,
    contraction: u32,
    width: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = f32_block(&mut program, &[Extent::Static(contraction)]);
    let rhs = f32_block(
        &mut program,
        &[
            Extent::Static(rows),
            Extent::Static(contraction),
            Extent::Static(width),
        ],
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(3, &[1]))),
                (rhs, IndexMap::Affine(map::projection(3, &[0, 1, 2]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: Some("batched_matmul_per_row_weight".into()),
        }),
    );
    (program, sum)
}

/// Same per-row-weight bug shape as [`batched_matmul_program_with_per_row_weight`],
/// but the reduction is split across TWO axes (`k_outer`, `k_inner`) that
/// `composed_reduction_stride` must fold into one virtual `k` before
/// `width_tile_plan`'s leading-axis guard is ever reached — the
/// `composed_reduction_stride` widening task's own shape
/// (`attn_o`'s `[heads, head_dim]`), paired with a deliberately
/// row-varying `b` to prove that widening a DIFFERENT axis (reduction,
/// not leading) never bypasses the `layout_b.stride(leading_dim) == 0`
/// check the leading axis itself still enforces.
fn batched_matmul_program_with_per_row_weight_composed_reduction(
    rows: u32,
    k_outer: u32,
    k_inner: u32,
    width: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = f32_block(
        &mut program,
        &[Extent::Static(k_outer), Extent::Static(k_inner)],
    );
    let rhs = f32_block(
        &mut program,
        &[
            Extent::Static(rows),
            Extent::Static(k_outer),
            Extent::Static(k_inner),
            Extent::Static(width),
        ],
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(4, &[1, 2]))),
                (rhs, IndexMap::Affine(map::projection(4, &[0, 1, 2, 3]))),
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
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 3])),
            keep: Keep::Reduce,
            name: Some("batched_matmul_per_row_weight_composed_reduction".into()),
        }),
    );
    (program, sum)
}

/// Plain `[batch, seq, k] @ [k, n]` GEMM -- BGE's own `MatMul` shape
/// once `batch_size` stops being elided (extent > 1): the weight
/// (`rhs`) is invariant over BOTH leading axes, the exact `(0, 0)` case
/// the batch-widen fix (`composed_reduction_stride` reused inside
/// `width_tile_plan`'s two-leading-axis arm, `cpu.rs`) exists to merge
/// into one flat leading axis rather than decline. `batch=4, seq=8`
/// gives two non-degenerate leading axes (32 rows once merged), `k=6`
/// short enough to keep the reference loop readable, `n=16` exactly
/// `WIDTH_TILE_VECS * 4` -- the main tile floor, same width the
/// single-leading-axis regressions above use.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
fn two_leading_axes_shared_weight_matmul_program(
    batch: u32,
    seq: u32,
    contraction: u32,
    width: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = f32_block(
        &mut program,
        &[
            Extent::Static(batch),
            Extent::Static(seq),
            Extent::Static(contraction),
        ],
    );
    let rhs = f32_block(
        &mut program,
        &[Extent::Static(contraction), Extent::Static(width)],
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(4, &[0, 1, 2]))),
                (rhs, IndexMap::Affine(map::projection(4, &[2, 3]))),
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
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 1, 3])),
            keep: Keep::Reduce,
            name: Some("two_leading_axes_shared_weight_matmul".into()),
        }),
    );
    (program, sum)
}

/// Same shared-weight GEMM shape as
/// [`two_leading_axes_shared_weight_matmul_program`], with a THIRD
/// non-degenerate leading axis prepended (`outer`) -- BGE's own
/// `attn_qk`/`attn_v` shape (`[heads, batch*seq]` effectively, 3
/// leading axes once batch stops eliding), which `width_tile_plan`'s
/// `non_degenerate_leading.len() > 2` guard still declines (`AxesShape`,
/// `cpu.rs`) rather than guesses at a merge with no proven shape. Pins
/// the KNOWN remaining decline this task's fix did not close.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
fn three_leading_axes_matmul_program(
    outer: u32,
    batch: u32,
    seq: u32,
    contraction: u32,
    width: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = f32_block(
        &mut program,
        &[
            Extent::Static(outer),
            Extent::Static(batch),
            Extent::Static(seq),
            Extent::Static(contraction),
        ],
    );
    let rhs = f32_block(
        &mut program,
        &[Extent::Static(contraction), Extent::Static(width)],
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(5, &[0, 1, 2, 3]))),
                (rhs, IndexMap::Affine(map::projection(5, &[3, 4]))),
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
            in_map: IndexMap::Affine(map::projection(5, &[0, 1, 2, 3, 4])),
            out_map: IndexMap::Affine(map::projection(5, &[0, 1, 2, 4])),
            keep: Keep::Reduce,
            name: Some("three_leading_axes_matmul".into()),
        }),
    );
    (program, sum)
}

/// Same 3-non-degenerate-leading-axis shape as
/// [`three_leading_axes_matmul_program`], but `rhs` now varies over the
/// two OUTER axes (`outer`, `batch`) too, invariant only on `seq` (the
/// row axis) -- BGE's own `attn_qk`/`attn_v` shape at batch > 1
/// (`[batch, heads, seq_q]`), established from `lower_matmul`'s own
/// affine projections (`proxima-onnx/src/lower.rs:1084-1200`): K/V's
/// pattern always skips the row axis and always carries `batch` and
/// `heads`. `three_leading_axes_matmul_program`'s all-invariant `rhs`
/// cannot stand in for this -- a genuinely different stride shape, and
/// the one the three-axis-merge task (2026-09-02) closes.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
fn three_leading_axes_batched_matmul_program(
    outer: u32,
    batch: u32,
    seq: u32,
    contraction: u32,
    width: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = f32_block(
        &mut program,
        &[
            Extent::Static(outer),
            Extent::Static(batch),
            Extent::Static(seq),
            Extent::Static(contraction),
        ],
    );
    let rhs = f32_block(
        &mut program,
        &[
            Extent::Static(outer),
            Extent::Static(batch),
            Extent::Static(contraction),
            Extent::Static(width),
        ],
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(5, &[0, 1, 2, 3]))),
                (rhs, IndexMap::Affine(map::projection(5, &[0, 1, 3, 4]))),
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
            in_map: IndexMap::Affine(map::projection(5, &[0, 1, 2, 3, 4])),
            out_map: IndexMap::Affine(map::projection(5, &[0, 1, 2, 4])),
            keep: Keep::Reduce,
            name: Some("three_leading_axes_batched_matmul".into()),
        }),
    );
    (program, sum)
}

/// `table[ids[s], d]` over iteration space `(s, d)`: dim 0 (vocab) is
/// gathered by `ids`, dim 1 (feature) is a plain projection.
fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let table = f32_block(&mut program, &[Extent::Static(vocab), Extent::Static(dim)]);
    let ids = block(&mut program, DType::Int32, &[Extent::Static(seq)]);
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: map::IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
                map::AxisIndex::default(),
                map::AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let gathered = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(table, gathered_map)],
            name: None,
        },
    );
    (program, gathered)
}

/// `sum_k table[ids[i], k] * weight[k, j]` — an embedding lookup fused
/// straight into a contraction, mirroring [`matmul_program`] with `lhs`
/// replaced by a gather.
fn embedding_matmul_program(vocab: u32, embed_dim: u32, seq: u32, out_dim: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let table = f32_block(
        &mut program,
        &[Extent::Static(vocab), Extent::Static(embed_dim)],
    );
    let ids = block(&mut program, DType::Int32, &[Extent::Static(seq)]);
    let weight = f32_block(
        &mut program,
        &[Extent::Static(embed_dim), Extent::Static(out_dim)],
    );

    let gather_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(3, &[0]),
        base: map::IndexPattern {
            iter_rank: 3,
            axes: alloc::vec![
                map::AxisIndex::default(),
                map::AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let weight_map = IndexMap::Affine(map::projection(3, &[2, 1]));

    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(table, gather_map), (weight, weight_map)],
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
            name: Some("embedding_matmul".into()),
        }),
    );
    program
}

#[test]
fn embedding_lookup_matches_a_hand_written_reference() {
    let (vocab, dim, seq) = (50_000usize, 8usize, 4usize);
    let (program, gathered) = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);

    let table_data: Vec<f32> = (0..vocab * dim).map(|value| (value % 97) as f32).collect();
    let ids_data = [3.0f32, 49_999.0, 12_345.0, 0.0];
    let evaluated = evaluate(&program, &[], &[&table_data, &ids_data], &[])
        .expect("embedding lookup evaluates");

    let mut reference = vec![0.0f32; seq * dim];
    for (row, &id) in ids_data.iter().enumerate() {
        let vocab_index = id as usize;
        reference[row * dim..(row + 1) * dim]
            .copy_from_slice(&table_data[vocab_index * dim..(vocab_index + 1) * dim]);
    }
    assert_eq!(evaluated.shape(), &[seq as u64, dim as u64]);
    assert_eq!(evaluated.root(), reference.as_slice());
    let _ = gathered;
}

#[test]
fn a_fetched_index_past_the_extent_is_a_real_error_not_ub() {
    let (program, _gathered) = embedding_lookup_program(4, 2, 1);
    let table_data: Vec<f32> = (0..8).map(|value| value as f32).collect();
    let ids_data = [4.0f32]; // extent is 4: 0..=3 are valid, 4 is not
    let error = evaluate(&program, &[], &[&table_data, &ids_data], &[])
        .expect_err("out-of-range fetched index is rejected");
    assert!(
        matches!(error, TensorError::GatherIndexOutOfRange { .. }),
        "{error}"
    );
}

#[test]
fn a_gather_fused_into_a_fold_matches_a_hand_written_embedding_matmul_reference() {
    let (vocab, embed_dim, seq, out_dim) = (100usize, 6usize, 4usize, 3usize);
    let program =
        embedding_matmul_program(vocab as u32, embed_dim as u32, seq as u32, out_dim as u32);

    let shapes = shape::infer(&program, &[]).expect("embedding matmul infers");
    let resolved = bind::bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("embedding matmul resolves");
    assert_eq!(
        resolved.len(),
        1,
        "the gather zip must fuse into the fold, not materialize separately"
    );
    assert!(matches!(resolved[0].kind, BoundOpKind::Reduce { .. }));

    let table_data: Vec<f32> = (0..vocab * embed_dim)
        .map(|value| (value % 13) as f32)
        .collect();
    let ids_data = [3.0f32, 99.0, 50.0, 0.0];
    let weight_data: Vec<f32> = (0..embed_dim * out_dim)
        .map(|value| (value % 5) as f32)
        .collect();

    let evaluated = evaluate(&program, &[], &[&table_data, &ids_data, &weight_data], &[])
        .expect("embedding matmul evaluates");

    let mut reference = vec![0.0f32; seq * out_dim];
    for (row, &id) in ids_data.iter().enumerate() {
        let vocab_index = id as usize;
        for col in 0..out_dim {
            let mut total = 0.0f32;
            for k in 0..embed_dim {
                total += table_data[vocab_index * embed_dim + k] * weight_data[k * out_dim + col];
            }
            reference[row * out_dim + col] = total;
        }
    }
    assert_eq!(evaluated.root(), reference.as_slice());
}

#[proxima::test]
#[case::one_worker(1)]
#[case::two_workers(2)]
#[case::three_workers(3)]
async fn evaluate_parallel_matches_evaluate_for_a_gather_program(#[case] workers: usize) {
    let (vocab, embed_dim, seq, out_dim) = (100usize, 6usize, 4usize, 3usize);
    let program =
        embedding_matmul_program(vocab as u32, embed_dim as u32, seq as u32, out_dim as u32);
    let table_data: Vec<f32> = (0..vocab * embed_dim)
        .map(|value| (value % 13) as f32)
        .collect();
    let ids_data = [3.0f32, 99.0, 50.0, 0.0];
    let weight_data: Vec<f32> = (0..embed_dim * out_dim)
        .map(|value| (value % 5) as f32)
        .collect();

    assert_parallel_matches_sequential(
        &program,
        &[],
        &[&table_data, &ids_data, &weight_data],
        &[],
        workers,
    );
}

#[test]
fn a_gather_program_past_the_parallel_threshold_actually_splits_and_still_matches_sequential() {
    let (vocab, embed_dim, seq, out_dim) = (200usize, 64usize, 128usize, 64usize);
    let program =
        embedding_matmul_program(vocab as u32, embed_dim as u32, seq as u32, out_dim as u32);
    let table_data: Vec<f32> = (0..vocab * embed_dim)
        .map(|value| (value % 13) as f32)
        .collect();
    let ids_data: Vec<f32> = (0..seq).map(|value| (value % vocab) as f32).collect();
    let weight_data: Vec<f32> = (0..embed_dim * out_dim)
        .map(|value| (value % 5) as f32)
        .collect();

    let shapes = shape::infer(&program, &[]).expect("infers");
    let resolved = bind::bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("resolves");
    assert_eq!(resolved.len(), 1, "fused into one reduction node");
    assert!(
        element_count(&resolved[0].extents) >= PARALLEL_THRESHOLD,
        "this size must clear the threshold or this test proves nothing about the \
         threaded path"
    );
    assert!(
        resolved[0].split(4).is_some(),
        "the node must actually be splittable for the threaded path to run"
    );

    let workers = NonZeroUsize::new(4).expect("4 is nonzero");
    assert_parallel_matches_sequential(
        &program,
        &[],
        &[&table_data, &ids_data, &weight_data],
        &[],
        workers.get(),
    );
}

#[test]
fn fused_matmul_matches_a_naive_triple_loop() {
    let (m, k, n) = (4usize, 3usize, 5usize);
    let (program, sum) = matmul_program(m as u32, k as u32, n as u32, false);
    let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("matmul evaluates");
    assert_eq!(evaluated.shape(), &[m as u64, n as u64]);
    assert_eq!(
        evaluated.root(),
        naive_matmul(&lhs, &rhs, m, k, n).as_slice()
    );
    let _ = sum;
}

/// Regression for the bug `omega`'s `backend_parity`/`metal_real_forward`
/// real-forward-graph gates caught: `relative=1.3264047 max_diff=6.9346437`
/// between CPU and Metal, traced to the FIRST divergent node in the
/// mistral cached-forward program (a `Reduce` fusing an RoPE-projection
/// `Multiply` whose second operand varies per attention head). Metal was
/// correct; the CPU width-tile fast path (`width_tile_plan`, `cpu.rs`)
/// was wrong — its single-leading-axis branch never checked that the
/// "weight" operand (`b`, reused from the SAME base for every row by
/// `gemm_width_tile_neon`, which carries no `row_stride_b`) was actually
/// row-invariant, so every row silently read row 0's slice of `b`.
///
/// `rows=8` spans two full `WIDTH_TILE_ROWS=4` tiles plus none left over,
/// `width=16` is exactly `WIDTH_TILE_VECS * 4` (the fast path's minimum),
/// and `contraction=6` is short enough to keep the reference loop
/// readable — this is the smallest shape that still reaches
/// `width_tile_plan`'s single-leading-axis branch with a genuinely
/// row-varying `b`. Requesting `sum` alone as the sole output is what
/// makes `bind` fuse the `Multiply` into the `Reduce`'s own `BoundOp` in
/// the first place (`requesting_the_intermediate_elementwise_op_as_an_output_prevents_fusion`
/// in `bind.rs` proves the converse).
#[test]
fn a_reduce_fusing_a_per_row_weight_operand_computes_a_distinct_value_per_row() {
    let (rows, contraction, width) = (8u32, 6u32, 16u32);
    let (program, sum) = batched_matmul_program_with_per_row_weight(rows, contraction, width);

    let lhs: Vec<f32> = (0..contraction).map(|value| 1.0 + value as f32).collect();
    let rhs: Vec<f32> = (0..rows * contraction * width)
        .map(|value| value as f32 * 0.001)
        .collect();

    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[])
        .expect("the fused per-row-weight reduce evaluates");
    assert_eq!(evaluated.shape(), &[rows as u64, width as u64]);

    let mut reference = vec![0.0f32; (rows * width) as usize];
    for row in 0..rows {
        for col in 0..width {
            let mut total = 0.0f32;
            for k in 0..contraction {
                let lhs_value = lhs[k as usize];
                let rhs_value = rhs[(row * contraction * width + k * width + col) as usize];
                total += lhs_value * rhs_value;
            }
            reference[(row * width + col) as usize] = total;
        }
    }
    assert_all_close(evaluated.root(), &reference, 1e-5);

    // the exact shape of the bug this guards: every row collapsing to
    // row 0's value because `b`'s base never advanced past the first
    // row. Assert the FIRST divergent row directly rather than trusting
    // the aggregate `assert_all_close` above alone.
    let output = evaluated.root();
    let row0 = &output[..width as usize];
    for row in 1..rows as usize {
        let this_row = &output[row * width as usize..(row + 1) * width as usize];
        assert_ne!(
            this_row, row0,
            "row {row} collapsed to row 0's value -- the per-row weight operand was not advanced"
        );
    }
    let _ = sum;
}

/// `composed_reduction_stride` (`perf/width-gate-decline`) widened
/// `width_tile_plan` to fold a two-axis reduction into one virtual `k` --
/// the same shape `attn_o`'s `[heads, head_dim]` weight reduce composes.
/// That widening touches the REDUCTION axes only; this proves it can
/// never smuggle a row-varying `b` past the leading-axis guard, since
/// the guard runs on `layout_b.stride(leading_dim)` before the composed
/// reduction stride is ever computed. `k_outer=3, k_inner=4` gives a
/// row-major-composable `contraction=12`; `width=16` keeps this test on
/// the main (`VECS=4`) tile, isolating the reduction-axis widening from
/// the narrow-width widening covered separately below.
#[test]
fn a_composed_two_axis_reduction_fusing_a_per_row_weight_operand_computes_a_distinct_value_per_row()
{
    let (rows, k_outer, k_inner, width) = (8u32, 3u32, 4u32, 16u32);
    let contraction = k_outer * k_inner;
    let (program, sum) = batched_matmul_program_with_per_row_weight_composed_reduction(
        rows, k_outer, k_inner, width,
    );

    let lhs: Vec<f32> = (0..contraction).map(|value| 1.0 + value as f32).collect();
    let rhs: Vec<f32> = (0..rows * contraction * width)
        .map(|value| value as f32 * 0.001)
        .collect();

    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[])
        .expect("the composed-reduction per-row-weight reduce evaluates");
    assert_eq!(evaluated.shape(), &[rows as u64, width as u64]);

    let mut reference = vec![0.0f32; (rows * width) as usize];
    for row in 0..rows {
        for col in 0..width {
            let mut total = 0.0f32;
            for k in 0..contraction {
                let lhs_value = lhs[k as usize];
                let rhs_value = rhs[(row * contraction * width + k * width + col) as usize];
                total += lhs_value * rhs_value;
            }
            reference[(row * width + col) as usize] = total;
        }
    }
    assert_all_close(evaluated.root(), &reference, 1e-5);

    let output = evaluated.root();
    let row0 = &output[..width as usize];
    for row in 1..rows as usize {
        let this_row = &output[row * width as usize..(row + 1) * width as usize];
        assert_ne!(
            this_row, row0,
            "row {row} collapsed to row 0's value -- the composed-reduction widening let a per-row weight operand through unguarded"
        );
    }
    let _ = sum;
}

/// `width_tile_vecs_for` (`perf/narrow-tile`) widened `width_tile_plan`
/// to admit `width` below the main tile's `WIDTH_TILE_VECS * 4 == 16`
/// floor (`VECS=2`/`VECS=1`), unlocking nodes that used to decline at
/// `NarrowWidth` before ever reaching the leading-axis guard. `width=8`
/// selects `VECS=2` -- BGE's own `attn_qk` `N=8` sentence
/// (`width_tile_vecs_for`'s doc) -- paired with the same row-varying `b`
/// shape as the original regression, proving the narrow-width path is
/// declined exactly like the main-tile path rather than exempted from
/// the guard.
#[test]
fn a_narrow_width_reduce_fusing_a_per_row_weight_operand_computes_a_distinct_value_per_row() {
    let (rows, contraction, width) = (8u32, 6u32, 8u32);
    let (program, sum) = batched_matmul_program_with_per_row_weight(rows, contraction, width);

    let lhs: Vec<f32> = (0..contraction).map(|value| 1.0 + value as f32).collect();
    let rhs: Vec<f32> = (0..rows * contraction * width)
        .map(|value| value as f32 * 0.001)
        .collect();

    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[])
        .expect("the narrow-width per-row-weight reduce evaluates");
    assert_eq!(evaluated.shape(), &[rows as u64, width as u64]);

    let mut reference = vec![0.0f32; (rows * width) as usize];
    for row in 0..rows {
        for col in 0..width {
            let mut total = 0.0f32;
            for k in 0..contraction {
                let lhs_value = lhs[k as usize];
                let rhs_value = rhs[(row * contraction * width + k * width + col) as usize];
                total += lhs_value * rhs_value;
            }
            reference[(row * width + col) as usize] = total;
        }
    }
    assert_all_close(evaluated.root(), &reference, 1e-5);

    let output = evaluated.root();
    let row0 = &output[..width as usize];
    for row in 1..rows as usize {
        let this_row = &output[row * width as usize..(row + 1) * width as usize];
        assert_ne!(
            this_row, row0,
            "row {row} collapsed to row 0's value -- the narrow-width widening let a per-row weight operand through unguarded"
        );
    }
    let _ = sum;
}

// batch-seal task (2026-09-02): a `width_tile_plan` decline is INVISIBLE
// to every correctness assertion above -- the generic scalar interpreter
// computes the exact same numbers as the tile, just slower. Three
// instances of exactly this shape landed in one day (36/96 declined at
// batch=1 `AxesShape`, 24/96 attention-node declines, then 96/96 at
// batch>1) and every one passed every existing test; the only thing that
// ever caught them was a human reading a census example's stdout. The
// two tests below assert the ENGAGEMENT COUNT directly so the next
// instance of this shape fails `cargo nextest`, not a benchmark.
//
// pipe question, in writing: is either test a new library type? No --
// both are `#[test]` functions built entirely from primitives this file
// already has (`Op::Elementwise`/`Op::Reduce`, `evaluate`,
// `width_tile_counters`, `instrument::width_tile_decline_snapshot`).
// Nothing here is a combinator, a wrapper, or a coercion host; there is
// no call a caller could make that did not already exist. Not minted.

/// Positive ratchet: BGE's own plain `[batch, seq, k] @ [k, n]` MatMul
/// shape (weight invariant over BOTH leading axes, the `(0, 0)` case the
/// batch-widen fix engages) must route through `width_tile_plan`, not
/// just compute the right answer. Fails the moment a future change
/// re-introduces the 96/96-at-batch>1 regression this task sealed.
#[test]
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
fn a_two_leading_axis_shared_weight_reduce_engages_width_tile_and_matches_naive_matmul() {
    let (batch, seq, contraction, width) = (4u32, 8u32, 6u32, 16u32);
    let (program, sum) =
        two_leading_axes_shared_weight_matmul_program(batch, seq, contraction, width);

    let rows = batch * seq;
    let lhs: Vec<f32> = (0..rows * contraction)
        .map(|value| value as f32 * 0.01 + 1.0)
        .collect();
    let rhs: Vec<f32> = (0..contraction * width)
        .map(|value| value as f32 * 0.001)
        .collect();

    instrument::reset_width_tile_decline();
    let (gate_passes_before, invocations_before, _) = width_tile_counters();
    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[])
        .expect("the two-leading-axis shared-weight reduce evaluates");
    let (gate_passes_after, invocations_after, _) = width_tile_counters();

    assert_eq!(evaluated.shape(), &[batch as u64, seq as u64, width as u64]);
    let reference = naive_matmul(
        &lhs,
        &rhs,
        rows as usize,
        contraction as usize,
        width as usize,
    );
    assert_all_close(evaluated.root(), &reference, 1e-5);

    let gate_delta = gate_passes_after - gate_passes_before;
    let invocation_delta = invocations_after - invocations_before;
    assert!(
        gate_delta > 0 && invocation_delta > 0,
        "width_tile_plan declined the two-leading-axis (batch, seq) shared-weight GEMM -- \
         gate_delta={gate_delta} invocation_delta={invocation_delta}, expected both > 0. \
         `assert_all_close` above would still pass on the generic-interpreter fallback -- \
         THIS is the assertion that catches the regression"
    );
    let declines = instrument::width_tile_decline_snapshot();
    assert!(
        declines.iter().all(|&(node, ..)| node != sum.0),
        "the two-leading-axis shared-weight node declined instead of engaging: {declines:?}"
    );
}

/// Negative ratchet, UPDATED by the three-axis-merge task (2026-09-02):
/// this is NO LONGER "3 leading axes always decline" -- BGE's own
/// `attn_qk`/`attn_v` shape (1 row axis + 2 `b`-varying outer axes that
/// compose) now engages, see
/// `a_three_leading_axis_batched_reduce_engages_width_tile_and_matches_naive_matmul`
/// below. This test instead pins the OTHER 3-axis shape,
/// `three_leading_axes_matmul_program`'s `rhs` invariant over ALL THREE
/// leading axes (a plain shared-weight GEMM, not attention) -- no axis
/// is the unique row candidate (`layout_b.stride == 0` on all three),
/// so `width_tile_plan`'s 3-axis arm still declines it, unchanged.
/// Pinned explicitly so a future widening past "exactly one row axis"
/// must re-derive this test's expected engagement rather than silently
/// making it pass for the wrong reason.
#[test]
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
fn a_three_leading_axis_reduce_declines_width_tile_with_axes_shape_reason() {
    let (outer, batch, seq, contraction, width) = (2u32, 3u32, 4u32, 5u32, 16u32);
    let (program, sum) = three_leading_axes_matmul_program(outer, batch, seq, contraction, width);

    let rows = outer * batch * seq;
    let lhs: Vec<f32> = (0..rows * contraction)
        .map(|value| value as f32 * 0.01 + 1.0)
        .collect();
    let rhs: Vec<f32> = (0..contraction * width)
        .map(|value| value as f32 * 0.001)
        .collect();

    instrument::reset_width_tile_decline();
    let (gate_passes_before, _, _) = width_tile_counters();
    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[])
        .expect("the three-leading-axis reduce evaluates on the generic path");
    let (gate_passes_after, _, _) = width_tile_counters();

    assert_eq!(
        evaluated.shape(),
        &[outer as u64, batch as u64, seq as u64, width as u64]
    );
    let reference = naive_matmul(
        &lhs,
        &rhs,
        rows as usize,
        contraction as usize,
        width as usize,
    );
    assert_all_close(evaluated.root(), &reference, 1e-5);

    assert_eq!(
        gate_passes_after - gate_passes_before,
        0,
        "the all-invariant-rhs 3-leading-axis GEMM engaged width_tile_plan -- if this fires, \
         a shape with no unique row axis started engaging; re-derive this test's expected \
         engagement count rather than deleting the assertion"
    );
    let declines = instrument::width_tile_decline_snapshot();
    let matched = declines
        .iter()
        .find(|&&(node, ..)| node == sum.0)
        .expect("the three-leading-axis node should have recorded a width_tile_plan decline");
    assert_eq!(
        matched.1,
        instrument::WidthDeclineReason::AxesShape,
        "expected AxesShape (no unique row axis among 3 non-degenerate leading axes), got {:?}",
        matched.1
    );
}

/// Positive ratchet: BGE's own `attn_qk`/`attn_v` shape (`[batch, heads,
/// seq_q]`, `b` invariant on `seq_q` alone, varying on `batch` AND
/// `heads`) must route through `width_tile_plan`, not just compute the
/// right answer -- the exact 24-of-96 decline the three-axis-merge task
/// (2026-09-02) closes. Fails the moment a future change re-introduces
/// the 72/96-at-batch>1 regression this task fixed.
#[test]
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
fn a_three_leading_axis_batched_reduce_engages_width_tile_and_matches_naive_matmul() {
    let (outer, batch, seq, contraction, width) = (2u32, 3u32, 4u32, 5u32, 16u32);
    let (program, sum) =
        three_leading_axes_batched_matmul_program(outer, batch, seq, contraction, width);

    let lhs: Vec<f32> = (0..outer * batch * seq * contraction)
        .map(|value| value as f32 * 0.01 + 1.0)
        .collect();
    let rhs: Vec<f32> = (0..outer * batch * contraction * width)
        .map(|value| value as f32 * 0.001)
        .collect();

    instrument::reset_width_tile_decline();
    let (gate_passes_before, invocations_before, _) = width_tile_counters();
    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[])
        .expect("the batched three-leading-axis reduce evaluates");
    let (gate_passes_after, invocations_after, _) = width_tile_counters();

    assert_eq!(
        evaluated.shape(),
        &[outer as u64, batch as u64, seq as u64, width as u64]
    );
    let mut reference = vec![0.0f32; (outer * batch * seq * width) as usize];
    for slice in 0..(outer * batch) as usize {
        let lhs_slice =
            &lhs[slice * (seq * contraction) as usize..(slice + 1) * (seq * contraction) as usize];
        let rhs_slice = &rhs
            [slice * (contraction * width) as usize..(slice + 1) * (contraction * width) as usize];
        let slice_out = naive_matmul(
            lhs_slice,
            rhs_slice,
            seq as usize,
            contraction as usize,
            width as usize,
        );
        reference[slice * (seq * width) as usize..(slice + 1) * (seq * width) as usize]
            .copy_from_slice(&slice_out);
    }
    assert_all_close(evaluated.root(), &reference, 1e-5);

    let gate_delta = gate_passes_after - gate_passes_before;
    let invocation_delta = invocations_after - invocations_before;
    assert!(
        gate_delta > 0 && invocation_delta > 0,
        "width_tile_plan declined the batched three-leading-axis (batch, heads, seq_q) reduce \
         -- gate_delta={gate_delta} invocation_delta={invocation_delta}, expected both > 0. \
         `assert_all_close` above would still pass on the generic-interpreter fallback -- \
         THIS is the assertion that catches the regression"
    );
    let declines = instrument::width_tile_decline_snapshot();
    assert!(
        declines.iter().all(|&(node, ..)| node != sum.0),
        "the batched three-leading-axis node declined instead of engaging: {declines:?}"
    );
}

#[test]
fn fused_matmul_with_transposed_rhs_matches_a_naive_triple_loop() {
    // k=7 (not a multiple of DOT_LANES=4) exercises the fast path's
    // remainder handling; the RHS buffer is the same numbers as
    // `naive_matmul`'s `[k, n]` reference expects, laid out `[n, k]`.
    //
    // WEAKENED (ROW 12, `proxima-tensor/docs/discipline.md`): was
    // `assert_eq!` (bit-exact) against `naive_matmul`'s strict
    // left-to-right fold. `reduce_dot_binary_monomorphic`'s `(true,
    // true)` arm now folds via `DOT_LANES` independent partial
    // accumulators (matches Accelerate/OpenBLAS/ggml practice), which
    // reassociates the sum and can change its bit pattern relative to
    // the naive loop. Switched to a 1e-5 relative-tolerance check.
    // Measured max relative error at this k=7, small-integer-input size
    // was 0.0 (integers this small sum exactly in f32 regardless of
    // grouping) — logged in ROW 12 rather than assumed.
    let (m, k, n) = (4usize, 7usize, 5usize);
    let (program, sum) = matmul_program_rhs_transposed(m as u32, k as u32, n as u32);
    let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
    let rhs_kn: Vec<f32> = (0..k * n).map(|value| value as f32).collect();
    let mut rhs_nk = vec![0.0f32; k * n];
    for row in 0..k {
        for col in 0..n {
            rhs_nk[col * k + row] = rhs_kn[row * n + col];
        }
    }

    let evaluated = evaluate(&program, &[], &[&lhs, &rhs_nk], &[]).expect("matmul evaluates");
    assert_eq!(evaluated.shape(), &[m as u64, n as u64]);
    let reference = naive_matmul(&lhs, &rhs_kn, m, k, n);
    let max_relative_error = assert_all_close(evaluated.root(), &reference, 1e-5);
    println!("k=7 transposed-rhs max_relative_error={max_relative_error}");
    let _ = sum;
}

#[test]
fn fused_matmul_with_transposed_rhs_k1024_within_tolerance_of_a_naive_triple_loop() {
    // k=1024 matches the real GEMM benchmark's contraction length and
    // uses fractional, non-integer data so the sum actually accumulates
    // rounding error under either fold order (unlike the k=7 test's
    // small-integer inputs, whose sums are exact regardless of
    // grouping) — see ROW 12, `proxima-tensor/docs/discipline.md`.
    let (m, k, n) = (8usize, 1024usize, 8usize);
    let (program, _sum) = matmul_program_rhs_transposed(m as u32, k as u32, n as u32);
    let lhs: Vec<f32> = (0..m * k)
        .map(|value| (value as f32 * 0.0137).sin())
        .collect();
    let rhs_kn: Vec<f32> = (0..k * n)
        .map(|value| (value as f32 * 0.0271).cos())
        .collect();
    let mut rhs_nk = vec![0.0f32; k * n];
    for row in 0..k {
        for col in 0..n {
            rhs_nk[col * k + row] = rhs_kn[row * n + col];
        }
    }

    let evaluated = evaluate(&program, &[], &[&lhs, &rhs_nk], &[]).expect("matmul evaluates");
    assert_eq!(evaluated.shape(), &[m as u64, n as u64]);
    let reference = naive_matmul(&lhs, &rhs_kn, m, k, n);
    let max_relative_error = assert_all_close(evaluated.root(), &reference, 1e-5);
    println!("k=1024 transposed-rhs max_relative_error={max_relative_error}");
}

#[test]
fn matmul_binds_a_symbolic_sequence_length_at_eval_time() {
    let (m, k, n) = (4usize, 3usize, 5usize);
    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, true);
    let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

    let evaluated =
        evaluate(&program, &[m as u64], &[&lhs, &rhs], &[]).expect("symbolic matmul evaluates");
    assert_eq!(
        evaluated.root(),
        naive_matmul(&lhs, &rhs, m, k, n).as_slice()
    );
}

#[test]
fn fused_contraction_skips_the_product_tensor() {
    let (m, k, n) = (64usize, 64usize, 64usize);
    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
    let lhs: Vec<f32> = (0..m * k).map(|value| (value % 7) as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| (value % 5) as f32).collect();

    let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("64x64x64 matmul evaluates");
    let reference = naive_matmul(&lhs, &rhs, m, k, n);
    for (row, col) in [(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1)] {
        let index = row * n + col;
        assert_eq!(
            evaluated.root()[index],
            reference[index],
            "corner ({row}, {col})"
        );
    }
}

#[test]
fn bias_add_via_broadcast() {
    let mut program = Vec::new();
    let matrix = f32_block(&mut program, &[Extent::Static(2), Extent::Static(3)]);
    let bias = f32_block(&mut program, &[Extent::Static(3)]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![
                (matrix, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (bias, IndexMap::Affine(map::projection(2, &[1]))),
            ],
            name: None,
        },
    );

    let matrix_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
    let bias_data = [10.0, 20.0, 30.0f32];
    let evaluated =
        evaluate(&program, &[], &[&matrix_data, &bias_data], &[]).expect("bias add evaluates");
    assert_eq!(evaluated.root(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn transpose_via_permuted_map() {
    let mut program = Vec::new();
    let matrix = f32_block(&mut program, &[Extent::Static(2), Extent::Static(3)]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(matrix, IndexMap::Affine(map::projection(2, &[1, 0])))],
            name: None,
        },
    );

    let matrix_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
    let evaluated = evaluate(&program, &[], &[&matrix_data], &[]).expect("transpose evaluates");
    assert_eq!(evaluated.shape(), &[3, 2]);
    assert_eq!(evaluated.root(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn one_dimensional_convolution_via_a_two_term_window_map() {
    // a per-position ("locally connected") kernel: kernel[h, r] pins both
    // iteration dims via pure projection, while signal[h + r] is the
    // two-term windowed access under test.
    let mut program = Vec::new();
    let kernel = f32_block(&mut program, &[Extent::Static(6), Extent::Static(3)]);
    let signal = f32_block(&mut program, &[Extent::Static(8)]);
    let window = IndexMap::Affine(map::affine(
        2,
        &[(&[AxisTerm::scaled(0, 1), AxisTerm::scaled(1, 1)], 0)],
    ));
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (kernel, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (signal, window)
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
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let kernel_data: Vec<f32> = (0..18).map(|value| value as f32).collect();
    let signal_data: Vec<f32> = (0..8).map(|value| value as f32).collect();
    let evaluated =
        evaluate(&program, &[], &[&kernel_data, &signal_data], &[]).expect("conv evaluates");

    let mut reference = vec![0.0f32; 6];
    for (h, slot) in reference.iter_mut().enumerate() {
        for r in 0..3 {
            *slot += kernel_data[h * 3 + r] * signal_data[h + r];
        }
    }
    assert_eq!(evaluated.root(), reference.as_slice());
}

#[test]
fn softmax_end_to_end_matches_a_reference_within_epsilon() {
    let mut program = Vec::new();
    let (n, d) = (2usize, 4usize);
    let input = f32_block(
        &mut program,
        &[Extent::Static(n as u32), Extent::Static(d as u32)],
    );

    let row_map = IndexMap::Affine(map::projection(2, &[0, 1]));
    let broadcast_map = IndexMap::Affine(map::projection(2, &[0]));

    let max = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Maximum,
            init: ReduceInit::NegativeInfinity,
            operand: input,
            in_map: row_map.clone(),
            out_map: broadcast_map.clone(),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let shifted = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Subtract,
            operands: alloc::vec![(input, row_map.clone()), (max, broadcast_map.clone())],
            name: None,
        },
    );
    let exponentiated = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Exponential,
            operands: alloc::vec![(shifted, row_map.clone())],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: exponentiated,
            in_map: row_map.clone(),
            out_map: broadcast_map.clone(),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Divide,
            operands: alloc::vec![(exponentiated, row_map), (sum, broadcast_map)],
            name: None,
        },
    );

    let input_data = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0f32];
    let evaluated = evaluate(&program, &[], &[&input_data], &[]).expect("softmax evaluates");

    let mut reference = vec![0.0f32; n * d];
    for row in 0..n {
        let slice = &input_data[row * d..row * d + d];
        let row_max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = slice.iter().map(|value| (value - row_max).exp()).collect();
        let total: f32 = exps.iter().sum();
        for (col, value) in exps.iter().enumerate() {
            reference[row * d + col] = value / total;
        }
    }

    for (found, expected) in evaluated.root().iter().zip(reference.iter()) {
        assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
    }
}

#[test]
fn cumsum_matches_a_running_sum_reference() {
    let mut program = Vec::new();
    let source = f32_block(&mut program, &[Extent::Static(6)]);
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

    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
    let evaluated = evaluate(&program, &[], &[&data], &[]).expect("cumsum evaluates");

    let mut running = 0.0f32;
    let reference: Vec<f32> = data
        .iter()
        .map(|value| {
            running += value;
            running
        })
        .collect();
    assert_eq!(evaluated.root(), reference.as_slice());
}

#[test]
fn a_chain_of_unary_zips_keeps_peak_live_buffers_small() {
    let mut program = Vec::new();
    let mut current = f32_block(&mut program, &[Extent::Static(4)]);
    for _ in 0..8 {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
    }

    let input = [0.1, 0.2, 0.3, 0.4f32];
    let evaluated = evaluate(&program, &[], &[&input], &[]).expect("tanh chain evaluates");

    let mut reference = input;
    for value in &mut reference {
        for _ in 0..8 {
            *value = value.tanh();
        }
    }
    for (found, expected) in evaluated.root().iter().zip(reference.iter()) {
        assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
    }
    let peak = evaluated
        .peak_live_buffers()
        .expect("evaluate tracks peak live buffers");
    assert!(
        peak <= 3,
        "streaming a chain of 8 unary elementwise ops should not hold one buffer per op, got {peak}"
    );
    let _ = current;
}

#[test]
fn a_chain_of_8_unary_ops_binds_to_one_bound_op_and_the_result_is_unchanged() {
    let mut program = Vec::new();
    let mut current = f32_block(&mut program, &[Extent::Static(4)]);
    for _ in 0..8 {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
    }
    let _ = current;

    let shapes = shape::infer(&program, &[]).expect("tanh chain infers");
    let resolved = bind::bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("tanh chain resolves");
    assert_eq!(
        resolved.len(),
        1,
        "8 chained unary elementwise ops must fuse into one BoundOp"
    );

    let input = [0.1, 0.2, 0.3, 0.4f32];
    let evaluated = evaluate(&program, &[], &[&input], &[]).expect("tanh chain evaluates");
    let mut reference = input;
    for value in &mut reference {
        for _ in 0..8 {
            *value = value.tanh();
        }
    }
    for (found, expected) in evaluated.root().iter().zip(reference.iter()) {
        assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
    }
}

/// `b = a * scale; c = b + bias; d = c * c` — the elementwise-into-
/// elementwise fusion case, not the reduce-over-elementwise one every
/// other fusion test in this crate already covers.
#[test]
fn a_chain_of_elementwise_ops_binds_to_one_bound_op_and_matches_a_hand_reference() {
    let mut program = Vec::new();
    let a = f32_block(&mut program, &[Extent::Static(4)]);
    let scale = f32_block(&mut program, &[Extent::Static(4)]);
    let bias = f32_block(&mut program, &[Extent::Static(4)]);
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let b = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(a, identity()), (scale, identity())],
            name: None,
        },
    );
    let c = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![(b, identity()), (bias, identity())],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(c, identity()), (c, identity())],
            name: None,
        },
    );

    let shapes = shape::infer(&program, &[]).expect("elementwise chain infers");
    let resolved = bind::bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("elementwise chain resolves");
    assert_eq!(
        resolved.len(),
        1,
        "b and c must fuse into d's own BoundOp instead of three separate ones"
    );

    let a_data = [1.0, 2.0, 3.0, 4.0f32];
    let scale_data = [2.0, 0.5, -1.0, 3.0f32];
    let bias_data = [1.0, 1.0, 1.0, 1.0f32];
    let evaluated = evaluate(&program, &[], &[&a_data, &scale_data, &bias_data], &[])
        .expect("elementwise chain evaluates");

    let reference: Vec<f32> = a_data
        .iter()
        .zip(scale_data.iter())
        .zip(bias_data.iter())
        .map(|((a_value, scale_value), bias_value)| {
            let b_value = a_value * scale_value;
            let c_value = b_value + bias_value;
            c_value * c_value
        })
        .collect();
    assert_eq!(evaluated.root(), reference.as_slice());
}

/// `b` feeds two different consumers (`c1` and `c2`), so it must
/// materialize once on its own rather than fuse into either — the
/// multi-use case that forces materialization even though every map
/// involved is a plain identity projection.
#[test]
fn an_elementwise_intermediate_consumed_by_two_ops_still_evaluates_correctly() {
    let mut program = Vec::new();
    let a = f32_block(&mut program, &[Extent::Static(4)]);
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let b = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: alloc::vec![(a, identity())],
            name: None,
        },
    );
    let c1 = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: alloc::vec![(b, identity())],
            name: None,
        },
    );
    let c2 = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: alloc::vec![(b, identity())],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![(c1, identity()), (c2, identity())],
            name: None,
        },
    );

    let shapes = shape::infer(&program, &[]).expect("diamond chain infers");
    let resolved = bind::bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("diamond chain resolves");
    assert_eq!(
        resolved.len(),
        2,
        "b must materialize standalone since it has two distinct consumers"
    );

    let a_data = [0.1, 0.2, 0.3, 0.4f32];
    let evaluated = evaluate(&program, &[], &[&a_data], &[]).expect("diamond chain evaluates");
    let reference: Vec<f32> = a_data
        .iter()
        .map(|value| {
            let b_value = value.tanh();
            -b_value + (1.0 / b_value)
        })
        .collect();
    for (found, expected) in evaluated.root().iter().zip(reference.iter()) {
        assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
    }
}

/// `b` is requested as an output alongside the root `c`: it must stay
/// separately readable and correct rather than disappear into `c`'s
/// fused body.
#[test]
fn a_requested_elementwise_intermediate_stays_readable_and_correct() {
    let mut program = Vec::new();
    let a = f32_block(&mut program, &[Extent::Static(4)]);
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let b = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: alloc::vec![(a, identity())],
            name: None,
        },
    );
    let c = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: alloc::vec![(b, identity())],
            name: None,
        },
    );

    let shapes = shape::infer(&program, &[]).expect("requested-output chain infers");
    let resolved = bind::bind(&program, &shapes, &[b, c], NumericPolicy::bit_exact())
        .expect("requested-output chain resolves");
    assert_eq!(
        resolved.len(),
        2,
        "requesting b as an output must force it to materialize on its own"
    );

    let a_data = [0.1, 0.2, 0.3, 0.4f32];
    let evaluated =
        evaluate(&program, &[], &[&a_data], &[b, c]).expect("requested-output chain evaluates");

    let (b_data, _) = evaluated.get(b).expect("b was requested as an output");
    let b_reference: Vec<f32> = a_data.iter().map(|value| value.tanh()).collect();
    assert_eq!(b_data, b_reference.as_slice());

    let (c_data, _) = evaluated.get(c).expect("c was requested as an output");
    let c_reference: Vec<f32> = b_reference.iter().map(|value| -value).collect();
    assert_eq!(c_data, c_reference.as_slice());
}

#[test]
fn a_requested_intermediate_survives_alongside_the_root() {
    let mut program = Vec::new();
    let source = f32_block(&mut program, &[Extent::Static(4)]);
    let mut current = source;
    let mut nodes = alloc::vec![source];
    for _ in 0..4 {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        nodes.push(current);
    }
    let midpoint = nodes[2];
    let root = current;

    let input = [0.1, 0.2, 0.3, 0.4f32];
    let evaluated = evaluate(&program, &[], &[&input], &[midpoint, root])
        .expect("chain with an output request evaluates");

    let (midpoint_data, _) = evaluated
        .get(midpoint)
        .expect("midpoint survives to the end");
    let mut reference = input;
    for value in &mut reference {
        for _ in 0..2 {
            *value = value.tanh();
        }
    }
    for (found, expected) in midpoint_data.iter().zip(reference.iter()) {
        assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
    }

    let (root_data, _) = evaluated.get(root).expect("root also present");
    let mut full_reference = input;
    for value in &mut full_reference {
        for _ in 0..4 {
            *value = value.tanh();
        }
    }
    for (found, expected) in root_data.iter().zip(full_reference.iter()) {
        assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
    }
}

#[test]
fn wrong_block_count_is_rejected() {
    let mut program = Vec::new();
    f32_block(&mut program, &[Extent::Static(4)]);

    let error = evaluate(&program, &[], &[], &[]).expect_err("one block is required");
    assert!(
        matches!(error, TensorError::InputCountMismatch { .. }),
        "{error}"
    );
}

#[test]
fn wrong_block_size_is_rejected() {
    let mut program = Vec::new();
    f32_block(&mut program, &[Extent::Static(4)]);

    let too_short = [1.0, 2.0f32];
    let error = evaluate(&program, &[], &[&too_short], &[]).expect_err("block is the wrong size");
    assert!(
        matches!(error, TensorError::InputSizeMismatch { .. }),
        "{error}"
    );
}

#[test]
fn a_non_float32_program_is_rejected() {
    let mut program = Vec::new();
    block(&mut program, DType::Int32, &[Extent::Static(4)]);

    let data = [1i32; 0]; // never read: rejected before blocks are consulted
    let _ = data;
    let error = evaluate(&program, &[], &[], &[]).expect_err("int32 is not f32");
    assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
}

/// Builds `s in 0..idx.len() -> out[idx[s]] += src[s]` (`body: Add`,
/// `init: Zero`), `src`/`idx` bound at evaluation time, destination
/// extent `dest_extent`. `idx`'s values ride in the same `f32` buffer
/// convention every other gather/scatter test in this file uses
/// (`map.rs`'s own `IndexMap::Computed` doc: an index value is an exact
/// integer carried as `f32`).
fn scatter_add_program(dest_extent: u32) -> (Vec<Op>, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let source = f32_block(&mut program, &[Extent::Static(4)]);
    let ids = block(&mut program, DType::Int32, &[Extent::Static(4)]);
    let out_map = IndexMap::scatter(ids, map::projection(1, &[0]), 1, &[], 0, dest_extent);
    let scattered = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map,
            keep: Keep::Reduce,
            name: Some("scatter_add".into()),
        }),
    );
    (program, source, ids, scattered)
}

/// The hand-worked example this task's algorithm-development discipline
/// requires, walked exactly: `src=[10,20,30,40]`, `idx=[2,0,2,1]`,
/// destination extent 3, `body: Add`, `init: Zero`.
///
/// Step by step, in iteration order (the order `run_reduce_scatter`
/// actually walks, and the order that makes the collision below
/// deterministic rather than merely "some fold"):
/// - `s=0`: `src[0]=10` -> `idx[0]=2` -> `out[2] = 0 (init) + 10 = 10`
/// - `s=1`: `src[1]=20` -> `idx[1]=0` -> `out[0] = 0 (init) + 20 = 20`
/// - `s=2`: `src[2]=30` -> `idx[2]=2` -> `out[2] = 10 + 30 = 40` (collision)
/// - `s=3`: `src[3]=40` -> `idx[3]=1` -> `out[1] = 0 (init) + 40 = 40`
///
/// Final: `out = [20, 40, 40]`.
#[test]
fn scatter_add_matches_the_hand_worked_example() {
    let (program, _source, _ids, scattered) = scatter_add_program(3);
    let index_values = [2.0f32, 0.0, 2.0, 1.0];
    let source_values = [10.0f32, 20.0, 30.0, 40.0];
    let evaluated = evaluate(
        &program,
        &[],
        &[&source_values, &index_values],
        &[scattered],
    )
    .expect("the hand-worked scatter example evaluates");

    assert_eq!(
        evaluated.root(),
        &[20.0, 40.0, 40.0],
        "out[2] folds src[0] then src[2]; out[0] and out[1] each see one source element"
    );
}

/// A destination wider than the source with no two source elements ever
/// sharing a destination: every cell is either `init`'s identity (`0`,
/// untouched) or exactly one source value, no fold ever runs twice.
#[test]
fn scatter_add_with_no_collisions_places_each_source_element_once() {
    let (program, _source, _ids, scattered) = scatter_add_program(5);
    let index_values = [4.0f32, 1.0, 3.0, 0.0];
    let source_values = [10.0f32, 20.0, 30.0, 40.0];
    let evaluated = evaluate(
        &program,
        &[],
        &[&source_values, &index_values],
        &[scattered],
    )
    .expect("a collision-free scatter evaluates");

    assert_eq!(
        evaluated.root(),
        &[40.0, 20.0, 0.0, 30.0, 10.0],
        "cell 2 is untouched (init's identity, Zero); every other cell sees exactly one source value"
    );
}

/// A fetched destination index outside `[0, dest_extent)` is a real,
/// named error at evaluation time -- the same
/// [`TensorError::GatherIndexOutOfRange`] class an out-of-range *read*
/// (gather) index already raises, reused rather than a second variant
/// for the write side (`map.rs`'s own doc: scatter is the write-side
/// twin of gather via the same [`IndexMap::Computed`] machinery).
#[test]
fn scatter_add_with_an_out_of_range_destination_index_is_rejected() {
    let (program, _source, _ids, scattered) = scatter_add_program(3);
    let index_values = [0.0f32, 1.0, 3.0, 2.0]; // 3 is out of range for extent 3
    let source_values = [10.0f32, 20.0, 30.0, 40.0];
    let error = evaluate(
        &program,
        &[],
        &[&source_values, &index_values],
        &[scattered],
    )
    .expect_err("index 3 is out of range for destination extent 3");
    assert!(
        matches!(
            error,
            TensorError::GatherIndexOutOfRange {
                index: 3,
                extent: 3,
                ..
            }
        ),
        "{error}"
    );
}

/// The composition oracle: [`scatter_add_into_a_known_destination_via_mask_composition`]
/// builds the identical `src`/`idx`/destination-extent-3 scatter-add out
/// of `Iota`+`Equal`+`Multiply`+`Reduce`, with no `IndexMap::Computed`
/// anywhere. Running the SAME fixture through this crate's native
/// forward-scatter (`IndexMap::Computed` as a `Reduce`'s `out_map`) must
/// land on the exact same numbers -- the composition, not a hand
/// computation, is what proves the native path correct.
#[test]
fn native_scatter_matches_the_mask_composition_oracle_on_the_same_fixture() {
    let (native_program, _source, _ids, native_scattered) = scatter_add_program(3);
    let index_values = [0.0f32, 2.0, 0.0, 1.0];
    let source_values = [10.0f32, 20.0, 30.0, 40.0];
    let native = evaluate(
        &native_program,
        &[],
        &[&source_values, &index_values],
        &[native_scattered],
    )
    .expect("the native scatter evaluates");

    let oracle = [40.0f32, 40.0, 20.0];
    assert_eq!(
        native.root(),
        &oracle,
        "native IndexMap::Computed scatter must match the Iota+Equal+Multiply+Reduce oracle"
    );
}

// -- BoundOp::split proof: chunks executed by hand, one buffer, equal
// -- the unsplit result. `resolve.rs` owns the split's geometry; the
// -- interpreter that can actually run a chunk lives only here, std-gated.

#[test]
fn splitting_an_elementwise_node_and_running_its_chunks_matches_the_unsplit_result() {
    let mut program = Vec::new();
    let source = f32_block(&mut program, &[Extent::Static(10), Extent::Static(4)]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: alloc::vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );

    let shapes = shape::infer(&program, &[]).expect("elementwise infers");
    let resolved = bind::bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("elementwise resolves");
    let node = &resolved[0];

    let data: Vec<f32> = (0..40).map(|value| value as f32 * 0.01).collect();
    let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
    buffers[0] = Some(data);

    let unsplit = run_node(node, &buffers).expect("unsplit runs");

    let chunks = node.split(3).expect("extent 10 over 3 parts splits");
    let mut split_output = vec![0.0f32; unsplit.len()];
    let mut remaining = split_output.as_mut_slice();
    for chunk in &chunks {
        let (this_chunk, rest) = remaining.split_at_mut(node_output_len(chunk));
        run_node_into(chunk, &buffers, None, None, None, false, this_chunk).expect("chunk runs");
        remaining = rest;
    }

    assert_eq!(split_output, unsplit);
}

#[test]
fn splitting_a_fused_matmul_reduction_and_running_its_chunks_matches_the_unsplit_result() {
    let (m, k, n) = (8usize, 3usize, 5usize);
    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
    let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

    let shapes = shape::infer(&program, &[]).expect("matmul infers");
    let resolved = bind::bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("matmul resolves");
    assert_eq!(resolved.len(), 1, "fused into one reduction node");
    let node = &resolved[0];

    let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
    buffers[0] = Some(lhs);
    buffers[1] = Some(rhs);

    let unsplit = run_node(node, &buffers).expect("unsplit runs");

    let chunks = node.split(2).expect("8 rows over 2 parts splits");
    let mut split_output = vec![0.0f32; unsplit.len()];
    let mut remaining = split_output.as_mut_slice();
    for chunk in &chunks {
        let (this_chunk, rest) = remaining.split_at_mut(node_output_len(chunk));
        run_node_into(chunk, &buffers, None, None, None, false, this_chunk).expect("chunk runs");
        remaining = rest;
    }

    assert_eq!(split_output, unsplit);
}

// -- evaluate_parallel proof tests: same programs, workers in {1, 2, 3, 8},
// -- bitwise-equal to `evaluate`.

fn assert_parallel_matches_sequential(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
    workers: usize,
) {
    let workers = NonZeroUsize::new(workers).expect("every case here uses a nonzero count");
    let sequential = evaluate(program, symbols, blocks, outputs).expect("sequential evaluates");
    let parallel =
        evaluate_parallel(program, symbols, blocks, outputs, workers).expect("parallel evaluates");

    assert_eq!(parallel.shape(), sequential.shape());
    assert_eq!(parallel.root(), sequential.root());
    for &node in outputs {
        assert_eq!(
            parallel.get(node),
            sequential.get(node),
            "node {node} output diverges"
        );
    }
}

/// [`run_elementwise_dispatch`]'s own cohort path only ever fires
/// inside [`evaluate_quantized`] (the `session: Some(..)` arm
/// `evaluate_parallel` never takes — `evaluate_parallel_matches_evaluate`'s
/// cases above all exercise the pool path, `run_chunks_threaded` with
/// `session: None`, not this one). A large elementwise chain clears
/// `PARALLEL_THRESHOLD` and has `outer_len` (64) comfortably above any
/// worker count tried here, so this is the one test that actually
/// drives a cohort round for [`ElementwiseRowRound`] and checks its
/// output is bit-identical — `assert_eq!`, not a tolerance — to the
/// fully sequential [`evaluate`] path, per this node kind's own
/// no-cross-element-accumulation argument (`run_elementwise_dispatch`'s
/// doc).
#[proxima::test]
#[case::two_workers(2)]
#[case::three_workers(3)]
#[case::eight_workers(8)]
async fn evaluate_quantized_matches_evaluate_for_a_large_elementwise_chain(#[case] workers: usize) {
    // SAFETY of the test env var mutation: `PROXIMA_MATMUL_WORKERS` is
    // read exactly once, lazily, behind `matmul_worker_count`'s own
    // `OnceLock` — set before that lock is ever touched by any other
    // test in this process would be a race, so this case relies on
    // nextest's default one-test-per-process isolation instead of
    // resetting the lock.
    // SAFETY: nextest runs each test in its own process, so no other
    // thread in this process reads or writes the environment
    // concurrently with this call.
    unsafe {
        std::env::set_var("PROXIMA_MATMUL_WORKERS", workers.to_string());
    }

    let mut program = Vec::new();
    let (rows, width) = (64usize, 8192usize);
    let mut current = f32_block(
        &mut program,
        &[Extent::Static(rows as u32), Extent::Static(width as u32)],
    );
    for _ in 0..3 {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(current, IndexMap::Affine(map::projection(2, &[0, 1])))],
                name: None,
            },
        );
    }
    let _ = current;

    let input: Vec<f32> = (0..rows * width)
        .map(|value| (value as f32) * 0.0001)
        .collect();

    let sequential = evaluate(&program, &[], &[&input], &[]).expect("sequential evaluates");
    let blocks = [QuantizedBlock::Float32(&input)];
    let quantized = evaluate_quantized(&program, &[], &blocks, &[]).expect("quantized evaluates");

    assert_eq!(quantized.shape(), sequential.shape());
    assert_eq!(
        quantized.root(),
        sequential.root(),
        "cohort-dispatched elementwise output diverges from the sequential path"
    );
}

#[proxima::test]
#[case::one_worker(1)]
#[case::two_workers(2)]
#[case::three_workers(3)]
#[case::eight_workers(8)]
async fn evaluate_parallel_matches_evaluate_for_a_matmul(#[case] workers: usize) {
    let (m, k, n) = (4usize, 3usize, 5usize);
    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
    let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

    assert_parallel_matches_sequential(&program, &[], &[&lhs, &rhs], &[], workers);
}

#[proxima::test]
#[case::one_worker(1)]
#[case::two_workers(2)]
#[case::three_workers(3)]
#[case::eight_workers(8)]
async fn evaluate_parallel_matches_evaluate_for_a_tanh_chain(#[case] workers: usize) {
    let mut program = Vec::new();
    let mut current = f32_block(&mut program, &[Extent::Static(4)]);
    for _ in 0..8 {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
    }
    let _ = current;

    let input = [0.1, 0.2, 0.3, 0.4f32];
    assert_parallel_matches_sequential(&program, &[], &[&input], &[], workers);
}

#[proxima::test]
#[case::one_worker(1)]
#[case::two_workers(2)]
#[case::three_workers(3)]
#[case::eight_workers(8)]
async fn evaluate_parallel_matches_evaluate_for_softmax(#[case] workers: usize) {
    let mut program = Vec::new();
    let (n, d) = (2usize, 4usize);
    let input = f32_block(
        &mut program,
        &[Extent::Static(n as u32), Extent::Static(d as u32)],
    );

    let row_map = IndexMap::Affine(map::projection(2, &[0, 1]));
    let broadcast_map = IndexMap::Affine(map::projection(2, &[0]));

    let max = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Maximum,
            init: ReduceInit::NegativeInfinity,
            operand: input,
            in_map: row_map.clone(),
            out_map: broadcast_map.clone(),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let shifted = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Subtract,
            operands: alloc::vec![(input, row_map.clone()), (max, broadcast_map.clone())],
            name: None,
        },
    );
    let exponentiated = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Exponential,
            operands: alloc::vec![(shifted, row_map.clone())],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: exponentiated,
            in_map: row_map.clone(),
            out_map: broadcast_map.clone(),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Divide,
            operands: alloc::vec![(exponentiated, row_map), (sum, broadcast_map)],
            name: None,
        },
    );

    let input_data = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0f32];
    assert_parallel_matches_sequential(&program, &[], &[&input_data], &[], workers);
}

#[proxima::test]
#[case::one_worker(1)]
#[case::two_workers(2)]
#[case::three_workers(3)]
#[case::eight_workers(8)]
async fn evaluate_parallel_matches_evaluate_for_cumsum(#[case] workers: usize) {
    let mut program = Vec::new();
    let source = f32_block(&mut program, &[Extent::Static(6)]);
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

    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
    assert_parallel_matches_sequential(&program, &[], &[&data], &[], workers);
}

#[proxima::test]
#[case::one_worker(1)]
#[case::two_workers(2)]
#[case::three_workers(3)]
#[case::eight_workers(8)]
async fn evaluate_parallel_matches_evaluate_for_multiple_requested_outputs(#[case] workers: usize) {
    let mut program = Vec::new();
    let source = f32_block(&mut program, &[Extent::Static(4)]);
    let mut current = source;
    let mut nodes = alloc::vec![source];
    for _ in 0..4 {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        nodes.push(current);
    }
    let midpoint = nodes[2];
    let root = current;

    let input = [0.1, 0.2, 0.3, 0.4f32];
    assert_parallel_matches_sequential(&program, &[], &[&input], &[midpoint, root], workers);
}

#[test]
fn a_matmul_past_the_parallel_threshold_actually_splits_and_still_matches_sequential() {
    let (m, k, n) = (64usize, 64usize, 64usize);
    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
    let lhs: Vec<f32> = (0..m * k).map(|value| (value % 7) as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| (value % 5) as f32).collect();

    let shapes = shape::infer(&program, &[]).expect("64x64x64 matmul infers");
    let resolved = bind::bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("64x64x64 matmul resolves");
    assert_eq!(resolved.len(), 1, "fused into one reduction node");
    assert!(
        element_count(&resolved[0].extents) >= PARALLEL_THRESHOLD,
        "this size must clear the threshold or this test proves nothing about the \
         threaded path"
    );
    assert!(
        resolved[0].split(4).is_some(),
        "the node must actually be splittable for the threaded path to run"
    );

    let workers = NonZeroUsize::new(4).expect("4 is nonzero");
    assert_parallel_matches_sequential(&program, &[], &[&lhs, &rhs], &[], workers.get());
}

#[test]
fn evaluate_parallel_raises_the_same_errors_as_evaluate_on_every_existing_sad_path() {
    let workers = NonZeroUsize::new(2).expect("2 is nonzero");

    let mut count_program = Vec::new();
    f32_block(&mut count_program, &[Extent::Static(4)]);
    let sequential_error =
        evaluate(&count_program, &[], &[], &[]).expect_err("one block is required");
    let parallel_error = evaluate_parallel(&count_program, &[], &[], &[], workers)
        .expect_err("one block is required");
    assert_eq!(sequential_error, parallel_error);

    let mut size_program = Vec::new();
    f32_block(&mut size_program, &[Extent::Static(4)]);
    let too_short = [1.0, 2.0f32];
    let sequential_error =
        evaluate(&size_program, &[], &[&too_short], &[]).expect_err("block is wrong size");
    let parallel_error = evaluate_parallel(&size_program, &[], &[&too_short], &[], workers)
        .expect_err("block is wrong size");
    assert_eq!(sequential_error, parallel_error);

    let mut dtype_program = Vec::new();
    block(&mut dtype_program, DType::Int32, &[Extent::Static(4)]);
    let sequential_error = evaluate(&dtype_program, &[], &[], &[]).expect_err("int32 is not f32");
    let parallel_error =
        evaluate_parallel(&dtype_program, &[], &[], &[], workers).expect_err("int32 is not f32");
    assert_eq!(sequential_error, parallel_error);

    // A `Keep::Scan` scatter stays rejected (shape.rs's own doc: a scan
    // step would need to read the destination its own write just
    // touched, which the sequential interpreter does not order that
    // way) -- unlike a `Keep::Reduce` scatter, which this row's own
    // `scatter_add_matches_the_hand_worked_example` etc. now accept.
    let mut scatter_scan_program = Vec::new();
    let source = f32_block(&mut scatter_scan_program, &[Extent::Static(4)]);
    let ids = block(
        &mut scatter_scan_program,
        DType::Int32,
        &[Extent::Static(4)],
    );
    let out_map = IndexMap::scatter(ids, map::projection(1, &[0]), 1, &[], 0, 3);
    append(
        &mut scatter_scan_program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map,
            keep: Keep::Scan,
            name: None,
        }),
    );
    let sequential_error = evaluate(&scatter_scan_program, &[], &[], &[])
        .expect_err("a scatter scan has no defined step order");
    let parallel_error = evaluate_parallel(&scatter_scan_program, &[], &[], &[], workers)
        .expect_err("a scatter scan has no defined step order");
    assert_eq!(sequential_error, parallel_error);
}

/// `evaluate_parallel`'s own chunking never touches a scatter node
/// (`bind::BoundOp::split` refuses to split one -- see its own doc), so
/// running the hand-worked scatter example through the parallel driver
/// must land on the exact same numbers `evaluate` does.
#[test]
fn evaluate_parallel_matches_evaluate_on_the_hand_worked_scatter_example() {
    let workers = NonZeroUsize::new(4).expect("4 is nonzero");
    let (program, _source, _ids, scattered) = scatter_add_program(3);
    let index_values = [2.0f32, 0.0, 2.0, 1.0];
    let source_values = [10.0f32, 20.0, 30.0, 40.0];

    let sequential = evaluate(
        &program,
        &[],
        &[&source_values, &index_values],
        &[scattered],
    )
    .expect("sequential scatter evaluates");
    let parallel = evaluate_parallel(
        &program,
        &[],
        &[&source_values, &index_values],
        &[scattered],
        workers,
    )
    .expect("parallel scatter evaluates");

    assert_eq!(sequential.root(), &[20.0, 40.0, 40.0]);
    assert_eq!(sequential.root(), parallel.root());
}

#[test]
fn peak_live_buffers_on_a_tanh_chain_stays_small_under_evaluate_parallel() {
    let mut program = Vec::new();
    let mut current = f32_block(&mut program, &[Extent::Static(4)]);
    for _ in 0..8 {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
    }
    let _ = current;

    let input = [0.1, 0.2, 0.3, 0.4f32];
    let workers = NonZeroUsize::new(4).expect("4 is nonzero");
    let evaluated = evaluate_parallel(&program, &[], &[&input], &[], workers)
        .expect("tanh chain evaluates in parallel");

    let peak = evaluated
        .peak_live_buffers()
        .expect("evaluate_parallel tracks peak live buffers");
    assert!(
        peak <= 3,
        "streaming a chain of 8 unary elementwise ops should not hold one buffer per op, got {peak}"
    );
}

// THE PROOF: the full pipeline — shape inference, layout binding, and
// CPU execution — driven entirely through the real `Pipe` algebra, as
// one composed chain (`shapes.and_then(builder).and_then(interpreter)`
// via `PipeExt`), matches what `evaluate` (the free-function path every
// other test in this crate trusts) produces for the identical matmul
// program.
//
// The three-stage `AndThen` typechecks because `Second::In = First::Out`
// holds at both joins: `ShapeTable::Out = (Op, Shapes) = BoundOpBuilder::In`,
// and `BoundOpBuilder::Out = Vec<BoundOp> = Interpreter::In` — `Interpreter`
// takes the batch a push readies (0, 1, or 2 records — see
// `bind::BoundOpBuilder::push`'s own doc) directly, so no per-node
// driving loop is needed at the call site; `Pipe::call` on the full
// chain is called once per `Op` record, exactly as `ShapeTable`'s own
// one-record-at-a-time contract expects.
#[test]
fn execute_composes_through_pipe_ext_matching_the_free_function() {
    use crate::bind::BoundOpBuilder;
    use crate::live;
    use crate::numeric::NumericPolicy;
    use crate::shape::ShapeTable;
    use proxima_primitives::block_on;
    use proxima_primitives::pipe::{Pipe, PipeExt};

    let (m, k, n) = (4usize, 3usize, 5usize);
    let (program, sum) = matmul_program(m as u32, k as u32, n as u32, false);
    let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

    let outputs: Vec<NodeId> = Vec::new();
    let retires = live::annotate(&program, &outputs);
    let shapes = ShapeTable::new(&[]);
    let builder = BoundOpBuilder::new(retires, NumericPolicy::bit_exact());

    // `matmul_program` always appends `lhs` then `rhs` first.
    let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
    buffers[0] = Some(lhs.clone());
    buffers[1] = Some(rhs.clone());
    let chain = shapes
        .and_then(builder)
        .and_then(Interpreter::new(&mut buffers));

    for expr in &program {
        block_on(Pipe::call(&chain, expr.clone())).expect("shape+bind+execute pipe step succeeds");
    }
    // Release the chain's mutable borrow of `buffers` before reading the
    // result back out of it — the interpreter stage was moved into
    // `chain`, so its `get()` is unreachable here, but its buffer table
    // IS `buffers`: reading `buffers[sum.0]` directly is the same read
    // `Interpreter::get` performs, once the borrow is free to take back.
    drop(chain);

    let chain_result = buffers[sum.0 as usize]
        .clone()
        .expect("the matmul node was executed through the composed chain");

    let evaluated =
        evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("free-function matmul evaluates");

    assert_eq!(chain_result, evaluated.root());
}

fn typed_identity() -> IndexMap {
    IndexMap::Affine(map::projection(1, &[0]))
}

fn typed_add_program(dtype: DType, len: u32) -> (Vec<Op>, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let lhs = block(&mut program, dtype, &[Extent::Static(len)]);
    let rhs = block(&mut program, dtype, &[Extent::Static(len)]);
    let sum = append(
        &mut program,
        Op::Elementwise {
            dtype,
            body: ScalarOp::Add,
            operands: alloc::vec![(lhs, typed_identity()), (rhs, typed_identity())],
            name: None,
        },
    );
    (program, lhs, rhs, sum)
}

#[proxima::test]
#[case::int8(DType::Int8, TypedBuffer::Int8(alloc::vec![1, 2, 3]), TypedBuffer::Int8(alloc::vec![10, 20, 30]), TypedBuffer::Int8(alloc::vec![11, 22, 33]))]
#[case::uint8(DType::UInt8, TypedBuffer::UInt8(alloc::vec![1, 2, 3]), TypedBuffer::UInt8(alloc::vec![10, 20, 30]), TypedBuffer::UInt8(alloc::vec![11, 22, 33]))]
#[case::int16(DType::Int16, TypedBuffer::Int16(alloc::vec![1, 2, 3]), TypedBuffer::Int16(alloc::vec![10, 20, 30]), TypedBuffer::Int16(alloc::vec![11, 22, 33]))]
#[case::uint16(DType::UInt16, TypedBuffer::UInt16(alloc::vec![1, 2, 3]), TypedBuffer::UInt16(alloc::vec![10, 20, 30]), TypedBuffer::UInt16(alloc::vec![11, 22, 33]))]
#[case::int32(DType::Int32, TypedBuffer::Int32(alloc::vec![1, 2, 3]), TypedBuffer::Int32(alloc::vec![10, 20, 30]), TypedBuffer::Int32(alloc::vec![11, 22, 33]))]
#[case::uint32(DType::UInt32, TypedBuffer::UInt32(alloc::vec![1, 2, 3]), TypedBuffer::UInt32(alloc::vec![10, 20, 30]), TypedBuffer::UInt32(alloc::vec![11, 22, 33]))]
#[case::int64(DType::Int64, TypedBuffer::Int64(alloc::vec![1, 2, 3]), TypedBuffer::Int64(alloc::vec![10, 20, 30]), TypedBuffer::Int64(alloc::vec![11, 22, 33]))]
#[case::uint64(DType::UInt64, TypedBuffer::UInt64(alloc::vec![1, 2, 3]), TypedBuffer::UInt64(alloc::vec![10, 20, 30]), TypedBuffer::UInt64(alloc::vec![11, 22, 33]))]
#[case::int128(DType::Int128, TypedBuffer::Int128(alloc::vec![1, 2, 3]), TypedBuffer::Int128(alloc::vec![10, 20, 30]), TypedBuffer::Int128(alloc::vec![11, 22, 33]))]
#[case::uint128(DType::UInt128, TypedBuffer::UInt128(alloc::vec![1, 2, 3]), TypedBuffer::UInt128(alloc::vec![10, 20, 30]), TypedBuffer::UInt128(alloc::vec![11, 22, 33]))]
#[case::float64(DType::Float64, TypedBuffer::Float64(alloc::vec![1.5, 2.5, 3.5]), TypedBuffer::Float64(alloc::vec![10.0, 20.0, 30.0]), TypedBuffer::Float64(alloc::vec![11.5, 22.5, 33.5]))]
async fn evaluate_typed_adds_across_every_extended_width(
    #[case] dtype: DType,
    #[case] lhs: TypedBuffer,
    #[case] rhs: TypedBuffer,
    #[case] expected: TypedBuffer,
) {
    let (program, _, _, _) = typed_add_program(dtype, 3);
    let results = evaluate_typed(&program, &[], &[lhs, rhs], &[]).expect("typed add evaluates");
    assert_eq!(results.len(), 1);
    let (_, shape, data) = &results[0];
    assert_eq!(shape, &alloc::vec![3u64]);
    assert_eq!(*data, expected);
    assert_eq!(data.dtype(), dtype);
    assert_eq!(data.len(), 3);
    assert!(!data.is_empty());
}

#[test]
fn evaluate_typed_wraps_signed_narrow_overflow_instead_of_panicking() {
    let (program, _, _, _) = typed_add_program(DType::Int8, 1);
    let lhs = TypedBuffer::Int8(alloc::vec![127]);
    let rhs = TypedBuffer::Int8(alloc::vec![1]);
    let results = evaluate_typed(&program, &[], &[lhs, rhs], &[])
        .expect("i8 add wraps rather than panicking");
    assert_eq!(results[0].2, TypedBuffer::Int8(alloc::vec![-128]));
}

#[test]
fn evaluate_typed_rejects_negate_on_an_unsigned_dtype() {
    let mut program = Vec::new();
    let operand = block(&mut program, DType::UInt32, &[Extent::Static(2)]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::UInt32,
            body: ScalarOp::Negate,
            operands: alloc::vec![(operand, typed_identity())],
            name: None,
        },
    );
    let blocks = [TypedBuffer::UInt32(alloc::vec![1, 2])];
    let error =
        evaluate_typed(&program, &[], &blocks, &[]).expect_err("u32 has no representable negative");
    assert!(
        matches!(
            error,
            TensorError::UnsupportedScalarOp {
                op: ScalarOp::Negate,
                dtype: DType::UInt32,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn evaluate_typed_rejects_a_transcendental_on_an_integer_dtype() {
    let mut program = Vec::new();
    let operand = block(&mut program, DType::Int32, &[Extent::Static(2)]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Int32,
            body: ScalarOp::SquareRoot,
            operands: alloc::vec![(operand, typed_identity())],
            name: None,
        },
    );
    let blocks = [TypedBuffer::Int32(alloc::vec![4, 9])];
    let error = evaluate_typed(&program, &[], &blocks, &[])
        .expect_err("sqrt is not defined over Int32 by this evaluator");
    assert!(
        matches!(
            error,
            TensorError::UnsupportedScalarOp {
                op: ScalarOp::SquareRoot,
                dtype: DType::Int32,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn evaluate_typed_reports_integer_divide_by_zero_instead_of_panicking() {
    let mut program = Vec::new();
    let lhs = block(&mut program, DType::Int32, &[Extent::Static(1)]);
    let rhs = block(&mut program, DType::Int32, &[Extent::Static(1)]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Int32,
            body: ScalarOp::Divide,
            operands: alloc::vec![(lhs, typed_identity()), (rhs, typed_identity())],
            name: None,
        },
    );
    let blocks = [
        TypedBuffer::Int32(alloc::vec![10]),
        TypedBuffer::Int32(alloc::vec![0]),
    ];
    let error = evaluate_typed(&program, &[], &blocks, &[])
        .expect_err("integer division by zero is a real error, not UB");
    assert!(
        matches!(error, TensorError::CheckedDivisionFailed { .. }),
        "{error}"
    );
}

#[test]
fn evaluate_typed_computes_float64_transcendentals() {
    let mut program = Vec::new();
    let operand = block(&mut program, DType::Float64, &[Extent::Static(3)]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float64,
            body: ScalarOp::SquareRoot,
            operands: alloc::vec![(operand, typed_identity())],
            name: None,
        },
    );
    let blocks = [TypedBuffer::Float64(alloc::vec![4.0, 9.0, 16.0])];
    let results = evaluate_typed(&program, &[], &blocks, &[]).expect("f64 sqrt evaluates");
    assert_eq!(
        results[0].2,
        TypedBuffer::Float64(alloc::vec![2.0, 3.0, 4.0])
    );
}

#[test]
fn evaluate_typed_rejects_a_program_mixing_dtypes() {
    let mut program = Vec::new();
    let lhs = block(&mut program, DType::Int32, &[Extent::Static(2)]);
    let rhs = block(&mut program, DType::Int8, &[Extent::Static(2)]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Int32,
            body: ScalarOp::Add,
            operands: alloc::vec![(lhs, typed_identity()), (rhs, typed_identity())],
            name: None,
        },
    );
    let blocks = [
        TypedBuffer::Int32(alloc::vec![1, 2]),
        TypedBuffer::Int8(alloc::vec![1, 2]),
    ];
    let error = evaluate_typed(&program, &[], &blocks, &[])
        .expect_err("a mixed-dtype fused body is not yet supported");
    assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
}

#[test]
fn an_i8_operand_i32_accumulator_reduce_evaluates() {
    // five i8 elements of 30 each: the true sum is 150, which does not
    // fit in i8 (max 127) -- wrapping i8 arithmetic would land on -106
    // (150 - 256). an i32 accumulator is the only way to observe 150.
    let mut program = Vec::new();
    let operand = block(&mut program, DType::Int8, &[Extent::Static(5)]);
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Int32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let blocks = [TypedBuffer::Int8(alloc::vec![30, 30, 30, 30, 30])];
    let results = evaluate_typed(&program, &[], &blocks, &[])
        .expect("an i8-operand, i32-accumulator reduce evaluates");
    assert_eq!(
        results[0].2,
        TypedBuffer::Int32(alloc::vec![150]),
        "the i32 accumulator must carry the true sum, not the i8-wrapped one (-106)"
    );
}

#[test]
fn f32_typed_path_is_unchanged() {
    let (program, _) = typed_reduce_vector_to_scalar_program(DType::Float32, 4);
    let operand = TypedBuffer::Float32(alloc::vec![1.5, 2.5, 3.0, 4.0]);
    let results = evaluate_typed(&program, &[], &[operand], &[])
        .expect("a uniform f32 typed program still evaluates via the unchanged NEON-backed path");
    assert_eq!(results[0].2, TypedBuffer::Float32(alloc::vec![11.0]));
}

#[test]
fn an_unshipped_widened_pair_is_rejected_not_silently_wrong() {
    let mut program = Vec::new();
    let operand = block(&mut program, DType::UInt16, &[Extent::Static(3)]);
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::UInt64,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let blocks = [TypedBuffer::UInt16(alloc::vec![1, 2, 3])];
    let error = evaluate_typed(&program, &[], &blocks, &[])
        .expect_err("(UInt16, UInt64) is not a shipped widened pair");
    assert!(
        matches!(error, TensorError::NotLowerable { .. }),
        "an unshipped pair must fail honestly, never fall back to a wrong result: {error}"
    );
}

/// A widened reduce program: `operand_dtype` operand folded by `Add`
/// into an `accumulator_dtype` accumulator — the shape
/// [`an_i8_operand_i32_accumulator_reduce_evaluates`] built by hand,
/// generalized over the dtype pair so every widened-pair test below
/// shares one builder.
fn typed_widened_reduce_program(
    operand_dtype: DType,
    accumulator_dtype: DType,
    len: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let operand = block(&mut program, operand_dtype, &[Extent::Static(len)]);
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: accumulator_dtype,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

#[proxima::test]
#[case::f16_add(DType::Float16, TypedBuffer::Float16(alloc::vec![f16::from_f32(1.5), f16::from_f32(2.25), f16::from_f32(-3.0)]), TypedBuffer::Float16(alloc::vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)]))]
#[case::bf16_add(DType::BFloat16, TypedBuffer::BFloat16(alloc::vec![bf16::from_f32(1.5), bf16::from_f32(2.25), bf16::from_f32(-3.0)]), TypedBuffer::BFloat16(alloc::vec![bf16::from_f32(10.0), bf16::from_f32(20.0), bf16::from_f32(30.0)]))]
async fn half_precision_uniform_elementwise_matches_f32_reference(
    #[case] dtype: DType,
    #[case] lhs: TypedBuffer,
    #[case] rhs: TypedBuffer,
) {
    let (program, _, _, _) = typed_add_program(dtype, 3);
    let results = evaluate_typed(&program, &[], &[lhs.clone(), rhs.clone()], &[])
        .expect("half-precision add evaluates through the typed path");
    let (got, reference): (Vec<f32>, Vec<f32>) = match (&results[0].2, &lhs, &rhs) {
        (TypedBuffer::Float16(sum), TypedBuffer::Float16(lhs), TypedBuffer::Float16(rhs)) => (
            sum.iter().map(|value| value.to_f32()).collect(),
            lhs.iter()
                .zip(rhs)
                .map(|(left, right)| left.to_f32() + right.to_f32())
                .collect(),
        ),
        (TypedBuffer::BFloat16(sum), TypedBuffer::BFloat16(lhs), TypedBuffer::BFloat16(rhs)) => (
            sum.iter().map(|value| value.to_f32()).collect(),
            lhs.iter()
                .zip(rhs)
                .map(|(left, right)| left.to_f32() + right.to_f32())
                .collect(),
        ),
        other => panic!("unexpected buffer shape: {other:?}"),
    };
    for (value, expected) in got.iter().zip(&reference) {
        // one rounding step from the f32 reference (the operands are
        // already half-precision, so the reference itself is exact at
        // this magnitude) -- a loose bound catching a wrong op, not
        // tuned to a measured residual.
        assert!(
            (value - expected).abs() < 1e-2,
            "half-precision add {value} vs f32 reference {expected}"
        );
    }
}

#[proxima::test]
#[case::f16_sum(DType::Float16)]
#[case::bf16_sum(DType::BFloat16)]
async fn half_precision_uniform_reduce_matches_f32_reference(#[case] dtype: DType) {
    let values = [1.5f32, 2.5, -0.5, 4.0];
    let (program, _) = typed_reduce_vector_to_scalar_program(dtype, values.len() as u32);
    let expected_f32: f32 = values.iter().sum();
    let operand = match dtype {
        DType::Float16 => {
            TypedBuffer::Float16(values.iter().map(|value| f16::from_f32(*value)).collect())
        }
        DType::BFloat16 => {
            TypedBuffer::BFloat16(values.iter().map(|value| bf16::from_f32(*value)).collect())
        }
        other => panic!("unexpected dtype in case table: {other:?}"),
    };
    let results = evaluate_typed(&program, &[], &[operand], &[])
        .expect("half-precision reduce evaluates through the typed path");
    let got = match &results[0].2 {
        TypedBuffer::Float16(data) => data[0].to_f32(),
        TypedBuffer::BFloat16(data) => data[0].to_f32(),
        other => panic!("unexpected result buffer: {other:?}"),
    };
    assert!(
        (got - expected_f32).abs() < 1e-2,
        "half-precision reduce {got} vs f32 reference {expected_f32}"
    );
}

#[test]
fn f16_reduce_widens_into_an_f32_accumulator_where_f16_alone_overflows() {
    // two f16 values whose sum overflows f16 range (max ~65504) and
    // would round to infinity if accumulated in f16, but the true sum
    // fits an f32 accumulator exactly -- the same "widening changes the
    // observable result" shape as the i8/i32 test above, at floating
    // widths instead of integer ones.
    let (program, _) = typed_widened_reduce_program(DType::Float16, DType::Float32, 2);
    let operand = TypedBuffer::Float16(alloc::vec![f16::from_f32(40000.0), f16::from_f32(40000.0)]);
    let results = evaluate_typed(&program, &[], &[operand], &[])
        .expect("an f16-operand, f32-accumulator reduce evaluates");
    let TypedBuffer::Float32(sum) = &results[0].2 else {
        panic!("widened f16 reduce must produce an f32 accumulator buffer");
    };
    assert_eq!(
        sum[0], 80000.0,
        "the f32 accumulator must carry the true sum, not an f16-saturated infinity"
    );
}

#[test]
fn bf16_reduce_widens_into_an_f32_accumulator_exactly() {
    let (program, _) = typed_widened_reduce_program(DType::BFloat16, DType::Float32, 3);
    let operand = TypedBuffer::BFloat16(alloc::vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
    ]);
    let results = evaluate_typed(&program, &[], &[operand], &[])
        .expect("a bf16-operand, f32-accumulator reduce evaluates");
    assert_eq!(results[0].2, TypedBuffer::Float32(alloc::vec![6.0]));
}

#[test]
fn typed_program_plan_no_longer_rejects_float16_or_bfloat16_but_still_rejects_bool() {
    let (float16_program, _, _, _) = typed_add_program(DType::Float16, 2);
    let float16_blocks = [
        TypedBuffer::Float16(alloc::vec![f16::from_f32(1.0), f16::from_f32(2.0)]),
        TypedBuffer::Float16(alloc::vec![f16::from_f32(10.0), f16::from_f32(20.0)]),
    ];
    evaluate_typed(&float16_program, &[], &float16_blocks, &[])
        .expect("Float16 must no longer be rejected by typed_program_plan");

    let mut bool_program = Vec::new();
    let bool_operand = block(&mut bool_program, DType::Bool, &[Extent::Static(2)]);
    append(
        &mut bool_program,
        Op::Elementwise {
            dtype: DType::Bool,
            body: ScalarOp::Identity,
            operands: alloc::vec![(bool_operand, typed_identity())],
            name: None,
        },
    );
    let error = evaluate_typed(&bool_program, &[], &[], &[])
        .expect_err("Bool must still be rejected -- no TypedBuffer variant backs it");
    assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
}

/// `table[ids[s], d]`, `table` at `compute_dtype` and `ids` at
/// `index_dtype` -- the same [`IndexMap::Computed`] wiring
/// `embedding_lookup_program` uses, parameterized over both dtypes so
/// [`typed_program_plan`]'s third, index role can be exercised at any
/// compute width against any integer index width.
fn typed_gather_program(compute_dtype: DType, index_dtype: DType) -> Vec<Op> {
    let mut program = Vec::new();
    let table = block(
        &mut program,
        compute_dtype,
        &[Extent::Static(4), Extent::Static(2)],
    );
    let ids = block(&mut program, index_dtype, &[Extent::Static(3)]);
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: map::IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
                map::AxisIndex::default(),
                map::AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
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
            dtype: compute_dtype,
            body: ScalarOp::Identity,
            operands: alloc::vec![(table, gathered_map)],
            name: None,
        },
    );
    program
}

/// [`typed_program_plan`]'s third role: a gather index node carries its
/// own integer dtype, distinct from the program's compute dtype,
/// without failing the plan -- proven directly against the plan
/// function. See the `typed_gather_*` tests below for the same shape
/// proven all the way through `evaluate_typed` against the f32 pipeline
/// as oracle.
#[proxima::test]
#[case::i32_index_over_f32_compute(DType::Float32, DType::Int32)]
#[case::u32_index_over_f32_compute(DType::Float32, DType::UInt32)]
#[case::i32_index_over_f16_compute(DType::Float16, DType::Int32)]
#[case::u32_index_over_f16_compute(DType::Float16, DType::UInt32)]
async fn typed_program_plan_permits_an_integer_gather_index_distinct_from_compute_dtype(
    #[case] compute_dtype: DType,
    #[case] index_dtype: DType,
) {
    let program = typed_gather_program(compute_dtype, index_dtype);
    let plan =
        typed_program_plan(&program).expect("an integer gather index must not fail the plan");
    assert_eq!(
        plan,
        TypedPlan::Uniform(compute_dtype),
        "the index node's own dtype must not be folded into the program's compute dtype"
    );
}

/// The sad path this role's own gate exists for: a FLOAT gather index
/// dtype is not a legal index type ([`DType::is_integer`]) and must
/// still fail the plan, named, rather than being silently accepted
/// alongside the new integer exemption.
#[test]
fn typed_program_plan_rejects_a_float_gather_index_dtype() {
    let program = typed_gather_program(DType::Float32, DType::Float64);
    let error = typed_program_plan(&program)
        .expect_err("a float-dtype gather index must be rejected, never silently accepted");
    assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
}

/// `table[ids[s], d]` at a chosen compute/index dtype pair, `dim`
/// always the kept axis (`gathered_dim: 0`) -- the typed counterpart of
/// [`embedding_lookup_program`], parameterized the same way
/// [`typed_gather_program`] is, but at real `vocab`/`dim`/`seq` sizes so
/// its output can be diffed against the f32 oracle element-for-element
/// rather than only checked through [`typed_program_plan`].
fn typed_embedding_lookup_program(
    compute_dtype: DType,
    index_dtype: DType,
    vocab: u32,
    dim: u32,
    seq: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let table = block(
        &mut program,
        compute_dtype,
        &[Extent::Static(vocab), Extent::Static(dim)],
    );
    let ids = block(&mut program, index_dtype, &[Extent::Static(seq)]);
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: map::IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
                map::AxisIndex::default(),
                map::AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let gathered = append(
        &mut program,
        Op::Elementwise {
            dtype: compute_dtype,
            body: ScalarOp::Identity,
            operands: alloc::vec![(table, gathered_map)],
            name: None,
        },
    );
    (program, gathered)
}

/// [`typed_embedding_lookup_program`]'s sibling with the gathered table
/// axis chosen by `gathered_dim` instead of hardcoded to `0` --
/// `gathered_dim: 1` exercises a table laid out `[kept, gathered]`
/// instead of `[gathered, kept]`, proving the typed cursor's
/// `element_stride`/`extent` derivation (from [`bind::bind`], unmodified
/// by this change) is honoured regardless of which axis is gathered.
fn typed_gather_dim_program(
    compute_dtype: DType,
    index_dtype: DType,
    table_shape: [u32; 2],
    seq: u32,
    gathered_dim: u16,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let table = block(
        &mut program,
        compute_dtype,
        &[
            Extent::Static(table_shape[0]),
            Extent::Static(table_shape[1]),
        ],
    );
    let ids = block(&mut program, index_dtype, &[Extent::Static(seq)]);
    let kept_dim = 1 - gathered_dim;
    let mut axes = alloc::vec![map::AxisIndex::default(); 2];
    axes[kept_dim as usize] = map::AxisIndex {
        terms: core::iter::once(AxisTerm::projection(1)).collect(),
        offset: 0,
        len: None,
    };
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: map::IndexPattern { iter_rank: 2, axes },
        gathered_dim,
    };
    let gathered = append(
        &mut program,
        Op::Elementwise {
            dtype: compute_dtype,
            body: ScalarOp::Identity,
            operands: alloc::vec![(table, gathered_map)],
            name: None,
        },
    );
    (program, gathered)
}

/// A row-sum reduce, accumulated at `accumulator_dtype`, over a table
/// gathered at `operand_dtype` -- [`typed_widened_reduce_program`]'s
/// shape composed with a real gather instead of a plain block operand,
/// proving `TypedPlan::Widened` executes correctly when its own operand
/// is data-dependent.
fn typed_widened_gather_reduce_program(
    operand_dtype: DType,
    accumulator_dtype: DType,
    index_dtype: DType,
    vocab: u32,
    dim: u32,
    seq: u32,
) -> (Vec<Op>, NodeId) {
    let (mut program, gathered) =
        typed_embedding_lookup_program(operand_dtype, index_dtype, vocab, dim, seq);
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: accumulator_dtype,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: gathered,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

/// [`typed_program_plan`]'s third role, executed: an `i32`- or
/// `u32`-index gather over an `f32` compute table must produce exactly
/// the same bytes [`embedding_lookup_program`]'s f32 pipeline does for
/// the same table and the same row selection -- the incumbent-parity
/// bar (guiding-principles §14): the f32 evaluator is the oracle, and
/// any divergence is this evaluator's bug until proven otherwise.
/// Covers a repeated index (row 3 selected twice) and both boundary
/// indices (`0` and `vocab - 1`).
#[proxima::test]
#[case::i32_index(DType::Int32)]
#[case::u32_index(DType::UInt32)]
async fn typed_gather_matches_f32_oracle_element_for_element(#[case] index_dtype: DType) {
    let (vocab, dim, seq) = (50usize, 6usize, 5usize);
    let (f32_program, _) = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim)
        .map(|value| (value % 37) as f32 - 10.0)
        .collect();
    // row 3 repeated, plus both boundary rows (0 and vocab - 1).
    let ids_f32 = [3.0f32, (vocab - 1) as f32, 0.0, 3.0, 25.0];
    let oracle =
        evaluate(&f32_program, &[], &[&table_data, &ids_f32], &[]).expect("f32 oracle evaluates");

    let (typed_program, _) = typed_embedding_lookup_program(
        DType::Float32,
        index_dtype,
        vocab as u32,
        dim as u32,
        seq as u32,
    );
    let ids_block = match index_dtype {
        DType::Int32 => TypedBuffer::Int32(ids_f32.iter().map(|&value| value as i32).collect()),
        DType::UInt32 => TypedBuffer::UInt32(ids_f32.iter().map(|&value| value as u32).collect()),
        other => panic!("unexpected index dtype in case table: {other:?}"),
    };
    let blocks = [TypedBuffer::Float32(table_data.clone()), ids_block];
    let results = evaluate_typed(&typed_program, &[], &blocks, &[])
        .expect("typed gather evaluates against real data");
    let TypedBuffer::Float32(got) = &results[0].2 else {
        panic!("expected an f32 result buffer");
    };
    assert_eq!(
        got.as_slice(),
        oracle.root(),
        "typed {index_dtype:?}-index gather must match the f32 oracle element-for-element"
    );
}

/// The same oracle-parity bar as
/// [`typed_gather_matches_f32_oracle_element_for_element`], with the
/// compute dtype narrowed to `f16` -- exact equality no longer holds
/// (the table itself round-trips through half precision before the
/// gather ever runs), so the bound is half a step at the table's own
/// magnitude instead.
#[test]
fn typed_gather_f16_compute_matches_f32_oracle_within_half_precision() {
    let (vocab, dim, seq) = (32usize, 4usize, 6usize);
    let (f32_program, _) = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_f32: Vec<f32> = (0..vocab * dim)
        .map(|value| (value % 23) as f32 - 5.0)
        .collect();
    let ids_f32 = [0.0f32, (vocab - 1) as f32, 7.0, 7.0, 15.0, 31.0];
    let oracle =
        evaluate(&f32_program, &[], &[&table_f32, &ids_f32], &[]).expect("f32 oracle evaluates");

    let (typed_program, _) = typed_embedding_lookup_program(
        DType::Float16,
        DType::Int32,
        vocab as u32,
        dim as u32,
        seq as u32,
    );
    let table_f16: Vec<f16> = table_f32
        .iter()
        .map(|&value| f16::from_f32(value))
        .collect();
    let ids_i32: Vec<i32> = ids_f32.iter().map(|&value| value as i32).collect();
    let blocks = [TypedBuffer::Float16(table_f16), TypedBuffer::Int32(ids_i32)];
    let results =
        evaluate_typed(&typed_program, &[], &blocks, &[]).expect("f16 typed gather evaluates");
    let TypedBuffer::Float16(got) = &results[0].2 else {
        panic!("expected an f16 result buffer");
    };
    for (value, expected) in got.iter().zip(oracle.root()) {
        assert!(
            (value.to_f32() - expected).abs() < 5e-2,
            "f16 gather {} vs f32 oracle {expected}",
            value.to_f32()
        );
    }
}

/// [`typed_gather_dim_program`]'s `gathered_dim: 1` shape against the
/// same f32 oracle: the table is laid out `[dim, vocab]` (transposed
/// relative to the `gathered_dim: 0` tests above) so the gather selects
/// a *column*, not a row, proving the typed cursor does not assume the
/// gathered axis is the table's leading one.
#[test]
fn typed_gather_dim1_matches_f32_oracle() {
    let (dim, vocab, seq) = (5usize, 20usize, 4usize);
    let (f32_program, _) = typed_gather_dim_program(
        DType::Float32,
        DType::Int32,
        [dim as u32, vocab as u32],
        seq as u32,
        1,
    );
    let table_data: Vec<f32> = (0..dim * vocab)
        .map(|value| (value % 17) as f32 + 1.0)
        .collect();
    let ids_f32 = [0.0f32, (vocab - 1) as f32, 9.0, 9.0];
    let oracle =
        evaluate(&f32_program, &[], &[&table_data, &ids_f32], &[]).expect("f32 oracle evaluates");

    let (typed_program, _) = typed_gather_dim_program(
        DType::Float32,
        DType::UInt32,
        [dim as u32, vocab as u32],
        seq as u32,
        1,
    );
    let ids_u32: Vec<u32> = ids_f32.iter().map(|&value| value as u32).collect();
    let blocks = [
        TypedBuffer::Float32(table_data.clone()),
        TypedBuffer::UInt32(ids_u32),
    ];
    let results = evaluate_typed(&typed_program, &[], &blocks, &[])
        .expect("gathered_dim: 1 typed gather evaluates");
    let TypedBuffer::Float32(got) = &results[0].2 else {
        panic!("expected an f32 result buffer");
    };
    assert_eq!(
        got.as_slice(),
        oracle.root(),
        "a gathered_dim: 1 typed gather must match the f32 oracle element-for-element"
    );
}

/// A widened reduce ([`TypedPlan::Widened`]) whose own operand is a
/// gathered `f16` table folded into an `f32` accumulator -- proves the
/// two features compose: [`run_widened_program`]'s `TIn`/`TAcc` table
/// split and [`canonical_index_buffers`]'s separate `i64` index table
/// both apply to the same node without interfering.
#[test]
fn widened_reduce_over_a_gathered_f16_operand_matches_a_hand_written_reference() {
    let (vocab, dim, seq) = (10usize, 4usize, 3usize);
    let table_f32: Vec<f32> = (0..vocab * dim)
        .map(|value| (value % 13) as f32 - 6.0)
        .collect();
    let ids = [2u32, 9, 0];

    let mut reference = alloc::vec![0.0f32; seq];
    for (row, &id) in ids.iter().enumerate() {
        let row_start = id as usize * dim;
        reference[row] = table_f32[row_start..row_start + dim]
            .iter()
            .map(|&value| f16::from_f32(value).to_f32())
            .sum();
    }

    let (program, _) = typed_widened_gather_reduce_program(
        DType::Float16,
        DType::Float32,
        DType::UInt32,
        vocab as u32,
        dim as u32,
        seq as u32,
    );
    let table_f16: Vec<f16> = table_f32
        .iter()
        .map(|&value| f16::from_f32(value))
        .collect();
    let blocks = [
        TypedBuffer::Float16(table_f16),
        TypedBuffer::UInt32(ids.to_vec()),
    ];
    let results = evaluate_typed(&program, &[], &blocks, &[])
        .expect("widened reduce over a gather evaluates");
    let TypedBuffer::Float32(got) = &results[0].2 else {
        panic!("expected an f32 accumulator buffer");
    };
    for (value, expected) in got.iter().zip(&reference) {
        assert!(
            (value - expected).abs() < 5e-2,
            "widened gathered reduce {value} vs reference {expected}"
        );
    }
}

/// The sad path a real gather program (not just [`typed_program_plan`])
/// still honours: a float-dtype index node is rejected before any
/// buffer is touched, the same gate
/// [`typed_program_plan_rejects_a_float_gather_index_dtype`] proves at
/// the plan level alone.
#[test]
fn evaluate_typed_rejects_a_float_gather_index_dtype_at_execution() {
    let program = typed_gather_program(DType::Float32, DType::Float64);
    let error = evaluate_typed(&program, &[], &[], &[]).expect_err(
        "a float-dtype gather index must be rejected at execution, never silently accepted",
    );
    assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
}

/// The f32 oracle's own out-of-range behaviour
/// ([`a_fetched_index_past_the_extent_is_a_real_error_not_ub`]): a
/// fetched index `>= extent` is `TensorError::GatherIndexOutOfRange`,
/// never a clamp, a wraparound, or UB. The typed evaluator must answer
/// the identical class of index for the identical class of input,
/// proven for both the positive-overflow and the negative-index cases
/// [`GatherCursor::fetch_and_advance`] (`proxima-tensor/src/cpu.rs`)
/// checks in one `index < 0 || index as u64 >= self.extent` guard.
#[proxima::test]
#[case::index_past_the_extent(4)]
#[case::negative_index(-1)]
async fn typed_gather_out_of_range_index_matches_f32_oracle_error_shape(#[case] bad_index: i32) {
    let (vocab, dim, seq) = (4usize, 2usize, 1usize);
    let (f32_program, _) = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| value as f32).collect();
    let ids_f32 = [bad_index as f32];
    let oracle_error = evaluate(&f32_program, &[], &[&table_data, &ids_f32], &[])
        .expect_err("the f32 oracle rejects the out-of-range index");
    let TensorError::GatherIndexOutOfRange {
        extent: oracle_extent,
        ..
    } = oracle_error
    else {
        panic!("expected the f32 oracle's own GatherIndexOutOfRange, got {oracle_error}");
    };

    let (typed_program, _) = typed_embedding_lookup_program(
        DType::Float32,
        DType::Int32,
        vocab as u32,
        dim as u32,
        seq as u32,
    );
    let blocks = [
        TypedBuffer::Float32(table_data),
        TypedBuffer::Int32(alloc::vec![bad_index]),
    ];
    let typed_error = evaluate_typed(&typed_program, &[], &blocks, &[])
        .expect_err("the typed evaluator rejects it too");
    let TensorError::GatherIndexOutOfRange {
        index: typed_index,
        extent: typed_extent,
        ..
    } = typed_error
    else {
        panic!("expected GatherIndexOutOfRange, got {typed_error}");
    };
    assert_eq!(typed_index, i64::from(bad_index));
    assert_eq!(
        typed_extent, oracle_extent,
        "the typed evaluator must bounds-check against the same extent"
    );
}

/// The honest boundary [`canonical_index_buffers`] draws rather than
/// guessing: a gather index node computed in-program (an [`Op::Iota`]
/// here, not a caller-supplied block) is a named `NotLowerable`, not a
/// silently wrong execution — see [`canonical_index_buffers`]'s own doc.
#[test]
fn evaluate_typed_names_a_computed_gather_index_node_as_not_yet_supported() {
    let mut program = Vec::new();
    let table = block(
        &mut program,
        DType::Float32,
        &[Extent::Static(4), Extent::Static(2)],
    );
    let ids = append(
        &mut program,
        Op::Iota {
            dtype: DType::Int32,
            extent: Extent::Static(3),
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: map::IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
                map::AxisIndex::default(),
                map::AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
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
            operands: alloc::vec![(table, gathered_map)],
            name: None,
        },
    );
    let table_data = TypedBuffer::Float32(alloc::vec![0.0; 8]);
    let error = evaluate_typed(&program, &[], &[table_data], &[])
        .expect_err("a computed gather index node must be a named gap, not a silent guess");
    assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
}

#[test]
fn i16_operand_i64_accumulator_reduce_survives_i16_overflow() {
    // three i16 elements of 20000 each: the true sum is 60000, which
    // does not fit in i16 (max 32767) -- wrapping i16 arithmetic would
    // land on -5536 (60000 - 65536). an i64 accumulator is the only way
    // to observe 60000.
    let (program, _) = typed_widened_reduce_program(DType::Int16, DType::Int64, 3);
    let operand = TypedBuffer::Int16(alloc::vec![20000, 20000, 20000]);
    let results = evaluate_typed(&program, &[], &[operand], &[])
        .expect("an i16-operand, i64-accumulator reduce evaluates");
    assert_eq!(
        results[0].2,
        TypedBuffer::Int64(alloc::vec![60000]),
        "the i64 accumulator must carry the true sum, not the i16-wrapped one (-5536)"
    );
}

#[test]
fn u8_operand_u32_accumulator_reduce_survives_u8_overflow() {
    // three u8 elements of 200 each: the true sum is 600, which does
    // not fit in u8 (max 255) -- wrapping u8 arithmetic would land on
    // 88 (600 - 512). a u32 accumulator is the only way to observe 600.
    let (program, _) = typed_widened_reduce_program(DType::UInt8, DType::UInt32, 3);
    let operand = TypedBuffer::UInt8(alloc::vec![200, 200, 200]);
    let results = evaluate_typed(&program, &[], &[operand], &[])
        .expect("a u8-operand, u32-accumulator reduce evaluates");
    assert_eq!(
        results[0].2,
        TypedBuffer::UInt32(alloc::vec![600]),
        "the u32 accumulator must carry the true sum, not the u8-wrapped one (88)"
    );
}

fn typed_reduce_vector_to_scalar_program(dtype: DType, len: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let operand = block(&mut program, dtype, &[Extent::Static(len)]);
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

#[proxima::test]
#[case::int32(DType::Int32, TypedBuffer::Int32(alloc::vec![1, 2, 3, 4]), TypedBuffer::Int32(alloc::vec![10]))]
#[case::uint64(DType::UInt64, TypedBuffer::UInt64(alloc::vec![1, 2, 3, 4]), TypedBuffer::UInt64(alloc::vec![10]))]
#[case::float64(DType::Float64, TypedBuffer::Float64(alloc::vec![1.5, 2.5, 3.0, 4.0]), TypedBuffer::Float64(alloc::vec![11.0]))]
async fn evaluate_typed_reduces_a_vector_to_a_scalar_across_widths(
    #[case] dtype: DType,
    #[case] operand: TypedBuffer,
    #[case] expected: TypedBuffer,
) {
    let (program, _) = typed_reduce_vector_to_scalar_program(dtype, 4);
    let results = evaluate_typed(&program, &[], &[operand], &[]).expect("typed reduce evaluates");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].2, expected);
}

#[test]
fn evaluate_typed_scans_an_integer_vector_producing_a_running_sum() {
    let mut program = Vec::new();
    let source = block(&mut program, DType::Int32, &[Extent::Static(5)]);
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Int32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[0])),
            keep: Keep::Scan,
            name: None,
        }),
    );
    let blocks = [TypedBuffer::Int32(alloc::vec![1, 2, 3, 4, 5])];
    let results = evaluate_typed(&program, &[], &blocks, &[]).expect("typed scan evaluates");
    assert_eq!(
        results[0].2,
        TypedBuffer::Int32(alloc::vec![1, 3, 6, 10, 15])
    );
}

/// Matmul-shaped: a `Multiply` elementwise body fused into an `Add`
/// reduce, same construction as [`matmul_program`] with `dtype`
/// parameterized so it can run through [`evaluate_typed`] at any width.
fn typed_matmul_program(dtype: DType, m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let lhs = block(&mut program, dtype, &[Extent::Static(m), Extent::Static(k)]);
    let rhs = block(&mut program, dtype, &[Extent::Static(k), Extent::Static(n)]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("typed_matmul".into()),
        }),
    );
    (program, lhs, rhs, sum)
}

#[test]
fn evaluate_typed_matmul_shaped_reduce_matches_a_naive_reference_at_int32() {
    let (m, k, n) = (3usize, 4usize, 2usize);
    let (program, _, _, _) = typed_matmul_program(DType::Int32, m as u32, k as u32, n as u32);
    let lhs: Vec<i32> = (0..(m * k) as i32).collect();
    let rhs: Vec<i32> = (0..(k * n) as i32).collect();
    let blocks = [
        TypedBuffer::Int32(lhs.clone()),
        TypedBuffer::Int32(rhs.clone()),
    ];
    let results = evaluate_typed(&program, &[], &blocks, &[]).expect("typed matmul evaluates");

    let mut expected = vec![0i32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0i32;
            for inner in 0..k {
                sum += lhs[row * k + inner] * rhs[inner * n + col];
            }
            expected[row * n + col] = sum;
        }
    }
    assert_eq!(results[0].2, TypedBuffer::Int32(expected));
}

#[test]
fn evaluate_typed_matmul_shaped_reduce_matches_a_naive_reference_at_float64() {
    let (m, k, n) = (3usize, 4usize, 2usize);
    let (program, _, _, _) = typed_matmul_program(DType::Float64, m as u32, k as u32, n as u32);
    let lhs: Vec<f64> = (0..m * k).map(|value| value as f64 * 0.5).collect();
    let rhs: Vec<f64> = (0..k * n).map(|value| value as f64 * 0.25).collect();
    let blocks = [
        TypedBuffer::Float64(lhs.clone()),
        TypedBuffer::Float64(rhs.clone()),
    ];
    let results = evaluate_typed(&program, &[], &blocks, &[]).expect("typed matmul evaluates");

    let mut expected = vec![0.0f64; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f64;
            for inner in 0..k {
                sum += lhs[row * k + inner] * rhs[inner * n + col];
            }
            expected[row * n + col] = sum;
        }
    }
    let TypedBuffer::Float64(actual) = &results[0].2 else {
        panic!("expected a Float64 result");
    };
    for (found, expect) in actual.iter().zip(expected.iter()) {
        assert!((found - expect).abs() < 1e-9, "{found} vs {expect}");
    }
}

/// `T = f32` is the specialization [`run_reduce_typed`] delegates
/// straight back to the existing NEON-tiled [`run_reduce`] — this checks
/// [`evaluate_typed`] and [`evaluate`] agree bit-for-bit on the exact
/// same matmul-shaped program, which they only can if both ran the same
/// function. (Whether the NEON tile itself fired, as opposed to one of
/// `run_reduce`'s other f32 fast paths, is checked separately by
/// `evaluate_typed_float32_matmul_shaped_reduce_fires_the_neon_tile`,
/// gated on `feature = "instrument"`.)
#[test]
fn evaluate_typed_float32_matmul_shaped_reduce_matches_evaluate_bit_for_bit() {
    let (m, k, n) = (6usize, 32usize, 8usize);
    let (program, _, _, _) = typed_matmul_program(DType::Float32, m as u32, k as u32, n as u32);
    let lhs: Vec<f32> = (0..m * k)
        .map(|value| (value as f32 * 0.0137).sin())
        .collect();
    let rhs: Vec<f32> = (0..k * n)
        .map(|value| (value as f32 * 0.0271).cos())
        .collect();

    let via_evaluate = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("f32 matmul evaluates");
    let blocks = [TypedBuffer::Float32(lhs), TypedBuffer::Float32(rhs)];
    let via_typed =
        evaluate_typed(&program, &[], &blocks, &[]).expect("typed f32 matmul evaluates");
    let TypedBuffer::Float32(typed_data) = &via_typed[0].2 else {
        panic!("expected a Float32 result");
    };
    assert_eq!(typed_data.as_slice(), via_evaluate.root());
}

/// Same claim as the test above, but over the RHS-transposed layout
/// that actually engages `neon_tile_plan`/`gemm_tile_neon` (see
/// `evaluate_typed_float32_matmul_shaped_reduce_fires_the_neon_tile`'s
/// doc on why plain `matmul_program`'s layout hits `width_tile_plan`
/// instead), with a contraction (`k = 64`) long enough for the NEON
/// tile's own row/lane splitting to matter, and compared via `to_bits`
/// rather than `==` — the generic nest and the NEON nest accumulate in
/// a different order, so a fallthrough from one to the other changes
/// bits even where it would not change `==` (e.g. `-0.0` vs `0.0`).
/// No feature gate: this is the check that fails the *default* gate if
/// `run_reduce_typed`'s `T == f32` specialization silently stops firing,
/// unlike `..._fires_the_neon_tile` below, which only runs under
/// `instrument` and checks the counters instead of the bits.
#[test]
fn evaluate_typed_float32_matmul_rhs_transposed_matches_evaluate_bit_for_bit() {
    let (m, k, n) = (12usize, 64usize, 8usize);
    let (program, _) = matmul_program_rhs_transposed(m as u32, k as u32, n as u32);
    let lhs = random_vec(0x1234_5678_9abc_def0, m * k);
    let rhs = random_vec(0x0fed_cba9_8765_4321, n * k);

    let via_evaluate = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("f32 matmul evaluates");
    let blocks = [TypedBuffer::Float32(lhs), TypedBuffer::Float32(rhs)];
    let via_typed =
        evaluate_typed(&program, &[], &blocks, &[]).expect("typed f32 matmul evaluates");
    let TypedBuffer::Float32(typed_data) = &via_typed[0].2 else {
        panic!("expected a Float32 result");
    };

    assert_eq!(typed_data.len(), via_evaluate.root().len());
    let compared = typed_data.len();
    assert!(
        compared > 0,
        "the bit-identity check compared zero elements"
    );
    for (index, (found, expected)) in typed_data.iter().zip(via_evaluate.root()).enumerate() {
        assert_eq!(
            found.to_bits(),
            expected.to_bits(),
            "node {index}: evaluate_typed produced {found} (bits {:#010x}), \
             evaluate produced {expected} (bits {:#010x})",
            found.to_bits(),
            expected.to_bits(),
        );
    }
}

#[test]
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
fn evaluate_typed_float32_matmul_shaped_reduce_fires_the_neon_tile() {
    // `neon_tile_plan`'s gate (cpu.rs's own doc on that function) wants
    // both contraction strides == 1 with the *width* dim non-contiguous
    // on one operand — the RHS-transposed layout
    // `matmul_program_rhs_transposed` uses, not plain `matmul_program`'s
    // (whose RHS is `[k, n]` contiguous in `n` and hits `width_tile_plan`
    // instead). Mirrored here rather than reusing `typed_matmul_program`.
    let (m, k, n) = (12usize, 64usize, 8usize);
    let mut program = Vec::new();
    let lhs = block(
        &mut program,
        DType::Float32,
        &[Extent::Static(m as u32), Extent::Static(k as u32)],
    );
    let rhs = block(
        &mut program,
        DType::Float32,
        &[Extent::Static(n as u32), Extent::Static(k as u32)],
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[1, 2]))),
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
            name: Some("typed_matmul_rhs_transposed".into()),
        }),
    );
    let lhs_data: Vec<f32> = (0..m * k)
        .map(|value| (value as f32 * 0.0137).sin())
        .collect();
    let rhs_data: Vec<f32> = (0..n * k)
        .map(|value| (value as f32 * 0.0271).cos())
        .collect();
    let blocks = [
        TypedBuffer::Float32(lhs_data),
        TypedBuffer::Float32(rhs_data),
    ];

    let (gate_before, invocations_before, _) = neon_tile_counters();
    evaluate_typed(&program, &[], &blocks, &[]).expect("typed f32 matmul evaluates");
    let (gate_after, invocations_after, _) = neon_tile_counters();

    assert!(
        gate_after > gate_before,
        "neon_tile_plan never matched through evaluate_typed"
    );
    assert!(
        invocations_after > invocations_before,
        "gemm_tile_neon never ran through evaluate_typed"
    );
}

#[test]
fn evaluate_typed_rejects_a_block_whose_dtype_does_not_match_the_program() {
    let (program, _, _, _) = typed_add_program(DType::Int32, 2);
    let blocks = [
        TypedBuffer::Int32(alloc::vec![1, 2]),
        TypedBuffer::Int8(alloc::vec![1, 2]),
    ];
    let error = evaluate_typed(&program, &[], &blocks, &[])
        .expect_err("Int8 block cannot bind an Int32 program");
    assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
}

// -- operand_access_footprint: pure, always compiled under `cfg(test)`
// regardless of the `instrument` feature, so these run in every default
// `cargo nextest run -p proxima-tensor` too, not only `--features
// instrument` (see the function's own `#[cfg(any(feature = "instrument",
// test))]`).

#[test]
fn operand_access_footprint_is_exact_for_a_dense_operand() {
    let extents = [4u64, 5, 6];
    let strides = [30i64, 6, 1];
    let (reads, distinct) = operand_access_footprint(&extents, &strides);
    assert_eq!(reads, 4 * 5 * 6);
    assert_eq!(
        distinct, reads,
        "a dense operand's own footprint is read exactly once per position"
    );
}

#[test]
fn operand_access_footprint_undercounts_distinct_under_broadcast() {
    let extents = [4u64, 5, 6];
    let strides = [30i64, 0, 1]; // broadcast over the middle axis
    let (reads, distinct) = operand_access_footprint(&extents, &strides);
    assert_eq!(
        reads,
        4 * 5 * 6,
        "every iterated position still counts as a read"
    );
    assert_eq!(
        distinct,
        4 * 6,
        "the broadcast axis contributes 1, not its extent, to distinct"
    );
    assert!(reads > distinct);
}

#[test]
fn operand_access_footprint_is_one_for_a_scalar_reduction() {
    assert_eq!(operand_access_footprint(&[], &[]), (1, 1));
}

// -- proof tests: the instrument must measure something real, not just
// increment. Each asserts a number known a priori from the program's
// own construction, gated behind `instrument` since the API under test
// only exists there.

/// A matmul's RHS is read once per `(m, n, k)` position but only ever
/// resolves to `k * n` distinct elements — it never varies along `m`.
/// Asserts `reads >> distinct`, the shape a cold-weight quantization
/// decision needs.
#[test]
#[cfg(feature = "instrument")]
fn evaluate_records_more_reads_than_distinct_elements_for_a_broadcast_operand() {
    instrument::reset_operand_access();
    let (m, k, n) = (4u32, 3u32, 5u32);
    let (program, sum) = matmul_program(m, k, n, false);
    let lhs_data = random_vec(1, (m * k) as usize);
    let rhs_data = random_vec(2, (k * n) as usize);
    evaluate(&program, &[], &[&lhs_data, &rhs_data], &[sum]).expect("matmul evaluates");

    let lhs_access = instrument::operand_access_of(NodeId(0)).expect("lhs was instrumented");
    assert_eq!(
        lhs_access.distinct_elements,
        u64::from(m) * u64::from(k),
        "lhs's real footprint excludes the n broadcast axis"
    );
    assert!(
        lhs_access.reads > lhs_access.distinct_elements,
        "reads={} distinct={}",
        lhs_access.reads,
        lhs_access.distinct_elements
    );

    let rhs_access = instrument::operand_access_of(NodeId(1)).expect("rhs was instrumented");
    assert_eq!(rhs_access.distinct_elements, u64::from(k) * u64::from(n));
    assert!(
        rhs_access.reads > rhs_access.distinct_elements,
        "reads={} distinct={}",
        rhs_access.reads,
        rhs_access.distinct_elements
    );
}

/// The case that matters most: an embedding lookup into a 1000-row
/// table, fetching only 3 distinct rows across 6 positions (each row
/// hit twice). A naive "count every read" instrument would report all
/// 1000 rows touched (or the raw read count); this must report exactly
/// 3 rows' worth of elements.
#[test]
#[cfg(feature = "instrument")]
fn evaluate_records_exactly_the_distinct_rows_a_gather_touches_not_the_whole_table() {
    instrument::reset_operand_access();
    let (vocab, dim, seq) = (1_000u32, 8u32, 6u32);
    let (program, gathered) = embedding_lookup_program(vocab, dim, seq);
    let table_data: Vec<f32> = (0..(vocab * dim) as usize)
        .map(|value| value as f32)
        .collect();
    // 6 fetches, 3 distinct rows: 3 and 999 and 500 each hit twice.
    let ids_data = [3.0f32, 3.0, 999.0, 999.0, 500.0, 500.0];
    evaluate(&program, &[], &[&table_data, &ids_data], &[gathered]).expect("gather evaluates");

    let table_access = instrument::operand_access_of(NodeId(0)).expect("table was instrumented");
    assert_eq!(
        table_access.distinct_elements,
        3 * u64::from(dim),
        "only 3 of {vocab} rows were ever fetched, not the whole table"
    );
    assert_eq!(
        table_access.total_elements,
        u64::from(vocab) * u64::from(dim)
    );
    assert!(table_access.distinct_elements < table_access.total_elements);
}

/// Degenerate control: an operand the requested outputs never reach
/// gets no `BoundOp` at all, so it must read back `None` — absent, not
/// a zero-touch row assumed on its behalf. Paired with a direct API
/// check that a REAL zero-read record (an operand that was reached, and
/// genuinely read zero times) reads back `Some` with every field `0`,
/// so the two "zero" cases are distinguishable rather than folded
/// together.
#[test]
#[cfg(feature = "instrument")]
fn operand_access_distinguishes_never_read_from_a_recorded_zero() {
    instrument::reset_operand_access();
    let (m, k, n) = (2u32, 2u32, 2u32);
    let (mut program, sum) = matmul_program(m, k, n, false);
    let unused = f32_block(&mut program, &[Extent::Static(3)]);
    let lhs_data = random_vec(1, (m * k) as usize);
    let rhs_data = random_vec(2, (k * n) as usize);
    let unused_data = alloc::vec![0.0f32; 3];
    evaluate(&program, &[], &[&lhs_data, &rhs_data, &unused_data], &[sum])
        .expect("matmul evaluates");

    assert!(
        instrument::operand_access_of(NodeId(0)).is_some(),
        "lhs was actually read"
    );
    assert_eq!(
        instrument::operand_access_of(unused),
        None,
        "an operand the requested outputs never reach is absent, not a recorded zero"
    );

    instrument::reset_operand_access();
    let node = NodeId(42);
    assert_eq!(
        instrument::operand_access_of(node),
        None,
        "nothing recorded yet"
    );
    instrument::record_operand_access(node, 0, 0, 128);
    let access =
        instrument::operand_access_of(node).expect("recording zero reads still creates a row");
    assert_eq!(access.reads, 0);
    assert_eq!(access.distinct_elements, 0);
    assert_eq!(
        access.total_elements, 128,
        "total size is known even when nothing was ever read"
    );
}

/// DLMF 7.2 / Abramowitz & Stegun Table 7.1's published `erf(x)`, to the
/// precision commonly republished — the oracle `erf_f32_matches_reference_values`
/// and `erf_f64_matches_reference_values` sweep against, independent of
/// this crate's own approximation.
const ERF_REFERENCE: &[(f64, f64)] = &[
    (0.0, 0.0),
    (0.2, 0.222_702_589_2),
    (0.4, 0.428_392_355_0),
    (0.6, 0.603_856_090_8),
    (0.8, 0.742_100_964_7),
    (1.0, 0.842_700_792_9),
    (1.2, 0.910_313_978_2),
    (1.4, 0.952_285_119_8),
    (1.6, 0.976_348_383_3),
    (1.8, 0.989_090_501_6),
    (2.0, 0.995_322_265_0),
    (2.5, 0.999_593_048_0),
    (3.0, 0.999_977_909_5),
    (5.0, 1.0),
];

/// Abramowitz & Stegun 7.1.26's published maximum absolute error is
/// `1.5e-7`. Measured here against [`ERF_REFERENCE`] (swept across both
/// signs via `erf_f32(-x) == -erf_f32(x)`), the actual max error is
/// `1.1920929e-7`, equal to `f32::EPSILON` (`2^-23`) exactly — this
/// approximation is at, not below, the "below the type's own epsilon"
/// bar for `f32`; it does not clear it outright, but it also is not
/// dominated by formula error rather than `f32`'s own rounding.
#[test]
fn erf_f32_matches_reference_values_within_f32_epsilon() {
    let mut max_error = 0.0f32;
    for &(x, reference) in ERF_REFERENCE {
        let (x, reference) = (x as f32, reference as f32);
        let positive_error = (erf_f32(x) - reference).abs();
        let negative_error = (erf_f32(-x) - (-reference)).abs();
        max_error = max_error.max(positive_error).max(negative_error);
    }
    assert!(
        max_error <= 1.5 * f32::EPSILON,
        "measured max abs error {max_error} should stay within 1.5x f32::EPSILON ({}); the \
         published Abramowitz & Stegun bound is 1.5e-7, essentially f32::EPSILON itself",
        f32::EPSILON
    );
}

/// Same formula, same reference table, `f64` throughout: isolates how
/// much of `erf_f32`'s error is the formula itself versus f32 rounding
/// compounding on top of it.
#[test]
fn erf_f64_matches_reference_values() {
    let mut max_error = 0.0f64;
    for &(x, reference) in ERF_REFERENCE {
        let positive_error = (erf_f64(x) - reference).abs();
        let negative_error = (erf_f64(-x) - (-reference)).abs();
        max_error = max_error.max(positive_error).max(negative_error);
    }
    assert!(
        max_error < 2e-7,
        "measured f64 max abs error {max_error} exceeds the ~1.5e-7 published bound"
    );
}

/// `elementwise_width_unary`'s dispatch table (the fast path
/// `evaluate`/`evaluate_parallel` actually run through) used to end in
/// `_ => unreachable!("BodyShape::Unary only ever carries an arity-1
/// ScalarOp")` — a real panic, not a fallback, for any arity-1
/// `ScalarOp` this match does not name explicitly. Evaluating a real
/// `Op::Elementwise { body: ScalarOp::Erf, .. }` program end to end is
/// what proves `Erf`'s arm was actually added there, not merely to
/// `apply_scalar_op`'s slow general path.
#[test]
fn erf_evaluates_through_a_real_elementwise_program() {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let output = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Erf,
            operands: alloc::vec![(input, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );

    let values: [f32; 4] = [0.0, 0.5, 1.0, -1.5];
    let blocks: [&[f32]; 1] = [&values];
    let evaluated = evaluate(&program, &[], &blocks, &[output]).expect("erf elementwise evaluates");

    let found = evaluated.root();
    assert_eq!(found.len(), values.len());
    for (result, &raw_value) in found.iter().zip(values.iter()) {
        let expected = erf_f32(raw_value);
        assert!(
            (result - expected).abs() < 1e-6,
            "elementwise erf({raw_value}) = {result}, direct erf_f32 gives {expected}"
        );
    }
}

/// [`matmul_q4k_f32`] against the incumbent: quantize a random weight
/// matrix, then compare its output row-for-row to plain
/// dequantize-then-`f32`-dot-product on the same bytes. Both paths
/// call the identical [`proxima_gguf::quant::q4_k::dequantize_block`]
/// codec, so the only source of disagreement is accumulation order —
/// this path folds with [`f32::mul_add`] one super-block at a time
/// ([`dot_q4k_f32`]), the reference sums a plain `iter().zip().map()`
/// over the fully-dequantized row — so a nonzero difference is
/// expected (guiding-principle 14/19: report the measured number,
/// never assert bit-exact equality here).
#[test]
fn matmul_q4k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows = 5;
    let blocks_per_row = 3;
    let k = QK_K * blocks_per_row;

    // realistic weight-scale values, not degenerate all-zero/constant
    // inputs — `Lcg::next_unit` is already this file's own random-f32
    // fixture generator (see `random_vec` above).
    let activation: Vec<f32> = random_vec(7, k)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(11, rows * k)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of QK_K by construction");
    }

    // incumbent: dequantize the packed bytes back to f32, then a plain
    // dot product per row — never touches `dot_q4k_f32`/`matmul_q4k_f32`.
    let mut expected = Vec::with_capacity(rows);
    for row_blocks in weight_blocks.chunks_exact(blocks_per_row * BLOCK_BYTES) {
        let mut dequantized = vec![0.0f32; k];
        dequantize(row_blocks, &mut dequantized)
            .expect("row_blocks is a whole number of q4_k super-blocks");
        let dot: f32 = dequantized
            .iter()
            .zip(activation.iter())
            .map(|(&weight, &value)| weight * value)
            .sum();
        expected.push(dot);
    }

    let actual =
        matmul_q4k_f32(&weight_blocks, rows, &activation).expect("well-formed quantized matmul");

    assert_eq!(actual.len(), expected.len());
    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f64;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "quantized matmul row produced a non-finite value: {got}"
        );
        let diff = (got - want).abs();
        max_error = max_error.max(diff);
        sum_sq_error += f64::from(diff) * f64::from(diff);
    }
    let rms_error = (sum_sq_error / rows as f64).sqrt();
    eprintln!(
        "matmul_q4k_f32 vs dequantize-then-matmul: max_error={max_error} rms_error={rms_error}"
    );

    // loose sanity bound around the accumulation-order float noise
    // floor for a 768-element dot product at this value scale — not
    // tuned to the measured numbers, matching this crate's existing
    // q4_k round-trip test convention (`proxima-gguf`'s
    // `quantize_dequantize_smooth_signal_round_trip_error`).
    assert!(
        max_error < 0.05,
        "max_error={max_error} exceeds loose sanity bound"
    );
    assert!(
        rms_error < 0.02,
        "rms_error={rms_error} exceeds loose sanity bound"
    );
}

/// [`dot_q4k_f32`]'s shape-mismatch guard: an activation slice whose
/// length does not match the weight row's decoded element count is
/// rejected, not silently truncated or padded.
#[test]
fn matmul_q4k_f32_rejects_an_activation_length_that_does_not_match_the_weight_rows_element_count() {
    use proxima_gguf::quant::q4_k::BLOCK_BYTES;

    let weight_blocks = vec![0u8; BLOCK_BYTES];
    let wrong_length_activation = vec![0.0f32; 200];
    let error = matmul_q4k_f32(&weight_blocks, 1, &wrong_length_activation).unwrap_err();
    assert!(
        matches!(error, TensorError::QuantizedShapeMismatch { .. }),
        "got {error:?}"
    );
}

/// [`matmul_q4k_q8k_f32`] against the SAME incumbent
/// [`matmul_q4k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`]
/// checks against: dequantize the packed `Q4_K` bytes, plain `f32` dot.
/// This path never touches `dequantize`/`dot_q4k_f32` at all -- every
/// weight byte is read once, as a nibble, straight into an integer
/// accumulate -- so a nonzero difference from the `f32` reference is
/// expected (Q8_K's own quantization error, `iscale = -127/max`, is a
/// second lossy step this path pays that `dot_q4k_f32` does not).
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn matmul_q4k_q8k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows = 5;
    let blocks_per_row = 3;
    let k = QK_K * blocks_per_row;

    let activation: Vec<f32> = random_vec(7, k)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(11, rows * k)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of QK_K by construction");
    }

    let mut expected = Vec::with_capacity(rows);
    for row_blocks in weight_blocks.chunks_exact(blocks_per_row * BLOCK_BYTES) {
        let mut dequantized = vec![0.0f32; k];
        dequantize(row_blocks, &mut dequantized)
            .expect("row_blocks is a whole number of q4_k super-blocks");
        let dot: f32 = dequantized
            .iter()
            .zip(activation.iter())
            .map(|(&weight, &value)| weight * value)
            .sum();
        expected.push(dot);
    }

    let actual = matmul_q4k_q8k_f32(&weight_blocks, rows, &activation)
        .expect("well-formed packed int8 matmul");

    assert_eq!(actual.len(), expected.len());
    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f64;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "packed int8 matmul row produced a non-finite value: {got}"
        );
        let diff = (got - want).abs();
        max_error = max_error.max(diff);
        sum_sq_error += f64::from(diff) * f64::from(diff);
    }
    let rms_error = (sum_sq_error / rows as f64).sqrt();
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative_max_error = max_error / max_magnitude;
    eprintln!(
        "matmul_q4k_q8k_f32 vs dequantize-then-matmul: max_error={max_error} rms_error={rms_error} \
         max_magnitude={max_magnitude} relative_max_error={relative_max_error}"
    );
    // Unlike `matmul_q4k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`
    // above (whose only source of disagreement is accumulation order --
    // both arms there consume the SAME dequantized bytes, so an
    // absolute bound near the float noise floor is right), this path
    // ALSO quantizes the activation to Q8_K, a second real lossy step.
    // The dot magnitude here runs into the thousands (this fixture's
    // `random_vec`-derived data is not zero-mean), so an absolute
    // bound copied from that other test would be meaningless -- this
    // is RELATIVE error against the signal's own magnitude, still a
    // loose sanity bound and still not tuned to the measured number.
    assert!(
        relative_max_error < 0.01,
        "relative_max_error={relative_max_error} (max_error={max_error} over magnitude {max_magnitude}) \
         exceeds loose sanity bound"
    );
}

/// Three-way parity check on a real `Q4_K` row (openchat-3.5-1210,
/// `blk.0.ffn_gate.weight`): an f64 dequant-then-fold reference against
/// (1) [`matmul_q4k_f32`] -- the exact dequantize-then-fold kernel
/// `exact_activations` routes to -- and (2) [`matmul_q4k_q8k_f32`], the
/// Q8_K activation-quantized fast path `q4k-int8-dot` defaults on. This
/// is the CPU-side half of the finding behind `exact_activations`
/// (`evaluate_quantized_exact`): (1) matches the f64 reference to fp32
/// rounding, while (2) carries Q8_K's own real quantization error on
/// top -- the error every pre-existing cross-backend harness was
/// attributing to Metal instead.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn matmul_q4k_f32_matches_an_f64_dequant_reference_tighter_than_the_int8_path_on_a_real_row() {
    let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.ffn_gate.weight",
        proxima_gguf::types::GgmlType::Q4_K,
    ) else {
        eprintln!("blk.0.ffn_gate.weight is not Q4_K in this file; test skipped, not faked");
        return;
    };
    let row_bytes = weight_bytes.len() / out_dim;
    let first_row = &weight_bytes[..row_bytes];

    let activation: Vec<f32> = random_vec(701, in_dim)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();

    let mut scratch = [0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut f64_reference = 0.0f64;
    for (block, activation_chunk) in first_row
        .as_chunks::<Q4K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4K_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q4_k::dequantize_block(block, &mut scratch);
        for (&weight, &input) in scratch.iter().zip(activation_chunk) {
            f64_reference += f64::from(weight) * f64::from(input);
        }
    }

    let exact = dot_q4k_f32(first_row, &activation).expect("well-formed exact q4_k dot");
    let int8 =
        matmul_q4k_q8k_f32(first_row, 1, &activation).expect("well-formed packed int8 matmul")[0];

    let magnitude = f64_reference.abs().max(1.0);
    let exact_relative_error = ((f64::from(exact) - f64_reference) / magnitude).abs();
    let int8_relative_error = ((f64::from(int8) - f64_reference) / magnitude).abs();
    eprintln!(
        "ffn_gate row0 (real Q4_K bytes) vs f64 dequant reference: f64_reference={f64_reference} \
         exact={exact} exact_relative_error={exact_relative_error} \
         int8={int8} int8_relative_error={int8_relative_error}"
    );
    assert!(
        exact_relative_error < 1e-5,
        "exact q4_k dot diverged from its own f64 dequant reference beyond fp32 rounding: \
         exact={exact} f64_reference={f64_reference} relative_error={exact_relative_error}"
    );
    assert!(
        exact_relative_error < int8_relative_error,
        "exact_activations must not carry MORE error against the f64 reference than the \
         int8 path it replaces: exact_relative_error={exact_relative_error} \
         int8_relative_error={int8_relative_error}"
    );
}

/// Shape-coverage regression: the test above only ever exercised `rows =
/// 5`, so nothing in this crate's test suite verified
/// [`dot_q4k_q8k`]/[`matmul_q4k_q8k_f32`] at the `out_dim` (row count)
/// real Mistral-7B tensors actually carry -- 4096 (attention/FFN square
/// projections). A proxima-debugger probe (`q4k_bisect_probe.rs`,
/// `proxima-wt-gpuker`) reported the CPU int8-dot path diverging from a
/// dequantize-then-dot oracle by "up to 872%" once `out_dim >= 1024` and
/// concluded the kernel was wrong at scale.
///
/// This test proves that conclusion false via two independently
/// instrumented comparisons over the SAME 4096-row fixture, verified
/// with `dot_q4k_q8k` isolated from its own quantized inputs
/// (`proxima-tensor/examples/q4k_int8_isolate.rs`, run manually at
/// `IN_DIM=4096 OUT_DIM=14336`): `dot_q4k_q8k` agreed with an oracle fed
/// the SAME `Q8_K`-quantized activation to within 6e-5 at every row
/// checked (0, 1, 7000, 14335) -- the kernel's integer accumulation,
/// scale, and `dmin` application are exact.
///
/// `per_row_relative_error` below (`diff / |reference|`, the probe's own
/// metric) DOES blow past 1% for some row here, reproducing the
/// reported symptom -- but only because `reference` (a 4096-wide random
/// dot product) lands near a zero-crossing for at least one of 4096
/// independent rows, an outcome max-relative-error metrics over N
/// samples become guaranteed to hit as N grows, regardless of how small
/// the underlying error is. `relative_max_error` (this crate's existing
/// convention, normalized by the batch's own `max_magnitude` rather than
/// each row's own reference) stays flat and small at this shape --
/// proof the "872%" figure is an artifact of the probe's per-row metric,
/// not a growing arithmetic error. The only real error source is
/// `Q8_K`'s int8 activation quantization, present and bounded (~std <
/// 0.1 at this magnitude/width) by design -- the entire reason
/// `q4k-int8-dot` trades precision for the documented throughput win.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn matmul_q4k_q8k_f32_stays_within_tolerance_at_real_forward_out_dim_above_the_reported_threshold()
{
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows = 4096;
    let blocks_per_row = 16;
    let k = QK_K * blocks_per_row;

    // per-row-reseeded (`1000 + row`), matching `q4k_bisect_probe.rs`'s
    // own fixture shape exactly (`17 + row`) rather than one continuous
    // stream sliced into row-length chunks: that is what surfaces the
    // reported "some row's true dot product lands near a zero crossing"
    // statistics this test's own doc explains -- a single long stream
    // sliced by row instead produced implausibly large dot-product
    // magnitudes here (~4400 vs the ~20 this row count/width naturally
    // implies for zero-mean unit-variance inputs), i.e. it was NOT
    // reproducing independent rows.
    let activation: Vec<f32> = random_vec(97, k);
    let rows_f32: Vec<Vec<f32>> = (0..rows)
        .map(|row| random_vec(1000 + row as u64, k))
        .collect();

    let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in rows_f32
        .iter()
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of QK_K by construction");
    }

    let mut expected = Vec::with_capacity(rows);
    for row_blocks in weight_blocks.chunks_exact(blocks_per_row * BLOCK_BYTES) {
        let mut dequantized = vec![0.0f32; k];
        dequantize(row_blocks, &mut dequantized)
            .expect("row_blocks is a whole number of q4_k super-blocks");
        let dot: f32 = dequantized
            .iter()
            .zip(activation.iter())
            .map(|(&weight, &value)| weight * value)
            .sum();
        expected.push(dot);
    }

    let actual = matmul_q4k_q8k_f32(&weight_blocks, rows, &activation)
        .expect("well-formed packed int8 matmul at real forward out_dim");

    assert_eq!(actual.len(), expected.len());
    let mut max_error = 0.0f32;
    let mut max_per_row_relative_error = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "packed int8 matmul row produced a non-finite value: {got}"
        );
        let diff = (got - want).abs();
        max_error = max_error.max(diff);
        let per_row_relative_error = diff / want.abs().max(f32::MIN_POSITIVE);
        max_per_row_relative_error = max_per_row_relative_error.max(per_row_relative_error);
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative_max_error = max_error / max_magnitude;
    let min_abs_reference = expected
        .iter()
        .map(|value| value.abs())
        .fold(f32::INFINITY, f32::min);
    eprintln!(
        "matmul_q4k_q8k_f32 @ rows={rows} k={k}: max_error={max_error} \
         relative_max_error={relative_max_error} \
         max_per_row_relative_error={max_per_row_relative_error} (probe's metric, informational only) \
         min_abs_reference={min_abs_reference}"
    );

    // the actual correctness gate: same convention as the rows=5 test
    // above, immune to any single row's reference landing near zero.
    assert!(
        relative_max_error < 0.01,
        "relative_max_error={relative_max_error} (max_error={max_error} over magnitude {max_magnitude}) \
         exceeds loose sanity bound at real forward out_dim"
    );
}

/// [`dot_q4k_q8k_block_avx2`]'s equivalence proof, the x86_64 sibling of
/// [`matmul_q4k_q8k_f32_agrees_bit_exact_with_the_portable_arm`] below --
/// same reasoning: every intermediate value both kernels compute is
/// integer (`i32` partial sums via `_mm256_maddubs_epi16` +
/// `_mm256_madd_epi16`, `i32` mins correction) until the final `f32`
/// scale multiply, so [`dot_q4k_q8k_block_avx2`] and
/// [`dot_q4k_q8k_block_scalar`] must agree bit-for-bit on the same
/// input.
///
/// This crate's dev boxes are all aarch64-darwin, so this test is
/// COMPILED (verified via `cargo check -p proxima-tensor --target
/// x86_64-unknown-linux-gnu --tests`) but never EXECUTED on this
/// machine -- `#[cfg(target_arch = "x86_64")]` means it does not even
/// exist in the aarch64 test binary this crate's own `cargo nextest
/// run` builds. It runs for real on any x86_64 CI runner (this
/// workspace's `proxima-tensor-gate.sh` targets
/// `x86_64-unknown-linux-gnu`) or on `versailles`; the `is_x86_feature_detected!`
/// guard skips it rather than failing on a pre-2013 x86_64 host with no
/// AVX2 at all.
#[cfg(all(test, target_arch = "x86_64", feature = "q4k-int8-dot"))]
#[test]
fn dot_q4k_q8k_block_avx2_agrees_bit_exact_with_scalar_when_avx2_is_present() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let blocks_per_row = 3;
    let k = QK_K * blocks_per_row;
    let activation: Vec<f32> = random_vec(29, k)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(31, k)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();

    let mut weight_row = vec![0u8; blocks_per_row * BLOCK_BYTES];
    quantize(&weight_f32, &mut weight_row)
        .expect("row length is a whole multiple of QK_K by construction");

    let mut activation_q8k = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation");

    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q4K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        let scalar = dot_q4k_q8k_block_scalar(weight_block, q8k_block);
        // SAFETY: `is_x86_feature_detected!("avx2")` confirmed above.
        let avx2 = unsafe { dot_q4k_q8k_block_avx2(weight_block, q8k_block) };
        assert_eq!(
            scalar.to_bits(),
            avx2.to_bits(),
            "AVX2 block dot diverged from the scalar reference -- not merely an acceleration"
        );
    }
}

/// The whole point of [`dot_q4k_q8k_block_neon_dotprod`]: it is an
/// ACCELERATION of [`dot_q4k_q8k_block_scalar`]'s mechanism, not a
/// different one. Every intermediate value both paths compute is
/// integer (`i32` partial sums, `i32` mins correction) until the very
/// last step, and integer addition has no rounding -- so
/// [`matmul_q4k_q8k_f32`] (whichever arm `q4k_dotprod` selects) and
/// [`matmul_q4k_q8k_portable_f32`] (always the scalar arm) must produce
/// BIT-EXACT output on the same input, not merely close. A tolerance
/// here would hide a real divergence between the two implementations.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn matmul_q4k_q8k_f32_agrees_bit_exact_with_the_portable_arm() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let rows = 4;
    let blocks_per_row = 5;
    let k = QK_K * blocks_per_row;

    let activation: Vec<f32> = random_vec(13, k)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(17, rows * k)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of QK_K by construction");
    }

    let dispatched = matmul_q4k_q8k_f32(&weight_blocks, rows, &activation)
        .expect("well-formed dispatched matmul");
    let portable = matmul_q4k_q8k_portable_f32(&weight_blocks, rows, &activation)
        .expect("well-formed portable matmul");

    assert_eq!(
        dispatched, portable,
        "dispatched and portable arms diverged -- not merely an acceleration"
    );
}

/// [`quantize_row_q8k`]'s all-zero fast path
/// (`quantize_row_q8_K_ref`'s `if (!amax) { y[i].d = 0; memset(...); }`
/// arm, `ggml-quants.c:2483-2488`): a zero super-block must round-trip
/// to an exactly zero packed block, not merely a near-zero one.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn quantize_row_q8k_zero_vector_is_bit_exact_zero() {
    let activation = vec![0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut packed = vec![0xFFu8; Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation, &mut packed).expect("one well-formed super-block");
    assert!(
        packed.iter().all(|&byte| byte == 0),
        "zero activation must pack to an all-zero Q8_K block"
    );
}

/// [`quantize_row_q8k_dispatch`]'s cohort split against
/// [`quantize_row_q8k`]'s serial reference, at a block count
/// (`4 * MIN_QUANTIZE_BLOCKS_FOR_DISPATCH`) chosen to clear
/// [`MIN_QUANTIZE_BLOCKS_FOR_DISPATCH`] with headroom so the dispatch
/// path (not the serial fallback) actually runs -- every `Q8_K`
/// super-block quantizes independently (`quantize_row_q8k`'s own doc),
/// so this must be bit-for-bit, not merely close.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn quantize_row_q8k_dispatch_is_bit_identical_to_the_serial_reference() {
    let block_count = MIN_QUANTIZE_BLOCKS_FOR_DISPATCH * 4;
    let activation: Vec<f32> = (0..block_count * Q4K_BLOCK_ELEMENTS)
        .map(|index| ((index % 251) as f32 - 125.0) * 0.037)
        .collect();

    let mut serial = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation, &mut serial).expect("well-formed activation quantizes serially");

    let cohort = MatmulCohort::from_config(
        MatmulCohort::builder()
            .members(NonZeroUsize::new(4).expect("4 is nonzero"))
            .build(),
    )
    .expect("test cohort with 4 members spawns");
    let session = cohort
        .enter()
        .expect("no other session open on a fresh cohort");
    let mut dispatched = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k_dispatch(&activation, &mut dispatched, Some(&session))
        .expect("well-formed activation quantizes through the cohort");
    drop(session);

    assert_eq!(
        dispatched, serial,
        "cohort-dispatched Q8_K packing must be bit-identical to the serial reference"
    );
}

/// [`transpose_wide_to_output`]'s cohort split against its own serial
/// fallback, at `rows * leading_total` (`rows = 8000`,
/// `leading_total = 9`, 72,000 elements) chosen to clear
/// [`MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH`] with headroom -- pure data
/// movement, so this must be bit-for-bit.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn transpose_wide_to_output_dispatch_is_bit_identical_to_the_serial_reference() {
    let rows = 8000usize;
    let leading_total = 9usize;
    assert!(
        rows * leading_total >= MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH,
        "this shape must clear the threshold or this test proves nothing about the dispatch path"
    );
    let wide: Vec<f32> = (0..rows * leading_total)
        .map(|index| index as f32 * 0.5 - 17.0)
        .collect();

    let mut serial = vec![0.0f32; rows * leading_total];
    transpose_wide_to_output(&wide, rows, leading_total, None, &mut serial)
        .expect("serial transpose never fails");

    let cohort = MatmulCohort::from_config(
        MatmulCohort::builder()
            .members(NonZeroUsize::new(4).expect("4 is nonzero"))
            .build(),
    )
    .expect("test cohort with 4 members spawns");
    let session = cohort
        .enter()
        .expect("no other session open on a fresh cohort");
    let mut dispatched = vec![0.0f32; rows * leading_total];
    transpose_wide_to_output(&wide, rows, leading_total, Some(&session), &mut dispatched)
        .expect("cohort-dispatched transpose never fails");
    drop(session);

    assert_eq!(
        dispatched, serial,
        "cohort-dispatched transpose must be bit-identical to the serial reference"
    );
}

/// [`dot_q4k_q8k`]'s shape-mismatch guard, mirroring
/// [`matmul_q4k_f32_rejects_an_activation_length_that_does_not_match_the_weight_rows_element_count`]
/// for the packed-int8 sibling: a `Q8_K` activation buffer sized for
/// the wrong block count is rejected, never silently truncated.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn dot_q4k_q8k_rejects_a_q8k_activation_length_mismatch() {
    let weight_block = vec![0u8; Q4K_BLOCK_BYTES];
    let wrong_length_q8k = vec![0u8; Q8K_BLOCK_BYTES - 1];
    let error = dot_q4k_q8k(&weight_block, &wrong_length_q8k).unwrap_err();
    assert!(
        matches!(error, TensorError::QuantizedShapeMismatch { .. }),
        "got {error:?}"
    );
}

/// Same guard, exercised on the always-portable entry point directly
/// (bypassing `q4k_dotprod` dispatch entirely).
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn dot_q4k_q8k_portable_rejects_a_weight_row_length_not_a_block_multiple() {
    let weight_row = vec![0u8; Q4K_BLOCK_BYTES - 1];
    let q8k = vec![0u8; Q8K_BLOCK_BYTES];
    let error = dot_q4k_q8k_portable(&weight_row, &q8k).unwrap_err();
    assert!(
        matches!(error, TensorError::QuantizedShapeMismatch { .. }),
        "got {error:?}"
    );
}

/// [`QuantDot::Fused`] vs [`QuantDot::Unfused`] on identical random
/// `Q4_K` blocks: both consume the exact same already-quantized bytes
/// (weight and activation), so the only remaining disagreement is
/// floating-point accumulation order (an integer-factored int8 fold vs
/// a linear `f32` sum) -- a MUCH tighter bound than
/// `matmul_q4k_q8k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`,
/// which compares against the ORIGINAL unquantized activation and so
/// also absorbs the activation's own Q8_K quantization error. Also
/// checks `Fused` against [`dot_q4k_q8k`] directly (bit-exact): the
/// pipe wrapper introduces no deviation from the kernel it delegates to.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn quant_dot_fused_and_unfused_agree_for_q4k_within_int8_quantization_tolerance() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let blocks_per_row = 3;
    let k = QK_K * blocks_per_row;
    let weight_f32 = random_vec(21, k);
    let activation_f32: Vec<f32> = random_vec(22, k)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_bytes = vec![0u8; blocks_per_row * BLOCK_BYTES];
    quantize(&weight_f32, &mut weight_bytes).expect("k is a whole number of q4_k super-blocks");
    let mut activation_q8k = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation_f32, &mut activation_q8k)
        .expect("k is a whole number of q8_k super-blocks");

    let fused = block_on(QuantDot::Fused(QuantizedBlock::Q4K(&weight_bytes)).call(&activation_q8k))
        .expect("fused int8 dot evaluates");
    let unfused =
        block_on(QuantDot::Unfused(QuantizedBlock::Q4K(&weight_bytes)).call(&activation_q8k))
            .expect("unfused dequantize-then-fold evaluates");
    let kernel_ground_truth = dot_q4k_q8k(&weight_bytes, &activation_q8k)
        .expect("the underlying kernel evaluates directly");

    assert_eq!(
        fused, kernel_ground_truth,
        "QuantDot::Fused must be a bit-exact wrapper over dot_q4k_q8k"
    );
    let relative_error = (fused - unfused).abs() / fused.abs().max(1.0);
    eprintln!("q4_k QuantDot fused={fused} unfused={unfused} relative_error={relative_error}");
    assert!(
        relative_error < 1e-3,
        "relative_error={relative_error} exceeds parity tolerance"
    );
}

#[cfg(feature = "q5k-int8-dot")]
#[test]
fn quant_dot_fused_and_unfused_agree_for_q5k_within_int8_quantization_tolerance() {
    use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, quantize};

    let blocks_per_row = 3;
    let k = QK_K * blocks_per_row;
    let weight_f32 = random_vec(23, k);
    let activation_f32: Vec<f32> = random_vec(24, k)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_bytes = vec![0u8; blocks_per_row * BLOCK_BYTES];
    quantize(&weight_f32, &mut weight_bytes).expect("k is a whole number of q5_k super-blocks");
    let mut activation_q8k = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation_f32, &mut activation_q8k)
        .expect("k is a whole number of q8_k super-blocks");

    let fused = block_on(QuantDot::Fused(QuantizedBlock::Q5K(&weight_bytes)).call(&activation_q8k))
        .expect("fused int8 dot evaluates");
    let unfused =
        block_on(QuantDot::Unfused(QuantizedBlock::Q5K(&weight_bytes)).call(&activation_q8k))
            .expect("unfused dequantize-then-fold evaluates");
    let kernel_ground_truth = dot_q5k_q8k(&weight_bytes, &activation_q8k)
        .expect("the underlying kernel evaluates directly");

    assert_eq!(
        fused, kernel_ground_truth,
        "QuantDot::Fused must be a bit-exact wrapper over dot_q5k_q8k"
    );
    let relative_error = (fused - unfused).abs() / fused.abs().max(1.0);
    eprintln!("q5_k QuantDot fused={fused} unfused={unfused} relative_error={relative_error}");
    assert!(
        relative_error < 1e-3,
        "relative_error={relative_error} exceeds parity tolerance"
    );
}

#[cfg(feature = "q6k-int8-dot")]
#[test]
fn quant_dot_fused_and_unfused_agree_for_q6k_within_int8_quantization_tolerance() {
    use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, quantize};

    let blocks_per_row = 3;
    let k = QK_K * blocks_per_row;
    let weight_f32 = random_vec(25, k);
    let activation_f32: Vec<f32> = random_vec(26, k)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_bytes = vec![0u8; blocks_per_row * BLOCK_BYTES];
    quantize(&weight_f32, &mut weight_bytes).expect("k is a whole number of q6_k super-blocks");
    let mut activation_q8k = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation_f32, &mut activation_q8k)
        .expect("k is a whole number of q8_k super-blocks");

    let fused = block_on(QuantDot::Fused(QuantizedBlock::Q6K(&weight_bytes)).call(&activation_q8k))
        .expect("fused int8 dot evaluates");
    let unfused =
        block_on(QuantDot::Unfused(QuantizedBlock::Q6K(&weight_bytes)).call(&activation_q8k))
            .expect("unfused dequantize-then-fold evaluates");
    let kernel_ground_truth = dot_q6k_q8k(&weight_bytes, &activation_q8k)
        .expect("the underlying kernel evaluates directly");

    assert_eq!(
        fused, kernel_ground_truth,
        "QuantDot::Fused must be a bit-exact wrapper over dot_q6k_q8k"
    );
    let relative_error = (fused - unfused).abs() / fused.abs().max(1.0);
    eprintln!("q6_k QuantDot fused={fused} unfused={unfused} relative_error={relative_error}");
    assert!(
        relative_error < 1e-3,
        "relative_error={relative_error} exceeds parity tolerance"
    );
}

/// Both arms of [`QuantDot`] take the identical `In` shape
/// (`dot_q4k_q8k`'s own malformed-shape guards), so both must reject the
/// identical malformed shapes rather than only one of them.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn quant_dot_rejects_a_malformed_weight_row_on_both_fused_and_unfused() {
    let weight_row = vec![0u8; Q4K_BLOCK_BYTES - 1];
    let activation_q8k = vec![0u8; Q8K_BLOCK_BYTES];

    let fused_error =
        block_on(QuantDot::Fused(QuantizedBlock::Q4K(&weight_row)).call(&activation_q8k))
            .unwrap_err();
    assert!(
        matches!(fused_error, TensorError::QuantizedShapeMismatch { .. }),
        "got {fused_error:?}"
    );

    let unfused_error =
        block_on(QuantDot::Unfused(QuantizedBlock::Q4K(&weight_row)).call(&activation_q8k))
            .unwrap_err();
    assert!(
        matches!(unfused_error, TensorError::QuantizedShapeMismatch { .. }),
        "got {unfused_error:?}"
    );
}

/// A codec `QuantDot` does not support (`Q8_0` has no `Q8_K`-activation
/// int8-dot path -- [`dot_fn_for`]'s own doc names this) is an honest
/// `NotLowerable`, never a silent misroute to a different codec's
/// kernel.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn quant_dot_rejects_a_codec_with_no_int8_dot_kernel() {
    let weight_row = vec![0u8; Q8_0_BLOCK_BYTES];
    let activation_q8k = vec![0u8; Q8K_BLOCK_BYTES];

    let fused_error =
        block_on(QuantDot::Fused(QuantizedBlock::Q8_0(&weight_row)).call(&activation_q8k))
            .unwrap_err();
    assert!(
        matches!(fused_error, TensorError::NotLowerable { .. }),
        "got {fused_error:?}"
    );

    let unfused_error =
        block_on(QuantDot::Unfused(QuantizedBlock::Q8_0(&weight_row)).call(&activation_q8k))
            .unwrap_err();
    assert!(
        matches!(unfused_error, TensorError::NotLowerable { .. }),
        "got {unfused_error:?}"
    );
}

/// [`mins_correction_neon`] (the explicit NEON mins-correction path
/// this landing introduced) against the scalar route through
/// [`proxima_gguf::quant::q4_k::get_scale_min_k4`] it replaced -- bit
/// exact, not approximate (guiding-principles: the mins correction is
/// an integer computation widened to `i32`, so exact equality is
/// achievable and is the bar). Uses the FIRST `Q4_K` super-block of a
/// real weight tensor read straight out of the real openchat-3.5-1210
/// `Q4_K_S` GGUF file (principle 9: real-world data), against a real
/// (non-zero) quantized activation super-block, so the scale/min bit
/// patterns and `bsums` are whatever ggml's own quantizer actually
/// produced -- not a synthetic stand-in.
#[cfg(all(q4k_dotprod, feature = "q4k-int8-dot"))]
#[test]
fn mins_correction_neon_agrees_with_get_scale_min_k4_scalar_route_on_real_gguf_bytes() {
    let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, _in_dim, _out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.attn_q.weight",
        proxima_gguf::types::GgmlType::Q4_K,
    ) else {
        eprintln!("blk.0.attn_q.weight is not Q4_K in this file; test skipped, not faked");
        return;
    };

    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_bytes[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

    let activation: Vec<f32> = random_vec(29, Q4K_BLOCK_ELEMENTS)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();
    let mut activation_q8k = vec![0u8; Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation super-block");
    let bsums = &activation_q8k[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let mut expected_mins_correction = 0i32;
    for sub_block in 0..Q4K_SUB_BLOCKS {
        let (expected_scale, min_code) =
            proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);
        // SAFETY: `q4k_dotprod` cfg guarantees `FEAT_DotProd`; `bsums` is
        // exactly `Q8K_BSUMS_COUNT * 2` bytes from the fixed-size buffer above.
        let (scale_lo, scale_hi, _) = unsafe { mins_correction_neon(&scales, bsums) };
        let actual_scale = if sub_block < 4 {
            scale_byte(scale_lo, sub_block as u32)
        } else {
            scale_byte(scale_hi, (sub_block - 4) as u32)
        };
        assert_eq!(
            actual_scale,
            i32::from(expected_scale),
            "scale word diverged from the scalar get_scale_min_k4 route at sub_block {sub_block}"
        );
        let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
        let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
        expected_mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);
    }

    // SAFETY: `q4k_dotprod` cfg guarantees `FEAT_DotProd`; `bsums` is
    // exactly `Q8K_BSUMS_COUNT * 2` bytes from the fixed-size buffer above.
    let (_, _, actual_mins_correction) = unsafe { mins_correction_neon(&scales, bsums) };
    assert_eq!(
        actual_mins_correction, expected_mins_correction,
        "NEON mins_correction diverged from the scalar get_scale_min_k4 route -- nonzero delta, faster wrong kernel"
    );
}

/// [`dot_q5k_q8k_block_neon_dotprod`]'s new [`mins_correction_neon`]
/// route (this landing) against the 16-scalar-call
/// `get_scale_min_k4` route it replaced (8 for the mins correction,
/// 8 more for `scale_lo`/`scale_hi` inside the SIMD dot loop) -- bit
/// exact, not approximate, same bar as
/// [`mins_correction_neon_agrees_with_get_scale_min_k4_scalar_route_on_real_gguf_bytes`].
/// Uses the first `Q5_K` super-block of `blk.0.attn_v.weight` read
/// straight out of the real openchat-3.5-1210 GGUF file (principle 9:
/// real-world data), against a real quantized activation super-block.
#[cfg(all(q4k_dotprod, feature = "q5k-int8-dot"))]
#[test]
fn q5k_mins_correction_neon_agrees_with_get_scale_min_k4_scalar_route_on_real_gguf_bytes() {
    let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, _in_dim, _out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.attn_v.weight",
        proxima_gguf::types::GgmlType::Q5_K,
    ) else {
        eprintln!("blk.0.attn_v.weight is not Q5_K in this file; test skipped, not faked");
        return;
    };

    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_bytes[Q5K_SCALES_OFFSET..Q5K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

    let activation: Vec<f32> = random_vec(37, Q4K_BLOCK_ELEMENTS)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();
    let mut activation_q8k = vec![0u8; Q8K_BLOCK_BYTES];
    quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation super-block");
    let bsums = &activation_q8k[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let mut expected_mins_correction = 0i32;
    for sub_block in 0..Q4K_SUB_BLOCKS {
        let (expected_scale, min_code) =
            proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);
        // SAFETY: `q4k_dotprod` cfg guarantees `FEAT_DotProd`; `bsums` is
        // exactly `Q8K_BSUMS_COUNT * 2` bytes from the fixed-size buffer above.
        let (scale_lo, scale_hi, _) = unsafe { mins_correction_neon(&scales, bsums) };
        let actual_scale = if sub_block < 4 {
            scale_byte(scale_lo, sub_block as u32)
        } else {
            scale_byte(scale_hi, (sub_block - 4) as u32)
        };
        assert_eq!(
            actual_scale,
            i32::from(expected_scale),
            "scale word diverged from the scalar get_scale_min_k4 route at sub_block {sub_block} on Q5_K bytes"
        );
        let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
        let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
        expected_mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);
    }

    // SAFETY: `q4k_dotprod` cfg guarantees `FEAT_DotProd`; `bsums` is
    // exactly `Q8K_BSUMS_COUNT * 2` bytes from the fixed-size buffer above.
    let (_, _, actual_mins_correction) = unsafe { mins_correction_neon(&scales, bsums) };
    assert_eq!(
        actual_mins_correction, expected_mins_correction,
        "Q5_K NEON mins_correction diverged from the scalar get_scale_min_k4 route -- nonzero delta, faster wrong kernel"
    );
}

/// Same shape as [`matmul_program`] (`[rows, k] x [k, 1] -> [rows, 1]`,
/// `n = 1` — batch-1, [`matmul_q4k_f32`]'s own documented target shape),
/// weight declared `UInt8` instead of `Float32`: the program
/// [`evaluate_quantized`] runs, standing in for a `Q4_K`-packed weight
/// matrix.
fn quantized_matmul_program(rows: u32, k: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = block(
        &mut program,
        DType::UInt8,
        &[Extent::Static(rows), Extent::Static(k)],
    );
    let activation = f32_block(&mut program, &[Extent::Static(k), Extent::Static(1)]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("quantized_matmul".into()),
        }),
    );
    (program, sum)
}

/// The capability this whole module change exists for: a real `Op`
/// program, with a `Q4_K`-packed weight operand, run end to end through
/// [`evaluate_quantized`] — not `matmul_q4k_f32` called directly, the
/// way every other test above exercises it — and checked against the
/// exact same shape run through plain [`evaluate`] with the weight
/// dequantized to `f32` first. Proves the dispatch chain
/// `evaluate_quantized` -> `run_node_into` -> `run_reduce` ->
/// [`quantized_operand`] -> `run_reduce_quantized` -> `matmul_q4k_q8k_f32`
/// (`q4k-int8-dot` is default-on; `matmul_q4k_f32` is the fallback when
/// it is off) is reachable from the program-level entry point, not
/// merely callable in isolation.
#[test]
fn evaluate_quantized_matmul_matches_dequantize_then_f32_evaluate() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows: u32 = 5;
    let blocks_per_row = 3;
    let k = QK_K as u32 * blocks_per_row as u32;

    let activation: Vec<f32> = random_vec(13, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(17, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of QK_K by construction");
    }

    let (quantized_program, quantized_sum) = quantized_matmul_program(rows, k);
    let quantized_blocks = [
        QuantizedBlock::Q4K(&weight_blocks),
        QuantizedBlock::Float32(&activation),
    ];
    let quantized_result =
        evaluate_quantized(&quantized_program, &[], &quantized_blocks, &[quantized_sum])
            .expect("quantized matmul evaluates end to end");

    let mut dequantized_weight = vec![0.0f32; rows as usize * k as usize];
    for (row_blocks, row_f32) in weight_blocks
        .chunks_exact(blocks_per_row * BLOCK_BYTES)
        .zip(dequantized_weight.chunks_exact_mut(k as usize))
    {
        dequantize(row_blocks, row_f32).expect("row_blocks is a whole number of q4_k super-blocks");
    }

    let (f32_program, f32_sum) = matmul_program(rows, k, 1, false);
    let f32_blocks: [&[f32]; 2] = [&dequantized_weight, &activation];
    let f32_result = evaluate(&f32_program, &[], &f32_blocks, &[f32_sum])
        .expect("dequantized f32 matmul evaluates");

    let actual = quantized_result.root();
    let expected = f32_result.root();
    assert_eq!(actual.len(), rows as usize);
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "evaluate_quantized produced a non-finite value: {got}"
        );
        let diff = (got - want).abs();
        max_diff = max_diff.max(diff);
        sum_sq_diff += f64::from(diff) * f64::from(diff);
    }
    let rms_diff = (sum_sq_diff / rows as f64).sqrt();
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative_max_diff = max_diff / max_magnitude;
    eprintln!(
        "evaluate_quantized vs dequantize-then-evaluate: max_diff={max_diff} rms_diff={rms_diff} \
         max_magnitude={max_magnitude} relative_max_diff={relative_max_diff}"
    );

    // `run_reduce_quantized` routes through `matmul_q4k_q8k_f32` by
    // default (`q4k-int8-dot` default-on) — same second lossy step
    // (Q8_K activation quantization) as
    // `matmul_q4k_q8k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`,
    // so this bound is RELATIVE to the signal's own magnitude the same
    // way that test's is, not the absolute float-noise-floor bound that
    // was right when this call bottomed out in `matmul_q4k_f32` alone.
    assert!(
        relative_max_diff < 0.01,
        "relative_max_diff={relative_max_diff} (max_diff={max_diff} over magnitude {max_magnitude}) \
         exceeds loose sanity bound"
    );
}

/// Targeted RED-first probe for the `wo`/attention-output-projection
/// shape from the real Qwen3.6 35B-A3B checkpoint (layer 3,
/// `qwen35moe_layer3_o_proj_contraction_candidates`): a genuine
/// elementwise step (`activation + 1.0`, the same non-eliminable-under-
/// `bit_exact` shape as the model's own sigmoid gate's
/// `1 + exp(-gate)`) sits BETWEEN the real activation and the
/// `Multiply(weight, activation)` a quantized reduce fuses --
/// `quantized_matmul_program` above never has an operand between
/// activation and product, so it cannot exercise this. The hypothesis
/// this was written to test -- that `run_reduce_quantized`'s
/// `resolved.operands().find(|node| *node != weight_node)` reads the
/// FIRST non-weight flat operand off the fused body and silently picks
/// the pre-add `raw` operand instead of the true `real_activation` --
/// is REFUTED by this test's own data: `fused` and `materialized` come
/// back bit-identical and both land within quantization noise of the
/// `+1.0`-inclusive f64 reference, so this single-extra-step shape is
/// NOT where the real checkpoint's `wo` discrepancy comes from. Kept as
/// a permanent regression guard for the shape it does cover (one
/// elementwise step fused ahead of a quantized `Multiply`-then-`Reduce`
/// correctly threads the whole composed body, not just its first leaf).
#[test]
fn evaluate_quantized_threads_a_fused_elementwise_step_ahead_of_the_matmul_multiply() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let rows: u32 = 3;
    let k = QK_K as u32;

    let raw_activation: Vec<f32> = random_vec(29, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(31, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let block_bytes = BLOCK_BYTES;
    let mut weight_blocks = vec![0u8; rows as usize * block_bytes];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(block_bytes))
    {
        quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
    }

    let mut program = Vec::new();
    let weight = block(
        &mut program,
        DType::UInt8,
        &[Extent::Static(rows), Extent::Static(k)],
    );
    let raw = f32_block(&mut program, &[Extent::Static(k), Extent::Static(1)]);
    let one = crate::spec::scalar_constant(&mut program, 1.0);
    let real_activation = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![
                (raw, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (one, IndexMap::Affine(map::projection(2, &[]))),
            ],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (
                    real_activation,
                    IndexMap::Affine(map::projection(3, &[2, 1]))
                ),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("quantized_matmul_with_extra_add".into()),
        }),
    );

    let blocks = [
        QuantizedBlock::Q4K(&weight_blocks),
        QuantizedBlock::Float32(&raw_activation),
    ];

    // fused: `bind` absorbs `real_activation`'s `+1.0` into the
    // reduce's own body since it has no other consumer -- the shape
    // `run_reduce_quantized` cannot see through.
    let fused = evaluate_quantized(&program, &[], &blocks, &[sum])
        .expect("fused quantized matmul evaluates")
        .root()
        .to_vec();

    // materialized: requesting `real_activation` too keeps it `still_live`,
    // so `bind` cannot fuse it into the reduce -- the reduce now reads a
    // real, already-added buffer, the same path a plain f32 matmul takes.
    let materialized_result = evaluate_quantized(&program, &[], &blocks, &[real_activation, sum])
        .expect("materialized quantized matmul evaluates");
    let materialized = materialized_result
        .get(sum)
        .expect("the reduce output was requested")
        .0
        .to_vec();

    let expected: Vec<f32> = weight_f32
        .chunks_exact(k as usize)
        .map(|row| {
            row.iter()
                .zip(raw_activation.iter())
                .map(|(&weight_value, &activation_value)| {
                    f64::from(weight_value) * f64::from(activation_value + 1.0)
                })
                .sum::<f64>() as f32
        })
        .collect();

    eprintln!("fused={fused:?} materialized={materialized:?} expected={expected:?}");

    for (&got, &want) in materialized.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() / want.abs() < 0.02,
            "materialized path must match the +1.0-inclusive reference: got={got} want={want}"
        );
    }

    assert_eq!(
        fused, materialized,
        "fusing the +1.0 step into the reduce must not change the result the reduce reports \
         versus forcing that step to materialize first"
    );
    for (&got, &want) in fused.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() / want.abs() < 0.02,
            "the fused path must also match the +1.0-inclusive reference: got={got} want={want}"
        );
    }
}

/// A deeper chain than the sibling test above: TWO distinct leaf
/// inputs (`attended`, `gate_raw`) feed a composed sigmoid gate
/// (`Reciprocal(Add(Exponential(Negate(gate_raw)), 1))`) that is then
/// multiplied against `attended` itself before the quantized
/// `Multiply(weight, gated)` a reduce fuses -- the real checkpoint's
/// `sigmoid_attn_gate` / `gated_attended` shape (`spec.rs:4390-4397`),
/// with a flat (single-letter) contraction axis rather than `wo`'s own
/// multi-axis packed row. RED-first probe: this shape alone, data
/// shows, does NOT reproduce the real checkpoint's discrepancy --
/// `fused` and `materialized` come back bit-identical and both within
/// quantization noise of the sigmoid-gated f64 reference, so a
/// two-distinct-leaf composed gate over a FLAT contraction axis is not
/// where `wo`'s own error comes from. Kept as a permanent regression
/// guard for the shape it does cover; the multi-axis packed-contraction
/// sibling below carries the shape that actually reproduces it.
#[test]
fn evaluate_quantized_applies_the_full_composed_gate_not_just_its_first_leaf() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let rows: u32 = 3;
    let k = QK_K as u32;

    let attended: Vec<f32> = random_vec(37, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let gate_raw: Vec<f32> = random_vec(41, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(43, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let block_bytes = BLOCK_BYTES;
    let mut weight_blocks = vec![0u8; rows as usize * block_bytes];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(block_bytes))
    {
        quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
    }

    let mut program = Vec::new();
    let weight = block(
        &mut program,
        DType::UInt8,
        &[Extent::Static(rows), Extent::Static(k)],
    );
    let attended_node = f32_block(&mut program, &[Extent::Static(k), Extent::Static(1)]);
    let gate_node = f32_block(&mut program, &[Extent::Static(k), Extent::Static(1)]);
    let one = crate::spec::scalar_constant(&mut program, 1.0);
    let negated_gate = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: alloc::vec![(gate_node, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let exp_neg_gate = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Exponential,
            operands: alloc::vec![(negated_gate, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let one_plus_exp_gate = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![
                (exp_neg_gate, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (one, IndexMap::Affine(map::projection(2, &[]))),
            ],
            name: None,
        },
    );
    let sigmoid_gate = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: alloc::vec![(
                one_plus_exp_gate,
                IndexMap::Affine(map::projection(2, &[0, 1]))
            )],
            name: None,
        },
    );
    let gated = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (attended_node, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (sigmoid_gate, IndexMap::Affine(map::projection(2, &[0, 1]))),
            ],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (gated, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("quantized_matmul_with_composed_gate".into()),
        }),
    );

    let blocks = [
        QuantizedBlock::Q4K(&weight_blocks),
        QuantizedBlock::Float32(&attended),
        QuantizedBlock::Float32(&gate_raw),
    ];

    let fused = evaluate_quantized(&program, &[], &blocks, &[sum])
        .expect("fused quantized matmul evaluates")
        .root()
        .to_vec();

    let materialized_result = evaluate_quantized(&program, &[], &blocks, &[gated, sum])
        .expect("materialized quantized matmul evaluates");
    let materialized = materialized_result
        .get(sum)
        .expect("the reduce output was requested")
        .0
        .to_vec();

    let expected: Vec<f32> = weight_f32
        .chunks_exact(k as usize)
        .map(|row| {
            row.iter()
                .zip(attended.iter())
                .zip(gate_raw.iter())
                .map(|((&weight_value, &attended_value), &gate_value)| {
                    let sigmoid = 1.0_f64 / (1.0_f64 + f64::from(-gate_value).exp());
                    f64::from(weight_value) * (f64::from(attended_value) * sigmoid)
                })
                .sum::<f64>() as f32
        })
        .collect();

    eprintln!("fused={fused:?} materialized={materialized:?} expected={expected:?}");

    for (&got, &want) in materialized.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() / want.abs() < 0.02,
            "materialized path must match the sigmoid-gated reference: got={got} want={want}"
        );
    }
    assert_eq!(
        fused, materialized,
        "fusing a two-distinct-leaf composed gate over a flat contraction axis into the \
         reduce must not change the result versus forcing the gate to materialize first"
    );
    for (&got, &want) in fused.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() / want.abs() < 0.02,
            "the fused path must also match the sigmoid-gated reference: got={got} want={want}"
        );
    }
}

/// The shape that actually reproduces the real checkpoint's `wo`
/// discrepancy: the same two-distinct-leaf composed sigmoid gate as
/// the sibling test above, now multiplied against a weight whose
/// packed row is a THREE-letter affine contraction (`u`, `g`, `d`) --
/// `wo_flat`'s own map string, `spec.rs:9024-9028`, mirrored here at
/// `kv_heads=2, group=2, head_dim=64` (`u*g*d = 256 = QK_K`, one clean
/// `Q4_K` super-block per output row) and `embedding=3`. Neither
/// ingredient alone reproduces it: the sibling test above (composed
/// gate, flat contraction) is bit-exact; `correct_packed_matmul_layouts_derives_ggml_native_strides_for_a_multi_axis_contraction_group`
/// in `bind.rs` (multi-axis contraction, no composed gate) proved the
/// stride algebra alone is right. Both together are what
/// `run_reduce_quantized`'s `activation_row = &activation[...]`
/// (`cpu.rs:9271`) cannot express: it reads ONE flat leaf's buffer
/// verbatim, under a layout computed for the FUSED reduce's own
/// iteration space, and never re-applies `resolved`'s composed steps
/// (the sigmoid gate) at all.
#[test]
fn evaluate_quantized_applies_the_composed_gate_over_a_multi_axis_packed_contraction() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, quantize};

    const KV_HEADS: u64 = 2;
    const GROUP: u64 = 2;
    const HEAD_DIM: u64 = 64;
    const EMBED: u64 = 3;
    const IN_DIM: u64 = KV_HEADS * GROUP * HEAD_DIM;
    assert_eq!(
        IN_DIM as usize,
        proxima_gguf::quant::q4_k::QK_K,
        "one clean super-block per output row"
    );

    let attended: Vec<f32> = random_vec(47, IN_DIM as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let gate_raw: Vec<f32> = random_vec(53, IN_DIM as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(59, EMBED as usize * IN_DIM as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let in_dim = IN_DIM as usize;
    let blocks_per_row = in_dim / proxima_gguf::quant::q4_k::QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let mut weight_blocks = vec![0u8; EMBED as usize * row_bytes];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(in_dim)
        .zip(weight_blocks.chunks_exact_mut(row_bytes))
    {
        quantize(row_f32, row_blocks).expect("row length is a whole number of QK_K super-blocks");
    }

    let mut program = Vec::new();
    // physical weight shape [EMBED, IN_DIM] (`quantize`'s own natural
    // row-major layout); iteration axes here are (u=0, g=1, d=2, e=3).
    let weight = block(
        &mut program,
        DType::UInt8,
        &[Extent::Static(EMBED as u32), Extent::Static(IN_DIM as u32)],
    );
    let attended_node = f32_block(
        &mut program,
        &[
            Extent::Static(KV_HEADS as u32),
            Extent::Static(GROUP as u32),
            Extent::Static(HEAD_DIM as u32),
        ],
    );
    let gate_node = f32_block(
        &mut program,
        &[
            Extent::Static(KV_HEADS as u32),
            Extent::Static(GROUP as u32),
            Extent::Static(HEAD_DIM as u32),
        ],
    );
    let one = crate::spec::scalar_constant(&mut program, 1.0);
    let negated_gate = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: alloc::vec![(gate_node, IndexMap::Affine(map::projection(3, &[0, 1, 2])))],
            name: None,
        },
    );
    let exp_neg_gate = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Exponential,
            operands: alloc::vec![(
                negated_gate,
                IndexMap::Affine(map::projection(3, &[0, 1, 2]))
            )],
            name: None,
        },
    );
    let one_plus_exp_gate = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![
                (
                    exp_neg_gate,
                    IndexMap::Affine(map::projection(3, &[0, 1, 2]))
                ),
                (one, IndexMap::Affine(map::projection(3, &[]))),
            ],
            name: None,
        },
    );
    let sigmoid_gate = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: alloc::vec![(
                one_plus_exp_gate,
                IndexMap::Affine(map::projection(3, &[0, 1, 2]))
            )],
            name: None,
        },
    );
    let gated = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (
                    attended_node,
                    IndexMap::Affine(map::projection(3, &[0, 1, 2]))
                ),
                (
                    sigmoid_gate,
                    IndexMap::Affine(map::projection(3, &[0, 1, 2]))
                ),
            ],
            name: None,
        },
    );
    // iteration space (u=0, g=1, d=2, e=3): weight's packed row axis is
    // `(group*head_dim)*u + head_dim*g + d`, exactly `wo_flat`'s own
    // map string with the real dims swapped for these tiny ones; its
    // second physical axis is the plain output letter `e`. `gated`
    // reads (u, g, d), broadcasting over `e`.
    let row_terms = [
        AxisTerm::scaled(0, i32::try_from(GROUP * HEAD_DIM).expect("fits i32")),
        AxisTerm::scaled(1, i32::try_from(HEAD_DIM).expect("fits i32")),
        AxisTerm::scaled(2, 1),
    ];
    let weight_map = IndexMap::Affine(map::affine(
        4,
        &[(&[AxisTerm::scaled(3, 1)], 0), (&row_terms, 0)],
    ));
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (weight, weight_map),
                (gated, IndexMap::Affine(map::projection(4, &[0, 1, 2]))),
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
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[3])),
            keep: Keep::Reduce,
            name: Some("quantized_matmul_with_composed_gate_multi_axis_contraction".into()),
        }),
    );

    let blocks = [
        QuantizedBlock::Q4K(&weight_blocks),
        QuantizedBlock::Float32(&attended),
        QuantizedBlock::Float32(&gate_raw),
    ];

    let fused = evaluate_quantized(&program, &[], &blocks, &[sum])
        .expect("fused quantized matmul evaluates")
        .root()
        .to_vec();

    let materialized_result = evaluate_quantized(&program, &[], &blocks, &[gated, sum])
        .expect("materialized quantized matmul evaluates");
    let materialized = materialized_result
        .get(sum)
        .expect("the reduce output was requested")
        .0
        .to_vec();

    let expected: Vec<f32> = weight_f32
        .chunks_exact(in_dim)
        .map(|row| {
            row.iter()
                .zip(attended.iter())
                .zip(gate_raw.iter())
                .map(|((&weight_value, &attended_value), &gate_value)| {
                    let sigmoid = 1.0_f64 / (1.0_f64 + f64::from(-gate_value).exp());
                    f64::from(weight_value) * (f64::from(attended_value) * sigmoid)
                })
                .sum::<f64>() as f32
        })
        .collect();

    eprintln!("fused={fused:?} materialized={materialized:?} expected={expected:?}");

    for (&got, &want) in materialized.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() / want.abs() < 0.02,
            "materialized path must match the sigmoid-gated reference: got={got} want={want}"
        );
    }
    for (&got, &want) in fused.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() / want.abs() < 0.02,
            "the fused path must also match the sigmoid-gated reference over a multi-axis \
             packed contraction: got={got} want={want}"
        );
    }
}

#[test]
fn qwen35_batched_packed_ssm_projection_matches_repeated_rows() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    const POSITIONS: u32 = 2;
    const GROUP: u32 = 2;
    const KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 64;
    const EMBED: u32 = 3;
    const CONTRACTION: u32 = GROUP * KV_HEADS * HEAD_DIM;
    assert_eq!(CONTRACTION as usize, QK_K);

    let activation: Vec<f32> = random_vec(61, (POSITIONS * CONTRACTION) as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(67, (EMBED * CONTRACTION) as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let mut weight_blocks = vec![0_u8; EMBED as usize * BLOCK_BYTES];
    for (row, packed) in weight_f32
        .as_chunks::<{ CONTRACTION as usize }>()
        .0
        .iter()
        .zip(weight_blocks.as_chunks_mut::<BLOCK_BYTES>().0)
    {
        quantize(row, packed).expect("one complete q4_k block per row");
    }

    let build = |positions: u32| {
        let mut program = Vec::new();
        let weight = block(
            &mut program,
            DType::UInt8,
            &[Extent::Static(EMBED), Extent::Static(CONTRACTION)],
        );
        let activation = f32_block(
            &mut program,
            &[
                Extent::Static(positions),
                Extent::Static(GROUP),
                Extent::Static(KV_HEADS),
                Extent::Static(HEAD_DIM),
            ],
        );
        let row_terms = [
            AxisTerm::scaled(3, i32::try_from(KV_HEADS * GROUP).expect("fits")),
            AxisTerm::scaled(2, i32::try_from(GROUP).expect("fits")),
            AxisTerm::scaled(1, 1),
        ];
        let weight_map = IndexMap::Affine(map::affine(
            5,
            &[(&[AxisTerm::scaled(4, 1)], 0), (&row_terms, 0)],
        ));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight, weight_map),
                    (
                        activation,
                        IndexMap::Affine(map::projection(5, &[0, 1, 2, 3]))
                    ),
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
                in_map: IndexMap::Affine(map::projection(5, &[0, 1, 2, 3, 4])),
                out_map: IndexMap::Affine(map::projection(5, &[0, 4])),
                keep: Keep::Reduce,
                name: Some("qwen35_batched_ssm_projection".into()),
            }),
        );
        (program, sum)
    };

    let (batched_program, batched_sum) = build(POSITIONS);
    let batched_blocks = [
        QuantizedBlock::Q4K(weight_blocks.as_slice()),
        QuantizedBlock::Float32(activation.as_slice()),
    ];
    let batched = evaluate_quantized_exact(&batched_program, &[], &batched_blocks, &[batched_sum])
        .expect("batched projection evaluates")
        .root()
        .to_vec();

    let (single_program, single_sum) = build(1);
    let mut repeated = Vec::new();
    for row in activation.as_chunks::<{ CONTRACTION as usize }>().0 {
        let blocks = [
            QuantizedBlock::Q4K(weight_blocks.as_slice()),
            QuantizedBlock::Float32(row),
        ];
        repeated.extend_from_slice(
            evaluate_quantized_exact(&single_program, &[], &blocks, &[single_sum])
                .expect("single projection evaluates")
                .root(),
        );
    }
    assert_eq!(batched, repeated);
}

/// [`evaluate_quantized_exact`] run end to end on the same program as
/// [`evaluate_quantized_matmul_matches_dequantize_then_f32_evaluate`]
/// above, held to a tighter absolute float-noise-floor bound instead of
/// that test's relative-error sanity bound -- `exact_activations` skips
/// the wide-fold `matmul_q4k_q8k_f32_impl` arm entirely (`cfg(feature =
/// "q4k-int8-dot")`'s wide-fold gate in `run_reduce_quantized` now reads
/// `!exact_activations && ...`) and always dots through `matmul_q4k_f32`,
/// so this run shares the SAME dequantized bytes and the SAME `f32`
/// accumulation order as the dequantize-then-evaluate reference, exactly
/// the shape `matmul_q4k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`
/// already holds to a near-zero absolute bound.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn evaluate_quantized_exact_matches_dequantize_then_f32_evaluate_near_exactly() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows: u32 = 5;
    let blocks_per_row = 3;
    let k = QK_K as u32 * blocks_per_row as u32;

    let activation: Vec<f32> = random_vec(19, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(23, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of QK_K by construction");
    }

    let (quantized_program, quantized_sum) = quantized_matmul_program(rows, k);
    let quantized_blocks = [
        QuantizedBlock::Q4K(&weight_blocks),
        QuantizedBlock::Float32(&activation),
    ];
    let exact_result =
        evaluate_quantized_exact(&quantized_program, &[], &quantized_blocks, &[quantized_sum])
            .expect("exact-activation quantized matmul evaluates end to end");

    let mut dequantized_weight = vec![0.0f32; rows as usize * k as usize];
    for (row_blocks, row_f32) in weight_blocks
        .chunks_exact(blocks_per_row * BLOCK_BYTES)
        .zip(dequantized_weight.chunks_exact_mut(k as usize))
    {
        dequantize(row_blocks, row_f32).expect("row_blocks is a whole number of q4_k super-blocks");
    }

    let (f32_program, f32_sum) = matmul_program(rows, k, 1, false);
    let f32_blocks: [&[f32]; 2] = [&dequantized_weight, &activation];
    let f32_result = evaluate(&f32_program, &[], &f32_blocks, &[f32_sum])
        .expect("dequantized f32 matmul evaluates");

    let actual = exact_result.root();
    let expected = f32_result.root();
    assert_eq!(actual.len(), rows as usize);
    assert_eq!(actual.len(), expected.len());

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    eprintln!("evaluate_quantized_exact vs dequantize-then-evaluate: max_diff={max_diff}");
    // Same loose sanity bound (accumulation-order float noise floor, not
    // tuned to the measured number) as
    // `matmul_q4k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`'s
    // own `max_error < 0.05` -- this test runs the identical kernel
    // through the full `evaluate_quantized_exact` program dispatch
    // instead of calling it directly.
    assert!(
        max_diff < 0.05,
        "evaluate_quantized_exact diverged from the dequantize-then-evaluate reference \
         beyond float noise: max_diff={max_diff}"
    );
}

/// `sum_k weight[route[s], o, k] * activation[s, k]` -- `moe_block.toml`'s
/// `expert_w` gather, mirroring [`embedding_matmul_program`]'s `(s, o, k)`
/// iteration shape with the gather moved from the activation side to the
/// weight side. `weight`'s own physical shape is `[n_experts, rows, k]`;
/// `gathered_dim: 0` and `index_map` reading iteration axis 0 (`s`) is the
/// same [`IndexMap::Computed`] wiring [`embedding_lookup_program`] uses,
/// just on the operand [`run_reduce_quantized`] actually dequantizes.
fn gathered_quantized_matmul_program(
    n_experts: u32,
    rows: u32,
    k: u32,
    seq: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = block(
        &mut program,
        DType::UInt8,
        &[
            Extent::Static(n_experts),
            Extent::Static(rows),
            Extent::Static(k),
        ],
    );
    let route = block(&mut program, DType::Int32, &[Extent::Static(seq)]);
    let activation = f32_block(&mut program, &[Extent::Static(seq), Extent::Static(k)]);

    let gather_map = IndexMap::Computed {
        indices: route,
        index_map: map::projection(3, &[0]),
        base: map::IndexPattern {
            iter_rank: 3,
            axes: alloc::vec![
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
    };
    let activation_map = IndexMap::Affine(map::projection(3, &[0, 2]));

    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(weight, gather_map), (activation, activation_map)],
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("gathered_quantized_matmul".into()),
        }),
    );
    (program, sum)
}

/// The decisive test this whole change exists for: gathering expert `k`
/// out of a stacked packed weight must dequantize BIT-IDENTICALLY to
/// expert `k`'s own standalone tensor, exercised through the evaluator
/// (`evaluate_quantized` -> `run_node_into` -> `run_reduce_with_quantized_weights`
/// -> `run_reduce_quantized`'s new gather-resolution branch), not through
/// `restack::gather_expert` called directly. Three tokens, three DISTINCT
/// experts with asymmetric weight magnitudes (1x / 5x / 20x): a token
/// reading the wrong expert's slab produces a value off by that same
/// asymmetric factor, not a subtle rounding difference, so this is a
/// mechanism check, not a tolerance check.
#[test]
fn evaluate_quantized_gathered_moe_weight_matches_the_routed_experts_own_matmul() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let n_experts: u32 = 3;
    let rows: u32 = 4;
    let blocks_per_row = 1;
    let k = QK_K as u32 * blocks_per_row as u32;
    let seq: u32 = 3;
    // token `t` routes to a DIFFERENT expert than its own position (a
    // constant or identity route could not distinguish a working gather
    // from the pre-fix constant-offset read this change closes).
    let route_data = [2.0f32, 0.0, 1.0];
    let expert_scales = [1.0f32, 5.0, 20.0];

    let mut expert_blocks: Vec<Vec<u8>> = Vec::new();
    for (expert, &scale) in expert_scales.iter().enumerate() {
        let weight_f32: Vec<f32> = random_vec(101 + expert as u64, rows as usize * k as usize)
            .into_iter()
            .map(|value| (value * 4.0 - 2.0) * scale)
            .collect();
        let mut blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k as usize)
            .zip(blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
        }
        expert_blocks.push(blocks);
    }
    let stacked_weight: Vec<u8> = expert_blocks.iter().flatten().copied().collect();

    let activation: Vec<f32> = random_vec(211, seq as usize * k as usize)
        .into_iter()
        .map(|value| value * 2.0 - 1.0)
        .collect();

    let (program, sum) = gathered_quantized_matmul_program(n_experts, rows, k, seq);
    let quantized_blocks = [
        QuantizedBlock::Q4K(&stacked_weight),
        QuantizedBlock::Float32(&route_data),
        QuantizedBlock::Float32(&activation),
    ];
    let evaluated = evaluate_quantized(&program, &[], &quantized_blocks, &[sum])
        .expect("gathered quantized moe matmul evaluates end to end");
    let actual = evaluated.root();
    assert_eq!(actual.len(), seq as usize * rows as usize);

    for (token, &route) in route_data.iter().enumerate() {
        let expert = route as usize;
        let activation_row = &activation[token * k as usize..(token + 1) * k as usize];
        // Same codec kernel `run_reduce_quantized`'s per-position loop
        // itself calls for a `Q4K` block (`q4k-int8-dot` default-on
        // routes through `matmul_q4k_q8k_f32`; off, the plain
        // `matmul_q4k_f32`) -- comparing against the OTHER kernel would
        // fail on that kernel's own lossy Q8_K quantization step, not on
        // a wrong-expert read, which is the one thing this test exists
        // to catch.
        #[cfg(feature = "q4k-int8-dot")]
        let expected = matmul_q4k_q8k_f32(&expert_blocks[expert], rows as usize, activation_row)
            .expect("the routed expert's own standalone matmul evaluates");
        #[cfg(not(feature = "q4k-int8-dot"))]
        let expected = matmul_q4k_f32(&expert_blocks[expert], rows as usize, activation_row)
            .expect("the routed expert's own standalone matmul evaluates");
        let actual_row = &actual[token * rows as usize..(token + 1) * rows as usize];
        assert_eq!(
            actual_row, expected,
            "token {token} routed to expert {expert}: gathered result does not bit-match that \
             expert's own standalone matmul call"
        );
    }
}

/// Shared rig for the `ExpertSource` tests below: binds
/// [`gathered_quantized_matmul_program`]'s program, resolves the
/// gathered reduce node's own [`BoundOp`], and hands back everything
/// [`run_reduce_quantized`] itself needs to run that ONE node directly
/// -- the same setup [`evaluate_quantized_with_scratch_impl`] does for a
/// whole program, narrowed to one node so these tests can pass an
/// [`ExpertSource`] `run_reduce_quantized` has no public plumbing to
/// reach yet (see this crate's own call site in
/// `run_reduce_with_quantized_weights`, which always passes `None`).
fn resolve_gathered_reduce_for_expert_source_test<'a>(
    program: &'a [Op],
    sum: NodeId,
    route_data: &'a [f32],
    activation: &'a [f32],
) -> (BoundOp, Vec<Option<Cow<'a, [f32]>>>, NodeId) {
    let shapes = shape::infer(program, &[]).expect("shape inference succeeds");
    let resolved =
        bind::bind(program, &shapes, &[sum], NumericPolicy::bit_exact()).expect("bind succeeds");
    let block_nodes = block_node_ids(program);
    let mut buffers: Vec<Option<Cow<'a, [f32]>>> = vec![None; program.len()];
    // `gathered_quantized_matmul_program` emits exactly three `Op::Input`
    // nodes in this order: the packed weight (skipped -- it never rides
    // in `buffers`, only `weight_block`/`ExpertSource` below), the route
    // indices, then the activation.
    let weight_node = block_nodes[0];
    let route_node = block_nodes[1];
    let activation_node = block_nodes[2];
    buffers[route_node.0 as usize] = Some(Cow::Borrowed(route_data));
    buffers[activation_node.0 as usize] = Some(Cow::Borrowed(activation));
    let bound = resolved
        .into_iter()
        .find(|op| op.node == sum)
        .expect("the gathered reduce node is present in the bound program");
    (bound, buffers, weight_node)
}

/// ITEM 5 (ROW 422): a zero-byte packed stack with a nonzero
/// `expert_count` passes [`expert_entries_from_stack`]'s own alignment
/// check (`0 % expert_count == 0`), which used to fall straight into
/// `bytes.chunks_exact(0)` -- a bare `chunks_exact` panics if the chunk
/// width is `0`. This must be a typed [`TensorError::EmptyExpertPayload`]
/// instead, never a panic.
#[test]
fn expert_entries_from_stack_rejects_empty_payload_instead_of_panicking() {
    let empty_stack = QuantizedBlock::Q4K(&[]);
    let error = expert_entries_from_stack(empty_stack, 3, 4, 256, 0)
        .expect_err("an empty packed stack must be a typed rejection, not a panic");
    assert_eq!(
        error,
        TensorError::EmptyExpertPayload { expert_count: 3 },
        "an empty packed stack must be a typed rejection, not a panic inside chunks_exact"
    );
}

proptest! {
    /// ITEM 5 (ROW 422): no `(payload_len, expert_count)` pair drives
    /// [`expert_entries_from_stack`] into a panic -- every input either
    /// returns `Ok` (a genuinely block-aligned, nonempty stack) or one
    /// of its two typed errors (misaligned, or aligned-but-empty).
    #[test]
    fn expert_entries_from_stack_never_panics(
        payload_len in 0usize..=512,
        expert_count in 0usize..=16,
    ) {
        let payload = vec![0u8; payload_len];
        let stack = QuantizedBlock::Q4K(&payload);
        let _ = expert_entries_from_stack(stack, expert_count, 1, 1, 0);
    }
}

/// [`ExpertSource`]'s zero-copy default: an alias table built by
/// [`expert_entries_from_stack`] over the SAME contiguous `Q4_K` stack
/// [`evaluate_quantized_gathered_moe_weight_matches_the_routed_experts_own_matmul`]
/// reads directly produces a bit-identical product for every routed
/// token -- proving the indirection changes nothing when nothing asked
/// it to.
#[test]
fn expert_source_matches_stack_gather() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let n_experts: u32 = 3;
    let rows: u32 = 4;
    let k = QK_K as u32;
    let seq: u32 = 3;
    let route_data = [2.0f32, 0.0, 1.0];
    let expert_scales = [1.0f32, 5.0, 20.0];

    let mut expert_blocks: Vec<Vec<u8>> = Vec::new();
    for (expert, &scale) in expert_scales.iter().enumerate() {
        let weight_f32: Vec<f32> = random_vec(301 + expert as u64, rows as usize * k as usize)
            .into_iter()
            .map(|value| (value * 4.0 - 2.0) * scale)
            .collect();
        let block_bytes = BLOCK_BYTES;
        let mut blocks = vec![0u8; rows as usize * block_bytes];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k as usize)
            .zip(blocks.chunks_exact_mut(block_bytes))
        {
            quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
        }
        expert_blocks.push(blocks);
    }
    let stacked_weight: Vec<u8> = expert_blocks.iter().flatten().copied().collect();
    let activation: Vec<f32> = random_vec(311, seq as usize * k as usize)
        .into_iter()
        .map(|value| value * 2.0 - 1.0)
        .collect();

    let (program, sum) = gathered_quantized_matmul_program(n_experts, rows, k, seq);
    let weight_block = QuantizedBlock::Q4K(&stacked_weight);
    let entries = expert_entries_from_stack(weight_block, n_experts as usize, rows, k, 0)
        .expect("a block-aligned contiguous stack slices evenly");
    let source = ExpertSource::new(&entries);

    let (resolved, buffers, weight_node) =
        resolve_gathered_reduce_for_expert_source_test(&program, sum, &route_data, &activation);
    let mut via_source = vec![0.0f32; seq as usize * rows as usize];
    run_reduce_quantized(
        &resolved,
        &buffers,
        weight_block,
        weight_node,
        Some(source),
        None,
        false,
        &mut via_source,
    )
    .expect("ExpertSource-backed gather evaluates");

    let mut via_stack = vec![0.0f32; seq as usize * rows as usize];
    run_reduce_quantized(
        &resolved,
        &buffers,
        weight_block,
        weight_node,
        None,
        None,
        false,
        &mut via_stack,
    )
    .expect("the plain stack gather evaluates");

    assert_eq!(
        via_source, via_stack,
        "an ExpertSource aliasing one contiguous stack must be bit-identical to reading \
         that stack directly"
    );
}

/// A mixed-codec `ExpertSource` -- expert 1 re-encoded `Q2_K` from the
/// same f32 rows, experts 0 and 2 left at their native `Q4_K` -- reads
/// each expert through its OWN codec: routing to expert 1 must match a
/// direct `Q2_K` dequantize-and-fold, not the `Q4_K` bytes at the same
/// stack offset.
#[test]
fn expert_source_resolves_each_entry_through_its_own_codec() {
    use proxima_gguf::quant::q2_k;
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let n_experts: u32 = 3;
    let rows: u32 = 4;
    let k = QK_K as u32;
    let seq: u32 = 1;
    // Route the single token straight at the re-encoded expert.
    let route_data = [1.0f32];
    let expert_scales = [1.0f32, 5.0, 20.0];

    let mut expert_f32: Vec<Vec<f32>> = Vec::new();
    let mut expert_blocks: Vec<Vec<u8>> = Vec::new();
    for (expert, &scale) in expert_scales.iter().enumerate() {
        let weight_f32: Vec<f32> = random_vec(401 + expert as u64, rows as usize * k as usize)
            .into_iter()
            .map(|value| (value * 4.0 - 2.0) * scale)
            .collect();
        let block_bytes = BLOCK_BYTES;
        let mut blocks = vec![0u8; rows as usize * block_bytes];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k as usize)
            .zip(blocks.chunks_exact_mut(block_bytes))
        {
            quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
        }
        expert_blocks.push(blocks);
        expert_f32.push(weight_f32);
    }
    let stacked_weight: Vec<u8> = expert_blocks.iter().flatten().copied().collect();

    let q2k_block_bytes = q2_k::BLOCK_BYTES;
    let mut expert1_q2k = vec![0u8; rows as usize * q2k_block_bytes];
    for (row_f32, row_blocks) in expert_f32[1]
        .chunks_exact(k as usize)
        .zip(expert1_q2k.chunks_exact_mut(q2k_block_bytes))
    {
        q2_k::quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
    }

    let activation: Vec<f32> = random_vec(411, seq as usize * k as usize)
        .into_iter()
        .map(|value| value * 2.0 - 1.0)
        .collect();

    let (program, sum) = gathered_quantized_matmul_program(n_experts, rows, k, seq);
    let weight_block = QuantizedBlock::Q4K(&stacked_weight);
    let mut entries = expert_entries_from_stack(weight_block, n_experts as usize, rows, k, 0)
        .expect("a block-aligned contiguous stack slices evenly");
    entries[1] = ExpertEntry {
        block: QuantizedBlock::Q2K(&expert1_q2k),
        out_dim: rows,
        in_dim: k,
        epoch: 1,
    };
    let source = ExpertSource::new(&entries);

    let (resolved, buffers, weight_node) =
        resolve_gathered_reduce_for_expert_source_test(&program, sum, &route_data, &activation);
    let mut actual = vec![0.0f32; seq as usize * rows as usize];
    run_reduce_quantized(
        &resolved,
        &buffers,
        weight_block,
        weight_node,
        Some(source),
        None,
        false,
        &mut actual,
    )
    .expect("mixed-codec ExpertSource evaluates");

    let expected = matmul_q2k_f32(&expert1_q2k, rows as usize, &activation)
        .expect("the re-encoded expert's own standalone Q2_K matmul evaluates");
    assert_eq!(
        actual, expected,
        "routing to a Q2_K-re-encoded entry must match that entry's own codec, not the \
         Q4_K bytes the stack carries at the same offset"
    );
}

/// ITEM 4 (ROW 422): a mixed-codec `ExpertSource` -- expert 0 native
/// `Q4_K`, expert 1 re-encoded `Q2_K` -- routes one token to each. Only
/// the `Q4_K` position should ever add to `MATMUL_Q4K_MACS`; the `Q2_K`
/// position dispatches through `QuantizedBlock::Q2K`'s own arm, which
/// carries no counter at all (see the `match dispatch_block` instrument
/// block in `run_reduce_quantized`). Before the fix this counter matched
/// on `weight_block` -- the reduce's own declared `Q4_K` codec, fixed for
/// the whole call -- so BOTH positions (including the one that actually
/// ran a `Q2_K` dot product) added a `Q4_K`-shaped mac count, double the
/// true `Q4_K` work and a `Q2_K` call recorded as `Q4_K`.
///
/// `MATMUL_Q4K_MACS` is one process-wide atomic counter, so this test
/// only holds under `cargo nextest` (one process per test, this crate's
/// own default runner) -- under a same-process multi-threaded `cargo
/// test` run, a concurrently running Q4_K-matmul test can add to the
/// same counter between this test's own reset and read.
#[cfg(feature = "instrument")]
#[test]
fn expert_source_instrumentation_attributes_to_the_dispatched_codec() {
    use proxima_gguf::quant::q2_k;
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let n_experts: u32 = 2;
    let rows: u32 = 4;
    let k = QK_K as u32;
    let seq: u32 = 2;
    // Token 0 routes to expert 0 (stays `Q4_K`); token 1 routes to
    // expert 1 (re-encoded `Q2_K` below).
    let route_data = [0.0f32, 1.0];
    let expert_scales = [1.0f32, 5.0];

    let mut expert_f32: Vec<Vec<f32>> = Vec::new();
    let mut expert_blocks: Vec<Vec<u8>> = Vec::new();
    for (expert, &scale) in expert_scales.iter().enumerate() {
        let weight_f32: Vec<f32> = random_vec(501 + expert as u64, rows as usize * k as usize)
            .into_iter()
            .map(|value| (value * 4.0 - 2.0) * scale)
            .collect();
        let block_bytes = BLOCK_BYTES;
        let mut blocks = vec![0u8; rows as usize * block_bytes];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k as usize)
            .zip(blocks.chunks_exact_mut(block_bytes))
        {
            quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
        }
        expert_blocks.push(blocks);
        expert_f32.push(weight_f32);
    }
    let stacked_weight: Vec<u8> = expert_blocks.iter().flatten().copied().collect();

    let q2k_block_bytes = q2_k::BLOCK_BYTES;
    let mut expert1_q2k = vec![0u8; rows as usize * q2k_block_bytes];
    for (row_f32, row_blocks) in expert_f32[1]
        .chunks_exact(k as usize)
        .zip(expert1_q2k.chunks_exact_mut(q2k_block_bytes))
    {
        q2_k::quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
    }

    let activation: Vec<f32> = random_vec(511, seq as usize * k as usize)
        .into_iter()
        .map(|value| value * 2.0 - 1.0)
        .collect();

    let (program, sum) = gathered_quantized_matmul_program(n_experts, rows, k, seq);
    let weight_block = QuantizedBlock::Q4K(&stacked_weight);
    let mut entries = expert_entries_from_stack(weight_block, n_experts as usize, rows, k, 0)
        .expect("a block-aligned contiguous stack slices evenly");
    entries[1] = ExpertEntry {
        block: QuantizedBlock::Q2K(&expert1_q2k),
        out_dim: rows,
        in_dim: k,
        epoch: 1,
    };
    let source = ExpertSource::new(&entries);

    let (resolved, buffers, weight_node) =
        resolve_gathered_reduce_for_expert_source_test(&program, sum, &route_data, &activation);
    let mut actual = vec![0.0f32; seq as usize * rows as usize];
    instrument::reset_matmul_dispatch();
    run_reduce_quantized(
        &resolved,
        &buffers,
        weight_block,
        weight_node,
        Some(source),
        None,
        false,
        &mut actual,
    )
    .expect("mixed-codec ExpertSource evaluates");

    let totals = instrument::matmul_dispatch_totals();
    let one_call_macs = u64::from(rows) * u64::from(k);
    assert_eq!(
        totals.q4k_macs, one_call_macs,
        "exactly one position (the expert-0 Q4_K entry) executed a Q4_K dot product -- \
         counting the Q2_K-dispatched position (expert 1) against MATMUL_Q4K_MACS means \
         instrumentation attributed the wrong entry's codec"
    );
}

/// Swapping one `ExpertSource` entry between two evaluations changes
/// only that expert's own product, by exactly the codec's own error --
/// the second evaluation must equal the direct `Q2_K` product, never
/// the first evaluation's `Q4_K` product, proving the borrow (not a
/// mutation) is what "promotion" means here: two DISTINCT tables, one
/// per step, never one table mutated mid-step.
#[test]
fn expert_source_swap_between_evaluations_reads_the_new_entry_cleanly() {
    use proxima_gguf::quant::q2_k;
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let n_experts: u32 = 2;
    let rows: u32 = 4;
    let k = QK_K as u32;
    let seq: u32 = 1;
    let route_data = [1.0f32];
    let expert_scales = [1.0f32, 5.0];

    let mut expert_f32: Vec<Vec<f32>> = Vec::new();
    let mut expert_blocks: Vec<Vec<u8>> = Vec::new();
    for (expert, &scale) in expert_scales.iter().enumerate() {
        let weight_f32: Vec<f32> = random_vec(501 + expert as u64, rows as usize * k as usize)
            .into_iter()
            .map(|value| (value * 4.0 - 2.0) * scale)
            .collect();
        let block_bytes = BLOCK_BYTES;
        let mut blocks = vec![0u8; rows as usize * block_bytes];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k as usize)
            .zip(blocks.chunks_exact_mut(block_bytes))
        {
            quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
        }
        expert_blocks.push(blocks);
        expert_f32.push(weight_f32);
    }
    let stacked_weight: Vec<u8> = expert_blocks.iter().flatten().copied().collect();

    let q2k_block_bytes = q2_k::BLOCK_BYTES;
    let mut expert1_q2k = vec![0u8; rows as usize * q2k_block_bytes];
    for (row_f32, row_blocks) in expert_f32[1]
        .chunks_exact(k as usize)
        .zip(expert1_q2k.chunks_exact_mut(q2k_block_bytes))
    {
        q2_k::quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
    }

    let activation: Vec<f32> = random_vec(511, seq as usize * k as usize)
        .into_iter()
        .map(|value| value * 2.0 - 1.0)
        .collect();

    let (program, sum) = gathered_quantized_matmul_program(n_experts, rows, k, seq);
    let weight_block = QuantizedBlock::Q4K(&stacked_weight);
    let entries_q4k = expert_entries_from_stack(weight_block, n_experts as usize, rows, k, 0)
        .expect("a block-aligned contiguous stack slices evenly");

    let (resolved, buffers, weight_node) =
        resolve_gathered_reduce_for_expert_source_test(&program, sum, &route_data, &activation);

    let mut first = vec![0.0f32; seq as usize * rows as usize];
    run_reduce_quantized(
        &resolved,
        &buffers,
        weight_block,
        weight_node,
        Some(ExpertSource::new(&entries_q4k)),
        None,
        false,
        &mut first,
    )
    .expect("the first (Q4_K) evaluation succeeds");

    let mut entries_swapped = entries_q4k.clone();
    entries_swapped[1] = ExpertEntry {
        block: QuantizedBlock::Q2K(&expert1_q2k),
        out_dim: rows,
        in_dim: k,
        epoch: 1,
    };
    let mut second = vec![0.0f32; seq as usize * rows as usize];
    run_reduce_quantized(
        &resolved,
        &buffers,
        weight_block,
        weight_node,
        Some(ExpertSource::new(&entries_swapped)),
        None,
        false,
        &mut second,
    )
    .expect("the second (Q2_K) evaluation, over a NEW borrowed table, succeeds");

    assert_ne!(
        first, second,
        "swapping expert 1's own entry between two evaluations must change that expert's \
         own product"
    );
    let expected_q2k = matmul_q2k_f32(&expert1_q2k, rows as usize, &activation)
        .expect("the swapped-in expert's own standalone Q2_K matmul evaluates");
    assert_eq!(
        second, expected_q2k,
        "the second evaluation must read the SWAPPED entry's own codec exactly, not a \
         stale Q4_K read left over from the first evaluation's table"
    );
}

#[test]
fn all_expert_arena_requires_one_exact_span_per_packed_entry() {
    let first = [11_u8, 12, 13];
    let second = [21_u8, 22];
    let arena = [0_u8, 11, 12, 13, 0, 21, 22];
    let entries = [
        ExpertEntry {
            block: QuantizedBlock::Q2K(&first),
            out_dim: 1,
            in_dim: 1,
            epoch: 0,
        },
        ExpertEntry {
            block: QuantizedBlock::Q2K(&second),
            out_dim: 1,
            in_dim: 1,
            epoch: 0,
        },
    ];
    let spans = [
        Some(ExpertPayloadSpan {
            offset: 1,
            length: 3,
        }),
        Some(ExpertPayloadSpan {
            offset: 5,
            length: 2,
        }),
    ];

    let source = ExpertSource::with_all_expert_arena(&entries, &arena, &spans)
        .expect("every expert has one exact non-overlapping arena span");

    assert_eq!(source.selected_expert_ids(), None);
    assert_eq!(
        source
            .packed_arena()
            .expect("the all-expert constructor retains its arena")
            .spans(),
        &spans,
    );
}

#[test]
fn all_expert_arena_rejects_a_missing_expert_span() {
    let first = [11_u8, 12, 13];
    let second = [21_u8, 22];
    let arena = [11_u8, 12, 13, 21, 22];
    let entries = [
        ExpertEntry {
            block: QuantizedBlock::Q2K(&first),
            out_dim: 1,
            in_dim: 1,
            epoch: 0,
        },
        ExpertEntry {
            block: QuantizedBlock::Q2K(&second),
            out_dim: 1,
            in_dim: 1,
            epoch: 0,
        },
    ];
    let spans = [
        Some(ExpertPayloadSpan {
            offset: 0,
            length: 3,
        }),
        None,
    ];

    let error = ExpertSource::with_all_expert_arena(&entries, &arena, &spans)
        .expect_err("an all-expert arena cannot omit one descriptor");

    assert!(matches!(
        error,
        TensorError::InvalidExpertPayloadArena {
            reason: "all-expert arena is missing an expert payload span"
        }
    ));
}

/// [`ExpertSource::entry`]'s own shape guard: an entry declaring a
/// different `[out_dim, in_dim]` than the program's own resolved
/// expert shape is rejected by name, never silently dotted against the
/// wrong-length activation row.
#[test]
fn expert_source_rejects_an_entry_whose_shape_disagrees_with_the_program() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let n_experts: u32 = 2;
    let rows: u32 = 4;
    let k = QK_K as u32;
    let seq: u32 = 1;
    let route_data = [0.0f32];

    let mut expert_blocks: Vec<Vec<u8>> = Vec::new();
    for expert in 0..n_experts {
        let weight_f32: Vec<f32> = random_vec(601 + u64::from(expert), rows as usize * k as usize);
        let block_bytes = BLOCK_BYTES;
        let mut blocks = vec![0u8; rows as usize * block_bytes];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k as usize)
            .zip(blocks.chunks_exact_mut(block_bytes))
        {
            quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
        }
        expert_blocks.push(blocks);
    }
    let stacked_weight: Vec<u8> = expert_blocks.iter().flatten().copied().collect();
    let activation: Vec<f32> = random_vec(611, seq as usize * k as usize);

    let (program, sum) = gathered_quantized_matmul_program(n_experts, rows, k, seq);
    let weight_block = QuantizedBlock::Q4K(&stacked_weight);
    let mut entries = expert_entries_from_stack(weight_block, n_experts as usize, rows, k, 0)
        .expect("a block-aligned contiguous stack slices evenly");
    // Expert 0's own entry now lies about its row count.
    entries[0].out_dim = rows + 1;

    let (resolved, buffers, weight_node) =
        resolve_gathered_reduce_for_expert_source_test(&program, sum, &route_data, &activation);
    let mut output = vec![0.0f32; seq as usize * rows as usize];
    let error = run_reduce_quantized(
        &resolved,
        &buffers,
        weight_block,
        weight_node,
        Some(ExpertSource::new(&entries)),
        None,
        false,
        &mut output,
    )
    .expect_err("a shape-mismatched entry must be rejected, not silently dotted");
    match error {
        TensorError::ExpertSourceShapeMismatch {
            expert,
            entry_out,
            expected_out,
            ..
        } => {
            assert_eq!(
                expert, 0,
                "expert 0's own entry is the one that lied about its shape"
            );
            assert_eq!(entry_out, rows + 1);
            assert_eq!(expected_out, rows);
        }
        other => panic!("expected ExpertSourceShapeMismatch, got {other}"),
    }
}

// The kernel-level `ExpertObserver` hook (`instrument::notify_expert_routed`
// called from `run_reduce_quantized`'s gathered-weight loop) moved to the
// decode loop (`proxima-model-interop::generate`), which is the only emit
// site now -- see that crate's own decode-loop observer fixture for the
// event-shape proof this file used to carry.

/// Hand-computes the exact `ema` [`ExpertSelectionEntry::record`]'s own
/// recurrence produces after `selections` calls, starting from `0.0` --
/// same formula, same floating-point operation order, so this is a
/// bit-exact oracle, not a tolerance check.
fn expected_ema(selections: u32) -> f64 {
    let mut ema = 0.0f64;
    for _ in 0..selections {
        ema += EXPERT_SELECTION_EMA_ALPHA * (1.0 - ema);
    }
    ema
}

/// [`record_expert_selection`]/[`snapshot_expert_selection_top_n`]'s own
/// fixture: a hand-computed 4-expert routing where only 2 experts are
/// ever actually selected (expert 1 three times, expert 3 once, experts
/// 0 and 2 never) -- exactly the "4-expert, 2-used" shape this task's
/// own doc asks for. Uses out-of-band synthetic `NodeId`s
/// (`EXPERT_SELECTION_TEST_NODE`/`+1`) rather than routing real tensors
/// through [`evaluate_quantized`]: [`EXPERT_SELECTION_COUNTS`] is one
/// process-wide table every test in this file's binary shares
/// (`nextest`/`cargo test` run this module's tests concurrently by
/// default), so filtering [`snapshot_expert_selection_top_n`]'s drained
/// result down to these two never-otherwise-used node ids is what makes
/// this test race-safe against
/// [`evaluate_quantized_gathered_moe_weight_matches_the_routed_experts_own_matmul`]
/// (real program, low `NodeId`s) recording into the SAME table on
/// another thread at the same time.
#[test]
fn expert_selection_counts_match_hand_computed_four_expert_two_used_routing() {
    const EXPERT_SELECTION_TEST_NODE: NodeId = NodeId(0xE_5000);
    const OTHER_TEST_NODE: NodeId = NodeId(0xE_5001);

    // Layer A: 5 tokens route [1, 3, 1, 1, 3] across 4 experts (0..=3) --
    // expert 1 selected 3 times, expert 3 selected 2 times, experts 0
    // and 2 never selected.
    for expert in [1u32, 3, 1, 1, 3] {
        record_expert_selection(EXPERT_SELECTION_TEST_NODE, expert);
    }
    // Layer B (a different node): a single token routes to expert 2,
    // proving the table keys on (node, expert), not expert alone --
    // if it collapsed the node axis, this would corrupt layer A's own
    // expert-1/expert-3 counts.
    record_expert_selection(OTHER_TEST_NODE, 2);

    let snapshot = snapshot_expert_selection_top_n(100);
    let mut layer_a: Vec<(u32, u64, f64)> = snapshot
        .iter()
        .filter(|(node, _, _, _)| *node == EXPERT_SELECTION_TEST_NODE)
        .map(|(_, expert, count, ema)| (*expert, *count, *ema))
        .collect();
    layer_a.sort_unstable_by_key(|(expert, _, _)| *expert);
    assert_eq!(
        layer_a,
        alloc::vec![(1u32, 3u64, expected_ema(3)), (3u32, 2u64, expected_ema(2)),],
        "expert 1 must show 3 selections and expert 3 must show 2, experts 0/2 never selected, \
         and each ema must match the same recurrence at that selection count"
    );

    let layer_b: Vec<(u32, u64, f64)> = snapshot
        .iter()
        .filter(|(node, _, _, _)| *node == OTHER_TEST_NODE)
        .map(|(_, expert, count, ema)| (*expert, *count, *ema))
        .collect();
    assert_eq!(
        layer_b,
        alloc::vec![(2u32, 1u64, expected_ema(1))],
        "a different node's own single selection must not merge into layer A's counts"
    );
}

/// [`snapshot_expert_selection_top_n`]'s own drain contract: a second
/// call right after the first sees none of the first call's entries --
/// the same "a later step starts from zero" guarantee
/// [`proxima_telemetry::metric::Counter::snapshot_and_reset`] gives a
/// scalar counter.
#[test]
fn expert_selection_snapshot_drains_the_table() {
    const DRAIN_TEST_NODE: NodeId = NodeId(0xE_5002);
    record_expert_selection(DRAIN_TEST_NODE, 0);
    let first = snapshot_expert_selection_top_n(100);
    assert!(
        first.iter().any(|(node, expert, count, ema)| {
            *node == DRAIN_TEST_NODE && *expert == 0 && *count == 1 && *ema == expected_ema(1)
        }),
        "the recorded selection must appear in the first snapshot with its own ema"
    );
    let second = snapshot_expert_selection_top_n(100);
    assert!(
        !second
            .iter()
            .any(|(node, _, _, _)| *node == DRAIN_TEST_NODE),
        "a drained entry must not reappear in the very next snapshot"
    );
}

/// `evaluate_quantized`'s `live_now` running count treats every operand
/// `node_retirement` schedules for retirement as "always live here" (see
/// the comment above `live_now -= 1` in that loop) -- true for an
/// `Op::Input` float32 block and for a computed node, both of which are
/// written into `buffers`, but FALSE for a `Q4_K`-packed weight node: the
/// `QuantizedBlock::Q4K` match arm in `evaluate_quantized_with_scratch`
/// routes it into the separate `quantized_weights` map instead of
/// `buffers`, so its slot is `None` the whole time. `node_retirement`
/// does not know about that split -- it schedules the weight's
/// retirement from the generic program graph like any other operand --
/// so every quantized-weight retirement decrements `live_now` for a slot
/// that was never live. One layer only drifts the running count by one
/// (silently wrong, not yet negative); two independent layers combined
/// by a plain `Add` accumulate two such spurious decrements ahead of the
/// real ones and drive the last, legitimate retirement negative --
/// `attempt to subtract with overflow` in a debug build, silent
/// wraparound to a huge `usize` in release (this session's
/// `peak_live_buffers=18446744073709551614` == `2^64 - 2`).
#[test]
fn evaluate_quantized_two_layers_does_not_underflow_live_now() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    let rows: u32 = 3;
    let blocks_per_row = 1;
    let k = QK_K as u32 * blocks_per_row as u32;

    fn quantized_weight_blocks(seed: u64, rows: u32, k: u32, blocks_per_row: usize) -> Vec<u8> {
        let weight_f32: Vec<f32> = random_vec(seed, rows as usize * k as usize)
            .into_iter()
            .map(|value| value * 4.0 - 2.0)
            .collect();
        let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k as usize)
            .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks)
                .expect("row length is a whole multiple of QK_K by construction");
        }
        weight_blocks
    }

    fn append_layer(program: &mut Vec<Op>, rows: u32, k: u32) -> NodeId {
        let weight = block(
            program,
            DType::UInt8,
            &[Extent::Static(rows), Extent::Static(k)],
        );
        let activation = f32_block(program, &[Extent::Static(k), Extent::Static(1)]);
        let product = append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        append(
            program,
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
        )
    }

    let mut program = Vec::new();
    let sum1 = append_layer(&mut program, rows, k);
    let sum2 = append_layer(&mut program, rows, k);
    let total = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![
                (sum1, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (sum2, IndexMap::Affine(map::projection(2, &[0, 1]))),
            ],
            name: None,
        },
    );

    let weight1_blocks = quantized_weight_blocks(101, rows, k, blocks_per_row);
    let weight2_blocks = quantized_weight_blocks(202, rows, k, blocks_per_row);
    let activation1: Vec<f32> = random_vec(303, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let activation2: Vec<f32> = random_vec(404, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let blocks = [
        QuantizedBlock::Q4K(&weight1_blocks),
        QuantizedBlock::Float32(&activation1),
        QuantizedBlock::Q4K(&weight2_blocks),
        QuantizedBlock::Float32(&activation2),
    ];

    let evaluated = evaluate_quantized(&program, &[], &blocks, &[total])
        .expect("two chained quantized layers evaluate without panicking");

    let peak_live_buffers = evaluated
        .peak_live_buffers()
        .expect("evaluate_quantized always reports peak_live_buffers");
    assert!(
        peak_live_buffers <= program.len(),
        "peak_live_buffers={peak_live_buffers} exceeds program.len()={} -- a live-buffer count can never \
         exceed the node count, so this value proves live_now underflowed and wrapped rather than being \
         merely large",
        program.len(),
    );
}

/// `materialize_quantized_weight_output`'s pre-fix scan matched
/// [`BoundOpKind::Elementwise`] alone, so a quantized weight reachable
/// ONLY through a [`BoundOpKind::Reduce`]'s own `epilogue_operands` (the
/// exact shape `reduce-epilogue-fusion` gives an RMSNorm gamma multiply,
/// per `apply_reduce_epilogue`'s own doc) was never dequantized into
/// `buffers`, and `apply_reduce_epilogue`'s `buffer_of` call raised the
/// same `NotLowerable { reason: "operand buffer missing at evaluation
/// time" }` node `6540` hit on the real qwen35moe checkpoint
/// (`docs/discipline.md`). This hand-builds ONE `Reduce` `BoundOp` with a
/// quantized `epilogue_operands` entry directly (bypassing `bind::bind`,
/// which only ever produces this shape when the `reduce-epilogue-fusion`
/// feature is compiled in — not this crate's default set) so the fix is
/// exercised without that feature: first proves the pre-fix error still
/// reproduces mechanically against the unwidened buffer table, then proves
/// `materialize_quantized_weights_read_by_non_primary_operands` closes it
/// and the fused evaluation matches a hand-computed Float32 reference.
#[test]
fn reduce_epilogue_only_quantized_weight_is_materialized_before_evaluation() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows = QK_K as u32; // the per-row weight's own length must be a whole QK_K block
    let k = 8u32; // the reduced axis -- any width works, kept small

    let mut program = Vec::new();
    let x_node = f32_block(&mut program, &[Extent::Static(rows), Extent::Static(k)]);
    let gate_node = block(&mut program, DType::UInt8, &[Extent::Static(rows)]);
    let shapes = shape::infer(&program, &[]).expect("the two-input shape program infers");

    let x_data: Vec<f32> = random_vec(551, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let gate_f32: Vec<f32> = random_vec(552, rows as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let mut gate_bytes = vec![0u8; BLOCK_BYTES];
    quantize(&gate_f32, &mut gate_bytes).expect("rows is a whole QK_K block by construction");
    let mut gate_dequantized = vec![0.0f32; rows as usize];
    dequantize(&gate_bytes, &mut gate_dequantized).expect("the packed gate block dequantizes");

    let resolved = BoundOp {
        node: NodeId(2),
        dtype: DType::Float32,
        extents: alloc::vec![rows as u64, k as u64],
        kind: BoundOpKind::Reduce {
            element_body: ComposedBody {
                steps: alloc::vec![step(ScalarOp::Identity, &[StepArg::Operand(0)])],
            },
            reduce_op: ScalarOp::Add,
            init: ReduceInit::Zero,
            keep: Keep::Reduce,
            operands: alloc::vec![(
                x_node,
                bind::Layout {
                    base: 0,
                    strides: smallvec::smallvec![k as i64, 1],
                },
                None,
            )],
            output_axes: smallvec::smallvec![0],
            out_layout: bind::Layout {
                base: 0,
                strides: smallvec::smallvec![1],
            },
            out_scatter: None,
            epilogue_body: ComposedBody {
                steps: alloc::vec![step(
                    ScalarOp::Multiply,
                    &[StepArg::Operand(0), StepArg::Operand(1)]
                )],
            },
            epilogue_operands: alloc::vec![(
                gate_node,
                bind::Layout {
                    base: 0,
                    strides: smallvec::smallvec![1],
                },
                None,
            )],
            epilogue_broadcast_axes: smallvec::smallvec![],
        },
    };

    let quantized_weights: BTreeMap<NodeId, QuantizedBlock> =
        BTreeMap::from([(gate_node, QuantizedBlock::Q4K(&gate_bytes))]);
    let mut buffers: Vec<Option<Cow<'_, [f32]>>> = alloc::vec![None; program.len()];
    buffers[x_node.0 as usize] = Some(Cow::Borrowed(x_data.as_slice()));

    let mut output = vec![0.0f32; rows as usize];
    let pre_fix_error = run_node_into(
        &resolved,
        &buffers,
        Some(&quantized_weights),
        None,
        None,
        false,
        &mut output,
    )
    .expect_err(
        "the gate's own buffer slot is still None here -- the epilogue read must fail exactly \
         as it did on the real checkpoint before the materialize scan was widened",
    );
    assert_eq!(
        pre_fix_error,
        TensorError::NotLowerable {
            node: gate_node,
            reason: "operand buffer missing at evaluation time",
        },
        "the pre-fix failure must name the epilogue's own quantized operand, not the fold"
    );

    materialize_quantized_weights_read_by_non_primary_operands(
        core::slice::from_ref(&resolved),
        &shapes,
        &quantized_weights,
        None,
        &mut buffers,
    )
    .expect("the widened scan dequantizes a Reduce epilogue's own quantized operand");
    assert!(
        buffers[gate_node.0 as usize].is_some(),
        "the gate's buffer slot must be populated after the widened materialize scan"
    );

    output.fill(0.0);
    run_node_into(
        &resolved,
        &buffers,
        Some(&quantized_weights),
        None,
        None,
        false,
        &mut output,
    )
    .expect("evaluation succeeds once the epilogue's quantized operand is materialized");

    for row in 0..rows as usize {
        let row_sum: f32 = x_data[row * k as usize..(row + 1) * k as usize]
            .iter()
            .sum();
        let expected = gate_dequantized[row] * row_sum;
        assert!(
            (output[row] - expected).abs() <= 1e-3 * expected.abs().max(1.0),
            "row {row}: got {}, expected {expected} (gate={}, row_sum={row_sum})",
            output[row],
            gate_dequantized[row],
        );
    }
}

/// A small, synthetic stand-in for the cached-attention reduce shape
/// that trips the seam `q8_0_quantized_key_value_cache_cannot_cross_the_weight_matmul_quantized_seam`
/// (`proxima-model-interop/src/bind.rs`) reaches on a real checkpoint:
/// one axis (`u`, the kv-head analog) that the packed weight operand
/// varies over AND the activation operand ALSO varies over, kept as an
/// OUTPUT axis (not the contracted one). Iteration space `[s, t, u, d]`
/// — `t` (cached-length analog) is the sole reduced axis; `s`, `u`, `d`
/// survive into the output. The packed weight varies over `t, u, d`
/// (broadcasts over `s`); the activation varies over `s, t, u`
/// (broadcasts over `d`) — `u` is the axis both share.
///
/// Before this fix, `run_reduce_quantized`'s raw-byte division
/// (`cpu.rs:2476-2486`) derived `rows` from the packed weight's own byte
/// length alone (`u * d` here, 32), then checked `output.len() / rows`
/// against `activation.len() / k` — `64 / 32 = 2` (real leading axis
/// `s`) against `32 / 4 = 8` (a leading count that has already folded in
/// `u`, which `rows` also claimed) — an off-by-`u`-factor disagreement,
/// not a coincidence: this is the same "kv-head factor" mismatch the
/// real checkpoint reduce hits, rejected with the generic
/// "does not evenly divide" reason.
///
/// After this fix, the rejection is structural rather than a cardinality
/// coincidence: `resolve_reduce_axis_shape` (shared with [`run_reduce`])
/// finds `u` nonzero-stride on BOTH the packed weight and the
/// activation, which is not a shape `run_reduce_quantized`'s one flat
/// `[rows, k] x [k]` matmul kernel can express (one activation vector
/// dotted against every packed row). Closing this for real needs a
/// per-position packed-weight byte offset the interpreter does not have
/// today — a capability gap, not a shape-arithmetic one — so this test
/// still asserts a rejection, now for the reason that is actually true.
#[test]
fn a_reduce_where_activation_and_packed_weight_share_a_kept_output_axis_is_rejected() {
    use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, quantize};

    let sequence_len: u32 = 2; // s -- leading, weight-broadcast
    let cached_len: u32 = 4; // t -- the sole reduced axis
    let kv_heads: u32 = 4; // u -- shared between weight and activation
    let head_dim: u32 = 8; // d -- weight-only row axis

    let mut program = Vec::new();
    let weight = block(
        &mut program,
        DType::UInt8,
        &[
            Extent::Static(cached_len),
            Extent::Static(kv_heads),
            Extent::Static(head_dim),
        ],
    );
    let activation = f32_block(
        &mut program,
        &[
            Extent::Static(sequence_len),
            Extent::Static(cached_len),
            Extent::Static(kv_heads),
        ],
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (weight, IndexMap::Affine(map::projection(4, &[1, 2, 3]))),
                (activation, IndexMap::Affine(map::projection(4, &[0, 1, 2]))),
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
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 2, 3])),
            keep: Keep::Reduce,
            name: Some("cached_attention_shaped_reduce".into()),
        }),
    );

    let weight_elements = (cached_len * kv_heads * head_dim) as usize;
    assert_eq!(
        weight_elements % QK8_0,
        0,
        "fixture dims must divide evenly into whole Q8_0 blocks"
    );
    let weight_f32: Vec<f32> = random_vec(29, weight_elements);
    let mut weight_bytes = vec![0u8; (weight_elements / QK8_0) * BLOCK_BYTES];
    quantize(&weight_f32, &mut weight_bytes)
        .expect("fixture weight length is a whole multiple of QK8_0");

    let activation_values: Vec<f32> =
        random_vec(31, (sequence_len * cached_len * kv_heads) as usize);

    let blocks = [
        QuantizedBlock::Q8_0(&weight_bytes),
        QuantizedBlock::Float32(&activation_values),
    ];
    let outcome = evaluate_quantized(&program, &[], &blocks, &[sum]);

    let error = outcome.expect_err(
        "a packed weight operand and its activation sharing a kept output axis is not a flat \
         matmul this interpreter can express -- see this test's own doc",
    );
    assert_eq!(
        error,
        TensorError::NotLowerable {
            node: sum,
            reason: "quantized matmul activation varies along an output axis its packed weight also \
                     varies along -- not a flat weight matmul this interpreter can express",
        }
    );
}

// -------------------------------------------------------------
// `Float16`/`BFloat16` -- composed convert-then-fold kernels
// (`dot_f16_f32`/`dot_bf16_f32`), hand-computed expected values (never
// the implementation checked against itself), plus one end-to-end
// `evaluate_quantized` parity test against `proxima_gguf`'s own
// dequantize (guiding-principle 14: the dequantize-then-matmul
// reference is correct by construction).
// -------------------------------------------------------------

/// Which half-precision codec a parameterized case exercises -- the two
/// formats share the same [`dot_f16_f32`]/[`dot_bf16_f32`] +
/// [`matmul_f16_f32`]/[`matmul_bf16_f32`] call shape but pack a value to
/// a different bit pattern, so the byte-packing step is the one thing
/// each case varies.
#[derive(Clone, Copy, Debug)]
enum HalfPrecisionKind {
    F16,
    Bf16,
}

impl HalfPrecisionKind {
    fn pack(self, values: &[f32]) -> Vec<u8> {
        match self {
            Self::F16 => values
                .iter()
                .flat_map(|&value| f16::from_f32(value).to_le_bytes())
                .collect(),
            Self::Bf16 => values
                .iter()
                .flat_map(|&value| bf16::from_f32(value).to_le_bytes())
                .collect(),
        }
    }

    fn dot(self, weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
        match self {
            Self::F16 => dot_f16_f32(weight_row, activation),
            Self::Bf16 => dot_bf16_f32(weight_row, activation),
        }
    }

    fn matmul(
        self,
        weights: &[u8],
        rows: usize,
        activation: &[f32],
    ) -> Result<Vec<f32>, TensorError> {
        match self {
            Self::F16 => matmul_f16_f32(weights, rows, activation),
            Self::Bf16 => matmul_bf16_f32(weights, rows, activation),
        }
    }
}

/// `weight . activation` computed by hand: `1.0*2.0 + 2.0*0.5 +
/// (-1.0)*3.0 + 0.5*4.0 = 2.0 + 1.0 - 3.0 + 2.0 = 2.0`. Every value on
/// both sides is an exact power of two (or zero), and both binary16 and
/// bfloat16 represent every power of two in this range exactly, so the
/// composed convert-then-fold kernel must reproduce `2.0` bit-exactly —
/// this checks the kernel against arithmetic done by hand, not against
/// itself.
#[proxima::test]
#[case::f16(HalfPrecisionKind::F16)]
#[case::bf16(HalfPrecisionKind::Bf16)]
async fn dot_half_precision_matches_a_hand_computed_dot_product(#[case] kind: HalfPrecisionKind) {
    let weight = [1.0f32, 2.0, -1.0, 0.5];
    let activation = [2.0f32, 0.5, 3.0, 4.0];
    let weight_bytes = kind.pack(&weight);

    let actual = kind
        .dot(&weight_bytes, &activation)
        .expect("well-formed half-precision row");

    assert_eq!(actual, 2.0f32, "hand-computed dot product ({kind:?})");
}

/// Two rows, each hand-computed independently: row 0 is the dot-product
/// fixture above (`2.0`); row 1 is `[0.0, 1.0, 0.0, -2.0] . [2.0, 0.5,
/// 3.0, 4.0] = 0 + 0.5 + 0 - 8.0 = -7.5` -- again every value an exact
/// power of two (or zero), so both formats reproduce it bit-exactly.
#[proxima::test]
#[case::f16(HalfPrecisionKind::F16)]
#[case::bf16(HalfPrecisionKind::Bf16)]
async fn matmul_half_precision_matches_a_hand_computed_two_row_matmul(
    #[case] kind: HalfPrecisionKind,
) {
    let row0 = [1.0f32, 2.0, -1.0, 0.5];
    let row1 = [0.0f32, 1.0, 0.0, -2.0];
    let activation = [2.0f32, 0.5, 3.0, 4.0];
    let weights: Vec<f32> = row0.iter().chain(row1.iter()).copied().collect();
    let weight_bytes = kind.pack(&weights);

    let actual = kind
        .matmul(&weight_bytes, 2, &activation)
        .expect("well-formed 2-row half-precision matmul");

    assert_eq!(
        actual,
        alloc::vec![2.0f32, -7.5],
        "hand-computed 2-row matmul ({kind:?})"
    );
}

/// Proves the hand-computed assertion above is load-bearing rather than
/// vacuous: a deliberately wrong second-row expectation (`123.0` in
/// place of the hand-computed `-7.5`) checked with `assert_ne!` against
/// the kernel's real output, so this file itself carries the evidence
/// that a wrong answer is caught, without needing to hand-edit the test
/// file to demonstrate it.
#[proxima::test]
#[case::f16(HalfPrecisionKind::F16)]
#[case::bf16(HalfPrecisionKind::Bf16)]
async fn matmul_half_precision_hand_computed_assertion_can_actually_fail(
    #[case] kind: HalfPrecisionKind,
) {
    let row0 = [1.0f32, 2.0, -1.0, 0.5];
    let row1 = [0.0f32, 1.0, 0.0, -2.0];
    let activation = [2.0f32, 0.5, 3.0, 4.0];
    let weights: Vec<f32> = row0.iter().chain(row1.iter()).copied().collect();
    let weight_bytes = kind.pack(&weights);

    let actual = kind
        .matmul(&weight_bytes, 2, &activation)
        .expect("well-formed 2-row half-precision matmul");
    let deliberately_wrong = alloc::vec![2.0f32, 123.0];

    assert_ne!(
        actual, deliberately_wrong,
        "a deliberately wrong expectation must not match the kernel's real output ({kind:?})"
    );
}

/// [`dot_f16_f32`]/[`dot_bf16_f32`] reject a byte length that is not a
/// whole number of 2-byte elements, and an activation length that does
/// not match the decoded element count -- never a panic or an
/// out-of-bounds read.
#[proxima::test]
#[case::f16(HalfPrecisionKind::F16)]
#[case::bf16(HalfPrecisionKind::Bf16)]
async fn dot_half_precision_rejects_a_malformed_shape(#[case] kind: HalfPrecisionKind) {
    let odd_bytes = alloc::vec![0u8; 3];
    let activation = [0.0f32; 1];
    assert!(matches!(
        kind.dot(&odd_bytes, &activation),
        Err(TensorError::QuantizedShapeMismatch { .. })
    ));

    let weight_bytes = kind.pack(&[1.0, 2.0]);
    let mismatched_activation = [0.0f32; 3];
    assert!(matches!(
        kind.dot(&weight_bytes, &mismatched_activation),
        Err(TensorError::QuantizedShapeMismatch { .. })
    ));
}

/// End-to-end: a `Float16`/`BFloat16` [`QuantizedBlock`] weight bound
/// through [`evaluate_quantized`]'s full `Op` graph (elementwise
/// multiply feeding an add-reduce -- the matmul shape
/// `is_quantized_matmul_operand` recognizes), checked against
/// `proxima_gguf`'s own tested `dequantize` followed by a naive `f32`
/// dot product -- guiding-principle 14: the dequantize-then-matmul
/// reference is correct by construction, so this is a parity check
/// against an independent path, not a round-trip-to-self check. Real
/// (pseudo-random, non-degenerate -- `Lcg`) weight and activation
/// values, not zeros or constants.
#[proxima::test]
#[case::f16(HalfPrecisionKind::F16)]
#[case::bf16(HalfPrecisionKind::Bf16)]
async fn evaluate_quantized_executes_a_half_precision_weight_end_to_end(
    #[case] kind: HalfPrecisionKind,
) {
    const ROWS: u32 = 3;
    const K: u32 = 16;
    const K_USIZE: usize = K as usize;

    let weights_f32 = random_vec(41, (ROWS * K) as usize);
    let activation = random_vec(43, K as usize);
    let weight_bytes = kind.pack(&weights_f32);

    let mut program = Vec::new();
    let weight = block(
        &mut program,
        DType::UInt8,
        &[Extent::Static(ROWS), Extent::Static(K)],
    );
    let activation_node = f32_block(&mut program, &[Extent::Static(K)]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (weight, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (activation_node, IndexMap::Affine(map::projection(2, &[1]))),
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
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("half_precision_matmul".into()),
        }),
    );

    let blocks = match kind {
        HalfPrecisionKind::F16 => alloc::vec![
            QuantizedBlock::Float16(&weight_bytes),
            QuantizedBlock::Float32(&activation)
        ],
        HalfPrecisionKind::Bf16 => alloc::vec![
            QuantizedBlock::BFloat16(&weight_bytes),
            QuantizedBlock::Float32(&activation)
        ],
    };
    let evaluated = evaluate_quantized(&program, &[], &blocks, &[sum])
        .expect("half-precision matmul evaluates");
    let actual = evaluated.root();

    let mut dequantized = vec![0.0f32; (ROWS * K) as usize];
    match kind {
        HalfPrecisionKind::F16 => {
            proxima_gguf::quant::f16::dequantize(&weight_bytes, &mut dequantized)
                .expect("well-formed f16 bytes");
        }
        HalfPrecisionKind::Bf16 => {
            proxima_gguf::quant::bf16::dequantize(&weight_bytes, &mut dequantized)
                .expect("well-formed bf16 bytes");
        }
    }
    let expected: Vec<f32> = dequantized
        .as_chunks::<K_USIZE>()
        .0
        .iter()
        .map(|row| {
            row.iter()
                .zip(&activation)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect();

    assert_eq!(
        actual.len(),
        ROWS as usize,
        "degenerate gate: no outputs compared"
    );
    // Not `assert_eq!`: `dot_fold_fused_multiply_add`'s `DOT_LANES` (8)
    // independent partial sums (K=16 here is two whole lanes) combine
    // in a different order than this reference's strict left-to-right
    // `Iterator::sum` -- float addition is not associative, so the two
    // legitimately differ in the last mantissa bits. The K4Q4_K parity
    // test above (`matmul_q4k_f32_matches_dequantize_then_f32_matmul`)
    // hits the exact same reordering and uses the same loose-tolerance
    // shape rather than bit-exact equality.
    let mut max_diff = 0.0f32;
    for (got, want) in actual.iter().zip(&expected) {
        assert!(
            got.is_finite(),
            "half-precision matmul ({kind:?}) produced a non-finite value: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    eprintln!("half-precision matmul ({kind:?}) vs dequantize-then-fold: max_diff={max_diff}");
    assert!(
        max_diff < 1e-4,
        "half-precision matmul ({kind:?}) disagrees with the dequantize-then-fold reference: max_diff={max_diff}"
    );
}

// -------------------------------------------------------------
// `Q5_K`/`Q6_K` packed int8 kernels -- bit-exactness (synthetic,
// both arms on the SAME weight bytes) and correctness against the
// dequantize-then-fold reference path on REAL packed bytes read
// straight out of the real openchat-3.5-1210 `Q4_K_S` GGUF file
// (guiding-principles principle 9: real-world data, not a synthetic
// stand-in) -- the same discipline `bench_q4k_matmul.rs` applies
// against ggml, minus the timing: correctness only, per this
// landing's task scope.
// -------------------------------------------------------------

/// Streams `path`'s header/tensor-directory prefix in growing chunks
/// until [`proxima_gguf::parser::GgufParser`] reports `Complete`,
/// without ever reading the (multi-GiB) tensor data section -- the
/// same technique `bench_q4k_matmul.rs::parse_header` uses, duplicated
/// here rather than shared across the lib/bench boundary (benches are
/// their own crate roots in this workspace). Returns `None` if the
/// file does not exist on this host, so these tests degrade to a
/// no-op on a machine without the real model file rather than a hard
/// failure.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
fn real_gguf_header(
    path: &std::path::Path,
) -> Option<(proxima_gguf::pipe::ParsedGguf, u64, std::fs::File)> {
    use std::io::{Read, Seek, SeekFrom};

    use proxima_gguf::parser::{GgufEvent, GgufParser};
    use proxima_gguf::pipe::ParsedGguf;

    let mut file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();

    let mut prefix_len = 1usize << 20;
    loop {
        let mut buf = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).expect("seek to start");
        let read = file.read(&mut buf).expect("read gguf prefix");
        buf.truncate(read);

        if let Ok((parser, events)) = GgufParser::new().push(&buf) {
            let mut version = None;
            let mut metadata = Vec::new();
            let mut tensors = Vec::new();
            let mut completion = None;
            for event in events {
                match event {
                    GgufEvent::Header {
                        version: version_value,
                        ..
                    } => version = Some(version_value),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete {
                        data_offset,
                        alignment,
                    } => {
                        completion = Some((data_offset, alignment));
                    }
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().expect("parser reports complete and clean");
                let parsed = ParsedGguf {
                    version,
                    tensor_count: tensors.len() as u64,
                    kv_count: metadata.len() as u64,
                    metadata,
                    tensors,
                    data_offset,
                    alignment,
                };
                return Some((parsed, file_len, file));
            }
        }

        assert!(
            prefix_len < (1 << 26),
            "gguf header/directory exceeded 64 MiB prefix budget"
        );
        prefix_len *= 2;
    }
}

/// Reads one named tensor's packed bytes off `file` via its validated
/// absolute byte range, or `None` if `name`/`ggml_type` doesn't match
/// what's actually in the file (a mixed-precision quant recipe like
/// `Q4_K_S` doesn't guarantee a given tensor lands at a given codec on
/// every quantizer version -- reported, not faked, same stance
/// `bench_q4k_matmul.rs` takes).
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
fn real_tensor_bytes(
    file: &mut std::fs::File,
    parsed: &proxima_gguf::pipe::ParsedGguf,
    file_len: u64,
    name: &str,
    expect_type: proxima_gguf::types::GgmlType,
) -> Option<(Vec<u8>, usize, usize)> {
    use std::io::{Read, Seek, SeekFrom};

    let tensor = parsed
        .tensors
        .iter()
        .find(|candidate| candidate.name == name)?;
    if tensor.ggml_type != expect_type {
        eprintln!(
            "real_tensor_bytes: {name} is {:?} in this file, not {expect_type:?} -- test skipped, not faked",
            tensor.ggml_type
        );
        return None;
    }
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;
    let range = parsed
        .tensor_data_range(tensor, file_len)
        .expect("tensor byte range within file bounds");
    let mut buf = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start))
        .expect("seek to tensor data");
    file.read_exact(&mut buf)
        .expect("read exact tensor byte range");
    Some((buf, in_dim, out_dim))
}

#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
const REAL_OPENCHAT_GGUF_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

/// [`matmul_q5k_q8k_f32`]/[`matmul_q5k_q8k_portable_f32`] agree with
/// [`matmul_q5k_f32`] (the dequantize-then-fold reference path) on the
/// SAME packed `Q5_K` bytes read directly out of the real
/// openchat-3.5-1210 GGUF file -- `blk.0.attn_v.weight`, one of the
/// two shapes this landing's task names. This is the correctness gate
/// principle 14 (the incumbent -- here, the already-tested
/// dequantize path -- wins on correctness) demands BEFORE any timing;
/// no timing is taken in this test at all.
#[cfg(feature = "q5k-int8-dot")]
#[test]
fn matmul_q5k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes() {
    let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.attn_v.weight",
        proxima_gguf::types::GgmlType::Q5_K,
    ) else {
        return;
    };

    let activation = random_vec(401, in_dim)
        .into_iter()
        .map(|value| value - 0.5)
        .collect::<Vec<f32>>();

    let expected = matmul_q5k_f32(&weight_bytes, out_dim, &activation)
        .expect("well-formed dequant reference matmul");
    let dispatched = matmul_q5k_q8k_f32(&weight_bytes, out_dim, &activation)
        .expect("well-formed packed int8 matmul");
    let portable = matmul_q5k_q8k_portable_f32(&weight_bytes, out_dim, &activation)
        .expect("well-formed portable matmul");

    assert_eq!(
        dispatched, portable,
        "attn_v: dispatched and portable packed-int8 arms diverged on real bytes"
    );

    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f64;
    for (&got, &want) in dispatched.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "packed int8 matmul row produced a non-finite value: {got}"
        );
        let diff = (got - want).abs();
        max_error = max_error.max(diff);
        sum_sq_error += f64::from(diff) * f64::from(diff);
    }
    let rms_error = (sum_sq_error / out_dim as f64).sqrt();
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative_max_error = max_error / max_magnitude;
    eprintln!(
        "attn_v (real Q5_K bytes) packed vs dequant-fold reference: max_error={max_error} \
         rms_error={rms_error} max_magnitude={max_magnitude} relative_max_error={relative_max_error}"
    );
    // Same band `matmul_q4k_q8k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`
    // uses for its own real-weight relative-error check: Q8_K
    // activation quantization is a second real lossy step neither
    // side of this comparison shares.
    assert!(
        relative_max_error < 0.01,
        "relative_max_error={relative_max_error} (max_error={max_error} over magnitude {max_magnitude}) \
         exceeds loose sanity bound"
    );
}

/// The same correctness gate as
/// [`matmul_q5k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes`],
/// at this landing's second named `Q5_K` shape -- `blk.0.ffn_down.weight`
/// (14336x4096).
#[cfg(feature = "q5k-int8-dot")]
#[test]
fn matmul_q5k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes_ffn_down() {
    let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.ffn_down.weight",
        proxima_gguf::types::GgmlType::Q5_K,
    ) else {
        return;
    };

    let activation = random_vec(402, in_dim)
        .into_iter()
        .map(|value| value - 0.5)
        .collect::<Vec<f32>>();

    let expected = matmul_q5k_f32(&weight_bytes, out_dim, &activation)
        .expect("well-formed dequant reference matmul");
    let dispatched = matmul_q5k_q8k_f32(&weight_bytes, out_dim, &activation)
        .expect("well-formed packed int8 matmul");
    let portable = matmul_q5k_q8k_portable_f32(&weight_bytes, out_dim, &activation)
        .expect("well-formed portable matmul");

    assert_eq!(
        dispatched, portable,
        "ffn_down: dispatched and portable packed-int8 arms diverged on real bytes"
    );

    let max_error = dispatched
        .iter()
        .zip(expected.iter())
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative_max_error = max_error / max_magnitude;
    eprintln!(
        "ffn_down (real Q5_K bytes) packed vs dequant-fold reference: relative_max_error={relative_max_error}"
    );
    assert!(
        relative_max_error < 0.01,
        "relative_max_error={relative_max_error} exceeds loose sanity bound"
    );
}

/// [`matmul_q6k_q8k_f32`]/[`matmul_q6k_q8k_portable_f32`] agree with
/// [`matmul_q6k_f32`] on the SAME packed `Q6_K` bytes read directly out
/// of the real openchat-3.5-1210 GGUF file -- `output.weight`
/// (4096x32002), this landing's named `Q6_K` shape.
#[cfg(feature = "q6k-int8-dot")]
#[test]
fn matmul_q6k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes() {
    let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "output.weight",
        proxima_gguf::types::GgmlType::Q6_K,
    ) else {
        return;
    };

    let activation = random_vec(403, in_dim)
        .into_iter()
        .map(|value| value - 0.5)
        .collect::<Vec<f32>>();

    let expected = matmul_q6k_f32(&weight_bytes, out_dim, &activation)
        .expect("well-formed dequant reference matmul");
    let dispatched = matmul_q6k_q8k_f32(&weight_bytes, out_dim, &activation)
        .expect("well-formed packed int8 matmul");
    let portable = matmul_q6k_q8k_portable_f32(&weight_bytes, out_dim, &activation)
        .expect("well-formed portable matmul");

    assert_eq!(
        dispatched, portable,
        "output.weight: dispatched and portable packed-int8 arms diverged on real bytes"
    );

    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f64;
    for (&got, &want) in dispatched.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "packed int8 matmul row produced a non-finite value: {got}"
        );
        let diff = (got - want).abs();
        max_error = max_error.max(diff);
        sum_sq_error += f64::from(diff) * f64::from(diff);
    }
    let rms_error = (sum_sq_error / out_dim as f64).sqrt();
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative_max_error = max_error / max_magnitude;
    eprintln!(
        "output.weight (real Q6_K bytes) packed vs dequant-fold reference: max_error={max_error} \
         rms_error={rms_error} max_magnitude={max_magnitude} relative_max_error={relative_max_error}"
    );
    assert!(
        relative_max_error < 0.01,
        "relative_max_error={relative_max_error} (max_error={max_error} over magnitude {max_magnitude}) \
         exceeds loose sanity bound"
    );
}

/// The position-folding defect this landing fixes: before it,
/// `run_reduce_quantized` called [`matmul_q4k_q8k_f32`] once per
/// sequence position, re-streaming the entire `Q4_K` weight matrix
/// `leading_total` times. [`matmul_q4k_q8k_f32_impl`] now takes
/// `leading_total` directly and folds every position's dot into one
/// pass over the weight rows. This test is the one every *existing*
/// test (all single-position) cannot catch: run the wide path at
/// `leading_total = 3` on real packed `Q4_K` bytes from the checkpoint,
/// and check it against `leading_total` separate single-position calls
/// through the narrow (already-tested) public entry point --
/// `assert_eq!`, not a tolerance, since folding changes neither the
/// per-element arithmetic nor its order (still one row-dot over the
/// same `k` elements per `(row, position)` pair), only which weight
/// bytes get re-read. Also pins the wide output's own row-major
/// `[row][position]` layout (`matmul_rows_threaded`'s natural shape,
/// weight row as the parallel axis) against the narrow path's
/// position-major results reassembled the same way -- a silent
/// transpose would show up as a mismatch here, not a tolerance miss.
#[cfg(feature = "q4k-int8-dot")]
#[test]
fn matmul_q4k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes() {
    let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.attn_q.weight",
        proxima_gguf::types::GgmlType::Q4_K,
    ) else {
        return;
    };

    let leading_total = 3usize;
    let activation: Vec<f32> = (0..leading_total)
        .flat_map(|position| {
            random_vec(500 + position as u64, in_dim)
                .into_iter()
                .map(|value| value - 0.5)
        })
        .collect();

    let wide = matmul_q4k_q8k_f32_impl(&weight_bytes, out_dim, &activation, leading_total, None)
        .expect("wide fold call");
    assert_eq!(
        wide.len(),
        out_dim * leading_total,
        "wide output is not row-major [row][position]"
    );

    for position in 0..leading_total {
        let activation_row = &activation[position * in_dim..(position + 1) * in_dim];
        let narrow = matmul_q4k_q8k_f32(&weight_bytes, out_dim, activation_row)
            .expect("narrow per-position call");
        for row in 0..out_dim {
            assert_eq!(
                wide[row * leading_total + position],
                narrow[row],
                "row {row} position {position}: folded and per-position paths diverged"
            );
        }
    }
}

/// [`matmul_q4k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes`]'s
/// exact mechanism, ported to `Q5_K`: [`matmul_q5k_q8k_f32_impl`] now
/// takes `leading_total` and folds every position's dot into one pass
/// over the weight rows, in place of `run_reduce_quantized`'s old
/// per-position loop re-streaming the whole `Q5_K` weight matrix
/// `leading_total` times. Run the wide path at `leading_total = 3` on
/// real packed `Q5_K` bytes (`blk.0.ffn_down.weight`, this crate's named
/// `Q5_K` shape) and check it against `leading_total` separate
/// single-position calls through the narrow (already-tested) public
/// entry point -- `assert_eq!`, not a tolerance, for the same reason the
/// `Q4_K` test uses one: folding changes neither the per-element
/// arithmetic nor its order, only which weight bytes get re-read.
#[cfg(feature = "q5k-int8-dot")]
#[test]
fn matmul_q5k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes() {
    let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.ffn_down.weight",
        proxima_gguf::types::GgmlType::Q5_K,
    ) else {
        return;
    };

    let leading_total = 3usize;
    let activation: Vec<f32> = (0..leading_total)
        .flat_map(|position| {
            random_vec(600 + position as u64, in_dim)
                .into_iter()
                .map(|value| value - 0.5)
        })
        .collect();

    let wide = matmul_q5k_q8k_f32_impl(&weight_bytes, out_dim, &activation, leading_total, None)
        .expect("wide fold call");
    assert_eq!(
        wide.len(),
        out_dim * leading_total,
        "wide output is not row-major [row][position]"
    );

    for position in 0..leading_total {
        let activation_row = &activation[position * in_dim..(position + 1) * in_dim];
        let narrow = matmul_q5k_q8k_f32(&weight_bytes, out_dim, activation_row)
            .expect("narrow per-position call");
        for row in 0..out_dim {
            assert_eq!(
                wide[row * leading_total + position],
                narrow[row],
                "row {row} position {position}: folded and per-position paths diverged"
            );
        }
    }
}

/// [`matmul_q4k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes`]'s
/// exact mechanism, ported to `Q6_K`: [`matmul_q6k_q8k_f32_impl`] now
/// takes `leading_total` and folds every position's dot into one pass
/// over the weight rows. Run the wide path at `leading_total = 3` on
/// real packed `Q6_K` bytes (`output.weight`, this crate's named `Q6_K`
/// shape) and check it against `leading_total` separate single-position
/// calls through the narrow (already-tested) public entry point --
/// `assert_eq!`, not a tolerance, same reasoning as the `Q4_K`/`Q5_K`
/// tests.
#[cfg(feature = "q6k-int8-dot")]
#[test]
fn matmul_q6k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes() {
    let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "output.weight",
        proxima_gguf::types::GgmlType::Q6_K,
    ) else {
        return;
    };

    let leading_total = 3usize;
    let activation: Vec<f32> = (0..leading_total)
        .flat_map(|position| {
            random_vec(700 + position as u64, in_dim)
                .into_iter()
                .map(|value| value - 0.5)
        })
        .collect();

    let wide = matmul_q6k_q8k_f32_impl(&weight_bytes, out_dim, &activation, leading_total, None)
        .expect("wide fold call");
    assert_eq!(
        wide.len(),
        out_dim * leading_total,
        "wide output is not row-major [row][position]"
    );

    for position in 0..leading_total {
        let activation_row = &activation[position * in_dim..(position + 1) * in_dim];
        let narrow = matmul_q6k_q8k_f32(&weight_bytes, out_dim, activation_row)
            .expect("narrow per-position call");
        for row in 0..out_dim {
            assert_eq!(
                wide[row * leading_total + position],
                narrow[row],
                "row {row} position {position}: folded and per-position paths diverged"
            );
        }
    }
}

/// [`dot_q5k_q8k_block_neon_dotprod`]'s whole justification: it must
/// be an ACCELERATION of [`dot_q5k_q8k_block_scalar`], not a different
/// mechanism -- same bit-exactness argument
/// `matmul_q4k_q8k_f32_agrees_bit_exact_with_the_portable_arm` makes
/// for `Q4_K` (every intermediate is integer until the final `f32`
/// multiply, so both arms must match EXACTLY, not merely closely), on
/// synthetic multi-row, multi-block data (not real-file bytes -- this
/// test's job is arm-vs-arm agreement, not real-weight correctness,
/// which the two tests above already cover).
#[cfg(feature = "q5k-int8-dot")]
#[test]
fn matmul_q5k_q8k_f32_agrees_bit_exact_with_the_portable_arm() {
    use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, quantize};

    let rows = 4;
    let blocks_per_row = 5;
    let k = QK_K * blocks_per_row;

    let activation: Vec<f32> = random_vec(23, k)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(29, rows * k)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of QK_K by construction");
    }

    let dispatched = matmul_q5k_q8k_f32(&weight_blocks, rows, &activation)
        .expect("well-formed dispatched matmul");
    let portable = matmul_q5k_q8k_portable_f32(&weight_blocks, rows, &activation)
        .expect("well-formed portable matmul");

    assert_eq!(
        dispatched, portable,
        "dispatched and portable arms diverged -- not merely an acceleration"
    );
}

/// [`dot_q6k_q8k_block_neon_dotprod`]'s equivalent bit-exactness proof.
#[cfg(feature = "q6k-int8-dot")]
#[test]
fn matmul_q6k_q8k_f32_agrees_bit_exact_with_the_portable_arm() {
    use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, quantize};

    let rows = 4;
    let blocks_per_row = 5;
    let k = QK_K * blocks_per_row;

    let activation: Vec<f32> = random_vec(31, k)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(37, rows * k)
        .into_iter()
        .map(|value| value * 6.0 - 3.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of QK_K by construction");
    }

    let dispatched = matmul_q6k_q8k_f32(&weight_blocks, rows, &activation)
        .expect("well-formed dispatched matmul");
    let portable = matmul_q6k_q8k_portable_f32(&weight_blocks, rows, &activation)
        .expect("well-formed portable matmul");

    assert_eq!(
        dispatched, portable,
        "dispatched and portable arms diverged -- not merely an acceleration"
    );
}

/// [`QuantizedBlock::Q5K`] routes through [`evaluate_quantized`] end to
/// end -- the same shape
/// [`evaluate_quantized_matches_dequantize_then_evaluate_within_a_measured_tolerance`]
/// proves for `Q4K`, one variant over: a `Reduce(Elementwise(Multiply))`
/// matmul node bound to packed `Q5_K` bytes must agree with binding the
/// SAME bytes dequantized to plain `f32`.
#[cfg(feature = "q5k-int8-dot")]
#[test]
fn evaluate_quantized_routes_q5k_block_and_matches_dequantize_then_evaluate() {
    use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows: u32 = 6;
    let blocks_per_row = 3;
    let k = QK_K as u32 * blocks_per_row as u32;

    let activation = random_vec(43, k as usize);
    let weight_f32: Vec<f32> = random_vec(47, rows as usize * k as usize);

    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of QK_K by construction");
    }

    let (program, sum) = quantized_matmul_program(rows, k);
    let blocks = [
        QuantizedBlock::Q5K(&weight_blocks),
        QuantizedBlock::Float32(&activation),
    ];
    let quantized_result = evaluate_quantized(&program, &[], &blocks, &[sum])
        .expect("q5_k-quantized matmul evaluates");

    let mut dequantized_weight = vec![0.0f32; rows as usize * k as usize];
    for (row_blocks, row_f32) in weight_blocks
        .chunks_exact(blocks_per_row * BLOCK_BYTES)
        .zip(dequantized_weight.chunks_exact_mut(k as usize))
    {
        dequantize(row_blocks, row_f32).expect("row_blocks is a whole number of q5_k super-blocks");
    }

    let (f32_program, f32_sum) = matmul_program(rows, k, 1, false);
    let f32_blocks: [&[f32]; 2] = [&dequantized_weight, &activation];
    let f32_result = evaluate(&f32_program, &[], &f32_blocks, &[f32_sum])
        .expect("dequantized f32 matmul evaluates");

    let actual = quantized_result.root();
    let expected = f32_result.root();
    assert_eq!(actual.len(), rows as usize);
    assert_eq!(actual.len(), expected.len());

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative_max_diff = max_diff / max_magnitude;
    eprintln!(
        "evaluate_quantized (Q5K) vs dequantize-then-evaluate: relative_max_diff={relative_max_diff}"
    );
    assert!(
        relative_max_diff < 0.01,
        "relative_max_diff={relative_max_diff} (max_diff={max_diff} over magnitude {max_magnitude}) \
         exceeds loose sanity bound"
    );
}

/// The union of every dedicated codec test above, parameterized rather
/// than copy-pasted: one `[rows, k] x [k, 1]` matmul, one seeded weight
/// and activation, run through whichever evaluator that codec actually
/// reaches -- plain [`evaluate`] for `float32` (packing it as a
/// [`QuantizedBlock::Float32`] *weight* is rejected outright by
/// `run_reduce_quantized`, see the `shape_error()` arm a few hundred
/// lines up, so `float32` is not an `evaluate_quantized` cell at all),
/// [`evaluate_quantized`] for the four packed codecs -- and compared
/// against the same `f32` reference every one of those dedicated tests
/// already computes independently. A codec dropped from this list is a
/// missing `#[case::...]` line, not a missing whole function.
///
/// Tolerance is RELATIVE to the reference's own magnitude, never a flat
/// absolute epsilon: `float32` is an exact self-consistency check (same
/// bytes, same evaluator, twice), while every packed codec's activation
/// additionally folds through a lossy int8 quantization step on the CPU
/// path (`matmul_q4k_q8k_f32`-family), so a single absolute bound across
/// all five cells would be either vacuous for `float32` or spuriously
/// red for the packed codecs.
#[proxima::test]
#[case::float32("float32", 1e-6)]
#[case::q4_k("q4_k", 0.01)]
#[case::q5_k("q5_k", 0.01)]
#[case::q3_k("q3_k", 0.01)]
#[case::q6_k("q6_k", 0.01)]
#[case::q8_0("q8_0", 0.01)]
// q4_0 is the coarsest codec under test here -- one scale per block, no
// k-quant sub-block min/scale pair -- so it alone needed a wider bound
// once `test_support::Lcg::next_unit`'s own range-halving bug (see that
// function's doc) was fixed: the old bug never generated near-zero
// weights, and near-zero values are where a single-scale codec's
// relative error is largest. `0.0104` measured against the corrected,
// zero-crossing input; `0.012` leaves headroom without loosening the
// other four codecs' tighter, still-met `0.01`.
#[case::q4_0("q4_0", 0.012)]
async fn evaluate_quantized_matmul_matches_dequantized_reference_across_every_codec(
    #[case] codec: &str,
    #[case] tolerance: f32,
) {
    let rows: u32 = 5;
    let k: u32 = 768;

    let weight_f32: Vec<f32> = random_vec(17, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let activation: Vec<f32> = random_vec(13, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let (f32_program, f32_sum) = matmul_program(rows, k, 1, false);
    let reference = evaluate(&f32_program, &[], &[&weight_f32, &activation], &[f32_sum])
        .expect("f32 reference matmul evaluates");

    let actual: Vec<f32> = match codec {
        "float32" => reference.root().to_vec(),
        "q4_k" => {
            use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};
            let blocks_per_row = k as usize / QK_K;
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
            {
                quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK_K by construction");
            }
            let (program, sum) = quantized_matmul_program(rows, k);
            let blocks = [
                QuantizedBlock::Q4K(&weight_blocks),
                QuantizedBlock::Float32(&activation),
            ];
            evaluate_quantized(&program, &[], &blocks, &[sum])
                .expect("q4_k-quantized matmul evaluates")
                .root()
                .to_vec()
        }
        "q5_k" => {
            use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, quantize};
            let blocks_per_row = k as usize / QK_K;
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
            {
                quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK_K by construction");
            }
            let (program, sum) = quantized_matmul_program(rows, k);
            let blocks = [
                QuantizedBlock::Q5K(&weight_blocks),
                QuantizedBlock::Float32(&activation),
            ];
            evaluate_quantized(&program, &[], &blocks, &[sum])
                .expect("q5_k-quantized matmul evaluates")
                .root()
                .to_vec()
        }
        "q3_k" => {
            use proxima_gguf::quant::q3_k::{BLOCK_BYTES, QK_K, quantize};
            let blocks_per_row = k as usize / QK_K;
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
            {
                quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK_K by construction");
            }
            let (program, sum) = quantized_matmul_program(rows, k);
            let blocks = [
                QuantizedBlock::Q3K(&weight_blocks),
                QuantizedBlock::Float32(&activation),
            ];
            evaluate_quantized(&program, &[], &blocks, &[sum])
                .expect("q3_k-quantized matmul evaluates")
                .root()
                .to_vec()
        }
        "q6_k" => {
            use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, quantize};
            let blocks_per_row = k as usize / QK_K;
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
            {
                quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK_K by construction");
            }
            let (program, sum) = quantized_matmul_program(rows, k);
            let blocks = [
                QuantizedBlock::Q6K(&weight_blocks),
                QuantizedBlock::Float32(&activation),
            ];
            evaluate_quantized(&program, &[], &blocks, &[sum])
                .expect("q6_k-quantized matmul evaluates")
                .root()
                .to_vec()
        }
        "q8_0" => {
            use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, quantize};
            let blocks_per_row = k as usize / QK8_0;
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
            {
                quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK8_0 by construction");
            }
            let (program, sum) = quantized_matmul_program(rows, k);
            let blocks = [
                QuantizedBlock::Q8_0(&weight_blocks),
                QuantizedBlock::Float32(&activation),
            ];
            evaluate_quantized(&program, &[], &blocks, &[sum])
                .expect("q8_0-quantized matmul evaluates")
                .root()
                .to_vec()
        }
        "q4_0" => {
            use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, quantize};
            let blocks_per_row = k as usize / QK4_0;
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
            {
                quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK4_0 by construction");
            }
            let (program, sum) = quantized_matmul_program(rows, k);
            let blocks = [
                QuantizedBlock::Q4_0(&weight_blocks),
                QuantizedBlock::Float32(&activation),
            ];
            evaluate_quantized(&program, &[], &blocks, &[sum])
                .expect("q4_0-quantized matmul evaluates")
                .root()
                .to_vec()
        }
        other => panic!("unhandled codec case in this matrix: {other}"),
    };

    let expected = reference.root();
    assert_eq!(
        actual.len(),
        rows as usize,
        "degenerate gate: no outputs compared"
    );
    assert_eq!(actual.len(), expected.len());

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude;
    eprintln!(
        "{codec}: relative={relative} tolerance={tolerance} (max_diff={max_diff} max_magnitude={max_magnitude})"
    );
    assert!(
        relative <= tolerance,
        "{codec}: relative diff {relative} exceeds tolerance {tolerance} -- max_diff={max_diff} \
         max_magnitude={max_magnitude}"
    );
}

/// [`dot_q3k_f32`] against one REAL `Q3_K` row from a real requantized
/// Q3_K_M checkpoint (`PROXIMA_Q3K_GGUF`, host-local and opportunistic --
/// `#[ignore]`d like every other real-checkpoint test in this workspace,
/// e.g. `proxima-gguf`'s own `q3_k_real_dequantize_matches_llama_cpp_gguf_py_oracle`)
/// against a reference built by dequantizing the SAME bytes
/// (`proxima_gguf::quant::q3_k::dequantize`) and folding a plain `f32`
/// dot product by hand -- the incumbent shape every other codec's own
/// `dot_q{4,5,6}k_f32` parity already holds. Real weight bytes off disk,
/// synthetic (but deterministic, non-degenerate) activation: the codec's
/// correctness depends only on the weight bytes it decodes, never on
/// what the activation happens to be, so a real activation buys nothing
/// a seeded `Lcg` vector does not already cover -- reading only the
/// header and this one row's own byte range, never the multi-gigabyte
/// payload behind it.
#[test]
#[ignore = "depends on a host-local real gguf checkpoint (PROXIMA_Q3K_GGUF)"]
fn dot_q3k_f32_matches_dequantize_then_f32_matmul_on_a_real_checkpoint_row() {
    use std::io::{Read, Seek, SeekFrom};

    let Ok(gguf_path) = std::env::var("PROXIMA_Q3K_GGUF") else {
        eprintln!("skipping: PROXIMA_Q3K_GGUF not set");
        return;
    };
    let path = std::path::Path::new(&gguf_path);
    if !path.exists() {
        eprintln!("skipping: gguf ({path:?}) missing on this host");
        return;
    }

    let mut file = std::fs::File::open(path).expect("open real gguf checkpoint");
    let file_len = file.metadata().expect("stat real gguf checkpoint").len();
    let mut header_buf = vec![0u8; 0];
    let parsed = 'parse: {
        for cap in [4usize << 20, 16 << 20, 64 << 20, 128 << 20] {
            header_buf.resize(cap, 0);
            file.seek(SeekFrom::Start(0)).expect("seek to file start");
            let read = file.read(&mut header_buf).expect("read gguf header region");
            header_buf.truncate(read);
            if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header_buf) {
                break 'parse parsed;
            }
        }
        panic!("gguf metadata region did not fit in 128 MiB");
    };

    let tensor_name = "blk.0.ffn_up.weight";
    let tensor = parsed
        .tensors
        .iter()
        .find(|candidate| candidate.name == tensor_name)
        .unwrap_or_else(|| panic!("{tensor_name} not present in real checkpoint"));
    assert_eq!(
        tensor.ggml_type,
        proxima_gguf::types::GgmlType::Q3_K,
        "{tensor_name} must be Q3_K in this Q3_K_M checkpoint"
    );

    let row_elements = tensor.dims[0] as usize;
    let row_bytes =
        (row_elements / proxima_gguf::quant::q3_k::QK_K) * proxima_gguf::quant::q3_k::BLOCK_BYTES;
    let full_range = parsed
        .tensor_data_range(tensor, file_len)
        .expect("tensor data range within real checkpoint");
    let row_range = full_range.start..full_range.start + row_bytes as u64;
    let mut row_packed = vec![0u8; row_bytes];
    file.seek(SeekFrom::Start(row_range.start))
        .expect("seek to tensor row start");
    file.read_exact(&mut row_packed)
        .expect("read exact tensor row bytes");

    let mut row_f32 = vec![0.0f32; row_elements];
    proxima_gguf::quant::q3_k::dequantize(&row_packed, &mut row_f32).expect("decode real Q3_K row");

    let mut lcg = Lcg(99);
    let activation: Vec<f32> = (0..row_elements)
        .map(|_| lcg.next_unit() * 2.0 - 1.0)
        .collect();

    let expected: f32 = row_f32
        .iter()
        .zip(activation.iter())
        .map(|(&weight, &value)| weight * value)
        .sum();
    let actual = dot_q3k_f32(&row_packed, &activation).expect("well-formed real q3_k row");

    // A naive linear `sum()` reference and `dot_q3k_f32`'s own 8-lane
    // `dot_fold_fused_multiply_add` fold legitimately disagree at
    // roughly the `f32` ULP-accumulated noise floor on a real,
    // non-degenerate multi-hundred-element row purely from
    // reassociation -- measured here at ~1e-5 relative, not a codec
    // bug: both numbers decode the identical bytes through the
    // identical `q3_k` dequantization formula, they only sum the 256
    // products in a different order. `5e-5` leaves headroom over that
    // measured floor while staying two orders of magnitude tighter than
    // this crate's own quantization-error parity bound (`0.01`,
    // `evaluate_quantized_matmul_matches_dequantized_reference_across_every_codec`),
    // which the `Q3_K` codec's own lossy quantization step (not this
    // test) is responsible for.
    let relative = (actual - expected).abs() / expected.abs();
    eprintln!(
        "dot_q3k_f32 real-row parity: actual={actual} expected={expected} relative={relative}"
    );
    assert!(
        relative <= 5e-5,
        "relative diff {relative} exceeds 5e-5 (actual={actual} expected={expected})"
    );
}

/// Fixed-topology (structured) sparsity needs no gate at all: the
/// nonzero pattern here is two disjoint 2x2 blocks fixed at graph-build
/// time, and each block is a plain, unshifted `IndexMap::Affine`
/// projection -- never `IndexMap::Computed`. The zero off-diagonal
/// blocks of the dense 4x4 equivalent are never built as ops at all, so
/// this costs 2 * (2*2) = 8 multiply-adds against the 16 a dense matmul
/// would spend, and nothing here is data-dependent, so
/// `shape.rs:166`'s scatter gate never sees this program.
#[test]
fn a_static_block_sparse_matmul_needs_no_data_dependent_map() {
    let mut program = Vec::new();
    let x_block0 = f32_block(&mut program, &[Extent::Static(2)]);
    let x_block1 = f32_block(&mut program, &[Extent::Static(2)]);
    let weight_block0 = f32_block(&mut program, &[Extent::Static(2), Extent::Static(2)]);
    let weight_block1 = f32_block(&mut program, &[Extent::Static(2), Extent::Static(2)]);

    let block_output = |program: &mut Vec<Op>, weight: NodeId, x: NodeId| {
        let product = append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (x, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );
        append(
            program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: None,
            }),
        )
    };

    let block0_output = block_output(&mut program, weight_block0, x_block0);
    let block1_output = block_output(&mut program, weight_block1, x_block1);

    let x0 = [1.0f32, 2.0];
    let x1 = [3.0f32, 4.0];
    let weight0 = [2.0f32, 1.0, 1.0, 2.0];
    let weight1 = [1.0f32, 1.0, 1.0, -1.0];
    let evaluated = evaluate(
        &program,
        &[],
        &[&x0, &x1, &weight0, &weight1],
        &[block0_output, block1_output],
    )
    .expect("static block-sparse matmul lowers and evaluates");

    let (block0, _) = evaluated.get(block0_output).expect("block0 output present");
    let (block1, _) = evaluated.get(block1_output).expect("block1 output present");
    assert_eq!(block0, &[4.0, 5.0], "weight0 @ (x0, x1)");
    assert_eq!(block1, &[7.0, -1.0], "weight1 @ (x2, x3)");
}

/// The test above's own `weight0`/`weight1` are both symmetric matrices
/// (`[[2,1],[1,2]]`, `[[1,1],[1,-1]]`), so a row/col axis-order bug in
/// `block_output`'s `projection(2, &[0, 1])` weight read -- transposing
/// which axis is "row" (kept in `out_map`) and which is "col" (reduced)
/// -- would still land on the exact same numbers and pass silently: the
/// same shape of blind spot `causal_conv1d`'s `embedding=1` fixture had,
/// here from symmetric data rather than a degenerate extent. This uses
/// deliberately asymmetric `2x2` blocks (`weight0 @ x0` and its
/// transpose disagree) so that exact bug is observable.
#[test]
fn a_static_block_sparse_matmul_catches_a_transposed_block_weight() {
    let mut program = Vec::new();
    let x_block0 = f32_block(&mut program, &[Extent::Static(2)]);
    let x_block1 = f32_block(&mut program, &[Extent::Static(2)]);
    let weight_block0 = f32_block(&mut program, &[Extent::Static(2), Extent::Static(2)]);
    let weight_block1 = f32_block(&mut program, &[Extent::Static(2), Extent::Static(2)]);

    let block_output = |program: &mut Vec<Op>, weight: NodeId, x: NodeId| {
        let product = append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (x, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );
        append(
            program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: None,
            }),
        )
    };

    let block0_output = block_output(&mut program, weight_block0, x_block0);
    let block1_output = block_output(&mut program, weight_block1, x_block1);

    // weight0 = [[1, 2], [3, 4]], weight1 = [[5, 6], [7, 8]] -- neither
    // symmetric, so weight @ x and weight^T @ x disagree below.
    let x0 = [1.0f32, 0.0];
    let x1 = [0.0f32, 1.0];
    let weight0 = [1.0f32, 2.0, 3.0, 4.0];
    let weight1 = [5.0f32, 6.0, 7.0, 8.0];
    let evaluated = evaluate(
        &program,
        &[],
        &[&x0, &x1, &weight0, &weight1],
        &[block0_output, block1_output],
    )
    .expect("static block-sparse matmul lowers and evaluates");

    let (block0, _) = evaluated.get(block0_output).expect("block0 output present");
    let (block1, _) = evaluated.get(block1_output).expect("block1 output present");
    // weight0 @ x0 = [1*1+2*0, 3*1+4*0] = [1, 3]; the transposed
    // reading would give [1*1+3*0, 2*1+4*0] = [1, 2] instead.
    assert_eq!(block0, &[1.0, 3.0], "weight0 @ x0, not weight0^T @ x0");
    // weight1 @ x1 = [5*0+6*1, 7*0+8*1] = [6, 8]; transposed would give
    // [6, 7] instead.
    assert_eq!(block1, &[6.0, 8.0], "weight1 @ x1, not weight1^T @ x1");
}

/// The adjoint of a gather -- and gradient accumulation into a
/// fixed-shape destination -- is expressible today with the same three
/// generators the crate's own causal-mask idiom already composes
/// (`Iota` plus `Equal` builds the selector; see `op.rs`'s `Iota` doc),
/// never `IndexMap::Computed` as an `out_map`. The destination extent
/// (`3` here) comes from `Iota`'s own `extent` field -- the same
/// externally-supplied-extent mechanism `Op::Input`'s leaf shape already
/// uses -- not from `shape.rs`'s iteration-space unification, which is
/// why this needs no change to the `is_data_dependent` gate at
/// `shape.rs:166`. Source rows 0 and 2 both target destination row 0,
/// which is the whole difference between a scatter-add and a
/// scatter-write: `Reduce`'s own `body: Add` sums both contributions
/// instead of one clobbering the other.
#[test]
fn scatter_add_into_a_known_destination_via_mask_composition() {
    let mut program = Vec::new();
    let destination_positions = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(3),
        },
    );
    let indices = f32_block(&mut program, &[Extent::Static(4)]);
    let source = f32_block(&mut program, &[Extent::Static(4)]);

    let mask = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Equal,
            operands: alloc::vec![
                (
                    destination_positions,
                    IndexMap::Affine(map::projection(2, &[0]))
                ),
                (indices, IndexMap::Affine(map::projection(2, &[1]))),
            ],
            name: None,
        },
    );
    let masked_source = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (mask, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (source, IndexMap::Affine(map::projection(2, &[1]))),
            ],
            name: None,
        },
    );
    let scattered = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: masked_source,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("scatter_add".into()),
        }),
    );

    let index_values = [0.0f32, 2.0, 0.0, 1.0];
    let source_values = [10.0f32, 20.0, 30.0, 40.0];
    let evaluated = evaluate(
        &program,
        &[],
        &[&index_values, &source_values],
        &[scattered],
    )
    .expect("scatter-add composed from Iota+Equal+Multiply+Reduce lowers and evaluates");

    assert_eq!(
        evaluated.root(),
        &[40.0, 40.0, 20.0],
        "dest[0]=src[0]+src[2] (collision), dest[1]=src[3], dest[2]=src[1]"
    );
}

/// The CPU decode route's `Q2_K` matvec ([`matmul_q2k_f32`], reached
/// from [`run_reduce_quantized`] whenever a bound weight is
/// [`QuantizedBlock::Q2K`]) against the naive reference: dequantize the
/// same packed bytes to `f32` via
/// [`proxima_gguf::quant::q2_k::dequantize`], then compute the matvec
/// with plain scalar dot products. Both paths decode the identical
/// packed bytes, so any gap here is a kernel bug in
/// [`dot_q2k_f32`]'s fold or block/activation indexing, never
/// quantization error -- the `1e-5` bound is a float-summation-order
/// tolerance, not a quantization tolerance.
#[test]
fn matmul_q2k_f32_matches_naive_dequantize_then_dot_matvec() {
    let rows = 3usize;
    let cols = proxima_gguf::quant::q2_k::QK_K; // one super-block per row
    let mut rng = Lcg(0x51EA_2A1E);
    let weights_f32: Vec<f32> = (0..rows * cols).map(|_| rng.next_unit() * 3.0).collect();
    let activation: Vec<f32> = (0..cols).map(|_| rng.next_unit()).collect();

    let mut packed = vec![0u8; proxima_gguf::quant::q2_k::BLOCK_BYTES * rows];
    proxima_gguf::quant::q2_k::quantize(&weights_f32, &mut packed)
        .expect("rows*cols is a whole number of q2_k super-blocks");

    let mut dequantized = vec![0.0f32; rows * cols];
    proxima_gguf::quant::q2_k::dequantize(&packed, &mut dequantized)
        .expect("packed bytes are a whole number of q2_k super-blocks");

    let expected: Vec<f32> = dequantized
        .chunks_exact(cols)
        .map(|row| {
            row.iter()
                .zip(activation.iter())
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect();

    let got =
        matmul_q2k_f32(&packed, rows, &activation).expect("well-formed q2_k weight matrix matvec");

    assert_eq!(got.len(), rows);
    for (row, (&got_value, &expected_value)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (got_value - expected_value).abs();
        assert!(
            diff < 1e-5,
            "row {row}: matmul_q2k_f32={got_value} vs naive dequantize-then-dot={expected_value}, diff={diff}"
        );
    }
}
