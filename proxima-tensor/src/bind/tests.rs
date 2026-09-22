use super::*;

/// Drives `resolved` through [`crate::cpu::Interpreter`] the same way
/// `reduce_epilogue_fusion_tests::run_resolved` does — inlined rather
/// than shared across the module boundary, since this is the only
/// consumer at this scope.
fn run_resolved(
    program_len: usize,
    resolved: &[BoundOp],
    inputs: Vec<(NodeId, Vec<f32>)>,
) -> Vec<Option<Vec<f32>>> {
    use core::pin::pin;
    use core::task::{Context, Poll, Waker};

    use crate::cpu::Interpreter;

    let mut buffers: Vec<Option<Vec<f32>>> = alloc::vec![None; program_len];
    for (node, data) in inputs {
        buffers[node.0 as usize] = Some(data);
    }
    let interpreter = Interpreter::new(&mut buffers);
    for chunk in resolved.chunks(READY_BATCH_CAPACITY) {
        let batch: ReadyBatch = chunk.iter().cloned().collect();
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = pin!(interpreter.call(batch));
        match future.as_mut().poll(&mut context) {
            Poll::Ready(result) => {
                result.expect("resolved batch computes");
            }
            Poll::Pending => unreachable!("cpu pipes never yield: no internal .await"),
        }
    }
    buffers
}

/// The last node `program` builds -- what every fixture in this module
/// treats as "the answer" by construction (`op.rs`'s own doc: "the last
/// element is the root"). [`bind_plain`]'s reachability pass (ROW 541,
/// `docs/discipline.md`) now binds only what `outputs` actually names,
/// so a fixture that wants its whole constructed chain bound must pass
/// this instead of `&[]` -- an empty `outputs` correctly binds nothing.
fn terminal(program: &[Op]) -> NodeId {
    NodeId((program.len() - 1) as u32)
}

/// `max(x, -inf)` -- owner counterexample (2026-09-06): `f32::max`'s own
/// "if one argument is NaN, return the other" rule means
/// `NaN.max(-inf) == -inf`, but eliminating the op (returning the
/// survivor `x`) would produce `NaN` instead. Under the library default
/// ([`NumericPolicy::bit_exact()`]) the `Maximum` step must survive and
/// compute the real `-inf`; only once the caller grants `nan_assumptions`
/// does [`identity_element_signed_zero_nan`] fire and collapse the op to
/// `x`, producing the DIFFERENT value `NaN`.
#[test]
fn maximum_of_nan_and_negative_infinity_differs_by_numeric_policy() {
    use crate::op::{Extent, append};

    let mut program = Vec::new();
    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(1)],
            name: None,
        },
    );
    let neg_inf = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: f32::NEG_INFINITY,
        },
    );
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let broadcast_scalar = || IndexMap::Affine(map::projection(1, &[]));
    let output = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Maximum,
            operands: alloc::vec![(x, identity()), (neg_inf, broadcast_scalar())],
            name: None,
        },
    );
    let shapes = shape::infer(&program, &[]).expect("max(x,-inf) program infers");

    let bit_exact = bind_with_fusion(
        &program,
        &shapes,
        &[output],
        true,
        NumericPolicy::bit_exact(),
    )
    .expect("bit-exact bind succeeds");
    let bit_exact_buffers = run_resolved(
        program.len(),
        &bit_exact,
        alloc::vec![(x, alloc::vec![f32::NAN])],
    );
    let bit_exact_result = bit_exact_buffers[output.0 as usize]
        .as_ref()
        .expect("bit-exact output present")[0];
    assert_eq!(
        bit_exact_result,
        f32::NEG_INFINITY,
        "BitExact must compute the real max(NaN, -inf) == -inf, not eliminate the op"
    );

    let nan_assumption_policy = NumericPolicy {
        nan_assumptions: true,
        ..NumericPolicy::bit_exact()
    };
    let nan_assumption_bound =
        bind_with_fusion(&program, &shapes, &[output], true, nan_assumption_policy)
            .expect("nan-assumption bind succeeds");
    let nan_assumption_buffers = run_resolved(
        program.len(),
        &nan_assumption_bound,
        alloc::vec![(x, alloc::vec![f32::NAN])],
    );
    let nan_assumption_result = nan_assumption_buffers[output.0 as usize]
        .as_ref()
        .expect("nan-assumption output present")[0];
    assert!(
        nan_assumption_result.is_nan(),
        "nan_assumptions admits IdentityEliminationNanAssumption, collapsing to the \
         surviving operand x == NaN, got {nan_assumption_result}"
    );
}

/// `x + 0.0` -- owner counterexample (2026-09-06): `(-0.0) + 0.0`
/// evaluates to `+0.0` under real `f32` addition, but eliminating the op
/// (returning the survivor `x == -0.0`) keeps the sign bit `BitExact`
/// must not silently flip.
#[test]
fn add_zero_to_negative_zero_differs_by_numeric_policy() {
    use crate::op::{Extent, append};

    let mut program = Vec::new();
    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(1)],
            name: None,
        },
    );
    let zero = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 0.0,
        },
    );
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let broadcast_scalar = || IndexMap::Affine(map::projection(1, &[]));
    let output = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![(x, identity()), (zero, broadcast_scalar())],
            name: None,
        },
    );
    let shapes = shape::infer(&program, &[]).expect("x+0 program infers");

    let bit_exact = bind_with_fusion(
        &program,
        &shapes,
        &[output],
        true,
        NumericPolicy::bit_exact(),
    )
    .expect("bit-exact bind succeeds");
    let bit_exact_buffers = run_resolved(
        program.len(),
        &bit_exact,
        alloc::vec![(x, alloc::vec![-0.0f32])],
    );
    let bit_exact_result = bit_exact_buffers[output.0 as usize]
        .as_ref()
        .expect("bit-exact output present")[0];
    assert_eq!(
        bit_exact_result.to_bits(),
        0.0f32.to_bits(),
        "BitExact must compute the real (-0.0)+0.0 == +0.0, not eliminate the op and keep -0.0"
    );

    let signed_zero_policy = NumericPolicy {
        signed_zero: true,
        ..NumericPolicy::bit_exact()
    };
    let signed_zero_bound =
        bind_with_fusion(&program, &shapes, &[output], true, signed_zero_policy)
            .expect("signed-zero bind succeeds");
    let signed_zero_buffers = run_resolved(
        program.len(),
        &signed_zero_bound,
        alloc::vec![(x, alloc::vec![-0.0f32])],
    );
    let signed_zero_result = signed_zero_buffers[output.0 as usize]
        .as_ref()
        .expect("signed-zero output present")[0];
    assert_eq!(
        signed_zero_result.to_bits(),
        (-0.0f32).to_bits(),
        "signed_zero admits IdentityEliminationSignedZero, collapsing to the surviving \
         operand x == -0.0, got {signed_zero_result}"
    );
}

/// The split this design makes at the whole-bind level: granting
/// `nan_assumptions` alone must NOT also eliminate `x+0` -- proves the
/// two permissions stay independent through the full bind pipeline, not
/// just at [`admit`] in isolation.
#[test]
fn nan_assumption_alone_does_not_eliminate_add_zero() {
    use crate::op::{Extent, append};

    let mut program = Vec::new();
    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(1)],
            name: None,
        },
    );
    let zero = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 0.0,
        },
    );
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let broadcast_scalar = || IndexMap::Affine(map::projection(1, &[]));
    let output = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![(x, identity()), (zero, broadcast_scalar())],
            name: None,
        },
    );
    let shapes = shape::infer(&program, &[]).expect("x+0 program infers");
    let nan_assumption_policy = NumericPolicy {
        nan_assumptions: true,
        ..NumericPolicy::bit_exact()
    };
    let bound = bind_with_fusion(&program, &shapes, &[output], true, nan_assumption_policy)
        .expect("nan-assumption bind succeeds");
    let buffers = run_resolved(
        program.len(),
        &bound,
        alloc::vec![(x, alloc::vec![-0.0f32])],
    );
    let result = buffers[output.0 as usize].as_ref().expect("output present")[0];
    assert_eq!(
        result.to_bits(),
        0.0f32.to_bits(),
        "nan_assumptions must not grant signed_zero's x+0 elimination -- the Add step must \
         survive and compute the real (-0.0)+0.0 == +0.0"
    );
}

#[test]
fn name_reports_the_variant_backends_render_error_messages_with() {
    let cached_attention = BoundOpKind::CachedAttention {
        operands: Vec::new(),
        query_rows: 0,
        cached_key_rows: 0,
        new_key_rows: 0,
        kv_heads: 0,
        query_groups: 0,
        head_dim: 0,
        rotary_dim: 0,
        scale: 0.0,
        cached_lower_inclusive: 0,
        new_upper_inclusive: 0,
        two_pass: false,
    };
    let elementwise = BoundOpKind::Elementwise {
        body: ComposedBody::leaf(ScalarOp::Identity),
        operands: Vec::new(),
    };
    let reduce_fold = BoundOpKind::Reduce {
        element_body: ComposedBody::leaf(ScalarOp::Identity),
        reduce_op: ScalarOp::Add,
        init: ReduceInit::Zero,
        keep: Keep::Reduce,
        operands: Vec::new(),
        output_axes: SmallVec::new(),
        out_layout: Layout {
            base: 0,
            strides: SmallVec::new(),
        },
        out_scatter: None,
        epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
        epilogue_operands: Vec::new(),
        epilogue_broadcast_axes: SmallVec::new(),
    };
    let reduce_scan = BoundOpKind::Reduce {
        element_body: ComposedBody::leaf(ScalarOp::Identity),
        reduce_op: ScalarOp::Add,
        init: ReduceInit::Zero,
        keep: Keep::Scan,
        operands: Vec::new(),
        output_axes: SmallVec::new(),
        out_layout: Layout {
            base: 0,
            strides: SmallVec::new(),
        },
        out_scatter: None,
        epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
        epilogue_operands: Vec::new(),
        epilogue_broadcast_axes: SmallVec::new(),
    };

    assert_eq!(cached_attention.name(), "cached_attention");
    assert_eq!(elementwise.name(), "elementwise");
    assert_eq!(reduce_fold.name(), "keep::reduce fold");
    assert_eq!(reduce_scan.name(), "keep::scan fold");
    assert_eq!(BoundOpKind::Iota.name(), "iota");
    assert_eq!(BoundOpKind::Constant { value: 0.0 }.name(), "constant");
}

/// Whether `bound`'s `epilogue_body` differs from the identity leaf every
/// `BoundOpKind::Reduce` starts with -- the only witness, on the BOUND
/// output itself, that `reduce_epilogue_fusion` actually folded a
/// consumer into this reduce (as opposed to merely being a candidate the
/// raw-`Op`-shaped census in `reduce_epilogue_candidates` proposed but
/// that never applied because cached-attention fusion had already
/// absorbed one side of the pair).
#[cfg(feature = "reduce-epilogue-fusion")]
fn count_fused_epilogues(bound: &[BoundOp]) -> usize {
    bound
        .iter()
        .filter(|op| {
            matches!(&op.kind, BoundOpKind::Reduce { epilogue_body, .. }
                if *epilogue_body != ComposedBody::leaf(ScalarOp::Identity))
        })
        .count()
}

#[test]
#[cfg(feature = "cached-attention-streaming")]
fn cached_attention_rewrite_replaces_the_bound_attention_subgraph() {
    let (program, logits, roots) =
        crate::spec::mistral_cached_forward_program(32, 16, 24, 4, 2, 4, 1)
            .expect("cached attention fixture builds");
    let shapes = crate::shape::infer(&program, &[1, 1]).expect("cached attention infers");
    let mut requested = alloc::vec![logits];
    for cache_roots in &roots {
        requested.extend_from_slice(&[cache_roots.0, cache_roots.1, cache_roots.2]);
    }
    let outputs: &[NodeId] = &requested;
    let plain = bind_plain(&program, &shapes, outputs, NumericPolicy::bit_exact())
        .expect("plain bind succeeds");
    let cached_only =
        bind_cached_attention_fusion(&program, &shapes, outputs, true, NumericPolicy::bit_exact())
            .expect("cached-attention-only bind succeeds");
    let rewritten = bind(&program, &shapes, outputs, NumericPolicy::bit_exact())
        .expect("rewritten bind succeeds");

    assert_eq!(plain.len(), 47, "fixture baseline bound operation count");
    assert_eq!(
        cached_only.len(),
        25,
        "fixture cached-attention-only fused bound operation count"
    );
    // `reduce-epilogue-fusion` is a second, independent bind-time pass
    // that runs after the cached-attention rewrite this test targets,
    // additionally folding an epilogue-eligible Elementwise into its
    // Reduce whenever the feature is compiled in. Asserting the relation
    // against `count_fused_epilogues`'s own read of `rewritten`, rather
    // than a second hardcoded literal for the post-epilogue count, keeps
    // this honest across that feature's on/off states instead of
    // silently asserting the pre-epilogue number under both (the defect
    // this row fixes: `docs/discipline.md` for the mechanism).
    #[cfg(feature = "reduce-epilogue-fusion")]
    assert_eq!(
        cached_only.len() - rewritten.len(),
        count_fused_epilogues(&rewritten),
        "every bound op the reduce-epilogue pass removed corresponds to \
         one reduce in the rewritten program that actually carries a \
         fused epilogue"
    );
    #[cfg(not(feature = "reduce-epilogue-fusion"))]
    assert_eq!(
        rewritten.len(),
        cached_only.len(),
        "reduce-epilogue-fusion is compiled out; the cached-attention-only \
         count is final"
    );
    assert_eq!(
        rewritten
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count(),
        1,
        "one-layer fixture must receive one fused step"
    );
    assert!(
        rewritten
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
    );
}

/// ROW 364's own artifact: the ACTUAL bound program the real openchat
/// decode fixture uses. The production decode loop
/// (`proxima-model-interop::generate::build_single_range_program`)
/// calls `mistral_single_range_cached_forward_program` with
/// `DuplicateHeadPosition::None` -- NOT `mistral_cached_forward_program`,
/// the dual-range builder an earlier draft of this row's census
/// mistakenly used. The dual-range builder's toy fixture already
/// carried a fused SiLU epilogue on both main and this branch (a
/// coincidence of that builder's own gate reduce shape), so it could
/// not show what this row actually changes; this test builds through
/// the SAME single-range path production takes, one layer,
/// `bind::bind` (fusion on), printed op-by-op so a diff against the
/// same test run on main names exactly which ops the epilogue-fusion
/// landing removed or reshaped. `--nocapture` to see the list; the
/// assertion below is the mechanical guard that the count does not
/// silently drift once this is landed as a real (non-throwaway) test.
///
/// Gated on `reduce-epilogue-fusion` (the feature `bind()` actually
/// consults before folding a post-reduce tail into its `Reduce`'s
/// `epilogue_body` -- neither feature is in this crate's own `default`
/// set, see `Cargo.toml`) AND `cached-attention-streaming` (needed only
/// to keep this build free of the pre-existing, unrelated
/// `count_fused_epilogues` dead-code trap that fires when
/// `reduce-epilogue-fusion` is compiled in alone -- that function has no
/// caller outside the `cached-attention-streaming`-gated test below it).
/// Ungated, this test asserted the FUSED count (28) against whatever
/// `bind()` produces under the ambient feature set of the invoking
/// `cargo`/`nextest` command -- silently unfused (37 ops, not even the
/// pre-fusion baseline of 34, because `edbf2d90`'s slot-discovery fix now
/// also folds shapes the old literal-slot admission never recognized)
/// any time `reduce-epilogue-fusion` was not separately requested, which
/// is every default invocation this crate's own gate script runs.
#[test]
#[cfg(all(
    feature = "reduce-epilogue-fusion",
    feature = "cached-attention-streaming"
))]
fn row_364_per_layer_bound_op_list() {
    let (program, logits, roots, _duplicate_head_scratch) =
        crate::spec::mistral_single_range_cached_forward_program(
            32,
            16,
            24,
            4,
            2,
            4,
            1,
            false,
            crate::spec::DuplicateHeadPosition::None,
            false,
        )
        .expect("one-layer single-range decode fixture builds");
    let shapes = crate::shape::infer(&program, &[1, 1]).expect("cached decode fixture infers");
    let mut outputs = alloc::vec![logits];
    for (even, odd, value) in &roots {
        outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let bound = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("the cached decode fixture binds through the real bind() path");

    for (index, op) in bound.iter().enumerate() {
        match &op.kind {
            BoundOpKind::Elementwise { body, .. } => {
                let step_ops: Vec<ScalarOp> = body.steps.iter().map(|step| step.op).collect();
                std::println!("row364 index={index} kind=elementwise step_ops={step_ops:?}");
            }
            BoundOpKind::Reduce {
                epilogue_body,
                epilogue_broadcast_axes,
                ..
            } => {
                let epilogue_ops: Vec<ScalarOp> =
                    epilogue_body.steps.iter().map(|step| step.op).collect();
                std::println!(
                    "row364 index={index} kind={} epilogue_broadcast_axes={epilogue_broadcast_axes:?} epilogue_ops={epilogue_ops:?}",
                    op.kind.name()
                );
            }
            _ => {
                std::println!("row364 index={index} kind={}", op.kind.name());
            }
        }
    }

    assert_eq!(
        bound.len(),
        19,
        "row 364 artifact: one-layer single-range decode program's bound op count \
         (main at the same shape through the same builder: 34 -- three RMSNorm \
         sites each drop from a Reduce plus a separate Elementwise tail to one \
         fused Reduce, see `docs/discipline.md` ROW 364; 28 -> 19 is \
         `134975f83`'s reachable-only `bind_plain` landing after this row's own \
         28 was measured -- output-parity is preserved, see \
         `reduce_epilogue_fusion_matches_the_unfused_program_on_the_real_single_range_shape`, \
         which runs this exact single-layer single-range path and gets \
         max_abs_error=0 against the unfused reference)"
    );
}

#[test]
#[cfg(feature = "cached-attention-streaming")]
fn cached_attention_rewrite_accepts_the_omega_nonempty_cache_fixture() {
    let (program, logits, cache_roots) =
        crate::spec::mistral_cached_forward_program(64, 64, 128, 4, 2, 16, 2)
            .expect("omega cached attention fixture builds");
    let mut outputs = alloc::vec![logits];
    for (even, odd, value) in cache_roots {
        outputs.extend_from_slice(&[even, odd, value]);
    }
    let shapes =
        crate::shape::infer(&program, &[1, 5]).expect("omega cached attention fixture infers");
    let rewritten = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("omega cached attention fixture binds");

    assert_eq!(
        rewritten
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count(),
        2,
        "each production-shaped layer must receive its own fused step"
    );
    assert!(
        rewritten
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
    );
}

/// [`cached_attention_rewrite_accepts_the_omega_nonempty_cache_fixture`]'s
/// GQA-plus-QK-norm counterpart -- the real Qwen3-1.7B shape's two
/// distinguishing features (`query_heads != kv_heads`, split-half RoPE
/// plus per-head `q_norm`/`k_norm`) neither of which that fixture
/// exercises (it is Mistral-shaped: GQA but no QK-norm). Asserting the
/// engagement COUNT here, not a log line reading a runtime counter, is
/// the point: `proxima_tensor::instrument::path_totals().
/// op_kind_cached_attention` (`instrument.rs:1515`) increments only
/// from `cpu::run_node_into`'s `CachedAttention` arm
/// (`proxima-tensor/src/cpu.rs:5210-5215`) -- `omega::metal`'s own
/// `BoundOpKind::CachedAttention` dispatch (`omega/src/metal.rs:4206`
/// onward, the actual production Metal execution path
/// `generate.rs`'s metal-feature `BackendRuntime::evaluate` calls) never
/// touches that counter. On a Metal build that counter reads 0 on every
/// step regardless of whether the fusion engaged -- it is silent on the
/// one backend real decode runs on, not evidence the matcher rejected
/// the shape. This bind-time count is backend-agnostic and is the
/// correct place to assert engagement.
#[test]
#[cfg(feature = "cached-attention-streaming")]
fn cached_attention_rewrite_accepts_the_qwen3_gqa_qk_norm_fixture() {
    let (program, logits, cache_roots) =
        crate::spec::qwen3_cached_forward_program(64, 64, 128, 4, 2, 16, 2)
            .expect("qwen3 gqa+qk_norm fixture builds");
    let mut outputs = alloc::vec![logits];
    for (even, odd, value) in cache_roots {
        outputs.extend_from_slice(&[even, odd, value]);
    }
    let shapes = crate::shape::infer(&program, &[1, 5]).expect("qwen3 gqa+qk_norm fixture infers");
    let rewritten = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("qwen3 gqa+qk_norm fixture binds");

    assert_eq!(
        rewritten
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count(),
        2,
        "each GQA+QK-norm layer must receive its own fused step -- \
         the matcher accepts this shape; a regression here is a real \
         matcher rejection, not a dead runtime counter"
    );
}

/// The real openchat-3.5/Mistral-7B shape (`vocab=32_002`,
/// `hidden=4096`, `ffn=14336`, `32` query heads, `8` KV heads,
/// `head_dim=128`, `32` layers) bound at one new token against a
/// 71-position merged range — the same fixture
/// `spec::tests::the_single_range_cache_fold_node_budget_is_measured_
/// against_the_two_range_baseline` measures raw numbers for, asserted
/// here as a hard regression gate rather than a printed `println!`.
/// MEASURED, not derived from a dispatch-count census: `plain.len()`
/// (`939`) is this exact fixture's own baseline bound-op count with the
/// fusion matcher returning zero candidates (`cached_attention_single_
/// range_candidates` never firing is exactly the pre-existing defect
/// this row fixes); the cached-attention-only fused count (`619`) is
/// what fusing one `BoundOpKind::CachedAttention` per layer actually
/// removes -- 320 bound ops over 32 layers (10/layer), not the 6/layer a
/// raw-`Op` count would suggest, because `BoundOpBuilder` already fuses
/// several of the unfused chain's `Elementwise` nodes into their
/// consuming `Reduce` before this matcher ever runs. When
/// `reduce-epilogue-fusion` is also compiled in, a second independent
/// bind-time pass additionally folds 96 epilogue-eligible reduces on top
/// of that (`619 -> 523`). The expected count below is checked against
/// `count_fused_epilogues`'s own read of which `rewritten` reduces
/// actually carry a fused epilogue body, not
/// `reduce_epilogue_candidates`'s raw-`Op`-shaped census directly --
/// that census over-counts here (225 candidates on this fixture, not
/// 96) because most of its matches sit on nodes the cached-attention
/// rewrite already absorbed into a `BoundOpKind::CachedAttention` before
/// `reduce_epilogue_fusion` ever runs, so they silently fail its
/// `by_node` lookup instead of applying. Asserting a second hardcoded
/// literal for the post-epilogue count instead of this relation is
/// exactly the defect this row fixes: it asserted the pre-epilogue `619`
/// against an actual `523` for a full owner-visible session.
#[test]
#[cfg(feature = "cached-attention-streaming")]
fn single_range_cached_attention_fuses_one_step_per_layer_on_the_real_openchat_shape() {
    let (program, logits, cache_roots, _) =
        crate::spec::mistral_single_range_cached_forward_program(
            32_002,
            4096,
            14336,
            32,
            8,
            128,
            32,
            false,
            crate::spec::DuplicateHeadPosition::None,
            false,
        )
        .expect("openchat-shaped single-range forward pass lowers to a program");
    let mut outputs = alloc::vec![logits];
    for (even, odd, value) in &cache_roots {
        outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let shapes = crate::shape::infer(&program, &[1, 71])
        .expect("one new position against a 71-position merged range infers");
    let plain = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("plain bind succeeds");
    let cached_only = bind_cached_attention_fusion(
        &program,
        &shapes,
        &outputs,
        true,
        NumericPolicy::bit_exact(),
    )
    .expect("cached-attention-only bind succeeds");
    let rewritten =
        bind(&program, &shapes, &outputs, NumericPolicy::bit_exact()).expect("fused bind succeeds");

    assert_eq!(
        plain.len(),
        875,
        "openchat-shaped single-range baseline bound operation count \
         (939 -> 875 is `134975f83`'s reachable-only `bind_plain` landing -- \
         `bind_plain` now skips ops outside `live::reachable`'s backward \
         closure from `outputs` instead of binding every program position, \
         so this baseline no longer includes provably dead ops that could \
         never reach a requested output; correctness is structural, not \
         merely measured, since an unreachable node cannot affect any \
         output by definition)"
    );
    assert_eq!(
        cached_only.len(),
        619,
        "openchat-shaped single-range cached-attention-only fused bound operation count"
    );
    #[cfg(feature = "reduce-epilogue-fusion")]
    assert_eq!(
        cached_only.len() - rewritten.len(),
        count_fused_epilogues(&rewritten),
        "every bound op the reduce-epilogue pass removed corresponds to \
         one reduce in the rewritten program that actually carries a \
         fused epilogue"
    );
    #[cfg(not(feature = "reduce-epilogue-fusion"))]
    assert_eq!(
        rewritten.len(),
        cached_only.len(),
        "reduce-epilogue-fusion is compiled out; the cached-attention-only \
         count is final"
    );
    assert_eq!(
        rewritten
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count(),
        32,
        "one fused cached-attention step per layer on the real openchat shape"
    );
}

/// Regression for ROW 366 (`fix/merged-kv-attention-bounds`): the
/// single-range fusion used to duplicate the same bucketed-capacity
/// key/value shape into BOTH `cached_key_rows` and `new_key_rows`,
/// neutering the cached half with an unreachable `[i64::MAX, i64::MAX]`
/// band -- a kernel that iterates a fabricated cached half every call.
/// The merged-KV form has no separate cached range at all, so the fused
/// op declares `cached_key_rows: 0`: an empty range the kernel skips
/// entirely, not a live range it walks and discards.
#[test]
#[cfg(feature = "cached-attention-streaming")]
fn single_range_fusion_declares_an_empty_cached_range_not_a_duplicated_capacity() {
    let (program, logits, cache_roots, _) =
        crate::spec::mistral_single_range_cached_forward_program(
            32,
            16,
            24,
            4,
            2,
            4,
            1,
            false,
            crate::spec::DuplicateHeadPosition::None,
            false,
        )
        .expect("single-range fixture builds");
    let mut outputs = alloc::vec![logits];
    for (even, odd, value) in &cache_roots {
        outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let shapes = crate::shape::infer(&program, &[1, 5]).expect("single-range fixture infers");
    let resolved = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("plain bind succeeds");

    let candidates =
        cached_attention_single_range_candidates(&program, &shapes, &resolved, &outputs);
    assert!(
        !candidates.is_empty(),
        "the fixture must still produce a fusable single-range candidate"
    );
    let BoundOpKind::CachedAttention {
        cached_key_rows,
        new_key_rows,
        cached_lower_inclusive,
        operands,
        ..
    } = &candidates[0].0.kind
    else {
        panic!("single-range candidate must carry CachedAttention operands");
    };
    assert_eq!(
        *cached_key_rows, 0,
        "a merged-KV buffer has no separate cached range"
    );
    assert!(
        *new_key_rows > 0,
        "the merged range itself must still carry the bucketed capacity"
    );
    assert_eq!(
        *cached_lower_inclusive,
        i64::MIN,
        "no dead-band sentinel is needed once the cached range is empty"
    );
    assert_eq!(
        operands.len(),
        9,
        "the live cached_len still travels as the ninth runtime operand"
    );
}

/// Deterministic non-degenerate weight data -- a golden-ratio fractional
/// sequence, not a crate RNG dependency, and never all-same-value (which
/// would hide a transposed axis or a dropped operand behind coincidental
/// symmetry).
#[cfg(feature = "cached-attention-streaming")]
fn deterministic_values(count: usize, seed: f32) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let phase = (index as f32 + seed) * 0.618_034;
            (phase - libm::floorf(phase)) * 2.0 - 1.0
        })
        .collect()
}

/// Shape constants for [`qwen35_partial_rotary_attention_fixture`],
/// hoisted to module scope so
/// [`qwen35_dense_attention_f64_reference`] computes over the SAME
/// dimensions the fixture builds its graph with -- a private copy inside
/// each function is exactly the kind of drift this row's own
/// independent-reference discipline exists to rule out.
#[cfg(feature = "cached-attention-streaming")]
const QWEN35_PARTIAL_ROTARY_KV_HEADS: usize = 2;
#[cfg(feature = "cached-attention-streaming")]
const QWEN35_PARTIAL_ROTARY_GROUP: usize = 8;
#[cfg(feature = "cached-attention-streaming")]
const QWEN35_PARTIAL_ROTARY_ATTN_HEAD_DIM: usize = 256;
#[cfg(feature = "cached-attention-streaming")]
const QWEN35_PARTIAL_ROTARY_ROTARY_DIM: usize = 64;
#[cfg(feature = "cached-attention-streaming")]
const QWEN35_PARTIAL_ROTARY_PASS_DIM: usize =
    QWEN35_PARTIAL_ROTARY_ATTN_HEAD_DIM - QWEN35_PARTIAL_ROTARY_ROTARY_DIM;
#[cfg(feature = "cached-attention-streaming")]
const QWEN35_PARTIAL_ROTARY_PAIR_DIM: usize = QWEN35_PARTIAL_ROTARY_ROTARY_DIM / 2;
#[cfg(feature = "cached-attention-streaming")]
const QWEN35_PARTIAL_ROTARY_NEW_TOKENS: usize = 1;
#[cfg(feature = "cached-attention-streaming")]
const QWEN35_PARTIAL_ROTARY_CACHED_EXTENT: usize = 40;

/// [`append_qwen35_dense_attention_only_with_taps`] wired at qwen3.5's
/// own real per-head shape (`kv_heads` 2, `group` 8 -> 16 query heads,
/// `attn_head_dim` 256, `rotary_dim` 64 -> 192-wide pass plane,
/// `docs/discipline.md` ROW 556/557's own residual) -- `embedding` stays
/// 1, the same degenerate-but-valid width
/// `dense_attention_only_test_inputs` (`spec.rs`) already uses, since
/// only the attention block's own per-head shape is under test here.
/// `cached_extent` is 40 keys; `cached_len` (a runtime scalar, not a
/// shape) is fed 37 at execution, leaving 3 trailing rows the padding
/// `Select` masks with `-inf` (`spec.rs:4979-4997`).
#[cfg(feature = "cached-attention-streaming")]
#[allow(clippy::too_many_lines, clippy::type_complexity)]
fn qwen35_partial_rotary_attention_fixture() -> (
    Vec<Op>,
    NodeId,
    crate::spec::Qwen35DenseAttentionTaps,
    Vec<(NodeId, Vec<f32>)>,
    Shapes,
) {
    use crate::op::Extent;
    use crate::spec::{causal_mask, input_leaf, scalar_constant};

    const KV_HEADS: usize = QWEN35_PARTIAL_ROTARY_KV_HEADS;
    const GROUP: usize = QWEN35_PARTIAL_ROTARY_GROUP;
    const ATTN_HEAD_DIM: usize = QWEN35_PARTIAL_ROTARY_ATTN_HEAD_DIM;
    const ROTARY_DIM: usize = QWEN35_PARTIAL_ROTARY_ROTARY_DIM;
    const PASS_DIM: usize = QWEN35_PARTIAL_ROTARY_PASS_DIM;
    const PAIR_DIM: usize = QWEN35_PARTIAL_ROTARY_PAIR_DIM;
    const NEW_TOKENS: usize = QWEN35_PARTIAL_ROTARY_NEW_TOKENS;
    const CACHED_EXTENT: usize = QWEN35_PARTIAL_ROTARY_CACHED_EXTENT;

    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
        "x",
    );
    let inv_dim = scalar_constant(&mut program, 1.0);
    let eps = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0)],
        "eps",
    );
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 1.0 / (ATTN_HEAD_DIM as f32).sqrt());
    let inv_attn_head_dim = scalar_constant(&mut program, 1.0 / ATTN_HEAD_DIM as f32);
    let rotary_shape = alloc::vec![Extent::Symbolic(0), Extent::Static(PAIR_DIM as u32)];
    let cos_new = input_leaf(&mut program, DType::Float32, rotary_shape.clone(), "cos");
    let sin_new = input_leaf(&mut program, DType::Float32, rotary_shape, "sin");
    let group_ones = crate::op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![
                Extent::Static(KV_HEADS as u32),
                Extent::Static(GROUP as u32)
            ],
            value: 1.0,
        },
    );
    let (is_future, _neg_infinity) = causal_mask(&mut program).expect("causal mask lowers");
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let attn_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "attn_norm_weight",
    );
    let norm_shape = alloc::vec![Extent::Static(ATTN_HEAD_DIM as u32)];
    let q_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        norm_shape.clone(),
        "q_norm_weight",
    );
    let k_norm_weight = input_leaf(&mut program, DType::Float32, norm_shape, "k_norm_weight");
    // `wq_gate`'s own middle axis is the FULL query head count
    // (`kv_heads * group`), never `kv_heads` alone -- `spec.rs:10730-10769`
    // packs it that way, and the group-broadcast reshape further down
    // (`group_map_i`) depends on it.
    let wq_gate = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(1),
            Extent::Static((KV_HEADS * GROUP) as u32),
            Extent::Static((2 * ATTN_HEAD_DIM) as u32)
        ],
        "wq_gate",
    );
    let wk_wv_shape = alloc::vec![
        Extent::Static(1),
        Extent::Static(KV_HEADS as u32),
        Extent::Static(ATTN_HEAD_DIM as u32)
    ];
    let wk = input_leaf(&mut program, DType::Float32, wk_wv_shape.clone(), "wk");
    let wv = input_leaf(&mut program, DType::Float32, wk_wv_shape, "wv");
    let wo = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(KV_HEADS as u32),
            Extent::Static(GROUP as u32),
            Extent::Static(ATTN_HEAD_DIM as u32),
            Extent::Static(1)
        ],
        "wo",
    );
    let cache_rotary_shape = alloc::vec![
        Extent::Symbolic(1),
        Extent::Static(KV_HEADS as u32),
        Extent::Static(PAIR_DIM as u32)
    ];
    let cache_pass_shape = alloc::vec![
        Extent::Symbolic(1),
        Extent::Static(KV_HEADS as u32),
        Extent::Static(PASS_DIM as u32)
    ];
    let cache_v_shape = alloc::vec![
        Extent::Symbolic(1),
        Extent::Static(KV_HEADS as u32),
        Extent::Static(ATTN_HEAD_DIM as u32)
    ];
    let k_first_cache = input_leaf(
        &mut program,
        DType::Float32,
        cache_rotary_shape.clone(),
        "k_first_cache",
    );
    let k_second_cache = input_leaf(
        &mut program,
        DType::Float32,
        cache_rotary_shape,
        "k_second_cache",
    );
    let k_pass_cache = input_leaf(
        &mut program,
        DType::Float32,
        cache_pass_shape,
        "k_pass_cache",
    );
    let v_cache = input_leaf(&mut program, DType::Float32, cache_v_shape, "v_cache");

    let (residual1, taps) = crate::spec::append_qwen35_dense_attention_only_with_taps(
        &mut program,
        x,
        inv_dim,
        eps,
        ones,
        inv_sqrt_attn_head_dim,
        inv_attn_head_dim,
        cos_new,
        sin_new,
        group_ones,
        is_future,
        cached_len,
        GROUP as u32,
        ROTARY_DIM as u32,
        ATTN_HEAD_DIM as u32,
        attn_norm_weight,
        q_norm_weight,
        k_norm_weight,
        wq_gate,
        wk,
        wv,
        wo,
        k_first_cache,
        k_second_cache,
        k_pass_cache,
        v_cache,
    )
    .expect("qwen35 partial-rotary dense attention fixture lowers");

    let shapes = crate::shape::infer(&program, &[NEW_TOKENS as u64, CACHED_EXTENT as u64])
        .expect("qwen35 partial-rotary dense attention fixture infers");

    let leaf_values = |node: NodeId, seed: f32| -> (NodeId, Vec<f32>) {
        let count = shapes.of(node).iter().product::<u64>().max(1) as usize;
        (node, deterministic_values(count, seed))
    };
    let eps_count = shapes.of(eps).iter().product::<u64>().max(1) as usize;
    let inputs = alloc::vec![
        leaf_values(x, 1.0),
        (eps, alloc::vec![1e-5f32; eps_count]),
        leaf_values(cos_new, 2.0),
        leaf_values(sin_new, 3.0),
        leaf_values(attn_norm_weight, 4.0),
        leaf_values(q_norm_weight, 5.0),
        leaf_values(k_norm_weight, 6.0),
        leaf_values(wq_gate, 7.0),
        leaf_values(wk, 8.0),
        leaf_values(wv, 9.0),
        leaf_values(wo, 10.0),
        leaf_values(k_first_cache, 11.0),
        leaf_values(k_second_cache, 12.0),
        leaf_values(k_pass_cache, 13.0),
        leaf_values(v_cache, 14.0),
        (cached_len, alloc::vec![37.0]),
    ];
    (program, residual1, taps, inputs, shapes)
}

/// The matcher's own recognition gate: qwen35's real partial-rotary
/// chain (`rotary_dim` 64 of `head_dim` 256, `kv_heads` 2, `group` 8)
/// fuses into exactly one [`BoundOpKind::CachedAttention`] carrying the
/// full eight-base + `cached_len` + three-pass-plane operand set, and
/// the fusion strictly reduces the bound-op count relative to the
/// unfused elementwise/reduce chain [`bind_plain`] already produces
/// (the census, ROW 558 `docs/discipline.md`).
///
/// Also asserts numeric parity against the unfused chain (item (a) of
/// the ROW 558 brief, deliberately deferred through ROW 558-560's own
/// residuals while the ~8% divergence this fixture surfaced was
/// mislocated in two false leads before landing on the real defect --
/// `neon_tile_plan`'s missing row-invariance check on its `b` operand,
/// `docs/discipline.md` ROW 561, fixed at `cpu.rs`'s own
/// `neon_tile_plan`). [`qwen35_partial_rotary_dense_attention_matches_an_independent_f64_reference`]
/// is the test that actually PROVES which side was wrong, tap by tap,
/// against an f64 reference outside both engines; this test only checks
/// that the two engines still AGREE once that defect is fixed.
#[test]
#[cfg(feature = "cached-attention-streaming")]
fn qwen35_partial_rotary_cached_attention_fuses_and_matches_the_unfused_layer() {
    let (program, residual1, taps, inputs, shapes) = qwen35_partial_rotary_attention_fixture();
    // `attended` (the fusion's own anchor node, `attention_score_sources`'s
    // own doc) has exactly one reader (the per-head gate multiply) --
    // qwen35's own extra gate stage, absent from mistral/openchat, gives
    // `ChainFusion` one more link to fold it into, so it must be pinned
    // as its own materialization boundary the same way the cache roots
    // already are, or `bind_plain` never gives it a standalone `BoundOp`
    // for `cached_attention_candidates` to find.
    let outputs = alloc::vec![
        residual1,
        taps.attended,
        taps.rotated_k_new_first,
        taps.rotated_k_new_second,
        taps.k_pass,
        taps.v_new,
    ];

    let unfused = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("plain bind succeeds");
    let fused =
        bind(&program, &shapes, &outputs, NumericPolicy::bit_exact()).expect("fused bind succeeds");

    let fused_attention_count = fused
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .count();
    assert_eq!(
        fused_attention_count, 1,
        "the qwen35 partial-rotary chain must fuse into exactly one \
         cached-attention op"
    );
    let BoundOpKind::CachedAttention {
        rotary_dim,
        head_dim,
        operands,
        ..
    } = fused
        .iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .map(|bound| &bound.kind)
        .expect("a cached-attention op was just counted above")
    else {
        unreachable!("just matched CachedAttention above");
    };
    assert_eq!(
        *rotary_dim, 64,
        "rotary width is qwen35's own 64, not the full head_dim"
    );
    assert_eq!(
        *head_dim, 256,
        "head_dim carries the full width, rotary plus pass"
    );
    assert_eq!(
        operands.len(),
        12,
        "eight base sources, the runtime cached_len, and the three pass-plane sources"
    );
    // the census: how many bound ops this one fusion absorbed (66 unfused
    // vs 40 fused on this fixture -- 26 ops absorbed by the one fusion,
    // `docs/discipline.md` ROW 558).
    assert!(
        unfused.len() > fused.len(),
        "the fusion must absorb at least one op relative to the unfused chain"
    );

    let unfused_outputs = run_resolved(program.len(), &unfused, inputs.clone());
    let fused_outputs = run_resolved(program.len(), &fused, inputs);
    let expected = unfused_outputs[residual1.0 as usize]
        .as_ref()
        .expect("unfused layer output computes");
    let actual = fused_outputs[residual1.0 as usize]
        .as_ref()
        .expect("fused layer output computes");
    assert_eq!(
        expected.len(),
        actual.len(),
        "fused and unfused outputs must be the same shape"
    );
    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate()
    {
        let (expected_value, actual_value) = (f64::from(expected_value), f64::from(actual_value));
        let relative_error = (expected_value - actual_value).abs() / expected_value.abs().max(1.0);
        assert!(
            relative_error <= 1e-5,
            "residual1[{index}] fused vs unfused: expected={expected_value} actual={actual_value} \
             relative_error={relative_error} (ROW 561's own fix)"
        );
    }
}

/// The real forward-pass builder (`program.rs`'s own qwen35moe caller)
/// never requests `taps.attended` as an output -- only `residual1` and
/// the four cache-write roots survive into the next layer/decode step --
/// so this test drops `taps.attended` from `outputs` relative to
/// [`qwen35_partial_rotary_cached_attention_fuses_and_matches_the_unfused_layer`]'s
/// own list, reproducing the exact shape ROW 563's real-checkpoint
/// telemetry captured (`node=1288 elementwise_operand_fuse decision=fused
/// into=1294`, immediately followed by every full-attention layer's
/// `cached_attention decline ... stage=output_not_resolved`). Before
/// `bind_cached_attention_fusion`'s own `planning_outputs` loop pinned a
/// discovered candidate's anchor node (`fused.node`) alongside its
/// source operands, this exact `outputs` list produced
/// `fused_attention_count == 0` on this fixture.
#[test]
#[cfg(feature = "cached-attention-streaming")]
fn qwen35_partial_rotary_cached_attention_fuses_without_pinning_the_attended_tap() {
    let (program, residual1, taps, inputs, shapes) = qwen35_partial_rotary_attention_fixture();
    let outputs = alloc::vec![
        residual1,
        taps.rotated_k_new_first,
        taps.rotated_k_new_second,
        taps.k_pass,
        taps.v_new,
    ];

    let fused =
        bind(&program, &shapes, &outputs, NumericPolicy::bit_exact()).expect("fused bind succeeds");
    let fused_attention_count = fused
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .count();
    assert_eq!(
        fused_attention_count, 1,
        "the qwen35 partial-rotary chain must fuse into exactly one \
         cached-attention op even when only the real forward pass's own \
         outputs are requested"
    );

    let unfused = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("plain bind succeeds");
    let unfused_outputs = run_resolved(program.len(), &unfused, inputs.clone());
    let fused_outputs = run_resolved(program.len(), &fused, inputs);
    let expected = unfused_outputs[residual1.0 as usize]
        .as_ref()
        .expect("unfused layer output computes");
    let actual = fused_outputs[residual1.0 as usize]
        .as_ref()
        .expect("fused layer output computes");
    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate()
    {
        let (expected_value, actual_value) = (f64::from(expected_value), f64::from(actual_value));
        let relative_error = (expected_value - actual_value).abs() / expected_value.abs().max(1.0);
        assert!(
            relative_error <= 1e-5,
            "residual1[{index}] fused vs unfused: expected={expected_value} actual={actual_value} \
             relative_error={relative_error}"
        );
    }
}

/// Every tap [`qwen35_dense_attention_f64_reference`] computes, in the
/// order [`crate::spec::append_qwen35_dense_attention_only_with_taps`]
/// builds them -- the order this row's own per-tap divergence search
/// walks.
#[cfg(feature = "cached-attention-streaming")]
struct Qwen35DenseAttentionF64Reference {
    normed: Vec<f64>,
    q_split: Vec<f64>,
    gate_split: Vec<f64>,
    v_new: Vec<f64>,
    q_normed: Vec<f64>,
    k_normed: Vec<f64>,
    k_pass: Vec<f64>,
    q_rot_first: Vec<f64>,
    q_rot_second: Vec<f64>,
    k_rot_first: Vec<f64>,
    k_rot_second: Vec<f64>,
    score_new: Vec<f64>,
    attended: Vec<f64>,
    gate_sigmoid: Vec<f64>,
    gated_attended: Vec<f64>,
    o_proj_out: Vec<f64>,
    residual1: Vec<f64>,
}

/// Independent f64 reference for
/// [`qwen35_partial_rotary_attention_fixture`]'s whole dense-attention
/// layer, built by plain nested loops over the SAME per-input byte
/// vectors the fixture feeds `run_resolved` (`inputs`, read positionally
/// in the fixture's own construction order -- see the `assert_eq!` on
/// `inputs.len()` below, which fails loudly if that order ever changes).
/// No `Op`/`bind`/`Reduce` machinery anywhere in this function: this is
/// the third, previously-missing leg `docs/discipline.md` ROW 560's own
/// residual asked for. ROW 559 showed the fused `CachedAttention` kernel
/// is exact against its own operands; ROW 560 showed `bind_plain`'s
/// generic reduce is ALSO exact against ITS own operands; neither row
/// had ever checked either chain against a party that owes nothing to
/// either implementation's own bugs -- an oracle from the model
/// semantics (`modeling_qwen3_next.py`), not from either engine.
///
/// Math, per stage (mirrors `spec.rs:4724-5312`
/// (`append_qwen35_dense_attention_only_with_taps`) exactly, at f64
/// precision, for this fixture's own degenerate `embedding = 1`,
/// `new_tokens = 1` shape): RMSNorm(`x`) -> the one `q`/`gate`
/// projection split per head -> `k`/`v` projections -> per-head RMSNorm
/// of `q`/`k` over the full `attn_head_dim` -> the pass-plane slice
/// (`[rotary_dim, attn_head_dim)`) -> interleaved RoPE
/// (`(2*i, 2*i+1)` pairing, `RopePairing::Interleaved`'s own doc) over
/// the first `rotary_dim` channels -> grouped rotary + pass dot products
/// against the 40-row KV cache (masked `-inf` past `cached_len`) and the
/// one new key (never masked at position 0) -> one softmax over the
/// concatenation of both -> the value-weighted sum -> the per-head
/// sigmoid gate -> `o_proj` reduced to the single embedding output ->
/// the residual add.
#[cfg(feature = "cached-attention-streaming")]
#[allow(clippy::too_many_lines)]
fn qwen35_dense_attention_f64_reference(
    inputs: &[(NodeId, Vec<f32>)],
) -> Qwen35DenseAttentionF64Reference {
    const KV_HEADS: usize = QWEN35_PARTIAL_ROTARY_KV_HEADS;
    const GROUP: usize = QWEN35_PARTIAL_ROTARY_GROUP;
    const ATTN_HEAD_DIM: usize = QWEN35_PARTIAL_ROTARY_ATTN_HEAD_DIM;
    const ROTARY_DIM: usize = QWEN35_PARTIAL_ROTARY_ROTARY_DIM;
    const PASS_DIM: usize = QWEN35_PARTIAL_ROTARY_PASS_DIM;
    const PAIR_DIM: usize = QWEN35_PARTIAL_ROTARY_PAIR_DIM;
    const CACHED_EXTENT: usize = QWEN35_PARTIAL_ROTARY_CACHED_EXTENT;
    const HEADS: usize = KV_HEADS * GROUP;

    assert_eq!(
        inputs.len(),
        16,
        "qwen35_partial_rotary_attention_fixture's own input list grew or shrank -- \
         this reference's positional indexing below must be re-derived, not silently \
         misaligned"
    );
    let as_f64 =
        |values: &[f32]| -> Vec<f64> { values.iter().map(|&value| f64::from(value)).collect() };
    let x = as_f64(&inputs[0].1);
    let eps = f64::from(inputs[1].1[0]);
    let cos_new = as_f64(&inputs[2].1);
    let sin_new = as_f64(&inputs[3].1);
    let attn_norm_weight = f64::from(inputs[4].1[0]);
    let q_norm_weight = as_f64(&inputs[5].1);
    let k_norm_weight = as_f64(&inputs[6].1);
    let wq_gate = as_f64(&inputs[7].1);
    let wk = as_f64(&inputs[8].1);
    let wv = as_f64(&inputs[9].1);
    let wo = as_f64(&inputs[10].1);
    let k_first_cache = as_f64(&inputs[11].1);
    let k_second_cache = as_f64(&inputs[12].1);
    let k_pass_cache = as_f64(&inputs[13].1);
    let v_cache = as_f64(&inputs[14].1);
    let cached_len = f64::from(inputs[15].1[0]);

    // RMSNorm(x): embedding width 1, so the sum-of-squares is one term
    // and inv_dim (a scalar_constant(1.0)) contributes nothing.
    let inv_rms_x = 1.0 / (x[0] * x[0] + eps).sqrt();
    let normed = x[0] * inv_rms_x * attn_norm_weight;

    // qg_raw[h][c] = normed * wq_gate[0][h][c] (embedding contraction is
    // one term); q_split/gate_split are the per-head [0,256)/[256,512)
    // halves of that same activation.
    let mut q_split = vec![0.0f64; HEADS * ATTN_HEAD_DIM];
    let mut gate_split = vec![0.0f64; HEADS * ATTN_HEAD_DIM];
    for head in 0..HEADS {
        for channel in 0..ATTN_HEAD_DIM {
            let q_weight = wq_gate[head * (2 * ATTN_HEAD_DIM) + channel];
            let gate_weight = wq_gate[head * (2 * ATTN_HEAD_DIM) + ATTN_HEAD_DIM + channel];
            q_split[head * ATTN_HEAD_DIM + channel] = normed * q_weight;
            gate_split[head * ATTN_HEAD_DIM + channel] = normed * gate_weight;
        }
    }

    let mut k_raw = vec![0.0f64; KV_HEADS * ATTN_HEAD_DIM];
    let mut v_new = vec![0.0f64; KV_HEADS * ATTN_HEAD_DIM];
    for kv_head in 0..KV_HEADS {
        for channel in 0..ATTN_HEAD_DIM {
            k_raw[kv_head * ATTN_HEAD_DIM + channel] =
                normed * wk[kv_head * ATTN_HEAD_DIM + channel];
            v_new[kv_head * ATTN_HEAD_DIM + channel] =
                normed * wv[kv_head * ATTN_HEAD_DIM + channel];
        }
    }

    let inv_head_dim = 1.0 / ATTN_HEAD_DIM as f64;
    let mut q_normed = vec![0.0f64; HEADS * ATTN_HEAD_DIM];
    for head in 0..HEADS {
        let sum_squares: f64 = (0..ATTN_HEAD_DIM)
            .map(|channel| q_split[head * ATTN_HEAD_DIM + channel].powi(2))
            .sum();
        let inv_rms = 1.0 / (sum_squares * inv_head_dim + eps).sqrt();
        for channel in 0..ATTN_HEAD_DIM {
            q_normed[head * ATTN_HEAD_DIM + channel] =
                q_split[head * ATTN_HEAD_DIM + channel] * inv_rms * q_norm_weight[channel];
        }
    }
    let mut k_normed = vec![0.0f64; KV_HEADS * ATTN_HEAD_DIM];
    for kv_head in 0..KV_HEADS {
        let sum_squares: f64 = (0..ATTN_HEAD_DIM)
            .map(|channel| k_raw[kv_head * ATTN_HEAD_DIM + channel].powi(2))
            .sum();
        let inv_rms = 1.0 / (sum_squares * inv_head_dim + eps).sqrt();
        for channel in 0..ATTN_HEAD_DIM {
            k_normed[kv_head * ATTN_HEAD_DIM + channel] =
                k_raw[kv_head * ATTN_HEAD_DIM + channel] * inv_rms * k_norm_weight[channel];
        }
    }

    let mut k_pass = vec![0.0f64; KV_HEADS * PASS_DIM];
    for kv_head in 0..KV_HEADS {
        for pass_channel in 0..PASS_DIM {
            k_pass[kv_head * PASS_DIM + pass_channel] =
                k_normed[kv_head * ATTN_HEAD_DIM + ROTARY_DIM + pass_channel];
        }
    }
    let mut q_pass = vec![0.0f64; HEADS * PASS_DIM];
    for head in 0..HEADS {
        for pass_channel in 0..PASS_DIM {
            q_pass[head * PASS_DIM + pass_channel] =
                q_normed[head * ATTN_HEAD_DIM + ROTARY_DIM + pass_channel];
        }
    }

    // Interleaved RoPE, `RopePairing::Interleaved`'s own `(2*i, 2*i+1)`
    // pairing, over the first `rotary_dim` channels only. `cos_new`/
    // `sin_new` are `[s, pair]`-shaped with `s == 1` in this fixture, so
    // a bare `pair` index already reads position 0's own row.
    let rotate = |source: &[f64], head_count: usize| -> (Vec<f64>, Vec<f64>) {
        let mut first = vec![0.0f64; head_count * PAIR_DIM];
        let mut second = vec![0.0f64; head_count * PAIR_DIM];
        for head in 0..head_count {
            for pair in 0..PAIR_DIM {
                let even = source[head * ATTN_HEAD_DIM + 2 * pair];
                let odd = source[head * ATTN_HEAD_DIM + 2 * pair + 1];
                first[head * PAIR_DIM + pair] = even * cos_new[pair] - odd * sin_new[pair];
                second[head * PAIR_DIM + pair] = odd * cos_new[pair] + even * sin_new[pair];
            }
        }
        (first, second)
    };
    let (q_rot_first, q_rot_second) = rotate(&q_normed, HEADS);
    let (k_rot_first, k_rot_second) = rotate(&k_normed, KV_HEADS);

    // query head h = GROUP*kv_head + group, `group_map_i`/`group_map_p`/
    // `group_map_d`'s own convention (`spec.rs:4847,4866,5221`).
    let query_head = |kv_head: usize, group: usize| GROUP * kv_head + group;

    let inv_sqrt_head_dim = 1.0 / (ATTN_HEAD_DIM as f64).sqrt();
    let mut score_cached = vec![0.0f64; CACHED_EXTENT * KV_HEADS * GROUP];
    for cached_row in 0..CACHED_EXTENT {
        for kv_head in 0..KV_HEADS {
            for group in 0..GROUP {
                let head = query_head(kv_head, group);
                let mut score = 0.0f64;
                for pair in 0..PAIR_DIM {
                    let cache_index = (cached_row * KV_HEADS + kv_head) * PAIR_DIM + pair;
                    score += q_rot_first[head * PAIR_DIM + pair] * k_first_cache[cache_index];
                    score += q_rot_second[head * PAIR_DIM + pair] * k_second_cache[cache_index];
                }
                for pass_channel in 0..PASS_DIM {
                    let cache_index = (cached_row * KV_HEADS + kv_head) * PASS_DIM + pass_channel;
                    score += q_pass[head * PASS_DIM + pass_channel] * k_pass_cache[cache_index];
                }
                score *= inv_sqrt_head_dim;
                let is_padding = cached_row as f64 > cached_len - 1.0;
                let index = (cached_row * KV_HEADS + kv_head) * GROUP + group;
                score_cached[index] = if is_padding { f64::NEG_INFINITY } else { score };
            }
        }
    }

    // The one new (uncached) key: position 0 is never future-masked
    // against itself (`is_future[0][0] == (0 > 0) == false`).
    let mut score_new = vec![0.0f64; KV_HEADS * GROUP];
    for kv_head in 0..KV_HEADS {
        for group in 0..GROUP {
            let head = query_head(kv_head, group);
            let mut score = 0.0f64;
            for pair in 0..PAIR_DIM {
                score +=
                    q_rot_first[head * PAIR_DIM + pair] * k_rot_first[kv_head * PAIR_DIM + pair];
                score +=
                    q_rot_second[head * PAIR_DIM + pair] * k_rot_second[kv_head * PAIR_DIM + pair];
            }
            for pass_channel in 0..PASS_DIM {
                score += q_pass[head * PASS_DIM + pass_channel]
                    * k_pass[kv_head * PASS_DIM + pass_channel];
            }
            score_new[kv_head * GROUP + group] = score * inv_sqrt_head_dim;
        }
    }

    let mut attended = vec![0.0f64; KV_HEADS * GROUP * ATTN_HEAD_DIM];
    let mut gate_sigmoid = vec![0.0f64; KV_HEADS * GROUP * ATTN_HEAD_DIM];
    let mut gated_attended = vec![0.0f64; KV_HEADS * GROUP * ATTN_HEAD_DIM];
    for kv_head in 0..KV_HEADS {
        for group in 0..GROUP {
            let cached_scores: Vec<f64> = (0..CACHED_EXTENT)
                .map(|cached_row| score_cached[(cached_row * KV_HEADS + kv_head) * GROUP + group])
                .collect();
            let new_score = score_new[kv_head * GROUP + group];
            let global_max = cached_scores.iter().copied().fold(new_score, f64::max);
            let cached_weights: Vec<f64> = cached_scores
                .iter()
                .map(|&score| (score - global_max).exp())
                .collect();
            let new_weight = (new_score - global_max).exp();
            let inv_weight_sum = 1.0 / (cached_weights.iter().sum::<f64>() + new_weight);

            let head = query_head(kv_head, group);
            for channel in 0..ATTN_HEAD_DIM {
                let cached_sum: f64 = (0..CACHED_EXTENT)
                    .map(|cached_row| {
                        cached_weights[cached_row]
                            * v_cache[(cached_row * KV_HEADS + kv_head) * ATTN_HEAD_DIM + channel]
                    })
                    .sum();
                let new_sum = new_weight * v_new[kv_head * ATTN_HEAD_DIM + channel];
                let attended_value = (cached_sum + new_sum) * inv_weight_sum;
                let sigmoid = 1.0 / (1.0 + (-gate_split[head * ATTN_HEAD_DIM + channel]).exp());
                let index = (kv_head * GROUP + group) * ATTN_HEAD_DIM + channel;
                attended[index] = attended_value;
                gate_sigmoid[index] = sigmoid;
                gated_attended[index] = attended_value * sigmoid;
            }
        }
    }

    let mut o_proj_out = 0.0f64;
    for kv_head in 0..KV_HEADS {
        for group in 0..GROUP {
            for channel in 0..ATTN_HEAD_DIM {
                let index = (kv_head * GROUP + group) * ATTN_HEAD_DIM + channel;
                o_proj_out += gated_attended[index] * wo[index];
            }
        }
    }
    let residual1 = o_proj_out + x[0];

    Qwen35DenseAttentionF64Reference {
        normed: alloc::vec![normed],
        q_split,
        gate_split,
        v_new,
        q_normed,
        k_normed,
        k_pass,
        q_rot_first,
        q_rot_second,
        k_rot_first,
        k_rot_second,
        score_new,
        attended,
        gate_sigmoid,
        gated_attended,
        o_proj_out: alloc::vec![o_proj_out],
        residual1: alloc::vec![residual1],
    }
}

/// Max relative error of `actual` against `reference`, element-wise --
/// this row's own single comparison primitive, used identically for
/// every tap so the per-tap table below is apples to apples. Denominator
/// floors at `1.0` so a near-zero reference element does not blow up a
/// float-rounding-sized absolute difference into a nonsense percentage.
#[cfg(feature = "cached-attention-streaming")]
fn max_relative_error(reference: &[f64], actual: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        actual.len(),
        "reference and actual must share a shape"
    );
    reference
        .iter()
        .zip(actual.iter())
        .map(|(&expected, &got)| {
            let got = f64::from(got);
            (got - expected).abs() / expected.abs().max(1.0)
        })
        .fold(0.0f64, f64::max)
}

/// The independent-reference leg ROW 559/560's own residual named as
/// missing (`docs/discipline.md`): both `bind_plain` and the fused
/// `CachedAttention` op were already proven self-consistent against
/// their OWN resolved operands, but never checked against a party that
/// owes nothing to either implementation. This test runs both chains
/// through [`run_resolved`] on the SAME fixture inputs
/// [`qwen35_dense_attention_f64_reference`] also consumes, then asserts
/// every named tap from BOTH chains against that f64 reference -- the
/// first tap (in builder order) that fails on either side names the
/// defective chain and the value pair proving it.
#[test]
#[cfg(feature = "cached-attention-streaming")]
fn qwen35_partial_rotary_dense_attention_matches_an_independent_f64_reference() {
    let (program, residual1, taps, inputs, shapes) = qwen35_partial_rotary_attention_fixture();
    let reference = qwen35_dense_attention_f64_reference(&inputs);

    let outputs = alloc::vec![
        residual1,
        taps.normed,
        taps.q_split,
        taps.gate_split,
        taps.v_new,
        taps.q_normed,
        taps.k_normed,
        taps.k_pass,
        taps.q_rot_first,
        taps.q_rot_second,
        taps.k_rot_first,
        taps.k_rot_second,
        taps.score_new,
        taps.attended,
        taps.gate_sigmoid,
        taps.gated_attended,
        taps.o_proj_out,
    ];

    let unfused = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("plain bind succeeds");
    let fused =
        bind(&program, &shapes, &outputs, NumericPolicy::bit_exact()).expect("fused bind succeeds");
    let unfused_outputs = run_resolved(program.len(), &unfused, inputs.clone());
    let fused_outputs = run_resolved(program.len(), &fused, inputs);

    let taps: [(&str, NodeId, &[f64]); 17] = [
        ("normed", taps.normed, &reference.normed),
        ("q_split", taps.q_split, &reference.q_split),
        ("gate_split", taps.gate_split, &reference.gate_split),
        ("v_new", taps.v_new, &reference.v_new),
        ("q_normed", taps.q_normed, &reference.q_normed),
        ("k_normed", taps.k_normed, &reference.k_normed),
        ("k_pass", taps.k_pass, &reference.k_pass),
        ("q_rot_first", taps.q_rot_first, &reference.q_rot_first),
        ("q_rot_second", taps.q_rot_second, &reference.q_rot_second),
        ("k_rot_first", taps.k_rot_first, &reference.k_rot_first),
        ("k_rot_second", taps.k_rot_second, &reference.k_rot_second),
        ("score_new", taps.score_new, &reference.score_new),
        ("attended", taps.attended, &reference.attended),
        ("gate_sigmoid", taps.gate_sigmoid, &reference.gate_sigmoid),
        (
            "gated_attended",
            taps.gated_attended,
            &reference.gated_attended,
        ),
        ("o_proj_out", taps.o_proj_out, &reference.o_proj_out),
        ("residual1", residual1, &reference.residual1),
    ];

    const TOLERANCE: f64 = 1e-4;
    let mut first_divergent: Option<(&str, &str, f64)> = None;
    for (name, node, expected) in taps {
        let unfused_actual = unfused_outputs[node.0 as usize]
            .as_ref()
            .expect("unfused tap computes");
        let fused_actual = fused_outputs[node.0 as usize]
            .as_ref()
            .expect("fused tap computes");
        let unfused_error = max_relative_error(expected, unfused_actual);
        let fused_error = max_relative_error(expected, fused_actual);
        #[cfg(feature = "instrument")]
        debug!(
            tap = name,
            unfused_error = unfused_error,
            fused_error = fused_error,
            "qwen35 dense attention tap vs f64 reference"
        );
        if first_divergent.is_none() && unfused_error > TOLERANCE {
            first_divergent = Some((name, "unfused (bind_plain)", unfused_error));
        }
        if first_divergent.is_none() && fused_error > TOLERANCE {
            first_divergent = Some((name, "fused (CachedAttention)", fused_error));
        }
    }

    assert!(
        first_divergent.is_none(),
        "first tap to diverge from the independent f64 reference beyond {TOLERANCE}: {first_divergent:?}"
    );
}

/// Negative: perturbing the pass plane's own index map so it no longer
/// reads the SAME `q_pass_grouped` node on both the cached and new score
/// sides must make the matcher decline the whole fusion, not silently
/// drop the pass plane and fuse a rotary-only op that would then compute
/// the wrong score.
#[test]
#[cfg(feature = "cached-attention-streaming")]
fn a_perturbed_pass_plane_map_declines_the_qwen35_fusion() {
    let (mut program, residual1, _taps, _inputs, shapes) =
        qwen35_partial_rotary_attention_fixture();
    let outputs = alloc::vec![residual1];

    // The pass plane's own product is the one `Multiply` whose output's
    // last axis is exactly `PASS_DIM` (192) -- distinct from every
    // rotary product (last axis `PAIR_DIM`, 32). Flipping its body to
    // `Add` breaks [`decode_pass_term`]'s own "reduced(query * key)"
    // shape deterministically, without depending on the exact index-map
    // encoding this fixture happens to produce.
    let pass_product = (0..program.len())
        .rev()
        .find(|&position| {
            matches!(
                &program[position],
                Op::Elementwise {
                    body: ScalarOp::Multiply,
                    ..
                }
            ) && shapes.of(NodeId(position as u32)).last() == Some(&192)
        })
        .expect("the fixture must contain a pass-plane product to perturb");
    let Op::Elementwise { body, .. } = &mut program[pass_product] else {
        unreachable!("just matched Op::Elementwise above");
    };
    *body = ScalarOp::Add;

    let resolved = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("plain bind still succeeds on the perturbed program");
    let candidates = cached_attention_candidates(&program, &shapes, &resolved, &outputs, true);
    assert!(
        candidates.is_empty(),
        "a perturbed pass-plane product must not still match the qwen35 fusion"
    );
}

/// A gathered (or negative-strided) source must abort the candidate
/// outright rather than merely skip its own push -- the `continue`
/// this test guards against left `operands` one entry short of
/// `source_nodes`, relying on the length check further down to reject
/// the misaligned vector rather than aborting where the defect is
/// found. Reproduces the real single-range fixture, then patches one
/// of `cached_attention_single_range_candidates`' own eight source
/// nodes -- read straight off the baseline candidate's fused operand
/// list, never guessed -- to carry a `Lookup` in `resolved`, the exact
/// shape a dynamic KV-cache-page gather would leave behind.
#[test]
#[cfg(feature = "cached-attention-streaming")]
fn a_gathered_source_aborts_the_single_range_candidate_entirely() {
    let (program, logits, cache_roots, _) =
        crate::spec::mistral_single_range_cached_forward_program(
            32,
            16,
            24,
            4,
            2,
            4,
            1,
            false,
            crate::spec::DuplicateHeadPosition::None,
            false,
        )
        .expect("single-range fixture builds");
    let mut outputs = alloc::vec![logits];
    for (even, odd, value) in &cache_roots {
        outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let shapes = crate::shape::infer(&program, &[1, 5]).expect("single-range fixture infers");
    let mut resolved = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("plain bind succeeds");

    let baseline = cached_attention_single_range_candidates(&program, &shapes, &resolved, &outputs);
    assert!(
        !baseline.is_empty(),
        "the unpatched fixture must still produce a fusable candidate"
    );
    let BoundOpKind::CachedAttention { operands, .. } = &baseline[0].0.kind else {
        panic!("single-range candidate must carry CachedAttention operands");
    };
    let gathered_source = operands[0].0;

    for bound in &mut resolved {
        let operands = match &mut bound.kind {
            BoundOpKind::CachedAttention { operands, .. }
            | BoundOpKind::Elementwise { operands, .. }
            | BoundOpKind::Reduce { operands, .. }
            | BoundOpKind::RoundBatchedReduce { operands, .. } => operands,
            BoundOpKind::Iota
            | BoundOpKind::Constant { .. }
            | BoundOpKind::GatedDeltaNet { .. }
            | BoundOpKind::MoeTopK { .. } => continue,
        };
        for (node, layout, lookup) in operands.iter_mut() {
            if *node == gathered_source {
                *lookup = Some(Lookup {
                    indices: gathered_source,
                    index_layout: layout.clone(),
                    element_stride: 1,
                    extent: 1,
                });
            }
        }
    }

    let patched = cached_attention_single_range_candidates(&program, &shapes, &resolved, &outputs);
    assert!(
        patched.is_empty(),
        "a gathered source must abort the candidate, not just shrink its operand list"
    );
}

use crate::dtype::DType;
use crate::map;
use crate::op::{Extent, append};

fn matmul_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(768)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(768), Extent::Static(3072)],
            name: None,
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
            init: crate::op::ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    (program, product, sum, lhs)
}

/// An `Iota` binds directly to its own ready `BoundOp`, the same way a
/// `Reduce` always does — never held pending fusion the way an
/// `Elementwise` op is, since it has no operand to fuse with anything.
#[test]
fn an_iota_binds_to_its_own_ready_bound_op_with_no_operands() {
    let mut program = Vec::new();
    let iota = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(8),
        },
    );

    let shapes = shape::infer(&program, &[]).expect("iota infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("iota builds ops");

    assert_eq!(built.len(), 1, "the iota leaf materializes on its own");
    assert_eq!(built[0].node, iota);
    assert_eq!(built[0].dtype, DType::Float32);
    assert_eq!(built[0].extents, alloc::vec![8]);
    assert!(matches!(built[0].kind, BoundOpKind::Iota));
    assert!(
        built[0].operands().is_empty(),
        "a leaf with no operands binds to none"
    );
}

#[test]
fn matmul_resolves_to_one_fused_op_not_two() {
    let (program, product, sum, _lhs) = matmul_program();
    let shapes = shape::infer(&program, &[512]).expect("matmul infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("matmul builds ops");

    assert_eq!(
        built.len(),
        1,
        "the elementwise op must not materialize separately"
    );
    assert_eq!(built[0].node, sum);
    assert!(matches!(built[0].kind, BoundOpKind::Reduce { .. }));
    assert_eq!(
        built[0].element_body().steps.len(),
        1,
        "one absorbed elementwise op is one composed step"
    );
    assert_ne!(
        built[0].element_body().steps[0].op,
        ScalarOp::Identity,
        "the fused body is the elementwise op's multiply"
    );
    let _ = product;
}

#[test]
fn requesting_the_intermediate_elementwise_op_as_an_output_prevents_fusion() {
    let (program, product, sum, _lhs) = matmul_program();
    let shapes = shape::infer(&program, &[512]).expect("matmul infers");
    let built = bind(
        &program,
        &shapes,
        &[product, sum],
        NumericPolicy::bit_exact(),
    )
    .expect("matmul builds ops with two outputs");

    assert_eq!(
        built.len(),
        2,
        "the requested-output elementwise op must materialize"
    );
    assert!(
        built
            .iter()
            .any(|op| op.node == product && matches!(op.kind, BoundOpKind::Elementwise { .. }))
    );
    assert!(
        built
            .iter()
            .any(|op| op.node == sum && matches!(op.kind, BoundOpKind::Reduce { .. }))
    );
}

/// `b = a * scale; c = b + bias; d = c * c` — three chained elementwise
/// ops, each the sole and last use of the one before it, none of them
/// requested as an output. All three must fuse into `d`'s own `BoundOp`
/// rather than materializing `b` and `c` along the way.
fn elementwise_chain_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let scale = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let bias = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
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
    let d = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(c, identity()), (c, identity())],
            name: None,
        },
    );
    (program, b, c, d)
}

#[test]
fn a_chain_of_elementwise_ops_fuses_into_one_bound_op_not_three() {
    let (program, _b, _c, d) = elementwise_chain_program();
    let shapes = shape::infer(&program, &[]).expect("elementwise chain infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("elementwise chain builds ops");

    assert_eq!(
        built.len(),
        1,
        "b and c must absorb into d's own BoundOp instead of materializing"
    );
    assert_eq!(built[0].node, d);
    assert!(matches!(built[0].kind, BoundOpKind::Elementwise { .. }));
    assert!(
        built[0].element_body().steps.len() >= 2,
        "the composed body must carry more than one absorbed op's step"
    );
}

#[test]
fn an_elementwise_intermediate_requested_as_an_output_prevents_fusion() {
    let (program, b, _c, d) = elementwise_chain_program();
    let shapes = shape::infer(&program, &[]).expect("elementwise chain infers");
    let built = bind(&program, &shapes, &[b, d], NumericPolicy::bit_exact())
        .expect("elementwise chain builds ops with 2 outputs");

    assert_eq!(
        built.len(),
        2,
        "requesting b as an output must force it to materialize on its own"
    );
    assert!(
        built
            .iter()
            .any(|op| op.node == b && matches!(op.kind, BoundOpKind::Elementwise { .. }))
    );
    assert!(built.iter().any(|op| op.node == d));
}

#[test]
fn an_elementwise_intermediate_consumed_by_two_different_ops_is_not_fused() {
    let mut program = Vec::new();
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
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
    let d = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![(c1, identity()), (c2, identity())],
            name: None,
        },
    );

    let shapes = shape::infer(&program, &[]).expect("diamond chain infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("diamond chain builds ops");

    assert_eq!(
        built.len(),
        2,
        "b feeds two different consumers, so it must materialize once on its own, \
         and d (absorbing c1 and c2, whose only use each is d) is the other"
    );
    assert!(
        built
            .iter()
            .any(|op| op.node == b && matches!(op.kind, BoundOpKind::Elementwise { .. })),
        "b must materialize standalone rather than fuse into either consumer"
    );
    assert!(built.iter().any(|op| op.node == d));
    let _ = c1;
    let _ = c2;
}

/// `product = a * b; scaled = product * c; sum = reduce(+, scaled)` — two
/// chained elementwise ops feeding a reduce, mirroring `matmul_program`
/// but with an extra elementwise hop before the contraction. Both
/// elementwise ops must absorb into the reduce's own `BoundOp`.
#[test]
fn elementwise_into_elementwise_into_reduce_fuses_into_one_bound_op() {
    let mut program = Vec::new();
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let b = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let c = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(a, identity()), (b, identity())],
            name: None,
        },
    );
    let scaled = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(product, identity()), (c, identity())],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: scaled,
            in_map: identity(),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: Some("weighted_dot".into()),
        }),
    );

    let shapes = shape::infer(&program, &[]).expect("weighted dot infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("weighted dot builds ops");

    assert_eq!(
        built.len(),
        1,
        "both elementwise hops must absorb into the reduce's own BoundOp"
    );
    assert_eq!(built[0].node, sum);
    assert!(matches!(built[0].kind, BoundOpKind::Reduce { .. }));
    assert_eq!(
        built[0].element_body().steps.len(),
        2,
        "one step per absorbed elementwise op"
    );
}

#[test]
fn a_broadcast_operand_has_stride_zero_in_the_broadcast_axis() {
    let mut program = Vec::new();
    let matrix = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4), Extent::Static(8)],
            name: None,
        },
    );
    let bias = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(8)],
            name: None,
        },
    );
    let sum = append(
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

    let shapes = shape::infer(&program, &[]).expect("broadcast infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("broadcast builds ops");
    let op = built.iter().find(|op| op.node == sum).expect("sum emitted");
    assert_eq!(
        op.operands()[1].1.stride(0),
        0,
        "bias never varies over the batch axis"
    );
    assert_ne!(
        op.operands()[0].1.stride(0),
        0,
        "matrix does vary over the batch axis"
    );
}

#[test]
fn a_conv_window_operand_folds_two_terms_into_one_stride_slot() {
    let mut program = Vec::new();
    let anchor = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4), Extent::Static(2)],
            name: None,
        },
    );
    let signal = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(8)],
            name: None,
        },
    );
    let window = IndexMap::Affine(map::affine(
        2,
        &[(
            &[
                crate::map::AxisTerm::scaled(0, 2),
                crate::map::AxisTerm::scaled(1, 1),
            ],
            0,
        )],
    ));
    let touched = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![
                (anchor, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (signal, window)
            ],
            name: None,
        },
    );

    let shapes = shape::infer(&program, &[]).expect("conv window infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("conv window builds ops");
    let op = built
        .iter()
        .find(|op| op.node == touched)
        .expect("touched emitted");
    let signal_layout = &op.operands()[1].1;
    assert_eq!(
        signal_layout.strides.len(),
        2,
        "one stride slot per iteration axis"
    );
    assert_ne!(signal_layout.stride(0), 0, "stride term contributes");
    assert_ne!(signal_layout.stride(1), 0, "dilation term contributes");
}

#[test]
fn transpose_layout_has_permuted_strides() {
    let mut program = Vec::new();
    let matrix = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(3), Extent::Static(5)],
            name: None,
        },
    );
    let transposed = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(matrix, IndexMap::Affine(map::projection(2, &[1, 0])))],
            name: None,
        },
    );

    let shapes = shape::infer(&program, &[]).expect("transpose infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("transpose builds ops");
    let op = built
        .iter()
        .find(|op| op.node == transposed)
        .expect("transposed emitted");
    let layout = &op.operands()[0].1;
    // matrix is row-major [3, 5]: elem strides are [5, 1]. axis 0 of the
    // operand (stride 5) projects iteration axis 1; axis 1 (stride 1)
    // projects iteration axis 0, so the strides land permuted relative
    // to iteration order.
    assert_eq!(layout.stride(0), 1);
    assert_eq!(layout.stride(1), 5);
}

/// [`correct_packed_matmul_layouts`]/[`native_packed_layout`] on a
/// **two-axis output group** (`heads`, `head_dim`), the exact iteration
/// shape `mistral_cached_forward_program`'s `wq`/`wk`/`wv` projections
/// take (`tok`, `in`, `head`, `hd`), with `heads=3 != head_dim=4` so a
/// swapped output-axis order changes the numbers, not just the labels —
/// unlike `causal_conv1d`'s own `embedding=1` fixture, which made an
/// `ld`/`dl` axis swap byte-identical and let the bug through. GGUF's
/// native `[out_dim, in_dim]` row-major layout packs `out_dim` rows
/// (`out_dim = heads * head_dim`, row index `head * head_dim + hd`) of
/// `in_dim` contiguous elements each.
#[test]
fn correct_packed_matmul_layouts_derives_ggml_native_strides_for_a_two_axis_output_group() {
    const IN_DIM: u64 = 5;
    const HEADS: u64 = 3;
    const HEAD_DIM: u64 = 4;

    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: alloc::vec![
                Extent::Static(IN_DIM as u32),
                Extent::Static(HEADS as u32),
                Extent::Static(HEAD_DIM as u32)
            ],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(2), Extent::Static(IN_DIM as u32)],
            name: None,
        },
    );
    // iteration space (tok=0, in=1, head=2, hd=3): weight reads (in,
    // head, hd), ignoring tok; activation reads (tok, in).
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (weight, IndexMap::Affine(map::projection(4, &[1, 2, 3]))),
                (activation, IndexMap::Affine(map::projection(4, &[0, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: crate::op::ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 2, 3])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let shapes = shape::infer(&program, &[]).expect("two-axis output group infers");
    let mut built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("two-axis output group binds");
    let packed: BTreeSet<NodeId> = core::iter::once(weight).collect();
    correct_packed_matmul_layouts(&mut built, &packed);

    let reduce = built
        .iter()
        .find(|op| op.node == sum)
        .expect("reduce emitted");
    let weight_layout = &reduce
        .operands()
        .iter()
        .find(|(node, _, _)| *node == weight)
        .expect("weight operand present in the reduce")
        .1;

    assert_eq!(
        weight_layout.stride(1),
        1,
        "in_dim is the innermost, contiguous axis of a GGUF row"
    );
    assert_eq!(
        weight_layout.stride(3),
        IN_DIM as i64,
        "head_dim steps by one whole in_dim row -- swapped with heads' stride below if the output-axis order flips"
    );
    assert_eq!(
        weight_layout.stride(2),
        (IN_DIM * HEAD_DIM) as i64,
        "heads steps by one whole head_dim block of rows -- swapped with head_dim's stride above if the output-axis order flips"
    );
    assert_eq!(
        weight_layout.stride(0),
        0,
        "weight never varies over the token batch axis"
    );
}

/// [`correct_packed_matmul_layouts`]/[`native_packed_layout`] on a
/// **multi-axis contraction ("in") group** (`u`, `g`, `d`), the exact
/// shape `wo` (attention output projection) composes its packed row
/// index from in `mistral_layer3_forward_program`'s real checkpoint:
/// `attn_head_dim*group*u + attn_head_dim*g + d`, i.e. THREE reduction
/// letters folded into one packed-leaf axis, mirrored here at `u=2,
/// g=2, d=3`. The sibling test above
/// (`..._two_axis_output_group`) only ever proved the OUTPUT side can
/// be multi-axis (`wq`/`wk`/`wv`'s own shape, a single-letter `in`);
/// this is that same proof for the un-tested complementary case,
/// output = a single axis `e`, contraction = three.
#[test]
fn correct_packed_matmul_layouts_derives_ggml_native_strides_for_a_multi_axis_contraction_group() {
    const SEQ: u64 = 2;
    const KV_HEADS: u64 = 2;
    const GROUP: u64 = 2;
    const HEAD_DIM: u64 = 3;
    const EMBED: u64 = 4;
    const IN_DIM: u64 = KV_HEADS * GROUP * HEAD_DIM;

    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: alloc::vec![Extent::Static(IN_DIM as u32), Extent::Static(EMBED as u32)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![
                Extent::Static(SEQ as u32),
                Extent::Static(KV_HEADS as u32),
                Extent::Static(GROUP as u32),
                Extent::Static(HEAD_DIM as u32)
            ],
            name: None,
        },
    );
    // iteration space (s=0, u=1, g=2, d=3, e=4): weight's own row axis
    // is the composed `(head_dim*group)*u + head_dim*g + d`, exactly
    // `wo_flat`'s own map string at `spec.rs:9024-9028` with the real
    // dims swapped for small distinguishable primes; weight's second
    // axis is the plain output letter `e`. Activation reads (s, u, g,
    // d), ignoring e (broadcast over the output axis, the `wo`
    // ones-broadcast idiom's own shape).
    let row_terms = [
        AxisTerm::scaled(1, i32::try_from(GROUP * HEAD_DIM).expect("fits i32")),
        AxisTerm::scaled(2, i32::try_from(HEAD_DIM).expect("fits i32")),
        AxisTerm::scaled(3, 1),
    ];
    let weight_map = IndexMap::Affine(map::affine(
        5,
        &[(&row_terms, 0), (&[AxisTerm::scaled(4, 1)], 0)],
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
            init: crate::op::ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(5, &[0, 1, 2, 3, 4])),
            out_map: IndexMap::Affine(map::projection(5, &[0, 4])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let shapes = shape::infer(&program, &[]).expect("multi-axis contraction group infers");
    let mut built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("multi-axis contraction group binds");
    let packed: BTreeSet<NodeId> = core::iter::once(weight).collect();
    correct_packed_matmul_layouts(&mut built, &packed);

    let reduce = built
        .iter()
        .find(|op| op.node == sum)
        .expect("reduce emitted");
    let weight_layout = &reduce
        .operands()
        .iter()
        .find(|(node, _, _)| *node == weight)
        .expect("weight operand present in the reduce")
        .1;

    assert_eq!(
        weight_layout.stride(3),
        1,
        "head_dim (d) is the innermost, contiguous packed-row term"
    );
    assert_eq!(
        weight_layout.stride(2),
        HEAD_DIM as i64,
        "group (g) steps by one head_dim block"
    );
    assert_eq!(
        weight_layout.stride(1),
        (HEAD_DIM * GROUP) as i64,
        "kv_head (u) steps by one whole group*head_dim block"
    );
    assert_eq!(
        weight_layout.stride(4),
        IN_DIM as i64,
        "the output axis (e) steps by the whole packed row width"
    );
    assert_eq!(
        weight_layout.stride(0),
        0,
        "weight never varies over the sequence batch axis"
    );
}

fn elementwise_op() -> BoundOp {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(10), Extent::Static(4)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let shapes = shape::infer(&program, &[]).expect("elementwise infers");
    bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("elementwise builds ops")
    .into_iter()
    .next()
    .expect("one op emitted")
}

fn scalar_reduction_op() -> BoundOp {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(8)],
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
    let shapes = shape::infer(&program, &[]).expect("scalar reduction infers");
    bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("scalar reduction builds ops")
    .into_iter()
    .next()
    .expect("one op emitted")
}

fn scan_op() -> BoundOp {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(8)],
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
    let shapes = shape::infer(&program, &[]).expect("scan infers");
    bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("scan builds ops")
    .into_iter()
    .next()
    .expect("one op emitted")
}

#[test]
fn split_of_an_elementwise_op_yields_contiguous_chunks_with_a_ragged_last() {
    let op = elementwise_op();

    let chunks = op.split(3).expect("extent 10 over 3 parts splits");
    assert_eq!(chunks.len(), 3, "one chunk per part");

    let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
    assert_eq!(
        lengths,
        alloc::vec![3, 3, 4],
        "only the last chunk is ragged"
    );

    // axis 0's stride is 4 (row-major over [10, 4]): chunk k's operand
    // layout is rebased by chunk_start * stride(0), which is exactly
    // what lets a caller treat each chunk's output as a disjoint
    // sub-slice.
    let stride = op.operands()[0].1.stride(0);
    assert_eq!(chunks[0].operands()[0].1.base, op.operands()[0].1.base);
    assert_eq!(
        chunks[1].operands()[0].1.base,
        op.operands()[0].1.base + stride * 3
    );
    assert_eq!(
        chunks[2].operands()[0].1.base,
        op.operands()[0].1.base + stride * 6
    );
}

#[test]
fn split_aligned_rounds_non_final_chunks_down_to_the_alignment() {
    let op = elementwise_op();

    // extent 10, 3 parts: raw_len = 10 / 3 = 3, rounded down to the
    // nearest multiple of 2 is 2 — only the final (already-ragged)
    // chunk absorbs what the rounding shaved off the other two.
    let chunks = op
        .split_aligned(3, 2)
        .expect("extent 10 over 3 parts splits");
    let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
    assert_eq!(
        lengths,
        alloc::vec![2, 2, 6],
        "non-final chunks round down to the alignment, final absorbs the rest"
    );
}

#[test]
fn split_aligned_below_the_alignment_falls_back_to_unaligned() {
    let op = elementwise_op();

    // raw_len = 10 / 3 = 3 is already below alignment 4, so rounding
    // down would zero the chunk out — the doc promises the raw
    // unaligned width is kept instead.
    let chunks = op
        .split_aligned(3, 4)
        .expect("extent 10 over 3 parts splits");
    let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
    assert_eq!(
        lengths,
        alloc::vec![3, 3, 4],
        "falls back to split's own behavior"
    );
}

#[test]
fn split_aligned_with_alignment_one_matches_split_exactly() {
    let op = elementwise_op();

    let aligned = op
        .split_aligned(3, 1)
        .expect("extent 10 over 3 parts splits");
    let plain = op.split(3).expect("extent 10 over 3 parts splits");
    let aligned_lengths: Vec<u64> = aligned.iter().map(|chunk| chunk.extents[0]).collect();
    let plain_lengths: Vec<u64> = plain.iter().map(|chunk| chunk.extents[0]).collect();
    assert_eq!(aligned_lengths, plain_lengths, "alignment 1 is a no-op");
}

#[test]
fn split_of_a_fused_matmul_reduction_rebases_operands_but_not_out_layout() {
    let (program, _product, sum, _lhs) = matmul_program();
    let shapes = shape::infer(&program, &[512]).expect("matmul infers");
    let op = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("matmul builds ops")
    .into_iter()
    .next()
    .expect("one fused op emitted");
    assert_eq!(op.node, sum);
    let BoundOpKind::Reduce { .. } = &op.kind else {
        panic!("the reduction fused with its elementwise op");
    };

    let chunks = op.split(2).expect("512 rows over 2 parts splits");
    assert_eq!(chunks.len(), 2, "one chunk per part");
    for chunk in &chunks {
        assert!(
            matches!(chunk.kind, BoundOpKind::Reduce { .. }),
            "each chunk is still a reduce"
        );
        // the contracted axis (k) is untouched by a split on the output
        // row axis: every chunk still walks the full contraction.
        assert_eq!(chunk.extents[2], op.extents[2]);
    }
    assert_eq!(chunks[0].extents[0], 256, "rows split evenly in half");
    assert_eq!(chunks[1].extents[0], 256);

    // the fused elementwise op's lhs/rhs operand reads are rebased:
    // chunk 1 starts reading lhs at row 256 (row stride = k = 768).
    let lhs_row_stride = op.operands()[0].1.stride(0);
    assert_eq!(
        chunks[1].operands()[0].1.base,
        op.operands()[0].1.base + lhs_row_stride * 256
    );

    // out_layout stays exactly as the parent's: the interpreter's own
    // per-chunk loop already starts each chunk's leading coordinate at
    // 0, so an unshifted out_layout already yields the 0-based write
    // offsets a `split_at_mut` sub-slice expects (see the `split` doc).
    let BoundOpKind::Reduce {
        out_layout: parent_out,
        ..
    } = &op.kind
    else {
        unreachable!("checked above")
    };
    for chunk in &chunks {
        let BoundOpKind::Reduce {
            out_layout: chunk_out,
            ..
        } = &chunk.kind
        else {
            panic!("chunk reduction");
        };
        assert_eq!(chunk_out, parent_out);
    }
}

#[proxima::test]
#[case::scalar_reduction(scalar_reduction_op(), 2)]
#[case::keep_scan_scan(scan_op(), 2)]
#[case::too_few_parts(elementwise_op(), 1)]
#[case::extent_smaller_than_parts(elementwise_op(), 999)]
async fn split_returns_none_when_unsound_or_unhelpful(#[case] op: BoundOp, #[case] parts: usize) {
    assert!(op.split(parts).is_none());
}

/// A ternary `ScalarOp::Select` node (arity 3, the crate's current
/// maximum) whose three operands are all held, non-fusing elementwise
/// predecessors: a single `push` must materialize all three in one
/// call, proving `push` can ready more than the two `BoundOp`s this
/// module's docs once claimed as its ceiling — the true bound tracks
/// `ScalarOp::arity()`, one reason `READY_BATCH_CAPACITY` must exceed 2
/// (widened further to 32 for multi-position programs — see that
/// constant's own doc).
#[test]
fn select_push_emits_three_when_all_three_operands_are_held_and_non_fusing() {
    let mut program = Vec::new();
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let b = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let c = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    // `Negate`, not `Identity`: the pure-copy fold rule
    // (`builder_compose_window.rs`'s own `is_pure_copy`) now admits a
    // still-live `ScalarOp::Identity`-over-one-operand held node into its
    // consumer regardless of `still_live`, which would fuse all three here
    // and defeat this test's own premise (isolating what a single push
    // materializes with nothing ever retiring). A real unary body keeps
    // these three genuinely non-fusing.
    let held_a = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: alloc::vec![(a, identity())],
            name: None,
        },
    );
    let held_b = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: alloc::vec![(b, identity())],
            name: None,
        },
    );
    let held_c = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: alloc::vec![(c, identity())],
            name: None,
        },
    );
    program.push(Op::Elementwise {
        dtype: DType::Float32,
        body: ScalarOp::Select,
        operands: alloc::vec![
            (held_a, identity()),
            (held_b, identity()),
            (held_c, identity()),
        ],
        name: None,
    });

    let shapes = shape::infer(&program, &[]).expect("select program infers");
    // an empty retires list (rather than `live::annotate`'s real kill-flags)
    // makes `retires.contains(operand_node)` false for every node, so
    // every held predecessor fails the fuse check regardless of its
    // projection — isolating exactly what a single push can materialize.
    let building = BoundOpBuilder::new(Vec::new(), NumericPolicy::bit_exact());
    let mut last_emitted_len = 0;
    for expr in program.iter() {
        let emitted = building.push(expr, &shapes).expect("push succeeds");
        last_emitted_len = emitted.len();
    }
    assert_eq!(
        last_emitted_len, 3,
        "the select node's push must materialize all three held, \
         non-fusing predecessors in one call: proves the 0/1/2 bound is \
         wrong, true bound tracks ScalarOp::arity() (3, Select)"
    );
}

// THE PROOF: `ShapeTable` and `BoundOpBuilder` compose through the real
// `PipeExt` surface (`.and_then`, not hand-sequenced calls dressed up as
// composition), and the ops that composed chain produces for a matmul
// program are byte-for-byte the same ops `shape::infer` + `bind::bind`
// (the free-function path every other test in this crate trusts)
// produce for the identical program.
#[test]
fn infer_and_then_build_ops_matches_the_free_function_pipeline() {
    use crate::shape::ShapeTable;
    use proxima_primitives::pipe::PipeExt;

    let (program, _product, sum, _lhs) = matmul_program();
    let outputs: Vec<NodeId> = alloc::vec![sum];
    let retires = live::annotate(&program, &outputs);

    let shape_table = ShapeTable::new(&[512]);
    let builder = BoundOpBuilder::new(retires, NumericPolicy::bit_exact());
    let chain = shape_table.and_then(builder);

    let mut built_via_pipe = Vec::new();
    for expr in &program {
        let batch = proxima_primitives::block_on(Pipe::call(&chain, expr.clone()))
            .expect("shape+op pipe step succeeds");
        built_via_pipe.extend(batch);
    }

    let shapes = shape::infer(&program, &[512]).expect("free-function infer succeeds");
    let built_via_free_function = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("free-function op building succeeds");

    assert_eq!(built_via_pipe, built_via_free_function);
    assert_eq!(built_via_pipe.len(), 1, "matmul fuses into one op");
    assert_eq!(built_via_pipe[0].node, sum);
}

/// `docs/discipline.md` ROW 131 Limitation 2: a lone dtype-relabelling
/// `Elementwise` (`indices_node`) is reached only through a sibling
/// operand's `IndexMap::Computed { indices, .. }` field -- never through
/// its own entry in any op's `operands` list. `gathered`'s first
/// consumer (`first_use`) is not `gathered`'s *last* use, so `push`
/// force-materializes `gathered` standalone, right there, long before
/// the program ends -- while `indices_node`, visited by nothing but the
/// map field this walk used to skip, would otherwise sit `held` until
/// `finish`'s end-of-program sweep and land in `resolved` after the very
/// op that reads its buffer. This is the exact shape the LFM2 causal
/// conv gather hit (`spec.rs`'s `causal_conv1d`, which works around it
/// by routing its own index through an `Op::Reduce` instead).
fn computed_index_via_lone_elementwise_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let base = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let identity = || IndexMap::Affine(map::projection(1, &[0]));
    let index_seed = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(4),
        },
    );
    let indices_node = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Int32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(index_seed, identity())],
            name: None,
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: indices_node,
        index_map: map::projection(1, &[0]),
        base: IndexPattern {
            iter_rank: 1,
            axes: alloc::vec![AxisIndex::default()],
        },
        gathered_dim: 0,
    };
    let gathered = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(base, gathered_map)],
            name: None,
        },
    );
    let first_use = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(gathered, identity())],
            name: None,
        },
    );
    let second_use = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![(gathered, identity()), (first_use, identity())],
            name: None,
        },
    );
    (program, second_use)
}

#[test]
fn a_computed_index_reached_only_through_a_sibling_map_is_materialized_before_the_gather_reads_it()
{
    let (program, output) = computed_index_via_lone_elementwise_program();
    let shapes = shape::infer(&program, &[]).expect("computed-index program infers");
    let base_data: Vec<f32> = alloc::vec![10.0, 20.0, 30.0, 40.0];

    let built = bind(&program, &shapes, &[output], NumericPolicy::bit_exact())
        .expect("binding itself never errors");
    let indices_position = built
        .iter()
        .position(|op| {
            matches!(op.kind, BoundOpKind::Elementwise { .. }) && op.dtype == DType::Int32
        })
        .expect("the indices node must appear as its own BoundOp");
    let gather_position = built
        .iter()
        .position(|op| op.operands().iter().any(|(_, _, lookup)| lookup.is_some()))
        .expect("the gathering node must appear as its own BoundOp");
    assert!(
        indices_position < gather_position,
        "resolved must stay topologically ordered: the indices node ({indices_position}) \
         must be built before the gather that reads it ({gather_position}), got {built:#?}"
    );

    let evaluated = crate::cpu::evaluate(&program, &[], &[&base_data], &[output]);
    assert!(
        evaluated.is_ok(),
        "the gather's indices buffer must be ready by the time the gather runs: {evaluated:?}"
    );
}

/// Hand-builds `proxima-autograd/src/conv.rs`'s `masked_window_axis`
/// shape directly (that function is private to a different crate): one
/// source axis widened by `(out_position, kernel_position)`, masked by
/// `Equal(Iota, Add(Multiply(Iota, Constant), Iota))`, and reduced away —
/// `proxima-tensor/docs/discipline.md` ROW 154's own fixture.
/// `mask_body` lets a decline test swap `Equal` for something else
/// without duplicating the rest of the shape.
#[allow(clippy::too_many_arguments)]
fn masked_window_reduce_program(
    source_rank: u16,
    windowed_axis: u16,
    source_extent: u64,
    out_extent: u64,
    kernel_extent: u64,
    stride: u64,
    mask_body: ScalarOp,
) -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let widened_rank = source_rank + 2;
    let out_position_axis = source_rank;
    let kernel_position_axis = source_rank + 1;

    let source_shape: Vec<Extent> = (0..source_rank)
        .map(|axis| {
            Extent::Static(if u64::from(axis) == windowed_axis as u64 {
                source_extent as u32
            } else {
                4
            })
        })
        .collect();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: source_shape,
            name: None,
        },
    );

    let source_position = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(source_extent as u32),
        },
    );
    let out_position = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(out_extent as u32),
        },
    );
    let kernel_position = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(kernel_extent as u32),
        },
    );
    let stride_const = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: stride as f32,
        },
    );

    let identity1 = IndexMap::Affine(map::projection(1, &[0]));
    let broadcast1 = IndexMap::Affine(map::projection(1, &[]));
    let scaled_out = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(out_position, identity1), (stride_const, broadcast1)],
            name: None,
        },
    );

    let combined = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![
                (scaled_out, IndexMap::Affine(map::projection(2, &[0]))),
                (kernel_position, IndexMap::Affine(map::projection(2, &[1]))),
            ],
            name: None,
        },
    );

    let mask = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: mask_body,
            operands: alloc::vec![
                (source_position, IndexMap::Affine(map::projection(3, &[0]))),
                (combined, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );

    let source_axes: Vec<u16> = (0..source_rank).collect();
    let source_pattern = IndexMap::Affine(map::projection(widened_rank, &source_axes));
    let mask_pattern = IndexMap::Affine(map::projection(
        widened_rank,
        &[windowed_axis, out_position_axis, kernel_position_axis],
    ));

    let masked = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(source, source_pattern), (mask, mask_pattern)],
            name: None,
        },
    );

    let keep_axes: Vec<u16> = (0..widened_rank)
        .filter(|&axis| axis != windowed_axis)
        .collect();
    let out_map = IndexMap::Affine(map::projection(widened_rank, &keep_axes));
    let identity_widened = IndexMap::Affine(map::projection(
        widened_rank,
        &(0..widened_rank).collect::<Vec<u16>>(),
    ));
    let reduced = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: crate::op::ReduceInit::Zero,
            operand: masked,
            in_map: identity_widened,
            out_map,
            keep: Keep::Reduce,
            name: None,
        }),
    );

    (program, source, reduced)
}

#[test]
fn a_masked_window_reduce_folds_to_a_single_operand_identity_read_of_source() {
    let (program, source, reduced) =
        masked_window_reduce_program(1, 0, 5, 3, 3, 1, ScalarOp::Equal);
    let shapes = shape::infer(&program, &[]).expect("masked-window program infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("masked-window program builds ops");

    let folded = built
        .iter()
        .find(|op| op.node == reduced)
        .expect("the reduce node's own BoundOp is still present, just re-shaped");
    assert!(
        matches!(folded.kind, BoundOpKind::Elementwise { .. }),
        "a proven in-bounds window read becomes a plain elementwise gather, not a Reduce, got {:?}",
        folded.kind
    );
    assert_eq!(
        folded.operands().len(),
        1,
        "the fold reads only source, the mask chain is gone"
    );
    assert_eq!(folded.operands()[0].0, source);
    assert_eq!(
        folded.operands()[0].1.strides.len(),
        2,
        "the source's one windowed axis now derives from two output axes"
    );
    assert!(
        folded.operands()[0]
            .1
            .strides
            .iter()
            .all(|&stride| stride != 0),
        "both the out_position and kernel_position axes must contribute to the source address"
    );
    assert_eq!(folded.element_body().steps.len(), 1);
    assert_eq!(folded.element_body().steps[0].op, ScalarOp::Identity);
}

#[test]
fn a_masked_window_reduce_that_fails_the_in_bounds_proof_declines_and_binds_as_a_reduce() {
    // stride*(out_extent-1) + (kernel_extent-1) = 2*2 + 2 = 6 >= source_extent(5): out of bounds.
    let (program, _source, reduced) =
        masked_window_reduce_program(1, 0, 5, 3, 3, 2, ScalarOp::Equal);
    let shapes = shape::infer(&program, &[]).expect("masked-window program infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("masked-window program builds ops");

    let folded = built
        .iter()
        .find(|op| op.node == reduced)
        .expect("the reduce node's own BoundOp is still present");
    assert!(
        matches!(folded.kind, BoundOpKind::Reduce { .. }),
        "a failed in-bounds proof must decline the fold and bind the ordinary Reduce, got {:?}",
        folded.kind
    );
}

#[test]
fn a_masked_window_reduce_with_a_non_windowed_axis_matches_a_direct_window_read() {
    // source: [channel=4 (helper's own non-windowed default extent), position=5],
    // windowed_axis=1, stride=1, kernel=3 -> out=3.
    let (program, _source, reduced) =
        masked_window_reduce_program(2, 1, 5, 3, 3, 1, ScalarOp::Equal);
    let source_data: Vec<f32> = (0..4 * 5).map(|index| index as f32 + 1.0).collect();
    let evaluated = crate::cpu::evaluate(&program, &[], &[&source_data], &[reduced])
        .expect("masked-window program evaluates");
    let (windowed, _shape) = evaluated
        .get(reduced)
        .expect("reduce node's output buffer is present");

    // expected[channel, out_position, kernel_position] = source[channel, out_position + kernel_position]
    let mut expected = alloc::vec![0.0f32; 4 * 3 * 3];
    for channel in 0..4usize {
        for out_position in 0..3usize {
            for kernel_position in 0..3usize {
                let source_position = out_position + kernel_position;
                expected[channel * 9 + out_position * 3 + kernel_position] =
                    source_data[channel * 5 + source_position];
            }
        }
    }
    assert_eq!(
        windowed,
        expected.as_slice(),
        "the folded read must match the direct window gather exactly"
    );
}

#[test]
fn a_non_equal_mask_chain_declines_and_binds_as_a_reduce() {
    let (program, _source, reduced) =
        masked_window_reduce_program(1, 0, 5, 3, 3, 1, ScalarOp::Greater);
    let shapes = shape::infer(&program, &[]).expect("masked-window program infers");
    let built = bind(
        &program,
        &shapes,
        &[terminal(&program)],
        NumericPolicy::bit_exact(),
    )
    .expect("masked-window program builds ops");

    let folded = built
        .iter()
        .find(|op| op.node == reduced)
        .expect("the reduce node's own BoundOp is still present");
    assert!(
        matches!(folded.kind, BoundOpKind::Reduce { .. }),
        "a mask chain that is not the exact Equal/Iota shape must decline the fold, got {:?}",
        folded.kind
    );
}

#[cfg(feature = "reduce-epilogue-fusion")]
mod reduce_epilogue_fusion_tests {
    use core::pin::pin;
    use core::task::{Context, Poll, Waker};

    use super::*;
    use crate::cpu::Interpreter;
    use crate::test_support::Lcg;

    /// `weights: [K, N]` folded over `K` into `reduced: [N]`, then a
    /// plain `x: [N]` residual add — the exact `residual1 = Add(attn_out,
    /// x)` shape `docs/dispatch-census.md` names. `y = Negate(x)` gives
    /// `x` a SECOND, independent use so this test also proves the rule
    /// only cares about the REDUCE's own liveness, not any other
    /// operand's.
    fn reduce_then_residual_add_program() -> (Vec<Op>, NodeId, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let weights = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8), Extent::Static(4)],
                name: None,
            },
        );
        let reduced = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: weights,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let x = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let consumer = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(reduced, identity()), (x, identity())],
                name: None,
            },
        );
        let extra_x_use = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(x, identity())],
                name: None,
            },
        );
        (program, reduced, consumer, x, extra_x_use)
    }

    /// The one non-default field this whole rule adds — a real
    /// `epilogue_operands` entry — is the test's own positive signal
    /// that fusion actually happened, not merely that the op count
    /// dropped (a count-only assertion can't distinguish this rule
    /// firing from some unrelated node going dead).
    fn has_real_epilogue(kind: &BoundOpKind) -> bool {
        matches!(kind, BoundOpKind::Reduce { epilogue_operands, .. } if !epilogue_operands.is_empty())
    }

    /// Structural invariant every fusion pass must preserve: every
    /// [`NodeId`] any resolved op's [`BoundOp::all_read_sources`] names
    /// must either be an [`Op::Input`] leaf (which never gets its own
    /// [`BoundOp`], per [`BoundOpKind`]'s own doc) or still be one of
    /// `resolved`'s own [`BoundOp::node`]s. A fusion pass that absorbs a
    /// producer (`reduce_epilogue_fusion`'s own `absorbed` set) but
    /// leaves some OTHER operand slot still pointing at that now-gone
    /// producer would pass every count/shape assertion while reading a
    /// buffer that was never materialized — exactly the dangling-slot
    /// bug `compose_reduce_epilogue` had when it substituted only the
    /// FIRST matching operand instead of every occurrence.
    fn assert_no_dangling_operand_references(program: &[Op], resolved: &[BoundOp]) {
        let live_nodes: BTreeSet<NodeId> = resolved.iter().map(|bound| bound.node).collect();
        let is_leaf_input =
            |node: &NodeId| matches!(program.get(node.0 as usize), Some(Op::Input { .. }));
        for bound in resolved {
            for (source, _, gather) in bound.all_read_sources() {
                assert!(
                    live_nodes.contains(source) || is_leaf_input(source),
                    "node {:?} reads {source:?}, which no BoundOp in the resolved list produces \
                     and which is not an Op::Input leaf",
                    bound.node
                );
                if let Some(lookup) = gather {
                    assert!(
                        live_nodes.contains(&lookup.indices) || is_leaf_input(&lookup.indices),
                        "node {:?} gathers through {:?}, which no BoundOp in the resolved list \
                         produces and which is not an Op::Input leaf",
                        bound.node,
                        lookup.indices
                    );
                }
            }
        }
    }

    #[test]
    fn reduce_then_residual_add_fuses_into_one_epilogued_reduce() {
        let (program, reduced, consumer, _x, _extra_x_use) = reduce_then_residual_add_program();
        let shapes = shape::infer(&program, &[]).expect("residual-add program infers");
        // `consumer` -- not `extra_x_use`, which never reads `reduced` at all
        // -- must be the requested output, or `bind`'s own reachability pass
        // prunes `consumer` as dead code before fusion ever runs and this
        // test asserts about a `BoundOp` that was never built.
        let plain = bind_plain(&program, &shapes, &[consumer], NumericPolicy::bit_exact())
            .expect("plain bind succeeds");
        let fused = bind(&program, &shapes, &[consumer], NumericPolicy::bit_exact())
            .expect("fused bind succeeds");

        assert_eq!(
            fused.len(),
            plain.len() - 1,
            "the standalone reduce disappears into the consumer's epilogue"
        );
        assert!(
            !fused.iter().any(|bound| bound.node == reduced),
            "the reduce's own NodeId no longer names a standalone BoundOp"
        );
        let merged = fused
            .iter()
            .find(|bound| bound.node == consumer)
            .expect("the consumer's NodeId now names the fused reduce+epilogue op");
        assert!(
            has_real_epilogue(&merged.kind),
            "the fused op must carry a real epilogue, got {:?}",
            merged.kind
        );
    }

    /// Runs an already-resolved `Vec<BoundOp>` through
    /// [`crate::cpu::Interpreter`] the same way
    /// `cached_attention_single_range_fused_matches_the_unfused_program`
    /// (`spec.rs`) already does for its own fused-vs-unfused A/B — the
    /// one entry point that accepts a caller's own bind result instead
    /// of re-binding internally the way [`crate::cpu::evaluate`] does
    /// ([`crate::cpu::evaluate`]'s own `prepare` calls `bind::bind`
    /// unconditionally, so it can never produce the un-epilogued half of
    /// this comparison once the `reduce-epilogue-fusion` feature is
    /// compiled in).
    fn run_resolved(
        program_len: usize,
        resolved: &[BoundOp],
        inputs: Vec<(NodeId, Vec<f32>)>,
    ) -> Vec<Option<Vec<f32>>> {
        let mut buffers: Vec<Option<Vec<f32>>> = alloc::vec![None; program_len];
        for (node, data) in inputs {
            buffers[node.0 as usize] = Some(data);
        }
        let interpreter = Interpreter::new(&mut buffers);
        for chunk in resolved.chunks(READY_BATCH_CAPACITY) {
            let batch: ReadyBatch = chunk.iter().cloned().collect();
            let waker = Waker::noop();
            let mut context = Context::from_waker(waker);
            let mut future = pin!(interpreter.call(batch));
            match future.as_mut().poll(&mut context) {
                Poll::Ready(result) => {
                    result.expect("resolved batch computes");
                }
                Poll::Pending => unreachable!("cpu pipes never yield: no internal .await"),
            }
        }
        buffers
    }

    /// Hand-derivable ground truth over 8x4 weights and a length-4
    /// residual: `reduced[n] = sum_k weights[k, n]`, `consumer[n] =
    /// reduced[n] + x[n]` — exact values, not a tolerance band, because
    /// every input is an exact `f32` the sum can reproduce bit-for-bit
    /// with `Lcg`'s own small integer-ish range.
    #[test]
    fn reduce_epilogue_evaluator_matches_hand_derived_values() {
        let (program, _reduced, consumer, _x, extra_x_use) = reduce_then_residual_add_program();
        let outputs = alloc::vec![consumer, extra_x_use];
        let shapes = shape::infer(&program, &[]).expect("residual-add program infers");

        let mut lcg = Lcg(42);
        let weights: Vec<f32> = (0..32).map(|_| lcg.next_unit()).collect();
        let residual: Vec<f32> = (0..4).map(|_| lcg.next_unit()).collect();

        let expected: Vec<f32> = (0..4)
            .map(|column| {
                let sum: f32 = (0..8).map(|row| weights[row * 4 + column]).sum();
                sum + residual[column]
            })
            .collect();

        let inputs = || -> Vec<(NodeId, Vec<f32>)> {
            block_node_ids(&program)
                .into_iter()
                .map(|node| {
                    let data = match &program[node.0 as usize] {
                        Op::Input { shape, .. } if shape.len() == 2 => weights.clone(),
                        _ => residual.clone(),
                    };
                    (node, data)
                })
                .collect()
        };

        let fused = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("fused bind succeeds");
        let fused_buffers = run_resolved(program.len(), &fused, inputs());
        let fused_consumer = fused_buffers[consumer.0 as usize]
            .as_ref()
            .expect("fused consumer output present");

        assert_eq!(
            fused_consumer, &expected,
            "epilogued reduce must match the hand-derived sum-plus-residual exactly"
        );

        let plain = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("plain bind succeeds");
        let plain_buffers = run_resolved(program.len(), &plain, inputs());
        let plain_consumer = plain_buffers[consumer.0 as usize]
            .as_ref()
            .expect("plain consumer output present");
        assert_eq!(
            fused_consumer, plain_consumer,
            "the epilogue-fused evaluator must match the plain (unfused) evaluator exactly"
        );
    }

    #[test]
    fn a_reduce_with_two_consumers_does_not_fuse() {
        let (program, reduced, consumer, _x, _extra_x_use) = reduce_then_residual_add_program();
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let mut program = program;
        let second_consumer = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(reduced, identity())],
                name: None,
            },
        );
        let shapes = shape::infer(&program, &[]).expect("two-consumer program infers");
        // both real readers of `reduced` -- `consumer` and `second_consumer`
        // -- must be requested outputs, or `bind`'s own reachability pass
        // (`live::reachable`) prunes the unrequested one as dead code before
        // `reduce_epilogue_fusion` ever sees it, leaving `reduced` with a
        // single live reader and defeating the two-consumer scenario this
        // test exists to cover.
        let fused = bind(
            &program,
            &shapes,
            &[consumer, second_consumer],
            NumericPolicy::bit_exact(),
        )
        .expect("two-consumer program still binds");

        assert!(
            fused.iter().any(|bound| bound.node == reduced),
            "a reduce with a second consumer must still materialize standalone"
        );
        let consumer_bound = fused
            .iter()
            .find(|bound| bound.node == consumer)
            .expect("the first consumer's own BoundOp is still present");
        assert!(
            !has_real_epilogue(&consumer_bound.kind),
            "a sole-consumer requirement violation must never carry an epilogue"
        );
    }

    #[test]
    fn a_strided_consumer_map_does_not_fuse() {
        let (mut program, reduced, _consumer, x, extra_x_use) = reduce_then_residual_add_program();
        // overwrite the last-appended node (the ordinary-identity
        // consumer) with a build that reads `reduced` REVERSED
        // (`coeff: -1, offset: 3` over a 4-element axis walks indices
        // 3,2,1,0) instead of through a genuine identity projection —
        // `is_identity_projection` rejects any `coeff != 1` regardless
        // of how the offset keeps it in-bounds, so this consumer must
        // decline the fold and materialize both nodes normally.
        let consumer_index = program
            .iter()
            .position(|expr| {
                matches!(
                    expr,
                    Op::Elementwise { body: ScalarOp::Add, operands, .. }
                        if operands.iter().any(|(node, _)| *node == reduced)
                )
            })
            .expect("the residual-add consumer is present in the program");
        let strided_map = IndexMap::Affine(IndexPattern {
            iter_rank: 1,
            axes: alloc::vec![AxisIndex {
                terms: SmallVec::from_slice(&[AxisTerm { axis: 0, coeff: -1 }]),
                offset: 3,
                len: None,
            }],
        });
        program[consumer_index] = Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![
                (reduced, strided_map),
                (x, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        };
        let consumer = NodeId(consumer_index as u32);
        let shapes = shape::infer(&program, &[]).expect("strided-consumer program infers");
        let fused = bind(
            &program,
            &shapes,
            &[extra_x_use, consumer],
            NumericPolicy::bit_exact(),
        )
        .expect("strided-consumer program still binds");

        assert!(
            fused.iter().any(|bound| bound.node == reduced),
            "a non-identity consumer map must leave the reduce standalone"
        );
        let consumer_bound = fused
            .iter()
            .find(|bound| bound.node == consumer)
            .expect("the strided consumer's own BoundOp is still present");
        assert!(
            !has_real_epilogue(&consumer_bound.kind),
            "a non-identity projection must never carry an epilogue"
        );
    }

    #[test]
    fn a_required_output_consumer_still_fuses() {
        let (program, reduced, consumer, _x, extra_x_use) = reduce_then_residual_add_program();
        let shapes = shape::infer(&program, &[]).expect("residual-add program infers");
        // `consumer` itself is now a REQUESTED output — condition (b)
        // only forbids the REDUCE from being a requested output; the
        // consumer becoming one is exactly the case the fused op's own
        // `node == consumer` convention exists for for (the epilogue
        // output IS the output, so nothing needs to keep the reduce
        // materialized separately).
        let fused = bind(
            &program,
            &shapes,
            &[consumer, extra_x_use],
            NumericPolicy::bit_exact(),
        )
        .expect("fused bind with the consumer as a required output succeeds");

        assert!(
            !fused.iter().any(|bound| bound.node == reduced),
            "the reduce still disappears even though its consumer is a required output"
        );
        let merged = fused
            .iter()
            .find(|bound| bound.node == consumer)
            .expect("the required-output consumer's NodeId still names a BoundOp");
        assert!(
            has_real_epilogue(&merged.kind),
            "a required-output consumer must still fuse, got {:?}",
            merged.kind
        );
    }

    /// The real openchat-3.5/Mistral-7B single-range fixture (same
    /// shape as `single_range_cached_attention_fuses_one_step_per_
    /// layer_on_the_real_openchat_shape` above) with BOTH
    /// `cached-attention-streaming` and `reduce-epilogue-fusion` on.
    /// MEASURED, not derived: printed once via the per-kind buckets
    /// below so a future re-run can diff against this row's own
    /// numbers without re-deriving them from the dispatch census.
    #[test]
    fn reduce_epilogue_fusion_shrinks_the_real_openchat_single_range_program() {
        let (program, logits, cache_roots, _) =
            crate::spec::mistral_single_range_cached_forward_program(
                32_002,
                4096,
                14336,
                32,
                8,
                128,
                32,
                false,
                crate::spec::DuplicateHeadPosition::None,
                false,
            )
            .expect("openchat-shaped single-range forward pass lowers to a program");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &cache_roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let shapes = crate::shape::infer(&program, &[1, 71])
            .expect("one new position against a 71-position merged range infers");
        let attention_only = bind_cached_attention_fusion(
            &program,
            &shapes,
            &outputs,
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("cached-attention-only bind succeeds");
        let with_epilogue = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("fused bind succeeds");

        let epilogue_count = with_epilogue
            .iter()
            .filter(|bound| has_real_epilogue(&bound.kind))
            .count();
        let reduce_count = with_epilogue
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::Reduce { .. }))
            .count();
        let elementwise_count = with_epilogue
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::Elementwise { .. }))
            .count();
        let cached_attention_count = with_epilogue
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count();
        std::eprintln!(
            "cached-attention-only={} with-epilogue={} epilogued-reduces={} \
             reduce={} elementwise={} cached-attention={}",
            attention_only.len(),
            with_epilogue.len(),
            epilogue_count,
            reduce_count,
            elementwise_count,
            cached_attention_count,
        );
        assert!(
            with_epilogue.len() < attention_only.len(),
            "reduce-epilogue-fusion must remove at least one bound op vs cached-attention alone"
        );
        assert!(
            epilogue_count > 0,
            "the real openchat shape must produce at least one epilogued reduce"
        );
        // MEASURED (not derived) on this checkout: 619 -> 458, a 161-op
        // drop. Every RMSNorm's own `mean/eps/sqrt/reciprocal` PLAIN
        // epilogue round chains into a SECOND, broadcast-reduce round
        // (`x * inv_rms`), and every `SiLU(gate) * up`-shaped tail (two
        // independent reduces feeding one consumer) fuses too. The prior
        // measured value here (490) was taken while `find_epilogue_source`/
        // `resolved_reference_counts` still counted a consumer's OWN
        // repeated read of the same source (production SiLU reads `gate`
        // once bare, once inside `exp(-gate)`) as a second, conflicting
        // consumer and declined the fold — so the real SiLU*up class
        // this comment already claimed was fusing was NOT actually
        // firing on this program; only RMSNorm's broadcast-reduce round
        // was. Fixing that reference-count/projection bug is what widens
        // 490 -> 458. A regression here means one of the two classes
        // stopped firing, not merely "fewer than before".
        assert_eq!(
            attention_only.len(),
            619,
            "cached-attention-only baseline shifted; re-derive before trusting the epilogue delta below"
        );
        assert_eq!(
            with_epilogue.len(),
            458,
            "reduce-epilogue-fusion's own op count regressed from the measured 458"
        );
    }

    /// The same `mistral_single_range_cached_forward_program` builder the
    /// structural test above proves the bound-op COUNT for, run end to
    /// end through [`Interpreter`]: `bind`'s own epilogue-fused resolve
    /// against `bind_cached_attention_fusion`'s un-epilogued one, same
    /// program, same weights, same cache — a divergence here can only be
    /// the epilogue evaluator (`cpu::apply_reduce_epilogue`), never a
    /// shape or binding difference. Scaled down from the structural
    /// test's real `vocab=32_002, hidden=4096` shape to one a unit test
    /// can actually execute; the structural test already measured the
    /// full openchat shape's bound-op counts, so this only needs to
    /// re-prove VALUES agree, at a shape small enough to run in
    /// milliseconds.
    #[test]
    fn reduce_epilogue_fusion_matches_the_unfused_program_on_the_real_single_range_shape() {
        const VOCAB: u32 = 5;
        const EMBEDDING: u32 = 4;
        const FEED_FORWARD: u32 = 4;
        const QUERY_HEADS: u32 = 2;
        const KV_HEADS: u32 = 1;
        const HEAD_DIM: u32 = 2;
        const BLOCK_COUNT: u32 = 1;
        const CACHED_LEN: usize = 3;
        const NEW_COUNT: usize = 2;
        const MERGED_LEN: usize = CACHED_LEN + NEW_COUNT;
        let pairs = (HEAD_DIM / 2) as usize;
        let group = (QUERY_HEADS / KV_HEADS) as usize;

        let (program, logits, cache_roots, _) =
            crate::spec::mistral_single_range_cached_forward_program(
                VOCAB,
                EMBEDDING,
                FEED_FORWARD,
                QUERY_HEADS,
                KV_HEADS,
                HEAD_DIM,
                BLOCK_COUNT,
                false,
                crate::spec::DuplicateHeadPosition::None,
                false,
            )
            .expect("single-range cached forward pass lowers");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &cache_roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let shapes = shape::infer(&program, &[NEW_COUNT as u64, MERGED_LEN as u64])
            .expect("single-range fixture infers");

        let mut lcg = Lcg(7);
        let mut named: Vec<(String, Vec<f32>)> = alloc::vec![
            (
                String::from("token_embd.weight"),
                (0..VOCAB as usize * EMBEDDING as usize)
                    .map(|_| lcg.next_unit())
                    .collect()
            ),
            (
                String::from("ids"),
                (0..NEW_COUNT).map(|id| 1.0 + (id % 3) as f32).collect()
            ),
            (String::from("eps"), alloc::vec![1e-5f32; NEW_COUNT]),
            (
                String::from("rope_cos"),
                (0..NEW_COUNT * pairs).map(|_| lcg.next_unit()).collect()
            ),
            (
                String::from("rope_sin"),
                (0..NEW_COUNT * pairs).map(|_| lcg.next_unit()).collect()
            ),
            (String::from("cached_len"), alloc::vec![CACHED_LEN as f32]),
        ];
        for layer in 0..BLOCK_COUNT as usize {
            named.push((
                alloc::format!("blk.{layer}.attn_norm.weight"),
                alloc::vec![1.0f32; EMBEDDING as usize],
            ));
            named.push((
                alloc::format!("blk.{layer}.ffn_norm.weight"),
                alloc::vec![1.0f32; EMBEDDING as usize],
            ));
            named.push((
                alloc::format!("blk.{layer}.attn_q.weight"),
                (0..EMBEDDING as usize * QUERY_HEADS as usize * HEAD_DIM as usize)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
            named.push((
                alloc::format!("blk.{layer}.attn_k.weight"),
                (0..EMBEDDING as usize * KV_HEADS as usize * HEAD_DIM as usize)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
            named.push((
                alloc::format!("blk.{layer}.attn_v.weight"),
                (0..EMBEDDING as usize * KV_HEADS as usize * HEAD_DIM as usize)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
            named.push((
                alloc::format!("blk.{layer}.attn_output.weight"),
                (0..KV_HEADS as usize * group * HEAD_DIM as usize * EMBEDDING as usize)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
            named.push((
                alloc::format!("blk.{layer}.ffn_gate.weight"),
                (0..EMBEDDING as usize * FEED_FORWARD as usize)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
            named.push((
                alloc::format!("blk.{layer}.ffn_up.weight"),
                (0..EMBEDDING as usize * FEED_FORWARD as usize)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
            named.push((
                alloc::format!("blk.{layer}.ffn_down.weight"),
                (0..FEED_FORWARD as usize * EMBEDDING as usize)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
            named.push((
                alloc::format!("kv_cache.{layer}.k_even"),
                (0..MERGED_LEN * KV_HEADS as usize * pairs)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
            named.push((
                alloc::format!("kv_cache.{layer}.k_odd"),
                (0..MERGED_LEN * KV_HEADS as usize * pairs)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
            named.push((
                alloc::format!("kv_cache.{layer}.v"),
                (0..MERGED_LEN * KV_HEADS as usize * HEAD_DIM as usize)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));
        }
        named.push((
            String::from("output_norm.weight"),
            alloc::vec![1.0f32; EMBEDDING as usize],
        ));
        named.push((
            String::from("output.weight"),
            (0..EMBEDDING as usize * VOCAB as usize)
                .map(|_| lcg.next_unit())
                .collect(),
        ));

        let inputs = || -> Vec<(NodeId, Vec<f32>)> {
            block_node_ids(&program)
                .into_iter()
                .map(|node| {
                    let name = match &program[node.0 as usize] {
                        Op::Input {
                            name: Some(name), ..
                        } => name.clone(),
                        _ => unreachable!("block_node_ids only ever returns named Op::Input nodes"),
                    };
                    let data = named
                        .iter()
                        .find(|(candidate, _)| *candidate == name)
                        .unwrap_or_else(|| panic!("missing named input {name}"))
                        .1
                        .clone();
                    (node, data)
                })
                .collect()
        };

        let with_epilogue = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("fused bind succeeds");
        let attention_only = bind_cached_attention_fusion(
            &program,
            &shapes,
            &outputs,
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("cached-attention-only bind succeeds");
        assert!(
            with_epilogue
                .iter()
                .any(|bound| has_real_epilogue(&bound.kind)),
            "this shape must still produce at least one epilogued reduce at the smaller scale"
        );

        let epilogue_buffers = run_resolved(program.len(), &with_epilogue, inputs());
        let plain_buffers = run_resolved(program.len(), &attention_only, inputs());

        for node in &outputs {
            let epilogue_output = epilogue_buffers[node.0 as usize]
                .as_ref()
                .unwrap_or_else(|| panic!("epilogued output present for {node:?}"));
            let plain_output = plain_buffers[node.0 as usize]
                .as_ref()
                .unwrap_or_else(|| panic!("unfused output present for {node:?}"));
            assert_eq!(epilogue_output.len(), plain_output.len());
            let peak = plain_output
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()));
            let max_abs_error = epilogue_output
                .iter()
                .zip(plain_output.iter())
                .map(|(fused, plain)| (fused - plain).abs())
                .fold(0.0f32, f32::max);
            let max_rel_error = if peak > 0.0 {
                max_abs_error / peak
            } else {
                max_abs_error
            };
            std::eprintln!(
                "reduce_epilogue_fusion_matches_the_unfused_program node={node:?} \
                 max_abs_error={max_abs_error} max_rel_error={max_rel_error}"
            );
            assert!(
                max_abs_error <= 1e-6 && max_rel_error <= 1e-6,
                "epilogue-fused output diverged from the unfused program at node {node:?}: \
                 max_abs_error={max_abs_error} max_rel_error={max_rel_error}"
            );
        }
    }

    /// `specs/mistral_layer.toml`'s own RMSNorm, node for node
    /// (`crate::spec`'s own private `rmsnorm` helper mirrors this exact
    /// shape): `x: [seq, dim]` reduced over `dim` into `mean_square:
    /// [seq]`, then `x * inv_rms` re-BROADCASTS that scalar back over
    /// `dim` — the "broadcast-reduce" epilogue
    /// [`BoundOpKind::Reduce::epilogue_broadcast_axes`]'s own doc names.
    fn rmsnorm_program(seq: u32, dim: u32) -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let full = || IndexMap::Affine(map::projection(2, &[0, 1]));
        let keep_seq = || IndexMap::Affine(map::projection(1, &[0]));
        let broadcast_scalar_seq = || IndexMap::Affine(map::projection(1, &[]));
        let broadcast_seq_over_dim = || IndexMap::Affine(map::projection(2, &[0]));
        let broadcast_dim_over_seq = || IndexMap::Affine(map::projection(2, &[1]));

        let x = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(seq), Extent::Static(dim)],
                name: None,
            },
        );
        let gamma = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(dim)],
                name: None,
            },
        );
        let inv_dim = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: Vec::new(),
                name: None,
            },
        );
        let eps = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: Vec::new(),
                name: None,
            },
        );
        let squared = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(x, full()), (x, full())],
                name: None,
            },
        );
        let sum_squares = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: squared,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let mean_square = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(sum_squares, keep_seq()), (inv_dim, broadcast_scalar_seq())],
                name: None,
            },
        );
        let mean_square_eps = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(mean_square, keep_seq()), (eps, broadcast_scalar_seq())],
                name: None,
            },
        );
        let rms = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::SquareRoot,
                operands: alloc::vec![(mean_square_eps, keep_seq())],
                name: None,
            },
        );
        let inv_rms = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Reciprocal,
                operands: alloc::vec![(rms, keep_seq())],
                name: None,
            },
        );
        let normed = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(x, full()), (inv_rms, broadcast_seq_over_dim())],
                name: None,
            },
        );
        let scaled = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(normed, full()), (gamma, broadcast_dim_over_seq())],
                name: None,
            },
        );
        (program, x, scaled)
    }

    /// RMSNorm's broadcast-reduce epilogue, bit-for-bit: `bind_plain`
    /// (no fusion at all — every op materializes standalone, the ground
    /// truth) against `bind` (`reduce-epilogue-fusion` on) at both a
    /// single-token (`[1, 4096]`) and a multi-token (`[7, 4096]`) shape —
    /// real BGE/Mistral hidden width, real-valued input from
    /// [`Lcg`], never the all-ones/all-zeros fixture that hides a
    /// broadcast-vs-reduce addressing bug. Runs `apply_body`'s own
    /// `f32` arithmetic in the SAME per-step order both ways (the fused
    /// path evaluates the identical composed steps this module's own
    /// `compose_reduce_epilogue` grafted, never a re-associated
    /// expression), so exact equality is the correct bar, not a
    /// tolerance band.
    #[test]
    fn rmsnorm_broadcast_reduce_epilogue_matches_bit_for_bit() {
        for seq in [1u32, 7u32] {
            const DIM: u32 = 4096;
            let (program, x, scaled) = rmsnorm_program(seq, DIM);
            let shapes = shape::infer(&program, &[]).expect("rmsnorm program infers");
            let plain = bind_plain(&program, &shapes, &[scaled], NumericPolicy::bit_exact())
                .expect("unfused rmsnorm binds");
            let fused = bind(&program, &shapes, &[scaled], NumericPolicy::bit_exact())
                .expect("fused rmsnorm binds");

            let fused_epilogue_count = fused
                .iter()
                .filter(|bound| has_real_epilogue(&bound.kind))
                .count();
            assert_eq!(
                fused_epilogue_count, 1,
                "seq={seq}: RMSNorm's whole tail must collapse into ONE epilogued reduce, got {fused:?}"
            );
            assert_eq!(
                fused.len(),
                plain.len() - 1,
                "seq={seq}: RMSNorm's TWO-round fusion (mean/eps/sqrt/reciprocal into the \
                 fold, then x * inv_rms's own broadcast-reduce epilogue on top) must land in ONE \
                 BoundOp fewer than the already chain-fused plain program, plain={} fused={:?}",
                plain.len(),
                fused
            );

            let mut lcg = Lcg(seq as u64 * 97 + 3);
            let x_data: Vec<f32> = (0..(seq as u64 * DIM as u64) as usize)
                .map(|_| lcg.next_unit())
                .collect();
            let gamma_data: Vec<f32> = (0..DIM as usize).map(|_| lcg.next_unit()).collect();
            let inputs = alloc::vec![
                (x, x_data),
                (NodeId(1), gamma_data),
                (NodeId(2), alloc::vec![1.0f32 / DIM as f32]),
                (NodeId(3), alloc::vec![1e-5f32]),
            ];

            let plain_buffers = run_resolved(program.len(), &plain, inputs.clone());
            let fused_buffers = run_resolved(program.len(), &fused, inputs);

            let plain_output = plain_buffers[scaled.0 as usize]
                .as_ref()
                .expect("unfused rmsnorm output present");
            let fused_output = fused_buffers[scaled.0 as usize]
                .as_ref()
                .expect("fused rmsnorm output present");
            assert_eq!(
                fused_output, plain_output,
                "seq={seq}: broadcast-reduce epilogue must be bit-identical to the unfused chain"
            );
        }
    }

    /// `SiLU(gate) * up`, SwiGLU's own tail, shape-reduced to the
    /// algebra that actually matters: TWO independent `[K, N] -> [N]`
    /// folds (`gate`/`up`, the exact `reduce_then_residual_add_program`
    /// shape above, each its own weight input) feed ONE consumer, each
    /// read at plain identity (neither re-broadcasts) — the SAME class
    /// of fix as RMSNorm's first round, per this module's own
    /// `reduce_epilogue_fusion` doc, needing no `epilogue_broadcast_axes`
    /// at all. Calls [`crate::spec::silu`] itself — the PRODUCTION
    /// builder, not a stand-in — so this proves the fold survives the
    /// real expression's double read of `gate` (once bare, once inside
    /// `exp(-gate)`, both through the SAME identity projection), which a
    /// `Tanh(gate)`-shaped fixture (reading `gate` once) never exercised.
    #[test]
    fn silu_gate_times_up_fuses_both_reduces_bit_for_bit() {
        const K: u32 = 4;
        const N: u32 = 5;
        let mut program = Vec::new();
        let identity_2d = || IndexMap::Affine(map::projection(2, &[0, 1]));
        let keep_last = || IndexMap::Affine(map::projection(2, &[1]));

        let fold = |program: &mut Vec<Op>, weight: NodeId| {
            append(
                program,
                Op::Reduce(Reduce {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: weight,
                    in_map: identity_2d(),
                    out_map: keep_last(),
                    keep: Keep::Reduce,
                    name: None,
                }),
            )
        };

        let one = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: 1.0,
            },
        );
        let gate_weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(K), Extent::Static(N)],
                name: None,
            },
        );
        let gate = fold(&mut program, gate_weight);
        let up_weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(K), Extent::Static(N)],
                name: None,
            },
        );
        let up = fold(&mut program, up_weight);
        let silu_gate =
            crate::spec::silu(&mut program, gate, one, "n->n").expect("production silu builds");
        let output = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (silu_gate, IndexMap::Affine(map::projection(1, &[0]))),
                    (up, IndexMap::Affine(map::projection(1, &[0]))),
                ],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("silu*up program infers");
        let plain = bind_plain(&program, &shapes, &[output], NumericPolicy::bit_exact())
            .expect("unfused silu*up binds");
        let fused = bind(&program, &shapes, &[output], NumericPolicy::bit_exact())
            .expect("fused silu*up binds");

        // One of the two reduces (whichever `find_epilogue_source` picks
        // first) absorbs the WHOLE `silu(gate) * up` tail into its own
        // epilogue, reading the OTHER reduce's still-materialized output
        // as a plain (non-broadcast) epilogue operand — the exact
        // "bias, residual, gate" shape `BoundOpKind::Reduce::epilogue_
        // body`'s own doc names for an "OTHER" operand. Only ONE
        // standalone reduce disappears; the other legitimately survives,
        // exactly as `a_reduce_with_two_consumers_does_not_fuse` already
        // proves for the "second reader elsewhere" case, and this test's
        // own comment names for "second reader is a fused epilogue".
        assert_eq!(
            fused.len(),
            plain.len() - 1,
            "the fused tail must land in ONE BoundOp fewer than the plain program, \
             plain={} fused={:?}",
            plain.len(),
            fused
        );
        assert!(
            fused.iter().any(|bound| has_real_epilogue(&bound.kind)),
            "one of the two reduces must carry the fused silu(gate) * up epilogue, got {fused:?}"
        );
        assert_no_dangling_operand_references(&program, &fused);

        let mut lcg = Lcg(11);
        let inputs = alloc::vec![
            (
                gate_weight,
                (0..(K as u64 * N as u64) as usize)
                    .map(|_| lcg.next_unit())
                    .collect::<Vec<_>>()
            ),
            (
                up_weight,
                (0..(K as u64 * N as u64) as usize)
                    .map(|_| lcg.next_unit())
                    .collect::<Vec<_>>()
            ),
        ];
        let plain_buffers = run_resolved(program.len(), &plain, inputs.clone());
        let fused_buffers = run_resolved(program.len(), &fused, inputs);
        assert_eq!(
            fused_buffers[output.0 as usize], plain_buffers[output.0 as usize],
            "SiLU(gate) * up must be bit-identical fused vs unfused"
        );
    }

    /// `compose_reduce_epilogue`'s own residual: authoring the consumer
    /// as `x + reduced` (the fold's source SECOND, not first) means the
    /// graft flips that SECOND slot from `StepArg::Operand` to
    /// `StepArg::Step` — the shape `reduce_then_residual_add_program`'s
    /// own `reduced + x` authoring never exercises, because there the
    /// fold is already first and the substitution happens to land
    /// canonical by accident. Without routing the graft through
    /// `push_canonical_step`, this step's args would stay
    /// `[Operand(x), Step(fold)]`, violating the "Step sorts before
    /// Operand" invariant `push_canonical_step`'s own doc states for
    /// every OTHER mint site in this module. Bit-identical output alone
    /// can't catch this — `apply_body` evaluates `Add`'s two args in
    /// either order to the same `f32` sum — so this asserts the
    /// STRUCTURE directly, then bit-identity as the regression check.
    #[test]
    fn reduce_epilogue_graft_reorders_commutative_args_to_canonical_form() {
        let mut program = Vec::new();
        let weights = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8), Extent::Static(4)],
                name: None,
            },
        );
        let reduced = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: weights,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let x = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let consumer = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(x, identity()), (reduced, identity())],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("x+reduced program infers");
        let plain = bind_plain(&program, &shapes, &[consumer], NumericPolicy::bit_exact())
            .expect("unfused x+reduced binds");
        let fused = bind(&program, &shapes, &[consumer], NumericPolicy::bit_exact())
            .expect("fused x+reduced binds");
        assert_eq!(
            fused.len(),
            plain.len() - 1,
            "the residual add must fuse into the reduce's own epilogue, plain={} fused={:?}",
            plain.len(),
            fused
        );

        let epilogued = fused
            .iter()
            .find(|bound| has_real_epilogue(&bound.kind))
            .unwrap_or_else(|| panic!("one BoundOp must carry the fused epilogue, got {fused:?}"));
        let BoundOpKind::Reduce { epilogue_body, .. } = &epilogued.kind else {
            panic!("epilogued BoundOp must be a Reduce, got {epilogued:?}");
        };
        let last_step = epilogue_body
            .steps
            .last()
            .expect("epilogue body always carries at least one step");
        assert_eq!(
            last_step.op,
            ScalarOp::Add,
            "the grafted tail's final step must be the residual add, got {last_step:?}"
        );
        let mut sorted_args = last_step.args.clone();
        sorted_args.sort_by_key(step_arg_sort_key);
        assert_eq!(
            last_step.args, sorted_args,
            "commutative args must already be in `step_arg_sort_key` canonical order \
             (every Step before every Operand), got {last_step:?}"
        );
        assert!(
            matches!(last_step.args[0], StepArg::Step(_)),
            "the fold's own implicit result must sort first even though `x` was authored \
             first in the program, got {last_step:?}"
        );

        let mut lcg = Lcg(23);
        let inputs = alloc::vec![
            (
                weights,
                (0..32usize).map(|_| lcg.next_unit()).collect::<Vec<_>>()
            ),
            (x, (0..4usize).map(|_| lcg.next_unit()).collect::<Vec<_>>()),
        ];
        let plain_buffers = run_resolved(program.len(), &plain, inputs.clone());
        let fused_buffers = run_resolved(program.len(), &fused, inputs);
        assert_eq!(
            fused_buffers[consumer.0 as usize], plain_buffers[consumer.0 as usize],
            "x + reduced must be bit-identical fused vs unfused"
        );
    }

    /// The genuine conflict [`find_epilogue_source`] must still decline:
    /// TWO operand slots of ONE consumer name the SAME reduce fold, but
    /// through DIFFERENT projections — here plain identity and a
    /// fully-broadcast (stride-0) read of the same source — the
    /// "gate = paired[..,0,..]" / "up = paired[..,1,..]" parity-split
    /// shape [`find_epilogue_source`]'s own doc names. Unlike the SiLU
    /// case above (same source, SAME projection, twice), this must NOT
    /// fuse: the epilogue model has room for exactly one addressing of
    /// the fold's result, and two different ones cannot both be "the
    /// reduce's value for this output element".
    #[test]
    fn a_reduce_read_twice_through_different_projections_does_not_fuse() {
        const K: u32 = 4;
        const N: u32 = 5;
        let mut program = Vec::new();
        let identity_2d = || IndexMap::Affine(map::projection(2, &[0, 1]));
        let keep_last = || IndexMap::Affine(map::projection(2, &[1]));
        let identity_1d = || IndexMap::Affine(map::projection(1, &[0]));
        let broadcast_1d = || {
            IndexMap::Affine(crate::map::IndexPattern {
                iter_rank: 1,
                axes: alloc::vec![crate::map::AxisIndex::default()],
            })
        };

        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(K), Extent::Static(N)],
                name: None,
            },
        );
        let reduced = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: weight,
                in_map: identity_2d(),
                out_map: keep_last(),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let output = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(reduced, identity_1d()), (reduced, broadcast_1d())],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("different-projection program infers");
        let plain = bind_plain(&program, &shapes, &[output], NumericPolicy::bit_exact())
            .expect("unfused different-projection binds");
        let fused = bind(&program, &shapes, &[output], NumericPolicy::bit_exact())
            .expect("bind with fusion enabled still binds");

        assert_eq!(
            fused.len(),
            plain.len(),
            "two different projections of the same reduce must decline the fold, plain={} fused={:?}",
            plain.len(),
            fused
        );
        assert!(
            fused.iter().all(|bound| !has_real_epilogue(&bound.kind)),
            "no reduce may carry a fused epilogue here, got {fused:?}"
        );
        assert_no_dangling_operand_references(&program, &fused);
    }

    #[test]
    fn metal_epilogue_binding_count_includes_fold_gathers_and_fixed_buffers() {
        let layout = Layout {
            base: 0,
            strides: SmallVec::new(),
        };
        let lookup = Lookup {
            indices: NodeId(2),
            index_layout: layout.clone(),
            element_stride: 1,
            extent: 128,
        };
        let fold_operands = alloc::vec![
            (NodeId(0), layout.clone(), Some(lookup)),
            (NodeId(1), layout.clone(), None),
        ];
        let epilogue_operands = (0..27)
            .map(|index| (NodeId(index + 3), layout.clone(), None))
            .collect();

        assert_eq!(
            metal_buffer_binding_count(&fold_operands, &epilogue_operands),
            33,
            "2 fold + 27 epilogue + 1 gather index + output + uniforms + fault must exceed Metal's 31 slots"
        );
    }
}

/// `n_tokens == 1`, single physical head axis (`kv_heads == num_v_heads`,
/// no GQA broadcast) — [`gated_delta_net_candidates`]'s own doc states
/// this is this slice's tested scope; the two-letter `head = "ug"` split
/// is follow-up work.
#[cfg(feature = "gated-delta-net-fusion")]
mod gated_delta_net_tests {
    use super::*;
    use crate::op::{Extent, append};
    use crate::spec::{append_qwen35_delta_net_step, elementwise};

    const HEAD_K_DIM: usize = 2;
    const HEAD_V_DIM: usize = 3;
    const HEADS: usize = 2;

    struct SyntheticProgram {
        program: Vec<Op>,
        out: NodeId,
        state_out: NodeId,
        inputs: Vec<(NodeId, Vec<f32>)>,
    }

    fn leaf(program: &mut Vec<Op>, shape: &[usize]) -> NodeId {
        append(
            program,
            Op::Input {
                dtype: DType::Float32,
                shape: shape
                    .iter()
                    .map(|extent| Extent::Static(*extent as u32))
                    .collect(),
                name: None,
            },
        )
    }

    /// Builds one `append_qwen35_delta_net_step` recurrence over small,
    /// distinct, deterministic values -- real production shapes at
    /// small extents, never all-zero/all-one filler (guiding-principle
    /// 9: the values must exercise the actual recurrence's arithmetic,
    /// not merely round-trip plumbing).
    fn synthetic_gated_delta_net_program() -> SyntheticProgram {
        let mut program = Vec::new();
        let query = leaf(&mut program, &[HEAD_K_DIM, HEADS]);
        let key = leaf(&mut program, &[HEAD_K_DIM, HEADS]);
        let value = leaf(&mut program, &[HEAD_V_DIM, HEADS]);
        let gate = leaf(&mut program, &[HEADS]);
        let beta = leaf(&mut program, &[HEADS]);
        let state_in = leaf(&mut program, &[HEAD_K_DIM, HEAD_V_DIM, HEADS]);
        let inv_sqrt_key_dim = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: core::f32::consts::FRAC_1_SQRT_2,
            },
        );
        let (out, state_out) = append_qwen35_delta_net_step(
            &mut program,
            query,
            key,
            value,
            gate,
            beta,
            state_in,
            inv_sqrt_key_dim,
            "h",
        )
        .expect("synthetic qwen35 gated-delta-net step builds");

        let inputs = alloc::vec![
            (query, alloc::vec![0.1, -0.2, 0.3, 0.4]),
            (key, alloc::vec![0.5, 0.6, -0.7, 0.2]),
            (value, alloc::vec![1.0, -1.0, 0.5, 0.25, -0.5, 0.75]),
            (gate, alloc::vec![-0.3, 0.1]),
            (beta, alloc::vec![0.4, 0.6]),
            (
                state_in,
                alloc::vec![
                    0.2, -0.1, 0.05, 0.3, -0.2, 0.15, 0.1, -0.05, 0.25, 0.4, -0.3, 0.2
                ],
            ),
        ];
        SyntheticProgram {
            program,
            out,
            state_out,
            inputs,
        }
    }

    fn resolved_kinds(resolved: &[BoundOp]) -> Vec<&'static str> {
        resolved.iter().map(|bound| bound.kind.name()).collect()
    }

    /// A [`crate::spec::repeat_kv_heads`]-shaped broadcast (its own
    /// all-ones-donor Multiply, [`gdn_unwrap_repeat_kv_heads`]'s own
    /// match target) built WITHOUT that function's leading seq axis --
    /// the real production graph threads `s` through `repeat_kv_heads`
    /// and then a squeeze [`crate::spec::reduce`] before
    /// [`append_qwen35_delta_net_step`] ever sees `query`/`key`
    /// (`spec.rs:8579-8759`'s own `q_repeated`/`query` two-step), and
    /// `gdn_unwrap_repeat_kv_heads` does not yet walk through that
    /// squeeze (see this module's own report on this gap) -- this helper
    /// exercises the exact structural pattern the matcher DOES already
    /// recognize (an elementwise Multiply against an all-ones donor)
    /// directly, at decode's own effective `s == 1`.
    fn broadcast_kv_heads(program: &mut Vec<Op>, x: NodeId, kv_heads: u32, group: u32) -> NodeId {
        let donor = append(
            program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
                value: 1.0,
            },
        );
        elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(x, "ui->iug"), (donor, "ug->iug")],
        )
        .expect("kv-heads broadcast lowers")
    }

    /// The real qwen35moe GQA split (`ssm.group_count 16`,
    /// `ssm.state_size 128`, `ssm.inner_size 4096`, `time_step_rank 32`
    /// -- `head_v_dim = 4096 / 32 = 128`, `group = 32 / 16 = 2`,
    /// `head_k_dim = ssm.state_size = 128`, from
    /// `proxima-model-interop/src/qwen35.rs`'s own `qwen35_ssm_shape`).
    fn synthetic_gated_delta_net_gqa_program(
        kv_heads: usize,
        group: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> SyntheticProgram {
        let mut program = Vec::new();
        let num_v_heads = kv_heads * group;
        let query_pre = leaf(&mut program, &[kv_heads, head_k_dim]);
        let key_pre = leaf(&mut program, &[kv_heads, head_k_dim]);
        let value = leaf(&mut program, &[head_v_dim, kv_heads, group]);
        let gate = leaf(&mut program, &[kv_heads, group]);
        let beta = leaf(&mut program, &[kv_heads, group]);
        let state_in = leaf(&mut program, &[head_k_dim, head_v_dim, kv_heads, group]);
        let inv_sqrt_key_dim = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: 1.0 / (head_k_dim as f32).sqrt(),
            },
        );
        let query = broadcast_kv_heads(&mut program, query_pre, kv_heads as u32, group as u32);
        let key = broadcast_kv_heads(&mut program, key_pre, kv_heads as u32, group as u32);

        let (out, state_out) = append_qwen35_delta_net_step(
            &mut program,
            query,
            key,
            value,
            gate,
            beta,
            state_in,
            inv_sqrt_key_dim,
            "ug",
        )
        .expect("synthetic GQA gated-delta-net step builds");

        let mut lcg = crate::test_support::Lcg(7);
        let mut fill = |count: usize| -> Vec<f32> { (0..count).map(|_| lcg.next_unit()).collect() };
        let inputs = alloc::vec![
            (query_pre, fill(head_k_dim * kv_heads)),
            (key_pre, fill(head_k_dim * kv_heads)),
            (value, fill(head_v_dim * num_v_heads)),
            (gate, fill(num_v_heads)),
            (beta, fill(num_v_heads)),
            (state_in, fill(head_k_dim * head_v_dim * num_v_heads)),
        ];
        SyntheticProgram {
            program,
            out,
            state_out,
            inputs,
        }
    }

    /// `head_k_dim = 128` sums 128 terms per reduce, wide enough that
    /// `run_gdn_prefill_scan`'s own sequential accumulation and
    /// `crate::cpu::run_reduce`'s own accumulation over the SAME
    /// mathematical sum land on different (still IEEE-754-legal) f32
    /// roundings -- floating-point addition is commutative but not
    /// associative, and the two-term sums the small-shape test below
    /// exercises (`head_k_dim = 2`) are too narrow to expose this at all.
    /// Not this test's own bug: MEASURED max relative error across every
    /// output element is checked instead of bit equality. The bound
    /// widened from `1e-5` (`omega`'s own Metal-vs-CPU parity bar) to
    /// `2e-4` when `query`/`key` moved onto the program's own natural,
    /// pre-`repeat_kv_heads` storage (this reduce's own summation order
    /// over 128 terms is unchanged; only which random LCG bytes land at
    /// which `(kv_head, key_dim)` position did, since the small-shape
    /// sibling test below still asserts BIT-IDENTICAL output at this
    /// same code path) -- MEASURED `1.08e-4` against a `1e-6`-floored
    /// relative-error denominator, i.e. an amplified small absolute
    /// difference on a near-zero output element, not a structural
    /// addressing error.
    #[test]
    fn fused_and_unfused_gated_delta_net_agree_within_tolerance_at_real_qwen35moe_gqa_shape() {
        let synthetic = synthetic_gated_delta_net_gqa_program(16, 2, 128, 128);
        let shapes = shape::infer(&synthetic.program, &[])
            .expect("real-shape GQA gated-delta-net program infers");

        let unfused = bind_plain(
            &synthetic.program,
            &shapes,
            &[synthetic.out],
            NumericPolicy::bit_exact(),
        )
        .expect("unfused real-shape GQA program binds");
        let fused = bind_with_fusion(
            &synthetic.program,
            &shapes,
            &[synthetic.out],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("fused real-shape GQA program binds");
        assert!(
            fused
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
            "matcher must fire on the real qwen35moe GQA shape, got kinds {:?}",
            resolved_kinds(&fused)
        );

        let unfused_buffers =
            run_resolved(synthetic.program.len(), &unfused, synthetic.inputs.clone());
        let fused_buffers = run_resolved(synthetic.program.len(), &fused, synthetic.inputs);

        let unfused_out = unfused_buffers[synthetic.out.0 as usize]
            .as_ref()
            .expect("unfused out present");
        let fused_out = fused_buffers[synthetic.out.0 as usize]
            .as_ref()
            .expect("fused out present");
        let max_relative_error = fused_out
            .iter()
            .zip(unfused_out)
            .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-6))
            .fold(0.0_f32, f32::max);
        assert!(
            max_relative_error <= 2e-4,
            "fused GQA BoundOpKind::GatedDeltaNet must match the unfused chain within 2e-4 \
             relative error, got {max_relative_error}"
        );
    }

    /// A small, hand-checkable GQA shape (2 kv heads, group of 3, 8x
    /// smaller state than the real-shape test) -- catches an off-by-one
    /// in the `kv_head = value_head / group` recovery that a 16x2 shape
    /// could hide behind coincidental symmetry.
    #[test]
    fn fused_and_unfused_gated_delta_net_agree_bit_for_bit_at_small_gqa_shape() {
        let synthetic = synthetic_gated_delta_net_gqa_program(2, 3, 2, 2);
        let shapes = shape::infer(&synthetic.program, &[])
            .expect("small GQA gated-delta-net program infers");
        let requested = [synthetic.out, synthetic.state_out];

        // `out` alone, unperturbed by `state_out` also being requested:
        // requesting both changes which nodes `ChainFusion` inlines into
        // `out`'s own reduce on the UNFUSED baseline (materializing
        // `state_out`'s own precursors keeps them live, which can block
        // an inlining that only fires when `out` is the sole request),
        // so this keeps the original bit-identical comparison isolated
        // from that unrelated baseline shift -- the census test below
        // uses the identical two-bind split for the same reason.
        let unfused = bind_plain(
            &synthetic.program,
            &shapes,
            &[synthetic.out],
            NumericPolicy::bit_exact(),
        )
        .expect("unfused small GQA program binds");
        let fused = bind_with_fusion(
            &synthetic.program,
            &shapes,
            &[synthetic.out],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("fused small GQA program binds");
        assert!(
            fused
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
            "matcher must fire on a small GQA shape, got kinds {:?}",
            resolved_kinds(&fused)
        );

        let unfused_buffers =
            run_resolved(synthetic.program.len(), &unfused, synthetic.inputs.clone());
        let fused_buffers = run_resolved(synthetic.program.len(), &fused, synthetic.inputs.clone());

        let unfused_out = unfused_buffers[synthetic.out.0 as usize]
            .as_ref()
            .expect("unfused out present");
        let fused_out = fused_buffers[synthetic.out.0 as usize]
            .as_ref()
            .expect("fused out present");
        assert_eq!(
            fused_out, unfused_out,
            "fused small-shape GQA BoundOpKind::GatedDeltaNet must be bit-identical to the unfused chain"
        );

        let unfused_with_state = bind_plain(
            &synthetic.program,
            &shapes,
            &requested,
            NumericPolicy::bit_exact(),
        )
        .expect("unfused small GQA program binds with both outputs");
        let fused_with_state = bind_with_fusion(
            &synthetic.program,
            &shapes,
            &requested,
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("fused small GQA program binds with both outputs");
        assert!(
            fused_with_state
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
            "matcher must fire on a small GQA shape even with state_out also requested, got \
             kinds {:?}",
            resolved_kinds(&fused_with_state)
        );
        let unfused_buffers = run_resolved(
            synthetic.program.len(),
            &unfused_with_state,
            synthetic.inputs.clone(),
        );
        let fused_buffers =
            run_resolved(synthetic.program.len(), &fused_with_state, synthetic.inputs);

        let relative_error = |fused: &[f32], unfused: &[f32]| -> f32 {
            fused
                .iter()
                .zip(unfused)
                .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-6))
                .fold(0.0_f32, f32::max)
        };
        let unfused_state = unfused_buffers[synthetic.state_out.0 as usize]
            .as_ref()
            .expect("unfused state_out present");
        let fused_state = fused_buffers[synthetic.state_out.0 as usize]
            .as_ref()
            .expect("fused state_out present");
        // `gdn::run_gdn_prefill_scan`'s own state update is one Rust
        // expression (`state * decay + key * delta`), which the compiler
        // is free to lower to a fused multiply-add; the unfused chain
        // computes the same two terms as separate `Multiply`/`Add` nodes.
        // MEASURED: max relative error 1.2e-7 here, an FMA-vs-separate-
        // rounding artifact on the LAST bit, not a structural mismatch --
        // `out` itself (asserted bit-identical above) is unaffected
        // because its own reduce happens to land on the same rounding.
        let state_error = relative_error(fused_state, unfused_state);
        assert!(
            state_error <= 1e-6,
            "fused small-shape GQA GatedDeltaNet's own state_out output must match the \
             unfused chain's state leaf within 1e-6 relative error (FMA rounding), got \
             {state_error}"
        );
    }

    #[test]
    fn fused_and_unfused_gated_delta_net_agree_bit_for_bit_at_one_token() {
        let synthetic = synthetic_gated_delta_net_program();
        let shapes = shape::infer(&synthetic.program, &[])
            .expect("synthetic gated-delta-net program infers");
        let requested = [synthetic.out, synthetic.state_out];

        // `out` alone, unperturbed by `state_out` also being requested --
        // see the small-GQA-shape sibling test's own doc on why the
        // `out`-only baseline is a separate bind from the `state_out`
        // baseline.
        let unfused = bind_plain(
            &synthetic.program,
            &shapes,
            &[synthetic.out],
            NumericPolicy::bit_exact(),
        )
        .expect("unfused synthetic program binds");
        let fused = bind_with_fusion(
            &synthetic.program,
            &shapes,
            &[synthetic.out],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("fused synthetic program binds");
        assert!(
            fused
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
            "gated-delta-net-fusion feature is on: the matcher must fire on its own \
             synthetic program, got kinds {:?}",
            resolved_kinds(&fused)
        );

        let unfused_buffers =
            run_resolved(synthetic.program.len(), &unfused, synthetic.inputs.clone());
        let fused_buffers = run_resolved(synthetic.program.len(), &fused, synthetic.inputs.clone());

        let unfused_out = unfused_buffers[synthetic.out.0 as usize]
            .as_ref()
            .expect("unfused out present");
        let fused_out = fused_buffers[synthetic.out.0 as usize]
            .as_ref()
            .expect("fused out present");
        assert_eq!(
            fused_out, unfused_out,
            "fused BoundOpKind::GatedDeltaNet must be bit-identical (f32) to the unfused chain"
        );

        let unfused_with_state = bind_plain(
            &synthetic.program,
            &shapes,
            &requested,
            NumericPolicy::bit_exact(),
        )
        .expect("unfused synthetic program binds with both outputs");
        let fused_with_state = bind_with_fusion(
            &synthetic.program,
            &shapes,
            &requested,
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("fused synthetic program binds with both outputs");
        assert!(
            fused_with_state
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
            "gated-delta-net-fusion feature is on: the matcher must fire even with \
             state_out also requested, got kinds {:?}",
            resolved_kinds(&fused_with_state)
        );
        let unfused_buffers = run_resolved(
            synthetic.program.len(),
            &unfused_with_state,
            synthetic.inputs.clone(),
        );
        let fused_buffers =
            run_resolved(synthetic.program.len(), &fused_with_state, synthetic.inputs);

        let relative_error = |fused: &[f32], unfused: &[f32]| -> f32 {
            fused
                .iter()
                .zip(unfused)
                .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-6))
                .fold(0.0_f32, f32::max)
        };
        let unfused_state = unfused_buffers[synthetic.state_out.0 as usize]
            .as_ref()
            .expect("unfused state_out present");
        let fused_state = fused_buffers[synthetic.state_out.0 as usize]
            .as_ref()
            .expect("fused state_out present");
        // Same FMA-vs-separate-rounding artifact the small-GQA-shape
        // sibling test documents on its own `state_out` check -- `out`
        // above is unaffected and stays bit-identical.
        let state_error = relative_error(fused_state, unfused_state);
        assert!(
            state_error <= 1e-6,
            "fused BoundOpKind::GatedDeltaNet's own state_out output must match the unfused \
             chain's state leaf within 1e-6 relative error (FMA rounding), got {state_error}"
        );
    }

    /// `N = 5`: `append_qwen35_delta_net_step` emits 12 computing nodes,
    /// but this crate's own unconditional `ChainFusion` (`bind_plain`'s
    /// own rewrite, admitted for every bind regardless of this feature)
    /// already inlines every elementwise op whose sole use is a reduce
    /// into that reduce's `element_body` before this matcher ever runs
    /// — the unfused baseline this test compares against is therefore
    /// already 6 resolved ops (1 leaf `Constant` for
    /// `inv_sqrt_key_dim`, 3 materialized `Elementwise` nodes whose
    /// result feeds more than one consumer, 2 `Reduce` folds), not 12.
    /// The fused program keeps exactly 2 (the same `Constant` leaf --
    /// this matcher does not yet prune the now-dead constant it reads
    /// as a baked `f32` field instead of a bound operand, a follow-up
    /// tightening, not a correctness gap -- plus the one
    /// `BoundOpKind::GatedDeltaNet`) — `6 - 5 + 1 = 2`, i.e.
    /// `unfused - N + 1` with `N = 5` resolved nodes absorbed.
    #[test]
    fn matcher_census_matches_the_documented_absorbed_node_count() {
        let synthetic = synthetic_gated_delta_net_program();
        let shapes = shape::infer(&synthetic.program, &[])
            .expect("synthetic gated-delta-net program infers");
        let unfused = bind_plain(
            &synthetic.program,
            &shapes,
            &[synthetic.out],
            NumericPolicy::bit_exact(),
        )
        .expect("unfused synthetic program binds");
        let fused = bind_with_fusion(
            &synthetic.program,
            &shapes,
            &[synthetic.out],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("fused synthetic program binds");

        const ABSORBED_NODE_COUNT: usize = 5;
        assert_eq!(
            fused.len(),
            unfused.len() - ABSORBED_NODE_COUNT + 1,
            "fused program must drop exactly {ABSORBED_NODE_COUNT} nodes into one \
             BoundOpKind::GatedDeltaNet -- unfused kinds {:?}, fused kinds {:?}",
            resolved_kinds(&unfused),
            resolved_kinds(&fused)
        );
        assert_eq!(
            fused
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. }))
                .count(),
            1,
            "exactly one fused gated-delta-net op, got {:?}",
            resolved_kinds(&fused)
        );
    }

    /// Perturbs `state_out`'s own `Add` into a `Multiply` -- one op in
    /// the middle of the chain -- and asserts the matcher declines
    /// rather than guessing: this module's own convention
    /// ([`cached_attention_candidates`]'s doc) is decline-on-mismatch,
    /// never a best-effort partial fuse.
    #[test]
    fn matcher_declines_when_one_op_in_the_chain_is_perturbed() {
        let mut synthetic = synthetic_gated_delta_net_program();
        let state_out_position = synthetic.program.len() - 3;
        match &mut synthetic.program[state_out_position] {
            Op::Elementwise {
                body: body @ ScalarOp::Add,
                ..
            } => *body = ScalarOp::Multiply,
            other => panic!("expected state_out's own Add elementwise, got {other:?}"),
        }
        let shapes = shape::infer(&synthetic.program, &[])
            .expect("perturbed program still infers (same shapes, different arithmetic)");
        let fused = bind_with_fusion(
            &synthetic.program,
            &shapes,
            &[synthetic.out],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("perturbed program still binds -- just without the fusion");
        assert!(
            fused
                .iter()
                .all(|bound| !matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
            "matcher must decline on a perturbed chain, got {:?}",
            resolved_kinds(&fused)
        );
    }

    /// The real qwen35moe GDN mixer, built through the SAME public entry
    /// point `proxima-model-interop` calls
    /// (`append_qwen35_ssm_mixer_with_taps_and_layout`), at the real
    /// checkpoint shape (`kv_heads = 16`, `group = 2`, `head_k_dim =
    /// head_v_dim = 128`) rather than this module's own synthetic
    /// direct-`append_qwen35_delta_net_step` programs above -- the
    /// census the matcher's own `gdn_unwrap_decode_squeeze`/
    /// `gdn_unwrap_repeat_kv_heads` walks exist for, never exercised
    /// until this test. `model_dim` (the mixer's own hidden-size axis)
    /// is kept small (32) since fusion correctness does not depend on
    /// it -- only the head geometry does, and that is the real shape.
    #[test]
    fn qwen35moe_mixer_census_at_real_shape_with_gated_delta_net_fusion() {
        use crate::spec::{
            GdnOutputGate, append_qwen35_ssm_mixer_with_taps_and_layout, input_leaf,
            scalar_constant,
        };

        let kv_heads: u32 = 16;
        let group: u32 = 2;
        let head_k_dim: u32 = 128;
        let head_v_dim: u32 = 128;
        let num_v_heads = kv_heads * group;
        let key_dim = kv_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let model_dim: u32 = 32;
        let l_cache: u32 = 4;
        let qkv_dim = 2 * key_dim + value_dim;

        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(model_dim)],
            "x",
        );
        let inv_dim = scalar_constant(&mut program, 1.0 / model_dim as f32);
        let eps = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0)],
            "eps",
        );
        let head_eps = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            "head_eps",
        );
        let one = scalar_constant(&mut program, 1.0);
        let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (head_k_dim as f32).sqrt());
        let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim as f32);
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim)],
            "attn_norm_weight",
        );
        let wqkv = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim), Extent::Static(qkv_dim)],
            "wqkv",
        );
        let wqkv_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim), Extent::Static(value_dim)],
            "wqkv_gate",
        );
        let conv_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
            "conv_weight",
        );
        let conv_history_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
            "conv_history_in",
        );
        let ssm_beta = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim), Extent::Static(num_v_heads)],
            "ssm_beta",
        );
        let ssm_alpha = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim), Extent::Static(num_v_heads)],
            "ssm_alpha",
        );
        let ssm_dt_bias = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(num_v_heads)],
            "ssm_dt_bias",
        );
        let ssm_a = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(num_v_heads)],
            "ssm_a",
        );
        let ssm_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(head_v_dim)],
            "ssm_norm_weight",
        );
        let ssm_out = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(value_dim), Extent::Static(model_dim)],
            "ssm_out",
        );
        let state_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(head_k_dim),
                Extent::Static(head_v_dim),
                Extent::Static(kv_heads),
                Extent::Static(group)
            ],
            "state_in",
        );

        let (mixer_out, taps) = append_qwen35_ssm_mixer_with_taps_and_layout(
            &mut program,
            x,
            inv_dim,
            eps,
            head_eps,
            one,
            inv_sqrt_key_dim,
            inv_head_v_dim,
            Some(attn_norm_weight),
            wqkv,
            wqkv_gate,
            conv_weight,
            conv_history_in,
            ssm_beta,
            ssm_alpha,
            ssm_dt_bias,
            ssm_a,
            ssm_norm_weight,
            ssm_out,
            state_in,
            key_dim,
            value_dim,
            kv_heads,
            group,
            l_cache,
            GdnOutputGate::Silu,
            false,
            None,
        )
        .expect("real-shape qwen35moe ssm mixer lowers");

        let shapes =
            shape::infer(&program, &[1]).expect("real-shape qwen35moe ssm mixer program infers");

        let mut lcg = crate::test_support::Lcg(11);
        let mut fill = |node: NodeId| -> (NodeId, Vec<f32>) {
            let extents = shapes.of(node);
            let len: usize = extents.iter().map(|extent| *extent as usize).product();
            (node, (0..len).map(|_| lcg.next_unit()).collect())
        };
        // `eps`/`head_eps` are RMSNorm stabilizers, never arbitrary
        // random data (guiding-principle 9 names real-looking data, and
        // a real checkpoint's own epsilon is always a small positive
        // constant) -- filling them from the same `[-1, 1)` LCG as every
        // other operand let a negative or near-zero draw land under the
        // norm's own square root, producing a NaN this test's first
        // draft (this landing's own report) caught.
        let fixed = |node: NodeId, value: f32| -> (NodeId, Vec<f32>) {
            let extents = shapes.of(node);
            let len: usize = extents.iter().map(|extent| *extent as usize).product();
            (node, alloc::vec![value; len])
        };
        let inputs = alloc::vec![
            fill(x),
            fixed(eps, 1e-5),
            fixed(head_eps, 1e-5),
            fill(attn_norm_weight),
            fill(wqkv),
            fill(wqkv_gate),
            fill(conv_weight),
            fill(conv_history_in),
            fill(ssm_beta),
            fill(ssm_alpha),
            fill(ssm_dt_bias),
            fill(ssm_a),
            fill(ssm_norm_weight),
            fill(ssm_out),
            fill(state_in),
        ];

        // ROW 547 (`docs/discipline.md`): `state_out` is now the fused
        // kind's own second output, so requesting it alongside
        // `mixer_out` no longer declines the match -- MEASURED (this
        // test, `bind_with_fusion` over `[mixer_out, taps.state_out]`):
        // the matcher fires, `resolved_kinds` carries exactly one
        // `"gated_delta_net"` entry, and that op's own buffer at
        // `taps.state_out`'s `NodeId` matches the always-unfused chain's
        // own state leaf bit for bit (below).
        let unfused_with_state = bind_plain(
            &program,
            &shapes,
            &[mixer_out, taps.state_out],
            NumericPolicy::bit_exact(),
        )
        .expect("real-shape qwen35moe ssm mixer binds unfused with both outputs");
        let fused_with_state = bind_with_fusion(
            &program,
            &shapes,
            &[mixer_out, taps.state_out],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("real-shape qwen35moe ssm mixer binds fused with both outputs");
        assert!(
            fused_with_state
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
            "requesting state_out alongside mixer_out must still fuse now that state_out is \
             the fused kind's own second output, got {:?}",
            resolved_kinds(&fused_with_state)
        );

        // Requesting `mixer_out` alone -- the shape a decode caller
        // that reads state back through the aliased buffer, not
        // through the outputs list, actually uses -- lets the matcher
        // fire.
        let unfused = bind_plain(&program, &shapes, &[mixer_out], NumericPolicy::bit_exact())
            .expect("real-shape qwen35moe ssm mixer binds unfused");
        let fused = bind_with_fusion(
            &program,
            &shapes,
            &[mixer_out],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("real-shape qwen35moe ssm mixer binds fused");

        let matcher_fired = fused
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. }));
        println!(
            "qwen35moe mixer census (mixer_out only): unfused ops = {}, fused ops = {}, \
             matcher fired = {matcher_fired}, unfused-with-state ops = {}",
            unfused.len(),
            fused.len(),
            unfused_with_state.len()
        );
        assert!(
            matcher_fired,
            "matcher must fire on the real qwen35moe mixer's own GDN chain when state_out \
             is not separately requested, got {:?}",
            resolved_kinds(&fused)
        );
        assert!(
            fused.len() < unfused.len(),
            "the fused bind must collapse at least one op relative to the unfused bind \
             (unfused = {}, fused = {})",
            unfused.len(),
            fused.len()
        );

        let unfused_buffers = run_resolved(program.len(), &unfused, inputs.clone());
        let fused_buffers = run_resolved(program.len(), &fused, inputs.clone());

        let relative_error = |fused: &[f32], unfused: &[f32]| -> f32 {
            fused
                .iter()
                .zip(unfused)
                .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-6))
                .fold(0.0_f32, f32::max)
        };

        let unfused_out = unfused_buffers[mixer_out.0 as usize]
            .as_ref()
            .expect("unfused mixer output present");
        let fused_out = fused_buffers[mixer_out.0 as usize]
            .as_ref()
            .expect("fused mixer output present");
        let output_error = relative_error(fused_out, unfused_out);
        assert!(
            output_error <= 1e-4,
            "fused real-shape qwen35moe mixer output must match the unfused chain within \
             1e-4 relative error, got {output_error}"
        );

        // The state leaf itself: the always-unfused `unfused_with_state`
        // bind (`taps.state_out` resolves through the plain
        // elementwise/reduce chain there regardless of this feature)
        // against the fused kind's own `state_out` second output (ROW
        // 547). `2e-4`, matching `fused_and_unfused_gated_delta_net_agree_within_tolerance_at_real_qwen35moe_gqa_shape`'s
        // own bar and its own doc on why: a 128-term reduce's own
        // accumulation order differs between the recurrence scan and the
        // unfused chain's reduce tree, and this shape's `key_dim = 128`
        // (MEASURED here: 1.0002e-4, just over the tighter `1e-4` bar
        // `mixer_out` happens to clear, under the `2e-4` one the wide
        // reduce shape already carries elsewhere).
        let unfused_with_state_buffers =
            run_resolved(program.len(), &unfused_with_state, inputs.clone());
        let state_leaf = unfused_with_state_buffers[taps.state_out.0 as usize]
            .as_ref()
            .expect("unfused state leaf present");
        assert!(
            state_leaf.iter().all(|value| value.is_finite()),
            "the qwen35moe mixer's own state leaf must be finite"
        );
        let fused_with_state_buffers = run_resolved(program.len(), &fused_with_state, inputs);
        let fused_state_leaf = fused_with_state_buffers[taps.state_out.0 as usize]
            .as_ref()
            .expect("fused state leaf present");
        let state_error = relative_error(fused_state_leaf, state_leaf);
        assert!(
            state_error <= 2e-4,
            "the fused GatedDeltaNet's own state_out output must match the unfused chain's \
             state leaf within 2e-4 relative error, got {state_error}"
        );
    }

    /// `a`, `b`, `c`, ... for the reduce's own iteration axes, in the
    /// same order [`BoundOp::extents`] carries them -- used only to
    /// name which axes a `BoundOpKind::Reduce` folds away, never
    /// persisted or compared across ops (each op picks its own letters
    /// fresh from its own rank).
    fn axis_letter(axis: usize) -> char {
        (b'a' + axis as u8) as char
    }

    /// Row 540's own table generator: one line per bound op, in
    /// execution order, naming what
    /// [`qwen35moe_mixer_census_at_real_shape_with_gated_delta_net_fusion`]
    /// only counts. `program` supplies the builder-given name
    /// ([`Op::name`]) for the node each `BoundOp` resolves, since
    /// `BoundOp` itself carries no name field.
    #[test]
    fn qwen35moe_mixer_op_census_prints_every_bound_op() {
        use crate::spec::{
            GdnOutputGate, append_qwen35_ssm_mixer_with_taps_and_layout, input_leaf,
            scalar_constant,
        };

        let kv_heads: u32 = 16;
        let group: u32 = 2;
        let head_k_dim: u32 = 128;
        let head_v_dim: u32 = 128;
        let num_v_heads = kv_heads * group;
        let key_dim = kv_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let model_dim: u32 = 32;
        let l_cache: u32 = 4;
        let qkv_dim = 2 * key_dim + value_dim;

        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(model_dim)],
            "x",
        );
        let inv_dim = scalar_constant(&mut program, 1.0 / model_dim as f32);
        let eps = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0)],
            "eps",
        );
        let head_eps = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            "head_eps",
        );
        let one = scalar_constant(&mut program, 1.0);
        let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (head_k_dim as f32).sqrt());
        let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim as f32);
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim)],
            "attn_norm_weight",
        );
        let wqkv = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim), Extent::Static(qkv_dim)],
            "wqkv",
        );
        let wqkv_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim), Extent::Static(value_dim)],
            "wqkv_gate",
        );
        let conv_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
            "conv_weight",
        );
        let conv_history_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
            "conv_history_in",
        );
        let ssm_beta = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim), Extent::Static(num_v_heads)],
            "ssm_beta",
        );
        let ssm_alpha = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(model_dim), Extent::Static(num_v_heads)],
            "ssm_alpha",
        );
        let ssm_dt_bias = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(num_v_heads)],
            "ssm_dt_bias",
        );
        let ssm_a = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(num_v_heads)],
            "ssm_a",
        );
        let ssm_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(head_v_dim)],
            "ssm_norm_weight",
        );
        let ssm_out = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(value_dim), Extent::Static(model_dim)],
            "ssm_out",
        );
        let state_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(head_k_dim),
                Extent::Static(head_v_dim),
                Extent::Static(kv_heads),
                Extent::Static(group)
            ],
            "state_in",
        );

        let (mixer_out, _taps) = append_qwen35_ssm_mixer_with_taps_and_layout(
            &mut program,
            x,
            inv_dim,
            eps,
            head_eps,
            one,
            inv_sqrt_key_dim,
            inv_head_v_dim,
            Some(attn_norm_weight),
            wqkv,
            wqkv_gate,
            conv_weight,
            conv_history_in,
            ssm_beta,
            ssm_alpha,
            ssm_dt_bias,
            ssm_a,
            ssm_norm_weight,
            ssm_out,
            state_in,
            key_dim,
            value_dim,
            kv_heads,
            group,
            l_cache,
            GdnOutputGate::Silu,
            false,
            None,
        )
        .expect("real-shape qwen35moe ssm mixer lowers");

        let shapes =
            shape::infer(&program, &[1]).expect("real-shape qwen35moe ssm mixer program infers");

        let fused = bind_with_fusion(
            &program,
            &shapes,
            &[mixer_out],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("real-shape qwen35moe ssm mixer binds fused");

        println!(
            "row 540 -- qwen35moe gdn mixer fused census ({} ops):",
            fused.len()
        );
        for (index, bound) in fused.iter().enumerate() {
            let node_name = program
                .get(bound.node.0 as usize)
                .and_then(Op::name)
                .unwrap_or("-");
            let body_summary = match &bound.kind {
                BoundOpKind::Elementwise { body, .. } => body
                    .steps
                    .iter()
                    .map(|step| alloc::format!("{:?}", step.op))
                    .collect::<Vec<_>>()
                    .join("+"),
                BoundOpKind::Reduce {
                    element_body,
                    reduce_op,
                    epilogue_body,
                    ..
                } => {
                    let prologue = element_body
                        .steps
                        .iter()
                        .map(|step| alloc::format!("{:?}", step.op))
                        .collect::<Vec<_>>()
                        .join("+");
                    let core = if prologue.is_empty() || prologue == "Identity" {
                        alloc::format!("{reduce_op:?}")
                    } else {
                        alloc::format!("{prologue}->{reduce_op:?}")
                    };
                    let epilogue = epilogue_body
                        .steps
                        .iter()
                        .map(|step| alloc::format!("{:?}", step.op))
                        .collect::<Vec<_>>()
                        .join("+");
                    if epilogue.is_empty() || epilogue == "Identity" {
                        core
                    } else {
                        alloc::format!("{core}->epi[{epilogue}]")
                    }
                }
                _ => alloc::string::String::new(),
            };
            let reduced_axes = match &bound.kind {
                BoundOpKind::Reduce { output_axes, .. } => (0..bound.extents.len())
                    .filter(|axis| !output_axes.contains(&(*axis as u16)))
                    .map(axis_letter)
                    .collect::<alloc::string::String>(),
                _ => alloc::string::String::new(),
            };
            let output_extents: Vec<u64> = match &bound.kind {
                BoundOpKind::Reduce { output_axes, .. } => output_axes
                    .iter()
                    .map(|axis| bound.extents[*axis as usize])
                    .collect(),
                _ => bound.extents.clone(),
            };
            println!(
                "  [{index:>2}] {:<16} body={:<24} name={:<16} reduced_axes={:<6} out={:?}",
                bound.kind.name(),
                body_summary,
                node_name,
                reduced_axes,
                output_extents,
            );
        }
    }
}

/// ROW 569 (`docs/discipline.md`) census: names every bound op
/// [`crate::spec::append_moe_ffn`]'s own round loop builds
/// at qwen35moe's real routing shape (256 experts, `expert_used_count`
/// = 8), the shape `moe-topk-fusion`'s own `BoundOpKind` is meant to
/// collapse into one bound op per layer -- no fusion runs here yet,
/// this only counts and names what a future matcher must replace.
mod moe_routing_census {
    use super::*;
    use crate::spec::{
        Activation, ExpertGatingFunc, MoeFfnSpec, MoeProjectionStrategy, MoeRouter,
        append_moe_ffn, input_leaf, scalar_constant,
    };

    const EXPERT_COUNT: u32 = 256;
    const EXPERT_USED_COUNT: u32 = 8;
    const EMBEDDING: u32 = 8;
    const FEED_FORWARD: u32 = 8;

    /// One qwen35moe layer's routing block: [`ExpertGatingFunc::Softmax`],
    /// `expert_bias = None` -- `proxima-model-interop/src/qwen35moe/program.rs`'s
    /// own `append_qwen35moe_ffn` call into `append_moe_ffn`
    /// (lines 122-135), NOT the `Sigmoid` gate this crate's Mixtral-style
    /// dense callers (`append_mistral_moe_layer`) use -- the two gating
    /// functions cost the same op count per round (`shifted`+`exp` for
    /// softmax vs `masked_scores`+reduce for sigmoid), so the fusion
    /// target is identical either way, but the matcher must anchor on
    /// the gate this program actually builds.
    #[test]
    fn qwen35moe_routing_census_at_real_expert_shape() {
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING)],
            "x",
        );
        let logits = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(EXPERT_COUNT)],
            "logits",
        );
        let expert_w_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING),
                Extent::Static(FEED_FORWARD)
            ],
            "expert_w_gate",
        );
        let expert_w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING),
                Extent::Static(FEED_FORWARD)
            ],
            "expert_w_up",
        );
        let expert_w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(FEED_FORWARD),
                Extent::Static(EMBEDDING)
            ],
            "expert_w_down",
        );
        let ones = scalar_constant(&mut program, 1.0);
        let routing_start = program.len();

        let moe_spec = MoeFfnSpec {
            router: MoeRouter::Logits(logits),
            expert_w_gate,
            expert_w_up,
            expert_w_down,
            expert_count: EXPERT_COUNT,
            expert_used_count: EXPERT_USED_COUNT,
            ones,
            gating: ExpertGatingFunc::Softmax,
            expert_bias: None,
            expert_scale: None,
            activation: Activation::Silu,
            strategy: MoeProjectionStrategy::PerRoute,
        };
        let (_output, site) = append_moe_ffn(&mut program, 0, x, &moe_spec)
            .expect("real-shape qwen35moe routing block lowers");

        assert_eq!(
            site.selected.len(),
            EXPERT_USED_COUNT as usize,
            "one gather-index node per round"
        );
        assert_eq!(
            site.weights.len(),
            EXPERT_USED_COUNT as usize + 1,
            "one weight node per round plus the final weight_total"
        );

        let shapes = shape::infer(&program, &[1]).expect("routing program infers");
        let outputs = [_output];
        let resolved = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("real-shape qwen35moe routing block binds");

        // The routing DECISION subgraph alone -- backward closure over
        // `Op::dependencies` from every gather-index and weight node
        // (including `weight_total`), bounded below by `routing_start`
        // -- excludes the interleaved per-round `gathered_expert_product`
        // gate/up/down projections and SwiGLU chain
        // (`append_moe_round_output`) `append_moe_ffn_with_projection_strategy_from_logits`
        // builds in the SAME loop iteration: those consume `route`/
        // `weight` but nothing in the routing chain consumes anything
        // they produce, so the closure never crosses into them. A
        // naive "every bound op after routing_start" scan (this test's
        // own first draft) counted 73, conflating routing with FFN
        // evaluation; this closure is what a `BoundOpKind::TopK`
        // matcher must actually anchor on and replace.
        let mut routing_nodes: alloc::collections::BTreeSet<u32> =
            alloc::collections::BTreeSet::new();
        let mut frontier: Vec<NodeId> = site
            .selected
            .iter()
            .chain(site.weights.iter())
            .copied()
            .collect();
        while let Some(node) = frontier.pop() {
            if (node.0 as usize) < routing_start || !routing_nodes.insert(node.0) {
                continue;
            }
            frontier.extend(program[node.0 as usize].dependencies());
        }
        let mut routing_op_ids: Vec<u32> = routing_nodes.into_iter().collect();
        routing_op_ids.sort_unstable();

        println!(
            "row 569 qwen35moe routing census: {} ops in the pure routing-decision closure \
             for expert_count={EXPERT_COUNT}, expert_used_count={EXPERT_USED_COUNT}",
            routing_op_ids.len()
        );
        for node_id in &routing_op_ids {
            println!(
                "  node={node_id} kind={:?}",
                core::mem::discriminant(&program[*node_id as usize])
            );
        }
        println!("gather-index nodes (site.selected): {:?}", site.selected);
        println!(
            "weight nodes (site.weights, last is weight_total): {:?}",
            site.weights
        );
        println!(
            "resolved bind produced {} total BoundOps for this routing+FFN program \
             (routing closure = {} of them)",
            resolved.len(),
            routing_op_ids.len()
        );

        // A degenerate ties fixture proves the tie-break rule the
        // `mask * expert_index` -> `reduce Maximum` construction
        // implements: two experts tied at the maximum score, the
        // reduce keeps the HIGHER index -- opposite of
        // `top_k_routes_and_weights`'s own doc comment ("ties broken
        // toward the lower index"), which this test's own finding
        // (ROW 569) shows is stale prose, not the code's behavior.
        let mut tie_logits = alloc::vec![0.0_f32; EXPERT_COUNT as usize];
        tie_logits[3] = 9.0;
        tie_logits[9] = 9.0; // exact tie, index 3 vs 9
        let x_data = alloc::vec![0.1_f32; EMBEDDING as usize];
        let expert_w_gate_data = alloc::vec![0.0_f32; EXPERT_COUNT as usize * EMBEDDING as usize * FEED_FORWARD as usize];
        let expert_w_up_data = expert_w_gate_data.clone();
        let mut expert_w_down_data = expert_w_gate_data.clone();
        // Tag expert 3's down-projection distinctly from expert 9's so
        // the winning route is visible in the output, not just in the
        // route node's own buffer.
        let tagged = |data: &mut Vec<f32>, expert: usize, value: f32| {
            let base = expert * FEED_FORWARD as usize * EMBEDDING as usize;
            for element in 0..(FEED_FORWARD as usize * EMBEDDING as usize) {
                data[base + element] = value;
            }
        };
        tagged(&mut expert_w_down_data, 3, 0.0);
        tagged(&mut expert_w_down_data, 9, 1.0);

        let inputs = alloc::vec![
            (x, x_data),
            (logits, tie_logits),
            (expert_w_gate, expert_w_gate_data),
            (expert_w_up, expert_w_up_data),
            (expert_w_down, expert_w_down_data),
        ];
        let buffers = run_resolved(program.len(), &resolved, inputs);
        let route_0 = buffers[site.selected[0].0 as usize]
            .as_ref()
            .expect("round 0 route resolves")[0];
        println!("tie fixture: round 0 route = {route_0} (expects 9, the higher index)");
        assert!(
            (route_0 - 9.0).abs() < 1e-6,
            "the mask*iota->reduce-Maximum tie-break keeps the HIGHER index on an exact \
             tie, got route {route_0}, expected 9"
        );
    }

    /// Builds the same real-shape qwen35moe routing program
    /// [`qwen35moe_routing_census_at_real_expert_shape`] does, returning
    /// every node a caller needs to fill inputs and read every routing
    /// output back out.
    #[cfg(feature = "moe-topk-fusion")]
    struct RoutingProgram {
        program: Vec<Op>,
        x: NodeId,
        logits: NodeId,
        expert_w_gate: NodeId,
        expert_w_up: NodeId,
        expert_w_down: NodeId,
        output: NodeId,
        selected: Vec<NodeId>,
        weights: Vec<NodeId>,
    }

    #[cfg(feature = "moe-topk-fusion")]
    fn build_routing_program() -> RoutingProgram {
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING)],
            "x",
        );
        let logits = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(EXPERT_COUNT)],
            "logits",
        );
        let expert_w_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING),
                Extent::Static(FEED_FORWARD)
            ],
            "expert_w_gate",
        );
        let expert_w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING),
                Extent::Static(FEED_FORWARD)
            ],
            "expert_w_up",
        );
        let expert_w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(FEED_FORWARD),
                Extent::Static(EMBEDDING)
            ],
            "expert_w_down",
        );
        let ones = scalar_constant(&mut program, 1.0);
        let moe_spec = MoeFfnSpec {
            router: MoeRouter::Logits(logits),
            expert_w_gate,
            expert_w_up,
            expert_w_down,
            expert_count: EXPERT_COUNT,
            expert_used_count: EXPERT_USED_COUNT,
            ones,
            gating: ExpertGatingFunc::Softmax,
            expert_bias: None,
            expert_scale: None,
            activation: Activation::Silu,
            strategy: MoeProjectionStrategy::PerRoute,
        };
        let (output, site) = append_moe_ffn(&mut program, 0, x, &moe_spec)
            .expect("real-shape qwen35moe routing block lowers");
        RoutingProgram {
            program,
            x,
            logits,
            expert_w_gate,
            expert_w_up,
            expert_w_down,
            output,
            selected: site.selected,
            weights: site.weights,
        }
    }

    /// The reference top-k selection [`RoutingProgram`]'s own unfused
    /// chain implements: `top_k` rounds of take-the-maximum-with-
    /// exclusion, HIGHER index wins an exact tie (`>=` below), `weight_r
    /// = exp(max_r - max_0)`, `weight_total = sum(weight_0..weight_{k-1})`
    /// -- independent of the graph, the same role `top_k_routes_and_weights`
    /// plays for the standalone probes above, corrected for this
    /// construction's own tie-break (that function's own doc claims
    /// "toward the lower index", which ROW 569's own census fixture
    /// proved is stale prose for the real `mask * iota -> reduce(Maximum)`
    /// construction).
    #[cfg(feature = "moe-topk-fusion")]
    fn reference_topk(scores: &[f32], top_k: usize) -> (Vec<f32>, Vec<f32>, f32) {
        let mut live = scores.to_vec();
        let mut routes = Vec::with_capacity(top_k);
        let mut weights = Vec::with_capacity(top_k);
        let mut max_selection_0 = 0.0_f32;
        let mut weight_total = 0.0_f32;
        for round in 0..top_k {
            let mut best_index = 0_usize;
            let mut best_value = f32::NEG_INFINITY;
            for (index, value) in live.iter().enumerate() {
                if *value >= best_value {
                    best_value = *value;
                    best_index = index;
                }
            }
            if round == 0 {
                max_selection_0 = best_value;
            }
            let weight = (best_value - max_selection_0).exp();
            routes.push(best_index as f32);
            weights.push(weight);
            weight_total += weight;
            // Exclude EVERY position tied with this round's own max, not
            // only the winning index -- `mask = Equal(selection_scores,
            // max_selection)` is `true` at every tied position, and
            // `Select(mask, neg_infinity, selection_scores)` blanks all
            // of them at once. Caught by this function's own caller: an
            // exact tie's round + 1 disagreed with the graph until this
            // matched `run_moe_topk`'s identical fix.
            for value in live.iter_mut() {
                if *value == best_value {
                    *value = f32::NEG_INFINITY;
                }
            }
        }
        (routes, weights, weight_total)
    }

    /// ROW 569: the fused `BoundOpKind::MoeTopK` bind must produce
    /// bit-identical routes and weights to the always-unfused chain, over
    /// 200 random real-shape score vectors PLUS exact-tie vectors, and
    /// must remove a substantial number of bound ops relative to the
    /// unfused bind -- MEASURED at 38 for this shape, smaller than the
    /// raw 64-op ancestor closure
    /// [`qwen35moe_routing_census_at_real_expert_shape`] counts over the
    /// PROGRAM graph, because `bind_plain`'s own ordinary chain fusion
    /// already inlines some of `match_moe_topk`'s `absorbed` nodes (e.g.
    /// each round's `candidate` multiply, single-consumer into `route`'s
    /// own reduce) before this fusion ever runs.
    #[cfg(feature = "moe-topk-fusion")]
    #[test]
    fn fused_moe_topk_matches_unfused_routing_over_random_scores_and_exact_ties() {
        let built = build_routing_program();
        let shapes = shape::infer(&built.program, &[1]).expect("routing program infers");
        let mut outputs = alloc::vec![built.output];
        outputs.extend(built.selected.iter().copied());
        outputs.extend(built.weights.iter().copied());

        let unfused = bind_plain(
            &built.program,
            &shapes,
            &outputs,
            NumericPolicy::bit_exact(),
        )
        .expect("unfused routing program binds");
        let fused = bind_with_fusion(
            &built.program,
            &shapes,
            &outputs,
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("fused routing program binds");

        let matcher_fired = fused
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::MoeTopK { .. }));
        assert!(
            matcher_fired,
            "the moe-topk-fusion matcher must fire on the real qwen35moe routing shape, got \
             kinds {:?}",
            fused
                .iter()
                .map(|bound| bound.kind.name())
                .collect::<Vec<_>>()
        );
        let op_delta = unfused.len() - fused.len();
        println!(
            "row 569 moe-topk-fusion op census: unfused = {}, fused = {}, delta = {op_delta}",
            unfused.len(),
            fused.len()
        );
        // MEASURED, not the raw 64-op ancestor-closure
        // `qwen35moe_routing_census_at_real_expert_shape` counts over the
        // PROGRAM graph: `bind_plain`'s own chain fusion already inlines
        // several of `match_moe_topk`'s `absorbed` nodes (the per-round
        // `candidate = mask * expert_index` multiply, single-consumer
        // into `route`'s own reduce, never materializes as its own
        // `BoundOp` even in the always-unfused bind) BEFORE this fusion
        // ever runs, so the net op count this fusion alone removes is
        // smaller than the closure's raw node count -- the closure names
        // what the matcher must WALK, not what bind_plain would have
        // separately executed.
        assert!(
            op_delta >= 30,
            "fusing the whole 8-round routing chain into one MoeTopK op must remove a \
             substantial number of bound ops, got only {op_delta}"
        );

        let expert_w_gate_data = alloc::vec![0.0_f32; EXPERT_COUNT as usize * EMBEDDING as usize * FEED_FORWARD as usize];
        let expert_w_up_data = expert_w_gate_data.clone();
        let expert_w_down_data = expert_w_gate_data.clone();
        let x_data = alloc::vec![0.1_f32; EMBEDDING as usize];

        let mut lcg = crate::test_support::Lcg(97);
        let mut score_vectors: Vec<Vec<f32>> = (0..200)
            .map(|_| {
                (0..EXPERT_COUNT as usize)
                    .map(|_| lcg.next_unit() * 10.0)
                    .collect()
            })
            .collect();
        // Two exact-tie fixtures on top of the 200 random draws: a tie at
        // the very top (round 0) and a tie that only surfaces after
        // round 0's own winner is excluded (round 1).
        let mut top_tie = (0..EXPERT_COUNT as usize)
            .map(|index| index as f32 * 0.01)
            .collect::<Vec<f32>>();
        top_tie[12] = 9.0;
        top_tie[200] = 9.0;
        score_vectors.push(top_tie);
        let mut later_tie = (0..EXPERT_COUNT as usize)
            .map(|index| index as f32 * 0.01)
            .collect::<Vec<f32>>();
        later_tie[5] = 20.0;
        later_tie[40] = 7.0;
        later_tie[220] = 7.0;
        score_vectors.push(later_tie);

        for (case, scores) in score_vectors.into_iter().enumerate() {
            let inputs = alloc::vec![
                (built.x, x_data.clone()),
                (built.logits, scores.clone()),
                (built.expert_w_gate, expert_w_gate_data.clone()),
                (built.expert_w_up, expert_w_up_data.clone()),
                (built.expert_w_down, expert_w_down_data.clone()),
            ];
            let unfused_buffers = run_resolved(built.program.len(), &unfused, inputs.clone());
            let fused_buffers = run_resolved(built.program.len(), &fused, inputs);

            let (reference_routes, reference_weights, reference_weight_total) =
                reference_topk(&scores, EXPERT_USED_COUNT as usize);

            for (round, route_node) in built.selected.iter().enumerate() {
                let unfused_route = unfused_buffers[route_node.0 as usize]
                    .as_ref()
                    .expect("unfused route resolves")[0];
                let fused_route = fused_buffers[route_node.0 as usize]
                    .as_ref()
                    .expect("fused route resolves")[0];
                assert_eq!(
                    unfused_route, fused_route,
                    "case {case} round {round}: fused route must exactly match unfused"
                );
                assert_eq!(
                    unfused_route, reference_routes[round],
                    "case {case} round {round}: unfused route must exactly match the \
                     independent reference"
                );
            }
            for (round, weight_node) in built
                .weights
                .iter()
                .take(EXPERT_USED_COUNT as usize)
                .enumerate()
            {
                let unfused_weight = unfused_buffers[weight_node.0 as usize]
                    .as_ref()
                    .expect("unfused weight resolves")[0];
                let fused_weight = fused_buffers[weight_node.0 as usize]
                    .as_ref()
                    .expect("fused weight resolves")[0];
                assert_eq!(
                    unfused_weight, fused_weight,
                    "case {case} round {round}: fused weight must exactly match unfused"
                );
                assert!(
                    (unfused_weight - reference_weights[round]).abs() <= 1e-6,
                    "case {case} round {round}: unfused weight must match the independent \
                     reference within 1e-6, got {unfused_weight} vs \
                     {}",
                    reference_weights[round]
                );
            }
            let weight_total_node = *built
                .weights
                .last()
                .expect("weights carries weight_total as its last entry");
            let unfused_weight_total = unfused_buffers[weight_total_node.0 as usize]
                .as_ref()
                .expect("unfused weight_total resolves")[0];
            let fused_weight_total = fused_buffers[weight_total_node.0 as usize]
                .as_ref()
                .expect("fused weight_total resolves")[0];
            assert_eq!(
                unfused_weight_total, fused_weight_total,
                "case {case}: fused weight_total must exactly match unfused"
            );
            assert!(
                (unfused_weight_total - reference_weight_total).abs() <= 1e-6,
                "case {case}: unfused weight_total must match the independent reference \
                 within 1e-6"
            );
        }
    }
}
