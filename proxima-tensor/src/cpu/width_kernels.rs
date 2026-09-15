use super::*;

/// The pre-ROW-12 strict left-to-right fold, kept verbatim as the
/// `len < DOT_LANES` fallback for [`dot_fold_multi_accumulator_binary`] —
/// too few terms for independent lanes to pay for themselves, and this
/// keeps tiny-`k` folds byte-for-byte identical to pre-ROW-12 behavior.
pub(super) fn dot_fold_scalar_binary<F, R>(
    op: F,
    reduce: R,
    slice_a: &[f32],
    slice_b: &[f32],
    fold: DotFold,
) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.seeded {
        let mut acc = fold.init;
        for (&value_a, &value_b) in slice_a.iter().zip(slice_b) {
            acc = reduce(acc, op(value_a, value_b));
        }
        acc
    } else if let ([first_a, rest_a @ ..], [first_b, rest_b @ ..]) = (slice_a, slice_b) {
        let mut acc = op(*first_a, *first_b);
        for (&value_a, &value_b) in rest_a.iter().zip(rest_b) {
            acc = reduce(acc, op(value_a, value_b));
        }
        acc
    } else {
        fold.init
    }
}

/// `DOT_LANES` independent partial accumulators, one per position in each
/// `DOT_LANES`-wide `chunks_exact` block of `slice_a`/`slice_b`, combined
/// via `reduce` (associative by construction: `Add`/`Multiply`/`Maximum`/
/// `Minimum`, the only four reduce ops this fast path ever specializes
/// for), then folded into one scalar via a single `DOT_LANES`-wide
/// horizontal combine at the end. Reassociates the sum relative to the
/// strict left-to-right fold — the numeric result differs from the naive
/// triple loop by float rounding, same as Accelerate/OpenBLAS/ggml (ROW
/// 12, `proxima-tensor/docs/discipline.md`). Operates on matching-length
/// slices via `chunks_exact` (not manual indexing) so the length relation
/// LLVM needs to elide bounds checks and vectorize is visible in the
/// source, the same technique [`reduce_width_binary_monomorphic`] already
/// relies on. `seeded == false` (the `ReduceInit::FirstElement` case)
/// seeds each lane with its own first block value instead of `fold.init`,
/// so no lane ever combines with a non-identity `fold.init` value.
#[inline(always)]
pub(super) fn dot_fold_multi_accumulator_binary<F, R>(
    op: F,
    reduce: R,
    slice_a: &[f32],
    slice_b: &[f32],
    fold: DotFold,
) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.len < DOT_LANES {
        return dot_fold_scalar_binary(op, reduce, slice_a, slice_b, fold);
    }
    let (chunks_a, remainder_a) = slice_a.as_chunks::<DOT_LANES>();
    let (chunks_b, remainder_b) = slice_b.as_chunks::<DOT_LANES>();
    let mut lanes = [fold.init; DOT_LANES];
    let mut seeded = fold.seeded;
    for (chunk_a, chunk_b) in chunks_a.iter().zip(chunks_b) {
        if seeded {
            for ((lane, &value_a), &value_b) in lanes.iter_mut().zip(chunk_a).zip(chunk_b) {
                *lane = reduce(*lane, op(value_a, value_b));
            }
        } else {
            for ((lane, &value_a), &value_b) in lanes.iter_mut().zip(chunk_a).zip(chunk_b) {
                *lane = op(value_a, value_b);
            }
            seeded = true;
        }
    }
    let mut acc = lanes[0];
    for &lane in &lanes[1..] {
        acc = reduce(acc, lane);
    }
    for (&value_a, &value_b) in remainder_a.iter().zip(remainder_b) {
        let value = op(value_a, value_b);
        acc = if seeded { reduce(acc, value) } else { value };
        seeded = true;
    }
    acc
}

/// The pre-ROW-12 strict left-to-right fold, kept verbatim as the
/// `len < DOT_LANES` fallback for [`dot_fold_multi_accumulator_unary`].
pub(super) fn dot_fold_scalar_unary<F, R>(op: F, reduce: R, slice: &[f32], fold: DotFold) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.seeded {
        let mut acc = fold.init;
        for &raw_value in slice {
            acc = reduce(acc, op(raw_value));
        }
        acc
    } else if let [first, rest @ ..] = slice {
        let mut acc = op(*first);
        for &raw_value in rest {
            acc = reduce(acc, op(raw_value));
        }
        acc
    } else {
        fold.init
    }
}

/// Same discipline as [`dot_fold_multi_accumulator_binary`], one operand.
#[inline(always)]
pub(super) fn dot_fold_multi_accumulator_unary<F, R>(op: F, reduce: R, slice: &[f32], fold: DotFold) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.len < DOT_LANES {
        return dot_fold_scalar_unary(op, reduce, slice, fold);
    }
    let (chunks, remainder) = slice.as_chunks::<DOT_LANES>();
    let mut lanes = [fold.init; DOT_LANES];
    let mut seeded = fold.seeded;
    for chunk in chunks {
        if seeded {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane = reduce(*lane, op(value));
            }
        } else {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane = op(value);
            }
            seeded = true;
        }
    }
    let mut acc = lanes[0];
    for &lane in &lanes[1..] {
        acc = reduce(acc, lane);
    }
    for &value in remainder {
        let mapped = op(value);
        acc = if seeded { reduce(acc, mapped) } else { mapped };
        seeded = true;
    }
    acc
}

/// The contraction-dim counterpart of [`reduce_width_fast`]: instead of
/// accumulating one value per width position across many calls (one per `k`),
/// folds the whole contraction range for ONE output position in a single
/// call. Used when the width dim's own stride disqualifies
/// [`reduce_width_fast`] but the contraction dim is affine on every operand
/// the body shape reads (transposed-B GEMM: `k` is contiguous on both
/// operands even though `n` is not).
#[inline(always)]
pub(super) fn reduce_dot_fast(
    shape: &BodyShape,
    reduce_op: ScalarOp,
    raw: &[&[f32]],
    running: &[i64],
    reduction_strides: &[i64],
    fold: DotFold,
) -> f32 {
    let span_of = |index: u16| {
        let index = index as usize;
        OperandSpan {
            data: raw[index],
            base: running[index] as usize,
            stride: reduction_strides[index] as usize,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => reduce_dot_unary(op, reduce_op, span_of(a), fold),
        BodyShape::Binary(op, a, b) => {
            reduce_dot_binary(op, reduce_op, span_of(a), span_of(b), fold)
        }
        BodyShape::FusedAdamUpdate(..) | BodyShape::Generic(_) => {
            unreachable!("fast path is never entered for a Generic or FusedAdamUpdate body shape")
        }
    }
}

/// Same op/reduce_op monomorphized-closure dispatch as [`reduce_width_unary`],
/// folding to one scalar instead of accumulating across a width slice.
pub(super) fn reduce_dot_unary(op: ScalarOp, reduce_op: ScalarOp, span: OperandSpan, fold: DotFold) -> f32 {
    macro_rules! unary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => {
                    reduce_dot_unary_monomorphic($f, |acc: f32, v: f32| acc + v, span, fold)
                }
                ScalarOp::Multiply => {
                    reduce_dot_unary_monomorphic($f, |acc: f32, v: f32| acc * v, span, fold)
                }
                ScalarOp::Maximum => {
                    reduce_dot_unary_monomorphic($f, |acc: f32, v: f32| acc.max(v), span, fold)
                }
                ScalarOp::Minimum => {
                    reduce_dot_unary_monomorphic($f, |acc: f32, v: f32| acc.min(v), span, fold)
                }
                _ => reduce_dot_unary_scalar_dispatch(op, reduce_op, span, fold),
            }
        };
    }
    match op {
        ScalarOp::Identity => unary_op_arm!(|a: f32| a),
        ScalarOp::Negate => unary_op_arm!(|a: f32| -a),
        ScalarOp::Reciprocal => unary_op_arm!(|a: f32| 1.0 / a),
        ScalarOp::Exponential => unary_op_arm!(|a: f32| a.exp()),
        ScalarOp::Logarithm => unary_op_arm!(|a: f32| a.ln()),
        ScalarOp::SquareRoot => unary_op_arm!(|a: f32| a.sqrt()),
        ScalarOp::Tanh => unary_op_arm!(|a: f32| a.tanh()),
        _ => reduce_dot_unary_scalar_dispatch(op, reduce_op, span, fold),
    }
}

/// `seeded` is branched on ONCE, outside the fold loop, same discipline as
/// [`reduce_width_unary_monomorphic`] — the loop body below contains exactly
/// one call to `op` and, past the first term, one call to `reduce`, both
/// inlined non-capturing closures. A strided span delegates to
/// [`reduce_dot_unary_monomorphic_strided`] before the stride-0/1 arms run.
#[inline(always)]
pub(super) fn reduce_dot_unary_monomorphic<F, R>(op: F, reduce: R, span: OperandSpan, fold: DotFold) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if span.is_strided() {
        return reduce_dot_unary_monomorphic_strided(op, reduce, span, fold);
    }
    if span.stride == 1 {
        let slice = &span.data[span.base..span.base + fold.len];
        dot_fold_multi_accumulator_unary(op, reduce, slice, fold)
    } else {
        let value = op(span.data[span.base]);
        if fold.seeded {
            let mut acc = fold.init;
            for _ in 0..fold.len {
                acc = reduce(acc, value);
            }
            acc
        } else if fold.len == 0 {
            fold.init
        } else {
            let mut acc = value;
            for _ in 1..fold.len {
                acc = reduce(acc, value);
            }
            acc
        }
    }
}

/// Mirrors [`reduce_dot_unary_monomorphic`]'s `seeded`/`fold.len == 0`
/// handling for a stride > 1 span, reading each term with
/// [`OperandSpan::at`] instead of a hoisted broadcast scalar — never routed
/// through [`dot_fold_multi_accumulator_unary`], which deliberately
/// reassociates and would silently change output for this newly-widened case.
#[inline(always)]
pub(super) fn reduce_dot_unary_monomorphic_strided<F, R>(
    op: F,
    reduce: R,
    span: OperandSpan,
    fold: DotFold,
) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.seeded {
        let mut acc = fold.init;
        for position in 0..fold.len {
            acc = reduce(acc, op(span.at(position)));
        }
        acc
    } else if fold.len == 0 {
        fold.init
    } else {
        let mut acc = op(span.at(0));
        for position in 1..fold.len {
            acc = reduce(acc, op(span.at(position)));
        }
        acc
    }
}

/// The unaccelerated fallback for a `reduce_op` outside {Add, Multiply,
/// Maximum, Minimum} — same numerical result as
/// [`reduce_dot_unary_monomorphic`], dispatched per term via
/// [`apply_scalar_op`]/[`combine_reduction`]. [`OperandSpan::at`] already
/// generalizes over every stride.
pub(super) fn reduce_dot_unary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    span: OperandSpan,
    fold: DotFold,
) -> f32 {
    let mut acc = fold.init;
    let mut seeded = fold.seeded;
    for step in 0..fold.len {
        let value = apply_scalar_op(op, &[span.at(step)]);
        acc = combine_reduction(reduce_op, acc, value, seeded);
        seeded = true;
    }
    acc
}

/// Same discipline as [`reduce_dot_unary`], for the two-operand case — the
/// contraction-dim counterpart of [`reduce_width_binary`].
pub(super) fn reduce_dot_binary(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    fold: DotFold,
) -> f32 {
    // the multiply-accumulate case — every contraction in every matmul —
    // taken before the generic closure dispatch, because `mul_add` has to be
    // asked for by name (see `dot_fold_fused_multiply_add`). `a.stride == 1
    // && b.stride == 1` already excludes any stride > 1 literally, so this
    // gate does not need widening alongside `operand_is_affine`.
    if FUSED_MULTIPLY_ADD
        && fold.seeded
        && fold.len >= DOT_LANES
        && a.stride == 1
        && b.stride == 1
        && matches!((op, reduce_op), (ScalarOp::Multiply, ScalarOp::Add))
    {
        let slice_a = &a.data[a.base..a.base + fold.len];
        let slice_b = &b.data[b.base..b.base + fold.len];
        return dot_fold_fused_multiply_add(slice_a, slice_b, fold);
    }
    macro_rules! binary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => {
                    reduce_dot_binary_monomorphic($f, |acc: f32, v: f32| acc + v, a, b, fold)
                }
                ScalarOp::Multiply => {
                    reduce_dot_binary_monomorphic($f, |acc: f32, v: f32| acc * v, a, b, fold)
                }
                ScalarOp::Maximum => {
                    reduce_dot_binary_monomorphic($f, |acc: f32, v: f32| acc.max(v), a, b, fold)
                }
                ScalarOp::Minimum => {
                    reduce_dot_binary_monomorphic($f, |acc: f32, v: f32| acc.min(v), a, b, fold)
                }
                _ => reduce_dot_binary_scalar_dispatch(op, reduce_op, a, b, fold),
            }
        };
    }
    match op {
        ScalarOp::Add => binary_op_arm!(|x: f32, y: f32| x + y),
        ScalarOp::Subtract => binary_op_arm!(|x: f32, y: f32| x - y),
        ScalarOp::Multiply => binary_op_arm!(|x: f32, y: f32| x * y),
        ScalarOp::Divide => binary_op_arm!(|x: f32, y: f32| x / y),
        ScalarOp::Maximum => binary_op_arm!(|x: f32, y: f32| x.max(y)),
        ScalarOp::Minimum => binary_op_arm!(|x: f32, y: f32| x.min(y)),
        ScalarOp::Greater => binary_op_arm!(|x: f32, y: f32| f32::from(u8::from(x > y))),
        ScalarOp::Equal => {
            binary_op_arm!(|x: f32, y: f32| f32::from(u8::from((x - y).abs() == 0.0)))
        }
        _ => reduce_dot_binary_scalar_dispatch(op, reduce_op, a, b, fold),
    }
}

/// The `(true, true)` arm folds via [`dot_fold_multi_accumulator_binary`]
/// (`DOT_LANES` independent partial sums, reassociated relative to the
/// naive triple loop — ROW 12) instead of one strict left-to-right chain.
/// This is the exact shape a transposed-B GEMM's per-output-element dot
/// product takes (`proxima-tensor/docs/discipline.md` ROW 10/11/12). A
/// strided operand delegates to [`reduce_dot_binary_monomorphic_strided`]
/// before this match runs.
#[inline(always)]
pub(super) fn reduce_dot_binary_monomorphic<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    fold: DotFold,
) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if a.is_strided() || b.is_strided() {
        return reduce_dot_binary_monomorphic_strided(op, reduce, a, b, fold);
    }
    match (a.stride == 1, b.stride == 1) {
        (true, true) => {
            let slice_a = &a.data[a.base..a.base + fold.len];
            let slice_b = &b.data[b.base..b.base + fold.len];
            dot_fold_multi_accumulator_binary(op, reduce, slice_a, slice_b, fold)
        }
        (true, false) => {
            let slice_a = &a.data[a.base..a.base + fold.len];
            let value_b = b.data[b.base];
            if fold.seeded {
                let mut acc = fold.init;
                for &value_a in slice_a {
                    acc = reduce(acc, op(value_a, value_b));
                }
                acc
            } else if let [first_a, rest_a @ ..] = slice_a {
                let mut acc = op(*first_a, value_b);
                for &value_a in rest_a {
                    acc = reduce(acc, op(value_a, value_b));
                }
                acc
            } else {
                fold.init
            }
        }
        (false, true) => {
            let value_a = a.data[a.base];
            let slice_b = &b.data[b.base..b.base + fold.len];
            if fold.seeded {
                let mut acc = fold.init;
                for &value_b in slice_b {
                    acc = reduce(acc, op(value_a, value_b));
                }
                acc
            } else if let [first_b, rest_b @ ..] = slice_b {
                let mut acc = op(value_a, *first_b);
                for &value_b in rest_b {
                    acc = reduce(acc, op(value_a, value_b));
                }
                acc
            } else {
                fold.init
            }
        }
        (false, false) => {
            let value_a = a.data[a.base];
            let value_b = b.data[b.base];
            let value = op(value_a, value_b);
            if fold.seeded {
                let mut acc = fold.init;
                for _ in 0..fold.len {
                    acc = reduce(acc, value);
                }
                acc
            } else if fold.len == 0 {
                fold.init
            } else {
                let mut acc = value;
                for _ in 1..fold.len {
                    acc = reduce(acc, value);
                }
                acc
            }
        }
    }
}

/// Mirrors [`reduce_dot_binary_monomorphic`]'s `seeded`/`fold.len == 0`
/// handling one position at a time via [`OperandSpan::at`], for the case at
/// least one of `a`/`b` has a stride > 1 — never routed through
/// [`dot_fold_multi_accumulator_binary`], which reassociates.
#[inline(always)]
pub(super) fn reduce_dot_binary_monomorphic_strided<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    fold: DotFold,
) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.seeded {
        let mut acc = fold.init;
        for position in 0..fold.len {
            acc = reduce(acc, op(a.at(position), b.at(position)));
        }
        acc
    } else if fold.len == 0 {
        fold.init
    } else {
        let mut acc = op(a.at(0), b.at(0));
        for position in 1..fold.len {
            acc = reduce(acc, op(a.at(position), b.at(position)));
        }
        acc
    }
}

/// The unaccelerated fallback for a `reduce_op` outside {Add, Multiply,
/// Maximum, Minimum}. [`OperandSpan::at`] already generalizes over every
/// stride, so both reads collapse to one expression regardless of stride.
pub(super) fn reduce_dot_binary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    fold: DotFold,
) -> f32 {
    let mut acc = fold.init;
    let mut seeded = fold.seeded;
    for step in 0..fold.len {
        let value = apply_scalar_op(op, &[a.at(step), b.at(step)]);
        acc = combine_reduction(reduce_op, acc, value, seeded);
        seeded = true;
    }
    acc
}

/// The width-loop fast path for [`run_elementwise`]: no accumulator, no
/// `reduce_op` — every position gets a fresh value written straight to
/// `out`. Same eligibility gate as `run_reduce`
/// ([`body_shape_is_affine_fast_path`]), same [`OperandSpan`] reads, same
/// monomorphized-closure-per-op dispatch technique as ROW 4
/// (`proxima-tensor/docs/discipline.md` ROW 5). `step_values` is only read
/// by the `Generic` arm ([`elementwise_width_generic`]); `Unary`/`Binary`
/// ignore it, same as [`eval_body_shape`]'s own split.
#[inline(always)]
pub(super) fn elementwise_width_fast(
    shape: &BodyShape,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    out: &mut [f32],
    step_values: &mut [f32],
) {
    let span_of = |index: u16| {
        let index = index as usize;
        OperandSpan {
            data: raw[index],
            base: running[index] as usize,
            stride: strides[index] as usize,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => elementwise_width_unary(op, span_of(a), out),
        BodyShape::Binary(op, a, b) => elementwise_width_binary(op, span_of(a), span_of(b), out),
        BodyShape::FusedAdamUpdate(roles, _) => {
            elementwise_width_fused_adam_update(roles, raw, running, out)
        }
        BodyShape::Generic(body) => {
            elementwise_width_generic(body, raw, running, strides, out, step_values)
        }
    }
}

/// The dedicated, register-resident kernel for [`BodyShape::FusedAdamUpdate`]
/// (`docs/discipline.md` ROW 179) — the eight bias-correction scalar steps
/// (`step*ln(beta) -> exp -> 1-that -> reciprocal`, once each for `m` and
/// `v`) are pure rank-0 arithmetic on values that never vary across the
/// element loop, so they are hoisted and computed ONCE, exactly like
/// `run_elementwise_range`'s own loop-invariant-stride doc already
/// establishes for a genuine broadcast operand — reducing the per-element
/// body to the same 8-op chain ROW 176's own standalone `adam_update`
/// microbench measured at 0.2612 ns/element (`m_hat = m*recip_bias1`
/// through `out = param-scaled_update`). `m`/`v`/`param` are read as plain
/// contiguous slices (never through [`OperandSpan`]'s stride-generalized
/// `at` accessor) because [`fused_adam_update_is_affine_fast_path`] already
/// guarantees stride 1 — this is what lets LLVM auto-vectorize the whole
/// per-element chain the same way it already does for that microbench,
/// instead of branching or calling out per step the way a
/// runtime-dispatched interpreter over an arbitrary op sequence must (ROW
/// 177's own `candidate_b`/`candidate_c`, both measured WORSE than the
/// shipped tiled path they were meant to replace). Every arithmetic step
/// and its operand order matches [`apply_scalar_op`] exactly (`Multiply`/
/// `Add` are commutative so operand order is moot there; `Subtract`/
/// `Reciprocal` are not, and every subtraction/reciprocal here matches its
/// own `BodyStep`'s `apply_scalar_op` argument order bit-for-bit, per
/// [`detect_adam_update_roles`]'s own step-by-step doc) — a pure reorder of
/// the SAME expression tree (scalar hoisting included: a rank-0 value
/// computed once outside the loop is bit-identical to the same value
/// recomputed, unchanged, at every position inside it), not a
/// reassociation, so output is bit-identical to `elementwise_width_generic`'s
/// own tiled walk of the identical [`ComposedBody`].
#[inline(always)]
pub(super) fn elementwise_width_fused_adam_update(
    roles: AdamUpdateRoles,
    raw: &[&[f32]],
    running: &[i64],
    out: &mut [f32],
) {
    let width = out.len();
    let slice_of = |index: u16| {
        let index = index as usize;
        let base = running[index] as usize;
        &raw[index][base..base + width]
    };
    let scalar_of = |index: u16| {
        let index = index as usize;
        raw[index][running[index] as usize]
    };
    let m = slice_of(roles.m);
    let v = slice_of(roles.v);
    let param = slice_of(roles.param);
    let learning_rate = scalar_of(roles.learning_rate);
    let epsilon = scalar_of(roles.epsilon);

    let bias1_power = (scalar_of(roles.step_for_bias1) * scalar_of(roles.ln_beta1)).exp();
    let recip_bias1 = 1.0 / (scalar_of(roles.one_for_bias1) - bias1_power);
    let bias2_power = (scalar_of(roles.step_for_bias2) * scalar_of(roles.ln_beta2)).exp();
    let recip_bias2 = 1.0 / (scalar_of(roles.one_for_bias2) - bias2_power);

    for index in 0..width {
        let m_hat = m[index] * recip_bias1;
        let v_hat = v[index] * recip_bias2;
        let sqrt_v_hat = v_hat.sqrt();
        let denominator = sqrt_v_hat + epsilon;
        let recip_denominator = 1.0 / denominator;
        let update = m_hat * recip_denominator;
        let scaled_update = learning_rate * update;
        out[index] = param[index] - scaled_update;
    }
}

/// The width-loop fast path for a fused multi-step [`ComposedBody`]
/// (`BodyShape::Generic`) — the same straight-line shape
/// [`elementwise_width_fast`]'s `Unary`/`Binary` arms already give a
/// single-`ScalarOp` body, generalized to [`apply_body`]'s own step-by-step
/// evaluation instead of a bespoke per-arity function.
///
/// Step-outer, position-inner (the reverse of the position-outer loop this
/// replaced): each step resolves its [`StepArg`]s to plain [`OperandSpan`]s
/// **once**, the same struct [`elementwise_width_unary`]/`_binary` already
/// read, then hands them to that step's arity-specific monomorphic function
/// — [`elementwise_width_unary_monomorphic`] and
/// [`elementwise_width_binary_monomorphic`] are reused verbatim for arity
/// 1/2, [`elementwise_width_ternary_monomorphic`] added for `Select`'s arity
/// 3. Each of those matches `contiguous`/`broadcast` per operand **once,
/// before** the position loop, so the loop body itself is a fixed slice (or
/// scalar) walk with no per-element branch — earlier `ArgKind::at` case-fell
/// back to a per-element match instead, which measured slower, not faster,
/// than the naive path it replaced. `step_values` is a
/// `body.steps.len() * out.len()` flat row table (row `index` holds step
/// `index`'s value at every position) instead of one scalar reused across
/// steps, because `StepArg::Step` is backwards-only (`BodyStep`'s own doc)
/// and every earlier row must survive until the last step reads it. A
/// `StepArg::Step` read is always a whole prior row, so it is always
/// `contiguous: true` — never the loop-invariant-broadcast case, which only
/// ever applies to a genuine stride-0 `StepArg::Operand`. Evaluation order
/// and every `apply_scalar_op` call match [`apply_body`]'s scalar path
/// exactly: output is bit-identical (`proxima-tensor/docs/discipline.md`
/// ROW 5).
#[inline(always)]
pub(super) fn elementwise_width_generic(
    body: &ComposedBody,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    out: &mut [f32],
    step_values: &mut [f32],
) {
    let mut tile_start = 0usize;
    while tile_start < out.len() {
        let tile_len = GENERIC_WIDTH_TILE.min(out.len() - tile_start);
        elementwise_width_generic_tile(
            body,
            raw,
            running,
            strides,
            tile_start,
            &mut out[tile_start..tile_start + tile_len],
            step_values,
        );
        tile_start += tile_len;
    }
}

/// Width block one [`elementwise_width_generic`] pass evaluates the whole
/// fused chain over. Step-outer/position-inner evaluation makes one full
/// pass across the width PER STEP, so a 6-step body on a 14336-wide row
/// streamed 6 x 56 KiB of intermediates through L2 and allocated a 344 KiB
/// `step_values` table per node call — measured by this crate's own
/// `ELEMENTWISE_STEP_VALUES_TICKS` at 1010.8 ns/call over 771 calls per
/// decode step. Blocking the row caps that scratch at `steps * 512` floats
/// whatever the row width, and keeps every intermediate the chain produces
/// L1-resident between the step that writes it and the step that reads it.
/// 512 `f32` is 2 KiB per step row, 12 KiB for the deepest body this
/// program builds, against this core's 128 KiB L1D.
pub(super) const GENERIC_WIDTH_TILE: usize = 512;

/// One [`GENERIC_WIDTH_TILE`] block of [`elementwise_width_generic`].
/// `tile_start` offsets each operand's own width span by its own stride —
/// the only thing blocking changes. Every output position is computed by
/// the same steps in the same order against the same inputs it would have
/// been at full width, so output is bit-identical.
#[inline(always)]
pub(super) fn elementwise_width_generic_tile(
    body: &ComposedBody,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    tile_start: usize,
    out: &mut [f32],
    step_values: &mut [f32],
) {
    let width = out.len();
    let empty: &[f32] = &[];
    for (index, step) in body.steps.iter().enumerate() {
        let (earlier, rest) = step_values.split_at_mut(index * width);
        let row = &mut rest[..width];

        let mut spans = [OperandSpan {
            data: empty,
            base: 0,
            stride: 1,
        }; 3];
        for (arg_slot, arg) in step.args.iter().enumerate() {
            spans[arg_slot] = match *arg {
                StepArg::Operand(operand_index) => {
                    let operand_index = operand_index as usize;
                    let stride = strides[operand_index] as usize;
                    OperandSpan {
                        data: raw[operand_index],
                        base: running[operand_index] as usize + tile_start * stride,
                        stride,
                    }
                }
                StepArg::Step(step_index) => {
                    let step_index = step_index as usize;
                    OperandSpan {
                        data: &earlier[step_index * width..(step_index + 1) * width],
                        base: 0,
                        stride: 1,
                    }
                }
            };
        }
        elementwise_width_generic_step(step.op, &spans, row);
    }
    let last = body.steps.len() - 1;
    out.copy_from_slice(&step_values[last * width..(last + 1) * width]);
}

/// Picks `step`'s `ScalarOp` **once** and dispatches to the matching arity's
/// monomorphic function — the `Generic`-body counterpart of
/// [`elementwise_width_unary`]/[`elementwise_width_binary`]'s own
/// once-per-call dispatch, generalized to a `Select`-only ternary case.
#[inline(always)]
pub(super) fn elementwise_width_generic_step(op: ScalarOp, spans: &[OperandSpan; 3], row: &mut [f32]) {
    match op {
        ScalarOp::Identity => elementwise_width_unary_monomorphic(|a: f32| a, spans[0], row),
        ScalarOp::Negate => elementwise_width_unary_monomorphic(|a: f32| -a, spans[0], row),
        ScalarOp::Reciprocal => {
            elementwise_width_unary_monomorphic(|a: f32| 1.0 / a, spans[0], row)
        }
        ScalarOp::Exponential => {
            elementwise_width_unary_monomorphic(|a: f32| a.exp(), spans[0], row)
        }
        ScalarOp::Logarithm => elementwise_width_unary_monomorphic(|a: f32| a.ln(), spans[0], row),
        ScalarOp::SquareRoot => {
            elementwise_width_unary_monomorphic(|a: f32| a.sqrt(), spans[0], row)
        }
        ScalarOp::Tanh => elementwise_width_unary_monomorphic(|a: f32| a.tanh(), spans[0], row),
        ScalarOp::Erf => elementwise_width_unary_monomorphic(erf_f32, spans[0], row),
        ScalarOp::Add => {
            elementwise_width_binary_monomorphic(|a: f32, b: f32| a + b, spans[0], spans[1], row)
        }
        ScalarOp::Subtract => {
            elementwise_width_binary_monomorphic(|a: f32, b: f32| a - b, spans[0], spans[1], row);
        }
        ScalarOp::Multiply => {
            elementwise_width_binary_monomorphic(|a: f32, b: f32| a * b, spans[0], spans[1], row);
        }
        ScalarOp::Divide => {
            elementwise_width_binary_monomorphic(|a: f32, b: f32| a / b, spans[0], spans[1], row)
        }
        ScalarOp::Maximum => {
            elementwise_width_binary_monomorphic(
                |a: f32, b: f32| a.max(b),
                spans[0],
                spans[1],
                row,
            );
        }
        ScalarOp::Minimum => {
            elementwise_width_binary_monomorphic(
                |a: f32, b: f32| a.min(b),
                spans[0],
                spans[1],
                row,
            );
        }
        ScalarOp::Greater => elementwise_width_binary_monomorphic(
            |a: f32, b: f32| f32::from(u8::from(a > b)),
            spans[0],
            spans[1],
            row,
        ),
        ScalarOp::Equal => elementwise_width_binary_monomorphic(
            |a: f32, b: f32| f32::from(u8::from((a - b).abs() == 0.0)),
            spans[0],
            spans[1],
            row,
        ),
        ScalarOp::Select => elementwise_width_ternary_monomorphic(
            |condition: f32, when_true: f32, when_false: f32| {
                if condition != 0.0 {
                    when_true
                } else {
                    when_false
                }
            },
            spans[0],
            spans[1],
            spans[2],
            row,
        ),
    }
}

/// The `Select`-arity counterpart of
/// [`elementwise_width_binary_monomorphic`]: every operand's
/// `contiguous`/`broadcast` case is matched **once**, before the position
/// loop, so each of the eight combinations runs a fixed slice-or-scalar walk
/// with no per-element branch. A fully-broadcast step (all three operands
/// stride-0) computes `op` exactly once and splats the single result, same
/// as the all-broadcast arm of the binary/unary cases — never re-evaluated
/// per position, since none of its inputs vary by position. Any operand with
/// a stride > 1 delegates to [`elementwise_width_ternary_monomorphic_strided`]
/// before this match runs, so the eight combinations below still only ever
/// see stride 0 or 1.
#[inline(always)]
pub(super) fn elementwise_width_ternary_monomorphic<F>(
    op: F,
    condition: OperandSpan,
    when_true: OperandSpan,
    when_false: OperandSpan,
    row: &mut [f32],
) where
    F: Fn(f32, f32, f32) -> f32,
{
    if condition.is_strided() || when_true.is_strided() || when_false.is_strided() {
        return elementwise_width_ternary_monomorphic_strided(
            op, condition, when_true, when_false, row,
        );
    }
    let width = row.len();
    match (
        condition.stride == 1,
        when_true.stride == 1,
        when_false.stride == 1,
    ) {
        (true, true, true) => {
            let condition_slice = &condition.data[condition.base..condition.base + width];
            let when_true_slice = &when_true.data[when_true.base..when_true.base + width];
            let when_false_slice = &when_false.data[when_false.base..when_false.base + width];
            for (((slot, &condition_value), &when_true_value), &when_false_value) in row
                .iter_mut()
                .zip(condition_slice)
                .zip(when_true_slice)
                .zip(when_false_slice)
            {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (true, true, false) => {
            let condition_slice = &condition.data[condition.base..condition.base + width];
            let when_true_slice = &when_true.data[when_true.base..when_true.base + width];
            let when_false_value = when_false.data[when_false.base];
            for ((slot, &condition_value), &when_true_value) in
                row.iter_mut().zip(condition_slice).zip(when_true_slice)
            {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (true, false, true) => {
            let condition_slice = &condition.data[condition.base..condition.base + width];
            let when_true_value = when_true.data[when_true.base];
            let when_false_slice = &when_false.data[when_false.base..when_false.base + width];
            for ((slot, &condition_value), &when_false_value) in
                row.iter_mut().zip(condition_slice).zip(when_false_slice)
            {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (true, false, false) => {
            let condition_slice = &condition.data[condition.base..condition.base + width];
            let when_true_value = when_true.data[when_true.base];
            let when_false_value = when_false.data[when_false.base];
            for (slot, &condition_value) in row.iter_mut().zip(condition_slice) {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (false, true, true) => {
            let condition_value = condition.data[condition.base];
            let when_true_slice = &when_true.data[when_true.base..when_true.base + width];
            let when_false_slice = &when_false.data[when_false.base..when_false.base + width];
            for ((slot, &when_true_value), &when_false_value) in
                row.iter_mut().zip(when_true_slice).zip(when_false_slice)
            {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (false, true, false) => {
            let condition_value = condition.data[condition.base];
            let when_true_slice = &when_true.data[when_true.base..when_true.base + width];
            let when_false_value = when_false.data[when_false.base];
            for (slot, &when_true_value) in row.iter_mut().zip(when_true_slice) {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (false, false, true) => {
            let condition_value = condition.data[condition.base];
            let when_true_value = when_true.data[when_true.base];
            let when_false_slice = &when_false.data[when_false.base..when_false.base + width];
            for (slot, &when_false_value) in row.iter_mut().zip(when_false_slice) {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (false, false, false) => {
            let value = op(
                condition.data[condition.base],
                when_true.data[when_true.base],
                when_false.data[when_false.base],
            );
            for slot in row.iter_mut() {
                *slot = value;
            }
        }
    }
}

/// Independent per-position writes, no accumulator to reorder — reads each
/// operand with [`OperandSpan::at`], which already generalizes over stride 0,
/// 1, or any wider constant stride.
#[inline(always)]
pub(super) fn elementwise_width_ternary_monomorphic_strided<F>(
    op: F,
    condition: OperandSpan,
    when_true: OperandSpan,
    when_false: OperandSpan,
    row: &mut [f32],
) where
    F: Fn(f32, f32, f32) -> f32,
{
    for (position, slot) in row.iter_mut().enumerate() {
        *slot = op(
            condition.at(position),
            when_true.at(position),
            when_false.at(position),
        );
    }
}

pub(super) fn elementwise_width_unary(op: ScalarOp, span: OperandSpan, out: &mut [f32]) {
    match op {
        ScalarOp::Identity => elementwise_width_unary_monomorphic(|a: f32| a, span, out),
        ScalarOp::Negate => elementwise_width_unary_monomorphic(|a: f32| -a, span, out),
        ScalarOp::Reciprocal => elementwise_width_unary_monomorphic(|a: f32| 1.0 / a, span, out),
        ScalarOp::Exponential => elementwise_width_unary_monomorphic(|a: f32| a.exp(), span, out),
        ScalarOp::Logarithm => elementwise_width_unary_monomorphic(|a: f32| a.ln(), span, out),
        ScalarOp::SquareRoot => elementwise_width_unary_monomorphic(|a: f32| a.sqrt(), span, out),
        ScalarOp::Tanh => elementwise_width_unary_monomorphic(|a: f32| a.tanh(), span, out),
        ScalarOp::Erf => elementwise_width_unary_monomorphic(erf_f32, span, out),
        ScalarOp::Add
        | ScalarOp::Subtract
        | ScalarOp::Multiply
        | ScalarOp::Divide
        | ScalarOp::Maximum
        | ScalarOp::Minimum
        | ScalarOp::Greater
        | ScalarOp::Equal
        | ScalarOp::Select => {
            unreachable!("BodyShape::Unary only ever carries an arity-1 ScalarOp")
        }
    }
}

#[inline(always)]
pub(super) fn elementwise_width_unary_monomorphic<F>(op: F, span: OperandSpan, out: &mut [f32])
where
    F: Fn(f32) -> f32,
{
    if span.is_strided() {
        return elementwise_width_unary_monomorphic_strided(op, span, out);
    }
    if span.stride == 1 {
        let slice = &span.data[span.base..span.base + out.len()];
        for (slot, &raw_value) in out.iter_mut().zip(slice) {
            *slot = op(raw_value);
        }
    } else {
        let value = op(span.data[span.base]);
        for slot in out.iter_mut() {
            *slot = value;
        }
    }
}

/// Independent per-position writes, no accumulator to reorder — one
/// [`OperandSpan::at`] read per position covers any stride > 1.
#[inline(always)]
pub(super) fn elementwise_width_unary_monomorphic_strided<F>(op: F, span: OperandSpan, out: &mut [f32])
where
    F: Fn(f32) -> f32,
{
    for (position, slot) in out.iter_mut().enumerate() {
        *slot = op(span.at(position));
    }
}

pub(super) fn elementwise_width_binary(op: ScalarOp, a: OperandSpan, b: OperandSpan, out: &mut [f32]) {
    match op {
        ScalarOp::Add => elementwise_width_binary_monomorphic(|x: f32, y: f32| x + y, a, b, out),
        ScalarOp::Subtract => {
            elementwise_width_binary_monomorphic(|x: f32, y: f32| x - y, a, b, out)
        }
        ScalarOp::Multiply => {
            elementwise_width_binary_monomorphic(|x: f32, y: f32| x * y, a, b, out)
        }
        ScalarOp::Divide => elementwise_width_binary_monomorphic(|x: f32, y: f32| x / y, a, b, out),
        ScalarOp::Maximum => {
            elementwise_width_binary_monomorphic(|x: f32, y: f32| x.max(y), a, b, out)
        }
        ScalarOp::Minimum => {
            elementwise_width_binary_monomorphic(|x: f32, y: f32| x.min(y), a, b, out)
        }
        ScalarOp::Greater => elementwise_width_binary_monomorphic(
            |x: f32, y: f32| f32::from(u8::from(x > y)),
            a,
            b,
            out,
        ),
        ScalarOp::Equal => elementwise_width_binary_monomorphic(
            |x: f32, y: f32| f32::from(u8::from((x - y).abs() == 0.0)),
            a,
            b,
            out,
        ),
        ScalarOp::Identity
        | ScalarOp::Negate
        | ScalarOp::Reciprocal
        | ScalarOp::Exponential
        | ScalarOp::Logarithm
        | ScalarOp::SquareRoot
        | ScalarOp::Tanh
        | ScalarOp::Erf
        | ScalarOp::Select => {
            unreachable!("BodyShape::Binary only ever carries an arity-2 ScalarOp")
        }
    }
}

#[inline(always)]
pub(super) fn elementwise_width_binary_monomorphic<F>(op: F, a: OperandSpan, b: OperandSpan, out: &mut [f32])
where
    F: Fn(f32, f32) -> f32,
{
    if a.is_strided() || b.is_strided() {
        return elementwise_width_binary_monomorphic_strided(op, a, b, out);
    }
    let width = out.len();
    match (a.stride == 1, b.stride == 1) {
        (true, true) => {
            let slice_a = &a.data[a.base..a.base + width];
            let slice_b = &b.data[b.base..b.base + width];
            for ((slot, &value_a), &value_b) in out.iter_mut().zip(slice_a).zip(slice_b) {
                *slot = op(value_a, value_b);
            }
        }
        (true, false) => {
            let slice_a = &a.data[a.base..a.base + width];
            let value_b = b.data[b.base];
            for (slot, &value_a) in out.iter_mut().zip(slice_a) {
                *slot = op(value_a, value_b);
            }
        }
        (false, true) => {
            let value_a = a.data[a.base];
            let slice_b = &b.data[b.base..b.base + width];
            for (slot, &value_b) in out.iter_mut().zip(slice_b) {
                *slot = op(value_a, value_b);
            }
        }
        (false, false) => {
            let value = op(a.data[a.base], b.data[b.base]);
            for slot in out.iter_mut() {
                *slot = value;
            }
        }
    }
}

/// Independent per-position writes, no accumulator to reorder — one
/// [`OperandSpan::at`] read per operand per position covers any stride > 1.
#[inline(always)]
pub(super) fn elementwise_width_binary_monomorphic_strided<F>(
    op: F,
    a: OperandSpan,
    b: OperandSpan,
    out: &mut [f32],
) where
    F: Fn(f32, f32) -> f32,
{
    for (position, slot) in out.iter_mut().enumerate() {
        *slot = op(a.at(position), b.at(position));
    }
}

/// The width-loop fast path for [`run_scan`]: unlike `run_elementwise`,
/// output at each position depends on the previous position's accumulated
/// value (`accumulator = reduce_op(accumulator, value)`), a genuine
/// sequential dependency the fold cannot be vectorized around without a
/// parallel-scan restructuring this row does not attempt. What IS removed,
/// same as `run_elementwise`/`run_reduce`: the per-element gather
/// `Option` check, the `operand_values` scratch copy, and the per-element
/// `op`/`reduce_op` dispatch — all replaced by [`OperandSpan`] reads and a
/// once-per-call monomorphized closure pair, restricted to the same four
/// accelerated `reduce_op`s ROW 4 used (`Add`/`Multiply`/`Maximum`/
/// `Minimum`). The `!seeded` special case for the very first element of
/// the very first call (across the whole scan, `seeded` is never reset
/// mid-run) is resolved ONCE before the loop, not re-checked per element.
///
/// `state` bundles `seeded`/`accumulator` — they always travel together —
/// keeping this under clippy's argument-count lint the same way
/// [`OperandSpan`] does for `reduce_width_binary` (ROW 3 addendum).
pub(super) struct ScanState {
    pub(super) seeded: bool,
    pub(super) accumulator: f32,
}

#[inline(always)]
pub(super) fn scan_width_fast(
    shape: &BodyShape,
    reduce_op: ScalarOp,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    out: &mut [f32],
    state: ScanState,
) -> f32 {
    let ScanState {
        seeded,
        accumulator,
    } = state;
    let span_of = |index: u16| {
        let index = index as usize;
        OperandSpan {
            data: raw[index],
            base: running[index] as usize,
            stride: strides[index] as usize,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => {
            scan_width_unary(op, reduce_op, span_of(a), out, seeded, accumulator)
        }
        BodyShape::Binary(op, a, b) => scan_width_binary(
            op,
            reduce_op,
            span_of(a),
            span_of(b),
            out,
            seeded,
            accumulator,
        ),
        BodyShape::FusedAdamUpdate(..) | BodyShape::Generic(_) => {
            unreachable!("fast path is never entered for a Generic or FusedAdamUpdate body shape")
        }
    }
}

pub(super) fn scan_width_unary(
    op: ScalarOp,
    reduce_op: ScalarOp,
    span: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32 {
    macro_rules! unary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => scan_width_unary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc + v,
                    span,
                    out,
                    seeded,
                    accumulator,
                ),
                ScalarOp::Multiply => scan_width_unary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc * v,
                    span,
                    out,
                    seeded,
                    accumulator,
                ),
                ScalarOp::Maximum => scan_width_unary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc.max(v),
                    span,
                    out,
                    seeded,
                    accumulator,
                ),
                ScalarOp::Minimum => scan_width_unary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc.min(v),
                    span,
                    out,
                    seeded,
                    accumulator,
                ),
                _ => {
                    scan_width_unary_scalar_dispatch(op, reduce_op, span, out, seeded, accumulator)
                }
            }
        };
    }
    match op {
        ScalarOp::Identity => unary_op_arm!(|a: f32| a),
        ScalarOp::Negate => unary_op_arm!(|a: f32| -a),
        ScalarOp::Reciprocal => unary_op_arm!(|a: f32| 1.0 / a),
        ScalarOp::Exponential => unary_op_arm!(|a: f32| a.exp()),
        ScalarOp::Logarithm => unary_op_arm!(|a: f32| a.ln()),
        ScalarOp::SquareRoot => unary_op_arm!(|a: f32| a.sqrt()),
        ScalarOp::Tanh => unary_op_arm!(|a: f32| a.tanh()),
        _ => scan_width_unary_scalar_dispatch(op, reduce_op, span, out, seeded, accumulator),
    }
}

#[inline(always)]
pub(super) fn scan_width_unary_monomorphic<F, R>(
    op: F,
    reduce: R,
    span: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    let width = out.len();
    let mut acc = accumulator;
    let mut start = 0usize;
    if !seeded && width > 0 {
        // position 0 is `.at(0)` regardless of stride (`0 * stride == 0`
        // for any stride) -- the shapes only diverge starting at position 1.
        acc = op(span.at(0));
        out[0] = acc;
        start = 1;
    }
    if span.is_strided() {
        return scan_width_unary_monomorphic_strided(op, reduce, span, out, start, acc);
    }
    if span.stride == 1 {
        let slice = &span.data[span.base..span.base + width];
        for (slot, &raw_value) in out[start..].iter_mut().zip(&slice[start..]) {
            acc = reduce(acc, op(raw_value));
            *slot = acc;
        }
    } else {
        let value = op(span.data[span.base]);
        for slot in out[start..].iter_mut() {
            acc = reduce(acc, value);
            *slot = acc;
        }
    }
    acc
}

/// Continues [`scan_width_unary_monomorphic`]'s fold from `start` via
/// [`OperandSpan::at`], for a stride > 1 span — same strict left-to-right
/// combine order, just read position by position instead of through a
/// contiguous slice or a hoisted broadcast scalar.
#[inline(always)]
pub(super) fn scan_width_unary_monomorphic_strided<F, R>(
    op: F,
    reduce: R,
    span: OperandSpan,
    out: &mut [f32],
    start: usize,
    accumulator: f32,
) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    let mut acc = accumulator;
    for (position, slot) in out.iter_mut().enumerate().skip(start) {
        acc = reduce(acc, op(span.at(position)));
        *slot = acc;
    }
    acc
}

pub(super) fn scan_width_unary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    span: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32 {
    let width = out.len();
    let mut acc = accumulator;
    let mut seeded = seeded;
    for (index, slot) in out.iter_mut().enumerate().take(width) {
        let value = apply_scalar_op(op, &[span.at(index)]);
        acc = combine_reduction(reduce_op, acc, value, seeded);
        seeded = true;
        *slot = acc;
    }
    acc
}

pub(super) fn scan_width_binary(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32 {
    macro_rules! binary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => scan_width_binary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc + v,
                    a,
                    b,
                    out,
                    seeded,
                    accumulator,
                ),
                ScalarOp::Multiply => scan_width_binary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc * v,
                    a,
                    b,
                    out,
                    seeded,
                    accumulator,
                ),
                ScalarOp::Maximum => scan_width_binary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc.max(v),
                    a,
                    b,
                    out,
                    seeded,
                    accumulator,
                ),
                ScalarOp::Minimum => scan_width_binary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc.min(v),
                    a,
                    b,
                    out,
                    seeded,
                    accumulator,
                ),
                _ => {
                    scan_width_binary_scalar_dispatch(op, reduce_op, a, b, out, seeded, accumulator)
                }
            }
        };
    }
    match op {
        ScalarOp::Add => binary_op_arm!(|x: f32, y: f32| x + y),
        ScalarOp::Subtract => binary_op_arm!(|x: f32, y: f32| x - y),
        ScalarOp::Multiply => binary_op_arm!(|x: f32, y: f32| x * y),
        ScalarOp::Divide => binary_op_arm!(|x: f32, y: f32| x / y),
        ScalarOp::Maximum => binary_op_arm!(|x: f32, y: f32| x.max(y)),
        ScalarOp::Minimum => binary_op_arm!(|x: f32, y: f32| x.min(y)),
        ScalarOp::Greater => binary_op_arm!(|x: f32, y: f32| f32::from(u8::from(x > y))),
        ScalarOp::Equal => {
            binary_op_arm!(|x: f32, y: f32| f32::from(u8::from((x - y).abs() == 0.0)))
        }
        _ => scan_width_binary_scalar_dispatch(op, reduce_op, a, b, out, seeded, accumulator),
    }
}

#[inline(always)]
pub(super) fn scan_width_binary_monomorphic<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    let width = out.len();
    let mut acc = accumulator;
    let mut start = 0usize;
    if !seeded && width > 0 {
        acc = op(a.at(0), b.at(0));
        out[0] = acc;
        start = 1;
    }
    if a.is_strided() || b.is_strided() {
        return scan_width_binary_monomorphic_strided(op, reduce, a, b, out, start, acc);
    }
    match (a.stride == 1, b.stride == 1) {
        (true, true) => {
            let slice_a = &a.data[a.base..a.base + width];
            let slice_b = &b.data[b.base..b.base + width];
            for ((slot, &value_a), &value_b) in out[start..]
                .iter_mut()
                .zip(&slice_a[start..])
                .zip(&slice_b[start..])
            {
                acc = reduce(acc, op(value_a, value_b));
                *slot = acc;
            }
        }
        (true, false) => {
            let slice_a = &a.data[a.base..a.base + width];
            let value_b = b.data[b.base];
            for (slot, &value_a) in out[start..].iter_mut().zip(&slice_a[start..]) {
                acc = reduce(acc, op(value_a, value_b));
                *slot = acc;
            }
        }
        (false, true) => {
            let value_a = a.data[a.base];
            let slice_b = &b.data[b.base..b.base + width];
            for (slot, &value_b) in out[start..].iter_mut().zip(&slice_b[start..]) {
                acc = reduce(acc, op(value_a, value_b));
                *slot = acc;
            }
        }
        (false, false) => {
            let value_a = a.data[a.base];
            let value_b = b.data[b.base];
            for slot in out[start..].iter_mut() {
                acc = reduce(acc, op(value_a, value_b));
                *slot = acc;
            }
        }
    }
    acc
}

/// Continues [`scan_width_binary_monomorphic`]'s fold from `start` via
/// [`OperandSpan::at`], for the case at least one of `a`/`b` has a stride > 1
/// — same strict left-to-right combine order as every other arm here.
#[inline(always)]
pub(super) fn scan_width_binary_monomorphic_strided<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    out: &mut [f32],
    start: usize,
    accumulator: f32,
) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    let mut acc = accumulator;
    for (position, slot) in out.iter_mut().enumerate().skip(start) {
        acc = reduce(acc, op(a.at(position), b.at(position)));
        *slot = acc;
    }
    acc
}

pub(super) fn scan_width_binary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32 {
    let width = out.len();
    let mut acc = accumulator;
    let mut seeded = seeded;
    for (index, slot) in out.iter_mut().enumerate().take(width) {
        let value = apply_scalar_op(op, &[a.at(index), b.at(index)]);
        acc = combine_reduction(reduce_op, acc, value, seeded);
        seeded = true;
        *slot = acc;
    }
    acc
}

/// Evaluates a (possibly fused) [`ComposedBody`] for one iteration step:
/// `operand_values[i]` is the freshly-read value of physical operand `i`,
/// `step_values` is scratch sized `body.steps.len()` the caller reuses
/// across every step of a run rather than allocating it per element — each
/// step's own value lands in `step_values[index]` as it is computed, so a
/// later step's `StepArg::Step` reference always reads an already-written
/// slot (steps only ever reference earlier steps).
///
/// Only reached through [`BodyShape::Generic`] now — [`eval_body_shape`]'s
/// `Unary`/`Binary` arms bypass this entirely for the common single-step
/// case, so this stays the slow-but-general path for real fused chains.
pub(super) fn apply_body(body: &ComposedBody, operand_values: &[f32], step_values: &mut [f32]) -> f32 {
    for (index, step) in body.steps.iter().enumerate() {
        let mut args = [0.0f32; 3];
        for (slot, arg) in step.args.iter().enumerate() {
            args[slot] = match arg {
                StepArg::Operand(operand_index) => operand_values[*operand_index as usize],
                StepArg::Step(step_index) => step_values[*step_index as usize],
            };
        }
        step_values[index] = apply_scalar_op(step.op, &args[..step.args.len()]);
    }
    step_values[body.steps.len() - 1]
}

/// Abramowitz & Stegun 7.1.26: a single-branch rational approximation to
/// `erf`, entire in `core` float ops (no `libm` dependency — see this
/// module's own doc for why the crate does not carry one). Published maximum
/// absolute error is `1.5e-7`; measured here in `f32` against 14 reference
/// points (`erf_f32_matches_reference_values_within_f32_epsilon`), the
/// actual max error is `1.1920929e-7` — equal to `f32::EPSILON` itself
/// (`2^-23`), i.e. this approximation is precision-limited by `f32`'s own
/// representable step at these points, not by the formula.
#[inline(always)]
pub(super) fn erf_f32(x: f32) -> f32 {
    const P: f32 = 0.327_591_1;
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let magnitude = x.abs();
    let t = 1.0 / P.mul_add(magnitude, 1.0);
    let poly = t * A5
        .mul_add(t, A4)
        .mul_add(t, A3)
        .mul_add(t, A2)
        .mul_add(t, A1);
    sign * poly.mul_add(-(-magnitude * magnitude).exp(), 1.0)
}

/// Same formula as [`erf_f32`], carried in `f64` for [`Element::apply`]'s
/// `f64` instantiation — not a wider-precision approximation, the same
/// published `1.5e-7` bound, just without f32's own rounding compounding on
/// top of it.
#[inline(always)]
pub(super) fn erf_f64(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let magnitude = x.abs();
    let t = 1.0 / P.mul_add(magnitude, 1.0);
    let poly = t * A5
        .mul_add(t, A4)
        .mul_add(t, A3)
        .mul_add(t, A2)
        .mul_add(t, A1);
    sign * poly.mul_add(-(-magnitude * magnitude).exp(), 1.0)
}

#[inline(always)]
pub(super) fn apply_scalar_op(op: ScalarOp, operands: &[f32]) -> f32 {
    match op {
        ScalarOp::Identity => operands[0],
        ScalarOp::Add => operands[0] + operands[1],
        ScalarOp::Subtract => operands[0] - operands[1],
        ScalarOp::Multiply => operands[0] * operands[1],
        ScalarOp::Divide => operands[0] / operands[1],
        ScalarOp::Maximum => operands[0].max(operands[1]),
        ScalarOp::Minimum => operands[0].min(operands[1]),
        ScalarOp::Negate => -operands[0],
        ScalarOp::Reciprocal => 1.0 / operands[0],
        ScalarOp::Exponential => operands[0].exp(),
        ScalarOp::Logarithm => operands[0].ln(),
        ScalarOp::SquareRoot => operands[0].sqrt(),
        ScalarOp::Tanh => operands[0].tanh(),
        ScalarOp::Erf => erf_f32(operands[0]),
        ScalarOp::Greater => f32::from(u8::from(operands[0] > operands[1])),
        ScalarOp::Equal => f32::from(u8::from((operands[0] - operands[1]).abs() == 0.0)),
        ScalarOp::Select => {
            if operands[0] != 0.0 {
                operands[1]
            } else {
                operands[2]
            }
        }
    }
}

