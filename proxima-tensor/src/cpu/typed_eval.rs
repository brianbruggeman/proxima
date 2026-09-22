use super::*;

// ---------------------------------------------------------------------
// Typed elementwise evaluator: every dtype `reject_non_float32` used to
// reject outright, restricted to elementwise-only programs.
//
// `evaluate`/`evaluate_parallel` stay f32-only by construction (their
// buffers, width-tiling, and dot-fold kernels are `Vec<f32>` end to end);
// regeneralizing that pipeline to every width is the "generic element
// parameter threaded through every kernel" option this module's author
// considered and did not take, for the reason `dtype.rs`'s own doc already
// gives for keeping `DType` a runtime field: a single node can mix several
// dtypes (quantized matmul is `i8 x i8 -> i32`), which one `T` cannot
// describe, and threading a type parameter through every SIMD/dot-fold
// kernel here would monomorphize each one per width — compile time and
// code size scale with the ~13-way product, for kernels most callers never
// invoke at most of those widths.
//
// What follows instead is the other option: one runtime-dispatched
// [`TypedBuffer`] enum, matched once at the entry point
// ([`evaluate_typed`]) to pick a monomorphized [`Element`] instantiation —
// same shape `DType` itself already uses. Every match arm still hands the
// kernel one contiguous `&[T]`/`Vec<T>`, never a per-element tag or a boxed
// scalar, which is what lets a future NEON kernel specialize per width
// later without changing how a buffer is stored: a 128-bit register packs
// 16 `i8` lanes, 8 `i16`, 4 `i32`, or 2 `i64`; `i128`/`u128` have no NEON
// lane width and would run scalar-only even with a kernel written. No SIMD
// kernel is written here — every op below is a scalar loop — this only
// leaves the representation ready for one.
// ---------------------------------------------------------------------

/// A CPU-native scalar [`evaluate_typed`] can execute: every operand and
/// every output is one contiguous `[Self]`, matching what [`TypedBuffer`]
/// stores. `apply` is fallible, not merely a closed match: an integer dtype
/// genuinely cannot execute a transcendental (`exp`/`ln`/`sqrt`/`tanh`/
/// `reciprocal`), an unsigned dtype cannot negate, and integer division has
/// a real undefined case (zero divisor, or `T::MIN / -1`) — each is a named
/// [`TensorError`] at the node it was found, not a panic or a silently wrong
/// answer.
///
/// `'static` is what lets [`run_reduce_typed`]/[`run_scan_typed`] compare
/// `TypeId::of::<T>()` against `TypeId::of::<f32>()` and, on a match,
/// reinterpret this evaluator's `Vec<T>` buffers as the `Vec<f32>` the
/// existing NEON reduce/scan (`run_reduce`/`run_scan`) already take — the
/// specialization that keeps the fast path a single implementation instead
/// of a second copy of the reduction nest.
///
/// Not a pipe, and not converted to one. `DTYPE` is a type-level fact, not
/// a transformation -- it is read only as an ordinary runtime struct field
/// (`dtype: Self::DTYPE` at the two `UnsupportedScalarOp` sites above),
/// never in a const item, array length, match pattern, or `const fn` body
/// anywhere in this workspace, so today the "a `Pipe` impl can't be const"
/// objection is theoretical, not load-bearing. The real reason `Element`
/// stays a trait: `unwrap_block`/`apply`/`reduce_seed`/`from_index` are a
/// per-type dispatch table the 11 `T: Element`-bound functions below
/// (`run_typed_program` through `run_scan_generic`) select at monomorphize
/// time, not a stream of values flowing through combinators -- each is
/// called once per site with its arguments already in hand, nothing is
/// composed. Splitting `apply` out as a pipe would still need a trait
/// bound naming that pipe per `T`, i.e. the same trait under a new name.
pub(super) trait Element: Copy + Default + 'static {
    const DTYPE: DType;

    fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]>;
    fn apply(node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError>;

    /// The reduce seed for `init`, in this element's own type — the typed
    /// counterpart of [`initial_value`]. `None` for [`ReduceInit::FirstElement`],
    /// same as [`initial_value`]: there is no synthetic identity, the first
    /// element visited seeds the accumulator instead.
    fn reduce_seed(init: ReduceInit) -> Option<Self>;

    /// [`BoundOpKind::Iota`]'s output value at position `index`, in this
    /// element's own type — the typed counterpart of [`run_iota`]'s
    /// `index as f32`.
    fn from_index(index: usize) -> Self;

    /// [`BoundOpKind::Constant`]'s literal in this element's own type — the
    /// typed counterpart of [`run_constant`]'s bare `f32`. An integer
    /// element truncates toward zero, the same `as` conversion
    /// [`Element::from_index`] uses in the other direction.
    fn from_literal(value: f32) -> Self;
}

macro_rules! impl_element_signed_integer {
    ($ty:ty, $dtype:expr, $variant:ident) => {
        impl Element for $ty {
            const DTYPE: DType = $dtype;

            fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
                match buffer {
                    TypedBuffer::$variant(data) => Some(data.as_slice()),
                    _ => None,
                }
            }

            fn apply(node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
                Ok(match op {
                    ScalarOp::Identity => args[0],
                    ScalarOp::Add => args[0].wrapping_add(args[1]),
                    ScalarOp::Subtract => args[0].wrapping_sub(args[1]),
                    ScalarOp::Multiply => args[0].wrapping_mul(args[1]),
                    ScalarOp::Divide => {
                        return args[0]
                            .checked_div(args[1])
                            .ok_or(TensorError::CheckedDivisionFailed { node });
                    }
                    ScalarOp::Maximum => args[0].max(args[1]),
                    ScalarOp::Minimum => args[0].min(args[1]),
                    ScalarOp::Negate => args[0].wrapping_neg(),
                    ScalarOp::Greater => Self::from(args[0] > args[1]),
                    ScalarOp::Equal => Self::from(args[0] == args[1]),
                    ScalarOp::Select => {
                        if args[0] != 0 {
                            args[1]
                        } else {
                            args[2]
                        }
                    }
                    ScalarOp::Reciprocal
                    | ScalarOp::Exponential
                    | ScalarOp::Logarithm
                    | ScalarOp::SquareRoot
                    | ScalarOp::Tanh
                    | ScalarOp::Erf => {
                        return Err(TensorError::UnsupportedScalarOp {
                            node,
                            op,
                            dtype: Self::DTYPE,
                        });
                    }
                })
            }

            fn reduce_seed(init: ReduceInit) -> Option<Self> {
                match init {
                    ReduceInit::Zero => Some(0),
                    ReduceInit::One => Some(1),
                    ReduceInit::NegativeInfinity => Some(Self::MIN),
                    ReduceInit::PositiveInfinity => Some(Self::MAX),
                    ReduceInit::FirstElement => None,
                }
            }

            fn from_index(index: usize) -> Self {
                index as $ty
            }

            fn from_literal(value: f32) -> Self {
                value as $ty
            }
        }
    };
}

macro_rules! impl_element_unsigned_integer {
    ($ty:ty, $dtype:expr, $variant:ident) => {
        impl Element for $ty {
            const DTYPE: DType = $dtype;

            fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
                match buffer {
                    TypedBuffer::$variant(data) => Some(data.as_slice()),
                    _ => None,
                }
            }

            fn apply(node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
                Ok(match op {
                    ScalarOp::Identity => args[0],
                    ScalarOp::Add => args[0].wrapping_add(args[1]),
                    ScalarOp::Subtract => args[0].wrapping_sub(args[1]),
                    ScalarOp::Multiply => args[0].wrapping_mul(args[1]),
                    ScalarOp::Divide => {
                        return args[0]
                            .checked_div(args[1])
                            .ok_or(TensorError::CheckedDivisionFailed { node });
                    }
                    ScalarOp::Maximum => args[0].max(args[1]),
                    ScalarOp::Minimum => args[0].min(args[1]),
                    ScalarOp::Greater => Self::from(args[0] > args[1]),
                    ScalarOp::Equal => Self::from(args[0] == args[1]),
                    ScalarOp::Select => {
                        if args[0] != 0 {
                            args[1]
                        } else {
                            args[2]
                        }
                    }
                    ScalarOp::Negate
                    | ScalarOp::Reciprocal
                    | ScalarOp::Exponential
                    | ScalarOp::Logarithm
                    | ScalarOp::SquareRoot
                    | ScalarOp::Tanh
                    | ScalarOp::Erf => {
                        return Err(TensorError::UnsupportedScalarOp {
                            node,
                            op,
                            dtype: Self::DTYPE,
                        });
                    }
                })
            }

            fn reduce_seed(init: ReduceInit) -> Option<Self> {
                match init {
                    ReduceInit::Zero | ReduceInit::NegativeInfinity => Some(0),
                    ReduceInit::One => Some(1),
                    ReduceInit::PositiveInfinity => Some(Self::MAX),
                    ReduceInit::FirstElement => None,
                }
            }

            fn from_index(index: usize) -> Self {
                index as $ty
            }

            fn from_literal(value: f32) -> Self {
                value as $ty
            }
        }
    };
}

impl_element_signed_integer!(i8, DType::Int8, Int8);
impl_element_signed_integer!(i16, DType::Int16, Int16);
impl_element_signed_integer!(i32, DType::Int32, Int32);
impl_element_signed_integer!(i64, DType::Int64, Int64);
impl_element_signed_integer!(i128, DType::Int128, Int128);

impl_element_unsigned_integer!(u8, DType::UInt8, UInt8);
impl_element_unsigned_integer!(u16, DType::UInt16, UInt16);
impl_element_unsigned_integer!(u32, DType::UInt32, UInt32);
impl_element_unsigned_integer!(u64, DType::UInt64, UInt64);
impl_element_unsigned_integer!(u128, DType::UInt128, UInt128);

impl Element for f32 {
    const DTYPE: DType = DType::Float32;

    fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
        match buffer {
            TypedBuffer::Float32(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn apply(_node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
        Ok(apply_scalar_op(op, args))
    }

    fn reduce_seed(init: ReduceInit) -> Option<Self> {
        initial_value(init)
    }

    fn from_index(index: usize) -> Self {
        index as Self
    }

    fn from_literal(value: f32) -> Self {
        value as Self
    }
}

impl Element for f64 {
    const DTYPE: DType = DType::Float64;

    fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
        match buffer {
            TypedBuffer::Float64(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn apply(_node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
        Ok(match op {
            ScalarOp::Identity => args[0],
            ScalarOp::Add => args[0] + args[1],
            ScalarOp::Subtract => args[0] - args[1],
            ScalarOp::Multiply => args[0] * args[1],
            ScalarOp::Divide => args[0] / args[1],
            ScalarOp::Maximum => args[0].max(args[1]),
            ScalarOp::Minimum => args[0].min(args[1]),
            ScalarOp::Negate => -args[0],
            ScalarOp::Reciprocal => 1.0 / args[0],
            ScalarOp::Exponential => args[0].exp(),
            ScalarOp::Logarithm => args[0].ln(),
            ScalarOp::SquareRoot => args[0].sqrt(),
            ScalarOp::Tanh => args[0].tanh(),
            ScalarOp::Erf => erf_f64(args[0]),
            ScalarOp::Greater => f64::from(u8::from(args[0] > args[1])),
            ScalarOp::Equal => f64::from(u8::from((args[0] - args[1]).abs() == 0.0)),
            ScalarOp::Select => {
                if args[0] != 0.0 {
                    args[1]
                } else {
                    args[2]
                }
            }
        })
    }

    fn reduce_seed(init: ReduceInit) -> Option<Self> {
        match init {
            ReduceInit::Zero => Some(0.0),
            ReduceInit::One => Some(1.0),
            ReduceInit::NegativeInfinity => Some(f64::NEG_INFINITY),
            ReduceInit::PositiveInfinity => Some(f64::INFINITY),
            ReduceInit::FirstElement => None,
        }
    }

    fn from_index(index: usize) -> Self {
        index as Self
    }

    fn from_literal(value: f32) -> Self {
        value as Self
    }
}

/// Shared body for [`f16`] and [`bf16`]'s [`Element`] impl: neither type has
/// stable-Rust arithmetic operators (`convert.rs`'s own doc), so every op
/// round-trips through `f32` — widen both operands, run the existing f32
/// scalar table ([`apply_scalar_op`]), narrow the result back. This is a
/// real semantic (one rounding step per op, not the fused half-precision
/// arithmetic a hardware FPU would give), documented here rather than
/// silently assumed by a caller.
macro_rules! impl_element_half_float {
    ($ty:ty, $dtype:expr, $variant:ident) => {
        impl Element for $ty {
            const DTYPE: DType = $dtype;

            fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
                match buffer {
                    TypedBuffer::$variant(data) => Some(data.as_slice()),
                    _ => None,
                }
            }

            fn apply(_node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
                let mut widened = [0.0f32; 3];
                for (slot, value) in widened.iter_mut().zip(args) {
                    *slot = value.to_f32();
                }
                let result = apply_scalar_op(op, &widened[..args.len()]);
                Ok(Self::from_f32(result))
            }

            fn reduce_seed(init: ReduceInit) -> Option<Self> {
                initial_value(init).map(Self::from_f32)
            }

            fn from_index(index: usize) -> Self {
                Self::from_f32(index as f32)
            }

            fn from_literal(value: f32) -> Self {
                Self::from_f32(value)
            }
        }
    };
}

impl_element_half_float!(f16, DType::Float16, Float16);
impl_element_half_float!(bf16, DType::BFloat16, BFloat16);

/// One contiguous typed buffer, tagged by which native type backs it — the
/// storage half of [`evaluate_typed`]'s runtime dispatch. Every variant is a
/// plain `Vec<T>`: a whole buffer is tagged, never a scalar, which is what
/// keeps every operand a contiguous, SIMD-ready slice once a kernel is
/// written for it (see this module's typed-evaluator doc). `Bool` has no
/// variant yet — its storage convention (packed bits vs. one byte per
/// element) is undecided; see `typed_program_plan` for the boundary this
/// actually enforces today. `Float16`/`BFloat16` route every arithmetic op
/// through an `f32` round-trip (`Element`'s half-float impl, above) since
/// neither has stable-Rust arithmetic operators of its own.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedBuffer {
    Int8(Vec<i8>),
    UInt8(Vec<u8>),
    Int16(Vec<i16>),
    UInt16(Vec<u16>),
    Int32(Vec<i32>),
    UInt32(Vec<u32>),
    Int64(Vec<i64>),
    UInt64(Vec<u64>),
    Int128(Vec<i128>),
    UInt128(Vec<u128>),
    Float16(Vec<f16>),
    BFloat16(Vec<bf16>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

impl TypedBuffer {
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Int8(_) => DType::Int8,
            Self::UInt8(_) => DType::UInt8,
            Self::Int16(_) => DType::Int16,
            Self::UInt16(_) => DType::UInt16,
            Self::Int32(_) => DType::Int32,
            Self::UInt32(_) => DType::UInt32,
            Self::Int64(_) => DType::Int64,
            Self::UInt64(_) => DType::UInt64,
            Self::Int128(_) => DType::Int128,
            Self::UInt128(_) => DType::UInt128,
            Self::Float16(_) => DType::Float16,
            Self::BFloat16(_) => DType::BFloat16,
            Self::Float32(_) => DType::Float32,
            Self::Float64(_) => DType::Float64,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Int8(data) => data.len(),
            Self::UInt8(data) => data.len(),
            Self::Int16(data) => data.len(),
            Self::UInt16(data) => data.len(),
            Self::Int32(data) => data.len(),
            Self::UInt32(data) => data.len(),
            Self::Int64(data) => data.len(),
            Self::UInt64(data) => data.len(),
            Self::Int128(data) => data.len(),
            Self::UInt128(data) => data.len(),
            Self::Float16(data) => data.len(),
            Self::BFloat16(data) => data.len(),
            Self::Float32(data) => data.len(),
            Self::Float64(data) => data.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// [`typed_program_plan`]'s answer: either every node in the program shares
/// one dtype (the only shape this evaluator supported before mixed
/// precision), or exactly one dtype change occurs, and only at a
/// [`Op::Reduce`] node's own accumulator — the quantized-accumulate shape
/// (`i8` operand folded into an `i32` accumulator) that a single uniform
/// dtype cannot express. See [`typed_program_plan`]'s own doc for the
/// structural check that produces this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TypedPlan {
    Uniform(DType),
    Widened { operand: DType, accumulator: DType },
}

/// Validates a program is executable by [`evaluate_typed`] and returns its
/// [`TypedPlan`].
///
/// A THIRD role sits alongside the operand/accumulator pair below: a
/// gather's `indices` node ([`index_node_ids`], the same structural
/// detection [`reject_non_float32`]'s f32 pipeline already uses to exempt
/// gather indices from its own uniform-dtype rule). An index node is
/// exempt from the uniform/widened dtype check entirely — it may carry any
/// [`DType::is_integer`] dtype regardless of what the rest of the program
/// runs at — but a non-integer index dtype (a float, or `Bool`) is an
/// honest `NotLowerable`, never silently coerced. [`run_typed_program`] and
/// [`run_widened_program`] execute this role for real: [`canonical_index_buffers`]
/// widens every index node's caller-supplied buffer into one canonical
/// `i64` table once, up front, and [`fill_gather_cursors_typed`] reads a
/// gathered operand's fetched index from that table instead of the
/// compute-dtype operand table — the plan only had to stop rejecting the
/// shape at the door once that table existed to back it.
///
/// Two shapes pass beyond that, everything else is
/// [`TensorError::NotLowerable`]:
///
/// - **uniform** — every non-index node shares one dtype. This is the
///   whole-program restriction this function always enforced; it is
///   unchanged for any program that never mixes dtypes, which is what keeps
///   the existing f32 NEON fast path (`run_reduce_typed`/`run_scan_typed`'s
///   `T = f32` specialization) reachable exactly as before.
/// - **widened** — every non-index node up to some position shares one
///   dtype (`operand`), the node at that position is an [`Op::Reduce`]
///   whose own dtype differs (`accumulator`), and every node from there on
///   shares `accumulator`. A `Reduce`'s `operand: NodeId` field only ever
///   points backwards (this crate's own SSA invariant — see [`Op::append`]'s
///   doc), so the dtype that changed at that node is provably the fold's
///   operand dtype widening into its accumulator, not an unrelated node
///   happening to differ. [`evaluate_typed`] dispatches this shape to
///   [`run_widened_program`], scoped to the pairs it ships a [`Convert`]
///   [`Pipe`] for — see that function's own doc.
///
/// Any dtype change outside those shapes (a third distinct non-index dtype,
/// or a change at a non-`Reduce` node) is rejected with an honest
/// `NotLowerable` rather than silently picked apart. `Bool` is out at any
/// non-index position — see [`TypedBuffer`]'s doc; `BFloat16`/`Float16` are
/// typed elements like any other (see `Element`'s half-float impl).
pub(super) fn typed_program_plan(program: &[Op]) -> Result<TypedPlan, TensorError> {
    let index_nodes = index_node_ids(program);
    let base_dtype = program
        .iter()
        .enumerate()
        .find(|(position, _)| !index_nodes.contains(&NodeId(*position as u32)))
        .map(|(_, expr)| expr.dtype())
        .ok_or(TensorError::Empty)?;
    let mut widen_at: Option<(usize, DType)> = None;
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        let dtype = expr.dtype();
        if index_nodes.contains(&node) {
            if !dtype.is_integer() {
                return Err(TensorError::NotLowerable {
                    node,
                    reason: "a gather index node must carry an integer dtype",
                });
            }
            continue;
        }
        if dtype == DType::Bool {
            return Err(TensorError::NotLowerable {
                node,
                reason: "the typed evaluator does not support Bool yet",
            });
        }
        if dtype == base_dtype {
            continue;
        }
        match widen_at {
            None => widen_at = Some((position, dtype)),
            Some((_, accumulator)) if accumulator == dtype => {}
            Some(_) => {
                return Err(TensorError::NotLowerable {
                    node,
                    reason: "the typed evaluator supports at most one dtype change per program",
                });
            }
        }
    }
    match widen_at {
        None => Ok(TypedPlan::Uniform(base_dtype)),
        Some((position, accumulator)) => {
            if !matches!(program[position], Op::Reduce(_)) {
                return Err(TensorError::NotLowerable {
                    node: NodeId(position as u32),
                    reason: "a dtype change may only occur at a Reduce node's own accumulator",
                });
            }
            Ok(TypedPlan::Widened {
                operand: base_dtype,
                accumulator,
            })
        }
    }
}

/// One requested output's node, shape, and data — [`evaluate_typed`]'s
/// per-dtype row, and [`run_typed_program`]'s own before it is wrapped into
/// a [`TypedBuffer`].
pub(super) type TypedRow<Data> = (NodeId, Vec<u64>, Data);

/// Run an elementwise-or-reduce tensor program against a caller-chosen
/// non-f32 (or f64) dtype — the full-width counterpart of [`evaluate`] for
/// the programs `reject_non_float32` used to reject outright. See
/// `typed_program_plan` for exactly which programs qualify, and for the
/// uniform case, `DType::Float32` dispatches `run_typed_program` the same as
/// every other width, but that function's own [`Op::Reduce`] handling
/// specializes straight back to the existing NEON `run_reduce`/`run_scan`
/// for `T = f32` — see `run_reduce_typed`'s doc. A `TypedPlan::Widened`
/// program dispatches to `run_widened_program` instead, over the
/// `(operand, accumulator)` pairs that section ships a [`Convert`] for.
pub fn evaluate_typed(
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<TypedBuffer>>, TensorError> {
    match typed_program_plan(program)? {
        TypedPlan::Uniform(dtype) => {
            evaluate_uniform_typed(dtype, program, symbols, blocks, outputs)
        }
        TypedPlan::Widened {
            operand,
            accumulator,
        } => evaluate_widened_typed(operand, accumulator, program, symbols, blocks, outputs),
    }
}

/// [`evaluate_typed`]'s [`TypedPlan::Uniform`] arm: the whole-program,
/// single-dtype dispatch this evaluator always had, unmodified.
pub(super) fn evaluate_uniform_typed(
    dtype: DType,
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<TypedBuffer>>, TensorError> {
    macro_rules! dispatch {
        ($ty:ty, $variant:ident) => {{
            run_typed_program::<$ty>(program, symbols, blocks, outputs)?
                .into_iter()
                .map(|(node, shape, data)| (node, shape, TypedBuffer::$variant(data)))
                .collect()
        }};
    }
    Ok(match dtype {
        DType::Int8 => dispatch!(i8, Int8),
        DType::UInt8 => dispatch!(u8, UInt8),
        DType::Int16 => dispatch!(i16, Int16),
        DType::UInt16 => dispatch!(u16, UInt16),
        DType::Int32 => dispatch!(i32, Int32),
        DType::UInt32 => dispatch!(u32, UInt32),
        DType::Int64 => dispatch!(i64, Int64),
        DType::UInt64 => dispatch!(u64, UInt64),
        DType::Int128 => dispatch!(i128, Int128),
        DType::UInt128 => dispatch!(u128, UInt128),
        DType::Float16 => dispatch!(f16, Float16),
        DType::BFloat16 => dispatch!(bf16, BFloat16),
        DType::Float32 => dispatch!(f32, Float32),
        DType::Float64 => dispatch!(f64, Float64),
        DType::Bool => unreachable!("typed_program_plan already rejected this dtype"),
    })
}

/// [`evaluate_typed`]'s [`TypedPlan::Widened`] arm — the `(operand,
/// accumulator)` dispatch table. Scoped to the pairs actually shipped, not
/// the full `DType x DType` cross product: `(Int8, Int32)` (the
/// quantized-accumulate case `typed_program_plan`'s doc names), `(Int16,
/// Int64)` and `(UInt8, UInt32)` (the same accumulation-overflow shape at
/// other integer widths), and `(Float16, Float32)`/`(BFloat16, Float32)`
/// (a half-precision reduce folded into an f32 accumulator, the same
/// widen-before-fold shape at floating-point widths). Any other pair is an
/// honest [`TensorError::NotLowerable`] — never a silent wrong result from
/// picking the nearer-available width.
pub(super) fn evaluate_widened_typed(
    operand: DType,
    accumulator: DType,
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<TypedBuffer>>, TensorError> {
    match (operand, accumulator) {
        (DType::Int8, DType::Int32) => Ok(run_widened_program::<i8, i32>(
            program, symbols, blocks, outputs,
        )?
        .into_iter()
        .map(|(node, shape, data)| (node, shape, TypedBuffer::Int32(data)))
        .collect()),
        (DType::Int16, DType::Int64) => Ok(run_widened_program::<i16, i64>(
            program, symbols, blocks, outputs,
        )?
        .into_iter()
        .map(|(node, shape, data)| (node, shape, TypedBuffer::Int64(data)))
        .collect()),
        (DType::UInt8, DType::UInt32) => Ok(run_widened_program::<u8, u32>(
            program, symbols, blocks, outputs,
        )?
        .into_iter()
        .map(|(node, shape, data)| (node, shape, TypedBuffer::UInt32(data)))
        .collect()),
        (DType::Float16, DType::Float32) => Ok(run_widened_program::<f16, f32>(
            program, symbols, blocks, outputs,
        )?
        .into_iter()
        .map(|(node, shape, data)| (node, shape, TypedBuffer::Float32(data)))
        .collect()),
        (DType::BFloat16, DType::Float32) => Ok(run_widened_program::<bf16, f32>(
            program, symbols, blocks, outputs,
        )?
        .into_iter()
        .map(|(node, shape, data)| (node, shape, TypedBuffer::Float32(data)))
        .collect()),
        _ => Err(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "the typed evaluator does not ship a mixed-precision reduce pair for this \
                     operand/accumulator combination",
        }),
    }
}

/// Widens one caller-supplied index block into [`GatherCursor`]'s canonical
/// `i64` width, matching the same lossy `raw as i64` truncation
/// [`GatherCursor::fetch_and_advance`]'s f32 sibling already performs at
/// every element read (`i128`/`u128`/`u64` values past `i64::MAX` truncate
/// exactly as they would there) — paid once per index buffer here instead
/// of once per gathered element. A non-integer `TypedBuffer` variant is an
/// honest `NotLowerable`: [`typed_program_plan`] already rejected a
/// non-integer index node's *declared* dtype, this rejects the buffer the
/// caller actually handed over disagreeing with that at the same gate.
pub(super) fn typed_buffer_to_index(
    node: NodeId,
    buffer: &TypedBuffer,
) -> Result<Vec<i64>, TensorError> {
    match buffer {
        TypedBuffer::Int8(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::UInt8(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::Int16(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::UInt16(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::Int32(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::UInt32(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::Int64(data) => Ok(data.clone()),
        TypedBuffer::UInt64(data) => Ok(data.iter().map(|&value| value as i64).collect()),
        TypedBuffer::Int128(data) => Ok(data.iter().map(|&value| value as i64).collect()),
        TypedBuffer::UInt128(data) => Ok(data.iter().map(|&value| value as i64).collect()),
        TypedBuffer::Float16(_)
        | TypedBuffer::BFloat16(_)
        | TypedBuffer::Float32(_)
        | TypedBuffer::Float64(_) => Err(TensorError::NotLowerable {
            node,
            reason: "a gather index buffer must carry an integer dtype",
        }),
    }
}

/// Builds [`fill_gather_cursors_typed`]'s canonical `i64` index-buffer
/// table, one entry per [`index_node_ids`] member, from the same
/// caller-supplied `blocks` [`run_typed_program`]/[`run_widened_program`]
/// bind their own compute-dtype operand table from — every index node this
/// evaluator supports is a caller-supplied [`Op::Input`] leaf, the shape
/// [`crate::spec`]'s `Gather` table construction and every differential
/// gather test in this module actually produce. A computed index node
/// (derived through `Elementwise`/`Reduce`/`Iota`/`Constant` instead of
/// supplied as a block) is an honest `NotLowerable` rather than a second,
/// unexercised execution nest guessed at without a test to prove it right —
/// see [`TensorError::NotLowerable`]'s own doc on preferring a named gap
/// over a silently wrong result.
pub(super) fn canonical_index_buffers(
    program: &[Op],
    shapes: &shape::Shapes,
    block_nodes: &[NodeId],
    blocks: &[TypedBuffer],
) -> Result<Vec<Option<Vec<i64>>>, TensorError> {
    let index_nodes = index_node_ids(program);
    let mut index_buffers: Vec<Option<Vec<i64>>> = vec![None; program.len()];
    for (node, buffer) in block_nodes.iter().zip(blocks.iter()) {
        if !index_nodes.contains(node) {
            continue;
        }
        let data = typed_buffer_to_index(*node, buffer)?;
        let expected = element_count(shapes.of(*node));
        if data.len() != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found: data.len(),
            });
        }
        index_buffers[node.0 as usize] = Some(data);
    }
    for node in &index_nodes {
        if index_buffers[node.0 as usize].is_none() {
            return Err(TensorError::NotLowerable {
                node: *node,
                reason: "a gather index node must be a caller-supplied input; a computed index \
                         node is not supported yet",
            });
        }
    }
    Ok(index_buffers)
}

/// The monomorphic body [`evaluate_typed`] dispatches into per dtype: shape
/// inference, block binding, and a scalar-or-`T=f32`-specialized walk of
/// every resolved node — the same three stages [`prepare`]/[`evaluate_pooled`]
/// run for f32, minus chunk splitting (this evaluator does not parallelize
/// yet).
///
/// Buffer handling mirrors [`prepare`]/[`evaluate_pooled`] rather than
/// forking it: an input block is held as `Cow::Borrowed` (no per-call copy
/// of the caller's data — the `data.to_vec()` this replaced copied every
/// input block on every call regardless of whether the program even used
/// it), a computed node's output comes from [`typed_take_or_allocate`]
/// (reusing a retired buffer's storage instead of a fresh `vec![..]` per
/// node), and [`node_retirement`] — already generic over the buffer type,
/// unmodified here — decides when a buffer is done being read and goes back
/// to the pool via [`typed_retire_into`].
pub(super) fn run_typed_program<T: Element>(
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<Vec<T>>>, TensorError> {
    let shapes = shape::infer(program, symbols)?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };

    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let index_nodes = index_node_ids(program);
    let index_buffers = canonical_index_buffers(program, &shapes, &block_nodes, blocks)?;

    let mut buffers: Vec<Option<Cow<'_, [T]>>> = vec![None; program.len()];
    for (node, buffer) in block_nodes.iter().zip(blocks.iter()) {
        if index_nodes.contains(node) {
            continue;
        }
        let data = T::unwrap_block(buffer).ok_or(TensorError::NotLowerable {
            node: *node,
            reason: "typed evaluator input dtype does not match the program's uniform dtype",
        })?;
        let expected = element_count(shapes.of(*node));
        if data.len() != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found: data.len(),
            });
        }
        buffers[node.0 as usize] = Some(Cow::Borrowed(data));
    }

    let resolved = bind::bind(
        program,
        &shapes,
        &effective_outputs,
        NumericPolicy::bit_exact(),
    )?;

    let retires = node_retirement(&resolved, &effective_outputs);
    let mut free_buffers: Vec<Vec<T>> = Vec::new();
    for (position, node) in resolved.iter().enumerate() {
        let mut output = typed_take_or_allocate(&mut free_buffers, node_output_len(node));
        match &node.kind {
            BoundOpKind::CachedAttention { .. } => {
                return Err(TensorError::NotLowerable {
                    node: node.node,
                    reason: "cached attention binding is not wired into the typed executor",
                });
            }
            BoundOpKind::CachedSoftmaxWeights { .. } => {
                return Err(TensorError::NotLowerable {
                    node: node.node,
                    reason: "cached softmax weights binding is not wired into the typed executor",
                });
            }
            BoundOpKind::GatedDeltaNet { .. } => {
                return Err(TensorError::NotLowerable {
                    node: node.node,
                    reason: "gated delta net binding is not wired into the typed executor",
                });
            }
            BoundOpKind::MoeTopK { .. } => {
                return Err(TensorError::NotLowerable {
                    node: node.node,
                    reason: "moe top-k binding is not wired into the typed executor",
                });
            }
            BoundOpKind::Elementwise { .. } => {
                run_elementwise_typed(node, &buffers, &index_buffers, &mut output)?
            }
            BoundOpKind::Reduce {
                keep: Keep::Reduce, ..
            } => {
                run_reduce_typed(node, &buffers, &index_buffers, &mut output)?;
            }
            BoundOpKind::Reduce {
                keep: Keep::Scan, ..
            } => {
                run_scan_typed(node, &buffers, &index_buffers, &mut output)?;
            }
            BoundOpKind::RoundBatchedReduce { .. } => {
                return Err(TensorError::NotLowerable {
                    node: node.node,
                    reason: "round-batched reduce binding is not wired into the typed executor",
                });
            }
            BoundOpKind::Iota => run_iota_typed(&mut output),
            BoundOpKind::Constant { value } => run_constant_typed(*value, &mut output),
        }
        buffers[node.node.0 as usize] = Some(Cow::Owned(output));
        for retired in &retires[position] {
            typed_retire_into(&mut buffers, *retired, &mut free_buffers);
        }
    }

    Ok(effective_outputs
        .iter()
        .map(|node| {
            let shape = shapes.of(*node).to_vec();
            let data = buffers[node.0 as usize]
                .clone()
                .map(Cow::into_owned)
                .unwrap_or_default();
            (*node, shape, data)
        })
        .collect())
}

/// [`run_typed_program`]'s two-dtype sibling: every node up to a
/// [`Op::Reduce`] runs in `TIn` (the operand width), the `Reduce` node
/// itself and everything downstream of it runs in `TAcc` (the accumulator
/// width) — [`typed_program_plan`]'s `Widened` shape, executed. This is the
/// case a single generic parameter structurally could not express: an `i8`
/// operand folded into an `i32` accumulator needs the accumulator to
/// actually be `i32`-wide in memory, not `i8` wrapped on overflow.
///
/// Two buffer tables instead of one (`buffers_in: [TIn]`, `buffers_out:
/// [TAcc]`), each indexed by [`NodeId`] exactly like [`run_typed_program`]'s
/// single table. A node's own dtype (`BoundOp::dtype`, mirroring
/// [`Op::dtype`]) decides which table it writes into. The one new step is
/// the crossing itself: immediately before running the `Reduce` node (or any
/// node reading a `TIn`-dtype operand from `TAcc` context), that operand's
/// buffer is widened once, elementwise, through [`Convert`]`<TIn,
/// TAcc>`::`call` (the same [`Pipe`] [`crate::convert`] ships for every
/// other conversion in this crate — no bespoke fold, the algebra already had
/// this piece) and the converted copy is stashed into `buffers_out` at that
/// same node id, so every downstream reader (including the `Reduce` itself)
/// sees an ordinary same-type `TAcc` operand from then on.
pub(super) fn run_widened_program<TIn, TAcc>(
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<Vec<TAcc>>>, TensorError>
where
    TIn: Element,
    TAcc: Element,
    Convert<TIn, TAcc>: Pipe<In = TIn, Out = TAcc, Err = core::convert::Infallible>,
{
    let shapes = shape::infer(program, symbols)?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };

    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let index_nodes = index_node_ids(program);
    let index_buffers = canonical_index_buffers(program, &shapes, &block_nodes, blocks)?;

    let mut buffers_in: Vec<Option<Cow<'_, [TIn]>>> = vec![None; program.len()];
    let mut buffers_out: Vec<Option<Cow<'_, [TAcc]>>> = vec![None; program.len()];
    for (node, buffer) in block_nodes.iter().zip(blocks.iter()) {
        if index_nodes.contains(node) {
            continue;
        }
        let expected = element_count(shapes.of(*node));
        if program[node.0 as usize].dtype() == TIn::DTYPE {
            let data = TIn::unwrap_block(buffer).ok_or(TensorError::NotLowerable {
                node: *node,
                reason: "typed evaluator input dtype does not match its node's own dtype",
            })?;
            if data.len() != expected {
                return Err(TensorError::InputSizeMismatch {
                    node: *node,
                    expected,
                    found: data.len(),
                });
            }
            buffers_in[node.0 as usize] = Some(Cow::Borrowed(data));
        } else {
            let data = TAcc::unwrap_block(buffer).ok_or(TensorError::NotLowerable {
                node: *node,
                reason: "typed evaluator input dtype does not match its node's own dtype",
            })?;
            if data.len() != expected {
                return Err(TensorError::InputSizeMismatch {
                    node: *node,
                    expected,
                    found: data.len(),
                });
            }
            buffers_out[node.0 as usize] = Some(Cow::Borrowed(data));
        }
    }

    let resolved = bind::bind(
        program,
        &shapes,
        &effective_outputs,
        NumericPolicy::bit_exact(),
    )?;

    let retires = node_retirement(&resolved, &effective_outputs);
    let mut free_in: Vec<Vec<TIn>> = Vec::new();
    let mut free_out: Vec<Vec<TAcc>> = Vec::new();
    let converter = Convert::<TIn, TAcc>::new();

    for (position, node) in resolved.iter().enumerate() {
        if node.dtype == TIn::DTYPE {
            let mut output = typed_take_or_allocate(&mut free_in, node_output_len(node));
            match &node.kind {
                BoundOpKind::CachedAttention { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "cached attention binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::CachedSoftmaxWeights { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "cached softmax weights binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::GatedDeltaNet { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "gated delta net binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::MoeTopK { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "moe top-k binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::Elementwise { .. } => {
                    run_elementwise_typed(node, &buffers_in, &index_buffers, &mut output)?;
                }
                BoundOpKind::Reduce {
                    keep: Keep::Reduce, ..
                } => {
                    run_reduce_typed(node, &buffers_in, &index_buffers, &mut output)?;
                }
                BoundOpKind::Reduce {
                    keep: Keep::Scan, ..
                } => {
                    run_scan_typed(node, &buffers_in, &index_buffers, &mut output)?;
                }
                BoundOpKind::RoundBatchedReduce { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "round-batched reduce binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::Iota => run_iota_typed(&mut output),
                BoundOpKind::Constant { value } => run_constant_typed(*value, &mut output),
            }
            buffers_in[node.node.0 as usize] = Some(Cow::Owned(output));
        } else {
            for (source, _, _) in node.operands() {
                let needs_widening = buffers_out[source.0 as usize].is_none()
                    && buffers_in[source.0 as usize].is_some();
                if needs_widening {
                    let narrow = buffers_in[source.0 as usize].as_deref().unwrap_or_default();
                    let mut widened: Vec<TAcc> = Vec::with_capacity(narrow.len());
                    for value in narrow {
                        widened.push(match block_on(converter.call(*value)) {
                            Ok(value) => value,
                            Err(never) => match never {},
                        });
                    }
                    buffers_out[source.0 as usize] = Some(Cow::Owned(widened));
                }
            }
            let mut output = typed_take_or_allocate(&mut free_out, node_output_len(node));
            match &node.kind {
                BoundOpKind::CachedAttention { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "cached attention binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::CachedSoftmaxWeights { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "cached softmax weights binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::GatedDeltaNet { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "gated delta net binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::MoeTopK { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "moe top-k binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::Elementwise { .. } => {
                    run_elementwise_typed(node, &buffers_out, &index_buffers, &mut output)?;
                }
                BoundOpKind::Reduce {
                    keep: Keep::Reduce, ..
                } => {
                    run_reduce_typed(node, &buffers_out, &index_buffers, &mut output)?;
                }
                BoundOpKind::Reduce {
                    keep: Keep::Scan, ..
                } => {
                    run_scan_typed(node, &buffers_out, &index_buffers, &mut output)?;
                }
                BoundOpKind::RoundBatchedReduce { .. } => {
                    return Err(TensorError::NotLowerable {
                        node: node.node,
                        reason: "round-batched reduce binding is not wired into the widened executor",
                    });
                }
                BoundOpKind::Iota => run_iota_typed(&mut output),
                BoundOpKind::Constant { value } => run_constant_typed(*value, &mut output),
            }
            buffers_out[node.node.0 as usize] = Some(Cow::Owned(output));
        }
        for retired in &retires[position] {
            typed_retire_into(&mut buffers_in, *retired, &mut free_in);
            typed_retire_into(&mut buffers_out, *retired, &mut free_out);
        }
    }

    Ok(effective_outputs
        .iter()
        .map(|node| {
            let shape = shapes.of(*node).to_vec();
            let data = buffers_out[node.0 as usize]
                .clone()
                .map(Cow::into_owned)
                .unwrap_or_default();
            (*node, shape, data)
        })
        .collect())
}

/// The typed counterpart of [`take_or_allocate`]: same best-fit-by-capacity
/// pool search, generic over [`Element`] instead of hardcoded to `f32`.
pub(super) fn typed_take_or_allocate<T: Element>(
    pool: &mut Vec<Vec<T>>,
    required: usize,
) -> Vec<T> {
    let best_fit = pool
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.capacity() >= required)
        .min_by_key(|(_, buffer)| buffer.capacity())
        .map(|(index, _)| index);

    match best_fit {
        Some(index) => {
            let mut buffer = pool.swap_remove(index);
            buffer.resize(required, T::default());
            buffer
        }
        None => vec![T::default(); required],
    }
}

/// The typed counterpart of [`retire_into`]: same take-and-stash, generic
/// over [`Element`].
pub(super) fn typed_retire_into<T: Element>(
    buffers: &mut [Option<Cow<'_, [T]>>],
    node: NodeId,
    pool: &mut Vec<Vec<T>>,
) {
    if let Some(Cow::Owned(buffer)) = buffers[node.0 as usize].take() {
        pool.push(buffer);
    }
}

/// The typed counterpart of [`operand_buffers`]: every node kind
/// ([`run_elementwise_typed`], [`run_reduce_generic`], [`run_scan_generic`])
/// reads its operands' physical buffers the same way, so this is the one
/// place that walk is written.
pub(super) fn typed_operand_buffers<'a, T: Element>(
    resolved: &BoundOp,
    buffers: &'a [Option<Cow<'_, [T]>>],
) -> Result<Vec<&'a [T]>, TensorError> {
    resolved
        .operands()
        .iter()
        .map(|(source, _, _)| {
            buffers[source.0 as usize]
                .as_deref()
                .ok_or(TensorError::NotLowerable {
                    node: *source,
                    reason: "operand buffer missing at evaluation time",
                })
        })
        .collect()
}

/// The typed counterpart of [`run_iota`]: `output[i] = T::from_index(i)`, at
/// whichever width `T` calls for rather than only f32.
pub(super) fn run_iota_typed<T: Element>(output: &mut [T]) {
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = T::from_index(index);
    }
}

/// The typed counterpart of [`run_constant`].
pub(super) fn run_constant_typed<T: Element>(value: f32, output: &mut [T]) {
    output.fill(T::from_literal(value));
}

/// The typed counterpart of [`run_elementwise`]: same coordinate walk
/// (`fill_running_offsets`/`unflatten_into`/`split_innermost` are pure
/// geometry over `&[u64]`/[`bind::Layout`], with no f32 dependence, so they
/// are shared verbatim) and the same [`GatherCursor`]/[`fill_gather_cursors_typed`]
/// gather step [`run_elementwise`]'s own generic loop uses, sourced from
/// `index_buffers` instead of `buffers` — no width-tile SIMD fast path,
/// which is still f32-only (see [`run_typed_program`]'s doc).
pub(super) fn run_elementwise_typed<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let innermost_dim = outer_extents.len() as u16;
    let raw = typed_operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let mut operand_values = vec![T::default(); raw.len()];
    let mut step_values = vec![T::default(); body.steps.len()];
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| view.stride(innermost_dim))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor<'_, i64>>> =
        (0..raw.len()).map(|_| None).collect();
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    for outer_position in 0..odometer_len(outer_extents) as usize {
        unflatten_into(outer_position as u64, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        fill_gather_cursors_typed(
            resolved,
            index_buffers,
            &outer_coordinate,
            Some(innermost_dim),
            &mut gather_cursors,
        )?;
        let out_base = outer_position * inner_len;

        for step in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(resolved.node)?;
                }
                operand_values[index] = data[offset as usize];
                running[index] += strides[index];
            }
            output[out_base + step] =
                eval_body_typed(resolved.node, body, &operand_values, &mut step_values)?;
        }
    }
    Ok(())
}

/// The typed counterpart of [`apply_body`]: same fused-step walk, fallible
/// per [`Element::apply`] instead of the f32 body's infallible one.
pub(super) fn eval_body_typed<T: Element>(
    node: NodeId,
    body: &ComposedBody,
    operand_values: &[T],
    step_values: &mut [T],
) -> Result<T, TensorError> {
    for (index, step) in body.steps.iter().enumerate() {
        let mut args = [T::default(); 3];
        for (slot, arg) in step.args.iter().enumerate() {
            args[slot] = match arg {
                StepArg::Operand(operand_index) => operand_values[*operand_index as usize],
                StepArg::Step(step_index) => step_values[*step_index as usize],
            };
        }
        step_values[index] = T::apply(node, step.op, &args[..step.args.len()])?;
    }
    Ok(step_values[body.steps.len() - 1])
}

/// Reinterprets a `&[T]` as `&[f32]` with no copy.
///
/// # Safety
/// The caller must have already confirmed `TypeId::of::<T>() ==
/// TypeId::of::<f32>()`; only then are `T` and `f32` provably the same type,
/// which is what makes this pointer reinterpretation sound.
pub(super) unsafe fn reinterpret_slice<T: 'static>(slice: &[T]) -> &[f32] {
    // SAFETY: forwarded from this function's own contract.
    unsafe { core::slice::from_raw_parts(slice.as_ptr().cast::<f32>(), slice.len()) }
}

/// The `&mut` counterpart of [`reinterpret_slice`]; same contract.
///
/// # Safety
/// See [`reinterpret_slice`].
pub(super) unsafe fn reinterpret_slice_mut<T: 'static>(slice: &mut [T]) -> &mut [f32] {
    // SAFETY: forwarded from this function's own contract.
    unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<f32>(), slice.len()) }
}

/// [`Op::Reduce`] with `Keep::Reduce`, at any width [`Element`] covers.
///
/// This is the specialization point the module doc promises: for `T = f32`
/// with no gathered operand, it does not run a second reduction nest at all
/// — it reinterprets the typed evaluator's own `Vec<f32>` buffers as the
/// `&[f32]` the existing NEON-tiled [`run_reduce`] already takes (sound
/// because [`Element`]'s `'static` bound lets [`TypeId`] prove `T` really is
/// `f32` first) and calls that function directly, so the GEMM tiling,
/// dot-fold, and width-fast paths all still fire exactly as they do for
/// [`evaluate`]. A gathered operand skips this specialization even at `T =
/// f32`: [`run_reduce`]'s own [`fill_gather_cursors`] reads an index node's
/// value out of the *same* `&[f32]` buffer table its operands live in, but
/// the typed evaluator's index nodes live in the separate `index_buffers`
/// table [`canonical_index_buffers`] builds — reinterpreting `buffers` alone
/// would leave `run_reduce` unable to see them. Every other width, and every
/// gathered node regardless of width, falls through to
/// [`run_reduce_generic`].
pub(super) fn run_reduce_typed<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    // Scatter is `f32`-only for now: `run_reduce_scatter` (the only
    // execution path this crate ships for a data-dependent `out_map`) reads
    // straight out of the `evaluate`/`evaluate_parallel` `&[f32]` buffer
    // table, not `evaluate_typed`'s per-width `Cow<'_, [T]>` one. Named,
    // never a silent fallback to a wrong (non-scatter) reduction.
    if let BoundOpKind::Reduce {
        out_scatter: Some(_),
        ..
    } = &resolved.kind
    {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "scatter is not yet supported by the typed (non-f32) evaluator",
        });
    }
    let has_gather = resolved
        .operands()
        .iter()
        .any(|(_, _, lookup)| lookup.is_some());
    if !has_gather && TypeId::of::<T>() == TypeId::of::<f32>() {
        let buffers_f32: Vec<Option<&[f32]>> = buffers
            .iter()
            .map(|slot| {
                slot.as_ref().map(|data| {
                    // SAFETY: the `TypeId` check above proves `T == f32`.
                    unsafe { reinterpret_slice(&data[..]) }
                })
            })
            .collect();
        // SAFETY: the `TypeId` check above proves `T == f32`.
        let output_f32 = unsafe { reinterpret_slice_mut(output) };
        return run_reduce(resolved, &buffers_f32, output_f32, None);
    }
    run_reduce_generic(resolved, buffers, index_buffers, output)
}

/// The scalar reduction nest generic over every [`Element`] width — the
/// same (leading, reduction) coordinate walk [`run_reduce`]'s own generic
/// fallback runs (its NEON/width-tile/dot-fold fast paths stay f32-only, so
/// this has no equivalent of them to port), rewritten against
/// [`Element::apply`]/[`eval_body_typed`] instead of `apply_scalar_op`/
/// `eval_body_shape` so it type-checks for every width, and fallible where
/// [`Element::apply`] is (an unsupported op, or an integer division that has
/// no representable result). Same [`GatherCursor`]/[`fill_gather_cursors_typed`]
/// step [`run_reduce`]'s own generic fallback uses, sourced from
/// `index_buffers`.
pub(super) fn run_reduce_generic<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        out_layout,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_reduce_generic is only called for a Keep::Reduce fold")
    };
    let raw = typed_operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let mut operand_values = vec![T::default(); raw.len()];
    let mut step_values = vec![T::default(); body.steps.len()];

    let reduction_dims: Vec<u16> = (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.as_slice().contains(dim))
        .collect();
    let (leading_output_axes, last_output_dim) = output_axes_split(output_axes.as_slice());
    let leading_extents: Vec<u64> = leading_output_axes
        .iter()
        .map(|dim| resolved.extents[*dim as usize])
        .collect();
    let reduction_extents: Vec<u64> = reduction_dims
        .iter()
        .map(|dim| resolved.extents[*dim as usize])
        .collect();
    let width = last_output_dim.map_or(1, |dim| resolved.extents[dim as usize] as usize);

    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| last_output_dim.map_or(0, |dim| view.stride(dim)))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor<'_, i64>>> =
        (0..raw.len()).map(|_| None).collect();
    let mut leading_coordinate = vec![0u64; leading_extents.len()];
    let mut reduction_coordinate = vec![0u64; reduction_extents.len()];
    let mut full_coordinate = vec![0u64; resolved.extents.len()];
    let reduction_total = odometer_len(&reduction_extents);
    let leading_total = odometer_len(&leading_extents);

    let seed = T::reduce_seed(*init).unwrap_or_default();
    let mut accumulator = vec![seed; width];

    for leading_flat in 0..leading_total {
        unflatten_into(leading_flat, &leading_extents, &mut leading_coordinate);
        accumulator.fill(seed);
        let mut seeded = !matches!(init, ReduceInit::FirstElement);

        for reduction_flat in 0..reduction_total {
            unflatten_into(
                reduction_flat,
                &reduction_extents,
                &mut reduction_coordinate,
            );
            merge_coordinates_into(
                leading_output_axes,
                &leading_coordinate,
                &reduction_dims,
                &reduction_coordinate,
                &mut full_coordinate,
            );
            fill_running_offsets(resolved, &full_coordinate, &mut running);
            fill_gather_cursors_typed(
                resolved,
                index_buffers,
                &full_coordinate,
                last_output_dim,
                &mut gather_cursors,
            )?;

            for slot in &mut accumulator {
                for (index, data) in raw.iter().enumerate() {
                    let mut offset = running[index];
                    if let Some(cursor) = gather_cursors[index].as_mut() {
                        offset += cursor.fetch_and_advance(resolved.node)?;
                    }
                    operand_values[index] = data[offset as usize];
                    running[index] += strides[index];
                }
                let value =
                    eval_body_typed(resolved.node, body, &operand_values, &mut step_values)?;
                *slot = if seeded {
                    T::apply(resolved.node, *reduce_op, &[*slot, value])?
                } else {
                    value
                };
            }
            seeded = true;
        }

        merge_coordinates_into(
            leading_output_axes,
            &leading_coordinate,
            &[],
            &[],
            &mut full_coordinate,
        );
        let out_prefix = out_layout.offset_of(&full_coordinate);
        let out_stride = last_output_dim.map_or(0, |dim| out_layout.stride(dim));
        for (slot, value) in accumulator.iter().enumerate() {
            output[(out_prefix + out_stride * slot as i64) as usize] = *value;
        }
    }
    Ok(())
}

/// [`Op::Reduce`] with `Keep::Scan`, at any width [`Element`] covers — the
/// scan counterpart of [`run_reduce_typed`], same `T = f32`, gather-free
/// specialization down to the existing NEON-aware [`run_scan`] (see
/// [`run_reduce_typed`]'s own doc for why a gathered operand skips it).
pub(super) fn run_scan_typed<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    let has_gather = resolved
        .operands()
        .iter()
        .any(|(_, _, lookup)| lookup.is_some());
    if !has_gather && TypeId::of::<T>() == TypeId::of::<f32>() {
        let buffers_f32: Vec<Option<&[f32]>> = buffers
            .iter()
            .map(|slot| {
                slot.as_ref().map(|data| {
                    // SAFETY: the `TypeId` check above proves `T == f32`.
                    unsafe { reinterpret_slice(&data[..]) }
                })
            })
            .collect();
        // SAFETY: the `TypeId` check above proves `T == f32`.
        let output_f32 = unsafe { reinterpret_slice_mut(output) };
        return run_scan(resolved, &buffers_f32, output_f32);
    }
    run_scan_generic(resolved, buffers, index_buffers, output)
}

/// The scalar scan nest generic over every [`Element`] width — [`run_scan`]'s
/// generic fallback (its width-fast SIMD path stays f32-only), rewritten
/// against [`Element::apply`]/[`eval_body_typed`] the same way
/// [`run_reduce_generic`] rewrites [`run_reduce`]'s, including the same
/// [`GatherCursor`]/[`fill_gather_cursors_typed`] step.
pub(super) fn run_scan_generic<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        out_layout,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_scan_generic is only called for a Keep::Scan fold")
    };
    let raw = typed_operand_buffers(resolved, buffers)?;
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let innermost_dim = outer_extents.len() as u16;
    let body = resolved.element_body();
    let mut operand_values = vec![T::default(); raw.len()];
    let mut step_values = vec![T::default(); body.steps.len()];
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| view.stride(innermost_dim))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor<'_, i64>>> =
        (0..raw.len()).map(|_| None).collect();
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    let mut accumulator = T::reduce_seed(*init).unwrap_or_default();
    let mut seeded = !matches!(init, ReduceInit::FirstElement);

    for outer_flat in 0..odometer_len(outer_extents) {
        unflatten_into(outer_flat, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        fill_gather_cursors_typed(
            resolved,
            index_buffers,
            &outer_coordinate,
            Some(innermost_dim),
            &mut gather_cursors,
        )?;
        let mut out_running = out_layout.offset_of(&outer_coordinate);
        let out_stride = out_layout.stride(innermost_dim);

        for _ in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(resolved.node)?;
                }
                operand_values[index] = data[offset as usize];
                running[index] += strides[index];
            }
            let value = eval_body_typed(resolved.node, body, &operand_values, &mut step_values)?;
            accumulator = if seeded {
                T::apply(resolved.node, *reduce_op, &[accumulator, value])?
            } else {
                value
            };
            seeded = true;
            output[out_running as usize] = accumulator;
            out_running += out_stride;
        }
    }
    Ok(())
}
