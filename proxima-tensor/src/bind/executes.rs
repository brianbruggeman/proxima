use super::{BoundOp, BoundOpKind};
use crate::op::Keep;

/// One shape of bound op a backend may or may not render. A variant names a
/// [`BoundOpKind`] (or, for the epilogue forms, a fusion of one) and nothing
/// finer: limits inside a form belong to the form's own admission, not here.
#[non_exhaustive]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Form {
    Elementwise,
    ElementwiseTwin,
    Reduce,
    Scan,
    ReduceEpilogue,
    ReduceBroadcastEpilogue,
    RoundBatchedReduce,
    Iota,
    Constant,
    BlockQuantize,
    CachedAttention,
    CachedSoftmaxWeights,
    GatedDeltaNet,
    MoeTopK,
    TopFractionSelect,
}

impl Form {
    const fn bit(self) -> u32 {
        1_u32 << (self as u8)
    }
}

/// A set of [`Form`]s as one `u32`: `Copy`, hashable, const-constructible, no heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FormSet(u32);

impl FormSet {
    pub const EMPTY: Self = Self(0);

    /// The unfused algebra: what every backend must render for a program to run at all.
    pub const BASE: Self = Self::EMPTY
        .with(Form::Elementwise)
        .with(Form::Reduce)
        .with(Form::Scan)
        .with(Form::Iota)
        .with(Form::Constant);

    #[must_use]
    pub const fn with(self, form: Form) -> Self {
        Self(self.0 | form.bit())
    }

    #[must_use]
    pub const fn without(self, form: Form) -> Self {
        Self(self.0 & !form.bit())
    }

    #[must_use]
    pub const fn contains(self, form: Form) -> bool {
        self.0 & form.bit() != 0
    }

    /// Whether `self` renders every form in `required`; the subset shape of
    /// `proxima_numeric::NumericPolicy::grants`.
    #[must_use]
    pub const fn grants(self, required: Self) -> bool {
        required.0 & !self.0 == 0
    }

    /// The forms in `required` that `self` does not render.
    #[must_use]
    pub const fn missing(self, required: Self) -> Self {
        Self(required.0 & !self.0)
    }

    #[must_use]
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Why an [`Executes`] declined an op.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Refusal {
    Form(Form),
}

/// Whether a backend reads the host's memory in place or from its own.
///
/// Decides where a gather-only table may live: on [`Self::Separate`] memory a
/// table read only by a row gather can stay on the host (see
/// [`crate::partition::row_gathers`]), on [`Self::Unified`] it never needs to.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceMemory {
    Unified,
    Separate,
}

/// Capacity facts about a backend's device, carried beside its [`FormSet`] in
/// [`Executes`] so plan identity hashes both.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceLimits {
    pub memory: DeviceMemory,
}

impl DeviceLimits {
    pub const UNIFIED: Self = Self {
        memory: DeviceMemory::Unified,
    };

    /// The tighter of two limits: a pair is `Separate` if either side is.
    #[must_use]
    pub const fn intersect(self, other: Self) -> Self {
        let separate = matches!(self.memory, DeviceMemory::Separate)
            || matches!(other.memory, DeviceMemory::Separate);
        Self {
            memory: if separate {
                DeviceMemory::Separate
            } else {
                DeviceMemory::Unified
            },
        }
    }
}

/// What one backend renders, as a value.
///
/// Composes the same subset shape as `proxima_numeric::NumericPolicy::grants`
/// ([`FormSet::grants`]), so a backend's capability and a numeric policy are
/// read the same way. It is a value rather than a trait because a value can be
/// intersected ([`Executes::intersect`]), hashed into plan identity, and later
/// loaded from data, so a new target is a table and not a recompile; a trait
/// object can do none of those. Build one from [`Executes::BASE`] with
/// [`Executes::with`], ask it with [`Executes::admit`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Executes {
    pub forms: FormSet,
    pub limits: DeviceLimits,
}

impl Executes {
    /// The reference executor: every form.
    pub const CPU: Self = Self {
        forms: FormSet::BASE
            .with(Form::ElementwiseTwin)
            .with(Form::ReduceEpilogue)
            .with(Form::ReduceBroadcastEpilogue)
            .with(Form::RoundBatchedReduce)
            .with(Form::BlockQuantize)
            .with(Form::CachedAttention)
            .with(Form::CachedSoftmaxWeights)
            .with(Form::GatedDeltaNet)
            .with(Form::MoeTopK)
            .with(Form::TopFractionSelect),
        limits: DeviceLimits::UNIFIED,
    };

    /// The unfused algebra only.
    pub const BASE: Self = Self {
        forms: FormSet::BASE,
        limits: DeviceLimits::UNIFIED,
    };

    #[must_use]
    pub const fn with(self, form: Form) -> Self {
        Self {
            forms: self.forms.with(form),
            limits: self.limits,
        }
    }

    #[must_use]
    pub const fn without(self, form: Form) -> Self {
        Self {
            forms: self.forms.without(form),
            limits: self.limits,
        }
    }

    /// The same forms on the named memory kind.
    #[must_use]
    pub const fn with_memory(self, memory: DeviceMemory) -> Self {
        Self {
            forms: self.forms,
            limits: DeviceLimits { memory },
        }
    }

    #[must_use]
    pub const fn memory(&self) -> DeviceMemory {
        self.limits.memory
    }

    #[must_use]
    pub const fn intersect(self, other: Self) -> Self {
        Self {
            forms: self.forms.intersect(other.forms),
            limits: self.limits.intersect(other.limits),
        }
    }

    /// # Errors
    /// [`Refusal::Form`] naming the form `op` needs and this value lacks.
    #[must_use = "a refusal that is dropped admits the op silently"]
    pub fn admit(&self, op: &BoundOp) -> Result<(), Refusal> {
        let form = form_of(&op.kind);
        if self.forms.contains(form) {
            Ok(())
        } else {
            Err(Refusal::Form(form))
        }
    }
}

fn form_of(kind: &BoundOpKind) -> Form {
    match kind {
        BoundOpKind::CachedAttention { .. } => Form::CachedAttention,
        BoundOpKind::CachedSoftmaxWeights { .. } => Form::CachedSoftmaxWeights,
        BoundOpKind::GatedDeltaNet { .. } => Form::GatedDeltaNet,
        BoundOpKind::MoeTopK { .. } => Form::MoeTopK,
        BoundOpKind::TopFractionSelect { .. } => Form::TopFractionSelect,
        BoundOpKind::Elementwise { .. } => Form::Elementwise,
        BoundOpKind::ElementwiseTwin { .. } => Form::ElementwiseTwin,
        BoundOpKind::Reduce { keep: Keep::Reduce, .. } => Form::Reduce,
        BoundOpKind::Reduce { keep: Keep::Scan, .. } => Form::Scan,
        BoundOpKind::RoundBatchedReduce { .. } => Form::RoundBatchedReduce,
        BoundOpKind::Iota => Form::Iota,
        BoundOpKind::Constant { .. } => Form::Constant,
        BoundOpKind::BlockQuantize { .. } => Form::BlockQuantize,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_numeric::{DType, NumericPolicy};

    use super::*;
    use crate::bind::bind;
    use crate::map::{self, IndexMap};
    use crate::op::{Extent, NodeId, Op, Reduce, ReduceInit, ScalarOp, append};
    use crate::shape::infer;

    fn scalar_op(kind: BoundOpKind) -> BoundOp {
        BoundOp {
            node: NodeId(0),
            dtype: DType::FLOAT32,
            extents: Vec::new(),
            kind,
        }
    }

    fn bound_reduce(keep: Keep) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(8)],
                name: None,
            },
        );
        let last = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep,
                name: None,
            }),
        );
        let shapes = infer(&program, &[]).expect("a static reduce infers");
        bind(&program, &shapes, &[last], NumericPolicy::default())
            .expect("a static reduce binds")
            .pop()
            .expect("the reduce binds to one op")
    }

    fn bound_elementwise() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(64)],
                name: None,
            },
        );
        let activated = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("a static elementwise infers");
        bind(&program, &shapes, &[activated], NumericPolicy::default())
            .expect("a static elementwise binds")
            .pop()
            .expect("the elementwise binds to one op")
    }

    fn cached_attention() -> BoundOp {
        scalar_op(BoundOpKind::CachedAttention {
            operands: Vec::new(),
            query_rows: 1,
            cached_key_rows: 5,
            new_key_rows: 1,
            kv_heads: 1,
            query_groups: 8,
            head_dim: 256,
            rotary_dim: 256,
            scale: 1.0,
            cached_lower_inclusive: i64::MIN,
            new_upper_inclusive: 0,
            arithmetic: crate::bind::CachedAttentionArithmetic::OnlineF32,
            attention_sink: false,
        })
    }

    fn constructible_ops() -> Vec<(Form, BoundOp)> {
        vec![
            (Form::Elementwise, bound_elementwise()),
            (Form::Reduce, bound_reduce(Keep::Reduce)),
            (Form::Scan, bound_reduce(Keep::Scan)),
            (Form::Iota, scalar_op(BoundOpKind::Iota)),
            (Form::Constant, scalar_op(BoundOpKind::Constant { value: 0.5 })),
            (Form::CachedAttention, cached_attention()),
        ]
    }

    #[test]
    fn form_set_grants_a_subset_and_names_what_is_missing() {
        let have = FormSet::BASE.with(Form::MoeTopK);
        let want = FormSet::EMPTY.with(Form::MoeTopK).with(Form::CachedAttention);

        assert!(have.grants(FormSet::BASE));
        assert!(!have.grants(want));
        assert_eq!(have.missing(want), FormSet::EMPTY.with(Form::CachedAttention));
        assert!(have.missing(FormSet::BASE).is_empty());
    }

    #[test]
    fn form_set_without_removes_only_the_named_form() {
        let trimmed = Executes::CPU.forms.without(Form::Scan);

        assert!(!trimmed.contains(Form::Scan));
        assert!(trimmed.contains(Form::Reduce));
        assert!(Executes::CPU.forms.grants(trimmed));
    }

    #[test]
    fn cpu_admits_every_constructible_kind_and_holds_every_form() {
        for (form, op) in constructible_ops() {
            assert_eq!(form_of(&op.kind), form, "{form:?} fixture maps to its own form");
            assert_eq!(Executes::CPU.admit(&op), Ok(()), "{form:?}");
        }
        for form in [
            Form::ElementwiseTwin,
            Form::ReduceEpilogue,
            Form::ReduceBroadcastEpilogue,
            Form::RoundBatchedReduce,
            Form::BlockQuantize,
            Form::CachedSoftmaxWeights,
            Form::GatedDeltaNet,
            Form::MoeTopK,
            Form::TopFractionSelect,
        ] {
            assert!(Executes::CPU.forms.contains(form), "{form:?}");
        }
    }

    #[test]
    fn base_refuses_cached_attention_with_its_form() {
        assert_eq!(
            Executes::BASE.admit(&cached_attention()),
            Err(Refusal::Form(Form::CachedAttention))
        );
        assert_eq!(Executes::BASE.admit(&bound_elementwise()), Ok(()));
    }

    #[test]
    fn intersect_keeps_only_forms_both_sides_render() {
        let narrowed = Executes::CPU.intersect(Executes::BASE.with(Form::MoeTopK));

        assert!(narrowed.forms.contains(Form::MoeTopK));
        assert!(!narrowed.forms.contains(Form::CachedAttention));
    }

    #[test]
    fn unified_memory_executes_is_the_default() {
        assert_eq!(Executes::CPU.memory(), DeviceMemory::Unified);
        assert_eq!(Executes::BASE.memory(), DeviceMemory::Unified);

        let discrete = Executes::BASE.with(Form::MoeTopK).with_memory(DeviceMemory::Separate);
        assert_eq!(discrete.memory(), DeviceMemory::Separate);
        assert!(discrete.forms.contains(Form::MoeTopK));
        assert_ne!(discrete, Executes::BASE.with(Form::MoeTopK));
        assert_eq!(
            discrete.intersect(Executes::CPU).memory(),
            DeviceMemory::Separate
        );
    }
}
