//! A tensor program: a flat, self-contained sequence of expressions.
//!
//! There is no arena, no builder, no interning — a program is a plain
//! `Vec<Op>`, and every `Op` owns its own data. That is deliberate: a
//! partition pass that ships part of a program across a wire is then just a
//! sub-slice plus renumbering, not a walk of a separate side-table. Two rules
//! make a slice safe to consume without a validation pass of its own:
//!
//! - **References point backwards only.** A [`NodeId`] is a position in the
//!   slice; an `Op` may only name positions built before it. That makes
//!   acyclicity an O(1) comparison per reference instead of a traversal.
//! - **The last element is the root**, by the same rule — nothing later can
//!   exist to consume it.
//!
//! Together these mean a linear scan of the slice is always a valid
//! topological order, which is what [`shape::infer`](crate::shape::infer) and
//! [`bind::bind`](crate::bind::bind) both rely on.

use alloc::string::String;
use alloc::vec::Vec;

use crate::dtype::DType;
use crate::map::IndexMap;

/// Index of an [`Op`] in the program slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(pub u32);

impl core::fmt::Display for NodeId {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "%{}", self.0)
    }
}

/// One dimension of a leaf's shape.
///
/// Symbolic extents are not an edge case: sequence length is unknown until a
/// request arrives, so a program that cannot express one cannot describe a
/// model that serves traffic. This is why rank and shape are values here and
/// not type parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum Extent {
    Static(u32),
    Symbolic(u16),
}

/// Scalar body of an [`Op::Elementwise`] or a [`Reduce`].
///
/// Closed on purpose, and it is the one closed set in this crate that stays
/// closed: these are scalar machine primitives, not an extension point.
/// Composite activations desugar into several expressions — `gelu` is a
/// handful of elementwise expressions, not a variant here — which costs
/// expressions and buys a vocabulary that never grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum ScalarOp {
    Identity,
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    Negate,
    Reciprocal,
    Exponential,
    Logarithm,
    SquareRoot,
    Tanh,
    Erf,
    Greater,
    Equal,
    Select,
}

impl ScalarOp {
    /// Operand count the body consumes. An elementwise expression whose
    /// operand count disagrees with this is malformed, and
    /// [`shape::infer`](crate::shape::infer) says so.
    #[must_use]
    pub const fn arity(self) -> usize {
        match self {
            Self::Identity
            | Self::Negate
            | Self::Reciprocal
            | Self::Exponential
            | Self::Logarithm
            | Self::SquareRoot
            | Self::Tanh
            | Self::Erf => 1,
            Self::Add
            | Self::Subtract
            | Self::Multiply
            | Self::Divide
            | Self::Maximum
            | Self::Minimum
            | Self::Greater
            | Self::Equal => 2,
            Self::Select => 3,
        }
    }

    /// Whether reducing with this body is order-independent. Only
    /// associative bodies may be reassociated into a tree or a parallel
    /// scan, so a scheduler reads this before choosing a reduction
    /// strategy.
    #[must_use]
    pub const fn is_associative(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Multiply | Self::Maximum | Self::Minimum
        )
    }
}

/// Seed value for a reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum ReduceInit {
    Zero,
    One,
    NegativeInfinity,
    PositiveInfinity,
    /// Seed from the first element — the form `argmax` needs, since no
    /// synthetic identity exists for a `(value, index)` accumulator.
    FirstElement,
}

/// Which prefixes of a reduce survive.
///
/// The only thing separating a reduction from a scan. Log-depth prefix sum is
/// a *scheduling* decision about a `Keep::Scan` reduce, not a different
/// operation, which is why there is no separate `Scan` expression form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum Keep {
    /// Reduce: only the final accumulator.
    Reduce,
    /// Scan: every prefix, preserving the reduced axis.
    Scan,
}

/// A reduce's parameters, named because eight of them travel together
/// through every pass and every backend.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct Reduce {
    pub dtype: DType,
    pub body: ScalarOp,
    pub init: ReduceInit,
    pub operand: NodeId,
    /// Addresses the operand from the iteration space.
    pub in_map: IndexMap,
    /// Addresses the result. Data-dependent here is what makes a scatter.
    pub out_map: IndexMap,
    pub keep: Keep,
    pub name: Option<String>,
}

/// The three generators.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub enum Op {
    /// A leaf. Where data enters.
    ///
    /// `name` is identity, not decoration: positional binding does not
    /// survive a partition pass that renumbers the slice, and a cut edge in a
    /// distributed program delivers a tensor to an `Input` over a wire keyed
    /// by name, not by index. It is also how weights load by name. Local,
    /// single-partition evaluation still binds `blocks` positionally in
    /// [`cpu::evaluate`](crate::cpu::evaluate) — the name is what survives a
    /// partition that positional binding does not.
    Input {
        dtype: DType,
        shape: Vec<Extent>,
        name: Option<String>,
    },

    /// N-ary elementwise. Each operand carries its own index pattern, so
    /// arity 1 with a permuting pattern is a transpose and an operand with a
    /// computed pattern is a gather.
    Elementwise {
        dtype: DType,
        body: ScalarOp,
        operands: Vec<(NodeId, IndexMap)>,
        name: Option<String>,
    },

    /// Reduce, scan, scatter, contraction and argmax, distinguished by
    /// [`Reduce::keep`] and by whether [`Reduce::out_map`] is
    /// data-dependent.
    Reduce(Reduce),

    /// A leaf like [`Op::Input`], but computed instead of externally
    /// supplied: `output[i] = i` for `i` in `0..extent`. Named `Iota` after
    /// XLA HLO's op of the same name and semantics, an established tensor-IR
    /// precedent for exactly this generator.
    ///
    /// This is what lets a caller build a causal mask: `Greater` compares
    /// two broadcast `Iota`s (a query-position one and a key-position one)
    /// into a 0/1 tensor, and `Select` routes that comparison into
    /// [`crate::op::ReduceInit::NegativeInfinity`]'s elementwise counterpart
    /// before a softmax — the composition `specs/causal_attention.toml`
    /// spells and `spec.rs`'s test evaluates. Before this variant, no
    /// composition of `Input`/`Elementwise`/`Reduce` could produce a tensor
    /// whose values are iteration indices rather than either external data
    /// or a fold over one: `Input` is the only leaf (this module's own
    /// enum), and [`crate::map::IndexMap`] only ever *consumes* an index to
    /// address an operand — nothing upstream of that turns an index into a
    /// value a body can compute over.
    ///
    /// `extent` (not `axis`) is this field's name: an axis *position* is
    /// already a `u16` elsewhere in this crate ([`crate::map::AxisTerm::axis`]),
    /// and this field carries the axis's *size*, the same quantity
    /// [`Op::Input`]'s `shape: Vec<Extent>` carries per dimension — collapsed
    /// to one [`Extent`] because an `Iota` is definitionally one axis; a
    /// multi-axis index tensor is two `Iota`s combined through the existing
    /// elementwise algebra, not a second field here.
    Iota { dtype: DType, extent: Extent },
}

impl Op {
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Input { dtype, .. } | Self::Elementwise { dtype, .. } | Self::Iota { dtype, .. } => {
                *dtype
            }
            Self::Reduce(reduce) => reduce.dtype,
        }
    }

    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Input { name, .. } | Self::Elementwise { name, .. } => name.as_deref(),
            Self::Reduce(reduce) => reduce.name.as_deref(),
            Self::Iota { .. } => None,
        }
    }
}

/// Append an expression, returning the [`NodeId`] it can be referenced by.
/// The id-is-index idiom in one place: every push returns an id greater than
/// any the program already contains, which is what keeps references
/// backwards-only by construction rather than by a check.
pub fn append(program: &mut Vec<Op>, expr: Op) -> NodeId {
    let id = NodeId(program.len() as u32);
    program.push(expr);
    id
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::unary(ScalarOp::Exponential, 1)]
    #[case::negate(ScalarOp::Negate, 1)]
    #[case::binary(ScalarOp::Add, 2)]
    #[case::compare(ScalarOp::Greater, 2)]
    #[case::ternary(ScalarOp::Select, 3)]
    fn arity_matches_the_body(#[case] body: ScalarOp, #[case] expected: usize) {
        assert_eq!(body.arity(), expected);
    }

    #[rstest]
    #[case::add(ScalarOp::Add, true)]
    #[case::multiply(ScalarOp::Multiply, true)]
    #[case::maximum(ScalarOp::Maximum, true)]
    #[case::subtract(ScalarOp::Subtract, false)]
    #[case::divide(ScalarOp::Divide, false)]
    fn only_associative_bodies_may_be_reassociated(
        #[case] body: ScalarOp,
        #[case] associative: bool,
    ) {
        assert_eq!(body.is_associative(), associative);
    }

    #[test]
    fn node_id_displays_in_ssa_form() {
        extern crate alloc;
        use alloc::string::ToString;
        assert_eq!(NodeId(12).to_string(), "%12");
    }

    #[test]
    fn append_returns_increasing_ids() {
        let mut program = Vec::new();
        let first = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let second = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        assert_eq!(first.0, 0);
        assert_eq!(second.0, 1);
        assert!(
            second.0 > first.0,
            "ids increase, so references point backwards"
        );
    }

    #[test]
    fn dtype_and_name_read_through_every_variant() {
        let leaf = Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: Some("x".into()),
        };
        assert_eq!(leaf.dtype(), DType::Float32);
        assert_eq!(leaf.name(), Some("x"));

        let reduce = Op::Reduce(Reduce {
            dtype: DType::Int32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: NodeId(0),
            in_map: IndexMap::Affine(crate::map::projection(1, &[0])),
            out_map: IndexMap::Affine(crate::map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        });
        assert_eq!(reduce.dtype(), DType::Int32);
        assert_eq!(reduce.name(), None);

        let iota = Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(4),
        };
        assert_eq!(iota.dtype(), DType::Float32);
        assert_eq!(
            iota.name(),
            None,
            "Iota carries no name field, unlike Input"
        );
    }
}
