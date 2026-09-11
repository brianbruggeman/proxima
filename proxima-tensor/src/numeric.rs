//! Numerical permission set for bind-time and plan-time rewrites.
//!
//! [`op::is_associative`](crate::op::ScalarOp::is_associative) has exactly
//! one caller today ([`shape::ShapeTable`](crate::shape::ShapeTable)'s
//! accumulator-width check) — nothing consults it before reassociating a
//! fold. [`NumericPolicy`] and [`admit`] close that gap: a rewrite that
//! changes which bits come out (fusing a multiply and add into one FMA,
//! reordering an associative reduction, approximating a transcendental)
//! must be classified with [`NumericRewrite`] and cleared through [`admit`]
//! before it fires, instead of firing unconditionally.

use crate::error::TensorError;

/// Five independent float-bit-changing permissions a bind (or GPU plan) may
/// grant, mirroring LLVM's own fast-math flags (`contract`, `reassoc`,
/// `nnan`, `nsz`, `afn`/`arcp`) rather than a single total order. Default
/// (all `false`) is bit-exact -- every rewrite this crate ships
/// unconditionally today (identity elimination of `x*1`, chain fusion,
/// reduce-epilogue fusion) needs none of these and is admitted regardless.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub struct NumericPolicy {
    /// Permits merging a multiply and an add into one hardware FMA (one
    /// rounding instead of two). LLVM's `contract`.
    pub contraction: bool,
    /// Permits reordering an associative fold: tree-reduce, the simdgroup
    /// context-chunk merge, quantization-scale factoring across a block.
    /// LLVM's `reassoc`.
    pub reassociation: bool,
    /// Permits assuming no operand is NaN, so an op whose *only* deviation
    /// from its algebraic identity is NaN propagation may be eliminated
    /// (`max(x, -inf)`, `min(x, +inf)`). LLVM's `nnan`.
    pub nan_assumptions: bool,
    /// Permits treating `+0.0`/`-0.0` as interchangeable, so `x + 0.0` may
    /// be eliminated. LLVM's `nsz`.
    pub signed_zero: bool,
    /// Permits approximate transcendentals/reciprocals with a bounded
    /// relative error. LLVM's `afn`/`arcp`.
    pub approx_functions: bool,
}

impl NumericPolicy {
    /// Bit-parity with [`crate::cpu::evaluate`] -- no permission granted.
    #[must_use]
    pub const fn bit_exact() -> Self {
        Self {
            contraction: false,
            reassociation: false,
            nan_assumptions: false,
            signed_zero: false,
            approx_functions: false,
        }
    }

    /// Metal `MTLMathMode::Relaxed`'s own documented contract (Apple's
    /// `MTLMathMode` header, quoted at `omega::metal`'s `MathMode` doc):
    /// "allows aggressive, unsafe floating-point optimizations but
    /// preserves infs and nans." Grants contraction and reassociation;
    /// withholds `nan_assumptions`/`signed_zero`/`approx_functions` because
    /// Relaxed's own contract explicitly preserves NaN/inf/zero behavior.
    #[must_use]
    pub const fn llama_relaxed() -> Self {
        Self {
            contraction: true,
            reassociation: true,
            nan_assumptions: false,
            signed_zero: false,
            approx_functions: false,
        }
    }

    /// Metal `MTLMathMode::Fast`'s contract: aggressive optimization with no
    /// NaN/inf/zero preservation. Every permission granted.
    #[must_use]
    pub const fn fast() -> Self {
        Self {
            contraction: true,
            reassociation: true,
            nan_assumptions: true,
            signed_zero: true,
            approx_functions: true,
        }
    }

    /// Whether `self` grants every permission `required` names -- a subset
    /// check, never a total-order comparison.
    #[must_use]
    pub const fn grants(self, required: Self) -> bool {
        (!required.contraction || self.contraction)
            && (!required.reassociation || self.reassociation)
            && (!required.nan_assumptions || self.nan_assumptions)
            && (!required.signed_zero || self.signed_zero)
            && (!required.approx_functions || self.approx_functions)
    }

    /// Fluent per-permission setters -- `#[non_exhaustive]` blocks a
    /// cross-crate struct literal (`NumericPolicy { .. }`) even with
    /// functional-update syntax, so a caller outside this crate composing a
    /// custom permission set (one not covered by [`Self::bit_exact`]/
    /// [`Self::llama_relaxed`]/[`Self::fast`]) needs a builder-shaped path
    /// on the type itself, never a second type: `NumericPolicy::bit_exact()
    /// .with_contraction(true)`.
    #[must_use]
    pub const fn with_contraction(mut self, contraction: bool) -> Self {
        self.contraction = contraction;
        self
    }

    #[must_use]
    pub const fn with_reassociation(mut self, reassociation: bool) -> Self {
        self.reassociation = reassociation;
        self
    }

    #[must_use]
    pub const fn with_nan_assumptions(mut self, nan_assumptions: bool) -> Self {
        self.nan_assumptions = nan_assumptions;
        self
    }

    #[must_use]
    pub const fn with_signed_zero(mut self, signed_zero: bool) -> Self {
        self.signed_zero = signed_zero;
        self
    }

    #[must_use]
    pub const fn with_approx_functions(mut self, approx_functions: bool) -> Self {
        self.approx_functions = approx_functions;
        self
    }
}

/// One rewrite class this crate or a GPU backend may apply, and the exact
/// [`NumericPolicy`] permissions it requires. `#[non_exhaustive]` -- every
/// future bit-changing rewrite adds a variant and a
/// [`NumericRewrite::required_permissions`] arm before it may fire, never a
/// second, parallel check.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericRewrite {
    /// [`mod@crate::bind`]'s identity-elimination fold, restricted to the ONE
    /// sub-case that is bit-exact for every `f32` including NaN and signed
    /// zero: `x * 1.0`. Needs nothing.
    IdentityElimination,
    /// [`mod@crate::bind`]'s identity-elimination fold for `x + 0.0` --
    /// changes bits only on signed zero (`(-0.0) + 0.0` evaluates to
    /// `+0.0`, while eliminating the op returns the survivor `-0.0`). Needs
    /// `signed_zero` alone; does NOT need `nan_assumptions` -- `NaN + 0.0`
    /// stays `NaN` whichever path is taken.
    IdentityEliminationSignedZero,
    /// [`mod@crate::bind`]'s identity-elimination fold for `max(x, -inf)` and
    /// `min(x, +inf)` -- changes bits only on NaN handling (IEEE 754
    /// `maxNum`/[`f32::max`]'s own "if one argument is NaN, return the
    /// other" rule: `max(NaN, -inf)` evaluates to `NaN`, while eliminating
    /// the op returns the survivor `-inf`). Needs `nan_assumptions` alone;
    /// does NOT need `signed_zero` -- no zero literal is involved.
    IdentityEliminationNanAssumption,
    /// [`mod@crate::bind`]'s elementwise/reduce chain fusion. Bit-exact by
    /// construction. Needs nothing.
    ChainFusion,
    /// [`mod@crate::bind`]'s reduce-epilogue fusion. Bit-exact by construction.
    /// Needs nothing.
    ReduceEpilogueFusion,
    /// Merging a multiply and an add into one hardware FMA.
    FmaContraction,
    /// Reordering an associative reduction as a tree instead of a left fold.
    TreeReduce,
    /// The GPU cross-simdgroup context-chunk merge in an attention kernel.
    ContextChunkMerge,
    /// The GPU cross-threadgroup key-split merge in an attention kernel --
    /// the same online-softmax combine as [`Self::ContextChunkMerge`], one
    /// hardware level up: partials cross a kernel-dispatch boundary (read
    /// from a scratch buffer another dispatch wrote) rather than a
    /// `threadgroup_barrier` inside one dispatch.
    ContextSplitMerge,
    /// Factoring a dequantization scale across a block instead of per element.
    DequantScaleFactoring,
    /// An approximate transcendental/reciprocal with bounded relative error.
    FastMathApprox,
}

impl NumericRewrite {
    /// The exact permission set this rewrite needs -- never a scalar
    /// minimum rung.
    #[must_use]
    pub const fn required_permissions(self) -> NumericPolicy {
        match self {
            Self::IdentityElimination | Self::ChainFusion | Self::ReduceEpilogueFusion => {
                NumericPolicy::bit_exact()
            }
            Self::IdentityEliminationSignedZero => NumericPolicy {
                signed_zero: true,
                ..NumericPolicy::bit_exact()
            },
            Self::IdentityEliminationNanAssumption => NumericPolicy {
                nan_assumptions: true,
                ..NumericPolicy::bit_exact()
            },
            Self::FmaContraction => NumericPolicy {
                contraction: true,
                ..NumericPolicy::bit_exact()
            },
            Self::TreeReduce
            | Self::ContextChunkMerge
            | Self::ContextSplitMerge
            | Self::DequantScaleFactoring => NumericPolicy {
                reassociation: true,
                ..NumericPolicy::bit_exact()
            },
            Self::FastMathApprox => NumericPolicy {
                approx_functions: true,
                ..NumericPolicy::bit_exact()
            },
        }
    }
}

/// The one admission check every rewrite site calls before firing — a typed
/// error, never a silent numerics change.
///
/// # Errors
/// [`TensorError::NumericPolicyTooStrict`] when `policy` does not grant
/// every permission `rewrite.required_permissions()` names.
pub fn admit(policy: NumericPolicy, rewrite: NumericRewrite) -> Result<(), TensorError> {
    let required = rewrite.required_permissions();
    if policy.grants(required) {
        Ok(())
    } else {
        Err(TensorError::NumericPolicyTooStrict {
            rewrite,
            required,
            granted: policy,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn presets_grant_exactly_their_documented_permissions() {
        assert_eq!(NumericPolicy::bit_exact(), NumericPolicy::default());
        assert!(NumericPolicy::llama_relaxed().contraction);
        assert!(NumericPolicy::llama_relaxed().reassociation);
        assert!(!NumericPolicy::llama_relaxed().nan_assumptions);
        assert!(!NumericPolicy::llama_relaxed().signed_zero);
        assert!(!NumericPolicy::llama_relaxed().approx_functions);
        assert!(NumericPolicy::fast().nan_assumptions);
        assert!(NumericPolicy::fast().signed_zero);
        assert!(NumericPolicy::fast().approx_functions);
    }

    #[test]
    fn bit_exact_rewrites_admitted_at_the_default_policy() {
        let policy = NumericPolicy::default();
        assert_eq!(policy, NumericPolicy::bit_exact());
        assert!(admit(policy, NumericRewrite::IdentityElimination).is_ok());
        assert!(admit(policy, NumericRewrite::ChainFusion).is_ok());
        assert!(admit(policy, NumericRewrite::ReduceEpilogueFusion).is_ok());
    }

    #[test]
    fn reassociating_rewrite_rejected_under_bit_exact_policy() {
        let error = admit(
            NumericPolicy::bit_exact(),
            NumericRewrite::ContextChunkMerge,
        )
        .expect_err("context-chunk merge reassociates and needs the reassociation permission");
        assert_eq!(
            error,
            TensorError::NumericPolicyTooStrict {
                rewrite: NumericRewrite::ContextChunkMerge,
                required: NumericPolicy {
                    reassociation: true,
                    ..NumericPolicy::bit_exact()
                },
                granted: NumericPolicy::bit_exact(),
            }
        );
    }

    #[test]
    fn reassociating_rewrite_admitted_once_the_policy_opts_up() {
        assert!(
            admit(
                NumericPolicy::llama_relaxed(),
                NumericRewrite::ContextChunkMerge
            )
            .is_ok()
        );
        assert!(admit(NumericPolicy::fast(), NumericRewrite::ContextChunkMerge).is_ok());
    }

    /// [`NumericRewrite::ContextSplitMerge`] -- the cross-THREADGROUP
    /// sibling of `ContextChunkMerge` (cross-simdgroup) -- joins the same
    /// `reassociation` permission arm, so it is rejected under `bit_exact`
    /// with the identical shape.
    #[test]
    fn context_split_merge_rejected_under_bit_exact_policy() {
        let error = admit(
            NumericPolicy::bit_exact(),
            NumericRewrite::ContextSplitMerge,
        )
        .expect_err("a cross-threadgroup online-softmax merge reassociates the fold");
        assert_eq!(
            error,
            TensorError::NumericPolicyTooStrict {
                rewrite: NumericRewrite::ContextSplitMerge,
                required: NumericPolicy {
                    reassociation: true,
                    ..NumericPolicy::bit_exact()
                },
                granted: NumericPolicy::bit_exact(),
            }
        );
    }

    #[test]
    fn context_split_merge_admitted_once_the_policy_opts_up() {
        assert!(
            admit(
                NumericPolicy::llama_relaxed(),
                NumericRewrite::ContextSplitMerge
            )
            .is_ok()
        );
        assert!(admit(NumericPolicy::fast(), NumericRewrite::ContextSplitMerge).is_ok());
    }

    /// The split this design makes: granting `signed_zero` alone eliminates
    /// `x+0`, but must NOT also eliminate `max(x,-inf)` -- proves the two
    /// permissions are actually independent, not just independently named.
    #[test]
    fn signed_zero_alone_does_not_grant_nan_assumption_elimination() {
        let policy = NumericPolicy {
            signed_zero: true,
            ..NumericPolicy::bit_exact()
        };
        let error = admit(policy, NumericRewrite::IdentityEliminationNanAssumption)
            .expect_err("signed_zero does not grant nan_assumptions");
        assert_eq!(
            error,
            TensorError::NumericPolicyTooStrict {
                rewrite: NumericRewrite::IdentityEliminationNanAssumption,
                required: NumericPolicy {
                    nan_assumptions: true,
                    ..NumericPolicy::bit_exact()
                },
                granted: policy,
            }
        );
    }

    /// The converse split: granting `nan_assumptions` alone must NOT also
    /// eliminate `x+0`.
    #[test]
    fn nan_assumption_alone_does_not_grant_signed_zero_elimination() {
        let policy = NumericPolicy {
            nan_assumptions: true,
            ..NumericPolicy::bit_exact()
        };
        assert!(admit(policy, NumericRewrite::IdentityEliminationSignedZero).is_err());
        assert!(admit(policy, NumericRewrite::IdentityEliminationNanAssumption).is_ok());
    }

    #[test]
    fn signed_zero_elimination_admitted_once_the_policy_opts_up() {
        let policy = NumericPolicy {
            signed_zero: true,
            ..NumericPolicy::bit_exact()
        };
        assert!(admit(policy, NumericRewrite::IdentityEliminationSignedZero).is_ok());
    }
}
