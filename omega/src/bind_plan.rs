//! The one bind entry every backend, the candidate-identity probe and the kv-padding probe
//! share: the binding policy comes from the config and the grant, fusion is decided once, and
//! the post-bind passes the grant calls for run here. A backend differs only where it drops to
//! kernel source and binary.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use proxima_numeric::NumericPolicy;
use proxima_tensor::bind::bind_with_fusion_and_packed_leaves;
use proxima_tensor::{
    BoundOp, Executes, Form, NodeId, Op, Shapes, TensorError, bind_unfused,
    correct_packed_matmul_layouts, derive_activation_quantize, prune_dead,
};

use crate::lowering::{Codec, PackedOperands};

#[derive(Debug, Clone, Copy)]
pub struct BindConfig<'a> {
    pub leaves: Option<&'a BTreeSet<NodeId>>,
    pub numeric: NumericPolicy,
    pub allow_fusion: bool,
    pub fuse_cached_attention: bool,
    pub packed: &'a PackedOperands,
    pub activation_storage: Option<Codec>,
    pub top_fraction_min_rows: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Bound {
    pub resolved: Vec<BoundOp>,
    pub packed: PackedOperands,
    pub policy: NumericPolicy,
}

impl<'a> BindConfig<'a> {
    /// Fusion follows the reassociation permission unless the caller overrides it.
    #[must_use]
    pub fn new(numeric: NumericPolicy, allow_fusion: bool, packed: &'a PackedOperands) -> Self {
        Self {
            numeric,
            allow_fusion,
            fuse_cached_attention: numeric.reassociation,
            packed,
            leaves: None,
            activation_storage: None,
            top_fraction_min_rows: None,
        }
    }

    #[must_use]
    pub fn with_fused_cached_attention(mut self, fuse: bool) -> Self {
        self.fuse_cached_attention = fuse;
        self
    }

    #[must_use]
    pub fn with_packed_leaves(mut self, leaves: &'a BTreeSet<NodeId>) -> Self {
        self.leaves = Some(leaves);
        self
    }

    /// Collapses top-fraction selections of at least `min_rows` score rows where the grant renders them.
    #[must_use]
    pub fn with_top_fraction(mut self, min_rows: u64) -> Self {
        self.top_fraction_min_rows = Some(min_rows);
        self
    }

    #[must_use]
    pub fn with_activation_storage(mut self, storage: Option<Codec>) -> Self {
        self.activation_storage = storage;
        self
    }
}

/// The policy a grant binds with: the reference executor keeps its accumulations, every device
/// renderer sums in float, and a grant of the integer dot reads flat activation blocks.
#[must_use]
pub fn binding_policy(numeric: NumericPolicy, executes: &Executes) -> NumericPolicy {
    if *executes == Executes::CPU {
        return numeric;
    }
    let narrowed = numeric
        .with_accumulator_narrowing()
        .with_flash_narrowing(true);
    narrowed.with_flat_activation_blocks(
        narrowed.flat_activation_blocks || executes.forms.contains(Form::Int8DotReduce),
    )
}

/// Binds `program` for `executes` and runs the passes that grant asks for.
///
/// # Errors
/// [`TensorError`] when the binder refuses the graph.
pub fn bind_plan(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    config: &BindConfig<'_>,
    executes: &Executes,
) -> Result<Bound, TensorError> {
    let policy = binding_policy(config.numeric, executes);
    let mut leaves: BTreeSet<NodeId> = config.packed.keys().copied().collect();
    leaves.extend(config.leaves.into_iter().flatten().copied());
    let bound = if config.allow_fusion {
        let fuse_attention =
            config.fuse_cached_attention && executes.forms.contains(Form::CachedAttention);
        bind_fused(
            program,
            shapes,
            outputs,
            (fuse_attention, policy),
            &leaves,
            config,
            executes,
        )?
    } else {
        bind_unfused(program, shapes, outputs, policy)?
    };
    Ok(finish_bound(
        prune_dead(bound, outputs),
        program,
        shapes,
        config,
        executes,
    ))
}

fn bind_fused(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    (fuse_attention, policy): (bool, NumericPolicy),
    leaves: &BTreeSet<NodeId>,
    config: &BindConfig<'_>,
    executes: &Executes,
) -> Result<Vec<BoundOp>, TensorError> {
    #[cfg(feature = "top-fraction-fusion")]
    if let Some(min_rows) = config
        .top_fraction_min_rows
        .filter(|_| executes.forms.contains(Form::TopFractionSelect))
    {
        return proxima_tensor::bind::bind_with_top_fraction_and_packed_leaves(
            program,
            shapes,
            outputs,
            fuse_attention,
            policy,
            min_rows,
            leaves,
            executes,
        );
    }
    let _ = config;
    bind_with_fusion_and_packed_leaves(
        program,
        shapes,
        outputs,
        fuse_attention,
        policy,
        leaves,
        executes,
    )
}

/// The post-bind passes alone, for a refit that rebinds nothing.
#[must_use]
pub fn finish_bound(
    mut resolved: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    config: &BindConfig<'_>,
    executes: &Executes,
) -> Bound {
    let policy = binding_policy(config.numeric, executes);
    let mut packed = config.packed.clone();
    if !packed.is_empty() {
        correct_packed_matmul_layouts(&mut resolved, &packed.keys().copied().collect());
    }
    if executes.forms.contains(Form::Int8DotReduce) {
        resolved = derive_through_tensor_pass(
            resolved,
            program,
            shapes,
            &mut packed,
            policy,
            config.activation_storage,
            executes,
        );
    }
    Bound {
        resolved,
        packed,
        policy,
    }
}

/// Runs `derive_activation_quantize` against the grant and folds the held codecs into `packed`.
pub(crate) fn derive_through_tensor_pass(
    resolved: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    packed: &mut PackedOperands,
    policy: NumericPolicy,
    storage: Option<Codec>,
    executes: &Executes,
) -> Vec<BoundOp> {
    packed.retain(|node, _| (node.0 as usize) < program.len());
    let (bound, held) =
        derive_activation_quantize(resolved, program, shapes, packed, policy, executes, storage);
    packed.extend(held);
    bound
}

#[cfg(all(test, feature = "cuda-driver"))]
mod tests {
    use super::*;
    use crate::cuda::{executes, target::Target};
    use proxima_tensor::BoundOpKind;

    #[test]
    fn the_grant_alone_decides_narrowing_and_flat_activation_blocks() {
        let numeric = NumericPolicy::bit_exact();
        assert_eq!(binding_policy(numeric, &Executes::CPU), numeric);
        let cuda = binding_policy(numeric, &executes(&Target::CUDA));
        assert!(cuda.flat_activation_blocks && cuda.accumulator_narrowing && cuda.flash_narrowing);
        let hip = binding_policy(numeric, &executes(&Target::HIP));
        assert!(!hip.flat_activation_blocks && hip.accumulator_narrowing && hip.flash_narrowing);
        assert_eq!(
            binding_policy(numeric, &executes(&Target::CUDA)),
            cuda,
            "binding twice changes nothing"
        );
    }

    #[cfg(feature = "cached-attention-streaming")]
    #[test]
    fn the_kv_padding_probe_agrees_with_the_entry_on_whether_attention_fuses() {
        let (program, root, _) =
            proxima_tensor::spec::gqa_cached_forward_program(32, 16, 24, 4, 2, 4, 2)
                .expect("fixture builds");
        let shapes = proxima_tensor::infer(&program, &[1, 6]).expect("infers");
        let packed = PackedOperands::new();
        let numeric = NumericPolicy::llama_relaxed();
        for target in [Target::CUDA, Target::HIP] {
            let grant = executes(&target);
            let config = BindConfig::new(numeric, true, &packed);
            let bound = bind_plan(&program, &shapes, &[root], &config, &grant).expect("binds");
            assert_eq!(
                has_cached_attention(&bound),
                grant.forms.contains(Form::CachedAttention),
                "{target:?}"
            );
            assert_eq!(bound.policy, binding_policy(numeric, &grant));
            let unfused = BindConfig::new(numeric, false, &packed);
            let bound = bind_plan(&program, &shapes, &[root], &unfused, &grant).expect("binds");
            assert!(
                !has_cached_attention(&bound),
                "{target:?} fused with fusion disallowed"
            );
        }
    }

    #[cfg(feature = "cached-attention-streaming")]
    #[test]
    fn the_config_can_withhold_cached_attention_from_a_grant_that_renders_it() {
        let (program, root, _) =
            proxima_tensor::spec::gqa_cached_forward_program(32, 16, 24, 4, 2, 4, 2)
                .expect("fixture builds");
        let shapes = proxima_tensor::infer(&program, &[1, 6]).expect("infers");
        let packed = PackedOperands::new();
        let grant = executes(&Target::CUDA);
        let config = BindConfig::new(NumericPolicy::llama_relaxed(), true, &packed);
        assert!(has_cached_attention(
            &bind_plan(&program, &shapes, &[root], &config, &grant).expect("binds")
        ));
        let withheld = config.with_fused_cached_attention(false);
        assert!(!has_cached_attention(
            &bind_plan(&program, &shapes, &[root], &withheld, &grant).expect("binds")
        ));
        assert_eq!(withheld.with_top_fraction(8).top_fraction_min_rows, Some(8));
        assert_eq!(withheld.top_fraction_min_rows, None);
    }

    fn has_cached_attention(bound: &Bound) -> bool {
        bound
            .resolved
            .iter()
            .any(|op| matches!(op.kind, BoundOpKind::CachedAttention { .. }))
    }
}

#[cfg(all(test, feature = "wgpu-backend", feature = "cached-attention-streaming"))]
mod wgpu_bind_tests {
    use super::*;
    use crate::wgpu_driver::executes;
    use proxima_tensor::{bind_without_reduce_epilogue_fusion, prune_dead};

    #[test]
    fn the_entry_binds_a_wgpu_grant_to_what_the_retired_binder_produced() {
        let (program, root, _) =
            proxima_tensor::spec::gqa_cached_forward_program(32, 16, 24, 4, 2, 4, 2)
                .expect("fixture builds");
        let shapes = proxima_tensor::infer(&program, &[1, 6]).expect("infers");
        let packed = PackedOperands::new();
        let numeric = NumericPolicy::llama_relaxed();
        let narrowed = numeric
            .with_accumulator_narrowing()
            .with_flash_narrowing(true);
        let retired = prune_dead(
            bind_without_reduce_epilogue_fusion(&program, &shapes, &[root], false, narrowed)
                .expect("binds"),
            &[root],
        );
        let config = BindConfig::new(numeric, true, &packed);
        let entry = bind_plan(&program, &shapes, &[root], &config, &executes()).expect("binds");
        assert!(!retired.is_empty());
        assert_eq!(entry.resolved, retired);
    }
}
