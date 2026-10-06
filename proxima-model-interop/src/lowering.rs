//! A checkpoint lowered from config: the family profile a checkpoint's own
//! `general.architecture` string keys ([`crate::profiles::family_profile`])
//! names the header reader that fills its [`ModelDescriptor`]
//! ([`proxima_tensor::spec::ScheduleSource`]), [`build_forward`] lowers that
//! descriptor, and [`crate::bind_leaves::bind_program_leaves`] binds the
//! weights its `Input` leaves name. There is no per-family type and no
//! registry: adding a family whose header reads as one of the existing
//! layouts is a profile file, and a different layout is a new reader behind
//! one more enum arm.
//!
//! Composes [`crate::bind`]'s public bind toolkit (`BoundWeights`,
//! `find_tensor`, `metadata_*`) and `proxima_tensor::spec`'s `build_forward`.
//! Teaching pointer: [`bind_checkpoint`] is the whole load-time pipeline in
//! one call; [`BoundProgram::lowered_from`] is the same pipeline starting from
//! an edited config.

use alloc::vec;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::{
    ForwardProgram, LayerKind, ModelDescriptor, LayerCacheRoots, ScheduleSource, build_forward,
};

use crate::bind::{BoundWeights, ModelHparams, metadata_str};
use crate::bind_leaves::{bind_missing_leaves, bind_program_leaves};
use crate::error::InteropError;
use crate::profiles::{binding_profile, family_profile};
use crate::recurrent_routed_interval::MoeLayerDiagnostics;

/// Names for every `Extent::Symbolic` slot the decode loop itself binds
/// before it evaluates a step -- `crate::generate::LoadedModel`'s own
/// `symbols = [new_count as u64, kv_bound_extent as u64]` array, positions
/// fixed by convention. A [`StepInput`] whose leaf has a symbolic extent claims
/// a slot starting at [`symbols::FIRST_FREE`]; naming
/// [`symbols::NEW_COUNT`]/[`symbols::KV_BOUND`] as reserved is what lets
/// [`bind_symbols`] reject a slot that collides with one of these instead of
/// silently overwriting it.
pub mod symbols {
    /// `next_ids.len()` this step -- the whole prompt on the prefill step,
    /// one token every step after.
    pub const NEW_COUNT: u16 = 0;
    /// `kv_extent`'s bucketed cache capacity for this step
    /// (`crate::generate`'s own `kv_bound_extent`) -- NOT the rows a
    /// per-step leaf carries.
    pub const KV_BOUND: u16 = 1;
    /// The rows a sliding-window layer's ring cache hands the program this
    /// step ([`super::KvLayout::SlidingRing`]): at most the window, and never
    /// more than [`KV_BOUND`]. Bound by the decode loop itself, so it is
    /// reserved from a [`super::StepInput`] slot too.
    pub const SLIDING_KV_BOUND: u16 = proxima_tensor::spec::SLIDING_KV_SYMBOL;
    /// The first slot number free for a [`super::StepInput`] to claim.
    pub const FIRST_FREE: u16 = 3;
}

/// Assembles the `symbols` slice [`proxima_tensor::infer`] and every
/// evaluator resolve a `proxima_tensor::op::Extent::Symbolic` extent
/// against, binding the decode loop's own [`symbols::NEW_COUNT`]/
/// [`symbols::KV_BOUND`] slots plus whatever slot each `step_inputs` entry
/// names via [`StepInput::symbol`]. Composes no pipe: this is plain
/// data assembly ahead of a pipe stage (`BackendRuntime::evaluate`), not a
/// step in the pipe itself.
///
/// # Errors
///
/// [`InteropError::ReservedSymbolSlot`] when a `step_input` names
/// [`symbols::NEW_COUNT`] or [`symbols::KV_BOUND`]; [`InteropError::MultiPositionStepUnsupported`]
/// when `single_position_step` is set and `new_count != 1` -- see
/// [`BoundProgram::single_position_step`]'s own doc for which architectures
/// set it and why.
pub fn bind_symbols(
    new_count: usize,
    kv_bound_extent: usize,
    step_inputs: &[StepInput],
    single_position_step: bool,
) -> Result<Vec<u64>, InteropError> {
    if single_position_step && new_count != 1 {
        return Err(InteropError::MultiPositionStepUnsupported { new_count });
    }
    let mut highest = symbols::FIRST_FREE.saturating_sub(1) as usize;
    for step_input in step_inputs {
        if let Some((slot, _)) = step_input.symbol {
            if slot == symbols::NEW_COUNT
                || slot == symbols::KV_BOUND
                || slot == symbols::SLIDING_KV_BOUND
            {
                return Err(InteropError::ReservedSymbolSlot { slot });
            }
            highest = highest.max(slot as usize);
        }
    }
    let mut bound = vec![0u64; highest + 1];
    bound[symbols::NEW_COUNT as usize] = new_count as u64;
    bound[symbols::KV_BOUND as usize] = kv_bound_extent as u64;
    for step_input in step_inputs {
        if let Some((slot, extent)) = step_input.symbol {
            bound[slot as usize] = extent as u64;
        }
    }
    Ok(bound)
}

/// Every weight tensor bound plus the compiled forward program, in the one
/// shape [`bind_checkpoint`] assembles for every family: `logits_root` is the
/// single terminal node every decode step reads
/// ([`crate::generate::LoadedModel`]'s own `logits_root` field), and
/// `layer_roots` is one entry per forward-program layer in layer order --
/// [`LayerCacheRoots::Attention`] for an attention layer and
/// [`LayerCacheRoots::Ssm`] for a recurrent one, exactly as `layer_roots`'s
/// own field doc in `generate.rs` describes. Every root is the one
/// [`proxima_tensor::spec::ForwardProgram`] names; a root the lowering engine
/// does not produce is the empty value of its type.
pub struct BoundProgram<'file> {
    pub weights: BoundWeights<'file>,
    pub architecture: ModelHparams,
    pub program: Vec<Op>,
    /// Evaluates to exactly ONE row of `vocab` logits -- `[1, vocab]` or a bare
    /// `[vocab]` -- the last new position, never the full `[new_count, vocab]`
    /// buffer: the program gathers on its own `lm_head_row` leaf (see
    /// `generate.rs`'s decode loop, which feeds that leaf `new_count - 1` every
    /// step). A program whose root leaves the ungathered `[new_count, vocab]`
    /// shape is rejected -- at decode time with
    /// [`crate::error::InteropError::LogitsShapeMismatch`], and at load time
    /// wherever the program's own output shape for this node is statically
    /// known -- rather than silently sampled at row 0 (position 0's logits)
    /// instead of the last token.
    pub logits_root: NodeId,
    /// [`proxima_tensor::spec::ForwardProgram::hidden`]: the last-norm
    /// activation `logits_root` projects from, which a pooled embedding reads.
    /// `None` for an engine that exposes no hidden node. See
    /// [`crate::generate::LoadedModel::hidden_root`].
    pub hidden_root: Option<NodeId>,
    /// One post-layer residual root per layer, when the lowering engine
    /// exposes them. This is a correctness seam for comparing CPU/CUDA/wgpu
    /// at the first divergent layer without guessing NodeId arithmetic.
    pub residual_roots: Vec<NodeId>,
    pub layer_roots: Vec<LayerCacheRoots>,
    /// One graph-level diagnostic boundary per routed recurrent layer, in layer
    /// order. Empty for every other lowering.
    pub moe_layer_diagnostics: Vec<MoeLayerDiagnostics>,
    /// Per-layer router-logit roots for a program that can expose a router
    /// prepass. Empty means the program has no routed layers or has not
    /// implemented the pre-gather execution contract.
    pub router_roots: Vec<NodeId>,
    /// One [`proxima_tensor::spec::MoeSite`] per MoE layer the lowering
    /// produced -- empty on a dense checkpoint. `crate::generate`'s decode loop
    /// reads this to know which extra nodes to request as step outputs when a
    /// routing observer (`proxima_tensor::instrument::ExpertObserver`,
    /// `instrument`-gated, hence not a doc link here -- it does not exist under
    /// a non-`instrument` build this crate still documents) is registered; see
    /// that module's own doc for why the loop, not this kernel-building step,
    /// decides whether to evaluate them.
    pub moe_sites: proxima_tensor::spec::MoeSites,
    /// `PROXIMA_HEAD_REPEATS`'s own scratch output
    /// ([`proxima_tensor::spec::ForwardProgram::duplicate_head_roots`]) -- the
    /// extra head-chain roots the lowering appends under
    /// `PROXIMA_HEAD_REPEATS=2|3` (`instrument`-gated, empty otherwise).
    /// Threaded through so the decode loop can request/verify these nodes by
    /// their real [`NodeId`]s instead of reconstructing them from
    /// `program.len()`.
    pub duplicate_head_roots: Vec<NodeId>,
    /// `true` when the forward program is only correct one position per
    /// evaluation: a layer is [`LayerKind::Gdn`], whose gated-DeltaNet mixer
    /// (`proxima_tensor::spec::append_qwen35_ssm_mixer`) `s`-axis reduce sums
    /// across positions rather than stepping through them (see
    /// `proxima_tensor::error::TensorError::SingleTokenStepOnly`'s own doc for
    /// the mechanism, and [`bind_symbols`] for where this flag is enforced).
    /// `false` for a schedule of attention layers, which batches an arbitrary
    /// `new_count` of positions into one evaluation. A chunked causal scan
    /// would let a `true` program batch its own prefill too; until one exists,
    /// `bind_symbols` refuses `new_count > 1` here rather than silently
    /// returning a wrong forward pass.
    pub single_position_step: bool,
}

/// [`crate::recurrent_interval::SsmShape`]'s own fixed sizes -- see that type's
/// doc for what each field measures. Lives here (not `generate.rs`) because
/// [`step_state`] is the seam that hands it out; `generate.rs` re-exports it
/// under its old name for the decode loop's own `SsmLayerCache::new` caller,
/// unchanged.
pub use crate::recurrent_interval::SsmShape;

/// Per-decode-step scratch shape only a schedule with recurrent layers needs
/// to size ahead of the first decode step --
/// [`crate::generate::LoadedModel`]'s own `qwen35_ssm_shape`/
/// `qwen35_attn_head_dim`/`ssm_state_bytes` fields. [`step_state`] returns
/// `None` for a schedule of attention layers, so "not applicable" is stated
/// once, there, and `load_inner` does not special-case it.
#[derive(Debug, Clone, Copy)]
pub struct StepState {
    pub ssm_shape: SsmShape,
    pub attn_head_dim: u32,
    /// `crate::recurrent_interval::qwen35_ssm_state_bytes`'s own resident-bytes
    /// total across every layer -- computed once, from the same header read
    /// that derived `ssm_shape`.
    pub ssm_state_bytes: u64,
}

pub use proxima_tensor::spec::{FfnRouting, KvCacheShape};

/// How a bound program lays out the KV cache of its sliding-window layers.
/// [`Self::Full`] stores every position and lets the window mask hide the
/// evicted ones, the layout [`bind_checkpoint`] binds;
/// [`Self::SlidingRing`] stores only the most recent `window` rows per
/// sliding layer ([`proxima_tensor::spec::SLIDING_KV_SYMBOL`]). The two are
/// numerically identical, because the mask already hides every row the ring
/// drops; the ring only changes how many rows are held.
///
/// [`crate::generate::LoadedModel::load`] asks for [`Self::SlidingRing`].
/// A schedule with no sliding layers ignores the choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvLayout {
    Full,
    #[default]
    SlidingRing,
}

/// One named leaf [`sliding_rope_inputs`] hands the decode loop for this step --
/// the value representation matches the decode loop's own builtin blocks
/// exactly ([`QuantizedBlock::Float32`], the same shape `ids`/`eps`/
/// `rope_cos`/`rope_sin` already bind as), so `run_decode_loop_observed_seeded`
/// pushes this straight into its `named_blocks` with no conversion.
/// `values` is owned (not borrowed) since it is derived fresh from token
/// ids each step, not read out of a resident buffer the way a weight
/// tensor is.
pub struct StepInput {
    pub name: &'static str,
    pub values: Vec<f32>,
    /// This leaf's own `Extent::Symbolic` slot and the extent to bind it
    /// to, when the forward program declares [`Self::name`] with a
    /// symbolic (not `Static`) shape -- `(slot, values.len())` for a
    /// leaf shaped `[Extent::Symbolic(slot)]`. `None` for a leaf whose
    /// shape is entirely `Static` (nothing to bind). Read by
    /// [`bind_symbols`], never by the runtime directly. Slot MUST be
    /// `>= symbols::FIRST_FREE`; [`bind_symbols`] returns
    /// [`InteropError::ReservedSymbolSlot`] otherwise.
    pub symbol: Option<(u16, usize)>,
}

impl StepInput {
    /// Borrows [`Self::values`] as the [`QuantizedBlock::Float32`] shape a
    /// `named_blocks` entry needs -- the one call site
    /// `crate::generate::LoadedModel`'s own decode loop makes per returned
    /// [`StepInput`], pulled out so that call site reads as "push this
    /// leaf" rather than reaching into the enum itself.
    #[must_use]
    pub fn as_named_block(&self) -> (&str, QuantizedBlock<'_>) {
        (self.name, QuantizedBlock::Float32(self.values.as_slice()))
    }
}

/// One header reader's output: the descriptor [`build_forward`] lowers and the
/// hyperparameters the decode loop sizes its caches and tables from.
type Header = (ModelDescriptor, ModelHparams);

/// `general.architecture`, the key every family profile is looked up by.
fn family(parsed: &ParsedGguf) -> Result<&str, InteropError> {
    metadata_str(parsed, "general.architecture")
}

/// The header reader the family profile names, run over `parsed`.
fn header(parsed: &ParsedGguf, layout: KvLayout) -> Result<Header, InteropError> {
    match family_profile(family(parsed)?)?.schedule_source {
        ScheduleSource::Uniform => crate::dense::header(parsed),
        ScheduleSource::SlidingPattern => crate::sliding_pattern::header(parsed, layout),
        ScheduleSource::RecurrentInterval => crate::recurrent_interval::header(parsed),
        ScheduleSource::RecurrentRoutedInterval => crate::recurrent_routed_interval::header(parsed),
    }
}

/// The checkpoint's own descriptor: what the family profile's header reader
/// derives from `parsed`, before any config layer edits it. Serialize it, edit
/// it, and hand it to `LoadedModel::load_with_descriptor`.
///
/// # Errors
///
/// The family has no profile, or its header reader refuses the header.
pub fn header_descriptor(parsed: &ParsedGguf, layout: KvLayout) -> Result<ModelDescriptor, InteropError> {
    header(parsed, layout).map(|(descriptor, _)| descriptor)
}

/// Binds every weight tensor and lowers the forward program in one pass,
/// borrowing from `file_bytes` -- the checkpoint's own mmap, never copied.
/// [`bind_checkpoint_with_kv_layout`] with the full-cache layout, which every
/// caller that drives its own `kv_cache.*` leaves expects.
///
/// # Errors
///
/// Whatever the family's header reader, the lowering or the weight bind can
/// fail with.
pub fn bind_checkpoint<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
) -> Result<BoundProgram<'file>, InteropError> {
    bind_checkpoint_with_kv_layout(parsed, file_bytes, KvLayout::Full)
}

/// [`bind_checkpoint`] with an explicit sliding-layer KV layout
/// ([`KvLayout`]); [`crate::generate::LoadedModel::load`] binds
/// [`KvLayout::SlidingRing`]. A family with no sliding layers ignores it.
///
/// # Errors
///
/// Whatever the family's header reader, the lowering or the weight bind can
/// fail with.
pub fn bind_checkpoint_with_kv_layout<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    layout: KvLayout,
) -> Result<BoundProgram<'file>, InteropError> {
    let (descriptor, architecture) = header(parsed, layout)?;
    bind_descriptor(parsed, file_bytes, architecture, &descriptor)
}

/// Lowers `descriptor` and binds the weights its `Input` leaves name: the one
/// pipeline every family takes after its header reader has run, and the one a
/// verify program takes after [`ModelDescriptor::verify`] has edited the decode
/// descriptor.
fn bind_descriptor<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    architecture: ModelHparams,
    descriptor: &ModelDescriptor,
) -> Result<BoundProgram<'file>, InteropError> {
    let ForwardProgram {
        program,
        logits,
        layer_roots,
        moe_sites,
        layer_residuals,
        hidden,
        duplicate_head_roots,
        layer_diagnostics,
    } = build_forward(descriptor)?;
    let weights = bind_program_leaves(
        parsed,
        file_bytes,
        &program,
        &binding_profile(&architecture.family)?,
        &[],
    )?;
    Ok(BoundProgram {
        weights,
        architecture,
        program,
        logits_root: logits,
        hidden_root: hidden,
        residual_roots: layer_residuals,
        layer_roots,
        router_roots: layer_diagnostics
            .iter()
            .map(|diagnostic| diagnostic.router_logits)
            .collect(),
        moe_layer_diagnostics: layer_diagnostics,
        moe_sites,
        duplicate_head_roots,
        single_position_step: descriptor.layers.iter().any(|layer| layer.kind == LayerKind::Gdn),
    })
}

/// The all-positions counterpart of [`bind_checkpoint_with_kv_layout`] for
/// speculative decode's verify step, lowered from the same descriptor with
/// every new position's logits row kept ([`ModelDescriptor::verify`]): its
/// `logits_root` gathers one row per drafted token. `config` is the descriptor
/// in force, or the header's own when `None`. `None` out when the descriptor
/// does not arm verify (`speculative_verify` is off, or a layer cannot rewind),
/// so the verify step stays off without a name comparison at the load site.
///
/// # Errors
///
/// The header reader refuses the header, or the verify descriptor does not
/// lower or bind.
pub fn bind_speculative_verify<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    layout: KvLayout,
    config: Option<&ModelDescriptor>,
) -> Result<Option<BoundProgram<'file>>, InteropError> {
    let (header_descriptor, architecture) = header(parsed, layout)?;
    config
        .unwrap_or(&header_descriptor)
        .verify()
        .map(|verify| bind_descriptor(parsed, file_bytes, architecture, &verify))
        .transpose()
}

/// The per-decode-step scratch shape a recurrent layer needs sized ahead of
/// the first step, re-derived off `parsed`'s own header (the same source the
/// header reader reads) rather than threaded through [`BoundProgram`]. `None`
/// for a family whose layers are all attention: nothing to size.
///
/// # Errors
///
/// Whatever the family's header reader can fail with.
pub fn step_state(parsed: &ParsedGguf) -> Result<Option<StepState>, InteropError> {
    match family_profile(family(parsed)?)?.schedule_source {
        ScheduleSource::RecurrentInterval => crate::recurrent_interval::step_state(parsed).map(Some),
        _ => Ok(None),
    }
}

/// The KV layout `crate::memory_fit::MemoryBudget::derive` prices: one
/// `(kv_heads, head_dim, window)` entry per layer that owns a KV cache,
/// `window` being `Some(rows)` for a sliding-window layer.
///
/// # Errors
///
/// Whatever the family's metadata reads can fail with.
pub fn kv_layers(parsed: &ParsedGguf) -> Result<Vec<(u32, u32, Option<u32>)>, InteropError> {
    match family_profile(family(parsed)?)?.schedule_source {
        ScheduleSource::SlidingPattern => crate::sliding_pattern::hparams::kv_layers_from_metadata(parsed),
        ScheduleSource::RecurrentRoutedInterval => crate::recurrent_routed_interval::hparams::kv_layers_from_metadata(parsed),
        ScheduleSource::Uniform | ScheduleSource::RecurrentInterval => {
            crate::bind::kv_layers_from_metadata(parsed)
        }
    }
}

/// `{general.architecture}.context_length` -- the context the checkpoint was
/// trained at. `None` when the key is absent or is not a `u32`.
#[must_use]
pub fn trained_context_length(parsed: &ParsedGguf) -> Option<u32> {
    let family = family(parsed).ok()?;
    crate::bind::metadata_u32(parsed, &alloc::format!("{family}.context_length")).ok()
}

/// The checkpoint's own per-pair RoPE frequency-scaling factor (GGUF
/// `ROPE_FREQS`, `rope_freqs.weight`) for the builtin `rope_cos`/`rope_sin`
/// table's `head_dim / 2` frequency pairs, when the binder bound one into
/// `weights`: `crate::generate::build_position_inputs` divides each pair's
/// angle by `factors[pair]` before taking `cos`/`sin`. `None` rotates every
/// pair undivided. The tensor's presence in the bound weights is the whole
/// condition: the binding profile names it for the family that carries it
/// (`1e30` on a pair collapses its angle to `cos=1, sin=0`, HF's `inv_freq=0`
/// convention for a partial-rotary model, with no separate identity path).
#[must_use]
pub fn rope_freq_factors<'weights>(weights: &'weights BoundWeights<'_>) -> Option<&'weights [f32]> {
    weights
        .owned
        .iter()
        .find(|(name, _)| name == "rope_freqs.weight")
        .map(|(_, values)| values.as_slice())
}

/// Feeds the sliding-window RoPE table the `rope_cos_swa`/`rope_sin_swa`
/// leaves declare (`LayerAttentionConfig::rope_table`): the decode loop's
/// builtin `rope_cos`/`rope_sin` blocks carry the full-layer table, so a model
/// whose [`ModelHparams::sliding_rope`] is set needs these two extra
/// leaves, one row per position `new_start..new_start + new_count`. Pushes
/// nothing when the header carries no sliding table.
pub fn sliding_rope_inputs(
    architecture: &ModelHparams,
    new_start: usize,
    new_count: usize,
    out: &mut Vec<StepInput>,
) {
    let Some(rope) = architecture.sliding_rope else {
        return;
    };
    let positions: Vec<usize> = (new_start..new_start + new_count).collect();
    let (cos, sin) = crate::sliding_pattern::program::gemma4_sliding_rope_table(&positions, rope.freq_base, rope.dimension_count);
    out.push(StepInput { name: "rope_cos_swa", values: cos, symbol: None });
    out.push(StepInput { name: "rope_sin_swa", values: sin, symbol: None });
}

impl<'file> BoundProgram<'file> {
    /// This bound program with its forward program lowered from `descriptor`
    /// instead of the descriptor the binder derived: the op graph and every
    /// root into it come from [`proxima_tensor::spec::build_forward`] over the
    /// config, and the weights gain whatever leaves that program names which the
    /// binder's own program did not ([`crate::bind_leaves::bind_missing_leaves`]). A root the
    /// binder chose not to expose (`hidden_root`, `residual_roots`) stays
    /// unexposed, so a decode loop reads the same outputs it always did.
    ///
    /// Teaching pointer: this is the config seam. Serialize a descriptor,
    /// edit it, and hand it back here (or through
    /// `LoadedModel::load_with_descriptor`) and the lowering follows the
    /// config with no per-family Rust.
    ///
    /// # Errors
    ///
    /// [`proxima_tensor::spec::build_forward`] refuses the config, or the
    /// engine returns a cache-root count that disagrees with the schedule.
    pub fn lowered_from(
        mut self,
        descriptor: &ModelDescriptor,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<Self, InteropError> {
        let ForwardProgram {
            program,
            logits,
            layer_roots,
            moe_sites,
            layer_residuals,
            hidden,
            duplicate_head_roots,
            layer_diagnostics,
        } = build_forward(descriptor)?;
        bind_missing_leaves(
            parsed,
            file_bytes,
            &program,
            &binding_profile(&self.architecture.family)?,
            &mut self.weights,
        )?;
        Ok(Self {
            architecture: self.architecture.reshaped_by(descriptor),
            program,
            logits_root: logits,
            hidden_root: self.hidden_root.and(hidden),
            residual_roots: if self.residual_roots.is_empty() { Vec::new() } else { layer_residuals },
            layer_roots,
            moe_sites,
            duplicate_head_roots,
            router_roots: if self.router_roots.is_empty() {
                Vec::new()
            } else {
                layer_diagnostics.iter().map(|diagnostic| diagnostic.router_logits).collect()
            },
            moe_layer_diagnostics: layer_diagnostics,
            ..self
        })
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;

    use proxima_gguf::value::MetadataValue as Value;

    use super::*;
    use crate::bind::SlidingRope;

    fn gemma4_shaped_hparams(sliding_rope: Option<SlidingRope>) -> ModelHparams {
        ModelHparams {
            vocab: 262_144,
            embedding: 1536,
            feed_forward: 6144,
            query_heads: 8,
            kv_heads: 1,
            kv_heads_by_layer: vec![1; 35],
            head_dim: 512,
            block_count: 35,
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: 1_000_000.0,
            rms_epsilon: 1e-6,
            tied_embeddings: true,
            family: "gemma4".to_string(),
            sliding_rope,
        }
    }

    #[test]
    fn bind_symbols_rejects_new_count_above_one_when_single_position_step_is_set() {
        match bind_symbols(2, 8, &[], true) {
            Err(InteropError::MultiPositionStepUnsupported { new_count }) => {
                assert_eq!(new_count, 2);
            }
            other => panic!("expected MultiPositionStepUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn bind_symbols_allows_new_count_one_when_single_position_step_is_set() {
        let bound = bind_symbols(1, 8, &[], true).expect("new_count == 1 is always allowed");
        assert_eq!(bound[symbols::NEW_COUNT as usize], 1);
        assert_eq!(bound[symbols::KV_BOUND as usize], 8);
    }

    #[test]
    fn bind_symbols_allows_new_count_above_one_when_single_position_step_is_unset() {
        let bound = bind_symbols(4, 8, &[], false)
            .expect("batched prefill is unaffected by the flag when it is false");
        assert_eq!(bound[symbols::NEW_COUNT as usize], 4);
    }

    #[test]
    fn a_family_with_no_profile_is_refused_by_name_before_anything_binds() {
        let parsed = crate::test_support::parsed_header(vec![(
            "general.architecture",
            Value::String("phi9".to_string()),
        )]);

        match bind_checkpoint(&parsed, &[]) {
            Err(InteropError::MissingFamilyProfile { family }) => assert_eq!(family, "phi9"),
            Ok(_) => panic!("a family with no profile must not bind"),
            Err(other) => panic!("expected MissingFamilyProfile, got {other}"),
        }
    }

    #[test]
    fn a_sliding_rope_adds_a_cos_and_sin_row_per_new_position() {
        let hparams = gemma4_shaped_hparams(Some(SlidingRope { freq_base: 10_000.0, dimension_count: 256 }));
        let mut inputs = Vec::new();

        sliding_rope_inputs(&hparams, 40, 3, &mut inputs);

        let names: Vec<&str> = inputs.iter().map(|input| input.name).collect();
        assert_eq!(names, ["rope_cos_swa", "rope_sin_swa"]);
        assert!(inputs.iter().all(|input| input.values.len() == 3 * 128));
        assert!(inputs.iter().all(|input| input.symbol.is_none()));
    }

    #[test]
    fn no_sliding_rope_adds_no_step_inputs() {
        let hparams = gemma4_shaped_hparams(None);
        let mut inputs = Vec::new();

        sliding_rope_inputs(&hparams, 0, 1, &mut inputs);

        assert!(inputs.is_empty());
    }

    #[test]
    fn rope_freq_factors_read_the_bound_tensor_by_name_and_none_when_absent() {
        let mut weights = BoundWeights::new(&[]);
        assert!(rope_freq_factors(&weights).is_none());

        weights.owned.push(("rope_freqs.weight".to_string(), vec![1.0, 1.0, 1.0e30]));

        assert_eq!(rope_freq_factors(&weights), Some([1.0, 1.0, 1.0e30].as_slice()));
    }

    /// The measured `context_length` of each real checkpoint (`ollama
    /// /api/show`, 2026-09-29), each written into a GGUF header by the real
    /// encoder and read back.
    #[proxima::test]
    #[case::gemma4_e2b_reads_131072("gemma4", 131_072)]
    #[case::qwen35moe_a3b_reads_262144("qwen35moe", 262_144)]
    #[case::qwen3_8b_dense_reads_40960("qwen3", 40_960)]
    async fn trained_context_read(#[case] family: &'static str, #[case] expected: u32) {
        let context_key = alloc::format!("{family}.context_length");
        let parsed = crate::test_support::parsed_header(vec![
            ("general.architecture", Value::String(family.to_string())),
            (context_key.as_str(), Value::U32(expected)),
        ]);

        assert_eq!(trained_context_length(&parsed), Some(expected));
    }

    #[test]
    fn trained_context_absent_key_reads_none() {
        let parsed = crate::test_support::parsed_header(vec![(
            "general.architecture",
            Value::String("qwen3".to_string()),
        )]);

        assert_eq!(
            trained_context_length(&parsed),
            None,
            "a header without {{arch}}.context_length must read None, not a default"
        );
    }
}
