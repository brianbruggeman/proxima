//! The checkpoint-family dispatch [`crate::generate::LoadedModel::load`]
//! used to hard-code as an `if general.architecture == "qwen35" { .. } else
//! { .. }` branch (`generate.rs`'s own `load_inner`, before this module
//! existed) -- closed to any checkpoint family a foreign crate might want
//! to add, since adding one meant editing this crate. [`Architecture`] is
//! that `if`/`else` arm's shape pulled out to a trait, [`ArchitectureRegistry`]
//! the table [`crate::generate::LoadedModel::load_with_registry`] resolves
//! against instead of a hard-coded string compare. `dense.rs`'s
//! `DenseArch` and `qwen35.rs`'s `Qwen35Arch` are the two arms this crate
//! ships as registered [`Architecture`] values -- read either as the
//! worked example a foreign architecture (registered the same way, from a
//! crate that has never sent proxima a PR) follows.
//!
//! Composes [`crate::bind`]'s now-public bind toolkit (`BoundWeights`,
//! `bind_dense`, `bind_matmul_weight`, `find_tensor`, `metadata_*`,
//! `vocab_from_token_embedding` -- the same functions `qwen35.rs` has
//! always used internally) and `proxima_tensor::spec`'s `pub fn
//! *_forward_program` builders, which were already public before this
//! module existed (`proxima-tensor/src/spec.rs`'s own module doc: program
//! assembly is "a plain `Vec<Op>`", no arena to gate).

use alloc::vec;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::Qwen35LayerRoots;

use crate::bind::{BoundWeights, ModelArchitecture, metadata_str};
use crate::error::InteropError;

/// Names for every `Extent::Symbolic` slot the decode loop itself binds
/// before it evaluates a step -- `crate::generate::LoadedModel`'s own
/// `symbols = [new_count as u64, kv_bound_extent as u64]` array, positions
/// fixed by convention and, until now, nowhere spelled out by name. A
/// foreign architecture's [`Architecture::step_inputs`] declares its
/// own leaf's symbolic extent starting at [`symbols::FIRST_FREE`]; naming
/// [`symbols::NEW_COUNT`]/[`symbols::KV_BOUND`] as reserved is what lets
/// [`bind_symbols`] reject a foreign slot that collides with one of these
/// instead of silently overwriting it.
pub mod symbols {
    /// `next_ids.len()` this step -- the whole prompt on the prefill step,
    /// one token every step after.
    pub const NEW_COUNT: u16 = 0;
    /// `kv_extent`'s bucketed cache capacity for this step
    /// (`crate::generate`'s own `kv_bound_extent`) -- NOT the rows a
    /// foreign architecture's own per-step leaf carries; see the defect
    /// this module's doc links back to.
    pub const KV_BOUND: u16 = 1;
    /// The first slot number free for a foreign
    /// [`super::Architecture::step_inputs`] override to claim.
    pub const FIRST_FREE: u16 = 2;
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
/// [`symbols::NEW_COUNT`] or [`symbols::KV_BOUND`].
pub fn bind_symbols(
    new_count: usize,
    kv_bound_extent: usize,
    step_inputs: &[StepInput],
) -> Result<Vec<u64>, InteropError> {
    let mut highest = symbols::FIRST_FREE.saturating_sub(1) as usize;
    for step_input in step_inputs {
        if let Some((slot, _)) = step_input.symbol {
            if slot == symbols::NEW_COUNT || slot == symbols::KV_BOUND {
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
/// shape `crate::generate::LoadedModel::load_inner`'s qwen35 and dense
/// arms each assembled by hand before this trait existed
/// (`generate.rs:1122-1276` on the pre-seam code): `logits_root` is the
/// single terminal node every decode step reads
/// ([`crate::generate::LoadedModel`]'s own `logits_root` field), and
/// `layer_roots` is one entry per forward-program layer in layer order --
/// [`Qwen35LayerRoots::Attention`] on the dense path (wrapping
/// `mistral_cached_forward_program_with_experts`'s plain
/// `CachedLayerRoots`) and a mix of `Attention`/[`Qwen35LayerRoots::Ssm`]
/// on the qwen35 path, exactly as `layer_roots`'s own field doc in
/// `generate.rs` already describes.
pub struct BoundProgram<'file> {
    pub weights: BoundWeights<'file>,
    pub architecture: ModelArchitecture,
    pub program: Vec<Op>,
    pub logits_root: NodeId,
    /// `proxima_tensor::spec::ForwardRoots::hidden` off this architecture's
    /// own forward-program builder, when it names one --
    /// [`crate::dense::DenseArch`] (`mistral_cached_forward_program_with_experts`
    /// returns a [`proxima_tensor::spec::ForwardRoots`]) sets this;
    /// [`crate::qwen35::Qwen35Arch`] (`qwen35_forward_program` returns a
    /// bare `logits` root with no named hidden-state counterpart yet)
    /// leaves it `None`. See [`crate::generate::LoadedModel::hidden_root`].
    pub hidden_root: Option<NodeId>,
    pub layer_roots: Vec<Qwen35LayerRoots>,
    /// One [`proxima_tensor::spec::MoeSite`] per MoE layer this
    /// architecture's forward-program builder produced -- empty on a dense
    /// checkpoint. `crate::generate`'s decode loop reads this to know which
    /// extra nodes to request as step outputs when a routing observer
    /// (`proxima_tensor::instrument::ExpertObserver`, `instrument`-gated,
    /// hence not a doc link here -- it does not exist under a
    /// non-`instrument` build this crate still documents) is registered;
    /// see that module's own doc for why the loop, not this kernel-building
    /// step, decides whether to evaluate them.
    pub moe_sites: proxima_tensor::spec::MoeSites,
}

/// [`crate::qwen35::Qwen35SsmShape`]'s own fixed sizes -- see that type's
/// doc for what each field measures. Lives here (not `generate.rs`) because
/// [`Architecture::step_state`] is the seam that hands it out; `generate.rs`
/// re-exports it under its old name for the decode loop's own
/// `SsmLayerCache::new` caller, unchanged.
pub use crate::qwen35::Qwen35SsmShape;

/// Per-decode-step scratch shape only a hybrid attention+state-space
/// architecture needs to size ahead of the first decode step --
/// [`crate::generate::LoadedModel`]'s own `qwen35_ssm_shape`/
/// `qwen35_attn_head_dim`/`ssm_state_bytes` fields, today set by hand
/// inside `load_inner`'s qwen35 `if` arm and left at their "not
/// applicable" value (`None`/`0`) on every other arm. Promoted to a trait
/// hook (default `None`) so a uniform per-layer-attention architecture
/// (dense, or a foreign one) states "not applicable" once, in its own
/// [`Architecture`] impl, rather than `load_inner` special-casing it.
#[derive(Debug, Clone, Copy)]
pub struct StepState {
    pub ssm_shape: Qwen35SsmShape,
    pub attn_head_dim: u32,
    /// `crate::qwen35::qwen35_ssm_state_bytes`'s own resident-bytes
    /// total across every layer -- computed here, once, by the same
    /// architecture that derived `ssm_shape`, rather than recomputed by
    /// `load_inner` for a shape it did not derive.
    pub ssm_state_bytes: u64,
}

/// One checkpoint family's bind + forward-program pipeline, registered
/// against [`crate::generate::LoadedModel::load`]'s dispatch instead of
/// hard-coded into it. See `qwen35.rs`'s `Qwen35Arch` for the worked
/// example this trait was extracted from -- every method here is a step
/// that file already took, promoted from a private `if` arm to a trait so
/// a foreign crate can add an architecture without a proxima PR (P2
/// teaching surface: read `qwen35.rs` first, this trait second).
pub trait Architecture: Send + Sync {
    /// Value read from `general.architecture`; matched by
    /// [`ArchitectureRegistry::resolve`] against every registered
    /// architecture's own name before falling back to the registry's
    /// default (see that method's own doc for why a fallback exists at
    /// all: most checkpoints' `general.architecture` names their own
    /// family -- `llama`, `mistral`, `qwen3`, `mixtral` -- not "dense",
    /// and `DenseArch`'s whole job is to bind whichever of those a
    /// foreign-architecture-unaware caller hands it, the way `load_inner`'s
    /// `else` arm always has).
    fn name(&self) -> &'static str;

    /// Binds every weight tensor and assembles the forward program in one
    /// pass, borrowing from `file_bytes` -- the checkpoint's own mmap,
    /// never copied (mirrors `qwen35::bind_qwen35_weights` +
    /// `qwen35::qwen35_forward_program`'s existing two-call shape, fused
    /// here so the trait has one entry point, not two callers that must
    /// sequence them in the right order).
    ///
    /// # Errors
    ///
    /// Whatever this architecture's own metadata derivation, weight bind,
    /// or forward-program builder can fail with.
    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError>;

    /// This architecture's per-decode-step scratch sizing, re-derived
    /// straight off `parsed`'s own metadata (the same source
    /// [`Architecture::bind`] itself reads) rather than threaded through
    /// [`BoundProgram`] -- metadata reads are cheap and pure, so a second
    /// read costs nothing next to the weight bind [`Architecture::bind`]
    /// already paid, and it means [`BoundProgram`] never needs an
    /// architecture-specific scratch field. Default `Ok(None)`: every
    /// architecture whose forward program is uniform per-layer attention
    /// with no interleaved state-space mixer has nothing to size here,
    /// which is every architecture this crate ships except qwen35.
    ///
    /// # Errors
    ///
    /// Whatever this architecture's own metadata derivation can fail with.
    fn step_state(&self, parsed: &ParsedGguf) -> Result<Option<StepState>, InteropError> {
        let _ = parsed;
        Ok(None)
    }

    /// One extra token-derived [`Op::Input`] leaf per name, appended to
    /// this step's named blocks alongside the builtin `ids`/`eps`/
    /// `rope_cos`/`rope_sin`/`cached_len`/`kv_cache.*` set
    /// (`crate::generate::LoadedModel::run_decode_loop_observed_seeded`'s
    /// own `named_blocks` build) -- the seam an n-gram hash table, a
    /// retrieval index, or any other architecture whose forward program
    /// declares a leaf this crate does not know about reaches to feed it,
    /// without `run_decode_loop_observed_seeded` special-casing that
    /// architecture by name. `out` is reused across steps (the caller
    /// `clear()`s it before each call), so pushing is the only allocation
    /// this default costs a caller that never overrides it: nothing.
    /// Default pushes nothing -- every architecture whose forward program
    /// declares no leaf beyond the builtin set (every architecture this
    /// crate ships) needs no override.
    fn step_inputs(&self, context: &StepInputContext<'_>, out: &mut Vec<StepInput>) {
        let _ = context;
        let _ = out;
    }
}

/// What [`Architecture::step_inputs`] reads to derive its own per-step
/// leaves -- the token history the decode loop already holds, sliced by
/// [`Self::new_start`]/[`Self::new_count`] into "already cached" vs "new
/// this step" the same way [`crate::generate::LoadedModel`]'s own
/// `cached_len`/`new_count` split already does for the KV cache.
/// `all_token_ids` is prompt + every token generated so far, INCLUDING the
/// tokens this step is about to evaluate (so an n-gram hash table lookup
/// over `tokens[i-k..=i]` can read the trailing context of the newest
/// token, not just tokens already cached).
pub struct StepInputContext<'ids> {
    pub all_token_ids: &'ids [u32],
    /// Index into [`Self::all_token_ids`] of the first token this step
    /// evaluates -- `0` on the prefill step (`new_count ==` the whole
    /// prompt), `all_token_ids.len() - 1` on every decode step after
    /// (single new token per step).
    pub new_start: usize,
    /// `all_token_ids.len() - new_start` -- carried directly rather than
    /// recomputed, since it is also the block length every
    /// [`Architecture::step_inputs`] override must produce per new
    /// position.
    pub new_count: usize,
}

/// One named leaf [`Architecture::step_inputs`] hands back for this step --
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

/// A `'static` table of registered architectures, resolved by
/// `general.architecture` string match, plus one designated fallback for
/// any checkpoint whose own architecture string names a real family
/// ([`Architecture::name`]'s own doc) that has not registered itself --
/// today, every non-qwen35 checkpoint `load_inner` has ever accepted.
/// `&'static dyn Architecture` (never `Box`/`Arc`): each registrant is a
/// `static` value the registering crate owns; the registry holds
/// references, not ownership, so building the table is the only
/// allocation (setup path, not hot path). An open/unbounded set of
/// foreign architecture crates is exactly the case guiding-principles §20
/// carries as the legitimate dynamic-dispatch exception.
pub struct ArchitectureRegistry {
    entries: Vec<&'static dyn Architecture>,
    default: Option<&'static dyn Architecture>,
}

impl Default for ArchitectureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchitectureRegistry {
    /// An empty table with no default -- [`Self::resolve`] returns
    /// [`InteropError::UnknownArchitecture`] for any checkpoint whose
    /// `general.architecture` does not match a name a caller has
    /// [`Self::register`]ed. The strict counterpart to [`Self::with_builtin`]:
    /// a caller that wants "reject anything not explicitly registered"
    /// rather than "fall back to dense" starts here instead.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            default: None,
        }
    }

    /// Registers this crate's own two architectures --
    /// `qwen35::QWEN35` by exact name, `dense::DENSE` as both a named
    /// entry and the registry's fallback (see [`Self::resolve`]'s doc for
    /// why a fallback is not the same thing as a third named entry).
    #[must_use]
    pub fn with_builtin() -> Self {
        let mut registry = Self {
            entries: Vec::new(),
            default: None,
        };
        registry.register(&crate::qwen35::QWEN35);
        registry.register(&crate::dense::DENSE);
        registry.default = Some(&crate::dense::DENSE);
        registry
    }

    /// Adds `architecture` to the table, matched by [`Architecture::name`]
    /// before the registry's default is consulted. Registering the same
    /// name twice keeps both entries; [`Self::resolve`] returns whichever
    /// was registered first, so a caller that wants to override a
    /// built-in name registers its own before calling
    /// [`Self::with_builtin`]'s equivalent, or builds the table from
    /// scratch rather than extending it.
    pub fn register(&mut self, architecture: &'static dyn Architecture) -> &mut Self {
        self.entries.push(architecture);
        self
    }

    /// The exact names this registry would match by [`Architecture::name`],
    /// in registration order -- test/introspection surface, not consulted
    /// by [`Self::resolve`] itself.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.entries.iter().map(|architecture| architecture.name()).collect()
    }

    /// Reads `general.architecture` off `parsed` and returns the first
    /// registered architecture whose [`Architecture::name`] matches it, or
    /// this registry's default (`None` unless [`Self::with_builtin`] or a
    /// caller's own setup provided one) when nothing matches.
    ///
    /// # Errors
    ///
    /// Whatever [`metadata_str`] fails with if `general.architecture` is
    /// absent, or [`InteropError::UnknownArchitecture`] if no registered
    /// name matches and no default is set.
    pub fn resolve(&self, parsed: &ParsedGguf) -> Result<&'static dyn Architecture, InteropError> {
        let name = metadata_str(parsed, "general.architecture")?;
        if let Some(matched) = self
            .entries
            .iter()
            .copied()
            .find(|architecture| architecture.name() == name)
        {
            return Ok(matched);
        }
        self.default
            .ok_or_else(|| InteropError::UnknownArchitecture { name: name.into() })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;
    use core::sync::atomic::{AtomicBool, Ordering};

    use proxima_gguf::value::MetadataValue as Value;
    use proxima_gguf::{GgufModel, write_complete};

    use super::*;

    /// A foreign crate's own [`Architecture`], registered against nothing
    /// this crate ships -- the seam
    /// [`ArchitectureRegistry::with_builtin`]'s own doc says a checkpoint
    /// family can add without a proxima PR. `BOUND` is the observable proof
    /// [`Architecture::bind`] actually ran (not just that `resolve` picked
    /// the right entry), the same shape `Qwen35Arch`/`DenseArch` would need
    /// if they wanted to assert the same thing from outside this crate.
    struct FakeArchitecture;

    static FAKE_BIND_CALLED: AtomicBool = AtomicBool::new(false);
    static FAKE: FakeArchitecture = FakeArchitecture;

    impl Architecture for FakeArchitecture {
        fn name(&self) -> &'static str {
            "fake-test-arch"
        }

        fn bind<'file>(
            &self,
            _parsed: &ParsedGguf,
            _file_bytes: &'file [u8],
        ) -> Result<BoundProgram<'file>, InteropError> {
            FAKE_BIND_CALLED.store(true, Ordering::SeqCst);
            Ok(BoundProgram {
                weights: BoundWeights::new(&[]),
                architecture: ModelArchitecture {
                    vocab: 0,
                    embedding: 0,
                    feed_forward: 0,
                    query_heads: 0,
                    kv_heads: 0,
                    head_dim: 0,
                    block_count: 0,
                    expert_count: 0,
                    expert_used_count: 0,
                    rope_freq_base: 0.0,
                    rms_epsilon: 0.0,
                    tied_embeddings: false,
                },
                program: Vec::new(),
                logits_root: NodeId(0),
                hidden_root: None,
                layer_roots: Vec::new(),
                moe_sites: proxima_tensor::spec::MoeSites::default(),
            })
        }
    }

    /// A minimal GGUF whose only load-bearing content is
    /// `general.architecture = name` -- every test in this module reads
    /// nothing else off it, since [`ArchitectureRegistry::resolve`] itself
    /// reads nothing else.
    fn gguf_with_architecture(name: &str) -> (proxima_gguf::pipe::ParsedGguf, Vec<u8>) {
        let model = GgufModel {
            version: 3,
            metadata: alloc::vec![(
                "general.architecture".to_string(),
                Value::String(name.to_string()),
            )],
            tensors: Vec::new(),
        };
        let file_bytes = write_complete(&model).expect("writes a minimal gguf");
        // leaked so the parsed borrow can outlive this function -- a test
        // fixture, not a hot path (the same reason
        // `bind.rs`'s own fixtures never worry about freeing this).
        let leaked: &'static [u8] = Vec::leak(file_bytes.clone());
        let parsed =
            proxima_gguf::parse_complete(leaked).expect("parses a minimal gguf");
        (parsed, file_bytes)
    }

    #[test]
    fn a_foreign_architecture_registers_and_its_bind_is_called_by_name() {
        FAKE_BIND_CALLED.store(false, Ordering::SeqCst);
        let mut registry = ArchitectureRegistry::with_builtin();
        registry.register(&FAKE);
        let (parsed, file_bytes) = gguf_with_architecture("fake-test-arch");

        let resolved = registry
            .resolve(&parsed)
            .expect("the registered fake architecture resolves by its own name");
        assert_eq!(resolved.name(), "fake-test-arch");

        resolved
            .bind(&parsed, &file_bytes)
            .expect("the fake architecture's own bind always succeeds");
        assert!(
            FAKE_BIND_CALLED.load(Ordering::SeqCst),
            "resolve must return the SAME architecture whose bind a caller then calls"
        );
    }

    #[test]
    fn resolve_on_an_unregistered_name_with_no_default_returns_the_typed_error() {
        let registry = ArchitectureRegistry::new();
        let (parsed, _file_bytes) = gguf_with_architecture("totally-unknown-checkpoint-family");

        match registry.resolve(&parsed) {
            Err(InteropError::UnknownArchitecture { name }) => {
                assert_eq!(name, "totally-unknown-checkpoint-family");
            }
            Ok(resolved) => panic!(
                "expected InteropError::UnknownArchitecture, resolved {:?} instead",
                resolved.name()
            ),
            Err(other) => panic!("expected InteropError::UnknownArchitecture, got {other}"),
        }
    }

    #[test]
    fn with_builtin_registers_exactly_dense_and_qwen35_by_name() {
        let registry = ArchitectureRegistry::with_builtin();
        assert_eq!(registry.names(), alloc::vec!["qwen35", "dense"]);
    }
}
