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

use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::Qwen35LayerRoots;

use crate::bind::{BoundWeights, ModelArchitecture, metadata_str};
use crate::error::InteropError;

/// Every weight tensor bound plus the compiled forward program, in the one
/// shape [`crate::generate::LoadedModel::load_inner`]'s qwen35 and dense
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
    /// [`crate::qwen35::qwen35_ssm_state_bytes`]'s own resident-bytes
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

impl ArchitectureRegistry {
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
