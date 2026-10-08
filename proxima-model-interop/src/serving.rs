//! Per-invocation serving knobs for the CPU forward path, plain Rust data
//! mirroring llama-server's CLI surface (`-c`, `-np`, `-ctk`, `-ctv`, `-fa`,
//! `-b`, `-ub`, `-ngl`, `-fit`, `--no-kv-offload`, `--no-mmproj`,
//! `--reasoning-budget`, `--min-p`, `--temp`, `--top-k`, `--top-p`,
//! `--repeat-last-n`, `--repeat-penalty`, `--frequency-penalty`,
//! `--presence-penalty`, `--seed`) one field per flag, plus a configurable
//! `model_path` in place of the forward test's former hardcoded fixture
//! constant.
//!
//! The eight sampling fields (`temperature` through `seed`) feed
//! `generate.rs`'s decode loop into `proxima_tokenizer::sample::
//! sample_next_token` instead of `proxima_tokenizer::greedy_pick` directly
//! -- that function's own doc is the sampler's filter chain, order, and
//! upstream citations; this module only carries the per-invocation values,
//! it does not reimplement the algorithm.
//!
//! No `serde`/`toml`/`bon`/`clap`/`conflaguration` in this module or this
//! crate's dependency graph -- an earlier runtime policy surface linked all
//! four into the runtime binary and regressed the 7B forward 31% (reverted
//! at `23a6688`); `ServingConfig` is a `derive(Debug, Clone, Copy,
//! PartialEq)` struct only, the shape proven zero-cost by
//! `proxima-tensor/src/sized.rs` for compile-time sizing -- this is the
//! same "plain data crosses the boundary" discipline applied to runtime
//! knobs instead, since these are per-invocation choices and cannot be
//! `build.rs` consts.
//! The section grammar in `serving_grammar.rs` derives serde; the `Copy` structs
//! in this module do not.
//!
//! `kv_cache_key_quant`/`kv_cache_value_quant` reuse
//! [`proxima_gguf::types::GgmlType`] rather than minting a parallel quant
//! enum -- llama.cpp's own `--cache-type-k`/`-ctv` accept exactly this
//! type-name vocabulary (`f16`, `q8_0`, `q4_0`, ...), and that is the same
//! enum [`crate::bind`] already decodes GGUF tensor bytes against.
//! `gpu_layers` reuses llama.cpp's own `n_gpu_layers` convention -- a plain
//! `i32` where `-1` means "every layer" and `>= 0` is an explicit count --
//! instead of a bespoke `enum { None, Count(u32), All }`; that sentinel is
//! not a shortcut invented for this crate, it is upstream's own
//! representation for `-ngl all`, so no new type is needed to say it.
//!
//! Every field reaches [`apply_serving_config`]: an implemented knob is
//! validated or folded into the forward, an unimplemented one returns
//! [`crate::error::InteropError::UnsupportedServingConfig`] naming what
//! implementing it requires and what happens instead. A field that is
//! neither validated, folded in, nor error-guarded here is a knob silently
//! ignored, which this module treats as a bug, not an omission -- see this
//! crate's own doc and the task that added this file.

use alloc::format;
use alloc::string::String;

#[cfg(all(feature = "metal", target_os = "macos"))]
use omega::{DispatchType, MathMode};
use proxima_gguf::types::GgmlType;
use proxima_tensor::NumericPolicy;

use crate::error::InteropError;
use crate::rope_scaling::RopeScaling;
use crate::serving_grammar::{AssembleStep, ReadSpec};

/// Which tensor names a [`WeightPrecisionRule`] applies to. Deliberately not
/// a glob: `crate::bind::bind_dense_as`/`bind_matmul_weight_as` (this rule's
/// two consumers) run once per tensor at bind time, so the match itself must
/// stay allocation-free and `no_std`-safe -- an exact name, a `blk.3.`-style
/// prefix, or a `_exps.weight`-style suffix cover every selection this
/// crate's own tensor-naming convention needs (per-layer, per-family, or
/// one specific tensor) without pulling in a glob/regex dependency for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamePattern<'model> {
    /// Matches one tensor name exactly, e.g. `"output.weight"`.
    Exact(&'model str),
    /// Matches every tensor name starting with this, e.g. `"blk.3."` for
    /// every tensor in layer 3.
    Prefix(&'model str),
    /// Matches every tensor name ending with this, e.g. `"_exps.weight"`
    /// for every mixture-of-experts stacked weight regardless of layer.
    Suffix(&'model str),
}

impl NamePattern<'_> {
    // sole caller chain is `matching_precision_target` ->
    // `precision_target_for` (`bind.rs`), both `feature = "std"`-gated --
    // without this gate a bare `cargo test` (no features) compiles this
    // method with no reachable caller and `-D dead-code` rejects it.
    #[cfg(feature = "std")]
    #[must_use]
    fn matches(&self, name: &str) -> bool {
        match self {
            NamePattern::Exact(pattern) => *pattern == name,
            NamePattern::Prefix(prefix) => name.starts_with(prefix),
            NamePattern::Suffix(suffix) => name.ends_with(suffix),
        }
    }
}

/// One per-tensor recode instruction: bind `name`-matching tensors at
/// `target` instead of whatever [`GgmlType`] the checkpoint stored them at
/// on disk. [`ServingConfig::weight_precision`] is an ordered list of these
/// -- the first rule whose [`NamePattern`] matches a given tensor name wins,
/// so a specific [`NamePattern::Exact`]/narrow-[`NamePattern::Prefix`] rule
/// must sit before a broader catch-all in the slice for it to take effect.
///
/// Composes two existing primitives rather than adding a new bind path:
/// [`proxima_gguf::quant`]'s decoder for the tensor's on-disk codec, then its
/// encoder for `target` (`crate::bind::bind_dense_as`/`bind_matmul_weight_as`'s
/// own doc names exactly where this rule set is consulted, and
/// `crate::error::InteropError::UnsupportedWeightPrecisionTarget`
/// (`std`-gated) for what
/// happens when `target` has no encoder).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightPrecisionRule<'model> {
    pub pattern: NamePattern<'model>,
    pub target: GgmlType,
}

impl<'model> WeightPrecisionRule<'model> {
    /// `true` when `name` matches this rule's [`NamePattern`].
    #[cfg(feature = "std")]
    #[must_use]
    pub(crate) fn matches(&self, name: &str) -> bool {
        self.pattern.matches(name)
    }
}

/// Walks `rules` in order and returns the first match's `target` --
/// [`WeightPrecisionRule`]'s own doc: first match wins, so callers never
/// need to know how many later rules would also have matched.
#[cfg(feature = "std")]
#[must_use]
pub(crate) fn matching_precision_target(
    rules: &[WeightPrecisionRule<'_>],
    name: &str,
) -> Option<GgmlType> {
    rules
        .iter()
        .find(|rule| rule.matches(name))
        .map(|rule| rule.target)
}

/// `-ngl all` (upstream's own `n_gpu_layers = -1` convention for "offload
/// every layer"), reused verbatim rather than adding an `enum` variant for
/// the same idea.
pub const GPU_LAYERS_ALL: i32 = -1;

/// [`ServingConfig::default`]'s `gpu_layers`: whole-model offload on a build
/// that has a GPU backend, CPU-only on one that does not --
/// [`apply_serving_config`] rejects [`GPU_LAYERS_ALL`] without the `metal`
/// feature, so a default of `-1` there would fail its own admission check.
pub const DEFAULT_GPU_LAYERS: i32 = if cfg!(feature = "metal") {
    GPU_LAYERS_ALL
} else {
    0
};

/// [`ServingConfig::default`]'s `resident_prefill_plan_bytes`: 512 MiB holds
/// one prefill plan of a 970-token prompt on a 35-layer decoder (402 MB of output slots).
pub const DEFAULT_RESIDENT_PREFILL_PLAN_BYTES: usize = 512 * 1024 * 1024;

/// [`ServingConfig::default`]'s `batch_size`: llama.cpp's own `-b` default
/// (`common/common.h` `n_batch`), the logical batch an `-ub` micro-batch is
/// clamped to.
pub const DEFAULT_BATCH_SIZE: u32 = 2048;

/// [`ServingConfig::default`]'s `ubatch_size`: llama.cpp's own `-ub` default
/// (`common/common.h` `n_ubatch`). It sits above `omega`'s
/// `TILED_GEMM_MIN_TOKENS` (`omega-runtime.toml` `[tiled_gemm].min_tokens`),
/// so a prefill chunk reaches the tiled GEMM and row-tiled attention kernels;
/// a smaller `-ub` lowers every chunk to the matvec path, which
/// `LoadedModel::generate` logs at debug level.
pub const DEFAULT_UBATCH_SIZE: u32 = 512;

/// `--reasoning-budget -1` (upstream's own sentinel for "unbounded").
pub const REASONING_BUDGET_UNBOUNDED: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GdnPrefillBackend {
    Cpu,
    Mlx,
}

/// llama.cpp's `common_speculative_type` (`common/common.h:173-186`), named
/// identically so this crate's config round-trips llama.cpp's own `--spec-type`
/// vocabulary (`common_speculative_type_to_str`,
/// `common/speculative.cpp:2229-2244`). Every upstream variant is present so
/// a config file naming a not-yet-wired type (e.g. `draft-mtp`) still parses
/// -- [`apply_serving_config`] is where an unwired selection is rejected,
/// not this enum (this crate's own convention: an unimplemented knob is a
/// per-field validation error, not a smaller enum). This crate's own
/// `speculative-decode-llama-parity` sub-specs own the draft-model
/// families' correctness; only the five n-gram types plus `None` run end to
/// end today (`generate/decode.rs`'s own speculative branch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeType {
    None,
    DraftSimple,
    DraftEagle3,
    DraftMtp,
    DraftDflash,
    DraftDspark,
    NgramSimple,
    NgramMapK,
    NgramMapK4v,
    NgramMod,
    NgramCache,
}

impl SpeculativeType {
    /// llama.cpp's own `--spec-type` string for this variant
    /// (`common_speculative_type_to_str`, `common/speculative.cpp:2229-2244`).
    #[must_use]
    pub const fn llama_name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DraftSimple => "draft-simple",
            Self::DraftEagle3 => "draft-eagle3",
            Self::DraftMtp => "draft-mtp",
            Self::DraftDflash => "draft-dflash",
            Self::DraftDspark => "draft-dspark",
            Self::NgramSimple => "ngram-simple",
            Self::NgramMapK => "ngram-map-k",
            Self::NgramMapK4v => "ngram-map-k4v",
            Self::NgramMod => "ngram-mod",
            Self::NgramCache => "ngram-cache",
        }
    }

    /// The inverse of [`Self::llama_name`]. `None` when `name` is not one of
    /// llama.cpp's own `--spec-type` strings.
    #[must_use]
    pub fn from_llama_name(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Self::None),
            "draft-simple" => Some(Self::DraftSimple),
            "draft-eagle3" => Some(Self::DraftEagle3),
            "draft-mtp" => Some(Self::DraftMtp),
            "draft-dflash" => Some(Self::DraftDflash),
            "draft-dspark" => Some(Self::DraftDspark),
            "ngram-simple" => Some(Self::NgramSimple),
            "ngram-map-k" => Some(Self::NgramMapK),
            "ngram-map-k4v" => Some(Self::NgramMapK4v),
            "ngram-mod" => Some(Self::NgramMod),
            "ngram-cache" => Some(Self::NgramCache),
            _ => None,
        }
    }
}

/// llama.cpp's fixed speculator priority order (`common/speculative.cpp:2617-2629`,
/// "this list here defines the priority of the speculators"): highest
/// priority first. Registration order in llama.cpp's own `--spec-type` list
/// never matters -- `common_get_enabled_speculative_configs` folds the
/// caller's `Vec<type>` into a bitset before this order is walked -- so
/// [`SpeculativeTypeSet::iter_priority_order`] reproduces the SAME set
/// semantics for any set a caller enables. `SpeculativeType::None` is
/// absent: llama.cpp's own `switch` in `common_speculative_init` treats that
/// variant as a no-op (`case COMMON_SPECULATIVE_TYPE_NONE: break;`), never
/// adding an implementation.
const PRIORITY_ORDER: [SpeculativeType; 10] = [
    SpeculativeType::NgramSimple,
    SpeculativeType::NgramMapK,
    SpeculativeType::NgramMapK4v,
    SpeculativeType::NgramMod,
    SpeculativeType::NgramCache,
    SpeculativeType::DraftSimple,
    SpeculativeType::DraftEagle3,
    SpeculativeType::DraftMtp,
    SpeculativeType::DraftDflash,
    SpeculativeType::DraftDspark,
];

/// llama.cpp's `std::vector<common_speculative_type> types`
/// (`common/common.h:373`) -- a SET of simultaneously-enabled speculators,
/// not a single active choice (`common_get_enabled_speculative_configs`,
/// `common/speculative.cpp:2310-2316`, folds the caller's list into exactly
/// this bitset before `common_speculative_init` walks `PRIORITY_ORDER`
/// over it). A `u16` bitmask keeps this `Copy` -- [`SpeculativeConfig`], and
/// therefore [`ServingConfig`], depend on that (this struct's own doc).
/// llama.cpp runs every enabled speculator per step in priority order until one
/// yields a non-empty draft for a position (`common_speculative_draft`,
/// `common/speculative.cpp:2802-2843`: each enabled impl's `draft()` is
/// tried in turn; the first to fill `dp.result` wins and the rest are
/// skipped via `dp.drafting = false`), then accepts through that
/// implementation and notifies every other enabled implementation with
/// `is_other = true` (`common_speculative_accept`, `:2915-2919`) so
/// stateful drafters (`ngram-mod`, `ngram-cache`) can still track
/// occupancy/acceptance across steps they did not win. This crate wires the
/// draft/verify loop for exactly one member of the set today
/// (`generate/decode.rs`'s own speculative branch checks
/// `contains(SpeculativeType::NgramSimple)`) -- the SET representation is
/// what lets a config round-trip any of llama.cpp's `--spec-type` combinations
/// even before every member is wired; [`apply_serving_config`] rejects only
/// the members that are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpeculativeTypeSet(u16);

impl SpeculativeTypeSet {
    /// llama.cpp's own default: `types = { COMMON_SPECULATIVE_TYPE_NONE }`
    /// (`common/common.h:373`) -- no speculator enabled.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// A set containing exactly one member.
    #[must_use]
    pub const fn single(type_id: SpeculativeType) -> Self {
        Self(1u16 << type_id as u16)
    }

    /// This set with `type_id` added, llama.cpp's own `types.push_back`.
    #[must_use]
    pub const fn insert(self, type_id: SpeculativeType) -> Self {
        Self(self.0 | (1u16 << type_id as u16))
    }

    /// Whether `type_id` is one of this set's enabled speculators.
    #[must_use]
    pub const fn contains(self, type_id: SpeculativeType) -> bool {
        self.0 & (1u16 << type_id as u16) != 0
    }

    /// No speculator enabled -- llama.cpp's own default, and this crate's
    /// speculation-off state.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// This set's members in llama.cpp's own fixed priority order
    /// (`PRIORITY_ORDER`'s own doc) -- the order `common_speculative_init`
    /// registers implementations in and `common_speculative_draft` tries
    /// them in, regardless of the order a caller named them in.
    pub fn iter_priority_order(self) -> impl Iterator<Item = SpeculativeType> {
        PRIORITY_ORDER
            .into_iter()
            .filter(move |&type_id| self.contains(type_id))
    }
}

/// llama.cpp's `common_params_speculative_ngram_map` (`common/common.h:361-365`),
/// shared verbatim by llama.cpp's own `ngram_simple`/`ngram_map_k`/
/// `ngram_map_k4v` fields -- one struct shape, three instances, matched here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramMapParams {
    /// llama.cpp's `size_n`: the n-gram size looked up in history.
    pub size_n: u16,
    /// llama.cpp's `size_m`: the m-gram size drafted after a match.
    pub size_m: u16,
    /// llama.cpp's `min_hits`: minimum hits before a match is proposed.
    pub min_hits: u16,
}

impl Default for NgramMapParams {
    /// llama.cpp's own default for all three of `ngram_simple`/`ngram_map_k`/
    /// `ngram_map_k4v` (`common/common.h:362-364`).
    fn default() -> Self {
        Self {
            size_n: 12,
            size_m: 48,
            min_hits: 1,
        }
    }
}

/// llama.cpp's `common_params_speculative_ngram_mod` (`common/common.h:354-359`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramModParams {
    pub n_match: u16,
    pub n_max: u16,
    pub n_min: u16,
}

impl Default for NgramModParams {
    fn default() -> Self {
        Self {
            n_match: 24,
            n_max: 64,
            n_min: 48,
        }
    }
}

/// [`ServingConfig::speculative`]'s own data: llama.cpp's `common_params_speculative`
/// (`common/common.h:372-389`), mirrored field for field -- including
/// `types`, llama.cpp's own `Vec<common_speculative_type>` SET of
/// simultaneously-enabled speculators ([`SpeculativeTypeSet`]'s own doc for
/// the set/priority/accept-notification semantics this reproduces). A `u16`
/// bitmask keeps [`Self::speculative_types`], and therefore this struct and
/// [`ServingConfig`], `Copy` (a `Vec` field never is, and [`ServingConfig`]'s
/// own `Copy` derive is load-bearing: `examples/speculative_decode_parity.rs`'s
/// OFF/ON pairs rely on it to see byte-identical input). This crate wires the
/// draft/verify loop for exactly one set member today
/// (`generate/decode.rs`'s own speculative branch checks
/// `contains(SpeculativeType::NgramSimple)`; the `Drafter` enum
/// `speculative-decode-llama-parity/TASKS.md` adds wires the rest) --
/// [`apply_serving_config`] rejects only the members not yet wired, naming
/// each. `ngram_cache_lookup_static`/`_dynamic` are borrowed (`&'model str`)
/// for the same reason `model_path` is -- the caller keeps the owned path
/// alive for `'model` (`SpeculativeSettings::as_speculative_config`,
/// `std`-gated, is the conflaguration-facing owner of that storage).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeculativeConfig<'model> {
    /// llama.cpp's `types` (`common/common.h:373`) -- the enabled-speculator set.
    pub speculative_types: SpeculativeTypeSet,
    /// llama.cpp's `common_params_speculative_draft::n_max` (`common/common.h:328`):
    /// maximum tokens to draft.
    pub n_max: i32,
    /// llama.cpp's `common_params_speculative_draft::n_min` (`common/common.h:329`):
    /// minimum draft tokens to keep.
    pub n_min: i32,
    /// llama.cpp's `common_params_speculative_draft::p_min` (`common/common.h:332`):
    /// minimum greedy-acceptance probability.
    pub p_min: f32,
    pub ngram_simple: NgramMapParams,
    pub ngram_map_k: NgramMapParams,
    pub ngram_map_k4v: NgramMapParams,
    pub ngram_mod: NgramModParams,
    /// llama.cpp's `common_params_speculative_ngram_cache::lookup_cache_static`.
    pub ngram_cache_lookup_static: Option<&'model str>,
    /// llama.cpp's `common_params_speculative_ngram_cache::lookup_cache_dynamic`.
    pub ngram_cache_lookup_dynamic: Option<&'model str>,
}

impl SpeculativeConfig<'static> {
    /// Speculation OFF: llama.cpp's own default `types = { COMMON_SPECULATIVE_TYPE_NONE }`
    /// (`common/common.h:373`) -- the empty set. This crate's [`Default`] is
    /// [`Self::ngram_simple`], not this; `none` is how a caller turns
    /// speculation off. Per-type
    /// param defaults still hold llama.cpp's own values so enabling any single
    /// member of [`Self::speculative_types`] alone reproduces llama.cpp's
    /// defaults for that type.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            speculative_types: SpeculativeTypeSet::empty(),
            n_max: 3,
            n_min: 0,
            p_min: 0.0,
            ngram_simple: NgramMapParams {
                size_n: 12,
                size_m: 48,
                min_hits: 1,
            },
            ngram_map_k: NgramMapParams {
                size_n: 12,
                size_m: 48,
                min_hits: 1,
            },
            ngram_map_k4v: NgramMapParams {
                size_n: 12,
                size_m: 48,
                min_hits: 1,
            },
            ngram_mod: NgramModParams {
                n_match: 24,
                n_max: 64,
                n_min: 48,
            },
            ngram_cache_lookup_static: None,
            ngram_cache_lookup_dynamic: None,
        }
    }

    /// Speculation on with the single `ngram-simple` drafter at llama.cpp's own
    /// per-type defaults (`size_n 12`, `size_m 48`, `min_hits 1`) -- this
    /// crate's shipped default ([`Default`]). Output is unchanged by it:
    /// every drafted token is checked against `select_decoded_token`'s own
    /// choice, so it only ever changes how many forwards a decode takes.
    /// [`Self::none`] is the off switch.
    #[must_use]
    pub const fn ngram_simple() -> Self {
        Self {
            speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramSimple),
            ..Self::none()
        }
    }
}

impl Default for SpeculativeConfig<'static> {
    fn default() -> Self {
        Self::ngram_simple()
    }
}

/// [`ServingConfig::prompt_cache`]'s data: the per-model prompt cache that
/// reuses the longest common token prefix across requests
/// (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md`). Plain `Copy`
/// numbers for the same reason [`SpeculativeConfig`] is: [`ServingConfig`]
/// stays `Copy` and this module stays free of `serde`/`bon`/`conflaguration`;
/// `PromptCacheSettings` (`std`-gated) is the env/TOML/builder owner
/// of the same fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptCacheConfig {
    /// Most host bytes the cache may hold across all entries; `0` turns the
    /// cache off, so a request prefills its whole prompt. The default is
    /// `2_147_483_648` (2 GiB). Measured on the E2B checkpoint, one entry holds 69.2
    /// MB after a 2,063-token request and 220.3 MB after an 8,207-token one:
    /// 18.9 MB of ring rows that never grow, plus the three full-attention
    /// layers' rows at 12,288 bytes per token, allocated by doubling. That
    /// puts four 8k-token conversations at 0.88 GB, with room for their
    /// checkpoints. The memory-fit gate lowers this to what the host limit
    /// has left after the model's own budget.
    pub byte_budget: u64,
    /// Most entries the cache holds, whatever the byte budget allows.
    pub max_entries: u32,
    /// Extra rows each sliding-window ring layer keeps past its window while
    /// the cache is on, so a rewind of up to this many tokens finds the rows
    /// the window needs. Raising it costs `slack` rows per ring layer per
    /// entry; the ring rewinds `stored_len - lcp` only when that is no more
    /// than the slack. Applied on top of any speculative-decode slack as a
    /// maximum, never a sum.
    pub ring_rewind_slack: u32,
    /// Tokens between sliding-window checkpoints (the spec): a request
    /// stops its prefill at every multiple of this inside the range it
    /// prefills and snapshots the ring layers, so a later rewind past
    /// [`Self::ring_rewind_slack`] restores the nearest one and prefills from
    /// there. `0` takes none at intervals (caller-marked turn ends and the
    /// resume point still count).
    pub checkpoint_interval: u32,
    /// Checkpoints kept per entry, `0` for none. The earliest is pinned and
    /// the oldest of the rest evicted first. One E2B checkpoint is 12
    /// MiB (12 ring layers x 512 rows x 2,048 bytes), counted against
    /// [`Self::byte_budget`].
    pub max_checkpoints: u32,
    /// Shortest run of tokens worth shifting after a divergence
    /// (llama.cpp's `n_cache_reuse`): a run of at least this many tokens that the
    /// prompt shares with the entry at another position is moved with its keys
    /// re-rotated by the position delta instead of prefilled, e.g. `256` for
    /// the turns kept after a summary replaced the middle of a conversation.
    /// The moved rows were computed under the entry's older context, so the
    /// layers past the first full-attention layer hold keys and values that
    /// differ from a fresh prefill of the new prompt, and the ids generated
    /// from them can differ too, as they do under llama-server; an entry built
    /// from a shift carries those rows into later requests. Needs
    /// [`Self::ring_rewind_slack`] above zero, and a run whose sliding-window
    /// rows the entry's ring has overwritten is prefilled instead. `0` keeps
    /// chunk reuse off like llama.cpp's `n_cache_reuse`.
    pub cache_reuse_min: u32,
    /// Tokens one anticipatory-prefill chunk covers (the spec): a prewarm
    /// stops at every multiple of this to check whether a request is waiting,
    /// so a request that arrives mid-prewarm waits at most one chunk. `0`
    /// runs the whole prewarm as one chunk, which a request cannot preempt.
    pub prewarm_chunk_tokens: u32,
    /// How many likely next user turns the model drafts from a finished
    /// answer, each prefilled as a branch entry behind the answer's
    /// turn-boundary suffix (optional deeper anticipation); `0`
    /// drafts none. Each branch holds a full copy of the entry's rows, so
    /// every one counts against [`Self::byte_budget`] and
    /// [`Self::max_entries`], and an unused branch is evicted before any
    /// entry a request produced.
    pub follow_up_branches: u32,
    /// Most tokens one drafted user turn runs to; the draft stops earlier
    /// at the end-of-turn token.
    pub follow_up_max_tokens: u32,
    /// Sampling temperature of the draft in thousandths (`800` is 0.8): the
    /// branches differ only by seed, so `0` makes every draft the same
    /// greedy one and all but the first are dropped as duplicates.
    pub follow_up_temperature_milli: u32,
    /// How much of a prompt, in thousandths, an entry's shared prefix must
    /// cover for the request to reuse that entry (`100` is 10%); below it the
    /// request builds an entry of its own and the older one stays cached.
    /// llama-server's `--slot-prompt-similarity` default, 0.1, and its strict
    /// greater-than. An entry the prompt extends whole is reused at any
    /// similarity; `0` reuses any entry sharing the first token.
    pub min_similarity_milli: u32,
    /// Tokens per block of the prefix index: each cached entry is cut into
    /// whole blocks of this many tokens, hashed with the block before it, and
    /// a request finds the entry sharing its longest prefix by looking its own
    /// blocks up instead of comparing every entry. Smaller blocks find shorter
    /// overlaps and cost more index entries per cached token.
    pub block_tokens: u32,
    /// How many rows behind the newest row a full block's last row must be before the block is sealed. A sealed block is never rewound by an in-flight decode, so this is the deepest rewind a decode may make. `0` seals a block as soon as it is full.
    pub seal_horizon_rows: u32,
    /// Bits of the bloom filter each entry keeps over its blocks' content, which
    /// says whether a block of a prompt probably appears anywhere in the entry
    /// once the prefix has stopped matching. One filter costs `bits / 8` bytes.
    pub bloom_bits_per_entry: u32,
    /// Probes per block in that filter.
    pub bloom_hashes: u32,
}

impl PromptCacheConfig {
    /// The shipped default: 2 GiB, four entries, 256 rows of ring slack, and
    /// up to four checkpoints per entry every 2,048 tokens, prewarmed in
    /// 256-token chunks.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            byte_budget: 2 << 30,
            max_entries: 4,
            ring_rewind_slack: 256,
            checkpoint_interval: 2048,
            max_checkpoints: 4,
            cache_reuse_min: 0,
            prewarm_chunk_tokens: 256,
            follow_up_branches: 0,
            follow_up_max_tokens: 48,
            follow_up_temperature_milli: 800,
            min_similarity_milli: 100,
            block_tokens: 64,
            seal_horizon_rows: 256,
            bloom_bits_per_entry: 4096,
            bloom_hashes: 4,
        }
    }

    /// The off switch (`byte_budget` `0`): nothing is cached and every
    /// request prefills in full. Set it per request, or through
    /// `PROXIMA_PROMPT_CACHE_BYTE_BUDGET=0`.
    #[must_use]
    pub const fn off() -> Self {
        Self {
            byte_budget: 0,
            ..Self::standard()
        }
    }

    /// Whether this config asks for any caching at all.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.byte_budget > 0 && self.max_entries > 0
    }

    /// Rows of ring slack a request under this config needs on top of its
    /// speculative slack: [`Self::ring_rewind_slack`] when the cache is on,
    /// `0` when it is off.
    #[must_use]
    pub const fn rewind_slack_rows(&self) -> usize {
        if self.is_enabled() {
            self.ring_rewind_slack as usize
        } else {
            0
        }
    }
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self::standard()
    }
}

/// The forward test's former hardcoded `FIXTURE_PATH`, kept as the
/// [`ServingConfig::default`] `model_path` so existing tests keep running
/// unmodified when no caller supplies their own checkpoint.
pub const DEFAULT_MODEL_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

/// I11's request-admission scheduling level: whether a request is accepted
/// at all before it occupies a sequence slot. Consulted at exactly one
/// site, [`apply_serving_config`]'s own top-level walk, independent of how
/// an admitted request is later phase-scheduled ([`PhaseSchedule`]) or how
/// its experts are kept resident ([`ExpertResidencySchedule`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdmissionSchedule {
    /// Hard ceiling on `ServingConfig::parallel_sequences`. `0` (this
    /// field's default) disables the check, matching today's behavior
    /// byte-for-byte -- unmeasured until a caller opts in.
    pub max_concurrent_requests: usize,
}

/// I11's phase-scheduling level: prefill vs decode ordering within one
/// sequence's step loop. Consulted at exactly one site,
/// `generate/decode.rs`'s `one_evaluation_prefill_requested` computation,
/// independent of [`AdmissionSchedule`] and [`ExpertResidencySchedule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseSchedule {
    /// `true` (this field's default) keeps today's behavior byte-for-byte:
    /// a sequence's own prefill runs to completion before its decode loop
    /// starts. `false` requests interleaving prefill and decode steps
    /// across sequences sharing a batch -- not yet implemented.
    pub prefill_before_decode: bool,
}

impl Default for PhaseSchedule {
    fn default() -> Self {
        Self {
            prefill_before_decode: true,
        }
    }
}

/// I11's per-layer expert-residency level, distinct from
/// [`ServingConfig::moe_residency_budget_bytes`]'s single pool shared
/// across every recurrent-routed layer. Consulted at exactly one site,
/// `generate/decode.rs`'s residency-pool construction, independent of
/// [`AdmissionSchedule`] and [`PhaseSchedule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpertResidencySchedule {
    /// Byte budget applied independently to each recurrent-routed layer's own
    /// resident expert set, rather than one pool shared across all layers.
    /// `0` (this field's default) disables the per-layer cap, matching
    /// today's behavior byte-for-byte -- unmeasured until a caller opts in.
    pub per_layer_budget_bytes: u64,
}

/// `-c`: the context length a call asks for, and how strictly it is held to
/// the limit [`resolve_context_length`] derives from the checkpoint (its
/// trained context, or `original_context x factor` under
/// [`RopeScaling`]). [`resolve_context_length`] is the one primitive that
/// turns this into a served length; the memory fit
/// (`crate::memory_fit::fit_context_length`) then clamps the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContextLength {
    /// Serve the limit itself, e.g. 131072 for sliding-pattern at its trained context.
    #[default]
    Native,
    /// Serve exactly `n` tokens; `n` above the limit is rejected with
    /// [`InteropError::ContextExceedsTrained`].
    Within(u32),
    /// Serve exactly `n` tokens even past the limit -- llama.cpp's silent
    /// behaviour, made explicit (e.g. `Extrapolate(131_072)` on an unscaled
    /// qwen3-8b whose limit is 40960).
    Extrapolate(u32),
}

impl ContextLength {
    /// The explicit token count, or `None` for [`Self::Native`], which is
    /// unresolved until [`resolve_context_length`] sees the checkpoint.
    #[must_use]
    pub const fn length(self) -> Option<u32> {
        match self {
            Self::Native => None,
            Self::Within(length) | Self::Extrapolate(length) => Some(length),
        }
    }

    /// `self` with its token count replaced by `length`, keeping whether
    /// extrapolation was opted into, so a resolved or memory-fit-reduced
    /// length re-resolves the same way. [`Self::Native`] becomes
    /// [`Self::Within`]: a length derived from the limit is within it.
    #[must_use]
    pub const fn resolved(self, length: u32) -> Self {
        match self {
            Self::Native | Self::Within(_) => Self::Within(length),
            Self::Extrapolate(_) => Self::Extrapolate(length),
        }
    }
}

/// The attention settings of a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AttentionConfig {
    /// Which cached rows a decode step reads.
    pub read: ReadSpec,
}

/// The prefill settings of a request.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PrefillConfig<'model> {
    /// Ordered stages that build the starting cache of a request; empty is
    /// today's behaviour: prefix reuse, plus shifted-chunk reuse when the
    /// prompt cache's `cache_reuse_min` and `ring_rewind_slack` are above 0.
    pub assemble: &'model [AssembleStep],
}

/// One field per llama-server flag the repo owner's invocation sets,
/// plus `model_path`. See the module doc for why each field's shape is
/// what it is and why none of this crate's dependencies grew to carry it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ServingConfig<'model> {
    /// GGUF checkpoint path. Not one of llama-server's flags in the task's
    /// invocation (`-m` is implied by the server's own model-loading
    /// flow); added because [`crate::bind::gguf_tensor_as_f32`]'s callers
    /// need a path from somewhere other than a source constant.
    pub model_path: &'model str,
    /// `-c`: maximum context length in tokens, and whether one past the
    /// checkpoint's limit is rejected or served ([`ContextLength`]; resolved
    /// by [`resolve_context_length`]). The memory fit then clamps whichever
    /// length results.
    pub context_length: ContextLength,
    /// Replaces the GGUF's own `{arch}.rope.scaling.*` for this call when
    /// `Some` (e.g. `Some(RopeScaling::yarn(4.0, 32_768))` to run qwen3-8b at
    /// 131072); `None` keeps whatever the checkpoint declares.
    pub rope_scaling: Option<RopeScaling>,
    /// `-np`: number of parallel sequence slots served at once.
    pub parallel_sequences: u32,
    /// `-ctk`: KV cache key-tensor storage type.
    pub kv_cache_key_quant: GgmlType,
    /// `-ctv`: KV cache value-tensor storage type.
    pub kv_cache_value_quant: GgmlType,
    /// `-fa`: fused flash-attention kernel instead of the naive
    /// multiply-then-reduce attention graph.
    pub flash_attention: bool,
    /// `-b`: logical prompt-processing batch size in tokens.
    pub batch_size: u32,
    /// `-ub`: physical micro-batch size in tokens. An architecture that is
    /// not `single_position_step` (sliding-pattern, dense uniform-schedule) evaluates a
    /// prompt longer than this in `ceil(prompt_tokens / ubatch_size)`
    /// evaluations of `ubatch_size` rows each (the last takes the
    /// remainder): `cached_len` and the sliding KV ring advance per chunk
    /// and only the last chunk requests logits, so the unfused two-range
    /// attention's `[rows, keys, heads]` score tensors are sized by
    /// `ubatch_size`, not by the prompt. `0` is the control: the whole
    /// prompt in one evaluation. Speculative verify steps are never
    /// chunked.
    pub ubatch_size: u32,
    /// `-ngl`: number of layers to offload to a GPU. [`GPU_LAYERS_ALL`]
    /// for "all", `0` for CPU-only, `N` for an explicit layer count.
    pub gpu_layers: i32,
    /// `-fit`: derive this checkpoint's device-memory budget from its own
    /// shape (weights bytes + placed-KV bytes at `context_length` + a fixed
    /// arena allowance) and refuse to run, or reduce `context_length` to
    /// the largest value that fits, before
    /// `crate::generate::LoadedModel::generate_with_serving_config`
    /// (`std`-gated) asks a
    /// device for a single buffer (`crate::memory_fit`'s own module doc).
    /// `true` (this field's own default) unlike the owner's real `-fit off`
    /// invocation ([`ServingConfig::default`]'s own doc) -- a load that
    /// silently exceeds the host's own memory rather than refusing or
    /// shrinking is the defect this field exists to close, so the safe
    /// behavior is what a caller gets without having to ask for it; pass
    /// `false` to opt back out, the same explicit-override shape every
    /// other knob on this struct already has.
    pub gpu_memory_fit: bool,
    /// Optional hard ceiling for model-serving memory. `None` uses the
    /// device-reported working-set limit; `Some(bytes)` applies the smaller
    /// of this value and that limit before any Metal plan can allocate.
    pub gpu_memory_limit_bytes: Option<u64>,
    /// `--no-kv-offload` inverted: `true` allows the KV cache to live on
    /// the GPU, `false` (the owner's `--no-kv-offload`) keeps it resident
    /// on the host.
    pub kv_offload: bool,
    /// `--no-mmproj` inverted: `true` loads a multimodal projector
    /// alongside the checkpoint, `false` (the owner's `--no-mmproj`)
    /// serves text-only.
    pub multimodal_projector: bool,
    /// `--reasoning-budget`: token budget reserved for a reasoning/
    /// thinking block. `0` disables it, [`REASONING_BUDGET_UNBOUNDED`]
    /// removes the cap, `N` bounds it.
    pub reasoning_budget: i32,
    /// `--temp`: sampling temperature. `<= 0.0` (this crate's own default,
    /// not upstream's `0.80`) samples greedily -- exact argmax over
    /// whatever survives the other filters, matching this forward's
    /// pre-sampler behavior byte-for-byte (`proxima_tokenizer::sample::
    /// sample_next_token`'s own doc).
    pub temperature: f32,
    /// `--top-k`. `<= 0` disables the filter (upstream's own "use vocab
    /// size" convention).
    pub top_k: i32,
    /// `--top-p`: nucleus sampling cutoff. `1.0` disables the filter.
    pub top_p: f32,
    /// `--min-p`: minimum-probability sampling cutoff. `0.0` (the owner's
    /// `--min-p 0`) disables the filter. Must be in `0.0..=1.0` --
    /// [`apply_serving_config`] rejects anything else, since a value
    /// outside that range feeds `ln()` a domain it was never meant to see
    /// (see `proxima_tokenizer::sample`'s own min-p filter doc).
    pub min_p: f32,
    /// `--repeat-last-n`: how many of the most recently seen tokens
    /// (prompt included, matching upstream) the repetition-penalty filter
    /// counts over. Must be `>= 0` -- upstream's `-1` ("context size")
    /// sentinel is not implemented; [`apply_serving_config`] rejects it.
    pub repeat_last_n: i32,
    /// `--repeat-penalty`. `1.0` disables the multiplicative half of the
    /// penalty filter.
    pub repeat_penalty: f32,
    /// `--frequency-penalty`. `0.0` disables the per-occurrence-count
    /// subtractive penalty.
    pub frequency_penalty: f32,
    /// `--presence-penalty`. `0.0` disables the flat once-per-distinct-
    /// recent-token subtractive penalty.
    pub presence_penalty: f32,
    /// `-s`/`--seed`: seeds the sampler's `fastrand::Rng` once per
    /// `crate::generate::LoadedModel::generate_with_serving_config` call.
    /// Unlike upstream's own `LLAMA_DEFAULT_SEED` (which resolves to real
    /// OS-sourced randomness when left at its sentinel value), this field
    /// has no such fallback -- determinism is required unconditionally, so
    /// every seed value, including this struct's own default, is always
    /// literal.
    pub seed: u64,
    /// Not an upstream llama-server flag -- a plan-cache key rounding
    /// policy for `generate.rs`'s placed-KV Metal decode loop
    /// (`generate::kv_extent`). The true `merged_len` (cached_len +
    /// new_count) strictly increases every decode step, so keying the
    /// Metal plan cache on it directly forces a re-plan (prepare +
    /// op-setup) on every step; rounding `merged_len` up to this many
    /// tokens before binding it as the plan's `Extent::Symbolic(1)` keeps
    /// the plan-cache key constant across a whole bucket of steps, at the
    /// cost of computing attention over the padded (always-masked) tail.
    /// `1` disables bucketing (`merged_len` unchanged). Must be `>= 1` --
    /// [`apply_serving_config`] rejects `0`.
    pub kv_bucket_tokens: usize,
    /// Not an upstream llama-server flag -- `omega::metal::MathMode` for
    /// every kernel this call's `Plan`s compile on the Metal backend
    /// (`generate.rs`'s `BackendRuntime::new` reads this once per call and
    /// sets it via `Plan::set_math_mode`). No effect when `gpu_layers`
    /// selects the Cpu engine. `Relaxed` (this field's default) is the
    /// measured winner: `proxima-tensor/docs/discipline.md` ROW 296/297
    /// found `Safe`'s 179.2 GB/s and `Relaxed`'s 240.9-247.3 GB/s produce
    /// identical decode output on the ROW 297 acceptance loop, so `Safe`
    /// stays reachable as an explicit override, not the default.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub math_mode: MathMode,
    /// Not an upstream llama-server flag -- `proxima_tensor::NumericPolicy`,
    /// the richer permission set `MathMode` only narrows
    /// (`omega::metal::numeric_policy_as_metal_math_mode`'s own doc).
    /// `generate.rs`'s `BackendRuntime` reads this once per call and passes
    /// it INTO `plan`/`plan_named` at construction -- the policy is fixed
    /// for a plan's whole life (`omega::metal::Plan::numeric_policy`'s own
    /// doc: there is no post-hoc setter), so this field, not the narrower
    /// `math_mode` above, is the one that actually governs which bind-time
    /// rewrites fire. [`NumericPolicy::llama_relaxed`] with
    /// `epilogue_sources` granted (this field's default; the grant is what
    /// lets the reduce-epilogue pass reach an RMSNorm whose first reduce
    /// operand is a multi-reader projection output, `NumericPolicy::
    /// epilogue_sources`'s own doc) is today's measured behavior:
    /// `proxima-tensor/docs/discipline.md` ROW 362 measured the
    /// context-chunk merge it admits (keys_per_chunk 16, generated text
    /// identical, -4.9% gpu_exec) under exactly this value -- this field
    /// exists so that behavior is visible and overridable at the app edge,
    /// not only as an internal default nothing names. Present
    /// unconditionally (unlike `math_mode`/`dispatch_type` below): unlike
    /// `MathMode`, [`NumericPolicy`] is not Metal-specific -- it is
    /// `crate::bind`'s own bit-changing-rewrite gate too, and a
    /// non-Metal build still has an `apply_serving_config` walk that should
    /// see it.
    pub numeric_policy: NumericPolicy,
    /// Not an upstream llama-server flag -- `omega::metal::DispatchType` for
    /// the one compute encoder every call's `Plan`s dispatch through on the
    /// Metal backend (`generate.rs`'s `BackendRuntime::new` reads this once
    /// per call and sets it via `Plan::set_dispatch_type`). No effect when
    /// `gpu_layers` selects the Cpu engine. `Serial` is the serving default
    /// because the current concurrent hazard schedule has not been proven
    /// for the recurrent/MoE graph; callers may opt into `Concurrent` only
    /// for a path with repeated exactness evidence. See
    /// `omega::metal::DispatchType`'s own doc.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub dispatch_type: DispatchType,
    /// Not an upstream llama-server flag -- routes the CPU forward's
    /// `Q4_K`/`Q5_K`/`Q6_K` dots through their exact dequantize-then-fold
    /// kernels instead of the `q{4,5,6}k-int8-dot` activation-quantized
    /// fast path those features default on
    /// (`proxima_tensor::cpu::evaluate_quantized_exact`'s own doc names the
    /// finding this exists for). `true` (this field's default) keeps the CPU
    /// on the exact dequantize-then-fold reference so a cross-backend quality
    /// harness comparing against Metal's own exact kernels does not
    /// misattribute either side's quantization error to the other backend.
    /// `false` opts into the `q{4,5,6}k-int8-dot` fast path.
    pub exact_activations: bool,
    /// Not an upstream llama-server flag -- per-tensor bind-time recode
    /// rules ([`WeightPrecisionRule`]'s own doc), applied by
    /// `crate::bind::bind_dense_as`/`bind_matmul_weight_as` before either
    /// binds a tensor at its on-disk [`GgmlType`]. Empty (this field's
    /// default) reproduces today's behavior byte-for-byte: every tensor
    /// binds at whatever codec the checkpoint stored it in, unchanged.
    /// [`apply_serving_config`] does not walk this field -- unlike every
    /// other field on this struct, it is a bind-time choice consulted once
    /// per tensor before the forward program ever compiles, not a
    /// per-sequence decode knob, so there is nothing here for a per-`sequence`
    /// check to validate; a rule naming a target with no encoder surfaces at
    /// bind time as `crate::error::InteropError::UnsupportedWeightPrecisionTarget`
    /// (`std`-gated) instead.
    pub weight_precision: &'model [WeightPrecisionRule<'model>],
    /// Requests the routed recurrent-routed execution seam that runs the router,
    /// residency transition, and expert gather as separate phases. The Metal
    /// path binds only the selected expert sources at the gather boundary;
    /// the full graph remains the explicit monolithic arm.
    pub moe_pre_gather: bool,
    /// Keeps router cut tensors in caller-owned Metal buffers across the
    /// router/gather boundary instead of reading them back to the host.
    pub moe_persistent_cuts: bool,
    pub gdn_prefill_backend: GdnPrefillBackend,
    /// Byte budget for the DynaExq high-precision expert residency pool.
    pub moe_residency_budget_bytes: u64,
    /// Load-time refusal cap for `crate::memory_fit::WeightClassBytes::dense_bytes`
    /// (`crate::generate::LoadedModel::apply_memory_fit_gate`'s per-class
    /// gate; ROW 501/I2 -- "separate budgets and placement owners for
    /// expert weights, dense layers, activations, and KV; they must not
    /// collapse into one cache counter"). `0` (this field's default) is
    /// unbounded, matching today's behavior byte-for-byte.
    pub dense_weights_budget_bytes: u64,
    /// Load-time refusal cap for `crate::memory_fit::WeightClassBytes::expert_bytes`,
    /// checked independently of [`Self::moe_residency_budget_bytes`]
    /// (that field sizes the DynaExq high-precision pool at decode time;
    /// this one is the load-time admission cap on the checkpoint's own
    /// on-disk expert weight bytes). `0` is unbounded.
    pub expert_weights_budget_bytes: u64,
    /// Load-time refusal cap for `crate::memory_fit::MemoryBudget::arena_allowance_bytes`
    /// -- the fixed `BufferArena` scratch allotment activations are placed
    /// into. `0` is unbounded.
    pub activations_budget_bytes: u64,
    /// Load-time refusal cap for `crate::memory_fit::MemoryBudget::kv_cache_bytes`
    /// at the requested context length. `0` is unbounded.
    pub kv_cache_budget_bytes: u64,
    /// Enables route-history advice for HOBBIT prefetching.
    pub moe_expert_prefetch: bool,
    /// Allows the all-low monolithic pre-gather diagnostic path.
    pub moe_monolithic_all_low: bool,
    /// Number of adjacent recurrent-routed layers to execute in one exact
    /// sidecar-backed pre-gather window. `1` is the existing router/gather
    /// boundary; `2` admits the bounded pair window, which exposes both
    /// router outputs only after the pair has completed.
    pub moe_layer_window: usize,
    /// Uses the original mmap-backed expert stacks in one Metal graph. The
    /// device performs the routed descriptor lookup; no low-precision copy is
    /// substituted, so this arm is an exactness/per-submission baseline.
    pub moe_monolithic_high_mmap: bool,
    /// When enabled in a build with the WGPU/Vulkan driver, run the serving
    /// graph through the CPU oracle even when `gpu_layers` requests the GPU.
    /// This is an explicit correctness escape hatch for models whose GPU
    /// f32 trajectory is not token-equivalent; the default keeps the GPU
    /// path selected so its performance remains measurable.
    pub gpu_correctness_fallback: bool,
    /// Not an upstream llama-server flag -- runtime switch for the
    /// `single_position_step` prefill's alt one-evaluation program
    /// (`generate/decode.rs`'s `one_evaluation_prefill_requested`), in place
    /// of that call site's own `PROXIMA_PREFILL_ONE_EVALUATION` env var
    /// (`5a4ac2c5`). `false` (this field's default: the default path
    /// regressed on the France checkpoint, `proxima-tensor/docs/discipline.md`
    /// ROW 590) keeps the `next_ids.len()`-way split loop; `true` evaluates
    /// the whole prompt in one call. `PROXIMA_PREFILL_SEQUENTIAL=1` forces
    /// the split loop, checked at the same call site regardless of this
    /// field's own value.
    pub prefill_one_evaluation: bool,
    /// Not an upstream llama-server flag -- caps how many prompt positions
    /// [`Self::prefill_one_evaluation`]'s alt program evaluates in one call
    /// (`generate/decode.rs`'s prefill batch loop, I9/Sarathi-style chunked
    /// prefill): the prompt is split into chunks of this many positions,
    /// each built at its own width via
    /// `crate::recurrent_routed_interval::recurrent_routed_interval_forward_program_at_width`, with
    /// `cached_len` carried across chunks the same way the existing
    /// one-position split loop already carries it. Bounds peak activation
    /// memory for a long prompt instead of the whole-prompt evaluation
    /// growing activations linearly with `prompt_token_count`. `0` (this
    /// field's default) evaluates the whole prompt in one call, matching
    /// today's `prefill_one_evaluation` behavior byte-for-byte; has no
    /// effect unless `prefill_one_evaluation` is also set.
    pub prefill_chunk_positions: usize,
    /// Not an upstream llama-server flag -- runtime toggle for
    /// `proxima_tensor::bind::bind_with_fusion`'s cached-attention fused
    /// kind, mirroring that function's own
    /// `PROXIMA_DISABLE_CACHED_ATTENTION_FUSION` env-var escape hatch so a
    /// caller can flip it per invocation without setting process env.
    /// `true` (this field's default) matches today's shipped behavior.
    pub cached_attention_fusion: bool,
    /// Not an upstream llama-server flag -- runtime toggle for
    /// `bind_with_fusion`'s `gated-delta-net-fusion` rewrite, mirroring the
    /// new `PROXIMA_DISABLE_GATED_DELTA_NET_FUSION` env var. `true` (this
    /// field's default) matches today's shipped behavior on a build with
    /// the `gated-delta-net-fusion` feature compiled in (`metal`'s own
    /// default set); has no effect when that feature is absent.
    pub gated_delta_net_fusion: bool,
    /// Not an upstream llama-server flag -- runtime toggle for
    /// `bind_with_fusion`'s `moe-topk-fusion` rewrite, mirroring the new
    /// `PROXIMA_DISABLE_MOE_TOPK_FUSION` env var. `true` (this field's
    /// default) matches today's shipped behavior on a build with the
    /// `moe-topk-fusion` feature compiled in (`metal`'s own default set);
    /// has no effect when that feature is absent.
    pub moe_topk_fusion: bool,
    /// Not an upstream llama-server flag -- marks every plan's own
    /// `Op::Constant` leaf (`omega::metal::Plan::
    /// mark_plan_time_constants_resident`'s own doc) resident the same way
    /// `crate::generate::BackendRuntime::build_placed_plan`'s
    /// `resident_names` already marks checkpoint weights: computed once by
    /// its first real dispatch, then read back from `Plan::device_buffers`
    /// on every later call against that plan instead of re-dispatching a
    /// kernel for it every token. `true` (this field's default) is the
    /// shipped behavior -- a resident position is written once, on the
    /// plan's cold call, and never rewritten again, so it can never take a
    /// slot from the arena's free list (`build_buffer_arena`'s own doc).
    pub plan_time_constants: bool,
    /// Not an upstream llama-server flag -- whether a plan-cache miss that
    /// differs from the cached plan only in `kv_bound_extent` (a
    /// `kv_bucket_tokens` crossing) refits that plan in place
    /// (`omega::backend::refit_symbols`) instead of building another. `true`
    /// (this field's default) is the shipped behavior; `false` rebuilds the
    /// plan on every miss, which is the A/B arm a harness needs to show that
    /// a refitted plan emits what a fresh one does.
    pub plan_refit: bool,
    /// Not an upstream llama-server flag -- how many `MTLCommandBuffer`s
    /// `omega::metal::Plan`'s placements executor
    /// (`execute_plan_with_placements_inner`'s own
    /// `command_buffer_chunk_count` doc) splits one decode step's dispatch
    /// sequence into, threaded the same way as [`Self::plan_time_constants`]
    /// above: `BackendRuntime::new` reads this once per call and applies it
    /// via `omega::backend::set_command_buffer_chunks`. `1` (this field's
    /// default) never splits, matching every call before Intervention 6.
    /// `PROXIMA_COMMAND_BUFFER_CHUNKS=K` still overrides this per-process
    /// when set -- the same A/B escape hatch this field now supplies a
    /// config-sourced default for, not a replacement for it. A checkpoint
    /// whose family profile declares its own
    /// non-default split count (`FamilyProfile::command_buffer_chunks`'s own
    /// doc -- the sliding-pattern family's is `8`, the Intervention 6 measured decode
    /// configuration) uses that value instead of this field's own default,
    /// but never overrides a caller who set this field explicitly.
    pub command_buffer_chunks: u32,
    /// Not an upstream llama-server flag -- a hard ceiling on
    /// `omega::metal::MetalStageTotals::gpu_exec_calls` (Metal command
    /// buffers committed) per decode step, checked in
    /// `generate::decode::run_decode_loop_placed_kv` against that step's
    /// own `metal_stage_totals()` snapshot. `0` (this field's default)
    /// disables the check -- unmeasured until a caller opts in. `N > 0`
    /// returns `InteropError::TooManyCommandBuffers` the first step that
    /// exceeds it, instead of silently letting a future full-graph
    /// regression multiply command-buffer submissions per token.
    pub max_command_buffers_per_token: usize,
    /// Not an upstream llama-server flag -- the most bytes of Metal output
    /// slots the plans for prompt-width shapes (`new_count > 1`) may hold
    /// resident between requests and between the shapes of one request.
    /// A plan is a function of `(new_count, kv bucket, PlanIdentity)`, so a
    /// request that repeats a shape skips the plan build, the arena build and
    /// the plan-time constant dispatch (23-27 ms at a 970-token prompt). The
    /// plans cost device memory for as long as they are held: a 970-token
    /// prefill plan is 402 MB. `0` keeps nothing past the call that built it
    /// and frees a prompt-width plan when the shape changes, which is the
    /// behaviour before this field existed. Decode-width (`new_count == 1`)
    /// plans are not counted here; they are always resident.
    pub resident_prefill_plan_bytes: usize,
    /// I3 (ROW 501 HeteGen/FlexGen): issue the next layer's expert-source
    /// uploads while the current layer's dispatches still run, instead of
    /// waiting for the dispatches to finish first. Gated on a lifetime
    /// trace proving no overlap exists today (`instrument::layer_transfer_compute_overlap`'s
    /// own doc) -- `false` (this field's default) is today's shipped
    /// serial ordering, unchanged; the overlap arm is not wired to any
    /// call site yet (ROW 587's own residual).
    pub overlap_transfer_compute: bool,
    /// I11 scheduling level 1 of 3: request admission. See
    /// `AdmissionSchedule`'s own doc for the one site that consults it.
    pub admission_schedule: AdmissionSchedule,
    /// I11 scheduling level 2 of 3: phase scheduling (prefill vs decode).
    /// See `PhaseSchedule`'s own doc for the one site that consults it.
    pub phase_schedule: PhaseSchedule,
    /// I11 scheduling level 3 of 3: per-layer expert residency. See
    /// `ExpertResidencySchedule`'s own doc for the one site that
    /// consults it.
    pub expert_residency_schedule: ExpertResidencySchedule,
    /// llama.cpp's `common_params_speculative` (`common/common.h:372-389`). See
    /// `SpeculativeConfig`'s own doc for the shape and the one narrowing
    /// from llama.cpp's own `Vec<type>`. Consulted by `generate/decode.rs`'s
    /// speculative branch -- the sole gate for whether speculation runs,
    /// replacing this crate's former process-env toggle. On by default
    /// (`ngram-simple`); `SpeculativeConfig::none()` is the off switch.
    pub speculative: SpeculativeConfig<'model>,
    /// The per-model prompt cache (`PromptCacheConfig`'s own doc). On by
    /// default (`PromptCacheConfig::standard()`, 2 GiB); turn it off with
    /// `PromptCacheConfig::off()` or `PROXIMA_PROMPT_CACHE_BYTE_BUDGET=0`.
    /// Consulted by `generate/decode.rs`'s decode-loop entry.
    pub prompt_cache: PromptCacheConfig,
    /// Which cached rows a decode step reads.
    pub attention: AttentionConfig,
    /// Ordered stages that build the starting cache of a request. Empty is
    /// today's behaviour: prefix reuse, plus shifted-chunk reuse when the
    /// prompt cache's `cache_reuse_min` and `ring_rewind_slack` are above 0.
    pub prefill: PrefillConfig<'model>,
}

impl<'model> ServingConfig<'model> {
    /// The fluent surface guiding-principle 4 asks every config type carry
    /// alongside its data surface: `ServingConfig { weight_precision: rules,
    /// ..config }` expressed as a chainable call instead of a struct-update
    /// literal. Consumes and returns `self` so it composes with other
    /// `with_*` calls on one line, the same shape a caller already gets from
    /// `..Default::default()`'s struct-update syntax -- this module's own
    /// doc: no bespoke builder type, no `bon`/`conflaguration` dependency.
    #[must_use]
    pub const fn with_weight_precision(
        mut self,
        weight_precision: &'model [WeightPrecisionRule<'model>],
    ) -> Self {
        self.weight_precision = weight_precision;
        self
    }

    /// Same shape as [`Self::with_weight_precision`], for the speculative
    /// section.
    #[must_use]
    pub const fn with_speculative(mut self, speculative: SpeculativeConfig<'model>) -> Self {
        self.speculative = speculative;
        self
    }

    /// Same shape as [`Self::with_weight_precision`], for the prompt-cache
    /// section.
    #[must_use]
    pub const fn with_prompt_cache(mut self, prompt_cache: PromptCacheConfig) -> Self {
        self.prompt_cache = prompt_cache;
        self
    }
}

impl Default for ServingConfig<'static> {
    /// The repo owner's invocation (`-c 131072 -np 1 -ctk q8_0 -ctv q8_0 -fa
    /// on -b 32 -ub 32 -ngl all -fit off --no-kv-offload --no-mmproj
    /// --reasoning-budget 1024 --min-p 0`) restricted to what
    /// [`apply_serving_config`] admits, with `-c` unset (`Native`: the
    /// checkpoint's own limit, [`resolve_context_length`]): F32 KV (`-ctk`/`-ctv`), `-fa off`,
    /// `--reasoning-budget 0`, and `-ngl` `DEFAULT_GPU_LAYERS`. A default
    /// the admission check rejects is a defect, so the deviations are the
    /// ones admission forces. `-b`/`-ub` are llama.cpp's own defaults ([`DEFAULT_BATCH_SIZE`], [`DEFAULT_UBATCH_SIZE`]), not the invocation's 32, so real prefill chunks reach the tiled kernels. `gpu_memory_fit` (`-fit`) also defaults `true`
    /// here, not the invocation's own `off` -- see that field's own doc for
    /// why the safe default won this argument over exact invocation
    /// fidelity. Every
    /// other field, and every sampling knob (the invocation names none, so
    /// each defaults to its own disabled value -- `temperature: 0.0`, not
    /// upstream's own `0.80`, see that field's own doc), remains the exact
    /// greedy path this forward has always run, byte-for-byte, proved in
    /// `generate.rs`'s own `real_openchat_file` acceptance test.
    fn default() -> Self {
        Self {
            model_path: DEFAULT_MODEL_PATH,
            context_length: ContextLength::Native,
            rope_scaling: None,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: DEFAULT_BATCH_SIZE,
            ubatch_size: DEFAULT_UBATCH_SIZE,
            gpu_layers: DEFAULT_GPU_LAYERS,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            // 2026-09-04 quiet round 1 winner: wall 40.89 vs off (bucket
            // 1) 42.54 ms/token, hits 5/8 (64 also wins at 41.18/hits
            // 6/8; 256 loses at 48.18 -- CARD 6.3's full table).
            kv_bucket_tokens: 32,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: MathMode::Relaxed,
            numeric_policy: NumericPolicy::llama_relaxed().with_epilogue_sources(true),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: DispatchType::Serial,
            // Correctness-first default: CPU uses the scalar/dequantized
            // reference path so its oracle is comparable with GPU kernels.
            // `PROXIMA_RELAXED_ACTIVATIONS` is an explicit performance
            // escape, never the implicit behavior.
            exact_activations: true,
            weight_precision: &[],
            moe_pre_gather: false,
            moe_persistent_cuts: false,
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            moe_expert_prefetch: false,
            moe_layer_window: 1,
            moe_monolithic_all_low: false,
            moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            // default flipped false: main's default path regressed on the
            // France checkpoint (garbage tokens / EmptyLogits) somewhere in
            // 78a313d7..HEAD; see proxima-tensor/docs/discipline.md ROW 590.
            prefill_one_evaluation: false,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: true,
            plan_refit: true,
            command_buffer_chunks: 1,
            max_command_buffers_per_token: 0,
            resident_prefill_plan_bytes: DEFAULT_RESIDENT_PREFILL_PLAN_BYTES,
            overlap_transfer_compute: false,
            admission_schedule: AdmissionSchedule {
                max_concurrent_requests: 0,
            },
            phase_schedule: PhaseSchedule {
                prefill_before_decode: true,
            },
            expert_residency_schedule: ExpertResidencySchedule {
                per_layer_budget_bytes: 0,
            },
            speculative: SpeculativeConfig::default(),
            prompt_cache: PromptCacheConfig::default(),
            attention: AttentionConfig::default(),
            prefill: PrefillConfig::default(),
        }
    }
}

/// Walks every [`ServingConfig`] field against a forward pass whose prompt
/// is `sequence` tokens long. Implemented knobs are validated or already
/// match what the current forward does; every other knob returns
/// [`InteropError::UnsupportedServingConfig`] naming what it means, what
/// implementing it requires, and what runs instead today. Called once per
/// forward, as early as the tokenized prompt length is known and before the
/// program evaluates, so an owner reproducing their exact invocation gets a
/// clear error at the first flag that is not wired yet rather than a
/// silently different forward.
///
/// # Errors
///
/// [`InteropError::SequenceExceedsContextLength`] if `sequence` exceeds an
/// explicit `config.context_length` (a `Native` one is unresolved: the loaded
/// model's own `serving_context_length` resolves it to the limit before any
/// step reaches this gate), or [`InteropError::UnsupportedServingConfig`] at
/// the first knob below whose value requests behavior this forward path
/// does not implement yet.
pub fn apply_serving_config(config: &ServingConfig, sequence: usize) -> Result<(), InteropError> {
    if matches!(config.gdn_prefill_backend, GdnPrefillBackend::Mlx) && !cfg!(feature = "mlx-gdn") {
        return Err(InteropError::UnsupportedServingConfig(
            "gdn_prefill_backend=mlx requires the mlx-gdn feature and an MLX installation".into(),
        ));
    }
    if let Some(context_length) = config.context_length.length()
        && sequence > context_length as usize
    {
        return Err(InteropError::SequenceExceedsContextLength {
            sequence,
            context_length,
        });
    }

    let max_concurrent_requests = config.admission_schedule.max_concurrent_requests;
    if max_concurrent_requests != 0 && config.parallel_sequences as usize > max_concurrent_requests
    {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "admission_schedule.max_concurrent_requests={max_concurrent_requests}: \
             parallel_sequences={} exceeds the request-admission ceiling",
            config.parallel_sequences
        )));
    }

    if config.parallel_sequences != 1 {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "parallel_sequences={} (-np): serving more than one sequence slot needs a \
             per-slot KV cache plus a request scheduler across slots; \
             `evaluate_quantized_named` runs exactly one sequence per call today",
            config.parallel_sequences
        )));
    }

    let key_quant_supported = config.kv_cache_key_quant == GgmlType::F32;
    let value_quant_supported = config.kv_cache_value_quant == GgmlType::F32;
    if !key_quant_supported || !value_quant_supported {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "kv_cache_key_quant={:?} kv_cache_value_quant={:?} (-ctk/-ctv): the per-layer \
             key/value context cache (`proxima-model-interop`'s cached decode loop, \
             `proxima_tensor::spec::gqa_cached_forward_program`) stores F32 unquantized \
             today; Q8_0 storage and its `matmul_q8_0_f32` kernel exist \
             (`proxima_tensor::cpu::QuantizedBlock::Packed` with `Codec::Q8_0`) but the read path does not work \
             end to end -- the quantized matmul dispatch only handles a flat \
             `weight[rows, k] x activation[batch, k]` matmul, while the cached-attention \
             reduces are batched reduces over a shared kv-head axis (the K-cache reduce \
             keeps the cached-length axis as an output axis, the V-cache reduce contracts \
             it), which the blocking check in \
             `proxima_tensor::cpu::run_reduce_quantized` (`proxima-tensor/src/cpu.rs:2485`) \
             rejects; F16/Q4_0/every other GgmlType has no packing or matmul kernel wired \
             in at all",
            config.kv_cache_key_quant, config.kv_cache_value_quant
        )));
    }

    if config.flash_attention {
        return Err(InteropError::UnsupportedServingConfig(
            "flash_attention=true (-fa on): `gqa_forward_program` lowers attention to \
             a naive multiply-then-reduce op graph, not a fused flash-attention kernel; \
             implementing this requires a new fused Op variant plus a matching cpu.rs \
             kernel that never materializes the full [seq, seq] score matrix"
                .into(),
        ));
    }

    if config.gpu_layers != 0 && config.gpu_layers != GPU_LAYERS_ALL {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "gpu_layers={} (-ngl): partial per-layer GPU offload needs a per-layer \
             placement decision this forward path does not make; only 0 (cpu-only) \
             and {GPU_LAYERS_ALL} (-ngl all, whole-model offload through \
             `omega::backend::Engine::Gpu`) are supported",
            config.gpu_layers
        )));
    }
    if config.gpu_layers == GPU_LAYERS_ALL && !cfg!(feature = "metal") {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "gpu_layers={GPU_LAYERS_ALL} (-ngl all): this build was not compiled with \
             proxima-model-interop's `metal` feature, so there is no GPU backend to \
             offload onto"
        )));
    }

    // `gpu_memory_fit` (`-fit`) is no longer rejected here: it is the load-time
    // memory-fit gate's own switch (`crate::generate::LoadedModel::apply_memory_fit_gate`,
    // `crate::memory_fit`'s own module doc), evaluated once per
    // `generate_with_serving_config` call, before this per-step
    // `apply_serving_config` validation walk ever runs -- there is nothing
    // left for THIS function to reject.

    if config.kv_offload {
        return Err(InteropError::UnsupportedServingConfig(
            "kv_offload=true: offloading the KV cache to a GPU presupposes both a GPU \
             backend and a KV cache, neither of which exists on this forward path yet; \
             kv_offload=false is a no-op today since every tensor already lives on the \
             host"
                .into(),
        ));
    }

    if config.multimodal_projector {
        return Err(InteropError::UnsupportedServingConfig(
            "multimodal_projector=true (mmproj enabled): this crate has no image/audio \
             encoder or projector-weight loader; implementing this requires a second \
             GGUF checkpoint's worth of tensors bound and run through a separate vision \
             or audio forward before the text model ever sees an embedding"
                .into(),
        ));
    }

    if config.reasoning_budget != 0 {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "reasoning_budget={} (--reasoning-budget): there is no reasoning/thinking \
             block in this forward -- every token greedy_pick sees is a final-answer \
             token, so implementing this requires a chat-template-aware split between \
             reasoning and answer segments plus a token-count budget enforced on the \
             reasoning segment specifically",
            config.reasoning_budget
        )));
    }

    if !(0.0..=1.0).contains(&config.min_p) {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "min_p={} (--min-p): must be in 0.0..=1.0 -- the min-p filter's threshold is \
             `max_logit + ln(min_p)` (`proxima_tokenizer::sample`'s own min-p filter doc), \
             and `ln` of a value outside that range is either undefined (negative) or \
             raises the threshold above every candidate's own logit (greater than 1.0), \
             silently dropping every candidate including the argmax",
            config.min_p
        )));
    }

    if config.repeat_last_n < 0 {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "repeat_last_n={} (--repeat-last-n): upstream's own `-1` (\"context size\") \
             sentinel is not implemented here -- the repetition-penalty filter's caller \
             (`generate.rs`'s decode loop) must slice a concrete non-negative window out \
             of its own token history",
            config.repeat_last_n
        )));
    }

    if config.command_buffer_chunks < 1 {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "command_buffer_chunks={}: must be >= 1 -- omega's `command_buffer_chunk_count` \
             reads 0 as \"no config value threaded\", not a literal 0-way split; 1 (the \
             default) keeps today's single command buffer",
            config.command_buffer_chunks
        )));
    }

    if config.kv_bucket_tokens < 1 {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "kv_bucket_tokens={}: must be >= 1 -- `generate::kv_extent` divides `merged_len` \
             by this value to compute the plan-cache-key bucket, so 0 would divide by zero; \
             1 disables bucketing",
            config.kv_bucket_tokens
        )));
    }

    if !matches!(config.moe_layer_window, 1 | 2) {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "moe_layer_window={}: only 1 (the existing boundary) or 2 (the bounded exact sidecar window) is supported",
            config.moe_layer_window
        )));
    }
    if config.moe_layer_window == 2 && !config.moe_pre_gather {
        return Err(InteropError::UnsupportedServingConfig(
            "moe_layer_window=2 requires moe_pre_gather=true".into(),
        ));
    }
    if config.moe_layer_window == 2 && config.moe_persistent_cuts {
        return Err(InteropError::UnsupportedServingConfig(
            "moe_layer_window=2 currently requires moe_persistent_cuts=false because the pair window returns both router roots after one command buffer".into(),
        ));
    }

    let mut unwired_types = String::new();
    for type_id in config.speculative.speculative_types.iter_priority_order() {
        if matches!(
            type_id,
            SpeculativeType::DraftSimple
                | SpeculativeType::DraftEagle3
                | SpeculativeType::DraftMtp
                | SpeculativeType::DraftDflash
                | SpeculativeType::DraftDspark
        ) {
            if !unwired_types.is_empty() {
                unwired_types.push(',');
            }
            unwired_types.push_str(type_id.llama_name());
        }
    }
    if !unwired_types.is_empty() {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "speculative.speculative_types={unwired_types} (--spec-type): draft-model \
             speculation needs a second GGUF checkpoint's forward path, owned by each named \
             type's own speculative-<type> sub-spec; only none and the five n-gram types \
             (ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache) run end to end \
             today"
        )));
    }

    Ok(())
}

/// The context length to serve at, before the memory fit clamps it
/// (`crate::memory_fit::fit_context_length`, which takes the result as its
/// requested length): the explicit length of `requested` ([`ContextLength::Within`]
/// or [`ContextLength::Extrapolate`]), or for [`ContextLength::Native`] the
/// limit `scaling` admits over `trained` ([`RopeScaling::limit`]). `trained` is the
/// checkpoint's own `{arch}.context_length`
/// (`crate::lowering::trained_context_length`), so a default request never
/// silently runs past what the checkpoint was trained for.
///
/// # Errors
///
/// [`InteropError::ContextExceedsTrained`] when `requested` is
/// [`ContextLength::Within`] a length above the limit.
pub fn resolve_context_length(
    requested: ContextLength,
    trained: u32,
    scaling: RopeScaling,
) -> Result<u32, InteropError> {
    let limit = scaling.limit(trained);
    match requested {
        ContextLength::Native => Ok(limit),
        ContextLength::Within(requested) if requested > limit => {
            Err(InteropError::ContextExceedsTrained {
                requested,
                limit,
                scaling,
            })
        }
        ContextLength::Within(requested) | ContextLength::Extrapolate(requested) => Ok(requested),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;

    /// The struct-literal surface (guiding-principle 4's config-as-mirror):
    /// every field the owner's invocation sets is present and matches the
    /// invocation's own values, checked flag by flag against the verbatim
    /// command line this module's doc quotes -- except `gpu_memory_fit`,
    /// this default's one deliberate deviation from the invocation
    /// (`ServingConfig::default`'s own doc, `gpu_memory_fit`'s own field
    /// doc), asserted `true` here instead.
    #[test]
    fn default_matches_owner_invocation_semantics() {
        let config = ServingConfig::default();

        assert_eq!(
            config.context_length,
            ContextLength::Native,
            "-c unset resolves to the limit"
        );
        assert_eq!(config.parallel_sequences, 1, "-np 1");
        assert_eq!(config.kv_cache_key_quant, GgmlType::F32, "-ctk f32");
        assert_eq!(config.kv_cache_value_quant, GgmlType::F32, "-ctv f32");
        assert!(!config.flash_attention, "-fa off");
        assert_eq!(config.batch_size, 2048, "-b 2048");
        assert_eq!(config.ubatch_size, 512, "-ub 512");
        assert_eq!(config.gpu_layers, DEFAULT_GPU_LAYERS, "-ngl all with metal");
        assert!(
            config.gpu_memory_fit,
            "gpu_memory_fit defaults true, deliberately overriding the owner's own -fit off"
        );
        assert!(!config.kv_offload, "--no-kv-offload");
        assert!(!config.multimodal_projector, "--no-mmproj");
        assert_eq!(config.reasoning_budget, 0, "--reasoning-budget 0");
        assert_eq!(config.min_p, 0.0, "--min-p 0");
        assert_eq!(config.model_path, DEFAULT_MODEL_PATH);
    }

    /// The kernel threshold is data (`omega-runtime.toml`
    /// `[tiled_gemm].min_tokens`); the serving default chunk must clear it, or
    /// real prefill lowers to the matvec path the benches never measured.
    #[cfg(feature = "metal")]
    #[test]
    fn default_ubatch_reaches_the_tiled_gemm_threshold() {
        let config = ServingConfig::default();

        assert!(
            u64::from(config.ubatch_size) >= omega::sized::TILED_GEMM_MIN_TOKENS,
            "default ubatch {} is below TILED_GEMM_MIN_TOKENS {}",
            config.ubatch_size,
            omega::sized::TILED_GEMM_MIN_TOKENS
        );
        assert!(
            config.ubatch_size <= config.batch_size,
            "llama.cpp clamps -ub to -b"
        );
    }

    /// Every sampling field defaults to its own disabled value -- the
    /// owner's invocation names no sampling flags, so this forward's
    /// pre-sampler greedy behavior must survive unchanged.
    #[test]
    fn default_sampling_config_is_fully_disabled() {
        let config = ServingConfig::default();

        assert_eq!(
            config.temperature, 0.0,
            "greedy, not upstream's 0.80 default"
        );
        assert_eq!(config.top_k, 0, "disabled");
        assert_eq!(config.top_p, 1.0, "disabled");
        assert_eq!(config.min_p, 0.0, "disabled");
        assert_eq!(config.repeat_penalty, 1.0, "disabled");
        assert_eq!(config.frequency_penalty, 0.0, "disabled");
        assert_eq!(config.presence_penalty, 0.0, "disabled");
        assert_eq!(config.seed, 0, "always literal, never OS-sourced");
    }

    /// A default the admission check rejects is a self-consistency defect:
    /// the config a caller gets without asking must pass the function that
    /// validates configs.
    #[test]
    fn serving_default_admission_accepts_the_default() {
        apply_serving_config(&ServingConfig::default(), 1)
            .expect("ServingConfig::default() must pass apply_serving_config");
    }

    /// The paired control: admission still refuses what it must, so the
    /// default cannot pass by loosening the check. `Q4_0` KV and a second
    /// parallel sequence are both unimplemented and must stay refused.
    #[test]
    fn serving_default_admission_rejects_the_controls() {
        let two_sequences = ServingConfig {
            parallel_sequences: 2,
            ..ServingConfig::default()
        };
        let q4_0_keys = ServingConfig {
            kv_cache_key_quant: GgmlType::Q4_0,
            ..ServingConfig::default()
        };

        let sequences_error = apply_serving_config(&two_sequences, 1)
            .expect_err("parallel_sequences=2 must be rejected");
        let quant_error =
            apply_serving_config(&q4_0_keys, 1).expect_err("Q4_0 key cache must be rejected");

        assert!(sequences_error.to_string().contains("parallel_sequences"));
        assert!(quant_error.to_string().contains("kv_cache_key_quant"));
    }

    /// A config with every unimplemented knob switched to its
    /// currently-supported value runs clean -- proves the walk is a real
    /// per-field gate, not a blanket error at the top.
    #[test]
    fn serving_section_attention_defaults_to_the_dense_read() {
        assert_eq!(ServingConfig::default().attention, AttentionConfig { read: ReadSpec::Dense });
    }

    #[test]
    fn serving_section_prefill_defaults_to_todays_assemble() {
        assert!(ServingConfig::default().prefill.assemble.is_empty());
    }

    #[test]
    fn fully_supported_config_applies_without_error() {
        let config = ServingConfig {
            model_path: DEFAULT_MODEL_PATH,
            context_length: ContextLength::Native,
            rope_scaling: None,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            gpu_memory_fit: false,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            kv_bucket_tokens: 32,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: MathMode::Relaxed,
            numeric_policy: NumericPolicy::llama_relaxed(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: DispatchType::Serial,
            exact_activations: true,
            weight_precision: &[],
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            moe_pre_gather: false,
            moe_persistent_cuts: false,
            moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            moe_expert_prefetch: false,
            moe_layer_window: 1,
            moe_monolithic_all_low: false,
            moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: false,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: true,
            plan_refit: true,
            command_buffer_chunks: 1,
            max_command_buffers_per_token: 0,
            resident_prefill_plan_bytes: DEFAULT_RESIDENT_PREFILL_PLAN_BYTES,
            overlap_transfer_compute: false,
            admission_schedule: AdmissionSchedule {
                max_concurrent_requests: 0,
            },
            phase_schedule: PhaseSchedule {
                prefill_before_decode: true,
            },
            expert_residency_schedule: ExpertResidencySchedule {
                per_layer_budget_bytes: 0,
            },
            speculative: SpeculativeConfig::default(),
            prompt_cache: PromptCacheConfig::default(),
            attention: AttentionConfig::default(),
            prefill: PrefillConfig::default(),
        };
        apply_serving_config(&config, 6).expect("fully supported config must apply cleanly");
    }

    #[test]
    fn multiple_parallel_sequences_reaches_its_error() {
        let config = ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            reasoning_budget: 0,
            parallel_sequences: 4,
            ..ServingConfig::default()
        };
        let error =
            apply_serving_config(&config, 6).expect_err("parallel_sequences != 1 must be rejected");
        assert!(error.to_string().contains("parallel_sequences"));
    }

    #[test]
    fn prompt_longer_than_context_length_errors() {
        let config = ServingConfig {
            context_length: ContextLength::Within(4),
            ..ServingConfig::default()
        };
        let error =
            apply_serving_config(&config, 6).expect_err("sequence longer than -c must error");
        assert!(matches!(
            error,
            InteropError::SequenceExceedsContextLength {
                sequence: 6,
                context_length: 4
            }
        ));
    }

    /// The gap this task closes: `min_p` in its valid range no longer
    /// errors -- it is folded into the sampler `generate.rs`'s decode loop
    /// calls, not rejected here.
    #[test]
    fn nonzero_min_p_in_range_applies_without_error() {
        let config = ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            reasoning_budget: 0,
            min_p: 0.1,
            ..ServingConfig::default()
        };
        apply_serving_config(&config, 6).expect("min_p within 0.0..=1.0 must apply cleanly");
    }

    #[test]
    fn min_p_outside_the_valid_range_reaches_its_error() {
        let config = ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            reasoning_budget: 0,
            min_p: 1.5,
            ..ServingConfig::default()
        };
        let error = apply_serving_config(&config, 6).expect_err("min_p > 1.0 must be rejected");
        assert!(error.to_string().contains("min_p"));
    }

    #[test]
    fn negative_repeat_last_n_reaches_its_error() {
        let config = ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            reasoning_budget: 0,
            repeat_last_n: -1,
            ..ServingConfig::default()
        };
        let error =
            apply_serving_config(&config, 6).expect_err("negative repeat_last_n must be rejected");
        assert!(error.to_string().contains("repeat_last_n"));
    }

    /// Guiding-principle 4's config-as-mirror: a config built as a full
    /// struct literal (the "data" surface) and one built by overriding a
    /// single field on [`ServingConfig::default`] (the fluent-update
    /// surface every call site in this crate actually uses) agree bit for
    /// bit on `kv_bucket_tokens` -- there is no bespoke builder in this
    /// crate (this module's own doc: no `bon`/`conflaguration` in this
    /// crate's dependency graph), so `..Default::default()` struct-update
    /// syntax is the fluent surface plain data interoperates with.
    #[test]
    fn kv_bucket_tokens_agrees_across_literal_and_default_override() {
        let via_default_override = ServingConfig {
            kv_bucket_tokens: 64,
            ..ServingConfig::default()
        };
        let via_full_literal = ServingConfig {
            model_path: DEFAULT_MODEL_PATH,
            context_length: ContextLength::Native,
            rope_scaling: None,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: DEFAULT_BATCH_SIZE,
            ubatch_size: DEFAULT_UBATCH_SIZE,
            gpu_layers: DEFAULT_GPU_LAYERS,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            kv_bucket_tokens: 64,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: MathMode::Relaxed,
            numeric_policy: NumericPolicy::llama_relaxed().with_epilogue_sources(true),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: DispatchType::Serial,
            exact_activations: true,
            weight_precision: &[],
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            moe_pre_gather: false,
            moe_persistent_cuts: false,
            moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            moe_expert_prefetch: false,
            moe_layer_window: 1,
            moe_monolithic_all_low: false,
            moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: false,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: true,
            plan_refit: true,
            command_buffer_chunks: 1,
            max_command_buffers_per_token: 0,
            resident_prefill_plan_bytes: DEFAULT_RESIDENT_PREFILL_PLAN_BYTES,
            overlap_transfer_compute: false,
            admission_schedule: AdmissionSchedule {
                max_concurrent_requests: 0,
            },
            phase_schedule: PhaseSchedule {
                prefill_before_decode: true,
            },
            expert_residency_schedule: ExpertResidencySchedule {
                per_layer_budget_bytes: 0,
            },
            speculative: SpeculativeConfig::default(),
            prompt_cache: PromptCacheConfig::default(),
            attention: AttentionConfig::default(),
            prefill: PrefillConfig::default(),
        };
        assert_eq!(via_default_override, via_full_literal);
        assert_eq!(via_default_override.kv_bucket_tokens, 64);
    }

    #[test]
    fn default_kv_bucket_tokens_is_the_measured_winner() {
        assert_eq!(ServingConfig::default().kv_bucket_tokens, 32);
    }

    /// The declared default is [`NumericPolicy::llama_relaxed`] with
    /// `epilogue_sources` granted, so the RMSNorm apply folds into its
    /// sum-of-squares reduce (one dispatch per norm row, llama.cpp's
    /// `kernel_rms_norm_fuse_impl` shape). A silent default change here
    /// would be exactly the regression this branch's own defect report
    /// named: the serving path narrowing back to a stricter policy nobody
    /// asked for.
    #[test]
    fn default_numeric_policy_is_llama_relaxed_with_epilogue_sources() {
        assert_eq!(
            ServingConfig::default().numeric_policy,
            NumericPolicy::llama_relaxed().with_epilogue_sources(true)
        );
    }

    /// The grant is what the reduce-epilogue pass reads to consider every
    /// reduce operand of an RMSNorm apply instead of only the first (the
    /// first is the two-reader projection output, which the single-reader
    /// gate rejects), so the serving default is the switch that takes the
    /// 170 per-token norm applies of a dense decode step to zero.
    #[test]
    fn default_numeric_policy_admits_the_widened_reduce_epilogue_fusion() {
        let widened = proxima_tensor::NumericRewrite::WidenedReduceEpilogueFusion;

        assert!(
            ServingConfig::default()
                .numeric_policy
                .grants(widened.required_permissions())
        );
    }

    /// Guiding-principle 4's config-as-mirror, `numeric_policy`'s own case:
    /// a config built as a full struct literal and one built by overriding
    /// a single field on [`ServingConfig::default`] agree bit for bit --
    /// same interoperability [`kv_bucket_tokens_agrees_across_literal_and_default_override`]
    /// already proves for that field.
    #[test]
    fn numeric_policy_agrees_across_literal_and_default_override() {
        let via_default_override = ServingConfig {
            numeric_policy: NumericPolicy::bit_exact(),
            ..ServingConfig::default()
        };
        let via_full_literal = ServingConfig {
            model_path: DEFAULT_MODEL_PATH,
            context_length: ContextLength::Native,
            rope_scaling: None,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: DEFAULT_BATCH_SIZE,
            ubatch_size: DEFAULT_UBATCH_SIZE,
            gpu_layers: DEFAULT_GPU_LAYERS,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            kv_bucket_tokens: 32,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: MathMode::Relaxed,
            numeric_policy: NumericPolicy::bit_exact(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: DispatchType::Serial,
            exact_activations: true,
            weight_precision: &[],
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            moe_pre_gather: false,
            moe_persistent_cuts: false,
            moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            moe_expert_prefetch: false,
            moe_layer_window: 1,
            moe_monolithic_all_low: false,
            moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: false,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: true,
            plan_refit: true,
            command_buffer_chunks: 1,
            max_command_buffers_per_token: 0,
            resident_prefill_plan_bytes: DEFAULT_RESIDENT_PREFILL_PLAN_BYTES,
            overlap_transfer_compute: false,
            admission_schedule: AdmissionSchedule {
                max_concurrent_requests: 0,
            },
            phase_schedule: PhaseSchedule {
                prefill_before_decode: true,
            },
            expert_residency_schedule: ExpertResidencySchedule {
                per_layer_budget_bytes: 0,
            },
            speculative: SpeculativeConfig::default(),
            prompt_cache: PromptCacheConfig::default(),
            attention: AttentionConfig::default(),
            prefill: PrefillConfig::default(),
        };
        assert_eq!(via_default_override, via_full_literal);
        assert_eq!(
            via_default_override.numeric_policy,
            NumericPolicy::bit_exact()
        );
    }

    /// `exact_activations` defaults to `true` so CPU and GPU comparisons share
    /// the same dequantized arithmetic reference by default.
    #[test]
    fn default_exact_activations_is_true() {
        assert!(ServingConfig::default().exact_activations);
    }

    /// Guiding-principle 4's config-as-mirror, `exact_activations`'s own
    /// case -- same interoperability
    /// [`numeric_policy_agrees_across_literal_and_default_override`]
    /// already proves for that field.
    #[test]
    fn exact_activations_agrees_across_literal_and_default_override() {
        let via_default_override = ServingConfig {
            exact_activations: true,
            ..ServingConfig::default()
        };
        let via_full_literal = ServingConfig {
            model_path: DEFAULT_MODEL_PATH,
            context_length: ContextLength::Native,
            rope_scaling: None,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: DEFAULT_BATCH_SIZE,
            ubatch_size: DEFAULT_UBATCH_SIZE,
            gpu_layers: DEFAULT_GPU_LAYERS,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            kv_bucket_tokens: 32,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: MathMode::Relaxed,
            numeric_policy: NumericPolicy::llama_relaxed().with_epilogue_sources(true),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: DispatchType::Serial,
            exact_activations: true,
            weight_precision: &[],
            moe_pre_gather: false,
            moe_persistent_cuts: false,
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            moe_expert_prefetch: false,
            moe_layer_window: 1,
            moe_monolithic_all_low: false,
            moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: false,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: true,
            plan_refit: true,
            command_buffer_chunks: 1,
            max_command_buffers_per_token: 0,
            resident_prefill_plan_bytes: DEFAULT_RESIDENT_PREFILL_PLAN_BYTES,
            overlap_transfer_compute: false,
            admission_schedule: AdmissionSchedule {
                max_concurrent_requests: 0,
            },
            phase_schedule: PhaseSchedule {
                prefill_before_decode: true,
            },
            expert_residency_schedule: ExpertResidencySchedule {
                per_layer_budget_bytes: 0,
            },
            speculative: SpeculativeConfig::default(),
            prompt_cache: PromptCacheConfig::default(),
            attention: AttentionConfig::default(),
            prefill: PrefillConfig::default(),
        };
        assert_eq!(via_default_override, via_full_literal);
        assert!(via_default_override.exact_activations);
    }

    /// Guiding-principle 4's config-as-mirror, `prefill_one_evaluation`'s own
    /// case -- same interoperability
    /// [`exact_activations_agrees_across_literal_and_default_override`]
    /// already proves for that field.
    #[test]
    fn prefill_one_evaluation_agrees_across_literal_and_default_override() {
        let via_default_override = ServingConfig {
            prefill_one_evaluation: true,
            ..ServingConfig::default()
        };
        let via_full_literal = ServingConfig {
            model_path: DEFAULT_MODEL_PATH,
            context_length: ContextLength::Native,
            rope_scaling: None,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: DEFAULT_BATCH_SIZE,
            ubatch_size: DEFAULT_UBATCH_SIZE,
            gpu_layers: DEFAULT_GPU_LAYERS,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            kv_bucket_tokens: 32,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: MathMode::Relaxed,
            numeric_policy: NumericPolicy::llama_relaxed().with_epilogue_sources(true),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: DispatchType::Serial,
            exact_activations: true,
            weight_precision: &[],
            moe_pre_gather: false,
            moe_persistent_cuts: false,
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            moe_expert_prefetch: false,
            moe_layer_window: 1,
            moe_monolithic_all_low: false,
            moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: true,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: true,
            plan_refit: true,
            command_buffer_chunks: 1,
            max_command_buffers_per_token: 0,
            resident_prefill_plan_bytes: DEFAULT_RESIDENT_PREFILL_PLAN_BYTES,
            overlap_transfer_compute: false,
            admission_schedule: AdmissionSchedule {
                max_concurrent_requests: 0,
            },
            phase_schedule: PhaseSchedule {
                prefill_before_decode: true,
            },
            expert_residency_schedule: ExpertResidencySchedule {
                per_layer_budget_bytes: 0,
            },
            speculative: SpeculativeConfig::default(),
            prompt_cache: PromptCacheConfig::default(),
            attention: AttentionConfig::default(),
            prefill: PrefillConfig::default(),
        };
        assert_eq!(via_default_override, via_full_literal);
        assert!(via_default_override.prefill_one_evaluation);
    }

    #[test]
    fn zero_kv_bucket_tokens_reaches_its_error() {
        let config = ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            reasoning_budget: 0,
            kv_bucket_tokens: 0,
            ..ServingConfig::default()
        };
        let error =
            apply_serving_config(&config, 6).expect_err("kv_bucket_tokens=0 must be rejected");
        assert!(error.to_string().contains("kv_bucket_tokens"));
    }

    fn supported_default() -> ServingConfig<'static> {
        ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            gpu_memory_fit: false,
            reasoning_budget: 0,
            ..ServingConfig::default()
        }
    }

    #[test]
    fn default_serving_config_enables_ngram_simple_with_llama_params() {
        let speculative = ServingConfig::default().speculative;

        assert_eq!(
            speculative.speculative_types,
            SpeculativeTypeSet::single(SpeculativeType::NgramSimple)
        );
        assert_eq!(speculative.ngram_simple.size_n, 12);
        assert_eq!(speculative.ngram_simple.size_m, 48);
        assert_eq!(speculative.ngram_simple.min_hits, 1);
        assert_eq!(speculative, SpeculativeConfig::default());
        apply_serving_config(&supported_default(), 6)
            .expect("the shipped default must pass serving-config validation");
    }

    #[test]
    fn none_turns_speculation_off_and_keeps_every_other_field() {
        let off = supported_default().with_speculative(SpeculativeConfig::none());

        assert!(off.speculative.speculative_types.is_empty());
        assert_eq!(
            off.speculative.ngram_simple,
            SpeculativeConfig::default().ngram_simple
        );
        apply_serving_config(&off, 6).expect("speculation off must pass serving-config validation");
    }

    #[test]
    fn draft_model_type_is_still_rejected_when_selected_over_the_default() {
        let draft_model = supported_default().with_speculative(SpeculativeConfig {
            speculative_types: SpeculativeTypeSet::single(SpeculativeType::DraftSimple),
            ..SpeculativeConfig::none()
        });

        let error = apply_serving_config(&draft_model, 6)
            .expect_err("an unwired draft-model type must be rejected");

        assert!(error.to_string().contains("draft-simple"));
    }

    /// I11: the three scheduling levels are independent structs consulted
    /// at independent sites -- changing one level's field must not move
    /// either of the other two levels' own values, and each level's
    /// non-default value must be rejected on its own terms
    /// (`admission_schedule`'s rejection names `parallel_sequences`, not
    /// `phase_schedule` or `expert_residency_schedule`).
    #[test]
    fn scheduling_levels_are_independent() {
        let baseline = ServingConfig::default();

        let admission_changed = ServingConfig {
            admission_schedule: AdmissionSchedule {
                max_concurrent_requests: 4,
            },
            ..baseline
        };
        assert_eq!(admission_changed.phase_schedule, baseline.phase_schedule);
        assert_eq!(
            admission_changed.expert_residency_schedule,
            baseline.expert_residency_schedule
        );

        let phase_changed = ServingConfig {
            phase_schedule: PhaseSchedule {
                prefill_before_decode: false,
            },
            ..baseline
        };
        assert_eq!(
            phase_changed.admission_schedule,
            baseline.admission_schedule
        );
        assert_eq!(
            phase_changed.expert_residency_schedule,
            baseline.expert_residency_schedule
        );

        let residency_changed = ServingConfig {
            expert_residency_schedule: ExpertResidencySchedule {
                per_layer_budget_bytes: 1024,
            },
            ..baseline
        };
        assert_eq!(
            residency_changed.admission_schedule,
            baseline.admission_schedule
        );
        assert_eq!(residency_changed.phase_schedule, baseline.phase_schedule);

        let over_admission = ServingConfig {
            admission_schedule: AdmissionSchedule {
                max_concurrent_requests: 1,
            },
            parallel_sequences: 2,
            ..baseline
        };
        let error = apply_serving_config(&over_admission, 6)
            .expect_err("parallel_sequences over the admission ceiling must be rejected");
        assert!(error.to_string().contains("max_concurrent_requests"));
        assert!(!error.to_string().contains("phase_schedule"));
    }

    const QWEN3_TRAINED_CONTEXT: u32 = 40_960;

    fn qwen3_yarn_4() -> RopeScaling {
        RopeScaling::Yarn {
            factor: 4.0,
            original_context: 32_768,
            extrapolation_factor: 1.0,
            attention_factor: 1.138_629_4,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    #[proxima::test]
    #[case::qwen3_unscaled_resolves_to_the_trained_context(RopeScaling::None, 40_960)]
    #[case::qwen3_yarn_4_resolves_to_original_context_times_factor(qwen3_yarn_4(), 131_072)]
    async fn context_default_resolves(#[case] scaling: RopeScaling, #[case] expected: u32) {
        let resolved =
            resolve_context_length(ContextLength::Native, QWEN3_TRAINED_CONTEXT, scaling)
                .expect("no explicit request must resolve to the limit");

        assert_eq!(resolved, expected);
    }

    #[proxima::test]
    #[case::one_past_trained_unscaled_is_rejected(
        ContextLength::Within(40_961),
        RopeScaling::None,
        Err(40_960)
    )]
    #[case::one_past_yarn_limit_is_rejected(
        ContextLength::Within(131_073),
        qwen3_yarn_4(),
        Err(131_072)
    )]
    #[case::extrapolation_admits_the_yarn_length_unscaled(
        ContextLength::Extrapolate(131_072),
        RopeScaling::None,
        Ok(131_072)
    )]
    async fn context_over_limit(
        #[case] requested: ContextLength,
        #[case] scaling: RopeScaling,
        #[case] expected: Result<u32, u32>,
    ) {
        let outcome = resolve_context_length(requested, QWEN3_TRAINED_CONTEXT, scaling);

        match (expected, outcome) {
            (Ok(admitted), Ok(resolved)) => assert_eq!(resolved, admitted),
            (
                Err(expected_limit),
                Err(InteropError::ContextExceedsTrained {
                    requested: reported_requested,
                    limit,
                    scaling: reported_scaling,
                }),
            ) => {
                assert_eq!(Some(reported_requested), requested.length());
                assert_eq!(limit, expected_limit);
                assert_eq!(reported_scaling, scaling);
            }
            (expected, outcome) => panic!("expected {expected:?}, got {outcome:?}"),
        }
    }
}
