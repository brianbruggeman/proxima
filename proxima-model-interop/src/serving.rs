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

#[cfg(all(feature = "metal", target_os = "macos"))]
use omega::{DispatchType, MathMode};
use proxima_gguf::types::GgmlType;
use proxima_tensor::NumericPolicy;

use crate::error::InteropError;

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

/// `--reasoning-budget -1` (upstream's own sentinel for "unbounded").
pub const REASONING_BUDGET_UNBOUNDED: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnPrefillBackend {
    Cpu,
    Mlx,
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
        Self { prefill_before_decode: true }
    }
}

/// I11's per-layer expert-residency level, distinct from
/// [`ServingConfig::qwen35moe_residency_budget_bytes`]'s single pool shared
/// across every qwen35moe layer. Consulted at exactly one site,
/// `generate/decode.rs`'s residency-pool construction, independent of
/// [`AdmissionSchedule`] and [`PhaseSchedule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpertResidencySchedule {
    /// Byte budget applied independently to each qwen35moe layer's own
    /// resident expert set, rather than one pool shared across all layers.
    /// `0` (this field's default) disables the per-layer cap, matching
    /// today's behavior byte-for-byte -- unmeasured until a caller opts in.
    pub per_layer_budget_bytes: u64,
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
    /// `-c`: maximum context length in tokens.
    pub context_length: u32,
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
    /// `-ub`: physical micro-batch size in tokens.
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
    /// rewrites fire. [`NumericPolicy::llama_relaxed`] (this field's
    /// default) is today's measured, already-shipping behavior:
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
    /// finding this exists for). `false` (this field's default) is today's
    /// shipping fast path, unchanged. A cross-backend quality harness
    /// comparing against Metal's own exact kernels sets this `true` on its
    /// CPU-reference side so neither side's own quantization error is
    /// misattributed to the other backend.
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
    /// Requests the routed qwen35moe execution seam that runs the router,
    /// residency transition, and expert gather as separate phases. The Metal
    /// path binds only the selected expert sources at the gather boundary;
    /// the full graph remains the explicit monolithic arm.
    pub qwen35moe_pre_gather: bool,
    /// Keeps router cut tensors in caller-owned Metal buffers across the
    /// router/gather boundary instead of reading them back to the host.
    pub qwen35moe_persistent_cuts: bool,
    pub gdn_prefill_backend: GdnPrefillBackend,
    /// Byte budget for the DynaExq high-precision expert residency pool.
    pub qwen35moe_residency_budget_bytes: u64,
    /// Load-time refusal cap for `crate::memory_fit::WeightClassBytes::dense_bytes`
    /// (`crate::generate::LoadedModel::apply_memory_fit_gate`'s per-class
    /// gate; ROW 501/I2 -- "separate budgets and placement owners for
    /// expert weights, dense layers, activations, and KV; they must not
    /// collapse into one cache counter"). `0` (this field's default) is
    /// unbounded, matching today's behavior byte-for-byte.
    pub dense_weights_budget_bytes: u64,
    /// Load-time refusal cap for `crate::memory_fit::WeightClassBytes::expert_bytes`,
    /// checked independently of [`Self::qwen35moe_residency_budget_bytes`]
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
    pub qwen35moe_expert_prefetch: bool,
    /// Requests the GDN prefill scan instead of the ordinary recurrent path.
    pub qwen35moe_gdn_prefill_scan: bool,
    /// Enables the explicit GDN comparison gate for the prefill scan.
    pub debug_gdn_compare: bool,
    /// Allows the all-low monolithic pre-gather diagnostic path.
    pub qwen35moe_monolithic_all_low: bool,
    /// Number of adjacent qwen35moe layers to execute in one exact
    /// sidecar-backed pre-gather window. `1` is the existing router/gather
    /// boundary; `2` admits the bounded pair window, which exposes both
    /// router outputs only after the pair has completed.
    pub qwen35moe_layer_window: usize,
    /// Uses the original mmap-backed expert stacks in one Metal graph. The
    /// device performs the routed descriptor lookup; no low-precision copy is
    /// substituted, so this arm is an exactness/per-submission baseline.
    pub qwen35moe_monolithic_high_mmap: bool,
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
    /// (`5a4ac2c5`). `true` (this field's default, `dec68d40`'s width fix
    /// having landed the real-checkpoint oracle) evaluates the whole prompt
    /// in one call; `PROXIMA_PREFILL_SEQUENTIAL=1` is the opt-out back to
    /// the old `next_ids.len()`-way split loop, checked at the same call
    /// site regardless of this field's own value.
    pub prefill_one_evaluation: bool,
    /// Not an upstream llama-server flag -- caps how many prompt positions
    /// [`Self::prefill_one_evaluation`]'s alt program evaluates in one call
    /// (`generate/decode.rs`'s prefill batch loop, I9/Sarathi-style chunked
    /// prefill): the prompt is split into chunks of this many positions,
    /// each built at its own width via
    /// `crate::qwen35moe::qwen35moe_forward_program_at_width`, with
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
    /// `Op::Iota`/`Op::Constant` leaf (`omega::metal::Plan::
    /// mark_plan_time_constants_resident`'s own doc) resident the same way
    /// [`crate::generate::BackendRuntime::build_placed_plan`]'s
    /// `resident_names` already marks checkpoint weights: computed once by
    /// its first real dispatch, then read back from [`Plan::device_buffers`]
    /// on every later call against that plan instead of re-dispatching a
    /// kernel for it every token. `false` (this field's default) is today's
    /// shipped behavior, unchanged.
    pub plan_time_constants: bool,
    /// Not an upstream llama-server flag -- a hard ceiling on
    /// `omega::metal::MetalStageTotals::gpu_exec_calls` (Metal command
    /// buffers committed) per decode step, checked in
    /// `generate::decode::run_decode_loop_placed_kv` against that step's
    /// own `metal_stage_totals()` snapshot. `0` (this field's default)
    /// disables the check -- unmeasured until a caller opts in. `N > 0`
    /// returns [`InteropError::TooManyCommandBuffers`] the first step that
    /// exceeds it, instead of silently letting a future full-graph
    /// regression multiply command-buffer submissions per token.
    pub max_command_buffers_per_token: usize,
    /// I11 scheduling level 1 of 3: request admission. See
    /// [`AdmissionSchedule`]'s own doc for the one site that consults it.
    pub admission_schedule: AdmissionSchedule,
    /// I11 scheduling level 2 of 3: phase scheduling (prefill vs decode).
    /// See [`PhaseSchedule`]'s own doc for the one site that consults it.
    pub phase_schedule: PhaseSchedule,
    /// I11 scheduling level 3 of 3: per-layer expert residency. See
    /// [`ExpertResidencySchedule`]'s own doc for the one site that
    /// consults it.
    pub expert_residency_schedule: ExpertResidencySchedule,
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
}

impl Default for ServingConfig<'static> {
    /// The repo owner's exact invocation, verbatim, with ONE deliberate
    /// deviation: `-c 131072 -np 1 -ctk q8_0 -ctv q8_0 -fa on -b 32 -ub 32
    /// -ngl all -fit off --no-kv-offload --no-mmproj --reasoning-budget 1024
    /// --min-p 0`. `gpu_memory_fit` (`-fit`) defaults `true` here, not the
    /// invocation's own `off` -- see that field's own doc for why the safe
    /// default won this argument over exact invocation fidelity. Every
    /// other field, and every sampling knob (the invocation names none, so
    /// each defaults to its own disabled value -- `temperature: 0.0`, not
    /// upstream's own `0.80`, see that field's own doc), remains the exact
    /// greedy path this forward has always run, byte-for-byte, proved in
    /// `generate.rs`'s own `real_openchat_file` acceptance test.
    fn default() -> Self {
        Self {
            model_path: DEFAULT_MODEL_PATH,
            context_length: 131_072,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::Q8_0,
            kv_cache_value_quant: GgmlType::Q8_0,
            flash_attention: true,
            batch_size: 32,
            ubatch_size: 32,
            gpu_layers: GPU_LAYERS_ALL,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 1024,
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
            numeric_policy: NumericPolicy::llama_relaxed(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: DispatchType::Serial,
            // Correctness-first default: CPU uses the scalar/dequantized
            // reference path so its oracle is comparable with GPU kernels.
            // `PROXIMA_RELAXED_ACTIVATIONS` is an explicit performance
            // escape, never the implicit behavior.
            exact_activations: true,
            weight_precision: &[],
            qwen35moe_pre_gather: false,
            qwen35moe_persistent_cuts: false,
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            qwen35moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            qwen35moe_expert_prefetch: false,
            qwen35moe_layer_window: 1,
            qwen35moe_gdn_prefill_scan: false,
            debug_gdn_compare: false,
            qwen35moe_monolithic_all_low: false,
            qwen35moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: true,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: false,
            max_command_buffers_per_token: 0,
            admission_schedule: AdmissionSchedule { max_concurrent_requests: 0 },
            phase_schedule: PhaseSchedule { prefill_before_decode: true },
            expert_residency_schedule: ExpertResidencySchedule { per_layer_budget_bytes: 0 },
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
/// [`InteropError::SequenceExceedsContextLength`] if `sequence` exceeds
/// `config.context_length`, or [`InteropError::UnsupportedServingConfig`] at
/// the first knob below whose value requests behavior this forward path
/// does not implement yet.
pub fn apply_serving_config(config: &ServingConfig, sequence: usize) -> Result<(), InteropError> {
    if matches!(config.gdn_prefill_backend, GdnPrefillBackend::Mlx) && !cfg!(feature = "mlx-gdn") {
        return Err(InteropError::UnsupportedServingConfig(
            "gdn_prefill_backend=mlx requires the mlx-gdn feature and an MLX installation".into(),
        ));
    }
    if sequence > config.context_length as usize {
        return Err(InteropError::SequenceExceedsContextLength {
            sequence,
            context_length: config.context_length,
        });
    }

    let max_concurrent_requests = config.admission_schedule.max_concurrent_requests;
    if max_concurrent_requests != 0 && config.parallel_sequences as usize > max_concurrent_requests {
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
             `proxima_tensor::spec::mistral_cached_forward_program`) stores F32 unquantized \
             today; Q8_0 storage and its `matmul_q8_0_f32` kernel exist \
             (`proxima_tensor::cpu::QuantizedBlock::Q8_0`) but the read path does not work \
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
            "flash_attention=true (-fa on): `mistral_forward_program` lowers attention to \
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

    if config.kv_bucket_tokens < 1 {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "kv_bucket_tokens={}: must be >= 1 -- `generate::kv_extent` divides `merged_len` \
             by this value to compute the plan-cache-key bucket, so 0 would divide by zero; \
             1 disables bucketing",
            config.kv_bucket_tokens
        )));
    }

    if !matches!(config.qwen35moe_layer_window, 1 | 2) {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "qwen35moe_layer_window={}: only 1 (the existing boundary) or 2 (the bounded exact sidecar window) is supported",
            config.qwen35moe_layer_window
        )));
    }
    if config.qwen35moe_layer_window == 2 && !config.qwen35moe_pre_gather {
        return Err(InteropError::UnsupportedServingConfig(
            "qwen35moe_layer_window=2 requires qwen35moe_pre_gather=true".into(),
        ));
    }
    if config.qwen35moe_layer_window == 2 && config.qwen35moe_persistent_cuts {
        return Err(InteropError::UnsupportedServingConfig(
            "qwen35moe_layer_window=2 currently requires qwen35moe_persistent_cuts=false because the pair window returns both router roots after one command buffer".into(),
        ));
    }
    if config.qwen35moe_layer_window == 2 && config.qwen35moe_gdn_prefill_scan {
        return Err(InteropError::UnsupportedServingConfig(
            "qwen35moe_layer_window=2 currently excludes the GDN prefill scan; its recurrent producer remains sequential by position".into(),
        ));
    }

    Ok(())
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

        assert_eq!(config.context_length, 131_072, "-c 131072");
        assert_eq!(config.parallel_sequences, 1, "-np 1");
        assert_eq!(config.kv_cache_key_quant, GgmlType::Q8_0, "-ctk q8_0");
        assert_eq!(config.kv_cache_value_quant, GgmlType::Q8_0, "-ctv q8_0");
        assert!(config.flash_attention, "-fa on");
        assert_eq!(config.batch_size, 32, "-b 32");
        assert_eq!(config.ubatch_size, 32, "-ub 32");
        assert_eq!(config.gpu_layers, GPU_LAYERS_ALL, "-ngl all");
        assert!(
            config.gpu_memory_fit,
            "gpu_memory_fit defaults true, deliberately overriding the owner's own -fit off"
        );
        assert!(!config.kv_offload, "--no-kv-offload");
        assert!(!config.multimodal_projector, "--no-mmproj");
        assert_eq!(config.reasoning_budget, 1024, "--reasoning-budget 1024");
        assert_eq!(config.min_p, 0.0, "--min-p 0");
        assert_eq!(config.model_path, DEFAULT_MODEL_PATH);
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

    /// `apply_serving_config` on the owner's own default invocation still
    /// reaches an unimplemented knob -- the owner's invocation is `-ctk
    /// q8_0 -ctv q8_0`, and only F32 is supported end to end, so
    /// `kv_cache_key_quant`/`kv_cache_value_quant` fires first, ahead of
    /// `-fa`, `-ngl`, `--reasoning-budget`. The placeholders are real, not
    /// decorative, even against the one config that matters most.
    #[test]
    fn owner_default_invocation_reaches_an_unimplemented_knob() {
        let error = apply_serving_config(&ServingConfig::default(), 6)
            .expect_err("owner's default invocation must reach an unimplemented knob");
        assert!(
            error.to_string().contains("kv_cache_key_quant"),
            "expected the kv-cache-quant gate to fire first, got: {error}"
        );
    }

    /// A config with every unimplemented knob switched to its
    /// currently-supported value runs clean -- proves the walk is a real
    /// per-field gate, not a blanket error at the top.
    #[test]
    fn fully_supported_config_applies_without_error() {
        let config = ServingConfig {
            model_path: DEFAULT_MODEL_PATH,
            context_length: 131_072,
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
            qwen35moe_pre_gather: false,
            qwen35moe_persistent_cuts: false,
            qwen35moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            qwen35moe_expert_prefetch: false,
            qwen35moe_layer_window: 1,
            qwen35moe_gdn_prefill_scan: false,
            debug_gdn_compare: false,
            qwen35moe_monolithic_all_low: false,
            qwen35moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: false,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: false,
            max_command_buffers_per_token: 0,
            admission_schedule: AdmissionSchedule { max_concurrent_requests: 0 },
            phase_schedule: PhaseSchedule { prefill_before_decode: true },
            expert_residency_schedule: ExpertResidencySchedule { per_layer_budget_bytes: 0 },
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
            context_length: 4,
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
            context_length: 131_072,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::Q8_0,
            kv_cache_value_quant: GgmlType::Q8_0,
            flash_attention: true,
            batch_size: 32,
            ubatch_size: 32,
            gpu_layers: GPU_LAYERS_ALL,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 1024,
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
            numeric_policy: NumericPolicy::llama_relaxed(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: DispatchType::Serial,
            exact_activations: true,
            weight_precision: &[],
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            qwen35moe_pre_gather: false,
            qwen35moe_persistent_cuts: false,
            qwen35moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            qwen35moe_expert_prefetch: false,
            qwen35moe_layer_window: 1,
            qwen35moe_gdn_prefill_scan: false,
            debug_gdn_compare: false,
            qwen35moe_monolithic_all_low: false,
            qwen35moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: true,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: false,
            max_command_buffers_per_token: 0,
            admission_schedule: AdmissionSchedule { max_concurrent_requests: 0 },
            phase_schedule: PhaseSchedule { prefill_before_decode: true },
            expert_residency_schedule: ExpertResidencySchedule { per_layer_budget_bytes: 0 },
        };
        assert_eq!(via_default_override, via_full_literal);
        assert_eq!(via_default_override.kv_bucket_tokens, 64);
    }

    #[test]
    fn default_kv_bucket_tokens_is_the_measured_winner() {
        assert_eq!(ServingConfig::default().kv_bucket_tokens, 32);
    }

    /// The declared default is [`NumericPolicy::llama_relaxed`] -- today's
    /// own already-shipping behavior made visible and overridable at this
    /// app edge, per this field's own doc. A silent default change here
    /// would be exactly the regression this branch's own defect report
    /// named: the serving path narrowing back to a stricter policy nobody
    /// asked for.
    #[test]
    fn default_numeric_policy_is_llama_relaxed() {
        assert_eq!(
            ServingConfig::default().numeric_policy,
            NumericPolicy::llama_relaxed()
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
            context_length: 131_072,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::Q8_0,
            kv_cache_value_quant: GgmlType::Q8_0,
            flash_attention: true,
            batch_size: 32,
            ubatch_size: 32,
            gpu_layers: GPU_LAYERS_ALL,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 1024,
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
            qwen35moe_pre_gather: false,
            qwen35moe_persistent_cuts: false,
            qwen35moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            qwen35moe_expert_prefetch: false,
            qwen35moe_layer_window: 1,
            qwen35moe_gdn_prefill_scan: false,
            debug_gdn_compare: false,
            qwen35moe_monolithic_all_low: false,
            qwen35moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: true,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: false,
            max_command_buffers_per_token: 0,
            admission_schedule: AdmissionSchedule { max_concurrent_requests: 0 },
            phase_schedule: PhaseSchedule { prefill_before_decode: true },
            expert_residency_schedule: ExpertResidencySchedule { per_layer_budget_bytes: 0 },
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
            context_length: 131_072,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::Q8_0,
            kv_cache_value_quant: GgmlType::Q8_0,
            flash_attention: true,
            batch_size: 32,
            ubatch_size: 32,
            gpu_layers: GPU_LAYERS_ALL,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 1024,
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
            qwen35moe_pre_gather: false,
            qwen35moe_persistent_cuts: false,
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            qwen35moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            qwen35moe_expert_prefetch: false,
            qwen35moe_layer_window: 1,
            qwen35moe_gdn_prefill_scan: false,
            debug_gdn_compare: false,
            qwen35moe_monolithic_all_low: false,
            qwen35moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: true,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: false,
            max_command_buffers_per_token: 0,
            admission_schedule: AdmissionSchedule { max_concurrent_requests: 0 },
            phase_schedule: PhaseSchedule { prefill_before_decode: true },
            expert_residency_schedule: ExpertResidencySchedule { per_layer_budget_bytes: 0 },
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
            context_length: 131_072,
            parallel_sequences: 1,
            kv_cache_key_quant: GgmlType::Q8_0,
            kv_cache_value_quant: GgmlType::Q8_0,
            flash_attention: true,
            batch_size: 32,
            ubatch_size: 32,
            gpu_layers: GPU_LAYERS_ALL,
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: None,
            kv_offload: false,
            multimodal_projector: false,
            reasoning_budget: 1024,
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
            qwen35moe_pre_gather: false,
            qwen35moe_persistent_cuts: false,
            gdn_prefill_backend: GdnPrefillBackend::Cpu,
            qwen35moe_residency_budget_bytes: 0,
            dense_weights_budget_bytes: 0,
            expert_weights_budget_bytes: 0,
            activations_budget_bytes: 0,
            kv_cache_budget_bytes: 0,
            qwen35moe_expert_prefetch: false,
            qwen35moe_layer_window: 1,
            qwen35moe_gdn_prefill_scan: false,
            debug_gdn_compare: false,
            qwen35moe_monolithic_all_low: false,
            qwen35moe_monolithic_high_mmap: false,
            gpu_correctness_fallback: false,
            prefill_one_evaluation: true,
            prefill_chunk_positions: 0,
            cached_attention_fusion: true,
            gated_delta_net_fusion: true,
            moe_topk_fusion: true,
            plan_time_constants: false,
            max_command_buffers_per_token: 0,
            admission_schedule: AdmissionSchedule { max_concurrent_requests: 0 },
            phase_schedule: PhaseSchedule { prefill_before_decode: true },
            expert_residency_schedule: ExpertResidencySchedule { per_layer_budget_bytes: 0 },
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
            admission_schedule: AdmissionSchedule { max_concurrent_requests: 4 },
            ..baseline
        };
        assert_eq!(admission_changed.phase_schedule, baseline.phase_schedule);
        assert_eq!(
            admission_changed.expert_residency_schedule,
            baseline.expert_residency_schedule
        );

        let phase_changed = ServingConfig {
            phase_schedule: PhaseSchedule { prefill_before_decode: false },
            ..baseline
        };
        assert_eq!(phase_changed.admission_schedule, baseline.admission_schedule);
        assert_eq!(
            phase_changed.expert_residency_schedule,
            baseline.expert_residency_schedule
        );

        let residency_changed = ServingConfig {
            expert_residency_schedule: ExpertResidencySchedule { per_layer_budget_bytes: 1024 },
            ..baseline
        };
        assert_eq!(residency_changed.admission_schedule, baseline.admission_schedule);
        assert_eq!(residency_changed.phase_schedule, baseline.phase_schedule);

        let over_admission = ServingConfig {
            admission_schedule: AdmissionSchedule { max_concurrent_requests: 1 },
            parallel_sequences: 2,
            ..baseline
        };
        let error = apply_serving_config(&over_admission, 6)
            .expect_err("parallel_sequences over the admission ceiling must be rejected");
        assert!(error.to_string().contains("max_concurrent_requests"));
        assert!(!error.to_string().contains("phase_schedule"));
    }
}
