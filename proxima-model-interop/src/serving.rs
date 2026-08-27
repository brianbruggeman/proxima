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

use proxima_gguf::types::GgmlType;

use crate::error::InteropError;

/// `-ngl all` (upstream's own `n_gpu_layers = -1` convention for "offload
/// every layer"), reused verbatim rather than adding an `enum` variant for
/// the same idea.
pub const GPU_LAYERS_ALL: i32 = -1;

/// `--reasoning-budget -1` (upstream's own sentinel for "unbounded").
pub const REASONING_BUDGET_UNBOUNDED: i32 = -1;

/// The forward test's former hardcoded `FIXTURE_PATH`, kept as the
/// [`ServingConfig::default`] `model_path` so existing tests keep running
/// unmodified when no caller supplies their own checkpoint.
pub const DEFAULT_MODEL_PATH: &str =
    "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

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
    /// `-fit`: automatically fit the KV cache size to available VRAM.
    pub gpu_memory_fit: bool,
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
}

impl Default for ServingConfig<'static> {
    /// The repo owner's exact invocation, verbatim:
    /// `-c 131072 -np 1 -ctk q8_0 -ctv q8_0 -fa on -b 32 -ub 32 -ngl all
    /// -fit off --no-kv-offload --no-mmproj --reasoning-budget 1024
    /// --min-p 0`. The owner's invocation names no sampling flags at all,
    /// so every sampling knob defaults to its own disabled value
    /// (`temperature: 0.0`, not upstream's own `0.80` default -- see that
    /// field's own doc for why) -- the exact greedy path this forward has
    /// always run, byte-for-byte, proved in `generate.rs`'s own
    /// `real_openchat_file` acceptance test.
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
            gpu_memory_fit: false,
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
    if sequence > config.context_length as usize {
        return Err(InteropError::SequenceExceedsContextLength {
            sequence,
            context_length: config.context_length,
        });
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

    if config.batch_size != 0 || config.ubatch_size != 0 {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "batch_size={} ubatch_size={} (-b/-ub): the interpreter evaluates the whole \
             prompt's sequence extent as one static tensor dimension with no batching \
             loop; implementing this requires chunking prefill into batch_size-token \
             windows and feeding the interpreter one micro-batch of ubatch_size at a \
             time",
            config.batch_size, config.ubatch_size
        )));
    }

    if config.gpu_layers != 0 && config.gpu_layers != GPU_LAYERS_ALL {
        return Err(InteropError::UnsupportedServingConfig(format!(
            "gpu_layers={} (-ngl): partial per-layer GPU offload needs a per-layer \
             placement decision this forward path does not make; only 0 (cpu-only) \
             and {GPU_LAYERS_ALL} (-ngl all, whole-model offload through \
             `omega::backend::Backend::Metal`) are supported",
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

    if config.gpu_memory_fit {
        return Err(InteropError::UnsupportedServingConfig(
            "gpu_memory_fit=true (-fit on): auto-fitting the KV cache to available VRAM \
             presupposes both a GPU backend and a KV cache to size, neither of which \
             exists on this forward path yet"
                .into(),
        ));
    }

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
    /// command line this module's doc quotes.
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
        assert!(!config.gpu_memory_fit, "-fit off");
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

        assert_eq!(config.temperature, 0.0, "greedy, not upstream's 0.80 default");
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
        let error = apply_serving_config(&config, 6)
            .expect_err("parallel_sequences != 1 must be rejected");
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
}
