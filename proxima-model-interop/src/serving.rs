//! Per-invocation serving knobs for the CPU forward path, plain Rust data
//! mirroring llama-server's CLI surface (`-c`, `-np`, `-ctk`, `-ctv`, `-fa`,
//! `-b`, `-ub`, `-ngl`, `-fit`, `--no-kv-offload`, `--no-mmproj`,
//! `--reasoning-budget`, `--min-p`) one field per flag, plus a configurable
//! `model_path` in place of the forward test's former hardcoded fixture
//! constant.
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
//! validated or folded into the forward, an unimplemented one fires
//! `todo!` naming what implementing it requires and what happens instead.
//! A field that is neither validated, folded in, nor `todo!`-guarded here
//! is a knob silently ignored, which this module treats as a bug, not an
//! omission -- see this crate's own doc and the task that added this file.

use proxima_gguf::types::GgmlType;

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
    /// `--min-p`: minimum-probability sampling cutoff. `0.0` (the owner's
    /// `--min-p 0`) disables the filter.
    pub min_p: f32,
}

impl Default for ServingConfig<'static> {
    /// The repo owner's exact invocation, verbatim:
    /// `-c 131072 -np 1 -ctk q8_0 -ctv q8_0 -fa on -b 32 -ub 32 -ngl all
    /// -fit off --no-kv-offload --no-mmproj --reasoning-budget 1024
    /// --min-p 0`.
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
            min_p: 0.0,
        }
    }
}

/// Walks every [`ServingConfig`] field against a forward pass whose prompt
/// is `sequence` tokens long. Implemented knobs are validated or already
/// match what the current forward does; every other knob fires `todo!`
/// naming what it means, what implementing it requires, and what runs
/// instead today. Called once per forward, as early as the tokenized
/// prompt length is known and before the program evaluates, so an owner
/// reproducing their exact invocation gets a clear failure at the first
/// flag that is not wired yet rather than a silently different forward.
///
/// # Panics
///
/// Panics (via `assert!`) if `sequence` exceeds `config.context_length`,
/// or (via `todo!`) at the first knob below whose value requests behavior
/// this forward path does not implement yet.
pub fn apply_serving_config(config: &ServingConfig, sequence: usize) {
    assert!(
        sequence <= config.context_length as usize,
        "prompt sequence {sequence} exceeds configured context_length {} (-c)",
        config.context_length
    );

    if config.parallel_sequences != 1 {
        todo!(
            "parallel_sequences={} (-np): serving more than one sequence slot needs a \
             per-slot KV cache plus a request scheduler across slots; \
             `evaluate_quantized_named` runs exactly one sequence per call today",
            config.parallel_sequences
        );
    }

    let key_quant_supported = config.kv_cache_key_quant == GgmlType::F32;
    let value_quant_supported = config.kv_cache_value_quant == GgmlType::F32;
    if !key_quant_supported || !value_quant_supported {
        todo!(
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
        );
    }

    if config.flash_attention {
        todo!(
            "flash_attention=true (-fa on): `mistral_forward_program` lowers attention to \
             a naive multiply-then-reduce op graph, not a fused flash-attention kernel; \
             implementing this requires a new fused Op variant plus a matching cpu.rs \
             kernel that never materializes the full [seq, seq] score matrix"
        );
    }

    if config.batch_size != 0 || config.ubatch_size != 0 {
        todo!(
            "batch_size={} ubatch_size={} (-b/-ub): the interpreter evaluates the whole \
             prompt's sequence extent as one static tensor dimension with no batching \
             loop; implementing this requires chunking prefill into batch_size-token \
             windows and feeding the interpreter one micro-batch of ubatch_size at a \
             time",
            config.batch_size, config.ubatch_size
        );
    }

    if config.gpu_layers != 0 {
        todo!(
            "gpu_layers={} (-ngl): this forward path only ever dispatches through \
             proxima-tensor's CPU interpreter; implementing this requires a GPU op \
             dispatch backend and a per-layer placement decision, not just a count",
            config.gpu_layers
        );
    }

    if config.gpu_memory_fit {
        todo!(
            "gpu_memory_fit=true (-fit on): auto-fitting the KV cache to available VRAM \
             presupposes both a GPU backend and a KV cache to size, neither of which \
             exists on this forward path yet"
        );
    }

    if config.kv_offload {
        todo!(
            "kv_offload=true: offloading the KV cache to a GPU presupposes both a GPU \
             backend and a KV cache, neither of which exists on this forward path yet; \
             kv_offload=false is a no-op today since every tensor already lives on the \
             host"
        );
    }

    if config.multimodal_projector {
        todo!(
            "multimodal_projector=true (mmproj enabled): this crate has no image/audio \
             encoder or projector-weight loader; implementing this requires a second \
             GGUF checkpoint's worth of tensors bound and run through a separate vision \
             or audio forward before the text model ever sees an embedding"
        );
    }

    if config.reasoning_budget != 0 {
        todo!(
            "reasoning_budget={} (--reasoning-budget): there is no reasoning/thinking \
             block in this forward -- every token greedy_pick sees is a final-answer \
             token, so implementing this requires a chat-template-aware split between \
             reasoning and answer segments plus a token-count budget enforced on the \
             reasoning segment specifically",
            config.reasoning_budget
        );
    }

    if config.min_p != 0.0 {
        todo!(
            "min_p={} (--min-p): token selection today is `greedy_pick`'s deterministic \
             argmax with no sampling distribution at all; implementing this requires a \
             softmax-then-filter sampler before greedy_pick ever runs, which min_p=0 \
             degenerates to (no filtering, so argmax is exact) and any nonzero value \
             does not",
            config.min_p
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

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

    /// `apply_serving_config` on the owner's own default invocation still
    /// reaches an unimplemented knob -- the owner's invocation is `-ctk
    /// q8_0 -ctv q8_0`, and only F32 is supported end to end, so
    /// `kv_cache_key_quant`/`kv_cache_value_quant` fires first, ahead of
    /// `-fa`, `-ngl`, `--reasoning-budget`. The placeholders are real, not
    /// decorative, even against the one config that matters most.
    #[test]
    #[should_panic(expected = "kv_cache_key_quant")]
    fn owner_default_invocation_reaches_an_unimplemented_knob() {
        apply_serving_config(&ServingConfig::default(), 6);
    }

    /// A config with every unimplemented knob switched to its
    /// currently-supported value runs clean -- proves the walk is a real
    /// per-field gate, not a blanket `todo!()` at the top.
    #[test]
    fn fully_supported_config_applies_without_panicking() {
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
            min_p: 0.0,
        };
        apply_serving_config(&config, 6);
    }

    #[test]
    #[should_panic(expected = "parallel_sequences")]
    fn multiple_parallel_sequences_reaches_its_todo() {
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
        apply_serving_config(&config, 6);
    }

    #[test]
    #[should_panic(expected = "context_length")]
    fn prompt_longer_than_context_length_panics() {
        let config = ServingConfig {
            context_length: 4,
            ..ServingConfig::default()
        };
        apply_serving_config(&config, 6);
    }

    #[test]
    #[should_panic(expected = "min_p")]
    fn nonzero_min_p_reaches_its_todo() {
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
        apply_serving_config(&config, 6);
    }
}
