//! Reachable text generation: bind a checkpoint's weights once, then
//! generate text repeatedly against the bound weights without re-paying
//! the load cost.
//!
//! [`LoadedModel`] is the transform pipe (`In = (String, usize), Out =
//! (Vec<u32>, String, bool)` -- `proxima_primitives::pipe::Pipe`,
//! `proxima-primitives/src/pipe/primitives.rs:91-102`'s general form,
//! since neither `In` nor `Out` is `()`). [`LoadedModel::load`] is a plain
//! constructor, not a pipe: it pays the expensive one-time cost (mmap +
//! parse + bind 226 tensors, ~4 GB / ~120 ms prefault on the real
//! openchat-3.5 checkpoint -- `crate::bind::bind_all_weights`'s own doc)
//! and hands back a value that [`Pipe::call`] is then cheap to invoke many
//! times against, one call per generation request, without rebinding.
//! That two-step shape is the direct answer to "load once, generate
//! repeatedly": a caller holds one `LoadedModel` and calls it as many
//! times as it wants, exactly the way a caller holds one bound
//! `TcpListener` and accepts many connections from it.
//!
//! `call`'s body is synchronous CPU work wrapped in `async move { .. }`
//! with no internal `.await` -- the same shape `Pipe`'s own doc's
//! `Double`/`Always`/`Discard`/`Echo` examples use. It is still the right
//! trait: the algebra's whole point is that combinators (retry, tee,
//! rate-limit, ...) compose over `Pipe` regardless of whether a given
//! impl happens to yield control anywhere inside.
//!
//! # Stopping: the model's own signal, not just the caller's budget
//!
//! `Out`'s third field is `true` exactly when decoding stopped because the
//! model emitted its own end-of-sequence token, `false` when it stopped
//! because `max_tokens` ran out first -- the two are otherwise
//! indistinguishable to a caller (`generated_ids.len() < max_tokens` is
//! not proof of an early stop if `max_tokens` itself was small). A plain
//! `bool` earns this over a new enum because this checkpoint's own
//! metadata defines exactly one stopping condition to check, confirmed by
//! reading it rather than assumed: on the real openchat-3.5-1210 fixture
//! (`~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf`),
//! `tokenizer.ggml.eos_token_id = 32000`, which is *not* the SentencePiece
//! `</s>` (id 2) -- it is `<|end_of_turn|>`, a [`proxima_tokenizer::vocab::TokenType::Control`]
//! entry, and the same id OpenChat's own `tokenizer.chat_template` emits
//! between turns. There is no separate `tokenizer.ggml.eot_token_id` (or
//! similar) key on this fixture; the GGUF writer already folded the
//! turn-boundary marker into the one `eos_token_id` slot
//! [`proxima_tokenizer::Vocab::eos_token_id`] reads. So checking a single
//! id against [`Vocab::eos_token_id`] is this fixture's whole stopping
//! condition -- a `bool` carries it exactly; an enum would be modeling a
//! multi-token-family case this checkpoint does not have.
//!
//! The stop token itself is excluded from both the returned ids and the
//! returned text (never pushed onto `generated_ids` before the loop
//! breaks) -- symmetric exclusion, not just from decoded text, because a
//! caller who re-feeds `generated_ids` as a future prompt's tokens should
//! never see a turn-boundary marker reappear as if it were generated
//! content.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;

use proxima_gguf::GgmlType;
use proxima_gguf::pipe::ParsedGguf;
use proxima_primitives::pipe::Pipe;
use proxima_tensor::cpu::{Evaluated, QuantizedBlock};
#[cfg(not(feature = "metal"))]
use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::{CachedLayerRoots, mistral_cached_forward_program_with_experts};
use proxima_tokenizer::{SamplingConfig, Vocab, sample_next_token};

#[cfg(feature = "metal")]
use omega::backend::{Backend, Plan, execute_plan_named, mark_resident, plan_named};
#[cfg(all(feature = "instrument", feature = "metal"))]
use omega::backend::execute_plan_named_metal_op_timed;
#[cfg(all(feature = "instrument", feature = "metal"))]
use omega::metal::OpGpuTiming;
#[cfg(all(feature = "instrument", feature = "metal"))]
use omega::metal::metal_stage_totals;
#[cfg(feature = "instrument")]
use proxima_tensor::instrument::{elapsed_ticks, read_ticks, ticks_to_nanos};

use crate::bind::{BoundWeights, ModelArchitecture, architecture_from_metadata, bind_all_weights};
use crate::error::InteropError;
use crate::hf_bind::bind_all_weights_from_safetensors;
use crate::serving::ServingConfig;
use crate::serving::apply_serving_config;
#[cfg(feature = "metal")]
use crate::serving::GPU_LAYERS_ALL;

const RMS_EPSILON: f32 = 1e-5;

/// How many of [`OpGpuTiming`]'s entries [`report_op_timings`] names
/// individually -- the discipline log's own "top 20 ops by GPU time" ask.
#[cfg(all(feature = "instrument", feature = "metal"))]
const OP_PROFILE_TOP_N: usize = 20;

/// Prints the per-op GPU attribution `run_decode_loop`'s
/// `PROXIMA_METAL_OP_PROFILE_STEP` branch gathers for exactly one decode
/// step: the op count and summed GPU time (asserting the count so a
/// degenerate empty profile reads as RED, not quiet), one line per
/// `OpGpuTiming::kind` bucket, and the top [`OP_PROFILE_TOP_N`] ops by GPU
/// time with their operand bytes and bytes/ns -- exactly what settles
/// whether GPU time tracks operand bytes or is flat per dispatch.
#[cfg(all(feature = "instrument", feature = "metal"))]
fn report_op_timings(step: usize, timings: &[OpGpuTiming]) {
    let op_count = timings.len();
    let total_gpu_ns: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
    let total_operand_bytes: u64 = timings.iter().map(|timing| timing.operand_bytes).sum();

    std::println!(
        "op_profile step={step} op_count={op_count} total_gpu_ns={total_gpu_ns} \
         total_gpu_ms={:.3} total_operand_bytes={total_operand_bytes}",
        total_gpu_ns as f64 / 1e6,
    );

    let mut by_kind: alloc::collections::BTreeMap<&'static str, (u64, u64, u64)> = alloc::collections::BTreeMap::new();
    for timing in timings {
        let entry = by_kind.entry(timing.kind).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += timing.gpu_ns;
        entry.2 += timing.operand_bytes;
    }
    for (kind, (count, ns, bytes)) in &by_kind {
        std::println!(
            "op_profile_bucket step={step} kind={kind} op_count={count} gpu_ms={:.3} \
             gpu_ns_per_op={:.1} operand_bytes={bytes}",
            *ns as f64 / 1e6,
            *ns as f64 / *count as f64,
        );
    }

    let mut ranked: Vec<&OpGpuTiming> = timings.iter().collect();
    ranked.sort_by_key(|timing| core::cmp::Reverse(timing.gpu_ns));
    for (rank, timing) in ranked.iter().take(OP_PROFILE_TOP_N).enumerate() {
        std::println!(
            "op_profile_top step={step} rank={} node={} kind={} weight_name={:?} \
             operand_bytes={} operand_count={} gpu_ns={} gpu_ns_per_byte={:.6}",
            rank + 1,
            timing.node.0,
            timing.kind,
            timing.weight_name,
            timing.operand_bytes,
            timing.operand_count,
            timing.gpu_ns,
            if timing.operand_bytes == 0 {
                0.0
            } else {
                timing.gpu_ns as f64 / timing.operand_bytes as f64
            },
        );
    }

    // one row per real weight FAMILY (`blk.N.ffn_down.weight` -> `ffn_down.weight`,
    // stripping the per-layer digit so 32 layers' worth of one matmul kind
    // aggregates into one line) -- the byte-share/time-share table the
    // discipline log's "by tensor name where you can recover it" ask ends
    // on, since a per-node top-20 line cannot show whether a whole KIND is
    // slow or just its biggest instance.
    let mut by_family: alloc::collections::BTreeMap<String, FamilyGpuStats> = alloc::collections::BTreeMap::new();
    for timing in timings {
        let family = match &timing.weight_name {
            Some(name) => strip_layer_index(name),
            None => String::from("(no named operand)"),
        };
        let entry = by_family.entry(family).or_default();
        entry.op_count += 1;
        entry.gpu_ns += timing.gpu_ns;
        entry.operand_bytes += timing.operand_bytes;
        entry.min_operand_count = entry.min_operand_count.min(timing.operand_count);
        entry.max_operand_count = entry.max_operand_count.max(timing.operand_count);
        match timing.packed_row_block_rejection.as_deref() {
            Some("PASS") => stats_pass(entry, timing.gpu_ns, timing.operand_bytes),
            Some(rejection) => stats_reject(entry, rejection, timing.gpu_ns, timing.operand_bytes),
            None => {}
        }
    }
    let mut family_ranked: Vec<(&String, &FamilyGpuStats)> = by_family.iter().collect();
    family_ranked.sort_by_key(|(_, stats)| core::cmp::Reverse(stats.gpu_ns));
    for (family, stats) in family_ranked {
        std::println!(
            "op_profile_family step={step} family={family:?} op_count={} gpu_ms={:.3} \
             operand_bytes={} gpu_ns_per_byte={:.6} min_operand_count={} max_operand_count={} \
             row_blocked_count={} rejected_count={} packed_row_block_gates={:?}",
            stats.op_count,
            stats.gpu_ns as f64 / 1e6,
            stats.operand_bytes,
            if stats.operand_bytes == 0 { 0.0 } else { stats.gpu_ns as f64 / stats.operand_bytes as f64 },
            stats.min_operand_count,
            stats.max_operand_count,
            stats.row_blocked_count,
            stats.rejected_count,
            stats.packed_row_block_gates,
        );
        // A family with BOTH a row-blocked verdict AND a rejected verdict
        // (`ffn_down`/`attn_v`: 28 already-packed `Q4_K` ops PASS, 4
        // still-dequantized `Q5_K` ops reject) is exactly the case the
        // aggregate line above cannot answer on its own -- "how much of
        // this family's cost is the codec gap, not the family" -- so print
        // the split explicitly rather than making a reader subtract two
        // numbers from a set that does not carry counts.
        if stats.row_blocked_count > 0 && stats.rejected_count > 0 {
            std::println!(
                "op_profile_family_split step={step} family={family:?} \
                 passed_op_count={} passed_gpu_ms={:.3} passed_operand_bytes={} passed_gpu_ns_per_byte={:.6} \
                 rejected_op_count={} rejected_gpu_ms={:.3} rejected_operand_bytes={} rejected_gpu_ns_per_byte={:.6}",
                stats.row_blocked_count,
                stats.passed_gpu_ns as f64 / 1e6,
                stats.passed_operand_bytes,
                if stats.passed_operand_bytes == 0 {
                    0.0
                } else {
                    stats.passed_gpu_ns as f64 / stats.passed_operand_bytes as f64
                },
                stats.rejected_count,
                stats.rejected_gpu_ns as f64 / 1e6,
                stats.rejected_operand_bytes,
                if stats.rejected_operand_bytes == 0 {
                    0.0
                } else {
                    stats.rejected_gpu_ns as f64 / stats.rejected_operand_bytes as f64
                },
            );
        }
    }
}

#[cfg(all(feature = "instrument", feature = "metal"))]
fn stats_pass(entry: &mut FamilyGpuStats, gpu_ns: u64, operand_bytes: u64) {
    entry.row_blocked_count += 1;
    entry.passed_gpu_ns += gpu_ns;
    entry.passed_operand_bytes += operand_bytes;
    entry.packed_row_block_gates.insert("PASS".to_string());
}

#[cfg(all(feature = "instrument", feature = "metal"))]
fn stats_reject(entry: &mut FamilyGpuStats, rejection: &str, gpu_ns: u64, operand_bytes: u64) {
    entry.rejected_count += 1;
    entry.rejected_gpu_ns += gpu_ns;
    entry.rejected_operand_bytes += operand_bytes;
    entry.packed_row_block_gates.insert(rejection.to_string());
}

/// One tensor family's aggregated per-op GPU cost across every layer that
/// carries it, plus the DISTINCT set of [`OpGpuTiming::packed_row_block_rejection`]
/// verdicts its ops reported -- printed as a set (rather than one value)
/// because a family's own gate can legitimately vary by shape (a family's
/// `Some("PASS")` alongside a rejection would mean SOME layers took the
/// fast path and others did not, which the aggregate ns/byte alone cannot
/// show).
#[cfg(all(feature = "instrument", feature = "metal"))]
struct FamilyGpuStats {
    op_count: u64,
    gpu_ns: u64,
    operand_bytes: u64,
    min_operand_count: usize,
    max_operand_count: usize,
    row_blocked_count: u64,
    rejected_count: u64,
    /// Sum of `gpu_ns`/`operand_bytes` over exactly this family's
    /// row-blocked (`PASS`) ops -- the already-packed slice, isolated from
    /// [`Self::rejected_gpu_ns`] so a mixed family's aggregate `gpu_ns`
    /// (which sums both) is never mistaken for either codec's own cost.
    passed_gpu_ns: u64,
    passed_operand_bytes: u64,
    /// Sum of `gpu_ns`/`operand_bytes` over exactly this family's rejected
    /// ops. Before `Q5_K`'s own row-blocked kernel landed, `ffn_down`/
    /// `attn_v` each carried 4 rejected ops (`NotExactlyOnePackedOperand`
    /// on a weight the loader had already dequantized back to plain
    /// `f32`) -- this field is what isolated that codec's own cost from
    /// the 28 already-fast `Q4_K` ops sharing its family (ROW 92). Kept
    /// as a general split rather than a one-off measurement: any future
    /// codec gap in a mixed family reproduces this exact shape.
    rejected_gpu_ns: u64,
    rejected_operand_bytes: u64,
    packed_row_block_gates: BTreeSet<String>,
}

#[cfg(all(feature = "instrument", feature = "metal"))]
impl Default for FamilyGpuStats {
    fn default() -> Self {
        Self {
            row_blocked_count: 0,
            rejected_count: 0,
            op_count: 0,
            gpu_ns: 0,
            operand_bytes: 0,
            min_operand_count: usize::MAX,
            max_operand_count: 0,
            passed_gpu_ns: 0,
            passed_operand_bytes: 0,
            rejected_gpu_ns: 0,
            rejected_operand_bytes: 0,
            packed_row_block_gates: BTreeSet::new(),
        }
    }
}

/// `blk.7.ffn_down.weight` -> `ffn_down.weight`: drops exactly one
/// `.`-delimited numeric segment (the layer index every `blk.N.*` weight
/// name carries) so [`report_op_timings`] can sum one matmul KIND across
/// all 32 layers instead of reporting 32 near-identical lines.
#[cfg(all(feature = "instrument", feature = "metal"))]
fn strip_layer_index(name: &str) -> String {
    name.split('.')
        .filter(|segment| segment.parse::<u32>().is_err())
        .collect::<Vec<&str>>()
        .join(".")
}

/// A checkpoint's weights, bound once from a caller-owned byte view, plus
/// its compiled cached forward program -- everything a generation request
/// needs that does not change between requests. Borrows `file_bytes` for
/// `'file` rather than owning it, matching the rest of this crate's
/// sans-IO discipline (this crate never opens a file itself): the caller
/// keeps its own `mmap`/`Vec<u8>` alive for as long as it holds a
/// `LoadedModel` borrowed from it.
pub struct LoadedModel<'file> {
    weights: BoundWeights<'file>,
    architecture: ModelArchitecture,
    vocab: Vocab,
    program: Vec<Op>,
    logits_root: NodeId,
    cache_roots: Vec<CachedLayerRoots>,
}

impl<'file> LoadedModel<'file> {
    /// Binds every weight the cached forward program needs out of
    /// `parsed`/`file_bytes` ([`crate::bind::bind_all_weights`]), derives
    /// [`ModelArchitecture`] from `parsed`'s own metadata
    /// ([`crate::bind::architecture_from_metadata`]), builds the vocab
    /// from the same metadata, and compiles the cached forward program
    /// once. Pays the whole load cost; every [`Pipe::call`] after reuses
    /// the result.
    ///
    /// # Errors
    ///
    /// Whatever [`crate::bind::architecture_from_metadata`],
    /// [`proxima_tokenizer::gguf::vocab_from_metadata`], or
    /// [`proxima_tensor::spec::mistral_cached_forward_program_with_experts`]
    /// can fail with.
    pub fn load(parsed: &ParsedGguf, file_bytes: &'file [u8]) -> Result<Self, InteropError> {
        let architecture = architecture_from_metadata(parsed)?;
        let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)?;
        let weights = bind_all_weights(parsed, file_bytes, &architecture)?;
        // `architecture.expert_count`/`expert_used_count` read `0` for every
        // dense checkpoint (`ModelArchitecture`'s own doc), which selects
        // exactly the dense program this crate has always built -- a
        // mixture-of-experts checkpoint (`expert_count > 0`) is the only case
        // that changes which program gets compiled here.
        let (program, logits_root, cache_roots) = mistral_cached_forward_program_with_experts(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.query_heads,
            architecture.kv_heads,
            architecture.head_dim,
            architecture.block_count,
            architecture.expert_count,
            architecture.expert_used_count,
        )?;
        Ok(Self {
            weights,
            architecture,
            vocab,
            program,
            logits_root,
            cache_roots,
        })
    }

    /// [`Self::load`]'s HF/safetensors counterpart: binds every weight out
    /// of a single safetensors buffer's [`proxima_safetensors::Manifest`]
    /// ([`crate::hf_bind::bind_all_weights_from_safetensors`]) instead of a
    /// `ParsedGguf` tensor directory, and takes `architecture`/`vocab`
    /// already built rather than deriving them from the checkpoint itself --
    /// unlike GGUF, safetensors carries neither: `architecture` comes from
    /// `config.json` ([`crate::hf_config::architecture_from_hf_config`]),
    /// and HF's own vocabulary lives in `tokenizer.json`/`tokenizer_config.json`,
    /// files this crate has no reader for yet (out of scope for this
    /// change -- a caller builds its own [`Vocab`] however it can, the same
    /// way any [`Pipe`] caller owns its own setup-path inputs).
    ///
    /// `data_start` (`8 + header_len`) is the byte offset into `file_bytes`
    /// where tensor data begins -- `manifest`'s own `data_offsets` are
    /// relative to that point, never to the start of the file (see
    /// [`crate::hf_bind::bind_all_weights_from_safetensors`]'s doc); a
    /// caller who just parsed `file_bytes`'s header into `manifest` already
    /// has this value.
    ///
    /// # Errors
    ///
    /// [`InteropError::HfMoeWeightsUnsupported`] if `architecture.expert_count`
    /// is nonzero; otherwise whatever
    /// [`crate::hf_bind::bind_all_weights_from_safetensors`] or
    /// [`mistral_cached_forward_program_with_experts`] can fail with.
    pub fn load_from_safetensors(
        manifest: &proxima_safetensors::Manifest,
        file_bytes: &'file [u8],
        data_start: u64,
        architecture: ModelArchitecture,
        vocab: Vocab,
    ) -> Result<Self, InteropError> {
        let weights = bind_all_weights_from_safetensors(manifest, file_bytes, data_start, &architecture)?;
        let (program, logits_root, cache_roots) = mistral_cached_forward_program_with_experts(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.query_heads,
            architecture.kv_heads,
            architecture.head_dim,
            architecture.block_count,
            architecture.expert_count,
            architecture.expert_used_count,
        )?;
        Ok(Self {
            weights,
            architecture,
            vocab,
            program,
            logits_root,
            cache_roots,
        })
    }
}

/// This call's growable per-layer key/value cache -- `F32` only:
/// [`apply_serving_config`]'s own gate rejects any other
/// `kv_cache_key_quant`/`kv_cache_value_quant` before [`LoadedModel::call`]
/// ever reaches this loop, so there is no second precision for this type
/// to carry (contrast `bind.rs`'s own `real_openchat_file::LayerCache`,
/// which still probes the rejected `Q8_0` path directly against the
/// tensor seam that gate exists to keep unreachable here).
struct LayerCache {
    k_even: Vec<f32>,
    k_odd: Vec<f32>,
    v: Vec<f32>,
}

impl LayerCache {
    fn new() -> Self {
        Self { k_even: Vec::new(), k_odd: Vec::new(), v: Vec::new() }
    }

    fn append(&mut self, even: &[f32], odd: &[f32], value: &[f32]) {
        self.k_even.extend_from_slice(even);
        self.k_odd.extend_from_slice(odd);
        self.v.extend_from_slice(value);
    }

    fn named_blocks<'cache>(
        &'cache self,
        k_even_name: &'cache str,
        k_odd_name: &'cache str,
        v_name: &'cache str,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 3] {
        [
            (k_even_name, QuantizedBlock::Float32(self.k_even.as_slice())),
            (k_odd_name, QuantizedBlock::Float32(self.k_odd.as_slice())),
            (v_name, QuantizedBlock::Float32(self.v.as_slice())),
        ]
    }
}

/// Every per-call input the cached forward program needs beyond the model
/// weights and the growing key/value cache: `ids_f32`/RoPE `cos`/`sin` for
/// only the `new` positions this call introduces, at their true absolute
/// angle (`start_position`, not 0 -- a generated token's position is
/// `cached_len`, never the start of the sequence), plus the
/// reduce-broadcast `eps` vector sized to match.
struct PositionInputs {
    ids_f32: Vec<f32>,
    epsilon: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn build_position_inputs(
    new_ids: &[u32],
    start_position: usize,
    head_dim: u32,
    rope_freq_base: f32,
) -> PositionInputs {
    let new_count = new_ids.len();
    let pairs = head_dim as usize / 2;
    let ids_f32: Vec<f32> = new_ids.iter().map(|&id| id as f32).collect();
    let epsilon = alloc::vec![RMS_EPSILON; new_count];

    let mut cos = alloc::vec![0.0f32; new_count * pairs];
    let mut sin = alloc::vec![0.0f32; new_count * pairs];
    for offset in 0..new_count {
        let position = (start_position + offset) as f32;
        for pair in 0..pairs {
            let theta = position * rope_freq_base.powf(-((2 * pair) as f32) / (head_dim as f32));
            cos[offset * pairs + pair] = theta.cos();
            sin[offset * pairs + pair] = theta.sin();
        }
    }

    PositionInputs { ids_f32, epsilon, cos, sin }
}

/// The fully-supported [`ServingConfig`]: every knob [`apply_serving_config`]
/// accepts today, `F32` key/value cache storage (the only precision the
/// cached-attention reduce's shared `kv_heads` axis can cross -- see
/// `bind.rs`'s own `q8_0_quantized_key_value_cache_cannot_cross_the_weight_matmul_quantized_seam`
/// for the gap this sidesteps by construction rather than by luck).
fn supported_serving_config() -> ServingConfig<'static> {
    ServingConfig {
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

/// `ServingConfig::gpu_layers` (`-ngl`, `serving.rs`) is this crate's
/// existing GPU-offload knob, so backend selection reads it rather than a
/// second mechanism -- `0` (cpu-only, [`supported_serving_config`]'s own
/// default) selects [`Backend::Cpu`]; [`GPU_LAYERS_ALL`] (`-ngl all`)
/// selects [`Backend::Metal`]. `apply_serving_config` rejects every other
/// value before a forward ever runs, so those are the only two this match
/// needs to distinguish.
#[cfg(feature = "metal")]
fn select_backend(config: &ServingConfig) -> Backend {
    if config.gpu_layers == GPU_LAYERS_ALL { Backend::Metal } else { Backend::Cpu }
}

// `dequantize_unsupported_metal_weights`/`resolve_packed_block` (the
// per-call "convert this packed weight back to f32 because Metal has no
// unpack kernel for it yet" step) were deleted here: `Q4_K`/`Q5_K`/`Q6_K`
// -- every packed codec this checkpoint's weights actually carry -- now
// all stay packed straight to the GPU (`omega::msl::Q5K_UNPACK_MSL` is the
// row-blocked kernel `Q5_K` was still missing), so the conversion step had
// zero remaining callers. `named_blocks` below pushes `*block` directly,
// the same value this mechanism always resolved to once its lookup found
// nothing to convert.

/// Everything a decode step needs to actually run the program that is
/// backend-specific: which [`Backend`] to run it on, and the reusable state
/// each call to [`Self::evaluate`] persists across steps. Owns the plan
/// cache directly rather than through a trait object -- [`Backend`] is
/// already a closed, non-`dyn` enum (`omega::backend`'s own doc), and this
/// struct's whole job is picking one arm of it once per
/// [`LoadedModel::generate_with_serving_config`] call.
#[cfg(feature = "metal")]
pub(crate) struct BackendRuntime {
    backend: Backend,
    /// Keyed by `(new_count, cached_len)` -- the two symbols
    /// `mistral_cached_forward_program`'s cached-attention read extent
    /// resolves against (`Extent::Symbolic(1) == cached_len`). A [`Plan`]
    /// bakes concrete shapes from those symbols (`omega::backend::plan_named`'s
    /// own doc), and `cached_len` grows by `new_count` every decode step, so
    /// a plan built for one step's shape is never valid for the next --
    /// this cache exists for the shape that DOES repeat (a caller replaying
    /// the same partial length twice), not for ordinary autoregressive
    /// decode, which visits a strictly increasing `cached_len` and so never
    /// hits it within one call. See this crate's own task report for the
    /// measured hit/miss count on a real 24-token decode.
    plans: alloc::collections::BTreeMap<(usize, usize), Plan>,
    pub(crate) plan_hits: usize,
    pub(crate) plan_misses: usize,
}

#[cfg(feature = "metal")]
impl BackendRuntime {
    pub(crate) fn new(config: &ServingConfig) -> Self {
        Self {
            backend: select_backend(config),
            plans: alloc::collections::BTreeMap::new(),
            plan_hits: 0,
            plan_misses: 0,
        }
    }

    /// `resident_names` -- the caller's own model-weight names, fixed for
    /// the whole [`LoadedModel::generate_with_serving_config`] call -- is
    /// handed to [`mark_resident`] exactly once per distinct [`Plan`] (right
    /// after it is built, never on a cache hit, since a hit reuses the SAME
    /// `Plan` object that was already marked). See `omega::metal::Plan::mark_resident`'s
    /// own doc for why this needs a name set the tensor program itself
    /// cannot derive.
    fn evaluate(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
    ) -> Result<Evaluated, InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize);
        if self.plans.contains_key(&shape) {
            self.plan_hits += 1;
        } else {
            self.plan_misses += 1;
            let mut plan = plan_named(self.backend, program, symbols, named, outputs)?;
            mark_resident(&mut plan, resident_names);
            self.plans.insert(shape, plan);
        }
        let plan = self
            .plans
            .get_mut(&shape)
            .ok_or(InteropError::PlanCacheEntryVanished { shape })?;
        Ok(execute_plan_named(plan, named)?)
    }

    /// Diagnostic counterpart of [`Self::evaluate`]: same plan-cache lookup,
    /// but the Metal driver commits and waits on ONE command buffer PER
    /// `BoundOp` instead of once for the whole program, so each op's own
    /// GPU-only execution time comes back alongside the result -- see
    /// `omega::metal::execute_plan_op_timed`'s own doc for the cost this
    /// pays and why it must never replace [`Self::evaluate`] on the serving
    /// loop. Reachable only behind the `instrument` feature and only from
    /// this crate's own diagnostic call sites (`run_decode_loop`'s
    /// `PROXIMA_METAL_OP_PROFILE_STEP` branch).
    #[cfg(feature = "instrument")]
    fn evaluate_op_timed(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize);
        if self.plans.contains_key(&shape) {
            self.plan_hits += 1;
        } else {
            self.plan_misses += 1;
            let mut plan = plan_named(self.backend, program, symbols, named, outputs)?;
            mark_resident(&mut plan, resident_names);
            self.plans.insert(shape, plan);
        }
        let plan = self
            .plans
            .get_mut(&shape)
            .ok_or(InteropError::PlanCacheEntryVanished { shape })?;
        Ok(execute_plan_named_metal_op_timed(plan, named)?)
    }
}

/// The CPU-direct runtime a build without the `metal` feature keeps --
/// `omega` is not even a dependency in that build (`Cargo.toml`'s `metal`
/// feature is the only thing that turns `dep:omega` on), so this calls
/// [`evaluate_quantized_named_with_scratch`] exactly as
/// [`LoadedModel::generate`] always has. `free_buffers`/`validated_weight_nodes`
/// are the same scratch this loop's local variables used to own directly --
/// moved onto this struct so [`LoadedModel::generate_with_serving_config`]'s
/// loop body reads identically whether or not `metal` is compiled in.
#[cfg(not(feature = "metal"))]
pub(crate) struct BackendRuntime {
    free_buffers: Vec<Vec<f32>>,
    validated_weight_nodes: Option<BTreeSet<NodeId>>,
}

#[cfg(not(feature = "metal"))]
impl BackendRuntime {
    pub(crate) fn new(_config: &ServingConfig) -> Self {
        Self { free_buffers: Vec::new(), validated_weight_nodes: None }
    }

    /// `resident_names` is unused on this backend: the CPU evaluator has no
    /// device buffer to cache, so there is nothing to mark resident. Carried
    /// anyway so both `BackendRuntime::evaluate` impls share one signature
    /// and the decode loop's call site never needs a `cfg` of its own.
    fn evaluate(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        _resident_names: &BTreeSet<&str>,
    ) -> Result<Evaluated, InteropError> {
        Ok(evaluate_quantized_named_with_scratch(
            program,
            symbols,
            named,
            outputs,
            &mut self.free_buffers,
            &mut self.validated_weight_nodes,
        )?)
    }
}

impl<'file> Pipe for LoadedModel<'file> {
    type In = (String, usize);
    type Out = (Vec<u32>, String, bool);
    type Err = InteropError;

    fn call(&self, input: (String, usize)) -> impl Future<Output = Result<(Vec<u32>, String, bool), InteropError>> {
        async move {
            let (prompt, max_tokens) = input;
            self.generate(&prompt, max_tokens)
        }
    }
}

/// The decode loop's termination policy, isolated from the forward pass
/// that produces each token: pulls up to `max_tokens` ids out of
/// `produce_next_token` (one call per step, `0`-indexed), appending each
/// to the result unless it is `vocab`'s end-of-sequence id, in which case
/// decoding stops immediately without appending that id. Returns the
/// accumulated ids plus whether the stop was the model's own signal
/// (`true`) rather than the budget running out (`false`).
///
/// Factored out so this policy -- the exact defect this module's
/// [`LoadedModel::generate`] fixed (a loop with no termination condition
/// besides the budget) -- is provable against a scripted token source,
/// without paying for a real forward pass per test.
fn decode_until_stop_or_budget(
    vocab: &Vocab,
    max_tokens: usize,
    mut produce_next_token: impl FnMut(usize) -> Result<u32, InteropError>,
) -> Result<(Vec<u32>, bool), InteropError> {
    let mut generated_ids = Vec::with_capacity(max_tokens);
    let mut stopped_by_eos = false;
    for step in 0..max_tokens {
        let token_id = produce_next_token(step)?;
        if vocab.eos_token_id() == Some(token_id) {
            stopped_by_eos = true;
            break;
        }
        generated_ids.push(token_id);
    }
    Ok((generated_ids, stopped_by_eos))
}

impl<'file> LoadedModel<'file> {
    /// [`Self::generate_with_serving_config`] against
    /// [`supported_serving_config`] -- the reachable path every existing
    /// caller and test uses, unchanged: `gpu_layers: 0` always selects the
    /// CPU backend, so this runs exactly the forward it always has, on
    /// CPU, regardless of whether this build was compiled with the
    /// `metal` feature.
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<(Vec<u32>, String, bool), InteropError> {
        self.generate_with_serving_config(prompt, max_tokens, supported_serving_config())
    }

    /// The greedy decode loop itself: `max_tokens` steps, each one call
    /// into [`BackendRuntime::evaluate`] against `new_positions == 1` after
    /// the first step (`new_positions == prompt_length` on the first),
    /// growing [`LayerCache`] by one call's worth of positions every step
    /// instead of re-running the whole sequence from scratch -- stopping
    /// early the moment the model emits its own end-of-sequence id (see
    /// this module's doc for what that id is on the real checkpoint),
    /// never running past `max_tokens` regardless.
    ///
    /// `serving_config` is a caller-supplied override of
    /// [`supported_serving_config`]'s default -- the same [`ServingConfig`]
    /// [`apply_serving_config`] already gates, never a second selection
    /// mechanism. Setting `gpu_layers` to `GPU_LAYERS_ALL` (`-ngl all`) on
    /// a build compiled with this crate's `metal` feature runs this same
    /// loop against the Metal backend instead of the CPU one; every other
    /// field must already satisfy [`apply_serving_config`]'s gate the same
    /// way [`supported_serving_config`]'s does.
    pub fn generate_with_serving_config(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: ServingConfig,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let mut runtime = BackendRuntime::new(&serving_config);
        self.run_decode_loop(prompt, max_tokens, &serving_config, &mut runtime)
    }

    /// Shared by [`Self::generate_with_serving_config`] and this crate's
    /// own metal-path tests, which need to read `runtime`'s plan-cache
    /// hit/miss counters after the loop finishes -- a caller reachable
    /// only through the public method above never sees `runtime` at all.
    pub(crate) fn run_decode_loop(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let ids = proxima_tokenizer::encode_with_bos_eos(prompt, &self.vocab, true, false)?;
        // The repetition-penalty filter's own window: prompt tokens included,
        // matching upstream (`tools/main/main.cpp:725` feeds prompt tokens
        // through the same `common_sampler_accept` generated tokens use), grown
        // by one id every decode step below. `sample_config`/`rng` are built
        // once and threaded through every step -- the same seeded
        // `fastrand::Rng` this workspace already uses for every other
        // deterministic-by-seed pipe, drawn from progressively rather than
        // reseeded per token, mirroring upstream's own one-`std::mt19937`-per-
        // sampler-chain lifetime (`proxima_tokenizer::sample`'s own doc).
        let mut token_history: Vec<u32> = ids.clone();
        let repeat_window = serving_config.repeat_last_n.max(0) as usize;
        let sample_config = SamplingConfig {
            temperature: serving_config.temperature,
            top_k: serving_config.top_k,
            top_p: serving_config.top_p,
            min_p: serving_config.min_p,
            repeat_penalty: serving_config.repeat_penalty,
            frequency_penalty: serving_config.frequency_penalty,
            presence_penalty: serving_config.presence_penalty,
        };
        let mut rng = fastrand::Rng::with_seed(serving_config.seed);

        let block_count = self.architecture.block_count as usize;
        let kv_cache_names: Vec<(String, String, String)> = (0..block_count)
            .map(|layer| {
                (
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                )
            })
            .collect();
        let mut layer_caches: Vec<LayerCache> = (0..block_count).map(|_| LayerCache::new()).collect();

        // The caller's own knowledge of which named blocks are STATIC --
        // bound once in `LoadedModel::load` and never mutated again -- fixed
        // for this whole call, unlike `ids`/`eps`/`rope_cos`/`rope_sin` and
        // the KV cache's own blocks below, which change every step. This is
        // exactly the distinction `BackendRuntime::evaluate` hands to
        // `mark_resident` so the Metal driver's device-buffer cache can tell
        // "same name, same bytes" apart from "same name, new bytes" without
        // ever keying on name itself (`omega::metal::Plan::mark_resident`'s
        // own doc). Computed once, not per token: these names never change.
        let resident_names: BTreeSet<&str> = self
            .weights
            .owned
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(self.weights.packed.iter().map(|(name, _)| name.as_str()))
            .collect();

        let mut cached_len = 0usize;
        let mut next_ids = ids;
        let vocab_size = self.architecture.vocab as usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(&self.vocab, max_tokens, |_step| {
            #[cfg(feature = "instrument")]
            let step_started = read_ticks();

            let new_count = next_ids.len();
            #[cfg(feature = "instrument")]
            let apply_serving_config_started = read_ticks();
            apply_serving_config(serving_config, cached_len + new_count)?;
            #[cfg(feature = "instrument")]
            let apply_serving_config_ticks = elapsed_ticks(apply_serving_config_started);

            #[cfg(feature = "instrument")]
            let build_position_inputs_started = read_ticks();
            let inputs = build_position_inputs(
                &next_ids,
                cached_len,
                self.architecture.head_dim,
                self.architecture.rope_freq_base,
            );
            #[cfg(feature = "instrument")]
            let build_position_inputs_ticks = elapsed_ticks(build_position_inputs_started);

            let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
                self.weights.owned.len() + self.weights.packed.len() + 3 + layer_caches.len() * 3,
            );
            #[cfg(feature = "instrument")]
            let named_blocks_weights_started = read_ticks();
            named_blocks.push(("ids", QuantizedBlock::Float32(inputs.ids_f32.as_slice())));
            for (name, data) in &self.weights.owned {
                named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
            }
            for (name, block) in &self.weights.packed {
                named_blocks.push((name.as_str(), *block));
            }
            named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
            named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
            named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));
            #[cfg(feature = "instrument")]
            let named_blocks_weights_ticks = elapsed_ticks(named_blocks_weights_started);

            // KV-cache HOST -> DEVICE traffic: every named block below is the
            // FULL accumulated history (`LayerCache::append` only grows these,
            // never truncates), so this is the full `cached_len`-sized array
            // re-bound as a model input every single step -- not the
            // `new_count`-sized increment. Measured directly as element
            // counts read off the `Vec`s themselves (a size, not a timing),
            // so it is exact and needs no instrumentation to be turned on.
            #[cfg(feature = "instrument")]
            let kv_cache_upload_elements: u64 = layer_caches
                .iter()
                .map(|cache| (cache.k_even.len() + cache.k_odd.len() + cache.v.len()) as u64)
                .sum();
            #[cfg(feature = "instrument")]
            let named_blocks_kv_started = read_ticks();
            for (layer, (k_even_name, k_odd_name, v_name)) in kv_cache_names.iter().enumerate() {
                named_blocks.extend(layer_caches[layer].named_blocks(k_even_name, k_odd_name, v_name));
            }
            #[cfg(feature = "instrument")]
            let named_blocks_kv_ticks = elapsed_ticks(named_blocks_kv_started);

            let symbols = [new_count as u64, cached_len as u64];
            let mut roots: Vec<NodeId> = Vec::with_capacity(1 + self.cache_roots.len() * 3);
            roots.push(self.logits_root);
            for (even, odd, value) in &self.cache_roots {
                roots.push(*even);
                roots.push(*odd);
                roots.push(*value);
            }

            #[cfg(feature = "instrument")]
            let evaluate_started = read_ticks();
            // `PROXIMA_METAL_OP_PROFILE_STEP` -- diagnostic-only, `instrument`-gated,
            // default-off: unset in every production run, so `runtime.evaluate`
            // is the only path a caller without this env var ever takes. When
            // set to this step's own index, this ONE step instead runs
            // `evaluate_op_timed` (per-op command buffers, see that method's own
            // doc for the cost) and prints the per-op GPU attribution this
            // crate's own discipline log needed to settle the `gpu_exec`
            // investigation. Every other step, and every run without the env
            // var, is byte-for-byte the pre-existing path.
            #[cfg(all(feature = "instrument", feature = "metal"))]
            let evaluated = match std::env::var("PROXIMA_METAL_OP_PROFILE_STEP")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
            {
                Some(target) if target == _step => {
                    let (evaluated, timings) =
                        runtime.evaluate_op_timed(&self.program, &symbols, &named_blocks, &roots, &resident_names)?;
                    report_op_timings(_step, &timings);
                    evaluated
                }
                _ => runtime.evaluate(&self.program, &symbols, &named_blocks, &roots, &resident_names)?,
            };
            #[cfg(not(all(feature = "instrument", feature = "metal")))]
            let evaluated = runtime.evaluate(&self.program, &symbols, &named_blocks, &roots, &resident_names)?;
            #[cfg(feature = "instrument")]
            let evaluate_ticks = elapsed_ticks(evaluate_started);
            #[cfg(all(feature = "instrument", feature = "metal"))]
            let metal_stage = metal_stage_totals();

            // KV-cache DEVICE -> HOST readback + host append: unlike the
            // upload above, `evaluated.get(*even)` etc. is this step's own
            // `new_count`-sized OUTPUT increment (what the forward computed
            // for the newly-added positions), which `LayerCache::append`
            // then extends onto the growing history -- so this side is
            // expected to stay FLAT across tokens where the upload side
            // grows. `layer_cache_append` ticks/bytes below are the pure
            // host `extend_from_slice` memcpy cost, distinct from the GPU
            // readback `metal_stage_totals` already reports.
            #[cfg(feature = "instrument")]
            let layer_cache_append_started = read_ticks();
            #[cfg(feature = "instrument")]
            let mut layer_cache_append_elements: u64 = 0;
            for (layer, (even, odd, value)) in self.cache_roots.iter().enumerate() {
                let (even_data, _) = evaluated.get(*even).ok_or(InteropError::MissingEvaluatedNode { node: *even })?;
                let (odd_data, _) = evaluated.get(*odd).ok_or(InteropError::MissingEvaluatedNode { node: *odd })?;
                let (value_data, _) = evaluated.get(*value).ok_or(InteropError::MissingEvaluatedNode { node: *value })?;
                #[cfg(feature = "instrument")]
                {
                    layer_cache_append_elements += (even_data.len() + odd_data.len() + value_data.len()) as u64;
                }
                layer_caches[layer].append(even_data, odd_data, value_data);
            }
            #[cfg(feature = "instrument")]
            let layer_cache_append_ticks = elapsed_ticks(layer_cache_append_started);
            cached_len += new_count;

            let (logits, _shape) = evaluated
                .get(self.logits_root)
                .ok_or(InteropError::MissingEvaluatedNode { node: self.logits_root })?;
            let last_position = &logits[(new_count - 1) * vocab_size..new_count * vocab_size];

            #[cfg(feature = "instrument")]
            let greedy_pick_started = read_ticks();
            let recent_window_start = token_history.len().saturating_sub(repeat_window);
            let recent_tokens = &token_history[recent_window_start..];
            let token_id =
                sample_next_token(last_position, recent_tokens, sample_config, &mut rng).ok_or(InteropError::EmptyLogits)?;
            token_history.push(token_id);
            #[cfg(feature = "instrument")]
            let greedy_pick_ticks = elapsed_ticks(greedy_pick_started);
            next_ids = alloc::vec![token_id];

            #[cfg(feature = "instrument")]
            {
                let ms = |ticks: u64| ticks_to_nanos(ticks) as f64 / 1e6;
                std::println!(
                    "token_breakdown step={_step} new_count={new_count} cached_len_before={} \
                     step_wall_ms={:.3} apply_serving_config_ms={:.3} build_position_inputs_ms={:.3} \
                     named_blocks_weights_ms={:.3} named_blocks_kv_ms={:.3} kv_cache_upload_bytes={} \
                     evaluate_ms={:.3} layer_cache_append_ms={:.3} layer_cache_append_bytes={} \
                     greedy_pick_ms={:.3}",
                    cached_len,
                    ms(elapsed_ticks(step_started)),
                    ms(apply_serving_config_ticks),
                    ms(build_position_inputs_ticks),
                    ms(named_blocks_weights_ticks),
                    ms(named_blocks_kv_ticks),
                    kv_cache_upload_elements * 4,
                    ms(evaluate_ticks),
                    ms(layer_cache_append_ticks),
                    layer_cache_append_elements * 4,
                    ms(greedy_pick_ticks),
                );
                #[cfg(feature = "metal")]
                std::println!(
                    "token_breakdown_metal step={_step} prepare_calls={} prepare_ms={:.3} \
                     emit_calls={} emit_ms={:.3} pipeline_hits={} pipeline_misses={} pipeline_compile_ms={:.3} \
                     block_upload_calls={} block_upload_ms={:.3} block_upload_bytes={} \
                     op_setup_calls={} op_setup_ms={:.3} \
                     pipeline_lookup_calls={} pipeline_lookup_ms={:.3} \
                     encode_dispatch_calls={} encode_dispatch_ms={:.3} \
                     gpu_exec_calls={} gpu_exec_ms={:.3} \
                     readback_calls={} readback_ms={:.3} readback_bytes={} \
                     nocopy_uploads={} copying_uploads={} nocopy_reuses={} \
                     resident_uploads={} resident_reuses={}",
                    metal_stage.prepare_calls,
                    ms(metal_stage.prepare_ticks),
                    metal_stage.emit_calls,
                    ms(metal_stage.emit_ticks),
                    metal_stage.pipeline_hits,
                    metal_stage.pipeline_misses,
                    ms(metal_stage.pipeline_compile_ticks),
                    metal_stage.block_upload_calls,
                    ms(metal_stage.block_upload_ticks),
                    metal_stage.block_upload_bytes,
                    metal_stage.op_setup_calls,
                    ms(metal_stage.op_setup_ticks),
                    metal_stage.pipeline_lookup_calls,
                    ms(metal_stage.pipeline_lookup_ticks),
                    metal_stage.encode_dispatch_calls,
                    ms(metal_stage.encode_dispatch_ticks),
                    metal_stage.gpu_exec_calls,
                    ms(metal_stage.gpu_exec_ticks),
                    metal_stage.readback_calls,
                    ms(metal_stage.readback_ticks),
                    metal_stage.readback_bytes,
                    metal_stage.nocopy_uploads,
                    metal_stage.copying_uploads,
                    metal_stage.nocopy_reuses,
                    metal_stage.resident_uploads,
                    metal_stage.resident_reuses,
                );
            }

            Ok(token_id)
        })?;

        let text = proxima_tokenizer::decode(&generated_ids, &self.vocab)?;
        Ok((generated_ids, text, stopped_by_eos))
    }
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::String;
    use alloc::vec::Vec;

    use proxima_tokenizer::Vocab;

    use super::decode_until_stop_or_budget;

    /// A minimal valid [`Vocab`] (every byte-level BPE vocab needs all 256
    /// base-byte tokens present or [`Vocab::new`] rejects it) plus one
    /// extra token at id `256` marked as this vocab's end-of-sequence id --
    /// enough to exercise [`decode_until_stop_or_budget`]'s stopping policy
    /// without a real checkpoint. Spells the base-byte alphabet as the
    /// SentencePiece `"<0xXX>"` fallback form directly (not through
    /// `proxima_tokenizer`'s private `byte_to_char`) since that spelling is
    /// public knowledge, not an internal detail this test needs to reach
    /// into the crate for.
    fn vocab_with_eos(eos_id: u32) -> Vocab {
        let mut tokens: Vec<String> = (0..=255u8).map(|byte| alloc::format!("<0x{byte:02X}>")).collect();
        tokens.push(String::from("<eos-marker>"));
        Vocab::new(tokens, &[], Some(0), Some(eos_id), None).expect("minimal vocab builds")
    }

    /// The defect this module exists to fix, proved directly: a scripted
    /// token source that would emit `999` on a 4th call never gets asked
    /// for it, because the 3rd call's token (`32000`, this vocab's eos id)
    /// stops the loop first. Also proves the eos id itself never lands in
    /// `generated_ids`.
    #[test]
    fn stops_early_when_eos_is_produced_and_excludes_it_from_ids() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [10u32, 20, 32_000, 999];
        let mut calls = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(&vocab, 4, |step| {
            calls += 1;
            Ok(scripted_tokens[step])
        })
        .expect("scripted token source never errors");

        assert_eq!(generated_ids, alloc::vec![10, 20], "eos id must not be appended to the generated ids");
        assert!(stopped_by_eos, "must report that the stop was the model's own eos signal");
        assert_eq!(calls, 3, "must not pull a 4th token once eos is seen on the 3rd");
    }

    /// The other half of the invariant: when the model never emits eos,
    /// decoding runs the full budget and reports that distinctly from an
    /// eos stop -- `stopped_by_eos == false` is the caller's only way to
    /// tell "ran out of budget" apart from "the model finished".
    #[test]
    fn exhausts_the_budget_and_reports_it_distinctly_from_an_eos_stop() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [10u32, 20, 30, 40];

        let (generated_ids, stopped_by_eos) =
            decode_until_stop_or_budget(&vocab, scripted_tokens.len(), |step| Ok(scripted_tokens[step]))
                .expect("scripted token source never errors");

        assert_eq!(generated_ids, alloc::vec![10, 20, 30, 40], "every scripted token is a real id, none is eos");
        assert!(!stopped_by_eos, "budget exhaustion must not be reported as an eos stop");
        assert_eq!(generated_ids.len(), scripted_tokens.len(), "budget exhaustion still runs every requested step");
    }

    /// Degenerate control: if the eos comparison were broken (e.g. always
    /// `false`), this test's scripted eos-first source would run the full
    /// budget instead of stopping on step 1 -- confirming the two tests
    /// above are not passing by coincidence of never actually comparing
    /// against `vocab.eos_token_id()`.
    #[test]
    fn stops_on_the_very_first_token_when_it_is_eos() {
        let vocab = vocab_with_eos(32_000);
        let mut calls = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(&vocab, 10, |_step| {
            calls += 1;
            Ok(32_000)
        })
        .expect("scripted token source never errors");

        assert!(generated_ids.is_empty(), "an immediate eos must produce zero generated ids");
        assert!(stopped_by_eos);
        assert_eq!(calls, 1, "must stop after exactly one call, not run toward the budget of 10");
    }
}
