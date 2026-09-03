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
#[cfg(not(feature = "metal"))]
use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::cpu::{Evaluated, QuantizedBlock};
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::{Qwen35LayerRoots, mistral_cached_forward_program_with_experts};
use proxima_tokenizer::{SamplingConfig, Vocab, sample_next_token};

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use omega::backend::execute_plan_named_metal_op_timed;
#[cfg(feature = "metal")]
use omega::backend::{Backend, Plan, execute_plan_named, mark_resident, plan_named};
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use omega::metal::OpGpuTiming;
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use omega::metal::metal_stage_totals;
#[cfg(feature = "instrument")]
use proxima_telemetry::debug;
#[cfg(feature = "instrument")]
use proxima_tensor::instrument::{elapsed_ticks, read_ticks, ticks_to_nanos};

use crate::bind::{BoundWeights, ModelArchitecture, architecture_from_metadata, bind_all_weights};
use crate::error::InteropError;
use crate::hf_bind::bind_all_weights_from_safetensors;
#[cfg(feature = "metal")]
use crate::serving::GPU_LAYERS_ALL;
use crate::serving::ServingConfig;
use crate::serving::apply_serving_config;

const RMS_EPSILON: f32 = 1e-5;

/// How many of [`OpGpuTiming`]'s entries [`report_op_timings`] names
/// individually -- the discipline log's own "top 20 ops by GPU time" ask.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
const OP_PROFILE_TOP_N: usize = 20;

/// Prints the per-op GPU attribution `run_decode_loop`'s
/// `PROXIMA_METAL_OP_PROFILE_STEP` branch gathers for exactly one decode
/// step: the op count and summed GPU time (asserting the count so a
/// degenerate empty profile reads as RED, not quiet), one line per
/// `OpGpuTiming::kind` bucket, and the top [`OP_PROFILE_TOP_N`] ops by GPU
/// time with their operand bytes and bytes/ns -- exactly what settles
/// whether GPU time tracks operand bytes or is flat per dispatch.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn report_op_timings(step: usize, timings: &[OpGpuTiming]) {
    let op_count = timings.len();
    let total_gpu_ns: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
    let total_operand_bytes: u64 = timings.iter().map(|timing| timing.operand_bytes).sum();

    std::println!(
        "op_profile step={step} op_count={op_count} total_gpu_ns={total_gpu_ns} \
         total_gpu_ms={:.3} total_operand_bytes={total_operand_bytes}",
        total_gpu_ns as f64 / 1e6,
    );

    let mut by_kind: alloc::collections::BTreeMap<&'static str, (u64, u64, u64)> =
        alloc::collections::BTreeMap::new();
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
    let mut by_family: alloc::collections::BTreeMap<String, FamilyGpuStats> =
        alloc::collections::BTreeMap::new();
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
            if stats.operand_bytes == 0 {
                0.0
            } else {
                stats.gpu_ns as f64 / stats.operand_bytes as f64
            },
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

/// `TASK_VM_INFO`'s `phys_footprint` field (`mach/task_info.h`) -- macOS's
/// own accounting of this process's compressed+resident memory charge, the
/// same number Activity Monitor's "Memory" column and `footprint(1)` read.
/// Read directly via `task_info` rather than `/usr/bin/time`'s whole-process
/// peak RSS because this is sampled PER DECODE STEP, from inside the
/// process, so growth can be attributed to a step boundary instead of only
/// a start/end delta. Declares its own `task_info`/`mach_task_self` FFI
/// rather than pulling in `mach2`/`mach` (neither is otherwise in this
/// workspace's dependency graph) for a struct that is a stable, versioned,
/// public part of the mach ABI (rev1, `TASK_VM_INFO_REV1_COUNT`) -- the
/// struct here mirrors the header exactly up to (and including)
/// `phys_footprint` and stops there, matching REV1's word count so the
/// kernel fills exactly the fields declared.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn phys_footprint_bytes() -> u64 {
    #[repr(C)]
    #[derive(Default)]
    struct TaskVmInfo {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
    }

    const TASK_VM_INFO: i32 = 22;

    unsafe extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: i32,
            task_info_out: *mut TaskVmInfo,
            task_info_out_count: *mut u32,
        ) -> i32;
    }

    let mut info = TaskVmInfo::default();
    let mut count = (core::mem::size_of::<TaskVmInfo>() / core::mem::size_of::<u32>()) as u32;
    let result = unsafe { task_info(mach_task_self(), TASK_VM_INFO, &mut info, &mut count) };
    if result != 0 {
        return 0;
    }
    info.phys_footprint
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn stats_pass(entry: &mut FamilyGpuStats, gpu_ns: u64, operand_bytes: u64) {
    entry.row_blocked_count += 1;
    entry.passed_gpu_ns += gpu_ns;
    entry.passed_operand_bytes += operand_bytes;
    entry.packed_row_block_gates.insert("PASS".to_string());
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
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
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
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

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
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
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
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
    /// One entry per forward-program layer, in layer order --
    /// [`Qwen35LayerRoots::Attention`] for every layer on the dense path
    /// (`Self::load`/`Self::load_from_safetensors` wrap
    /// `mistral_cached_forward_program_with_experts`'s own
    /// [`CachedLayerRoots`] in that variant so both checkpoint families
    /// share one cache-threading loop, [`Self::run_decode_loop`]), and a mix
    /// of [`Qwen35LayerRoots::Attention`]/[`Qwen35LayerRoots::Ssm`] on the
    /// qwen35 path (`crate::qwen35::qwen35_forward_program`'s own return).
    layer_roots: Vec<Qwen35LayerRoots>,
    /// [`Some`] only for a qwen35-architecture checkpoint -- the SSM cache
    /// shapes [`SsmLayerCache::new`] needs (`Self::run_decode_loop`'s own
    /// per-layer state-space cache), derived once at load time rather than
    /// recomputed every decode step. `None` on the dense path, which never
    /// has an [`Qwen35LayerRoots::Ssm`] entry to size.
    qwen35_ssm_shape: Option<Qwen35SsmShape>,
}

/// [`SsmLayerCache`]'s own fixed sizes, all derived from
/// [`crate::qwen35::Qwen35Architecture`]'s ssm hyperparameters at load time
/// -- `qwen35.cpp:57-60`'s same derivation
/// `crate::qwen35::bind_qwen35_attn_qkv_split`'s own doc already walks
/// through for the fused `attn_qkv.weight` split.
#[derive(Debug, Clone, Copy)]
struct Qwen35SsmShape {
    /// `2 * ssm_key_dim + ssm_d_inner` -- one `qkv_mixed` row's width,
    /// matching `proxima_tensor::spec::qwen35_forward_program`'s own
    /// `ssm_cache.{layer}.conv_history` leaf shape's second axis.
    qkv_dim: usize,
    /// `ssm_d_conv - 1` -- the rolling conv-history window's fixed row
    /// count [`append_qwen35_ssm_mixer`]'s doc names (the causal conv1d
    /// kernel's own left-context width).
    conv_rows: usize,
    /// `ssm_d_state * head_v_dim * ssm_n_group * ssm_group` -- the gated
    /// DeltaNet recurrent state's flat element count, matching
    /// `qwen35_forward_program`'s own `ssm_cache.{layer}.state` leaf shape.
    state_len: usize,
}

impl<'file> LoadedModel<'file> {
    /// Binds every weight the cached forward program needs out of
    /// `parsed`/`file_bytes` (`crate::bind::bind_all_weights`), derives
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
        // registers `file_bytes` -- the checkpoint's own mmap, page-aligned
        // at its base by construction -- as the single mapping every packed
        // tensor's borrowed slice can be addressed into by OFFSET instead of
        // copied into its own device buffer; see
        // `omega::metal::register_checkpoint_mapping`'s own doc. A no-op
        // when the Metal backend is not compiled in.
        omega::backend::register_checkpoint_mapping(file_bytes);
        // `general.architecture` read directly, before `architecture_from_metadata`
        // (which assumes the dense per-layer shape every other checkpoint this
        // crate binds has) -- qwen35's hybrid attention+state-space layers
        // (`crate::qwen35`'s own module doc) are not that shape, so this
        // checkpoint gets its own bind + forward-program seam instead of being
        // handed to the dense path, which would either fail bind on an SSM
        // layer's tensors or, worse, silently misbind them as dense attention.
        if crate::bind::metadata_str(parsed, "general.architecture")? == "qwen35" {
            let qwen_architecture = crate::qwen35::qwen35_architecture_from_metadata(parsed)?;
            let weights = crate::qwen35::bind_qwen35_weights(parsed, file_bytes, &qwen_architecture)?;
            let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)?;
            let (program, logits_root, layer_roots) =
                crate::qwen35::qwen35_forward_program(&qwen_architecture)?;
            let ssm_shape = qwen35_ssm_shape(&qwen_architecture);
            let architecture = ModelArchitecture {
                vocab: qwen_architecture.vocab,
                embedding: qwen_architecture.embedding,
                feed_forward: qwen_architecture.feed_forward,
                query_heads: qwen_architecture.query_heads,
                kv_heads: qwen_architecture.kv_heads,
                head_dim: qwen_architecture.head_dim,
                block_count: qwen_architecture.block_count,
                // Qwen3.5 never routes FFN through experts
                // (`crate::qwen35::qwen35_forward_program`'s own doc,
                // `qwen35.cpp:471`), so this checkpoint reads the same
                // `expert_count == 0` dense-FFN branch every other checkpoint
                // without a `{architecture}.expert_count` key does.
                expert_count: 0,
                expert_used_count: 0,
                rope_freq_base: qwen_architecture.rope_freq_base,
                tied_embeddings: false,
            };
            return Ok(Self {
                weights,
                architecture,
                vocab,
                program,
                logits_root,
                layer_roots,
                qwen35_ssm_shape: Some(ssm_shape),
            });
        }

        let architecture = architecture_from_metadata(parsed)?;
        let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)?;
        let weights = bind_all_weights(parsed, file_bytes, &architecture)?;
        // `architecture.expert_count`/`expert_used_count` read `0` for every
        // dense checkpoint (`ModelArchitecture`'s own doc), which selects
        // exactly the dense program this crate has always built -- a
        // mixture-of-experts checkpoint (`expert_count > 0`) is the only case
        // that changes which program gets compiled here. `qk_norm` is Qwen3's
        // own per-head QK-norm (`crate::bind::checkpoint_has_qk_norm`'s own
        // doc) -- `false` reproduces the identical program this call has
        // always compiled for a checkpoint that carries no
        // `attn_q_norm.weight` tensor.
        let qk_norm = crate::bind::checkpoint_has_qk_norm(parsed);
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
            qk_norm,
        )?;
        Ok(Self {
            weights,
            architecture,
            vocab,
            program,
            logits_root,
            layer_roots: cache_roots.into_iter().map(Qwen35LayerRoots::Attention).collect(),
            qwen35_ssm_shape: None,
        })
    }

    /// [`Self::load`]'s HF/safetensors counterpart: binds every weight out
    /// of a single safetensors buffer's [`proxima_safetensors::Manifest`]
    /// (`crate::hf_bind::bind_all_weights_from_safetensors`) instead of a
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
    /// `crate::hf_bind::bind_all_weights_from_safetensors`'s doc); a
    /// caller who just parsed `file_bytes`'s header into `manifest` already
    /// has this value.
    ///
    /// # Errors
    ///
    /// [`InteropError::HfMoeWeightsUnsupported`] if `architecture.expert_count`
    /// is nonzero; otherwise whatever
    /// `crate::hf_bind::bind_all_weights_from_safetensors` or
    /// [`mistral_cached_forward_program_with_experts`] can fail with.
    pub fn load_from_safetensors(
        manifest: &proxima_safetensors::Manifest,
        file_bytes: &'file [u8],
        data_start: u64,
        architecture: ModelArchitecture,
        vocab: Vocab,
    ) -> Result<Self, InteropError> {
        let weights =
            bind_all_weights_from_safetensors(manifest, file_bytes, data_start, &architecture)?;
        // safetensors carries no GGUF tensor directory to probe for
        // `attn_q_norm.weight`, and no HF/safetensors checkpoint this crate
        // binds today needs QK-norm -- see [`Self::load`]'s own `qk_norm` for
        // the GGUF path that does.
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
            false,
        )?;
        Ok(Self {
            weights,
            architecture,
            vocab,
            program,
            logits_root,
            layer_roots: cache_roots.into_iter().map(Qwen35LayerRoots::Attention).collect(),
            qwen35_ssm_shape: None,
        })
    }
}

/// [`Qwen35SsmShape`]'s own derivation off a real checkpoint's ssm
/// hyperparameters -- `qwen35.cpp:57-60`'s same arithmetic
/// `crate::qwen35::Qwen35Architecture::ssm_key_dim`/`ssm_value_dim` already
/// use for the fused `attn_qkv.weight` row split, plus `head_v_dim =
/// ssm_inner_size / ssm_time_step_rank` and `ssm_group = ssm_time_step_rank
/// / ssm_group_count` (`proxima_tensor::spec::qwen35_forward_program`'s own
/// `head_v_dim`/`ssm_group` locals).
fn qwen35_ssm_shape(architecture: &crate::qwen35::Qwen35Architecture) -> Qwen35SsmShape {
    let ssm_key_dim = architecture.ssm_state_size * architecture.ssm_group_count;
    let head_v_dim = architecture.ssm_inner_size / architecture.ssm_time_step_rank;
    let ssm_group = architecture.ssm_time_step_rank / architecture.ssm_group_count;
    Qwen35SsmShape {
        qkv_dim: (2 * ssm_key_dim + architecture.ssm_inner_size) as usize,
        conv_rows: (architecture.ssm_conv_kernel.saturating_sub(1)) as usize,
        state_len: (architecture.ssm_state_size
            * head_v_dim
            * architecture.ssm_group_count
            * ssm_group) as usize,
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
        Self {
            k_even: Vec::new(),
            k_odd: Vec::new(),
            v: Vec::new(),
        }
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

/// [`LayerCache`]'s 4-wide counterpart for a
/// [`Qwen35LayerRoots::DenseAttention`] layer -- this checkpoint's own
/// partial-rotary gap (`proxima_tensor::spec::append_qwen35_dense_attention_layer`'s
/// own doc) needs a third K component (`k_pass`, the untouched
/// `rotary_dim..attn_head_dim` remainder) alongside the rotated
/// `k_first`/`k_second` halves [`LayerCache`]'s `k_even`/`k_odd` already
/// name for the plain single-section-RoPE checkpoints.
struct Qwen35DenseAttentionCache {
    k_first: Vec<f32>,
    k_second: Vec<f32>,
    k_pass: Vec<f32>,
    v: Vec<f32>,
}

impl Qwen35DenseAttentionCache {
    fn new() -> Self {
        Self {
            k_first: Vec::new(),
            k_second: Vec::new(),
            k_pass: Vec::new(),
            v: Vec::new(),
        }
    }

    fn append(&mut self, first: &[f32], second: &[f32], pass: &[f32], value: &[f32]) {
        self.k_first.extend_from_slice(first);
        self.k_second.extend_from_slice(second);
        self.k_pass.extend_from_slice(pass);
        self.v.extend_from_slice(value);
    }

    fn named_blocks<'cache>(
        &'cache self,
        k_first_name: &'cache str,
        k_second_name: &'cache str,
        k_pass_name: &'cache str,
        v_name: &'cache str,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 4] {
        [
            (k_first_name, QuantizedBlock::Float32(self.k_first.as_slice())),
            (k_second_name, QuantizedBlock::Float32(self.k_second.as_slice())),
            (k_pass_name, QuantizedBlock::Float32(self.k_pass.as_slice())),
            (v_name, QuantizedBlock::Float32(self.v.as_slice())),
        ]
    }
}

/// [`LayerCache`]'s counterpart for a [`Qwen35LayerRoots::Ssm`] layer --
/// `conv_history` is a fixed-size rolling window (the causal conv1d
/// kernel's own left context, `Qwen35SsmShape::conv_rows` rows of
/// `Qwen35SsmShape::qkv_dim` elements each, oldest row dropped as each new
/// one is appended) rather than [`LayerCache`]'s unbounded grow-forever
/// history; `state` is the gated DeltaNet recurrent state, fully replaced
/// every step (never appended to) because the mixer already folds every
/// past position into it.
struct SsmLayerCache {
    conv_history: Vec<f32>,
    state: Vec<f32>,
}

impl SsmLayerCache {
    fn new(shape: Qwen35SsmShape) -> Self {
        Self {
            conv_history: alloc::vec![0.0f32; shape.conv_rows * shape.qkv_dim],
            state: alloc::vec![0.0f32; shape.state_len],
        }
    }

    /// `qkv_mixed_new` is this step's own `new_count`-many freshly computed
    /// `qkv_mixed` rows (`shape.qkv_dim` elements each); `state_new` is the
    /// mixer's full replacement state. Keeps only `shape.conv_rows`' worth
    /// of the most recent `qkv_mixed` rows -- older rows fall out of the
    /// causal conv1d kernel's left context and are never read again.
    fn advance(&mut self, qkv_mixed_new: &[f32], state_new: &[f32], shape: Qwen35SsmShape) {
        self.conv_history.extend_from_slice(qkv_mixed_new);
        let keep = shape.conv_rows * shape.qkv_dim;
        let drop = self.conv_history.len().saturating_sub(keep);
        self.conv_history.drain(0..drop);
        self.state.clear();
        self.state.extend_from_slice(state_new);
    }

    fn named_blocks<'cache>(
        &'cache self,
        conv_history_name: &'cache str,
        state_name: &'cache str,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 2] {
        [
            (
                conv_history_name,
                QuantizedBlock::Float32(self.conv_history.as_slice()),
            ),
            (state_name, QuantizedBlock::Float32(self.state.as_slice())),
        ]
    }
}

/// [`LayerCache::new`]/[`SsmLayerCache::new`] threaded per forward-program
/// layer, matching [`LoadedModel::layer_roots`]'s own per-layer discriminant
/// -- an attention layer's cache append/readback shape genuinely differs
/// from an ssm layer's, the same reason [`Qwen35LayerRoots`] itself is an
/// enum rather than a fixed-shape tuple.
enum LayerCacheState {
    Attention(LayerCache),
    DenseAttention(Qwen35DenseAttentionCache),
    Ssm(SsmLayerCache),
}

/// This call's own [`Op::Input`] names for one layer's cache, matching
/// [`LayerCacheState`]'s discriminant one-to-one -- built once per
/// [`LoadedModel::run_decode_loop`] call (never per step) since layer kind
/// and layer index never change within a call.
enum LayerCacheNames {
    Attention {
        k_even: String,
        k_odd: String,
        v: String,
    },
    DenseAttention {
        k_first: String,
        k_second: String,
        k_pass: String,
        v: String,
    },
    Ssm {
        conv_history: String,
        state: String,
    },
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

    PositionInputs {
        ids_f32,
        epsilon,
        cos,
        sin,
    }
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
    if config.gpu_layers == GPU_LAYERS_ALL {
        Backend::Metal
    } else {
        Backend::Cpu
    }
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
    /// hits it within one call.
    ///
    /// [`Self::resolve_plan`] clears this on every miss instead of
    /// accumulating entries: measured on a real decode (`plan_cache_len` /
    /// `plan_misses` in `token_breakdown_metal`) this map grew 1:1 with the
    /// step index and `plan_hits` never left 0, so every step but the first
    /// was retaining a `Plan` that could never be looked up again for the
    /// rest of the call -- a Rust-heap leak (`phys_footprint_bytes` climbed
    /// while `omega::metal::current_allocated_size()` stayed flat over the
    /// same steps, proving the growth was not GPU-side). Clearing on miss
    /// keeps exactly the one entry the field's own rationale above says is
    /// worth keeping.
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
        let shape = self.resolve_plan(program, symbols, named, outputs, resident_names)?;
        let plan = self
            .plans
            .get_mut(&shape)
            .ok_or(InteropError::PlanCacheEntryVanished { shape })?;
        Ok(execute_plan_named(plan, named)?)
    }

    /// [`Self::evaluate`]/[`Self::evaluate_op_timed`]'s shared cache-lookup
    /// step, split out so the eviction policy lives in exactly one place.
    ///
    /// This struct's own [`Self::plans`] field comment already proved
    /// ordinary autoregressive decode's `cached_len` strictly increases, so
    /// a `Plan` keyed on it is NEVER looked up again once superseded --
    /// measured directly: `plan_cache_len`/`plan_misses` both grew 1:1 with
    /// the step index (`plan_hits` stayed 0) across a real decode, while
    /// `phys_footprint_bytes` climbed and `omega::metal::current_allocated_size()`
    /// stayed flat over the same steps -- the growth is a Rust-heap leak of
    /// superseded `Plan`s, not a Metal-driver allocation. Clearing the map on
    /// every miss keeps the one entry the doc's own rationale says is worth
    /// keeping (an immediate same-shape replay lands as a hit BEFORE the
    /// next miss would evict it) while making superseded entries collectible
    /// instead of retained for the rest of the call.
    fn resolve_plan(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
    ) -> Result<(usize, usize), InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize);
        if self.plans.contains_key(&shape) {
            self.plan_hits += 1;
        } else {
            self.plan_misses += 1;
            let mut plan = plan_named(self.backend, program, symbols, named, outputs)?;
            mark_resident(&mut plan, resident_names);
            self.plans.clear();
            self.plans.insert(shape, plan);
        }
        Ok(shape)
    }

    /// Live entry count in [`Self::plans`] -- the direct witness that
    /// [`Self::resolve_plan`]'s clear-on-miss policy keeps this bounded at 1
    /// through ordinary autoregressive decode's strictly increasing
    /// `cached_len`, rather than growing 1:1 with the step index as it did
    /// before that policy landed. See `token_breakdown_metal`'s
    /// `plan_cache_len` field.
    #[cfg(feature = "instrument")]
    pub(crate) fn plans_len(&self) -> usize {
        self.plans.len()
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
    #[cfg(all(feature = "instrument", target_os = "macos"))]
    fn evaluate_op_timed(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let shape = self.resolve_plan(program, symbols, named, outputs, resident_names)?;
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
        Self {
            free_buffers: Vec::new(),
            validated_weight_nodes: None,
        }
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

    fn call(
        &self,
        input: (String, usize),
    ) -> impl Future<Output = Result<(Vec<u32>, String, bool), InteropError>> {
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

/// [`Vocab::add_bos_token`]'s own fallback when the checkpoint's metadata
/// carries no `tokenizer.ggml.add_bos_token` opinion at all: default to
/// requesting BOS only when the vocab actually HAS a
/// [`Vocab::bos_token_id`] to add. Every dense checkpoint this crate has
/// bound so far declares one (openchat-3.5, SmolLM2), so this reproduces
/// this crate's pre-existing unconditional `true` default for them
/// byte-for-byte; the real Qwen3.5 checkpoint declares neither the policy
/// key nor a `tokenizer.ggml.bos_token_id` key at all (confirmed via
/// `strings` on the real file -- Qwen's own tokenizer has no BOS token,
/// chat turns open on `<|im_start|>` instead), so defaulting to `true`
/// there would ask [`proxima_tokenizer::encode_with_bos_eos`] to prepend an
/// id that does not exist, surfacing
/// [`proxima_tokenizer::TokenizerError::MissingMetadataKey`] on every
/// prompt rather than the tokenizer's own real, silent policy.
fn wants_bos(vocab: &Vocab) -> bool {
    vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some())
}

impl<'file> LoadedModel<'file> {
    /// [`Self::generate_with_serving_config`] against
    /// [`supported_serving_config`] -- the reachable path every existing
    /// caller and test uses, unchanged: `gpu_layers: 0` always selects the
    /// CPU backend, so this runs exactly the forward it always has, on
    /// CPU, regardless of whether this build was compiled with the
    /// `metal` feature.
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        self.generate_with_serving_config(prompt, max_tokens, supported_serving_config())
    }

    /// The greedy decode loop itself: `max_tokens` steps, each one call
    /// into `BackendRuntime::evaluate` against `new_positions == 1` after
    /// the first step (`new_positions == prompt_length` on the first),
    /// growing `LayerCache` by one call's worth of positions every step
    /// instead of re-running the whole sequence from scratch -- stopping
    /// early the moment the model emits its own end-of-sequence id (see
    /// this module's doc for what that id is on the real checkpoint),
    /// never running past `max_tokens` regardless.
    ///
    /// `serving_config` is a caller-supplied override of
    /// `supported_serving_config`'s default -- the same [`ServingConfig`]
    /// [`apply_serving_config`] already gates, never a second selection
    /// mechanism. Setting `gpu_layers` to `GPU_LAYERS_ALL` (`-ngl all`) on
    /// a build compiled with this crate's `metal` feature runs this same
    /// loop against the Metal backend instead of the CPU one; every other
    /// field must already satisfy [`apply_serving_config`]'s gate the same
    /// way `supported_serving_config`'s does.
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
        let ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
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

        let cache_names: Vec<LayerCacheNames> = self
            .layer_roots
            .iter()
            .enumerate()
            .map(|(layer, roots)| match roots {
                Qwen35LayerRoots::Attention(_) => LayerCacheNames::Attention {
                    k_even: alloc::format!("kv_cache.{layer}.k_even"),
                    k_odd: alloc::format!("kv_cache.{layer}.k_odd"),
                    v: alloc::format!("kv_cache.{layer}.v"),
                },
                Qwen35LayerRoots::DenseAttention(_) => LayerCacheNames::DenseAttention {
                    k_first: alloc::format!("kv_cache.{layer}.k_first"),
                    k_second: alloc::format!("kv_cache.{layer}.k_second"),
                    k_pass: alloc::format!("kv_cache.{layer}.k_pass"),
                    v: alloc::format!("kv_cache.{layer}.v"),
                },
                Qwen35LayerRoots::Ssm { .. } => LayerCacheNames::Ssm {
                    conv_history: alloc::format!("ssm_cache.{layer}.conv_history"),
                    state: alloc::format!("ssm_cache.{layer}.state"),
                },
            })
            .collect();
        let mut layer_caches: Vec<LayerCacheState> = self
            .layer_roots
            .iter()
            .map(|roots| match roots {
                Qwen35LayerRoots::Attention(_) => LayerCacheState::Attention(LayerCache::new()),
                Qwen35LayerRoots::DenseAttention(_) => {
                    LayerCacheState::DenseAttention(Qwen35DenseAttentionCache::new())
                }
                Qwen35LayerRoots::Ssm { .. } => LayerCacheState::Ssm(SsmLayerCache::new(
                    self.qwen35_ssm_shape.unwrap_or(Qwen35SsmShape {
                        qkv_dim: 0,
                        conv_rows: 0,
                        state_len: 0,
                    }),
                )),
            })
            .collect();

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
            .chain(
                self.weights
                    .packed_owned
                    .iter()
                    .map(|(name, _, _)| name.as_str()),
            )
            .collect();

        let mut cached_len = 0usize;
        let mut next_ids = ids;
        let vocab_size = self.architecture.vocab as usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &self.vocab,
            max_tokens,
            |_step| {
                // ROW 130's own fix, built: every counter this step's
                // `evaluate_ms` decomposition reads is zeroed HERE, at step
                // start, and read back after `evaluate_ticks` below is computed
                // -- a single step's own cost, measured directly inside one
                // process, never inferred by differencing two independent
                // launches' cumulative-since-start counters (that differencing
                // is exact for the integer counts ROW 129 used it for, and NOT
                // for timings -- ROW 130's own postmortem on why it produced a
                // sub-bucket larger than its parent and a negative duration).
                #[cfg(feature = "instrument")]
                proxima_tensor::instrument::reset_step();
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
                    self.weights.owned.len()
                        + self.weights.packed.len()
                        + self.weights.packed_owned.len()
                        + 3
                        + layer_caches.len() * 3,
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
                for (name, bytes, kind) in &self.weights.packed_owned {
                    named_blocks.push((name.as_str(), kind.as_block(bytes)));
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
                    .map(|cache| match cache {
                        LayerCacheState::Attention(cache) => {
                            (cache.k_even.len() + cache.k_odd.len() + cache.v.len()) as u64
                        }
                        LayerCacheState::DenseAttention(cache) => {
                            (cache.k_first.len()
                                + cache.k_second.len()
                                + cache.k_pass.len()
                                + cache.v.len()) as u64
                        }
                        LayerCacheState::Ssm(cache) => {
                            (cache.conv_history.len() + cache.state.len()) as u64
                        }
                    })
                    .sum();
                #[cfg(feature = "instrument")]
                let named_blocks_kv_started = read_ticks();
                for (layer, names) in cache_names.iter().enumerate() {
                    match (names, &layer_caches[layer]) {
                        (
                            LayerCacheNames::Attention { k_even, k_odd, v },
                            LayerCacheState::Attention(cache),
                        ) => {
                            named_blocks.extend(cache.named_blocks(k_even, k_odd, v));
                        }
                        (
                            LayerCacheNames::DenseAttention { k_first, k_second, k_pass, v },
                            LayerCacheState::DenseAttention(cache),
                        ) => {
                            named_blocks.extend(cache.named_blocks(k_first, k_second, k_pass, v));
                        }
                        (
                            LayerCacheNames::Ssm { conv_history, state },
                            LayerCacheState::Ssm(cache),
                        ) => {
                            named_blocks.extend(cache.named_blocks(conv_history, state));
                        }
                        _ => unreachable!(
                            "cache_names/layer_caches built from the same layer_roots, in lockstep"
                        ),
                    }
                }
                #[cfg(feature = "instrument")]
                let named_blocks_kv_ticks = elapsed_ticks(named_blocks_kv_started);

                let symbols = [new_count as u64, cached_len as u64];
                let mut roots: Vec<NodeId> = Vec::with_capacity(1 + self.layer_roots.len() * 3);
                roots.push(self.logits_root);
                for roots_for_layer in &self.layer_roots {
                    match roots_for_layer {
                        Qwen35LayerRoots::Attention((even, odd, value)) => {
                            roots.push(*even);
                            roots.push(*odd);
                            roots.push(*value);
                        }
                        Qwen35LayerRoots::DenseAttention((first, second, pass, value)) => {
                            roots.push(*first);
                            roots.push(*second);
                            roots.push(*pass);
                            roots.push(*value);
                        }
                        Qwen35LayerRoots::Ssm { qkv_mixed, state_out } => {
                            roots.push(*qkv_mixed);
                            roots.push(*state_out);
                        }
                    }
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
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let evaluated = match std::env::var("PROXIMA_METAL_OP_PROFILE_STEP")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                {
                    Some(target) if target == _step => {
                        let (evaluated, timings) = runtime.evaluate_op_timed(
                            &self.program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                        )?;
                        report_op_timings(_step, &timings);
                        evaluated
                    }
                    _ => runtime.evaluate(
                        &self.program,
                        &symbols,
                        &named_blocks,
                        &roots,
                        &resident_names,
                    )?,
                };
                #[cfg(not(all(feature = "instrument", feature = "metal", target_os = "macos")))]
                let evaluated = runtime.evaluate(
                    &self.program,
                    &symbols,
                    &named_blocks,
                    &roots,
                    &resident_names,
                )?;
                #[cfg(feature = "instrument")]
                let evaluate_ticks = elapsed_ticks(evaluate_started);
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
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
                for (layer, roots_for_layer) in self.layer_roots.iter().enumerate() {
                    match (roots_for_layer, &mut layer_caches[layer]) {
                        (
                            Qwen35LayerRoots::Attention((even, odd, value)),
                            LayerCacheState::Attention(cache),
                        ) => {
                            let (even_data, _) = evaluated
                                .get(*even)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *even })?;
                            let (odd_data, _) = evaluated
                                .get(*odd)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *odd })?;
                            let (value_data, _) = evaluated
                                .get(*value)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *value })?;
                            #[cfg(feature = "instrument")]
                            {
                                layer_cache_append_elements +=
                                    (even_data.len() + odd_data.len() + value_data.len()) as u64;
                            }
                            cache.append(even_data, odd_data, value_data);
                        }
                        (
                            Qwen35LayerRoots::DenseAttention((first, second, pass, value)),
                            LayerCacheState::DenseAttention(cache),
                        ) => {
                            let (first_data, _) = evaluated
                                .get(*first)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *first })?;
                            let (second_data, _) = evaluated
                                .get(*second)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *second })?;
                            let (pass_data, _) = evaluated
                                .get(*pass)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *pass })?;
                            let (value_data, _) = evaluated
                                .get(*value)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *value })?;
                            #[cfg(feature = "instrument")]
                            {
                                layer_cache_append_elements += (first_data.len()
                                    + second_data.len()
                                    + pass_data.len()
                                    + value_data.len())
                                    as u64;
                            }
                            cache.append(first_data, second_data, pass_data, value_data);
                        }
                        (
                            Qwen35LayerRoots::Ssm { qkv_mixed, state_out },
                            LayerCacheState::Ssm(cache),
                        ) => {
                            let (qkv_mixed_data, _) = evaluated.get(*qkv_mixed).ok_or(
                                InteropError::MissingEvaluatedNode { node: *qkv_mixed },
                            )?;
                            let (state_out_data, _) = evaluated.get(*state_out).ok_or(
                                InteropError::MissingEvaluatedNode { node: *state_out },
                            )?;
                            #[cfg(feature = "instrument")]
                            {
                                layer_cache_append_elements +=
                                    (qkv_mixed_data.len() + state_out_data.len()) as u64;
                            }
                            // `Self::load`'s own invariant: an `Ssm` entry in
                            // `layer_roots` exists only when `qwen35_ssm_shape`
                            // was derived alongside it (both come from the same
                            // `crate::qwen35::Qwen35Architecture`), so this
                            // fallback shape is never actually read.
                            let shape = self.qwen35_ssm_shape.unwrap_or(Qwen35SsmShape {
                                qkv_dim: 0,
                                conv_rows: 0,
                                state_len: 0,
                            });
                            cache.advance(qkv_mixed_data, state_out_data, shape);
                        }
                        _ => unreachable!(
                            "layer_roots/layer_caches built from the same layer_roots, in lockstep"
                        ),
                    }
                }
                #[cfg(feature = "instrument")]
                let layer_cache_append_ticks = elapsed_ticks(layer_cache_append_started);
                cached_len += new_count;

                let (logits, _shape) =
                    evaluated
                        .get(self.logits_root)
                        .ok_or(InteropError::MissingEvaluatedNode {
                            node: self.logits_root,
                        })?;
                let last_position = &logits[(new_count - 1) * vocab_size..new_count * vocab_size];

                if _step == 0 {
                    let stats = |data: &[f32]| {
                        let len = data.len();
                        let nan_count = data.iter().filter(|value| value.is_nan()).count();
                        let inf_count = data.iter().filter(|value| value.is_infinite()).count();
                        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let all_zero = data.iter().all(|value| *value == 0.0);
                        (len, nan_count, inf_count, min, max, all_zero)
                    };
                    let (len, nan_count, inf_count, min, max, all_zero) = stats(last_position);
                    std::println!(
                        "DIAG_PROBE logits len={len} nan_count={nan_count} inf_count={inf_count} min={min} max={max} all_zero={all_zero}"
                    );
                    for (layer_index, roots_for_layer) in self.layer_roots.iter().enumerate() {
                        match roots_for_layer {
                            Qwen35LayerRoots::Attention((_even, _odd, value)) => {
                                match evaluated.get(*value) {
                                    Some((data, _shape)) => {
                                        let (len, nan_count, inf_count, min, max, all_zero) = stats(data);
                                        std::println!(
                                            "DIAG_PROBE layer={layer_index} kind=Attention node=value len={len} nan_count={nan_count} inf_count={inf_count} min={min} max={max} all_zero={all_zero}"
                                        );
                                    }
                                    None => std::println!(
                                        "DIAG_PROBE layer={layer_index} kind=Attention node=value MISSING"
                                    ),
                                }
                            }
                            Qwen35LayerRoots::DenseAttention((_first, _second, _pass, value)) => {
                                match evaluated.get(*value) {
                                    Some((data, _shape)) => {
                                        let (len, nan_count, inf_count, min, max, all_zero) = stats(data);
                                        std::println!(
                                            "DIAG_PROBE layer={layer_index} kind=DenseAttention node=value len={len} nan_count={nan_count} inf_count={inf_count} min={min} max={max} all_zero={all_zero}"
                                        );
                                    }
                                    None => std::println!(
                                        "DIAG_PROBE layer={layer_index} kind=DenseAttention node=value MISSING"
                                    ),
                                }
                            }
                            Qwen35LayerRoots::Ssm { qkv_mixed, state_out } => {
                                match evaluated.get(*qkv_mixed) {
                                    Some((qkv_data, _shape)) => {
                                        let (qkv_len, qkv_nan, qkv_inf, qkv_min, qkv_max, qkv_all_zero) = stats(qkv_data);
                                        std::println!(
                                            "DIAG_PROBE layer={layer_index} kind=Ssm node=qkv_mixed len={qkv_len} nan_count={qkv_nan} inf_count={qkv_inf} min={qkv_min} max={qkv_max} all_zero={qkv_all_zero}"
                                        );
                                    }
                                    None => std::println!(
                                        "DIAG_PROBE layer={layer_index} kind=Ssm node=qkv_mixed MISSING"
                                    ),
                                }
                                match evaluated.get(*state_out) {
                                    Some((state_data, _shape)) => {
                                        let (state_len, state_nan, state_inf, state_min, state_max, state_all_zero) = stats(state_data);
                                        std::println!(
                                            "DIAG_PROBE layer={layer_index} kind=Ssm node=state_out len={state_len} nan_count={state_nan} inf_count={state_inf} min={state_min} max={state_max} all_zero={state_all_zero}"
                                        );
                                    }
                                    None => std::println!(
                                        "DIAG_PROBE layer={layer_index} kind=Ssm node=state_out MISSING"
                                    ),
                                }
                            }
                        }
                    }
                }

                #[cfg(feature = "instrument")]
                let greedy_pick_started = read_ticks();
                let recent_window_start = token_history.len().saturating_sub(repeat_window);
                let recent_tokens = &token_history[recent_window_start..];
                let token_id =
                    sample_next_token(last_position, recent_tokens, sample_config, &mut rng)
                        .ok_or(InteropError::EmptyLogits)?;
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
                    // ROW 130's per-step-reset attribution: kernel / dispatch+
                    // setup / park+spin+wake, all on the CALLING thread's own
                    // wall clock (never summed across the cohort's other worker
                    // threads, which run concurrently with it, not serially
                    // inside it -- see `CohortLeaderAttribution`'s own doc).
                    // `evaluate_ns` is this step's own tick-based total, already
                    // reset per step by `reset_step`; `residual_ns` is
                    // everything `evaluate_ms` paid for that these three terms
                    // do not name -- non-matmul ops (elementwise/reduce/scan),
                    // quantize/transpose bookkeeping, and staged-batch setup
                    // outside the cohort round itself. `saturating_sub` so a
                    // negative residual is impossible to construct by
                    // arithmetic; reported as 0 with `residual_underflow=true`
                    // if the three named terms would have exceeded the parent,
                    // which is itself a sanity-gate failure worth seeing rather
                    // than silently wrapping.
                    let attribution = proxima_tensor::instrument::cohort_leader_attribution();
                    let evaluate_ns = ticks_to_nanos(evaluate_ticks);
                    let named_ns = attribution.kernel_nanos
                        + attribution.dispatch_nanos
                        + attribution.park_spin_wake_nanos;
                    let residual_underflow = named_ns > evaluate_ns;
                    let residual_ns = evaluate_ns.saturating_sub(named_ns);
                    std::println!(
                        "token_attribution step={_step} evaluate_ms={:.3} kernel_ms={:.3} dispatch_ms={:.3} \
                     park_spin_wake_ms={:.3} residual_ms={:.3} residual_underflow={residual_underflow} \
                     named_plus_residual_ms={:.3}",
                        evaluate_ns as f64 / 1e6,
                        attribution.kernel_nanos as f64 / 1e6,
                        attribution.dispatch_nanos as f64 / 1e6,
                        attribution.park_spin_wake_nanos as f64 / 1e6,
                        residual_ns as f64 / 1e6,
                        (named_ns + residual_ns) as f64 / 1e6,
                    );
                    // ROW 140's own redundant-activation-quantize hypothesis
                    // check: `total_calls` vs `distinct_nodes` across every
                    // matmul reduce node this step evaluated. 1:1 kills the
                    // hypothesis; a ratio near the QKV/gate-up fan-out (2-3x)
                    // confirms it.
                    let (quantize_total_calls, quantize_distinct_nodes) =
                        proxima_tensor::instrument::quantize_activation_call_stats();
                    let quantize_cache_hits =
                        proxima_tensor::instrument::QUANTIZE_ACTIVATION_CACHE_HITS.get();
                    std::println!(
                        "token_quantize_calls step={_step} total_calls={quantize_total_calls} distinct_nodes={quantize_distinct_nodes} \
                     cache_hits={quantize_cache_hits}"
                    );
                    #[cfg(all(feature = "metal", target_os = "macos"))]
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
                     resident_uploads={} resident_reuses={} mapping_offset_uploads={} \
                     nocopy_cache_len={} phys_footprint_bytes={} device_allocated_bytes={} \
                     plan_cache_len={} plan_hits={} plan_misses={}",
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
                        metal_stage.mapping_offset_uploads,
                        omega::metal::nocopy_cache_len(),
                        phys_footprint_bytes(),
                        omega::metal::current_allocated_size().unwrap_or(0),
                        runtime.plans_len(),
                        runtime.plan_hits,
                        runtime.plan_misses,
                    );
                }

                Ok(token_id)
            },
        )?;

        let text = proxima_tokenizer::decode(&generated_ids, &self.vocab)?;
        Ok((generated_ids, text, stopped_by_eos))
    }

    /// A one-shot forward pass over `prompt` (BOS forced, fresh KV state,
    /// same input-binding shape as `Self::run_decode_loop`'s own first
    /// step) that returns the raw values for each requested `NodeId`
    /// instead of sampling a token from the final logits.
    ///
    /// Exists as this crate's cross-oracle diagnostic surface: a decoded
    /// token is an argmax, and an argmax destroys exactly the information
    /// that tells a near-tied bf16-vs-f16 rounding gap apart from a gross
    /// defect. [`Self::forward_logits`] is the `node_ids == [logits_root]`
    /// convenience most callers want; this general form additionally lets a
    /// caller bisect a numeric divergence by depth: build
    /// [`proxima_tensor::spec::mistral_cached_forward_program_with_experts`]
    /// again at a shorter `block_count` against this same architecture and
    /// read off the last shared `NodeId` (`proxima_tensor::op::append`'s
    /// id-is-index invariant guarantees the two programs agree on every
    /// `NodeId` up to the point they diverge) -- that id is the residual
    /// stream's value right after that layer, comparable directly against
    /// an oracle's own per-layer tensor dump.
    /// `examples/smollm2_logit_oracle_diff.rs` is the worked tool.
    ///
    /// # Errors
    ///
    /// Whatever tokenizing `prompt` against this checkpoint's own
    /// [`Vocab`] or evaluating its forward program can fail with, plus
    /// [`InteropError::MissingEvaluatedNode`] if any `node_ids` entry was
    /// never computed by this checkpoint's own forward program (a caller
    /// passed a `NodeId` from a differently-shaped program).
    pub fn forward_node_values(
        &self,
        prompt: &str,
        node_ids: &[NodeId],
    ) -> Result<Vec<Vec<f32>>, InteropError> {
        let serving_config = supported_serving_config();
        let mut runtime = BackendRuntime::new(&serving_config);

        let ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
        apply_serving_config(&serving_config, ids.len())?;
        let inputs = build_position_inputs(
            &ids,
            0,
            self.architecture.head_dim,
            self.architecture.rope_freq_base,
        );

        let block_count = self.architecture.block_count as usize;
        let empty_cache = LayerCache::new();
        let kv_cache_names: Vec<(String, String, String)> = (0..block_count)
            .map(|layer| {
                (
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                )
            })
            .collect();

        let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
            self.weights.owned.len()
                + self.weights.packed.len()
                + self.weights.packed_owned.len()
                + 3
                + block_count * 3,
        );
        named_blocks.push(("ids", QuantizedBlock::Float32(inputs.ids_f32.as_slice())));
        for (name, data) in &self.weights.owned {
            named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
        }
        for (name, block) in &self.weights.packed {
            named_blocks.push((name.as_str(), *block));
        }
        for (name, bytes, kind) in &self.weights.packed_owned {
            named_blocks.push((name.as_str(), kind.as_block(bytes)));
        }
        named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
        named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
        named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));
        for (k_even_name, k_odd_name, v_name) in &kv_cache_names {
            named_blocks.extend(empty_cache.named_blocks(k_even_name, k_odd_name, v_name));
        }

        let resident_names: BTreeSet<&str> = self
            .weights
            .owned
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(self.weights.packed.iter().map(|(name, _)| name.as_str()))
            .chain(
                self.weights
                    .packed_owned
                    .iter()
                    .map(|(name, _, _)| name.as_str()),
            )
            .collect();

        let symbols = [ids.len() as u64, 0u64];
        let evaluated = runtime.evaluate(
            &self.program,
            &symbols,
            &named_blocks,
            node_ids,
            &resident_names,
        )?;

        node_ids
            .iter()
            .map(|node| {
                evaluated
                    .get(*node)
                    .map(|(data, _shape)| data.to_vec())
                    .ok_or(InteropError::MissingEvaluatedNode { node: *node })
            })
            .collect()
    }

    /// [`Self::forward_node_values`] against `[Self::logits_root]`, sliced
    /// to just the LAST prompt position -- the convenience a caller
    /// cross-checking a decoded token's own logits (not an intermediate
    /// layer) wants. See that method's own doc for why a raw logit vector,
    /// not a sampled token, is what this crate's cross-oracle diagnostics
    /// need.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::forward_node_values`] can fail with.
    pub fn forward_logits(&self, prompt: &str) -> Result<Vec<f32>, InteropError> {
        let ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
        let mut values = self.forward_node_values(prompt, &[self.logits_root])?;
        let logits = values.remove(0);
        let vocab_size = self.architecture.vocab as usize;
        let new_count = ids.len();
        let last_position = logits[(new_count - 1) * vocab_size..new_count * vocab_size].to_vec();

        #[cfg(feature = "instrument")]
        {
            let mut ranked: Vec<usize> = (0..last_position.len()).collect();
            ranked.sort_by(|left, right| {
                last_position[*right]
                    .total_cmp(&last_position[*left])
                    .then_with(|| left.cmp(right))
            });
            let top1_token = ranked[0] as u64;
            let top1_logit = f64::from(last_position[ranked[0]]);
            debug!(
                prompt_tokens = ids.len() as u64,
                top1_token,
                top1_logit,
                "computed one-shot forward logits for cross-oracle comparison"
            );
        }

        Ok(last_position)
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
        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
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

        assert_eq!(
            generated_ids,
            alloc::vec![10, 20],
            "eos id must not be appended to the generated ids"
        );
        assert!(
            stopped_by_eos,
            "must report that the stop was the model's own eos signal"
        );
        assert_eq!(
            calls, 3,
            "must not pull a 4th token once eos is seen on the 3rd"
        );
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
            decode_until_stop_or_budget(&vocab, scripted_tokens.len(), |step| {
                Ok(scripted_tokens[step])
            })
            .expect("scripted token source never errors");

        assert_eq!(
            generated_ids,
            alloc::vec![10, 20, 30, 40],
            "every scripted token is a real id, none is eos"
        );
        assert!(
            !stopped_by_eos,
            "budget exhaustion must not be reported as an eos stop"
        );
        assert_eq!(
            generated_ids.len(),
            scripted_tokens.len(),
            "budget exhaustion still runs every requested step"
        );
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

        assert!(
            generated_ids.is_empty(),
            "an immediate eos must produce zero generated ids"
        );
        assert!(stopped_by_eos);
        assert_eq!(
            calls, 1,
            "must stop after exactly one call, not run toward the budget of 10"
        );
    }
}
