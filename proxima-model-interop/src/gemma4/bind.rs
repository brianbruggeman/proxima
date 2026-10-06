//! [`crate::architecture::Architecture`] registration for the `gemma4`
//! checkpoint family. `Gemma4Arch::bind` is a DESCRIPTOR consumer: it hands the
//! parsed header to [`proxima_tensor::spec::gemma4_descriptor_from_gguf`],
//! which builds the whole [`proxima_tensor::spec::ModelDescriptor`] (per-layer
//! [`proxima_tensor::spec::LayerAttentionConfig`]/
//! [`proxima_tensor::spec::LayerFfnConfig`] schedule included), lowers it
//! through [`proxima_tensor::spec::build_forward`], and binds the weights that
//! program's `Input` leaves name ([`crate::bind_leaves::bind_program_leaves`])
//! -- there is no bespoke gemma4 forward-graph builder, no gemma4 schedule and
//! no gemma4 tensor-name table in this crate.
//! Teaching pointer: read `proxima_tensor::spec::attention_forward`'s own doc
//! on `lfm2_forward_program_with_experts` before touching this file -- every
//! knob the descriptor sets is documented there, not here.

use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::{
    CacheStrategy, ModelDescriptor, Qwen35LayerRoots, build_forward,
    gemma4_descriptor_from_gguf,
};
use crate::architecture::{
    Architecture as ArchitectureTrait, BoundProgram, KvLayout, StepInput, StepInputContext, rebuild_layer_roots,
};
use crate::bind::{BoundWeights, ModelArchitecture, SlidingRope, find_tensor, metadata_str};
use crate::bind_leaves::bind_program_leaves;
use crate::error::InteropError;
use crate::profiles::{binding_profile, family_profile};

use super::hparams::from_metadata;
use super::program::gemma4_sliding_rope_table;

/// The binding profile key and the registry name: the architecture that lowers this program
/// is what names its leaves, so a delegating foreign architecture binds the same way.
const FAMILY: &str = "gemma4";

/// Marker registered for `general.architecture = "gemma4"`.
pub struct Gemma4Arch;

/// The builtin `gemma4` registration value.
pub static GEMMA4: Gemma4Arch = Gemma4Arch;

/// The checkpoint's descriptor: [`gemma4_descriptor_from_gguf`] over the family
/// profile `general.architecture` names, so the values GGUF does not carry come
/// from `crate::profiles` and never from this module.
#[cfg(feature = "std")]
pub fn descriptor_from_gguf(
    parsed: &ParsedGguf,
    sliding_kv_ring: bool,
) -> Result<ModelDescriptor, InteropError> {
    let profile = family_profile(metadata_str(parsed, "general.architecture")?)?;
    Ok(gemma4_descriptor_from_gguf(parsed, sliding_kv_ring, &profile)?)
}

/// `PROXIMA_HEAD_REPEATS=1|2|3` (unset or unparsable reads as `1`): the
/// head-cost measurement knob, layered into [`proxima_tensor::spec::ModelDescriptor::head_repeats`]
/// here so the op-graph builders stay pure functions of the descriptor.
#[cfg(feature = "instrument")]
fn head_repeats_from_env() -> u32 {
    std::env::var("PROXIMA_HEAD_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
}

/// [`Gemma4Arch::bind`]'s body, parameterized on `last_row_only`
/// (`lfm2_two_range_cached_forward_program_with_experts`'s own trailing
/// flag -- see its doc: `true` gathers the LM head to the last new
/// position only, `false` leaves every new position's own row in
/// `logits_root`, `[new_count, vocab]`). `Gemma4Arch::bind` above calls
/// this with `true` unchanged, so the registered decode/prefill path is
/// byte-for-byte what it was before this function existed.
/// [`bind_gemma4_all_positions_logits`] is the only other caller, with
/// `false` -- the additive SLICE 2a readout a speculative-decode verify
/// step needs (per-position logits for every candidate token from ONE
/// forward, not just the sampled last position).
#[cfg(feature = "std")]
fn bind_gemma4_with_last_row_only<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    last_row_only: bool,
    layout: KvLayout,
) -> Result<BoundProgram<'file>, InteropError> {
    let architecture = from_metadata(parsed)?;

    // every gemma4 shape routes through the padded-mask cached engine (all layers are `LayerKind::Attention`);
    // the two-range engine, not single-range, because the first step processes the whole
    // prompt as one `cached_len=0` call (`lfm2_single_range_cached.rs`'s own module doc)
    let descriptor = ModelDescriptor {
        last_row_only,
        ..descriptor_from_gguf(parsed, layout == KvLayout::SlidingRing)?
    };
    #[cfg(feature = "instrument")]
    let descriptor = ModelDescriptor {
        head_repeats: head_repeats_from_env(),
        ..descriptor
    };
    let schedule = &descriptor.layers;
    let cache_strategy = descriptor.cache_strategy;
        let (program, logits, cache_roots, moe_sites, _layer_residuals, _hidden, duplicate_head_roots) =
            build_forward(&descriptor)?;
        let weights = bind_program_leaves(
            parsed,
            file_bytes,
            &program,
            &binding_profile(FAMILY)?,
            &[],
        )?;
        let layer_roots: Vec<Qwen35LayerRoots> = match cache_strategy {
            CacheStrategy::Cached => {
                // `cache_roots` holds one entry per REAL cache-owning layer,
                // in layer order (`lfm2_two_range_cached_forward_program_with_experts`'s
                // own `stored_kv`/`cache_roots.push` doc: a
                // `KeySourceKind::SharedFromLayer` layer owns none) -- this
                // zips it back against `schedule`'s own per-layer
                // discriminant to rebuild the full, positionally-real
                // `Qwen35LayerRoots` vec `LoadedModel::declared_layer_cache_names_and_widths`
                // needs (one entry per layer index, `SharedFromLayer`
                // included).
                rebuild_layer_roots(
                    schedule.iter().map(|entry| entry.attention.key_source_kind),
                    cache_roots,
                )?
            }
            CacheStrategy::Cacheless => Vec::new(),
        };

        let tied_embeddings = find_tensor(parsed, "output.weight").is_err();
        let full_head_dim = architecture.key_length;
        let model_architecture = ModelArchitecture {
            vocab: architecture.vocab,
            embedding: architecture.embedding,
            feed_forward: architecture.feed_forward,
            query_heads: architecture.head_count,
            kv_heads: *architecture.kv_heads_by_layer.last().unwrap_or(&0),
            kv_heads_by_layer: architecture.kv_heads_by_layer.clone(),
            head_dim: full_head_dim,
            block_count: architecture.block_count,
            expert_count: architecture.expert_count,
            expert_used_count: architecture.expert_used_count,
            rope_freq_base: architecture.rope_freq_base,
            rms_epsilon: architecture.rms_epsilon,
            tied_embeddings,
            family: metadata_str(parsed, "general.architecture")?.into(),
            sliding_rope: Some(SlidingRope {
                freq_base: architecture.rope_freq_base_swa,
                dimension_count: architecture.rope_dimension_count_swa,
            }),
        };

        Ok(BoundProgram {
            weights,
            architecture: model_architecture,
            program,
            logits_root: logits,
            hidden_root: None,
            residual_roots: Vec::new(),
            layer_roots,
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites,
            duplicate_head_roots,
            single_position_step: false,
        })
}

/// SLICE 2a verify readout: the additive, opt-in counterpart to
/// [`Gemma4Arch::bind`] (which always gathers `logits_root` to the last new
/// position -- `bind_gemma4_with_last_row_only`'s own doc). Binds the exact
/// same weights/program shape with `last_row_only: false`, so `logits_root`
/// evaluates to every new position's own row, `[new_count, vocab]`, not
/// just the last. A speculative-decode verify step feeds this a K-token
/// candidate span (`context.new_start = cached_len`, one forward call) and
/// reads back K rows of logits instead of K separate single-position
/// decode steps -- `Gemma4Arch::bind`'s own program, decode loop, and
/// `logits_root` shape are untouched by this function existing.
#[cfg(feature = "std")]
pub fn bind_gemma4_all_positions_logits<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
) -> Result<BoundProgram<'file>, InteropError> {
    bind_gemma4_with_last_row_only(parsed, file_bytes, false, KvLayout::Full)
}

impl ArchitectureTrait for Gemma4Arch {
    fn name(&self) -> &'static str {
        FAMILY
    }

    fn kv_cache_shape(&self) -> crate::architecture::KvCacheShape {
        crate::architecture::KvCacheShape::Custom
    }

    fn kv_layers(&self, parsed: &ParsedGguf) -> Result<Vec<(u32, u32, Option<u32>)>, InteropError> {
        super::hparams::kv_layers_from_metadata(parsed)
    }

    /// Intervention 6's measured decode configuration (RUN.md "Intervention
    /// 6" / "INTEGRATION"): whole-token latency 18.83 -> 16.60 ms (-11.9%) at
    /// K=8, bytes identical to K=1 on the six-prompt corpus, uninstrumented
    /// rollout check confirming the same order of magnitude
    /// (-1.98 ms/token). Scoped to gemma4 alone -- every other architecture
    /// keeps [`ArchitectureTrait::command_buffer_chunks`]'s own default of `1`;
    /// this measurement does not establish a universal placements-path
    /// default (the owner's own integration recommendation).
    fn command_buffer_chunks(&self) -> u32 {
        8
    }

    #[cfg(feature = "std")]
    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        bind_gemma4_with_last_row_only(parsed, file_bytes, true, KvLayout::Full)
    }

    /// [`Self::bind`] with the sliding layers' cache laid out as the ring
    /// [`KvLayout::SlidingRing`] names -- what
    /// [`crate::generate::LoadedModel::load`] binds. [`Self::bind`] itself
    /// stays the full-cache layout, so every caller that drives its own
    /// `kv_cache.*` leaves against it is unchanged.
    #[cfg(feature = "std")]
    fn bind_with_kv_layout<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        layout: KvLayout,
    ) -> Result<BoundProgram<'file>, InteropError> {
        bind_gemma4_with_last_row_only(parsed, file_bytes, true, layout)
    }

    #[cfg(not(feature = "std"))]
    fn bind<'file>(
        &self,
        _parsed: &ParsedGguf,
        _file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        Err(InteropError::HybridMoeProgramUnsupported {
            name: self.name().into(),
        })
    }

    /// [`bind_gemma4_all_positions_logits`]'s own trait-level entry point --
    /// this crate's only [`ArchitectureTrait::speculative_verify_program`]
    /// override (that method's own doc on why a capability method, never a
    /// `name() == "gemma4"` check, is what gates speculative decode's
    /// verify step).
    #[cfg(feature = "std")]
    fn speculative_verify_program<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<Option<BoundProgram<'file>>, InteropError> {
        self.speculative_verify_program_with_kv_layout(parsed, file_bytes, KvLayout::Full)
    }

    #[cfg(feature = "std")]
    fn speculative_verify_program_with_kv_layout<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        layout: KvLayout,
    ) -> Result<Option<BoundProgram<'file>>, InteropError> {
        let header = descriptor_from_gguf(parsed, layout == KvLayout::SlidingRing)?;
        header
            .verify()
            .map(|_| bind_gemma4_with_last_row_only(parsed, file_bytes, false, layout))
            .transpose()
    }

    /// Feeds the sliding-window RoPE table the `rope_cos_swa`/`rope_sin_swa`
    /// leaves declare (`LayerAttentionConfig::rope_table`,
    /// `gemma4_descriptor_from_gguf`) -- the decode loop's builtin
    /// `rope_cos`/`rope_sin` blocks always carry the FULL-layer table
    /// (`Gemma4Arch::bind`'s own `ModelArchitecture::head_dim`/
    /// `rope_freq_base` are the full-layer values), so this is the one
    /// extra leaf this architecture needs from
    /// [`crate::architecture::Architecture::step_inputs`]'s own seam.
    /// Positions are `context.new_start + offset` for
    /// `offset in 0..context.new_count`, matching
    /// `crate::generate::residency_caches::build_position_inputs`'s own
    /// absolute-angle convention for the builtin table.
    fn step_inputs(&self, context: &StepInputContext<'_>, out: &mut Vec<StepInput>) {
        let positions: Vec<usize> = (0..context.new_count)
            .map(|offset| context.new_start + offset)
            .collect();
        let Some(rope) = context.architecture.sliding_rope else {
            return;
        };
        let (cos, sin) =
            gemma4_sliding_rope_table(&positions, rope.freq_base, rope.dimension_count);
        out.push(StepInput {
            name: "rope_cos_swa",
            values: cos,
            symbol: None,
        });
        out.push(StepInput {
            name: "rope_sin_swa",
            values: sin,
            symbol: None,
        });
    }

    /// llama.cpp's authoritative gemma4 graph (`src/models/gemma4.cpp`)
    /// applies `ggml_rope_ext` to full/global-attention layers with
    /// `n_rot=head_dim` (matching this checkpoint's `rope.dimension_count`
    /// metadata) but ALSO a per-pair `freq_factors` tensor
    /// (`rope_freqs.weight`, GGUF `ROPE_FREQS`) that ggml divides each
    /// pair's angle by -- `bind_rope_freqs` binds that tensor's own values
    /// verbatim into [`BoundWeights::owned`] at bind time, and this method
    /// hands the same slice straight back to
    /// `crate::generate::build_position_inputs`, which does the dividing.
    /// The real checkpoint's `rope_freqs.weight` holds `[1.0]*64 +
    /// [1e30]*192` (confirmed shape: 256 = `head_dim/2` pair entries):
    /// dividing by `1.0` is a no-op for the first 64 pairs, and dividing by
    /// `1e30` collapses the remaining 192 pairs' `theta` far enough below
    /// one radian that `cos` rounds to exactly `1.0f32` and `sin` rounds to
    /// float noise -- the data-driven replacement for what this method used
    /// to do by returning the literal `64` and letting
    /// `build_position_inputs` truncate its loop there. `rope.
    /// dimension_count=512` is `head_dim`, i.e. `n_rot`, NOT the rotary
    /// count -- the earlier `head_dim / 2` (full rotation of every pair)
    /// read that metadata field as the rotary width, which was the
    /// original bug this checkpoint's own `rope_freqs.weight` now fixes
    /// directly, with no hard-coded pair count anywhere in this crate. The
    /// SWA table ([`Self::step_inputs`]/[`gemma4_sliding_rope_table`])
    /// carries no `freq_factors` in the real graph and is untouched --
    /// `rope_freqs.weight` is never bound into its own separate table.
    fn rope_freq_factors<'weights>(
        &self,
        weights: &'weights BoundWeights<'_>,
    ) -> Option<&'weights [f32]> {
        weights
            .owned
            .iter()
            .find(|(name, _)| name == "rope_freqs.weight")
            .map(|(_, values)| values.as_slice())
    }
}

/// Regression coverage for the bug this crate shipped once: the binder and the
/// forward program each independently gated `attn_k.weight`/
/// `attn_k_norm.weight`/`attn_v.weight` per layer, and nothing forced the
/// two gates to agree -- `gemma4_descriptor_from_gguf` used to gate
/// `ValueSourceKind::ProjectedV` (and therefore the forward program's own
/// `attn_v.weight` [`proxima_tensor::op::Op::Input`] leaf) on `is_sliding`
/// alone, the MoE convention, while E2B/E4B's own-KV FULL layers (`blk.4`,
/// `blk.9`, `blk.14` on the real `gemma4:e2b-it-qat` checkpoint) carry a
/// real `attn_v.weight` regardless of sliding vs full. This test builds the
/// ACTUAL forward program `Gemma4Arch::bind` builds (not a hand-simulated
/// stand-in) for a synthetic E2B-shaped [`Architecture`] whose
/// `sliding_window_pattern`/`shared_kv_layers`/`block_count` are the real
/// checkpoint's own measured values (a real header dump against
/// `~/.ollama/models/blobs/sha256-3646b4c...` on 2026-09-20), then asserts
/// the forward program's own declared `Input` leaf names for these three
/// per-layer weights equal exactly the layers that own those tensors in the
/// real checkpoint (derived below from the header pattern, never from the
/// lowering under test).
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod declared_leaves_match_bound_leaves_tests {
    use super::*;
    use crate::gemma4::Architecture;
    use arrayvec::ArrayVec;
    use proxima_tensor::spec::KeySourceKind;
    use proxima_gguf::types::GgmlType;
    use proxima_gguf::value::{MetadataArray, MetadataValue};
    use proxima_gguf::{GgufModel, TensorPayload, parse_complete, write_complete};
    use proxima_tensor::op::Op;

    /// The real `gemma4:e2b-it-qat` checkpoint's own measured
    /// `sliding_window_pattern`/`shared_kv_layers`/`block_count` (header
    /// dump, 2026-09-20) -- every other key is a plausible dense-E2B
    /// value uninvolved in the attn_k/attn_k_norm/attn_v declare-vs-bind
    /// gate this test exercises. Written by the real GGUF encoder and
    /// parsed back by the real decoder; `token_embd.weight` is a 1-row
    /// table so vocab resolves.
    fn e2b_shaped(shared_kv_layers: u32) -> (ParsedGguf, Architecture) {
        let mut feed_forward = alloc::vec![6144u32; 15];
        feed_forward.extend(alloc::vec![12288u32; 20]);
        let u32_array = |values: Vec<u32>| MetadataValue::Array(MetadataArray::U32(values));
        let metadata = alloc::vec![
            ("general.architecture", MetadataValue::String("gemma4".into())),
            ("gemma4.embedding_length", MetadataValue::U32(1536)),
            ("gemma4.block_count", MetadataValue::U32(35)),
            ("gemma4.attention.head_count", MetadataValue::U32(8)),
            ("gemma4.attention.head_count_kv", u32_array(alloc::vec![1; 35])),
            (
                "gemma4.attention.sliding_window_pattern",
                MetadataValue::Array(MetadataArray::Bool((0..35u32).map(|index| (index + 1) % 5 != 0).collect())),
            ),
            ("gemma4.attention.shared_kv_layers", MetadataValue::U32(shared_kv_layers)),
            ("gemma4.attention.key_length", MetadataValue::U32(512)),
            ("gemma4.attention.value_length", MetadataValue::U32(512)),
            ("gemma4.attention.key_length_swa", MetadataValue::U32(256)),
            ("gemma4.attention.value_length_swa", MetadataValue::U32(256)),
            ("gemma4.attention.sliding_window", MetadataValue::U32(512)),
            ("gemma4.attention.layer_norm_rms_epsilon", MetadataValue::F32(1e-6)),
            ("gemma4.feed_forward_length", u32_array(feed_forward)),
            ("gemma4.rope.freq_base", MetadataValue::F32(1_000_000.0)),
            ("gemma4.rope.freq_base_swa", MetadataValue::F32(10_000.0)),
            ("gemma4.rope.dimension_count", MetadataValue::U32(512)),
            ("gemma4.rope.dimension_count_swa", MetadataValue::U32(256)),
            ("gemma4.final_logit_softcapping", MetadataValue::F32(30.0)),
            ("gemma4.embedding_length_per_layer_input", MetadataValue::U32(256)),
        ];
        let table = [0u8; 1536 * 4];
        let model = GgufModel {
            version: 3,
            metadata: metadata
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
            tensors: alloc::vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: ArrayVec::from_iter([1536u64, 1]),
                ggml_type: GgmlType::F32,
                data: &table,
            }],
        };
        let bytes = write_complete(&model).expect("the e2b-shaped model encodes");
        let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
        let architecture = from_metadata(&parsed).expect("gemma4 hparams parse from the e2b-shaped header");
        (parsed, architecture)
    }

    /// The leaf names of `suffix` a checkpoint with this header stores a
    /// tensor for: own-KV layers carry K and its norm, and V where the
    /// sliding pattern or a shared-KV header says so.
    fn stored_leaf_names(
        architecture: &Architecture,
        suffix: &str,
    ) -> alloc::collections::BTreeSet<String> {
        let first_shared_idx = architecture
            .block_count
            .saturating_sub(architecture.shared_kv_layers);
        architecture
            .sliding_window_pattern
            .iter()
            .enumerate()
            .filter_map(|(layer_index, &is_sliding)| {
                let layer = layer_index as u32;
                let is_shared_kv = layer >= first_shared_idx;
                let bound = match suffix {
                    "attn_k.weight" | "attn_k_norm.weight" => !is_shared_kv,
                    "attn_v.weight" => {
                        (is_sliding || architecture.shared_kv_layers > 0) && !is_shared_kv
                    }
                    other => unreachable!("unexpected suffix {other}"),
                };
                bound.then(|| format!("blk.{layer}.{suffix}"))
            })
            .collect()
    }

    /// The name set the ACTUAL forward program `Gemma4Arch::bind` lowers --
    /// `gemma4_descriptor_from_gguf`'s own output, built through the
    /// cacheless engine so the check is independent of which
    /// `CacheStrategy` bind picks -- declares as an `Input` leaf for `suffix`.
    fn declared_leaf_names(
        parsed: &ParsedGguf,
        suffix: &str,
    ) -> alloc::collections::BTreeSet<String> {
        let mut descriptor = descriptor_from_gguf(parsed, false)
            .expect("the e2b-shaped header carries every key the descriptor reads");
        descriptor.cache_strategy = CacheStrategy::Cacheless;
        let (program, ..) = build_forward(&descriptor).expect("gemma4 e2b-shaped forward program lowers");

        program
            .iter()
            .filter_map(|op| match op {
                Op::Input {
                    name: Some(name), ..
                } => Some(name.clone()),
                _ => None,
            })
            .filter(|name| name.ends_with(suffix) && name.starts_with("blk."))
            .collect()
    }

    #[test]
    fn e2b_declared_attn_v_leaves_equal_bound_attn_v_leaves() {
        let (parsed, architecture) = e2b_shaped(20);
        let declared = declared_leaf_names(&parsed, "attn_v.weight");
        let bound = stored_leaf_names(&architecture, "attn_v.weight");
        assert_eq!(
            declared, bound,
            "forward program declares attn_v.weight leaves the binder does not bind (or vice versa)"
        );
        // The three real full own-KV layers this bug silently dropped --
        // pins the invariant to the actual header fact, not just set
        // equality (an empty-vs-empty pair would also satisfy `assert_eq`
        // above).
        for full_own_kv_layer in [4, 9, 14] {
            let name = format!("blk.{full_own_kv_layer}.attn_v.weight");
            assert!(
                declared.contains(&name),
                "expected {name} to be declared (full own-KV E2B layer has a real attn_v.weight)"
            );
        }
    }

    #[test]
    fn e2b_declared_attn_k_and_attn_k_norm_leaves_equal_bound_leaves() {
        let (parsed, architecture) = e2b_shaped(20);
        for suffix in ["attn_k.weight", "attn_k_norm.weight"] {
            let declared = declared_leaf_names(&parsed, suffix);
            let bound = stored_leaf_names(&architecture, suffix);
            assert_eq!(declared, bound, "{suffix} declare/bind set mismatch");
        }
    }

    /// The MoE path (`shared_kv_layers == 0`) must keep the exact prior
    /// gate: only sliding layers declare/bind `attn_v.weight` -- proves the
    /// E2B fix above did not widen MoE's own set.
    #[test]
    fn moe_declared_attn_v_leaves_stay_gated_on_is_sliding_only() {
        let (parsed, architecture) = e2b_shaped(0);
        let declared = declared_leaf_names(&parsed, "attn_v.weight");
        let bound = stored_leaf_names(&architecture, "attn_v.weight");
        assert_eq!(declared, bound);
        for full_layer in [4, 9, 14] {
            let name = format!("blk.{full_layer}.attn_v.weight");
            assert!(
                !declared.contains(&name),
                "MoE full layer {name} must stay SharedWithKey (no attn_v.weight)"
            );
        }
    }

    /// Hand-derived from ollama's `mlxrunner/model/gemma4/gemma4.go`
    /// `TextConfig` KV-sharing-map build (`gemma4.go:590-611`: a shared layer
    /// reuses "the last non-shared layer of the same type") for
    /// `block_count=35`, `shared_kv_layers=20` (`first_shared_idx=15`): the 16
    /// sliding shared layers all reuse own-KV layer 13, the 4 full shared
    /// layers (19, 24, 29, 34) reuse layer 14 -- exactly 20 pairs, matching
    /// the real header's `attention.shared_kv_layers=20`. Read off the
    /// production descriptor, not a private helper.
    #[test]
    fn gemma4_e2b_shared_kv_reuse_map_matches_hand_derived_table() {
        let (parsed, _) = e2b_shaped(20);
        let descriptor = descriptor_from_gguf(&parsed, false)
            .expect("the e2b-shaped header carries every key the descriptor reads");
        let shared_sources: [(usize, u32); 20] = [
            (15, 13), (16, 13), (17, 13), (18, 13), (19, 14),
            (20, 13), (21, 13), (22, 13), (23, 13), (24, 14),
            (25, 13), (26, 13), (27, 13), (28, 13), (29, 14),
            (30, 13), (31, 13), (32, 13), (33, 13), (34, 14),
        ];

        for (layer, source) in shared_sources {
            let attention = descriptor.layers[layer].attention.clone();
            assert_eq!(
                attention.key_source_kind,
                KeySourceKind::SharedFromLayer(source),
                "layer {layer} expected to read the K of layer {source}"
            );
        }
        assert!(
            descriptor.layers[..15]
                .iter()
                .all(|entry| entry.attention.key_source_kind == KeySourceKind::ProjectedK),
            "own-KV layers 0..15 project their own K"
        );
    }

    /// The header's pattern must itself imply exactly 7 full-attention
    /// layers over 35 blocks (`35 / 5`), independent of the reuse-map
    /// assertion above, so a broken pattern cannot pass it by accident.
    #[test]
    fn gemma4_e2b_header_pattern_has_seven_full_attention_layers_over_thirty_five_blocks() {
        let (parsed, _) = e2b_shaped(20);
        let descriptor = descriptor_from_gguf(&parsed, false)
            .expect("the e2b-shaped header carries every key the descriptor reads");
        let full_layers = descriptor
            .layers
            .iter()
            .filter(|entry| entry.attention.mask_window.is_none())
            .count();
        assert_eq!(full_layers, 7);
    }
}
