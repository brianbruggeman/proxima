//! Header configuration for the `sliding-pattern` mixture-of-experts checkpoint
//! family -- `recurrent_routed_interval::hparams`'s own shape (family-prefixed metadata
//! reads, a per-layer array read into `Vec`) applied to Gemma 4's own
//! alternating sliding-window/full-attention schedule instead of Qwen 3.6's
//! GDN/attention schedule. Layer kind (sliding vs full) lives entirely in
//! `sliding_window_pattern` -- there is no bespoke `LayerKind` here, since
//! [`crate::lowering::bind_checkpoint`] reads that array directly to
//! build [`proxima_tensor::spec::LayerAttentionConfig`] per layer for the
//! generic [`proxima_tensor::spec::scheduled_forward_program_with_experts`]
//! engine, and every sliding-pattern layer is
//! [`proxima_tensor::spec::LayerKind::Attention`] (this family has no
//! `ShortConv` layers).

use alloc::format;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::MetadataValue;

use crate::bind::{
    metadata_bool_per_layer, metadata_f32_optional, metadata_str, metadata_u32,
    metadata_u32_optional, metadata_u32_optional_or, metadata_u32_per_layer,
    vocab_from_token_embedding,
};
use crate::error::InteropError;

#[derive(Debug, Clone)]
pub struct SlidingPatternHparams {
    pub vocab: u32,
    pub embedding: u32,
    pub block_count: u32,
    /// `{family}.feed_forward_length`'s first entry -- kept for callers that
    /// still want a single representative dense-FFN width (e.g. parity
    /// logging). The authoritative per-layer widths this matformer-style
    /// checkpoint (Gemma 4 E2B/E4B) actually carries live in
    /// [`Self::feed_forward_by_layer`]; a uniform checkpoint (12B/26B/31B)
    /// has every entry equal to this one.
    pub feed_forward: u32,
    /// `{family}.feed_forward_length`, one entry per block --
    /// `metadata_u32_per_layer` broadcasts a scalar (12B/26B/31B) to
    /// `block_count` equal entries, and preserves an array (E2B/E4B, whose
    /// matformer variable-width dense FFN this key stores per layer) as-is.
    pub feed_forward_by_layer: Vec<u32>,
    pub expert_feed_forward: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub head_count: u32,
    pub kv_heads_by_layer: Vec<u32>,
    pub rms_epsilon: f32,
    pub key_length: u32,
    pub value_length: u32,
    pub sliding_window: u32,
    pub key_length_swa: u32,
    pub value_length_swa: u32,
    pub sliding_window_pattern: Vec<bool>,
    /// `{family}.attention.shared_kv_layers` -- the count of TRAILING
    /// layers (`block_count - shared_kv_layers` through `block_count - 1`)
    /// that carry no `attn_k.weight`/`attn_v.weight`/`attn_k_norm.weight`
    /// tensors of their own at all (confirmed against the real
    /// `e2b-it-qat` header: an `UnknownTensor` load error on
    /// `blk.15.attn_k_norm.weight` when `block_count=35` and this key reads
    /// `20`). `0` (the default `metadata_u32_optional` returns when the key
    /// is absent, e.g. E4B/12B/26B/31B) means every layer owns its own
    /// K/V -- this crate's prior behaviour, unchanged.
    pub shared_kv_layers: u32,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub rope_dimension_count: u32,
    pub rope_dimension_count_swa: u32,
    pub final_logit_softcapping: f32,
    /// `{family}.embedding_length_per_layer_input` -- Gemma 4 E2B/E4B's
    /// per-layer-embedding (PLE) width (`256` on the real E2B checkpoint,
    /// the reference Go source's own `HiddenSizePerLayer`). `0` (the default
    /// `metadata_u32_optional` returns when the key is absent, e.g.
    /// 12B/26B/31B) means this checkpoint carries no PLE tensors at all --
    /// `crate::sliding_pattern_descriptor_from_gguf`'s own `ple_dim > 0`
    /// check is what gates both the tensor names and the schedule's
    /// `LayerFfnConfig::ple` flag on it.
    pub ple_dim: u32,
    /// `{family}.expert_shared_count` -- the header's shared-expert count (`2`
    /// on the captured sliding-pattern checkpoint). `0` when the key is absent.
    /// The capture's `ffn_*_shexp` rows are one `(2048, 1024)` triple per routed
    /// layer with no expert axis, i.e. a single 1024-wide shared FFN, so this
    /// count is a header fact rather than a width multiplier here; nothing
    /// consumes it yet (the bind takes the descriptor, not this struct) and
    /// whether it becomes one is the shared-expert lowering's decision.
    pub expert_shared_count: u32,
}

pub fn from_metadata(parsed: &ParsedGguf) -> Result<SlidingPatternHparams, InteropError> {
    let family = metadata_str(parsed, "general.architecture")?;
    let prefix = |name: &str| format!("{family}.{name}");

    let embedding = metadata_u32(parsed, &prefix("embedding_length"))?;
    let block_count = metadata_u32(parsed, &prefix("block_count"))?;
    let kv_heads_by_layer =
        metadata_u32_per_layer(parsed, &prefix("attention.head_count_kv"), block_count)?;
    let sliding_window_pattern = metadata_bool_per_layer(
        parsed,
        &prefix("attention.sliding_window_pattern"),
        block_count,
    )?;
    let rope_freq_base = metadata_f32_optional(
        parsed,
        &prefix("rope.freq_base"),
        proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
    );
    let head_count = metadata_u32(parsed, &prefix("attention.head_count"))?;
    // llama.cpp seeds both head widths with `n_embd / n_head` (and 0 when there
    // are no heads) before `attention.key_length` / `attention.value_length`
    // override them (`llama-model.cpp`, `n_embd_head_k_full` / `_v_full`)
    let default_head_width = embedding.checked_div(head_count).unwrap_or(0);
    let key_length =
        metadata_u32_optional_or(parsed, &prefix("attention.key_length"), default_head_width);
    let value_length = metadata_u32_optional_or(
        parsed,
        &prefix("attention.value_length"),
        default_head_width,
    );
    // a checkpoint that attends every layer at one head width omits the swa
    // overrides and the rope width; each falls back to the width it pairs
    // against, which is what makes the captured sliding-pattern header read as
    // 128 throughout
    let key_length_swa =
        metadata_u32_optional_or(parsed, &prefix("attention.key_length_swa"), key_length);
    let value_length_swa =
        metadata_u32_optional_or(parsed, &prefix("attention.value_length_swa"), value_length);
    let rope_dimension_count =
        metadata_u32_optional_or(parsed, &prefix("rope.dimension_count"), key_length);
    let feed_forward_by_layer =
        metadata_u32_per_layer(parsed, &prefix("feed_forward_length"), block_count)?;
    let feed_forward = *feed_forward_by_layer.first().unwrap_or(&0);

    Ok(SlidingPatternHparams {
        vocab: vocab_from_token_embedding(parsed, embedding)?,
        embedding,
        block_count,
        feed_forward,
        feed_forward_by_layer,
        // MoE-only keys (12B/26B/31B carry all three; E2B/E4B are dense and
        // carry none) -- absent means "this checkpoint is not
        // mixture-of-experts", not a malformed file, the same shape
        // `crate::bind`'s own `metadata_u32_optional` already uses for
        // every other family's dense-vs-MoE split.
        expert_feed_forward: metadata_u32_optional(parsed, &prefix("expert_feed_forward_length")),
        expert_count: metadata_u32_optional(parsed, &prefix("expert_count")),
        expert_used_count: metadata_u32_optional(parsed, &prefix("expert_used_count")),
        expert_shared_count: metadata_u32_optional(parsed, &prefix("expert_shared_count")),
        head_count,
        kv_heads_by_layer,
        rms_epsilon: metadata_f32_optional(
            parsed,
            &prefix("attention.layer_norm_rms_epsilon"),
            1e-6,
        ),
        key_length,
        value_length,
        sliding_window: metadata_u32(parsed, &prefix("attention.sliding_window"))?,
        key_length_swa,
        value_length_swa,
        sliding_window_pattern,
        shared_kv_layers: metadata_u32_optional(parsed, &prefix("attention.shared_kv_layers")),
        rope_freq_base,
        rope_freq_base_swa: metadata_f32_optional(
            parsed,
            &prefix("rope.freq_base_swa"),
            rope_freq_base,
        ),
        rope_dimension_count,
        rope_dimension_count_swa: metadata_u32_optional_or(
            parsed,
            &prefix("rope.dimension_count_swa"),
            rope_dimension_count,
        ),
        final_logit_softcapping: metadata_f32_optional(
            parsed,
            &prefix("final_logit_softcapping"),
            0.0,
        ),
        ple_dim: metadata_u32_optional(parsed, &prefix("embedding_length_per_layer_input")),
    })
}

/// One `(kv_heads, head_dim, window)` entry per layer that OWNS a KV cache:
/// layers `0..block_count - shared_kv_layers`. The trailing shared layers
/// read a source layer's cache and are charged nothing. A sliding layer
/// carries `Some(sliding_window)` and `key_length_swa`, a full layer `None`
/// and `key_length`. Reads only the keys it needs, so it works on a
/// header-only GGUF (no `token_embd.weight` lookup as in [`from_metadata`]).
///
/// # Errors
///
/// [`InteropError::MissingMetadataKey`] or
/// [`InteropError::MetadataArrayLengthMismatch`] from the underlying reads.
pub fn kv_layers_from_metadata(
    parsed: &ParsedGguf,
) -> Result<Vec<(u32, u32, Option<u32>)>, InteropError> {
    let family = metadata_str(parsed, "general.architecture")?;
    let prefix = |name: &str| format!("{family}.{name}");

    let block_count = metadata_u32(parsed, &prefix("block_count"))?;
    let kv_heads_by_layer =
        metadata_u32_per_layer(parsed, &prefix("attention.head_count_kv"), block_count)?;
    let sliding_window_pattern = metadata_bool_per_layer(
        parsed,
        &prefix("attention.sliding_window_pattern"),
        block_count,
    )?;
    let own_layers = block_count.saturating_sub(metadata_u32_optional(
        parsed,
        &prefix("attention.shared_kv_layers"),
    ));
    // same optional-read semantics as `from_metadata`: absent or not a `u32`
    // falls back, so every reader of this key agrees on one header
    let key_length = match parsed
        .metadata_value(&prefix("attention.key_length"))
        .and_then(MetadataValue::as_u32)
    {
        Some(width) => width,
        // llama.cpp seeds the width with `n_embd / n_head` when the key is absent
        None => metadata_u32(parsed, &prefix("embedding_length"))?
            .checked_div(metadata_u32(parsed, &prefix("attention.head_count"))?)
            .unwrap_or(0),
    };
    let key_length_swa =
        metadata_u32_optional_or(parsed, &prefix("attention.key_length_swa"), key_length);
    let sliding_window = metadata_u32(parsed, &prefix("attention.sliding_window"))?;

    Ok(sliding_window_pattern
        .into_iter()
        .zip(kv_heads_by_layer)
        .take(own_layers as usize)
        .map(|(is_sliding, kv_heads)| {
            if is_sliding {
                (kv_heads, key_length_swa, Some(sliding_window))
            } else {
                (kv_heads, key_length, None)
            }
        })
        .collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_gguf::pipe::ParsedGguf;
    use proxima_gguf::tensor::TensorInfo;
    use proxima_gguf::types::GgmlType;
    use proxima_gguf::value::{MetadataArray, MetadataValue};
    use proxima_tensor::op::Op;
    use proxima_tensor::spec::{
        ForwardProgram, LayerSchedule, ModelDescriptor, RopePairing, ValueSourceKind, build_forward,
        sliding_pattern_descriptor_from_gguf,
    };

    use crate::bind_leaves::{BindCensus, bind_census};
    use crate::error::InteropError;
    use crate::lowering::step_input_names;
    use crate::memory_fit::{MemoryBudget, WeightClassBytes};
    use crate::oracle_fixtures::read_fixture;
    use crate::profiles::{binding_profile, family_profile};
    use proxima_tensor::spec::ExpertGatingFunc;

    use crate::test_support::{
        flint_layer_pattern, flint_metadata, flint_parsed, flint_parsed_with, flint_tensor_rows,
        gemma4_e2b_header as e2b_header,
    };

    use super::{SlidingPatternHparams, from_metadata, kv_layers_from_metadata};

    const CONTEXT: u32 = 131_072;

    /// 12 sliding layers at 2048 B/position and 3 full layers at 4096, no
    /// window cap: (12 x 2048 + 3 x 4096) x 131072. Today's formula charges
    /// all 35 layers at the last layer's shape and lands far above it.
    #[test]
    fn memory_budget_gemma4_own_layers() {
        let parsed = e2b_header();
        let layers = crate::lowering::kv_layers(&parsed)
            .expect("the e2b header carries every key kv_layers reads");
        let uncapped: alloc::vec::Vec<_> = layers
            .iter()
            .map(|&(kv_heads, head_dim, _)| (kv_heads, head_dim, None))
            .collect();

        let own = MemoryBudget::derive(WeightClassBytes::default(), &uncapped, CONTEXT, 0, 0);
        let every_layer = vec![(1u32, 512u32, None); 35];
        let all_layers =
            MemoryBudget::derive(WeightClassBytes::default(), &every_layer, CONTEXT, 0, 0);

        assert_eq!(layers.len(), 15, "35 blocks minus 20 shared-KV layers");
        assert_eq!(own.kv_cache_bytes, 4_831_838_208);
        assert!(all_layers.kv_cache_bytes > own.kv_cache_bytes);
    }

    /// The same 15 layers with each sliding layer capped at the 512-row
    /// window: 12 x 2048 x 512 + 3 x 4096 x 131072.
    #[test]
    fn memory_budget_gemma4_window_cap() {
        let parsed = e2b_header();
        let layers = crate::lowering::kv_layers(&parsed)
            .expect("the e2b header carries every key kv_layers reads");

        let capped = MemoryBudget::derive(WeightClassBytes::default(), &layers, CONTEXT, 0, 0);

        assert_eq!(capped.kv_cache_bytes, 1_623_195_648);
    }

    /// R2's own literals for the flint capture. Deliberately not derived from
    /// the capture the header is built from, or the comparison could not fail.
    const FLINT_WIDTH: u32 = 128;
    const FLINT_WINDOW: u32 = 513;

    fn flint_sink_tensors() -> Vec<(String, u64)> {
        flint_tensor_rows(".attn_sinks.weight")
    }

    fn flint_sink_refs(sinks: &[(String, u64)]) -> Vec<(&str, u64)> {
        sinks
            .iter()
            .map(|(name, dim)| (name.as_str(), *dim))
            .collect()
    }

    /// The capture's own 779-row tensor directory in the shape the parser hands
    /// the binder. Payload offsets stay `0` because the census compares names,
    /// and the row set, ggml types and dims are the committed capture's own.
    fn flint_tensor_directory() -> Vec<TensorInfo> {
        crate::oracle_fixtures::parse_tensor_rows(&read_fixture("tensor_dir.txt"))
            .into_iter()
            .map(|row| TensorInfo {
                name: row.name,
                dims: row.dimensions.iter().copied().collect(),
                ggml_type: ggml_type_named(&row.ggml_type),
                offset: 0,
            })
            .collect()
    }

    /// The eight ggml types the capture stores, named by the fixture's own
    /// column. A type this map does not name is a changed capture, not a test
    /// this fixture should silently skip.
    fn ggml_type_named(name: &str) -> GgmlType {
        match name {
            "f32" => GgmlType::F32,
            "q8_0" => GgmlType::Q8_0,
            "q5_K" => GgmlType::Q5_K,
            "q4_K" => GgmlType::Q4_K,
            "q6_K" => GgmlType::Q6_K,
            "bf16" => GgmlType::Bf16,
            "iq4_xs" => GgmlType::Iq4Xs,
            "iq3_s" => GgmlType::Iq3S,
            other => {
                panic!("tensor_dir.txt carries ggml type {other:?}, which this map does not name")
            }
        }
    }

    /// The captured header plus the captured directory: the checkpoint a real
    /// load reads, minus the 23 GiB of payload the census does not touch.
    fn flint_checkpoint() -> ParsedGguf {
        let mut parsed = flint_parsed_with(flint_metadata(), &[]);
        parsed.tensors = flint_tensor_directory();
        parsed.tensor_count = parsed.tensors.len() as u64;
        parsed
    }

    /// The descriptor and the forward program a real flint load lowers.
    fn flint_forward(parsed: &ParsedGguf) -> (ModelDescriptor, ForwardProgram) {
        let profile = family_profile("flint").expect("the shipped flint profile parses");
        let descriptor = sliding_pattern_descriptor_from_gguf(parsed, false, &profile)
            .expect("the captured header and the shipped profile lower");
        let forward =
            build_forward(&descriptor).expect("the captured schedule builds a forward program");
        (descriptor, forward)
    }

    fn flint_census(
        parsed: &ParsedGguf,
        descriptor: &ModelDescriptor,
        forward: &ForwardProgram,
    ) -> BindCensus {
        let binding = binding_profile("flint").expect("the binding defaults parse");
        let runtime = step_input_names(descriptor, &forward.program);
        bind_census(parsed, &forward.program, &binding, &runtime)
    }

    /// R7: the real capture's 779 tensors are exactly the weights the lowered
    /// program reads -- no leaf without a tensor, no tensor without a leaf.
    ///
    /// The census is the load path's own gate (`crate::lowering`'s
    /// `bind_descriptor` runs it before returning), so this asserts the same
    /// comparison a `LoadedModel::load` on this checkpoint makes.
    #[test]
    fn flint_bind_all_tensors_() {
        let parsed = flint_checkpoint();
        assert_eq!(
            parsed.tensors.len(),
            779,
            "the capture's own tensor directory is the oracle this test reads"
        );
        let (descriptor, forward) = flint_forward(&parsed);

        let declared: alloc::collections::BTreeSet<&str> = forward
            .program
            .iter()
            .filter_map(|op| match op {
                proxima_tensor::op::Op::Input {
                    name: Some(name), ..
                } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        let runtime = step_input_names(&descriptor, &forward.program);
        assert!(
            runtime.iter().any(|name| name == "ids") && runtime.iter().any(|name| name == "eps"),
            "the step feeds its own scalar and RoPE leaves; a census that classified nothing as \
             runtime would report every one of them unbound"
        );
        assert_eq!(
            runtime
                .iter()
                .filter(|name| name.starts_with("kv_cache."))
                .count(),
            49 * 3,
            "every one of the 49 blocks owns its k_even/k_odd/v cache row"
        );
        assert_eq!(
            declared.len(),
            779 + runtime.len(),
            "the program names the capture's 779 weights and nothing else beyond the step inputs"
        );

        let census = flint_census(&parsed, &descriptor, &forward);

        assert_eq!(
            census.unbound_leaves,
            Vec::<String>::new(),
            "every weight leaf the program reads must be in the capture"
        );
        assert_eq!(
            census.unclaimed_tensors,
            Vec::<String>::new(),
            "every tensor the capture stores must be read by some leaf"
        );
        assert_eq!(census, BindCensus::default());
    }

    /// AC27: the census the test above trusts must reject a directory one
    /// tensor short, and name the tensor -- otherwise it proves nothing about
    /// the real one. Both directions are checked: a dropped weight and a
    /// planted extra.
    #[test]
    fn control_rejects_missing_tensor() {
        let parsed = flint_checkpoint();
        let (descriptor, forward) = flint_forward(&parsed);
        assert_eq!(
            flint_census(&parsed, &descriptor, &forward),
            BindCensus::default(),
            "the capture passes its own comparison"
        );

        // a norm the profile declares, not the directory: dropping it cannot
        // also drop the leaf, which is what makes it a missing-tensor control
        let dropped = "blk.0.attn_q_norm.weight";
        let mut short = parsed.clone();
        short.tensors.retain(|tensor| tensor.name != dropped);
        short.tensor_count = short.tensors.len() as u64;
        let census = flint_census(&short, &descriptor, &forward);
        assert_eq!(
            census.unbound_leaves,
            alloc::vec![String::from(dropped)],
            "the census names exactly the leaf the short directory cannot satisfy"
        );
        assert!(
            census.unclaimed_tensors.is_empty(),
            "dropping a tensor leaves nothing unclaimed, got {:?}",
            census.unclaimed_tensors
        );
        assert!(matches!(
            census.first_problem(),
            Some(InteropError::UnboundWeightLeaf { leaf }) if leaf == dropped
        ));

        let mut planted = parsed.clone();
        let extra = String::from("blk.0.ffn_up_extra.weight");
        planted.tensors.push(TensorInfo {
            name: extra.clone(),
            dims: alloc::vec![2048u64, 6144].into_iter().collect(),
            ggml_type: GgmlType::Q8_0,
            offset: 0,
        });
        planted.tensor_count = planted.tensors.len() as u64;
        let census = flint_census(&planted, &descriptor, &forward);
        assert_eq!(
            census.unclaimed_tensors,
            alloc::vec![extra.clone()],
            "a tensor no leaf reads is reported by name"
        );
        assert!(census.unbound_leaves.is_empty());
        assert!(matches!(
            census.first_problem(),
            Some(InteropError::UnclaimedTensor { tensor }) if tensor == extra
        ));
    }

    /// Every fact R3 names, read back off the descriptor and the header carrier,
    /// as one predicate so the control proves the same comparison rejects a
    /// mutated header.
    fn flint_dense_shared_problem(
        hparams: &SlidingPatternHparams,
        descriptor: &ModelDescriptor,
    ) -> Option<String> {
        if hparams.expert_shared_count != 2 {
            return Some(format!(
                "expert_shared_count = {}, expected 2",
                hparams.expert_shared_count
            ));
        }
        if descriptor.leading_dense_block_count != 2 {
            return Some(format!(
                "leading_dense_block_count = {}, expected 2",
                descriptor.leading_dense_block_count
            ));
        }
        if descriptor.expert_shared_feed_forward != 1_024 {
            return Some(format!(
                "expert_shared_feed_forward = {}, expected 1024",
                descriptor.expert_shared_feed_forward
            ));
        }
        for (index, layer) in descriptor.layers.iter().enumerate() {
            if layer.ffn.routed_gating != ExpertGatingFunc::Sigmoid {
                return Some(format!(
                    "layer {index} routed_gating {:?}, expected sigmoid from expert_gating_func = 2",
                    layer.ffn.routed_gating
                ));
            }
            if !layer.attention.attention_sinks {
                return Some(format!(
                    "layer {index} declares no sink leaf, expected blk.{index}.attn_sinks.weight"
                ));
            }
        }
        None
    }

    /// The header reader under test is generic; the profile only supplies the
    /// layout knobs the header does not carry (RoPE pairing, score scale,
    /// activation, the FFN combination). This is the checkpoint's own profile,
    /// so the descriptor these assertions read is the one a load builds.
    fn flint_descriptor(parsed: &ParsedGguf) -> ModelDescriptor {
        flint_descriptor_result(parsed).expect("the flint header must build a descriptor")
    }

    fn flint_descriptor_result(
        parsed: &ParsedGguf,
    ) -> Result<ModelDescriptor, proxima_tensor::error::TensorError> {
        flint_descriptor_with_profile(parsed, "flint")
    }

    /// The reader is generic and takes any profile; naming the family lets a
    /// test pick a profile whose own routing function differs from the header's,
    /// which is what makes the value table's arms distinguishable.
    fn flint_descriptor_with_profile(
        parsed: &ParsedGguf,
        family: &str,
    ) -> Result<ModelDescriptor, proxima_tensor::error::TensorError> {
        let profile = family_profile(family)
            .unwrap_or_else(|_| panic!("{family} supplies the seating knobs this header lacks"));
        sliding_pattern_descriptor_from_gguf(parsed, false, &profile)
    }

    fn flint_descriptor_error(parsed: &ParsedGguf) -> proxima_tensor::error::TensorError {
        flint_descriptor_result(parsed).expect_err("this header is outside the engine's shape")
    }

    /// Every width R2 names, plus the per-layer schedule, as one predicate so
    /// the control proves the same comparison rejects a mutated header. Four of
    /// the six widths are the fallback chain (the capture omits those keys), so
    /// they move together; the assertions that can fail on their own are
    /// `key_length` against the capture and each layer's window against the
    /// captured array.
    fn sliding_pattern_header_problem(
        hparams: &SlidingPatternHparams,
        descriptor: &ModelDescriptor,
        captured: &[bool],
    ) -> Option<String> {
        for (name, found) in [
            ("key_length", hparams.key_length),
            ("value_length", hparams.value_length),
            ("key_length_swa", hparams.key_length_swa),
            ("value_length_swa", hparams.value_length_swa),
            ("rope_dimension_count", hparams.rope_dimension_count),
            ("rope_dimension_count_swa", hparams.rope_dimension_count_swa),
        ] {
            if found != FLINT_WIDTH {
                return Some(format!("{name} = {found}, expected {FLINT_WIDTH}"));
            }
        }

        if hparams.sliding_window_pattern != captured {
            return Some(String::from(
                "sliding_window_pattern differs from the captured 49-entry array",
            ));
        }
        for (index, layer) in descriptor.layers.iter().enumerate() {
            let attention = &layer.attention;
            if attention.head_dim != FLINT_WIDTH {
                return Some(format!(
                    "layer {index} head_dim {}, expected {FLINT_WIDTH}",
                    attention.head_dim
                ));
            }
            let expected_pairs = RopePairing::SplitHalf {
                pairs: FLINT_WIDTH / 2,
            };
            if attention.rope_pairing != expected_pairs {
                return Some(format!(
                    "layer {index} rope pairing {:?}, expected {expected_pairs:?}",
                    attention.rope_pairing
                ));
            }
            let expected_window = captured[index].then_some(FLINT_WINDOW);
            if attention.mask_window != expected_window {
                return Some(format!(
                    "layer {index} window {:?}, expected {expected_window:?} from the capture",
                    attention.mask_window
                ));
            }
        }
        None
    }

    /// The flint capture omits four width keys. R2: the reader still builds a
    /// descriptor, every layer reports the capture's own 128 rather than a
    /// missing-key error, and a header that does carry the swa keys keeps them.
    #[test]
    fn sliding_pattern_header_derives_the_flint_capture_widths_and_keeps_explicit_swa_keys() {
        let parsed = flint_parsed(flint_metadata());
        let hparams = from_metadata(&parsed)
            .expect("the flint capture carries every key from_metadata reads");
        let descriptor = flint_descriptor(&parsed);
        let captured = flint_layer_pattern("flint.attention.sliding_window_pattern");

        if let Some(problem) = sliding_pattern_header_problem(&hparams, &descriptor, &captured) {
            panic!("{problem}");
        }

        let kv_layers = kv_layers_from_metadata(&parsed)
            .expect("the flint capture carries every key kv_layers_from_metadata reads");
        assert_eq!(
            kv_layers.len(),
            captured.len(),
            "one KV-owning layer per block"
        );
        for (index, &(kv_heads, head_dim, window)) in kv_layers.iter().enumerate() {
            assert_eq!(kv_heads, 4, "layer {index} keeps four kv heads");
            assert_eq!(head_dim, FLINT_WIDTH, "layer {index} attends at 128");
            let expected_window = captured[index].then_some(FLINT_WINDOW);
            assert_eq!(window, expected_window, "layer {index} window");
        }

        let mut explicit = flint_metadata();
        explicit.push(("flint.attention.key_length_swa", MetadataValue::U32(96)));
        explicit.push(("flint.attention.value_length_swa", MetadataValue::U32(96)));
        let parsed = flint_parsed(explicit);
        let hparams = from_metadata(&parsed)
            .expect("an explicit swa width does not remove a key from_metadata reads");
        assert_eq!(hparams.key_length_swa, 96, "an explicit swa key wins");
        assert_eq!(
            hparams.value_length_swa, 96,
            "an explicit swa value key wins"
        );
        assert_eq!(
            hparams.rope_dimension_count_swa, FLINT_WIDTH,
            "an absent rope.dimension_count_swa falls back to the full rotary width, not to the \
             narrower swa head (llama-model.cpp: n_rot_swa = n_rot_full)"
        );

        let descriptor = flint_descriptor(&parsed);
        let sliding: Vec<&LayerSchedule> = descriptor
            .layers
            .iter()
            .filter(|layer| layer.attention.mask_window.is_some())
            .collect();
        assert!(
            !sliding.is_empty(),
            "the captured pattern must select at least one sliding layer"
        );
        for layer in sliding {
            assert_eq!(
                layer.attention.head_dim, 96,
                "an explicit key_length_swa overrides the derived head width"
            );
            assert_eq!(
                layer.attention.rope_pairing,
                RopePairing::SplitHalf {
                    pairs: FLINT_WIDTH / 2
                },
                "the sliding rotary width still comes from the full one, so it does not follow \
                 the explicit swa head width"
            );
        }

        let mut explicit_rope = flint_metadata();
        explicit_rope.push(("flint.rope.dimension_count_swa", MetadataValue::U32(64)));
        let parsed = flint_parsed(explicit_rope);
        let hparams = from_metadata(&parsed).expect("an explicit swa rotary width reads");
        assert_eq!(
            hparams.rope_dimension_count_swa, 64,
            "an explicit swa rope key wins"
        );
        let descriptor = flint_descriptor(&parsed);
        for layer in &descriptor.layers {
            let expected_pairs = if layer.attention.mask_window.is_some() {
                32
            } else {
                FLINT_WIDTH / 2
            };
            assert_eq!(
                layer.attention.rope_pairing,
                RopePairing::SplitHalf {
                    pairs: expected_pairs
                },
                "only the sliding layers take the explicit swa rotary width"
            );
        }

        let mut no_value_length = flint_metadata();
        no_value_length.retain(|(key, _)| *key != "flint.attention.value_length");
        let parsed = flint_parsed(no_value_length);
        let hparams = from_metadata(&parsed).expect("an absent value_length reads");
        assert_eq!(
            hparams.value_length, 64,
            "flint: 2048 / 32, the same fallback llama.cpp computes before the key is read"
        );
        assert_eq!(
            hparams.value_length_swa, 64,
            "an absent value_length_swa then falls back to the derived value_length"
        );

        let mut zero_rope = flint_metadata();
        zero_rope.push(("flint.rope.dimension_count", MetadataValue::U32(0)));
        let parsed = flint_parsed(zero_rope);
        let hparams = from_metadata(&parsed).expect("an explicit zero rotary width reads");
        assert_eq!(
            hparams.rope_dimension_count, 0,
            "a present zero wins over the head width, as llama.cpp's key read does"
        );
        assert_eq!(hparams.rope_dimension_count_swa, 0);
        let descriptor = flint_descriptor(&parsed);
        for (index, layer) in descriptor.layers.iter().enumerate() {
            assert_eq!(
                layer.attention.rope_pairing,
                RopePairing::SplitHalf { pairs: 0 },
                "layer {index}: the descriptor keeps the explicit zero, so both readers agree"
            );
        }

        // llama.cpp defaults the head width to n_embd / n_head when the key is absent
        let mut no_widths = flint_metadata();
        no_widths.retain(|(key, _)| {
            *key != "flint.attention.key_length" && *key != "flint.attention.value_length"
        });
        let parsed = flint_parsed(no_widths);
        let hparams = from_metadata(&parsed).expect("an absent head width reads");
        assert_eq!(
            hparams.key_length, 64,
            "flint: 2048 / 32, the width llama.cpp seeds before the key is read"
        );
        let descriptor = flint_descriptor(&parsed);
        for (index, layer) in descriptor.layers.iter().enumerate() {
            assert_eq!(
                layer.attention.head_dim, 64,
                "layer {index}: the descriptor derives the same width"
            );
            assert_eq!(
                layer.attention.rope_pairing,
                RopePairing::SplitHalf { pairs: 32 },
                "layer {index}: the derived rotary width follows the derived head width"
            );
        }

        // one head width for K and V is an engine boundary, refused rather than mis-scaled
        let mut unequal_full = flint_metadata();
        for entry in &mut unequal_full {
            if entry.0 == "flint.attention.value_length" {
                entry.1 = MetadataValue::U32(96);
            }
        }
        let parsed = flint_parsed(unequal_full);
        let error = flint_descriptor_error(&parsed);
        assert!(
            matches!(
                error,
                proxima_tensor::error::TensorError::GgufUnequalValueWidth {
                    kind: "full",
                    key_length: FLINT_WIDTH,
                    value_length: 96
                }
            ),
            "an unequal value width must be refused by name, got {error:?}"
        );

        let mut unequal_sliding = flint_metadata();
        unequal_sliding.push(("flint.attention.key_length_swa", MetadataValue::U32(96)));
        let parsed = flint_parsed(unequal_sliding);
        let hparams = from_metadata(&parsed)
            .expect("the reviewable header still yields widths; the descriptor is what refuses");
        assert_eq!(
            hparams.value_length_swa, FLINT_WIDTH,
            "an absent value_length_swa falls back to value_length, which is why the pair is unequal"
        );
        let error = flint_descriptor_error(&parsed);
        assert!(
            matches!(
                error,
                proxima_tensor::error::TensorError::GgufUnequalValueWidth {
                    kind: "sliding",
                    key_length: 96,
                    value_length: 128
                }
            ),
            "the sliding pair is refused on the same rule, got {error:?}"
        );
    }

    /// R5: the shared expert's gate is a per-layer tensor, so the reader must
    /// take its presence from the directory the way the sink leaf does -- flint
    /// carries `ffn_{gate,up,down}_shexp.weight` on every routed block and no
    /// `ffn_gate_inp_shexp.weight` anywhere, so every block is ungated.
    #[test]
    fn flint_routed_blocks_declare_no_shared_expert_gate() {
        let sinks = flint_sink_tensors();
        let refs = flint_sink_refs(&sinks);
        let plain = flint_descriptor(&flint_parsed_with(flint_metadata(), &refs));
        assert!(
            plain
                .layers
                .iter()
                .all(|layer| !layer.ffn.shared_expert_gate),
            "the capture has no gate row, so no layer declares one"
        );

        // planting the gate row on one block turns that block's gate on and
        // leaves every other block alone -- the same per-layer rule the sink
        // test asserts, and the control that the flag can move at all
        let mut planted: Vec<(&str, u64)> = refs.clone();
        planted.push(("blk.2.ffn_gate_inp_shexp.weight", 2048));
        let gated = flint_descriptor(&flint_parsed_with(flint_metadata(), &planted));
        assert!(
            gated.layers[2].ffn.shared_expert_gate,
            "blk.2 declares its own row"
        );
        assert!(
            gated
                .layers
                .iter()
                .enumerate()
                .all(|(layer, entry)| layer == 2 || !entry.ffn.shared_expert_gate),
            "no other block declares a gate"
        );
    }

    /// R5's sibling defect: a full layer's value source is a fact about the
    /// directory, not about `attention.shared_kv_layers`. flint declares no
    /// shared-KV count and ships `attn_v.weight` on all 49 blocks, so every
    /// layer projects its own value; dropping one block's row makes exactly
    /// that block share the key projection.
    #[test]
    fn flint_full_layers_project_their_own_value_from_the_directory() {
        let mut rows = flint_sink_tensors();
        rows.extend(flint_tensor_rows(".attn_v.weight"));
        let refs: Vec<(&str, u64)> = rows
            .iter()
            .map(|(name, width)| (name.as_str(), *width))
            .collect();
        let projected = flint_descriptor(&flint_parsed_with(flint_metadata(), &refs));
        assert_eq!(
            projected
                .layers
                .iter()
                .filter(|layer| layer.attention.value_source_kind == ValueSourceKind::ProjectedV)
                .count(),
            49,
            "every block carries its own attn_v row"
        );

        let without_block_zero: Vec<(&str, u64)> = refs
            .iter()
            .copied()
            .filter(|(name, _)| *name != "blk.0.attn_v.weight")
            .collect();
        let shared = flint_descriptor(&flint_parsed_with(flint_metadata(), &without_block_zero));
        assert_eq!(
            shared.layers[0].attention.value_source_kind,
            ValueSourceKind::SharedWithKey,
            "a full block with no attn_v row shares the key projection"
        );
        assert_eq!(
            shared.layers[1].attention.value_source_kind,
            ValueSourceKind::ProjectedV,
            "block 1 still has its own row"
        );
        assert_eq!(
            shared
                .layers
                .iter()
                .filter(|layer| layer.attention.value_source_kind == ValueSourceKind::SharedWithKey)
                .count(),
            1,
            "exactly the block whose row was removed"
        );
    }

    /// R3: the four header facts flint states as data, read back through the
    /// descriptor and the header carrier, with every absent key keeping the
    /// pre-change default.
    #[test]
    fn flint_reader_leading_dense_and_shared_() {
        let sinks = flint_sink_tensors();
        let parsed = flint_parsed_with(flint_metadata(), &flint_sink_refs(&sinks));
        let hparams = from_metadata(&parsed)
            .expect("the flint capture carries every key from_metadata reads");
        let descriptor = flint_descriptor(&parsed);

        if let Some(problem) = flint_dense_shared_problem(&hparams, &descriptor) {
            panic!("{problem}");
        }

        // one row only: the leaf is resolved per layer, not "any sink row sets
        // every layer"
        let partial: Vec<(String, u64)> = sinks.iter().take(1).cloned().collect();
        let parsed = flint_parsed_with(flint_metadata(), &flint_sink_refs(&partial));
        let descriptor = flint_descriptor(&parsed);
        assert!(
            descriptor.layers[0].attention.attention_sinks,
            "blk.0's own row declares layer 0's leaf"
        );
        assert!(
            descriptor.layers[1..]
                .iter()
                .all(|layer| !layer.attention.attention_sinks),
            "a layer with no row declares no leaf"
        );

        // the value table's other arms, under a profile whose own routing
        // function is sigmoid so the header's value is the only difference
        let refs = flint_sink_refs(&sinks);
        let mut absent = flint_metadata();
        absent.retain(|(key, _)| *key != "flint.expert_gating_func");
        for (label, value, expected) in [
            ("absent", None, ExpertGatingFunc::Sigmoid),
            ("zero", Some(0), ExpertGatingFunc::Sigmoid),
            ("softmax", Some(1), ExpertGatingFunc::Softmax),
            ("sigmoid", Some(2), ExpertGatingFunc::Sigmoid),
        ] {
            let mut metadata = absent.clone();
            if let Some(value) = value {
                metadata.push(("flint.expert_gating_func", MetadataValue::U32(value)));
            }
            let parsed = flint_parsed_with(metadata, &refs);
            let descriptor = flint_descriptor_with_profile(&parsed, "lfm2moe")
                .expect("a supported gating value builds");
            for layer in &descriptor.layers {
                assert_eq!(
                    layer.ffn.routed_gating, expected,
                    "expert_gating_func {label}: the header value decides the routing function"
                );
            }
        }

        let mut bare = flint_metadata();
        bare.retain(|(key, _)| {
            !matches!(
                *key,
                "flint.leading_dense_block_count"
                    | "flint.expert_shared_feed_forward_length"
                    | "flint.expert_gating_func"
                    | "flint.expert_shared_count"
            )
        });
        let parsed = flint_parsed_with(bare, &[]);
        let hparams = from_metadata(&parsed).expect("the four keys are optional");
        let descriptor = flint_descriptor(&parsed);

        assert_eq!(
            hparams.expert_shared_count, 0,
            "an absent shared-expert count is 0, not a guess"
        );
        assert_eq!(
            descriptor.leading_dense_block_count, 0,
            "a mixture-of-experts checkpoint with no leading dense count keeps the \
             pre-change default of no dense layers"
        );
        assert_eq!(
            descriptor.expert_shared_feed_forward, 0,
            "an absent shared-expert width is 0, not the dense width"
        );

        let profile_gating = family_profile("flint")
            .expect("flint is embedded")
            .ffn
            .routed_gating;
        for (index, layer) in descriptor.layers.iter().enumerate() {
            assert_eq!(
                layer.ffn.routed_gating, profile_gating,
                "layer {index}: an absent gating value keeps the profile's own function"
            );
            assert!(
                !layer.attention.attention_sinks,
                "layer {index}: no sink tensor row, no sink leaf"
            );
        }

        // the other arm of the fallback: a dense checkpoint has no leading
        // routed layers, which is spelled block_count
        let mut dense = flint_metadata();
        dense.retain(|(key, _)| {
            !matches!(
                *key,
                "flint.expert_count" | "flint.leading_dense_block_count"
            )
        });
        let parsed = flint_parsed_with(dense, &[]);
        let dense_descriptor = flint_descriptor(&parsed);
        assert_eq!(
            dense_descriptor.leading_dense_block_count, 49,
            "a checkpoint with no experts is dense on every layer"
        );
    }

    /// AC24 control: the same comparison must reject a mutated capture, or the
    /// header it trusts proves nothing.
    #[test]
    fn control_flint_reader_leading_dense_() {
        let sinks = flint_sink_tensors();
        let refs = flint_sink_refs(&sinks);
        let parsed = flint_parsed_with(flint_metadata(), &refs);
        let hparams = from_metadata(&parsed).expect("the capture reads");
        let descriptor = flint_descriptor(&parsed);
        assert_eq!(
            flint_dense_shared_problem(&hparams, &descriptor),
            None,
            "the captured header passes its own comparison"
        );

        for (key, value, names) in [
            ("flint.expert_shared_count", 3_u32, "expert_shared_count"),
            (
                "flint.leading_dense_block_count",
                1,
                "leading_dense_block_count",
            ),
            (
                "flint.expert_shared_feed_forward_length",
                512,
                "expert_shared_feed_forward",
            ),
            ("flint.expert_gating_func", 1, "routed_gating"),
        ] {
            let mut mutated = flint_metadata();
            for entry in &mut mutated {
                if entry.0 == key {
                    entry.1 = MetadataValue::U32(value);
                }
            }
            let parsed = flint_parsed_with(mutated, &refs);
            let hparams = from_metadata(&parsed).expect("a mutated value still reads");
            let descriptor = flint_descriptor(&parsed);
            let problem = flint_dense_shared_problem(&hparams, &descriptor)
                .unwrap_or_else(|| panic!("{key} = {value} must fail the comparison"));
            assert!(
                problem.contains(names),
                "the failure names {names}, got: {problem}"
            );
        }

        // llama.cpp defines 3 (SOFTMAX_WEIGHT) and 4 (SQRT_SOFTPLUS); this
        // engine has no such function, so naming one is refused rather than
        // served with the profile's default
        let mut unsupported = flint_metadata();
        for entry in &mut unsupported {
            if entry.0 == "flint.expert_gating_func" {
                entry.1 = MetadataValue::U32(3);
            }
        }
        let parsed = flint_parsed_with(unsupported, &refs);
        let error = flint_descriptor_error(&parsed);
        assert!(
            matches!(
                error,
                proxima_tensor::error::TensorError::GgufUnsupportedExpertGating { value: 3 }
            ),
            "an unimplemented gating function must be refused by name, got {error:?}"
        );

        let parsed = flint_parsed_with(flint_metadata(), &[]);
        let hparams = from_metadata(&parsed).expect("the capture reads without tensor rows");
        let descriptor = flint_descriptor(&parsed);
        let problem = flint_dense_shared_problem(&hparams, &descriptor)
            .expect("dropping the sink rows must fail the comparison");
        assert!(
            problem.contains("declares no sink leaf"),
            "the failure names the missing sink leaf, got: {problem}"
        );
    }

    /// R2's other half: the committed gemma4 fixture reports its own explicit
    /// widths (256 sliding / 512 full), not the flint fallback. The fixture is
    /// the KV-relevant subset of that header, so this is the reader that carries
    /// every key it has.
    #[test]
    fn sliding_pattern_header_reports_the_gemma4_fixture_widths() {
        let layers = kv_layers_from_metadata(&e2b_header())
            .expect("the e2b header carries every key kv_layers_from_metadata reads");

        assert_eq!(layers.len(), 15, "35 blocks minus 20 shared-KV layers");
        for (index, &(kv_heads, head_dim, window)) in layers.iter().enumerate() {
            assert_eq!(kv_heads, 1);
            if index % 5 == 4 {
                assert_eq!((head_dim, window), (512, None), "full layer {index}");
            } else {
                assert_eq!(
                    (head_dim, window),
                    (256, Some(512)),
                    "sliding layer {index}"
                );
            }
        }
    }

    /// AC23 control: the same comparison the test above trusts must reject a
    /// mutated capture, or it proves nothing about the real one.
    #[test]
    fn control_rejects_mutated_sliding_pattern_header() {
        let captured = flint_layer_pattern("flint.attention.sliding_window_pattern");
        let parsed = flint_parsed(flint_metadata());
        let hparams = from_metadata(&parsed).expect("the capture reads");
        let descriptor = flint_descriptor(&parsed);
        assert_eq!(
            sliding_pattern_header_problem(&hparams, &descriptor, &captured),
            None,
            "the captured header passes its own comparison"
        );

        // both widths move together, so the descriptor still builds and the
        // header comparison is what rejects the mutant
        let mut narrowed = flint_metadata();
        for entry in &mut narrowed {
            if entry.0 == "flint.attention.key_length" || entry.0 == "flint.attention.value_length"
            {
                entry.1 = MetadataValue::U32(64);
            }
        }
        let parsed = flint_parsed(narrowed);
        let hparams = from_metadata(&parsed).expect("a narrower head width still reads");
        let descriptor = flint_descriptor(&parsed);
        let problem = sliding_pattern_header_problem(&hparams, &descriptor, &captured)
            .expect("a mutated key_length must fail the width comparison");
        assert!(
            problem.starts_with("key_length = 64"),
            "the failure names the mutated width, got: {problem}"
        );

        let mut flipped = flint_metadata();
        for entry in &mut flipped {
            if entry.0 == "flint.attention.sliding_window_pattern"
                && let MetadataValue::Array(MetadataArray::Bool(pattern)) = &mut entry.1
            {
                pattern[0] = !pattern[0];
            }
        }
        let parsed = flint_parsed(flipped);
        let hparams = from_metadata(&parsed).expect("a flipped schedule entry still reads");
        let descriptor = flint_descriptor(&parsed);
        let problem = sliding_pattern_header_problem(&hparams, &descriptor, &captured)
            .expect("a flipped schedule entry must fail the capture comparison");
        assert!(
            problem.contains("sliding_window_pattern"),
            "the failure names the schedule it compared, got: {problem}"
        );
    }

    /// R4's own comparison: every layer's rope selection against the capture's
    /// 49-entry array, as one predicate so the control proves the same
    /// comparison rejects a mutated header.
    fn flint_rope_pattern_problem(descriptor: &ModelDescriptor, captured: &[bool]) -> Option<String> {
        if descriptor.layers.len() != captured.len() {
            return Some(format!(
                "{} layers, expected one per captured entry ({})",
                descriptor.layers.len(),
                captured.len()
            ));
        }
        for (index, layer) in descriptor.layers.iter().enumerate() {
            if layer.attention.rope != captured[index] {
                return Some(format!(
                    "layer {index} rope {}, expected {} from the captured rope_pattern",
                    layer.attention.rope, captured[index]
                ));
            }
        }
        None
    }

    /// R4: `{arch}.attention.rope_pattern` selects rope per layer from data. The
    /// capture answers both ways -- its sliding layers rotate, its full ones do
    /// not -- so a reader that returned one constant for the whole schedule
    /// cannot pass; the lowered program is the other half of the same fact, with
    /// the full layers declaring no table leaf at all.
    #[test]
    fn flint_reader_rope_pattern_selects_rope_per_layer() {
        let captured = flint_layer_pattern("flint.attention.rope_pattern");
        let parsed = flint_parsed(flint_metadata());
        let descriptor = flint_descriptor(&parsed);

        if let Some(problem) = flint_rope_pattern_problem(&descriptor, &captured) {
            panic!("{problem}");
        }
        assert!(
            captured.contains(&true) && captured.contains(&false),
            "the captured pattern states both answers, or the per-layer assertion is vacuous"
        );

        // the descriptor's answer reaches the engine: the sliding layers bind
        // their own table, and no full-layer table is declared by anything
        let forward =
            build_forward(&descriptor).expect("the flint schedule builds the padded cached program");
        let mut leaves: Vec<&str> = forward
            .program
            .iter()
            .filter_map(|op| match op {
                Op::Input {
                    name: Some(name), ..
                } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        leaves.sort_unstable();
        leaves.dedup();
        assert!(
            leaves.contains(&"rope_cos_swa") && leaves.contains(&"rope_sin_swa"),
            "the sliding layers rotate, so their table is bound: {leaves:?}"
        );
        assert!(
            !leaves.contains(&"rope_cos") && !leaves.contains(&"rope_sin"),
            "every full layer is rope-off, so no full-layer table may be declared: {leaves:?}"
        );

        // llama.cpp seeds every entry to 1 before the key is read, so an absent
        // pattern leaves the schedule it had before the key was data
        let mut absent = flint_metadata();
        absent.retain(|(key, _)| *key != "flint.attention.rope_pattern");
        let parsed = flint_parsed(absent);
        let descriptor = flint_descriptor(&parsed);
        assert!(
            descriptor
                .layers
                .iter()
                .all(|layer| layer.attention.rope),
            "a checkpoint with no rope_pattern rotates every layer"
        );
    }

    /// AC25 control: the same comparison must reject a mutated capture, or the
    /// header it trusts proves nothing.
    #[test]
    fn control_rejects_mutated_rope_pattern() {
        let captured = flint_layer_pattern("flint.attention.rope_pattern");
        let parsed = flint_parsed(flint_metadata());
        let descriptor = flint_descriptor(&parsed);
        assert_eq!(
            flint_rope_pattern_problem(&descriptor, &captured),
            None,
            "the captured header passes its own comparison"
        );

        // one flipped entry in the header's own array must be visible
        let mut flipped = flint_metadata();
        for entry in &mut flipped {
            if entry.0 == "flint.attention.rope_pattern"
                && let MetadataValue::Array(MetadataArray::Bool(pattern)) = &mut entry.1
            {
                pattern[5] = !pattern[5];
            }
        }
        let parsed = flint_parsed(flipped);
        let descriptor = flint_descriptor(&parsed);
        let problem = flint_rope_pattern_problem(&descriptor, &captured)
            .expect("a flipped rope entry must fail the capture comparison");
        assert!(
            problem.contains("layer 5 rope"),
            "the failure names the layer it compared, got: {problem}"
        );

        // a short array is refused by name rather than padded or truncated
        let mut short = flint_metadata();
        for entry in &mut short {
            if entry.0 == "flint.attention.rope_pattern"
                && let MetadataValue::Array(MetadataArray::Bool(pattern)) = &mut entry.1
            {
                pattern.truncate(48);
            }
        }
        let parsed = flint_parsed(short);
        let error = flint_descriptor_error(&parsed);
        assert!(
            matches!(
                error,
                proxima_tensor::error::TensorError::GgufPerLayerLengthMismatch {
                    ref key,
                    expected: 49,
                    found: 48,
                } if key.as_str() == "flint.attention.rope_pattern"
            ),
            "a 48-entry rope pattern must be refused by name, got {error:?}"
        );
    }

}
