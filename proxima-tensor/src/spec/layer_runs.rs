use alloc::vec::Vec;

use serde::de::Error;
use serde::{Deserialize, Deserializer};

use super::{LayerAttentionConfig, LayerFfnConfig, LayerKind, LayerSchedule};

const MAX_LAYERS: usize = u16::MAX as usize;

const fn once() -> u32 {
    1
}

/// One `[[layers]]` entry as a person writes it: a layer (`kind`, `attention`,
/// `ffn`) or a `pattern` of entries, either one repeated `repeat` times. A plain
/// serialized `LayerSchedule` is the `repeat = 1` layer, so expanded files still load.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerRun {
    #[serde(default = "once")]
    repeat: u32,
    kind: Option<LayerKind>,
    attention: Option<LayerAttentionConfig>,
    ffn: Option<LayerFfnConfig>,
    #[serde(default)]
    pattern: Vec<LayerRun>,
}

impl LayerRun {
    fn expand<E: Error>(self, layers: &mut Vec<LayerSchedule>) -> Result<(), E> {
        let mut unit = Vec::new();
        match (self.kind, self.attention, self.ffn, self.pattern.is_empty()) {
            (Some(kind), Some(attention), Some(ffn), true) => unit.push(LayerSchedule { kind, attention, ffn }),
            (None, None, None, false) => self.pattern.into_iter().try_for_each(|run| run.expand::<E>(&mut unit))?,
            _ => return Err(E::custom("a layers entry is one layer (kind, attention, ffn) or a pattern, not both and not neither")),
        }
        let total = unit.len().saturating_mul(self.repeat as usize).saturating_add(layers.len());
        if total > MAX_LAYERS {
            return Err(E::custom(format_args!("layers expand to {total} entries, more than the {MAX_LAYERS} a model may hold")));
        }
        (0..self.repeat).for_each(|_| layers.extend(unit.iter().cloned()));
        Ok(())
    }
}

pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<LayerSchedule>, D::Error> {
    let mut layers = Vec::new();
    Vec::<LayerRun>::deserialize(deserializer)?
        .into_iter()
        .try_for_each(|run| run.expand::<D::Error>(&mut layers))?;
    Ok(layers)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::format;
    use alloc::string::String;

    use super::super::{ModelDescriptor, RopePairing};

    const HEADER: &str = r#"
vocab = 262144
embedding = 1536
feed_forward = 6144
expert_feed_forward = 0
query_heads = 8
block_count = 6
expert_count = 0
expert_used_count = 0
leading_dense_block_count = 6
l_cache = 0
cache_strategy = "TwoRange"
sliding_kv_ring = false
qk_norm = false
qkv_biases = false
paired_gate_up_reduce = false
fused_qkv_reduce = false
head_repeats = 1
last_row_only = true
"#;

    const SLIDING: &str = r#"
kind = "Attention"

[layers.attention]
head_dim = 256
kv_heads = 1
mask_window = 512
value_source_kind = "ProjectedV"
key_source_kind = "ProjectedK"
value_norm = true

[layers.attention.rope_table]
cos_name = "rope_cos_swa"
sin_name = "rope_sin_swa"

[layers.attention.rope_pairing.SplitHalf]
pairs = 128

[layers.attention.score_scale]
Unscaled = []

[layers.ffn]
post_attention_norm = true
combination = "Exclusive"
output_scale = true
routed_gating = "Softmax"
routed_expert_bias = false
activation = "GeluTanh"
exclusive_dense_post_norm = false
ple = true
"#;

    fn full_layer() -> String {
        SLIDING
            .replace("head_dim = 256", "head_dim = 512")
            .replace("mask_window = 512\n", "")
            .replace("rope_cos_swa", "rope_cos")
            .replace("rope_sin_swa", "rope_sin")
            .replace("pairs = 128", "pairs = 256")
    }

    fn parse(layers: &str) -> Result<ModelDescriptor, toml::de::Error> {
        toml::from_str(&format!("{HEADER}{layers}"))
    }

    fn window_of(descriptor: &ModelDescriptor) -> alloc::vec::Vec<Option<u32>> {
        descriptor.layers.iter().map(|layer| layer.attention.mask_window).collect()
    }

    #[test]
    fn a_repeated_layer_expands_to_that_many_identical_blocks() {
        let text = format!("[[layers]]\nrepeat = 6\n{SLIDING}");

        let descriptor = parse(&text).expect("a repeated layer parses");

        assert_eq!(descriptor.layers.len(), 6);
        assert!(descriptor.layers.iter().all(|layer| *layer == descriptor.layers[0]));
        assert_eq!(descriptor.layers[0].attention.rope_pairing, RopePairing::SplitHalf { pairs: 128 });
    }

    #[test]
    fn a_pattern_repeats_in_order_like_gemma4_sliding_then_full() {
        let text = format!(
            "[[layers]]\nrepeat = 2\n[[layers.pattern]]\nrepeat = 2\n{sliding}\n[[layers.pattern]]\n{full}",
            sliding = SLIDING.replace("[layers.", "[layers.pattern."),
            full = full_layer().replace("[layers.", "[layers.pattern."),
        );

        let descriptor = parse(&text).expect("a pattern parses");

        assert_eq!(
            window_of(&descriptor),
            [Some(512), Some(512), None, Some(512), Some(512), None]
        );
    }

    #[test]
    fn the_expanded_form_a_dump_produces_still_loads_and_equals_the_runs() {
        let compact = parse(&format!("[[layers]]\nrepeat = 3\n{SLIDING}[[layers]]\n{}", full_layer())).expect("runs parse");
        let expanded_text = toml::to_string(&compact).expect("a descriptor serializes");

        let restored: ModelDescriptor = toml::from_str(&expanded_text).expect("the expanded dump parses");

        assert_eq!(restored.layers.len(), 4);
        assert_eq!(restored, compact);
    }

    #[test]
    fn an_entry_that_is_both_a_layer_and_a_pattern_is_refused() {
        let text = format!("[[layers]]\n{SLIDING}[[layers.pattern]]\n{}", SLIDING.replace("[layers.", "[layers.pattern."));

        let error = parse(&text).expect_err("a layer and a pattern in one entry is ambiguous");

        assert!(error.to_string().contains("one layer"), "got: {error}");
    }

    #[test]
    fn an_entry_with_neither_a_layer_nor_a_pattern_is_refused() {
        let error = parse("[[layers]]\nrepeat = 4\n").expect_err("an empty run has nothing to repeat");

        assert!(error.to_string().contains("one layer"), "got: {error}");
    }

    #[test]
    fn an_unknown_key_in_a_layer_entry_is_an_error_not_a_silent_default() {
        let error = parse(&format!("[[layers]]\nrepeats = 2\n{SLIDING}")).expect_err("a misspelled repeat must not load");

        assert!(error.to_string().contains("unknown field"), "got: {error}");
    }

    #[test]
    fn a_repeat_that_would_exhaust_memory_is_refused_before_it_allocates() {
        let error = parse(&format!("[[layers]]\nrepeat = 4000000000\n{SLIDING}")).expect_err("four billion layers must not expand");

        assert!(error.to_string().contains("more than"), "got: {error}");
    }
}
