use serde::{Deserialize, Serialize};

use super::{Decode, Input, Stack};

/// A whole network: what enters, the body that transforms it, and how it runs.
/// `heads` and `embed` arrive with the cards that define them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "config", derive(bon::Builder, conflaguration::Settings))]
#[cfg_attr(feature = "config", settings(prefix = "PROXIMA_NETWORK"))]
#[serde(deny_unknown_fields)]
pub struct Network {
    /// The first block's input.
    #[cfg_attr(feature = "config", setting(skip))]
    pub input: Input,
    /// The transformation from `input` to the network's outputs.
    #[cfg_attr(feature = "config", setting(skip))]
    pub body: Stack,
    /// Execution options; defaults when absent from config.
    #[serde(default)]
    #[cfg_attr(feature = "config", setting(nested))]
    #[cfg_attr(feature = "config", builder(default))]
    pub decode: Decode,
}

#[cfg(feature = "config")]
impl conflaguration::Validate for Network {
    fn validate(&self) -> conflaguration::Result<()> {
        conflaguration::Validate::validate(&self.input)?;
        conflaguration::Validate::validate(&self.body)?;
        conflaguration::Validate::validate(&self.decode)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::super::{Block, Decode, Input, Norm, NormKind, Stack};
    use super::Network;
    use crate::spec::ModelDescriptor;

    const FLAT_HEADER: &str = r#"
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
cache_strategy = "Cached"
cache_mask = "Padded"
sliding_kv_ring = false
qk_norm = false
qkv_biases = false
paired_gate_up_reduce = false
fused_qkv_reduce = false
head_repeats = 1
last_row_only = true
"#;

    const FLAT_LAYER: &str = r#"
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

    #[test]
    fn network_default_is_none_and_existing_descriptor_toml_still_loads() {
        let text = format!("{FLAT_HEADER}[[layers]]\nrepeat = 6\n{FLAT_LAYER}");
        let descriptor: ModelDescriptor = toml::from_str(&text).expect(
            "the flat descriptor toml every existing config is written in must still parse",
        );
        assert_eq!(descriptor.network, None);
    }

    #[test]
    fn network_roundtrips_through_toml() {
        let network = Network {
            input: Input::Features { width: 8 },
            body: Stack {
                blocks: vec![Block::Norm(Norm {
                    kind: NormKind::Rms,
                    eps: 1e-5,
                    scaled: true,
                    group_width: None,
                })],
            },
            decode: Decode::default(),
        };
        let text = toml::to_string(&network).expect("a network with one norm serializes to toml");
        assert!(
            text.contains("norm = \"rms\""),
            "the norm kind must be spelled `norm`, not `kind`, under the block tag: {text}"
        );
        let parsed: Network = toml::from_str(&text).expect("the serialized network parses back");
        assert_eq!(parsed, network);
    }

    #[test]
    fn unknown_network_field_is_rejected() {
        let text = "bogus = 1\ninput = { kind = \"features\", width = 8 }\n[body]\nblocks = []\n";
        let error = toml::from_str::<Network>(text)
            .expect_err("an unknown top-level network key must not be silently dropped");
        assert!(
            error.to_string().contains("bogus"),
            "the error must name the unknown key: {error}"
        );
    }
}
