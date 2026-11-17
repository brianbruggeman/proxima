// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

use proxima_gguf::pipe::parse_complete;
use proxima_tensor::spec::ModelDescriptor;

use super::recurrent_checkpoint_fixture::{dense_checkpoint_stating_experts, recurrent_checkpoint};
use crate::error::InteropError;
use crate::lowering::{KvLayout, header_descriptor};

fn read(bytes: &[u8]) -> Result<ModelDescriptor, InteropError> {
    let parsed = parse_complete(bytes).expect("the synthetic checkpoint parses");
    header_descriptor(&parsed, KvLayout::Full)
}

#[test]
fn a_routed_header_with_expert_keys_reads_through_the_routed_reader() {
    let descriptor = read(&recurrent_checkpoint(true)).expect("qwen35moe states experts and its profile is routed");

    assert!(descriptor.expert_count > 0, "routed reader keeps the header's experts");
}

#[test]
fn a_dense_header_without_expert_keys_reads_through_the_dense_reader() {
    let descriptor = read(&recurrent_checkpoint(false)).expect("qwen35 states no experts and its profile is dense");

    assert_eq!(descriptor.expert_count, 0);
}

#[test]
fn a_dense_profile_with_expert_count_in_the_header_names_the_key_and_the_combination() {
    let refusal = read(&dense_checkpoint_stating_experts()).expect_err("header and profile disagree");

    assert!(
        matches!(refusal, InteropError::HeaderExpertKeysContradictProfile { value: 2, .. }),
        "{refusal:?}"
    );
    let message = refusal.to_string();
    assert!(message.contains("\"qwen35.expert_count\" = 2"), "{message}");
    assert!(message.contains("Exclusive"), "{message}");
    assert!(message.contains("\"qwen35\""), "{message}");
}
