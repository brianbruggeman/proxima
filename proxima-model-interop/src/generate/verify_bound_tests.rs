// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

use alloc::vec::Vec;

use conflaguration::Validate;
use proxima_gguf::pipe::parse_complete;
use proxima_tensor::shape;
use proxima_tensor::spec::{
    FamilyProfile, LayerKind, ModelDescriptor, ScheduleSource, build_forward,
    gqa_descriptor_from_shape,
};

use super::LoadedModel;
use super::chunked_prefill_tests::gemma4_checkpoint;
use super::recurrent_checkpoint_fixture::{VOCAB, recurrent_checkpoint};
use crate::lowering::{KvLayout, header_descriptor};
use crate::profiles::{FAMILY_PROFILES, family_profile};

const VERIFY_WIDTH: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Bound,
    Refused,
}

const SHIPPED: [(&str, Outcome); 14] = [
    ("gemma4", Outcome::Bound),
    ("llama", Outcome::Bound),
    ("mistral", Outcome::Bound),
    ("mixtral", Outcome::Bound),
    ("granitemoe", Outcome::Bound),
    ("qwen2", Outcome::Bound),
    ("qwen3", Outcome::Bound),
    ("qwen3moe", Outcome::Bound),
    ("qwen3_moe", Outcome::Bound),
    ("qwen35", Outcome::Bound),
    ("qwen35moe", Outcome::Bound),
    ("lfm2", Outcome::Refused),
    ("lfm2moe", Outcome::Refused),
    ("ouro", Outcome::Bound),
];

fn header_of(bytes: &[u8]) -> ModelDescriptor {
    let parsed = parse_complete(bytes).expect("the synthetic checkpoint parses");
    header_descriptor(&parsed, KvLayout::Full).expect("the family's header reader reads its header")
}

fn uniform_descriptor(profile: &FamilyProfile, outcome: Outcome) -> ModelDescriptor {
    let descriptor = gqa_descriptor_from_shape(
        VOCAB as u32,
        16,
        32,
        2,
        1,
        8,
        2,
        0,
        0,
        false,
        false,
        false,
        false,
        profile,
    );
    let mut descriptor = profile.recurring(descriptor, 2);
    if outcome == Outcome::Refused {
        descriptor.layers[0].kind = LayerKind::ShortConv;
    }
    descriptor
}

fn descriptor_for(family: &str, outcome: Outcome) -> ModelDescriptor {
    let profile = family_profile(family).expect("every shipped family has an embedded profile");
    match profile.schedule_source {
        ScheduleSource::Uniform => uniform_descriptor(&profile, outcome),
        ScheduleSource::SlidingPattern => header_of(&gemma4_checkpoint()),
        ScheduleSource::RecurrentInterval => header_of(&recurrent_checkpoint(false)),
        ScheduleSource::RecurrentRoutedInterval => header_of(&recurrent_checkpoint(true)),
    }
}

fn armed(descriptor: ModelDescriptor) -> ModelDescriptor {
    ModelDescriptor {
        speculative_verify: true,
        last_row_only: true,
        ..descriptor
    }
}

fn assert_bound(family: &str, decode: &ModelDescriptor) {
    let verify = decode
        .verify_at_width(VERIFY_WIDTH as u32)
        .unwrap_or_else(|| panic!("{family} arms a verify descriptor"));
    let forward = build_forward(&verify)
        .unwrap_or_else(|error| panic!("{family} lowers its verify program: {error:?}"));
    let shapes = shape::infer(&forward.program, &[VERIFY_WIDTH, 0, 0])
        .unwrap_or_else(|error| panic!("{family} verify program infers at width 3: {error:?}"));
    assert_eq!(
        shapes.of(forward.logits),
        &[VERIFY_WIDTH, u64::from(decode.vocab)],
        "{family} keeps one logits row per new position"
    );
}

fn assert_refused(family: &str, decode: &ModelDescriptor) {
    assert_eq!(
        decode.verify_at_width(VERIFY_WIDTH as u32),
        None,
        "{family} has no verify descriptor"
    );
    let refusal = decode
        .validate()
        .expect_err("a layer that cannot rewind by truncation cannot be armed for verify");
    assert!(
        refusal.to_string().contains("speculative_verify"),
        "{family}: {refusal}"
    );
}

#[test]
fn verify_program_bound_for_every_architecture() {
    let embedded: Vec<&str> = FAMILY_PROFILES.iter().map(|(family, _)| *family).collect();
    let expected: Vec<&str> = SHIPPED.iter().map(|(family, _)| *family).collect();
    assert_eq!(
        expected, embedded,
        "SHIPPED names every embedded profile, in order"
    );

    for (family, outcome) in SHIPPED {
        let decode = armed(descriptor_for(family, outcome));
        match outcome {
            Outcome::Bound => assert_bound(family, &decode),
            Outcome::Refused => assert_refused(family, &decode),
        }
    }
    std::println!(
        "archs_asserted = {} registry = {}",
        SHIPPED.len(),
        FAMILY_PROFILES.len()
    );
}

#[test]
fn the_loaded_gemma4_checkpoint_binds_its_verify_program() {
    let bytes = gemma4_checkpoint();
    let parsed = parse_complete(&bytes).expect("the gemma4 fixture parses");

    let model = LoadedModel::load(&parsed, &bytes).expect("the gemma4 fixture loads");

    assert!(model.speculative_verify_program.is_some());
}
