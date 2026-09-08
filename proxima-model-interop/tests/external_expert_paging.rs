//! Proves the decode loop actually evaluates through `LoadedModel`'s own
//! [`proxima_model_interop::expert_slab::ExpertSlab`] rather than reading
//! `weights.packed` directly: paging an expert's bytes BETWEEN two decode
//! steps changes what a later step reads (the epoch bumps, and the swap is
//! rejected while a step is running), the exact contract
//! `crate::expert_slab::ExpertSlab`'s own module doc names.
//!
//! Uses [`support::checkpoint_bytes_moe`] with `weight_codec: Q4_K` (not the
//! capability matrix's own `F32` MoE cell) specifically so the stacked
//! `_exps.weight` tensors bind through [`proxima_model_interop::bind::build_expert_slab`]'s
//! zero-copy [`proxima_tensor::cpu::QuantizedBlock`] arm -- an `F32` MoE
//! stack has no [`proxima_model_interop::PackedOwnedKind`] tag at all
//! (that enum's own doc), so it is never bound into the slab, and would
//! prove nothing about paging.
//!
//! The fixture's own [`support::push_output_projection`] zeroes every
//! output-projection row so every decode step's greedy id is `0`
//! regardless of which expert's bytes actually ran (that all-zero
//! projection is deliberate fixture design -- see its own doc -- so a
//! broken upstream codec still surfaces as a shape/NaN/panic, not a wrong
//! id). That means THIS fixture cannot show a paged expert changing the
//! decoded token id; what it proves instead is the plumbing a wrong id
//! would otherwise depend on: the epoch bump, the mid-step rejection, and
//! that decoding still runs to completion, unchanged in shape, after a
//! page.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[allow(dead_code)]
mod support;

use proxima_gguf::GgmlType;
use proxima_model_interop::{InteropError, LoadedModel, PackedOwnedKind};
use proxima_primitives::pipe::Pipe;

const PROMPT: &str = "The capital of France is";

/// `blk.0.ffn_gate_exps.weight` -- the first [`support::checkpoint_bytes_moe`]
/// projection [`proxima_model_interop::bind::build_expert_slab`]'s own
/// enumeration order binds, so this is always slab SITE `0` regardless of
/// `support::BLOCK_COUNT` (see [`LoadedModel::page_expert`]'s own doc on
/// what `layer` indexes there).
const GATE_SITE: usize = 0;

fn load_moe_fixture() -> LoadedModel<'static> {
    let file_bytes: &'static [u8] =
        Box::leak(support::checkpoint_bytes_moe(GgmlType::Q4_K, support::EXPERT_COUNT, support::EXPERT_USED_COUNT).into_boxed_slice());
    let parsed: &'static proxima_gguf::pipe::ParsedGguf =
        Box::leak(Box::new(proxima_gguf::parse_complete(file_bytes).expect(
            "parses the synthetic Q4_K MoE checkpoint",
        )));
    LoadedModel::load(parsed, file_bytes).expect("loads the synthetic Q4_K MoE checkpoint")
}

/// One expert's own `[out_dim, in_dim]` Q4_K-encoded weight slab, freshly
/// generated (not required to match the fixture's own original bytes --
/// [`LoadedModel::page_expert`] accepts any correctly-shaped, correctly-
/// coded buffer, exactly the same contract a real residency policy has).
fn one_expert_q4k_bytes(out_dim: u32, in_dim: u32, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let values: Vec<f32> = (0..(out_dim as usize * in_dim as usize))
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    support::encode_weights(GgmlType::Q4_K, &values)
}

/// Paging an expert BETWEEN two decode steps bumps its epoch and the
/// decode loop keeps running, unchanged in shape, on both sides of the
/// swap -- the CPU decode path (`LoadedModel::run_decode_loop_observed_seeded`)
/// really does call `ExpertSlab::begin_step`/`sources_for_step`/`end_step`
/// around every evaluation instead of reading `weights.packed` unconditionally.
#[proxima::test]
async fn paging_an_expert_between_steps_bumps_its_epoch_and_decode_continues() {
    let model = load_moe_fixture();

    assert_eq!(
        model.expert_epoch(GATE_SITE, 0),
        Some(0),
        "a freshly loaded checkpoint's slab starts every bound expert at epoch 0"
    );

    let (first_ids, _first_text, _stopped) = Pipe::call(&model, (PROMPT.to_string(), 2))
        .await
        .expect("the first two decode steps run before any paging");
    assert_eq!(first_ids.len(), 2, "an unbounded budget runs the full 2 steps");

    let paged_bytes = one_expert_q4k_bytes(support::FEED_FORWARD, support::EMBEDDING, 7);
    let epoch = model
        .page_expert(
            GATE_SITE,
            0,
            PackedOwnedKind::Q4K,
            &paged_bytes,
            support::FEED_FORWARD,
            support::EMBEDDING,
        )
        .expect("paging between two `Pipe::call`s -- no step is running -- succeeds");
    assert_eq!(epoch, 1, "the first page bumps epoch 0 -> 1");
    assert_eq!(model.expert_epoch(GATE_SITE, 0), Some(1));

    let (second_ids, second_text, stopped_by_eos) = Pipe::call(&model, (PROMPT.to_string(), 2))
        .await
        .expect("decoding after a page runs exactly as it did before one");
    assert_eq!(
        second_ids.len(),
        if stopped_by_eos { second_ids.len() } else { 2 },
        "paging must not change how many tokens a fresh call produces"
    );
    assert_eq!(
        second_ids, first_ids,
        "this fixture's own all-zero output projection makes every greedy id 0 \
         regardless of which expert's bytes ran (see this file's own module doc) -- \
         so the paged run's ids equal the unpaged run's, proving the swap did not \
         corrupt the forward pass rather than proving it changed the routed content"
    );
    assert!(!second_text.is_empty(), "a non-empty id sequence decodes to non-empty text");
}

/// [`LoadedModel::evict_expert`] then reselecting that expert's layer
/// surfaces a typed error instead of reading stale or garbage bytes --
/// [`proxima_model_interop::expert_slab::ExpertSlab::sources_for_step`]'s
/// own doc: an absent expert surfaces as
/// `proxima_tensor::TensorError::GatherIndexOutOfRange` from the gather,
/// wrapped into an [`InteropError`] by the same `?` every other tensor
/// error in this decode loop already propagates through. Every expert of
/// `GATE_SITE`'s layer is evicted (not just one) so this is true
/// regardless of which of `support::EXPERT_USED_COUNT` experts the
/// fixture's router happens to select for this prompt.
#[proxima::test]
async fn evicting_every_expert_of_a_layer_fails_the_next_decode() {
    let model = load_moe_fixture();

    for expert in 0..support::EXPERT_COUNT as usize {
        model
            .evict_expert(GATE_SITE, expert)
            .expect("every fixture expert index is in range for a freshly loaded slab");
        assert_eq!(
            model.expert_epoch(GATE_SITE, expert),
            None,
            "an evicted expert reports no epoch"
        );
    }

    let result = Pipe::call(&model, (PROMPT.to_string(), 2)).await;
    assert!(
        result.is_err(),
        "decoding through a layer with every expert evicted must fail, got {result:?}"
    );
}

/// [`LoadedModel::page_expert`]/[`LoadedModel::evict_expert`] reject an
/// out-of-range `(layer, expert)` pair with the SAME typed error
/// [`proxima_model_interop::expert_slab::ExpertSlab::page_expert`]'s own
/// doc names, proving the public forward changes nothing about that
/// contract.
#[proxima::test]
async fn paging_an_out_of_range_expert_index_is_rejected() {
    let model = load_moe_fixture();

    let result = model.page_expert(
        GATE_SITE,
        support::EXPERT_COUNT as usize,
        PackedOwnedKind::Q4K,
        &one_expert_q4k_bytes(support::FEED_FORWARD, support::EMBEDDING, 11),
        support::FEED_FORWARD,
        support::EMBEDDING,
    );

    assert!(
        matches!(
            result,
            Err(InteropError::ExpertSlabIndexOutOfRange { layer: GATE_SITE, .. })
        ),
        "an out-of-range expert index must be rejected, got {result:?}"
    );
}
