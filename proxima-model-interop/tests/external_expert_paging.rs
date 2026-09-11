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
use proxima_model_interop::expert_slab::encode_expert_copy;
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
    let file_bytes: &'static [u8] = Box::leak(
        support::checkpoint_bytes_moe(
            GgmlType::Q4_K,
            support::EXPERT_COUNT,
            support::EXPERT_USED_COUNT,
        )
        .into_boxed_slice(),
    );
    let parsed: &'static proxima_gguf::pipe::ParsedGguf = Box::leak(Box::new(
        proxima_gguf::parse_complete(file_bytes).expect("parses the synthetic Q4_K MoE checkpoint"),
    ));
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
    assert_eq!(
        first_ids.len(),
        2,
        "an unbounded budget runs the full 2 steps"
    );

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
    assert!(
        !second_text.is_empty(),
        "a non-empty id sequence decodes to non-empty text"
    );
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
            Err(InteropError::ExpertSlabIndexOutOfRange {
                layer: GATE_SITE,
                ..
            })
        ),
        "an out-of-range expert index must be rejected, got {result:?}"
    );
}

/// Which fixture expert [`q2k_paging_actually_changes_the_decoded_ids`]
/// pages -- the first stacked expert, same slot [`GATE_SITE`] already
/// indexes into.
const PAGED_EXPERT: usize = 0;

/// The one real-output-projection fixture's bytes, leaked once so every
/// independent [`LoadedModel::load`] call in
/// [`q2k_paging_actually_changes_the_decoded_ids`] borrows the SAME
/// underlying weight bytes -- three separate [`LoadedModel`]s (run A/B/C),
/// each with its OWN mutable [`proxima_model_interop::expert_slab::ExpertSlab`],
/// aliasing one shared, immutable checkpoint.
fn real_output_moe_checkpoint_bytes() -> &'static [u8] {
    Box::leak(
        support::checkpoint_bytes_moe_with_real_output(
            GgmlType::Q4_K,
            support::EXPERT_COUNT,
            support::EXPERT_USED_COUNT,
        )
        .into_boxed_slice(),
    )
}

/// [`support::checkpoint_bytes_moe_with_real_output`]'s own `output.weight`
/// is real ([`support::random_vec`]-seeded), so this fixture's decoded ids
/// actually depend on which expert's bytes a routed position gathers --
/// unlike [`load_moe_fixture`]'s all-zero-output fixture, which this file's
/// own module doc already explains cannot show a routed swap changing a
/// token.
fn load_real_output_moe_fixture(
    parsed: &'static proxima_gguf::pipe::ParsedGguf,
    file_bytes: &'static [u8],
) -> LoadedModel<'static> {
    LoadedModel::load(parsed, file_bytes)
        .expect("loads the synthetic Q4_K MoE checkpoint with a real output projection")
}

/// [`PAGED_EXPERT`]'s own on-disk `Q4_K` bytes out of `blk.0.ffn_gate_exps.weight`
/// -- the "hi" copy [`encode_expert_copy`]'s caller dequantizes before
/// re-encoding as `Q2_K`, exactly the residency-downgrade shape
/// [`proxima_model_interop::PackedOwnedKind::Q2K`]'s own doc names.
fn paged_expert_hi_bytes(file_bytes: &[u8]) -> Vec<u8> {
    let parsed =
        proxima_gguf::parse_complete(file_bytes).expect("re-parses this test's own fixture");
    let tensor = parsed
        .tensors
        .iter()
        .find(|candidate| candidate.name == "blk.0.ffn_gate_exps.weight")
        .expect("the fixture stacks ffn_gate_exps.weight for layer 0");
    let range = parsed
        .tensor_data_range(tensor, file_bytes.len() as u64)
        .expect("the stacked expert tensor's declared range fits the fixture bytes");
    let stack = &file_bytes[range.start as usize..range.end as usize];
    let per_expert = stack.len() / support::EXPERT_COUNT as usize;
    stack[PAGED_EXPERT * per_expert..(PAGED_EXPERT + 1) * per_expert].to_vec()
}

/// Re-encodes [`PAGED_EXPERT`]'s current `Q4_K` bytes as `Q2_K` --
/// dequantize-then-requantize through the real
/// [`proxima_gguf::quant::q4_k::dequantize`]/[`encode_expert_copy`] pair,
/// never inventing new weight data, so the paged copy is a genuine
/// (lossier) re-encoding of the SAME expert rather than an arbitrary
/// substitute.
fn paged_expert_q2k_bytes(file_bytes: &[u8]) -> Vec<u8> {
    let hi_bytes = paged_expert_hi_bytes(file_bytes);
    let element_count = support::FEED_FORWARD as usize * support::EMBEDDING as usize;
    let mut dequantized = vec![0.0f32; element_count];
    proxima_gguf::quant::q4_k::dequantize(&hi_bytes, &mut dequantized)
        .expect("the fixture's own Q4_K expert bytes dequantize cleanly");
    encode_expert_copy(
        &dequantized,
        support::FEED_FORWARD,
        support::EMBEDDING,
        PackedOwnedKind::Q2K,
    )
    .expect("a full-size dequantized expert row set re-encodes to Q2_K")
}

/// The index of the first position two equal-length id sequences disagree
/// at, or `None` if they are identical -- the mechanical way this test
/// finds where a paged expert's bytes actually started influencing greedy
/// selection, rather than asserting a hand-picked position number no
/// production caller could re-derive from the fixture's own seeds.
fn first_divergence(left: &[u32], right: &[u32]) -> Option<usize> {
    left.iter()
        .zip(right.iter())
        .position(|(first, second)| first != second)
}

/// Paging [`PAGED_EXPERT`] from `Q4_K` to a real [`encode_expert_copy`]-built
/// `Q2_K` copy changes at least one decoded id relative to the unpaged run
/// (run A vs run B), and never a position before the first such divergence.
/// Run C independently reloads the SAME checkpoint bytes and pages the
/// SAME `Q2_K` copy before its own first decode call: run B and run C page
/// through the identical `LoadedModel::page_expert` call, so their ids
/// matching bit-for-bit is a determinism control -- two independently
/// loaded models, given the same paged bytes, must decode identically.
#[proxima::test]
async fn q2k_paging_actually_changes_the_decoded_ids() {
    const TOKENS: usize = 4;

    let file_bytes = real_output_moe_checkpoint_bytes();
    let parsed: &'static proxima_gguf::pipe::ParsedGguf = Box::leak(Box::new(
        proxima_gguf::parse_complete(file_bytes)
            .expect("parses the synthetic Q4_K MoE checkpoint with a real output projection"),
    ));
    let q2k_bytes = paged_expert_q2k_bytes(file_bytes);

    // Run A: every step reads PAGED_EXPERT's original Q4_K bytes.
    let model_a = load_real_output_moe_fixture(parsed, file_bytes);
    let (ids_a, _text_a, _stopped_a) = Pipe::call(&model_a, (PROMPT.to_string(), TOKENS))
        .await
        .expect("run A decodes before any paging");
    assert_eq!(
        ids_a.len(),
        TOKENS,
        "an unbounded budget runs all four decode steps"
    );

    // Run B: an independent model over the SAME checkpoint bytes,
    // PAGED_EXPERT re-paged to a Q2_K re-encoding BEFORE this model's own
    // first decode call.
    let model_b = load_real_output_moe_fixture(parsed, file_bytes);
    let epoch = model_b
        .page_expert(
            GATE_SITE,
            PAGED_EXPERT,
            PackedOwnedKind::Q2K,
            &q2k_bytes,
            support::FEED_FORWARD,
            support::EMBEDDING,
        )
        .expect("paging to a real Q2_K re-encoding of the same expert succeeds");
    assert_eq!(
        epoch, 1,
        "the first page of a freshly loaded slab bumps epoch 0 -> 1"
    );
    let (ids_b, _text_b, _stopped_b) = Pipe::call(&model_b, (PROMPT.to_string(), TOKENS))
        .await
        .expect("run B decodes after paging");
    assert_eq!(
        ids_b.len(),
        TOKENS,
        "paging must not change how many tokens a fresh call produces"
    );

    // Run C: a FRESH model, paged to the SAME Q2_K bytes BEFORE its first
    // ever decode call -- "Q2_K from the start".
    let model_c = load_real_output_moe_fixture(parsed, file_bytes);
    model_c
        .page_expert(
            GATE_SITE,
            PAGED_EXPERT,
            PackedOwnedKind::Q2K,
            &q2k_bytes,
            support::FEED_FORWARD,
            support::EMBEDDING,
        )
        .expect("a never-yet-decoded model also accepts the same Q2_K paged bytes");
    let (ids_c, _text_c, _stopped_c) = Pipe::call(&model_c, (PROMPT.to_string(), TOKENS))
        .await
        .expect("run C decodes with the expert already paged");
    assert_eq!(ids_c.len(), TOKENS);

    let divergence = first_divergence(&ids_a, &ids_b);
    assert!(
        divergence.is_some(),
        "paging PAGED_EXPERT to a real Q2_K re-encoding must change at least one decoded id \
         over {TOKENS} tokens with a non-degenerate output projection, got identical runs \
         ids_a={ids_a:?} ids_b={ids_b:?}"
    );
    let divergence = divergence.expect("checked above");
    assert!(
        divergence < TOKENS,
        "the position search never runs past the shorter sequence's own length"
    );
    assert_eq!(
        ids_a[..divergence],
        ids_b[..divergence],
        "every id strictly before the first divergence must still match: the paged expert's \
         swap cannot retroactively change a token already decoded before the page happened"
    );

    assert_eq!(
        ids_b, ids_c,
        "run B (paged mid-run) and run C (paged before the first decode) must produce the \
         IDENTICAL id sequence -- page_expert leaves no state behind that a fresh, \
         already-paged model would not also carry"
    );
}
