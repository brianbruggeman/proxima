//! The medium-independent boundary between `serving_fsm::ServingState<Cache>`
//! and a real evaluation medium (CPU, Metal, ...), for this crate's own
//! forward programs.
//!
//! `ServingState<Cache>` (`crate::serving_fsm`) is already generic over
//! `Cache` -- the bug the prior wiring attempt ran into was never the FSM
//! itself, it was that `LoadedModel::run_decode_loop_observed_seeded`
//! (`decode.rs`, roughly lines 1250-1650) has no such boundary: its one
//! closure owns `layer_caches: Vec<LayerCacheState>` (logical, host-only
//! `Vec<f32>` state -- already medium-independent) ALONGSIDE
//! `dense_attention_buffers: Vec<Option<Qwen35DenseAttentionBuffers>>` and
//! `ssm_state_buffers: Vec<Option<(PlacedBuffer, PlacedBuffer)>>` (Metal
//! device-resident, correctness depends on stable addresses -- `decode.rs`'s
//! own doc on `ROW 531 invariant 2`), plus a stateful residency policy
//! (`qwen35moe_residency`), all threaded through the same 14-local closure
//! with no line between "logical state a `ServingState` transition may
//! carry" and "device resource a backend must own privately". This module
//! draws that line, as real, compiling, tested types -- not wired into
//! `decode.rs` (a separate, later step: the live loop's `Op::Input`
//! derivation, chunked prefill, expert residency, and instrumentation all
//! still belong to `LoadedModel::run_decode_loop_observed_seeded` and are
//! out of scope here).
//!
//! # The two halves
//!
//! [`ServingCache`] is the concrete `Cache` this crate's `ServingState`
//! should be instantiated with: `Vec<LayerCacheState>`, the exact host-only
//! per-layer state `decode.rs` already builds via
//! [`LoadedModel::fresh_layer_caches`] and advances via
//! [`LayerCache::append`]/[`Qwen35DenseAttentionCache::append`]/
//! [`SsmLayerCache::advance`] -- no new type, because none is needed: this
//! state was already medium-independent, just not yet named as the thing a
//! `ServingState` carries.
//!
//! [`ServingBackend`] is the boundary itself: one method, `evaluate`, the
//! medium-independent op the module doc on `serving_fsm` names ("evaluate
//! program at positions, advance the logical cache"). An implementor owns
//! whatever the medium needs on `Self`, never inside [`ServingCache`], so
//! cloning a `ServingCache` for `ServingState::Verify`'s `snapshot`
//! (`serving_fsm.rs`'s `enter_verify`) never clones or aliases a device
//! buffer -- there is no device buffer inside it to clone.
//!
//! [`MetalPlacementResources`] shows where `decode.rs`'s two device-resident
//! locals belong once a Metal `ServingBackend` is written: owned fields on a
//! handle a backend impl holds behind `&mut self`, mutated in place across
//! `evaluate` calls (stable addresses satisfied by construction -- the
//! handle itself is never cloned, only `&mut`-borrowed), instead of bare
//! locals threaded by layer index through one closure. It is deliberately
//! inert: no allocation, no `omega::metal` call, no `ServingBackend` impl --
//! wiring a real Metal backend means proving `evaluate` against the real
//! `qwen35moe` program, which is exactly the live-loop migration this step
//! does not attempt.

// not yet wired into `decode.rs`'s live loop (a later, separate step); this
// module's own tests below are its only caller until then, the same reason
// `serving_fsm.rs` (`ServingState<Cache>` itself) carries this allow.
#![allow(dead_code)]

use alloc::vec::Vec;

use super::*;

/// The concrete `Cache` for `serving_fsm::ServingState<Cache>` against this
/// crate's real forward programs -- logical KV/recurrent/sequence state
/// only, one entry per layer, matching [`LoadedModel::layer_roots`]'s own
/// per-layer discriminant. Never holds a device handle: a [`ServingBackend`]
/// mutates its own private resources and returns only this.
pub(super) type ServingCache = Vec<LayerCacheState>;

/// One medium-independent evaluation: place `positions` (prompt rows on a
/// prefill call, one row on a decode call, `K` rows on a verify call --
/// `serving_fsm::ServingState`'s own three evaluating variants), advance
/// `cache` by however many rows this evaluation consumed, and return
/// whatever per-step output the caller needs (`Step`: greedy token ids,
/// logits, or both -- `ServingState` itself never inspects this).
///
/// An implementor owns any device resources itself (Metal `PlacedBuffer`s
/// at stable addresses, a CPU scratch buffer, a stateful residency policy)
/// on `Self`, mutated through `&mut self` across calls -- never smuggled
/// into `cache` or the return value, so nothing here can leak a device
/// handle into a `ServingState` variant.
pub(super) trait ServingBackend {
    type Step;
    type Error;

    fn evaluate(
        &mut self,
        positions: &[u32],
        cache: ServingCache,
    ) -> Result<(Self::Step, ServingCache), Self::Error>;
}

/// Where `decode.rs`'s two Metal device-resident locals belong once a real
/// Metal [`ServingBackend`] exists: `dense_attention` mirrors
/// `dense_attention_buffers: Vec<Option<Qwen35DenseAttentionBuffers>>`
/// (`decode.rs` line ~1355, allocated at ~1370-1403), `ssm_state` mirrors
/// `ssm_state_buffers: Vec<Option<(PlacedBuffer, PlacedBuffer)>>`
/// (`decode.rs` line ~1469, allocated at ~1476-1487) -- both indexed by
/// layer, both currently bare locals threaded through the decode closure by
/// hand. Owned here instead: a Metal backend holds one `MetalPlacementResources`
/// behind `&mut self` and mutates its buffers in place every `evaluate`
/// call, so their addresses stay stable for exactly the reason `decode.rs`'s
/// `ROW 531 invariant 2` comment requires, without either buffer ever
/// crossing into a `ServingCache` that `ServingState::Verify` clones.
///
/// Deliberately unconstructed: no `allocate_placed_buffer`, no
/// `omega::metal` call, no `ServingBackend` impl. Wiring this in means
/// proving a real Metal `evaluate` against the `qwen35moe` program, out of
/// scope for this step (`serving_fsm.rs`'s own doc on why that oracle is
/// separately blocked).
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) struct MetalPlacementResources {
    pub(super) dense_attention: Vec<Option<Qwen35DenseAttentionBuffers>>,
    pub(super) ssm_state: Vec<Option<(PlacedBuffer, PlacedBuffer)>>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::serving_fsm::ServingState;

    /// A CPU backend proving [`ServingBackend`]'s contract is satisfiable by
    /// the real per-layer cache mutators (`LayerCache::append`,
    /// `SsmLayerCache::advance`) rather than a stand-in -- `evaluate` places
    /// `positions`, appends deterministic synthetic K/V/state, and predicts
    /// the next token as `positions.last() + 1` (mirrors `serving_fsm.rs`'s
    /// own `fake_greedy_next`), so a caller can drive `ServingState`
    /// transitions directly off this backend's own return values.
    struct FakeCpuBackend {
        vocab: u32,
    }

    impl ServingBackend for FakeCpuBackend {
        type Step = u32;
        type Error = core::convert::Infallible;

        fn evaluate(
            &mut self,
            positions: &[u32],
            mut cache: ServingCache,
        ) -> Result<(u32, ServingCache), Self::Error> {
            for (layer, state) in cache.iter_mut().enumerate() {
                let synthetic: Vec<f32> = positions
                    .iter()
                    .map(|&position| (position + layer as u32) as f32)
                    .collect();
                match state {
                    LayerCacheState::Attention(layer_cache) => {
                        layer_cache.append(&synthetic, &synthetic, &synthetic);
                    }
                    LayerCacheState::DenseAttention(layer_cache) => {
                        layer_cache.append(&synthetic, &synthetic, &synthetic, &synthetic);
                    }
                    LayerCacheState::Ssm(layer_cache) => {
                        layer_cache.advance(&synthetic, &synthetic, synthetic.len());
                    }
                    LayerCacheState::SharedFromLayer => {}
                }
            }
            let last = *positions.last().expect("evaluate is never called with zero positions");
            let next_token = (last + 1) % self.vocab;
            Ok((next_token, cache))
        }
    }

    fn fresh_cache() -> ServingCache {
        alloc::vec![
            LayerCacheState::Attention(LayerCache::new()),
            LayerCacheState::Ssm(SsmLayerCache::new(4, 2)),
        ]
    }

    /// `evaluate`'s output drives `ServingState::advance_prefill` /
    /// `advance_decode` directly, and every layer's logical cache grew by
    /// exactly the rows each call placed -- proving `ServingCache` threads
    /// through `ServingState` transitions the same way `serving_fsm.rs`'s
    /// own `FakeCache` walkthrough proves `Cache` does, except here `Cache`
    /// is the real per-layer state this crate's decode loop already builds,
    /// not a stand-in.
    #[test]
    fn serving_backend_drives_serving_state_through_prefill_and_decode() {
        let mut backend = FakeCpuBackend { vocab: 11 };
        let prompt = alloc::vec![3_u32, 5, 7];
        let prompt_rows = prompt.len();

        let state = ServingState::start(prompt.clone(), fresh_cache());
        // A real driver reads `positions`/`cache` back off the FSM's own
        // current variant, calls the backend, then feeds the result to the
        // matching transition -- `evaluate` never sees `ServingState`
        // itself, only the plain values the variant carries.
        let (positions, cache) = match &state {
            ServingState::Prefill { positions, cache } => (positions.clone(), cache.clone()),
            _ => panic!("start always enters Prefill"),
        };
        let (next_token, cache) = backend
            .evaluate(&positions, cache)
            .expect("fake backend never errors");
        assert_eq!(next_token, 8, "(7 + 1) % 11 == 8");
        let LayerCacheState::Attention(attention) = &cache[0] else {
            panic!("layer 0 is Attention");
        };
        assert_eq!(
            attention.k_even.len(),
            prompt_rows,
            "prefill placed every prompt row into layer 0's cache"
        );
        let LayerCacheState::Ssm(ssm) = &cache[1] else {
            panic!("layer 1 is Ssm");
        };
        assert_eq!(
            ssm.state.len(),
            prompt_rows,
            "prefill's synthetic state replaced layer 1's state wholesale"
        );

        let state = state
            .advance_prefill(next_token, cache)
            .expect("prefill -> decode is legal");
        let (last, cache) = match &state {
            ServingState::Decode { last, cache } => (*last, cache.clone()),
            _ => panic!("advance_prefill always enters Decode"),
        };
        assert_eq!(last, 8);
        let (next_token, cache) = backend
            .evaluate(&[last], cache)
            .expect("fake backend never errors");
        assert_eq!(next_token, 9, "(8 + 1) % 11 == 9");
        let LayerCacheState::Attention(attention) = &cache[0] else {
            panic!("layer 0 is Attention");
        };
        assert_eq!(
            attention.k_even.len(),
            prompt_rows + 1,
            "3 prefill rows plus this one decode row"
        );

        let state = state
            .advance_decode(next_token, cache)
            .expect("decode -> decode is legal");
        assert!(matches!(state, ServingState::Decode { last: 9, .. }));
    }
}
