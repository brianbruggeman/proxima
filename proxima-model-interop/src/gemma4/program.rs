//! The one piece of Gemma 4's forward pass the generic
//! [`proxima_tensor::spec::lfm2_forward_program_with_experts`] engine cannot
//! build itself: the sliding-window RoPE table's own values. Every other
//! node this checkpoint needs (attention mixer, dense/routed FFN, per-layer
//! norms, embedding scale, logit softcap) is now a config value
//! [`crate::gemma4::bind::Gemma4Arch::bind`] hands that engine directly --
//! see that module's own doc for the descriptor it builds. This file used to
//! hold a bespoke `gemma4_forward_program`/`gemma4_attention`/
//! `gemma4_ffn_block` graph-building layer; that layer is deleted, not
//! moved, now that the generic engine's `LayerAttentionConfig`/
//! `LayerFfnConfig`/`ValueSource`/`RopePairing` knobs (added the slice
//! before this one) express Gemma 4's own schedule directly.

use alloc::vec::Vec;

/// Builds the sliding-window layers' own RoPE `cos`/`sin` table --
/// [`crate::gemma4::bind::Gemma4Arch::step_inputs`]'s own seam feeds this
/// into the `rope_cos_swa`/`rope_sin_swa` leaves
/// [`proxima_tensor::spec::lfm2_forward_program_with_experts`] declares once
/// a sliding layer's [`proxima_tensor::spec::RopeTableSel`] names them,
/// since the FULL-layer table those leaves' own `rope_cos`/`rope_sin`
/// siblings read comes from the decode loop's builtin per-position table at
/// the checkpoint's own full-layer base/dimension instead
/// (`Gemma4Arch::bind`'s own `ModelArchitecture::head_dim`/
/// `rope_freq_base`).
#[must_use]
pub fn gemma4_sliding_rope_table(
    positions: &[usize],
    freq_base: f32,
    dimension_count: u32,
) -> (Vec<f32>, Vec<f32>) {
    let pairs = dimension_count as usize / 2;
    let mut cos = alloc::vec![0.0f32; positions.len() * pairs];
    let mut sin = alloc::vec![0.0f32; positions.len() * pairs];
    for (offset, &position) in positions.iter().enumerate() {
        for pair in 0..pairs {
            let theta =
                position as f32 * freq_base.powf(-((2 * pair) as f32) / dimension_count as f32);
            cos[offset * pairs + pair] = theta.cos();
            sin[offset * pairs + pair] = theta.sin();
        }
    }
    (cos, sin)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_sliding_rope_table_produces_identity_angles_at_position_zero() {
        let (cos, sin) = gemma4_sliding_rope_table(&[0], 1.0e4, 4);
        assert_eq!(
            cos,
            alloc::vec![1.0, 1.0],
            "theta = 0 at position 0 for every pair"
        );
        assert_eq!(sin, alloc::vec![0.0, 0.0]);
    }

    #[test]
    fn gemma4_sliding_rope_table_varies_by_position() {
        let (cos_zero, _) = gemma4_sliding_rope_table(&[0], 1.0e4, 4);
        let (cos_one, _) = gemma4_sliding_rope_table(&[1], 1.0e4, 4);
        assert_ne!(
            cos_zero, cos_one,
            "distinct positions rotate by distinct angles"
        );
    }
}
