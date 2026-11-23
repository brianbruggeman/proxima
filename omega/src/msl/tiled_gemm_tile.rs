//! The `BLOCK_M x BLOCK_N x BLOCK_K` tile one tiled-GEMM dispatch runs.
//!
//! A tile is a trade between three costs for an output of `M` features by `N`
//! tokens: the weight bytes `M * K` re-read once per `BLOCK_N`-token column
//! tile, the activation bytes `N * K` re-read once per `BLOCK_M`-feature row
//! tile, and the threadgroup's own memory. Total traffic for one GEMM is
//! `M * N * K * (weight_bytes_per_element / BLOCK_N + activation_bytes_per_element
//! / BLOCK_M)`, so both edges want to be as wide as the hardware allows.
//!
//! What the hardware allows is what bounds [`Self::for_feature_extent`]: a
//! threadgroup holds at most [`TILE_THREADGROUP_BYTES`] of `weight_tile`/
//! `act_tile`/`out_tile` staging.
//!
//! The threadgroup budget is also the binding cost, measured (PROXNUM-588):
//! the flux2 VAE's `3x3` convolution reduces at `128` features by 1,052,676
//! tokens ran at `41`-`46` s with the configured `64 x 32 x 32` tile (8 KB, four
//! threadgroups per core), `55` s at `128 x 64 x 32` (32 KB, one per core) and
//! `158` s at `128 x 64 x 32` backed by a 16 KB store (two per core). Halving
//! the activation re-read is worth less than the resident threadgroups it
//! costs, so the column edge is always the configured one; only the row edge
//! narrows, to the feature extent the op actually has. [`Self::tiles`] rounds
//! the token boundary up to whole column tiles, so a partial last column is
//! still a full `block_n`-wide tile of zero-padded reads rather than a short
//! one the staged operand and fragment loop cannot address.

use crate::sized::{TILED_GEMM_BLOCK_K, TILED_GEMM_BLOCK_M, TILED_GEMM_BLOCK_N};

/// Edge of one `simdgroup_matrix` fragment: `TILE_DIM` on each side, the fixed
/// 8x8 of `simdgroup_float8x8`.
const TILE_DIM: u64 = 8;
/// Smallest tile edge: two fragment rows per row-half, so both the row and the
/// column edge split evenly between the [`crate::msl::TILED_GEMM_NSG`] = 4
/// simdgroups' 2x2 halves (`row_half`/`col_half` each own half an edge, and
/// `simdgroup_load`/`simdgroup_store` address fragments, never half of one).
pub(crate) const TILE_MIN_EDGE: u64 = 2 * 2 * TILE_DIM;
/// Metal's threadgroup memory ceiling: the slim staging store
/// ([`Self::threadgroup_bytes`]) must fit inside it.
pub(crate) const TILE_THREADGROUP_BYTES: u64 = 32 * 1024;

/// One tiled-GEMM launch geometry. Every consumer -- the dispatch grid, the
/// threadgroup's staged arrays, the fragment loop bounds and the kernel
/// identity -- reads this one value for an op, so two ops that pick different
/// tiles can never share a compiled pipeline or a launch shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TiledGemmTile {
    /// Output feature rows one threadgroup owns (`BLOCK_SIZE_M`).
    pub(crate) block_m: u64,
    /// Activation tokens one threadgroup owns (`BLOCK_SIZE_N`).
    pub(crate) block_n: u64,
    /// Reduction extent one barrier pair amortizes over (`BLOCK_SIZE_K`).
    pub(crate) block_k: u64,
}

impl TiledGemmTile {
    /// The build-time configured tile (`omega-runtime.toml`'s
    /// `[tiled_gemm]`): what every op that [`Self::for_feature_extent`] leaves alone
    /// dispatches with, and what the dense-batched GEMM keeps using.
    pub(crate) const CONFIGURED: Self = Self {
        block_m: TILED_GEMM_BLOCK_M,
        block_n: TILED_GEMM_BLOCK_N,
        block_k: TILED_GEMM_BLOCK_K,
    };

    /// The tile for a GEMM whose output has `feature_extent` features.
    ///
    /// The row edge narrows to the feature extent the op actually has (a
    /// `3`-channel output decoder carve is one fragment row, not 61 zero rows
    /// of work) and never widens past the configured tile. The column edge is
    /// the configured one, never a shape-selected width: the measurement in
    /// this module's own doc is that the threadgroup store a wider edge costs
    /// is worth more than the re-read it saves. [`Self::tiles`] cuts the token
    /// extent into whole column tiles, rounding the boundary up.
    pub(crate) fn for_feature_extent(feature_extent: u64) -> Self {
        let configured = Self::CONFIGURED;
        let mut tile = Self {
            block_m: round_up_edge(feature_extent).clamp(TILE_MIN_EDGE, configured.block_m),
            block_n: configured.block_n,
            block_k: configured.block_k,
        };
        while tile.block_m > TILE_MIN_EDGE && tile.threadgroup_bytes() > TILE_THREADGROUP_BYTES {
            tile.block_m -= TILE_MIN_EDGE;
        }
        tile
    }

    /// Bytes of the threadgroup staging store this tile declares: the `half`
    /// weight tile and the `float` activation tile side by side, or the
    /// `float` output staging the writeback aliases over them, whichever is
    /// larger.
    ///
    /// The output term is ONE COLUMN HALF, not the whole tile: the writeback
    /// restages the accumulators a column half at a time (each simdgroup's own
    /// `col_half` owns `block_n / 2` tokens), so the buffer is
    /// `block_m * block_n / 2` floats. Staging the whole tile instead would
    /// make the output term `block_m * block_n * 4` — 32 KB at `128 x 64` —
    /// and a 32 KB threadgroup halves the number of threadgroups one core can
    /// keep resident, which measured slower than every traffic saving the
    /// wider tile bought.
    pub(crate) fn threadgroup_bytes(self) -> u64 {
        let weight_and_activation =
            self.block_m * self.block_k * 2 + self.block_n * self.block_k * 4;
        let output = self.block_m * (self.block_n / 2) * 4;
        if weight_and_activation > output {
            weight_and_activation
        } else {
            output
        }
    }

    /// `(row_tiles, col_tiles)` this tile cuts an `M` by `N` output into.
    pub(crate) fn tiles(self, feature_extent: u64, token_extent: u64) -> (u64, u64) {
        (
            feature_extent.div_ceil(self.block_m),
            token_extent.div_ceil(self.block_n),
        )
    }
}

/// `edge` rounded up to the smallest legal tile edge at or above it.
fn round_up_edge(edge: u64) -> u64 {
    edge.div_ceil(TILE_MIN_EDGE) * TILE_MIN_EDGE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_configured_tile_is_a_legal_geometry() {
        let tile = TiledGemmTile::CONFIGURED;
        assert!(tile.block_m >= TILE_MIN_EDGE && tile.block_m.is_multiple_of(TILE_MIN_EDGE));
        assert!(tile.block_n >= TILE_MIN_EDGE && tile.block_n.is_multiple_of(TILE_MIN_EDGE));
        assert!(tile.block_k.is_multiple_of(TILE_MIN_EDGE));
        assert!(tile.threadgroup_bytes() <= TILE_THREADGROUP_BYTES);
    }

    /// The convolution this rule exists for: `128` features over a million
    /// tokens takes the configured `64`-row edge (two row tiles, the
    /// threadgroup residency the measurement above pays for) and the
    /// configured column edge.
    #[test]
    fn a_narrow_feature_axis_never_widens_past_the_configured_tile() {
        let configured = TiledGemmTile::CONFIGURED;
        let tile = TiledGemmTile::for_feature_extent(128);
        assert_eq!(tile.block_m, configured.block_m);
        assert_eq!(tile.block_n, configured.block_n);
        assert_eq!(tile.block_k, configured.block_k);
        assert_eq!(
            tile.tiles(128, 1_052_676),
            (2, 1_052_676u64.div_ceil(configured.block_n))
        );
    }

    /// A feature axis narrower than the configured row edge narrows with it —
    /// `32` output channels are one row tile, not half of one plus 32 zero
    /// rows of work.
    #[test]
    fn a_feature_axis_below_the_configured_edge_narrows_to_it() {
        let tile = TiledGemmTile::for_feature_extent(32);
        assert_eq!(tile.block_m, 32);
        assert_eq!(tile.tiles(32, 1_048_576).0, 1);
    }

    /// A wide feature axis keeps the configured edge, so an op that already
    /// amortizes its activation over many row tiles is unchanged.
    #[test]
    fn a_wide_feature_axis_keeps_the_configured_row_edge() {
        let configured = TiledGemmTile::CONFIGURED;
        let tile = TiledGemmTile::for_feature_extent(27_648);
        assert_eq!(tile.block_m, configured.block_m);
        assert_eq!(tile.block_n, configured.block_n);
    }

    /// A feature axis narrower than one fragment still rounds up to a legal
    /// edge: `3` output channels become one fragment-row tile, never a
    /// zero-height one.
    #[test]
    fn a_feature_axis_below_one_fragment_rounds_up() {
        let tile = TiledGemmTile::for_feature_extent(3);
        assert_eq!(tile.block_m, TILE_MIN_EDGE);
        assert!(tile.block_n >= TILE_MIN_EDGE);
        assert!(tile.threadgroup_bytes() <= TILE_THREADGROUP_BYTES);
    }

    /// The boundary column keeps the configured edge and rounds up to a whole
    /// tile: a token extent that is not a multiple of `block_n` gets one more
    /// `block_n`-wide tile, never a short one the staged operand and fragment
    /// loop cannot address.
    #[test]
    fn a_token_boundary_rounds_up_to_whole_column_tiles() {
        let configured = TiledGemmTile::CONFIGURED;
        let tile = TiledGemmTile::for_feature_extent(128);
        assert_eq!(tile.block_n, configured.block_n);
        for (tokens, expected) in [
            (16u64, 1u64),
            (32, 1),
            (33, 2),
            (40, 2),
            (70, 3),
            (1_052_676, 32_897),
        ] {
            assert_eq!(
                tile.tiles(128, tokens).1,
                expected,
                "{tokens} tokens cut into {expected} column tiles"
            );
        }
    }

    /// Every selected geometry stays inside the threadgroup budget and the
    /// fragment cap, across the shapes a decoder actually reaches.
    #[test]
    fn every_selected_tile_fits_the_threadgroup_and_register_budget() {
        let fragment_cap =
            (TILED_GEMM_BLOCK_M / TILE_MIN_EDGE) * (TILED_GEMM_BLOCK_N / TILE_MIN_EDGE);
        for feature_extent in [3u64, 16, 32, 128, 256, 512, 3072, 27_648] {
            for token_extent in [64u64, 4608, 16_384, 1_052_676] {
                let tile = TiledGemmTile::for_feature_extent(feature_extent);
                assert!(
                    tile.threadgroup_bytes() <= TILE_THREADGROUP_BYTES,
                    "{feature_extent}x{token_extent}: {tile:?} needs {} bytes",
                    tile.threadgroup_bytes()
                );
                let fragments = (tile.block_m / TILE_MIN_EDGE) * (tile.block_n / TILE_MIN_EDGE);
                assert!(
                    fragments <= fragment_cap,
                    "{feature_extent}x{token_extent}: {tile:?} holds {fragments} fragments"
                );
            }
        }
    }
}
